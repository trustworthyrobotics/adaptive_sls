# sqp.py
from jax import jit, lax, vmap
import jax
import jax.numpy as jnp

from functools import partial

from trajax.optimizers import linearize, quadratize,vectorize
from gpu_sls.gpu_sls import (
    SLSConfig,
    posterior_trace_information_gain,
    sls_solve_gpu,
)
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass

@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start, self.line_search)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def information_augmented_cost(
    base_cost,
    dynamics,
    disturbance,
    G_lagged,
    measurement_matrix,
    sls_config,
    horizon,
):
    """Add the optional lagged-generator information term to a stage cost."""
    if not sls_config.enable_information_cost:
        return base_cost

    if not sls_config.adaptive or sls_config.n_theta <= 0:
        raise ValueError(
            "enable_information_cost requires adaptive dynamics with parameter states"
        )

    n_x = sls_config.n_x
    n_theta = sls_config.n_theta

    def cost_with_information(x, u, t):
        nominal_cost = base_cost(x, u, t)

        def transition_information(_):
            dynamics_jacobian = jax.jacfwd(dynamics, argnums=0)(x, u, t)
            E_t = dynamics_jacobian[:n_x, n_x : n_x + n_theta]
            # The first implementation intentionally uses only the stage-wise
            # exogenous map. Horizon-coupled remainder generators are excluded.
            F_t = disturbance(x[None, :])[0][:n_x]
            G_t = G_lagged[t]
            information_gain = posterior_trace_information_gain(
                E_t,
                F_t,
                G_t,
                measurement_matrix,
                gain_regularization=(
                    sls_config.information_gain_regularization
                ),
                enable_posterior_gate=sls_config.enable_posterior_gate,
            )
            discount = jnp.asarray(
                sls_config.information_cost_discount,
                dtype=x.dtype,
            ) ** t
            return (
                jnp.asarray(
                    sls_config.information_cost_weight,
                    dtype=x.dtype,
                )
                * discount
                * information_gain
            )

        information_reward = lax.cond(
            t < horizon,
            transition_information,
            lambda _: jnp.asarray(0.0, dtype=x.dtype),
            operand=None,
        )
        # The helper returns positive absolute uncertainty reduction, so it is
        # subtracted from the nominal minimization objective.
        return nominal_cost - information_reward

    return cost_with_information

def lagrangian(cost, dynamics, constraints, x0, obstacles, backoffs):
    """Full constrained SQP Lagrangian used by the GPU-SLS baseline."""

    def fun(x, u, t, v, v_prev, lam):
        c1 = cost(x, u, t)
        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))

        g_base = constraints(x, u, t)
        n_tightened = backoffs.shape[1]
        g_base_tight = jnp.concatenate(
            [
                g_base[:n_tightened] + backoffs[t],
                g_base[n_tightened:],
            ]
        )

        if obstacles.shape[0] == 0:
            g_obs_tight = jnp.empty((0,), dtype=g_base.dtype)
        else:
            centers = obstacles[:, :2]
            radii = obstacles[:, 2]
            diff = x[:2][None, :] - centers
            dist = jnp.linalg.norm(diff, axis=-1) + 1e-6
            normal = diff / dist[:, None]
            hx = jnp.abs(backoffs[t, 0])
            hy = jnp.abs(backoffs[t, 1])
            obs_backoff = jnp.abs(normal[:, 0]) * hx + jnp.abs(normal[:, 1]) * hy
            g_obs_tight = radii - dist + obs_backoff

        g_all = jnp.concatenate([g_base_tight, g_obs_tight], axis=0)
        c4 = jnp.dot(lam, g_all)
        return c1 + c2 + c3 + c4
    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    pos = x_curr[:, :2]
    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]
    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    
    return C_all, D_all, f_all

@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
def compute_search_direction(
    sls_config: SLSConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    limited_memory,
    constraints, disturbance, remainder_func, splts_cfg,
    measurement_matrix,
    obstacles,
    x0, X, U, V, c,
    w, y, rho,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev, K_prev, G_prev, G_0, P_inv_prev, nominal_parameter, dynamics_parameter,
    Q_bar, R_bar, Q_f_bar,
):
    T = U.shape[0]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])
    stage_cost = information_augmented_cost(
        cost,
        dynamics,
        disturbance,
        G_prev,
        measurement_matrix,
        sls_config,
        T,
    )

    if hessian_approx is None or sls_config.enable_information_cost:
        quadratizer = quadratize(stage_cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]
        
    linearizer = linearize(
        lagrangian(stage_cost, dynamics, constraints, x0, obstacles, h_ct_ws),
        argnums=6,
    )
    dynamics_linearizer = linearize(dynamics)

    q, r_pad = linearizer(
        X,
        pad(U),
        jnp.arange(T + 1),
        pad(V[1:]),
        V,
        y,
    )
    r = r_pad[:-1]

    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]
    nx = A.shape[1]
    nu = B.shape[2]

    U_pad = pad(U)
    t = jnp.arange(X.shape[0])

    g = vectorize(constraints)(X, U_pad, t)
    f = -g

    C, D = linearize(constraints)(X, U_pad, t)

    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)

    terminal_control_rows = jnp.linalg.norm(D_all[-1], axis=-1) > 1e-7

    D_all = D_all.at[-1].set(jnp.zeros_like(D_all[-1]))

    f_all = f_all.at[-1].set(
        jnp.where(
            terminal_control_rows,
            jnp.asarray(1e6, dtype=f_all.dtype),
            f_all[-1],
        )
    )

    if sls_config.adaptive:
        E = disturbance(X)
        sens = None
    else:
        if len(disturbance) == 4:
            E = disturbance[0](
                X, U, nominal_parameter, G_0, dynamics_parameter
            )
            sens = disturbance[1](X, U, nominal_parameter, dynamics_parameter)
        else:
            E = disturbance[0](X, U, nominal_parameter, G_0)
            sens = disturbance[1](X, U, nominal_parameter)
    cfg = admm_config

    if sls_config.enable_fastsls:
        nom_param = nominal_parameter if not sls_config.adaptive else None
        (
            dX, dU, dV, w, y, rho, converged, converged_admm,
            backoffs, Phi_x, Phi_u, K_kjN, betaN, muN, EN, KN, LN,
            GN, p_inv, sls_residual, admm_iterations,
            admm_primal_residual, admm_dual_residual,
            admm_primal_tolerance, admm_dual_tolerance,
            admm_primal_worst_flat_index, admm_primal_worst_z,
            admm_primal_worst_w,
        ) = sls_solve_gpu(
            cfg, remainder_func,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, sls_config,
            splts_cfg, E, E_prev, Q_bar, R_bar, Q_f_bar,
            obstacles, X, h_ct_ws, beta_ws, mu_ws,
            Phi_x_ws, Phi_u_ws, X, U, K_prev, G_prev, G_0, P_inv_prev, sens, nom_param,
            measurement_matrix,
        )
    else:
        (
            dX, dU, dV, w, y, rho, _, converged_admm,
            admm_iterations, admm_primal_residual, admm_dual_residual,
            admm_primal_tolerance, admm_dual_tolerance,
            admm_primal_worst_flat_index, admm_primal_worst_z,
            admm_primal_worst_w,
        ) = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        converged = converged_admm
        sls_residual = jnp.asarray(jnp.inf, dtype=Q.dtype)
        backoffs = jnp.zeros_like(h_ct_ws)
        Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u = jnp.zeros((T, T + 1, nu, nx))
        betaN = jnp.ones_like(beta_ws) * 1e-10
        muN = jnp.zeros((T + 1, nc))
        K_kjN = jnp.zeros((T, T + 1, nu, nx))
        EN = jnp.zeros_like(E_prev)
        KN  = jnp.zeros_like(K_prev)
        LN = jnp.zeros_like(K_prev)
        GN = jnp.zeros_like(G_prev)
        p_inv = jnp.zeros_like(P_inv_prev)

    return (
        dX, dU, dV, q, r, w, y, rho, backoffs, Phi_x, Phi_u,
        K_kjN, betaN, muN, EN, KN, LN, GN, p_inv, nominal_parameter,
        converged, sls_residual, converged_admm, admm_iterations,
        admm_primal_residual, admm_dual_residual, admm_primal_tolerance,
        admm_dual_tolerance, admm_primal_worst_flat_index,
        admm_primal_worst_z, admm_primal_worst_w,
    )

@jit
def merit_rho(c, dV):
    c2 = jnp.sum(c * c)
    dV2 = jnp.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * jnp.sqrt(dV2 / c2), 1e-2)

@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics,x0, X, U):
    T = U.shape[0]
    costs = vmap(cost)(X, jnp.pad(U, [[0, 1], [0, 0]]), jnp.arange(T + 1))
    g = jnp.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = jnp.vstack([x0 - X[0], vmap(residual_fn)(jnp.arange(T))])

    return g, c

def merit_function_factory(rho_merit):
    def merit_fn(V, g, c):
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function, model_evaluator,
    X_in, U_in, V_in,
    dX, dU, dV,
    current_merit, current_g, current_c,
    merit_slope, armijo_factor,
    alpha_0, alpha_mult, alpha_min,
):
    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        return jnp.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = jnp.where(jnp.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, jnp.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min

    return X, U, V, new_g, new_c, no_errors

@jit
def slope(dX, dU, dV, c, q, r, rho):
    return jnp.sum(q * dX) + jnp.sum(r * dU) + 2*jnp.sum(dV * c) - rho * jnp.sum(c * c)

@partial(jit, static_argnums=(0,1,2,3,4,5,6,7,8,9,10))
def mpc(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    limited_mempory,
    constraints, disturbance,
    remainder_func, splts_cfg,
    measurement_matrix,
    reference, parameter,
    W,
    x0, X_in, U_in, V_in,
    w, y, rho,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev, K_prev, G_prev, G_0, P_inv_prev, nominal_parameter,
    Q_bar=None, R_bar=None, Q_f_bar=None,
):
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    if sls_config.adaptive:
        _dynamics = partial(dynamics, parameter=parameter)
    else:
        _dynamics = partial(dynamics, parameter=parameter, nominal_param=nominal_parameter)

    horizon = U_in.shape[0]
    state_dim = X_in.shape[1]
    control_dim = U_in.shape[1]
    if Q_bar is None:
        Q_bar = jnp.broadcast_to(
            jnp.eye(state_dim, dtype=X_in.dtype),
            (horizon, state_dim, state_dim),
        )
    if R_bar is None:
        R_bar = jnp.broadcast_to(
            jnp.eye(control_dim, dtype=U_in.dtype),
            (horizon, control_dim, control_dim),
        )
    if Q_f_bar is None:
        Q_f_bar = jnp.eye(state_dim, dtype=X_in.dtype)

    def body(i, carry):
        (
            i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs,
            Phi_x, Phi_u, _, beta_ws, mu_w, E_prev, K_prev, L_prev,
            G_prev, P_inv_prev, sls_converged, sls_residual,
            admm_converged, admm_iterations, admm_primal_residual,
            admm_dual_residual, admm_primal_tolerance, admm_dual_tolerance,
            admm_primal_worst_flat_index, admm_primal_worst_z,
            admm_primal_worst_w,
        ) = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            iteration_cost = information_augmented_cost(
                _cost,
                _dynamics,
                disturbance,
                G_prev,
                measurement_matrix,
                sls_config,
                U_curr.shape[0],
            )
            model_evaluator = partial(
                model_evaluator_helper,
                iteration_cost,
                _dynamics,
                x0,
            )
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            warm_flag = jnp.array(bool(sqp_config.warm_start))

            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            rho0 = lax.select(
                warm_flag,
                rho,
                jnp.asarray(admm_config.initial_rho, dtype=rho.dtype),
            )
            h_ct_ws = backoffs
            (
                dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1,
                Phi_u1, K_kjN, betaN, muN, EN, KN, LN, GN, P_inv, _,
                sls_converged1, sls_residual1, admm_converged1,
                admm_iterations1, admm_primal_residual1,
                admm_dual_residual1, admm_primal_tolerance1,
                admm_dual_tolerance1,
                admm_primal_worst_flat_index1, admm_primal_worst_z1,
                admm_primal_worst_w1,
            ) = compute_search_direction(
                sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                limited_mempory,
                constraints, disturbance,
                remainder_func, splts_cfg,
                measurement_matrix,
                obstacles,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0,
                h_ct_ws, beta_ws, mu_w, Phi_x, Phi_u, E_prev, K_prev, G_prev, G_0, P_inv_prev, nominal_parameter, parameter,
                Q_bar, R_bar, Q_f_bar,
            )

            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            converged1 = jnp.logical_and(feas_ok, step_ok)
            X_next = lax.select(converged1, X_curr, X_curr + dX)
            U_next = lax.select(converged1, U_curr, U_curr + dU)
            V_next = lax.select(converged1, V_curr, V_curr + dV)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)
            merit_fn  = merit_function_factory(rho_merit)
            current_merit = merit_fn(V_curr, g, c)
            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)
            last_iter = (i == (sqp_config.max_sqp_iterations - 1))
            do_ls = jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter))

            def ls_branch(_):
                Xn, Un, Vn, g_new, c_new, ok = line_search(
                    merit_fn, model_evaluator,
                    X_curr, U_curr, V_curr,
                    dX, dU, dV,
                    current_merit, g, c,
                    merit_slope,
                    armijo_factor=1e-4,
                    alpha_0=1.0,
                    alpha_mult=0.5,
                    alpha_min=1e-6,
                )
                return Xn, Un, Vn

            def fullstep_branch(_):
                return (X_curr + dX, U_curr + dU, V_curr + dV)

            # X_next, U_next, V_next = lax.cond(do_ls, ls_branch, fullstep_branch, operand=None)

            # w_next = lax.select(converged1, w, w1)
            # y_next = lax.select(converged1, y, y1)
            # rho_next = lax.select(converged1, rho, rho1)
            # backoffs_next = lax.select(converged1, backoffs, backoffs1)
            # Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            # Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            # return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
            #         jnp.logical_or(converged, converged1),
            #         backoffs_next, Phi_x_next, Phi_u_next, K_kjN, betaN, muN, EN, KN, GN, P_inv)

            # Inside sqp.py -> do_iter()
            
            X_next, U_next, V_next = lax.cond(do_ls, ls_branch, fullstep_branch, operand=None)

            # FIX: Unconditionally accept the newly computed matrices!
            w_next = w1
            y_next = y1
            rho_next = rho1
            backoffs_next = backoffs1
            Phi_x_next = Phi_x1
            Phi_u_next = Phi_u1

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    jnp.logical_or(converged, converged1),
                    backoffs_next, Phi_x_next, Phi_u_next, K_kjN, betaN,
                    muN, EN, KN, LN, GN, P_inv, sls_converged1,
                    sls_residual1, admm_converged1, admm_iterations1,
                    admm_primal_residual1, admm_dual_residual1,
                    admm_primal_tolerance1, admm_dual_tolerance1,
                    admm_primal_worst_flat_index1, admm_primal_worst_z1,
                    admm_primal_worst_w1)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    T, Tp1, nu, nw = Phi_u_ws.shape
    nx = Phi_x_ws.shape[2]
    K_0 = jnp.zeros((T, Tp1, nu, nx))
    L_0 = jnp.zeros_like(K_prev)
    diagnostics_initial = (
        jnp.array(False),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.array(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
        jnp.asarray(jnp.inf, dtype=X_in.dtype),
    )
    carry0 = (
        0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0,
        Phi_x_ws, Phi_u_ws, K_0, beta_ws, mu_ws, E_prev, K_prev, L_0,
        G_prev, P_inv_prev, *diagnostics_initial,
    )
    (
        total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out,
        converged, backoffs, Phi_x, Phi_u, K_kjN, betaN, muN, EN, KN,
        LN, GN, P_inv, sls_converged, sls_residual, admm_converged,
        admm_iterations, admm_primal_residual, admm_dual_residual,
        admm_primal_tolerance, admm_dual_tolerance,
        admm_primal_worst_flat_index, admm_primal_worst_z,
        admm_primal_worst_w,
    ) = lax.fori_loop(
        0, sqp_config.max_sqp_iterations, body, carry0
    )
    return (
        X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x,
        Phi_u, K_kjN, betaN, muN, EN, KN, LN, GN, P_inv,
        sls_converged, sls_residual, admm_converged, admm_iterations,
        admm_primal_residual, admm_dual_residual, admm_primal_tolerance,
        admm_dual_tolerance, admm_primal_worst_flat_index,
        admm_primal_worst_z, admm_primal_worst_w,
    )
