# generic_mpc_wrapper.py
from __future__ import annotations
from functools import partial
from typing import Any
from typing import Callable
import jax
import jax.numpy as jnp
import gpu_sls.sqp as sqp
from gpu_sls.gpu_sls import girard_reduction_opt
from linearization_sls.test_path import remainder_bound_path_based
from linearization_sls.src.interval import Interval

# IMPORT JAXPR TRACERS
from linearization_sls.src.rhs_eval import _trace_prog, build_rhs_interval_from_prog

def pack_dynamics_as_single_input(dynamics, nx: int, nu: int, *, parameter, t_dim: int = 1, t_as_scalar: bool = True, adaptive=True, ntheta=None):
    dyn = partial(dynamics, parameter=parameter) 
    D = nx + nu + t_dim
    if not adaptive:
        D += ntheta

        def f_flat(z: jnp.ndarray) -> jnp.ndarray:
            x = z[:nx]
            nom_param = z[nx: nx+ntheta]
            u_start = nx + ntheta
            u = z[u_start:u_start+nu]
            t_slice = z[nx+nu+ntheta:nx+nu+t_dim+ntheta]
            t = t_slice[0] if (t_dim == 1 and t_as_scalar) else t_slice
            return dyn(x, u, t, nom_param)
    else:
        def f_flat(z: jnp.ndarray) -> jnp.ndarray:
            x = z[:nx]
            u = z[nx:nx+nu]
            t_slice = z[nx+nu:nx+nu+t_dim]
            t = t_slice[0] if (t_dim == 1 and t_as_scalar) else t_slice
            return dyn(x, u, t)

    return f_flat, D


class GenericMPCControllerWrapper:
    def __init__(
        self,
        sls_config, sqp_config, admm_config,
        config,
        dynamics, constraints, obstacles,
        cost,
        num_constraints: int,
        disturbance,
        X_in, U_in,
        limited_memory: bool = False,
        shift: int = 1,
        hessian_approx: Callable | None = None,
        measurement_matrix: jnp.ndarray | None = None,
        Q_bar: jnp.ndarray | None = None,
        R_bar: jnp.ndarray | None = None,
        Q_f_bar: jnp.ndarray | None = None,
        terminal_constraints: Callable | None = None,
        num_terminal_constraints: int = 0,
        fuse_rti_updates: bool = False,
    ):
        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        # Keep the legacy mutable Python carry by default.  Real-time callers
        # can opt into a single JIT that includes solve, carry update, and
        # warm-start shifting, avoiding hundreds of small GPU dispatches.
        self.fuse_rti_updates = fuse_rti_updates
        self.obstacles = obstacles
        if terminal_constraints is None and num_terminal_constraints != 0:
            raise ValueError(
                "num_terminal_constraints must be zero when terminal_constraints is None"
            )
        if terminal_constraints is not None and num_terminal_constraints <= 0:
            raise ValueError(
                "num_terminal_constraints must be positive when terminal_constraints is provided"
            )

        if terminal_constraints is None:
            solver_constraints = constraints
        else:
            def solver_constraints(x, u, t):
                base_values = constraints(x, u, t)
                terminal_values = terminal_constraints(x, u, t)
                inactive_values = -jnp.ones_like(terminal_values)
                terminal_values = jax.lax.cond(
                    t == config.N,
                    lambda _: terminal_values,
                    lambda _: inactive_values,
                    operand=None,
                )
                return jnp.concatenate([base_values, terminal_values])

        if measurement_matrix is None:
            measurement_matrix = jnp.eye(sls_config.n_x, dtype=X_in.dtype)
        measurement_matrix = jnp.asarray(measurement_matrix, dtype=X_in.dtype)
        expected_measurement_shape = (sls_config.n_x, sls_config.n_x)
        if measurement_matrix.shape != expected_measurement_shape:
            raise ValueError(
                "measurement_matrix must be a square physical-state gate with shape "
                f"{expected_measurement_shape}; got {measurement_matrix.shape}."
            )
        self.measurement_matrix = measurement_matrix

        def stage_weights(
            value: jnp.ndarray | None,
            *,
            horizon: int,
            size: int,
            dtype,
            name: str,
        ) -> jnp.ndarray:
            if value is None:
                value = jnp.eye(size, dtype=dtype)
            value = jnp.asarray(value, dtype=dtype)
            if value.shape == (size, size):
                return jnp.broadcast_to(value, (horizon, size, size))
            expected_shape = (horizon, size, size)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {(size, size)} or "
                    f"{expected_shape}; got {value.shape}."
                )
            return value

        self.Q_bar = stage_weights(
            Q_bar,
            horizon=config.N,
            size=config.n,
            dtype=X_in.dtype,
            name="Q_bar",
        )
        self.R_bar = stage_weights(
            R_bar,
            horizon=config.N,
            size=config.nu,
            dtype=U_in.dtype,
            name="R_bar",
        )
        if Q_f_bar is None:
            Q_f_bar = jnp.eye(config.n, dtype=X_in.dtype)
        self.Q_f_bar = jnp.asarray(Q_f_bar, dtype=X_in.dtype)
        expected_terminal_shape = (config.n, config.n)
        if self.Q_f_bar.shape != expected_terminal_shape:
            raise ValueError(
                "Q_f_bar must have shape "
                f"{expected_terminal_shape}; got {self.Q_f_bar.shape}."
            )

        num_obstacles = self.obstacles.shape[0]
        total_num_constraints = num_constraints + num_terminal_constraints
        self.h_ct_ws = jnp.zeros((config.N + 1, num_constraints - num_obstacles))
        self.beta_ws = jnp.ones((config.N + 1, config.N + 1, num_constraints - num_obstacles)) * 1e-10
        self.mu_ws = jnp.zeros((config.N + 1, total_num_constraints))
        self.Phi_x_ws = jnp.zeros((config.N + 1, config.N + 1, config.n, config.n))
        self.Phi_u_ws = jnp.zeros((config.N, config.N + 1, config.nu, config.n))
        
        if sls_config.adaptive:
            dist_size = (sls_config.n_w + sls_config.n_x) + 2*(sls_config.n_theta*sls_config.q_max)
        else:
            dist_size = (sls_config.n_w + config.n) + (sls_config.num_param*sls_config.q_max)

        self.E_prev = jnp.zeros((config.N + 1, config.n, dist_size))
        self.K_prev = jnp.zeros((config.N, sls_config.num_param, sls_config.num_param))
        self.L_prev = jnp.zeros_like(self.K_prev)
        self.G_prev = jnp.zeros((config.N, sls_config.num_param, sls_config.num_param*sls_config.q_max)) 
        self._has_generator_history = False
        self.P_inv_prev = jnp.zeros((config.N, sls_config.num_param, sls_config.n_x))

        self.U0 = U_in
        self.X0 = X_in
        self.V0 = jnp.zeros((config.N + 1, config.n))
        self.w = jnp.zeros((config.N + 1, total_num_constraints))
        self.y = jnp.zeros((config.N + 1, total_num_constraints))
        self.rho = jnp.asarray(self.admm_config.initial_rho, dtype=self.w.dtype)

        self.dynamics = dynamics
        self.constraints = solver_constraints
        self.cost = cost
        self.disturbance = disturbance

        f_flat, D_flat = pack_dynamics_as_single_input(
            dynamics,
            nx=config.n,
            nu=config.nu,
            parameter=config.dt,
            t_dim=1,
            t_as_scalar=True,
            adaptive=sls_config.adaptive,
            ntheta=sls_config.num_param if not sls_config.adaptive else None
        )

        # 1. Define the base linearization error function
        base_remainder_func = partial(remainder_bound_path_based, f_flat, state_dim=config.n)
        
        # 2. Vmap the base function over the time dimension
        vmap_base_remainder = jax.vmap(base_remainder_func, in_axes=(0, 0))

        # ---------------------------------------------------------
        # AUTOMATIC JAXPR TRACING FOR DISTURBANCE FUNCTIONS
        # ---------------------------------------------------------
        if (
            sls_config.enable_linearization_error
            and sls_config.adaptive
            and sls_config.enable_disturbance_variation_error
        ):
            def flat_dist(x):
                # Remove [0] to bypass the multi-dimensional slice. 
                # (1, nx, nx) flattens directly to (nx*nx,)
                return self.disturbance(x[None, :]).flatten()
            self.iv_dist_evaluator = build_rhs_interval_from_prog(_trace_prog(flat_dist, D=config.n))
        elif (
            sls_config.enable_linearization_error
            and not sls_config.adaptive
            and sls_config.enable_disturbance_variation_error
        ):
            param_dist_fn, sens_fn, exog_dist_fn = self.disturbance
            
            def flat_sens(z_flat):
                x_eval = z_flat[:config.n]
                u_eval = z_flat[config.n : config.n + config.nu]
                nom_p  = z_flat[config.n + config.nu :]
                sens = sens_fn(x_eval[None, :], u_eval[None, :], nom_p)
                # Remove [0] here as well
                return sens.flatten()

            self.iv_sens_evaluator = build_rhs_interval_from_prog(_trace_prog(flat_sens, D=config.n + config.nu + sls_config.num_param))

            def flat_exog(x):
                # Remove [0] here as well
                return exog_dist_fn(x[None, :]).flatten()
            self.iv_exog_evaluator = build_rhs_interval_from_prog(_trace_prog(flat_exog, D=config.n))

        # ---------------------------------------------------------
        # 3. CREATE THE UNIFIED TOTAL RESIDUAL BOUNDER
        # ---------------------------------------------------------
        def total_residual_bound(X_nom, U_nom, tube_x, tube_u, G_0, nominal_param):
            T = U_nom.shape[0]

            if not self.sls_config.enable_linearization_error:
                return jnp.zeros_like(X_nom)
            
            # --- A. Taylor Linearization Error ---
            # U_nom is still length T, so the nominal control trajectory DOES need padding
            U_pad = jnp.concatenate([U_nom, U_nom[-1:]], axis=0)
            t_arr = jnp.arange(T + 1, dtype=X_nom.dtype)[:, None]
            t_width = jnp.zeros_like(t_arr)

            # Concatenate to match the D_flat expected by the JAXPR tracer
            z_center = jnp.concatenate([X_nom, U_pad, t_arr], axis=-1)
            
            # tube_u is now properly passed in as length T+1!
            z_width  = jnp.concatenate([tube_x, tube_u, t_width], axis=-1)
            
            z_lo = z_center - z_width
            z_up = z_center + z_width
            
            # The base remainder function returns a single array of bounds.
            r_lin_mag = jnp.abs(vmap_base_remainder(z_lo, z_up))
            
            # --- B. Disturbance Variation Error ---
            if not self.sls_config.enable_disturbance_variation_error:
                E_nom = self.disturbance(X_nom) if self.sls_config.adaptive else self.disturbance[0](X_nom, U_nom, nominal_param, G_0)
                r_dist_mag = jnp.zeros_like(r_lin_mag)

            elif self.sls_config.adaptive:
                E_nom = self.disturbance(X_nom)
                
                X_iv = Interval(X_nom - tube_x, X_nom + tube_x)

                # MUST use jax.vmap here so JAX handles the time-batch dimension natively!
                E_iv_flat = jax.vmap(self.iv_dist_evaluator)(X_iv)
                
                nx = self.sls_config.n_x
                E_iv_lo = E_iv_flat.lo.reshape((T + 1, nx, nx))
                E_iv_hi = E_iv_flat.hi.reshape((T + 1, nx, nx))
                
            else:
                param_dist_fn, sens_fn, exog_dist_fn = self.disturbance
                E_nom = param_dist_fn(X_nom, U_nom, nominal_param, G_0)
                
                nx = self.sls_config.n_x
                
                # 1. Evaluate ONLY the Exogenous Interval using the Physical State
                # X_nom currently has shape (T+1, nx + num_params) because of X_temp concatenation
                # We MUST slice it to only pass the physical states to the exogenous evaluator!
                X_nom_phys = X_nom[..., :nx]
                tube_x_phys = tube_x[..., :nx]
                
                X_iv_phys = Interval(X_nom_phys - tube_x_phys, X_nom_phys + tube_x_phys)
                
                # VMAP the exogenous evaluator using the cleanly sliced shape
                exog_iv_flat = jax.vmap(self.iv_exog_evaluator)(X_iv_phys)
                exog_iv_lo = exog_iv_flat.lo.reshape((T + 1, nx, nx))
                exog_iv_hi = exog_iv_flat.hi.reshape((T + 1, nx, nx))
                
                # 2. Extract the nominal parameter sensitivity block from E_nom
                # E_nom shape is (T+1, nx, nx + num_param), so we slice everything after nx
                param_nom = E_nom[..., nx:]
                
                # 3. Concatenate to match E_nom's shape exactly
                # Because param_nom is identical to the block in E_nom,
                # (E_iv_lo - E_nom) will be exactly 0 for the parameter dimensions!
                E_iv_lo = jnp.concatenate([exog_iv_lo, param_nom], axis=-1)
                E_iv_hi = jnp.concatenate([exog_iv_hi, param_nom], axis=-1)
                
            if self.sls_config.enable_disturbance_variation_error:
                # C. Compute max deviation from nominal
                diff_lo = jnp.abs(E_iv_lo - E_nom)
                diff_hi = jnp.abs(E_iv_hi - E_nom)
                max_E_diff = jnp.maximum(diff_lo, diff_hi)
                
                # D. Scale by L-infinity unit ball
                r_dist_mag = jnp.sum(max_E_diff, axis=-1)
                
                # Pad adaptive case to match r_lin_mag state dimensions
                if self.sls_config.adaptive and self.sls_config.n_theta > 0:
                    r_dist_mag = jnp.concatenate([r_dist_mag, jnp.zeros((T + 1, self.sls_config.n_theta))], axis=-1)
            
            # E. Combine the absolute magnitude bounds
            return r_lin_mag + r_dist_mag

        # 4. Pass the unified bounder to the solver
        splts_cfg = (4, 4, 4, 4)
        work = partial(
            sqp.mpc,
            self.sls_config, self.sqp_config, self.admm_config,
            cost, dynamics,
            hessian_approx,
            limited_memory,
            solver_constraints, disturbance,
            total_residual_bound, splts_cfg,
            self.measurement_matrix,
            Q_bar=self.Q_bar,
            R_bar=self.R_bar,
            Q_f_bar=self.Q_f_bar,
        )

        self._solve = jax.jit(work)

        @jax.jit
        def update_and_extract(U, X, V, x0, X_tail):
            def safe():
                s = self.shift
                new_U0 = jnp.concatenate([U[s:], jnp.tile(U[-1:], (s, 1))], axis=0)
                new_X0 = jnp.concatenate([X[s:], jnp.tile(X_tail[None, :], (s, 1))], axis=0)
                new_V0 = jnp.concatenate([V[s:], jnp.tile(V[-1:], (s, 1))], axis=0)
                return new_U0, new_X0, new_V0

            def unsafe():
                new_U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
                new_X0 = jnp.tile(x0, (self.config.N + 1, 1))
                new_V0 = jnp.zeros((self.config.N + 1, self.config.n))
                return new_U0, new_X0, new_V0

            return jax.lax.cond(jnp.isnan(U[0, 0]), unsafe, safe)

        self._update_and_extract = update_and_extract

        if fuse_rti_updates:
            @jax.jit
            def fused_run(
                reference, parameter, x0, X_tail, G_0, nominal_param,
                X0, U0, V0, w, y, rho, h_ct_ws, beta_ws, mu_ws,
                Phi_x_ws, Phi_u_ws, E_prev, K_prev, G_prev, P_inv_prev,
            ):
                (
                    X, U, V, w_next, y_next, rho_next, backoffs,
                    Phi_x, Phi_u, K_kjN, betaN, muN, EN, KN, LN, GN, P_inv,
                    _sls_converged, _sls_residual, _admm_converged,
                    _admm_iterations, _admm_primal_residual,
                    _admm_dual_residual, _admm_primal_tolerance,
                    _admm_dual_tolerance,
                    _admm_primal_worst_flat_index, _admm_primal_worst_z,
                    _admm_primal_worst_w,
                ) = self._solve(
                    reference, parameter, self.config.W,
                    x0, X0, U0, V0, w, y, rho,
                    self.obstacles, h_ct_ws, beta_ws, mu_ws, Phi_x_ws,
                    Phi_u_ws, E_prev, K_prev, G_prev, G_0, P_inv_prev,
                    nominal_param,
                )
                G_solved = GN
                if self.sls_config.enable_information_cost:
                    G_next = jnp.concatenate(
                        [GN[self.shift:], jnp.tile(GN[-1:], (self.shift, 1, 1))],
                        axis=0,
                    )
                else:
                    G_next = GN
                h_ct_next = jnp.concatenate(
                    [backoffs[self.shift:], jnp.tile(backoffs[-1:], (self.shift, 1))],
                    axis=0,
                )
                beta_next = jnp.concatenate(
                    [betaN[self.shift:], jnp.tile(betaN[-1:], (self.shift, 1))],
                    axis=0,
                )
                beta_next = jnp.concatenate(
                    [
                        beta_next[:, self.shift:],
                        jnp.tile(beta_next[:, -1:], (1, self.shift, 1)),
                    ],
                    axis=1,
                )
                mu_next = jnp.concatenate(
                    [muN[self.shift:], jnp.tile(muN[-1:], (self.shift, 1))],
                    axis=0,
                )
                w_next = jnp.concatenate(
                    [w_next[self.shift:], jnp.tile(w_next[-1:], (self.shift, 1))],
                    axis=0,
                )
                y_next = jnp.concatenate(
                    [y_next[self.shift:], jnp.tile(y_next[-1:], (self.shift, 1))],
                    axis=0,
                )
                U0_next, X0_next, V0_next = self._update_and_extract(
                    U, X, V, x0, X_tail
                )
                return (
                    U[0], X, U, V, backoffs, Phi_x, Phi_u, K_kjN, G_solved,
                    X0_next, U0_next, V0_next, w_next, y_next, rho_next,
                    h_ct_next, beta_next, mu_next, EN, KN, LN, G_next, P_inv,
                )

            self._fused_run = fused_run

    def run(
        self,
        x0: jnp.ndarray,
        G_0: jnp.ndarray,
        reference: jnp.ndarray,
        nominal_param: jnp.ndarray,
        parameter: Any,
        X_tail: jnp.ndarray | None = None,
    ):
        if (
            self.sls_config.enable_information_cost
            and not self._has_generator_history
        ):
            G_seed = girard_reduction_opt(G_0, self.sls_config.q_max)
            self.G_prev = jnp.broadcast_to(G_seed, self.G_prev.shape)

        if self.fuse_rti_updates:
            if X_tail is None:
                raise ValueError("fuse_rti_updates requires an explicit X_tail")
            values = self._fused_run(
                reference, parameter, x0, X_tail, G_0, nominal_param,
                self.X0, self.U0, self.V0, self.w, self.y, self.rho,
                self.h_ct_ws, self.beta_ws, self.mu_ws, self.Phi_x_ws,
                self.Phi_u_ws, self.E_prev, self.K_prev, self.G_prev,
                self.P_inv_prev,
            )
            (
                u0, X, U, V, backoffs, Phi_x, Phi_u, K_kjN, G_solved,
                self.X0, self.U0, self.V0, self.w, self.y, self.rho,
                self.h_ct_ws, self.beta_ws, self.mu_ws, self.E_prev,
                self.K_prev, self.L_prev, self.G_prev, self.P_inv_prev,
            ) = values
            self.Phi_u_ws = Phi_u
            self.Phi_x_ws = Phi_x
            self._has_generator_history = True
            return (
                u0, X, U, V, backoffs, Phi_x, Phi_u, K_kjN,
                self.E_prev, self.K_prev, G_solved, self.P_inv_prev,
            )

        (
            X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, K_kjN,
            betaN, muN, EN, KN, LN, GN, P_inv, _sls_converged,
            _sls_residual, _admm_converged, _admm_iterations,
            _admm_primal_residual, _admm_dual_residual,
            _admm_primal_tolerance, _admm_dual_tolerance,
            _admm_primal_worst_flat_index, _admm_primal_worst_z,
            _admm_primal_worst_w,
        ) = self._solve(
            reference,
            parameter,
            self.config.W,
            x0, self.X0, self.U0, self.V0,
            self.w, self.y, self.rho,
            self.obstacles,
            self.h_ct_ws, self.beta_ws, self.mu_ws, self.Phi_x_ws, self.Phi_u_ws, self.E_prev, self.K_prev, self.G_prev, G_0, self.P_inv_prev, nominal_param
        )
        self.E_prev = EN
        self.K_prev = KN 
        self.L_prev = LN
        # Preserve the unshifted sequence for this solve's return value.  When
        # active information gathering is enabled, the next MPC call evaluates
        # its shifted warm-start trajectory against the correspondingly shifted
        # lagged generator sequence.
        G_solved = GN
        if self.sls_config.enable_information_cost:
            self.G_prev = jnp.concatenate(
                [
                    GN[self.shift:],
                    jnp.tile(GN[-1:], (self.shift, 1, 1)),
                ],
                axis=0,
            )
        else:
            self.G_prev = GN
        self._has_generator_history = True
        self.P_inv_prev = P_inv
        self.h_ct_ws = jnp.concatenate(
            [backoffs[self.shift:], jnp.tile(backoffs[-1:], (self.shift, 1))],
            axis=0
        )
        self.beta_ws = jnp.concatenate(
            [betaN[self.shift:], jnp.tile(betaN[-1:], (self.shift, 1))],
            axis=0
        )
        self.beta_ws = jnp.concatenate(
            [
                self.beta_ws[:, self.shift:],
                jnp.tile(self.beta_ws[:, -1:], (1, self.shift, 1)),
            ],
            axis=1,
        )
        self.mu_ws = jnp.concatenate(
            [muN[self.shift:], jnp.tile(muN[-1:], (self.shift, 1))],
            axis=0
        )

        self.w = jnp.concatenate([w[self.shift:], jnp.tile(w[-1:], (self.shift, 1))], axis=0)
        self.y = jnp.concatenate([y[self.shift:], jnp.tile(y[-1:], (self.shift, 1))], axis=0)

        self.rho = jnp.asarray(rho, dtype=self.rho.dtype)

        if X_tail is None:
            X_tail = X[-1]
        self.U0, self.X0, self.V0 = self._update_and_extract(U, X, V, x0, X_tail)
        self.Phi_u_ws = Phi_u
        self.Phi_x_ws = Phi_x
        return U[0], X, U, V, backoffs, Phi_x, Phi_u, K_kjN, self.E_prev, self.K_prev, G_solved, self.P_inv_prev

    def block_until_ready(self) -> None:
        """Wait for queued controller workspace work before a timing boundary."""
        for value in (
            self.X0, self.U0, self.V0, self.w, self.y, self.h_ct_ws,
            self.beta_ws, self.mu_ws, self.Phi_x_ws, self.Phi_u_ws,
            self.E_prev, self.K_prev, self.L_prev, self.G_prev, self.P_inv_prev,
        ):
            value.block_until_ready()
