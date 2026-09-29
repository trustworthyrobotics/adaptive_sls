from jax import debug, grad, jit, lax, scipy, vmap
import jax
import jax.numpy as np

from functools import partial

from trajax.optimizers import evaluate, linearize, quadratize,vectorize

from .kkt_helpers import compute_search_direction_kkt, tvlqr_kkt

from .dual_tvlqr import dual_lqr, dual_lqr_backward, dual_lqr_gpu,dual_lqr_backward_constrained

from .linalg_helpers import (
    invert_symmetric_positive_definite_matrix,
    project_psd_cone,
)

from .primal_tvlqr import (
    tvlqr,
    tvlqr_gpu,
    rollout,
    rollout_gpu,
    non_linear_rollout,
    rollout_controls,
    direct_optimal_control_rollout_with_defects,
    tvlqr_gpu_constrained,
    rollout_gpu_constrained,
    tvlqr_constrained,
    rollout_constrained,
)
import time

@jit
def regularize(Q, R, M, make_psd, psd_delta):
    """Regularizes the Q and R matrices.

    Args:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
      M:             [T+1, n, m]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
    """
    T, n, m = M.shape
    psd = vmap(partial(project_psd_cone, delta=psd_delta))

    # This is done to ensure that the R are positive definite.
    R = lax.cond(make_psd, psd, lambda x: x, R)

    # This is done to ensure that the Q - M R^(-1) M^T are positive semi-definite.
    Rinv = vmap(lambda t: invert_symmetric_positive_definite_matrix(R[t]))(np.arange(T))
    MRinvMT = vmap(lambda t: M[t] @ Rinv[t] @ M[t].T)(np.arange(T))
    QMRinvMT = vmap(lambda t: Q[t] - MRinvMT[t])(np.arange(T))
    QMRinvMT = lax.cond(make_psd, psd, lambda x: x, QMRinvMT)
    Q_T = Q[T].reshape([1, n, n])
    Q_T = lax.cond(make_psd, psd, lambda x: x, Q_T)
    Q = np.concatenate([QMRinvMT + MRinvMT, Q_T])

    return Q, R

def linearize_scan(fun, argnums=3):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically np.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u, t, *args):
        inputs = (x, u, t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def linearize_obj_scan(fun, argnums=5):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically np.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u,  v, v1, t, *args):
        inputs = (x, u, v, v1,t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate the associated Lagrangian."""

    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = np.dot(v, dynamics(x, u, t))
        c3 = np.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun


@partial(jit, static_argnums=(0, 1, 2, 3))
def compute_search_direction(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    x0,
    X,
    U,
    V,
    c,
    regularization = 1e-3
):
    """Computes the SQP search direction.

    Args:
      cost:          cost function with signature cost(x, u, t).
      dynamics:      dynamics function with signature dynamics(x, u, t).
      x0:            [n]           numpy array.
      X:             [T+1, n]      numpy array.
      U:             [T, m]        numpy array.
      V:             [T+1, n]      numpy array.
      c:             [T+1, n]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      q: [T+1, n]  numpy array.
      r: [T, m]    numpy array.
    """
    T = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), np.arange(T + 1))

    M = M_pad[:-1]

    # Q, R = regularize(Q, R, M, make_psd=True, psd_delta=1e-2)
    Q = Q + regularization * np.eye(n)[None, :, :]
    R = R_pad[:-1] + regularization * np.eye(m)[None, :, :]

    linearizer = linearize(lagrangian(cost, dynamics, x0),argnums = 5)
    dynamics_linearizer = linearize(dynamics)

    q, r_pad = linearizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]

    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    if limited_memory:
        K, k, P, p = tvlqr(Q, q, R, r, M, A, B, c[1:])
        dX, dU = rollout(K, k, c[0], A, B, c[1:])
    else:
        K, k, P, p = tvlqr_gpu(Q, q, R, r, M, A, B, c[1:])
        dX, dU = rollout_gpu(K, k, c[0], A, B, c[1:])
    
    dV = dual_lqr(dX, P, p)

    return dX, dU, dV, q, r


@partial(jit, static_argnums=(13,))
def _solve_equality_lqr(
    Q, q, R, r, M, A, B, c, hx, hu, h, regularization, dX0, limited_memory
):
    """Solves a pre-linearized equality-constrained LQ subproblem."""
    n = Q.shape[1]
    m = R.shape[1]
    Q = Q + regularization * np.eye(n)[None, :, :]
    R = R + regularization * np.eye(m)[None, :, :]
    if limited_memory:
        K, k, P, p, K_eq, k_eq = tvlqr_constrained(
            Q, q, R, r, M, A, B, c, hx, hu, h
        )
        dX, dU, dVeq = rollout_constrained(
            K, k, dX0, A, B, c, K_eq, k_eq
        )
    else:
        K, k, P, p, K_eq, k_eq = tvlqr_gpu_constrained(
            Q, q, R, r, M, A, B, c, hx, hu, h
        )
        dX, dU, dVeq = rollout_gpu_constrained(
            K, k, dX0, A, B, c, K_eq, k_eq
        )
    dV = dual_lqr_backward_constrained(Q, q, M, A, hx, dX, dU, dVeq)
    return dX, dU, dV, dVeq


@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def compute_equality_search_direction(
    cost,
    dynamics,
    equality,
    hessian_approx,
    limited_memory,
    x0,
    X,
    U,
    V,
    Veq,
    c,
    h,
    regularization=1e-6,
):
    """Computes the SQP direction for dynamics plus stage equalities.

    The stage equality is linearized as
      equality(X[t], U[t], t) + hx[t] dX[t] + hu[t] dU[t] = 0.
    """

    T = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), np.arange(T + 1))

    M = M_pad[:-1]
    R = R_pad[:-1]

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    equality_linearizer = linearize(equality)
    hx_pad, hu_pad = equality_linearizer(X, pad(U), np.arange(T + 1))
    hx = hx_pad[:-1]
    hu = hu_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    q, r_pad = linearizer(
        X,
        pad(U),
        np.arange(T + 1),
        pad(V[1:]),
        V,
    )
    r = r_pad[:-1]

    eq_q = vmap(lambda hxt, veqt: hxt.T @ veqt)(hx, Veq)
    eq_r = vmap(lambda hut, veqt: hut.T @ veqt)(hu, Veq)
    q = q.at[:-1].add(eq_q)
    r = r + eq_r

    dX, dU, dV, dVeq = _solve_equality_lqr(
        Q,
        q,
        R,
        r,
        M,
        A,
        B,
        c[1:],
        hx,
        hu,
        h,
        regularization,
        c[0],
        limited_memory,
    )

    return dX, dU, dV, dVeq, q, r


@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def compute_equality_fddp_search_direction(
    cost,
    dynamics,
    equality,
    hessian_approx,
    limited_memory,
    x0,
    X,
    U,
    V,
    Veq,
    defects,
    h,
    regularization=1e-6,
):
    """Computes the equality-constrained FDDP search direction.

    This is the same constrained Riccati direction as ``mpc_equality`` but it
    exposes the feedback policy and value-function derivatives needed by the
    feasibility-driven nonlinear rollout and Goldstein merit test.
    """

    T = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), np.arange(T + 1))

    M = M_pad[:-1]
    Q = Q + regularization * np.eye(n)[None, :, :]
    R = R_pad[:-1] + regularization * np.eye(m)[None, :, :]

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    equality_linearizer = linearize(equality)
    hx_pad, hu_pad = equality_linearizer(X, pad(U), np.arange(T + 1))
    hx = hx_pad[:-1]
    hu = hu_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    q, r_pad = linearizer(
        X,
        pad(U),
        np.arange(T + 1),
        pad(V[1:]),
        V,
    )
    r = r_pad[:-1]

    eq_q = vmap(lambda hxt, veqt: hxt.T @ veqt)(hx, Veq)
    eq_r = vmap(lambda hut, veqt: hut.T @ veqt)(hu, Veq)
    q = q.at[:-1].add(eq_q)
    r = r + eq_r

    if limited_memory:
        K, k, P, p, K_eq, k_eq = tvlqr_constrained(
            Q, q, R, r, M, A, B, defects[1:], hx, hu, h
        )
        dX, dU, dVeq = rollout_constrained(
            K, k, defects[0], A, B, defects[1:], K_eq, k_eq
        )
    else:
        K, k, P, p, K_eq, k_eq = tvlqr_gpu_constrained(
            Q, q, R, r, M, A, B, defects[1:], hx, hu, h
        )
        dX, dU, dVeq = rollout_gpu_constrained(
            K, k, defects[0], A, B, defects[1:], K_eq, k_eq
        )

    dV = dual_lqr_backward_constrained(Q, q, M, A, hx, dX, dU, dVeq)

    return K, k, dX, dU, dV, dVeq, q, r, R, B, P, p
@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used.

    Args:
      c:             [T+1, n]  numpy array.
      dV:            [T+1, n]  numpy array.

    Returns:
        rho: the penalty parameter.
    """
    c2 = np.sum(c * c)
    dV2 = np.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * np.sqrt(dV2 / c2), 1e-2)


@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Determines the directional derivative of the merit function.

    Args:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      c:  [T+1, n] numpy array.
      q:  [T+1, n] numpy array.
      r:  [T, m] numpy array.
      rho: the penalty parameter of the merit function.

    Returns:
        dir_derivative: the directional derivative.
    """
    return np.sum(q * dX) + np.sum(r * dU) + 2*np.sum(dV * c) - rho * np.sum(c * c)

@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    alpha_0,
    alpha_mult,
    alpha_min,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        # debug.print(f"{new_merit=}, {current_merit=}, {alpha=}, {merit_slope=}")\
        return np.logical_and(
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
        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, np.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min


    return X, U, V, new_g, new_c, no_errors

@partial(jit, static_argnums=(0, 1))
def parallel_line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,

):
    """Performs a primal-dual line search on an augmented Lagrangian merit function in parralel fixing the number of steps.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """
    def step_acceptance(merit,alpha):
        return merit > current_merit + alpha * armijo_factor * merit_slope
    alpha_values = np.exp2(-np.arange(11))
    def body(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new) #this cam br probly avoided
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit

    X, U, V, new_g, new_c, new_merit = vmap(body)(alpha_values)
    acceptance = vmap(step_acceptance)(new_merit,alpha_values)
    any_accepted = np.any(acceptance)
    best_index = np.where(any_accepted,np.argmin(acceptance),0)
    X_best = np.where(any_accepted, X[best_index], X_in)
    U_best = np.where(any_accepted, U[best_index], U_in)
    V_best = np.where(any_accepted, V[best_index], V_in)
    g_best = np.where(any_accepted, new_g[best_index], current_g)
    c_best = np.where(any_accepted, new_c[best_index], current_c)
    return X_best, U_best, V_best, g_best, c_best

@partial(jit, static_argnums=(0))
def filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    alpha_min=1e-4,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
    gamma_alpha=0.5,
):
    """Performs a backtracking line search.

    Args:
      X_in: [T+1, n] numpy array of current states.
      U_in: [T, m] numpy array of current controls.
      V_in: [T+1, n] numpy array of current multipliers.
      dX: [T+1, n] numpy array of state search direction.
      dU: [T, m] numpy array of control search direction.
      dV: [T+1, n] numpy array of multiplier search direction.
      current_cost: Current cost value (phi_k).
      current_c: Current constraint residuals used to evaluate theta_k.
      q: Objective gradient with respect to the states.
      r: Objective gradient with respect to the controls.
      alpha_min: Minimum step size.
      theta_max: Maximum acceptable constraint violation.
      theta_min: Minimum constraint violation worth considering.
      eta: Armijo parameter for sufficient decrease.
      gamma_phi: Cost reduction parameter.
      gamma_theta: Constraint violation reduction parameter.
      gamma_alpha: Step size reduction factor.

    Returns:
      X: [T+1, n] numpy array of updated states.
      U: [T, m] numpy array of updated controls.
      V: [T+1, n] numpy array of updated multipliers.
      new_cost: Updated cost value.
      new_c: Updated constraint values.
      accepted: Whether a step was accepted.
    """
    # Initial values
    alpha = 1.0
    theta_k =  np.sqrt(np.sum(current_c * current_c))
    phi_k = current_cost
    slope = np.sum(q * dX) + np.sum(r * dU)

    def continuation_criterion(inputs):
        _, _, _, alpha, accepted = inputs
        return np.logical_and(np.logical_not(accepted), alpha >= alpha_min)

    def body(inputs):
        _, _, _, alpha, _ = inputs

        # Compute trial point
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV

        # Evaluate at new point
        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new =  np.sqrt(np.sum(new_c * new_c))
        phi_new = new_cost
        finite = np.logical_and(np.all(np.isfinite(X_new)), np.all(np.isfinite(U_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(V_new)))
        finite = np.logical_and(finite, np.isfinite(phi_new))
        finite = np.logical_and(finite, np.all(np.isfinite(new_c)))

        # Case 1: Large constraint violation but improving
        condition1 = theta_new > theta_max
        case1 = np.logical_and(
            condition1,
            theta_new < (1 - gamma_theta) * theta_k
        )

        # Case 2: Small constraint violations and cost is decreasing
        condition2 =  np.logical_and(
                np.maximum(theta_new, theta_k) < theta_min,
                slope < 0
            )
        case2 = np.logical_and(
           condition2,
            phi_new < phi_k + eta * alpha * slope
        )

        # Case 3: Either cost or constraint violation is significantly reduced
        condition3 = np.logical_not(np.logical_or(condition1, condition2))
        case3 = np.logical_and(condition3, np.logical_or(
            phi_new < phi_k - gamma_phi * theta_k,
            theta_new < (1 - gamma_theta) * theta_k
        ))

        # Accept if any case is satisfied
        new_accepted = np.logical_and(
            finite, np.logical_or(np.logical_or(case1, case2), case3)
        )

        # If not accepted, reduce alpha
        alpha = np.where(new_accepted, alpha, gamma_alpha * alpha)

        return X_new, U_new, V_new, alpha, new_accepted

    # Run the backtracking loop
    X, U, V, alpha, accepted = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, alpha, False)
    )

    X = np.where(accepted, X, X_in)
    U = np.where(accepted, U, U_in)
    V = np.where(accepted, V, V_in)

    return X, U, V
@partial(jit, static_argnums=(0,11))
def parallel_filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    num_alpha=11,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
    gamma_alpha=0.5,
):
    """Performs a backtracking line search.

    Args:
      X_in: [T+1, n] numpy array of current states.
      U_in: [T, m] numpy array of current controls.
      V_in: [T+1, n] numpy array of current multipliers.
      dX: [T+1, n] numpy array of state search direction.
      dU: [T, m] numpy array of control search direction.
      dV: [T+1, n] numpy array of multiplier search direction.
      current_cost: Current cost value (phi_k).
      current_c: Current constraint violation (theta_k).
      alpha_min: Minimum step size.
      theta_max: Maximum acceptable constraint violation.
      theta_min: Minimum constraint violation worth considering.
      eta: Armijo parameter for sufficient decrease.
      gamma_phi: Cost reduction parameter.
      gamma_theta: Constraint violation reduction parameter.
      gamma_alpha: Step size reduction factor.

    Returns:
      X: [T+1, n] numpy array of updated states.
      U: [T, m] numpy array of updated controls.
      V: [T+1, n] numpy array of updated multipliers.
      new_cost: Updated cost value.
      new_c: Updated constraint values.
      accepted: Whether a step was accepted.
    """
    # Initial values
    alpha_values = np.exp2(-np.arange(num_alpha, dtype=X_in.dtype))
    theta_k =  np.sqrt(np.sum(current_c * current_c))
    phi_k = current_cost
    slope = np.sum(q * dX) + np.sum(r * dU)

    def body(alpha):
        # _, _, _, alpha, _ = inputs

        # Compute trial point
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV

        # Evaluate at new point
        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new =  np.sqrt(np.sum(new_c * new_c))
        phi_new = new_cost
        finite = np.logical_and(np.all(np.isfinite(X_new)), np.all(np.isfinite(U_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(V_new)))
        finite = np.logical_and(finite, np.isfinite(phi_new))
        finite = np.logical_and(finite, np.all(np.isfinite(new_c)))

        # Case 1: Large constraint violation but improving
        condition1 = theta_new > theta_max
        case1 = np.logical_and(
            condition1,
            theta_new < (1 - gamma_theta) * theta_k
        )

        # Case 2: Small constraint violations and cost is decreasing
        condition2 =  np.logical_and(
                np.maximum(theta_new, theta_k) < theta_min,
                slope < 0
            )
        case2 = np.logical_and(
           condition2,
            phi_new < phi_k + eta * alpha * slope
        )

        # Case 3: Either cost or constraint violation is significantly reduced
        condition3 = np.logical_not(np.logical_or(condition1, condition2))
        case3 = np.logical_and(condition3,np.logical_or(
            phi_new < phi_k - gamma_phi * phi_k,
            theta_new < (1 - gamma_theta) * theta_k
        ))

        # Accept if any case is satisfied
        new_accepted = np.logical_and(
            finite, np.logical_or(np.logical_or(case1, case2), case3)
        )

        return X_new, U_new, V_new, new_accepted

    # Run the backtracking loop
    X, U, V,accepted = vmap(body)(alpha_values)
    best_index = np.where(np.any(accepted), np.argmax(accepted), -1)

    X_new = X[best_index]
    U_new = U[best_index]
    V_new = V[best_index]
    
    return X_new, U_new, V_new


@partial(jit, static_argnums=(0, 13))
def parallel_filter_line_search_equality(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    Veq_in,
    dX,
    dU,
    dV,
    dVeq,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    num_alpha=11,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
):
    """Parallel filter line search for dynamics and stage equalities."""

    alpha_values = np.exp2(-np.arange(num_alpha, dtype=X_in.dtype))
    theta_k = np.sqrt(np.sum(current_c * current_c))
    phi_k = current_cost
    slope = np.sum(q * dX) + np.sum(r * dU)

    def body(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        Veq_new = Veq_in + alpha * dVeq

        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new = np.sqrt(np.sum(new_c * new_c))
        phi_new = new_cost
        finite = np.logical_and(np.all(np.isfinite(X_new)), np.all(np.isfinite(U_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(V_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(Veq_new)))
        finite = np.logical_and(finite, np.isfinite(phi_new))
        finite = np.logical_and(finite, np.all(np.isfinite(new_c)))

        condition1 = theta_new > theta_max
        case1 = np.logical_and(
            condition1,
            theta_new < (1 - gamma_theta) * theta_k,
        )

        condition2 = np.logical_and(
            np.maximum(theta_new, theta_k) < theta_min,
            slope < 0,
        )
        case2 = np.logical_and(
            condition2,
            phi_new < phi_k + eta * alpha * slope,
        )

        condition3 = np.logical_not(np.logical_or(condition1, condition2))
        case3 = np.logical_and(
            condition3,
            np.logical_or(
                phi_new < phi_k - gamma_phi * phi_k,
                theta_new < (1 - gamma_theta) * theta_k,
            ),
        )

        accepted = np.logical_and(
            finite, np.logical_or(np.logical_or(case1, case2), case3)
        )

        return X_new, U_new, V_new, Veq_new, accepted

    X, U, V, Veq, accepted = vmap(body)(alpha_values)
    best_index = np.where(np.any(accepted), np.argmax(accepted), -1)

    return X[best_index], U[best_index], V[best_index], Veq[best_index]

@partial(jit, static_argnums=(0, 13))
def parallel_l1_merit_line_search_equality(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    Veq_in,
    dX,
    dU,
    dV,
    dVeq,
    current_cost,
    current_c,
    q,
    r,
    num_alpha=11,
    armijo=1e-4,
    gamma_min=1.0,
    gamma_mult=1.5,
    eps=1e-12,
    c_scale=1.0,
):
    """Parallel exact-L1 merit line search.

    model_evaluator(X, U) should return:
      cost, constraint_vector

    constraint_vector should include all defects/equalities in one flat or
    broadcast-compatible array.
    """

    alpha_values = np.exp2(-np.arange(num_alpha, dtype=X_in.dtype))

    c0 = current_c * c_scale
    c0_l1 = np.sum(np.abs(c0))

    # Use primal-dual multipliers to set the exact-L1 penalty.
    lambda_inf = np.maximum(np.max(np.abs(V_in)), np.max(np.abs(Veq_in)))
    gamma = np.maximum(gamma_min, gamma_mult * lambda_inf)

    # If the objective directional derivative is positive, gamma must be
    # large enough for the merit directional derivative to be negative,
    # assuming the SQP direction approximately satisfies C + C_z dz = 0.
    gradJ_d = np.sum(q * dX) + np.sum(r * dU)
    gamma_needed = np.where(c0_l1 > eps, (gradJ_d + 1.0) / (c0_l1 + eps), gamma)
    gamma = np.maximum(gamma, gamma_needed)

    merit0 = current_cost + gamma * c0_l1

    # If the SQP direction satisfies the linearized constraints,
    # C_z dz ≈ -C, then the L1 merit directional derivative is:
    merit_slope = np.where(c0_l1 > eps, gradJ_d - gamma * c0_l1, gradJ_d)

    def body(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        Veq_new = Veq_in + alpha * dVeq

        cost_new, c_new_raw = model_evaluator(X_new, U_new)
        c_new = c_new_raw * c_scale
        merit_new = cost_new + gamma * np.sum(np.abs(c_new))

        finite = (
            np.all(np.isfinite(X_new))
            & np.all(np.isfinite(U_new))
            & np.all(np.isfinite(V_new))
            & np.all(np.isfinite(Veq_new))
            & np.isfinite(cost_new)
            & np.all(np.isfinite(c_new_raw))
            & np.isfinite(merit_new)
        )

        accepted = finite & (merit_slope < 0.0) & (
            merit_new <= merit0 + armijo * alpha * merit_slope
        )

        safe_merit = np.where(finite, merit_new, np.inf)
        return X_new, U_new, V_new, Veq_new, cost_new, c_new_raw, safe_merit, accepted

    X, U, V, Veq, costs, constraints, merits, accepted = vmap(body)(alpha_values)

    any_accepted = np.any(accepted)
    best_index = np.where(any_accepted, np.argmax(accepted), np.argmin(merits))
    alpha_best = alpha_values[best_index]

    X_best = np.where(any_accepted, X[best_index], X_in)
    U_best = np.where(any_accepted, U[best_index], U_in)
    V_best = np.where(any_accepted, V[best_index], V_in)
    Veq_best = np.where(any_accepted, Veq[best_index], Veq_in)
    cost_best = np.where(any_accepted, costs[best_index], current_cost)
    c_best = np.where(any_accepted, constraints[best_index], current_c)
    merit_best = np.where(any_accepted, merits[best_index], merit0)

    return (
        X_best,
        U_best,
        V_best,
        Veq_best,
        any_accepted,
        alpha_best
    )

@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics,x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X:               [T+1, n]      numpy array.
      U:               [T, m]        numpy array.

    Returns:
      g: the cost value (a scalar).
      c: the constraint values (a [T+1, n] numpy array).
    """
    T = U.shape[0]
    costs = vmap(cost)(X, np.pad(U, [[0, 1], [0, 0]]), np.arange(T + 1))
    g = np.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])

    return g, c


@partial(jit, static_argnums=(0, 1, 2))
def model_evaluator_equality_helper(cost, dynamics, equality, x0, X, U):
    """Evaluates cost and flattened dynamics/stage-equality residuals."""

    T = U.shape[0]
    costs = vmap(cost)(X, np.pad(U, [[0, 1], [0, 0]]), np.arange(T + 1))
    g = np.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])
    h = vmap(equality)(X[:-1], U, np.arange(T))

    return g, np.concatenate([c.reshape([-1]), h.reshape([-1])])


@partial(jit, static_argnums=(0,))
def stage_equality_evaluator(equality, X, U):
    """Evaluates stage equality constraints for t = 0, ..., T - 1."""

    return vmap(equality)(X[:-1], U, np.arange(U.shape[0]))


@partial(jit, static_argnums=(0,))
def direct_cost_evaluator_helper(cost, X, U):
    """Evaluates the trajectory cost for a dynamically feasible rollout."""

    T = U.shape[0]
    costs = vmap(cost)(X, np.pad(U, [[0, 1], [0, 0]]), np.arange(T + 1))
    return np.sum(costs)


@partial(jit, static_argnums=(0,))
def direct_dynamics_defect_helper(dynamics, x0, X, U):
    """Evaluates the multiple-shooting defects of a nominal trajectory."""

    T = U.shape[0]
    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    return np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])

@jit
def direct_model_improvement(Q, q, R, r, M, dX, dU):
    """Predicted decrease of the quadratic model for a unit step."""

    state_linear = np.sum(q * dX)
    control_linear = np.sum(r * dU)
    state_quad = 0.5 * np.sum(
        vmap(lambda x, q_mat: x.T @ q_mat @ x)(dX, Q)
    )
    control_quad = 0.5 * np.sum(
        vmap(lambda u, r_mat: u.T @ r_mat @ u)(dU, R)
    )
    cross_quad = np.sum(
        vmap(lambda x, m_mat, u: x.T @ m_mat @ u)(dX[:-1], M, dU)
    )
    model_value = state_linear + control_linear + state_quad + control_quad + cross_quad
    return np.maximum(-model_value, 0.0)


@jit
def direct_model_delta_terms(Q, q, R, r, M, dX, dU):
    """Return the quadratic-model coefficients used in the Goldstein test.

    The FDDP paper writes the expected cost variation of a trial step alpha as

      ΔJ(alpha) = Δ1 * alpha + 0.5 * Δ2 * alpha^2

    where Δ1 is the first-order term and Δ2 collects the second-order terms.
    """

    delta1 = np.sum(q * dX) + np.sum(r * dU)
    delta2_state = np.sum(vmap(lambda x, q_mat: x.T @ q_mat @ x)(dX, Q))
    delta2_control = np.sum(vmap(lambda u, r_mat: u.T @ r_mat @ u)(dU, R))
    delta2_cross = 2.0 * np.sum(
        vmap(lambda x, m_mat, u: x.T @ m_mat @ u)(dX[:-1], M, dU)
    )
    delta2 = delta2_state + delta2_control + delta2_cross
    return delta1, delta2


@jit
def equality_fddp_model_delta_terms(R, r, B, P, p, defects, k, dX):
    """Feasibility-aware expected cost variation from the paper.

    The constrained Riccati recursion uses the dynamics defects in its linear
    rollout. These terms implement Eqs. (24)-(25) in the paper, adapted to this
    codebase's sign convention where the feed-forward control increment is
    ``k`` and the expected cost variation is
    ``alpha * (delta1 + 0.5 * alpha * delta2)``.
    """

    T = k.shape[0]

    def terms(t):
        BtP = B[t].T @ P[t + 1]
        Qu = r[t] + B[t].T @ p[t + 1] + BtP @ defects[t + 1]
        Quu = R[t] + BtP @ B[t]
        f_bar = defects[t + 1]
        dx_next = dX[t + 1]

        delta1 = k[t].T @ Qu + f_bar.T @ (p[t + 1] - P[t + 1] @ dx_next)
        delta2 = (
            k[t].T @ Quu @ k[t]
            + f_bar.T @ (2.0 * P[t + 1] @ dx_next - P[t + 1] @ f_bar)
        )
        return delta1, delta2

    delta1, delta2 = vmap(terms)(np.arange(T))
    return np.sum(delta1), np.sum(delta2)


@jit
def fddp_l1_infeasibility(defects, h):
    """L1 infeasibility used by the equality-constrained DDP merit function."""

    return np.sum(np.abs(defects)) + np.sum(np.abs(h))


@jit
def update_fddp_merit_penalty(
    merit_penalty,
    expected_delta_unit,
    current_infeasibility,
    merit_rho=0.1,
    eps=1e-4,
):
    """Penalty update from Eq. (27), with this implementation's sign convention."""

    penalty_needed = np.where(
        current_infeasibility > eps,
        np.maximum(expected_delta_unit, 0.0)
        / ((1.0 - merit_rho) * current_infeasibility + eps),
        merit_penalty,
    )
    return np.maximum(merit_penalty, penalty_needed)


@partial(jit, static_argnums=(0, 1, 2, 4))
def parallel_equality_fddp_line_search(
    cost,
    dynamics,
    equality,
    x0,
    num_alpha,
    X_in,
    U_in,
    V_in,
    Veq_in,
    K,
    k,
    dV,
    dVeq,
    current_cost,
    current_defects,
    current_h,
    delta1,
    delta2,
    merit_penalty,
    eta1=0.1,
    eta2=2.0,
):
    """Paper-style FDDP line search for stage equality constraints.

    The trial states are produced by the feasibility-driven nonlinear rollout,
    which contracts dynamics defects by ``1 - alpha``. Acceptance follows the
    Goldstein-inspired test on the L1 merit function from Eqs. (26)-(29).
    """

    alpha_values = np.exp2(-np.arange(num_alpha, dtype=X_in.dtype))
    current_infeasibility = fddp_l1_infeasibility(current_defects, current_h)
    merit0 = current_cost + merit_penalty * current_infeasibility

    def evaluate_alpha(alpha):
        X_new, U_new = direct_optimal_control_rollout_with_defects(
            dynamics, x0, X_in, U_in, K, k, current_defects, alpha
        )
        V_new = V_in + alpha * dV
        Veq_new = Veq_in + alpha * dVeq

        cost_new = direct_cost_evaluator_helper(cost, X_new, U_new)
        defects_new = direct_dynamics_defect_helper(dynamics, x0, X_new, U_new)
        h_new = stage_equality_evaluator(equality, X_new, U_new)
        infeasibility_new = fddp_l1_infeasibility(defects_new, h_new)
        merit_new = cost_new + merit_penalty * infeasibility_new

        expected_delta = alpha * (delta1 + 0.5 * alpha * delta2)
        expected_merit_delta = (
            expected_delta - merit_penalty * alpha * current_infeasibility
        )
        goldstein_bound = np.where(
            expected_merit_delta <= 0.0,
            eta1 * expected_merit_delta,
            eta2 * expected_delta,
        )
        actual_merit_delta = merit_new - merit0

        finite = np.logical_and(np.all(np.isfinite(X_new)), np.all(np.isfinite(U_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(V_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(Veq_new)))
        finite = np.logical_and(finite, np.isfinite(cost_new))
        finite = np.logical_and(finite, np.all(np.isfinite(defects_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(h_new)))
        finite = np.logical_and(finite, np.isfinite(merit_new))

        accepted = np.logical_and(finite, actual_merit_delta <= goldstein_bound)
        safe_merit = np.where(finite, merit_new, np.inf)
        return (
            X_new,
            U_new,
            V_new,
            Veq_new,
            defects_new,
            h_new,
            cost_new,
            safe_merit,
            accepted,
        )

    (
        X_candidates,
        U_candidates,
        V_candidates,
        Veq_candidates,
        defect_candidates,
        h_candidates,
        cost_candidates,
        merit_candidates,
        accepted,
    ) = vmap(evaluate_alpha)(alpha_values)

    any_accepted = np.any(accepted)
    best_index = np.where(any_accepted, np.argmax(accepted), np.argmin(merit_candidates))
    alpha_best = alpha_values[best_index]

    X_best = np.where(any_accepted, X_candidates[best_index], X_in)
    U_best = np.where(any_accepted, U_candidates[best_index], U_in)
    V_best = np.where(any_accepted, V_candidates[best_index], V_in)
    Veq_best = np.where(any_accepted, Veq_candidates[best_index], Veq_in)
    defects_best = np.where(any_accepted, defect_candidates[best_index], current_defects)
    h_best = np.where(any_accepted, h_candidates[best_index], current_h)
    cost_best = np.where(any_accepted, cost_candidates[best_index], current_cost)

    return (
        X_best,
        U_best,
        V_best,
        Veq_best,
        defects_best,
        h_best,
        cost_best,
        alpha_best,
        any_accepted,
    )


@partial(jit, static_argnums=(0, 1, 2, 3))
def compute_fddp_search_direction(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    x0,
    X,
    U,
    defects,
    regularization = 1e-3
):
    """Computes the direct optimal control iLQR search direction."""

    T = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), np.arange(T + 1))

    linearizer = linearize(cost)
    q, r_pad = linearizer(X, pad(U), np.arange(T + 1))
    r = r_pad[:-1]

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    q_terminal = q[-1]
    q = q.at[-1].set(q_terminal)
    Q = Q + regularization * np.eye(n)[None, :, :]
    R = R_pad[:-1] + regularization * np.eye(m)[None, :, :]
    M = M_pad[:-1]
    c = defects[1:]
    dx0 = defects[0]

    if limited_memory:
        K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)
        dX, dU = rollout(K, k, dx0, A, B, c)
    else:
        K, k, _, _ = tvlqr_gpu(Q, q, R, r, M, A, B, c)
        dX, dU = rollout_gpu(K, k, dx0, A, B, c)

    delta1, delta2 = direct_model_delta_terms(Q, q, R, r, M, dX, dU)
    return K, k, dX, dU, q, r, delta1, delta2


@partial(jit, static_argnums=(0, 1, 3))
def parallel_goldstein_line_search(
    cost,
    dynamics,
    x0,
    num_alpha,
    X_in,
    U_in,
    K,
    k,
    current_cost,
    current_defects,
    delta1,
    delta2,
    b1=0.1,
    b2=2.0,
):
    """Evaluate a fixed alpha grid with the Goldstein test from FDDP.

    The paper accepts a trial step when the actual cost variation is bounded by
    a scaled version of the expected variation ΔJ(alpha). Since our alpha grid is
    ordered from 1 down to 2^(-num_alpha+1), taking the first accepted element
    reproduces a backtracking-style Goldstein search without a while loop.
    """

    alpha_values = np.exp2(-np.arange(num_alpha, dtype=X_in.dtype))

    def evaluate_alpha(alpha):
        X_new, U_new = direct_optimal_control_rollout_with_defects(
            dynamics, x0, X_in, U_in, K, k, current_defects, alpha
        )
        new_cost = direct_cost_evaluator_helper(cost, X_new, U_new)
        new_defects = direct_dynamics_defect_helper(dynamics, x0, X_new, U_new)
        theta_new = np.sum(new_defects * new_defects)

        expected_delta = delta1 * alpha + 0.5 * delta2 * alpha * alpha
        actual_delta = new_cost - current_cost
        goldstein_bound = np.where(
            expected_delta <= 0.0,
            b1 * expected_delta,
            b2 * expected_delta,
        )

        finite = np.logical_and(np.all(np.isfinite(X_new)), np.all(np.isfinite(U_new)))
        finite = np.logical_and(finite, np.all(np.isfinite(new_defects)))
        accepted = np.logical_and(finite, actual_delta <= goldstein_bound)
        safe_cost = np.where(finite, new_cost, np.inf)
        safe_theta = np.where(finite, theta_new, np.inf)
        return X_new, U_new, new_defects, safe_cost, safe_theta, accepted

    (
        X_candidates,
        U_candidates,
        defect_candidates,
        candidate_costs,
        candidate_thetas,
        accepted,
    ) = vmap(evaluate_alpha)(alpha_values)

    any_accepted = np.any(accepted)
    best_index = np.where(any_accepted, np.argmax(accepted), 0)

    X_best = np.where(any_accepted, X_candidates[best_index], X_in)
    U_best = np.where(any_accepted, U_candidates[best_index], U_in)
    defects_best = np.where(
        any_accepted, defect_candidates[best_index], current_defects
    )
    cost_best = np.where(any_accepted, candidate_costs[best_index], current_cost)
    theta_best = np.where(
        any_accepted,
        candidate_thetas[best_index],
        np.sum(current_defects * current_defects),
    )
    alpha_best = np.where(any_accepted, alpha_values[best_index], 0.0)

    return X_best, U_best, defects_best, cost_best, theta_best, alpha_best, any_accepted


@partial(jit, static_argnums=(0, 1, 2, 3, 10, 11, 12))
def fddp_mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    num_alpha=11,
    goldstein_b1=0.1,
    goldstein_b2=2.0,
):
    """Single direct optimal control iLQR step with FDDP defect closing."""

    _cost = partial(cost, W, reference)
    _hessian_approx = (
        partial(hessian_approx, W, reference) if hessian_approx is not None else None
    )
    _dynamics = partial(dynamics, parameter=parameter)

    X0 = X_in
    defects0 = direct_dynamics_defect_helper(_dynamics, x0, X0, U_in)
    cost0 = direct_cost_evaluator_helper(_cost, X0, U_in)

    K, k, _, _, _, _, delta1, delta2 = compute_fddp_search_direction(
        _cost,
        _dynamics,
        _hessian_approx,
        limited_mempory,
        x0,
        X0,
        U_in,
        defects0,
    )

    X_new, U_new, defects_new, _, _, _, accepted = parallel_goldstein_line_search(
        _cost,
        _dynamics,
        x0,
        num_alpha,
        X0,
        U_in,
        K,
        k,
        cost0,
        defects0,
        delta1,
        delta2,
        b1=goldstein_b1,
        b2=goldstein_b2,
    )

    return X_new, U_new, defects_new


@partial(jit, static_argnums=(0, 1, 2, 3, 11, 15))
def mpc_equality_fddp(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    equality,
    Veq_in,
    regularization,
    merit_penalty=1e-6,
    num_alpha=11,
    merit_rho=0.1,
    eta1=0.1,
    eta2=2.0,
):
    """Single equality-constrained FDDP step.

    This follows the equality-constrained DDP forward-pass logic from the
    inverse-dynamics MPC paper: constrained Riccati search direction,
    feasibility-driven nonlinear rollout, merit penalty update, and the
    Goldstein-inspired acceptance test.

    Returns:
      X, U, V, Veq, regularization, merit_penalty, alpha_best, accepted.
    """

    _equality = partial(equality, parameter=parameter)
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None
    _dynamics = partial(dynamics, parameter=parameter)

    defects0 = direct_dynamics_defect_helper(_dynamics, x0, X_in, U_in)
    h0 = stage_equality_evaluator(_equality, X_in, U_in)
    cost0 = direct_cost_evaluator_helper(_cost, X_in, U_in)

    (
        K,
        k,
        dX,
        dU,
        dV,
        dVeq,
        q,
        r,
        R,
        B,
        P,
        p,
    ) = compute_equality_fddp_search_direction(
        _cost,
        _dynamics,
        _equality,
        _hessian_approx,
        limited_mempory,
        x0,
        X_in,
        U_in,
        V_in,
        Veq_in,
        defects0,
        h0,
        regularization,
    )

    delta1, delta2 = equality_fddp_model_delta_terms(
        R, r, B, P, p, defects0, k, dX
    )
    expected_delta_unit = delta1 + 0.5 * delta2
    current_infeasibility = fddp_l1_infeasibility(defects0, h0)
    merit_penalty_new = update_fddp_merit_penalty(
        merit_penalty,
        expected_delta_unit,
        current_infeasibility,
        merit_rho=merit_rho,
    )

    (
        X_new,
        U_new,
        V_new,
        Veq_new,
        defects_new,
        h_new,
        cost_new,
        alpha_best,
        any_accepted,
    ) = parallel_equality_fddp_line_search(
        _cost,
        _dynamics,
        _equality,
        x0,
        num_alpha,
        X_in,
        U_in,
        V_in,
        Veq_in,
        K,
        k,
        dV,
        dVeq,
        cost0,
        defects0,
        h0,
        delta1,
        delta2,
        merit_penalty_new,
        eta1=eta1,
        eta2=eta2,
    )

    regularization_new = np.where(
        any_accepted, regularization * 0.75, regularization * 1.25
    )
    regularization_new = np.clip(regularization_new, 1e-6, 1e1)

    return (
        X_new,
        U_new,
        V_new,
        Veq_new,
        regularization_new,
        merit_penalty_new,
        alpha_best,
        any_accepted,
    )


@partial(jit, static_argnums=(0,1,2,3,11))
def mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    num_alpha=11
    ):

    _cost = partial(cost,W,reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx,W,reference)
    else:
        _hessian_approx = None
    _dynamics = partial(dynamics,parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics,x0)
    g, c = model_evaluator(X_in, U_in)
    dX,dU, dV, q, r = compute_search_direction(
            _cost,
            _dynamics,
            _hessian_approx,
            limited_mempory,
            x0,
            X_in,
            U_in,
            V_in,
            c,
        )
    # @jit
    # def merit_function(V, g, c, rho):
    #     return g + np.sum((V + 0.5 * rho * c) * c)

    # dV2 = np.sum(dV * dV)
    # c2 = np.sum(c * c)
    # rho  = 2.0 * np.sqrt(dV2 / c2)
    # merit = merit_function(V_in, g, c, rho)

    # merit_slope = slope(
    #     dX,
    #     dU,
    #     dV,
    #     c,
    #     q,
    #     r,
    #     rho,
    # )
    # X_new, U_new, V_new, g_new, c_new = parallel_line_search(
    #         partial(merit_function, rho=rho),
    #         model_evaluator,
    #         X_in,
    #         U_in,
    #         V_in,
    #         dX,
    #         dU,
    #         dV,
    #         merit,
    #         g,
    #         c,
    #         merit_slope,
    #         armijo_factor=1e-4,
    #     )
    X_new, U_new, V_new = parallel_filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    g,
    c,
    q,
    r,
    num_alpha=num_alpha,)

    return X_new, U_new, V_new

def mpc_equality(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    equality,
    Veq_in,
    regularization,
    num_alpha=11,
):
    """Single MPC SQP step with stage equality constraints.

    ``equality`` follows the same convention as ``cost``:
    ``equality(W, reference, x, u, t) -> residual`` for t = 0, ..., T - 1.
    Passing ``Veq_in`` warm-starts the equality multipliers; when omitted they
    are initialized to zeros with the stage equality residual shape.
    """

    _equality = partial(equality, parameter=parameter)
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None
    _dynamics = partial(dynamics, parameter=parameter)

    dynamics_evaluator = partial(model_evaluator_helper, _cost, _dynamics, x0)
    g, c = dynamics_evaluator(X_in, U_in)
    h = stage_equality_evaluator(_equality, X_in, U_in)

    dX, dU, dV, dVeq, q, r = compute_equality_search_direction(
        _cost,
        _dynamics,
        _equality,
        _hessian_approx,
        limited_mempory,
        x0,
        X_in,
        U_in,
        V_in,
        Veq_in,
        c,
        h,
        regularization
    )

    model_evaluator = partial(
        model_evaluator_equality_helper,
        _cost,
        _dynamics,
        _equality,
        x0,
    )
    _, residual = model_evaluator(X_in, U_in)

    X_new, U_new, V_new, Veq_new, any_accepted, alpha_best = parallel_l1_merit_line_search_equality(
        model_evaluator,
        X_in,
        U_in,
        V_in,
        Veq_in,
        dX,
        dU,
        dV,
        dVeq,
        g,
        residual,
        q,
        r,
        num_alpha=num_alpha,
    )
    regularization_new = np.where(any_accepted, regularization * 0.75, regularization * 1.25)
    regularization_new = np.clip(regularization_new, 1e-6, 1e1)
    return X_new, U_new, V_new, Veq_new, regularization_new, alpha_best, any_accepted


def ip_mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    equality,
    Veq_in,
    inequality,
    Vineq_in,
    slack_in,
    regularization,
    barrier_parameter=1.0,
    multiplier_regularization=1e-8,
    num_alpha=11,
):
    """One HIPPO-style primal-dual interior-point SQP iteration.

    Inequalities use ``g(x, u) <= 0`` with positive multipliers and slacks.
    The regularized IPM system is condensed using Eq. (19) from HIPPO, then
    solved by the existing equality-constrained Riccati recursion.
    """
    horizon = U_in.shape[0]
    pad = lambda value: np.pad(value, [[0, 1], [0, 0]])
    _cost = partial(cost, W, reference)
    _dynamics = partial(dynamics, parameter=parameter)
    _equality = partial(equality, parameter=parameter)
    _inequality = partial(inequality, parameter=parameter)

    g = vmap(_inequality)(X_in[:-1], U_in, np.arange(horizon))
    inequality_linearizer = linearize(_inequality)
    gx, gu = inequality_linearizer(X_in[:-1], U_in, np.arange(horizon))
    slack = np.maximum(slack_in, 1e-8)
    vineq = np.maximum(Vineq_in, 1e-8)

    defects = direct_dynamics_defect_helper(_dynamics, x0, X_in, U_in)
    h = stage_equality_evaluator(_equality, X_in, U_in)

    if hessian_approx is None:
        Q, R_pad, M_pad = quadratize(_cost)(
            X_in, pad(U_in), np.arange(horizon + 1)
        )
    else:
        _hessian_approx = partial(hessian_approx, W, reference)
        Q, R_pad, M_pad = vmap(_hessian_approx)(
            X_in, pad(U_in), np.arange(horizon + 1)
        )
    R = R_pad[:-1]
    M = M_pad[:-1]

    dynamics_linearizer = linearize(_dynamics)
    A_pad, B_pad = dynamics_linearizer(
        X_in, pad(U_in), np.arange(horizon + 1)
    )
    A, B = A_pad[:-1], B_pad[:-1]
    equality_linearizer = linearize(_equality)
    hx_pad, hu_pad = equality_linearizer(
        X_in, pad(U_in), np.arange(horizon + 1)
    )
    hx, hu = hx_pad[:-1], hu_pad[:-1]
    lagrangian_linearizer = linearize(lagrangian(_cost, _dynamics, x0), argnums=5)
    q, r_pad = lagrangian_linearizer(
        X_in,
        pad(U_in),
        np.arange(horizon + 1),
        pad(V_in[1:]),
        V_in,
    )
    q = q.at[:-1].add(np.einsum("tpi,tp->ti", hx, Veq_in))
    r = r_pad[:-1] + np.einsum("tpi,tp->ti", hu, Veq_in)

    t_rho = slack + multiplier_regularization * vineq
    diagonal = vineq / t_rho
    Q_ip = Q.at[:-1].add(np.einsum("tpi,tp,tpj->tij", gx, diagonal, gx))
    R_ip = R + np.einsum("tpi,tp,tpj->tij", gu, diagonal, gu)
    M_ip = M + np.einsum("tpi,tp,tpj->tij", gx, diagonal, gu)

    def direction(mu, corrector):
        linear_term = diagonal * g + (mu - corrector) / t_rho
        q_ip = q.at[:-1].add(np.einsum("tpi,tp->ti", gx, linear_term))
        r_ip = r + np.einsum("tpi,tp->ti", gu, linear_term)
        dX, dU, dV, dVeq = _solve_equality_lqr(
            Q_ip,
            q_ip,
            R_ip,
            r_ip,
            M_ip,
            A,
            B,
            defects[1:],
            hx,
            hu,
            h,
            regularization,
            defects[0],
            limited_mempory,
        )
        dg = vmap(lambda a, b, dx, du: a @ dx + b @ du)(gx, gu, dX[:-1], dU)
        primal_residual = g + slack
        complementarity = vineq * slack - mu + corrector
        dVineq = (
            -complementarity + vineq * (primal_residual + dg)
        ) / t_rho
        dslack = (
            -primal_residual
            - dg
            + multiplier_regularization * dVineq
        )
        return dX, dU, dV, dVeq, dVineq, dslack

    def fraction_to_boundary(value, step, tau=0.995):
        ratios = np.where(step < 0.0, -tau * value / step, np.inf)
        return np.minimum(1.0, np.min(ratios))

    affine = direction(0.0, np.zeros_like(slack))
    alpha_affine = fraction_to_boundary(slack, affine[5])
    alpha_v_affine = fraction_to_boundary(vineq, affine[4])
    slack_affine = slack + alpha_affine * affine[5]
    vineq_affine = vineq + alpha_v_affine * affine[4]
    complementarity = np.sum(vineq * slack)
    affine_complementarity = np.sum(vineq_affine * slack_affine)
    sigma = np.clip(
        affine_complementarity / np.maximum(complementarity, 1e-12), 0.0, 1.0
    ) ** 3
    mu = sigma * complementarity / slack.size
    corrector = affine[4] * affine[5]
    use_corrector = np.sum(np.abs(vineq_affine * slack_affine - mu)) < np.sum(
        np.abs(vineq * slack - mu)
    )
    corrector = np.where(use_corrector, corrector, np.zeros_like(corrector))
    dX, dU, dV, dVeq, dVineq, dslack = direction(mu, corrector)

    alpha_max = fraction_to_boundary(slack, dslack)
    alpha_v_max = fraction_to_boundary(vineq, dVineq)
    scales = 1.0 - np.arange(num_alpha, dtype=X_in.dtype) / num_alpha
    alpha_values = alpha_max * scales
    alpha_v_values = alpha_v_max * scales
    primal0 = (
        np.sum(np.abs(defects))
        + np.sum(np.abs(h))
        + np.sum(np.abs(g + slack))
    )
    complementarity0 = np.sum(np.abs(vineq * slack - mu))

    def trial(alpha, alpha_v):
        X = X_in + alpha * dX
        U = U_in + alpha * dU
        V = V_in + alpha * dV
        Veq = Veq_in + alpha * dVeq
        Vineq = vineq + alpha_v * dVineq
        trial_slack = slack + alpha * dslack
        trial_defects = direct_dynamics_defect_helper(_dynamics, x0, X, U)
        trial_h = stage_equality_evaluator(_equality, X, U)
        trial_g = vmap(_inequality)(X[:-1], U, np.arange(horizon))
        primal = (
            np.sum(np.abs(trial_defects))
            + np.sum(np.abs(trial_h))
            + np.sum(np.abs(trial_g + trial_slack))
        )
        complementarity_residual = np.sum(
            np.abs(Vineq * trial_slack - mu)
        )
        finite = (
            np.all(np.isfinite(X))
            & np.all(np.isfinite(U))
            & np.all(np.isfinite(V))
            & np.all(np.isfinite(Veq))
            & np.all(np.isfinite(Vineq))
            & np.all(np.isfinite(trial_slack))
        )
        positive = np.all(Vineq > 0.0) & np.all(trial_slack > 0.0)
        accepted = finite & positive & (
            (primal < primal0) | (complementarity_residual < complementarity0)
        )
        return X, U, V, Veq, Vineq, trial_slack, accepted

    candidates = vmap(trial)(alpha_values, alpha_v_values)
    accepted = candidates[-1]
    any_accepted = np.any(accepted)
    best = np.argmax(accepted)

    def select(values, nominal):
        return np.where(any_accepted, values[best], nominal)

    X_new = select(candidates[0], X_in)
    U_new = select(candidates[1], U_in)
    V_new = select(candidates[2], V_in)
    Veq_new = select(candidates[3], Veq_in)
    Vineq_new = select(candidates[4], vineq)
    slack_new = select(candidates[5], slack)
    alpha_best = np.where(any_accepted, alpha_values[best], 0.0)
    regularization_new = np.clip(
        np.where(any_accepted, regularization * 0.75, regularization * 1.25),
        1e-6,
        1e1,
    )
    return (
        X_new,
        U_new,
        V_new,
        Veq_new,
        Vineq_new,
        slack_new,
        regularization_new,
        mu,
        alpha_best,
        any_accepted,
    )
    
