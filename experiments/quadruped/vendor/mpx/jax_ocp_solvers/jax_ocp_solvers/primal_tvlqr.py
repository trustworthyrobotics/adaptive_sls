import jax.numpy as np

from jax import jit, lax, scipy, vmap

from .linalg_helpers import solve_symmetric_positive_definite_system

from functools import partial

import jax

def lqr_step(P, p, Q, q, R, r, M, A, B, c):
    """Single LQR Step.

    Args:
      P: [n, n] numpy array.
      p: [n]    numpy array.
      Q: [n, n] numpy array.
      q: [n]    numpy array.
      R: [m, m] numpy array.
      r: [m]    numpy array.
      M: [n, m] numpy array.
      A: [n, n] numpy array.
      B: [n, m] numpy array.
      c: [n]    numpy array.

    Returns:
      K, k: state feedback gain and affine term.
      P, p: updated matrices encoding quadratic value function.
    """
    symmetrize = lambda x: 0.5 * (x + x.T)

    AtP = A.T @ P
    AtPA = symmetrize(AtP @ A)
    BtP = B.T @ P
    BtPA = BtP @ A

    H = BtPA + M.T
    h = B.T @ p + BtP @ c + r

    G = symmetrize(R + BtP @ B)

    K_k = solve_symmetric_positive_definite_system(
        G, -np.hstack((H, h.reshape([-1, 1])))
    )

    K = K_k[:, :-1]
    k = K_k[:, -1]

    P = symmetrize(Q + AtPA + K.T @ H)
    p = q + A.T @ p + AtP @ c + K.T @ h

    return K, k, P, p

def constrained_lqr_step(P, p, Q, q, R, r, M, A, B, c, hx, hu, h_bar):
    """Single LQR Step.

    Args:
      P: [n, n] numpy array.
      p: [n]    numpy array.
      Q: [n, n] numpy array.
      q: [n]    numpy array.
      R: [m, m] numpy array.
      r: [m]    numpy array.
      M: [n, m] numpy array.
      A: [n, n] numpy array.
      B: [n, m] numpy array.
      c: [n]    numpy array.

    Returns:
      K, k: state feedback gain and affine term.
      P, p: updated matrices encoding quadratic value function.
    """
    symmetrize = lambda x: 0.5 * (x + x.T)
    AtP = A.T @ P
    AtPA = symmetrize(AtP @ A)
    BtP = B.T @ P
    BtPA = BtP @ A

    H = BtPA + M.T
    h = B.T @ p + BtP @ c + r

    G = symmetrize(R + BtP @ B)
    unconstrained_policy = scipy.linalg.solve(
        G, -np.hstack((H, h.reshape([-1, 1])))
    )
    K_u = unconstrained_policy[:, :-1]
    k_u = unconstrained_policy[:, -1]

    psi = scipy.linalg.solve(G, hu.T)
    constraint_metric = hu @ psi
    eta_policy = scipy.linalg.solve(
        constraint_metric,
        np.hstack(
            (
                hx + hu @ K_u,
                (h_bar + hu @ k_u).reshape([-1, 1]),
            )
        ),
    )
    K_eta = eta_policy[:, :-1]
    k_eta = eta_policy[:, -1]

    K = K_u - psi @ K_eta
    k = k_u - psi @ k_eta

    P = symmetrize(Q + AtPA + K.T @ G @ K + K.T @ H + H.T @ K)
    p = q + A.T @ p + AtP @ c + K.T @ G @ k + K.T @ h + H.T @ k

    return K, k, P, p, K_eta, k_eta
@jit
def tvlqr(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    Args:
      Q: [T+1, n, n]  numpy array.
      q: [T+1, n]     numpy array.
      R: [T, m, m]    numpy array.
      r: [T, m]       numpy array.
      M: [T, n, m]    numpy array.
      A: [T, n, n]    numpy array.
      B: [T, n, m]    numpy array.
      c: [T, n]       numpy array.

    Returns:
      K: [T, m, n]    Gains
      k: [T, m]       Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n]  numpy array encoding initial value function.
      p: [T+1, n]     numpy array encoding initial value function.
    """

    T = Q.shape[0] - 1
    n = Q.shape[1]
    def f(carry, elem):
        P, p = carry
        t = elem

        K, k, P, p = lqr_step(P, p, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], c[t])

        new_carry = (P, p)
        new_output = (K, k, P, p)

        return new_carry, new_output

    K, k, P, p = lax.scan(f, (Q[T], q[T]), np.arange(T), T, reverse=True)[1]

    return (
        K,
        k,
        np.concatenate([P, Q[T].reshape([1, n, n])]),
        np.concatenate([p, q[T].reshape([1, n])]),
    )

@jit
def tvlqr_constrained(Q, q, R, r, M, A, B, c,hx,hu,h_bar):
    """Discrete-time Finite Horizon Time-varying LQR.

    Args:
      Q: [T+1, n, n]  numpy array.
      q: [T+1, n]     numpy array.
      R: [T, m, m]    numpy array.
      r: [T, m]       numpy array.
      M: [T, n, m]    numpy array.
      A: [T, n, n]    numpy array.
      B: [T, n, m]    numpy array.
      c: [T, n]       numpy array.

    Returns:
      K: [T, m, n]    Gains
      k: [T, m]       Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n]  numpy array encoding initial value function.
      p: [T+1, n]     numpy array encoding initial value function.
    """

    T = Q.shape[0] - 1
    n = Q.shape[1]
    def f(carry, elem):
        P, p = carry
        t = elem

        K, k, P, p, Ks, ks = constrained_lqr_step(P, p, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], c[t],hx[t],hu[t],h_bar[t])

        new_carry = (P, p)
        new_output = (K, k, P, p, Ks, ks)

        return new_carry, new_output

    K, k, P, p, Ks, ks = lax.scan(f, (Q[T], q[T]), np.arange(T), T, reverse=True)[1]

    return (
        K,
        k,
        np.concatenate([P, Q[T].reshape([1, n, n])]),
        np.concatenate([p, q[T].reshape([1, n])]),
        Ks,
        ks,
    )

tvlqr_equality = tvlqr_constrained


@jit
def tvlqr_gpu_constrained(Q, q, R, r, M, A, B, c,hx,hu,h_bar):
    """Associative-scan equality-constrained finite-horizon TVLQR.

    Args:
      Q: [T+1, n, n] numpy array.
      q: [T+1, n]    numpy array.
      R: [T, m, m]   numpy array.
      r: [T, m]      numpy array.
      M: [T, n, m]   numpy array.
      A: [T, n, n]   numpy array.
      B: [T, n, m]   numpy array.
      c: [T, n]      numpy array.
      hx: [T, p, n]   Equality Jacobian with respect to state.
      hu: [T, p, m]   Equality Jacobian with respect to control.
      h_bar: [T, p]   Equality residual.

    Returns:
      K: [T, m, n] Gains
      k: [T, m] Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n] numpy array encoding initial value function.
      p: [T+1, n] numpy array encoding initial value function.
    """
    T = Q.shape[0] - 1
    n = Q.shape[1]
    m = R.shape[1]
    p_dim = hu.shape[1]

    def fn(next, prev):
        def decompose(elem):
            return (
                elem[:n],
                elem[n],
                elem[n + 1 : 2 * n + 1],
                elem[2 * n + 1],
                elem[-n:],
            )

        A_l, c_l, C_l, p_l, P_l = decompose(prev)
        A_r, c_r, C_r, p_r, P_r = decompose(next)

        ArIClPr_inv = A_r @ np.linalg.inv(np.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ np.linalg.inv(np.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return np.concatenate(
            [
                A_new,
                c_new.reshape(1, n),
                C_new,
                p_new.reshape(1, n),
                P_new,
            ]
        )

    def constrained_conditional_value(t):
        zero_pp = np.zeros([p_dim, p_dim], dtype=R.dtype)
        zero_pn = np.zeros([p_dim, n], dtype=R.dtype)
        kkt = np.concatenate(
            [
                np.concatenate([R[t], hu[t].T], axis=1),
                np.concatenate([hu[t], zero_pp], axis=1),
            ],
            axis=0,
        )
        kkt_inv = np.linalg.solve(kkt, np.eye(m + p_dim, dtype=R.dtype))

        D_x = np.concatenate([M[t].T, hx[t]], axis=0)
        D_lambda = np.concatenate([B[t].T, zero_pn], axis=0)
        d0 = np.concatenate([r[t], h_bar[t]], axis=0)

        DlambdaT_kkt_inv = D_lambda.T @ kkt_inv
        DxT_kkt_inv = D_x.T @ kkt_inv

        A_bar = A[t] - DlambdaT_kkt_inv @ D_x
        c_bar = c[t] - DlambdaT_kkt_inv @ d0
        C_bar = DlambdaT_kkt_inv @ D_lambda
        p_bar = q[t] - DxT_kkt_inv @ d0
        P_bar = Q[t] - DxT_kkt_inv @ D_x

        return A_bar, c_bar, C_bar, p_bar, P_bar

    A_bar, c_bar, C_bar, p_bar, P_bar = vmap(constrained_conditional_value)(
        np.arange(T)
    )

    elems = np.concatenate(
        [
            np.concatenate([A_bar, np.zeros([1, n, n])]),
            np.concatenate([c_bar.reshape([T, 1, n]), np.zeros([1, 1, n])]),
            np.concatenate([C_bar, np.zeros([1, n, n])]),
            np.concatenate([p_bar.reshape([T, 1, n]), q[T].reshape([1, 1, n])]),
            np.concatenate([P_bar, Q[T].reshape([1, n, n])]),
        ],
        axis=1,
    )

    result = lax.associative_scan(lambda r, l: vmap(fn)(r, l), elems, reverse=True)

    P = result[:, -n:, :]
    p = result[:, 2 * n + 1, :]

    def getKs(t):
        K, k, _, _, K_eq, k_eq = constrained_lqr_step(
            P[t + 1],
            p[t + 1],
            Q[t],
            q[t],
            R[t],
            r[t],
            M[t],
            A[t],
            B[t],
            c[t],
            hx[t],
            hu[t],
            h_bar[t],
        )
        return K, k, K_eq, k_eq

    K, k, K_eq, k_eq = vmap(getKs)(np.arange(T))

    return K, k, P, p, K_eq, k_eq


tvlqr_gpu_equality = tvlqr_gpu_constrained


@jit
def tvlqr_gpu(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    This is a O(log T) parallel time complexity implementation, based on
    https://ieeexplore.ieee.org/document/9697418.

    Args:
      Q: [T+1, n, n] numpy array.
      q: [T+1, n]    numpy array.
      R: [T, m, m]   numpy array.
      r: [T, m]      numpy array.
      M: [T, n, m]   numpy array.
      A: [T, n, n]   numpy array.
      B: [T, n, m]   numpy array.
      c: [T, n]      numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

    Returns:
      K: [T, m, n] Gains
      k: [T, m] Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n] numpy array encoding initial value function.
      p: [T+1, n] numpy array encoding initial value function.
    """
    T = Q.shape[0] - 1
    n = Q.shape[1]

    def fn(next, prev):
        def decompose(elem):
            return (
                elem[:n],
                elem[n],
                elem[n + 1 : 2 * n + 1],
                elem[2 * n + 1],
                elem[-n:],
            )

        A_l, c_l, C_l, p_l, P_l = decompose(prev)
        A_r, c_r, C_r, p_r, P_r = decompose(next)

        ArIClPr_inv = A_r @ np.linalg.inv(np.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ np.linalg.inv(np.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return np.concatenate(
            [
                A_new,
                c_new.reshape(1, n),
                C_new,
                p_new.reshape(1, n),
                P_new,
            ]
        )

    def chol_inv(t):
        f = scipy.linalg.cho_factor(R[t])
        m = R[t].shape[0]
        return scipy.linalg.cho_solve(f, np.eye(m))

    Rinv = vmap(chol_inv)(np.arange(T))
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(np.arange(T))
    MRinv = vmap(lambda t: M[t] @ Rinv[t])(np.arange(T))

    elems = np.concatenate(
        [
            # The A matrices.
            np.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ M[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
            # The c vectors (b, in the notation of https://ieeexplore.ieee.org/document/9697418).
            np.concatenate(
                [
                    (c - vmap(lambda t: BRinv[t] @ r[t])(np.arange(T))).reshape(
                        [T, 1, n]
                    ),
                    np.zeros([1, 1, n]),
                ]
            ),
            # The C matrices.
            np.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
            # The p vectors (-eta, in the notation of https://ieeexplore.ieee.org/document/9697418).
            q.reshape([T + 1, 1, n])
            - np.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ r[t])(np.arange(T)).reshape([T, 1, n]),
                    np.zeros([1, 1, n]),
                ]
            ),
            # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
            Q
            - np.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ M[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    result = lax.associative_scan(lambda r, l: vmap(fn)(r, l), elems, reverse=True)

    P = result[:, -n:, :]
    p = result[:, 2 * n + 1, :]

    def getKs(t):
        # symmetrize = lambda x: 0.5 * (x + x.T)

        BtP = B[t].T @ P[t + 1]
        BtPA = BtP @ A[t]

        H = BtPA + M[t].T
        h = B[t].T @ p[t + 1] + BtP @ c[t] + r[t]

        # G = symmetrize(R[t] + BtP @ B[t])

        # f = scipy.linalg.cho_factor(G)
        K_k = scipy.linalg.solve(R[t] + BtP @ B[t], -np.hstack((H, h.reshape([-1, 1]))))
        K = K_k[:, :-1]
        k = K_k[:, -1]

        return K, k

    K, k = vmap(getKs)(np.arange(T))

    return K, k, P, p


@jit
def rollout(K, k, x0, A, B, c, alpha=1.0):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

    T, n = c.shape

    def f(carry, elem):
        t = elem

        x = carry
        u = K[t] @ x + alpha * k[t]
        next_x = A[t] @ x + B[t] @ u + c[t]

        new_carry = next_x
        new_output = (next_x, u)

        return new_carry, new_output

    (X, U) = lax.scan(f, x0, np.arange(T), T)[1]

    return (np.concatenate([x0.reshape([1, n]), X]), U)

@partial(jit, static_argnums=(0))
def non_linear_rollout(dynamics, x0, K, k, U, alpha=1.0):
    """Rolls out a nonlinear policy around an open-loop control sequence."""

    T = k.shape[0]
    n = x0.shape[0]

    def f(carry, elem):
        t = elem

        x = carry
        u = K[t] @ x + alpha * k[t] + U[t]
        next_x = dynamics(x, u, t)

        new_carry = next_x
        new_output = (next_x, u)

        return new_carry, new_output

    (X, U) = lax.scan(f, x0, np.arange(T), T)[1]

    return (np.concatenate([x0.reshape([1, n]), X]), U)


@partial(jit, static_argnums=(0))
def rollout_controls(dynamics, x0, U):
    """Roll out dynamics under an open-loop control sequence."""

    T = U.shape[0]
    n = x0.shape[0]

    def f(carry, elem):
        u, t = elem
        next_x = dynamics(carry, u, t)
        return next_x, next_x

    X = lax.scan(f, x0, (U, np.arange(T)))[1]
    return np.concatenate([x0.reshape([1, n]), X.reshape(T, n)])


@partial(jit, static_argnums=(0))
def direct_optimal_control_rollout(dynamics, x0, X_ref, U_ref, K, k, alpha):
    """Nonlinear forward pass for direct optimal control iLQR/DDP.

    The control law matches the standard DDP/iLQR forward rollout:
    u_t = u_ref_t + alpha * k_t + K_t (x_t - x_ref_t).
    """

    T = U_ref.shape[0]
    n = x0.shape[0]

    def f(carry, elem):
        t = elem
        x = carry
        dx = x - X_ref[t]
        u = U_ref[t] + alpha * k[t] + K[t] @ dx
        next_x = dynamics(x, u, t)
        return next_x, (next_x, u)

    X, U = lax.scan(f, x0, np.arange(T), T)[1]
    return np.concatenate([x0.reshape([1, n]), X.reshape(T, n)]), U


@partial(jit, static_argnums=(0))
def direct_optimal_control_rollout_with_defects(
    dynamics, x0, X_ref, U_ref, K, k, defects, alpha
):
    """FDDP-style nonlinear forward pass that contracts shooting defects.

    The initial node and each subsequent node retain a fraction ``1-alpha`` of the
    current dynamics defect instead of forcing a fully feasible rollout for
    exploratory steps.
    """

    T = U_ref.shape[0]
    n = x0.shape[0]
    x_init = x0 - (1.0 - alpha) * defects[0]

    def f(carry, elem):
        t = elem
        x = carry
        dx = x - X_ref[t]
        u = U_ref[t] + alpha * k[t] + K[t] @ dx
        next_x = dynamics(x, u, t) - (1.0 - alpha) * defects[t + 1]
        return next_x, (next_x, u)

    X, U = lax.scan(f, x_init, np.arange(T), T)[1]
    return np.concatenate([x_init.reshape([1, n]), X.reshape(T, n)]), U

@jit
def rollout_gpu(K, k, x0, A, B, c, alpha=1.0):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
    T, _, n = K.shape

    def fn(prev, next):
        F = prev[:-1]
        f = prev[-1]
        G = next[:-1]
        g = next[-1]
        return np.concatenate([G @ F, (g + G @ f).reshape([1, n])])

    get_elem = lambda t: np.concatenate(
        [A[t] + B[t] @ K[t], (c[t] + alpha * (B[t] @ k[t])).reshape([1, n])]
    )
    elems = vmap(get_elem)(np.arange(T))
    comp = lax.associative_scan(lambda l, r: vmap(fn)(l, r), elems)
    X = np.concatenate(
        [
            x0.reshape(1, n),
            vmap(lambda t: comp[t, :-1, :] @ x0 + comp[t, -1, :])(np.arange(T)),
        ]
    )

    U = vmap(lambda t: K[t] @ X[t] + alpha * k[t])(np.arange(T))

    return X, U

def rollout_gpu_constrained(K, k, x0, A, B, c, Ks, ks):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
    T, _, n = K.shape

    def fn(prev, next):
        F = prev[:-1]
        f = prev[-1]
        G = next[:-1]
        g = next[-1]
        return np.concatenate([G @ F, (g + G @ f).reshape([1, n])])

    get_elem = lambda t: np.concatenate(
        [A[t] + B[t] @ K[t], (c[t] + B[t] @ k[t]).reshape([1, n])]
    )
    elems = vmap(get_elem)(np.arange(T))
    comp = lax.associative_scan(lambda l, r: vmap(fn)(l, r), elems)
    X = np.concatenate(
        [
            x0.reshape(1, n),
            vmap(lambda t: comp[t, :-1, :] @ x0 + comp[t, -1, :])(np.arange(T)),
        ]
    )

    U = vmap(lambda t: K[t] @ X[t] + k[t])(np.arange(T))
    Veq = vmap(lambda t: Ks[t] @ X[t] + ks[t])(np.arange(T))

    return X, U, Veq


@jit
def rollout_constrained(K, k, x0, A, B, c, Ks, ks, alpha=1.0):
    """Roll out a linear equality-constrained policy and multiplier policy."""

    X, U = rollout(K, k, x0, A, B, c, alpha)
    Veq = vmap(lambda t: Ks[t] @ X[t] + alpha * ks[t])(np.arange(k.shape[0]))
    return X, U, Veq


rollout_equality = rollout_constrained
rollout_gpu_equality = rollout_gpu_constrained
