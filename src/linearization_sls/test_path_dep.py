import jax
jax.config.update("jax_platforms", "cpu")  # Force CPU for testing
import jax.numpy as jnp

import time

from src.interval import Interval
from src.polynomial import LinPoly
from src.taylor_model import LinTM
from src.helper import make_step_boxes, build_linear_tm, prepare_initial_set
from src.rhs_eval import build_auto_rhs_analytic_int


# State z = [q0, q1, q2, q3, w1, w2, w3]   (7,)
# Input v = [u1, u2, u3]                    (3,)
#
# Paper dynamics:
#   q_dot = Omega(w) q
#   w_dot = I^{-1} ( v - w x (I w) )
# with I = diag(5,2,1)

def satellite_ct_dynamics(x):
    # unpack state and input
    q0, q1, q2, q3, w1, w2, w3, u1, u2, u3 = [x[i] for i in range(10)]

    # --------------------------
    # Quaternion dynamics q_dot = Omega(w) q (expanded dim-wise)
    # Omega(w) = 0.5 * [[0,-w1,-w2,-w3],
    #                   [w1,0,w3,-w2],
    #                   [w2,-w3,0,w1],
    #                   [w3,w2,-w1,0]]
    # --------------------------
    dq0 = 0.5 * (      - w1*q1 - w2*q2 - w3*q3)
    dq1 = 0.5 * (w1*q0         + w3*q2 - w2*q3)
    dq2 = 0.5 * (w2*q0 - w3*q1         + w1*q3)
    dq3 = 0.5 * (w3*q0 + w2*q1 - w1*q2        )

    # --------------------------
    # Angular-rate dynamics:
    # w_dot = I^{-1}(v - w x (I w)), I = diag(5,2,1)
    # Iw = [5w1, 2w2, 1w3]
    # w x (Iw) =
    #   [ w2*w3*(1-2),  w1*w3*(5-1),  w1*w2*(2-5) ]
    # = [ -w2*w3,       4*w1*w3,      -3*w1*w2 ]
    #
    # Therefore:
    # dw1 = (u1 - (-w2*w3))/5 = (u1 + w2*w3)/5
    # dw2 = (u2 - 4*w1*w3)/2
    # dw3 = (u3 - (-3*w1*w2))/1 = u3 + 3*w1*w2
    # --------------------------
    dw1 = (u1 + w2 * w3) / 5.0
    dw2 = (u2 - 4.0 * w1 * w3) / 2.0
    dw3 = (u3 + 3.0 * w1 * w2)

    du1 = jnp.zeros_like(u1)
    du2 = jnp.zeros_like(u2)
    du3 = jnp.zeros_like(u3)

    return jnp.stack([dq0, dq1, dq2, dq3, dw1, dw2, dw3, du1, du2, du3], axis=0)


def rk4_step_ct(f_ct, x, h=1.0):
    """One RK4 step for x_dot = f_ct(x), with constant input over the step."""
    k1 = f_ct(x)
    k2 = f_ct(x + 0.5 * h * k1)
    k3 = f_ct(x + 0.5 * h * k2)
    k4 = f_ct(x + h * k3)
    x_next = x + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return x_next


def satellite_dt_dynamics(x, h=1.0):
    """
    Discrete-time dynamics from CT via RK4.
    Paper uses h = 1.0.
    """
    x_next = rk4_step_ct(satellite_ct_dynamics, x, h=h)
    return x_next


# Optional: add disturbance in the DT model (paper case study uses additive disturbance)
def satellite_dt_dynamics_with_w(x, w=None, h=1.0):
    """
    x_{k+1} = f_d(x_k) + w_k
    w: shape (10,), optional
    """
    x_next = satellite_dt_dynamics(x, h=h)
    if w is not None:
        x_next = x_next + w
    return x_next


def euler_zyx_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles (yaw-pitch-roll) to quaternion [q0,q1,q2,q3]
    with q0 = scalar part.
    """
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    # q = [w, x, y, z]
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return jnp.stack([q0, q1, q2, q3], axis=0)

def get_initial_state():
    # Paper initial Euler angles and angular rate (degrees -> radians)
    euler_deg = jnp.array([180.0, 45.0, 45.0])   # (roll, pitch, yaw) if using this convention
    euler_rad = euler_deg * jnp.pi / 180.0

    omega0 = jnp.array([-1.0, -4.5, 4.5]) * jnp.pi / 180.0  # rad/s

    # Initial quaternion from Euler angles
    q0 = euler_zyx_to_quat(euler_rad[0], euler_rad[1], euler_rad[2])

    # Initial state z0 = [q0,q1,q2,q3,w1,w2,w3]
    z0 = jnp.concatenate([q0, omega0], axis=0)

    return z0

def interval_mul_lo_up(a_lo, a_up, b_lo, b_up):
    p1 = a_lo * b_lo
    p2 = a_lo * b_up
    p3 = a_up * b_lo
    p4 = a_up * b_up
    lo = jnp.minimum(jnp.minimum(p1, p2), jnp.minimum(p3, p4))
    up = jnp.maximum(jnp.maximum(p1, p2), jnp.maximum(p3, p4))
    return lo, up


def remainder_bound_classic_hessian(rhs_int_fn, x_lo, x_up):
    x0 = 0.5 * (x_lo + x_up)                      # midpoint nominal
    d_lo, d_up = x_lo - x0, x_up - x0             # delta interval (n,)
    n = x_lo.shape[0]

    # Hessian interval for vector-output f: (m,n,n)
    X = Interval(x_lo, x_up)
    H = rhs_int_fn(X)
    H_lo, H_up = H.lo, H.hi

    # delta^2 interval (n,)
    d2_lo, d2_up = interval_mul_lo_up(d_lo, d_up, d_lo, d_up)

    # diagonal contribution: 0.5 * sum_k H[:,k,k] * d2[k]
    Hdiag_lo = jnp.diagonal(H_lo, axis1=1, axis2=2)  # (m,n)
    Hdiag_up = jnp.diagonal(H_up, axis1=1, axis2=2)  # (m,n)
    diag_lo, diag_up = interval_mul_lo_up(Hdiag_lo, Hdiag_up, d2_lo[None, :], d2_up[None, :])
    diag_lo = 0.5 * jnp.sum(diag_lo, axis=1)         # (m,)
    diag_up = 0.5 * jnp.sum(diag_up, axis=1)         # (m,)

    # cross contribution: sum_{j<k} H[:,j,k] * (d[j] d[k])
    # build dd interval (n,n)
    dj_lo, dj_up = d_lo[:, None], d_up[:, None]
    dk_lo, dk_up = d_lo[None, :], d_up[None, :]
    dd_lo, dd_up = interval_mul_lo_up(dj_lo, dj_up, dk_lo, dk_up)  # (n,n)

    cross_lo, cross_up = interval_mul_lo_up(H_lo, H_up, dd_lo[None, :, :], dd_up[None, :, :])  # (m,n,n)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    cross_lo = jnp.where(mask[None, :, :], cross_lo, 0.0)
    cross_up = jnp.where(mask[None, :, :], cross_up, 0.0)
    cross_lo = jnp.sum(cross_lo, axis=(1, 2))  # (m,)
    cross_up = jnp.sum(cross_up, axis=(1, 2))  # (m,)

    r_lo = diag_lo + cross_lo
    r_up = diag_up + cross_up
    return r_lo, r_up

def main():
    # Test at a random state
    state_dim = 7
    input_dim = 3
    z_eps = 0.05
    v_eps = 0.1

    dynamics = satellite_dt_dynamics

    x0 = jnp.concatenate([get_initial_state(), jnp.zeros(input_dim)], axis=0)  # initial state + zero input
    eps = jnp.concatenate([jnp.full(state_dim, z_eps), jnp.full(input_dim, v_eps)], axis=0)  # shape (10,)
    x0_lo = (x0 - eps)[None]
    x0_hi = (x0 + eps)[None]
    print(f"x0_lo: {x0_lo}")
    print(f"x0_hi: {x0_hi}")

    D = state_dim + input_dim
    V = D + 1

    # rhs_int_fn = build_auto_rhs_analytic_int(dynamics, D=D, V=V)
    # X = rhs_int_fn(Interval(x0_lo.repeat(2, axis=0), x0_hi.repeat(2, axis=0)))
    # print(f"Output interval shape: {X.lo.shape}")  # Should be (10,10,10)

    rhs_int_fn = build_auto_rhs_analytic_int(jax.hessian(dynamics), D=D, V=V)

    print("\nComputing classic interval-Hessian remainder bound...")
    r_lo, r_up = remainder_bound_classic_hessian(rhs_int_fn, x0_lo, x0_hi)
    print(f"r_lo: {r_lo}")
    print(f"r_up: {r_up}")


if __name__ == "__main__":
    main()
