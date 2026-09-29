import jax
# jax.config.update("jax_platforms", "cpu")  # Force CPU for testing
import jax.numpy as jnp

import time

from src.interval import Interval
from src.polynomial import LinPoly
from src.taylor_model import LinTM
from src.helper import make_step_boxes, build_linear_tm, prepare_initial_set
from src.rhs_eval import build_auto_rhs_analytic


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

def estimate_mu_satellite_mc(key, num_samples=10_000):
    """
    Monte Carlo estimate of mu in Eq. (9) for the satellite case.

    Assumes:
      - satellite_dt_dynamics(x_aug, h=...) is already defined
      - x_aug = [q0,q1,q2,q3,w1,w2,w3,u1,u2,u3] (shape (10,))
      - first 7 outputs are the physical next-state dimensions

    Sampling set P (satellite case):
      - q: random unit quaternion
      - w_i in [-0.1, 0.1]
      - u_i in [-0.1, 0.1]

    Direction h in Eq. (9):
      - sampled uniformly from [-1,1]^10 so ||h||_inf <= 1

    Returns:
      mu: shape (7,), where mu[i] ~= 0.5 * max |h^T H_i(xi) h|
    """

    # ---- sample xi = [q,w,u] in P ----
    kq, kw, ku, kh = jax.random.split(key, 4)

    # unit quaternion samples (Gaussian then normalize)
    # q = jax.random.normal(kq, (num_samples, 4))
    # q = q / (jnp.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    q = jax.random.uniform(kq, (num_samples, 4), minval=-1.0, maxval=1.0)

    # angular rate and input samples
    w = jax.random.uniform(kw, (num_samples, 3), minval=-0.1, maxval=0.1)
    u = jax.random.uniform(ku, (num_samples, 3), minval=-0.1, maxval=0.1)

    # xi samples in P, shape (N,10)
    xi_batch = jnp.concatenate([q, w, u], axis=1)

    # h samples for quadratic form, shape (N,10), ||h||_inf <= 1
    # h_batch = jax.random.uniform(kh, (num_samples, 10), minval=-1.0, maxval=1.0)
    # h as Rademacher (+/-1), better for max over ||h||_inf <= 1
    h_batch = jnp.where(
        jax.random.bernoulli(kh, p=0.5, shape=(num_samples, 10)),
        1.0,
        -1.0,
    )

    # physical DT map (first 7 dims only)
    def f_phys(x_aug):
        return satellite_dt_dynamics(x_aug)[:7]

    # estimate each mu_i independently
    mu_list = []
    for i in range(10):
        # scalar output f_i
        fi = lambda x: f_phys(x)[i]

        # Hessian H_i(x) wrt augmented variable [q,w,u] (10D)
        hess_fi = jax.hessian(fi)

        # evaluate 0.5 * |h^T H_i(xi) h| for one sample
        def one_sample_val(xi, hvec):
            H = hess_fi(xi)               # (10,10)
            return 0.5 * jnp.abs(hvec @ H @ hvec)

        # vectorize over samples and take max
        vals = jax.vmap(one_sample_val)(xi_batch, h_batch)  # (N,)
        mu_i = jnp.max(vals)
        mu_list.append(mu_i)

    mu = jnp.stack(mu_list, axis=0)  # shape (7,)
    return mu

def estimate_mu(dynamics, global_x_lo, global_x_hi, key, num_samples: int):
    """
    Monte Carlo estimate of mu in Eq. (9) for a general discrete-time dynamics map.

    Args:
      dynamics: callable f(x) -> y, with x shape (n,), y shape (m,)
      global_x_lo: array, shape (n,), lower bound of sampling box P
      global_x_hi: array, shape (n,), upper bound of sampling box P
      key: jax.random.PRNGKey
      num_samples: number of MC samples

    Returns:
      mu: array, shape (m,), where
          mu[i] ~= 0.5 * max_{samples} | h^T H_i(x) h |
          with x ~ Uniform(P), h ~ Rademacher({-1,+1}^n) (so ||h||_inf = 1)
    """
    assert global_x_lo.shape == global_x_hi.shape

    # infer dims
    n = global_x_lo.shape[0]
    kx, kh = jax.random.split(key, 2)

    # x ~ Uniform([lo,hi])
    x_batch = jax.random.uniform(
        kx, (num_samples, n),
        minval=global_x_lo[None, :],
        maxval=global_x_hi[None, :],
    )

    # h ~ Rademacher (+/-1), good for max over ||h||_inf <= 1
    h_batch = jnp.where(
        jax.random.bernoulli(kh, p=0.5, shape=(num_samples, n)),
        1.0,
        -1.0,
    ).astype(x_batch.dtype)

    # Hessian of vector-output dynamics: H(x) has shape (m, n, n)
    hess_dyn = jax.jit(jax.jacfwd(jax.jacrev(dynamics)))

    # H_b: (B, m, n, n)
    H_batch = jax.vmap(hess_dyn)(x_batch)

    # quad_b: (B, m), where quad_b[b,i] = h_b^T H_i(x_b) h_b
    quad = jnp.einsum("ni,nmij,nj->nm", h_batch, H_batch, h_batch)

    vals_b = 0.5 * jnp.abs(quad)  # (B, m)
    mu = jnp.max(vals_b, axis=0)

    return mu

def estimate_remainder_paper(dynamics, mu, x_lo, x_up, state_dim):
    def _f(x_lo, x_up):
        c = 0.5 * (x_lo + x_up)
        S = 0.5 * (x_up - x_lo)
        c_next = dynamics(c)
        J = jax.jacfwd(dynamics)(c)
        A = J[:state_dim, :state_dim]
        B = J[:state_dim, state_dim:]
        tau = jnp.max(S, axis=-1)
        r_bound = mu * tau**2
        return r_bound, c_next, A, B
    
    r_bound, c, A, B = jax.vmap(_f)(x_lo, x_up)
    return_dict = {
        "c_paper": c,
        "A_paper": A,
        "B_paper": B,
    }

    return r_bound[..., :state_dim], return_dict

def estimate_remainder_sampling(f, x_lo, x_up, state_dim, num_samples=1000, key=jax.random.PRNGKey(0)):
    """
    Estimate nonlinear remainder r by sampling in boxes [x_lo, x_up] (batched).

      x_nom = (x_lo + x_up) / 2
      J = df/dx at x_nom
      r_b(x) = f(x) - f(x_nom) - J @ (x - x_nom)

    Args:
      f: callable f(x)->y, x shape (D,), y shape (M,)
      x_lo, x_up: shape (B, D)
      num_samples: samples per box
      key: PRNGKey

    Returns:
      r_bound: shape (B, M), componentwise sampled bound per batch
    """
    assert x_lo.shape == x_up.shape
    B, D = x_lo.shape

    x_nom = 0.5 * (x_lo + x_up)  # (B,D)

    # per-batch nominal outputs and Jacobians
    f_nom = jax.vmap(f)(x_nom)                       # (B,M)
    J = jax.vmap(jax.jacfwd(f))(x_nom)               # (B,M,D)

    # sample u ~ U[0,1] for each batch box: (B,N,D)
    u01 = jax.random.uniform(key, shape=(B, num_samples, D), minval=0.0, maxval=1.0)
    x_samples = x_lo[:, None, :] + (x_up - x_lo)[:, None, :] * u01  # (B,N,D)

    # remainder for one batch element (vectorize over N)
    def remainder_for_batch(x_nom_b, f_nom_b, J_b, x_samps_b):
        # x_samps_b: (N,D)
        def one_r(x):
            dx = x - x_nom_b                      # (D,)
            return f(x) - f_nom_b - J_b @ dx      # (M,)
        r_samps = jax.vmap(one_r)(x_samps_b)      # (N,M)
        r_min = jnp.min(r_samps, axis=0)          # (M,)
        r_max = jnp.max(r_samps, axis=0)          # (M,)
        return r_min, r_max   # (M,)

    r_min, r_max = jax.vmap(remainder_for_batch)(x_nom, f_nom, J, x_samples)  # (B,M)
    r_bound = jnp.maximum(jnp.abs(r_min), jnp.abs(r_max))

    # calculate the lower and upper bounds
    x_out = jax.vmap(f)(x_samples.reshape(-1, D)).reshape((B, num_samples, -1))  # (B,N,M)
    x_out_min = jnp.min(x_out, axis=1)  # (B,M)
    x_out_max = jnp.max(x_out, axis=1)  # (B,M)

    return_dict = {
        "output_lb": x_out_min[:, :state_dim],
        "output_ub": x_out_max[:, :state_dim],
    }


    return r_bound[..., :state_dim], return_dict

def estimate_remainder_TM(tm_fn, x_lo, x_up, state_dim, splits_cfg={}):
    B, D = x_lo.shape
    step_lo, step_hi, _, _ = make_step_boxes(B, D, h=1.0)
    # calculate TM around the nominal point
    c = 0.5 * (x_lo + x_up)
    S = 0.5 * (x_up - x_lo)
    x_tm0 = build_linear_tm(c, S)
    x_tmnext_nom = tm_fn(x_tm0, step_lo, step_hi)
    c_tm = x_tmnext_nom.P.c
    J_tm = x_tmnext_nom.P.L[:, :, 1:] / S
    A_tm = J_tm[:, :state_dim, :state_dim]
    B_tm = J_tm[:, :state_dim, state_dim:]
    # remainder = x_tmnext_nom.R
    # r_bound = jnp.maximum(jnp.abs(remainder.lo), jnp.abs(remainder.hi))
    # return r_bound[..., :state_dim], c_tm, A_tm, B_tm

    # split initial box to refine remainder estimate
    step_lo_split, step_hi_split = prepare_initial_set(step_lo[:, 1:], step_hi[:, 1:], splits_cfg)
    step_lo_split = jnp.concatenate([jnp.zeros((step_lo_split.shape[0], 1)), step_lo_split], axis=-1)  # add back addtional time dim.
    step_hi_split = jnp.concatenate([jnp.zeros((step_lo_split.shape[0], 1)), step_hi_split], axis=-1)
    n_splits = step_lo_split.shape[0] // B
    x_tm0_split = LinTM.repeat(x_tm0, n_splits)  # repeat nominal TM for each split box
    x_tmnext_split:LinTM = tm_fn(x_tm0_split, step_lo_split, step_hi_split)
    remainder = x_tmnext_split.R
    agg_r_lb = jnp.min(remainder.lo.reshape((B, -1, D)), axis=1)  # aggregate lower bounds over all splits
    agg_r_ub = jnp.max(remainder.hi.reshape((B, -1, D)), axis=1)
    r_bound = jnp.maximum(jnp.abs(agg_r_lb), jnp.abs(agg_r_ub))

    output_interval = x_tmnext_split.eval_interval(step_lo, step_hi)
    agg_output_lb = jnp.min(output_interval.lo.reshape((B, -1, D)), axis=1)
    agg_output_ub = jnp.max(output_interval.hi.reshape((B, -1, D)), axis=1)

    return_dict = {
        "c_tm": c_tm,
        "A_tm": A_tm,
        "B_tm": B_tm,
        "output_lb": agg_output_lb[:, :state_dim],
        "output_ub": agg_output_ub[:, :state_dim],
    }

    return r_bound[..., :state_dim], return_dict

    # x_lo_split, x_up_split = prepare_initial_set(x_lo, x_up, splits_cfg)
    # c_split = 0.5 * (x_lo_split + x_up_split)
    # S_split = 0.5 * (x_up_split - x_lo_split)
    # x_tm0_split = build_linear_tm(c_split, S_split)
    # x_tmnext_nom_split = tm_fn(x_tm0_split, step_lo, step_hi)
    # agg_r_lb = jnp.min(x_tmnext_split.eval_interval(step_lo_split, step_hi_split).lo.reshape((B, -1, D)), axis=1)
    # agg_r_ub = jnp.max(x_tmnext_split.eval_interval(step_lo_split, step_hi_split).hi.reshape((B, -1, D)), axis=1)

def estimate_remainder_TM_new(tm_fn, x_lo, x_up, state_dim, splits_cfg={}):
    B, D = x_lo.shape
    step_lo, step_hi, _, _ = make_step_boxes(B, D, h=1.0)
    x_lo_split, x_up_split = prepare_initial_set(x_lo, x_up, splits_cfg)
    c_split = 0.5 * (x_lo_split + x_up_split)
    S_split = 0.5 * (x_up_split - x_lo_split)
    x_tm0_split = build_linear_tm(c_split, S_split)
    x_tmnext_nom_split:LinTM = tm_fn(x_tm0_split, step_lo, step_hi)
    remainder = x_tmnext_nom_split.R
    agg_r_lb = jnp.min(remainder.lo.reshape((B, -1, D)), axis=1)  # aggregate lower bounds over all splits
    agg_r_ub = jnp.max(remainder.hi.reshape((B, -1, D)), axis=1)
    r_bound = jnp.maximum(jnp.abs(agg_r_lb), jnp.abs(agg_r_ub))

    output_interval = x_tmnext_nom_split.eval_interval(step_lo, step_hi)
    agg_output_lb = jnp.min(output_interval.lo.reshape((B, -1, D)), axis=1)
    agg_output_ub = jnp.max(output_interval.hi.reshape((B, -1, D)), axis=1)

    return_dict = {
        "c_tm": None,
        "A_tm": None,
        "B_tm": None,
        "output_lb": agg_output_lb[:, :state_dim],
        "output_ub": agg_output_ub[:, :state_dim],
    }

    return r_bound[..., :state_dim], return_dict

def main():
    # Test at a random state
    state_dim = 7
    input_dim = 3
    z_eps = 0.05
    v_eps = 0.1
    batch_size = 1

    dynamics = satellite_dt_dynamics

    x0 = jnp.concatenate([get_initial_state(), jnp.zeros(input_dim)], axis=0)  # initial state + zero input

    D = state_dim + input_dim
    V = D + 1
    rhs_tm_fn = build_auto_rhs_analytic(dynamics, D=D, V=V)

    eps = jnp.concatenate([jnp.full(state_dim, z_eps), jnp.full(input_dim, v_eps)], axis=0)  # shape (10,)
    # eps = eps.at[0].set(0.1)
    print(f"eps: {eps.tolist()}")

    x0_lo = (x0 - eps)[None].repeat(batch_size, axis=0)  # shape (batch_size, 10)
    x0_hi = (x0 + eps)[None].repeat(batch_size, axis=0)  # shape (batch_size, 10)

    # key = jax.random.PRNGKey(0)
    # key, subkey = jax.random.split(key)
    # # mu_estimate = estimate_mu_satellite_mc(subkey, num_samples=10000)
    # # global_x_lo = jnp.array([-1, -1, -1, -1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])[None]
    # # global_x_hi = jnp.array([1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[None]
    # global_x_lo = x0_lo
    # global_x_hi = x0_hi
    # mu = estimate_mu(dynamics, global_x_lo[0], global_x_hi[0], subkey, num_samples=10000)[:state_dim]
    # print("Estimated mu from Monte Carlo:\n", mu)

    mu = jnp.array([3.699,3.703,3.717,3.635,0.649,4.608,5.635])
    # 10K: [3.3224475 3.374932  3.3931732 3.296236  0.5580268 4.4366107 5.3266587]
    # 100K:[3.459942  3.4696774 3.4886975 3.4031327 0.5819119 4.5157175 5.439615 ]
    # 10Kq:[3.7818341 4.005538  4.254002  3.7110872 0.5580268 4.4366107 5.3266587]

    r_bound_paper, return_dict_paper = estimate_remainder_paper(dynamics, mu, x0_lo, x0_hi, state_dim)
    c_paper, A_paper, B_paper = return_dict_paper["c_paper"], return_dict_paper["A_paper"], return_dict_paper["B_paper"]

    
    # print("Center from paper:\n", c_paper.tolist())
    # print("Jacobian A from paper:\n", A_paper.tolist())
    # print("Jacobian B from paper:\n", B_paper.tolist())

    splits_cfg = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}
    # splits_cfg = {4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}
    # splits_cfg = {7: 8, 8: 8, 9: 8}
    # splits_cfg = {4: 8, 5: 8, 6: 8}
    # splits_cfg = {0: 8, 1: 8, 2: 8, 3: 8}
    # splits_cfg = {0: 1024}
    # splits_cfg = {}
    # splits_cfg = {0: 2}

    print(f"splits_cfg: {splits_cfg}")

    @jax.jit
    def jit_estimate_remainder_TM(x_lo, x_up):
        return estimate_remainder_TM(rhs_tm_fn, x_lo, x_up, state_dim, splits_cfg)

    compile_start_time = time.time()
    r_bound_tm, return_dict_tm = jit_estimate_remainder_TM(x0_lo, x0_hi)
    jax.block_until_ready(r_bound_tm)
    compile_end_time = time.time()
    print(f"Compile time: {compile_end_time - compile_start_time} seconds")

    run_start_time = time.time()
    r_bound_tm, return_dict_tm = jit_estimate_remainder_TM(x0_lo, x0_hi)
    jax.block_until_ready(r_bound_tm)
    run_end_time = time.time()
    print(f"Run time: {(run_end_time - run_start_time) * 1000 } milliseconds")

    c_tm, A_tm, B_tm = return_dict_tm["c_tm"], return_dict_tm["A_tm"], return_dict_tm["B_tm"]
    output_lb_tm, output_ub_tm = return_dict_tm["output_lb"], return_dict_tm["output_ub"]

    # print("Center from TM:\n", c_tm.tolist())
    # print("Jacobian A from TM:\n", A_tm.tolist())
    # print("Jacobian B from TM:\n", B_tm.tolist())

    assert jnp.allclose(c_paper, c_tm), "Center mismatch between paper and TM"
    assert jnp.allclose(A_paper, A_tm), "Jacobian A mismatch between paper and TM"
    assert jnp.allclose(B_paper, B_tm), "Jacobian B mismatch between paper and TM"

    r_bound_sampling, return_dict_sampling = estimate_remainder_sampling(dynamics, x0_lo, x0_hi, state_dim, num_samples=10000, key=jax.random.PRNGKey(1))

    print("Remainder bound from sampling:\n", r_bound_sampling.tolist())
    print("Remainder bound from paper:\n", r_bound_paper.tolist())
    print("Remainder bound from TM:\n", r_bound_tm.tolist())

    # @jax.jit
    # def jit_estimate_remainder_TM_new(x_lo, x_up):
    #     return estimate_remainder_TM_new(rhs_tm_fn, x_lo, x_up, state_dim, splits_cfg)


    # compile_start_time = time.time()
    # r_bound_tm_new, return_dict_tm_new = jit_estimate_remainder_TM_new(x0_lo, x0_hi)
    # jax.block_until_ready(r_bound_tm_new)
    # compile_end_time = time.time()
    # print(f"Compile time for TM new: {compile_end_time - compile_start_time} seconds")

    # run_start_time = time.time()
    # r_bound_tm_new, return_dict_tm_new = jit_estimate_remainder_TM_new(x0_lo, x0_hi)
    # jax.block_until_ready(r_bound_tm_new)
    # run_end_time = time.time()
    # print(f"Run time for TM new: {(run_end_time - run_start_time) * 1000 } milliseconds")

    # c_tm_new, A_tm_new, B_tm_new = return_dict_tm_new["c_tm"], return_dict_tm_new["A_tm"], return_dict_tm_new["B_tm"]
    # output_lb_tm_new, output_ub_tm_new = return_dict_tm_new["output_lb"], return_dict_tm_new["output_ub"]

    # print("Output bounds from sampling:\n", return_dict_sampling["output_lb"].tolist())
    # print("Output lb from TM:\n", output_lb_tm.tolist())
    # print("Output lb from TM new:\n", output_lb_tm_new.tolist())
    
    # print("Output bounds from sampling:\n", return_dict_sampling["output_ub"].tolist())
    # print("Output ub from TM:\n", output_ub_tm.tolist())
    # print("Output ub from TM new:\n", output_ub_tm_new.tolist())


if __name__ == "__main__":
    main()