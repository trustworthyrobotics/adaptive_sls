"""Shared model and utilities for the adaptive quadrotor experiments.

Physical state:
    [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]

Adaptive state:
    [x_phys, Fx, Fy, Fz]

The nominal model carries a constant wind-force estimate.  The executed plant
uses ``TRUE_FORCE``.  Both use the reference quadrotor's ZYX Euler convention
and thrust/gravity signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from jax import config as jax_config
import matplotlib.pyplot as plt
import numpy as np

from gpu_sls.gpu_sls import (
    adaptive_posterior_factors,
    get_adaptive_betas,
    observable_subspace_factorizer,
)

jax_config.update("jax_enable_x64", False)


N_PHYS = 12
N_THETA = 3
N_AUG = N_PHYS + N_THETA
N_CONTROL = 4

MASS = 1.0
GRAVITY = 9.81
JX = 0.02
JY = 0.02
JZ = 0.04
INERTIA = jnp.diag(jnp.array([JX, JY, JZ], dtype=jnp.float32))
INERTIA_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=jnp.float32))

DT = 0.03
FULL_HORIZON = 110
MPC_HORIZON = 30
MAX_MPC_STEPS = 110
GOAL_TOL = 0.25

NOMINAL_FORCE = jnp.zeros((N_THETA,), dtype=jnp.float32)
TRUE_FORCE = jnp.array([0.20, -0.15, 0.10], dtype=jnp.float32) * 1
FORCE_HALF_WIDTH = jnp.array([0.125, 0.125, 0.125], dtype=jnp.float32)

DISTURBANCE_MAGNITUDE = 0.06
ROLLOUT_DISTURBANCE_SCALE = 1.0
CORNER_ROLLOUTS_PER_CORNER = 10
NUM_ADVERSARIAL_ROLLOUTS = (2**N_THETA) * CORNER_ROLLOUTS_PER_CORNER
NUM_RANDOM_ROLLOUTS = 120

X0_PHYS = jnp.array(
    [
        -0.75, -0.75, 0.4,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ],
    dtype=jnp.float32,
)
X_GOAL_PHYS = jnp.array(
    [
        1.0, 0.8, 0.4,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ],
    dtype=jnp.float32,
)

PHYSICAL_STATE_WEIGHTS = jnp.array(
    [
        25.0, 25.0, 25.0,
        2.0, 2.0, 0.5,
        0.5, 0.5, 0.5,
        0.05, 0.05, 0.05,
    ],
    dtype=jnp.float32,
)
CONTROL_WEIGHTS = jnp.array([0.001, 0.01, 0.01, 0.01], dtype=jnp.float32)
W = jnp.concatenate(
    [PHYSICAL_STATE_WEIGHTS, jnp.zeros((N_THETA,), dtype=jnp.float32), CONTROL_WEIGHTS]
)
R_BAR = 0.5 * jnp.eye(N_CONTROL, dtype=jnp.float32)

T_HOVER = MASS * GRAVITY
TAU_MAX = 10.0
U_MIN = jnp.array([0.0, -TAU_MAX, -TAU_MAX, -TAU_MAX], dtype=jnp.float32)
U_MAX = jnp.array([2.0 * T_HOVER, TAU_MAX, TAU_MAX, TAU_MAX], dtype=jnp.float32)

X_MAX_PHYS = jnp.array(
    [
        15.0, 15.0, 15.0,
        jnp.pi / 2.0, jnp.pi / 2.0, 10.0 * jnp.pi,
        5.0, 5.0, 5.0,
        8.0, 8.0, 8.0,
    ],
    dtype=jnp.float32,
)
X_MIN_PHYS = (-X_MAX_PHYS).at[2].set(-1.0)
X_MIN = jnp.concatenate([X_MIN_PHYS, -FORCE_HALF_WIDTH])
X_MAX = jnp.concatenate([X_MAX_PHYS, FORCE_HALF_WIDTH])

OBSTACLES = jnp.array([[0.1, -0.1, 0.25]], dtype=jnp.float32)
MEASUREMENT_MATRIX_C = jnp.eye(N_PHYS, dtype=jnp.float32)


@dataclass(frozen=True)
class QuadrotorConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray
    dt: float


def rotation_matrix(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """Body-to-world rotation Rz(psi) Ry(theta) Rx(phi)."""
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    return jnp.array(
        [
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth, cth * sphi, cth * cphi],
        ],
        dtype=phi.dtype,
    )


def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Map [p, q, r] to [phi_dot, theta_dot, psi_dot]."""
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    ttheta = jnp.tan(theta)
    ctheta = jnp.cos(theta)
    return jnp.array(
        [
            [1.0, sphi * ttheta, cphi * ttheta],
            [0.0, cphi, -sphi],
            [0.0, sphi / ctheta, cphi / ctheta],
        ],
        dtype=phi.dtype,
    )


def physical_step(
    x_phys: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Forward-Euler step of the reference rigid-body model plus wind force."""
    phi, theta, psi = x_phys[3], x_phys[4], x_phys[5]
    vx, vy, vz = x_phys[6], x_phys[7], x_phys[8]
    p, q, r = x_phys[9], x_phys[10], x_phys[11]
    thrust = u[0]
    tau_phi, tau_theta, tau_psi = u[1], u[2], u[3]

    # These scalar expressions are algebraically identical to
    # Rz(psi) Ry(theta) Rx(phi) @ [0, 0, thrust].  Keeping them scalar also
    # lets the optional interval-Hessian linearization-error path trace them.
    phi_dot = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = (q * jnp.sin(phi) + r * jnp.cos(phi)) / jnp.cos(theta)

    vx_dot = (
        thrust
        * (jnp.sin(phi) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi) * jnp.sin(theta))
        / MASS
        + force[0] / MASS
    )
    vy_dot = (
        thrust
        * (-jnp.sin(phi) * jnp.cos(psi) + jnp.cos(phi) * jnp.sin(psi) * jnp.sin(theta))
        / MASS
        + force[1] / MASS
    )
    vz_dot = thrust * jnp.cos(phi) * jnp.cos(theta) / MASS - GRAVITY + force[2] / MASS

    p_dot = ((JY - JZ) / JX) * q * r + tau_phi / JX
    q_dot = ((JZ - JX) / JY) * p * r + tau_theta / JY
    r_dot = ((JX - JY) / JZ) * p * q + tau_psi / JZ

    x_dot = jnp.stack(
        [
            vx,
            vy,
            vz,
            phi_dot,
            theta_dot,
            psi_dot,
            vx_dot,
            vy_dot,
            vz_dot,
            p_dot,
            q_dot,
            r_dot,
        ]
    )
    return x_phys + jnp.asarray(dt, dtype=x_phys.dtype) * x_dot


def dynamics(
    x_aug: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: float,
) -> jnp.ndarray:
    """Adaptive dynamics with constant force-estimate states."""
    del t
    force = x_aug[N_PHYS:]
    x_next_phys = physical_step(x_aug[:N_PHYS], u, force, parameter)
    return jnp.concatenate([x_next_phys, force])


def nonadaptive_dynamics(
    x_phys: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    nominal_param: jnp.ndarray,
    *,
    parameter: float,
) -> jnp.ndarray:
    """Physical nominal model with a fixed, non-learning force parameter."""
    del t
    return physical_step(x_phys, u, nominal_param, parameter)


def cost(W: jnp.ndarray, reference: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
    """Reference cost from the original quadrotor example, lifted to 15 states."""
    x_ref = reference[t]
    state_weights = W[:N_AUG]
    control_weights = W[N_AUG:]

    delta = x - x_ref
    angle_cost = jnp.sum(
        state_weights[3:6] * (1.0 - jnp.cos(delta[3:6]))
    )
    state_cost = (
        jnp.sum(state_weights[:3] * delta[:3] ** 2)
        + angle_cost
        + jnp.sum(state_weights[6:N_AUG] * delta[6:N_AUG] ** 2)
    )
    delta_u = u - jnp.array([T_HOVER, 0.0, 0.0, 0.0], dtype=x.dtype)
    return state_cost + jnp.sum(control_weights * delta_u**2)


def nonadaptive_cost(
    W: jnp.ndarray,
    reference: jnp.ndarray,
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
):
    """Physical-state version of the quadrotor tracking cost."""
    x_ref = reference[t]
    delta = x - x_ref
    angle_cost = jnp.sum(
        W[3:6] * (1.0 - jnp.cos(delta[3:6]))
    )
    state_cost = (
        jnp.sum(W[:3] * delta[:3] ** 2)
        + angle_cost
        + jnp.sum(W[6:N_PHYS] * delta[6:N_PHYS] ** 2)
    )
    delta_u = u - jnp.array([T_HOVER, 0.0, 0.0, 0.0], dtype=x.dtype)
    return state_cost + jnp.sum(W[N_PHYS:] * delta_u**2)


def make_state_box_constraints(x_min: jnp.ndarray, x_max: jnp.ndarray) -> Callable:
    """Constrain physical states only; adaptive force states remain unconstrained."""
    x_min_phys = x_min[:N_PHYS]
    x_max_phys = x_max[:N_PHYS]

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del u, t
        x_phys = x[:N_PHYS]
        return jnp.concatenate([x_phys - x_max_phys, x_min_phys - x_phys])

    return constraints


def make_control_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray) -> Callable:
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del x, t
        return jnp.concatenate([u - u_max, u_min - u])

    return constraints


def make_exogenous_disturbance(
    dt: float = DT,
    information_center_z: float | None = None,
    disturbance_z_off: float | None = None,
    disturbance_z_sharpness: float = 10.0,
) -> Callable:
    """Build the 12-dimensional exogenous disturbance map.

    The disturbance depends only on altitude. ``disturbance_z_off`` is the
    midpoint of the smooth tanh transition; the disturbance is effectively
    off above that transition. ``information_center_z`` is accepted as the
    default midpoint for convenience; there is no XY center.
    """
    alpha = DISTURBANCE_MAGNITUDE * dt

    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        identity = jnp.eye(N_PHYS, dtype=X_prefix.dtype)
        if information_center_z is None and disturbance_z_off is None:
            E = alpha * identity
            # Keep the constant map connected to the traced input. Without
            # this zero-valued dependency, the interval interpreter sees a
            # constant 144-vector and incorrectly tries to broadcast it to
            # the 12-vector input shape.
            X_phys = X_prefix[..., :N_PHYS]
            zero_from_state = 0.0 * jnp.reshape(
                X_phys,
                (X_phys.shape[0], N_PHYS, 1),
            )
            return (
                jnp.broadcast_to(E, (X_prefix.shape[0], N_PHYS, N_PHYS))
                + zero_from_state
            )

        z_off_value = (
            disturbance_z_off
            if disturbance_z_off is not None
            else information_center_z
        )
        z_off = jnp.asarray(z_off_value, dtype=X_prefix.dtype)
        gate = 0.5 * (
            1.0
            - jnp.tanh(
                jnp.asarray(disturbance_z_sharpness, dtype=X_prefix.dtype)
                * (X_prefix[:, 2] - z_off)
            )
        )
        return alpha * jnp.einsum("t,ij->tij", gate, identity)

    return disturbance


def make_nonadaptive_disturbance(
    dt: float = DT,
) -> tuple[Callable, Callable, Callable]:
    """Return combined, sensitivity, and exogenous maps for non-adaptive SLS."""
    exogenous_disturbance = make_exogenous_disturbance(dt)

    def sensitivity(
        X_prefix: jnp.ndarray,
        U_prefix: jnp.ndarray,
        nominal_force: jnp.ndarray,
    ) -> jnp.ndarray:
        # Trajectory calls provide T controls for T+1 states, whereas the
        # interval tracer provides one already-aligned state/control pair.
        pad_len = X_prefix.shape[0] - U_prefix.shape[0]
        U_pad = (
            jnp.pad(U_prefix, ((0, pad_len), (0, 0)))
            if pad_len > 0
            else U_prefix
        )
        X_phys = X_prefix[..., :N_PHYS]

        def step_sensitivity(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            return jax.jacfwd(
                lambda force: physical_step(x, u, force, dt)
            )(nominal_force)

        return jax.vmap(step_sensitivity)(X_phys, U_pad)

    def combined_disturbance(
        X_prefix: jnp.ndarray,
        U_prefix: jnp.ndarray,
        nominal_force: jnp.ndarray,
        G_0: jnp.ndarray,
    ) -> jnp.ndarray:
        parameter_disturbance = sensitivity(
            X_prefix,
            U_prefix,
            nominal_force,
        ) @ G_0
        return jnp.concatenate(
            [exogenous_disturbance(X_prefix), parameter_disturbance],
            axis=-1,
        )

    return combined_disturbance, sensitivity, exogenous_disturbance


def build_physical_reference(
    x_start: jnp.ndarray,
    x_goal: jnp.ndarray,
    horizon: int,
    dt: float = DT,
) -> jnp.ndarray:
    progress = jnp.linspace(0.0, 1.0, horizon + 1, dtype=x_start.dtype)
    reference = jnp.zeros((horizon + 1, N_PHYS), dtype=x_start.dtype)
    position = (1.0 - progress[:, None]) * x_start[:3] + progress[:, None] * x_goal[:3]
    yaw = x_start[5] + progress * (x_goal[5] - x_start[5])
    velocity = (x_goal[:3] - x_start[:3]) / (horizon * dt)
    reference = reference.at[:, :3].set(position)
    reference = reference.at[:, 5].set(yaw)
    reference = reference.at[:, 6:9].set(velocity)
    return reference


def augment_reference(reference_phys: jnp.ndarray, force_estimate: jnp.ndarray) -> jnp.ndarray:
    force_ref = jnp.broadcast_to(force_estimate, (reference_phys.shape[0], N_THETA))
    return jnp.concatenate([reference_phys, force_ref], axis=-1)


def make_seed(reference_aug: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    U_seed = jnp.zeros((reference_aug.shape[0] - 1, N_CONTROL), dtype=reference_aug.dtype)
    U_seed = U_seed.at[:, 0].set(T_HOVER)
    return reference_aug, U_seed


def make_config(horizon: int, dt: float = DT) -> QuadrotorConfig:
    return QuadrotorConfig(
        n=N_AUG,
        nu=N_CONTROL,
        N=horizon,
        W=W,
        u_ref=jnp.array([T_HOVER, 0.0, 0.0, 0.0], dtype=jnp.float32),
        dt=dt,
    )


def make_nonadaptive_config(horizon: int, dt: float = DT) -> QuadrotorConfig:
    return QuadrotorConfig(
        n=N_PHYS,
        nu=N_CONTROL,
        N=horizon,
        W=jnp.concatenate([PHYSICAL_STATE_WEIGHTS, CONTROL_WEIGHTS]),
        u_ref=jnp.array([T_HOVER, 0.0, 0.0, 0.0], dtype=jnp.float32),
        dt=dt,
    )


@partial(jax.jit, static_argnames=("q_max",))
def girard_reduce(G: jnp.ndarray, q_max: int) -> jnp.ndarray:
    n_theta = G.shape[0]
    keep_count = n_theta * (q_max - 1)
    target_count = n_theta * q_max
    order = jnp.argsort(jnp.linalg.norm(G, axis=0))[::-1]
    sorted_G = G[:, order]
    kept = sorted_G[:, :keep_count]
    tail = jnp.diag(jnp.sum(jnp.abs(sorted_G[:, keep_count:]), axis=1))
    reduced = jnp.concatenate([kept, tail], axis=1)
    return jnp.pad(reduced, ((0, 0), (0, target_count - reduced.shape[1])))


@jax.jit
def observable_residual_map(
    E_force: jnp.ndarray,
    C: jnp.ndarray,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-10,
) -> jnp.ndarray:
    """Compute ``(C E_force)^dagger C`` using an SVD rank threshold."""
    H = C @ E_force
    U, singular_values, Vh = jnp.linalg.svd(H, full_matrices=False)
    tolerance = jnp.maximum(
        jnp.asarray(atol, dtype=H.dtype),
        jnp.asarray(rtol, dtype=H.dtype) * jnp.max(singular_values),
    )
    inverse_singular_values = jnp.where(
        singular_values >= tolerance,
        1.0 / singular_values,
        0.0,
    )
    H_pinv = (Vh.T * inverse_singular_values) @ U.T
    return H_pinv @ C


def make_estimator_update(
    disturbance: Callable,
    *,
    dt: float = DT,
    q_max: int = 20,
    C: jnp.ndarray = MEASUREMENT_MATRIX_C,
) -> Callable:
    """Build the signed-force observable-subspace set-membership update."""

    @jax.jit
    def estimator_update(
        previous_x: jnp.ndarray,
        u: jnp.ndarray,
        x_next: jnp.ndarray,
        G: jnp.ndarray,
        linearization_state: jnp.ndarray,
        linearization_control: jnp.ndarray,
        measurement_disturbance: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta = previous_x[N_PHYS:]

        def predicted_physical_state(candidate_theta: jnp.ndarray) -> jnp.ndarray:
            return physical_step(previous_x[:N_PHYS], u, candidate_theta, dt)

        def planned_physical_state(candidate_theta: jnp.ndarray) -> jnp.ndarray:
            return physical_step(
                linearization_state[:N_PHYS],
                linearization_control,
                candidate_theta,
                dt,
            )

        prediction = predicted_physical_state(theta)
        E_force = jax.jacfwd(planned_physical_state)(theta)
        P_obs, Pi, W_obs = observable_subspace_factorizer(
            E_force,
            measurement_disturbance,
            C,
        )
        _, L, G_next, _ = adaptive_posterior_factors(
            G,
            Pi,
            W_obs,
            gain_regularization=1.0e-7,
            enable_posterior_gate=True,
        )

        innovation = P_obs @ (x_next[:N_PHYS] - prediction)
        theta_next = theta + L @ innovation
        G_next = girard_reduce(G_next, q_max=q_max)
        x_updated = x_next.at[N_PHYS:].set(theta_next)
        singular_values = jnp.linalg.svd(C @ E_force, compute_uv=False)
        return x_updated, G_next, singular_values, innovation, L

    return estimator_update


def initial_augmented_state() -> jnp.ndarray:
    return jnp.concatenate([X0_PHYS, NOMINAL_FORCE])


def initial_force_generator(q_max: int) -> jnp.ndarray:
    G = jnp.diag(FORCE_HALF_WIDTH)
    return jnp.pad(G, ((0, 0), (0, N_THETA * q_max - N_THETA)))


def reached_goal(x: jnp.ndarray, tolerance: float = GOAL_TOL) -> bool:
    delta = x[:3] - X_GOAL_PHYS[:3]
    return bool(delta @ delta <= tolerance**2)


def state_feedback_gains(
    Phi_x: jnp.ndarray,
    Phi_u: jnp.ndarray,
) -> jnp.ndarray:
    """Recover causal K = Phi_u Phi_x^{-1} from block response tensors."""
    import jax.scipy.linalg as jla

    tx, tw = Phi_x.shape[:2]
    tu = Phi_u.shape[0]
    n_state = Phi_x.shape[-1]
    Phi_x_matrix = jnp.swapaxes(Phi_x, 1, 2).reshape(
        tx * n_state,
        tw * n_state,
    )
    Phi_u_matrix = jnp.swapaxes(Phi_u, 1, 2).reshape(
        tu * N_CONTROL,
        tw * n_state,
    )
    K_transpose = jla.solve_triangular(Phi_x_matrix.T, Phi_u_matrix.T, lower=False)
    K = K_transpose.T.reshape(
        tu,
        N_CONTROL,
        tw,
        n_state,
    ).swapaxes(1, 2)
    causal_mask = jnp.tril(jnp.ones((tu, tw), dtype=K.dtype))[:, :, None, None]
    return K * causal_mask


def sample_unit_ball(key: jax.Array, dimension: int, dtype: jnp.dtype):
    key, direction_key, radius_key = jax.random.split(key, 3)
    direction = jax.random.normal(direction_key, (dimension,), dtype=dtype)
    direction = direction / (jnp.linalg.norm(direction) + 1.0e-12)
    radius = jax.random.uniform(radius_key, (), dtype=dtype) ** (1.0 / dimension)
    return key, radius * direction


def rollout_disturbance(
    key: jax.Array,
    *,
    adversarial_index: int | None,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jnp.ndarray]:
    """Sample a random direction or select one of 26 boundary directions."""
    key, w = sample_unit_ball(key, N_PHYS, dtype)
    if adversarial_index is not None:
        direction_index = adversarial_index % (2 * N_PHYS + 2)
        if direction_index < N_PHYS:
            w = jnp.zeros((N_PHYS,), dtype=dtype).at[direction_index].set(1.0)
        elif direction_index < 2 * N_PHYS:
            w = jnp.zeros((N_PHYS,), dtype=dtype).at[direction_index - N_PHYS].set(-1.0)
        elif direction_index == 2 * N_PHYS:
            w = jnp.ones((N_PHYS,), dtype=dtype) / jnp.sqrt(N_PHYS)
        else:
            w = -jnp.ones((N_PHYS,), dtype=dtype) / jnp.sqrt(N_PHYS)
    return key, ROLLOUT_DISTURBANCE_SCALE * w


def simulate_true_step(
    key: jax.Array,
    x_aug: jnp.ndarray,
    u: jnp.ndarray,
    disturbance: Callable,
    *,
    true_force: jnp.ndarray = TRUE_FORCE,
    adversarial_index: int | None = None,
    dt: float = DT,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    x_next_phys = physical_step(x_aug[:N_PHYS], u, true_force, dt)
    key, w = rollout_disturbance(
        key,
        adversarial_index=adversarial_index,
        dtype=x_aug.dtype,
    )
    x_next_phys = x_next_phys + disturbance(x_aug[None, :])[0] @ w
    return key, jnp.concatenate([x_next_phys, x_aug[N_PHYS:]]), w


def state_tube_widths(
    phi_x: jnp.ndarray,
    adaptive_disturbance: jnp.ndarray,
) -> jnp.ndarray:
    """Return augmented-state tube widths from the adaptive SLS response.

    This is independent of the constraint rows: applying an identity output
    map gives each state component's backoff directly from Phi_x and the
    adaptive disturbance generators.
    """
    response = jnp.einsum(
        "kjxn,jne->kjxe",
        phi_x,
        adaptive_disturbance,
    )
    contributions = jnp.sum(jnp.abs(response), axis=-1)
    k = jnp.arange(contributions.shape[0])[:, None]
    j = jnp.arange(contributions.shape[1])[None, :]
    causal = j <= k
    return jnp.sum(
        jnp.where(causal[..., None], contributions, 0.0),
        axis=1,
    )


def structured_adaptive_state_tube_widths(
    phi_x: jnp.ndarray,
    phi_u: jnp.ndarray,
    adaptive_disturbance: jnp.ndarray,
    estimator_error_gains: jnp.ndarray,
    estimator_innovation_gains: jnp.ndarray,
    observable_residual_maps: jnp.ndarray,
    initial_parameter_generators: jnp.ndarray,
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    *,
    dt: float = DT,
) -> jnp.ndarray:
    """Return correlation-aware adaptive tubes for every augmented state.

    ``state_tube_widths`` is appropriate when each disturbance-generator
    block is independent.  The adaptive parameter-update blocks are not:
    they share the initial parameter uncertainty and propagate through the
    estimator recursion.  Treating those blocks independently repeatedly
    adds ``delta_G`` and can make the force-state tubes grow spuriously.

    This function applies the controller's structured ``get_adaptive_betas``
    construction to an identity output map, then converts its per-source
    squared widths into state-component half-widths.
    """
    horizon = nominal_controls.shape[0]
    n_aug = phi_x.shape[-2]
    n_theta = initial_parameter_generators.shape[0]

    # E_hat stores the original physical disturbance generators first and
    # appends two parameter-generator blocks of this size.
    parameter_generator_columns = initial_parameter_generators.shape[1]
    original_disturbance_columns = (
        adaptive_disturbance.shape[-1] - 2 * parameter_generator_columns
    )
    physical_disturbance = adaptive_disturbance[
        :, :N_PHYS, :original_disturbance_columns
    ]

    def force_sensitivity(x_aug: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(
            lambda force: physical_step(x_aug[:N_PHYS], u, force, dt)
        )(x_aug[N_PHYS:])

    sensitivity = jax.vmap(force_sensitivity)(
        nominal_states[:-1],
        nominal_controls,
    )
    sensitivity = jnp.concatenate(
        [
            sensitivity,
            jnp.zeros((1, N_PHYS, n_theta), dtype=sensitivity.dtype),
        ],
        axis=0,
    )

    observable_noise = jnp.einsum(
        "tij,tjk->tik",
        observable_residual_maps,
        physical_disturbance[:horizon],
    )
    identity_output = jnp.broadcast_to(
        jnp.eye(n_aug, dtype=phi_x.dtype),
        (horizon + 1, n_aug, n_aug),
    )
    zero_control_output = jnp.zeros(
        (horizon, n_aug, phi_u.shape[-2]),
        dtype=phi_u.dtype,
    )
    beta = get_adaptive_betas(
        identity_output,
        zero_control_output,
        phi_x,
        phi_u,
        physical_disturbance,
        sensitivity,
        estimator_error_gains,
        estimator_innovation_gains,
        observable_noise,
        initial_parameter_generators,
        nx=N_PHYS,
        n_theta=n_theta,
    )

    k = jnp.arange(horizon + 1)[:, None]
    j = jnp.arange(horizon + 1)[None, :]
    causal = j <= k
    return jnp.sum(
        jnp.where(causal[..., None], jnp.sqrt(jnp.maximum(beta, 0.0)), 0.0),
        axis=1,
    )


def save_parameter_plot(
    output_path: Path,
    estimates: np.ndarray,
    widths: np.ndarray,
    dt: float,
    true_force: np.ndarray | None = None,
) -> None:
    if true_force is None:
        true_force = np.asarray(TRUE_FORCE)
    labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    time = np.arange(estimates.shape[0]) * dt
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(time, estimates[:, index], label="estimate")
        axis.fill_between(
            time,
            estimates[:, index] - widths[:, index],
            estimates[:, index] + widths[:, index],
            alpha=0.18,
            label="uncertainty set",
        )
        axis.axhline(float(true_force[index]), color="tab:red", linestyle="--", label="true")
        axis.set_ylabel(f"{labels[index]} (N)")
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_xy_plot(
    output_path: Path,
    states: np.ndarray,
    plans: list[np.ndarray],
    *,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    for index, plan in enumerate(plans):
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            linestyle="--",
            alpha=0.35,
            label="planned" if index == 0 else None,
        )
    axis.plot(states[..., 0].T, states[..., 1].T, color="tab:blue", alpha=0.55)
    axis.scatter([float(X0_PHYS[0])], [float(X0_PHYS[1])], marker="o", color="black", label="start")
    axis.scatter([float(X_GOAL_PHYS[0])], [float(X_GOAL_PHYS[1])], marker="x", color="black", label="goal")
    circle = plt.Circle(
        (float(OBSTACLES[0, 0]), float(OBSTACLES[0, 1])),
        float(OBSTACLES[0, 2]),
        color="tab:red",
        alpha=0.25,
        label="obstacle",
    )
    axis.add_patch(circle)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title)
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_xy_tube_plot(
    output_path: Path,
    states: np.ndarray,
    nominal: np.ndarray,
    tubes: np.ndarray,
    *,
    title: str,
    max_tube_boxes: int = 24,
) -> None:
    """Plot XY rollouts and sampled adaptive axis-aligned tube cross-sections."""
    steps = min(nominal.shape[0], tubes.shape[0])
    stride = max(1, int(np.ceil(steps / max_tube_boxes)))
    sampled_steps = list(range(0, steps, stride))
    if sampled_steps[-1] != steps - 1:
        sampled_steps.append(steps - 1)

    fig, axis = plt.subplots(figsize=(8, 7))
    for plot_index, step in enumerate(sampled_steps):
        center = nominal[step, :2]
        half_width = tubes[step, :2]
        box = plt.Rectangle(
            center - half_width,
            2.0 * half_width[0],
            2.0 * half_width[1],
            facecolor="tab:green",
            edgecolor="tab:green",
            linewidth=0.8,
            alpha=0.09,
            label="adaptive tube" if plot_index == 0 else None,
        )
        axis.add_patch(box)

    axis.plot(
        nominal[:, 0],
        nominal[:, 1],
        color="tab:orange",
        linestyle="--",
        alpha=0.8,
        label="planned",
    )
    axis.plot(states[..., 0].T, states[..., 1].T, color="tab:blue", alpha=0.55)
    axis.scatter([float(X0_PHYS[0])], [float(X0_PHYS[1])], marker="o", color="black", label="start")
    axis.scatter([float(X_GOAL_PHYS[0])], [float(X_GOAL_PHYS[1])], marker="x", color="black", label="goal")
    circle = plt.Circle(
        (float(OBSTACLES[0, 0]), float(OBSTACLES[0, 1])),
        float(OBSTACLES[0, 2]),
        color="tab:red",
        alpha=0.25,
        label="obstacle",
    )
    axis.add_patch(circle)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title)
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_xz_plot(
    output_path: Path,
    states: np.ndarray,
    plans: list[np.ndarray],
    *,
    title: str,
    information_center_z: float | None = None,
) -> None:
    """Plot altitude progression, including the 3-D information center."""
    fig, axis = plt.subplots(figsize=(8, 6))
    for index, plan in enumerate(plans):
        axis.plot(
            plan[:, 0], plan[:, 2], color="tab:orange", linestyle="--", alpha=0.45,
            label="planned" if index == 0 else None,
        )
    axis.plot(states[..., 0].T, states[..., 2].T, color="tab:blue", alpha=0.55)
    axis.scatter([float(X0_PHYS[0])], [float(X0_PHYS[2])], marker="o", color="black", label="start")
    axis.scatter([float(X_GOAL_PHYS[0])], [float(X_GOAL_PHYS[2])], marker="x", color="black", label="goal")
    if information_center_z is not None:
        axis.axhline(
            float(information_center_z), color="tab:green", linestyle=":", linewidth=1.5,
            label=f"information altitude (z={information_center_z:.2f} m)",
        )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title(title)
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_z_time_plot(
    output_path: Path,
    states: np.ndarray,
    plans: list[np.ndarray],
    dt: float,
    *,
    title: str,
    information_center_z: float | None = None,
    tubes: np.ndarray | None = None,
    disturbance_z_off: float | None = None,
    disturbance_z_sharpness: float = 10.0,
) -> None:
    """Plot altitude, nominal z-tubes, rollouts, and disturbance magnitude."""
    fig, axis = plt.subplots(figsize=(8, 6))
    time = np.arange(states.shape[-2]) * dt
    plan_time = np.arange(plans[0].shape[0]) * dt if plans else time

    if disturbance_z_off is not None:
        # The background is a vertical altitude field. At every time, the
        # color at altitude z is ||E(z)||_F, where
        # E(z) = alpha * gate(z) * I and gate(z) is the tanh gate.
        z_values = [states[..., 2].reshape(-1)]
        if plans:
            z_values.extend(plan[:, 2].reshape(-1) for plan in plans)
        if tubes is not None and plans:
            tube_steps = min(plans[0].shape[0], tubes.shape[0])
            z_values.extend(
                [
                    plans[0][:tube_steps, 2] - tubes[:tube_steps, 2],
                    plans[0][:tube_steps, 2] + tubes[:tube_steps, 2],
                ]
            )
        finite_z = np.concatenate(z_values)
        finite_z = finite_z[np.isfinite(finite_z)]
        if finite_z.size:
            z_low = float(np.min(finite_z))
            z_high = float(np.max(finite_z))
            z_margin = max(0.02, 0.05 * (z_high - z_low))
            z_low -= z_margin
            z_high += z_margin
            z_edges = np.linspace(z_low, z_high, 256)
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            gate = 0.5 * (
                1.0
                - np.tanh(
                    disturbance_z_sharpness
                    * (z_centers - disturbance_z_off)
                )
            )
            disturbance_magnitude = (
                DISTURBANCE_MAGNITUDE * dt * gate * np.sqrt(N_PHYS)
            )
            time_high = max(
                float(time[-1]) if time.size else 0.0,
                float(plan_time[-1]) if plan_time.size else 0.0,
                dt,
            )
            color_field = disturbance_magnitude[:, None]
            image = axis.pcolormesh(
                [0.0, time_high],
                z_edges,
                color_field,
                cmap="magma",
                alpha=0.28,
                shading="auto",
                zorder=0,
            )
            colorbar = fig.colorbar(image, ax=axis, pad=0.02)
            colorbar.set_label(r"$\|E(z)\|_F$")

    if plans and tubes is not None:
        tube_steps = min(plans[0].shape[0], tubes.shape[0])
        nominal_z = plans[0][:tube_steps, 2]
        z_tube = tubes[:tube_steps, 2]
        axis.fill_between(
            plan_time[:tube_steps],
            nominal_z - z_tube,
            nominal_z + z_tube,
            color="tab:orange",
            alpha=0.20,
            label="nominal z-tube",
            zorder=1,
        )
    for index, plan in enumerate(plans):
        axis.plot(
            plan_time,
            plan[:, 2],
            color="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label="planned" if index == 0 else None,
        )
    axis.plot(time, states[..., 2].T, color="tab:blue", alpha=0.35)
    axis.axhline(float(X0_PHYS[2]), color="black", linestyle="-", alpha=0.35, label="start/goal altitude")
    if information_center_z is not None:
        axis.axhline(
            float(information_center_z),
            color="tab:green",
            linestyle=":",
            linewidth=1.5,
            label=f"information altitude (z={information_center_z:.2f} m)",
        )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("z (m)")
    axis.set_title(title)
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_tube_plot(
    output_path: Path,
    deviations: np.ndarray,
    tubes: np.ndarray,
    dt: float,
    *,
    title: str,
) -> None:
    indices = [0, 1, 2]
    labels = ["x", "y", "z"]
    if deviations.shape[-1] >= N_PHYS + N_THETA and tubes.shape[-1] >= N_PHYS + N_THETA:
        indices.extend([N_PHYS, N_PHYS + 1, N_PHYS + 2])
        labels.extend(["Fx", "Fy", "Fz"])
    steps = min(deviations.shape[-2], tubes.shape[0])
    time = np.arange(steps) * dt
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 12), sharex=True)
    for axis, index, label in zip(axes, indices, labels):
        deviation = np.nanmax(deviations[..., :steps, index], axis=0)
        axis.plot(time, deviation, label=f"max |{label} deviation|")
        axis.plot(time, tubes[:steps, index], linestyle="--", label=f"{label} tube")
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_all_rollouts_tube_plot(
    output_path: Path,
    deviations: np.ndarray,
    tubes: np.ndarray,
    dt: float,
    *,
    title: str,
) -> None:
    """Plot every rollout's absolute deviation against adaptive tube sizes."""
    indices = [0, 1, 2]
    labels = ["x", "y", "z"]
    units = ["m", "m", "m"]
    if deviations.shape[-1] >= N_PHYS + N_THETA and tubes.shape[-1] >= N_PHYS + N_THETA:
        indices.extend([N_PHYS, N_PHYS + 1, N_PHYS + 2])
        labels.extend(["Fx", "Fy", "Fz"])
        units.extend(["N", "N", "N"])
    steps = min(deviations.shape[-2], tubes.shape[0])
    time = np.arange(steps) * dt

    fig, axes = plt.subplots(len(indices), 1, figsize=(11, 14), sharex=True)
    fig.suptitle(title)

    for axis, index, label, unit in zip(axes, indices, labels, units):
        tube_line = tubes[:steps, index]
        rollout_lines = deviations[..., :steps, index].reshape(-1, steps)

        axis.plot(
            time,
            tube_line,
            color="tab:blue",
            linewidth=3.0,
            label=f"tube size ({label})",
            zorder=3,
        )
        for rollout_index, deviation in enumerate(rollout_lines):
            finite = np.isfinite(deviation)
            axis.plot(
                time[finite],
                deviation[finite],
                linewidth=1.0,
                alpha=0.65,
                label=(
                    f"|{label} - nominal| ({rollout_lines.shape[0]} rollouts)"
                    if rollout_index == 0
                    else None
                ),
                zorder=2,
            )

        axis.set_ylabel(unit)
        axis.set_title(f"{label}: Deviation vs Tube Size")
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
