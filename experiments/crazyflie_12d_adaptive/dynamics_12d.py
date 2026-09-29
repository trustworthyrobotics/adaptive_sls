"""JAX-compatible 12D, high-level-command Crazyflie dynamics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rotation_zyx(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """Return ``Rz(psi) @ Ry(theta) @ Rx(phi)``."""
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    return jnp.array(
        [
            [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
            [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
            [-stheta, ctheta * sphi, ctheta * cphi],
        ]
    )


def thrust_direction(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """Body +z expressed in the world frame."""
    return rotation_zyx(phi, theta, psi)[:, 2]


def euler_rate_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """ZYX Euler-rate map from body rates to ``[phi_dot, theta_dot, psi_dot]``."""
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    ctheta = jnp.cos(theta)
    return jnp.array(
        [
            [1.0, sphi * jnp.tan(theta), cphi * jnp.tan(theta)],
            [0.0, cphi, -sphi],
            [0.0, sphi / ctheta, cphi / ctheta],
        ]
    )


def continuous_dynamics(
    x_phys: jnp.ndarray,
    u: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    gravity: float,
    k_phi_p: float,
    k_phi_d: float,
    k_theta_p: float,
    k_theta_d: float,
    k_r: float,
) -> jnp.ndarray:
    """12D dynamics with an effective thrust gain ``beta``.

    ``beta`` has units of inverse mass only for a calibrated Newton-thrust
    actuator.  Here it is intentionally treated as an effective gain.
    """
    position, velocity = x_phys[:3], x_phys[3:6]
    phi, theta, psi = x_phys[6:9]
    omega = x_phys[9:12]
    thrust, phi_cmd, theta_cmd, yawrate_cmd = u
    del position
    direction = thrust_direction(phi, theta, psi)
    acceleration = jnp.array([0.0, 0.0, -gravity], dtype=x_phys.dtype) + beta[0] * thrust * direction
    euler_dot = euler_rate_matrix(phi, theta) @ omega
    p_dot = k_phi_p * (phi_cmd - phi) - k_phi_d * omega[0]
    q_dot = k_theta_p * (theta_cmd - theta) - k_theta_d * omega[1]
    r_dot = k_r * (yawrate_cmd - omega[2])
    return jnp.concatenate([velocity, acceleration, euler_dot, jnp.array([p_dot, q_dot, r_dot])])


def rk4_step(x_phys: jnp.ndarray, u: jnp.ndarray, beta: jnp.ndarray, dt: float, **model) -> jnp.ndarray:
    """Integrate one held-control transition with RK4."""
    dt = jnp.asarray(dt, dtype=x_phys.dtype)
    fn = lambda xx: continuous_dynamics(xx, u, beta, **model)
    k1 = fn(x_phys)
    k2 = fn(x_phys + 0.5 * dt * k1)
    k3 = fn(x_phys + 0.5 * dt * k2)
    k4 = fn(x_phys + dt * k3)
    return x_phys + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def augmented_step(x: jnp.ndarray, u: jnp.ndarray, dt: float, **model) -> jnp.ndarray:
    """Step physical state and retain constant effective-gain planner state."""
    beta = x[12:13]
    return jnp.concatenate([rk4_step(x[:12], u, beta, dt, **model), beta])


def discrete_jacobians(x_phys: jnp.ndarray, u: jnp.ndarray, beta: jnp.ndarray, dt: float, **model):
    """Return JAX derivatives ``A, B, E`` of the physical RK4 transition."""
    step = lambda xx, uu, bb: rk4_step(xx, uu, bb, dt, **model)
    return (
        jax.jacfwd(lambda xx: step(xx, u, beta))(x_phys),
        jax.jacfwd(lambda uu: step(x_phys, uu, beta))(u),
        jax.jacfwd(lambda bb: step(x_phys, u, bb))(beta),
    )
