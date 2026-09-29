"""Certified-paper planar quadrotor with inverse-mass uncertainty."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import ExperimentConfig


Array = np.ndarray


def continuous_dynamics(
    state: Array,
    control: Array,
    inverse_mass_error: Array | float,
    config: ExperimentConfig,
    disturbance: float = 0.0,
) -> Array:
    """Return the derivative for the authors' ``[d, theta]`` model."""
    _, _, phi, v1, v2, omega = np.asarray(state)
    u1, u2 = np.asarray(control)
    theta = float(np.asarray(inverse_mass_error).reshape(-1)[0])
    inverse_mass = config.nominal_inverse_mass + theta
    c, s = np.cos(phi), np.sin(phi)
    return np.array(
        [
            v1 * c - v2 * s,
            v1 * s + v2 * c,
            omega,
            v2 * omega - config.gravity * s + c * disturbance,
            -v1 * omega - config.gravity * c
            + (u1 + u2) * inverse_mass - s * disturbance,
            config.arm_length * (u1 - u2) / config.inertia,
        ],
        dtype=float,
    )


def rk4_step(
    state: Array,
    control: Array,
    inverse_mass_error: Array | float,
    config: ExperimentConfig,
    disturbance: float = 0.0,
) -> Array:
    """One zero-order-hold RK4 step, matching the CCM paper."""
    h = config.dt
    f: Callable[[Array], Array] = lambda x: continuous_dynamics(
        x, control, inverse_mass_error, config, disturbance
    )
    k1 = f(state)
    k2 = f(state + 0.5 * h * k1)
    k3 = f(state + 0.5 * h * k2)
    k4 = f(state + h * k3)
    return np.asarray(state) + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def true_step(
    state: Array,
    control: Array,
    inverse_mass_error: Array | float,
    disturbance_coefficients: Array,
    config: ExperimentConfig,
) -> Array:
    """Execute one step with a held scalar bounded disturbance."""
    coefficients = np.asarray(disturbance_coefficients, dtype=float)
    if coefficients.size != 1 or np.any(np.abs(coefficients) > 1.0 + 1e-12):
        raise ValueError("the scalar disturbance coefficient must lie in [-1,1]")
    disturbance = config.disturbance_acceleration_half_width * float(coefficients.flat[0])
    return rk4_step(
        state, control, inverse_mass_error, config, disturbance=disturbance
    )


def finite_difference_linearization(
    state: Array,
    control: Array,
    inverse_mass_error: Array | float,
    config: ExperimentConfig,
    epsilon: float = 1e-5,
) -> tuple[Array, Array]:
    """Central-difference discrete-time Jacobians of the RK4 map."""
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    a = np.empty((6, 6), dtype=float)
    b = np.empty((6, 2), dtype=float)
    for index in range(6):
        delta = np.zeros(6)
        delta[index] = epsilon
        a[:, index] = (
            rk4_step(state + delta, control, inverse_mass_error, config)
            - rk4_step(state - delta, control, inverse_mass_error, config)
        ) / (2.0 * epsilon)
    for index in range(2):
        delta = np.zeros(2)
        delta[index] = epsilon
        b[:, index] = (
            rk4_step(state, control + delta, inverse_mass_error, config)
            - rk4_step(state, control - delta, inverse_mass_error, config)
        ) / (2.0 * epsilon)
    return a, b


def parameter_sensitivity(
    state: Array,
    control: Array,
    parameter_center: Array,
    config: ExperimentConfig,
    epsilon: float = 1e-5,
) -> Array:
    """Sensitivity of one RK4 step to the inverse-mass error."""
    center = float(np.asarray(parameter_center).reshape(-1)[0])
    return (
        rk4_step(state, control, center + epsilon, config)
        - rk4_step(state, control, center - epsilon, config)
    )[:, None] / (2.0 * epsilon)


def disturbance_sensitivity(
    state: Array,
    control: Array,
    inverse_mass_error: Array | float,
    config: ExperimentConfig,
    epsilon: float = 1e-5,
) -> Array:
    """Sensitivity of one RK4 step to the held scalar disturbance."""
    return (
        rk4_step(state, control, inverse_mass_error, config, disturbance=epsilon)
        - rk4_step(state, control, inverse_mass_error, config, disturbance=-epsilon)
    )[:, None] / (2.0 * epsilon)


def rollout_nominal(
    initial_state: Array,
    inputs: Array,
    inverse_mass_error: Array | float,
    config: ExperimentConfig,
) -> Array:
    states = np.empty((len(inputs) + 1, 6), dtype=float)
    states[0] = initial_state
    for index, control in enumerate(inputs):
        states[index + 1] = rk4_step(
            states[index], control, inverse_mass_error, config
        )
    return states


def goal_distance(state: Array, goal: Array) -> float:
    return float(np.linalg.norm(np.asarray(state)[:2] - np.asarray(goal)[:2]))


def obstacle_clearance(state: Array, obstacles: Array) -> float:
    if np.asarray(obstacles).size == 0:
        return float("inf")
    distances = np.linalg.norm(np.asarray(state)[:2] - obstacles[:, :2], axis=1)
    return float(np.min(distances - obstacles[:, 2]))
