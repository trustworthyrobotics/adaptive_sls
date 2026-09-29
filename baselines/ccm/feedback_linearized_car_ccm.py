"""Certified CCM ingredients for the positive-speed dynamic unicycle.

The coordinate map

    psi(px, py, heading, speed) = (px, py, speed*cos(heading), speed*sin(heading))

is a diffeomorphism on ``speed > 0`` and any non-wrapping heading sector.  With
the input transformation implemented below, the nominal dynamics become two
decoupled double integrators.  A constant contraction metric in these
coordinates therefore pulls back to a true state-dependent CCM for the car.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class FeedbackLinearizedCarCCM:
    metric: np.ndarray
    differential_gain: np.ndarray
    contraction_rate: float
    disturbance_bound: float
    position_support: float
    velocity_support: float
    virtual_input_support: float
    certificate_max_eigenvalue: float
    maximum_disturbance_metric_norm: float

    @property
    def closed_loop_matrix(self) -> np.ndarray:
        A = np.block(
            [
                [np.zeros((2, 2)), np.eye(2)],
                [np.zeros((2, 2)), np.zeros((2, 2))],
            ]
        )
        B = np.vstack([np.zeros((2, 2)), np.eye(2)])
        return A + B @ self.differential_gain

    def radius(self, time_s: float | np.ndarray) -> float | np.ndarray:
        return (
            self.disturbance_bound
            * (1.0 - np.exp(-self.contraction_rate * time_s))
            / self.contraction_rate
        )

    def physical_half_widths(self, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            radii * self.position_support,
            radii * self.velocity_support,
            radii[:-1] * self.virtual_input_support,
        )


def car_to_linearizing_coordinates(state: np.ndarray) -> np.ndarray:
    px, py, heading, speed = np.asarray(state, dtype=float)
    return np.array([px, py, speed * np.cos(heading), speed * np.sin(heading)])


def linearizing_to_car_state(linear_state: np.ndarray) -> np.ndarray:
    px, py, velocity_x, velocity_y = np.asarray(linear_state, dtype=float)
    speed = np.hypot(velocity_x, velocity_y)
    if speed <= 0.0:
        raise ValueError("the feedback-linearizing coordinates require positive speed")
    return np.array([px, py, np.arctan2(velocity_y, velocity_x), speed])


def virtual_to_physical_input(state: np.ndarray, virtual_input: np.ndarray) -> np.ndarray:
    heading, speed = float(state[2]), float(state[3])
    if speed <= 0.0:
        raise ValueError("feedback linearization is singular at zero speed")
    cosine, sine = np.cos(heading), np.sin(heading)
    acceleration = cosine * virtual_input[0] + sine * virtual_input[1]
    turn_rate = (-sine * virtual_input[0] + cosine * virtual_input[1]) / speed
    return np.array([turn_rate, acceleration])


def physical_to_virtual_input(state: np.ndarray, physical_input: np.ndarray) -> np.ndarray:
    heading, speed = float(state[2]), float(state[3])
    turn_rate, acceleration = physical_input
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array(
        [
            acceleration * cosine - speed * turn_rate * sine,
            acceleration * sine + speed * turn_rate * cosine,
        ]
    )


def linearizing_jacobian(state: np.ndarray) -> np.ndarray:
    heading, speed = float(state[2]), float(state[3])
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -speed * sine, cosine],
            [0.0, 0.0, speed * cosine, sine],
        ]
    )


def pulled_back_metric(state: np.ndarray, certificate: FeedbackLinearizedCarCCM) -> np.ndarray:
    jacobian = linearizing_jacobian(state)
    return jacobian.T @ certificate.metric @ jacobian


def ccm_feedback(
    state: np.ndarray,
    nominal_linear_state: np.ndarray,
    nominal_virtual_input: np.ndarray,
    certificate: FeedbackLinearizedCarCCM,
) -> np.ndarray:
    error = car_to_linearizing_coordinates(state) - nominal_linear_state
    commanded_virtual_input = nominal_virtual_input + certificate.differential_gain @ error
    return virtual_to_physical_input(state, commanded_virtual_input)


def design_feedback_linearized_car_ccm(
    *,
    parameter_half_width: float,
    continuous_disturbance_bound: float,
    certified_maximum_speed: float,
    proportional_gain: float = 4.0,
    derivative_gain: float = 4.0,
    design_contraction_rate: float = 1.0,
    certified_contraction_rate: float = 0.99,
) -> FeedbackLinearizedCarCCM:
    """Compute and numerically verify a constant dual-coordinate CCM.

    The SDP normalizes the worst vertex of a conservative disturbance box to
    unit metric norm and minimizes position, velocity, and feedback support.
    The returned contraction rate is slightly below the design rate, leaving a
    strict numerical certificate margin.
    """
    if not 0.0 < certified_contraction_rate < design_contraction_rate:
        raise ValueError("the certified rate must be positive and below the design rate")
    A = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [np.zeros((2, 2)), np.zeros((2, 2))],
        ]
    )
    B = np.vstack([np.zeros((2, 2)), np.eye(2)])
    K = np.concatenate(
        [-proportional_gain * np.eye(2), -derivative_gain * np.eye(2)], axis=1
    )
    closed_loop = A + B @ K
    position_selector = np.concatenate([np.eye(2), np.zeros((2, 2))], axis=1)
    velocity_selector = np.concatenate([np.zeros((2, 2)), np.eye(2)], axis=1)

    # The position mismatch is the Minkowski sum of the fixed velocity bias
    # and continuous position disturbance.  Heading/speed disturbances induce
    # a velocity-vector mismatch bounded by d_v + vmax*d_heading per axis.
    position_mismatch = parameter_half_width + continuous_disturbance_bound
    velocity_mismatch = continuous_disturbance_bound * (1.0 + certified_maximum_speed)
    half_widths = np.array(
        [position_mismatch, position_mismatch, velocity_mismatch, velocity_mismatch]
    )
    disturbance_vertices = np.asarray(list(product((-1.0, 1.0), repeat=4))) * half_widths

    metric = cp.Variable((4, 4), symmetric=True)
    position_support_squared = cp.Variable(nonneg=True)
    velocity_support_squared = cp.Variable(nonneg=True)
    input_support_squared = cp.Variable(nonneg=True)
    constraints = [
        metric >> 1e-5 * np.eye(4),
        closed_loop.T @ metric
        + metric @ closed_loop
        + 2.0 * design_contraction_rate * metric
        << -1e-7 * np.eye(4),
        cp.bmat(
            [
                [position_support_squared * np.eye(2), position_selector],
                [position_selector.T, metric],
            ]
        )
        >> 0,
        cp.bmat(
            [
                [velocity_support_squared * np.eye(2), velocity_selector],
                [velocity_selector.T, metric],
            ]
        )
        >> 0,
        cp.bmat(
            [
                [input_support_squared * np.eye(2), K],
                [K.T, metric],
            ]
        )
        >> 0,
    ]
    constraints.extend(cp.quad_form(vertex, metric) <= 1.0 for vertex in disturbance_vertices)
    problem = cp.Problem(
        cp.Minimize(
            position_support_squared
            + 10.0 * velocity_support_squared
            + input_support_squared
        ),
        constraints,
    )
    problem.solve(solver="CLARABEL")
    if metric.value is None or not str(problem.status).startswith("optimal"):
        raise RuntimeError(f"CCM certificate SDP failed with status {problem.status}")

    metric_value = 0.5 * (metric.value + metric.value.T)
    inverse_metric = np.linalg.inv(metric_value)
    maximum_disturbance_norm = max(
        float(np.sqrt(vertex @ metric_value @ vertex)) for vertex in disturbance_vertices
    )
    contraction_residual = (
        closed_loop.T @ metric_value
        + metric_value @ closed_loop
        + 2.0 * certified_contraction_rate * metric_value
    )
    certificate_max_eigenvalue = float(np.linalg.eigvalsh(contraction_residual)[-1])
    if certificate_max_eigenvalue >= -1e-8:
        raise RuntimeError("computed metric has insufficient strict contraction margin")

    position_support = float(
        np.sqrt(np.linalg.eigvalsh(position_selector @ inverse_metric @ position_selector.T)[-1])
    )
    velocity_support = float(
        np.sqrt(np.linalg.eigvalsh(velocity_selector @ inverse_metric @ velocity_selector.T)[-1])
    )
    virtual_input_support = float(
        np.sqrt(np.linalg.eigvalsh(K @ inverse_metric @ K.T)[-1])
    )
    return FeedbackLinearizedCarCCM(
        metric=metric_value,
        differential_gain=K,
        contraction_rate=certified_contraction_rate,
        disturbance_bound=maximum_disturbance_norm,
        position_support=position_support,
        velocity_support=velocity_support,
        virtual_input_support=virtual_input_support,
        certificate_max_eigenvalue=certificate_max_eigenvalue,
        maximum_disturbance_metric_norm=maximum_disturbance_norm,
    )
