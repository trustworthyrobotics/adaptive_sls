"""Residual-gated inverse-mass estimators for the planar quadrotor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .model import disturbance_sensitivity, parameter_sensitivity, rk4_step


Array = np.ndarray
LEARNING_CHANNEL_DIAGONAL = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
LEARNING_CHANNEL = np.diag(LEARNING_CHANNEL_DIAGONAL)


def zonotope_from_interval(lower: float, upper: float) -> tuple[Array, Array]:
    if lower > upper:
        raise ValueError("empty inverse-mass interval")
    return np.array([(lower + upper) / 2.0]), np.array([[(upper - lower) / 2.0]])


def pad_generators(generators: Array, order: int) -> Array:
    target = generators.shape[0] * order
    if generators.shape[1] > target:
        raise ValueError("generator reduction is required before padding")
    return np.pad(generators, ((0, 0), (0, target - generators.shape[1])))


@dataclass
class SMEState:
    lower: float
    upper: float
    center: Array
    generators: Array
    nonlinear_remainder: Array

    @classmethod
    def initial(cls, config: ExperimentConfig) -> "SMEState":
        half = config.parameter_half_width
        center, generators = zonotope_from_interval(-half, half)
        return cls(-half, half, center, generators, np.zeros(6))

    @property
    def vertices(self) -> Array:
        return np.array([[self.lower], [self.upper]])

    @property
    def half_width(self) -> Array:
        return np.sum(np.abs(self.generators), axis=1)

    @property
    def area(self) -> float:
        # One-dimensional measure, retained under the legacy output key.
        return self.upper - self.lower

    def update(
        self,
        previous_state: Array,
        control: Array,
        next_state: Array,
        config: ExperimentConfig,
    ) -> "SMEState":
        """Intersect the interval using only the C-selected residual rows."""
        center_scalar = float(self.center[0])
        base = rk4_step(previous_state, control, center_scalar, config)
        sensitivity = parameter_sensitivity(previous_state, control, self.center, config)
        errors = []
        for theta in (self.lower, self.upper):
            affine = base + sensitivity[:, 0] * (theta - center_scalar)
            for disturbance in (
                -config.disturbance_acceleration_half_width,
                config.disturbance_acceleration_half_width,
            ):
                exact = rk4_step(
                    previous_state,
                    control,
                    theta,
                    config,
                    disturbance=disturbance,
                )
                errors.append(np.abs(exact - affine))
        remainder = np.max(errors, axis=0) + 1e-10
        residual_at_zero = next_state - base + sensitivity[:, 0] * center_scalar
        lower, upper = self.lower, self.upper
        for row in np.flatnonzero(LEARNING_CHANNEL_DIAGONAL):
            normal = float(sensitivity[row, 0])
            if abs(normal) < 1e-12:
                continue
            strip = sorted(
                (
                    (residual_at_zero[row] - remainder[row]) / normal,
                    (residual_at_zero[row] + remainder[row]) / normal,
                )
            )
            lower, upper = max(lower, strip[0]), min(upper, strip[1])
        if lower > upper + 1e-10:
            return SMEState(
                self.lower,
                self.upper,
                self.center.copy(),
                self.generators.copy(),
                remainder,
            )
        lower = min(lower, upper)
        center, generators = zonotope_from_interval(lower, upper)
        return SMEState(lower, upper, center, generators, remainder)


@dataclass
class GatedGainState:
    center: Array
    generators: Array
    contraction_gate: Array
    posterior_gate: Array

    @classmethod
    def initial(cls, config: ExperimentConfig) -> "GatedGainState":
        generators = np.array([[config.parameter_half_width]])
        return cls(np.zeros(1), generators, np.ones(1, bool), np.ones(1, bool))

    @property
    def half_width(self) -> Array:
        return np.sum(np.abs(self.generators), axis=1)

    def update(
        self,
        previous_state: Array,
        control: Array,
        next_state: Array,
        config: ExperimentConfig,
    ) -> "GatedGainState":
        prediction = rk4_step(previous_state, control, self.center, config)
        sensitivity = parameter_sensitivity(previous_state, control, self.center, config)
        measured_sensitivity = LEARNING_CHANNEL @ sensitivity
        residual_map = np.linalg.pinv(measured_sensitivity, rcond=1e-7) @ LEARNING_CHANNEL
        pi = residual_map @ sensitivity
        disturbance_width = (
            np.abs(
                disturbance_sensitivity(previous_state, control, self.center, config)[:, 0]
            )
            * config.disturbance_acceleration_half_width
        )
        noise_map = residual_map @ np.diag(disturbance_width + 1e-10)
        covariance = self.generators @ self.generators.T
        innovation_gram = (
            pi @ covariance @ pi.T
            + noise_map @ noise_map.T
            + 1e-12 * np.eye(1)
        )
        gain = covariance @ pi.T @ np.linalg.inv(innovation_gram)

        scaled = (gain @ pi) @ self.generators
        contraction_gate = np.sum(np.abs(self.generators - scaled), axis=1) < np.sum(
            np.abs(self.generators), axis=1
        )
        gain = np.diag(contraction_gate.astype(float)) @ gain
        pre = np.concatenate(
            [(np.eye(1) - gain @ pi) @ self.generators, gain @ noise_map], axis=1
        )
        posterior_gate = np.sum(np.abs(pre), axis=1) < np.sum(
            np.abs(self.generators), axis=1
        )
        gain = np.diag(posterior_gate.astype(float)) @ gain
        center = self.center + gain @ residual_map @ (next_state - prediction)
        generators = np.concatenate(
            [(np.eye(1) - gain @ pi) @ self.generators, gain @ noise_map], axis=1
        )
        target = generators.shape[0] * config.zonotope_order
        if generators.shape[1] > target:
            order = np.argsort(np.linalg.norm(generators, axis=0))[::-1]
            keep = target - generators.shape[0]
            sorted_generators = generators[:, order]
            tail = np.array([[np.sum(np.abs(sorted_generators[:, keep:]))]])
            generators = np.concatenate([sorted_generators[:, :keep], tail], axis=1)
        return GatedGainState(center, generators, contraction_gate, posterior_gate)
