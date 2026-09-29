"""Robust TVLQR tube used as a trajectory-local CCM approximation.

This module deliberately does not claim a global CCM certificate.  It follows
the online architecture of Sasfi, Zeilinger, and Koehler: a nominal trajectory
is surrounded by a scalar, ellipsoidal tube and tracked with differential
feedback.  Here the metric and feedback are computed along one trajectory by a
finite-horizon Riccati recursion.  Nonlinear tube propagation is sampled over
the boundary of the ellipsoid and the vertices of the uncertainty box.

The resulting controller is useful as a solve-once, closed-loop baseline.  Its
tube is a deterministic numerical over-approximation of the sampled points,
not an SOS-certified global reachable set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

Array = np.ndarray
StepFunction = Callable[[Array, Array], Array]
LinearizationFunction = Callable[[Array, Array], tuple[Array, Array]]
UncertainStepFunction = Callable[[Array, Array, Array, Array], Array]


@dataclass(frozen=True)
class BoxBounds:
    """State and input box constraints."""

    state_lower: Array
    state_upper: Array
    input_lower: Array
    input_upper: Array


@dataclass(frozen=True)
class UncertaintySet:
    """Independent centered boxes for parameters and per-step disturbance."""

    parameter_half_width: Array
    disturbance_half_width: Array

    def vertices(self) -> tuple[Array, Array]:
        total_dimension = self.parameter_half_width.size + self.disturbance_half_width.size
        signs = np.asarray(list(product((-1.0, 1.0), repeat=total_dimension)))
        p = self.parameter_half_width.size
        return (
            signs[:, :p] * self.parameter_half_width[None, :],
            signs[:, p:] * self.disturbance_half_width[None, :],
        )


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for successive convexification and tube sampling."""

    max_iterations: int = 18
    convergence_tolerance: float = 2e-3
    state_trust_region: float = 0.75
    input_trust_region: float = 1.25
    virtual_control_weight: float = 2e5
    obstacle_slack_weight: float = 5e5
    tube_directions: int = 32
    tube_safety_factor: float = 1.10
    metric_regularization: float = 1e-7
    solver: str = "CLARABEL"
    verbose: bool = False
    random_seed: int = 7


@dataclass
class TrajectoryLocalCCMResult:
    states: Array
    inputs: Array
    gains: Array
    metrics: Array
    radii: Array
    state_half_widths: Array
    input_half_widths: Array
    obstacle_backoffs: Array
    objective: float
    converged: bool
    iterations: int
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)


class TrajectoryLocalCCMSolver:
    """Plan once with robust tube tightening and return a TVLQR policy."""

    def __init__(
        self,
        *,
        nominal_step: StepFunction,
        uncertain_step: UncertainStepFunction,
        linearize: LinearizationFunction,
        state_cost: Array,
        input_cost: Array,
        terminal_cost: Array,
        feedback_state_cost: Array | None = None,
        feedback_input_cost: Array | None = None,
        feedback_terminal_cost: Array | None = None,
        bounds: BoxBounds,
        uncertainty: UncertaintySet,
        config: SolverConfig | None = None,
    ) -> None:
        self.nominal_step = nominal_step
        self.uncertain_step = uncertain_step
        self.linearize = linearize
        self.Q = np.asarray(state_cost, dtype=float)
        self.R = np.asarray(input_cost, dtype=float)
        self.Qf = np.asarray(terminal_cost, dtype=float)
        self.Q_feedback = np.asarray(
            state_cost if feedback_state_cost is None else feedback_state_cost,
            dtype=float,
        )
        self.R_feedback = np.asarray(
            input_cost if feedback_input_cost is None else feedback_input_cost,
            dtype=float,
        )
        self.Qf_feedback = np.asarray(
            terminal_cost if feedback_terminal_cost is None else feedback_terminal_cost,
            dtype=float,
        )
        self.bounds = bounds
        self.uncertainty = uncertainty
        self.config = config or SolverConfig()
        self.nx = self.Q.shape[0]
        self.nu = self.R.shape[0]
        self._validate()

    def _validate(self) -> None:
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("state_cost must be square")
        if self.Qf.shape != (self.nx, self.nx):
            raise ValueError("terminal_cost must match state_cost")
        if self.R.shape != (self.nu, self.nu):
            raise ValueError("input_cost must be square")
        if self.Q_feedback.shape != self.Q.shape or self.Qf_feedback.shape != self.Q.shape:
            raise ValueError("feedback state costs must match state_cost")
        if self.R_feedback.shape != self.R.shape:
            raise ValueError("feedback_input_cost must match input_cost")
        if self.config.tube_safety_factor < 1.0:
            raise ValueError("tube_safety_factor must be at least one")
        if self.config.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

    def rollout(self, initial_state: Array, inputs: Array) -> Array:
        states = np.empty((inputs.shape[0] + 1, self.nx), dtype=float)
        states[0] = initial_state
        for k, control in enumerate(inputs):
            states[k + 1] = self.nominal_step(states[k], control)
        return states

    def tvlqr(self, states: Array, inputs: Array) -> tuple[Array, Array]:
        """Return feedback gains K and quadratic metrics P for u=v+K(x-z)."""
        horizon = inputs.shape[0]
        gains = np.zeros((horizon, self.nu, self.nx), dtype=float)
        metrics = np.zeros((horizon + 1, self.nx, self.nx), dtype=float)
        metrics[-1] = self.Qf_feedback + self.config.metric_regularization * np.eye(self.nx)
        for k in range(horizon - 1, -1, -1):
            A, B = self.linearize(states[k], inputs[k])
            Pn = metrics[k + 1]
            control_hessian = self.R_feedback + B.T @ Pn @ B
            gain = -np.linalg.solve(control_hessian, B.T @ Pn @ A)
            P = self.Q_feedback + A.T @ Pn @ A + A.T @ Pn @ B @ gain
            metrics[k] = 0.5 * (P + P.T) + self.config.metric_regularization * np.eye(self.nx)
            gains[k] = gain
        return gains, metrics

    def _sample_directions(self) -> Array:
        rng = np.random.default_rng(self.config.random_seed)
        random_directions = rng.normal(size=(self.config.tube_directions, self.nx))
        random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
        axes = np.concatenate([np.eye(self.nx), -np.eye(self.nx)], axis=0)
        boundary = np.concatenate([axes, random_directions, -random_directions], axis=0)
        # Interior samples catch non-monotone nonlinear remainder behavior.
        return np.concatenate([boundary, 0.5 * boundary], axis=0)

    @staticmethod
    def _metric_inverse_sqrt(metric: Array) -> Array:
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T

    def propagate_tube(
        self,
        states: Array,
        inputs: Array,
        gains: Array,
        metrics: Array,
        obstacle_count: int,
    ) -> tuple[Array, Array, Array, Array]:
        """Sample one-step reachable sets and enclose them in metric balls."""
        horizon = inputs.shape[0]
        radii = np.zeros(horizon + 1, dtype=float)
        state_widths = np.zeros((horizon + 1, self.nx), dtype=float)
        input_widths = np.zeros((horizon, self.nu), dtype=float)
        obstacle_backoffs = np.zeros((horizon + 1, obstacle_count), dtype=float)
        parameter_vertices, disturbance_vertices = self.uncertainty.vertices()
        directions = self._sample_directions()

        for k in range(horizon + 1):
            metric_inverse = np.linalg.inv(metrics[k])
            state_widths[k] = radii[k] * np.sqrt(np.maximum(np.diag(metric_inverse), 0.0))
            position_covariance = metric_inverse[:2, :2]
            radial_width = radii[k] * np.sqrt(
                max(float(np.linalg.eigvalsh(position_covariance)[-1]), 0.0)
            )
            obstacle_backoffs[k, :] = radial_width
            if k == horizon:
                continue

            feedback_covariance = gains[k] @ metric_inverse @ gains[k].T
            input_widths[k] = radii[k] * np.sqrt(
                np.maximum(np.diag(feedback_covariance), 0.0)
            )

            inverse_sqrt = self._metric_inverse_sqrt(metrics[k])
            errors = radii[k] * directions @ inverse_sqrt.T
            errors = np.concatenate([np.zeros((1, self.nx)), errors], axis=0)
            next_metric = metrics[k + 1]
            nominal_next = self.nominal_step(states[k], inputs[k])
            maximum_radius = 0.0
            one_step_uncertainty_radius = 0.0
            for error_index, error in enumerate(errors):
                control = inputs[k] + gains[k] @ error
                for parameter, disturbance in zip(parameter_vertices, disturbance_vertices):
                    next_state = self.uncertain_step(
                        states[k] + error,
                        control,
                        parameter,
                        disturbance,
                    )
                    next_error = next_state - nominal_next
                    candidate = float(np.sqrt(max(next_error @ next_metric @ next_error, 0.0)))
                    maximum_radius = max(maximum_radius, candidate)
                    if error_index == 0:
                        one_step_uncertainty_radius = max(one_step_uncertainty_radius, candidate)
            # Inflate the newly injected uncertainty rather than the complete
            # accumulated radius.  Multiplying the full radius at every step
            # would compound a numerical sampling margin exponentially.
            radii[k + 1] = maximum_radius + (
                self.config.tube_safety_factor - 1.0
            ) * one_step_uncertainty_radius

        return radii, state_widths, input_widths, obstacle_backoffs

    def _objective(self, states: Array, inputs: Array, reference: Array) -> float:
        state_errors = states[:-1] - reference[:-1]
        terminal_error = states[-1] - reference[-1]
        return float(
            np.einsum("ti,ij,tj->", state_errors, self.Q, state_errors)
            + np.einsum("ti,ij,tj->", inputs, self.R, inputs)
            + terminal_error @ self.Qf @ terminal_error
        )

    def _convex_subproblem(
        self,
        initial_state: Array,
        reference: Array,
        states: Array,
        inputs: Array,
        state_widths: Array,
        input_widths: Array,
        obstacle_backoffs: Array,
        obstacles: Array,
    ) -> tuple[Array, Array, float, str]:
        import cvxpy as cp

        horizon = inputs.shape[0]
        X = cp.Variable((horizon + 1, self.nx))
        U = cp.Variable((horizon, self.nu))
        virtual = cp.Variable((horizon, self.nx))
        obstacle_slack = (
            cp.Variable((horizon + 1, obstacles.shape[0]), nonneg=True)
            if obstacles.shape[0]
            else None
        )
        constraints: list[cp.Constraint] = [X[0] == initial_state]

        for k in range(horizon):
            A, B = self.linearize(states[k], inputs[k])
            f_bar = self.nominal_step(states[k], inputs[k])
            constraints.append(
                X[k + 1]
                == f_bar + A @ (X[k] - states[k]) + B @ (U[k] - inputs[k]) + virtual[k]
            )

        finite_lower = np.isfinite(self.bounds.state_lower)
        finite_upper = np.isfinite(self.bounds.state_upper)
        for k in range(horizon + 1):
            if np.any(finite_lower):
                constraints.append(
                    X[k, finite_lower]
                    >= self.bounds.state_lower[finite_lower] + state_widths[k, finite_lower]
                )
            if np.any(finite_upper):
                constraints.append(
                    X[k, finite_upper]
                    <= self.bounds.state_upper[finite_upper] - state_widths[k, finite_upper]
                )
        constraints.extend(
            [
                U >= self.bounds.input_lower[None, :] + input_widths,
                U <= self.bounds.input_upper[None, :] - input_widths,
                cp.abs(X - states) <= self.config.state_trust_region,
                cp.abs(U - inputs) <= self.config.input_trust_region,
            ]
        )

        for k in range(horizon + 1):
            for obstacle_index, (center_x, center_y, radius) in enumerate(obstacles):
                center = np.array([center_x, center_y])
                difference = states[k, :2] - center
                distance = np.linalg.norm(difference)
                if distance < 1e-7:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = difference / distance
                constraints.append(
                    normal @ (X[k, :2] - center)
                    + obstacle_slack[k, obstacle_index]
                    >= radius + obstacle_backoffs[k, obstacle_index]
                )

        objective = 0
        for k in range(horizon):
            objective += cp.quad_form(X[k] - reference[k], self.Q)
            objective += cp.quad_form(U[k], self.R)
        objective += cp.quad_form(X[horizon] - reference[horizon], self.Qf)
        objective += self.config.virtual_control_weight * cp.norm1(virtual)
        if obstacle_slack is not None:
            objective += self.config.obstacle_slack_weight * cp.sum(obstacle_slack)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver=self.config.solver, verbose=self.config.verbose)
        except cp.error.SolverError:
            problem.solve(solver="SCS", verbose=self.config.verbose, max_iters=20_000)
        if X.value is None or U.value is None:
            raise RuntimeError(f"trajectory-local CCM subproblem failed with status {problem.status}")
        maximum_slack = float(np.max(obstacle_slack.value)) if obstacle_slack is not None else 0.0
        return np.asarray(X.value), np.asarray(U.value), maximum_slack, str(problem.status)

    def solve(
        self,
        *,
        initial_state: Array,
        reference: Array,
        initial_inputs: Array,
        obstacles: Array | None = None,
        initial_states: Array | None = None,
    ) -> TrajectoryLocalCCMResult:
        """Solve one robust plan and synthesize its closed-loop tube policy."""
        inputs = np.asarray(initial_inputs, dtype=float).copy()
        initial_state = np.asarray(initial_state, dtype=float)
        reference = np.asarray(reference, dtype=float)
        obstacles = np.empty((0, 3)) if obstacles is None else np.asarray(obstacles, dtype=float)
        states = (
            self.rollout(initial_state, inputs)
            if initial_states is None
            else np.asarray(initial_states, dtype=float).copy()
        )
        states[0] = initial_state
        converged = False
        maximum_slack = np.inf
        status = "not_solved"

        for iteration in range(1, self.config.max_iterations + 1):
            gains, metrics = self.tvlqr(states, inputs)
            radii, state_widths, input_widths, obstacle_backoffs = self.propagate_tube(
                states, inputs, gains, metrics, obstacles.shape[0]
            )
            new_states, new_inputs, maximum_slack, status = self._convex_subproblem(
                initial_state,
                reference,
                states,
                inputs,
                state_widths,
                input_widths,
                obstacle_backoffs,
                obstacles,
            )
            change = max(
                float(np.max(np.abs(new_states - states))),
                float(np.max(np.abs(new_inputs - inputs))),
            )
            states, inputs = new_states, new_inputs
            if change <= self.config.convergence_tolerance and maximum_slack <= 1e-5:
                converged = True
                break

        # Use the exact nonlinear nominal rollout for the final controller and
        # recompute its metric/tube.  The dynamics defect is exposed below.
        optimized_states = states
        states = self.rollout(initial_state, inputs)
        dynamics_defect = float(np.max(np.abs(states - optimized_states)))
        gains, metrics = self.tvlqr(states, inputs)
        radii, state_widths, input_widths, obstacle_backoffs = self.propagate_tube(
            states, inputs, gains, metrics, obstacles.shape[0]
        )
        state_margins: list[float] = []
        finite_lower = np.isfinite(self.bounds.state_lower)
        finite_upper = np.isfinite(self.bounds.state_upper)
        if np.any(finite_lower):
            state_margins.append(
                float(
                    np.min(
                        states[:, finite_lower]
                        - state_widths[:, finite_lower]
                        - self.bounds.state_lower[finite_lower]
                    )
                )
            )
        if np.any(finite_upper):
            state_margins.append(
                float(
                    np.min(
                        self.bounds.state_upper[finite_upper]
                        - states[:, finite_upper]
                        - state_widths[:, finite_upper]
                    )
                )
            )
        input_margin = float(
            min(
                np.min(inputs - input_widths - self.bounds.input_lower),
                np.min(self.bounds.input_upper - inputs - input_widths),
            )
        )
        obstacle_margin = np.inf
        if obstacles.size:
            distances = np.linalg.norm(
                states[:, None, :2] - obstacles[None, :, :2], axis=-1
            )
            obstacle_margin = float(
                np.min(distances - obstacles[None, :, 2] - obstacle_backoffs)
            )
        state_margin = min(state_margins, default=np.inf)
        robust_feasible = min(state_margin, input_margin, obstacle_margin) >= -1e-6
        return TrajectoryLocalCCMResult(
            states=states,
            inputs=inputs,
            gains=gains,
            metrics=metrics,
            radii=radii,
            state_half_widths=state_widths,
            input_half_widths=input_widths,
            obstacle_backoffs=obstacle_backoffs,
            objective=self._objective(states, inputs, reference),
            converged=converged,
            iterations=iteration,
            diagnostics={
                "solver_status": status,
                "maximum_obstacle_slack": maximum_slack,
                "maximum_dynamics_defect": dynamics_defect,
                "minimum_robust_state_margin": state_margin,
                "minimum_robust_input_margin": input_margin,
                "minimum_robust_obstacle_margin": obstacle_margin,
                "final_sampled_tube_constraints_feasible": robust_feasible,
                "tube_kind": "sampled trajectory-local CCM (TVLQR approximation)",
            },
        )
