"""Finite-horizon robust MPC with causal affine disturbance feedback.

This implements the policy class in (5)--(6) of Sinha et al.,
"Safe Adaptive Robust MPC: a Generalized Formulation" (the Pavone RAMPC
paper):

    u_k = v_k + sum_{j < k} K[k, j] d_j.

The nominal trajectory and every block of the strictly causal disturbance
gain are optimized together.  For a nonlinear plant, the robust response is
formed from the LTV linearization about the current nominal iterate.  No
bound on the omitted nonlinear linearization error is added here.  This is
intentional: it makes this a faithful no-LEB comparison baseline rather than
a certificate for the nonlinear plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np
from scipy import sparse

Array = np.ndarray
StepFunction = Callable[[Array, Array], Array]
LinearizationFunction = Callable[[Array, Array], tuple[Array, Array]]


@dataclass(frozen=True)
class BoxBounds:
    state_lower: Array
    state_upper: Array
    input_lower: Array
    input_upper: Array


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int = 12
    convergence_tolerance: float = 2e-3
    state_trust_region: float = 0.75
    input_trust_region: float = 1.25
    virtual_control_weight: float = 1e4
    obstacle_slack_weight: float = 1e5
    tube_size_weight: float = 2.0
    policy_regularization: float = 1e-5
    feedback_memory: int | None = 12
    # The same Problem instance is parameterized and warm-started. CVXPY's DPP
    # cache is disabled at solve time because its lifted parameter map requires
    # hundreds of GiB at the 75-step horizon.
    solver: str = "CLARABEL"
    verbose: bool = False


@dataclass
class AffineDisturbanceFeedbackResult:
    states: Array
    inputs: Array
    disturbance_gains: Array
    state_response: Array
    state_half_widths: Array
    input_half_widths: Array
    objective: float
    converged: bool
    iterations: int
    linearization_a: Array
    linearization_b: Array
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)
    convergence_history: list[dict[str, float | str | bool]] = field(default_factory=list)


@dataclass
class _SubproblemTemplate:
    problem: cp.Problem
    x: cp.Variable
    u: cp.Variable
    virtual: cp.Variable
    policy: cp.Expression
    response: cp.Variable
    obstacle_slack: cp.Variable | None
    a_parameters: list[cp.Parameter]
    b_parameters: list[cp.Parameter]
    dynamics_offset: cp.Parameter
    state_center: cp.Parameter
    input_center: cp.Parameter
    g_lifted: cp.Parameter
    h_lifted: cp.Parameter
    obstacle_normals: list[list[cp.Parameter]]
    disturbance_widths: Array
    active_solver: str | None = None


class AffineDisturbanceFeedbackSolver:
    """Optimize one robust plan and a full-horizon disturbance policy."""

    def __init__(
        self,
        *,
        nominal_step: StepFunction,
        linearize: LinearizationFunction,
        state_cost: Array,
        input_cost: Array,
        terminal_cost: Array,
        bounds: BoxBounds,
        disturbance_half_width: Array,
        config: SolverConfig | None = None,
    ) -> None:
        self.nominal_step = nominal_step
        self.linearize = linearize
        self.Q = np.asarray(state_cost, dtype=float)
        self.R = np.asarray(input_cost, dtype=float)
        self.Qf = np.asarray(terminal_cost, dtype=float)
        self.bounds = bounds
        self.disturbance_half_width = np.asarray(disturbance_half_width, dtype=float)
        self.config = config or SolverConfig()
        self.nx = self.Q.shape[0]
        self.nu = self.R.shape[0]
        self._validate()

    def _validate(self) -> None:
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("state_cost must be square")
        if self.Qf.shape != self.Q.shape:
            raise ValueError("terminal_cost must match state_cost")
        if self.R.ndim != 2 or self.R.shape[0] != self.R.shape[1]:
            raise ValueError("input_cost must be square")
        if self.disturbance_half_width.shape != (self.nx,):
            raise ValueError("disturbance_half_width must have one entry per state")
        if np.any(self.disturbance_half_width < 0.0):
            raise ValueError("disturbance half-widths must be nonnegative")
        if self.config.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.config.feedback_memory is not None and self.config.feedback_memory < 1:
            raise ValueError("feedback_memory must be positive or None")

    def _policy_variable(self, horizon: int) -> cp.Expression:
        """Embed only active policy coefficients into the full lifted matrix."""
        total_rows = horizon * self.nu
        total_columns = horizon * self.nx
        output_indices: list[int] = []
        for k in range(horizon):
            first_active = 0
            if self.config.feedback_memory is not None:
                first_active = max(0, k - self.config.feedback_memory)
            for j in range(first_active, k):
                for input_index in range(self.nu):
                    row = k * self.nu + input_index
                    for disturbance_index in range(self.nx):
                        column = j * self.nx + disturbance_index
                        # CVXPY reshape below uses Fortran/column-major order.
                        output_indices.append(row + total_rows * column)
        coefficients = cp.Variable(len(output_indices))
        selector = sparse.coo_matrix(
            (
                np.ones(len(output_indices)),
                (output_indices, np.arange(len(output_indices))),
            ),
            shape=(total_rows * total_columns, len(output_indices)),
        ).tocsc()
        return cp.reshape(
            selector @ coefficients,
            (total_rows, total_columns),
            order="F",
        )

    def rollout(self, initial_state: Array, inputs: Array) -> Array:
        states = np.empty((inputs.shape[0] + 1, self.nx), dtype=float)
        states[0] = initial_state
        for k, control in enumerate(inputs):
            states[k + 1] = self.nominal_step(states[k], control)
        return states

    def _lifted_matrices(
        self, a_matrices: Array, b_matrices: Array
    ) -> tuple[Array, Array]:
        """Return open-loop disturbance and input-to-state lifted maps."""
        horizon = a_matrices.shape[0]
        g = np.zeros((horizon * self.nx, horizon * self.nx))
        h = np.zeros((horizon * self.nx, horizon * self.nu))
        for disturbance_step in range(horizon):
            transition = np.eye(self.nx)
            for state_step in range(disturbance_step + 1, horizon + 1):
                row = slice((state_step - 1) * self.nx, state_step * self.nx)
                column = slice(disturbance_step * self.nx, (disturbance_step + 1) * self.nx)
                g[row, column] = transition
                if state_step < horizon:
                    transition = a_matrices[state_step] @ transition
        for input_step in range(horizon):
            transition = b_matrices[input_step]
            for state_step in range(input_step + 1, horizon + 1):
                row = slice((state_step - 1) * self.nx, state_step * self.nx)
                column = slice(input_step * self.nu, (input_step + 1) * self.nu)
                h[row, column] = transition
                if state_step < horizon:
                    transition = a_matrices[state_step] @ transition
        return g, h

    def _build_subproblem(
        self,
        *,
        initial_state: Array,
        reference: Array,
        horizon: int,
        obstacles: Array,
        terminal_position_tolerance: float | None,
    ) -> _SubproblemTemplate:
        x = cp.Variable((horizon + 1, self.nx))
        u = cp.Variable((horizon, self.nu))
        virtual = cp.Variable((horizon, self.nx))
        policy = self._policy_variable(horizon)
        obstacle_slack = (
            cp.Variable((horizon + 1, obstacles.shape[0]), nonneg=True)
            if obstacles.size
            else None
        )
        constraints: list[cp.Constraint] = [x[0] == initial_state]

        a_parameters = [cp.Parameter((self.nx, self.nx)) for _ in range(horizon)]
        b_parameters = [cp.Parameter((self.nx, self.nu)) for _ in range(horizon)]
        dynamics_offset = cp.Parameter((horizon, self.nx))
        state_center = cp.Parameter((horizon + 1, self.nx))
        input_center = cp.Parameter((horizon, self.nu))
        g_lifted = cp.Parameter((horizon * self.nx, horizon * self.nx))
        h_lifted = cp.Parameter((horizon * self.nx, horizon * self.nu))
        # An explicit response variable keeps robust-support canonicalization
        # sparse; applying abs() directly to G + H K creates a very large CVXPY
        # expression tree at the 75-step comparison horizon.
        response = cp.Variable((horizon * self.nx, horizon * self.nx))
        constraints.append(response == g_lifted + h_lifted @ policy)
        disturbance_widths = np.tile(self.disturbance_half_width, horizon)
        state_width_rows = cp.abs(response) @ disturbance_widths
        input_width_rows = cp.abs(policy) @ disturbance_widths

        for k in range(horizon):
            constraints.append(
                x[k + 1]
                == a_parameters[k] @ x[k]
                + b_parameters[k] @ u[k]
                + dynamics_offset[k]
                + virtual[k]
            )

        finite_lower = np.isfinite(self.bounds.state_lower)
        finite_upper = np.isfinite(self.bounds.state_upper)
        zero_width = np.zeros(self.nx)
        obstacle_normals: list[list[cp.Parameter]] = []
        for k in range(horizon + 1):
            step_normals: list[cp.Parameter] = []
            width = zero_width if k == 0 else state_width_rows[(k - 1) * self.nx : k * self.nx]
            if np.any(finite_lower):
                constraints.append(
                    x[k, finite_lower]
                    >= self.bounds.state_lower[finite_lower] + width[finite_lower]
                )
            if np.any(finite_upper):
                constraints.append(
                    x[k, finite_upper]
                    <= self.bounds.state_upper[finite_upper] - width[finite_upper]
                )

            for obstacle_index, (center_x, center_y, radius) in enumerate(obstacles):
                center = np.array([center_x, center_y])
                normal = cp.Parameter(2)
                step_normals.append(normal)
                if k == 0:
                    support = 0.0
                else:
                    response_block = response[(k - 1) * self.nx : k * self.nx]
                    support = cp.abs(normal @ response_block[:2]) @ disturbance_widths
                constraints.append(
                    normal @ (x[k, :2] - center)
                    + obstacle_slack[k, obstacle_index]
                    >= radius + support
                )
            obstacle_normals.append(step_normals)

        for k in range(horizon):
            width = input_width_rows[k * self.nu : (k + 1) * self.nu]
            constraints.extend(
                [
                    u[k] >= self.bounds.input_lower + width,
                    u[k] <= self.bounds.input_upper - width,
                ]
            )
        constraints.extend(
            [
                cp.abs(x - state_center) <= self.config.state_trust_region,
                cp.abs(u - input_center) <= self.config.input_trust_region,
            ]
        )
        if terminal_position_tolerance is not None:
            constraints.append(
                cp.norm_inf(x[-1, :2] - reference[-1, :2]) <= terminal_position_tolerance
            )

        def matrix_sqrt(weight: Array) -> Array:
            eigenvalues, eigenvectors = np.linalg.eigh(weight)
            return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T

        objective: cp.Expression = cp.sum_squares(
            (x[:-1] - reference[:-1]) @ matrix_sqrt(self.Q)
        )
        objective += cp.sum_squares(u @ matrix_sqrt(self.R))
        objective += cp.sum_squares((x[-1] - reference[-1]) @ matrix_sqrt(self.Qf))
        objective += self.config.virtual_control_weight * cp.norm1(virtual)
        objective += self.config.tube_size_weight * (
            cp.sum(state_width_rows) + 0.25 * cp.sum(input_width_rows)
        )
        objective += self.config.policy_regularization * cp.sum_squares(policy)
        if obstacle_slack is not None:
            objective += self.config.obstacle_slack_weight * cp.sum(obstacle_slack)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        return _SubproblemTemplate(
            problem=problem,
            x=x,
            u=u,
            virtual=virtual,
            policy=policy,
            response=response,
            obstacle_slack=obstacle_slack,
            a_parameters=a_parameters,
            b_parameters=b_parameters,
            dynamics_offset=dynamics_offset,
            state_center=state_center,
            input_center=input_center,
            g_lifted=g_lifted,
            h_lifted=h_lifted,
            obstacle_normals=obstacle_normals,
            disturbance_widths=disturbance_widths,
        )

    def _solve_subproblem(
        self,
        *,
        template: _SubproblemTemplate,
        states: Array,
        inputs: Array,
        obstacles: Array,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        float,
        str,
        dict[str, float | bool | str],
    ]:
        horizon = inputs.shape[0]
        a_matrices = np.asarray(
            [self.linearize(states[k], inputs[k])[0] for k in range(horizon)]
        )
        b_matrices = np.asarray(
            [self.linearize(states[k], inputs[k])[1] for k in range(horizon)]
        )
        g_lifted, h_lifted = self._lifted_matrices(a_matrices, b_matrices)
        template.state_center.value = states
        template.input_center.value = inputs
        template.g_lifted.value = g_lifted
        template.h_lifted.value = h_lifted
        offsets = np.empty((horizon, self.nx))
        for k in range(horizon):
            template.a_parameters[k].value = a_matrices[k]
            template.b_parameters[k].value = b_matrices[k]
            offsets[k] = (
                self.nominal_step(states[k], inputs[k])
                - a_matrices[k] @ states[k]
                - b_matrices[k] @ inputs[k]
            )
        template.dynamics_offset.value = offsets
        for k, step_normals in enumerate(template.obstacle_normals):
            for obstacle_index, normal_parameter in enumerate(step_normals):
                difference = states[k, :2] - obstacles[obstacle_index, :2]
                distance = np.linalg.norm(difference)
                normal_parameter.value = (
                    difference / distance if distance > 1e-8 else np.array([1.0, 0.0])
                )

        # Seed the first solve; later calls retain all primal and dual values on
        # this same Problem instance for CVXPY/solver warm starting.
        if template.x.value is None:
            template.x.value = states
            template.u.value = inputs
            template.virtual.value = np.zeros((horizon, self.nx))

        def solution_is_usable() -> bool:
            values = [
                template.x.value,
                template.u.value,
                template.virtual.value,
                template.policy.value,
                template.response.value,
            ]
            if template.obstacle_slack is not None:
                values.append(template.obstacle_slack.value)
            return (
                template.problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                and all(value is not None and np.all(np.isfinite(value)) for value in values)
                and template.problem.value is not None
                and np.isfinite(template.problem.value)
            )

        solver_attempts: list[str] = []
        primary_solver = template.active_solver or self.config.solver
        solvers = [primary_solver]
        for fallback in ("CLARABEL", "SCS"):
            if fallback != primary_solver:
                solvers.append(fallback)
        for solver_name in solvers:
            solve_options: dict[str, float | int | bool | str] = {
                "solver": solver_name,
                "verbose": self.config.verbose,
                "warm_start": solver_name == primary_solver,
                "ignore_dpp": True,
            }
            if solver_name == "OSQP":
                solve_options.update(
                    eps_abs=5e-4,
                    eps_rel=5e-4,
                    max_iter=20_000,
                    polishing=False,
                )
            elif solver_name == "CLARABEL":
                solve_options.update(max_iter=1_000)
            elif solver_name == "SCS":
                solve_options.update(max_iters=30_000, eps=5e-4)
            try:
                template.problem.solve(**solve_options)
                solver_attempts.append(f"{solver_name}:{template.problem.status}")
            except cp.error.SolverError as error:
                solver_attempts.append(f"{solver_name}:error({type(error).__name__})")
                continue
            if solution_is_usable():
                template.active_solver = solver_name
                break
            print(
                f"Pavone {solver_name} result rejected: "
                f"status={template.problem.status}; trying a fallback solver"
            )
        else:
            raise RuntimeError(
                "affine disturbance-feedback problem produced no finite optimal solution; "
                f"attempts: {', '.join(solver_attempts)}"
            )

        response_value = np.asarray(template.response.value)
        state_widths = np.zeros((horizon + 1, self.nx))
        state_widths[1:] = (
            np.abs(response_value) @ template.disturbance_widths
        ).reshape(horizon, self.nx)
        input_widths = (
            np.abs(template.policy.value) @ template.disturbance_widths
        ).reshape(horizon, self.nu)
        gains = np.asarray(template.policy.value).reshape(
            horizon, self.nu, horizon, self.nx
        ).swapaxes(1, 2)
        maximum_slack = (
            float(np.max(template.obstacle_slack.value))
            if template.obstacle_slack is not None
            else 0.0
        )
        solver_stats = template.problem.solver_stats
        subproblem_diagnostics = {
            "convex_objective": float(template.problem.value),
            "virtual_control_l1": float(np.sum(np.abs(template.virtual.value))),
            "virtual_control_linf": float(np.max(np.abs(template.virtual.value))),
            "solver_iterations": float(solver_stats.num_iters or 0),
            "solver_solve_time_seconds": float(solver_stats.solve_time or 0.0),
            "solver_used": str(solver_stats.solver_name),
            "solver_attempts": "; ".join(solver_attempts),
            "parameterized_problem_dpp": bool(template.problem.is_dpp()),
            "dpp_canonicalization_enabled": False,
        }
        return (
            np.asarray(template.x.value),
            np.asarray(template.u.value),
            gains,
            response_value.reshape(horizon, self.nx, horizon, self.nx),
            state_widths,
            input_widths,
            a_matrices,
            b_matrices,
            maximum_slack,
            str(template.problem.status),
            subproblem_diagnostics,
        )

    def solve(
        self,
        *,
        initial_state: Array,
        reference: Array,
        initial_inputs: Array,
        initial_states: Array,
        obstacles: Array | None = None,
        terminal_position_tolerance: float | None = None,
    ) -> AffineDisturbanceFeedbackResult:
        states = np.asarray(initial_states, dtype=float).copy()
        inputs = np.asarray(initial_inputs, dtype=float).copy()
        initial_state = np.asarray(initial_state, dtype=float)
        reference = np.asarray(reference, dtype=float)
        obstacles = np.empty((0, 3)) if obstacles is None else np.asarray(obstacles, dtype=float)
        states[0] = initial_state
        converged = False
        status = "not_solved"
        maximum_slack = np.inf
        convergence_history: list[dict[str, float | str | bool]] = []
        template = self._build_subproblem(
            initial_state=initial_state,
            reference=reference,
            horizon=inputs.shape[0],
            obstacles=obstacles,
            terminal_position_tolerance=terminal_position_tolerance,
        )

        for iteration in range(1, self.config.max_iterations + 1):
            result = self._solve_subproblem(
                template=template,
                states=states,
                inputs=inputs,
                obstacles=obstacles,
            )
            new_states, new_inputs = result[:2]
            state_change = float(np.max(np.abs(new_states - states)))
            input_change = float(np.max(np.abs(new_inputs - inputs)))
            change = max(state_change, input_change)
            states, inputs = new_states, new_inputs
            maximum_slack = result[-3]
            status = result[-2]
            subproblem_diagnostics = result[-1]
            iterate_ok = change <= self.config.convergence_tolerance
            slack_ok = maximum_slack <= 1e-5
            convergence_history.append(
                {
                    "iteration": iteration,
                    "solver_status": status,
                    "state_change_inf": state_change,
                    "input_change_inf": input_change,
                    "iterate_change_inf": change,
                    "iterate_tolerance": self.config.convergence_tolerance,
                    "iterate_tolerance_ratio": change / self.config.convergence_tolerance,
                    "maximum_obstacle_slack": maximum_slack,
                    "obstacle_slack_tolerance": 1e-5,
                    "obstacle_slack_tolerance_ratio": maximum_slack / 1e-5,
                    "iterate_converged": iterate_ok,
                    "obstacle_slack_converged": slack_ok,
                    **subproblem_diagnostics,
                }
            )
            if self.config.verbose:
                print(
                    "Pavone SCP iteration "
                    f"{iteration}/{self.config.max_iterations}: status={status}, "
                    f"state_change={state_change:.3e}, input_change={input_change:.3e}, "
                    f"obstacle_slack={maximum_slack:.3e}, "
                    f"virtual_control_linf={subproblem_diagnostics['virtual_control_linf']:.3e}"
                )
            if iterate_ok and slack_ok:
                converged = True
                break

        gains, response, state_widths, input_widths, a_matrices, b_matrices = result[2:8]
        nonlinear_rollout = self.rollout(initial_state, inputs)
        dynamics_defect = float(np.max(np.abs(nonlinear_rollout - states)))
        obstacle_margin = np.inf
        if obstacles.size:
            # Match the directional support used by the convex obstacle
            # constraints above.  The former post-check used
            # norm([x_half_width, y_half_width]), which combines maxima from
            # different disturbance realizations and can falsely reject an
            # otherwise feasible affine-response tube.
            disturbance_widths = np.tile(
                self.disturbance_half_width, inputs.shape[0]
            )
            obstacle_margins: list[float] = []
            for k in range(states.shape[0]):
                for center_x, center_y, radius in obstacles:
                    difference = states[k, :2] - np.array([center_x, center_y])
                    distance = float(np.linalg.norm(difference))
                    if k == 0:
                        support = 0.0
                    else:
                        normal = (
                            difference / distance
                            if distance > 1e-8
                            else np.array([1.0, 0.0])
                        )
                        directional_response = np.einsum(
                            "i,ijk->jk", normal, response[k - 1, :2]
                        )
                        support = float(
                            np.sum(
                                np.abs(directional_response.reshape(-1))
                                * disturbance_widths
                            )
                        )
                    obstacle_margins.append(distance - radius - support)
            obstacle_margin = float(np.min(obstacle_margins))
        state_margin = float(
            min(
                np.min(states - state_widths - self.bounds.state_lower),
                np.min(self.bounds.state_upper - states - state_widths),
            )
        )
        input_margin = float(
            min(
                np.min(inputs - input_widths - self.bounds.input_lower),
                np.min(self.bounds.input_upper - inputs - input_widths),
            )
        )
        objective = float(
            np.einsum("ti,ij,tj->", states[:-1] - reference[:-1], self.Q, states[:-1] - reference[:-1])
            + np.einsum("ti,ij,tj->", inputs, self.R, inputs)
            + (states[-1] - reference[-1]) @ self.Qf @ (states[-1] - reference[-1])
        )
        final_iteration = convergence_history[-1]
        blockers: list[str] = []
        if not final_iteration["iterate_converged"]:
            if final_iteration["state_change_inf"] >= final_iteration["input_change_inf"]:
                blockers.append("state iterate change above tolerance")
            else:
                blockers.append("input iterate change above tolerance")
        if not final_iteration["obstacle_slack_converged"]:
            blockers.append("obstacle slack above tolerance")
        if not blockers:
            blockers.append("none")
        iterate_ratios = [float(item["iterate_tolerance_ratio"]) for item in convergence_history]
        progress_ratio = (
            iterate_ratios[-1] / iterate_ratios[0]
            if iterate_ratios[0] > 0.0
            else 0.0
        )
        return AffineDisturbanceFeedbackResult(
            states=states,
            inputs=inputs,
            disturbance_gains=gains,
            state_response=response,
            state_half_widths=state_widths,
            input_half_widths=input_widths,
            objective=objective,
            converged=converged,
            iterations=iteration,
            linearization_a=a_matrices,
            linearization_b=b_matrices,
            diagnostics={
                "solver_status": status,
                "termination_reason": "converged" if converged else "maximum iterations reached",
                "primary_convergence_blocker": blockers[0],
                "convergence_blockers": "; ".join(blockers),
                "final_state_change_inf": final_iteration["state_change_inf"],
                "final_input_change_inf": final_iteration["input_change_inf"],
                "final_iterate_change_inf": final_iteration["iterate_change_inf"],
                "iterate_convergence_tolerance": self.config.convergence_tolerance,
                "final_iterate_tolerance_ratio": final_iteration["iterate_tolerance_ratio"],
                "obstacle_slack_tolerance": 1e-5,
                "final_obstacle_slack_tolerance_ratio": final_iteration[
                    "obstacle_slack_tolerance_ratio"
                ],
                "iterate_tolerance_ratio_final_over_initial": progress_ratio,
                "final_virtual_control_l1": final_iteration["virtual_control_l1"],
                "final_virtual_control_linf": final_iteration["virtual_control_linf"],
                "parameterized_problem_dpp": final_iteration["parameterized_problem_dpp"],
                "warm_start_enabled": True,
                "total_solver_iterations": sum(
                    float(item["solver_iterations"]) for item in convergence_history
                ),
                "total_solver_solve_time_seconds": sum(
                    float(item["solver_solve_time_seconds"])
                    for item in convergence_history
                ),
                "maximum_obstacle_slack": maximum_slack,
                "maximum_nonlinear_dynamics_defect": dynamics_defect,
                "minimum_robust_state_margin": state_margin,
                "minimum_robust_input_margin": input_margin,
                "minimum_radial_obstacle_margin": obstacle_margin,
                "linearization_error_bound_enabled": False,
                "tube_guarantee_scope": "LTV prediction model only",
                "disturbance_feedback_memory": (
                    "full" if self.config.feedback_memory is None else self.config.feedback_memory
                ),
            },
            convergence_history=convergence_history,
        )
