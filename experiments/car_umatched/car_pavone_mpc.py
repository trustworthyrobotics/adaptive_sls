"""Solve-once Pavone-style affine disturbance-feedback baseline for the car."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import cvxpy as cp
import numpy as np

from baselines.pavone_mpc import (
    AffineDisturbanceFeedbackSolver,
    BoxBounds,
    SolverConfig,
)


OBSTACLES = np.array(
    [
        [-0.25, 0.20, 0.23],
        [0.25, -0.25, 0.23],
    ]
)


def car_step(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    x, y, heading, speed = state
    turn_rate, acceleration = control
    return np.array(
        [
            x + dt * speed * np.cos(heading),
            y + dt * speed * np.sin(heading),
            heading + dt * turn_rate,
            speed + dt * acceleration,
        ]
    )


def car_linearization(
    state: np.ndarray, control: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    del control
    heading, speed = state[2], state[3]
    a = np.eye(4)
    a[0, 2] = -dt * speed * np.sin(heading)
    a[0, 3] = dt * np.cos(heading)
    a[1, 2] = dt * speed * np.cos(heading)
    a[1, 3] = dt * np.sin(heading)
    b = np.zeros((4, 2))
    b[2, 0] = dt
    b[3, 1] = dt
    return a, b


def geometric_seed(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    horizon: int,
    dt: float,
    route_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, horizon + 1)
    smooth = 3.0 * progress**2 - 2.0 * progress**3
    positions = (
        (1.0 - smooth[:, None]) * initial_state[None, :2]
        + smooth[:, None] * goal_state[None, :2]
    )
    positions[:, 0] += route_offset * np.sin(2.0 * np.pi * progress)
    velocity = np.gradient(positions, dt, axis=0)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    speed = np.linalg.norm(velocity, axis=1)
    states = np.column_stack((positions, heading, speed))
    states[0] = initial_state
    states[-1] = goal_state
    inputs = np.column_stack(
        (
            np.diff(states[:, 2]) / dt,
            np.diff(states[:, 3]) / dt,
        )
    )
    inputs[:, 0] = np.clip(inputs[:, 0], -4.0, 4.0)
    inputs[:, 1] = np.clip(inputs[:, 1], -1.0, 1.0)
    return states, inputs


def load_ccm_seed(
    path: Path,
    *,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    horizon: int,
    dt: float,
    terminal_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    """Load and time-align a CCM nominal plan for Pavone SCP initialization."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = {
        "time_s",
        "x_m",
        "y_m",
        "heading_rad",
        "speed_m_per_s",
    }
    if not rows or not required_columns.issubset(rows[0]):
        missing = sorted(required_columns - (set(rows[0]) if rows else set()))
        raise ValueError(f"CCM warm-start CSV is empty or missing columns: {missing}")

    source_times = np.asarray([float(row["time_s"]) for row in rows])
    source_states = np.asarray(
        [
            [
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["heading_rad"]),
                float(row["speed_m_per_s"]),
            ]
            for row in rows
        ]
    )
    if not np.all(np.isfinite(source_times)) or not np.all(np.isfinite(source_states)):
        raise ValueError("CCM warm-start CSV contains non-finite values")
    if len(source_times) < 2 or np.any(np.diff(source_times) <= 0.0):
        raise ValueError("CCM warm-start times must be strictly increasing")

    target_times = np.arange(horizon + 1) * dt
    time_tolerance = max(1e-9, 1e-6 * target_times[-1])
    if (
        source_times[0] > target_times[0] + time_tolerance
        or source_times[-1] < target_times[-1] - time_tolerance
    ):
        raise ValueError(
            "CCM warm-start trajectory does not cover the Pavone time horizon "
            f"[0, {target_times[-1]:.6g}] s"
        )

    states = np.column_stack(
        [
            np.interp(target_times, source_times, source_states[:, 0]),
            np.interp(target_times, source_times, source_states[:, 1]),
            np.interp(target_times, source_times, np.unwrap(source_states[:, 2])),
            np.interp(target_times, source_times, source_states[:, 3]),
        ]
    )
    initial_error = float(np.max(np.abs(states[0] - initial_state)))
    terminal_position_error = float(np.max(np.abs(states[-1, :2] - goal_state[:2])))
    if initial_error > 1e-5:
        raise ValueError(
            f"CCM warm start has incompatible initial state (inf error {initial_error:.3e})"
        )
    if terminal_position_error > terminal_tolerance + 1e-5:
        raise ValueError(
            "CCM warm start has incompatible terminal position "
            f"(inf error {terminal_position_error:.3e})"
        )
    states[0] = initial_state

    raw_inputs = np.column_stack(
        (
            np.diff(states[:, 2]) / dt,
            np.diff(states[:, 3]) / dt,
        )
    )
    inputs = raw_inputs.copy()
    inputs[:, 0] = np.clip(inputs[:, 0], -4.0, 4.0)
    inputs[:, 1] = np.clip(inputs[:, 1], -1.0, 1.0)
    seed_rollout = np.empty_like(states)
    seed_rollout[0] = initial_state
    for k in range(horizon):
        seed_rollout[k + 1] = car_step(seed_rollout[k], inputs[k], dt)
    return states, inputs, {
        "warmstart_source": "CCM nominal trajectory",
        "ccm_warmstart_path": str(path.resolve()),
        "ccm_warmstart_interpolated": bool(
            len(source_times) != len(target_times)
            or not np.allclose(source_times, target_times)
        ),
        "ccm_warmstart_initial_state_error": initial_error,
        "ccm_warmstart_terminal_position_error": terminal_position_error,
        "ccm_warmstart_dynamics_defect": float(np.max(np.abs(seed_rollout - states))),
        "ccm_warmstart_input_clipping": bool(np.any(np.abs(raw_inputs - inputs) > 1e-12)),
    }


def refine_nominal_seed(
    *,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    seed_states: np.ndarray,
    seed_inputs: np.ndarray,
    dt: float,
    iterations: int,
    x_min: float,
    x_max: float,
    minimum_speed: float,
    maximum_speed: float,
    heading_min: float,
    heading_max: float,
    terminal_tolerance: float,
    nominal_obstacle_reserve: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """Cheap nominal SCP pass before the large robust policy synthesis."""
    states = seed_states.copy()
    inputs = seed_inputs.copy()
    horizon = len(inputs)
    progress = np.linspace(0.0, 1.0, horizon + 1)
    reference = (1.0 - progress[:, None]) * initial_state + progress[:, None] * goal_state
    reference[:, 2] = -0.5 * np.pi
    converged = False
    maximum_slack = np.inf
    for _ in range(iterations):
        x = cp.Variable((horizon + 1, 4))
        u = cp.Variable((horizon, 2))
        virtual = cp.Variable((horizon, 4))
        obstacle_slack = cp.Variable((horizon + 1, len(OBSTACLES)), nonneg=True)
        constraints: list[cp.Constraint] = [x[0] == initial_state]
        for k in range(horizon):
            a, b = car_linearization(states[k], inputs[k], dt)
            constraints.append(
                x[k + 1]
                == car_step(states[k], inputs[k], dt)
                + a @ (x[k] - states[k])
                + b @ (u[k] - inputs[k])
                + virtual[k]
            )
        constraints.extend(
            [
                x[:, 0] >= x_min,
                x[:, 0] <= x_max,
                x[:, 1] >= -5.0,
                x[:, 1] <= 5.0,
                x[:, 2] >= heading_min,
                x[:, 2] <= heading_max,
                x[:, 3] >= minimum_speed,
                x[:, 3] <= maximum_speed,
                u[:, 0] >= -4.0,
                u[:, 0] <= 4.0,
                u[:, 1] >= -1.0,
                u[:, 1] <= 1.0,
                cp.abs(x - states) <= 0.75,
                cp.abs(u - inputs) <= 1.25,
                cp.norm_inf(x[-1, :2] - goal_state[:2]) <= terminal_tolerance,
            ]
        )
        for k in range(horizon + 1):
            for obstacle_index, (center_x, center_y, radius) in enumerate(OBSTACLES):
                center = np.array([center_x, center_y])
                difference = states[k, :2] - center
                distance = np.linalg.norm(difference)
                normal = difference / distance if distance > 1e-8 else np.array([1.0, 0.0])
                constraints.append(
                    normal @ (x[k, :2] - center) + obstacle_slack[k, obstacle_index]
                    >= radius + nominal_obstacle_reserve
                )
        objective = (
            0.5 * cp.sum_squares(x[:-1, :2] - reference[:-1, :2])
            + 0.1 * cp.sum_squares(x[:-1, 2:] - reference[:-1, 2:])
            + cp.sum_squares(u[:, 0])
            + 10.0 * cp.sum_squares(u[:, 1])
            + 10.0 * cp.sum_squares(x[-1, :2] - goal_state[:2])
            + 1e4 * cp.norm1(virtual)
            + 1e5 * cp.sum(obstacle_slack)
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver="CLARABEL")
        if x.value is None or u.value is None:
            raise RuntimeError(f"nominal Pavone seed refinement failed: {problem.status}")
        new_states = np.asarray(x.value)
        new_inputs = np.asarray(u.value)
        change = max(
            float(np.max(np.abs(new_states - states))),
            float(np.max(np.abs(new_inputs - inputs))),
        )
        states, inputs = new_states, new_inputs
        maximum_slack = float(np.max(obstacle_slack.value))
        if change <= 2e-3 and maximum_slack <= 1e-5:
            converged = True
            break
    nonlinear_rollout = np.empty_like(states)
    nonlinear_rollout[0] = initial_state
    for k in range(horizon):
        nonlinear_rollout[k + 1] = car_step(nonlinear_rollout[k], inputs[k], dt)
    defect = float(np.max(np.abs(nonlinear_rollout - states)))
    # Use the exact nominal rollout as the policy center. The remaining goal
    # error is exposed in diagnostics and the robust solve can make a final
    # small correction.
    return nonlinear_rollout, inputs, {
        "nominal_refinement_converged": converged,
        "nominal_refinement_maximum_obstacle_slack": maximum_slack,
        "nominal_refinement_dynamics_defect_before_rollout": defect,
    }


def simulate_policy(
    *,
    nominal_states: np.ndarray,
    nominal_inputs: np.ndarray,
    gains: np.ndarray,
    a_matrices: np.ndarray,
    b_matrices: np.ndarray,
    dt: float,
    true_parameter: np.ndarray,
    disturbance_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = nominal_inputs.shape[0]
    states = np.empty_like(nominal_states)
    controls = np.empty_like(nominal_inputs)
    reconstructed_disturbances = np.zeros((horizon, 4))
    states[0] = nominal_states[0]
    for k in range(horizon):
        correction = np.zeros(2)
        for j in range(k):
            correction += gains[k, j] @ reconstructed_disturbances[j]
        controls[k] = nominal_inputs[k] + correction
        nominal_next_from_actual = car_step(states[k], controls[k], dt)
        injected = np.array(
            [
                dt * true_parameter[0] + disturbance_scale,
                dt * true_parameter[1] + disturbance_scale,
                disturbance_scale,
                disturbance_scale,
            ]
        )
        states[k + 1] = nominal_next_from_actual + injected

        deviation = states[k] - nominal_states[k]
        input_deviation = controls[k] - nominal_inputs[k]
        next_deviation = states[k + 1] - nominal_states[k + 1]
        reconstructed_disturbances[k] = (
            next_deviation
            - a_matrices[k] @ deviation
            - b_matrices[k] @ input_deviation
        )
    return states, controls, reconstructed_disturbances


def write_outputs(
    run_dir: Path,
    nominal_states: np.ndarray,
    nominal_inputs: np.ndarray,
    rollout: np.ndarray,
    applied_inputs: np.ndarray,
    state_widths: np.ndarray,
    input_widths: np.ndarray,
    convergence_history: list[dict[str, float | str | bool]],
    dt: float,
) -> None:
    with (run_dir / "nominal_plan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for k, state in enumerate(nominal_states):
            writer.writerow([k, k * dt, *map(float, state)])
    with (run_dir / "tube_widths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time_s",
                "x_half_width_m",
                "y_half_width_m",
                "heading_half_width_rad",
                "speed_half_width_m_per_s",
            ]
        )
        for k, widths in enumerate(state_widths):
            writer.writerow([k, k * dt, *map(float, widths)])
    with (run_dir / "rollout.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time_s",
                "x_m",
                "y_m",
                "heading_rad",
                "speed_m_per_s",
                "omega_rad_per_s",
                "acceleration_m_per_s2",
            ]
        )
        for k, state in enumerate(rollout):
            control = applied_inputs[k] if k < len(applied_inputs) else (np.nan, np.nan)
            writer.writerow([k, k * dt, *map(float, state), *map(float, control)])
    with (run_dir / "nominal_inputs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time_s",
                "omega_rad_per_s",
                "acceleration_m_per_s2",
                "omega_tube_half_width",
                "acceleration_tube_half_width",
            ]
        )
        for k, (control, widths) in enumerate(zip(nominal_inputs, input_widths)):
            writer.writerow([k, k * dt, *map(float, control), *map(float, widths)])
    with (run_dir / "convergence_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(convergence_history[0]))
        writer.writeheader()
        writer.writerows(convergence_history)


def plot_results(
    run_dir: Path,
    nominal: np.ndarray,
    rollout: np.ndarray,
    widths: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 9))
    axis.plot(nominal[:, 0], nominal[:, 1], "--", lw=2.2, label="Pavone RMPC nominal")
    axis.plot(rollout[:, 0], rollout[:, 1], lw=2.2, label="Pavone RMPC rollout")
    for index, (center_x, center_y, radius) in enumerate(OBSTACLES):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                color="tab:red",
                alpha=0.35,
                label="obstacle" if index == 0 else None,
            )
        )
    for k in range(0, len(nominal), 5):
        axis.add_patch(
            Rectangle(
                nominal[k, :2] - widths[k, :2],
                2.0 * widths[k, 0],
                2.0 * widths[k, 1],
                color="tab:green",
                alpha=0.10,
            )
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title("Solve-Once Affine Disturbance-Feedback RMPC (No LEB)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "rollout_with_tubes.png", dpi=250)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    if args.nominal_obstacle_reserve < 0.0:
        raise ValueError("nominal obstacle reserve must be nonnegative")
    run_dir = args.output_dir / "pavone_affine_df"
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_state = np.array([0.0, 1.0, -0.5 * np.pi, args.initial_speed])
    goal_state = np.array([args.goal_x, args.goal_y, -0.5 * np.pi, args.goal_speed])
    warmstart_diagnostics: dict[str, float | str | bool]
    ccm_warmstart = args.ccm_warmstart
    ccm_warmstart_explicit = ccm_warmstart is not None
    if ccm_warmstart is None and not args.no_ccm_warmstart:
        candidates = (
            args.output_dir / "certified_ccm" / "nominal_plan.csv",
            Path(__file__).with_name("comparison_results")
            / "certified_ccm"
            / "nominal_plan.csv",
        )
        ccm_warmstart = next((path for path in candidates if path.is_file()), None)
    if ccm_warmstart is not None and not args.no_ccm_warmstart:
        try:
            seed_states, seed_inputs, warmstart_diagnostics = load_ccm_seed(
                ccm_warmstart,
                initial_state=initial_state,
                goal_state=goal_state,
                horizon=args.horizon,
                dt=args.dt,
                terminal_tolerance=args.terminal_position_tolerance,
            )
            print(f"Using CCM nominal trajectory warm start: {ccm_warmstart}")
        except (OSError, ValueError) as error:
            if ccm_warmstart_explicit:
                raise
            print(f"CCM warm start rejected ({error}); using geometric seed")
            seed_states, seed_inputs = geometric_seed(
                initial_state, goal_state, args.horizon, args.dt, args.route_offset
            )
            warmstart_diagnostics = {
                "warmstart_source": "geometric seed",
                "ccm_warmstart_rejection": str(error),
            }
    else:
        seed_states, seed_inputs = geometric_seed(
            initial_state, goal_state, args.horizon, args.dt, args.route_offset
        )
        warmstart_diagnostics = {"warmstart_source": "geometric seed"}
    progress = np.linspace(0.0, 1.0, args.horizon + 1)
    reference = (1.0 - progress[:, None]) * initial_state + progress[:, None] * goal_state
    reference[:, 2] = -0.5 * np.pi
    heading_center = args.certified_velocity_direction_rad
    heading_half_width = np.arccos(args.certified_minimum_speed / args.maximum_speed)
    seed_states, seed_inputs, nominal_diagnostics = refine_nominal_seed(
        initial_state=initial_state,
        goal_state=goal_state,
        seed_states=seed_states,
        seed_inputs=seed_inputs,
        dt=args.dt,
        iterations=args.planner_iterations,
        x_min=args.x_min,
        x_max=args.x_max,
        minimum_speed=args.certified_minimum_speed,
        maximum_speed=args.maximum_speed,
        heading_min=heading_center - heading_half_width,
        heading_max=heading_center + heading_half_width,
        terminal_tolerance=args.terminal_position_tolerance,
        nominal_obstacle_reserve=args.nominal_obstacle_reserve,
    )
    disturbance_half_width = np.array(
        [
            args.dt * args.parameter_uncertainty_bound + args.exogenous_disturbance_scale,
            args.dt * args.parameter_uncertainty_bound + args.exogenous_disturbance_scale,
            args.exogenous_disturbance_scale,
            args.exogenous_disturbance_scale,
        ]
    )
    solver = AffineDisturbanceFeedbackSolver(
        nominal_step=lambda state, control: car_step(state, control, args.dt),
        linearize=lambda state, control: car_linearization(state, control, args.dt),
        state_cost=np.diag([0.5, 0.5, 0.1, 0.1]),
        input_cost=np.diag([1.0, 10.0]),
        terminal_cost=np.diag([10.0, 10.0, 0.0, 0.0]),
        bounds=BoxBounds(
            state_lower=np.array(
                [args.x_min, -5.0, heading_center - heading_half_width, args.certified_minimum_speed]
            ),
            state_upper=np.array(
                [args.x_max, 5.0, heading_center + heading_half_width, args.maximum_speed]
            ),
            input_lower=np.array([-4.0, -1.0]),
            input_upper=np.array([4.0, 1.0]),
        ),
        disturbance_half_width=disturbance_half_width,
        config=SolverConfig(
            max_iterations=args.robust_policy_iterations,
            convergence_tolerance=args.convergence_tolerance,
            feedback_memory=(None if args.feedback_memory == 0 else args.feedback_memory),
            verbose=args.solver_verbose,
        ),
    )
    start = time.perf_counter()
    result = solver.solve(
        initial_state=initial_state,
        reference=reference,
        initial_inputs=seed_inputs,
        initial_states=seed_states,
        obstacles=OBSTACLES,
        terminal_position_tolerance=args.terminal_position_tolerance,
    )
    solve_seconds = time.perf_counter() - start
    rollout, applied_inputs, reconstructed = simulate_policy(
        nominal_states=result.states,
        nominal_inputs=result.inputs,
        gains=result.disturbance_gains,
        a_matrices=result.linearization_a,
        b_matrices=result.linearization_b,
        dt=args.dt,
        true_parameter=np.array([-0.05, 0.05]),
        disturbance_scale=args.exogenous_disturbance_scale,
    )
    deviations = np.abs(rollout - result.states)
    obstacle_clearance = (
        np.linalg.norm(rollout[:, None, :2] - OBSTACLES[None, :, :2], axis=-1)
        - OBSTACLES[None, :, 2]
    )
    write_outputs(
        run_dir,
        result.states,
        result.inputs,
        rollout,
        applied_inputs,
        result.state_half_widths,
        result.input_half_widths,
        result.convergence_history,
        args.dt,
    )
    np.savez_compressed(
        run_dir / "policy.npz",
        nominal_states=result.states,
        nominal_inputs=result.inputs,
        disturbance_gains=result.disturbance_gains,
        state_response=result.state_response,
        linearization_a=result.linearization_a,
        linearization_b=result.linearization_b,
        disturbance_half_width=disturbance_half_width,
    )
    plot_results(run_dir, result.states, rollout, result.state_half_widths)
    summary = {
        "method": "Pavone affine disturbance-feedback robust MPC",
        "solve_once": True,
        "replanned_during_rollout": False,
        "causal_disturbance_feedback": True,
        "linearization_error_bound_enabled": False,
        "nonlinear_tube_certified": False,
        "tube_guarantee_scope": "optimized LTV prediction model",
        "parameter_learning_enabled": False,
        "disturbance_feedback_memory_steps": (
            "full" if args.feedback_memory == 0 else args.feedback_memory
        ),
        "nominal_obstacle_reserve_m": args.nominal_obstacle_reserve,
        "solve_seconds": solve_seconds,
        "iterations": result.iterations,
        "converged": result.converged,
        "configured_max_iterations": args.robust_policy_iterations,
        "convergence_history": result.convergence_history,
        "rollout_inside_predicted_ltv_tube": bool(
            np.all(deviations <= result.state_half_widths + 1e-8)
        ),
        "maximum_tube_violation": float(np.max(deviations - result.state_half_widths)),
        "maximum_reconstructed_disturbance_ratio": float(
            np.max(np.abs(reconstructed) / disturbance_half_width[None, :])
        ),
        "minimum_actual_speed": float(np.min(rollout[:, 3])),
        "maximum_actual_speed": float(np.max(rollout[:, 3])),
        "maximum_absolute_turn_rate": float(np.max(np.abs(applied_inputs[:, 0]))),
        "maximum_absolute_acceleration": float(np.max(np.abs(applied_inputs[:, 1]))),
        "minimum_actual_obstacle_clearance": float(np.min(obstacle_clearance)),
        "distance_to_goal": float(np.linalg.norm(rollout[-1, :2] - goal_state[:2])),
        **warmstart_diagnostics,
        **nominal_diagnostics,
        **result.diagnostics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {run_dir}")
    return run_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the solve-once Pavone affine disturbance-feedback car baseline."
    )
    result.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("pavone_results"))
    result.add_argument("--horizon", type=int, default=75)
    result.add_argument("--dt", type=float, default=0.05)
    result.add_argument("--initial-speed", type=float, default=0.38)
    result.add_argument("--goal-speed", type=float, default=0.60)
    result.add_argument("--goal-x", type=float, default=0.25)
    result.add_argument("--goal-y", type=float, default=-1.0)
    result.add_argument("--x-min", type=float, default=-1.5)
    result.add_argument("--x-max", type=float, default=1.5)
    result.add_argument("--maximum-speed", type=float, default=2.0)
    result.add_argument("--certified-minimum-speed", type=float, default=0.10)
    result.add_argument(
        "--certified-velocity-direction-rad",
        type=float,
        default=float(np.arctan2(-2.0, 0.25)),
    )
    result.add_argument("--terminal-position-tolerance", type=float, default=0.01)
    result.add_argument("--parameter-uncertainty-bound", type=float, default=0.075)
    result.add_argument("--exogenous-disturbance-scale", type=float, default=0.00030)
    result.add_argument("--planner-iterations", type=int, default=8)
    result.add_argument(
        "--robust-policy-iterations",
        type=int,
        default=12,
        help="maximum sequential robust-policy refinements (default: 12)",
    )
    result.add_argument("--convergence-tolerance", type=float, default=2e-3)
    result.add_argument("--route-offset", type=float, default=-0.75)
    warmstart = result.add_mutually_exclusive_group()
    warmstart.add_argument(
        "--ccm-warmstart",
        type=Path,
        help="CCM nominal_plan.csv to use as the Pavone nominal SCP warm start",
    )
    warmstart.add_argument(
        "--no-ccm-warmstart",
        action="store_true",
        help="disable automatic CCM warm-start discovery and use the geometric seed",
    )
    result.add_argument(
        "--nominal-obstacle-reserve",
        type=float,
        default=0.12,
        help="extra obstacle clearance used only by the nominal seed refinement",
    )
    result.add_argument(
        "--feedback-memory",
        type=int,
        default=12,
        help="past disturbance steps used by each input (0 selects full history)",
    )
    result.add_argument("--solver-verbose", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
