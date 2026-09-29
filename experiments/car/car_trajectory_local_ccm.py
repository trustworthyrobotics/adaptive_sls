"""Solve once and roll out a robust TVLQR (trajectory-local CCM) car tube.

The physical model, uncertainty, disturbance convention, constraints, and
default map match ``car_adaptive_unmatched_v2.py`` and its
``run_adaptive_nonadaptive_comparison.sh`` invocation.  The optimization is
performed once.  Every simulated step uses the frozen feedback policy

    u[k] = v_star[k] + K[k] (x[k] - z_star[k]).

This is a numerical trajectory-local CCM approximation, not a global CCM
certificate; see ``baselines/ccm/trajectory_local_ccm.py``.
"""

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
from matplotlib.patches import Circle, Ellipse
import numpy as np

from baselines.ccm import (
    BoxBounds,
    SolverConfig,
    TrajectoryLocalCCMSolver,
    UncertaintySet,
)


DEFAULT_OBSTACLES = (
    (-0.25, 0.20, 0.23),
    (0.25, -0.35, 0.23),
    (-0.26, -0.90, 0.23),
    (0.14, -1.45, 0.23),
)
STATE_LABELS = ("x", "y", "heading", "speed")


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_car_model(dt: float):
    def nominal_step(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        px, py, heading, speed = state
        omega, acceleration = control
        return np.array(
            [
                px + dt * speed * np.cos(heading),
                py + dt * speed * np.sin(heading),
                heading + dt * omega,
                speed + dt * acceleration,
            ],
            dtype=float,
        )

    def uncertain_step(
        state: np.ndarray,
        control: np.ndarray,
        parameter: np.ndarray,
        disturbance: np.ndarray,
    ) -> np.ndarray:
        result = nominal_step(state, control)
        result[0] += dt * parameter[0]
        result[1] += dt * parameter[1]
        return result + disturbance

    def linearize(state: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del control
        heading, speed = state[2], state[3]
        A = np.eye(4)
        A[0, 2] = -dt * speed * np.sin(heading)
        A[0, 3] = dt * np.cos(heading)
        A[1, 2] = dt * speed * np.cos(heading)
        A[1, 3] = dt * np.sin(heading)
        B = np.zeros((4, 2))
        B[2, 0] = dt
        B[3, 1] = dt
        return A, B

    return nominal_step, uncertain_step, linearize


def geometric_initial_guess(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    horizon: int,
    dt: float,
    route_offset: float,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a smooth left-route state/control seed for convexification."""
    progress = np.linspace(0.0, 1.0, horizon + 1)
    smooth_progress = 3.0 * progress**2 - 2.0 * progress**3
    states = np.zeros((horizon + 1, 4))
    states[:, 0] = (
        initial_state[0] * (1.0 - smooth_progress)
        + goal_state[0] * smooth_progress
        + route_offset * np.sin(np.pi * progress) ** 2
    )
    states[:, 1] = (
        initial_state[1] * (1.0 - smooth_progress)
        + goal_state[1] * smooth_progress
    )
    dx = np.gradient(states[:, 0], dt)
    dy = np.gradient(states[:, 1], dt)
    headings = np.unwrap(np.arctan2(dy, dx))
    speeds = np.hypot(dx, dy)
    states[:, 2] = headings
    states[:, 3] = speeds
    states[0] = initial_state
    states[-1, 2:] = goal_state[2:]

    controls = np.column_stack(
        [
            np.diff(states[:, 2]) / dt,
            np.diff(states[:, 3]) / dt,
        ]
    )
    controls = np.clip(controls, input_lower, input_upper)
    return states, controls


def save_csvs(result, run_dir: Path, dt: float, rollout: np.ndarray, applied_inputs: np.ndarray) -> None:
    with (run_dir / "nominal_plan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for k, state in enumerate(result.states):
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
                "method",
            ]
        )
        for k, widths in enumerate(result.state_half_widths):
            writer.writerow(
                [k, k * dt, *map(float, widths), "trajectory_local_ccm_tvlqr"]
            )

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
            control = applied_inputs[k] if k < applied_inputs.shape[0] else (np.nan, np.nan)
            writer.writerow([k, k * dt, *map(float, state), *map(float, control)])


def plot_results(
    result,
    rollout: np.ndarray,
    obstacles: np.ndarray,
    goal: np.ndarray,
    dt: float,
    run_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 9))
    axis.plot(result.states[:, 0], result.states[:, 1], "--", lw=2.2, label="robust nominal plan")
    axis.plot(rollout[:, 0], rollout[:, 1], lw=2.2, label="closed-loop rollout")
    for obstacle_index, (center_x, center_y, radius) in enumerate(obstacles):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                facecolor="tab:red",
                edgecolor="black",
                alpha=0.35,
                label="obstacle" if obstacle_index == 0 else None,
            )
        )
    stride = max(1, (result.states.shape[0] - 1) // 20)
    for k in range(0, result.states.shape[0], stride):
        axis.add_patch(
            Ellipse(
                result.states[k, :2],
                2.0 * result.state_half_widths[k, 0],
                2.0 * result.state_half_widths[k, 1],
                color="tab:blue",
                alpha=0.10,
            )
        )
    axis.scatter([result.states[0, 0]], [result.states[0, 1]], marker="o", s=80, label="start")
    axis.scatter([goal[0]], [goal[1]], marker="*", s=180, label="goal")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title("Robust TVLQR Tube (Trajectory-Local CCM Approximation)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "rollout_with_tubes.png", dpi=250)
    plt.close(figure)

    times = np.arange(result.states.shape[0]) * dt
    deviations = np.abs(rollout - result.states)
    deviations[:, 2] = np.abs(wrap_to_pi(rollout[:, 2] - result.states[:, 2]))
    figure, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    units = ("m", "m", "rad", "m/s")
    for index, axis in enumerate(axes):
        axis.plot(times, result.state_half_widths[:, index], lw=3, label="tube half-width")
        axis.plot(times, deviations[:, index], lw=1.8, label="realized deviation")
        axis.set_ylabel(units[index])
        axis.set_title(STATE_LABELS[index])
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(run_dir / "deviation_vs_tube_width.png", dpi=250)
    plt.close(figure)


def run_experiment(args: argparse.Namespace) -> Path:
    if args.horizon < 1 or args.dt <= 0.0:
        raise ValueError("horizon and dt must be positive")
    if args.parameter_uncertainty_bound < 0.0 or args.exogenous_disturbance_scale < 0.0:
        raise ValueError("uncertainty bounds must be nonnegative")

    run_dir = args.output_dir / "trajectory_local_ccm_tvlqr"
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_state = np.array([0.0, 1.0, -0.5 * np.pi, 0.0])
    goal = np.array([args.goal_x, args.goal_y, -0.5 * np.pi, 0.0])
    input_lower = np.array([-4.0, -1.0])
    input_upper = np.array([4.0, 1.0])
    obstacles = np.asarray(
        [
            (args.obstacle_x, args.obstacle_y, args.obstacle_radius),
            (args.second_obstacle_x, args.second_obstacle_y, args.second_obstacle_radius),
            (args.third_obstacle_x, args.third_obstacle_y, args.third_obstacle_radius),
            *(args.extra_obstacle or [DEFAULT_OBSTACLES[3]]),
        ],
        dtype=float,
    )
    nominal_step, uncertain_step, linearize = make_car_model(args.dt)
    seed_states, seed_inputs = geometric_initial_guess(
        initial_state,
        goal,
        args.horizon,
        args.dt,
        args.route_offset,
        input_lower,
        input_upper,
    )
    solver = TrajectoryLocalCCMSolver(
        nominal_step=nominal_step,
        uncertain_step=uncertain_step,
        linearize=linearize,
        state_cost=np.diag([0.5, 0.5, 0.1, 0.1]),
        input_cost=np.diag([1.0, 10.0]),
        terminal_cost=np.diag([10.0, 10.0, 0.1, 0.1]),
        # Match the adaptive-SLS experiment's stronger x/y response design
        # while keeping its nominal tracking objective unchanged.
        feedback_state_cost=np.diag([10.0, 10.0, 1.0, 1.0]),
        feedback_input_cost=np.diag([0.2, 2.0]),
        feedback_terminal_cost=np.diag([20.0, 20.0, 5.0, 5.0]),
        bounds=BoxBounds(
            state_lower=np.array([args.x_min, -5.0, -np.inf, -1.0]),
            state_upper=np.array([args.x_max, 5.0, np.inf, 10.0]),
            input_lower=input_lower,
            input_upper=input_upper,
        ),
        uncertainty=UncertaintySet(
            parameter_half_width=np.full(2, args.parameter_uncertainty_bound),
            disturbance_half_width=np.full(4, args.exogenous_disturbance_scale),
        ),
        config=SolverConfig(
            max_iterations=args.planner_iterations,
            tube_directions=args.tube_directions,
            tube_safety_factor=args.tube_safety_factor,
            verbose=args.verbose_solver,
        ),
    )
    reference = np.tile(goal, (args.horizon + 1, 1))
    print("Solving one robust TVLQR/trajectory-local CCM plan...")
    start_time = time.perf_counter()
    result = solver.solve(
        initial_state=initial_state,
        reference=reference,
        initial_inputs=seed_inputs,
        initial_states=seed_states,
        obstacles=obstacles,
    )
    solve_seconds = time.perf_counter() - start_time

    # Match car_adaptive_unmatched_v2.py exactly: fixed true bias and w=ones.
    true_parameter = np.array([-0.05, 0.05])
    disturbance = np.full(4, args.exogenous_disturbance_scale)
    number_of_rollouts = 4  # Matches NUM_RANDOM + NUM_ADV in the adaptive experiment.
    rollouts = np.empty((number_of_rollouts, *result.states.shape), dtype=float)
    applied_inputs_all = np.empty((number_of_rollouts, *result.inputs.shape), dtype=float)
    saturation_count = 0
    for rollout_index in range(number_of_rollouts):
        rollouts[rollout_index, 0] = initial_state
        for k in range(args.horizon):
            error = rollouts[rollout_index, k] - result.states[k]
            error[2] = wrap_to_pi(error[2])
            raw_control = result.inputs[k] + result.gains[k] @ error
            control = np.clip(raw_control, input_lower, input_upper)
            saturation_count += int(not np.allclose(raw_control, control, atol=1e-9))
            applied_inputs_all[rollout_index, k] = control
            rollouts[rollout_index, k + 1] = uncertain_step(
                rollouts[rollout_index, k], control, true_parameter, disturbance
            )

    deviations = np.abs(rollouts - result.states[None, :, :])
    deviations[:, :, 2] = np.abs(
        wrap_to_pi(rollouts[:, :, 2] - result.states[None, :, 2])
    )
    containment_margin = result.state_half_widths[None, :, :] - deviations
    contained = bool(np.all(containment_margin >= -1e-8))
    rollout = rollouts[0]
    applied_inputs = applied_inputs_all[0]
    save_csvs(result, run_dir, args.dt, rollout, applied_inputs)
    plot_results(result, rollout, obstacles, goal, args.dt, run_dir)
    summary = {
        "method": "robust TVLQR tube (trajectory-local CCM approximation)",
        "global_ccm_certificate": False,
        "solve_once": True,
        "solve_seconds": solve_seconds,
        "planner_converged": result.converged,
        "planner_iterations": result.iterations,
        "objective": result.objective,
        "true_parameter": true_parameter.tolist(),
        "parameter_half_width": args.parameter_uncertainty_bound,
        "disturbance_half_width": args.exogenous_disturbance_scale,
        "rollout_disturbance_vertex": disturbance.tolist(),
        "number_of_identical_rollouts": number_of_rollouts,
        "rollout_contained_in_sampled_tube": contained,
        "minimum_containment_margin_after_initial_state": float(
            np.min(containment_margin[:, 1:, :])
        ),
        "control_saturation_count": saturation_count,
        **result.diagnostics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the solve-once robust TVLQR (trajectory-local CCM) car baseline."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("ccm_car_results"))
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--exogenous-disturbance-scale", type=float, default=0.00030)
    parser.add_argument("--parameter-uncertainty-bound", type=float, default=0.10)
    parser.add_argument("--goal-x", type=float, default=-0.75)
    parser.add_argument("--goal-y", type=float, default=-2.25)
    parser.add_argument("--x-min", type=float, default=-1.5)
    parser.add_argument("--x-max", type=float, default=0.6)
    parser.add_argument("--route-offset", type=float, default=-0.75)
    parser.add_argument("--obstacle-x", type=float, default=DEFAULT_OBSTACLES[0][0])
    parser.add_argument("--obstacle-y", type=float, default=DEFAULT_OBSTACLES[0][1])
    parser.add_argument("--obstacle-radius", type=float, default=DEFAULT_OBSTACLES[0][2])
    parser.add_argument("--second-obstacle-x", type=float, default=DEFAULT_OBSTACLES[1][0])
    parser.add_argument("--second-obstacle-y", type=float, default=DEFAULT_OBSTACLES[1][1])
    parser.add_argument("--second-obstacle-radius", type=float, default=DEFAULT_OBSTACLES[1][2])
    parser.add_argument("--third-obstacle-x", type=float, default=DEFAULT_OBSTACLES[2][0])
    parser.add_argument("--third-obstacle-y", type=float, default=DEFAULT_OBSTACLES[2][1])
    parser.add_argument("--third-obstacle-radius", type=float, default=DEFAULT_OBSTACLES[2][2])
    parser.add_argument(
        "--extra-obstacle",
        nargs=3,
        action="append",
        type=float,
        default=None,
        metavar=("X", "Y", "RADIUS"),
    )
    parser.add_argument("--planner-iterations", type=int, default=18)
    parser.add_argument("--tube-directions", type=int, default=32)
    parser.add_argument("--tube-safety-factor", type=float, default=1.10)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
