"""Faithful solve-once CCM-RMPC baseline for a positive-speed car domain."""

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

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from baselines.ccm import (
    car_to_linearizing_coordinates,
    ccm_feedback,
    design_feedback_linearized_car_ccm,
    linearizing_to_car_state,
    pulled_back_metric,
)


DEFAULT_OBSTACLES = np.array(
    [
        [-0.25, 0.20, 0.23],
        [0.25, -0.25, 0.23],
    ]
)


def geometric_seed(
    initial_y: np.ndarray, goal_y: np.ndarray, horizon: int, route_offset: float
) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, horizon + 1)
    smooth = 3.0 * progress**2 - 2.0 * progress**3
    seed = (1.0 - smooth[:, None]) * initial_y + smooth[:, None] * goal_y
    # S-shaped seed: pass right of the upper-left obstacle and then left of
    # the middle-right obstacle before returning to the goal.
    seed[:, 0] += route_offset * np.sin(2.0 * np.pi * progress)
    return seed


def solve_robust_plan(
    *,
    initial_y: np.ndarray,
    goal_y: np.ndarray,
    obstacles: np.ndarray,
    horizon: int,
    dt: float,
    certificate,
    x_min: float,
    x_max: float,
    maximum_speed: float,
    certified_minimum_speed: float,
    certified_velocity_direction_rad: float,
    terminal_position_tolerance: float,
    iterations: int,
    route_offset: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    times = np.arange(horizon + 1) * dt
    radii = np.asarray(certificate.radius(times))
    position_radius, velocity_radius, input_radius = certificate.physical_half_widths(radii)
    seed = geometric_seed(initial_y, goal_y, horizon, route_offset)
    status = "not_solved"
    numerical_margin = 1e-6
    velocity_direction = np.array(
        [
            np.cos(certified_velocity_direction_rad),
            np.sin(certified_velocity_direction_rad),
        ]
    )

    for _ in range(iterations):
        Y = cp.Variable((horizon + 1, 4))
        virtual_inputs = cp.Variable((horizon, 2))
        obstacle_slack = cp.Variable((horizon + 1, obstacles.shape[0]), nonneg=True)
        constraints = [Y[0] == initial_y]
        for k in range(horizon):
            constraints.extend(
                [
                    Y[k + 1, :2]
                    == Y[k, :2] + dt * Y[k, 2:] + 0.5 * dt**2 * virtual_inputs[k],
                    Y[k + 1, 2:] == Y[k, 2:] + dt * virtual_inputs[k],
                    cp.norm(virtual_inputs[k], 2)
                    <= 1.0 - input_radius[k] - numerical_margin,
                ]
            )
        for k in range(horizon + 1):
            constraints.extend(
                [
                    Y[k, 0] >= x_min + position_radius[k],
                    Y[k, 0] <= x_max - position_radius[k],
                    Y[k, 1] >= -5.0 + position_radius[k],
                    Y[k, 1] <= 5.0 - position_radius[k],
                    cp.norm(Y[k, 2:], 2) <= maximum_speed - velocity_radius[k],
                    velocity_direction @ Y[k, 2:] - velocity_radius[k]
                    >= certified_minimum_speed + numerical_margin,
                ]
            )
            for obstacle_index, (center_x, center_y, radius) in enumerate(obstacles):
                center = np.array([center_x, center_y])
                difference = seed[k, :2] - center
                norm = np.linalg.norm(difference)
                normal = difference / norm if norm > 1e-9 else np.array([-1.0, 0.0])
                constraints.append(
                    normal @ (Y[k, :2] - center) + obstacle_slack[k, obstacle_index]
                    >= radius + position_radius[k] + numerical_margin
                )

        constraints.append(
            cp.norm_inf(Y[-1, :2] - goal_y[:2]) <= terminal_position_tolerance
        )

        position_error = Y[:, :2] - goal_y[None, :2]
        velocity_error = Y[:, 2:] - goal_y[None, 2:]
        objective = (
            0.5 * cp.sum_squares(position_error[:-1])
            + 0.1 * cp.sum_squares(velocity_error[:-1])
            + cp.sum_squares(virtual_inputs)
            + 10.0 * cp.sum_squares(position_error[-1])
            + 0.1 * cp.sum_squares(velocity_error[-1])
            + 1e6 * cp.sum(obstacle_slack)
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver="CLARABEL")
        status = str(problem.status)
        if Y.value is None or virtual_inputs.value is None:
            raise RuntimeError(f"CCM robust planning problem failed with status {status}")
        new_seed = np.asarray(Y.value)
        if np.max(np.abs(new_seed - seed)) < 1e-6:
            seed = new_seed
            break
        seed = new_seed

    nominal_y = seed
    nominal_virtual_inputs = np.asarray(virtual_inputs.value)
    obstacle_distances = np.linalg.norm(
        nominal_y[:, None, :2] - obstacles[None, :, :2], axis=-1
    )
    minimum_obstacle_margin = float(
        np.min(obstacle_distances - obstacles[None, :, 2] - position_radius[:, None])
    )
    maximum_virtual_input_margin = float(
        np.min(1.0 - input_radius - np.linalg.norm(nominal_virtual_inputs, axis=1))
    )
    minimum_directional_speed_margin = float(
        np.min(
            nominal_y[:, 2:] @ velocity_direction
            - velocity_radius
            - certified_minimum_speed
        )
    )
    diagnostics = {
        "solver_status": status,
        "maximum_obstacle_slack": float(np.max(obstacle_slack.value)),
        "minimum_robust_obstacle_margin": minimum_obstacle_margin,
        "minimum_robust_virtual_input_margin": maximum_virtual_input_margin,
        "minimum_certified_speed_margin": minimum_directional_speed_margin,
        "robust_plan_feasible": min(
            minimum_obstacle_margin,
            maximum_virtual_input_margin,
            minimum_directional_speed_margin,
        )
        >= -1e-7,
    }
    return nominal_y, nominal_virtual_inputs, diagnostics


def nominal_state_inside_interval(
    nominal_y: np.ndarray, virtual_input: np.ndarray, tau: float
) -> np.ndarray:
    return np.concatenate(
        [
            nominal_y[:2] + tau * nominal_y[2:] + 0.5 * tau**2 * virtual_input,
            nominal_y[2:] + tau * virtual_input,
        ]
    )


def simulate_closed_loop(
    *,
    initial_state: np.ndarray,
    nominal_y: np.ndarray,
    nominal_virtual_inputs: np.ndarray,
    certificate,
    dt: float,
    true_parameter: np.ndarray,
    continuous_disturbance: np.ndarray,
    integration_substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = nominal_virtual_inputs.shape[0]
    states = np.empty((horizon + 1, 4))
    sample_inputs = np.empty((horizon, 2))
    states[0] = initial_state
    substep = dt / integration_substeps

    def derivative(state: np.ndarray, nominal: np.ndarray, virtual_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical_input = ccm_feedback(state, nominal, virtual_input, certificate)
        heading, speed = state[2], state[3]
        return (
            np.array(
                [
                    speed * np.cos(heading) + true_parameter[0] + continuous_disturbance[0],
                    speed * np.sin(heading) + true_parameter[1] + continuous_disturbance[1],
                    physical_input[0] + continuous_disturbance[2],
                    physical_input[1] + continuous_disturbance[3],
                ]
            ),
            physical_input,
        )

    for k in range(horizon):
        state = states[k].copy()
        sample_inputs[k] = ccm_feedback(
            state, nominal_y[k], nominal_virtual_inputs[k], certificate
        )
        for substep_index in range(integration_substeps):
            tau = substep_index * substep
            z1 = nominal_state_inside_interval(nominal_y[k], nominal_virtual_inputs[k], tau)
            k1, _ = derivative(state, z1, nominal_virtual_inputs[k])
            z2 = nominal_state_inside_interval(
                nominal_y[k], nominal_virtual_inputs[k], tau + 0.5 * substep
            )
            k2, _ = derivative(state + 0.5 * substep * k1, z2, nominal_virtual_inputs[k])
            k3, _ = derivative(state + 0.5 * substep * k2, z2, nominal_virtual_inputs[k])
            z4 = nominal_state_inside_interval(
                nominal_y[k], nominal_virtual_inputs[k], tau + substep
            )
            k4, _ = derivative(state + substep * k3, z4, nominal_virtual_inputs[k])
            state += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states[k + 1] = state
    return states, sample_inputs


def write_outputs(
    run_dir: Path,
    nominal_y: np.ndarray,
    nominal_x: np.ndarray,
    rollout: np.ndarray,
    sample_inputs: np.ndarray,
    radii: np.ndarray,
    position_radius: np.ndarray,
    velocity_radius: np.ndarray,
    dt: float,
) -> None:
    with (run_dir / "nominal_plan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for k, state in enumerate(nominal_x):
            writer.writerow([k, k * dt, *map(float, state)])
    with (run_dir / "tube_widths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time_s",
                "ccm_radius",
                "position_radial_bound_m",
                "velocity_radial_bound_m_per_s",
            ]
        )
        for k in range(nominal_y.shape[0]):
            writer.writerow(
                [k, k * dt, radii[k], position_radius[k], velocity_radius[k]]
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
            control = sample_inputs[k] if k < sample_inputs.shape[0] else [np.nan, np.nan]
            writer.writerow([k, k * dt, *map(float, state), *map(float, control)])


def plot_results(
    run_dir: Path,
    nominal_x: np.ndarray,
    rollout: np.ndarray,
    obstacles: np.ndarray,
    position_radius: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 9))
    axis.plot(nominal_x[:, 0], nominal_x[:, 1], "--", lw=2.2, label="CCM nominal plan")
    axis.plot(rollout[:, 0], rollout[:, 1], lw=2.2, label="CCM closed-loop rollout")
    for index, (center_x, center_y, radius) in enumerate(obstacles):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                color="tab:red",
                alpha=0.35,
                label="obstacle" if index == 0 else None,
            )
        )
    for k in range(0, nominal_x.shape[0], 5):
        axis.add_patch(
            Circle(
                nominal_x[k, :2],
                position_radius[k],
                color="tab:blue",
                alpha=0.10,
            )
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title("Certified Feedback-Linearized CCM Tube")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "rollout_with_tubes.png", dpi=250)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    run_dir = args.output_dir / "certified_ccm"
    run_dir.mkdir(parents=True, exist_ok=True)
    disturbance_rate = args.exogenous_disturbance_scale / args.dt
    certificate = design_feedback_linearized_car_ccm(
        parameter_half_width=args.parameter_uncertainty_bound,
        continuous_disturbance_bound=disturbance_rate,
        certified_maximum_speed=args.maximum_speed,
    )
    initial_state = np.array([0.0, 1.0, -0.5 * np.pi, args.initial_speed])
    goal_state = np.array([args.goal_x, args.goal_y, -0.5 * np.pi, args.goal_speed])
    initial_y = car_to_linearizing_coordinates(initial_state)
    goal_y = car_to_linearizing_coordinates(goal_state)
    start = time.perf_counter()
    nominal_y, nominal_virtual_inputs, planning_diagnostics = solve_robust_plan(
        initial_y=initial_y,
        goal_y=goal_y,
        obstacles=DEFAULT_OBSTACLES,
        horizon=args.horizon,
        dt=args.dt,
        certificate=certificate,
        x_min=args.x_min,
        x_max=args.x_max,
        maximum_speed=args.maximum_speed,
        certified_minimum_speed=args.certified_minimum_speed,
        certified_velocity_direction_rad=args.certified_velocity_direction_rad,
        terminal_position_tolerance=args.terminal_position_tolerance,
        iterations=args.planner_iterations,
        route_offset=args.route_offset,
    )
    solve_seconds = time.perf_counter() - start
    nominal_x = np.asarray([linearizing_to_car_state(state) for state in nominal_y])
    true_parameter = np.array([-0.05, 0.05])
    true_disturbance = np.full(4, disturbance_rate)
    rollout, sample_inputs = simulate_closed_loop(
        initial_state=initial_state,
        nominal_y=nominal_y,
        nominal_virtual_inputs=nominal_virtual_inputs,
        certificate=certificate,
        dt=args.dt,
        true_parameter=true_parameter,
        continuous_disturbance=true_disturbance,
        integration_substeps=args.integration_substeps,
    )
    times = np.arange(args.horizon + 1) * args.dt
    radii = np.asarray(certificate.radius(times))
    position_radius, velocity_radius, _ = certificate.physical_half_widths(radii)
    metric_errors = np.asarray(
        [
            np.sqrt(
                (car_to_linearizing_coordinates(state) - nominal) @ certificate.metric
                @ (car_to_linearizing_coordinates(state) - nominal)
            )
            for state, nominal in zip(rollout, nominal_y)
        ]
    )
    obstacle_clearances = np.linalg.norm(
        rollout[:, None, :2] - DEFAULT_OBSTACLES[None, :, :2], axis=-1
    ) - DEFAULT_OBSTACLES[None, :, 2]
    write_outputs(
        run_dir,
        nominal_y,
        nominal_x,
        rollout,
        sample_inputs,
        radii,
        position_radius,
        velocity_radius,
        args.dt,
    )
    plot_results(run_dir, nominal_x, rollout, DEFAULT_OBSTACLES, position_radius)
    summary = {
        "method": "certified feedback-linearized CCM homothetic tube",
        "solve_once": True,
        "sampled_tube_propagation": False,
        "sample_time_constraint_guarantee": True,
        "continuous_time_ccm_certificate": True,
        "solve_seconds": solve_seconds,
        "contraction_rate": certificate.contraction_rate,
        "certificate_max_eigenvalue": certificate.certificate_max_eigenvalue,
        "maximum_disturbance_metric_norm": certificate.maximum_disturbance_metric_norm,
        "rollout_inside_certified_tube": bool(np.all(metric_errors <= radii + 1e-7)),
        "minimum_tube_margin": float(np.min(radii[1:] - metric_errors[1:])),
        "minimum_actual_speed": float(np.min(rollout[:, 3])),
        "actual_heading_min": float(np.min(rollout[:, 2])),
        "actual_heading_max": float(np.max(rollout[:, 2])),
        "maximum_absolute_turn_rate": float(np.max(np.abs(sample_inputs[:, 0]))),
        "maximum_absolute_acceleration": float(np.max(np.abs(sample_inputs[:, 1]))),
        "minimum_actual_obstacle_clearance": float(np.min(obstacle_clearances)),
        "distance_to_goal": float(np.linalg.norm(rollout[-1, :2] - goal_state[:2])),
        **planning_diagnostics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {run_dir}")
    return run_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the certified solve-once CCM car baseline.")
    result.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("ccm_results"))
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
    result.add_argument("--planner-iterations", type=int, default=6)
    result.add_argument("--route-offset", type=float, default=-0.75)
    result.add_argument("--integration-substeps", type=int, default=10)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
