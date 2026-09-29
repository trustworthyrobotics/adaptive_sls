"""Adaptive+LEB comparison using the certified CCM experiment's domain."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != REPO_ROOT]
if str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))
source_path = REPO_ROOT / "experiments" / "car" / "car_adaptive_unmatched_v2.py"
module_spec = importlib.util.spec_from_file_location("car_adaptive_unmatched_v2", source_path)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"could not load adaptive experiment from {source_path}")
adaptive_module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = adaptive_module
module_spec.loader.exec_module(adaptive_module)
main = adaptive_module.main

OBSTACLES = np.array(
    [
        [-0.25, 0.20, 0.23],
        [0.25, -0.25, 0.23],
    ]
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_run(tube_csv: Path) -> dict[str, float | bool]:
    run_dir = tube_csv.parent
    state_columns = ("x_m", "y_m", "heading_rad", "speed_m_per_s")
    tube_columns = (
        "x_half_width_m",
        "y_half_width_m",
        "heading_half_width_rad",
        "speed_half_width_m_per_s",
    )
    rollout = np.asarray(
        [[float(row[column]) for column in state_columns] for row in _csv_rows(run_dir / "rollout.csv")]
    )
    nominal = np.asarray(
        [[float(row[column]) for column in state_columns] for row in _csv_rows(run_dir / "nominal_plan.csv")]
    )
    tube = np.asarray(
        [[float(row[column]) for column in tube_columns] for row in _csv_rows(tube_csv)]
    )
    if rollout.shape != nominal.shape or rollout.shape != tube.shape:
        raise RuntimeError("adaptive rollout, nominal plan, and tube shapes do not match")

    heading_center = np.arctan2(-2.0, 0.25)
    heading_half_width = np.arccos(0.10 / 2.0)
    obstacle_clearance = np.linalg.norm(
        rollout[:, None, :2] - OBSTACLES[None, :, :2], axis=-1
    ) - OBSTACLES[None, :, 2]
    finite = bool(
        np.all(np.isfinite(rollout))
        and np.all(np.isfinite(nominal))
        and np.all(np.isfinite(tube))
    )
    summary = {
        "all_values_finite": finite,
        "rollout_inside_leb_tube": bool(np.all(np.abs(rollout - nominal) <= tube + 1e-8)),
        "minimum_robust_speed_margin": float(np.min(nominal[:, 3] - tube[:, 3] - 0.10)),
        "minimum_actual_speed": float(np.min(rollout[:, 3])),
        "maximum_actual_speed": float(np.max(rollout[:, 3])),
        "actual_heading_min": float(np.min(rollout[:, 2])),
        "actual_heading_max": float(np.max(rollout[:, 2])),
        "minimum_actual_obstacle_clearance": float(np.min(obstacle_clearance)),
        "distance_to_goal": float(np.linalg.norm(rollout[-1, :2] - np.array([0.25, -1.0]))),
        "maximum_x_tube_half_width": float(np.max(tube[:, 0])),
        "maximum_y_tube_half_width": float(np.max(tube[:, 1])),
        "mean_position_tube_area": float(np.mean(4.0 * tube[:, 0] * tube[:, 1])),
    }
    valid = (
        finite
        and summary["rollout_inside_leb_tube"]
        and summary["minimum_robust_speed_margin"] >= -1e-8
        and summary["minimum_actual_speed"] >= 0.10 - 1e-8
        and summary["maximum_actual_speed"] <= 2.0 + 1e-8
        and summary["actual_heading_min"] >= heading_center - heading_half_width - 1e-8
        and summary["actual_heading_max"] <= heading_center + heading_half_width + 1e-8
        and summary["minimum_actual_obstacle_clearance"] >= -1e-8
    )
    summary["verified_on_ccm_domain"] = bool(valid)
    (run_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not valid:
        raise RuntimeError(f"adaptive+LEB verification failed: {summary}")
    print(json.dumps(summary, indent=2))
    return summary


def run(
    output_dir: Path,
    horizon: int,
    dt: float,
    parameter_uncertainty_bound: float,
    disturbance_scale: float,
    q_bar_scale: float,
) -> Path:
    # Projection along the start-to-goal direction is at least 0.10. Together with
    # ||velocity|| <= 2 this implies the following non-wrapping heading sector.
    heading_center = np.arctan2(-2.0, 0.25)
    heading_half_width = np.arccos(0.10 / 2.0)
    tube_csv = main(
        adaptive=True,
        enable_leb=True,
        output_dir=output_dir,
        horizon=horizon,
        time_step_s=dt,
        exogenous_disturbance_scale=disturbance_scale,
        parameter_uncertainty_bound=parameter_uncertainty_bound,
        sls_q_bar_scale=q_bar_scale,
        model_label="Positive-Speed Adaptive Dubins Car",
        terminal_state_weights=(10.0, 10.0, 0.0, 0.0),
        enforce_terminal_position=True,
        terminal_position_tolerance=0.01,
        goal_x_m=0.25,
        goal_y_m=-1.0,
        initial_speed_m_per_s=0.38,
        goal_speed_m_per_s=0.60,
        minimum_speed_m_per_s=0.10,
        # Retain the original inactive upper bound in the adaptive ADMM
        # formulation; its verified rollout remains below the CCM limit 2.0.
        maximum_speed_m_per_s=10.0,
        heading_min_rad=heading_center - heading_half_width,
        heading_max_rad=heading_center + heading_half_width,
        x_min_m=-1.5,
        x_max_m=1.5,
        enable_information_cost=False,
        enable_edagger_f_cost=False,
        shared_physical_nominal_solve=True,
        track_nominal_during_robust_solve=True,
        obstacle=(-0.25, 0.20, 0.23),
        additional_obstacles=(
            (0.25, -0.25, 0.23),
        ),
    )
    verify_run(tube_csv)
    return tube_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive+LEB on the positive-speed CCM comparison domain."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("adaptive_leb_results"),
    )
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--parameter-uncertainty-bound", type=float, default=0.075)
    parser.add_argument("--exogenous-disturbance-scale", type=float, default=0.00030)
    parser.add_argument("--q-bar-scale", type=float, default=10.0)
    arguments = parser.parse_args()
    run(
        arguments.output_dir,
        arguments.horizon,
        arguments.dt,
        arguments.parameter_uncertainty_bound,
        arguments.exogenous_disturbance_scale,
        arguments.q_bar_scale,
    )
