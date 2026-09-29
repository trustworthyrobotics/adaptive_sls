#!/usr/bin/env python3
"""Generate obstacle-backoff-circle plots from saved quadruped sweep data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = {
    "adaptive": (
        "quadruped_adaptive_damping_mjx_rollout.npz",
        "quadruped_adaptive_damping_xy_obstacle_backoff_circles.png",
        "quadruped_adaptive_damping_xy_box_tubes.png",
        "Adaptive",
    ),
    "nonadaptive": (
        "quadruped_nonadaptive_damping_mjx_rollout.npz",
        "quadruped_nonadaptive_damping_xy_obstacle_backoff_circles.png",
        "quadruped_nonadaptive_damping_xy_box_tubes.png",
        "Non-adaptive",
    ),
}


def load_plot_data(checkpoint: Path) -> dict[str, np.ndarray | float]:
    with np.load(checkpoint) as data:
        return {
            "states": np.asarray(data["xs"], dtype=float),
            "planned_xy": np.asarray(data["planned_xy"], dtype=float),
            "one_step_plans": np.asarray(data["one_step_plans"], dtype=float),
            "horizon_tubes": np.asarray(data["horizon_tubes"], dtype=float),
            "obstacle_backoffs": np.asarray(data["obstacle_backoffs"], dtype=float),
            "obstacle_center": np.asarray(data["obstacle_center"], dtype=float),
            "obstacle_radius": float(np.asarray(data["obstacle_radius"])),
            "admm_tolerance": float(np.asarray(data["admm_tolerance"])),
        }


def generate_plot(
    checkpoint: Path,
    output: Path,
    *,
    mode_title: str,
    forecast_stride: int,
    dpi: int,
) -> None:
    data = load_plot_data(checkpoint)
    states = data["states"]
    planned_xy = data["planned_xy"]
    one_step_plans = data["one_step_plans"]
    horizon_tubes = data["horizon_tubes"]
    obstacle_backoffs = data["obstacle_backoffs"]
    obstacle_center = data["obstacle_center"]
    obstacle_radius = data["obstacle_radius"]
    admm_tolerance = data["admm_tolerance"]

    assert isinstance(states, np.ndarray)
    assert isinstance(planned_xy, np.ndarray)
    assert isinstance(one_step_plans, np.ndarray)
    assert isinstance(horizon_tubes, np.ndarray)
    assert isinstance(obstacle_backoffs, np.ndarray)
    assert isinstance(obstacle_center, np.ndarray)
    assert isinstance(obstacle_radius, float)
    assert isinstance(admm_tolerance, float)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(states[:, 0], states[:, 1], label="executed")
    axis.plot(
        one_step_plans[:, 0],
        one_step_plans[:, 1],
        linestyle="--",
        label="one-step plan",
    )

    forecast_count = min(
        len(planned_xy), len(horizon_tubes), len(obstacle_backoffs)
    )
    for forecast_plot_index, forecast_index in enumerate(
        range(0, forecast_count, forecast_stride)
    ):
        plan = planned_xy[forecast_index]
        tubes = horizon_tubes[forecast_index]
        forecast_backoffs = obstacle_backoffs[forecast_index]
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            alpha=0.2,
            linewidth=0.8,
            label="projected horizons" if forecast_plot_index == 0 else None,
        )
        horizon_count = min(
            plan.shape[0] - 1,
            tubes.shape[0],
            forecast_backoffs.shape[0],
        )
        for horizon_index in range(horizon_count):
            center = np.asarray(plan[horizon_index + 1], dtype=float)
            backoff_values = np.asarray(
                forecast_backoffs[horizon_index], dtype=float
            ).reshape(-1)
            if not backoff_values.size or not np.isfinite(backoff_values[0]):
                continue
            axis.add_patch(
                plt.Circle(
                    center,
                    abs(float(backoff_values[0])),
                    facecolor="tab:purple",
                    edgecolor="tab:purple",
                    linewidth=0.35,
                    alpha=0.06,
                    label=(
                        "obstacle-backoff circles"
                        if forecast_plot_index == 0 and horizon_index == 0
                        else None
                    ),
                )
            )

    deflated_obstacle_radius = max(obstacle_radius - admm_tolerance, 0.0)
    axis.add_patch(
        plt.Circle(
            obstacle_center,
            deflated_obstacle_radius,
            fill=False,
            color="tab:red",
            label="obstacle radius - ADMM tolerance",
        )
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    horizon_length = horizon_tubes.shape[1] if horizon_tubes.ndim >= 2 else 0
    axis.set_title(
        f"{mode_title}: steps 1-{horizon_length} obstacle-backoff circles "
        f"(obstacle radius - {admm_tolerance:g})"
    )
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def generate_box_plot(
    checkpoint: Path,
    output: Path,
    *,
    mode_title: str,
    forecast_stride: int,
    dpi: int,
) -> None:
    data = load_plot_data(checkpoint)
    states = np.asarray(data["states"], dtype=float)
    planned_xy = np.asarray(data["planned_xy"], dtype=float)
    one_step_plans = np.asarray(data["one_step_plans"], dtype=float)
    horizon_tubes = np.asarray(data["horizon_tubes"], dtype=float)
    obstacle_center = np.asarray(data["obstacle_center"], dtype=float)
    obstacle_radius = float(data["obstacle_radius"])
    admm_tolerance = float(data["admm_tolerance"])

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(states[:, 0], states[:, 1], label="executed")
    axis.plot(
        one_step_plans[:, 0],
        one_step_plans[:, 1],
        linestyle="--",
        label="one-step plan",
    )

    forecast_count = min(len(planned_xy), len(horizon_tubes))
    for forecast_plot_index, forecast_index in enumerate(
        range(0, forecast_count, forecast_stride)
    ):
        plan = planned_xy[forecast_index]
        tubes = horizon_tubes[forecast_index]
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            alpha=0.2,
            linewidth=0.8,
            label="projected horizons" if forecast_plot_index == 0 else None,
        )
        horizon_count = min(plan.shape[0] - 1, tubes.shape[0])
        for horizon_index in range(horizon_count):
            center = np.asarray(plan[horizon_index + 1], dtype=float)
            half_width = np.abs(np.asarray(tubes[horizon_index, :2], dtype=float))
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_width)):
                continue
            axis.add_patch(
                plt.Rectangle(
                    (center[0] - half_width[0], center[1] - half_width[1]),
                    2.0 * half_width[0],
                    2.0 * half_width[1],
                    facecolor="tab:green",
                    edgecolor="tab:green",
                    linewidth=0.35,
                    alpha=0.06,
                    label=(
                        "horizon XY box tubes"
                        if forecast_plot_index == 0 and horizon_index == 0
                        else None
                    ),
                )
            )

    deflated_obstacle_radius = max(obstacle_radius - admm_tolerance, 0.0)
    axis.add_patch(
        plt.Circle(
            obstacle_center,
            deflated_obstacle_radius,
            fill=False,
            color="tab:red",
            label="obstacle radius - ADMM tolerance",
        )
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    horizon_length = horizon_tubes.shape[1] if horizon_tubes.ndim >= 2 else 0
    axis.set_title(
        f"{mode_title}: steps 1-{horizon_length} projected XY box tubes "
        f"(obstacle radius - {admm_tolerance:g})"
    )
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_root",
        type=Path,
        nargs="?",
        default=(
            Path(__file__).resolve().parent
            / "sweeps"
            / "damping_40_halfwidth_1p6_admm200_rho20"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=40)
    parser.add_argument("--forecast-stride", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace plots that already exist",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("circles", "boxes", "both"),
        default="circles",
        help="plot obstacle-backoff circles, XY box tubes, or both",
    )
    args = parser.parse_args()

    if args.num_seeds < 1 or args.forecast_stride < 1 or args.dpi < 1:
        raise SystemExit("num-seeds, forecast-stride, and dpi must be positive")

    generated = 0
    skipped = 0
    missing: list[Path] = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for mode, (
            checkpoint_name,
            circle_output_name,
            box_output_name,
            mode_title,
        ) in MODES.items():
            run_dir = args.sweep_root / f"seed_{seed:04d}" / mode
            checkpoint = run_dir / checkpoint_name
            if not checkpoint.exists():
                missing.append(checkpoint)
                continue
            if args.plot_kind in ("circles", "both"):
                output = run_dir / circle_output_name
                if output.exists() and not args.overwrite:
                    skipped += 1
                else:
                    generate_plot(
                        checkpoint,
                        output,
                        mode_title=mode_title,
                        forecast_stride=args.forecast_stride,
                        dpi=args.dpi,
                    )
                    generated += 1
                    print(output)
            if args.plot_kind in ("boxes", "both"):
                output = run_dir / box_output_name
                if output.exists() and not args.overwrite:
                    skipped += 1
                else:
                    generate_box_plot(
                        checkpoint,
                        output,
                        mode_title=mode_title,
                        forecast_stride=args.forecast_stride,
                        dpi=args.dpi,
                    )
                    generated += 1
                    print(output)

    print(f"Generated {generated} plots; skipped {skipped} existing plots.")
    if missing:
        print(f"Missing {len(missing)} checkpoints:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
