"""Overlay every saved rollout on a car-ablation x/y tube plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


def plot(case_dir: Path, output: Path) -> None:
    with np.load(case_dir / "rollout_metrics.npz") as archive:
        rollouts = np.asarray(archive["rollout_states"], dtype=float)
        nominal = np.asarray(archive["nominal_states"], dtype=float)
        tube = np.asarray(archive["state_tube_half_widths"], dtype=float)
        goal = np.asarray(archive["goal_state"], dtype=float)
    with np.load(case_dir / "controller.npz") as controller:
        obstacles = np.asarray(controller["obstacles"], dtype=float)

    horizon = min(rollouts.shape[1], nominal.shape[0], tube.shape[0])
    figure, axis = plt.subplots(figsize=(7, 6.2), layout="constrained")
    for step in range(0, horizon, 5):
        width, height = 2.0 * tube[step, 0], 2.0 * tube[step, 1]
        if np.isfinite(width) and np.isfinite(height):
            axis.add_patch(
                Rectangle(
                    nominal[step, :2] - tube[step, :2], width, height,
                    facecolor="tab:orange", edgecolor="none", alpha=0.13,
                    zorder=1,
                )
            )
    axis.plot(nominal[:horizon, 0], nominal[:horizon, 1], "--", color="tab:orange", linewidth=2.2, label="nominal plan", zorder=3)
    for index, rollout in enumerate(rollouts):
        finite = np.all(np.isfinite(rollout[:horizon, :2]), axis=1)
        axis.plot(
            rollout[:horizon, 0][finite], rollout[:horizon, 1][finite],
            color="tab:blue", alpha=0.35, linewidth=1.3,
            label="saved rollout" if index == 0 else None, zorder=2,
        )
    for index, (x, y, radius) in enumerate(obstacles):
        axis.add_patch(Circle((x, y), radius, color="tab:red", alpha=0.35, label="obstacle" if index == 0 else None, zorder=4))
    axis.scatter(nominal[0, 0], nominal[0, 1], color="tab:green", marker="o", s=40, label="start", zorder=5)
    axis.scatter(goal[0], goal[1], color="black", marker="*", s=100, label="goal", zorder=5)
    axis.plot([], [], color="tab:orange", linewidth=7, alpha=0.13, label="tube")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title(f"{case_dir.parent.name.title()}: {case_dir.name.replace('_', ' ')}")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path(__file__).with_name("results") / "unmatched" / "adaptive_true__leb_true",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.case_dir / "rollout_with_tubes_all_rollouts.png"
    plot(args.case_dir, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
