"""Overlay CCM, Pavone affine-DF RMPC, and adaptive+LEB results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np


OBSTACLES = (
    (-0.25, 0.20, 0.23),
    (0.25, -0.25, 0.23),
)
GOAL = (0.25, -1.0)


def read_columns(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        values = np.asarray(
            [[float(row[column]) for column in columns] for row in reader], dtype=float
        )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains empty or nonfinite comparison data")
    return values


def add_ccm_tubes(
    axis: plt.Axes, nominal_xy: np.ndarray, radii: np.ndarray, stride: int
) -> None:
    for step in range(0, min(len(nominal_xy), len(radii)), stride):
        axis.add_patch(
            Circle(
                nominal_xy[step],
                radii[step],
                facecolor="tab:orange",
                edgecolor="tab:orange",
                linewidth=0.8,
                alpha=0.12,
            )
        )


def add_adaptive_tubes(
    axis: plt.Axes,
    nominal_xy: np.ndarray,
    half_widths: np.ndarray,
    stride: int,
    color: str = "tab:blue",
) -> None:
    for step in range(0, min(len(nominal_xy), len(half_widths)), stride):
        center = nominal_xy[step]
        widths = half_widths[step]
        axis.add_patch(
            Rectangle(
                center - widths,
                2.0 * widths[0],
                2.0 * widths[1],
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.11,
            )
        )


def plot_comparison(
    results_dir: Path,
    output: Path,
    tube_stride: int = 5,
    pavone_dir: Path | None = None,
) -> Path:
    ccm_dir = results_dir / "certified_ccm"
    adaptive_dir = results_dir / "adaptive_true__leb_true"
    ccm_nominal = read_columns(ccm_dir / "nominal_plan.csv", ("x_m", "y_m"))
    ccm_rollout = read_columns(ccm_dir / "rollout.csv", ("x_m", "y_m"))
    ccm_radius = read_columns(
        ccm_dir / "tube_widths.csv", ("position_radial_bound_m",)
    )[:, 0]
    adaptive_nominal = read_columns(
        adaptive_dir / "nominal_plan.csv", ("x_m", "y_m")
    )
    adaptive_rollout = read_columns(adaptive_dir / "rollout.csv", ("x_m", "y_m"))
    adaptive_half_widths = read_columns(
        adaptive_dir / "tube_widths.csv",
        ("x_half_width_m", "y_half_width_m"),
    )
    pavone_dir = pavone_dir or results_dir / "pavone_affine_df"
    have_pavone = pavone_dir.is_dir()
    if have_pavone:
        pavone_nominal = read_columns(pavone_dir / "nominal_plan.csv", ("x_m", "y_m"))
        pavone_rollout = read_columns(pavone_dir / "rollout.csv", ("x_m", "y_m"))
        pavone_half_widths = read_columns(
            pavone_dir / "tube_widths.csv", ("x_half_width_m", "y_half_width_m")
        )

    figure, axis = plt.subplots(figsize=(9, 9))
    add_ccm_tubes(axis, ccm_nominal, ccm_radius, tube_stride)
    add_adaptive_tubes(axis, adaptive_nominal, adaptive_half_widths, tube_stride)
    if have_pavone:
        add_adaptive_tubes(
            axis,
            pavone_nominal,
            pavone_half_widths,
            tube_stride,
            color="tab:green",
        )
    axis.plot(
        ccm_nominal[:, 0], ccm_nominal[:, 1], "--", color="tab:orange", lw=2.0,
        label="CCM nominal",
    )
    axis.plot(
        ccm_rollout[:, 0], ccm_rollout[:, 1], color="tab:orange", lw=2.5,
        label="CCM rollout",
    )
    axis.plot(
        adaptive_nominal[:, 0], adaptive_nominal[:, 1], "--", color="tab:blue", lw=2.0,
        label="Adaptive + LEB nominal",
    )
    axis.plot(
        adaptive_rollout[:, 0], adaptive_rollout[:, 1], color="tab:blue", lw=2.5,
        label="Adaptive + LEB rollout",
    )
    if have_pavone:
        axis.plot(
            pavone_nominal[:, 0],
            pavone_nominal[:, 1],
            "--",
            color="tab:green",
            lw=2.0,
            label="Pavone affine-DF nominal",
        )
        axis.plot(
            pavone_rollout[:, 0],
            pavone_rollout[:, 1],
            color="tab:green",
            lw=2.5,
            label="Pavone affine-DF rollout (no LEB)",
        )

    for index, (center_x, center_y, radius) in enumerate(OBSTACLES):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                facecolor="tab:red",
                edgecolor="black",
                alpha=0.35,
                label="obstacle" if index == 0 else None,
            )
        )
    axis.scatter([0.0], [1.0], color="black", s=55, zorder=8, label="start")
    axis.scatter(
        [GOAL[0]], [GOAL[1]], color="black", marker="*", s=180, zorder=8,
        label="goal",
    )

    all_xy = np.vstack((ccm_nominal, ccm_rollout, adaptive_nominal, adaptive_rollout))
    if have_pavone:
        all_xy = np.vstack((all_xy, pavone_nominal, pavone_rollout))
    obstacle_extents = np.asarray(
        [
            point
            for center_x, center_y, radius in OBSTACLES
            for point in ((center_x - radius, center_y - radius), (center_x + radius, center_y + radius))
        ]
    )
    ccm_extents = np.vstack(
        (ccm_nominal - ccm_radius[:, None], ccm_nominal + ccm_radius[:, None])
    )
    adaptive_extents = np.vstack(
        (adaptive_nominal - adaptive_half_widths, adaptive_nominal + adaptive_half_widths)
    )
    all_xy = np.vstack(
        (all_xy, ccm_extents, adaptive_extents, obstacle_extents, np.asarray([GOAL]))
    )
    if have_pavone:
        pavone_extents = np.vstack(
            (pavone_nominal - pavone_half_widths, pavone_nominal + pavone_half_widths)
        )
        all_xy = np.vstack((all_xy, pavone_extents))
    margin = 0.25
    axis.set_xlim(np.min(all_xy[:, 0]) - margin, np.max(all_xy[:, 0]) + margin)
    axis.set_ylim(np.min(all_xy[:, 1]) - margin, np.max(all_xy[:, 1]) + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title(
        "CCM vs Pavone Affine-DF RMPC vs Adaptive + LEB"
        if have_pavone
        else "Certified CCM vs Adaptive + LEB"
    )
    axis.grid(True, alpha=0.3)

    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Patch(
                facecolor="tab:orange", edgecolor="tab:orange", alpha=0.12,
                label="CCM circular tubes",
            ),
            Patch(
                facecolor="tab:blue", edgecolor="tab:blue", alpha=0.11,
                label="Adaptive + LEB rectangular tubes",
            ),
        ]
    )
    labels.extend(("CCM circular tubes", "Adaptive + LEB rectangular tubes"))
    if have_pavone:
        handles.append(
            Patch(
                facecolor="tab:green",
                edgecolor="tab:green",
                alpha=0.11,
                label="Pavone LTV tubes (no LEB)",
            )
        )
        labels.append("Pavone LTV tubes (no LEB)")
    axis.legend(handles, labels, loc="best", fontsize=9)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Combined comparison plot written to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pavone-dir",
        type=Path,
        help="Pavone run directory when it is outside --results-dir",
    )
    parser.add_argument("--tube-stride", type=int, default=5)
    args = parser.parse_args()
    if args.tube_stride <= 0:
        parser.error("--tube-stride must be positive")
    plot_comparison(
        args.results_dir,
        args.output or args.results_dir / "ccm_vs_adaptive_leb.png",
        args.tube_stride,
        args.pavone_dir,
    )
