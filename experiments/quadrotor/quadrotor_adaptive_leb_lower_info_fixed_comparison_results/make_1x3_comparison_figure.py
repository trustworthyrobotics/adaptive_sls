"""Create a compact 1x3 publication figure for the fixed comparison."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

QUADROTOR_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = QUADROTOR_DIR.parents[1] / "src"
sys.path.insert(0, str(QUADROTOR_DIR))
sys.path.insert(0, str(SRC_DIR))

from quadrotor_common import (  # noqa: E402
    DISTURBANCE_MAGNITUDE,
    DT,
    N_PHYS,
    OBSTACLES,
    X0_PHYS,
    X_GOAL_PHYS,
)


ROOT = Path(__file__).resolve().parent
PDF_OUTPUT = ROOT / "quadrotor_adaptive_sls_1x3_comparison.pdf"
PNG_OUTPUT = ROOT / "quadrotor_adaptive_sls_1x3_comparison.png"

STYLES = {
    "without": {
        "nominal": "#174A7E",
        "rollout": "#72A9D5",
        "tube_alpha": 0.05,
        "label": "w/o active information",
    },
    "with": {
        "nominal": "#6A3D8F",
        "rollout": "#B995D1",
        "tube_alpha": 0.16,
        "label": r"w/ active information ($E^\dagger F$)",
    },
}

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.sans-serif": ["Helvetica"],
        "font.monospace": ["Courier"],
        "figure.dpi": 300,
        "font.size": 7.8,
        "axes.titlesize": 9.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.75,
        "lines.linewidth": 0.65,
    }
)


def load_case(directory: str) -> dict[str, np.ndarray | float]:
    path = ROOT / directory / "quadrotor_adaptive_sls_rollout.npz"
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def spatially_sampled_indices(path: np.ndarray, max_patches: int = 16) -> np.ndarray:
    """Select tube patches at approximately uniform XY arc-length intervals."""
    segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cumulative_distance = np.concatenate(
        [np.zeros(1), np.cumsum(segment_lengths)]
    )
    if cumulative_distance[-1] <= 0.0:
        return np.linspace(0, path.shape[0] - 1, max_patches, dtype=int)
    targets = np.linspace(0.0, cumulative_distance[-1], max_patches)
    return np.unique(
        np.clip(np.searchsorted(cumulative_distance, targets), 0, path.shape[0] - 1)
    )


def add_xy_panel(axis: plt.Axes, cases: dict[str, dict]) -> None:
    for name, case in cases.items():
        style = STYLES[name]
        states = case["states"]
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        steps = prediction.shape[0]

        for rollout in states:
            axis.plot(
                rollout[:, 0], rollout[:, 1], color=style["rollout"],
                alpha=0.30, linewidth=0.55, zorder=2,
            )
        for step in spatially_sampled_indices(prediction[:steps]):
            center = prediction[step, :2]
            half_width = tubes[step, :2]
            axis.add_patch(
                plt.Rectangle(
                    center - half_width,
                    2.0 * half_width[0],
                    2.0 * half_width[1],
                    facecolor=style["nominal"],
                    edgecolor=style["nominal"],
                    linewidth=0.45,
                    alpha=style["tube_alpha"],
                    zorder=3,
                )
            )
        axis.plot(
            prediction[:steps, 0], prediction[:steps, 1],
            color=style["nominal"], linestyle="--", linewidth=1.15, zorder=4,
        )

    axis.scatter([X0_PHYS[0]], [X0_PHYS[1]], color="black", s=8, zorder=6)
    axis.scatter([X_GOAL_PHYS[0]], [X_GOAL_PHYS[1]], color="black", marker="*", s=35, zorder=6)
    for obstacle in OBSTACLES:
        axis.add_patch(
            Circle(
                (float(obstacle[0]), float(obstacle[1])), float(obstacle[2]),
                facecolor="tab:red", edgecolor="tab:red", alpha=0.22, zorder=1,
            )
        )
    axis.set_xlabel(r"$p_x$ (m)")
    axis.set_ylabel(r"$p_y$ (m)")
    axis.axis("equal")
    axis.grid(True, alpha=0.45, linewidth=0.35)
    axis.text(0.97, 0.97, r"\textbf{(a)}", transform=axis.transAxes, ha="right", va="top", fontsize=13.5, zorder=10)


def add_z_panel(
    axis: plt.Axes,
    cases: dict[str, dict],
    colorbar_axis: plt.Axes,
) -> None:
    z_values = []
    for case in cases.values():
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        z_values.extend(
            [
                case["states"][..., 2].reshape(-1),
                prediction[:, 2],
                prediction[:, 2] - tubes[:, 2],
                prediction[:, 2] + tubes[:, 2],
            ]
        )
    finite_z = np.concatenate(z_values)
    finite_z = finite_z[np.isfinite(finite_z)]
    z_low, z_high = float(finite_z.min()), float(finite_z.max())
    margin = max(0.015, 0.05 * (z_high - z_low))
    z_low, z_high = z_low - margin, z_high + margin

    first_case = next(iter(cases.values()))
    z_off = float(first_case["disturbance_z_off"])
    sharpness = float(first_case["disturbance_z_sharpness"])
    z_edges = np.linspace(z_low, z_high, 256)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    gate = 0.5 * (1.0 - np.tanh(sharpness * (z_centers - z_off)))
    map_norm = DISTURBANCE_MAGNITUDE * DT * gate * np.sqrt(N_PHYS)
    time_high = max(float(case["prediction"].shape[0] - 1) * DT for case in cases.values())
    image = axis.pcolormesh(
        [0.0, time_high], z_edges, map_norm[:, None], cmap="magma",
        alpha=0.25, shading="auto", zorder=0,
    )
    colorbar = axis.figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.ax.text(
        1.04,
        0.5,
        r"$\|F(x)\|_F$",
        transform=colorbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.6,
    )
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(labelsize=6.0, pad=1)

    for name, case in cases.items():
        style = STYLES[name]
        states = case["states"]
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        time = np.arange(states.shape[1]) * DT
        plan_time = np.arange(prediction.shape[0]) * DT
        for rollout in states:
            axis.plot(time, rollout[:, 2], color=style["rollout"], alpha=0.30, linewidth=0.55, zorder=2)
        axis.fill_between(
            plan_time,
            prediction[:, 2] - tubes[:, 2],
            prediction[:, 2] + tubes[:, 2],
            color=style["nominal"],
            alpha=style["tube_alpha"] + 0.06,
            zorder=3,
        )
        axis.plot(plan_time, prediction[:, 2], color=style["nominal"], linestyle="--", linewidth=1.15, zorder=4)

    axis.axhline(float(X0_PHYS[2]), color="black", alpha=0.35, linewidth=0.65)
    axis.axhline(float(first_case["information_center_z"]), color="tab:green", linestyle=":", linewidth=0.8)
    axis.set_xlabel(r"time (s)")
    axis.set_ylabel(r"$p_z$ (m)")
    axis.grid(True, alpha=0.45, linewidth=0.35)
    axis.text(0.97, 0.97, r"\textbf{(b)}", transform=axis.transAxes, ha="right", va="top", fontsize=13.5, zorder=10)


def add_parameter_panel(axis: plt.Axes, cases: dict[str, dict]) -> None:
    for name, case in cases.items():
        style = STYLES[name]
        estimates = case["force_estimates"]
        true_forces = case["true_forces"]
        half_widths = case["planned_force_widths"][:, 0]
        errors = np.abs(estimates[:, :, 0] - true_forces[:, None, 0])
        time = np.arange(estimates.shape[1]) * DT
        for error in errors:
            axis.plot(time, error, color=style["rollout"], alpha=0.28, linewidth=0.5, zorder=2)
        axis.fill_between(
            time,
            np.zeros_like(half_widths),
            half_widths,
            color=style["nominal"],
            alpha=style["tube_alpha"],
            zorder=1,
        )
        axis.plot(
            time,
            half_widths,
            color=style["nominal"],
            linewidth=1.25,
            zorder=3,
        )
    axis.set_xlabel(r"time (s)")
    axis.set_ylabel(r"$|F_x|$ (N)")
    axis.grid(True, alpha=0.45, linewidth=0.35)
    axis.text(0.97, 0.97, r"\textbf{(c)}", transform=axis.transAxes, ha="right", va="top", fontsize=13.5, zorder=10)


def main() -> None:
    cases = {
        "without": load_case("adaptive__leb_true__edagger_f_false"),
        "with": load_case("adaptive__leb_true__edagger_f_true"),
    }
    figure = plt.figure(figsize=(4.4, 1.65))
    # Keep the z(t) panel shorter and centered, leaving room for its
    # horizontal disturbance-map colorbar above it.
    axes = [
        figure.add_axes([0.055, 0.24, 0.255, 0.58]),
        figure.add_axes([0.405, 0.24, 0.205, 0.50]),
        figure.add_axes([0.735, 0.24, 0.205, 0.58]),
    ]
    colorbar_axis = figure.add_axes([0.405, 0.765, 0.155, 0.035])
    add_xy_panel(axes[0], cases)
    add_z_panel(axes[1], cases, colorbar_axis)
    add_parameter_panel(axes[2], cases)

    legend_handles = [
        Line2D([0], [0], color=STYLES["without"]["nominal"], linestyle="--", label="A-SLS"),
        Line2D([0], [0], color=STYLES["without"]["rollout"], label="A-SLS rollouts"),
        Patch(facecolor=STYLES["without"]["nominal"], alpha=STYLES["without"]["tube_alpha"], label="A-SLS tube"),
        Line2D([0], [0], color=STYLES["with"]["nominal"], linestyle="--", label="A-SLS+"),
        Line2D([0], [0], color=STYLES["with"]["rollout"], label="A-SLS+ rollouts"),
        Patch(facecolor=STYLES["with"]["nominal"], alpha=STYLES["with"]["tube_alpha"], label="A-SLS+ tube"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.46, -0.12),
        ncol=6,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    figure.savefig(PDF_OUTPUT, dpi=300, bbox_inches="tight")
    figure.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {PDF_OUTPUT}")
    print(f"Saved {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
