"""Plot all sampled car rollouts together with the three nominal trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # Use installed LaTeX PSNFSS families. Matplotlib's default auxiliary
        # Computer Modern font lists require type1ec.sty, which is absent on
        # this machine even though the LaTeX executable itself is available.
        "font.serif": ["Times"],
        "font.sans-serif": ["Helvetica"],
        "font.monospace": ["Courier"],
        "figure.dpi": 300,
        "font.size": 6.6,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.6,
        "xtick.labelsize": 5.76,
        "ytick.labelsize": 5.76,
        "legend.fontsize": 5.4,
        "lines.linewidth": 0.8,
    }
)


OBSTACLES = (
    (-0.25, 0.20, 0.23),
    (0.25, -0.25, 0.23),
)
START = (0.0, 1.0)
GOAL = (0.25, -1.0)

# Dark nominal and light rollout shades within the requested method colors.
STYLES = {
    "adaptive_leb": {
        "label": "Adaptive + LEB",
        "nominal": "#174A7E",
        "rollout": "#72A9D5",
    },
    "ccm": {
        "label": "CCM",
        "nominal": "#B45309",
        "rollout": "#F2AE72",
    },
    "pavone": {
        "label": "Pavone",
        "nominal": "#176B38",
        "rollout": "#79BE8B",
    },
}
PLOT_ORDER = ("adaptive_leb", "ccm", "pavone")
CONTROLLER_FILES = {
    "adaptive_leb": "adaptive_leb_controller.npz",
    "ccm": "ccm_controller.npz",
    "pavone": "pavone_controller.npz",
}


def load_rollouts(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"method_names", "states", "is_adversarial", "dt"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        archive_method_names = [str(name) for name in archive["method_names"]]
        states = np.asarray(archive["states"], dtype=float)
        is_adversarial = np.asarray(archive["is_adversarial"], dtype=bool)
        dt = float(archive["dt"])
    if states.ndim != 4 or states.shape[0] != len(archive_method_names):
        raise ValueError(
            "states must have shape (method, rollout, time, physical_state)"
        )
    if states.shape[1] != len(is_adversarial) or states.shape[-1] < 2:
        raise ValueError("rollout metadata does not match the state array")
    unsupported = set(archive_method_names).difference(STYLES)
    if unsupported:
        raise ValueError(f"no plotting style configured for methods: {unsupported}")
    method_names = [method for method in PLOT_ORDER if method in archive_method_names]
    method_indices = [archive_method_names.index(method) for method in method_names]
    states = states[method_indices]
    return method_names, states, is_adversarial, dt


def load_controller_geometry(
    controller_dir: Path, method: str
) -> tuple[np.ndarray, np.ndarray]:
    path = controller_dir / CONTROLLER_FILES[method]
    with np.load(path, allow_pickle=False) as controller:
        if "nominal_states" not in controller.files:
            raise ValueError(f"{path} does not contain nominal_states")
        nominal = np.asarray(controller["nominal_states"], dtype=float)
        if method == "adaptive_leb":
            tube_widths = np.asarray(
                controller["state_tube_half_widths"][:, :2], dtype=float
            )
        elif method == "ccm":
            times = np.arange(len(nominal), dtype=float) * float(controller["dt"])
            contraction_rate = float(controller["contraction_rate"])
            radii = (
                float(controller["disturbance_bound"])
                * (1.0 - np.exp(-contraction_rate * times))
                / contraction_rate
            )
            tube_widths = radii * float(controller["position_support"])
        else:
            response = np.asarray(controller["state_response"], dtype=float)
            disturbance_width = np.asarray(
                controller["disturbance_half_width"], dtype=float
            )
            transition_widths = np.einsum(
                "tijk,k->ti", np.abs(response), disturbance_width
            )
            tube_widths = np.vstack(
                [np.zeros((1, transition_widths.shape[1])), transition_widths]
            )[:, :2]
    if nominal.ndim != 2 or nominal.shape[1] < 2:
        raise ValueError(f"invalid nominal state array in {path}: {nominal.shape}")
    if len(tube_widths) != len(nominal) or not np.all(np.isfinite(tube_widths)):
        raise ValueError(f"invalid tube array in {path}: {tube_widths.shape}")
    return nominal, tube_widths


def finite_xy_segments(states: np.ndarray) -> list[np.ndarray]:
    """Return contiguous finite prefixes so one bad sample cannot spoil a line."""
    segments = []
    for rollout in states:
        finite = np.all(np.isfinite(rollout[:, :2]), axis=1)
        first_bad = int(np.flatnonzero(~finite)[0]) if np.any(~finite) else len(rollout)
        if first_bad >= 2:
            segments.append(rollout[:first_bad, :2])
    return segments


def add_tube_patches(
    axis: plt.Axes,
    method: str,
    nominal: np.ndarray,
    tube_widths: np.ndarray,
    *,
    stride: int,
    alpha: float,
) -> None:
    color = STYLES[method]["rollout"]
    for step in range(0, len(nominal), stride):
        center = nominal[step, :2]
        if method == "ccm":
            patch = Circle(center, float(tube_widths[step]))
        else:
            half_width = tube_widths[step]
            patch = Rectangle(
                center - half_width,
                2.0 * half_width[0],
                2.0 * half_width[1],
            )
        patch.set(
            facecolor=color,
            edgecolor=color,
            linewidth=0.25,
            alpha=alpha,
            zorder=1,
        )
        axis.add_patch(patch)


def position_tube_width(tube_widths: np.ndarray, coordinate: int) -> np.ndarray:
    return tube_widths if tube_widths.ndim == 1 else tube_widths[:, coordinate]


def add_deviation_panel(
    axis: plt.Axes,
    coordinate: int,
    method_names: list[str],
    states: np.ndarray,
    controller_geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    rollout_order: np.ndarray,
    *,
    rollout_linewidth: float,
    rollout_alpha: float,
    tube_linewidth: float,
) -> None:
    coordinate_name = "x" if coordinate == 0 else "y"
    horizon = np.arange(states.shape[2])
    for method_index, method in enumerate(method_names):
        nominal, tube_widths = controller_geometry[method]
        style = STYLES[method]
        deviations = np.abs(
            states[method_index, rollout_order, :, coordinate]
            - nominal[None, :, coordinate]
        )
        for deviation in deviations:
            finite = np.isfinite(deviation)
            axis.plot(
                horizon[finite],
                deviation[finite],
                color=style["rollout"],
                linewidth=max(0.25, rollout_linewidth - 0.15),
                alpha=max(0.28, rollout_alpha - 0.16),
                zorder=2,
            )
        axis.plot(
            horizon,
            position_tube_width(tube_widths, coordinate),
            color=style["nominal"],
            linewidth=tube_linewidth,
            label=rf'{style["label"]} tube',
            zorder=4,
        )
    axis.set_ylabel(rf"$|\Delta p_{coordinate_name}|$ (m)", labelpad=1.5)
    axis.grid(True, color="0.65", alpha=0.65, linewidth=0.4)
    axis.set_xlim(horizon[0], horizon[-1])
    axis.locator_params(axis="x", nbins=4)
    axis.locator_params(axis="y", nbins=4)
    axis.tick_params(length=2.0, width=0.45, pad=1.5)


def style_compact_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.025,
        0.975,
        rf"\textbf{{({label})}}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        zorder=10,
    )


def plot_rollouts(
    rollout_path: Path,
    controller_dir: Path,
    output: Path,
    *,
    rollout_linewidth: float = 0.45,
    rollout_alpha: float = 0.58,
    nominal_linewidth: float = 1.4,
    tube_stride: int = 2,
    tube_alpha: float = 0.065,
    figure_width: float = 3.0,
    figure_height: float = 2.0,
    show_panel_labels: bool = True,
) -> Path:
    method_names, states, is_adversarial, _dt = load_rollouts(rollout_path)
    controller_geometry = {
        method: load_controller_geometry(controller_dir, method)
        for method in method_names
    }
    nominals = {method: values[0] for method, values in controller_geometry.items()}

    figure = plt.figure(
        figsize=(figure_width, figure_height), layout="constrained"
    )
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.65, 0.85),
        height_ratios=(1.0, 1.0),
        wspace=0.06,
        hspace=0.05,
    )
    axis = figure.add_subplot(grid[:, 0])
    px_axis = figure.add_subplot(grid[0, 1])
    py_axis = figure.add_subplot(grid[1, 1], sharex=px_axis)
    all_xy: list[np.ndarray] = []
    for method in method_names:
        nominal, tube_widths = controller_geometry[method]
        add_tube_patches(
            axis,
            method,
            nominal,
            tube_widths,
            stride=tube_stride,
            alpha=tube_alpha,
        )
        if tube_widths.ndim == 1:
            extent_widths = np.column_stack([tube_widths, tube_widths])
        else:
            extent_widths = tube_widths
        all_xy.extend(
            [nominal[:, :2] - extent_widths, nominal[:, :2] + extent_widths]
        )

    # Random trajectories go down first so the boundary stress tests remain
    # visible, but all sampled trajectories deliberately share one light style.
    rollout_order = np.concatenate(
        [np.flatnonzero(~is_adversarial), np.flatnonzero(is_adversarial)]
    )
    for method_index, method in enumerate(method_names):
        style = STYLES[method]
        ordered_states = states[method_index, rollout_order]
        for xy in finite_xy_segments(ordered_states):
            axis.plot(
                xy[:, 0],
                xy[:, 1],
                color=style["rollout"],
                linewidth=rollout_linewidth,
                alpha=rollout_alpha,
                solid_capstyle="round",
                zorder=2,
            )
            all_xy.append(xy)

    for method in method_names:
        style = STYLES[method]
        nominal_xy = nominals[method][:, :2]
        axis.plot(
            nominal_xy[:, 0],
            nominal_xy[:, 1],
            color=style["nominal"],
            linestyle="--",
            linewidth=nominal_linewidth,
            dash_capstyle="round",
            zorder=5,
        )
        all_xy.append(nominal_xy)

    for obstacle_index, (center_x, center_y, radius) in enumerate(OBSTACLES):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                facecolor="tab:red",
                edgecolor="black",
                linewidth=0.45,
                alpha=0.35,
                zorder=4,
                label="Obstacle" if obstacle_index == 0 else None,
            )
        )
    axis.scatter(
        [START[0]], [START[1]], color="black", s=16, zorder=7, label="Start"
    )
    axis.scatter(
        [GOAL[0]],
        [GOAL[1]],
        color="black",
        marker="*",
        s=48,
        zorder=7,
        label="Goal",
    )

    obstacle_extents = np.asarray(
        [
            point
            for center_x, center_y, radius in OBSTACLES
            for point in (
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            )
        ]
    )
    plot_extents = np.vstack([*all_xy, obstacle_extents, np.asarray([START, GOAL])])
    horizontal_margin = 0.18
    lower_vertical_margin = 0.25
    upper_vertical_margin = 0.08
    axis.set_xlim(
        float(np.nanmin(plot_extents[:, 0]) - horizontal_margin),
        float(np.nanmax(plot_extents[:, 0]) + horizontal_margin),
    )
    axis.set_ylim(
        float(np.nanmin(plot_extents[:, 1]) - lower_vertical_margin),
        float(np.nanmax(plot_extents[:, 1]) + upper_vertical_margin),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$p_x$ (m)")
    axis.set_ylabel(r"$p_y$ (m)")
    axis.grid(True, color="0.65", alpha=0.65, linewidth=0.4)
    axis.locator_params(axis="x", nbins=5)
    axis.locator_params(axis="y", nbins=5)
    axis.tick_params(length=2.0, width=0.45, pad=1.5)

    add_deviation_panel(
        px_axis,
        0,
        method_names,
        states,
        controller_geometry,
        rollout_order,
        rollout_linewidth=rollout_linewidth,
        rollout_alpha=rollout_alpha,
        tube_linewidth=nominal_linewidth,
    )
    add_deviation_panel(
        py_axis,
        1,
        method_names,
        states,
        controller_geometry,
        rollout_order,
        rollout_linewidth=rollout_linewidth,
        rollout_alpha=rollout_alpha,
        tube_linewidth=nominal_linewidth,
    )
    px_axis.tick_params(labelbottom=False)
    py_axis.set_xlabel(r"Horizon", labelpad=1.5)
    for subplot_axis in (axis, px_axis, py_axis):
        style_compact_axis(subplot_axis)
    if show_panel_labels:
        add_panel_label(axis, "a")
        add_panel_label(px_axis, "b")
        add_panel_label(py_axis, "c")

    method_handles = []
    for method in method_names:
        style = STYLES[method]
        method_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=style["nominal"],
                    linestyle="--",
                    linewidth=nominal_linewidth,
                    label=f'{style["label"]} nominal',
                ),
                Line2D(
                    [0],
                    [0],
                    color=style["rollout"],
                    linewidth=rollout_linewidth + 0.35,
                    alpha=rollout_alpha,
                    label=f'{style["label"]} rollouts',
                ),
            ]
        )
    # At journal-column scale, one color-coded entry per method remains
    # legible; line weight/shade semantics are visible directly in the panel.
    compact_method_handles = [method_handles[index] for index in range(0, 6, 2)]
    compact_labels = {
        "adaptive_leb": "A-SLS",
        "ccm": "CCM",
        "pavone": "DF",
    }
    for handle, method in zip(compact_method_handles, method_names):
        handle.set_label(compact_labels[method])
    tube_handles = [
        Patch(
            facecolor=STYLES[method]["rollout"],
            edgecolor=STYLES[method]["rollout"],
            alpha=max(tube_alpha, 0.24),
            label=f'{compact_labels[method]} tube',
        )
        for method in method_names
    ]
    figure.legend(
        handles=[*compact_method_handles, *tube_handles],
        loc="center",
        # Each +0.035 vertical figure offset is approximately +0.1 m in the
        # planar panel's current p_y scale; x is nudged right by 0.02.
        bbox_to_anchor=(0.54, 0.57),
        borderaxespad=0.0,
        framealpha=0.88,
        borderpad=0.18,
        handlelength=1.25,
        labelspacing=0.2,
        ncol=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Plotted {states.shape[1]} rollouts for each of {len(method_names)} "
        f"methods to {output}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=script_dir / "controller_rollouts.npz",
    )
    parser.add_argument(
        "--controller-dir",
        type=Path,
        default=script_dir / "saved_controllers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "all_controller_rollouts.png",
    )
    parser.add_argument("--rollout-linewidth", type=float, default=0.45)
    parser.add_argument("--rollout-alpha", type=float, default=0.58)
    parser.add_argument("--nominal-linewidth", type=float, default=1.4)
    parser.add_argument("--tube-stride", type=int, default=2)
    parser.add_argument("--tube-alpha", type=float, default=0.065)
    parser.add_argument("--figure-width", type=float, default=3.0)
    parser.add_argument("--figure-height", type=float, default=2.0)
    parser.add_argument(
        "--no-panel-labels",
        action="store_true",
        help="omit the (a), (b), and (c) labels from the subplot corners",
    )
    args = parser.parse_args()
    if not 0.0 < args.rollout_alpha <= 1.0:
        parser.error("--rollout-alpha must lie in (0, 1]")
    if not 0.0 < args.tube_alpha <= 1.0:
        parser.error("--tube-alpha must lie in (0, 1]")
    if args.tube_stride <= 0:
        parser.error("--tube-stride must be positive")
    if args.rollout_linewidth <= 0.0 or args.nominal_linewidth <= 0.0:
        parser.error("line widths must be positive")
    if args.figure_width <= 0.0 or args.figure_height <= 0.0:
        parser.error("figure dimensions must be positive")
    plot_rollouts(
        args.rollouts,
        args.controller_dir,
        args.output,
        rollout_linewidth=args.rollout_linewidth,
        rollout_alpha=args.rollout_alpha,
        nominal_linewidth=args.nominal_linewidth,
        tube_stride=args.tube_stride,
        tube_alpha=args.tube_alpha,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_panel_labels=not args.no_panel_labels,
    )


if __name__ == "__main__":
    main()
