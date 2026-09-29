"""Plot CCM and Adaptive + LEB tube half-widths on shared axes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CCM_CSV = (
    SCRIPT_DIR / "ccm_car_results/trajectory_local_ccm_tvlqr/tube_widths.csv"
)
DEFAULT_ADAPTIVE_LEB_CSV = (
    SCRIPT_DIR
    / "adaptive_nonadaptive_car_results/adaptive_true__leb_true/tube_widths.csv"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "adaptive_nonadaptive_car_results/ccm_vs_adaptive_leb.png"
)
DEFAULT_CCM_DIR = SCRIPT_DIR / "ccm_car_results/trajectory_local_ccm_tvlqr"
DEFAULT_ADAPTIVE_LEB_DIR = (
    SCRIPT_DIR / "adaptive_nonadaptive_car_results/adaptive_true__leb_true"
)
DEFAULT_ROLLOUT_OUTPUT = (
    SCRIPT_DIR
    / "adaptive_nonadaptive_car_results/ccm_vs_adaptive_leb_rollout_with_tubes.png"
)

OBSTACLES = (
    (-0.25, 0.20, 0.23),
    (0.25, -0.35, 0.23),
    (-0.26, -0.90, 0.23),
    (0.14, -1.45, 0.23),
)

QUANTITIES = (
    ("x_half_width_m", "X position", "m"),
    ("y_half_width_m", "Y position", "m"),
    ("heading_half_width_rad", "Heading", "rad"),
    ("speed_half_width_m_per_s", "Speed", "m/s"),
)


def read_tube_widths(path: Path) -> dict[str, list[float]]:
    """Read the time and tube-width columns used by the comparison figure."""
    required_columns = {"time_s", *(column for column, _, _ in QUANTITIES)}
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {missing_list}")

        values = {column: [] for column in required_columns}
        for row in reader:
            for column in required_columns:
                values[column].append(float(row[column]))

    if not values["time_s"]:
        raise ValueError(f"{path} contains no tube-width samples")
    return values


def read_xy(path: Path) -> dict[str, list[float]]:
    """Read finite x-y positions from a nominal-plan or rollout CSV."""
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {"x_m", "y_m"}.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {missing_list}")
        x_values, y_values = [], []
        for row in reader:
            x_value, y_value = float(row["x_m"]), float(row["y_m"])
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)

    if not x_values:
        raise ValueError(f"{path} contains no finite x-y samples")
    return {"x_m": x_values, "y_m": y_values}


def plot_comparison(
    ccm_csv: Path,
    adaptive_leb_csv: Path,
    output: Path,
) -> None:
    """Save a four-panel shared-axis comparison of the two methods."""
    ccm = read_tube_widths(ccm_csv)
    adaptive_leb = read_tube_widths(adaptive_leb_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis, (column, title, unit) in zip(axes.flat, QUANTITIES):
        axis.plot(
            ccm["time_s"],
            ccm[column],
            color="tab:orange",
            linestyle="--",
            linewidth=2.5,
            label="CCM",
        )
        axis.plot(
            adaptive_leb["time_s"],
            adaptive_leb[column],
            color="tab:blue",
            linewidth=2.5,
            label="Adaptive + LEB",
        )
        axis.set_title(title)
        axis.set_ylabel(f"tube half-width ({unit})")
        axis.grid(True, alpha=0.35)

    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Tube Half-Width Comparison", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CCM vs Adaptive + LEB comparison to {output}")


def plot_rollouts_with_tubes(
    ccm_dir: Path,
    adaptive_leb_dir: Path,
    output: Path,
) -> None:
    """Overlay both nominal paths, closed-loop rollouts, and position tubes."""
    methods = (
        (
            "CCM",
            "tab:orange",
            ccm_dir,
            read_tube_widths(ccm_dir / "tube_widths.csv"),
            "ellipse",
        ),
        (
            "Adaptive + LEB",
            "tab:blue",
            adaptive_leb_dir,
            read_tube_widths(adaptive_leb_dir / "tube_widths.csv"),
            "rectangle",
        ),
    )

    figure, axis = plt.subplots(figsize=(9, 9))
    all_x, all_y = [], []
    for label, color, run_dir, tubes, tube_shape in methods:
        nominal = read_xy(run_dir / "nominal_plan.csv")
        rollout = read_xy(run_dir / "rollout.csv")
        axis.plot(
            nominal["x_m"],
            nominal["y_m"],
            color=color,
            linestyle="--",
            linewidth=2.2,
            label=f"{label} nominal",
        )
        axis.plot(
            rollout["x_m"],
            rollout["y_m"],
            color=color,
            linewidth=2.5,
            label=f"{label} rollout",
        )

        sample_count = min(
            len(nominal["x_m"]),
            len(tubes["x_half_width_m"]),
            len(tubes["y_half_width_m"]),
        )
        stride = max(1, (sample_count - 1) // 20)
        for step in range(0, sample_count, stride):
            x_width = tubes["x_half_width_m"][step]
            y_width = tubes["y_half_width_m"][step]
            center_x = nominal["x_m"][step]
            center_y = nominal["y_m"][step]
            patch_options = {
                "facecolor": color,
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.10,
            }
            if tube_shape == "ellipse":
                tube_patch = Ellipse(
                    (center_x, center_y),
                    2.0 * x_width,
                    2.0 * y_width,
                    **patch_options,
                )
            else:
                tube_patch = Rectangle(
                    (center_x - x_width, center_y - y_width),
                    2.0 * x_width,
                    2.0 * y_width,
                    **patch_options,
                )
            axis.add_patch(tube_patch)
            all_x.extend((center_x - x_width, center_x + x_width))
            all_y.extend((center_y - y_width, center_y + y_width))
        all_x.extend(nominal["x_m"])
        all_x.extend(rollout["x_m"])
        all_y.extend(nominal["y_m"])
        all_y.extend(rollout["y_m"])

    for obstacle_index, (center_x, center_y, radius) in enumerate(OBSTACLES):
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
        all_x.extend((center_x - radius, center_x + radius))
        all_y.extend((center_y - radius, center_y + radius))

    start = read_xy(ccm_dir / "nominal_plan.csv")
    axis.scatter(
        [start["x_m"][0]], [start["y_m"][0]], color="black", s=65, zorder=6,
        label="start",
    )
    axis.scatter(
        [-0.75], [-2.25], color="black", marker="*", s=180, zorder=6,
        label="goal",
    )
    all_x.append(-0.75)
    all_y.append(-2.25)

    margin = 0.18
    axis.set_xlim(min(all_x) - margin, max(all_x) + margin)
    axis.set_ylim(min(all_y) - margin, max(all_y) + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title("CCM vs Adaptive + LEB: Rollouts and Position Tubes")
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        (
            Patch(
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.10,
                label="CCM elliptical tubes",
            ),
            Patch(
                facecolor="tab:blue",
                edgecolor="tab:blue",
                alpha=0.10,
                label="Adaptive + LEB rectangular tubes",
            ),
        )
    )
    labels.extend(("CCM elliptical tubes", "Adaptive + LEB rectangular tubes"))
    axis.legend(handles, labels, loc="best")
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved rollout and tube overlay to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CCM and Adaptive + LEB tube widths on the same axes."
    )
    parser.add_argument("--ccm-csv", type=Path, default=DEFAULT_CCM_CSV)
    parser.add_argument(
        "--adaptive-leb-csv", type=Path, default=DEFAULT_ADAPTIVE_LEB_CSV
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ccm-dir", type=Path, default=DEFAULT_CCM_DIR)
    parser.add_argument(
        "--adaptive-leb-dir", type=Path, default=DEFAULT_ADAPTIVE_LEB_DIR
    )
    parser.add_argument(
        "--rollout-output", type=Path, default=DEFAULT_ROLLOUT_OUTPUT
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot_comparison(
        ccm_csv=arguments.ccm_csv,
        adaptive_leb_csv=arguments.adaptive_leb_csv,
        output=arguments.output,
    )
    plot_rollouts_with_tubes(
        ccm_dir=arguments.ccm_dir,
        adaptive_leb_dir=arguments.adaptive_leb_dir,
        output=arguments.rollout_output,
    )
