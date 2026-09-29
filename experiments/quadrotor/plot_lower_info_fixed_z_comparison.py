"""Plot the fixed lower-information-altitude quadrotor comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quadrotor_common import (
    DISTURBANCE_MAGNITUDE,
    DT,
    N_PHYS,
    OBSTACLES,
    X0_PHYS,
    X_GOAL_PHYS,
)


ROOT = Path(__file__).resolve().parent / (
    "quadrotor_adaptive_leb_lower_info_fixed_comparison_results"
)
OUTPUT = ROOT / "quadrotor_adaptive_sls_z_comparison.png"
XY_OUTPUT = ROOT / "quadrotor_adaptive_sls_xy_tubes_comparison.png"
PARAMETER_OUTPUT = ROOT / "quadrotor_adaptive_sls_parameter_comparison.png"

STYLES = {
    "without": {
        "nominal": "#174A7E",
        "rollout": "#72A9D5",
        "tube_alpha": 0.05,
        "label": r"w/o active information gathering",
    },
    "with": {
        "nominal": "#6A3D8F",
        "rollout": "#B995D1",
        "tube_alpha": 0.16,
        "label": r"w/ active information gathering ($E^\dagger F$)",
    },
}


def load_case(name: str) -> dict[str, np.ndarray | float]:
    path = ROOT / name / "quadrotor_adaptive_sls_rollout.npz"
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def spatially_sampled_indices(path: np.ndarray, max_patches: int = 16) -> np.ndarray:
    """Select approximately equally spaced patches by XY arc length."""
    if path.shape[0] <= max_patches:
        return np.arange(path.shape[0])

    segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cumulative_distance = np.concatenate(
        [np.zeros(1), np.cumsum(segment_lengths)]
    )
    total_distance = cumulative_distance[-1]
    if total_distance <= 0.0:
        return np.linspace(0, path.shape[0] - 1, max_patches, dtype=int)

    target_distances = np.linspace(0.0, total_distance, max_patches)
    indices = np.searchsorted(cumulative_distance, target_distances)
    indices = np.clip(indices, 0, path.shape[0] - 1)
    return np.unique(indices)


def plot_xy_comparison(cases: dict[str, dict[str, np.ndarray | float]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 8))

    for name, case in cases.items():
        style = STYLES[name]
        states = case["states"]
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        steps = min(prediction.shape[0], tubes.shape[0])
        sampled_steps = spatially_sampled_indices(
            prediction[:steps],
            max_patches=20,
        )

        for rollout_index, rollout in enumerate(states):
            axis.plot(
                rollout[:, 0],
                rollout[:, 1],
                color=style["rollout"],
                linewidth=0.9,
                alpha=0.30,
                label=f"{style['label']} rollouts" if rollout_index == 0 else None,
                zorder=2,
            )

        for tube_index, step in enumerate(sampled_steps):
            center = prediction[step, :2]
            half_width = tubes[step, :2]
            box = plt.Rectangle(
                center - half_width,
                2.0 * half_width[0],
                2.0 * half_width[1],
                facecolor=style["nominal"],
                edgecolor=style["nominal"],
                linewidth=0.8,
                alpha=style["tube_alpha"],
                label=f"{style['label']} XY tube" if tube_index == 0 else None,
                zorder=3,
            )
            axis.add_patch(box)

        axis.plot(
            prediction[:steps, 0],
            prediction[:steps, 1],
            color=style["nominal"],
            linestyle="--",
            linewidth=2.5,
            label=f"{style['label']} nominal",
            zorder=4,
        )

    axis.scatter(
        [float(X0_PHYS[0])],
        [float(X0_PHYS[1])],
        marker="o",
        color="black",
        label="start",
        zorder=6,
    )
    axis.scatter(
        [float(X_GOAL_PHYS[0])],
        [float(X_GOAL_PHYS[1])],
        marker="x",
        color="black",
        label="goal",
        zorder=6,
    )
    for obstacle_index, obstacle in enumerate(OBSTACLES):
        circle = plt.Circle(
            (float(obstacle[0]), float(obstacle[1])),
            float(obstacle[2]),
            color="tab:red",
            alpha=0.25,
            label="obstacle" if obstacle_index == 0 else None,
            zorder=1,
        )
        axis.add_patch(circle)

    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Adaptive SLS + LEB: Lower Information-Altitude XY Comparison")
    axis.axis("equal")
    axis.grid(True, alpha=0.45)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(XY_OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {XY_OUTPUT}")


def plot_parameter_comparison(
    cases: dict[str, dict[str, np.ndarray | float]],
) -> None:
    labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for parameter_index, (axis, parameter_label) in enumerate(zip(axes, labels)):
        for name, case in cases.items():
            style = STYLES[name]
            estimates = case["force_estimates"]
            true_forces = case["true_forces"]
            half_widths = case["planned_force_widths"][:, parameter_index]
            errors = np.abs(
                estimates[:, :, parameter_index]
                - true_forces[:, None, parameter_index]
            )
            time = np.arange(estimates.shape[1]) * DT

            for rollout_index, error in enumerate(errors):
                axis.plot(
                    time,
                    error,
                    color=style["rollout"],
                    linewidth=0.8,
                    alpha=0.28,
                    label=(
                        f"{style['label']} parameter errors"
                        if rollout_index == 0
                        else None
                    ),
                    zorder=2,
                )
            axis.fill_between(
                time,
                np.zeros_like(half_widths),
                half_widths,
                color=style["nominal"],
                alpha=style["tube_alpha"],
                label=f"{style['label']} half-width",
                zorder=1,
            )
            axis.plot(
                time,
                half_widths,
                color=style["nominal"],
                linestyle="--",
                linewidth=2.0,
                label=None,
                zorder=3,
            )

        axis.set_ylabel(f"|{parameter_label}| (N)")
        axis.grid(True, alpha=0.45)
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Adaptive SLS + LEB: Parameter Half-Widths and Errors")
    fig.tight_layout()
    fig.savefig(PARAMETER_OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PARAMETER_OUTPUT}")


def main() -> None:
    cases = {
        "without": load_case("adaptive__leb_true__edagger_f_false"),
        "with": load_case("adaptive__leb_true__edagger_f_true"),
    }

    z_values = []
    for case in cases.values():
        states = case["states"]
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        z_values.extend(
            [
                states[..., 2].reshape(-1),
                prediction[:, 2],
                prediction[:, 2] - tubes[:, 2],
                prediction[:, 2] + tubes[:, 2],
            ]
        )
    finite_z = np.concatenate(z_values)
    finite_z = finite_z[np.isfinite(finite_z)]
    z_low = float(np.min(finite_z))
    z_high = float(np.max(finite_z))
    z_margin = max(0.02, 0.05 * (z_high - z_low))
    z_low -= z_margin
    z_high += z_margin

    first_case = next(iter(cases.values()))
    information_center_z = float(first_case["information_center_z"])
    disturbance_z_off = float(first_case["disturbance_z_off"])
    disturbance_z_sharpness = float(first_case["disturbance_z_sharpness"])
    z_edges = np.linspace(z_low, z_high, 256)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    gate = 0.5 * (
        1.0
        - np.tanh(disturbance_z_sharpness * (z_centers - disturbance_z_off))
    )
    disturbance_map_norm = DISTURBANCE_MAGNITUDE * DT * gate * np.sqrt(N_PHYS)
    time_high = max(
        float(case["prediction"].shape[0] - 1) * DT for case in cases.values()
    )

    fig, axis = plt.subplots(figsize=(10, 7))
    image = axis.pcolormesh(
        [0.0, time_high],
        z_edges,
        disturbance_map_norm[:, None],
        cmap="magma",
        alpha=0.25,
        shading="auto",
        zorder=0,
    )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"$\|E(z)\|_F$")

    for name, case in cases.items():
        style = STYLES[name]
        states = case["states"]
        prediction = case["prediction"]
        tubes = case["state_tubes"]
        time = np.arange(states.shape[1]) * DT
        plan_time = np.arange(prediction.shape[0]) * DT

        for rollout_index, rollout in enumerate(states):
            axis.plot(
                time,
                rollout[:, 2],
                color=style["rollout"],
                linewidth=0.9,
                alpha=0.30,
                label=f"{style['label']} rollouts" if rollout_index == 0 else None,
                zorder=2,
            )
        axis.fill_between(
            plan_time,
            prediction[:, 2] - tubes[:, 2],
            prediction[:, 2] + tubes[:, 2],
            color=style["nominal"],
            alpha=0.18,
            label=f"{style['label']} z-tube",
            zorder=3,
        )
        axis.plot(
            plan_time,
            prediction[:, 2],
            color=style["nominal"],
            linestyle="--",
            linewidth=2.5,
            label=f"{style['label']} nominal",
            zorder=4,
        )

    axis.axhline(
        float(first_case["states"][0, 0, 2]),
        color="black",
        linestyle="-",
        alpha=0.35,
        label="start/goal altitude",
        zorder=5,
    )
    axis.axhline(
        information_center_z,
        color="tab:green",
        linestyle=":",
        linewidth=1.5,
        label=f"information altitude (z={information_center_z:.2f} m)",
        zorder=5,
    )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("z (m)")
    axis.set_title("Adaptive SLS + LEB: Lower Information-Altitude Comparison")
    axis.grid(True, alpha=0.45)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    plot_xy_comparison(cases)
    plot_parameter_comparison(cases)


if __name__ == "__main__":
    main()
