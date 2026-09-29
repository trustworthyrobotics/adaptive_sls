"""Plot trajectories and aggregate diagnostics from a completed comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np

from .config import METHODS, OBSTACLES, ROOT, X0, X_GOAL


COLORS = {
    "adaptive_sls_gain": "tab:blue",
    "adaptive_sls_sme": "tab:cyan",
    "ccm_rampc": "tab:orange",
    "pavone_rampc": "tab:green",
}
LABELS = {
    "adaptive_sls_gain": "A-SLS gain",
    "adaptive_sls_sme": "A-SLS SME",
    "ccm_rampc": "CCM-RAMPC",
    "pavone_rampc": "Pavone ARMPC",
}


def _result_obstacles(results_dir: Path) -> np.ndarray:
    config_path = results_dir / "config.json"
    if config_path.exists():
        import json

        with config_path.open() as handle:
            config = json.load(handle)
        if all(
            key in config for key in ("obstacle_x", "obstacle_y", "obstacle_radius")
        ):
            return np.array(
                [[config["obstacle_x"], config["obstacle_y"], config["obstacle_radius"]]],
                dtype=float,
            )
    return OBSTACLES


def plot_results(
    results_dir: str | Path,
    output: str | Path | None = None,
    pavone_results_dir: str | Path | None = None,
) -> Path:
    results_dir = Path(results_dir)
    pavone_results_dir = (
        None if pavone_results_dir is None else Path(pavone_results_dir)
    )
    output = Path(output or (results_dir / "comparison.png"))
    obstacles = _result_obstacles(results_dir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    trajectory_axis, solve_axis, tube_axis, parameter_axis = axes.ravel()
    for method in METHODS:
        method_root = (
            pavone_results_dir
            if method == "pavone_rampc" and pavone_results_dir is not None
            else results_dir
        )
        method_label = LABELS[method]
        if method == "pavone_rampc" and pavone_results_dir is not None:
            method_label += " (6 SCP)"
        artifacts = sorted((method_root / method).glob("run_*.npz"))
        if not artifacts:
            continue
        solve_series, tube_series, parameter_series = [], [], []
        for artifact_index, artifact in enumerate(artifacts):
            with np.load(artifact) as data:
                states = data["states"]
                trajectory_axis.plot(
                    states[:, 0], states[:, 1], color=COLORS[method], alpha=0.22,
                    label=method_label if artifact_index == 0 else None,
                )
                solve_series.append(data["solve_times_seconds"])
                tubes = data["forecast_state_tube_widths"]
                tube_series.append(np.max(tubes[:, :, :2], axis=(1, 2)) if len(tubes) else np.array([]))
                parameter_series.append(np.linalg.norm(data["parameter_errors"], axis=1))
        for axis, series in (
            (solve_axis, solve_series),
            (tube_axis, tube_series),
            (parameter_axis, parameter_series),
        ):
            usable = [values for values in series if len(values)]
            if not usable:
                continue
            length = min(map(len, usable))
            stacked = np.stack([values[:length] for values in usable])
            mean = np.mean(stacked, axis=0)
            low, high = np.quantile(stacked, [0.1, 0.9], axis=0)
            steps = np.arange(length)
            axis.plot(steps, mean, color=COLORS[method], label=method_label)
            axis.fill_between(steps, low, high, color=COLORS[method], alpha=0.15)

    trajectory_axis.scatter(*X0[:2], color="black", marker="o", label="start")
    trajectory_axis.scatter(*X_GOAL[:2], color="black", marker="x", label="goal")
    for index, obstacle in enumerate(obstacles):
        trajectory_axis.add_patch(
            Circle(obstacle[:2], obstacle[2], color="tab:red", alpha=0.2,
                   label="obstacle" if index == 0 else None)
        )
    trajectory_axis.set(xlabel="x (m)", ylabel="y (m)", title="Closed-loop trajectories")
    trajectory_axis.axis("equal")
    solve_axis.set(xlabel="MPC step", ylabel="seconds", title="Solve time")
    solve_axis.set_yscale("log")
    solve_axis.axhline(
        0.05, color="black", linestyle="--", linewidth=1.0, label="50 ms deadline"
    )
    tube_axis.set(xlabel="MPC step", ylabel="half-width (m)", title="Maximum forecast XY tube")
    parameter_axis.set(
        xlabel="MPC step",
        ylabel=r"kg$^{-1}$",
        title="Inverse-mass parameter error norm",
    )
    for axis in axes.ravel():
        axis.grid(True, alpha=0.35)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mpc_rollout(
    results_dir: str | Path,
    run_index: int = 0,
    output: str | Path | None = None,
    pavone_results_dir: str | Path | None = None,
) -> Path:
    """Plot representative receding-horizon predictions and XY tube sections."""
    results_dir = Path(results_dir)
    pavone_results_dir = (
        None if pavone_results_dir is None else Path(pavone_results_dir)
    )
    output = Path(output or (results_dir / f"mpc_rollout_run_{run_index:03d}.png"))
    obstacles = _result_obstacles(results_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    forecast_steps = (0, 5, 10, 15, 20, 25)

    for axis, method in zip(axes.ravel(), METHODS):
        method_root = (
            pavone_results_dir
            if method == "pavone_rampc" and pavone_results_dir is not None
            else results_dir
        )
        method_label = LABELS[method]
        if method == "pavone_rampc" and pavone_results_dir is not None:
            method_label += " (6 SCP)"
        artifact = method_root / method / f"run_{run_index:03d}.npz"
        if not artifact.exists():
            axis.set_visible(False)
            continue
        with np.load(artifact) as data:
            states = data["states"]
            forecasts = data["forecast_states"]
            widths = data["forecast_state_tube_widths"]
            goal_errors = data["goal_errors"]
            clearances = data["obstacle_clearances"]

        axis.plot(
            states[:, 0], states[:, 1], color=COLORS[method], linewidth=2.5,
            label="closed loop", zorder=4,
        )
        for sequence_index, step in enumerate(forecast_steps):
            if step >= len(forecasts):
                continue
            forecast = forecasts[step]
            tube = widths[step]
            alpha = 0.25 + 0.1 * sequence_index
            axis.plot(
                forecast[:, 0], forecast[:, 1], linestyle="--",
                color=COLORS[method], alpha=min(alpha, 0.8), linewidth=1.2,
                label="MPC forecasts" if sequence_index == 0 else None,
                zorder=2,
            )
            # The stored XY widths are axis-aligned half-widths. Show the
            # terminal cross-section of each selected forecast as an ellipse.
            terminal = forecast[-1, :2]
            terminal_width = tube[-1, :2]
            axis.add_patch(
                Ellipse(
                    terminal,
                    width=2.0 * terminal_width[0],
                    height=2.0 * terminal_width[1],
                    edgecolor=COLORS[method],
                    facecolor=COLORS[method],
                    linewidth=0.8,
                    alpha=0.12,
                    zorder=1,
                )
            )

        axis.scatter(*X0[:2], color="black", s=25, marker="o", label="start", zorder=5)
        axis.scatter(*X_GOAL[:2], color="black", s=45, marker="x", label="goal", zorder=5)
        for obstacle_index, obstacle in enumerate(obstacles):
            axis.add_patch(
                Circle(
                    obstacle[:2], obstacle[2], color="tab:red", alpha=0.18,
                    label="obstacle" if obstacle_index == 0 else None,
                )
            )
        axis.set_title(
            f"{method_label}\n"
            f"{len(states) - 1} steps, final error {goal_errors[-1]:.3f} m, "
            f"clearance {np.min(clearances):.3f} m"
        )
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=7, loc="upper right")

    for axis in axes[-1]:
        axis.set_xlabel("x (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("y (m)")
    fig.suptitle(
        f"Planar quadrotor MPC rollout {run_index}: selected horizon forecasts "
        "and terminal XY tubes",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rollout-run", type=int)
    parser.add_argument("--rollout-output", type=Path)
    parser.add_argument(
        "--pavone-results-dir",
        type=Path,
        help="read Pavone artifacts from a separate results directory",
    )
    args = parser.parse_args()
    print(plot_results(args.results_dir, args.output, args.pavone_results_dir))
    if args.rollout_run is not None:
        print(
            plot_mpc_rollout(
                args.results_dir,
                args.rollout_run,
                args.rollout_output,
                args.pavone_results_dir,
            )
        )


if __name__ == "__main__":
    main()
