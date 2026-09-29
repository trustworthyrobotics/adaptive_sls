"""Plot receding-horizon forecast positions and XY tube boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import METHODS
from .plot_results import COLORS, LABELS


def plot_forecast_xy_tubes(
    results_dir: str | Path,
    output: str | Path | None = None,
    forecast_stride: int = 5,
    tube_stride: int = 5,
) -> Path:
    results_dir = Path(results_dir)
    output = Path(output or (results_dir / "forecast_xy_tubes.png"))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False)
    for column, method in enumerate(METHODS):
        x_axis = axes[0, column]
        y_axis = axes[1, column]
        method_root = results_dir
        if method == "pavone_rampc":
            scp6_root = results_dir.parent / (
                results_dir.name + "_pavone_scp6"
            )
            if (scp6_root / method).exists():
                method_root = scp6_root
        artifacts = sorted((method_root / method).glob("run_*.npz"))
        if not artifacts:
            x_axis.set_visible(False)
            y_axis.set_visible(False)
            continue

        # Use the first rollout as the representative forecast visualization.
        # The stored forecasts are receding-horizon predictions for that rollout.
        with np.load(artifacts[0]) as data:
            states = np.asarray(data["states"])
            forecasts = np.asarray(data["forecast_states"])
            widths = np.asarray(data["forecast_state_tube_widths"])

        time = np.arange(len(states))
        for axis, state_index, label in (
            (x_axis, 0, "x position (m)"),
            (y_axis, 1, "y position (m)"),
        ):
            axis.plot(
                time, states[:, state_index], color="black", linewidth=2.0,
                marker="o", markersize=2.5, label="executed", zorder=5,
            )
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)

        forecast_indices = range(0, len(forecasts), max(1, forecast_stride))
        forecast_indices = list(forecast_indices)
        for sequence_index, time_step in enumerate(forecast_indices):
            forecast = forecasts[time_step]
            tube = widths[time_step]
            line_alpha = 0.18 + 0.16 * (
                1.0 - sequence_index / max(1, len(forecast_indices) - 1)
            )
            for axis, state_index in ((x_axis, 0), (y_axis, 1)):
                horizon = min(len(forecast), len(tube))
                forecast_time = time_step + np.arange(horizon)
                axis.plot(
                    forecast_time, forecast[:horizon, state_index], linestyle="--",
                    color=COLORS[method], alpha=line_alpha, linewidth=1.2,
                    label="forecast" if sequence_index == 0 else None, zorder=3,
                )

                # Sample tube shading using the same stride convention while
                # fading the opacity from the near to the far horizon.
                for start in range(0, max(0, horizon - 1), max(1, tube_stride)):
                    stop = min(start + max(1, tube_stride), horizon - 1)
                    fraction = start / max(1, horizon - 1)
                    band_alpha = 0.18 - 0.13 * fraction
                    segment_time = forecast_time[start : stop + 1]
                    center = forecast[start : stop + 1, state_index]
                    half_width = tube[start : stop + 1, state_index]
                    axis.fill_between(
                        segment_time, center - half_width, center + half_width,
                        color=COLORS[method], alpha=band_alpha, linewidth=0, zorder=2,
                    )

        x_axis.set_title(f"{LABELS.get(method, method)} — run 0")
        x_axis.legend(fontsize=8, loc="best")
        y_axis.legend(fontsize=8, loc="best")

    for axis in axes[-1, :]:
        axis.set_xlabel("MPC time step")
    fig.suptitle(
        "Receding-horizon forecast positions and tube half-widths\n"
        f"forecast stride = {forecast_stride}, tube stride = {tube_stride}; "
        "tube opacity fades across each forecast horizon",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forecast-stride", type=int, default=5)
    parser.add_argument("--tube-stride", type=int, default=5)
    args = parser.parse_args()
    print(plot_forecast_xy_tubes(
        args.results_dir, args.output, args.forecast_stride, args.tube_stride
    ))


if __name__ == "__main__":
    main()
