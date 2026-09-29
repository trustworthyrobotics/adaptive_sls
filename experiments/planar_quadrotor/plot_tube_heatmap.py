"""Plot forecast XY tube widths as time-by-horizon heat maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_results import LABELS


def plot_tube_heatmap(results_dir: str | Path, output: str | Path | None = None) -> Path:
    results_dir = Path(results_dir)
    output = Path(output or (results_dir / "tube_width_heatmap.png"))
    methods = ("adaptive_sls_gain", "adaptive_sls_sme", "ccm_rampc", "pavone_rampc")

    maps: dict[str, np.ndarray] = {}
    for method in methods:
        values = []
        for artifact in sorted((results_dir / method).glob("run_*.npz")):
            with np.load(artifact) as data:
                widths = np.asarray(data["forecast_state_tube_widths"], dtype=float)
            if widths.size:
                # XY widths are axis-aligned half-widths. Use the larger one
                # at each forecast point to summarize the planar tube width.
                values.append(np.max(widths[:, :, :2], axis=2))
        if values:
            rows = max(v.shape[0] for v in values)
            cols = max(v.shape[1] for v in values)
            stack = np.full((len(values), rows, cols), np.nan)
            for i, value in enumerate(values):
                stack[i, : value.shape[0], : value.shape[1]] = value
            maps[method] = np.nanmean(stack, axis=0)

    max_rows = max(v.shape[0] for v in maps.values())
    max_cols = max(v.shape[1] for v in maps.values())
    for method, values in list(maps.items()):
        padded = np.full((max_rows, max_cols), np.nan)
        padded[: values.shape[0], : values.shape[1]] = values
        maps[method] = padded

    finite = np.concatenate([v[np.isfinite(v)] for v in maps.values()])
    vmax = float(np.max(finite))

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, method in zip(axes.ravel(), methods):
        values = maps.get(method)
        if values is None:
            axis.set_visible(False)
            continue
        image = axis.imshow(
            values.T,
            origin="lower",
            interpolation="none",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(LABELS.get(method, method))
        axis.set_xlabel("MPC time step")
        axis.set_ylabel("Forecast horizon step")
        axis.set_xticks(np.arange(0, max_rows, max(1, max_rows // 6)))
        axis.set_yticks(np.arange(max_cols))
        axis.grid(False)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02, shrink=0.9)
        colorbar.set_label("maximum forecast XY tube half-width (m)")
    fig.suptitle("Forecasted planar tube widths over MPC time and horizon", fontsize=14)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(plot_tube_heatmap(args.results_dir, args.output))


if __name__ == "__main__":
    main()
