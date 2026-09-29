#!/usr/bin/env python3
"""Plot the saved restricted sensitivity matrices H_t = C_t E_t over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "figure_data/parameter_provenance_seed_0033/run_01/quadruped_adaptive_damping_mjx_rollout.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name("h_t_over_time.png")

    with np.load(args.input, allow_pickle=False) as rollout:
        H = np.asarray(rollout["estimator_measured_sensitivities"], dtype=float)
        labels = np.asarray(rollout["theta_labels"]).astype(str)
        contacts = np.asarray(rollout["actual_contact_pairs"], dtype=bool)
        dt = float(rollout["controller_dt"])

    column_norms = np.linalg.norm(H, axis=1)  # (time, parameter)
    singular_values = np.linalg.svd(H, compute_uv=False)
    positive = column_norms[column_norms > 0.0]
    norm = LogNorm(
        vmin=max(float(np.min(positive)), 1.0e-9),
        vmax=float(np.max(positive)),
    )
    heatmap = np.ma.masked_where(column_norms.T == 0.0, column_norms.T)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("black")

    fig, (ax_h, ax_s) = plt.subplots(
        2, 1, figsize=(13, 6.8), sharex=True, height_ratios=(1.25, 1.0),
        constrained_layout=True,
    )
    image = ax_h.imshow(
        heatmap,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(0.0, H.shape[0] * dt, H.shape[2], 0.0),
    )
    readable = [label.replace("_damping", "").replace("_", " ") for label in labels]
    ax_h.set_yticks(np.arange(H.shape[2]) + 0.5, labels=readable)
    ax_h.set_ylabel("Parameter")
    ax_h.set_title(r"Restricted sensitivity $H_t=C_tE_t$: column norm $\|H_t[:,j]\|_2$")
    for boundary in range(2, H.shape[2], 2):
        ax_h.axhline(boundary, color="white", linewidth=1.0)
    fig.colorbar(image, ax=ax_h, pad=0.012, fraction=0.025).set_label(
        r"$\|H_t[:,j]\|_2$ (log scale)"
    )

    time = np.arange(H.shape[0]) * dt
    for index in range(singular_values.shape[1]):
        ax_s.semilogy(time, singular_values[:, index], linewidth=1.0, label=rf"$\sigma_{index + 1}$")
    ax_s.set_ylabel("Singular value")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_title("Singular values of $H_t$ (the estimator retains values above its rank threshold)")
    ax_s.grid(True, which="both", alpha=0.25)
    ax_s.legend(ncol=4, fontsize=8)

    # Report the strict, paper-level per-leg availability separately from H's
    # whole-body column norms, which can remain nonzero via another leg.
    own_leg_swing = np.repeat(np.all(~contacts, axis=1), 2, axis=1)
    fractions = own_leg_swing.mean(axis=0)
    fig.text(
        0.5, 0.002,
        "Own-leg swing availability: " + ", ".join(
            f"{label.split('_')[0]} {fraction:.1%}" for label, fraction in zip(labels[::2], fractions[::2])
        ),
        ha="center", fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
