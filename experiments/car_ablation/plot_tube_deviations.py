"""Plot car-ablation tube half-widths and saved-policy rollout deviations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHOD_LABELS = {
    "adaptive_true__leb_true": "Adaptive SLS + LEB",
    "adaptive_true__leb_false": "Adaptive SLS w/o LEB",
    "adaptive_false__leb_true": "SLS + LEB",
    "adaptive_false__leb_false": "SLS w/o LEB",
}
STATE_LABELS = (
    ("$|\\Delta x|$ (m)", 0),
    ("$|\\Delta y|$ (m)", 1),
    ("$|\\Delta \\psi|$ (rad)", 2),
    ("$|\\Delta v|$ (m/s)", 3),
)
CONTAINMENT_TOLERANCE = 1e-5


def plot_case(case_dir: Path, output: Path) -> None:
    with np.load(case_dir / "rollout_metrics.npz") as archive:
        rollouts = np.asarray(archive["rollout_states"], dtype=float)
        nominal = np.asarray(archive["nominal_states"], dtype=float)
        tube = np.asarray(archive["state_tube_half_widths"], dtype=float)
        kinds = np.asarray(archive["rollout_kinds"]) if "rollout_kinds" in archive.files else None
    with np.load(case_dir / "controller.npz") as controller:
        dt = float(controller["dt"])

    horizon = min(rollouts.shape[1], nominal.shape[0], tube.shape[0])
    times = np.arange(horizon) * dt
    rollouts, nominal, tube = rollouts[:, :horizon], nominal[:horizon], tube[:horizon]
    figure, axes = plt.subplots(4, 1, figsize=(3.35, 5.55), sharex=True, layout="constrained")
    for axis, (ylabel, state) in zip(axes, STATE_LABELS):
        signed_deviation = rollouts[:, :, state] - nominal[None, :, state]
        if state == 2:
            signed_deviation = (signed_deviation + np.pi) % (2.0 * np.pi) - np.pi
        deviations = np.abs(signed_deviation)
        for rollout_index, deviation in enumerate(deviations):
            is_corner = kinds is not None and kinds[rollout_index] == "adversarial_corner"
            color = "#D55E00" if is_corner else "#56B4E9"
            alpha = 0.38 if is_corner else 0.18
            axis.plot(times, deviation, color=color, alpha=alpha, linewidth=0.48, zorder=2)
            violations = deviation > tube[:, state] + CONTAINMENT_TOLERANCE
            if np.any(violations):
                axis.scatter(times[violations], deviation[violations], color="#CC0000", s=4, zorder=4)
        if kinds is not None:
            for kind, color in (("random", "#0072B2"), ("adversarial_corner", "#D55E00")):
                group = deviations[kinds == kind]
                if len(group):
                    axis.plot(times, np.nanmax(group, axis=0), color=color, linewidth=1.05, zorder=3)
        axis.plot(times, tube[:, state], color="black", linewidth=1.35, label="tube half-width", zorder=3)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3, linewidth=0.4)
        axis.set_ylim(bottom=0.0)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"{case_dir.parent.name.title()}: {METHOD_LABELS[case_dir.name]}", fontsize=9)
    figure.legend(
        handles=[
            Line2D([], [], color="black", linewidth=1.35, label="certified half-width"),
            Line2D([], [], color="#0072B2", linewidth=1.0, label="max. random deviation"),
            Line2D([], [], color="#D55E00", linewidth=1.0, label="max. corner deviation"),
            Line2D([], [], color="#CC0000", marker="o", linestyle="None", markersize=3, label="violation"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--format", choices=("png", "pdf"), default="pdf")
    args = parser.parse_args()
    for model in ("matched", "unmatched"):
        for case_name in METHOD_LABELS:
            case_dir = args.results_dir / model / case_name
            output = case_dir / f"all_rollouts_deviation_vs_tube_width.{args.format}"
            plot_case(case_dir, output)
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
