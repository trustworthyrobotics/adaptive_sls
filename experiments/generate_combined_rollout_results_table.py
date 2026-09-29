#!/usr/bin/env python3
"""Combine the car-ablation, car-baseline, and quadrotor rollout tables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRICS = (
    ("Rand. contain. (\\%)", "random_containment_pct", True),
    ("Corner contain. (\\%)", "corner_containment_pct", True),
    ("Goal dist. (m)", "final_goal_distance_m", False),
    ("Tracking RMSE (m)", "tracking_rmse_m", False),
    ("Log pos. tube volume", "log_position_tube_volume", False),
    ("Log param. tube volume", "log_parameter_tube_volume", False),
)

EXPERIMENTS = (
    (
        "Dubins' Matched Ablation",
        "car_ablation",
        "matched",
        (
            "Adaptive SLS + LEB",
            "Adaptive SLS w/o LEB",
            "SLS + LEB",
            "SLS w/o LEB",
        ),
    ),
    (
        "Dubins' Unmatched Ablation",
        "car_ablation",
        "unmatched",
        (
            "Adaptive SLS + LEB",
            "Adaptive SLS w/o LEB",
            "SLS + LEB",
            "SLS w/o LEB",
        ),
    ),
    (
        "Dubins' Unmatched Baselines",
        "car_baselines",
        None,
        ("adaptive_leb", "ccm", "pavone"),
    ),
    (
        "Quadrotor Info Gathering",
        "quadrotor",
        None,
        (
            "adaptive__leb_true__edagger_f_false",
            "adaptive__leb_true__edagger_f_true",
        ),
    ),
)

METHOD_LABELS = {
    "adaptive_leb": "Adaptive SLS + LEB",
    "ccm": "CCM",
    "pavone": "DF (Pavone)",
    "adaptive__leb_true__edagger_f_false": "w/o active information gathering",
    "adaptive__leb_true__edagger_f_true": (
        "w/ active information gathering ($E^\\dagger F$)"
    ),
}

EXPERIMENT_LABELS = {
    "Dubins' Matched Ablation": r"\shortstack[l]{Dubins' Matched\\Ablation}",
    "Dubins' Unmatched Ablation": r"\shortstack[l]{Dubins' Unmatched\\Ablation}",
    "Dubins' Unmatched Baselines": r"\shortstack[l]{Dubins' Unmatched\\Baselines}",
    "Quadrotor Info Gathering": r"\shortstack[l]{Quadrotor Info\\Gathering}",
}

KEY_MAP = {
    "random_tube_containment_pct": "random_containment_pct",
    "adversarial_tube_containment_pct": "corner_containment_pct",
    "random_xy_containment_pct": "random_containment_pct",
    "corner_xy_containment_pct": "corner_containment_pct",
}


@dataclass(frozen=True)
class Value:
    mean: float | None
    std: float | None = None


def optional_float(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    value = float(text)
    return value if np.isfinite(value) else None


def load_csv(
    path: Path,
    *,
    source: str,
) -> dict[tuple[str | None, str, str], Value]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source metrics: {path}")
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model") if source == "car_ablation" else None
            key = KEY_MAP.get(row["metric"], row["metric"])
            values[(model, row["method"], key)] = Value(
                optional_float(row.get("mean")),
                optional_float(row.get("std")),
            )
    return values


def normalize_results(
    root: Path,
) -> dict[str, list[tuple[str, dict[str, Value]]]]:
    sources = {
        "car_ablation": load_csv(
            root / "car_ablation/results/car_ablation_metrics.csv",
            source="car_ablation",
        ),
        "car_baselines": load_csv(
            root
            / "car_umatched/comparison_results/car_unmatched_baseline_metrics.csv",
            source="car_baselines",
        ),
        "quadrotor": load_csv(
            root
            / "quadrotor/quadrotor_adaptive_leb_lower_info_fixed_comparison_results"
            / "quadrotor_active_information_metrics.csv",
            source="quadrotor",
        ),
    }
    normalized = {}
    for experiment, source, model, methods in EXPERIMENTS:
        rows = []
        for method in methods:
            metrics = {
                key: sources[source].get((model, method, key), Value(None))
                for _, key, _ in METRICS
            }
            rows.append((METHOD_LABELS.get(method, method), metrics))
        normalized[experiment] = rows
    return normalized


def format_value(value: Value, bold: bool) -> str:
    if value.mean is None:
        return "{\\small --}"
    number = f"{value.mean:.3g}"
    body = number if value.std is None else f"{number} \\pm {value.std:.2g}"
    if bold:
        body = f"\\bm{{{body}}}"
    return f"{{\\small ${body}$}}"


def winners(
    results: dict[str, list[tuple[str, dict[str, Value]]]],
) -> set[tuple[str, str, str]]:
    best = set()
    for experiment, rows in results.items():
        for _, key, higher_is_better in METRICS:
            candidates = [
                (method, values[key].mean)
                for method, values in rows
                if values[key].mean is not None
            ]
            if not candidates:
                continue
            target = (max if higher_is_better else min)(
                value for _, value in candidates
            )
            best.update(
                (experiment, method, key)
                for method, value in candidates
                if np.isclose(value, target)
            )
    return best


def write_csv(
    results: dict[str, list[tuple[str, dict[str, Value]]]], output: Path
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "method", "metric", "mean", "std"])
        for experiment, rows in results.items():
            for method, values in rows:
                for _, key, _ in METRICS:
                    value = values[key]
                    writer.writerow(
                        [experiment, method, key, value.mean, value.std]
                    )


def write_latex(
    results: dict[str, list[tuple[str, dict[str, Value]]]], output: Path
) -> None:
    best = winners(results)
    headers = " & ".join(name for name, _, _ in METRICS)
    rows = []
    experiments = list(results)
    for experiment_index, experiment in enumerate(experiments):
        experiment_rows = results[experiment]
        for row_index, (method, values) in enumerate(experiment_rows):
            cells = [
                (
                    f"\\multirow{{{len(experiment_rows)}}}{{*}}{{"
                    f"{EXPERIMENT_LABELS[experiment]}}}"
                    if row_index == 0
                    else ""
                ),
                method,
            ]
            cells.extend(
                format_value(
                    values[key], (experiment, method, key) in best
                )
                for _, key, _ in METRICS
            )
            rows.append(" & ".join(cells) + r" \\")
        if experiment_index != len(experiments) - 1:
            rows.append("\\midrule")
    table = f"""\\begin{{table*}}[!t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\medmuskip=1mu
\\thickmuskip=2mu
\\caption{{Unified rollout comparison. The Dubins ablations use 30 random and 10 adversarial-corner rollouts, the Dubins baseline comparison uses 30 random and 10 adversarial-corner rollouts, and the quadrotor comparison uses 120 random and 80 adversarial-corner rollouts. Containment pools valid coordinates across rollouts and timesteps with a $10^{{-5}}$ tolerance; the baseline CCM comparison reports XY coordinate containment, while the ablations and quadrotor report full physical-state coordinate containment. Goal distance is XY for Dubins and 3D position for the quadrotor. Tracking reports pooled position RMSE and the population standard deviation of per-step error magnitudes. Tube entries report the timestep-level mean $\\pm$ population standard deviation of the natural log of the product of coordinate half-widths; nonpositive-width samples are omitted. Parameter-tube volume is unavailable for the saved Dubins baseline artifacts and is reported as ``--''. Bold values are best only within the corresponding experiment. Containment is higher is better; all remaining metrics are lower is better.}}
\\label{{tab:combined_rollout_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}ll{'c' * len(METRICS)}@{{}}}}
\\toprule
Experiment & Method & {headers} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}
"""
    output.write_text(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = normalize_results(args.experiments_dir)
    csv_path = output_dir / "combined_rollout_metrics.csv"
    tex_path = output_dir / "combined_rollout_results_table.tex"
    write_csv(results, csv_path)
    write_latex(results, tex_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
