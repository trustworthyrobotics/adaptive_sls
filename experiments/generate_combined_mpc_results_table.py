#!/usr/bin/env python3
"""Combine simulation and hardware receding-horizon MPC results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRICS = (
    ("Solve $Q_1$/Med./$Q_3$ (ms)", "runtime_milliseconds", False),
    ("Tracking RMSE (m)", "tracking_rmse_m", False),
    ("Log forecast pos. tube volume", "log_forecast_position_tube_volume", False),
    ("Control effort", "control_effort", False),
    ("Param. error", "final_parameter_error", False),
    ("Log final $\\theta$-tube volume", "log_final_parameter_tube_volume", False),
)

EXPERIMENTS = (
    (
        "Planar Quadrotor",
        "planar",
        (
            ("A-SLS", "adaptive_sls_gain"),
            ("A-SLS SME", "adaptive_sls_sme"),
            ("CCM", "ccm_rampc"),
            ("DF (Pavone)", "pavone_rampc"),
        ),
    ),
    (
        "Quadruped Damping",
        "quadruped",
        (
            ("Adaptive SLS", "adaptive"),
            ("Non-adaptive SLS", "nonadaptive"),
        ),
    ),
    (
        "Crazyflie Hardware",
        "hardware",
        (("A-SLS", "adaptive_sls_hardware"),),
    ),
)

EXPERIMENT_LABELS = {
    "Planar Quadrotor": r"\shortstack[l]{Planar\\Quadrotor}",
    "Quadruped Damping": r"\shortstack[l]{Quadruped\\Damping}",
    "Crazyflie Hardware": r"\shortstack[l]{Crazyflie\\Hardware}",
}


@dataclass(frozen=True)
class Value:
    mean: float | None
    std: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    count: int = 0


def optional_float(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    value = float(text)
    return value if np.isfinite(value) else None


def load_metrics(path: Path) -> dict[tuple[str, str], Value]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source metrics: {path}")
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["method"], row["metric"])] = Value(
                mean=optional_float(row.get("mean")),
                std=optional_float(row.get("std")),
                q1=optional_float(row.get("q1")),
                median=optional_float(row.get("median")),
                q3=optional_float(row.get("q3")),
                count=int(row.get("sample_count") or 0),
            )
    return values


def normalize_results(
    repo_root: Path,
) -> dict[str, list[tuple[str, dict[str, Value]]]]:
    planar_dir = repo_root / "experiments/planar_quadrotor/results"
    planar = load_metrics(planar_dir / "planar_quadrotor_metrics.csv")
    quadruped = load_metrics(
        repo_root
        / "experiments/quadruped/sweeps/damping_40/summary"
        / "quadruped_damping_metrics.csv"
    )
    hardware = load_metrics(
        repo_root
        / "experiments/crazyflie_12d_adaptive/results/repeat_beta135_width2"
        / "hardware_metrics.csv"
    )
    sources = {"planar": planar, "quadruped": quadruped, "hardware": hardware}

    results = {}
    for experiment, source, methods in EXPERIMENTS:
        rows = []
        for display_name, method in methods:
            values = {}
            for _, key, _ in METRICS:
                values[key] = sources[source].get((method, key), Value(None))
            rows.append((display_name, values))
        results[experiment] = rows
    return results


def winners(
    results: dict[str, list[tuple[str, dict[str, Value]]]],
) -> set[tuple[str, str, str]]:
    best = set()
    for experiment, rows in results.items():
        for _, key, higher_is_better in METRICS:
            candidates = []
            for method, values in rows:
                value = values[key]
                comparison = value.median if key == "runtime_milliseconds" else value.mean
                if comparison is not None:
                    candidates.append((method, comparison))
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


def format_value(value: Value, key: str, bold: bool) -> str:
    if value.mean is None:
        return "{\\small --}"
    if key == "runtime_milliseconds":
        body = f"{value.q1:.3g}/{value.median:.3g}/{value.q3:.3g}"
    else:
        body = f"{value.mean:.3g}"
        if value.std is not None:
            body += f" \\pm {value.std:.2g}"
    if bold:
        body = f"\\bm{{{body}}}"
    return f"{{\\small ${body}$}}"


def write_csv(
    results: dict[str, list[tuple[str, dict[str, Value]]]], output: Path
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["experiment", "method", "metric", "mean", "std", "q1", "median", "q3", "sample_count"]
        )
        for experiment, rows in results.items():
            for method, values in rows:
                for _, key, _ in METRICS:
                    value = values[key]
                    writer.writerow(
                        [
                            experiment,
                            method,
                            key,
                            value.mean,
                            value.std,
                            value.q1,
                            value.median,
                            value.q3,
                            value.count,
                        ]
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
            if row_index != 0:
                experiment_cell = ""
            elif len(experiment_rows) == 1:
                experiment_cell = EXPERIMENT_LABELS[experiment]
            else:
                experiment_cell = (
                    f"\\multirow{{{len(experiment_rows)}}}{{*}}{{"
                    f"{EXPERIMENT_LABELS[experiment]}}}"
                )
            cells = [experiment_cell, method]
            cells.extend(
                format_value(
                    values[key], key, (experiment, method, key) in best
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
\\caption{{Unified receding-horizon MPC comparison. Solve time reports $Q_1$/median/$Q_3$ over finite post-warm-up solves, excluding the first three planar-quadrotor and Crazyflie hardware solves and first four quadruped solves per rollout. Tracking reports pooled position RMSE (XY in simulation and XYZ in hardware) and the population standard deviation of per-step error magnitudes. Forecast position-tube volume pools the natural log of the product of positional half-widths over every future state in every receding-horizon forecast. Planar control effort is $\\lVert u-u_{{\\mathrm{{hover}}}}\\rVert_2^2$, Crazyflie hardware effort is $\\lVert u\\rVert_2^2$, and quadruped effort is $\\lVert u\\rVert_2$ but is unavailable for legacy checkpoints that predate torque logging. Parameter error is the final Euclidean estimation error. Final parameter-tube volume is the natural log of the product of final parameter half-widths. Nonpositive-width samples are omitted. Bold values are best only within each experiment; lower is better.}}
\\label{{tab:combined_mpc_results}}
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
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = normalize_results(args.repo_root)
    csv_path = output_dir / "combined_mpc_metrics.csv"
    tex_path = output_dir / "combined_mpc_results_table.tex"
    write_csv(results, csv_path)
    write_latex(results, tex_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
