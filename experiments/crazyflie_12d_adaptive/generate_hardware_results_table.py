#!/usr/bin/env python3
"""Build reproducible Crazyflie hardware metrics and a LaTeX table.

Only trials containing one hardware log and an ``mpc_full_metrics.npz`` with
``metric_goal_reached=True`` are included. Timestep and forecast metrics are
pooled across successful trials; final parameter metrics contribute one sample
per trial.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WARMUP_SOLVES = 3
METHOD_KEY = "adaptive_sls_hardware"
METHOD_NAME = "A-SLS"

METRICS = (
    ("Solve $Q_1$/Med./$Q_3$ (ms)", "runtime_milliseconds"),
    ("XYZ tracking RMSE (m)", "tracking_rmse_m"),
    ("Log forecast pos. tube volume", "log_forecast_position_tube_volume"),
    ("Control effort", "control_effort"),
    ("Param. error", "final_parameter_error"),
    ("Log final $\\theta$-tube volume", "log_final_parameter_tube_volume"),
)


@dataclass(frozen=True)
class Value:
    mean: float
    std: float
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    count: int = 0


def finite(values: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def mean_std(values: np.ndarray | list[float]) -> Value:
    values = finite(values)
    if not values.size:
        return Value(np.nan, np.nan)
    return Value(float(np.mean(values)), float(np.std(values, ddof=0)), count=len(values))


def tracking_rmse_std(error_norms: np.ndarray | list[float]) -> Value:
    error_norms = finite(error_norms)
    if not error_norms.size:
        return Value(np.nan, np.nan)
    return Value(
        float(np.sqrt(np.mean(np.square(error_norms)))),
        float(np.std(error_norms, ddof=0)),
        count=len(error_norms),
    )


def successful_trials(root: Path):
    for directory in sorted(root.glob("run_*")):
        metrics = directory / "mpc_full_metrics.npz"
        logs = sorted(directory.glob("hardware_run_*.npz"))
        if not metrics.exists() or len(logs) != 1:
            continue
        with np.load(metrics, allow_pickle=False) as data:
            if "metric_goal_reached" not in data or not bool(data["metric_goal_reached"]):
                continue
        yield metrics, logs[0]


def last_solve_indices(sample_indices: np.ndarray) -> np.ndarray:
    sample_indices = np.asarray(sample_indices, dtype=int)
    return np.asarray(
        [
            index
            for index, sample in enumerate(sample_indices)
            if sample >= 1 and index == np.flatnonzero(sample_indices == sample)[-1]
        ],
        dtype=int,
    )


def aggregate_metrics(root: Path) -> tuple[dict[str, Value], int]:
    solve_times: list[float] = []
    tracking_errors: list[float] = []
    position_log_volumes: list[float] = []
    control_efforts: list[float] = []
    final_parameter_errors: list[float] = []
    final_parameter_log_volumes: list[float] = []
    trial_count = 0

    for _, log_path in successful_trials(root):
        trial_count += 1
        with np.load(log_path, allow_pickle=False) as data:
            solve_times.extend(
                (finite(data["solve_compute_seconds"])[WARMUP_SOLVES:] * 1e3).tolist()
            )

            predicted = np.asarray(data["estimate_one_step_prediction"], dtype=float)
            measured = np.asarray(data["estimate_measured_next"], dtype=float)
            valid = np.all(
                np.isfinite(predicted[:, :3]) & np.isfinite(measured[:, :3]), axis=1
            )
            tracking_errors.extend(
                np.linalg.norm(measured[valid, :3] - predicted[valid, :3], axis=1)
            )

            keep = last_solve_indices(data["solve_sample_index"])
            widths = np.abs(
                np.asarray(data["solve_constraint_backoffs"], dtype=float)[keep, 1:, :3]
            ).reshape(-1, 3)
            valid_widths = np.all(np.isfinite(widths) & (widths > 0.0), axis=1)
            position_log_volumes.extend(
                np.sum(np.log(widths[valid_widths]), axis=1).tolist()
            )

            commands = np.asarray(data["estimate_applied_command"], dtype=float)
            valid_commands = np.all(np.isfinite(commands), axis=1)
            control_efforts.extend(
                np.sum(np.square(commands[valid_commands]), axis=1).tolist()
            )

            final_error = finite(data["metric_parameter_error_final_from_calibrated"])
            if final_error.size:
                final_parameter_errors.append(abs(float(final_error[-1])))

            final_width = finite(data["metric_parameter_tube_width_final"])
            if final_width.size and final_width[-1] > 0.0:
                # This experiment learns one parameter, so its product of final
                # parameter half-widths is the scalar width itself.
                final_parameter_log_volumes.append(float(np.log(final_width[-1])))

    if not trial_count:
        raise SystemExit(f"No successful trials found in {root}")

    solve = finite(solve_times)
    values = {
        "runtime_milliseconds": Value(
            float(np.mean(solve)),
            float(np.std(solve, ddof=0)),
            float(np.quantile(solve, 0.25)),
            float(np.quantile(solve, 0.50)),
            float(np.quantile(solve, 0.75)),
            len(solve),
        ),
        "tracking_rmse_m": tracking_rmse_std(tracking_errors),
        "log_forecast_position_tube_volume": mean_std(position_log_volumes),
        "control_effort": mean_std(control_efforts),
        "final_parameter_error": mean_std(final_parameter_errors),
        "log_final_parameter_tube_volume": mean_std(final_parameter_log_volumes),
    }
    return values, trial_count


def format_value(value: Value, key: str) -> str:
    if key == "runtime_milliseconds":
        body = f"{value.q1:.3g}/{value.median:.3g}/{value.q3:.3g}"
    else:
        body = f"{value.mean:.3g} \\pm {value.std:.2g}"
    return f"{{\\small ${body}$}}"


def write_csv(values: dict[str, Value], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "metric", "mean", "std", "q1", "median", "q3", "sample_count"]
        )
        for _, key in METRICS:
            value = values[key]
            writer.writerow(
                [
                    METHOD_KEY,
                    key,
                    value.mean,
                    value.std,
                    value.q1,
                    value.median,
                    value.q3,
                    value.count,
                ]
            )


def write_latex(values: dict[str, Value], trial_count: int, output: Path) -> None:
    headers = " & ".join(label for label, _ in METRICS)
    cells = " & ".join(format_value(values[key], key) for _, key in METRICS)
    table = f"""\\begin{{table*}}[!t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\caption{{Crazyflie hardware results over {trial_count} successful rollouts. Solve time reports $Q_1$/median/$Q_3$ after excluding the first three solves of each rollout. Tracking reports pooled XYZ RMSE and the population standard deviation of individual error magnitudes. Forecast position-tube volume pools $\\log(h_xh_yh_z)$ over future forecast states. Control effort pools $\\lVert u\\rVert_2^2$. Final parameter metrics contribute one sample per rollout. Lower is better.}}
\\label{{tab:crazyflie_hardware_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}l{'c' * len(METRICS)}@{{}}}}
\\toprule
Method & {headers} \\\\
\\midrule
{METHOD_NAME} & {cells} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}
"""
    output.write_text(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "repeat_beta135_width2",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    values, trial_count = aggregate_metrics(args.results_dir)
    table_path = args.output or args.results_dir / "hardware_results_table.tex"
    csv_path = args.csv_output or args.results_dir / "hardware_metrics.csv"
    write_latex(values, trial_count, table_path)
    write_csv(values, csv_path)
    print(f"included {trial_count} successful trials")
    print(csv_path)
    print(table_path)


if __name__ == "__main__":
    main()
