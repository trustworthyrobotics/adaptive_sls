#!/usr/bin/env python3
"""Generate quadruped damping-sweep metrics and a LaTeX comparison table.

Solve-time statistics pool all finite post-warm-up MPC solves from successful
rollouts and are reported as Q1/median/Q3.  Other continuous metrics pool their
underlying samples and are reported as mean plus/minus population standard
deviation.  The non-finite success rate uses every requested seed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = (
    ("Adaptive SLS", "adaptive", "quadruped_adaptive_damping_mjx_rollout.npz"),
    ("Non-adaptive SLS", "nonadaptive", "quadruped_nonadaptive_damping_mjx_rollout.npz"),
)
WARMUP_SOLVES = 4
METRICS = (
    ("Solve $Q_1$/Med./$Q_3$ (ms)", "runtime_milliseconds", False),
    ("Tracking RMSE (m)", "tracking_rmse_m", False),
    ("Log forecast pos. tube volume", "log_forecast_position_tube_volume", False),
    ("Control effort", "control_effort", False),
    ("Param. error", "final_parameter_error", False),
    ("Log final $\\theta$-tube volume", "log_final_parameter_tube_volume", False),
)


@dataclass(frozen=True)
class Value:
    mean: float | None
    std: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    count: int = 0


@dataclass(frozen=True)
class MethodResult:
    metrics: dict[str, Value]
    successes: int
    trials: int


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def summarize(values: list[float] | np.ndarray) -> Value:
    values = finite(np.asarray(values, dtype=float))
    if not len(values):
        return Value(None)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return Value(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=0)) if len(values) > 1 else None,
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        count=len(values),
    )


def tracking_rmse_std(values: list[float] | np.ndarray) -> Value:
    """Summarize pooled per-step XY error magnitudes as RMSE plus/minus std."""
    values = finite(np.asarray(values, dtype=float))
    if not len(values):
        return Value(None)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return Value(
        mean=float(np.sqrt(np.mean(np.square(values)))),
        std=float(np.std(values, ddof=0)) if len(values) > 1 else None,
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        count=len(values),
    )


def log_volume_samples(widths: np.ndarray) -> np.ndarray:
    """Return log products for strictly positive finite half-width rows."""
    widths = np.asarray(widths, dtype=float)
    valid = np.all(np.isfinite(widths) & (widths > 0.0), axis=-1)
    if not np.any(valid):
        return np.empty(0)
    return np.sum(np.log(widths[valid]), axis=-1).reshape(-1)


def load_successful_rollout(
    path: Path,
    *,
    expected_steps: int,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Return per-rollout metrics, or ``None`` for a partial/failed rollout."""
    with np.load(path) as data:
        states = np.asarray(data["xs"], dtype=float)
        one_step_plans = np.asarray(data["one_step_plans"], dtype=float)
        horizon_tubes = np.asarray(data["horizon_tubes"], dtype=float)
        solve_times = np.asarray(data["transition_times"], dtype=float)
        theta_estimates = np.asarray(data["theta_estimates"], dtype=float)
        theta_widths = np.asarray(data["theta_widths"], dtype=float)
        true_theta = np.asarray(data["true_theta"], dtype=float)
        nominal_controls = (
            np.asarray(data["nominal_controls"], dtype=float)
            if "nominal_controls" in data.files
            else np.empty((0, 0))
        )

    if len(solve_times) != expected_steps or len(states) != expected_steps + 1:
        return None
    if len(one_step_plans) != expected_steps or len(horizon_tubes) != expected_steps:
        return None

    solve_times_ms = finite(1.0e3 * solve_times[WARMUP_SOLVES:])
    if not len(solve_times_ms):
        return None

    executed_xy = states[1:, :2]
    predicted_xy = one_step_plans[:, :2]
    valid_tracking = np.all(
        np.isfinite(executed_xy) & np.isfinite(predicted_xy), axis=1
    )
    if not np.any(valid_tracking):
        return None
    tracking_errors = executed_xy[valid_tracking] - predicted_xy[valid_tracking]
    tracking_error_norms = np.linalg.norm(tracking_errors, axis=1)

    # horizon_tubes already contains forecast horizons 1 through N, so every
    # entry is a future state rather than the current state at horizon zero.
    position_log_volumes = log_volume_samples(horizon_tubes[..., :2])
    control_effort = (
        finite(np.linalg.norm(nominal_controls, axis=-1))
        if nominal_controls.size
        else np.empty(0)
    )
    final_parameter_error = np.array(
        [np.linalg.norm(theta_estimates[-1] - true_theta)], dtype=float
    )
    final_parameter_log_volume = log_volume_samples(theta_widths[-1:])

    metrics = {
        "tracking_rmse_m": tracking_error_norms,
        "log_forecast_position_tube_volume": position_log_volumes,
        "control_effort": control_effort,
        "final_parameter_error": final_parameter_error,
        "log_final_parameter_tube_volume": final_parameter_log_volume,
    }
    return metrics, solve_times_ms


def read_method(
    sweep_root: Path,
    mode: str,
    checkpoint_name: str,
    seeds: range,
    expected_steps: int,
) -> MethodResult:
    rollouts: list[dict[str, np.ndarray]] = []
    solve_time_arrays: list[np.ndarray] = []
    missing: list[int] = []
    for seed in seeds:
        path = sweep_root / f"seed_{seed:04d}" / mode / checkpoint_name
        if not path.exists():
            missing.append(seed)
            continue
        loaded = load_successful_rollout(
            path,
            expected_steps=expected_steps,
        )
        if loaded is not None:
            rollout, solve_times_ms = loaded
            rollouts.append(rollout)
            solve_time_arrays.append(solve_times_ms)

    if missing:
        raise FileNotFoundError(
            f"{mode}: missing checkpoints for seeds {', '.join(map(str, missing))}"
        )

    metrics = {}
    for _, key, _ in METRICS:
        if key == "runtime_milliseconds":
            continue
        pooled = (
            np.concatenate([rollout[key] for rollout in rollouts])
            if rollouts
            else np.empty(0)
        )
        metrics[key] = (
            tracking_rmse_std(pooled)
            if key == "tracking_rmse_m"
            else summarize(pooled)
        )
    metrics["runtime_milliseconds"] = summarize(
        np.concatenate(solve_time_arrays) if solve_time_arrays else np.empty(0)
    )
    return MethodResult(metrics=metrics, successes=len(rollouts), trials=len(seeds))


def format_value(value: Value, bold: bool, *, quartile_format: bool) -> str:
    if value.mean is None:
        return "{\\small --}"
    if quartile_format:
        body = f"{value.q1:.3g}/{value.median:.3g}/{value.q3:.3g}"
    else:
        body = f"{value.mean:.3g}"
        if value.std is not None:
            body += f" \\pm {value.std:.2g}"
    if bold:
        body = f"\\bm{{{body}}}"
    return f"{{\\small ${body}$}}"


def format_success(result: MethodResult, bold: bool) -> str:
    percent = 100.0 * result.successes / result.trials
    body = f"{percent:.1f}\\%\\;({result.successes}/{result.trials})"
    if bold:
        body = f"\\bm{{{body}}}"
    return f"{{\\small ${body}$}}"


def winners(results: dict[str, MethodResult]) -> set[tuple[str, str]]:
    best: set[tuple[str, str]] = set()
    for _, key, higher_is_better in METRICS:
        candidates = [
            (
                mode,
                result.metrics[key].median
                if key == "runtime_milliseconds"
                else result.metrics[key].mean,
            )
            for mode, result in results.items()
            if result.metrics[key].mean is not None
        ]
        if not candidates:
            continue
        target = (max if higher_is_better else min)(value for _, value in candidates)
        best.update(
            (mode, key) for mode, value in candidates if np.isclose(value, target)
        )
    success_target = max(result.successes / result.trials for result in results.values())
    best.update(
        (mode, "success_rate")
        for mode, result in results.items()
        if np.isclose(result.successes / result.trials, success_target)
    )
    return best


def write_csv(results: dict[str, MethodResult], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "metric", "mean", "std", "q1", "median", "q3", "sample_count"]
        )
        for mode, result in results.items():
            writer.writerow(
                [
                    mode,
                    "nonfinite_success_rate",
                    result.successes / result.trials,
                    "",
                    "",
                    "",
                    "",
                    result.trials,
                ]
            )
            for _, key, _ in METRICS:
                value = result.metrics[key]
                writer.writerow(
                    [
                        mode,
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
    results: dict[str, MethodResult],
    output: Path,
    *,
    half_width: float,
    expected_steps: int,
) -> None:
    best = winners(results)
    headers = " & ".join(name for name, _, _ in METRICS)
    rows = []
    for display_name, mode, _ in METHODS:
        result = results[mode]
        cells = [display_name, format_success(result, (mode, "success_rate") in best)]
        cells.extend(
            format_value(
                result.metrics[key],
                (mode, key) in best,
                quartile_format=key == "runtime_milliseconds",
            )
            for _, key, _ in METRICS
        )
        rows.append(" & ".join(cells) + r" \\")

    table = f"""\\begin{{table*}}[!t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\medmuskip=1mu
\\thickmuskip=2mu
\\caption{{Quadruped comparison over 40 matched randomized-damping scenarios with parameter half-width {half_width:g}. Non-finite success is the fraction completing all {expected_steps} MPC steps without a non-finite solution. Solve time reports $Q_1$/median/$Q_3$ over all finite MPC solves from successful rollouts after excluding the first four solves of each rollout for JAX warm-up. All other continuous entries pool their underlying samples across successful rollouts and report the pooled value $\\pm$ population standard deviation. Tracking reports pooled XY RMSE and the population standard deviation of individual per-step XY error magnitudes. Forecast position-tube volume pools $\\log(h_xh_y)$ over every future state in every receding-horizon plan. Control effort pools $\\lVert u\\rVert_2$ over saved nominal torque commands; it is unavailable for legacy checkpoints that predate torque logging. Parameter error is $\\lVert\\hat{{\\theta}}-\\theta^\\star\\rVert_2$ at the final step, and the final parameter-tube entry reports $\\log(\\prod_i h_{{\\theta_i}})$ with one final sample per completed rollout. Natural logs of saved half-width products are used, and nonpositive-width samples are omitted. Lower is better except for success rate.}}
\\label{{tab:quadruped_damping_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}l{'c' * (len(METRICS) + 1)}@{{}}}}
\\toprule
Method & Non-finite success & {headers} \\\\
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
        "--sweep-root",
        type=Path,
        default=Path(__file__).resolve().parent / "sweeps" / "damping_40",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=40)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--half-width", type=float, default=1.6)
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.num_seeds)
    results = {
        mode: read_method(
            args.sweep_root,
            mode,
            checkpoint_name,
            seeds,
            args.steps,
        )
        for _, mode, checkpoint_name in METHODS
    }

    output_dir = args.output_dir or args.sweep_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quadruped_damping_metrics.csv"
    latex_path = output_dir / "quadruped_damping_results_table.tex"
    write_csv(results, csv_path)
    write_latex(
        results,
        latex_path,
        half_width=args.half_width,
        expected_steps=args.steps,
    )

    for display_name, mode, _ in METHODS:
        result = results[mode]
        print(f"{display_name}: success {result.successes}/{result.trials}")
        for name, key, _ in METRICS:
            value = result.metrics[key]
            if key == "runtime_milliseconds":
                summary = f"{value.q1:.8g}/{value.median:.8g}/{value.q3:.8g}"
            elif value.mean is None:
                summary = "unavailable"
            else:
                summary = f"{value.mean:.8g} +/- {value.std:.8g}"
            print(f"  {name}: {summary} (n={value.count})")
    print(csv_path)
    print(latex_path)


if __name__ == "__main__":
    main()
