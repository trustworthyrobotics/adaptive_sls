"""Generate planar-quadrotor comparison metrics and a LaTeX table.

The inputs are the archives written by :mod:`experiments.planar_quadrotor.run_experiment`.
For every MPC call, the archived plan predicts the state at the next time step.
This script uses the saved one-step prediction to calculate XY tracking RMSE,
so it does not rely on a separately saved, open-loop nominal trajectory.

The forecasted-sequence tube widths average every *future* state in every
receding-horizon plan (horizons 1 through N).

Example
-------
``conda run -n adaptive_sls python -m experiments.planar_quadrotor.generate_results_table``

For the centered-obstacle experiment, pass its directory.  When a sibling
``*_pavone_scp6`` directory exists, its successful Pavone rollouts replace the
strict-RTI Pavone results automatically (or pass ``--pavone-results-dir``).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


METHODS = (
    ("A-SLS", "adaptive_sls_gain"),
    ("A-SLS SME", "adaptive_sls_sme"),
    ("CCM", "ccm_rampc"),
    ("DF (Pavone)", "pavone_rampc"),
)
WARMUP_SOLVES = 3
HOVER_INPUT = ExperimentConfig().hover_input
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
    count: int = 0
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    samples: np.ndarray | None = field(default=None, repr=False, compare=False)


def finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def mean_std(values: list[float] | np.ndarray) -> Value:
    values = finite(np.asarray(values, dtype=float)).reshape(-1)
    if not len(values):
        return Value(None)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return Value(
        float(np.mean(values)),
        float(np.std(values, ddof=0)) if len(values) > 1 else None,
        len(values),
        float(q1),
        float(median),
        float(q3),
        values,
    )


def tracking_rmse_std(values: list[float] | np.ndarray) -> Value:
    """Summarize pooled per-step XY error magnitudes as RMSE plus/minus std."""
    values = finite(np.asarray(values, dtype=float)).reshape(-1)
    if not len(values):
        return Value(None)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    return Value(
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.std(values, ddof=0)) if len(values) > 1 else None,
        len(values),
        float(q1),
        float(median),
        float(q3),
        values,
    )


def log_volume_stats(widths: np.ndarray) -> Value:
    """Summarize log products of strictly positive finite half-width rows."""
    widths = np.asarray(widths, dtype=float)
    valid = np.all(np.isfinite(widths) & (widths > 0.0), axis=-1)
    if not np.any(valid):
        return Value(None)
    return mean_std(np.sum(np.log(widths[valid]), axis=-1))


def load_rollout(path: Path) -> dict[str, Value]:
    """Compute one-run quantities from its saved receding-horizon data."""
    with np.load(path) as data:
        failed = bool(data["failed"])
        states = np.asarray(data["states"], dtype=float)
        forecasts = np.asarray(data["forecast_states"], dtype=float)
        tubes = np.asarray(data["forecast_state_tube_widths"], dtype=float)
        solve_times = np.asarray(data["solve_times_seconds"], dtype=float)
        controls = np.asarray(data["controls"], dtype=float)
        parameter_errors = np.asarray(data["parameter_errors"], dtype=float)
        parameter_widths = np.asarray(data["parameter_half_widths"], dtype=float)

    # A rejected/failed rollout is not a valid controller execution, so do not
    # mix its partial plan with successful-rollout metrics.
    if failed:
        return {key: Value(None) for _, key, _ in METRICS}

    steps = min(len(forecasts), len(tubes), max(0, len(states) - 1))
    executed_next = states[1 : steps + 1]
    predicted_next = forecasts[:steps, 1, :]
    xy_valid = np.all(np.isfinite(executed_next[:, :2]) & np.isfinite(predicted_next[:, :2]), axis=1)
    xy_deviation = executed_next[:, :2] - predicted_next[:, :2]
    tracking = (
        tracking_rmse_std(np.linalg.norm(xy_deviation[xy_valid], axis=1))
        if np.any(xy_valid)
        else Value(None)
    )
    final_parameter_error = finite(np.linalg.norm(parameter_errors, axis=1))
    future_tubes = tubes[:, 1:, :2]
    return {
        "runtime_milliseconds": mean_std(1e3 * solve_times[WARMUP_SOLVES:]),
        "tracking_rmse_m": tracking,
        "log_forecast_position_tube_volume": log_volume_stats(future_tubes),
        "control_effort": mean_std(
            np.sum((controls - HOVER_INPUT) ** 2, axis=1)
        ),
        "final_parameter_error": (
            mean_std([final_parameter_error[-1]])
            if len(final_parameter_error)
            else Value(None)
        ),
        "log_final_parameter_tube_volume": (
            log_volume_stats(parameter_widths[-1:])
            if len(parameter_widths)
            else Value(None)
        ),
    }


def aggregate(rollouts: list[dict[str, Value]]) -> dict[str, Value]:
    result: dict[str, Value] = {}
    for _, key, _ in METRICS:
        pooled_samples = [
                rollout[key].samples
                for rollout in rollouts
                if rollout[key].samples is not None
        ]
        pooled = (
            np.concatenate(pooled_samples) if pooled_samples else np.empty(0)
        )
        result[key] = (
            tracking_rmse_std(pooled)
            if key == "tracking_rmse_m"
            else mean_std(pooled)
        )
    return result


def read_method(directory: Path) -> dict[str, Value]:
    artifacts = sorted(directory.glob("run_*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No rollout archives found in {directory}")
    return aggregate([load_rollout(path) for path in artifacts])


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


def winners(results: dict[str, dict[str, Value]]) -> set[tuple[str, str]]:
    result = set()
    for _, key, higher_is_better in METRICS:
        candidates = [
            (
                method,
                values[key].median
                if key == "runtime_milliseconds"
                else values[key].mean,
            )
            for method, values in results.items()
        ]
        candidates = [(method, value) for method, value in candidates if value is not None]
        if candidates:
            target = (max if higher_is_better else min)(value for _, value in candidates)
            result.update((method, key) for method, value in candidates if np.isclose(value, target))
    return result


def write_csv(results: dict[str, dict[str, Value]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "metric", "mean", "std", "q1", "median", "q3", "sample_count"]
        )
        for method, values in results.items():
            for _, key, _ in METRICS:
                value = values[key]
                writer.writerow(
                    [
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


def write_latex(results: dict[str, dict[str, Value]], output: Path) -> None:
    best = winners(results)
    headers = " & ".join(name for name, _, _ in METRICS)
    rows = []
    for display_name, method in METHODS:
        cells = [display_name]
        cells.extend(
            format_value(
                results[method][key],
                (method, key) in best,
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
\\caption{{Planar-quadrotor comparison over the shared rollout scenarios. Solve time reports $Q_1$/median/$Q_3$ over all finite MPC solves after excluding the first three solves of each rollout to remove JAX warm-up. All other entries pool their underlying samples across all successful rollouts and report the pooled value $\\pm$ population standard deviation. Tracking reports the pooled XY RMSE and the population standard deviation of the individual per-step XY error magnitudes between the executed state and the preceding MPC plan's one-step prediction. Forecast position-tube volume pools $\\log(h_xh_y)$ over every future state in every receding-horizon plan. Control effort pools $\\lVert u-u_{{\\mathrm{{hover}}}}\\rVert_2^2$, where $u_{{\\mathrm{{hover}}}}=(mg/2,mg/2)$. The final parameter-tube entry reports $\\log(\\prod_i h_{{\\theta_i}})$ with one final sample per rollout. Natural logs of saved half-width products are used, and nonpositive-width samples are omitted. Lower is better.}}
\\label{{tab:planar_quadrotor_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}l{'c' * len(METRICS)}@{{}}}}
\\toprule
Method & {headers} \\\\
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
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--pavone-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    results_dir = args.results_dir
    pavone_dir = args.pavone_results_dir
    if pavone_dir is None:
        candidate = results_dir.parent / f"{results_dir.name}_pavone_scp6"
        pavone_dir = candidate if (candidate / "pavone_rampc").exists() else results_dir
    output_dir = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        method: read_method((pavone_dir if method == "pavone_rampc" else results_dir) / method)
        for _, method in METHODS
    }
    write_csv(results, output_dir / "planar_quadrotor_metrics.csv")
    write_latex(results, output_dir / "planar_quadrotor_results_table.tex")
    print(output_dir / "planar_quadrotor_metrics.csv")
    print(output_dir / "planar_quadrotor_results_table.tex")


if __name__ == "__main__":
    main()
