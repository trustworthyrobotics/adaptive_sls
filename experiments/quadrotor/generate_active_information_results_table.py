#!/usr/bin/env python3
"""Generate the w/ versus w/o active-information quadrotor rollout table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = (
    ("w/o active information gathering", "adaptive__leb_true__edagger_f_false"),
    ("w/ active information gathering ($E^\\dagger F$)", "adaptive__leb_true__edagger_f_true"),
)
N_PHYS = 12
GOAL_POSITION = np.asarray([1.0, 0.8, 0.4])
CONTAINMENT_TOLERANCE = 1.0e-5
METRICS = (
    ("Rand. contain. (\\%)", "random_containment_pct", True),
    ("Corner contain. (\\%)", "corner_containment_pct", True),
    ("Goal dist. (m)", "final_goal_distance_m", False),
    ("Tracking RMSE (m)", "tracking_rmse_m", False),
    ("Log pos. tube volume", "log_position_tube_volume", False),
    ("Log param. tube volume", "log_parameter_tube_volume", False),
)


@dataclass(frozen=True)
class Value:
    mean: float | None
    std: float | None = None
    count: int = 0


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def mean_std(values: np.ndarray) -> Value:
    values = finite(values)
    if not len(values):
        return Value(None)
    return Value(
        float(np.mean(values)),
        float(np.std(values, ddof=0)) if len(values) > 1 else None,
        len(values),
    )


def rmse_std(error_magnitudes: np.ndarray) -> Value:
    values = finite(error_magnitudes)
    if not len(values):
        return Value(None)
    return Value(
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.std(values, ddof=0)) if len(values) > 1 else None,
        len(values),
    )


def log_volume_stats(widths: np.ndarray) -> Value:
    """Summarize log products of strictly positive finite half-width rows."""
    widths = np.asarray(widths, dtype=float)
    valid = np.all(np.isfinite(widths) & (widths > 0.0), axis=1)
    if not np.any(valid):
        return Value(None)
    return mean_std(np.sum(np.log(widths[valid]), axis=1))


def containment(
    deviations: np.ndarray,
    tubes: np.ndarray,
    scenario_types: np.ndarray,
    scenario: str,
) -> Value:
    selected = scenario_types == scenario
    values = deviations[selected, :, :N_PHYS]
    bounds = tubes[None, :, :N_PHYS]
    valid = np.isfinite(values) & np.isfinite(bounds)
    count = int(np.count_nonzero(valid))
    if not count:
        return Value(None)
    contained = values <= bounds + CONTAINMENT_TOLERANCE
    return Value(100.0 * float(np.count_nonzero(contained & valid)) / count, count=count)


def final_position_distances(states: np.ndarray) -> np.ndarray:
    distances = []
    for rollout in states:
        valid = np.all(np.isfinite(rollout[:, :3]), axis=1)
        if np.any(valid):
            final_position = rollout[np.flatnonzero(valid)[-1], :3]
            distances.append(np.linalg.norm(final_position - GOAL_POSITION))
    return np.asarray(distances)


def read_case(path: Path) -> dict[str, Value]:
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=float)
        prediction = np.asarray(data["prediction"], dtype=float)
        tubes = np.asarray(data["state_tubes"], dtype=float)
        deviations = np.asarray(data["deviations"], dtype=float)
        scenario_types = np.asarray(data["scenario_types"])
        force_tubes = np.asarray(data["planned_force_widths"], dtype=float)

    steps = min(states.shape[1], prediction.shape[0], tubes.shape[0])
    states = states[:, :steps]
    prediction = prediction[:steps]
    tubes = tubes[:steps]
    deviations = deviations[:, :steps]
    force_tubes = force_tubes[:steps]
    position_errors = states[:, :, :3] - prediction[None, :, :3]
    valid_position = np.all(np.isfinite(position_errors), axis=-1)
    error_magnitudes = np.linalg.norm(position_errors[valid_position], axis=-1)

    return {
        "random_containment_pct": containment(
            deviations, tubes, scenario_types, "random"
        ),
        "corner_containment_pct": containment(
            deviations, tubes, scenario_types, "corner_adversarial"
        ),
        "final_goal_distance_m": mean_std(final_position_distances(states)),
        "tracking_rmse_m": rmse_std(error_magnitudes),
        "log_position_tube_volume": log_volume_stats(tubes[:, :3]),
        "log_parameter_tube_volume": log_volume_stats(force_tubes),
    }


def format_value(value: Value, bold: bool) -> str:
    if value.mean is None:
        return "{\\small --}"
    number = f"{value.mean:.3g}"
    body = number if value.std is None else f"{number} \\pm {value.std:.2g}"
    if bold:
        body = f"\\bm{{{body}}}"
    return f"{{\\small ${body}$}}"


def winners(results: dict[str, dict[str, Value]]) -> set[tuple[str, str]]:
    best = set()
    for _, key, higher_is_better in METRICS:
        candidates = [
            (method, values[key].mean)
            for method, values in results.items()
            if values[key].mean is not None
        ]
        target = (max if higher_is_better else min)(value for _, value in candidates)
        best.update(
            (method, key)
            for method, value in candidates
            if np.isclose(value, target)
        )
    return best


def write_csv(results: dict[str, dict[str, Value]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "mean", "std", "sample_count"])
        for method, values in results.items():
            for _, key, _ in METRICS:
                value = values[key]
                writer.writerow([method, key, value.mean, value.std, value.count])


def write_latex(results: dict[str, dict[str, Value]], output: Path) -> None:
    best = winners(results)
    headers = " & ".join(name for name, _, _ in METRICS)
    rows = []
    for display_name, method in METHODS:
        cells = [display_name]
        cells.extend(
            format_value(results[method][key], (method, key) in best)
            for _, key, _ in METRICS
        )
        rows.append(" & ".join(cells) + r" \\")
    table = f"""\\begin{{table*}}[!t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\medmuskip=1mu
\\thickmuskip=2mu
\\caption{{Quadrotor comparison with 120 random and 80 adversarial-corner rollouts per controller. Containment pools all valid physical-state coordinates across rollouts and timesteps using a $10^{{-5}}$ numerical tolerance. Goal distance reports the mean $\\pm$ population standard deviation of final 3D position errors. Tracking reports the pooled 3D position RMSE and the population standard deviation of individual per-step error magnitudes. Position and force-parameter tube entries report the timestep-level mean $\\pm$ population standard deviation of $\\log(\\prod_i h_i)$, where $h_i$ are the saved coordinate half-widths. The natural logarithm is used, and samples with a nonpositive width are omitted because their log volume is undefined. Containment is higher is better; all remaining metrics are lower is better.}}
\\label{{tab:quadrotor_active_information_results}}
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).with_name(
            "quadrotor_adaptive_leb_lower_info_fixed_comparison_results"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        method: read_case(
            args.results_dir / method / "quadrotor_adaptive_sls_rollout.npz"
        )
        for _, method in METHODS
    }
    csv_path = output_dir / "quadrotor_active_information_metrics.csv"
    tex_path = output_dir / "quadrotor_active_information_results_table.tex"
    write_csv(results, csv_path)
    write_latex(results, tex_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
