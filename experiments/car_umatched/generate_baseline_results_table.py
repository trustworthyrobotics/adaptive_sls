#!/usr/bin/env python3
"""Generate the common-rollout table for the unmatched-car baselines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = (
    ("Adaptive SLS + LEB", "adaptive_leb", "adaptive_leb_controller.npz"),
    ("CCM", "ccm", "ccm_controller.npz"),
    ("DF (Pavone)", "pavone", "pavone_controller.npz"),
)
CONTAINMENT_TOLERANCE = 1.0e-5
METRICS = (
    ("Rand. XY contain. (\\%)", "random_xy_containment_pct", True),
    ("Corner XY contain. (\\%)", "corner_xy_containment_pct", True),
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


def load_controller_geometry(path: Path, method: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as controller:
        nominal = np.asarray(controller["nominal_states"], dtype=float)
        if method == "adaptive_leb":
            tube = np.asarray(controller["state_tube_half_widths"][:, :2], dtype=float)
        elif method == "ccm":
            times = np.arange(len(nominal), dtype=float) * float(controller["dt"])
            rate = float(controller["contraction_rate"])
            radius = float(controller["disturbance_bound"]) * (
                1.0 - np.exp(-rate * times)
            ) / rate
            position_radius = radius * float(controller["position_support"])
            tube = np.repeat(position_radius[:, None], 2, axis=1)
        else:
            response = np.asarray(controller["state_response"], dtype=float)
            disturbance_width = np.asarray(
                controller["disturbance_half_width"], dtype=float
            )
            transition_width = np.einsum(
                "tijk,k->ti", np.abs(response), disturbance_width
            )
            tube = np.vstack(
                [np.zeros((1, transition_width.shape[1])), transition_width]
            )[:, :2]
    if nominal.ndim != 2 or nominal.shape[1] < 2 or tube.shape != (len(nominal), 2):
        raise ValueError(f"invalid controller geometry in {path}")
    return nominal, tube


def containment(
    states: np.ndarray,
    nominal: np.ndarray,
    tube: np.ndarray,
    scenario_types: np.ndarray,
    successful: np.ndarray,
    scenario: str,
) -> Value:
    selected = (scenario_types == scenario) & successful
    deviations = np.abs(states[selected, :, :2] - nominal[None, :, :2])
    valid = np.isfinite(deviations) & np.isfinite(tube[None, :, :])
    count = int(np.count_nonzero(valid))
    if not count:
        return Value(None)
    contained = deviations <= tube[None, :, :] + CONTAINMENT_TOLERANCE
    return Value(100.0 * float(np.count_nonzero(contained & valid)) / count, count=count)


def read_method(
    states: np.ndarray,
    failed: np.ndarray,
    distances: np.ndarray,
    scenario_types: np.ndarray,
    nominal: np.ndarray,
    tube: np.ndarray,
) -> dict[str, Value]:
    steps = min(states.shape[1], nominal.shape[0], tube.shape[0])
    states, nominal, tube = states[:, :steps], nominal[:steps], tube[:steps]
    successful = ~failed
    errors = states[successful, :, :2] - nominal[None, :, :2]
    valid_errors = np.all(np.isfinite(errors), axis=-1)
    error_magnitudes = np.linalg.norm(errors[valid_errors], axis=-1)
    return {
        "random_xy_containment_pct": containment(
            states, nominal, tube, scenario_types, successful, "random"
        ),
        "corner_xy_containment_pct": containment(
            states, nominal, tube, scenario_types, successful, "adversarial"
        ),
        "final_goal_distance_m": mean_std(distances[successful]),
        "tracking_rmse_m": rmse_std(error_magnitudes),
        "log_position_tube_volume": log_volume_stats(tube),
        "log_parameter_tube_volume": Value(None),
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
        if not candidates:
            continue
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
    for display_name, method, _ in METHODS:
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
\\caption{{Unmatched-car baseline comparison over 30 random and 10 adversarial-corner rollouts per controller. XY containment pools both position coordinates across rollouts and timesteps using a $10^{{-5}}$ numerical tolerance. Goal distance reports the mean $\\pm$ population standard deviation of final XY distances. Tracking reports the pooled XY RMSE and the population standard deviation of individual per-step error magnitudes. The position-tube entry reports the timestep-level mean $\\pm$ population standard deviation of $\\log(h_xh_y)$ using saved half-widths. The natural logarithm is used, and samples with a nonpositive width are omitted because their log volume is undefined. Parameter-tube volume is unavailable for these saved baseline artifacts and is reported as ``--''. Containment is higher is better; all remaining metrics are lower is better.}}
\\label{{tab:car_unmatched_baseline_results}}
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
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--rollouts", type=Path, default=root / "controller_rollouts.npz"
    )
    parser.add_argument(
        "--controller-dir", type=Path, default=root / "saved_controllers"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "comparison_results"
    )
    args = parser.parse_args()

    with np.load(args.rollouts, allow_pickle=False) as data:
        method_names = [str(name) for name in data["method_names"]]
        states = np.asarray(data["states"], dtype=float)
        failed = np.asarray(data["failed"], dtype=bool)
        distances = np.asarray(data["distance_to_goal"], dtype=float)
        scenario_types = np.asarray(data["scenario_types"])

    results = {}
    for _, method, filename in METHODS:
        method_index = method_names.index(method)
        nominal, tube = load_controller_geometry(
            args.controller_dir / filename, method
        )
        results[method] = read_method(
            states[method_index],
            failed[method_index],
            distances[method_index],
            scenario_types,
            nominal,
            tube,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "car_unmatched_baseline_metrics.csv"
    tex_path = args.output_dir / "car_unmatched_baseline_results_table.tex"
    write_csv(results, csv_path)
    write_latex(results, tex_path)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
