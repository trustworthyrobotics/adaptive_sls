"""Generate reproducible matched/unmatched car-ablation metrics and LaTex.

Run after ``run_car_ablation.py``.  Each experiment case must contain the
``rollout_metrics.npz`` and ``summary.json`` files written by the runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = (
    ("Adaptive SLS + LEB", "adaptive_true__leb_true"),
    ("Adaptive SLS w/o LEB", "adaptive_true__leb_false"),
    ("SLS + LEB", "adaptive_false__leb_true"),
    ("SLS w/o LEB", "adaptive_false__leb_false"),
)
CONTAINMENT_TOLERANCE = 1e-5
METRICS = (
    ("Rand. contain. (\\%)", "random_tube_containment_pct", True),
    ("Corner contain. (\\%)", "adversarial_tube_containment_pct", True),
    ("Goal dist. (m)", "final_goal_distance_m", False),
    ("Tracking RMSE (m)", "tracking_rmse_m", False),
    ("Log pos. tube volume", "log_position_tube_volume", False),
    ("Log param. tube volume", "log_parameter_tube_volume", False),
)


@dataclass(frozen=True)
class Value:
    mean: float | None
    std: float | None = None


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def mean_std(values: np.ndarray) -> Value:
    values = finite(np.asarray(values, dtype=float).reshape(-1))
    if not len(values):
        return Value(None)
    return Value(float(np.mean(values)), float(np.std(values, ddof=0)) if len(values) > 1 else None)


def tracking_rmse_std(error_magnitudes: np.ndarray) -> Value:
    """Return pooled XY RMSE and std of the per-step XY error magnitudes."""
    values = finite(np.asarray(error_magnitudes, dtype=float).reshape(-1))
    if not len(values):
        return Value(None)
    return Value(
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.std(values, ddof=0)) if len(values) > 1 else None,
    )


def log_volume_stats(widths: np.ndarray) -> Value:
    """Summarize log products of strictly positive finite half-width rows."""
    widths = np.asarray(widths, dtype=float)
    valid = np.all(np.isfinite(widths) & (widths > 0.0), axis=1)
    if not np.any(valid):
        return Value(None)
    return mean_std(np.sum(np.log(widths[valid]), axis=1))


def last_finite_xy(trajectory: np.ndarray) -> np.ndarray | None:
    valid = np.all(np.isfinite(trajectory[:, :2]), axis=1)
    return trajectory[np.flatnonzero(valid)[-1], :2] if np.any(valid) else None


def read_case(case_dir: Path) -> dict[str, Value]:
    metrics_path = case_dir / "rollout_metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{case_dir} lacks rollout_metrics.npz. Run "
            "experiments/car_ablation/sample_saved_controller_rollouts.py."
        )
    with np.load(metrics_path) as data:
        rollouts = np.asarray(data["rollout_states"], dtype=float)
        nominal = np.asarray(data["nominal_states"], dtype=float)
        tube = np.asarray(data["state_tube_half_widths"], dtype=float)
        parameter_tube = (
            np.asarray(data["parameter_tube_half_widths"], dtype=float)
            if "parameter_tube_half_widths" in data.files
            else np.empty((0, 0))
        )
        goal_state = np.asarray(data["goal_state"], dtype=float)
        rollout_kinds = np.asarray(data["rollout_kinds"]) if "rollout_kinds" in data.files else None
    horizon = min(rollouts.shape[1], nominal.shape[0], tube.shape[0])
    rollouts, nominal, tube = rollouts[:, :horizon], nominal[:horizon], tube[:horizon]
    physical_dim = min(4, rollouts.shape[-1], nominal.shape[-1], tube.shape[-1])
    deviations = np.abs(rollouts[:, :, :physical_dim] - nominal[None, :, :physical_dim])
    valid = np.isfinite(deviations) & np.isfinite(tube[None, :, :physical_dim])
    contained = deviations <= tube[None, :, :physical_dim] + CONTAINMENT_TOLERANCE
    containment = 100.0 * np.count_nonzero(contained & valid) / np.count_nonzero(valid)

    def containment_for(kind: str) -> Value:
        if rollout_kinds is None:
            return Value(float(containment))
        mask = rollout_kinds == kind
        selected_valid = valid[mask]
        if not np.count_nonzero(selected_valid):
            return Value(None)
        return Value(100.0 * np.count_nonzero(contained[mask] & selected_valid) / np.count_nonzero(selected_valid))

    final_distances = []
    tracking_error_magnitudes = []
    for rollout in rollouts:
        last_xy = last_finite_xy(rollout)
        if last_xy is not None:
            final_distances.append(np.linalg.norm(last_xy - goal_state[:2]))
        delta_xy = rollout[:, :2] - nominal[:, :2]
        valid_xy = np.all(np.isfinite(delta_xy), axis=1)
        if np.any(valid_xy):
            tracking_error_magnitudes.extend(
                np.linalg.norm(delta_xy[valid_xy], axis=1)
            )

    return {
        "random_tube_containment_pct": containment_for("random"),
        "adversarial_tube_containment_pct": containment_for("adversarial_corner"),
        "final_goal_distance_m": mean_std(np.asarray(final_distances)),
        "tracking_rmse_m": tracking_rmse_std(
            np.asarray(tracking_error_magnitudes)
        ),
        "log_position_tube_volume": log_volume_stats(tube[:, :2]),
        "log_parameter_tube_volume": (
            log_volume_stats(parameter_tube) if parameter_tube.size else Value(None)
        ),
    }


def format_value(value: Value, bold: bool) -> str:
    if value.mean is None:
        return "{\\small --}"
    number = f"{value.mean:.3g}"
    body = number if value.std is None else f"{number} \\pm {value.std:.2g}"
    body = f"\\bm{{{body}}}" if bold else body
    return f"{{\\small ${body}$}}"


def best_values(results: dict[str, dict[str, dict[str, Value]]]) -> set[tuple[str, str, str]]:
    winners = set()
    for model in results:
        for _, key, higher_is_better in METRICS:
            candidates = [(method, results[model][method][key].mean) for method, _ in METHODS]
            candidates = [(method, value) for method, value in candidates if value is not None]
            if candidates:
                target = (max if higher_is_better else min)(value for _, value in candidates)
                winners.update((model, method, key) for method, value in candidates if np.isclose(value, target))
    return winners


def write_csv(results: dict[str, dict[str, dict[str, Value]]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "method", "metric", "mean", "std"])
        for model, methods in results.items():
            for method, values in methods.items():
                for _, key, _ in METRICS:
                    value = values[key]
                    writer.writerow([model, method, key, value.mean, value.std])


def write_latex(results: dict[str, dict[str, dict[str, Value]]], output: Path) -> None:
    winners = best_values(results)
    metric_headers = " & ".join(name for name, _, _ in METRICS)
    rows = []
    for model in results:
        rows.append(f"\\multicolumn{{{len(METRICS) + 1}}}{{@{{}}l}}{{\\textit{{{model.title()} dynamics}}}} \\\\")
        for method, _ in METHODS:
            cells = [method]
            cells.extend(
                format_value(results[model][method][key], (model, method, key) in winners)
                for _, key, _ in METRICS
            )
            rows.append(" & ".join(cells) + r" \\")
        if model != list(results)[-1]:
            rows.append("\\midrule")
    table = f"""\\begin{{table*}}[!t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\medmuskip=1mu
\\thickmuskip=2mu
\\caption{{Car ablation with 30 random and 10 adversarial-corner rollouts per controller. Containment pools all valid physical-state coordinates across rollouts and timesteps and uses a $10^{{-5}}$ numerical tolerance; random and corner containment are higher is better. Goal distance reports the mean $\\pm$ population standard deviation of the final distances from all rollouts. Tracking reports the pooled XY RMSE and the population standard deviation of the individual per-step XY error magnitudes. Position and parameter tube entries report the timestep-level mean $\\pm$ population standard deviation of $\\log(\\prod_i h_i)$, where $h_i$ are the saved coordinate half-widths. The natural logarithm is used, and samples with a nonpositive width are omitted because their log volume is undefined. All remaining metrics are lower is better.}}
\\label{{tab:car_ablation_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}l{'c' * len(METRICS)}@{{}}}}
\\toprule
Method & {metric_headers} \\\\
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        model: {method: read_case(args.results_dir / model / directory) for method, directory in METHODS}
        for model in ("matched", "unmatched")
    }
    # Do not mix fresh parameter-generator data with legacy controllers.  The
    # parameter-volume column becomes available only after all adaptive
    # policies in a complete ablation run have saved their generator histories.
    parameter_keys = ("log_parameter_tube_volume",)
    adaptive_methods = ("Adaptive SLS + LEB", "Adaptive SLS w/o LEB")
    if not all(
        results[model][method][key].mean is not None
        for model in results
        for method in adaptive_methods
        for key in parameter_keys
    ):
        for model in results:
            for method in results[model]:
                for key in parameter_keys:
                    results[model][method][key] = Value(None)
    write_csv(results, args.output_dir / "car_ablation_metrics.csv")
    write_latex(results, args.output_dir / "car_ablation_results_table.tex")


if __name__ == "__main__":
    main()
