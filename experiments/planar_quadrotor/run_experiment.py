"""Run the four-arm planar-quadrotor receding-horizon comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.planar_quadrotor.config import ExperimentConfig, METHODS, ROOT, U_LOWER, U_UPPER, X0, X_GOAL
    from experiments.planar_quadrotor.controllers import AdaptiveSLSController, CCMController, PavoneController
    from experiments.planar_quadrotor.estimators import GatedGainState, SMEState
    from experiments.planar_quadrotor.model import goal_distance, obstacle_clearance, true_step
    from experiments.planar_quadrotor.scenarios import generate_scenarios
else:
    from .config import ExperimentConfig, METHODS, ROOT, U_LOWER, U_UPPER, X0, X_GOAL
    from .controllers import AdaptiveSLSController, CCMController, PavoneController
    from .estimators import GatedGainState, SMEState
    from .model import goal_distance, obstacle_clearance, true_step
    from .scenarios import generate_scenarios


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _inside_interval(point: np.ndarray, vertices: np.ndarray, tolerance: float = 1e-9) -> bool:
    value = float(np.asarray(point).reshape(-1)[0])
    endpoints = np.asarray(vertices, dtype=float).reshape(-1)
    return bool(endpoints.min() - tolerance <= value <= endpoints.max() + tolerance)


def _inside_zonotope(
    point: np.ndarray,
    center: np.ndarray,
    generators: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    active = generators[:, np.linalg.norm(generators, axis=0) > 1e-14]
    if active.shape[1] == 0:
        return bool(np.linalg.norm(point - center) <= tolerance)
    if active.shape[0] == 1:
        return bool(
            abs(float(np.asarray(point - center).reshape(-1)[0]))
            <= float(np.sum(np.abs(active))) + tolerance
        )
    coefficients = np.linalg.lstsq(active, point - center, rcond=None)[0]
    reconstruction = active @ coefficients
    return bool(
        np.linalg.norm(reconstruction - (point - center)) <= tolerance
        and np.max(np.abs(coefficients)) <= 1.0 + tolerance
    )


def _make_controller(method: str, config: ExperimentConfig, allow_uncertified_ccm: bool):
    if method.startswith("adaptive_sls"):
        return AdaptiveSLSController(config)
    if method == "ccm_rampc":
        return CCMController(config, allow_uncertified=allow_uncertified_ccm)
    if method == "pavone_rampc":
        return PavoneController(config)
    raise ValueError(f"unknown method: {method}")


def _make_estimator(method: str, config: ExperimentConfig):
    if method == "adaptive_sls_gain":
        return GatedGainState.initial(config)
    return SMEState.initial(config)


def run_rollout(
    *,
    method: str,
    run_index: int,
    true_parameter: np.ndarray,
    disturbances: np.ndarray,
    config: ExperimentConfig,
    output_dir: Path,
    allow_uncertified_ccm: bool,
) -> dict[str, Any]:
    controller = _make_controller(method, config, allow_uncertified_ccm)
    estimator = _make_estimator(method, config)
    state = X0.copy()
    states = [state.copy()]
    controls: list[np.ndarray] = []
    estimates = [estimator.center.copy()]
    parameter_widths = [estimator.half_width.copy()]
    parameter_errors = [estimator.center - true_parameter]
    solve_times: list[float] = []
    plant_step_times: list[float] = []
    estimator_update_times: list[float] = []
    forecast_states: list[np.ndarray] = []
    forecast_inputs: list[np.ndarray] = []
    forecast_state_tubes: list[np.ndarray] = []
    forecast_input_tubes: list[np.ndarray] = []
    forecast_parameter_tubes: list[np.ndarray] = []
    statuses: list[str] = []
    plan_successes: list[bool] = []
    diagnostics: list[dict[str, Any]] = []
    clearances = [obstacle_clearance(state, config.obstacles)]
    goal_errors = [goal_distance(state, X_GOAL)]
    contraction_gates: list[np.ndarray] = []
    posterior_gates: list[np.ndarray] = []
    exact_set_areas: list[float] = [estimator.area if isinstance(estimator, SMEState) else np.nan]
    exact_set_contains_true: list[bool] = [
        _inside_interval(true_parameter, estimator.vertices)
        if isinstance(estimator, SMEState)
        else False
    ]
    zonotope_contains_true = [
        _inside_zonotope(true_parameter, estimator.center, estimator.generators)
    ]
    estimator_vertices_json = [
        json.dumps(estimator.vertices.tolist()) if isinstance(estimator, SMEState) else "[]"
    ]
    failed = False
    failure_reason = ""

    for step in range(config.max_steps):
        if goal_errors[-1] <= config.goal_tolerance:
            break
        try:
            plan = controller.solve(state, estimator.center, estimator.generators)
        except Exception as error:  # retain partial artifacts for long Monte-Carlo jobs
            failed = True
            failure_reason = f"{type(error).__name__}: {error}"
            diagnostics.append({"exception": failure_reason, "traceback": traceback.format_exc()})
            break
        statuses.append(plan.status)
        plan_successes.append(plan.success)
        solve_times.append(plan.solve_time_seconds)
        forecast_states.append(plan.states)
        forecast_inputs.append(plan.inputs)
        forecast_state_tubes.append(plan.state_tube_widths)
        forecast_input_tubes.append(plan.input_tube_widths)
        forecast_parameter_tubes.append(plan.parameter_tube_widths)
        diagnostics.append(_jsonable(plan.diagnostics))
        if not np.all(np.isfinite(plan.control)):
            failed = True
            failure_reason = "nonfinite control"
            break
        if not plan.success:
            failed = True
            failure_reason = f"robust MPC plan rejected: {plan.status}"
            diagnostics[-1]["rollout_terminated_on_rejected_plan"] = True
            break

        control = np.clip(plan.control, U_LOWER, U_UPPER)
        plant_start = time.perf_counter()
        next_state = true_step(
            state, control, true_parameter, disturbances[step], config
        )
        plant_step_times.append(time.perf_counter() - plant_start)
        estimator_start = time.perf_counter()
        estimator = estimator.update(state, control, next_state, config)
        estimator_update_times.append(time.perf_counter() - estimator_start)
        if isinstance(estimator, GatedGainState):
            contraction_gates.append(estimator.contraction_gate.copy())
            posterior_gates.append(estimator.posterior_gate.copy())
            exact_set_areas.append(np.nan)
            exact_set_contains_true.append(False)
            estimator_vertices_json.append("[]")
        else:
            exact_set_areas.append(estimator.area)
            exact_set_contains_true.append(
                _inside_interval(true_parameter, estimator.vertices)
            )
            estimator_vertices_json.append(json.dumps(estimator.vertices.tolist()))
        zonotope_contains_true.append(
            _inside_zonotope(true_parameter, estimator.center, estimator.generators)
        )
        state = next_state
        states.append(state.copy())
        controls.append(control.copy())
        estimates.append(estimator.center.copy())
        parameter_widths.append(estimator.half_width.copy())
        parameter_errors.append(estimator.center - true_parameter)
        clearances.append(obstacle_clearance(state, config.obstacles))
        goal_errors.append(goal_distance(state, X_GOAL))

    rollout_dir = output_dir / method
    rollout_dir.mkdir(parents=True, exist_ok=True)
    artifact = rollout_dir / f"run_{run_index:03d}.npz"
    np.savez_compressed(
        artifact,
        method=np.asarray(method),
        run_index=np.asarray(run_index),
        true_inverse_mass_error=true_parameter,
        disturbance_coefficients=disturbances[: len(controls)],
        states=np.asarray(states),
        controls=np.asarray(controls),
        parameter_estimates=np.asarray(estimates),
        parameter_half_widths=np.asarray(parameter_widths),
        parameter_errors=np.asarray(parameter_errors),
        solve_times_seconds=np.asarray(solve_times),
        plant_step_times_seconds=np.asarray(plant_step_times),
        estimator_update_times_seconds=np.asarray(estimator_update_times),
        forecast_states=np.asarray(forecast_states),
        forecast_inputs=np.asarray(forecast_inputs),
        forecast_state_tube_widths=np.asarray(forecast_state_tubes),
        forecast_input_tube_widths=np.asarray(forecast_input_tubes),
        forecast_parameter_tube_widths=np.asarray(forecast_parameter_tubes),
        solver_statuses=np.asarray(statuses),
        plan_successes=np.asarray(plan_successes),
        obstacle_clearances=np.asarray(clearances),
        goal_errors=np.asarray(goal_errors),
        contraction_gates=np.asarray(contraction_gates, dtype=bool).reshape(-1, 1),
        posterior_gates=np.asarray(posterior_gates, dtype=bool).reshape(-1, 1),
        exact_parameter_set_areas=np.asarray(exact_set_areas),
        exact_set_contains_true=np.asarray(exact_set_contains_true),
        zonotope_contains_true=np.asarray(zonotope_contains_true),
        estimator_vertices_json=np.asarray(estimator_vertices_json),
        diagnostics_json=np.asarray([json.dumps(item, sort_keys=True) for item in diagnostics]),
        failed=np.asarray(failed),
        failure_reason=np.asarray(failure_reason),
    )
    per_solve_path = rollout_dir / f"run_{run_index:03d}_per_solve.csv"
    with per_solve_path.open("w", newline="") as handle:
        fields = [
            "step", "solve_time_seconds", "solver_status", "plan_success",
            "plant_step_time_seconds", "estimator_update_time_seconds",
            "realized_parameter_error_norm", "realized_parameter_width",
            "forecast_max_x_tube_width",
            "forecast_max_y_tube_width", "forecast_max_input_tube_width",
            "forecast_terminal_parameter_width", "obstacle_clearance",
            "goal_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(len(solve_times)):
            writer.writerow({
                "step": step,
                "solve_time_seconds": solve_times[step],
                "solver_status": statuses[step],
                "plan_success": plan_successes[step],
                "plant_step_time_seconds": (
                    plant_step_times[step] if step < len(plant_step_times) else np.nan
                ),
                "estimator_update_time_seconds": (
                    estimator_update_times[step]
                    if step < len(estimator_update_times)
                    else np.nan
                ),
                "realized_parameter_error_norm": float(np.linalg.norm(parameter_errors[step])),
                "realized_parameter_width": parameter_widths[step][0],
                "forecast_max_x_tube_width": float(np.max(forecast_state_tubes[step][:, 0])),
                "forecast_max_y_tube_width": float(np.max(forecast_state_tubes[step][:, 1])),
                "forecast_max_input_tube_width": float(np.max(forecast_input_tubes[step])),
                "forecast_terminal_parameter_width": forecast_parameter_tubes[step][-1, 0],
                "obstacle_clearance": clearances[step],
                "goal_error": goal_errors[step],
            })
    summary = {
        "method": method,
        "run_index": run_index,
        "steps": len(controls),
        "reached_goal": bool(goal_errors[-1] <= config.goal_tolerance),
        "failed": failed,
        "failure_reason": failure_reason,
        "final_goal_error": goal_errors[-1],
        "minimum_obstacle_clearance": float(np.min(clearances)),
        "total_solve_time_seconds": float(np.sum(solve_times)),
        "mean_solve_time_seconds": float(np.mean(solve_times)) if solve_times else np.nan,
        "maximum_solve_time_seconds": float(np.max(solve_times)) if solve_times else np.nan,
        "total_plant_step_time_seconds": float(np.sum(plant_step_times)),
        "mean_plant_step_time_seconds": (
            float(np.mean(plant_step_times)) if plant_step_times else np.nan
        ),
        "total_estimator_update_time_seconds": float(
            np.sum(estimator_update_times)
        ),
        "mean_estimator_update_time_seconds": (
            float(np.mean(estimator_update_times))
            if estimator_update_times
            else np.nan
        ),
        "maximum_estimator_update_time_seconds": (
            float(np.max(estimator_update_times))
            if estimator_update_times
            else np.nan
        ),
        "rejected_plan_count": int(np.sum(np.logical_not(plan_successes))),
        "final_parameter_error_norm": float(np.linalg.norm(parameter_errors[-1])),
        "final_parameter_width_sum": float(np.sum(parameter_widths[-1])),
        "exact_set_retained_true_parameter": bool(all(exact_set_contains_true))
        if isinstance(estimator, SMEState)
        else False,
        "zonotope_retained_true_parameter": bool(all(zonotope_contains_true)),
        "artifact": str(artifact),
        "per_solve_csv": str(per_solve_path),
    }
    with (rollout_dir / f"run_{run_index:03d}_summary.json").open("w") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
    return summary


def run_experiment(
    config: ExperimentConfig,
    *,
    methods: tuple[str, ...] = METHODS,
    output_dir: str | Path | None = None,
    allow_uncertified_ccm: bool = False,
    run_start: int = 0,
    run_stop: int | None = None,
    resume: bool = False,
) -> Path:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    run_stop = config.runs if run_stop is None else run_stop
    if not (0 <= run_start <= run_stop <= config.runs):
        raise ValueError(
            f"run range must satisfy 0 <= start <= stop <= {config.runs}; "
            f"got [{run_start}, {run_stop})"
        )
    output_dir = Path(output_dir or (ROOT / "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    serialized_config = config.as_json()
    if resume and config_path.exists():
        with config_path.open() as handle:
            existing_config = json.load(handle)
        if existing_config != serialized_config:
            raise ValueError(
                "cannot resume: existing output config does not match this run"
            )
    with config_path.open("w") as handle:
        json.dump(serialized_config, handle, indent=2, sort_keys=True)
    parameters, disturbances = generate_scenarios(config)
    np.savez_compressed(
        output_dir / "scenarios.npz",
        true_inverse_mass_errors=parameters,
        disturbance_coefficients=disturbances,
        seed=np.asarray(config.seed),
        disturbance_acceleration_half_width=np.asarray(
            config.disturbance_acceleration_half_width
        ),
    )

    summaries: list[dict[str, Any]] = []
    for run_index in range(run_start, run_stop):
        for method in methods:
            existing_summary_path = (
                output_dir / method / f"run_{run_index:03d}_summary.json"
            )
            existing_artifact_path = output_dir / method / f"run_{run_index:03d}.npz"
            if resume and existing_summary_path.exists() and existing_artifact_path.exists():
                with existing_summary_path.open() as handle:
                    existing_summary = json.load(handle)
                artifact_matches = False
                if not existing_summary.get("failed", True):
                    try:
                        with np.load(existing_artifact_path) as artifact:
                            artifact_matches = bool(
                                int(artifact["run_index"]) == run_index
                                and str(artifact["method"]) == method
                                and np.allclose(
                                    artifact["true_inverse_mass_error"],
                                    parameters[run_index],
                                )
                            )
                    except (OSError, KeyError, ValueError):
                        artifact_matches = False
                if artifact_matches:
                    summaries.append(existing_summary)
                    print(
                        f"[{run_index + 1:02d}/{config.runs:02d}] {method} "
                        "already complete; skipping",
                        flush=True,
                    )
                    continue
            print(f"[{run_index + 1:02d}/{config.runs:02d}] {method}", flush=True)
            summary = run_rollout(
                method=method,
                run_index=run_index,
                true_parameter=parameters[run_index],
                disturbances=disturbances[run_index],
                config=config,
                output_dir=output_dir,
                allow_uncertified_ccm=allow_uncertified_ccm,
            )
            summaries.append(summary)
            print(
                f"  steps={summary['steps']} goal={summary['final_goal_error']:.3f} "
                f"clearance={summary['minimum_obstacle_clearance']:.3f} "
                f"failed={summary['failed']}",
                flush=True,
            )

    # Rebuild aggregate files from every completed per-rollout summary. This
    # makes them correct after a resumed or batched run, even if an earlier
    # native-process crash occurred before aggregate output was written.
    summaries = []
    for method in METHODS:
        method_dir = output_dir / method
        if not method_dir.exists():
            continue
        for summary_path in sorted(method_dir.glob("run_*_summary.json")):
            with summary_path.open() as handle:
                summaries.append(json.load(handle))
    summaries.sort(key=lambda item: (int(item["run_index"]), str(item["method"])))
    columns = list(dict.fromkeys(key for item in summaries for key in item))
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(_jsonable(summaries), handle, indent=2, sort_keys=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--obstacle-x", type=float, default=-0.1)
    parser.add_argument("--obstacle-y", type=float, default=1.0)
    parser.add_argument("--obstacle-radius", type=float, default=0.5)
    parser.add_argument("--obstacle-inflation", type=float, default=0.0)
    parser.add_argument("--sls-primal-tolerance", type=float, default=1e-2)
    parser.add_argument("--sqp-feasibility-tolerance", type=float, default=1e-2)
    parser.add_argument("--sls-iterations", type=int, default=1)
    parser.add_argument("--sqp-iterations", type=int, default=1)
    parser.add_argument("--pavone-scp-iterations", type=int, default=1)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--run-stop", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip matching successful per-run artifacts already on disk",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--allow-uncertified-ccm",
        action="store_true",
        help="run the official CCM candidate even if it fails the changed-model sampled audit",
    )
    parser.add_argument(
        "--fail-on-rollout-error",
        action="store_true",
        help="exit nonzero after saving artifacts if any rollout failed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        ExperimentConfig(),
        runs=args.runs,
        seed=args.seed,
        max_steps=args.max_steps,
        obstacle_x=args.obstacle_x,
        obstacle_y=args.obstacle_y,
        obstacle_radius=args.obstacle_radius,
        obstacle_inflation=args.obstacle_inflation,
        rti_sls_primal_tolerance=args.sls_primal_tolerance,
        rti_sqp_feasibility_tolerance=args.sqp_feasibility_tolerance,
        rti_sls_iterations=args.sls_iterations,
        rti_sqp_iterations=args.sqp_iterations,
        pavone_scp_iterations=args.pavone_scp_iterations,
    )
    output = run_experiment(
        config,
        methods=tuple(args.methods),
        output_dir=args.output_dir,
        allow_uncertified_ccm=args.allow_uncertified_ccm,
        run_start=args.run_start,
        run_stop=args.run_stop,
        resume=args.resume,
    )
    print(f"Saved comparison to {output}")
    if args.fail_on_rollout_error:
        with (output / "summary.json").open() as handle:
            summaries = json.load(handle)
        selected_stop = config.runs if args.run_stop is None else args.run_stop
        failures = [
            item
            for item in summaries
            if item.get("failed", False)
            and item.get("method") in args.methods
            and args.run_start <= int(item.get("run_index", -1)) < selected_stop
        ]
        if failures:
            counts: dict[str, int] = {}
            for item in failures:
                method = str(item.get("method", "unknown"))
                counts[method] = counts.get(method, 0) + 1
            raise SystemExit(
                "Comparison saved with failed rollouts: "
                + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            )


if __name__ == "__main__":
    main()
