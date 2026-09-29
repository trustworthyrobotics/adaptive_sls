"""Four-way matched/unmatched Dubins-car SLS ablation.

The experiment runs the same full-horizon pipeline for
non-adaptive/adaptive SLS, each with and without the linearization-error
bound (LEB).  The matched and unmatched cases differ only in where the two
constant parameter errors enter the plant.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from plot_tube_deviations import plot_case


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

source_path = REPO_ROOT / "experiments" / "car" / "car_adaptive_unmatched_v2.py"
module_spec = importlib.util.spec_from_file_location("car_adaptive_unmatched_v2", source_path)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"could not load shared car runner from {source_path}")
v2 = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = v2
module_spec.loader.exec_module(v2)


def matched_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Matched model: parameter errors enter the actuated channels."""
    px, py, heading, speed = x[:4]
    omega_bias, accel_bias = x[4:6]
    omega, accel = u
    return jnp.array(
        [
            px + dt * speed * jnp.cos(heading),
            py + dt * speed * jnp.sin(heading),
            heading + dt * (omega + omega_bias),
            speed + dt * (accel + accel_bias),
            omega_bias,
            accel_bias,
        ],
        dtype=x.dtype,
    )


def matched_dynamics(
    x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any
) -> jnp.ndarray:
    del t
    return matched_step(x, u, parameter)


def matched_nonadaptive_dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    nominal_param: jnp.ndarray,
    *,
    parameter: Any,
) -> jnp.ndarray:
    del t
    return matched_step(jnp.concatenate([x, nominal_param]), u, parameter)[:4]


def matched_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,
    u: jnp.ndarray,
    true_param: jnp.ndarray,
    nominal_param: jnp.ndarray,
    G_0: jnp.ndarray,
    disturbance: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
    i: int,
    adaptive: bool = True,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    del nominal_param, G_0, i
    physical = matched_step(jnp.concatenate([x[:4], true_param]), u, dt)[:4]
    E = disturbance(x[None, :])[0]
    key, _ = jax.random.split(key, 2)
    w = jnp.ones((E.shape[1],), dtype=x.dtype)
    x_next = physical + E @ w
    if adaptive:
        x_next = jnp.concatenate([x_next, x[-2:]])
    return key, x_next, w


def install_matched_model() -> None:
    v2.dubins_step_impl = matched_step
    v2.dubins_step = jax.jit(matched_step)
    v2.dynamics = matched_dynamics
    v2.nonadaptive_dynamics = matched_nonadaptive_dynamics
    v2.dubins_step_with_disturbance = matched_step_with_disturbance


def run_case(
    *, matched: bool, adaptive: bool, leb: bool, output_dir: Path, args: argparse.Namespace
) -> Path:
    if matched:
        install_matched_model()
    tube_csv = v2.main(
        adaptive=adaptive,
        enable_leb=leb,
        output_dir=output_dir,
        horizon=args.horizon,
        time_step_s=args.dt,
        exogenous_disturbance_scale=args.disturbance_scale,
        parameter_uncertainty_bound=args.parameter_bound,
        sls_q_bar_scale=10.0,
        sls_r_bar_scale=1.0,
        model_label=("Matched" if matched else "Unmatched") + " Car Ablation",
        state_weights=(0.5, 0.5, 0.1, 0.1),
        control_weights=(1.0, 10.0),
        terminal_state_weights=(10.0, 10.0, 0.0, 0.0),
        deterministic_nominal_solve=True,
        shared_physical_nominal_solve=True,
        track_nominal_during_robust_solve=True,
        start_x_m=args.start_x,
        start_y_m=args.start_y,
        goal_x_m=args.goal_x,
        goal_y_m=args.goal_y,
        initial_speed_m_per_s=args.initial_speed,
        goal_speed_m_per_s=args.goal_speed,
        minimum_speed_m_per_s=args.minimum_speed,
        maximum_speed_m_per_s=args.maximum_speed,
        x_min_m=args.x_min,
        x_max_m=args.x_max,
        obstacle=(args.obstacle_x, args.obstacle_y, args.obstacle_radius),
    )
    # Keep the all-rollout diagnostic synchronized with this newly generated
    # controller archive; it initially contains the runner's internal rollouts.
    plot_case(tube_csv.parent, tube_csv.parent / "all_rollouts_deviation_vs_tube_width.png")
    return tube_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--horizon", type=int, default=70)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--start-x", type=float, default=1.0)
    parser.add_argument("--start-y", type=float, default=1.0)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=-1.0)
    parser.add_argument("--obstacle-x", type=float, default=0.95)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-radius", type=float, default=0.35)
    parser.add_argument("--initial-speed", type=float, default=0.0)
    parser.add_argument("--goal-speed", type=float, default=0.60)
    parser.add_argument("--minimum-speed", type=float, default=-1.0)
    parser.add_argument("--maximum-speed", type=float, default=3.0)
    parser.add_argument("--parameter-bound", type=float, default=0.075)
    parser.add_argument("--disturbance-scale", type=float, default=0.00030)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=3.5)
    args = parser.parse_args()
    if args.horizon < 1 or args.dt <= 0.0:
        parser.error("--horizon must be positive and --dt must be positive")

    for matched in (False, True):
        model_dir = args.output_dir / ("matched" if matched else "unmatched")
        paths = [
            run_case(
                matched=matched,
                adaptive=adaptive,
                leb=leb,
                output_dir=model_dir,
                args=args,
            )
            for adaptive in (False, True)
            for leb in (False, True)
        ]
        combined = model_dir / "tube_widths_all_combinations.csv"
        with combined.open("w", newline="") as target:
            writer = None
            for path in paths:
                with path.open(newline="") as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
        manifest = {
            "schema_version": 1,
            "model": "matched" if matched else "unmatched",
            "horizon": args.horizon,
            "dt": args.dt,
            "start": [args.start_x, args.start_y],
            "goal": [args.goal_x, args.goal_y],
            "obstacle": [args.obstacle_x, args.obstacle_y, args.obstacle_radius],
            "initial_speed": args.initial_speed,
            "speed_limits": [args.minimum_speed, args.maximum_speed],
            "controllers": {
                f"adaptive_{adaptive}__leb_{leb}": str(
                    (model_dir / f"adaptive_{adaptive}__leb_{leb}" / "controller.npz").resolve()
                )
                for adaptive in (False, True)
                for leb in (False, True)
            },
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {model_dir} ablation results and {combined}")


if __name__ == "__main__":
    main()
