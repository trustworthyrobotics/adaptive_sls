"""Receding-horizon adaptive-SLS MPC quadrotor rollout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
from jax import config as jax_config

jax_config.update("jax_platform_name", "gpu")
jax_config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gpu_sls.generic_mpc_wrapper import GenericMPCControllerWrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.sqp import SQPConfig
from utils.contraint_utils import combine_constraints

from quadrotor_common import (
    DT,
    MAX_MPC_STEPS,
    MEASUREMENT_MATRIX_C,
    MPC_HORIZON,
    N_AUG,
    N_CONTROL,
    N_PHYS,
    N_THETA,
    NOMINAL_FORCE,
    OBSTACLES,
    TRUE_FORCE,
    U_MAX,
    U_MIN,
    X0_PHYS,
    X_GOAL_PHYS,
    X_MAX_PHYS,
    X_MIN_PHYS,
    augment_reference,
    build_physical_reference,
    cost,
    dynamics,
    initial_augmented_state,
    initial_force_generator,
    make_config,
    make_control_box_constraints,
    make_estimator_update,
    make_exogenous_disturbance,
    make_seed,
    make_state_box_constraints,
    reached_goal,
    save_parameter_plot,
    save_tube_plot,
    save_xy_plot,
    simulate_true_step,
    state_tube_widths,
)

def reference_window(
    global_reference: jnp.ndarray,
    start: int,
    horizon: int,
    force_estimate: jnp.ndarray,
) -> jnp.ndarray:
    end = min(start + horizon + 1, global_reference.shape[0])
    physical = global_reference[start:end]
    if physical.shape[0] < horizon + 1:
        physical = jnp.concatenate(
            [physical, jnp.tile(physical[-1:], (horizon + 1 - physical.shape[0], 1))],
            axis=0,
        )
    return augment_reference(physical, force_estimate)


def run(
    *,
    horizon: int = MPC_HORIZON,
    max_steps: int = MAX_MPC_STEPS,
    output_dir: str | Path = "quadrotor_adaptive_mpc_results",
    enable_linearization_error: bool = False,
) -> Path:
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu":
        raise RuntimeError(f"Quadrotor MPC requires a GPU backend, got {backend}: {devices}")
    print(f"jax_backend: {backend}")
    print(f"devices: {devices}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_max = 20
    cfg = make_config(horizon)
    disturbance = make_exogenous_disturbance()
    estimator_update = make_estimator_update(disturbance, q_max=q_max)
    x = initial_augmented_state()
    G = initial_force_generator(q_max)
    force_estimate = NOMINAL_FORCE

    global_reference = build_physical_reference(X0_PHYS, X_GOAL_PHYS, max_steps)
    initial_reference = reference_window(global_reference, 0, horizon, force_estimate)
    X_seed, U_seed = make_seed(initial_reference)
    constraints = combine_constraints(
        make_state_box_constraints(X_MIN_PHYS, X_MAX_PHYS),
        make_control_box_constraints(U_MIN, U_MAX),
    )
    num_constraints = 2 * N_PHYS + 2 * N_CONTROL + OBSTACLES.shape[0]

    controller = GenericMPCControllerWrapper(
        SLSConfig(
            max_sls_iterations=1,
            sls_primal_tol=1.0e-2,
            enable_fastsls=True,
            warm_start=False,
            n_x=N_PHYS,
            n_w=N_PHYS,
            n_theta=N_THETA,
            num_param=N_THETA,
            q_max=q_max,
            adaptive=True,
            enable_linearization_error=enable_linearization_error,
            enable_disturbance_variation_error=False,
        ),
        SQPConfig(
            max_sqp_iterations=1,
            warm_start=False,
            feas_tol=1.0e-2,
            step_tol=1.0e-4,
            line_search=False,
        ),
        ADMMConfig(
            eps_abs=5.0e-2,
            eps_rel=1.0e-4,
            rho_max=1.0e4,
            max_iterations=400,
            rho_update_frequency=20,
        ),
        config=cfg,
        dynamics=dynamics,
        constraints=constraints,
        obstacles=OBSTACLES,
        cost=cost,
        num_constraints=num_constraints,
        disturbance=disturbance,
        X_in=X_seed,
        U_in=U_seed,
        limited_memory=False,
        shift=1,
        measurement_matrix=MEASUREMENT_MATRIX_C,
    )

    key = jax.random.PRNGKey(1)
    states = [np.asarray(x)]
    controls = []
    disturbances = []
    force_estimates = [np.asarray(force_estimate)]
    force_widths = [np.asarray(jnp.sum(jnp.abs(G), axis=1))]
    one_step_tubes = []
    one_step_deviations = []
    plans = []
    solve_times = []

    for step in range(max_steps):
        if reached_goal(x):
            break

        reference = reference_window(global_reference, step, horizon, force_estimate)
        solve_start = time.perf_counter()
        (
            u,
            X_pred,
            U_pred,
            _,
            backoffs,
            Phi_x,
            _,
            _,
            adaptive_disturbance,
            estimator_error_gains,
            _,
            observable_residual_maps,
        ) = controller.run(
            x0=x,
            G_0=G,
            reference=reference,
            nominal_param=force_estimate,
            parameter=DT,
            X_tail=reference[-1],
        )
        u.block_until_ready()
        solve_times.append(time.perf_counter() - solve_start)

        if not np.all(np.isfinite(np.asarray(X_pred))) or not np.all(np.isfinite(np.asarray(u))):
            raise FloatingPointError(f"The MPC solve returned non-finite values at step {step}.")

        u = jnp.clip(u, U_MIN, U_MAX)
        previous_x = x
        key, x_next, w = simulate_true_step(key, x, u, disturbance)
        x, G, singular_values, innovation, _ = estimator_update(
            previous_x,
            u,
            x_next,
            G,
            previous_x,
            u,
            disturbance(previous_x[None, :])[0],
        )
        force_estimate = x[N_PHYS:]

        tube = state_tube_widths(Phi_x, adaptive_disturbance)[1]
        deviation = jnp.abs(x - X_pred[1])
        states.append(np.asarray(x))
        controls.append(np.asarray(u))
        disturbances.append(np.asarray(w))
        force_estimates.append(np.asarray(force_estimate))
        force_widths.append(np.asarray(jnp.sum(jnp.abs(G), axis=1)))
        one_step_tubes.append(np.asarray(tube))
        one_step_deviations.append(np.asarray(deviation))
        plans.append(np.asarray(X_pred[:, :2]))

        print(
            f"step={step:03d} position={np.asarray(x[:3])} "
            f"force_estimate={np.asarray(force_estimate)} "
            f"width={np.asarray(jnp.sum(jnp.abs(G), axis=1))} "
            f"singular_values(C@E)={np.asarray(singular_values)} "
            f"innovation={np.asarray(innovation)}"
        )

    states_np = np.asarray(states)
    estimates_np = np.asarray(force_estimates)
    widths_np = np.asarray(force_widths)
    tubes_np = np.asarray(one_step_tubes)
    deviations_np = np.asarray(one_step_deviations)

    np.savez(
        output_dir / "quadrotor_adaptive_mpc_rollout.npz",
        states=states_np,
        controls=np.asarray(controls),
        disturbances=np.asarray(disturbances),
        force_estimates=estimates_np,
        force_widths=widths_np,
        true_force=np.asarray(TRUE_FORCE),
        one_step_tubes=tubes_np,
        one_step_deviations=deviations_np,
        plans=np.asarray(plans),
        measurement_matrix_C=np.asarray(MEASUREMENT_MATRIX_C),
        solve_times=np.asarray(solve_times),
        total_solve_time=np.asarray(np.sum(solve_times)),
    )
    save_parameter_plot(
        output_dir / "quadrotor_adaptive_mpc_parameters.png",
        estimates_np,
        widths_np,
        DT,
    )
    save_xy_plot(
        output_dir / "quadrotor_adaptive_mpc_xy.png",
        states_np[None, ...],
        plans[::10],
        title="Adaptive SLS MPC Quadrotor Rollout",
    )
    if deviations_np.size:
        save_tube_plot(
            output_dir / "quadrotor_adaptive_mpc_tube_vs_deviation.png",
            deviations_np[None, ...],
            tubes_np,
            DT,
            title="MPC One-Step Tube vs Executed Deviation",
        )
    print(
        f"Finished {states_np.shape[0] - 1} MPC steps; "
        f"total solve time={np.sum(solve_times):.3f}s."
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=MPC_HORIZON)
    parser.add_argument("--max-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--output-dir", default="quadrotor_adaptive_mpc_results")
    parser.add_argument("--enable-linearization-error", action="store_true")
    args = parser.parse_args()
    run(
        horizon=args.horizon,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        enable_linearization_error=args.enable_linearization_error,
    )


if __name__ == "__main__":
    main()
