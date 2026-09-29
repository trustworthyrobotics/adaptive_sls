"""One-shot full-horizon adaptive-SLS quadrotor rollout."""

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
    CORNER_ROLLOUTS_PER_CORNER,
    DT,
    FORCE_HALF_WIDTH,
    FULL_HORIZON,
    MEASUREMENT_MATRIX_C,
    N_CONTROL,
    N_PHYS,
    N_THETA,
    NUM_RANDOM_ROLLOUTS,
    NOMINAL_FORCE,
    OBSTACLES,
    R_BAR,
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
    make_nonadaptive_config,
    make_nonadaptive_disturbance,
    make_seed,
    make_state_box_constraints,
    nonadaptive_cost,
    nonadaptive_dynamics,
    save_parameter_plot,
    save_all_rollouts_tube_plot,
    save_tube_plot,
    save_xy_plot,
    save_xy_tube_plot,
    save_xz_plot,
    save_z_time_plot,
    simulate_true_step,
    state_feedback_gains,
    state_tube_widths,
    structured_adaptive_state_tube_widths,
)

DEFAULT_INFORMATION_CENTER_Z = 0.7
DEFAULT_DISTURBANCE_Z_OFF = 0.5
DEFAULT_DISTURBANCE_Z_SHARPNESS = 15.0
DEFAULT_EDAGGER_F_COST_WEIGHT = 1000.0
DEFAULT_TERMINAL_POSITION_TOLERANCE = 1.0e-2


def make_terminal_position_constraints(
    goal_position: jnp.ndarray,
    tolerance: float,
):
    """Return paired terminal inequalities for the physical XYZ position."""
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del u, t
        position_error = x[:3] - goal_position
        tolerance_array = jnp.asarray(tolerance, dtype=x.dtype)
        return jnp.concatenate(
            [
                position_error - tolerance_array,
                -position_error - tolerance_array,
            ]
        )

    return constraints


def build_rollout_scenarios(
    *,
    num_random_rollouts: int,
    corner_rollouts_per_corner: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build corner/adversarial and random-parameter rollout metadata."""
    if num_random_rollouts < 0 or corner_rollouts_per_corner < 0:
        raise ValueError("Rollout counts must be nonnegative.")

    corner_signs = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )
    corner_forces = (
        np.asarray(NOMINAL_FORCE)[None, :]
        + corner_signs * np.asarray(FORCE_HALF_WIDTH)[None, :]
    )

    true_forces = []
    adversarial_indices = []
    corner_indices = []
    scenario_types = []
    num_adversarial_directions = 2 * N_PHYS + 2
    for corner_index, corner_force in enumerate(corner_forces):
        for repetition in range(corner_rollouts_per_corner):
            true_forces.append(corner_force)
            adversarial_indices.append(
                (corner_index * corner_rollouts_per_corner + repetition)
                % num_adversarial_directions
            )
            corner_indices.append(corner_index)
            scenario_types.append("corner_adversarial")

    parameter_rng = np.random.default_rng(seed)
    random_forces = (
        np.asarray(NOMINAL_FORCE)[None, :]
        + parameter_rng.uniform(
            -1.0,
            1.0,
            size=(num_random_rollouts, N_THETA),
        ).astype(np.float32)
        * np.asarray(FORCE_HALF_WIDTH)[None, :]
    )
    for random_force in random_forces:
        true_forces.append(random_force)
        adversarial_indices.append(-1)
        corner_indices.append(-1)
        scenario_types.append("random")

    return (
        np.asarray(true_forces, dtype=np.float32).reshape(-1, N_THETA),
        np.asarray(adversarial_indices, dtype=np.int32),
        np.asarray(corner_indices, dtype=np.int32),
        np.asarray(scenario_types),
    )


def run(
    *,
    horizon: int = FULL_HORIZON,
    output_dir: str | Path | None = None,
    enable_linearization_error: bool = False,
    non_adaptive: bool = False,
    num_random_rollouts: int = NUM_RANDOM_ROLLOUTS,
    corner_rollouts_per_corner: int = CORNER_ROLLOUTS_PER_CORNER,
    enable_information_cost: bool = False,
    information_cost_weight: float = 1.0,
    information_cost_discount: float = 1.0,
    enable_edagger_f_cost: bool = False,
    edagger_f_cost_weight: float = DEFAULT_EDAGGER_F_COST_WEIGHT,
    information_center_z: float = DEFAULT_INFORMATION_CENTER_Z,
    disturbance_z_off: float | None = DEFAULT_DISTURBANCE_Z_OFF,
    disturbance_z_sharpness: float = DEFAULT_DISTURBANCE_Z_SHARPNESS,
    enforce_terminal_position: bool = True,
    terminal_position_tolerance: float = DEFAULT_TERMINAL_POSITION_TOLERANCE,
) -> Path:
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu":
        raise RuntimeError(f"Quadrotor SLS requires a GPU backend, got {backend}: {devices}")
    print(f"jax_backend: {backend}")
    print(f"devices: {devices}")

    adaptive = not non_adaptive
    if enable_information_cost and not adaptive:
        raise ValueError("The information cost requires the adaptive formulation.")
    if enable_edagger_f_cost and not adaptive:
        raise ValueError("The E-dagger-F cost requires the adaptive formulation.")
    if enable_information_cost and enable_edagger_f_cost:
        raise ValueError(
            "The posterior-trace and E-dagger-F costs are mutually exclusive."
        )
    if information_cost_weight < 0.0:
        raise ValueError("information_cost_weight must be nonnegative.")
    if not 0.0 < information_cost_discount <= 1.0:
        raise ValueError("information_cost_discount must lie in (0, 1].")
    if edagger_f_cost_weight < 0.0:
        raise ValueError("edagger_f_cost_weight must be nonnegative.")
    if disturbance_z_sharpness <= 0.0:
        raise ValueError("disturbance_z_sharpness must be positive.")
    if terminal_position_tolerance < 0.0:
        raise ValueError("terminal_position_tolerance must be nonnegative.")
    if output_dir is None:
        if adaptive and enable_edagger_f_cost:
            directory_name = "edagger_f_adaptive_sls_quadrotor_results"
        elif adaptive and enable_information_cost:
            directory_name = "active_adaptive_sls_quadrotor_results"
        elif adaptive:
            directory_name = "quadrotor_adaptive_sls_results"
        else:
            directory_name = "quadrotor_non-adaptive_sls_results"
        if enable_linearization_error:
            directory_name += "_with_lin_err"
        output_dir = Path(__file__).resolve().parent / directory_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_max = 20
    cfg = make_config(horizon) if adaptive else make_nonadaptive_config(horizon)
    exogenous_disturbance = make_exogenous_disturbance(
        disturbance_z_off=disturbance_z_off,
        disturbance_z_sharpness=disturbance_z_sharpness,
    )
    disturbance = (
        exogenous_disturbance
        if adaptive
        else make_nonadaptive_disturbance()
    )
    estimator_update = (
        make_estimator_update(exogenous_disturbance, q_max=q_max)
        if adaptive
        else None
    )
    x0 = initial_augmented_state() if adaptive else X0_PHYS
    G_initial = initial_force_generator(q_max)
    reference_phys = build_physical_reference(X0_PHYS, X_GOAL_PHYS, horizon)
    reference = (
        augment_reference(reference_phys, NOMINAL_FORCE)
        if adaptive
        else reference_phys
    )
    X_seed, U_seed = make_seed(reference)

    if enable_edagger_f_cost:
        def cost_fn(W, reference, x, u, t):
            nominal_cost = cost(W, reference, x, u, t)

            def stage_edagger_f(_):
                dynamics_jacobian = jax.jacfwd(dynamics, argnums=0)(
                    x,
                    u,
                    t,
                    parameter=DT,
                )
                E_t = dynamics_jacobian[
                    :N_PHYS,
                    N_PHYS : N_PHYS + N_THETA,
                ]
                F_t = exogenous_disturbance(x[None, :])[0]
                E_dagger_F = jnp.linalg.pinv(E_t, rtol=1.0e-6) @ F_t
                return (
                    jnp.asarray(edagger_f_cost_weight, dtype=x.dtype)
                    * jnp.sum(E_dagger_F * E_dagger_F)
                )

            return nominal_cost + jax.lax.cond(
                t < horizon,
                stage_edagger_f,
                lambda _: jnp.asarray(0.0, dtype=x.dtype),
                operand=None,
            )
    else:
        cost_fn = cost if adaptive else nonadaptive_cost

    print(
        "information_gathering: "
        f"posterior_trace={enable_information_cost}, "
        f"edagger_f={enable_edagger_f_cost}, "
        f"weight={edagger_f_cost_weight}, "
        f"center_z={information_center_z}, "
        f"z_off={disturbance_z_off if disturbance_z_off is not None else information_center_z}, "
        f"z_sharpness={disturbance_z_sharpness}"
    )

    constraints = combine_constraints(
        make_state_box_constraints(X_MIN_PHYS, X_MAX_PHYS),
        make_control_box_constraints(U_MIN, U_MAX),
    )
    num_constraints = 2 * N_PHYS + 2 * N_CONTROL + OBSTACLES.shape[0]
    terminal_constraint_fn = (
        make_terminal_position_constraints(
            X_GOAL_PHYS[:3],
            terminal_position_tolerance,
        )
        if enforce_terminal_position
        else None
    )
    num_terminal_constraints = 6 if enforce_terminal_position else 0

    # Tighten the planned z-tube and make vertical thrust corrections cheaper.
    # The wrapper's default Q_bar is identity and R_BAR is 0.5 * I, so these
    # changes are a 10x increase in Q_bar's z entry and a 10x reduction in the
    # thrust entry of R_bar, respectively.
    q_bar = jnp.eye(cfg.n, dtype=X_seed.dtype).at[2, 2].set(10.0)
    r_bar = R_BAR.at[0, 0].set(R_BAR[0, 0] / 10.0)

    # Match the car experiment's two-pass structure: first use the GPUSLS SQP
    # path with FastSLS disabled to obtain a nominal trajectory, then use that
    # trajectory as the reference and initial iterate for the robust solve.
    # ``cost_fn`` includes E-dagger-F when requested, so active information
    # gathering can shape this first-pass trajectory directly.
    nominal_controller = GenericMPCControllerWrapper(
        SLSConfig(
            max_sls_iterations=2,
            sls_primal_tol=1.0e-2,
            enable_fastsls=False,
            warm_start=False,
            n_x=N_PHYS,
            n_w=N_PHYS,
            n_theta=N_THETA if adaptive else 0,
            num_param=N_THETA,
            q_max=q_max,
            adaptive=adaptive,
            enable_linearization_error=False,
            enable_disturbance_variation_error=False,
            enable_information_cost=enable_information_cost,
            information_cost_weight=information_cost_weight,
            information_cost_discount=information_cost_discount,
        ),
        SQPConfig(
            max_sqp_iterations=20,
            warm_start=False,
            feas_tol=1.0e-1,
            line_search=True,
        ),
        ADMMConfig(
            eps_abs=1.0e-1,
            eps_rel=1.0e-2,
            rho_max=1.0e6,
            max_iterations=100,
            rho_update_frequency=25,
            initial_rho=1.0e2,
        ),
        config=cfg,
        dynamics=dynamics if adaptive else nonadaptive_dynamics,
        constraints=constraints,
        obstacles=OBSTACLES,
        cost=cost_fn,
        num_constraints=num_constraints,
        disturbance=disturbance,
        X_in=X_seed,
        U_in=U_seed,
        Q_bar=q_bar,
        R_bar=r_bar,
        limited_memory=False,
        shift=1,
        measurement_matrix=MEASUREMENT_MATRIX_C,
        terminal_constraints=terminal_constraint_fn,
        num_terminal_constraints=num_terminal_constraints,
    )
    nominal_start = time.perf_counter()
    (
        nominal_u0,
        X_nominal,
        U_nominal,
        *_,
    ) = nominal_controller.run(
        x0=x0,
        G_0=G_initial,
        reference=reference,
        nominal_param=NOMINAL_FORCE,
        parameter=DT,
    )
    nominal_u0.block_until_ready()
    nominal_solve_time = time.perf_counter() - nominal_start
    if not np.all(np.isfinite(np.asarray(X_nominal))):
        raise FloatingPointError("The FastSLS-disabled nominal solve returned non-finite states.")
    if not np.all(np.isfinite(np.asarray(U_nominal))):
        raise FloatingPointError("The FastSLS-disabled nominal solve returned non-finite controls.")

    nominal_prediction = np.asarray(X_nominal)
    X_seed = X_nominal
    U_seed = U_nominal
    reference = X_nominal.at[-1].set(reference[-1])
    print(
        "nominal_reference_solve: "
        f"fastsls=False, solve_time={nominal_solve_time:.3f}s, "
        f"edagger_f={enable_edagger_f_cost}, "
        f"final_position={nominal_prediction[-1, :3]}"
    )

    controller = GenericMPCControllerWrapper(
        SLSConfig(
            max_sls_iterations=2,
            sls_primal_tol=1.0e-2,
            enable_fastsls=True,
            warm_start=False,
            n_x=N_PHYS,
            n_w=N_PHYS,
            n_theta=N_THETA if adaptive else 0,
            num_param=N_THETA,
            q_max=q_max,
            adaptive=adaptive,
            enable_linearization_error=enable_linearization_error,
            # A state-varying F(x) introduces an additional remainder across
            # the state tube. Keep it coupled to the LEB switch so enabling
            # linearization-error protection accounts for both effects.
            enable_disturbance_variation_error=enable_linearization_error,
            enable_information_cost=enable_information_cost,
            information_cost_weight=information_cost_weight,
            information_cost_discount=information_cost_discount,
        ),
        SQPConfig(max_sqp_iterations=20, warm_start=True, feas_tol=1.0e-1, line_search=True),
        ADMMConfig(
            eps_abs=1.0e-1,
            eps_rel=1.0e-2,
            rho_max=1.0e6,
            max_iterations=100,
            rho_update_frequency=25,
            initial_rho=1.0e2,
        ),
        config=cfg,
        dynamics=dynamics if adaptive else nonadaptive_dynamics,
        constraints=constraints,
        obstacles=OBSTACLES,
        cost=cost_fn,
        num_constraints=num_constraints,
        disturbance=disturbance,
        X_in=X_seed,
        U_in=U_seed,
        Q_bar=q_bar,
        R_bar=r_bar,
        limited_memory=False,
        shift=1,
        measurement_matrix=MEASUREMENT_MATRIX_C,
        terminal_constraints=terminal_constraint_fn,
        num_terminal_constraints=num_terminal_constraints,
    )

    solve_start = time.perf_counter()
    (
        u0,
        X_pred,
        U_pred,
        _,
        backoffs,
        Phi_x,
        Phi_u,
        _,
        adaptive_disturbance,
        estimator_error_gains,
        parameter_generators,
        observable_residual_maps,
    ) = controller.run(
        x0=x0,
        G_0=G_initial,
        reference=reference,
        nominal_param=NOMINAL_FORCE,
        parameter=DT,
    )
    u0.block_until_ready()
    solve_time = time.perf_counter() - solve_start

    if not np.all(np.isfinite(np.asarray(X_pred))) or not np.all(np.isfinite(np.asarray(U_pred))):
        mode = "adaptive" if adaptive else "non-adaptive"
        raise FloatingPointError(
            f"The full-horizon {mode} SLS solve returned non-finite values."
        )

    feedback_gains = state_feedback_gains(Phi_x, Phi_u)
    # The solver returns G_t for each transition, with G_0 at index zero and
    # one fewer entry than the state trajectory. Repeat the final generator
    # so the planned parameter-width sequence aligns with T+1 state samples.
    planned_parameter_generators = (
        jnp.concatenate([parameter_generators, parameter_generators[-1:]], axis=0)
        if parameter_generators.shape[0] == horizon
        else parameter_generators[: horizon + 1]
    )
    planned_force_widths = jnp.sum(
        jnp.abs(planned_parameter_generators), axis=-1
    )
    independent_tube = state_tube_widths(Phi_x, adaptive_disturbance)
    if adaptive:
        structured_tube = structured_adaptive_state_tube_widths(
            Phi_x,
            Phi_u,
            adaptive_disturbance,
            estimator_error_gains,
            controller.L_prev,
            observable_residual_maps,
            G_initial,
            X_pred,
            U_pred,
        )
        # The physical response uses independent process/remainder disturbances.
        # Only the adaptive force coordinates require the structured estimator
        # recursion to avoid repeatedly counting correlated delta_G updates.
        tube = independent_tube.at[:, N_PHYS:].set(
            structured_tube[:, N_PHYS:]
        )
    else:
        # This is the original Phi_x E response paired with get_betas; no
        # estimator recursion or adaptive M-matrix tightening is involved.
        tube = independent_tube
    key = jax.random.PRNGKey(0)
    rollout_states = []
    rollout_forces = []
    rollout_widths = []
    rollout_disturbances = []
    rollout_true_forces = []
    rollout_generator_histories = []
    rollout_innovation_gain_histories = []
    scenario_types = []
    corner_indices = []
    adversarial_direction_indices = []

    (
        true_forces,
        adversarial_indices,
        scenario_corner_indices,
        scenario_labels,
    ) = build_rollout_scenarios(
        num_random_rollouts=num_random_rollouts,
        corner_rollouts_per_corner=corner_rollouts_per_corner,
    )

    num_rollouts = len(true_forces)
    print(
        f"rollout_suite: total={num_rollouts} "
        f"corner_adversarial={8 * corner_rollouts_per_corner} "
        f"random={num_random_rollouts}"
    )

    for rollout_index, (
        scenario_true_force,
        adversarial_index,
        corner_index,
        scenario_label,
    ) in enumerate(
        zip(
            true_forces,
            adversarial_indices,
            scenario_corner_indices,
            scenario_labels,
        )
    ):
        adversarial_index_or_none = (
            None if adversarial_index < 0 else int(adversarial_index)
        )
        x = x0
        G = G_initial
        states = [np.asarray(x)]
        force_estimates = [np.asarray(NOMINAL_FORCE)]
        widths = [np.asarray(planned_force_widths[0])]
        generator_history = [np.asarray(G)]
        innovation_gain_history = []
        disturbances = []
        deviation_history = []

        for step in range(horizon):
            delta_x = x - X_pred[step]
            deviation_history.append(delta_x)
            feedback = jnp.zeros((N_CONTROL,), dtype=x.dtype)
            for previous_step in range(step + 1):
                feedback = feedback + feedback_gains[step, previous_step] @ deviation_history[previous_step]
            u = jnp.clip(U_pred[step] + feedback, U_MIN, U_MAX)

            previous_x = x
            simulation_state = (
                x
                if adaptive
                else jnp.concatenate([x, NOMINAL_FORCE])
            )
            key, x_next_aug, w = simulate_true_step(
                key,
                simulation_state,
                u,
                exogenous_disturbance,
                true_force=jnp.asarray(scenario_true_force, dtype=x.dtype),
                adversarial_index=adversarial_index_or_none,
            )
            if adaptive:
                x, G, singular_values, _, L_rollout = estimator_update(
                    previous_x,
                    u,
                    x_next_aug,
                    G,
                    X_pred[step],
                    U_pred[step],
                    adaptive_disturbance[step, :N_PHYS, :2 * N_PHYS],
                )
                innovation_gain_history.append(np.asarray(L_rollout))
                force_estimate = x[N_PHYS:]
            else:
                x = x_next_aug[:N_PHYS]
                singular_values = jnp.zeros((N_THETA,), dtype=x.dtype)
                force_estimate = NOMINAL_FORCE

            if not np.all(np.isfinite(np.asarray(x))):
                raise FloatingPointError(f"Non-finite full-horizon rollout state at step {step}.")
            states.append(np.asarray(x))
            force_estimates.append(np.asarray(force_estimate))
            widths.append(np.asarray(planned_force_widths[step + 1]))
            generator_history.append(np.asarray(G))
            disturbances.append(np.asarray(w))

        rollout_states.append(np.asarray(states))
        rollout_forces.append(np.asarray(force_estimates))
        rollout_widths.append(np.asarray(widths))
        rollout_disturbances.append(np.asarray(disturbances))
        rollout_true_forces.append(np.asarray(scenario_true_force))
        rollout_generator_histories.append(np.asarray(generator_history))
        rollout_innovation_gain_histories.append(np.asarray(innovation_gain_history))
        scenario_types.append(scenario_label)
        corner_indices.append(corner_index)
        adversarial_direction_indices.append(adversarial_index)
        print(
            f"rollout={rollout_index:03d} scenario={scenario_label} "
            f"corner={corner_index} "
            f"adversarial_direction={adversarial_index_or_none} "
            f"true_force={np.asarray(scenario_true_force)} "
            f"final_position={np.asarray(x[:3])} "
            f"force_estimate={np.asarray(force_estimate)} "
            f"singular_values(C@E)={np.asarray(singular_values)}"
        )

    states_np = np.asarray(rollout_states)
    forces_np = np.asarray(rollout_forces)
    widths_np = np.asarray(rollout_widths)
    disturbances_np = np.asarray(rollout_disturbances)
    true_forces_np = np.asarray(rollout_true_forces)
    prediction_np = np.asarray(X_pred)
    deviations_np = np.abs(states_np - prediction_np[None, :, :])

    filename_prefix = (
        "quadrotor_adaptive_sls"
        if adaptive
        else "quadrotor_non-adaptive_sls"
    )
    np.savez(
        output_dir / f"{filename_prefix}_rollout.npz",
        states=states_np,
        controls=np.asarray(U_pred),
        disturbances=disturbances_np,
        force_estimates=forces_np,
        force_widths=widths_np,
        true_force=true_forces_np[0],
        true_forces=true_forces_np,
        scenario_types=np.asarray(scenario_types),
        corner_indices=np.asarray(corner_indices, dtype=np.int32),
        adversarial_direction_indices=np.asarray(
            adversarial_direction_indices,
            dtype=np.int32,
        ),
        prediction=prediction_np,
        backoffs=np.asarray(backoffs),
        state_tubes=np.asarray(tube),
        deviations=deviations_np,
        measurement_matrix_C=np.asarray(MEASUREMENT_MATRIX_C),
        adaptive=np.asarray(adaptive),
        enable_linearization_error=np.asarray(enable_linearization_error),
        enable_information_cost=np.asarray(enable_information_cost),
        enable_edagger_f_cost=np.asarray(enable_edagger_f_cost),
        edagger_f_cost_weight=np.asarray(edagger_f_cost_weight),
        information_center_z=np.asarray(information_center_z),
        disturbance_z_off=np.asarray(
            disturbance_z_off if disturbance_z_off is not None else information_center_z
        ),
        disturbance_z_sharpness=np.asarray(disturbance_z_sharpness),
        enforce_terminal_position=np.asarray(enforce_terminal_position),
        terminal_position_tolerance=np.asarray(terminal_position_tolerance),
        nominal_reference_prediction=nominal_prediction,
        nominal_solve_time=np.asarray(nominal_solve_time),
        solve_time=np.asarray(solve_time),
        parameter_generators=np.asarray(planned_parameter_generators),
        planned_force_widths=np.asarray(planned_force_widths),
        solver_estimator_error_gains=np.asarray(estimator_error_gains),
        solver_estimator_innovation_gains=np.asarray(controller.L_prev),
        rollout_parameter_generators=np.asarray(rollout_generator_histories),
        rollout_innovation_gains=np.asarray(rollout_innovation_gain_histories),
    )
    if adaptive:
        save_parameter_plot(
            output_dir / f"{filename_prefix}_parameters.png",
            forces_np[0],
            np.asarray(planned_force_widths),
            DT,
            true_force=true_forces_np[0],
        )
    save_xy_plot(
        output_dir / f"{filename_prefix}_xy.png",
        states_np,
        [prediction_np[:, :2]],
        title=(
            "Full-Horizon Adaptive SLS Quadrotor Rollouts"
            if adaptive
            else "Full-Horizon Non-Adaptive SLS Quadrotor Rollouts"
        ),
    )
    save_xy_tube_plot(
        output_dir / f"{filename_prefix}_xy_tubes.png",
        states_np,
        prediction_np,
        np.asarray(tube),
        title=(
            "Full-Horizon Adaptive SLS Quadrotor Rollouts with XY Tubes"
            if adaptive
            else "Full-Horizon Non-Adaptive SLS Quadrotor Rollouts with XY Tubes"
        ),
    )
    save_xz_plot(
        output_dir / f"{filename_prefix}_xz.png",
        states_np,
        [prediction_np],
        title=(
            "Full-Horizon Adaptive SLS Quadrotor Altitude"
            if adaptive
            else "Full-Horizon Non-Adaptive SLS Quadrotor Altitude"
        ),
        information_center_z=information_center_z,
    )
    save_z_time_plot(
        output_dir / f"{filename_prefix}_z.png",
        states_np,
        [prediction_np],
        DT,
        title=(
            "Full-Horizon Adaptive SLS Quadrotor Altitude vs Time"
            if adaptive
            else "Full-Horizon Non-Adaptive SLS Quadrotor Altitude vs Time"
        ),
        information_center_z=information_center_z,
        tubes=np.asarray(tube),
        disturbance_z_off=(
            disturbance_z_off
            if disturbance_z_off is not None
            else information_center_z
        ),
        disturbance_z_sharpness=disturbance_z_sharpness,
    )
    save_tube_plot(
        output_dir / f"{filename_prefix}_tube_vs_deviation.png",
        deviations_np,
        np.asarray(tube),
        DT,
        title=(
            "Full-Horizon Adaptive Tube vs Executed Deviation"
            if adaptive
            else "Full-Horizon Non-Adaptive Tube vs Executed Deviation"
        ),
    )
    save_all_rollouts_tube_plot(
        output_dir / f"{filename_prefix}_all_rollouts_tube_vs_deviation.png",
        deviations_np,
        np.asarray(tube),
        DT,
        title=(
            "Full-Horizon Adaptive Tube vs All Executed Rollouts"
            if adaptive
            else "Full-Horizon Non-Adaptive Tube vs All Executed Rollouts"
        ),
    )
    print(f"Finished {num_rollouts} full-horizon rollouts; solve time={solve_time:.3f}s.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FULL_HORIZON)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-random-rollouts", type=int, default=NUM_RANDOM_ROLLOUTS)
    parser.add_argument(
        "--corner-rollouts-per-corner",
        type=int,
        default=CORNER_ROLLOUTS_PER_CORNER,
    )
    parser.add_argument(
        "--enable-linearization-error",
        "--enable-lin-err",
        "--leb",
        dest="enable_linearization_error",
        action="store_true",
    )
    parser.add_argument(
        "--non-adaptive",
        action="store_true",
        help="Use fixed parameter uncertainty and original non-adaptive SLS backoffs.",
    )
    parser.add_argument(
        "--information-cost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the lagged-generator posterior-contraction stage cost.",
    )
    parser.add_argument("--information-cost-weight", type=float, default=1.0)
    parser.add_argument("--information-cost-discount", type=float, default=1.0)
    parser.add_argument(
        "--edagger-f-cost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Penalize the stage-wise squared Frobenius norm of E-dagger F.",
    )
    parser.add_argument(
        "--edagger-f-cost-weight",
        type=float,
        default=DEFAULT_EDAGGER_F_COST_WEIGHT,
    )
    parser.add_argument("--information-center-z", type=float, default=DEFAULT_INFORMATION_CENTER_Z)
    parser.add_argument(
        "--disturbance-z-off",
        type=float,
        default=DEFAULT_DISTURBANCE_Z_OFF,
        help="Midpoint of the tanh disturbance transition (defaults to 0.5 m).",
    )
    parser.add_argument(
        "--disturbance-z-sharpness",
        type=float,
        default=DEFAULT_DISTURBANCE_Z_SHARPNESS,
        help="Positive tanh sharpness in 1/m for the altitude disturbance gate.",
    )
    parser.add_argument(
        "--terminal-position-constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain the terminal nominal XYZ position to the goal.",
    )
    parser.add_argument(
        "--terminal-position-tolerance",
        type=float,
        default=DEFAULT_TERMINAL_POSITION_TOLERANCE,
    )
    args = parser.parse_args()
    if args.information_cost and args.non_adaptive:
        parser.error("--information-cost cannot be used with --non-adaptive")
    if args.edagger_f_cost and args.non_adaptive:
        parser.error("--edagger-f-cost cannot be used with --non-adaptive")
    if args.information_cost and args.edagger_f_cost:
        parser.error("--information-cost and --edagger-f-cost are mutually exclusive")
    if args.information_cost_weight < 0.0:
        parser.error("--information-cost-weight must be nonnegative")
    if not 0.0 < args.information_cost_discount <= 1.0:
        parser.error("--information-cost-discount must lie in (0, 1]")
    if args.edagger_f_cost_weight < 0.0:
        parser.error("--edagger-f-cost-weight must be nonnegative")
    if args.terminal_position_tolerance < 0.0:
        parser.error("--terminal-position-tolerance must be nonnegative")
    run(
        horizon=args.horizon,
        output_dir=args.output_dir,
        enable_linearization_error=args.enable_linearization_error,
        non_adaptive=args.non_adaptive,
        num_random_rollouts=args.num_random_rollouts,
        corner_rollouts_per_corner=args.corner_rollouts_per_corner,
        enable_information_cost=args.information_cost,
        information_cost_weight=args.information_cost_weight,
        information_cost_discount=args.information_cost_discount,
        enable_edagger_f_cost=args.edagger_f_cost,
        edagger_f_cost_weight=args.edagger_f_cost_weight,
        information_center_z=args.information_center_z,
        disturbance_z_off=args.disturbance_z_off,
        disturbance_z_sharpness=args.disturbance_z_sharpness,
        enforce_terminal_position=args.terminal_position_constraint,
        terminal_position_tolerance=args.terminal_position_tolerance,
    )


if __name__ == "__main__":
    main()
