# Adapted from https://github.com/iit-DLSLab/mpx/blob/main/mpx/examples/mjx_quad.py

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable

dir_path = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.join(dir_path, "..", ".."))
vendor_root = os.path.join(dir_path, "vendor")
for path in (os.path.join(repo_root, "src"), os.path.join(repo_root, "src", "quad_mpc")):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
if vendor_root in sys.path:
    sys.path.remove(vendor_root)
sys.path.insert(0, vendor_root)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
from matplotlib.patches import Rectangle
import mpx
import mpx.utils.sim as sim_utils

import config_go2_nonadaptive as config
import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.sqp import SQPConfig

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def make_stage_dependent_disturbance(
    n_x: int,
    *,
    base_z_disturbance_enabled: bool,
    first_base_position_alpha: float,
    first_thigh_position_alpha: float,
    first_calf_position_alpha: float,
    later_base_position_alpha: float,
    later_thigh_position_alpha: float,
    later_calf_position_alpha: float,
) -> Callable:
    def position_diagonal(
        dtype,
        *,
        base_alpha,
        thigh_alpha,
        calf_alpha,
    ):
        diagonal = jnp.zeros(n_x, dtype=dtype)
        diagonal = diagonal.at[:2].set(base_alpha)
        if base_z_disturbance_enabled:
            diagonal = diagonal.at[2].set(base_alpha)
        joint_start = 7
        joint_stop = joint_start + config.n_joints
        diagonal = diagonal.at[joint_start + 1 : joint_stop:3].set(
            thigh_alpha
        )
        diagonal = diagonal.at[joint_start + 2 : joint_stop:3].set(
            calf_alpha
        )
        return diagonal

    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        first_diagonal = position_diagonal(
            X_prefix.dtype,
            base_alpha=first_base_position_alpha,
            thigh_alpha=first_thigh_position_alpha,
            calf_alpha=first_calf_position_alpha,
        )
        later_diagonal = position_diagonal(
            X_prefix.dtype,
            base_alpha=later_base_position_alpha,
            thigh_alpha=later_thigh_position_alpha,
            calf_alpha=later_calf_position_alpha,
        )
        first_E = jnp.diag(first_diagonal)
        later_E = jnp.diag(later_diagonal)
        horizon_E = jnp.broadcast_to(
            later_E,
            (X_prefix.shape[0], n_x, n_x),
        )
        first_E = first_E.at[19:22, 19:22].set(0.000 * jnp.eye(3, dtype=first_E.dtype))
        later_E = later_E.at[19:22, 19:22].set(0.000 * jnp.eye(3, dtype=later_E.dtype))
        return (
            horizon_E
            if X_prefix.shape[0] == 0
            else horizon_E.at[0].set(first_E)
        )

    return disturbance


def make_nonadaptive_disturbance(exogenous_disturbance, dynamics):
    """Represent fixed damping uncertainty without damping states."""

    def padded_controls(X, U):
        pad_length = X.shape[0] - U.shape[0]
        return (
            jnp.pad(U, ((0, pad_length), (0, 0)))
            if pad_length > 0
            else U
        )

    def sensitivity(X, U, nominal_param, dynamics_parameter):
        U_padded = padded_controls(X, U)
        time_indices = jnp.arange(X.shape[0])

        def stage_sensitivity(x, u, t):
            return jax.jacrev(dynamics, argnums=3)(
                x,
                u,
                t,
                nominal_param,
                parameter=dynamics_parameter,
            )

        return jax.vmap(stage_sensitivity)(X, U_padded, time_indices)

    def parametric_disturbance(
        X,
        U,
        nominal_param,
        G_0,
        dynamics_parameter,
    ):
        parameter_map = sensitivity(X, U, nominal_param, dynamics_parameter)
        return jnp.concatenate(
            [exogenous_disturbance(X), parameter_map @ G_0],
            axis=-1,
        )

    # The fourth entry selects the legged interface that also supplies the
    # gait/contact parameter used by the dynamics at every horizon stage.
    return parametric_disturbance, sensitivity, exogenous_disturbance, True


@jax.jit
def contact_gated_transition_matrix(
    base_measurement_matrix: jnp.ndarray,
    endpoint_contacts: jnp.ndarray,
) -> jnp.ndarray:
    """Build the transition channel matrix as a sum of per-leg channels.

    A leg contributes its thigh and calf joint-position rows only when it is
    in swing at both endpoints. Hip rows never contribute. Non-joint rows
    retain the static channels supplied by ``base_measurement_matrix``.
    """
    base_measurement_matrix = jnp.asarray(base_measurement_matrix)
    endpoint_contacts = jnp.asarray(endpoint_contacts)
    expected_matrix_shape = (config.n_phys, config.n_phys)
    expected_contact_shape = (2, config.n_contact)
    if base_measurement_matrix.shape != expected_matrix_shape:
        raise ValueError(
            "base_measurement_matrix must have shape "
            f"{expected_matrix_shape}; got {base_measurement_matrix.shape}"
        )
    if endpoint_contacts.shape != expected_contact_shape:
        raise ValueError(
            f"endpoint_contacts must have shape {expected_contact_shape}; "
            f"got {endpoint_contacts.shape}"
        )

    joint_start = 7
    joint_stop = joint_start + config.n_joints
    transition_matrix = base_measurement_matrix.at[
        joint_start:joint_stop, :
    ].set(0.0)
    swing_at_both_endpoints = jnp.prod(
        1.0 - endpoint_contacts.astype(base_measurement_matrix.dtype),
        axis=0,
    )
    for leg_index, leg in enumerate(config.contact_frame):
        thigh_calf_rows = joint_start + config.LEG_JOINT_INDEX[leg][1:]
        leg_matrix = jnp.zeros_like(base_measurement_matrix)
        leg_matrix = leg_matrix.at[thigh_calf_rows, :].set(
            base_measurement_matrix[thigh_calf_rows, :]
        )
        transition_matrix = (
            transition_matrix
            + swing_at_both_endpoints[leg_index] * leg_matrix
        )
    return transition_matrix


def augmented_state_labels() -> list[str]:
    """Labels matching the 61 physical non-adaptive states."""
    labels = [
        "base_pos_x",
        "base_pos_y",
        "base_pos_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]
    joint_names = [
        f"{leg}_{joint}"
        for leg in config.contact_frame
        for joint in ("hip", "thigh", "calf")
    ]
    labels.extend(f"q_{name}" for name in joint_names)
    labels.extend(("base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"))
    labels.extend(("base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"))
    labels.extend(f"dq_{name}" for name in joint_names)
    labels.extend(
        f"foot_{leg}_{axis}" for leg in config.contact_frame for axis in "xyz"
    )
    labels.extend(
        f"grf_{leg}_{axis}" for leg in config.contact_frame for axis in "xyz"
    )
    if len(labels) != config.n:
        raise ValueError(
            f"Constructed {len(labels)} labels for {config.n} augmented states."
        )
    return labels


@partial(jax.jit, static_argnames=("q_max",))
def girard_reduce(G: jnp.ndarray, q_max: int) -> jnp.ndarray:
    """Equation (17): retain the largest generators and boxify the rest."""
    n_theta = G.shape[0]
    keep_count = n_theta * (q_max - 1)
    target_count = n_theta * q_max
    order = jnp.argsort(jnp.linalg.norm(G, axis=0))[::-1]
    sorted_G = G[:, order]
    kept = sorted_G[:, :keep_count]
    tail_box = jnp.diag(jnp.sum(jnp.abs(sorted_G[:, keep_count:]), axis=1))
    reduced = jnp.concatenate([kept, tail_box], axis=1)
    return jnp.pad(reduced, ((0, 0), (0, target_count - reduced.shape[1])))


@partial(
    jax.jit,
    static_argnames=(
        "dynamics",
        "disturbance",
        "q_max",
        "enable_posterior_gate",
    ),
)
def estimate_damping(
    dynamics: Callable,
    disturbance: Callable,
    previous_x: jnp.ndarray,
    x_next_phys: jnp.ndarray,
    u: jnp.ndarray,
    parameter: jnp.ndarray,
    G_0: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    q_max: int,
    enable_posterior_gate: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Observable-subspace update on the disturbance-supported state rows."""
    theta = previous_x[config.n_phys :]

    def physical_step(candidate_theta):
        candidate_x = previous_x.at[config.n_phys :].set(candidate_theta)
        return dynamics(candidate_x, u, 0, parameter=parameter)[: config.n_phys]

    prediction = physical_step(theta)
    E_t = jax.jacfwd(physical_step)(theta)
    measured_E_t = measurement_matrix @ E_t
    U_t, singular_values, Vh_t = jnp.linalg.svd(
        measured_E_t, full_matrices=False
    )
    tolerance = jnp.maximum(1.0e-7, 1.0e-2 * jnp.max(singular_values))
    rank = jnp.sum(singular_values >= tolerance)

    inverse_singular_values = jnp.where(
        singular_values >= tolerance,
        1.0 / singular_values,
        0.0,
    )
    measured_E_pinv = (Vh_t.T * inverse_singular_values) @ U_t.T
    residual_map = measured_E_pinv @ measurement_matrix
    Pi_t = residual_map @ E_t
    W_t = residual_map @ disturbance(previous_x[None, :])[0]
    covariance = G_0 @ G_0.T
    regularization = 1.0e-7 * jnp.eye(config.n_theta, dtype=G_0.dtype)
    K_t = covariance @ Pi_t.T @ jnp.linalg.inv(
        Pi_t @ covariance @ Pi_t.T + W_t @ W_t.T + regularization
    )

    scaled_G = (K_t @ Pi_t) @ G_0
    contraction_gate = (
        jnp.abs(G_0 - scaled_G).sum(axis=1)
        < jnp.abs(G_0).sum(axis=1)
    )
    K_t = jnp.diag(contraction_gate.astype(G_0.dtype)) @ K_t

    scaled_G = (K_t @ Pi_t) @ G_0
    scaled_innovation = K_t @ W_t
    G_pre = jnp.concatenate(
        [G_0 - scaled_G, scaled_innovation],
        axis=1,
    )
    if enable_posterior_gate:
        posterior_gate = (
            jnp.abs(G_pre).sum(axis=1)
            < jnp.abs(G_0).sum(axis=1)
        )
        K_t = jnp.diag(posterior_gate.astype(G_0.dtype)) @ K_t

    innovation = residual_map @ (x_next_phys - prediction)
    theta_next = jnp.maximum(theta + K_t @ innovation, 0.0)
    G_pre = jnp.concatenate(
        [
            (jnp.eye(config.n_theta, dtype=G_0.dtype) - K_t @ Pi_t) @ G_0,
            K_t @ W_t,
        ],
        axis=1,
    )
    return theta_next, girard_reduce(G_pre, q_max=q_max), rank, singular_values, innovation


def save_diagnostic_plots(
    *,
    states,
    theta_estimates,
    theta_widths,
    true_theta,
    planned_xy,
    one_step_plans,
    one_step_tubes,
    horizon_tubes,
    obstacle_backoffs,
    obstacle_center,
    obstacle_radius,
    admm_tolerance,
    sample_dt,
):
    time_axis = np.arange(theta_estimates.shape[0]) * sample_dt
    labels = tuple(label.replace("_", " ") for label in config.THETA_LABELS)
    fig, axes = plt.subplots(config.n_theta, 1, figsize=(9, 14), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(time_axis, theta_estimates[:, index], label="estimate")
        axis.fill_between(
            time_axis,
            theta_estimates[:, index] - theta_widths[:, index],
            theta_estimates[:, index] + theta_widths[:, index],
            alpha=0.18,
            label="uncertainty",
        )
        axis.axhline(true_theta[index], color="tab:red", linestyle="--", label="true")
        axis.set_ylabel(labels[index])
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "quadruped_nonadaptive_damping_parameters.png"), dpi=250)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(states[:, 0], states[:, 1], marker="o", markersize=2, label="executed")
    for index, plan in enumerate(planned_xy[::10]):
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            alpha=0.25,
            linestyle="--",
            label="MPC plans" if index == 0 else None,
        )
    axis.add_patch(
        plt.Circle(obstacle_center, obstacle_radius, fill=False, color="tab:red", label="obstacle")
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Non-adaptive damping MuJoCo rollout")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "quadruped_nonadaptive_damping_xy.png"), dpi=250)
    plt.close(fig)

    if one_step_tubes is None or one_step_plans is None:
        return

    one_step_tubes = np.array(one_step_tubes, copy=True)

    deviations = np.abs(states[1:] - one_step_plans)
    diagnostic_indices = (0, 1, 2)
    diagnostic_labels = ("base x", "base y", "base z")
    tube_time = np.arange(one_step_tubes.shape[0]) * sample_dt
    fig, axes = plt.subplots(
        len(diagnostic_indices),
        1,
        figsize=(9, 2.0 * len(diagnostic_indices)),
        sharex=True,
    )
    for axis, state_index, label in zip(axes, diagnostic_indices, diagnostic_labels):
        axis.plot(
            tube_time,
            deviations[:, state_index],
            label=f"|{label} deviation|",
        )
        axis.plot(tube_time, one_step_tubes[:, state_index], linestyle="--", label="tube")
        axis.fill_between(tube_time, 0.0, one_step_tubes[:, state_index], alpha=0.12)
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "quadruped_nonadaptive_damping_tube_vs_deviation.png"), dpi=250)
    plt.close(fig)

    all_labels = augmented_state_labels()
    ncols = 4
    nrows = int(np.ceil(config.n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.15 * nrows),
        sharex=True,
        squeeze=False,
    )
    for state_index, axis in enumerate(axes.flat):
        if state_index >= config.n:
            axis.set_visible(False)
            continue
        axis.plot(
            tube_time,
            deviations[:, state_index],
            color="tab:blue",
            linewidth=1.1,
        )
        axis.plot(
            tube_time,
            one_step_tubes[:, state_index],
            color="tab:orange",
            linestyle="--",
            linewidth=1.1,
        )
        axis.fill_between(
            tube_time,
            0.0,
            one_step_tubes[:, state_index],
            color="tab:orange",
            alpha=0.12,
        )
        axis.set_title(f"{state_index}: {all_labels[state_index]}", fontsize=8)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=7)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("time (s)")
    fig.suptitle("Non-adaptive SLS One-Step Tubes for All Physical States", fontsize=14)
    fig.legend(
        ["absolute executed deviation", "one-step tube"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.998),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(
        os.path.join(dir_path, "quadruped_nonadaptive_damping_all_state_tubes.png"),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(states[:, 0], states[:, 1], label="executed")
    axis.plot(one_step_plans[:, 0], one_step_plans[:, 1], linestyle="--", label="one-step plan")
    forecast_indices = range(
        0,
        min(len(planned_xy), len(horizon_tubes), len(obstacle_backoffs)),
        10,
    )
    for forecast_plot_index, forecast_index in enumerate(forecast_indices):
        plan = planned_xy[forecast_index]
        tubes = horizon_tubes[forecast_index]
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            alpha=0.2,
            linewidth=0.8,
            label="projected horizons" if forecast_plot_index == 0 else None,
        )
        horizon_count = min(plan.shape[0] - 1, tubes.shape[0])
        for horizon_index in range(horizon_count):
            center = plan[horizon_index + 1]
            tube = tubes[horizon_index]
            axis.add_patch(
                Rectangle(
                    (center[0] - tube[0], center[1] - tube[1]),
                    2.0 * tube[0],
                    2.0 * tube[1],
                    facecolor="tab:green",
                    edgecolor="tab:green",
                    linewidth=0.35,
                    alpha=0.06,
                    label=(
                        "horizon XY tubes"
                        if forecast_plot_index == 0 and horizon_index == 0
                        else None
                    ),
                )
            )
    axis.add_patch(
        plt.Circle(obstacle_center, obstacle_radius, fill=False, color="tab:red", label="obstacle")
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Steps 1-{config.N} projected XY tubes")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "quadruped_nonadaptive_damping_xy_tubes.png"), dpi=250)
    plt.close(fig)

    # Visualize the exact scalar backoff used to tighten the circular obstacle
    # constraint. It is evaluated on the SQP obstacle-linearization trajectory
    # and returned with the MPC output rather than reconstructed from the
    # post-step projected trajectory.
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(states[:, 0], states[:, 1], label="executed")
    axis.plot(
        one_step_plans[:, 0],
        one_step_plans[:, 1],
        linestyle="--",
        label="one-step plan",
    )
    for forecast_plot_index, forecast_index in enumerate(forecast_indices):
        plan = planned_xy[forecast_index]
        tubes = horizon_tubes[forecast_index]
        forecast_obstacle_backoffs = obstacle_backoffs[forecast_index]
        axis.plot(
            plan[:, 0],
            plan[:, 1],
            color="tab:orange",
            alpha=0.2,
            linewidth=0.8,
            label="projected horizons" if forecast_plot_index == 0 else None,
        )
        horizon_count = min(plan.shape[0] - 1, tubes.shape[0])
        for horizon_index in range(horizon_count):
            center = np.asarray(plan[horizon_index + 1])
            obstacle_backoff = forecast_obstacle_backoffs[horizon_index, 0]
            axis.add_patch(
                plt.Circle(
                    center,
                    float(obstacle_backoff),
                    facecolor="tab:purple",
                    edgecolor="tab:purple",
                    linewidth=0.35,
                    alpha=0.06,
                    label=(
                        "obstacle-backoff circles"
                        if forecast_plot_index == 0 and horizon_index == 0
                        else None
                    ),
                )
            )
    deflated_obstacle_radius = max(
        float(obstacle_radius) - float(admm_tolerance),
        0.0,
    )
    axis.add_patch(
        plt.Circle(
            obstacle_center,
            deflated_obstacle_radius,
            fill=False,
            color="tab:red",
            label="obstacle radius - ADMM tolerance",
        )
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        f"Steps 1-{config.N} obstacle-backoff circles "
        f"(obstacle radius - {float(admm_tolerance):g})"
    )
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            dir_path,
            "quadruped_nonadaptive_damping_xy_obstacle_backoff_circles.png",
        ),
        dpi=250,
    )
    plt.close(fig)


def main(
    headless=False,
    steps=500,
    seed=None,
    log_every=10,
    save_plots=True,
    enable_posterior_gate=True,
    fuse_rti_updates=True,
    randomize_damping=False,
    base_z_disturbance=False,
    output_dir=None,
    initial_theta_half_width=None,
    admm_max_iterations=1000,
    rho_update_frequency=25,
):
    if steps < 1:
        raise ValueError("steps must be positive")
    if log_every < 0:
        raise ValueError("log_every must be nonnegative")
    if initial_theta_half_width is not None and initial_theta_half_width <= 0.0:
        raise ValueError("initial_theta_half_width must be positive")
    if admm_max_iterations < 1:
        raise ValueError("admm_max_iterations must be positive")
    if rho_update_frequency < 1:
        raise ValueError("rho_update_frequency must be positive")
    result_dir = os.path.abspath(output_dir or dir_path)
    os.makedirs(result_dir, exist_ok=True)
    theta_half_width = (
        config.initial_theta_half_width.astype(config.q0.dtype)
        if initial_theta_half_width is None
        else jnp.full(
            (config.n_theta,), initial_theta_half_width, dtype=config.q0.dtype
        )
    )

    admm_tol = 0.1
    obstacle_center = jnp.array([1.0, 0.25], dtype=config.q0.dtype)
    physical_obstacle_radius = 0.15
    robot_footprint_radius = 0.28
    obstacle_radius = physical_obstacle_radius + robot_footprint_radius + admm_tol

    def outside_circle_constraint(x, u, t):
        del u, t
        distance = jnp.linalg.norm(x[:2] - obstacle_center)
        return jnp.array([obstacle_radius - distance], dtype=x.dtype)

    # Restore the broad state/control boxes used by the earlier v2 rollout.
    # Besides enforcing the actuator limit, these constraints make the SLS
    # state backoffs available for one-step tube diagnostics.
    state_bound = jnp.full((config.n,), 1.0e6, dtype=config.q0.dtype)
    state_bound = state_bound.at[:3].set(
        jnp.array([20.0, 20.0, 2.0], dtype=config.q0.dtype)
    )
    state_bound = state_bound.at[config.n_phys :].set(100.0)
    control_bound = jnp.full(
        (config.m,), config.max_torque, dtype=config.q0.dtype
    )

    def constraints(x, u, t):
        return jnp.concatenate(
            [
                x - state_bound,
                -state_bound - x,
                u - control_bound,
                -control_bound - u,
                outside_circle_constraint(x, u, t),
            ]
        )

    # Use the larger model-mismatch allowance on the immediately executed
    # transition, then reduce it at every later predicted horizon stage.
    exogenous_disturbance = make_stage_dependent_disturbance(
        config.n_phys,
        base_z_disturbance_enabled=base_z_disturbance,
        # Immediately executed transition
        first_base_position_alpha=0.002 * 2.0,
        first_thigh_position_alpha=0.010 * 2.0,
        first_calf_position_alpha=0.0125 * 2.0,
        # Later MPC horizon transitions
        later_base_position_alpha=0.002 * 2.0,
        later_thigh_position_alpha=0.010 * 2.0,
        later_calf_position_alpha=0.0125 * 2.0,
    )
    # Temporary diagnostic: retain the full E_w in planning/estimation but do
    # not inject a sampled exogenous disturbance into the executed plant.
    exogenous_rollout_scale = jnp.asarray(0.0, dtype=config.q0.dtype)
    disturbance_probe = exogenous_disturbance(
        jnp.zeros((1, config.n_phys), dtype=config.q0.dtype)
    )[0]
    measurement_mask = jnp.any(jnp.abs(disturbance_probe) > 0.0, axis=1).astype(config.q0.dtype)
    # Keep base-velocity uncertainty in E for MPC tube propagation, but do not
    # expose vx, vy, or vz as measurement channels to the parameter estimator.
    measurement_mask = measurement_mask.at[19:22].set(0.0)
    measurement_matrix = jnp.diag(measurement_mask)
    # Hip joint-position rows are zero because hip disturbance was removed.
    # The shared MPC wrapper gates the remaining joint rows at each contact
    # endpoint; GPU-SLS multiplies adjacent endpoint matrices so a thigh/calf
    # row is active only when its leg is in swing before and after a step.
    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=True,
        enable_linearization_error=False,
        enable_disturbance_variation_error=False,
        warm_start=True,
        n_x=config.n_phys,
        n_w=config.n_phys,
        n_theta=0,
        num_param=config.n_theta,
        q_max=3,
        adaptive=False,
        enable_posterior_gate=enable_posterior_gate,
    )
    admm_config = ADMMConfig(
        max_iterations=admm_max_iterations,
        rho_update_frequency=rho_update_frequency,
        eps_abs=admm_tol,
        eps_rel=1e-4,
    )

    @jax.jit
    def tightened_constraint_values(X, U, backoffs):
        """Evaluate the returned forecast against the current tightened constraints."""
        terminal_u = jnp.zeros((1, U.shape[1]), dtype=U.dtype)
        U_padded = jnp.concatenate([U, terminal_u], axis=0)
        time_indices = jnp.arange(X.shape[0])
        nominal_values = jax.vmap(constraints)(X, U_padded, time_indices)
        # gpu_admm.constrained_solve currently uses ``f = f``; eps_abs is a
        # residual stopping tolerance, not an additional constraint margin.
        tightened_values = nominal_values + jnp.abs(backoffs)
        return nominal_values, tightened_values

    state_names = augmented_state_labels()

    def constraint_name(constraint_index: int) -> str:
        if constraint_index < config.n:
            return f"state_upper[{state_names[constraint_index]}]"
        if constraint_index < 2 * config.n:
            state_index = constraint_index - config.n
            return f"state_lower[{state_names[state_index]}]"
        if constraint_index < 2 * config.n + config.m:
            return f"control_upper[joint_{constraint_index - 2 * config.n}]"
        if constraint_index < 2 * config.n + 2 * config.m:
            joint_index = constraint_index - 2 * config.n - config.m
            return f"control_lower[joint_{joint_index}]"
        if constraint_index == 2 * config.n + 2 * config.m:
            return "obstacle"
        return "unknown"

    mpx_root = Path(mpx.__file__).parent
    model = mujoco.MjModel.from_xml_path(
        str(mpx_root / "data" / "go2" / "scene_mjx.xml")
    )
    data = mujoco.MjData(model)
    planning_dynamics = config.dynamics(
        model,
        mjx.put_model(model),
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ],
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ],
    )
    disturbance = make_nonadaptive_disturbance(
        exogenous_disturbance,
        planning_dynamics,
    )

    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=sls_config,
        sqp_config=SQPConfig(),
        admm_config=admm_config,
        constraints=constraints,
        num_constraints=2 * config.n + 2 * config.m + 1,
        disturbance=disturbance,
        measurement_matrix=measurement_matrix,
        contact_gated_joint_measurements=True,
        fuse_rti_updates=fuse_rti_updates,
    )

    # Execute the same discrete interval assumed by the SLS dynamics in one
    # MuJoCo step.  One torque, disturbance sample, and estimator measurement
    # therefore correspond to one complete 0.02 s transition.
    sim_frequency = float(config.mpc_frequency)
    model.opt.timestep = 1.0 / sim_frequency
    controller_period = int(sim_frequency / config.mpc_frequency)
    if controller_period != 1:
        raise ValueError(
            "The MuJoCo plant and SLS controller must execute one shared 0.02 s transition; "
            f"computed period={controller_period}."
        )

    data.qpos[:] = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom_to_index = {
        int(geom_id): index for index, geom_id in enumerate(contact_ids)
    }
    contact_force_threshold = 1.0e-3

    def foot_normal_grfs() -> np.ndarray:
        """Sum each foot's normal contact-force magnitude against the floor."""
        normal_forces = np.zeros(config.n_contact, dtype=float)
        contact_force = np.zeros(6, dtype=float)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == floor_geom_id:
                foot_index = foot_geom_to_index.get(geom2)
            elif geom2 == floor_geom_id:
                foot_index = foot_geom_to_index.get(geom1)
            else:
                continue
            if foot_index is None:
                continue
            contact_force.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            normal_forces[foot_index] += abs(float(contact_force[0]))
        return normal_forces

    def measured_physical_state() -> jnp.ndarray:
        foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids), dtype=config.q0.dtype)
        return jnp.concatenate(
            [
                jnp.asarray(data.qpos, dtype=config.q0.dtype),
                jnp.asarray(data.qvel, dtype=config.q0.dtype),
                foot,
                jnp.zeros(3 * config.n_contact, dtype=config.q0.dtype),
            ]
        )

    actual_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    theta_hat = jnp.zeros(config.n_theta, dtype=config.q0.dtype)
    if randomize_damping:
        # Use a separate deterministic stream so enabling damping
        # randomization does not change the rollout-disturbance sequence.
        theta_seed = np.random.SeedSequence([actual_seed, 0xD4A9])
        theta_rng = np.random.default_rng(theta_seed)
        true_theta = jnp.asarray(
            theta_rng.uniform(
                low=0.0,
                high=np.asarray(theta_half_width),
            ),
            dtype=config.q0.dtype,
        )
    else:
        true_theta = config.default_true_theta
    print(
        f"true damping ({'randomized' if randomize_damping else 'fixed'}): "
        f"{dict(zip(config.THETA_LABELS, np.asarray(true_theta).tolist()))}"
    )
    G_0 = jnp.pad(
        jnp.diag(theta_half_width),
        ((0, 0), (0, config.n_theta * sls_config.q_max - config.n_theta)),
    )
    x_phys = measured_physical_state()
    x = x_phys
    mpc_data = mpc.reset(
        mpc.make_data(),
        data.qpos.copy(),
        data.qvel.copy(),
        x_phys[mpc.foot_slice],
    )

    command = jnp.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    # Compile once, exactly as the GPU-SLS example does, then reset the carry
    # so compilation does not advance the receding-horizon state.
    mpc_data, warm_output = mpc.run(
        mpc_data,
        x,
        command,
        G_0,
        theta_hat,
        warm_contact,
    )
    warm_output.tau.block_until_ready()
    mpc_data = mpc.reset(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        x_phys[mpc.foot_slice],
    )

    states = [x]
    theta_estimates = [theta_hat]
    theta_widths = [jnp.sum(jnp.abs(G_0), axis=-1)]
    nominal_controls = []
    solve_times = []
    admm_rho_values = []
    admm_convergence_history = []
    admm_iteration_history = []
    admm_primal_residual_history = []
    admm_dual_residual_history = []
    admm_primal_tolerance_history = []
    admm_dual_tolerance_history = []
    sls_convergence_history = []
    sls_residual_history = []
    admm_primal_worst_flat_index_history = []
    admm_primal_worst_z_history = []
    admm_primal_worst_w_history = []
    planned_xy = []
    one_step_plans = []
    one_step_tubes = []
    horizon_tubes = []
    obstacle_backoff_history = []
    constraint_backoff_history = []
    disturbance_rng = np.random.default_rng(actual_seed)
    disturbance_samples = []
    applied_disturbances = []
    estimator_ranks = []
    estimator_singular_values = []
    estimator_innovations = []
    estimator_update_applied = []
    normal_grfs = []
    actual_contact_pairs = []
    realized_measurement_masks = []
    tau_nominal = jnp.zeros(config.n_joints, dtype=config.q0.dtype)
    previous_x = None
    previous_u = None
    previous_parameter = None
    previous_contact = None
    damped_joint_indices = np.asarray(config.DAMPED_JOINT_INDEX, dtype=int)

    def controller_update(controller_index: int) -> None:
        nonlocal x, theta_hat, G_0, mpc_data, tau_nominal
        nonlocal previous_x, previous_u, previous_parameter, previous_contact

        x_phys_now = measured_physical_state()
        measured_x = x_phys_now
        contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
        if previous_x is not None:
            current_normal_grfs = foot_normal_grfs()
            realized_measurement_matrix = contact_gated_transition_matrix(
                measurement_matrix,
                jnp.stack([previous_contact, contact]),
            )
            # Non-adaptive baseline: the nominal damping and its generator set
            # remain exactly equal to their initial values for the full run.
            estimator_rank = jnp.array(0, dtype=jnp.int32)
            estimator_svals = jnp.zeros(config.n_theta, dtype=config.q0.dtype)
            estimator_innovation = jnp.zeros(
                config.n_theta, dtype=config.q0.dtype
            )
            apply_estimator_update = jnp.array(False)
            states.append(measured_x)
            theta_estimates.append(theta_hat)
            theta_widths.append(jnp.sum(jnp.abs(G_0), axis=-1))
            estimator_ranks.append(estimator_rank)
            estimator_singular_values.append(estimator_svals)
            estimator_innovations.append(estimator_innovation)
            estimator_update_applied.append(apply_estimator_update)
            normal_grfs.append(current_normal_grfs)
            actual_contact_pairs.append(
                jnp.stack([previous_contact, contact])
            )
            realized_measurement_masks.append(
                jnp.diag(realized_measurement_matrix)
            )

        solve_start = timer()
        mpc_data, output = mpc.run(
            mpc_data,
            measured_x,
            command,
            G_0,
            theta_hat,
            contact * 0.0,
        )
        output.tau.block_until_ready()
        solve_time = timer() - solve_start
        if not bool(jnp.all(jnp.isfinite(output.X))) or not bool(jnp.all(jnp.isfinite(output.tau))):
            raise FloatingPointError(f"non-finite MPC solution at controller step {controller_index}")

        tau_nominal = output.tau
        nominal_controls.append(tau_nominal)
        planned_xy.append(output.X[:, :2])
        one_step_plans.append(output.X[1])
        upper_tubes = jnp.abs(output.backoffs[1:, : config.n])
        lower_tubes = jnp.abs(output.backoffs[1:, config.n : 2 * config.n])
        full_horizon_tubes = jnp.maximum(upper_tubes, lower_tubes)
        horizon_tubes.append(full_horizon_tubes)
        one_step_tubes.append(full_horizon_tubes[0])
        # The circular obstacle is the final row of ``constraints``. Its
        # exact robust tightening is therefore the final h_ct/backoff column.
        obstacle_backoff_history.append(jnp.abs(output.backoffs[1:, -1:]))
        constraint_backoff_history.append(output.backoffs)
        solve_times.append(solve_time)
        admm_rho_values.append(mpc_data.rho)
        admm_convergence_history.append(bool(np.asarray(output.admm_converged)))
        admm_iteration_history.append(output.admm_iterations)
        admm_primal_residual_history.append(output.admm_primal_residual)
        admm_dual_residual_history.append(output.admm_dual_residual)
        admm_primal_tolerance_history.append(output.admm_primal_tolerance)
        admm_dual_tolerance_history.append(output.admm_dual_tolerance)
        sls_convergence_history.append(output.sls_converged)
        sls_residual_history.append(output.sls_residual)
        admm_primal_worst_flat_index_history.append(
            output.admm_primal_worst_flat_index
        )
        admm_primal_worst_z_history.append(output.admm_primal_worst_z)
        admm_primal_worst_w_history.append(output.admm_primal_worst_w)
        if not admm_convergence_history[-1]:
            admm_constraint_count = int(mpc_data.w.shape[1])
            admm_worst_flat_index = int(output.admm_primal_worst_flat_index)
            admm_worst_horizon_step = admm_worst_flat_index // admm_constraint_count
            admm_worst_constraint_index = admm_worst_flat_index % admm_constraint_count
            print(
                "solver_nonconvergence "
                f"controller_step={controller_index:04d} "
                f"admm_iterations={int(output.admm_iterations)} "
                f"primal={float(output.admm_primal_residual):.3e}/"
                f"{float(output.admm_primal_tolerance):.3e} "
                f"dual={float(output.admm_dual_residual):.3e}/"
                f"{float(output.admm_dual_tolerance):.3e} "
                f"sls_converged={bool(output.sls_converged)} "
                f"sls_residual={float(output.sls_residual):.3e}/"
                f"{float(sls_config.sls_primal_tol):.3e}"
            )
            print(
                "admm_primal_argmax "
                f"controller_step={controller_index:04d} "
                f"horizon_step={admm_worst_horizon_step:02d} "
                f"index={admm_worst_constraint_index} "
                f"name={constraint_name(admm_worst_constraint_index)} "
                f"z={float(output.admm_primal_worst_z):.6e} "
                f"w={float(output.admm_primal_worst_w):.6e} "
                f"abs_z_minus_w="
                f"{abs(float(output.admm_primal_worst_z) - float(output.admm_primal_worst_w)):.6e}"
            )
            nominal_values, tightened_values = tightened_constraint_values(
                output.X,
                output.U,
                output.backoffs,
            )
            nominal_values = np.asarray(nominal_values)
            tightened_values = np.asarray(tightened_values)
            violating_entries = np.argwhere(tightened_values > 0.0)
            if violating_entries.size == 0:
                print(
                    f"admm_nonconverged controller_step={controller_index:04d}: "
                    "no positive post-solve tightened nonlinear constraint; "
                    "failure is due to the ADMM residual tolerance"
                )
            else:
                violation_sizes = tightened_values[
                    violating_entries[:, 0], violating_entries[:, 1]
                ]
                descending_order = np.argsort(violation_sizes)[::-1]
                max_reported_violations = 10
                for entry_index in descending_order[:max_reported_violations]:
                    horizon_step, constraint_index = violating_entries[entry_index]
                    nominal_value = nominal_values[horizon_step, constraint_index]
                    backoff = abs(
                        float(output.backoffs[horizon_step, constraint_index])
                    )
                    print(
                        "constraint_violation "
                        f"controller_step={controller_index:04d} "
                        f"horizon_step={int(horizon_step):02d} "
                        f"index={int(constraint_index)} "
                        f"name={constraint_name(int(constraint_index))} "
                        f"tightened_violation={float(violation_sizes[entry_index]):.3e} "
                        f"nominal_value={float(nominal_value):.3e} "
                        f"backoff={backoff:.3e}"
                    )
                if len(descending_order) > max_reported_violations:
                    print(
                        "constraint_violation "
                        f"additional_count={len(descending_order) - max_reported_violations}"
                    )
        previous_x = measured_x
        previous_u = tau_nominal
        previous_parameter = output.parameter
        previous_contact = contact
        x = measured_x

        if log_every > 0 and controller_index % log_every == 0:
            recent_admm_convergence = admm_convergence_history[-10:]
            recent_admm_failures = sum(
                not converged for converged in recent_admm_convergence
            )
            print(
                f"controller_step={controller_index:04d} "
                f"base=({float(x[0]): .3f}, {float(x[1]): .3f}, {float(x[2]): .3f}) "
                f"theta_hat={np.asarray(theta_hat)} max|u_nom|={float(jnp.max(jnp.abs(tau_nominal))):.2f} "
                f"solve={1e3 * solve_time:.1f} ms rho={float(mpc_data.rho):.3g} "
                f"admm_fail_last10={recent_admm_failures > 0} "
                f"({recent_admm_failures}/{len(recent_admm_convergence)})"
            )

    def plant_step(sim_step: int) -> None:
        if sim_step % controller_period == 0:
            controller_update(sim_step // controller_period)

        actual_tau = np.asarray(tau_nominal, dtype=float).copy()
        damped_joint_velocity = np.asarray(
            data.qvel[6 + damped_joint_indices],
            dtype=float,
        )
        actual_tau[damped_joint_indices] -= (
            np.asarray(true_theta) * damped_joint_velocity
        )
        data.ctrl[:] = actual_tau
        mujoco.mj_step(model, data)

        if (sim_step + 1) % controller_period == 0:
            disturbance_map = np.asarray(exogenous_disturbance(x[None, :])[0])
            w = disturbance_rng.uniform(-1.0, 1.0, size=disturbance_map.shape[1])
            state_disturbance = float(exogenous_rollout_scale) * (disturbance_map @ w)
            data.qpos[:] += state_disturbance[: model.nq]
            data.qvel[:] += state_disturbance[model.nq : model.nq + model.nv]
            mujoco.mj_forward(model, data)
            disturbance_samples.append(w)
            applied_disturbances.append(state_disturbance)

    def append_final_measurement_if_complete() -> None:
        nonlocal x, theta_hat, G_0
        if steps % controller_period != 0 or previous_x is None:
            return
        x_phys_now = measured_physical_state()
        contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
        current_normal_grfs = foot_normal_grfs()
        realized_measurement_matrix = contact_gated_transition_matrix(
            measurement_matrix,
            jnp.stack([previous_contact, contact]),
        )
        estimator_rank = jnp.array(0, dtype=jnp.int32)
        estimator_svals = jnp.zeros(config.n_theta, dtype=config.q0.dtype)
        estimator_innovation = jnp.zeros(config.n_theta, dtype=config.q0.dtype)
        apply_estimator_update = jnp.array(False)
        x = x_phys_now
        states.append(x)
        theta_estimates.append(theta_hat)
        theta_widths.append(jnp.sum(jnp.abs(G_0), axis=-1))
        estimator_ranks.append(estimator_rank)
        estimator_singular_values.append(estimator_svals)
        estimator_innovations.append(estimator_innovation)
        estimator_update_applied.append(apply_estimator_update)
        normal_grfs.append(current_normal_grfs)
        actual_contact_pairs.append(jnp.stack([previous_contact, contact]))
        realized_measurement_masks.append(
            jnp.diag(realized_measurement_matrix)
        )

    def save_rollout_checkpoint():
        completed_steps = len(states) - 1
        if completed_steps == 0:
            print("No completed controller intervals are available to save or plot.")
            return

        warmup_solves_to_ignore = 4
        timed_solves = np.asarray(solve_times[warmup_solves_to_ignore:], dtype=float)
        if timed_solves.size > 0:
            print(
                "Solve-time summary "
                f"(excluding first {warmup_solves_to_ignore} solves): "
                f"avg={1e3 * np.mean(timed_solves):.2f} ms, "
                f"median={1e3 * np.median(timed_solves):.2f} ms, "
                f"Q1={1e3 * np.percentile(timed_solves, 25):.2f} ms, "
                f"Q3={1e3 * np.percentile(timed_solves, 75):.2f} ms"
            )
        else:
            print(
                "Solve-time summary unavailable: fewer than "
                f"{warmup_solves_to_ignore + 1} solves completed."
            )

        states_np = np.asarray(jnp.stack(states))
        theta_estimates_np = np.asarray(jnp.stack(theta_estimates))
        theta_widths_np = np.asarray(jnp.stack(theta_widths))
        nominal_controls_np = np.asarray(
            jnp.stack(nominal_controls[:completed_steps])
        )
        planned_xy_np = np.asarray(jnp.stack(planned_xy[:completed_steps]))
        one_step_plans_np = np.asarray(
            jnp.stack(one_step_plans[:completed_steps])
        )
        one_step_tubes_np = np.asarray(
            jnp.stack(one_step_tubes[:completed_steps])
        )
        horizon_tubes_np = np.asarray(
            jnp.stack(horizon_tubes[:completed_steps])
        )
        obstacle_backoffs_np = np.asarray(
            jnp.stack(obstacle_backoff_history[:completed_steps])
        )
        constraint_backoffs_np = np.asarray(
            jnp.stack(constraint_backoff_history[:completed_steps])
        )
        disturbance_samples_np = np.asarray(disturbance_samples[:completed_steps])
        applied_disturbances_np = np.asarray(applied_disturbances[:completed_steps])

        np.savez(
            os.path.join(result_dir, "quadruped_nonadaptive_damping_mjx_rollout.npz"),
            xs=states_np,
            theta_estimates=theta_estimates_np,
            theta_widths=theta_widths_np,
            nominal_controls=nominal_controls_np,
            true_theta=np.asarray(true_theta),
            initial_theta_half_width=np.asarray(theta_half_width),
            admm_max_iterations=np.asarray(admm_max_iterations),
            rho_update_frequency=np.asarray(rho_update_frequency),
            disturbance_samples=disturbance_samples_np,
            applied_disturbances=applied_disturbances_np,
            disturbance_matrix=np.asarray(exogenous_disturbance(x[None, :])[0]),
            later_horizon_disturbance_matrix=np.asarray(
                exogenous_disturbance(jnp.stack([x, x]))[1]
            ),
            rollout_disturbance_matrix=np.asarray(
                exogenous_rollout_scale * exogenous_disturbance(x[None, :])[0]
            ),
            exogenous_rollout_scale=np.asarray(exogenous_rollout_scale),
            information_matrix=np.asarray(measurement_matrix),
            state_labels=np.asarray(augmented_state_labels()),
            estimator_sensitivity_ranks=np.asarray(estimator_ranks),
            estimator_singular_values=np.asarray(estimator_singular_values),
            estimator_innovations=np.asarray(estimator_innovations),
            estimator_update_applied=np.asarray(estimator_update_applied),
            foot_normal_grfs=np.asarray(normal_grfs),
            contact_force_threshold=np.asarray(contact_force_threshold),
            posterior_gate_enabled=np.asarray(enable_posterior_gate),
            fused_rti_updates=np.asarray(fuse_rti_updates),
            randomized_damping=np.asarray(randomize_damping),
            base_z_disturbance_enabled=np.asarray(base_z_disturbance),
            actual_contact_pairs=np.asarray(actual_contact_pairs),
            realized_measurement_masks=np.asarray(realized_measurement_masks),
            transition_times=np.asarray(solve_times),
            admm_rho=np.asarray(admm_rho_values[:completed_steps]),
            admm_converged=np.asarray(
                admm_convergence_history[:completed_steps], dtype=bool
            ),
            admm_iterations=np.asarray(
                admm_iteration_history[:completed_steps]
            ),
            admm_primal_residual=np.asarray(
                admm_primal_residual_history[:completed_steps]
            ),
            admm_dual_residual=np.asarray(
                admm_dual_residual_history[:completed_steps]
            ),
            admm_primal_tolerance=np.asarray(
                admm_primal_tolerance_history[:completed_steps]
            ),
            admm_dual_tolerance=np.asarray(
                admm_dual_tolerance_history[:completed_steps]
            ),
            sls_converged=np.asarray(
                sls_convergence_history[:completed_steps]
            ),
            sls_residual=np.asarray(
                sls_residual_history[:completed_steps]
            ),
            sls_primal_tolerance=np.asarray(sls_config.sls_primal_tol),
            planned_xy=planned_xy_np,
            one_step_plans=one_step_plans_np,
            one_step_tubes=one_step_tubes_np,
            horizon_tubes=horizon_tubes_np,
            obstacle_backoffs=obstacle_backoffs_np,
            constraint_backoffs=constraint_backoffs_np,
            admm_primal_worst_flat_index=np.asarray(
                admm_primal_worst_flat_index_history[:completed_steps]
            ),
            admm_primal_worst_z=np.asarray(
                admm_primal_worst_z_history[:completed_steps]
            ),
            admm_primal_worst_w=np.asarray(
                admm_primal_worst_w_history[:completed_steps]
            ),
            seed=np.asarray(actual_seed),
            theta_labels=np.asarray(config.THETA_LABELS),
            obstacle_center=np.asarray(obstacle_center),
            obstacle_radius=np.asarray(obstacle_radius),
            admm_tolerance=np.asarray(admm_config.eps_abs),
            plant_backend=np.asarray("mujoco_single_20ms_step"),
            planner_dt=np.asarray(config.dt),
            controller_dt=np.asarray(1.0 / config.mpc_frequency),
            simulation_dt=np.asarray(1.0 / sim_frequency),
        )
        if save_plots:
            save_diagnostic_plots(
                states=states_np,
                theta_estimates=theta_estimates_np,
                theta_widths=theta_widths_np,
                true_theta=np.asarray(true_theta),
                planned_xy=planned_xy_np,
                one_step_plans=one_step_plans_np,
                one_step_tubes=one_step_tubes_np,
                horizon_tubes=horizon_tubes_np,
                obstacle_backoffs=obstacle_backoffs_np,
                obstacle_center=np.asarray(obstacle_center),
                obstacle_radius=float(obstacle_radius),
                admm_tolerance=float(admm_config.eps_abs),
                sample_dt=1.0 / config.mpc_frequency,
            )
        print(f"Saved rollout checkpoint with {completed_steps} completed steps.")

    try:
        if headless:
            for sim_step in range(steps):
                plant_step(sim_step)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                with viewer.lock():
                    obstacle = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        obstacle,
                        mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([physical_obstacle_radius, 1.0, 0.0]),
                        np.array([float(obstacle_center[0]), float(obstacle_center[1]), 1.0]),
                        np.eye(3).ravel(),
                        np.array([0.9, 0.1, 0.1, 0.35], dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1
                for sim_step in range(steps):
                    if not viewer.is_running():
                        break
                    tic = timer()
                    plant_step(sim_step)
                    viewer.sync()
                    elapsed = timer() - tic
                    if elapsed < model.opt.timestep:
                        time.sleep(model.opt.timestep - elapsed)
        append_final_measurement_if_complete()
    except BaseException:
        try:
            save_rollout_checkpoint()
        except Exception as checkpoint_error:
            print(f"Failed to save partial rollout checkpoint: {checkpoint_error}")
        raise
    else:
        save_rollout_checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="number of 50 Hz / 0.02 s MuJoCo transitions")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for this run's checkpoint",
    )
    parser.add_argument(
        "--posterior-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject estimator rows whose posterior uncertainty would increase",
    )
    parser.add_argument(
        "--fused-rti-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fuse the MPC solve and warm-start carry updates into one JIT",
    )
    parser.add_argument(
        "--randomize-damping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "sample each true damping parameter independently from Uniform(0, "
            "initial_theta_half_width)"
        ),
    )
    parser.add_argument(
        "--base-z-disturbance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the direct base-position z disturbance channel",
    )
    parser.add_argument(
        "--initial-theta-half-width",
        type=float,
        default=None,
        help="override the shared initial damping half-width",
    )
    parser.add_argument(
        "--admm-max-iterations",
        type=int,
        default=1000,
        help="maximum ADMM iterations per MPC solve (default: 1000)",
    )
    parser.add_argument(
        "--rho-update-frequency",
        type=int,
        default=25,
        help="ADMM rho update frequency in iterations (default: 25)",
    )
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        seed=args.seed,
        log_every=args.log_every,
        save_plots=not args.no_plots,
        enable_posterior_gate=args.posterior_gate,
        fuse_rti_updates=args.fused_rti_updates,
        randomize_damping=args.randomize_damping,
        base_z_disturbance=args.base_z_disturbance,
        output_dir=args.output_dir,
        initial_theta_half_width=args.initial_theta_half_width,
        admm_max_iterations=args.admm_max_iterations,
        rho_update_frequency=args.rho_update_frequency,
    )
