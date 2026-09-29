"""Adaptive-SLS rollout for the 12D effective-thrust-gain Crazyflie model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from functools import partial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.append(str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_sls_matplotlib")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from gpu_sls.generic_mpc_wrapper import GenericMPCControllerWrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.sqp import SQPConfig

try:
    from . import config_crazyflie_12d as config
    from .dynamics_12d import rk4_step
    from .sim_adapter import high_level_to_actuators, make_plant, measured_state
    from .plots import save_diagnostic_plots
except ImportError:
    import config_crazyflie_12d as config
    from dynamics_12d import rk4_step
    from sim_adapter import high_level_to_actuators, make_plant, measured_state
    from plots import save_diagnostic_plots


def make_disturbance(scale: float):
    def disturbance(prefix):
        first = jnp.diag(scale * config.first_disturbance.astype(prefix.dtype))
        later = jnp.diag(scale * config.later_disturbance.astype(prefix.dtype))
        matrices = jnp.broadcast_to(later, (prefix.shape[0], config.n_phys, config.n_phys))
        return matrices if prefix.shape[0] == 0 else matrices.at[0].set(first)
    return disturbance


@jax.jit
def _reduce_generator(G):
    # Scalar beta: one interval generator is both exact and sufficient.
    return jnp.array([[jnp.sum(jnp.abs(G))]], dtype=G.dtype).reshape(1, 1).repeat(config.q_max, axis=1).at[:, 1:].set(0.)


@partial(jax.jit, static_argnames=("dynamics", "disturbance"))
def estimate_beta(dynamics, disturbance, previous_x, measured_next, previous_u, dt, beta, G):
    """Velocity-gated observable-subspace update for the effective gain."""
    def physical_step(candidate):
        return dynamics(previous_x.at[12:13].set(candidate), previous_u, 0, parameter=dt)[:12]
    predicted = physical_step(beta)
    E = jax.jacfwd(physical_step)(beta)
    C = config.measurement_matrix
    H = C @ E
    U, s, Vh = jnp.linalg.svd(H, full_matrices=False)
    threshold = jnp.maximum(1e-7, 1e-2 * jnp.max(s))
    rank = jnp.sum(s >= threshold)
    inv_s = jnp.where(s >= threshold, 1. / s, 0.)
    H_pinv = (Vh.T * inv_s) @ U.T
    residual_map = H_pinv @ C
    Pi = residual_map @ E
    W = residual_map @ disturbance(previous_x[None, :])[0]
    covariance = G @ G.T
    gain = covariance @ Pi.T @ jnp.linalg.inv(Pi @ covariance @ Pi.T + W @ W.T + 1e-7 * jnp.eye(1))
    # Match the quadruped estimator: do not apply an observable-subspace gain
    # unless it contracts the prior generator, then require posterior shrinkage.
    scaled_G = (gain @ Pi) @ G
    contraction_applied = (
        jnp.sum(jnp.abs(G - scaled_G), axis=1)
        < jnp.sum(jnp.abs(G), axis=1)
    )
    gain = jnp.diag(contraction_applied.astype(G.dtype)) @ gain
    G_pre = jnp.concatenate([G - (gain @ Pi) @ G, gain @ W], axis=1)
    posterior_applied = (
        jnp.sum(jnp.abs(G_pre), axis=1)
        < jnp.sum(jnp.abs(G), axis=1)
    )
    gain = jnp.diag(posterior_applied.astype(G.dtype)) @ gain
    innovation = residual_map @ (measured_next - predicted)
    candidate = beta + gain @ innovation
    # Require a real observable direction and retain the configured interval.
    supported = rank > 0
    updated = jnp.where(supported, candidate, beta)
    updated = jnp.clip(updated, config.beta_min, config.beta_max)
    G_next = _reduce_generator(jnp.concatenate([(jnp.eye(1) - gain @ Pi) @ G, gain @ W], axis=1))
    return updated, G_next, predicted, E, rank, s, innovation, supported, contraction_applied, posterior_applied


def _constraints(x, u, t):
    del t
    return jnp.concatenate([x - config.state_upper, config.state_lower - x, u - config.control_upper, config.control_lower - u])


def _reset(controller, X, U):
    """Reset state mutated by the compilation solve."""
    controller.X0, controller.U0 = X, U
    controller.V0 = jnp.zeros_like(controller.V0)
    controller.w, controller.y = jnp.zeros_like(controller.w), jnp.zeros_like(controller.y)
    controller.rho = jnp.asarray(controller.admm_config.initial_rho, dtype=controller.rho.dtype)
    controller.h_ct_ws = jnp.zeros_like(controller.h_ct_ws)
    controller.beta_ws = jnp.ones_like(controller.beta_ws) * 1e-10
    controller.mu_ws = jnp.zeros_like(controller.mu_ws)
    controller.Phi_x_ws, controller.Phi_u_ws = jnp.zeros_like(controller.Phi_x_ws), jnp.zeros_like(controller.Phi_u_ws)
    controller.E_prev, controller.K_prev, controller.L_prev = jnp.zeros_like(controller.E_prev), jnp.zeros_like(controller.K_prev), jnp.zeros_like(controller.L_prev)
    controller.G_prev, controller.P_inv_prev = jnp.zeros_like(controller.G_prev), jnp.zeros_like(controller.P_inv_prev)
    controller._has_generator_history = False


def main(*, steps=267, true_mass_scale=1.2, disturbance_scale=1., log_every=10, output_dir=None, save_plots=True):
    if steps < 1 or true_mass_scale <= 0. or disturbance_scale <= 0.:
        raise ValueError("steps, true_mass_scale, and disturbance_scale must be positive")
    output_dir = Path(output_dir or HERE / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    disturbance = make_disturbance(disturbance_scale)
    sls = SLSConfig(max_sls_iterations=1, sls_primal_tol=5e-2, enable_fastsls=True, enable_linearization_error=False, enable_disturbance_variation_error=False, warm_start=True, n_x=config.n_phys, n_w=config.n_w, n_theta=1, num_param=1, q_max=config.q_max, adaptive=True, enable_posterior_gate=True)
    beta = jnp.array([config.beta_hat_initial], dtype=jnp.float32)
    G = config.initial_generator()
    X_seed, U_seed = config.initial_seed(beta)
    controller = GenericMPCControllerWrapper(sls, SQPConfig(), ADMMConfig(max_iterations=80, rho_update_frequency=20), config=config, dynamics=config.dynamics, constraints=_constraints, obstacles=jnp.empty((0, 3), dtype=jnp.float32), cost=config.cost, num_constraints=2 * config.n + 2 * config.m, disturbance=disturbance, X_in=X_seed, U_in=U_seed, limited_memory=False, shift=1, measurement_matrix=config.measurement_matrix, fuse_rti_updates=True)
    model, data, body_id, true_mass = make_plant(true_mass_scale)
    n_substeps = round(config.dt / model.opt.timestep)
    if not np.isclose(n_substeps * model.opt.timestep, config.dt):
        raise ValueError("controller interval must be an integer number of physics steps")
    x0 = jnp.concatenate([measured_state(data, body_id), beta])
    ref = config.goal_horizon(beta)
    warm = controller.run(x0=x0, G_0=G, reference=ref, nominal_param=beta, parameter=config.dt, X_tail=ref[-1])
    warm[0].block_until_ready()
    _reset(controller, X_seed, U_seed)
    x, previous_x, previous_u = x0, None, None
    command = jnp.array([config.gravity / beta[0], 0., 0., 0.], dtype=jnp.float32)
    yaw_target = float(x[8])
    states, betas, widths, controls, predictions, errors, ranks, singular_values, innovations, solve_times = [x], [beta], [jnp.sum(jnp.abs(G), axis=1)], [], [], [], [], [], [], []
    contraction_gates, posterior_gates = [], []
    reached_goal = False
    for k in range(steps):
        physical_now = measured_state(data, body_id)
        if previous_x is not None:
            beta, G, prediction, _, rank, s, innovation, supported, contraction_gate, posterior_gate = estimate_beta(config.dynamics, disturbance, previous_x, physical_now, previous_u, config.dt, beta, G)
            predictions.append(prediction); errors.append(physical_now - prediction); ranks.append(rank); singular_values.append(s); innovations.append(innovation)
            contraction_gates.append(contraction_gate); posterior_gates.append(posterior_gate)
        x = jnp.concatenate([physical_now, beta])
        goal_distance = float(jnp.linalg.norm(physical_now[:3] - config.goal_position))
        if goal_distance <= config.goal_radius:
            reached_goal = True
            print(f"goal reached at step={k:04d}: distance={goal_distance:.3f} m; terminating rollout")
            break
        ref = config.goal_horizon(beta)
        tic = timer()
        out = controller.run(x0=x, G_0=G, reference=ref, nominal_param=beta, parameter=config.dt, X_tail=ref[-1])
        out[0].block_until_ready()
        solve_times.append(timer() - tic)
        probe = config.probe_amplitude * jnp.sin(2. * jnp.pi * config.probe_frequency_hz * k * config.dt)
        command = jnp.clip(out[0].at[0].add(probe), config.control_lower, config.control_upper)
        if not bool(jnp.all(jnp.isfinite(command))):
            np.savez(
                output_dir / "partial_rollout_before_nonfinite.npz",
                xs=np.asarray(jnp.stack(states)),
                beta_estimates=np.asarray(jnp.stack(betas)),
                beta_widths=np.asarray(jnp.stack(widths)),
                controls=np.asarray(jnp.stack(controls)) if controls else np.empty((0, config.m)),
                one_step_predictions=np.asarray(jnp.stack(predictions)) if predictions else np.empty((0, config.n_phys)),
                one_step_errors=np.asarray(jnp.stack(errors)) if errors else np.empty((0, config.n_phys)),
                estimator_ranks=np.asarray(ranks),
                estimator_singular_values=np.asarray(jnp.stack(singular_values)) if singular_values else np.empty((0, 1)),
                estimator_innovations=np.asarray(jnp.stack(innovations)) if innovations else np.empty((0, 1)),
                estimator_contraction_gates=np.asarray(jnp.stack(contraction_gates)) if contraction_gates else np.empty((0, 1), dtype=bool),
                estimator_posterior_gates=np.asarray(jnp.stack(posterior_gates)) if posterior_gates else np.empty((0, 1), dtype=bool),
                solve_times=np.asarray(solve_times),
            )
            np.savez(
                output_dir / "nonfinite_mpc_diagnostic.npz",
                step=np.asarray(k),
                state=np.asarray(x),
                reference=np.asarray(ref),
                raw_control=np.asarray(out[0]),
                planned_states=np.asarray(out[1]),
                planned_controls=np.asarray(out[2]),
                beta=np.asarray(beta),
                beta_width=np.asarray(jnp.sum(jnp.abs(G), axis=1)),
            )
            raise FloatingPointError(f"non-finite command at step {k}")
        previous_x, previous_u = x, command
        for _ in range(n_substeps):
            low_level, yaw_target = high_level_to_actuators(model, data, body_id, np.asarray(command), yaw_target)
            data.ctrl[:] = low_level
            mujoco.mj_step(model, data)
        states.append(jnp.concatenate([measured_state(data, body_id), beta]))
        betas.append(beta); widths.append(jnp.sum(jnp.abs(G), axis=1)); controls.append(command)
        if log_every and k % log_every == 0:
            print(f"step={k:04d} pos={np.asarray(x[:3])} beta={float(beta[0]):.3f} rank={int(ranks[-1]) if ranks else 0} u={np.asarray(command)} solve_ms={1e3*solve_times[-1]:.1f}")
    path = output_dir / "crazyflie_12d_effective_gain_rollout.npz"
    np.savez(path, xs=np.asarray(jnp.stack(states)), beta_estimates=np.asarray(jnp.stack(betas)), beta_widths=np.asarray(jnp.stack(widths)), true_beta=np.asarray(1. / true_mass), controls=np.asarray(jnp.stack(controls)), probe_thrust=np.asarray([config.probe_amplitude * np.sin(2. * np.pi * config.probe_frequency_hz * k * config.dt) for k in range(len(controls))]), one_step_predictions=np.asarray(jnp.stack(predictions)), one_step_errors=np.asarray(jnp.stack(errors)), estimator_ranks=np.asarray(ranks), estimator_singular_values=np.asarray(jnp.stack(singular_values)), estimator_innovations=np.asarray(jnp.stack(innovations)), estimator_contraction_gates=np.asarray(jnp.stack(contraction_gates)), estimator_posterior_gates=np.asarray(jnp.stack(posterior_gates)), solve_times=np.asarray(solve_times), goal_position=np.asarray(config.goal_position), goal_radius=np.asarray(config.goal_radius), reached_goal=np.asarray(reached_goal), model_kwargs=np.asarray(config.model_kwargs, dtype=object), controller_dt=np.asarray(config.dt), simulation_dt=np.asarray(model.opt.timestep))
    if save_plots:
        save_diagnostic_plots(path)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=267)
    parser.add_argument("--true-mass-scale", type=float, default=1.2)
    parser.add_argument("--disturbance-scale", type=float, default=1.)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    arguments = vars(args)
    arguments["save_plots"] = not arguments.pop("no_plots")
    main(**arguments)
