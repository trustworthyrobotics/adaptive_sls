"""
dubins_car_mpc_experiment.py

End-to-end experiment: use mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control a Dubins car with box constraints on controls.

State:      x = [px, py, theta, v]
Control:    u = [omega, accel]
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from typing import Any, Callable

import argparse
import csv
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import config
import jax.scipy.linalg as jla

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sls import girard_reduction_opt as girard_reduction
from gpu_sls.sqp import SQPConfig
from gpu_sls.generic_mpc_wrapper import GenericMPCControllerWrapper
from utils.contraint_utils import combine_constraints
from utils.sls_visual import get_trajectory_tubes, get_adaptive_trajectory_tubes_exact

config.update("jax_enable_x64", True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

NUM_RANDOM = 2
NUM_ADV = 2
GOAL_TOL = 0.001
DEFAULT_INFORMATION_CENTER = (0.25, 0.0)
DEFAULT_TERMINAL_STATE_WEIGHTS = (10.0, 10.0, 0.0, 0.0)
# The start and goal lie on x=0, so this obstacle forces a meaningful
# left/right avoidance maneuver in both controller variants.
DEFAULT_OBSTACLE = (0.2, 0.0, 0.3)
DEFAULT_SECOND_OBSTACLE = (-0.075, -1.5, 0.3)
DEFAULT_THIRD_OBSTACLE = (-1.3, -0.7, 0.3)

def reached_goal_xy(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dxy = x[:2] - x_goal[:2]
    return (dxy @ dxy) <= (tol * tol)

def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi



def compute_state_feedback_gains(Phi_x, Phi_u, n_x_aug, n_u):
    # Infer sequence lengths from the tensors directly (e.g., T+1 vs T)
    T_x, T_w = Phi_x.shape[0], Phi_x.shape[1]
    T_u = Phi_u.shape[0]
    
    # 1. Flatten 4D block tensors into 2D block matrices
    # Phi_x becomes (T_x * n, T_w * n)
    Phi_x_2d = jnp.swapaxes(Phi_x, 1, 2).reshape(T_x * n_x_aug, T_w * n_x_aug)
    
    # Phi_u becomes (T_u * m, T_w * n)
    Phi_u_2d = jnp.swapaxes(Phi_u, 1, 2).reshape(T_u * n_u, T_w * n_x_aug)
    
    # 2. Solve the transposed system: (Phi_x_2d.T) @ K_2d.T = Phi_u_2d.T
    K_T_2d = jla.solve_triangular(Phi_x_2d.T, Phi_u_2d.T, lower=False)
    K_2d = K_T_2d.T  # Shape: (T_u * n_u, T_w * n_x_aug)
    
    # 3. Fold the 2D matrix back into a 4D block tensor (T_u, T_w, n_u, n_x_aug)
    K_kjN = K_2d.reshape(T_u, n_u, T_w, n_x_aug).swapaxes(1, 2)
    
    # 4. Enforce strict causality (mask out non-causal elements)
    mask = jnp.tril(jnp.ones((T_u, T_w)))[:, :, None, None]
    K_kjN = K_kjN * mask
    
    return K_kjN


def verify_final_sls_response(
    *,
    Phi_x: jnp.ndarray,
    Phi_u: jnp.ndarray,
    X: jnp.ndarray,
    U: jnp.ndarray,
    adaptive: bool,
    nominal_param: jnp.ndarray,
    dt: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """Re-linearize the returned plan and check its finite-horizon SLS identity."""
    time_indices = jnp.arange(U.shape[0])
    if adaptive:
        def adaptive_a(x, u, t):
            return jax.jacfwd(dynamics, argnums=0)(x, u, t, parameter=dt)

        def adaptive_b(x, u, t):
            return jax.jacfwd(dynamics, argnums=1)(x, u, t, parameter=dt)

        A = jax.vmap(adaptive_a)(X[:-1], U, time_indices)
        B = jax.vmap(adaptive_b)(X[:-1], U, time_indices)
    else:
        def nonadaptive_a(x, u, t):
            return jax.jacfwd(nonadaptive_dynamics, argnums=0)(
                x, u, t, nominal_param, parameter=dt
            )

        def nonadaptive_b(x, u, t):
            return jax.jacfwd(nonadaptive_dynamics, argnums=1)(
                x, u, t, nominal_param, parameter=dt
            )

        A = jax.vmap(nonadaptive_a)(X[:-1], U, time_indices)
        B = jax.vmap(nonadaptive_b)(X[:-1], U, time_indices)

    A_np, B_np = np.asarray(A), np.asarray(B)
    phi_x, phi_u = np.asarray(Phi_x), np.asarray(Phi_u)
    transitions = phi_u.shape[0]
    # Column zero is the initial-condition response.  As used by the tube
    # routine, columns 1:T+1 map disturbances at transitions 0:T; validate
    # that shifted block system separately.
    residual = (
        phi_x[1 : transitions + 1, 1 : transitions + 1]
        - np.einsum("tij,tkjl->tkil", A_np, phi_x[:transitions, 1 : transitions + 1])
        - np.einsum("tij,tkjl->tkil", B_np, phi_u[:, 1 : transitions + 1])
    )
    residual[np.arange(transitions), np.arange(transitions)] -= np.eye(phi_x.shape[-1])
    causal_x = np.triu(np.ones((transitions, transitions), dtype=bool), k=1)
    causal_u = np.triu(np.ones((transitions, transitions), dtype=bool), k=1)
    causality_error = max(
        float(np.max(np.abs(phi_x[1:, 1:][causal_x]))) if np.any(causal_x) else 0.0,
        float(np.max(np.abs(phi_u[:, 1:][causal_u]))) if np.any(causal_u) else 0.0,
    )
    maximum = float(np.max(np.abs(residual)))
    return A_np, B_np, {
        "response_residual_max": maximum,
        "response_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "causality_residual_max": causality_error,
        "verification_tolerance": tolerance,
        "verification_passed": bool(maximum <= tolerance and causality_error <= tolerance),
    }

# -----------------------------
# Dubins car dynamics
# -----------------------------
def dubins_step_impl(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, th, ve = x[0], x[1], x[2], x[3]
    om_bias, ac_bias = x[4], x[5]
    om, ac = u[0], u[1]
    
    # The unknown x/y velocity biases are additive, independent of speed.
    px_next = px + dt * (ve * jnp.cos(th) + om_bias)
    py_next = py + dt * (ve * jnp.sin(th) + ac_bias)
    # px_next = px + dt * (ve + ac_bias) * jnp.cos(th + om_bias)
    # py_next = py + dt * (ve + ac_bias) * jnp.sin(th + om_bias)
    th_next = th + dt * om
    ve_next = ve + dt * ac
    
    om_bias_next = om_bias
    ac_bias_next = ac_bias
    return jnp.array([px_next, py_next, th_next, ve_next, om_bias_next, ac_bias_next], dtype=x.dtype)

dubins_step = jax.jit(dubins_step_impl)  

def dubins_step_with_disturbance(
    key: jax.Array, x: jnp.ndarray, u: jnp.ndarray, true_param: jnp.ndarray,
    nominal_parameter: jnp.ndarray, G_0: jnp.ndarray,
    disturbance: Callable[[jnp.ndarray], jnp.ndarray], dt: float, i: int, adaptive: bool = True,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    
    px, py, th, ve = x[0], x[1], x[2], x[3]
    om_bias, ac_bias = true_param[0], true_param[1]
    om, ac = u[0], u[1]

    px_next = px + dt * (ve * jnp.cos(th) + om_bias)
    py_next = py + dt * (ve * jnp.sin(th) + ac_bias)
    # px_next = px + dt * (ve + ac_bias) * jnp.cos(th + om_bias)
    # py_next = py + dt * (ve + ac_bias) * jnp.sin(th + om_bias)
    th_next = th + dt * om
    ve_next = ve + dt * ac
    x_nom = jnp.array([px_next, py_next, th_next, ve_next], dtype=x.dtype)

    E_val = disturbance(x[None, :])
    E = E_val[0]

    key, subkey = jax.random.split(key, 2)
    n_w = E.shape[1]

    w = jnp.ones((n_w,), dtype=x.dtype)

    x_next = x_nom[:4] + E @ w
    if adaptive:
        x_next = jnp.concatenate([x_next, x[-2:]])
    return key, x_next, w

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dt = parameter
    return dubins_step_impl(x, u, dt)

def nonadaptive_dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, nominal_param: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dt = parameter
    x_aug = jnp.concatenate([x, nominal_param])
    num_params = nominal_param.shape[0]
    x = dubins_step_impl(x_aug, u, dt)
    return x[:-num_params]

def cost(
    W,
    reference,
    x,
    u,
    t,
    adaptive: bool = True,
    terminal_state_weights: jnp.ndarray | None = None,
    terminal_time: int | None = None,
):
    if adaptive:
        wx, wy, wtheta, wv, _, _, womega, waccel = W
    else:
        wx, wy, wtheta, wv, womega, waccel = W

    xref = reference[t]
    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = x[2] - xref[2]
    dv = x[3] - xref[3]
    om, ac = u[0], u[1]

    stage_state_cost = (
        wx * dx * dx
        + wy * dy * dy
        + wtheta * dth * dth
        + wv * dv * dv
    )
    stage_control_cost = womega * om * om + waccel * ac * ac

    if terminal_state_weights is None or terminal_time is None:
        return stage_state_cost + stage_control_cost

    terminal_state_cost = (
        terminal_state_weights[0] * dx * dx
        + terminal_state_weights[1] * dy * dy
        + terminal_state_weights[2] * dth * dth
        + terminal_state_weights[3] * dv * dv
    )
    is_terminal = t == terminal_time
    return jnp.where(
        is_terminal,
        terminal_state_cost,
        stage_state_cost + stage_control_cost,
    )

def make_control_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray):
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

def make_state_box_constraints(x_min: jnp.ndarray, x_max: jnp.ndarray):
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)
    return constraints


def make_terminal_position_constraints(
    goal_xy: jnp.ndarray,
    tolerance: float,
):
    """Return paired inequalities for the nominal terminal x-y position."""
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del u, t
        error_xy = x[:2] - goal_xy
        tolerance_array = jnp.asarray(tolerance, dtype=x.dtype)
        return jnp.concatenate(
            [
                error_xy - tolerance_array,
                -error_xy - tolerance_array,
            ]
        )

    return constraints

def make_state_varying_disturbance(
    n: int,
    alpha: float,
    adaptive: bool = True,
    dt: float = 0.1,
    information_center: tuple[float, float] = DEFAULT_INFORMATION_CENTER,
):
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        # A fixed, isotropic exogenous disturbance.  ``alpha`` is retained so
        # deterministic nominal solves can request the same map with alpha=0.
        # Keep a zero-valued state dependency so the interval-program tracer
        # represents this vector output per input sample rather than folding
        # the complete 4x4 matrix into one constant literal.
        dist_val = jnp.asarray(alpha, dtype=X_prefix.dtype) + 0.0 * X_prefix[:, 0]
        return jnp.einsum("t,ij->tij", dist_val, jnp.eye(n, dtype=X_prefix.dtype))

    if adaptive:
        return disturbance
    
    def sensitivity(X_prefix: jnp.ndarray, U_prefix: jnp.ndarray, nominal_param: jnp.ndarray) -> jnp.ndarray:
        def step_sensitivity(x, u, t):
            return jax.jacrev(nonadaptive_dynamics, argnums=3)(x, u, t, nominal_param, parameter=dt)
        # Trajectory calls provide T controls for T+1 states. The interval
        # tracer instead provides one already-aligned state/control pair.
        pad_len = X_prefix.shape[0] - U_prefix.shape[0]
        U_pad = jnp.pad(U_prefix, ((0, pad_len), (0, 0))) if pad_len > 0 else U_prefix
        t_arr = jnp.arange(X_prefix.shape[0])
        return jax.vmap(step_sensitivity)(X_prefix, U_pad, t_arr)
        
    def parametric_disturbance(X_prefix: jnp.ndarray, U_prefix: jnp.ndarray, nominal_param: jnp.ndarray, G_0: jnp.ndarray) -> jnp.ndarray:
        def step_sensitivity(x, u, t):
            return jax.jacrev(nonadaptive_dynamics, argnums=3)(x, u, t, nominal_param, parameter=dt)
        pad_len = X_prefix.shape[0] - U_prefix.shape[0]
        U_pad = jnp.pad(U_prefix, ((0, pad_len), (0, 0))) if pad_len > 0 else U_prefix
        t_arr = jnp.arange(X_prefix.shape[0])
        
        sensitivity = jax.vmap(step_sensitivity)(X_prefix, U_pad, t_arr)
        param_dist = sensitivity @ G_0
        exog_dist = disturbance(X_prefix)
        return jnp.concatenate([exog_dist, param_dist], axis=-1)

    return parametric_disturbance, sensitivity, disturbance

def measurement(x_pred_lin: jnp.ndarray, x_next: jnp.ndarray, kalman_gain: jnp.ndarray, sensitivity: jnp.ndarray, current_param: jnp.ndarray):
    # x_pred_lin is the prediction from the LTV system: X_pred[k+1] + A_k * \Delta x_k + B_k * \Delta u_k
    innovation = sensitivity @ (x_next - x_pred_lin)
    delta_param = kalman_gain @ innovation
    return current_param + delta_param, delta_param

@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray
    dt: float

# -----------------------------
# Main experiment
# -----------------------------
def main(
    *,
    adaptive: bool = True,
    enable_leb: bool = True,
    output_dir: str | Path = "active_car_information_center_results",
    horizon: int = 100,
    time_step_s: float = 0.05,
    exogenous_disturbance_scale: float = 0.0004,
    parameter_uncertainty_bound: float = 0.10,
    sls_q_bar_scale: float = 1.0,
    sls_r_bar_scale: float = 1.0,
    model_label: str = "Active Adaptive Dubins Car",
    state_weights: tuple[float, float, float, float] = (0.5, 0.5, 0.1, 0.1),
    control_weights: tuple[float, float] = (1.0, 10.0),
    terminal_state_weights: tuple[float, float, float, float] = DEFAULT_TERMINAL_STATE_WEIGHTS,
    enforce_terminal_position: bool = True,
    terminal_position_tolerance: float = 1e-2,
    deterministic_nominal_solve: bool = False,
    interpolated_reference: bool = False,
    route_bias_m: float = 0.0,
    shared_physical_nominal_solve: bool = False,
    track_nominal_during_robust_solve: bool = False,
    goal_x_m: float = 0.0,
    goal_y_m: float = -1.0,
    start_x_m: float = 0.0,
    start_y_m: float = 1.0,
    start_heading_rad: float | None = None,
    initial_speed_m_per_s: float = 0.0,
    goal_speed_m_per_s: float = 0.0,
    minimum_speed_m_per_s: float = -1.0,
    maximum_speed_m_per_s: float = 10.0,
    heading_min_rad: float = -jnp.inf,
    heading_max_rad: float = jnp.inf,
    x_min_m: float = -5.0,
    x_max_m: float = 5.0,
    enable_information_cost: bool = False,
    information_cost_weight: float = 1.0,
    information_cost_discount: float = 1.0,
    enable_edagger_f_cost: bool = False,
    edagger_f_cost_weight: float = 1.0,
    information_center: tuple[float, float] = DEFAULT_INFORMATION_CENTER,
    obstacle: tuple[float, float, float] = DEFAULT_OBSTACLE,
    additional_obstacles: tuple[tuple[float, float, float], ...] = (),
) -> Path:
    """Run the straight-down active-information Dubins experiment."""
    if enable_information_cost and not adaptive:
        raise ValueError("The information cost requires adaptive=True.")
    if enable_edagger_f_cost and not adaptive:
        raise ValueError("The E-dagger-F cost requires adaptive=True.")
    if enable_information_cost and enable_edagger_f_cost:
        raise ValueError(
            "The posterior-trace and E-dagger-F costs are mutually exclusive."
        )
    if information_cost_weight < 0.0:
        raise ValueError("information_cost_weight must be nonnegative.")
    if edagger_f_cost_weight < 0.0:
        raise ValueError("edagger_f_cost_weight must be nonnegative.")
    if not 0.0 < information_cost_discount <= 1.0:
        raise ValueError("information_cost_discount must lie in (0, 1].")
    if len(terminal_state_weights) != 4:
        raise ValueError("terminal_state_weights must contain four values.")
    if any(weight < 0.0 for weight in terminal_state_weights):
        raise ValueError("terminal_state_weights must be nonnegative.")
    if terminal_position_tolerance < 0.0:
        raise ValueError("terminal_position_tolerance must be nonnegative.")
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive.")
    if exogenous_disturbance_scale < 0.0:
        raise ValueError("exogenous_disturbance_scale must be nonnegative.")
    if parameter_uncertainty_bound < 0.0:
        raise ValueError("parameter_uncertainty_bound must be nonnegative.")
    if sls_q_bar_scale <= 0.0:
        raise ValueError("sls_q_bar_scale must be positive.")
    if sls_r_bar_scale <= 0.0:
        raise ValueError("sls_r_bar_scale must be positive.")
    if x_min_m >= x_max_m:
        raise ValueError("x_min_m must be smaller than x_max_m.")
    if heading_min_rad >= heading_max_rad:
        raise ValueError("heading_min_rad must be smaller than heading_max_rad.")
    if minimum_speed_m_per_s >= maximum_speed_m_per_s:
        raise ValueError(
            "minimum_speed_m_per_s must be smaller than maximum_speed_m_per_s."
        )
    if not minimum_speed_m_per_s <= initial_speed_m_per_s <= maximum_speed_m_per_s:
        raise ValueError("initial_speed_m_per_s must lie inside the speed bounds.")
    if not minimum_speed_m_per_s <= goal_speed_m_per_s <= maximum_speed_m_per_s:
        raise ValueError("goal_speed_m_per_s must lie inside the speed bounds.")
    obstacle_specs = (obstacle, *additional_obstacles)
    if any(obstacle_spec[2] <= 0.0 for obstacle_spec in obstacle_specs):
        raise ValueError("all obstacle radii must be positive.")

    output_dir = Path(output_dir)
    if enable_edagger_f_cost:
        active_prefix = "edagger_f_adaptive_sls_"
    elif enable_information_cost:
        active_prefix = "active_adaptive_sls_"
    else:
        active_prefix = ""
    run_name = (
        f"{active_prefix}adaptive_{str(adaptive).lower()}"
        f"__leb_{str(enable_leb).lower()}"
    )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # By default, align the car with the straight-line start-to-goal path.
    # Callers can still provide an explicit heading for legacy experiments.
    straight_line_heading = float(np.arctan2(goal_y_m - start_y_m, goal_x_m - start_x_m))
    path_heading = (
        straight_line_heading
        if start_heading_rad is None
        else float(start_heading_rad)
    )

    print(f"Running car experiment: adaptive={adaptive}, LEB={enable_leb}, output={run_dir}")
    print(
        f"Geometry: start=({start_x_m}, {start_y_m}), goal=({goal_x_m}, {goal_y_m}), "
        f"heading={path_heading}, "
        f"information_center={information_center}, obstacles={obstacle_specs}"
    )
    print(f"Nominal terminal state weights: {terminal_state_weights}")
    print(
        "Terminal nominal x-y constraint: "
        f"enabled={enforce_terminal_position}, tolerance={terminal_position_tolerance}"
    )
    print(
        "Stage information costs: "
        f"posterior_trace={enable_information_cost}, "
        f"edagger_f={enable_edagger_f_cost}, "
        f"edagger_f_weight={edagger_f_cost_weight}"
    )
    print(f"JAX backend={jax.default_backend()}, devices={jax.devices()}")
    n_x = 4
    n_u = 2    
    num_params = 2
    
    # Restored Original Dubins Setup
    nominal_param = jnp.array([0.0, 0.0], dtype=jnp.float64)  
    true_param = jnp.array([-0.05, 0.05], dtype=jnp.float64)
    # Initial independent bounds on the two unknown velocity-bias parameters.
    G_0 = jnp.eye(num_params, dtype=jnp.float64) * parameter_uncertainty_bound

    N = horizon
    dt = time_step_s
    parameter = dt
    q_max = 40
    
    om_max = 4.0
    ac_max = 1.0
    large_inf = jnp.inf 

    u_min = jnp.array([-om_max, -ac_max], dtype=jnp.float64)
    u_max = jnp.array([om_max, ac_max], dtype=jnp.float64)
    x_max = jnp.array(
        [x_max_m, 5.0, heading_max_rad, maximum_speed_m_per_s], dtype=jnp.float64
    )
    x_min = jnp.array(
        [x_min_m, -5.0, heading_min_rad, minimum_speed_m_per_s], dtype=jnp.float64
    )

    x0 = jnp.array(
        [start_x_m, start_y_m, path_heading, initial_speed_m_per_s],
        dtype=jnp.float64,
    )
    x_goal = jnp.array(
        [goal_x_m, goal_y_m, path_heading, goal_speed_m_per_s],
        dtype=jnp.float64,
    )

    # Note: Position weights set to 5.0 to force controller to correct deterministic drift!
    W_state = jnp.asarray(state_weights, dtype=jnp.float64)
    W_ctrl = jnp.asarray(control_weights, dtype=jnp.float64)

    if adaptive:
        W_param = jnp.array([0.0] * num_params, dtype=jnp.float64)
        W = jnp.concatenate([W_state, W_param, W_ctrl])
        n_theta = num_params
        n_w = n_x
        dyn_fn = dynamics
        param_bounds = jnp.array([jnp.inf] * num_params, dtype=jnp.float64)
        x_max = jnp.concatenate([x_max, param_bounds])
        x_min = jnp.concatenate([x_min, -param_bounds])
        x0 = jnp.concat([x0, nominal_param])
        x_goal = jnp.concat([x_goal, nominal_param])
    else:
        W = jnp.concatenate([W_state, W_ctrl])
        n_theta = 0
        n_w = n_x + num_params
        dyn_fn = nonadaptive_dynamics
        
    n = n_x + n_theta
    alpha_sim = exogenous_disturbance_scale
    disturbance = make_state_varying_disturbance(
        n=n_x,
        alpha=alpha_sim,
        adaptive=adaptive,
        dt=dt,
        information_center=information_center,
    )

    if not adaptive:
        exog_dist = disturbance[2]
        # disturbance = (disturbance[0], disturbance[1])
    else:
        exog_dist = disturbance
    terminal_weights_array = jnp.asarray(
        terminal_state_weights,
        dtype=jnp.float64,
    )
    base_cost_fn = partial(
        cost,
        adaptive=adaptive,
        terminal_state_weights=terminal_weights_array,
        terminal_time=N,
    )
    if enable_edagger_f_cost:
        def cost_fn(W, reference, x, u, t):
            nominal_cost = base_cost_fn(W, reference, x, u, t)

            def stage_edagger_f(_):
                dynamics_jacobian = jax.jacfwd(dyn_fn, argnums=0)(
                    x,
                    u,
                    t,
                    parameter=dt,
                )
                E_t = dynamics_jacobian[:n_x, n_x : n_x + n_theta]
                F_t = exog_dist(x[None, :])[0][:n_x]
                E_dagger_F = jnp.linalg.pinv(E_t, rtol=1e-6) @ F_t
                return (
                    jnp.asarray(edagger_f_cost_weight, dtype=x.dtype)
                    * jnp.sum(E_dagger_F * E_dagger_F)
                )

            return nominal_cost + jax.lax.cond(
                t < N,
                stage_edagger_f,
                lambda _: jnp.asarray(0.0, dtype=x.dtype),
                operand=None,
            )
    else:
        cost_fn = base_cost_fn

    nominal_disturbance = (
        make_state_varying_disturbance(
            n=n_x,
            alpha=0.0,
            adaptive=adaptive,
            dt=dt,
            information_center=information_center,
        )
        if deterministic_nominal_solve
        else disturbance
    )
    nominal_G_0 = jnp.zeros_like(G_0) if deterministic_nominal_solve else G_0

    cfg = MPCConfig(n=n, nu=n_u, N=N, W=W, u_ref=jnp.zeros((n_u,), dtype=jnp.float64), dt=dt)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=0, rho_max=1e5, max_iterations=1000)
    sls_cfg = SLSConfig(max_sls_iterations=2, sls_primal_tol=1e-2, enable_fastsls=False, n_x=n_x, n_w=n_x, n_theta=n_theta, num_param=num_params, q_max=q_max, adaptive=adaptive, enable_linearization_error=enable_leb)
    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        line_search=True,
    )

    constraints_u = make_control_box_constraints(u_min, u_max)
    constraints_x = make_state_box_constraints(x_min, x_max)
    constraints_all = combine_constraints(constraints_x, constraints_u)
    terminal_constraint_fn = (
        make_terminal_position_constraints(
            x_goal[:2],
            terminal_position_tolerance,
        )
        if enforce_terminal_position
        else None
    )
    num_terminal_constraints = 4 if enforce_terminal_position else 0

    obstacles = jnp.asarray(obstacle_specs, dtype=jnp.float64)
    n_obs = obstacles.shape[0]
    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    nc = 2 * n_u + 2 * n + n_obs

    if interpolated_reference:
        progress = jnp.linspace(0.0, 1.0, N + 1, dtype=x0.dtype)
        X_ref = (
            (1.0 - progress[:, None]) * x0[None, :]
            + progress[:, None] * x_goal[None, :]
        )
        # A zero-endpoint lateral bias breaks the otherwise symmetric left/right
        # Dubins local minima without changing the initial state or final goal.
        X_ref = X_ref.at[:, 0].add(route_bias_m * jnp.sin(jnp.pi * progress))
    else:
        X_ref = jnp.tile(x_goal[None, :], (N + 1, 1))
    reference = X_ref
    T_steps = N

    if shared_physical_nominal_solve:
        # Compute one formulation-independent physical nominal plan. This
        # avoids allowing augmented-state bookkeeping to select a different
        # left/right local minimum before the robust formulations diverge.
        nominal_cfg = MPCConfig(
            n=n_x,
            nu=n_u,
            N=N,
            W=jnp.concatenate([W_state, W_ctrl]),
            u_ref=jnp.zeros((n_u,), dtype=jnp.float64),
            dt=dt,
        )
        nominal_sls_cfg = SLSConfig(
            max_sls_iterations=2,
            sls_primal_tol=1e-2,
            enable_fastsls=False,
            n_x=n_x,
            n_w=n_x,
            n_theta=0,
            num_param=num_params,
            q_max=q_max,
            adaptive=False,
            enable_linearization_error=False,
        )
        nominal_constraints = combine_constraints(
            make_state_box_constraints(x_min[:n_x], x_max[:n_x]),
            constraints_u,
        )
        nominal_disturbance = make_state_varying_disturbance(
            n=n_x,
            alpha=0.0,
            adaptive=False,
            dt=dt,
            information_center=information_center,
        )
        controller = GenericMPCControllerWrapper(
            nominal_sls_cfg,
            sqp_cfg,
            admm_cfg,
            config=nominal_cfg,
            dynamics=nonadaptive_dynamics,
            constraints=nominal_constraints,
            obstacles=obstacles,
            cost=partial(
                cost,
                adaptive=False,
                terminal_state_weights=terminal_weights_array,
                terminal_time=N,
            ),
            num_constraints=2 * n_u + 2 * n_x + n_obs,
            disturbance=nominal_disturbance,
            limited_memory=False,
            shift=1,
            X_in=jnp.zeros((N + 1, n_x), dtype=jnp.float64),
            U_in=jnp.zeros((N, n_u), dtype=jnp.float64),
            terminal_constraints=terminal_constraint_fn,
            num_terminal_constraints=num_terminal_constraints,
        )
        nominal_x0 = x0[:n_x]
        nominal_reference = reference[:, :n_x]
    else:
        controller = GenericMPCControllerWrapper(
            sls_cfg, sqp_cfg, admm_cfg, config=cfg, dynamics=dyn_fn,
            constraints=constraints_all, obstacles=obstacles, cost=cost_fn,
            num_constraints=nc, disturbance=nominal_disturbance,
            limited_memory=False, shift=1,
            X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
            U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64),
            terminal_constraints=terminal_constraint_fn,
            num_terminal_constraints=num_terminal_constraints,
        )
        nominal_x0 = x0
        nominal_reference = reference

    key = jax.random.PRNGKey(0)
    plans_xy, lowers_xy, uppers_xy = [], [], []
    total_time = 0
    
    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, K_kjN, E_prev, K_prev, G_prev, P_inv_prev= controller.run(
        x0=nominal_x0, reference=nominal_reference, parameter=parameter,
        G_0=nominal_G_0, nominal_param=nominal_param
    )
    # JAX dispatch is asynchronous.  Synchronize before stopping the timer so
    # the recorded runtime is the actual solve time rather than dispatch time.
    u0.block_until_ready()
    end = time.perf_counter()
    total_time += (end - start)

    initial_nominal_plan_csv = run_dir / "initial_nominal_plan.csv"
    with initial_nominal_plan_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for step, state in enumerate(np.asarray(X_pred[:, :n_x])):
            writer.writerow([step, step * dt, *map(float, state)])

    if shared_physical_nominal_solve and adaptive:
        nominal_parameters = jnp.tile(nominal_param, (N + 1, 1))
        X_pred = jnp.concatenate([X_pred, nominal_parameters], axis=-1)
        V_pred = jnp.concatenate([V_pred, jnp.zeros_like(nominal_parameters)], axis=-1)

    if track_nominal_during_robust_solve:
        reference = X_pred.at[-1].set(x_goal)

    # -----------------------------
    # Update configs for robust run
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e6,
        max_iterations=150 if adaptive else 400,
    )
    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        warm_start=True,
        n_x=n_x,
        n_w=n_x,
        n_theta=n_theta,
        num_param=num_params,
        q_max=q_max,
        adaptive=adaptive,
        enable_linearization_error=enable_leb,
        enable_information_cost=enable_information_cost,
        information_cost_weight=information_cost_weight,
        information_cost_discount=information_cost_discount,
    )
    sqp_cfg = SQPConfig(warm_start=True, max_sqp_iterations=100, line_search=True)
    Q_f_bar = (
        jnp.zeros((n, n), dtype=jnp.float64)
        .at[0, 0].set(1.0)
        .at[1, 1].set(1.0)
    )
    # Bias the SLS response design toward rejecting x/y deviations.  The
    # remaining augmented-state coordinates retain the unit baseline weight.
    Q_bar = sls_q_bar_scale * (
        jnp.eye(n, dtype=jnp.float64)
        .at[0, 0].set(10.0)
        .at[1, 1].set(10.0)
    )
    # Keep the input-response regularization at its original scale. Increasing
    # Q_bar relative to R_bar prioritizes smaller state tubes.
    R_bar = sls_r_bar_scale * jnp.eye(n_u, dtype=jnp.float64)

    controller = GenericMPCControllerWrapper(
        sls_cfg, sqp_cfg, admm_cfg, config=cfg, dynamics=dyn_fn, constraints=constraints_all,
        obstacles=obstacles, cost=cost_fn, num_constraints=nc, disturbance=disturbance,
        limited_memory=False, shift=1, X_in=X_pred, U_in=U_pred,
        Q_bar=Q_bar,
        R_bar=R_bar,
        Q_f_bar=Q_f_bar,
        terminal_constraints=terminal_constraint_fn,
        num_terminal_constraints=num_terminal_constraints,
    )

    N_ROLLOUTS = NUM_RANDOM + NUM_ADV

    if not adaptive:
        G_0_reduced = girard_reduction(G_0, q_max)
        G_0 = G_0_reduced

    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, K_kjN, E_prev, K_prev, G_prev, P_inv_prev = controller.run(
        x0=x0, reference=reference, parameter=parameter, G_0=G_0, nominal_param=nominal_param
    )
    u0.block_until_ready()
    end = time.perf_counter()
    total_time += end - start
    K_kjN = compute_state_feedback_gains(Phi_x, Phi_u, n_x + n_theta, n_u)
    final_A, final_B, verification = verify_final_sls_response(
        Phi_x=Phi_x,
        Phi_u=Phi_u,
        X=X_pred,
        U=U_pred,
        adaptive=adaptive,
        nominal_param=nominal_param,
        dt=dt,
        tolerance=1e-2,
    )
    verification.update(
        {
            "max_sls_iterations": 2,
            "sls_primal_tolerance": 1e-2,
            "tube_uses_returned_response": True,
        }
    )
    (run_dir / "verification_summary.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    # -----------------------------
    # Prepare Variables for Visualizer
    # -----------------------------
    if adaptive:
        sensitivty_inv = P_inv_prev 
        sensitivity = jnp.linalg.pinv(sensitivty_inv)
        exog = E_prev[:, :n_x, :n_x]
        W_prev = jnp.einsum("tij,tjk->tik", sensitivty_inv, exog[:-1])
        
        # REMOVED the front-padding of K_prev and sensitivity here! 
        # K_prev[0] is now correctly the gain for the first step.

    if adaptive:
        sens_inv = P_inv_prev 
        sensitivity = jnp.linalg.pinv(sens_inv)
        
        n_w_full = n_x * 2
        exog = E_prev[:, :n_x, :n_w_full] 
        W_prev = jnp.einsum("tij,tjk->tik", sens_inv, exog[:-1]) 
        
        # Pad transition-indexed estimator quantities at the back to length
        # T+1.  ``exog[j]`` already describes the disturbance applied during
        # transition j -> j+1; the visualizer performs the corresponding Phi
        # column shift internally, so front-padding exog would delay the tube
        # disturbance by one step relative to the rollout.
        K_padded = jnp.concatenate([K_prev, jnp.zeros((1, n_theta, n_theta))], axis=0) 
        W_padded = jnp.concatenate([W_prev, jnp.zeros((1, n_theta, n_w_full))], axis=0) 
        E_padded = jnp.concatenate([sensitivity, jnp.zeros((1, n_x, n_theta))], axis=0) 
        F_padded = exog

        tube = get_adaptive_trajectory_tubes_exact(Phi_x, F_padded, E_padded, K_padded, W_padded, G_prev[0])
    else:
        tube = get_trajectory_tubes(Phi_x, E_prev)    

    saved_arrays = {
        "nominal_states": X_pred,
        "nominal_inputs": U_pred,
        "state_response": Phi_x,
        "input_response": Phi_u,
        "state_feedback_gains": K_kjN,
        "disturbance_map": E_prev,
        "state_tube_half_widths": tube,
    }
    nonfinite_arrays = [
        name for name, value in saved_arrays.items()
        if not np.all(np.isfinite(np.asarray(value)))
    ]
    verification["saved_arrays_finite"] = not nonfinite_arrays
    verification["nonfinite_saved_arrays"] = nonfinite_arrays
    (run_dir / "verification_summary.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    if nonfinite_arrays:
        raise FloatingPointError(
            f"Refusing to save an invalid controller; non-finite arrays: {nonfinite_arrays}"
        )

    # Save the complete solve-once feedback law needed for reproducible
    # out-of-sample rollouts. CSV outputs alone do not contain the causal SLS
    # gains or the adaptive estimator update matrices.
    np.savez_compressed(
        run_dir / "controller.npz",
        controller_type=np.asarray("adaptive_leb_sls" if adaptive else "nonadaptive_sls"),
        nominal_states=np.asarray(X_pred),
        nominal_inputs=np.asarray(U_pred),
        state_feedback_gains=np.asarray(K_kjN),
        state_response=np.asarray(Phi_x),
        input_response=np.asarray(Phi_u),
        final_linearization_A=final_A,
        final_linearization_B=final_B,
        estimator_gains=np.asarray(K_prev),
        estimator_sensitivity_inverses=np.asarray(P_inv_prev),
        parameter_generators=np.asarray(G_prev),
        initial_parameter_center=np.asarray(nominal_param),
        initial_parameter_generators=np.asarray(G_0),
        state_tube_half_widths=np.asarray(tube),
        model_label=np.asarray(model_label),
        start_state=np.asarray(x0[:n_x]),
        goal_state=np.asarray(x_goal[:n_x]),
        obstacles=np.asarray(obstacle_specs),
        speed_limits=np.asarray([minimum_speed_m_per_s, maximum_speed_m_per_s]),
        dt=np.asarray(dt),
        parameter_uncertainty_bound=np.asarray(parameter_uncertainty_bound),
        exogenous_disturbance_scale=np.asarray(exogenous_disturbance_scale),
        input_lower=np.asarray(u_min),
        input_upper=np.asarray(u_max),
        adaptive=np.asarray(adaptive),
        linearization_error_bound_enabled=np.asarray(enable_leb),
    )
    
    plan_xy = X_pred[:, :2]
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    # This is the robust nominal trajectory used to center the plotted tubes
    # and to compute the closed-loop rollout deviations.  Keep the preliminary
    # nominal solve separately in ``initial_nominal_plan.csv``.
    nominal_plan_csv = run_dir / "nominal_plan.csv"
    with nominal_plan_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for step, state in enumerate(np.asarray(X_pred[:, :n_x])):
            writer.writerow([step, step * dt, *map(float, state)])

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    # -----------------------------
    # Rollout simulations
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, n_x), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32) 
    for i in range(N_ROLLOUTS):
        # We now track state deviations (\Delta x) instead of disturbances (w)
        state_deviation_history = [] 
        x = x0

        for k in range(T_steps):
            if bool(reached_goal_xy(x, x_goal, GOAL_TOL)):
                stop_steps[i] = k
                break

            # ---------------------------------------------------------
            # 1. Calculate Current State Deviation
            # ---------------------------------------------------------
            delta_x = x - X_pred[k]
            
            # Wrap the angular deviation to remain in the linear tangent space
            angle_err = wrap_to_pi(delta_x[2]) 
            delta_x = delta_x.at[2].set(angle_err)
            
            state_deviation_history.append(delta_x)

            # ---------------------------------------------------------
            # 2. Apply SLS Control Policy using K = Phi_u @ Phi_x^-1
            # ---------------------------------------------------------
            state_feedback = jnp.zeros((n_u,), dtype=jnp.float64)
            for j in range(0, k+1):
                # K_kjN contains the optimized state feedback gains
                state_feedback = state_feedback + K_kjN[k, j] @ state_deviation_history[j]

            u = U_pred[k] + state_feedback

            # ---------------------------------------------------------
            # 3. Step True Physics
            # ---------------------------------------------------------
            prev_x = x
            key, x, w = dubins_step_with_disturbance(key, x, u, true_param, nominal_param, G_0, exog_dist, dt, i, adaptive)

            # ---------------------------------------------------------
            # 4. Measurement Update w.r.t the Linearized System
            # ---------------------------------------------------------
            if adaptive:
                # Evaluate Jacobians of the dynamics exactly at the nominal plan
                A_k = jax.jacfwd(dynamics, argnums=0)(X_pred[k], U_pred[k], None, parameter=dt)
                B_k = jax.jacfwd(dynamics, argnums=1)(X_pred[k], U_pred[k], None, parameter=dt)
                
                # Compute the linear prediction of the state deviation
                delta_u = u - U_pred[k]
                expected_linear_delta_x = A_k @ delta_x + B_k @ delta_u
                
                # Reconstruct the absolute linear prediction of the physical state
                x_pred_lin = X_pred[k + 1][:n_x] + expected_linear_delta_x[:n_x]
                
                # The measurement natively calculates: innovation = sensitivity @ (x_next - x_pred_lin)
                new_param, delta_param = measurement(
                    x_pred_lin=x_pred_lin, 
                    x_next=x[:n_x], 
                    kalman_gain=K_prev[k], 
                    sensitivity=P_inv_prev[k], 
                    current_param=prev_x[n_x:n_x+n_theta]
                ) 
                # Inject updated parameter into the true augmented state
                x = x.at[n_x:].set(new_param)

            # ---------------------------------------------------------
            # 5. Log Deviations
            # ---------------------------------------------------------
            disturbed[i, k, :2] = np.abs(np.asarray(X_pred[k + 1, :2] - x[:2]))
            disturbed[i, k, 2]  = np.abs(np.asarray(X_pred[k + 1, 2] - x[2]))
            disturbed[i, k, 3]  = np.abs(np.asarray(X_pred[k + 1, 3] - x[3]))
            
            xs[i, k] = np.asarray(x)

    # -----------------------------
    # Deviation vs tube size plots
    # -----------------------------
    # Prepend the initial state x0 to the rollout history so it matches t=0
    x0_expanded = np.tile(x0, (N_ROLLOUTS, 1, 1))
    xs = np.concatenate([x0_expanded, xs], axis=1)

    # FIRST extract the arrays from disturbed
    dx_np_all  = disturbed[:, :, 0]
    dy_np_all  = disturbed[:, :, 1]
    dth_np_all = disturbed[:, :, 2]
    dv_np_all  = disturbed[:, :, 3]

    # THEN prepend the true initial state (0.0 deviation)
    dx_np_all = np.concatenate([np.zeros((N_ROLLOUTS, 1)), dx_np_all], axis=1)
    dy_np_all = np.concatenate([np.zeros((N_ROLLOUTS, 1)), dy_np_all], axis=1)
    dth_np_all = np.concatenate([np.zeros((N_ROLLOUTS, 1)), dth_np_all], axis=1)
    dv_np_all = np.concatenate([np.zeros((N_ROLLOUTS, 1)), dv_np_all], axis=1)

    tube_x_np  = np.asarray(tube[:, 0])
    tube_y_np  = np.asarray(tube[:, 1])
    tube_th_np = np.asarray(tube[:, 2])
    tube_v_np  = np.asarray(tube[:, 3])

    tube_csv = run_dir / "tube_widths.csv"
    with tube_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "step", "time_s", "x_half_width_m", "y_half_width_m",
            "heading_half_width_rad", "speed_half_width_m_per_s",
            "adaptive", "leb",
        ])
        for step, widths in enumerate(np.asarray(tube[:, :4])):
            writer.writerow([
                step, step * dt, *map(float, widths), adaptive, enable_leb,
            ])

    t = np.arange(dx_np_all.shape[1]) * dt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    ax1.plot(t, tube_x_np, label="tube half-width (x position)", linewidth=4)
    for r, dx_np in enumerate(dx_np_all):
        m = np.isfinite(dx_np)
        ax1.plot(t[m], dx_np[m], label="Rollout" if r == 0 else None)
    ax1.set_ylabel("meters")
    ax1.set_title("X Position: Deviation vs Tube Half-Width")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, tube_y_np, label="tube half-width (y position)", linewidth=4)
    for r, dy_np in enumerate(dy_np_all):
        m = np.isfinite(dy_np)
        ax2.plot(t[m], dy_np[m], label="Rollout" if r == 0 else None)
    ax2.set_ylabel("meters")
    ax2.set_title("Y Position: Deviation vs Tube Half-Width")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(t, tube_th_np, label="tube half-width (heading)", linewidth=4)
    for r, dth_np in enumerate(dth_np_all):
        m = np.isfinite(dth_np)
        ax3.plot(t[m], dth_np[m], label="Rollout" if r == 0 else None)
    ax3.set_ylabel("radians")
    ax3.set_title("Heading: Deviation vs Tube Half-Width")
    ax3.grid(True)
    ax3.legend()

    ax4.plot(t, tube_v_np, label="tube half-width (speed)", linewidth=4)
    for r, dv_np in enumerate(dv_np_all):
        m = np.isfinite(dv_np)
        ax4.plot(t[m], dv_np[m], label="Rollout" if r == 0 else None)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("m/s")
    ax4.set_title("Speed: Deviation vs Tube Half-Width")
    ax4.grid(True)
    ax4.legend()
    plt.tight_layout()
    fig.suptitle(f"{model_label} Tubes (adaptive={adaptive}, LEB={enable_leb})", y=1.002)
    plt.savefig(run_dir / "deviation_vs_tube_width.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if adaptive:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for r in range(N_ROLLOUTS):
            m = np.isfinite(xs[r, :, 4])
            ax1.plot(t[m], xs[r, m, 4], label="om_bias" if r == 0 else None, color="tab:blue", alpha=0.5)
            m2 = np.isfinite(xs[r, :, 5])
            ax2.plot(t[m2], xs[r, m2, 5], label="ac_bias" if r == 0 else None, color="tab:orange", alpha=0.5)

        ax1.axhline(y=float(true_param[0]), color='r', linestyle='--', label='true om_bias')
        ax2.axhline(y=float(true_param[1]), color='r', linestyle='--', label='true ac_bias')

        ax1.set_ylabel("om_bias")
        ax1.set_title("Omega Bias over Time")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("ac_bias")
        ax2.set_title("Accel Bias over Time")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(run_dir / "parameter_estimates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    xs0 = xs[0]
    rollout_csv = run_dir / "rollout.csv"
    with rollout_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"]
        )
        for step, state in enumerate(xs0[:, :n_x]):
            writer.writerow([step, step * dt, *map(float, state)])

    # Preserve all rollout states (including adaptive parameter estimates) and
    # the solve runtime.  The compact CSV above remains convenient for plots;
    # this archive makes paper metrics exactly reproducible.
    np.savez_compressed(
        run_dir / "rollout_metrics.npz",
        rollout_states=xs,
        nominal_states=np.asarray(X_pred),
        state_tube_half_widths=np.asarray(tube),
        goal_state=np.asarray(x_goal[:n_x]),
        true_parameters=np.asarray(true_param),
        parameter_tube_half_widths=(
            np.concatenate(
                [
                    np.sum(np.abs(np.asarray(G_prev)), axis=-1),
                    np.sum(np.abs(np.asarray(G_prev[-1:])), axis=-1),
                ],
                axis=0,
            )
            if adaptive
            else np.empty((0, num_params))
        ),
        runtime_s=np.asarray(total_time),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_s": float(total_time),
                "num_rollouts": int(N_ROLLOUTS),
                "adaptive": bool(adaptive),
                "linearization_error_bound_enabled": bool(enable_leb),
            },
            indent=2,
        )
        + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs0[:, 0], xs0[:, 1], label="rollout trajectory", color="tab:blue", linewidth=2.25)
    ax.plot(plan_xy[:, 0], plan_xy[:, 1], label="planned trajectory", color="tab:orange", linestyle="--", linewidth=2.25)
    ax.plot(
        [float(x0[0]), float(x_goal[0])],
        [float(x0[1]), float(x_goal[1])],
        color="black",
        linestyle=":",
        linewidth=1.75,
        label="direct path",
    )
    ax.scatter(
        [information_center[0]],
        [information_center[1]],
        marker="*",
        s=220,
        color="tab:green",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label="information center",
    )
    for obstacle_index, (center, radius) in enumerate(
        zip(np.asarray(centers), np.asarray(radii))
    ):
        ax.add_patch(
            Circle(
                tuple(center),
                float(radius),
                facecolor="tab:red",
                edgecolor="black",
                alpha=0.35,
                zorder=5,
                label="obstacle" if obstacle_index == 0 else None,
            )
        )
    for k in range(0, tube.shape[0], 5):
        w = tube[k, 0] * 2
        h = tube[k, 1] * 2
        if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
            continue
        rect = Rectangle((plan_xy[k, 0] - tube[k, 0], plan_xy[k, 1] - tube[k, 1]), w, h, alpha=0.1)
        ax.add_patch(rect)
    ax.add_patch(Rectangle((0, 0), 0, 0, alpha=0.1, label="tube"))
# ... inside main() ...
    ax.set_xlabel("x position (m)", fontsize=22)
    ax.set_ylabel("y position (m)", fontsize=22)
    mode_label = "Adaptive" if adaptive else "Non-adaptive"
    leb_label = "with LEB" if enable_leb else "without LEB"
    ax.set_title(f"{model_label}: {mode_label}, {leb_label}", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    
    # Frame the complete experiment: nominal/rollout paths, tubes, targets,
    # and every obstacle (including its radius).
    obstacle_extents = np.concatenate(
        [np.asarray(centers) - np.asarray(radii)[:, None],
         np.asarray(centers) + np.asarray(radii)[:, None]],
        axis=0,
    )
    visual_points = np.concatenate(
        [
            np.asarray(plan_xy),
            np.asarray(lower),
            np.asarray(upper),
            xs[:, :, :2].reshape(-1, 2),
            np.asarray([[x0[0], x0[1]], [x_goal[0], x_goal[1]], information_center]),
            obstacle_extents,
        ],
        axis=0,
    )
    visual_points = visual_points[np.all(np.isfinite(visual_points), axis=1)]
    margin = 0.5
    ax.set_xlim(np.min(visual_points[:, 0]) - margin, np.max(visual_points[:, 0]) + margin)
    ax.set_ylim(np.min(visual_points[:, 1]) - margin, np.max(visual_points[:, 1]) + margin)
    
    plt.tight_layout()
    plt.savefig(run_dir / "rollout_with_tubes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved tube widths to {tube_csv}")
    return tube_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the straight-down active-information Dubins-car experiment."
    )
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leb", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the nonlinear linearization-error bound.")
    parser.add_argument("--all-combinations", action="store_true",
                        help="Run adaptive/non-adaptive with and without LEB.")
    parser.add_argument(
        "--leb-only",
        action="store_true",
        help="Run only adaptive and non-adaptive cases with LEB enabled.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("active_car_information_center_results"))
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Discrete dynamics time step in seconds.")
    parser.add_argument(
        "--exogenous-disturbance-scale",
        type=float,
        default=0.0004,
        help="Per-step isotropic exogenous disturbance-generator scale.",
    )
    parser.add_argument(
        "--parameter-uncertainty-bound",
        type=float,
        default=0.10,
        help="Initial independent half-width for each unknown parameter.",
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
        help="Penalize the squared Frobenius norm of E-dagger F stage-wise.",
    )
    parser.add_argument("--edagger-f-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--terminal-position-constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce the nominal terminal x-y position at the goal.",
    )
    parser.add_argument("--terminal-position-tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--terminal-x-weight",
        type=float,
        default=DEFAULT_TERMINAL_STATE_WEIGHTS[0],
    )
    parser.add_argument(
        "--terminal-y-weight",
        type=float,
        default=DEFAULT_TERMINAL_STATE_WEIGHTS[1],
    )
    parser.add_argument(
        "--nonadaptive-terminal-x-weight",
        type=float,
        default=None,
        help="Optional terminal x-cost weight used only by non-adaptive runs.",
    )
    parser.add_argument(
        "--nonadaptive-terminal-y-weight",
        type=float,
        default=None,
        help="Optional terminal y-cost weight used only by non-adaptive runs.",
    )
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=-1.0)
    parser.add_argument("--initial-speed", type=float, default=0.0)
    parser.add_argument("--goal-speed", type=float, default=0.0)
    parser.add_argument("--heading-min", type=float, default=-float("inf"))
    parser.add_argument("--heading-max", type=float, default=float("inf"))
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--information-center-x", type=float, default=DEFAULT_INFORMATION_CENTER[0])
    parser.add_argument("--information-center-y", type=float, default=DEFAULT_INFORMATION_CENTER[1])
    parser.add_argument("--obstacle-x", type=float, default=DEFAULT_OBSTACLE[0])
    parser.add_argument("--obstacle-y", type=float, default=DEFAULT_OBSTACLE[1])
    parser.add_argument("--obstacle-radius", type=float, default=DEFAULT_OBSTACLE[2])
    parser.add_argument("--second-obstacle-x", type=float, default=None)
    parser.add_argument("--second-obstacle-y", type=float, default=None)
    parser.add_argument("--second-obstacle-radius", type=float, default=DEFAULT_SECOND_OBSTACLE[2])
    parser.add_argument("--third-obstacle-x", type=float, default=None)
    parser.add_argument("--third-obstacle-y", type=float, default=None)
    parser.add_argument("--third-obstacle-radius", type=float, default=DEFAULT_THIRD_OBSTACLE[2])
    parser.add_argument(
        "--extra-obstacle",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "RADIUS"),
        default=[],
        help="Additional circular obstacle; may be specified more than once.",
    )
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.exogenous_disturbance_scale < 0.0:
        parser.error("--exogenous-disturbance-scale must be nonnegative")
    if args.parameter_uncertainty_bound < 0.0:
        parser.error("--parameter-uncertainty-bound must be nonnegative")
    if args.x_min >= args.x_max:
        parser.error("--x-min must be smaller than --x-max")
    if args.heading_min >= args.heading_max:
        parser.error("--heading-min must be smaller than --heading-max")
    if args.information_cost and not args.adaptive:
        parser.error("--information-cost requires --adaptive")
    if args.edagger_f_cost and not args.adaptive:
        parser.error("--edagger-f-cost requires --adaptive")
    if args.information_cost and args.edagger_f_cost:
        parser.error("--information-cost and --edagger-f-cost are mutually exclusive")
    if args.information_cost and (args.all_combinations or args.leb_only):
        parser.error("--information-cost cannot be combined with a multi-run mode")
    if args.edagger_f_cost and (args.all_combinations or args.leb_only):
        parser.error("--edagger-f-cost cannot be combined with a multi-run mode")
    if args.information_cost_weight < 0.0:
        parser.error("--information-cost-weight must be nonnegative")
    if args.edagger_f_cost_weight < 0.0:
        parser.error("--edagger-f-cost-weight must be nonnegative")
    if not 0.0 < args.information_cost_discount <= 1.0:
        parser.error("--information-cost-discount must lie in (0, 1]")
    if args.terminal_position_tolerance < 0.0:
        parser.error("--terminal-position-tolerance must be nonnegative")
    if args.terminal_x_weight < 0.0 or args.terminal_y_weight < 0.0:
        parser.error("terminal x/y weights must be nonnegative")
    if (
        args.nonadaptive_terminal_x_weight is not None
        and args.nonadaptive_terminal_x_weight < 0.0
    ):
        parser.error("--nonadaptive-terminal-x-weight must be nonnegative")
    if (
        args.nonadaptive_terminal_y_weight is not None
        and args.nonadaptive_terminal_y_weight < 0.0
    ):
        parser.error("--nonadaptive-terminal-y-weight must be nonnegative")
    if args.obstacle_radius <= 0.0:
        parser.error("--obstacle-radius must be positive")
    if (args.second_obstacle_x is None) != (args.second_obstacle_y is None):
        parser.error("--second-obstacle-x and --second-obstacle-y must be provided together")
    if args.second_obstacle_x is not None and args.second_obstacle_radius <= 0.0:
        parser.error("--second-obstacle-radius must be positive")
    if (args.third_obstacle_x is None) != (args.third_obstacle_y is None):
        parser.error("--third-obstacle-x and --third-obstacle-y must be provided together")
    if args.third_obstacle_x is not None and args.third_obstacle_radius <= 0.0:
        parser.error("--third-obstacle-radius must be positive")
    if any(radius <= 0.0 for _, _, radius in args.extra_obstacle):
        parser.error("--extra-obstacle RADIUS values must be positive")

    additional_obstacles = tuple(
        obstacle for obstacle in (
            (args.second_obstacle_x, args.second_obstacle_y, args.second_obstacle_radius),
            (args.third_obstacle_x, args.third_obstacle_y, args.third_obstacle_radius),
            *args.extra_obstacle,
        ) if obstacle[0] is not None
    )

    if args.leb_only:
        combinations = [(True, True), (False, True)]
    elif args.all_combinations:
        combinations = [(adaptive, leb) for adaptive in (True, False) for leb in (True, False)]
    else:
        combinations = [(args.adaptive, args.leb)]
    csv_paths = []
    for adaptive_value, leb_value in combinations:
        terminal_x_weight = (
            args.terminal_x_weight
            if adaptive_value or args.nonadaptive_terminal_x_weight is None
            else args.nonadaptive_terminal_x_weight
        )
        terminal_y_weight = (
            args.terminal_y_weight
            if adaptive_value or args.nonadaptive_terminal_y_weight is None
            else args.nonadaptive_terminal_y_weight
        )
        csv_paths.append(main(
            adaptive=adaptive_value,
            enable_leb=leb_value,
            output_dir=args.output_dir,
            horizon=args.horizon,
            time_step_s=args.dt,
            exogenous_disturbance_scale=args.exogenous_disturbance_scale,
            parameter_uncertainty_bound=args.parameter_uncertainty_bound,
            goal_x_m=args.goal_x,
            goal_y_m=args.goal_y,
            initial_speed_m_per_s=args.initial_speed,
            goal_speed_m_per_s=args.goal_speed,
            heading_min_rad=args.heading_min,
            heading_max_rad=args.heading_max,
            x_min_m=args.x_min,
            x_max_m=args.x_max,
            terminal_state_weights=(
                terminal_x_weight,
                terminal_y_weight,
                DEFAULT_TERMINAL_STATE_WEIGHTS[2],
                DEFAULT_TERMINAL_STATE_WEIGHTS[3],
            ),
            enable_information_cost=args.information_cost,
            information_cost_weight=args.information_cost_weight,
            information_cost_discount=args.information_cost_discount,
            enable_edagger_f_cost=args.edagger_f_cost,
            edagger_f_cost_weight=args.edagger_f_cost_weight,
            enforce_terminal_position=args.terminal_position_constraint,
            terminal_position_tolerance=args.terminal_position_tolerance,
            information_center=(
                args.information_center_x,
                args.information_center_y,
            ),
            obstacle=(
                args.obstacle_x,
                args.obstacle_y,
                args.obstacle_radius,
            ),
            additional_obstacles=additional_obstacles,
        ))

    if args.all_combinations or args.leb_only:
        combined_csv = args.output_dir / (
            "tube_widths_leb_only.csv" if args.leb_only else "tube_widths_all_combinations.csv"
        )
        with combined_csv.open("w", newline="") as combined_file:
            writer = None
            for csv_path in csv_paths:
                with csv_path.open(newline="") as source_file:
                    reader = csv.DictReader(source_file)
                    if writer is None:
                        writer = csv.DictWriter(combined_file, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
        print(f"Saved combined tube widths to {combined_csv}")
