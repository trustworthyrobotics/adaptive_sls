"""Configuration for the simulation-first 12D effective-gain experiment."""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import jax

try:  # Package import for tests and installed use.
    from .dynamics_12d import augmented_step
except ImportError:  # Direct-script import from this experiment directory.
    from dynamics_12d import augmented_step


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = str(Path(dir_path) / "vendor" / "bitcraze_crazyflie_2" / "scene.xml")
source_model_mass = 0.027
nominal_mass = 0.029
gravity = 9.81

dt = 0.030
simulation_dt = 0.002
# Match the quadruped RTI horizon for a direct timing comparison.  The
# controller period remains 30 ms, so this is a 360 ms prediction window.
N = 12
mpc_frequency = 1.0 / dt
n_phys, n_theta, n, m, nu, n_w, q_max = 12, 1, 13, 4, 4, 12, 3

# The simulator uses Newton thrust, but beta is deliberately named and logged
# as an effective gain so the same experiment remains honest on hardware.
beta_min = 1.0 / (1.2 * nominal_mass)
beta_max = 1.0 / (0.8 * nominal_mass)
beta_hat_initial = 0.5 * (beta_min + beta_max)
beta_half_width = 0.5 * (beta_max - beta_min)

start_position = jnp.array([0.0, 0.0, 0.50], dtype=jnp.float32)
goal_position = jnp.array([2.0, 0.0, 0.70], dtype=jnp.float32)
trajectory_duration = 8.0
goal_radius = 0.20
# The goal itself is higher than the start.  Unlike the older experiment, it
# is not a time-indexed tracking trajectory.
probe_amplitude = 0.010  # N, added after the solve and logged exactly
probe_frequency_hz = 0.5
initial_state = jnp.concatenate([start_position, jnp.zeros(9), jnp.array([beta_hat_initial])])

# Initial values; validate_model.py is responsible for fitting these against
# the simulator before interpreting beta estimates.
k_phi_p = 625.0
k_phi_d = 50.0
k_theta_p = 625.0
k_theta_d = 50.0
k_r = 12.0
model_kwargs = dict(gravity=gravity, k_phi_p=k_phi_p, k_phi_d=k_phi_d, k_theta_p=k_theta_p, k_theta_d=k_theta_d, k_r=k_r)

angle_max = jnp.deg2rad(jnp.asarray(5.0, dtype=jnp.float32))
yawrate_max = jnp.deg2rad(jnp.asarray(45.0, dtype=jnp.float32))
thrust_min, thrust_max = 0.0, 0.060 * gravity
moment_gear_scale = 300.0
control_lower = jnp.array([thrust_min, -angle_max, -angle_max, -yawrate_max])
control_upper = jnp.array([thrust_max, angle_max, angle_max, yawrate_max])
state_lower = jnp.array([-2., -2., .20, -3., -3., -3., -angle_max, -angle_max, -jnp.pi, -4., -4., -4., beta_min])
state_upper = jnp.array([2.5, 2., 2., 3., 3., 3., angle_max, angle_max, jnp.pi, 4., 4., 4., beta_max])

# Ported from experiments/quadrotor/quadrotor_common.py, reordered from its
# [position, Euler, velocity, body-rate] layout to ours
# [position, velocity, Euler, body-rate].
position_weights = jnp.array([25., 25., 25.], dtype=jnp.float32)
velocity_weights = jnp.array([.5, .5, .5], dtype=jnp.float32)
attitude_weights = jnp.array([2., 2., .5], dtype=jnp.float32)
rate_weights = jnp.array([.05, .05, .05], dtype=jnp.float32)
control_weights = jnp.array([.001, .01, .01, .01], dtype=jnp.float32)
Q = jnp.diag(jnp.concatenate([position_weights, velocity_weights, attitude_weights, rate_weights]))
R = jnp.diag(control_weights)
W = {"state": Q, "control": R}
u_ref = jnp.array([gravity / beta_hat_initial, 0., 0., 0.], dtype=jnp.float32)
# The MPC wrapper requires a square gate.  Its only nonzero rows select
# world-frame velocity, which is equivalent to the intended 3-by-12 C gate.
measurement_matrix = jnp.diag(jnp.array([0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.], dtype=jnp.float32))

first_disturbance = jnp.array([.002, .002, .002, .008, .020, .020, .004, .004, .004, .08, .08, .08])
later_disturbance = first_disturbance


def dynamics(x, u, t, *, parameter):
    del t
    return augmented_step(x, u, parameter, **model_kwargs)


def cost(W, reference, x, u, t):
    # ``reference`` is retained solely for the generic-wrapper signature.
    # This is goal regulation, not trajectory tracking.
    del reference, t
    goal = jnp.concatenate([goal_position, jnp.zeros(9, dtype=x.dtype)])
    error = x[:n_phys] - goal
    hover = jnp.array([gravity / jnp.maximum(x[12], 1e-6), 0., 0., 0.], dtype=x.dtype)
    control_error = u - hover
    # Treat Euler error periodically, as in the tuned 12D quadrotor cost.
    quadratic_error = jnp.concatenate([error[:6], jnp.zeros(3, dtype=x.dtype), error[9:12]])
    quadratic_cost = quadratic_error @ W["state"] @ quadratic_error
    attitude_cost = jnp.sum(attitude_weights * (1. - jnp.cos(error[6:9])))
    return quadratic_cost + attitude_cost + control_error @ W["control"] @ control_error


@jax.jit
def goal_horizon(beta_hat: jnp.ndarray) -> jnp.ndarray:
    """Structurally required horizon array; the cost itself is goal-only."""
    goal = jnp.concatenate([goal_position, jnp.zeros(9, dtype=jnp.float32), jnp.ravel(beta_hat)])
    return jnp.broadcast_to(goal, (N + 1, n))


def initial_generator():
    return jnp.pad(jnp.array([[beta_half_width]], dtype=jnp.float32), ((0, 0), (0, q_max - 1)))


def initial_seed(beta_hat=None):
    beta_hat = jnp.array([beta_hat_initial], dtype=jnp.float32) if beta_hat is None else beta_hat
    # A feasible hover seed is more useful than a fictitious path to goal.
    start = jnp.concatenate([start_position, jnp.zeros(9, dtype=jnp.float32), jnp.ravel(beta_hat)])
    X_seed = jnp.broadcast_to(start, (N + 1, n))
    U_seed = jnp.zeros((N, m), dtype=jnp.float32).at[:, 0].set(gravity / beta_hat[0])
    return X_seed, U_seed
