# Adapted from https://github.com/iit-DLSLab/mpx/blob/main/mpx/config/config_go2.py

import os
import sys
from functools import partial
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
vendor_root = os.path.join(dir_path, "vendor")
if vendor_root not in sys.path:
    sys.path.insert(0, vendor_root)

import jax.numpy as jnp
import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives
import mpx

mpx_root = Path(mpx.__file__).parent
model_path = str(mpx_root / "data" / "go2" / "go2_mjx.xml")  # Path to the MuJoCo model XML file
# Contact frame names and body names for feet (or calves)
contact_frame = ['FL', 'FR', 'RL', 'RR']
body_name = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

# Time and stage parameters
# dt = 0.02  # Time step in seconds
# N = 25         # Number of stages
# mpc_frequency = 50  # One control and measurement per 0.02 s SLS transition

dt = 0.032  # Time step in seconds
N = 14         # Number of stages
mpc_frequency = 31.25  # One control and measurement per 0.02 s SLS transition


# # Timer values (make sure the values match your intended configuration)
# timer_t =  jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg galop jnp.array([0.25, 0.5, 0.75, 0.0]) crawl jnp.array([0.25, 0.75, 0.0, 0.5])
# duty_factor = 0.65 #0.65  # Duty factor for the gait
# step_freq = 1.35 #1.4   # Step frequency in Hz
# step_height = 0.065 # Step height in meters
# initial_height = 0.1  # Initial height of the robot's base in meters
# robot_height = 0.27  # Height of the robot's base in meters
# clearance_speed = 0.2


# Timer values (make sure the values match your intended configuration)
timer_t =  jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg galop jnp.array([0.25, 0.5, 0.75, 0.0]) crawl jnp.array([0.25, 0.75, 0.0, 0.5])
duty_factor = 0.65 #0.65  # Duty factor for the gait
step_freq = 1.6 #1.4   # Step frequency in Hz
step_height = 0.1 # Step height in meters
initial_height = 0.1  # Initial height of the robot's base in meters
robot_height = 0.27  # Height of the robot's base in meters
clearance_speed = 0.2

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, robot_height])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)   
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])  # Initial joint angles
q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

p_legs0 = jnp.array([
    0.192, 0.142, .0,  # Initial position of the front left leg
    0.192, -0.142, .0, # Initial position of the front right leg
   -0.195, 0.142, .0,  # Initial position of the rear left leg
   -0.195, -0.142, .0  # Initial position of the rear right leg
])

# Determine number of joints and contacts from the lists
n_joints = 12  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n_phys = 13 + 2 * n_joints + 6 * n_contact
n_theta = 2 * n_contact
n = n_phys + n_theta
initial_theta_half_width = jnp.full(n_theta, 1.6, dtype=jnp.float32)
m = n_joints  # Number of controls (F)
grf_as_state = True
foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * n_contact)
leg_slice = foot_slice

# GPU-SLS tube-controller regularization weights. These are separate from
# ``W`` below, which controls the nominal whole-body tracking objective. The
# identity defaults preserve the prior implicit solver defaults.
Q_bar = jnp.eye(n, dtype=jnp.float32).at[:2,:2].set(jnp.eye(2, dtype=jnp.float32) * 10)
R_bar = jnp.eye(m, dtype=jnp.float32)
# R_bar = jnp.diag(
#     jnp.ones(m, dtype=jnp.float32).at[::3].set(0.1)
# )

# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([500, 1000, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1 # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1.5e3, 1e3, 5e3]))  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1e2, 1e2, 1e2]))  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1 # 1e-1  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-2  # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))

W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "contact": Qleg, "tau": Qtau, "grf": Q_grf}

use_terrain_estimation = True  # Flag to use terrain estimation

initial_state = jnp.concatenate(
    [
        p0,
        quat0,
        q0,
        jnp.zeros(6 + n_joints),
        p_legs0,
        jnp.zeros(3 * n_contact),
        jnp.zeros(n_theta),
    ]
)

_physical_cost = partial(
    mpc_objectives.quadruped_wb_obj,
    True,
    n_joints,
    n_contact,
    n_contact,
    N,
)


def cost(W, reference, x, u, t):
    """Whole-body objective with unpenalized constant damping states."""
    return _physical_cost(W, reference, x[:n_phys], u, t)


hessian_approx = None
reference_generator = partial(
    mpc_utils.reference_generator,
    use_terrain_estimation,
    N,
    dt,
    n_joints,
    n_contact,
    foot0=p_legs0,
    q0=q0,
    clearence_speed=clearance_speed,
)

LEG_JOINT_INDEX = {
    "FL": jnp.array([0, 1, 2], dtype=jnp.int32),
    "FR": jnp.array([3, 4, 5], dtype=jnp.int32),
    "RL": jnp.array([6, 7, 8], dtype=jnp.int32),
    "RR": jnp.array([9, 10, 11], dtype=jnp.int32),
}
DAMPED_JOINT_INDEX = jnp.array(
    [
        3 * leg_index + joint_offset
        for leg_index in range(n_contact)
        for joint_offset in (1, 2)
    ],
    dtype=jnp.int32,
)
THETA_LABELS = tuple(
    f"{leg}_{joint}_damping"
    for leg in contact_frame
    for joint in ("thigh", "calf")
)
default_true_theta = jnp.tile(
    jnp.array([1.2, 1.4], dtype=q0.dtype),
    n_contact,
)


def damping_mask(*, dtype=jnp.float32):
    """Map the eight leg-major thigh/calf parameters to joint torques."""
    mask = jnp.zeros((n_joints, n_theta), dtype=dtype)
    return mask.at[DAMPED_JOINT_INDEX, jnp.arange(n_theta)].set(1.0)


def dynamics(model, mjx_model, contact_id, body_id):
    physical_dynamics = partial(
        mpc_dyn_model.quadruped_wb_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        dt,
    )

    mask = damping_mask(dtype=q0.dtype)

    def adaptive_dynamics(x, u, t, *, parameter):
        x_phys = x[:n_phys]
        theta = x[n_phys:]
        joint_velocity = x_phys[13 + n_joints : 13 + 2 * n_joints]
        damped_u = u - (mask @ theta) * joint_velocity
        x_next = physical_dynamics(x_phys, damped_u, t, parameter)
        return jnp.concatenate([x_next, theta])

    return adaptive_dynamics
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
max_torque = 35
min_torque = -35
