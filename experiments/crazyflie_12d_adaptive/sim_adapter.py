"""MuJoCo adapter from 12D planner commands to the existing CF2 plant."""

from __future__ import annotations

import numpy as np
import mujoco
import sys
from pathlib import Path
import jax
import jax.numpy as jnp

try:
    from . import config_crazyflie_12d as config
except ImportError:
    import config_crazyflie_12d as config


def _actuator_layout(model: mujoco.MjModel):
    names = ("body_thrust", "x_moment", "y_moment", "z_moment")
    ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names],
        dtype=int,
    )
    if np.any(ids < 0) or len(np.unique(ids)) != 4:
        raise ValueError(f"Unexpected Crazyflie actuator names/ids: {dict(zip(names, ids))}")
    thrust_gear = float(model.actuator_gear[ids[0], 2])
    moment_gears = np.array(
        [model.actuator_gear[ids[index], 2 + index] for index in range(1, 4)],
        dtype=float,
    )
    if not np.isclose(thrust_gear, 1.0) or np.any(moment_gears == 0.0):
        raise ValueError(
            f"Unexpected actuator gears: thrust={thrust_gear}, moments={moment_gears}"
        )
    return ids, thrust_gear, moment_gears


def make_plant(true_mass_scale: float):
    """Load the vendored CF2.1 plant and scale its mass and inertia."""
    if true_mass_scale <= 0.0:
        raise ValueError("true_mass_scale must be positive")
    model = mujoco.MjModel.from_xml_path(config.model_path)
    if (model.nq, model.nv, model.nu) != (7, 6, 4):
        raise ValueError(
            "Expected a single-free-joint, four-actuator Crazyflie; "
            f"got nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    model.opt.timestep = config.simulation_dt
    ids, _, _ = _actuator_layout(model)
    model.actuator_ctrlrange[ids[0]] = np.array([config.thrust_min, config.thrust_max])
    for actuator_id, gear_index in zip(ids[1:], (3, 4, 5)):
        model.actuator_gear[actuator_id, gear_index] *= config.moment_gear_scale
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    if body_id < 0:
        raise ValueError("Crazyflie body cf2 was not found in the MuJoCo model")
    source_mass = float(model.body_mass[body_id])
    if not np.isclose(source_mass, config.source_model_mass, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Expected source model mass {config.source_model_mass}, got {source_mass}")
    true_mass = true_mass_scale * config.nominal_mass
    inertial_scale = true_mass / source_mass
    model.body_mass[body_id] = true_mass
    model.body_inertia[body_id] *= inertial_scale
    mujoco.mj_setConst(model, data)
    data.qpos[:] = np.array([*np.asarray(config.start_position), 1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, body_id, true_mass


def _desired_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cphi, sphi = np.cos(roll), np.sin(roll)
    ctheta, stheta = np.cos(pitch), np.sin(pitch)
    cpsi, spsi = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta, ctheta * sphi, ctheta * cphi],
    ])


@jax.jit
def pack_measured_state(qpos, qvel, xmat, body_gyro):
    """Device-side Euler extraction and 12D state packing."""
    R = jnp.reshape(xmat, (3, 3))
    phi = jnp.arctan2(R[2, 1], R[2, 2])
    theta = jnp.arcsin(jnp.clip(-R[2, 0], -1., 1.))
    psi = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.concatenate([
        qpos[:3], qvel[:3], jnp.array([phi, theta, psi]), body_gyro,
    ])


def measured_state(data: mujoco.MjData, body_id: int) -> jnp.ndarray:
    """Read MuJoCo telemetry on host, then pack it in the JITted state map."""
    return pack_measured_state(
        jnp.asarray(data.qpos),
        jnp.asarray(data.qvel),
        jnp.asarray(data.xmat[body_id]),
        jnp.asarray(data.sensor("body_gyro").data),
    )


def high_level_to_actuators(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, u: np.ndarray, yaw_target: float) -> np.ndarray:
    """Adapt ``[T, roll, pitch, yawrate]`` through a full attitude loop."""
    ids, thrust_gear, moment_gears = _actuator_layout(model)
    thrust, roll_ref, pitch_ref, yawrate_ref = np.asarray(u, dtype=float)
    R = np.asarray(data.xmat[body_id]).reshape(3, 3)
    yaw_now = np.arctan2(R[1, 0], R[0, 0])
    # A bounded yaw target is advanced at the requested yaw-rate command.
    yaw_desired = yaw_target + model.opt.timestep * yawrate_ref
    desired = _desired_rotation(roll_ref, pitch_ref, yaw_desired)
    error_rotation = R.T @ desired
    attitude_error = .5 * np.array([error_rotation[2, 1] - error_rotation[1, 2], error_rotation[0, 2] - error_rotation[2, 0], error_rotation[1, 0] - error_rotation[0, 1]])
    omega = np.asarray(data.sensor("body_gyro").data)
    moments = np.array([3.e-3, 3.e-3, 1.e-3]) * attitude_error - np.array([2.e-4, 2.e-4, 8.e-5]) * omega
    command = np.zeros(model.nu)
    command[ids[0]] = thrust / thrust_gear
    command[ids[1:]] = moments / moment_gears
    return np.clip(command, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]), yaw_desired
