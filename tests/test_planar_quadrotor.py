from __future__ import annotations

import numpy as np

from experiments.planar_quadrotor.config import ExperimentConfig, OBSTACLES, Q, QF, R
from experiments.planar_quadrotor.estimators import LEARNING_CHANNEL, SMEState
from experiments.planar_quadrotor.model import continuous_dynamics, true_step
from experiments.planar_quadrotor.official_ccm import OfficialCCM, audit_candidate_metric
from experiments.planar_quadrotor.scenarios import generate_scenarios


def test_certified_uncertainty_bounds_match_authors():
    config = ExperimentConfig()
    assert np.isclose(config.parameter_half_width, 0.01 / 0.486)
    assert config.disturbance_acceleration_half_width == 0.1


def test_common_mpc_cost_uses_balanced_rti_scaling():
    np.testing.assert_allclose(np.diag(Q), [10.0, 10.0, 2.0, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(QF, 10.0 * Q)
    np.testing.assert_allclose(R, np.eye(2))


def test_obstacle_geometry_is_configurable_without_changing_default():
    default = ExperimentConfig()
    shifted = ExperimentConfig(
        obstacle_x=1.0,
        obstacle_y=-0.1,
        obstacle_radius=0.35,
        obstacle_inflation=0.01,
    )
    np.testing.assert_allclose(default.obstacles, OBSTACLES)
    np.testing.assert_allclose(shifted.obstacles, [[1.0, -0.1, 0.35]])
    np.testing.assert_allclose(shifted.solver_obstacles, [[1.0, -0.1, 0.36]])


def test_scalar_disturbance_uses_authors_rotated_horizontal_channel():
    config = ExperimentConfig()
    state = np.zeros(6)
    control = np.full(2, config.hover_input)
    derivative = continuous_dynamics(state, control, 0.0, config, disturbance=0.1)
    assert np.isclose(derivative[3], 0.1)
    assert np.isclose(derivative[4], 0.0)
    state[2] = np.pi / 2
    derivative = continuous_dynamics(state, control, 0.0, config, disturbance=0.1)
    nominal = continuous_dynamics(state, control, 0.0, config, disturbance=0.0)
    assert np.isclose(derivative[3] - nominal[3], 0.0, atol=1e-12)
    assert np.isclose(derivative[4] - nominal[4], -0.1)


def test_learning_channel_selects_only_body_velocity_residuals():
    np.testing.assert_array_equal(
        np.diag(LEARNING_CHANNEL), np.array([0, 0, 0, 1, 1, 0])
    )


def test_scenarios_are_reproducible_and_shared_shape():
    config = ExperimentConfig(runs=3, max_steps=7, seed=11)
    first = generate_scenarios(config)
    second = generate_scenarios(config)
    assert first[0].shape == (3, 1)
    assert first[1].shape == (3, 7, 1)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_sme_retains_true_inverse_mass_error():
    config = ExperimentConfig()
    estimator = SMEState.initial(config)
    state = np.zeros(6)
    control = np.full(2, config.hover_input)
    true_parameter = np.array([0.4 * config.parameter_half_width])
    next_state = true_step(state, control, true_parameter, np.array([0.7]), config)
    posterior = estimator.update(state, control, next_state, config)
    assert posterior.lower <= true_parameter[0] <= posterior.upper
    assert posterior.half_width[0] <= estimator.half_width[0]


def test_official_metric_is_valid_at_stored_rate_for_certified_model():
    config = ExperimentConfig()
    ccm = OfficialCCM()
    state = np.zeros(6)
    np.testing.assert_allclose(ccm.W(state), ccm.W(state).T, atol=1e-10)
    assert np.linalg.eigvalsh(ccm.W(state))[0] > 0.0
    audit = audit_candidate_metric(
        ccm, config, samples=128, requested_rate=ccm.original_rho
    )
    assert audit.passed
