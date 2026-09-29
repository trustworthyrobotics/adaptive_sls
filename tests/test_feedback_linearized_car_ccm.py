import numpy as np

from baselines.ccm import (
    car_to_linearizing_coordinates,
    design_feedback_linearized_car_ccm,
    linearizing_jacobian,
    linearizing_to_car_state,
    physical_to_virtual_input,
    pulled_back_metric,
    virtual_to_physical_input,
)


def test_coordinate_and_input_maps_round_trip_on_positive_speed_domain():
    state = np.array([-0.2, 0.7, -1.8, 0.38])
    physical_input = np.array([0.6, -0.25])

    recovered_state = linearizing_to_car_state(car_to_linearizing_coordinates(state))
    virtual_input = physical_to_virtual_input(state, physical_input)
    recovered_input = virtual_to_physical_input(state, virtual_input)

    np.testing.assert_allclose(recovered_state, state, atol=1e-12)
    np.testing.assert_allclose(recovered_input, physical_input, atol=1e-12)
    assert abs(np.linalg.det(linearizing_jacobian(state))) > 0.0


def test_sdp_certificate_is_strict_and_pullback_is_positive_definite():
    certificate = design_feedback_linearized_car_ccm(
        parameter_half_width=0.05,
        continuous_disturbance_bound=0.00030 / 0.05,
        certified_maximum_speed=2.0,
    )
    state = np.array([0.0, 1.0, -0.5 * np.pi, 0.38])

    assert certificate.certificate_max_eigenvalue < -1e-8
    assert np.linalg.eigvalsh(certificate.metric)[0] > 0.0
    assert np.linalg.eigvalsh(pulled_back_metric(state, certificate))[0] > 0.0
    assert certificate.maximum_disturbance_metric_norm <= 1.0 + 1e-6
    assert certificate.radius(1.0) > 0.0
