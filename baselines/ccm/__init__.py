"""Trajectory-local control-contraction-metric baselines."""

from .trajectory_local_ccm import (
    BoxBounds,
    SolverConfig,
    TrajectoryLocalCCMResult,
    TrajectoryLocalCCMSolver,
    UncertaintySet,
)
from .feedback_linearized_car_ccm import (
    FeedbackLinearizedCarCCM,
    car_to_linearizing_coordinates,
    ccm_feedback,
    design_feedback_linearized_car_ccm,
    linearizing_jacobian,
    linearizing_to_car_state,
    physical_to_virtual_input,
    pulled_back_metric,
    virtual_to_physical_input,
)

__all__ = [
    "BoxBounds",
    "SolverConfig",
    "TrajectoryLocalCCMResult",
    "TrajectoryLocalCCMSolver",
    "UncertaintySet",
    "FeedbackLinearizedCarCCM",
    "car_to_linearizing_coordinates",
    "ccm_feedback",
    "design_feedback_linearized_car_ccm",
    "linearizing_jacobian",
    "linearizing_to_car_state",
    "physical_to_virtual_input",
    "pulled_back_metric",
    "virtual_to_physical_input",
]
