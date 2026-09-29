"""Shared numerical configuration for the planar-quadrotor comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 10
    dt: float = 0.05
    max_steps: int = 100
    goal_tolerance: float = 0.2
    runs: int = 40
    seed: int = 0
    obstacle_x: float = -0.1
    obstacle_y: float = 1.0
    obstacle_radius: float = 0.5
    obstacle_inflation: float = 0.0

    mass: float = 0.486
    gravity: float = 9.81
    arm_length: float = 0.25
    inertia: float = 0.00383

    # Authors' certified uncertainty set: theta is the additive error in
    # inverse mass, with a one-percent relative half-width.
    inverse_mass_relative_half_width: float = 0.01
    # Authors' scalar inertial-horizontal acceleration disturbance.
    disturbance_acceleration_half_width: float = 0.1

    rti_sls_iterations: int = 1
    rti_sqp_iterations: int = 1
    rti_admm_iterations: int = 400
    rti_sls_primal_tolerance: float = 1e-2
    rti_sqp_feasibility_tolerance: float = 1e-2
    enable_linearization_error: bool = True
    enable_disturbance_variation_error: bool = True
    ccm_nlp_iterations: int = 80
    pavone_scp_iterations: int = 1
    zonotope_order: int = 4

    @property
    def nominal_inverse_mass(self) -> float:
        return 1.0 / self.mass

    @property
    def parameter_half_width(self) -> float:
        return self.inverse_mass_relative_half_width * self.nominal_inverse_mass

    @property
    def hover_input(self) -> float:
        return self.mass * self.gravity / 2.0

    @property
    def obstacles(self) -> np.ndarray:
        return np.array(
            [[self.obstacle_x, self.obstacle_y, self.obstacle_radius]],
            dtype=float,
        )

    @property
    def solver_obstacles(self) -> np.ndarray:
        obstacles = self.obstacles
        obstacles[:, 2] += self.obstacle_inflation
        return obstacles

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        result["nominal_inverse_mass"] = self.nominal_inverse_mass
        result["parameter_half_width"] = self.parameter_half_width
        result["learning_channel_diagonal"] = [0, 0, 0, 1, 1, 0]
        result["hover_input"] = self.hover_input
        return result


X0 = np.zeros(6, dtype=float)
X_GOAL = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
OBSTACLES = np.array([[-0.1, 1.0, 0.5]], dtype=float)

# Paper domain: [p_x, p_y, phi, v_body_x, v_body_y, phi_dot]. Positions are
# intentionally unbounded; the remaining bounds are exactly those in Sec. 4.
X_LOWER = np.array([-np.inf, -np.inf, -np.pi / 3, -2.0, -1.0, -np.pi])
X_UPPER = np.array([np.inf, np.inf, np.pi / 3, 2.0, 1.0, np.pi])
U_LOWER = np.array([-1.0, -1.0])
U_UPPER = np.array([3.5, 3.5])

# A balanced state/input scaling is important for the RTI inequality-dual
# warm start.  The earlier Q_xy=30, R=0.1 choice drove the velocity/rate box
# constraints hard enough to make the shifted mu/sqrt(beta) weight singular.
Q = np.diag([10.0, 10.0, 2.0, 1.0, 1.0, 0.5])
QF = 10.0 * Q
R = np.eye(2)

METHODS = (
    "adaptive_sls_gain",
    "adaptive_sls_sme",
    "ccm_rampc",
    "pavone_rampc",
)
