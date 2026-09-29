"""Four receding-horizon controller adapters with a common result schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import (
    ExperimentConfig,
    Q,
    QF,
    R,
    ROOT,
    U_LOWER,
    U_UPPER,
    X_GOAL,
    X_LOWER,
    X_UPPER,
)
from .model import (
    disturbance_sensitivity,
    finite_difference_linearization,
    parameter_sensitivity,
    rk4_step,
    rollout_nominal,
)
from .official_ccm import (
    CCMAudit,
    OfficialCCM,
    audit_candidate_metric,
    parameter_lipschitz_constants,
    tube_derivative,
    tube_widths,
)


Array = np.ndarray
REPO_ROOT = ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"


@dataclass
class PlanResult:
    control: Array
    states: Array
    inputs: Array
    state_tube_widths: Array
    input_tube_widths: Array
    parameter_tube_widths: Array
    solve_time_seconds: float
    success: bool
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _shift_inputs(inputs: Array) -> Array:
    return np.concatenate([inputs[1:], inputs[-1:]], axis=0)


class CCMController:
    """Homothetic CCM tube MPC using the authors' polynomial W and Y."""

    def __init__(self, config: ExperimentConfig, *, allow_uncertified: bool = False) -> None:
        self.config = config
        self.ccm = OfficialCCM()
        self.original_rate_audit: CCMAudit = audit_candidate_metric(self.ccm, config)
        if self.original_rate_audit.passed:
            self.audit = self.original_rate_audit
        else:
            adapted_rate = 0.95 * self.original_rate_audit.minimum_sampled_contraction_rate
            self.audit = audit_candidate_metric(
                self.ccm, config, requested_rate=max(0.0, adapted_rate)
            )
        if not self.audit.passed and not allow_uncertified:
            raise RuntimeError(
                "official CCM failed its exact-model contraction audit; "
                "rerun with --allow-uncertified-ccm only for a clearly labeled numerical baseline"
            )
        self.parameter_lipschitz = parameter_lipschitz_constants(self.ccm, config)
        # Never report a rate larger than either the original design value or
        # the minimum observed in the changed-model audit.
        self.rate = self.audit.requested_rate
        if not self.audit.passed:
            # The numerical baseline still needs a finite propagation rate;
            # using zero is conservative relative to claiming contraction.
            self.rate = min(0.0, self.rate)
        self.inputs = np.full((config.horizon, 2), config.hover_input)
        self.first_solve = True

    def _evaluate(
        self,
        initial_state: Array,
        parameter_center: Array,
        parameter_generators: Array,
        flat_inputs: Array,
    ) -> tuple[float, Array, Array, Array, Array, Array]:
        inputs = flat_inputs.reshape(self.config.horizon, 2)
        states = rollout_nominal(initial_state, inputs, parameter_center, self.config)
        deltas = np.zeros(self.config.horizon + 1)
        state_width = np.zeros((self.config.horizon + 1, 6))
        input_width = np.zeros((self.config.horizon, 2))
        radial_width = np.zeros(self.config.horizon + 1)
        for k in range(self.config.horizon + 1):
            sw, uw, rw = tube_widths(deltas[k], states[k], self.ccm)
            state_width[k] = sw
            radial_width[k] = rw
            if k < self.config.horizon:
                input_width[k] = uw
                derivative = tube_derivative(
                    deltas[k], states[k], inputs[k], parameter_center,
                    parameter_generators,
                    self.ccm, self.config, self.rate, self.parameter_lipschitz,
                )
                deltas[k + 1] = max(0.0, deltas[k] + self.config.dt * derivative)
        state_error = states[:-1] - X_GOAL
        terminal_error = states[-1] - X_GOAL
        input_error = inputs - self.config.hover_input
        objective = float(
            np.einsum("ti,ij,tj->", state_error, Q, state_error)
            + np.einsum("ti,ij,tj->", input_error, R, input_error)
            + terminal_error @ QF @ terminal_error
        )
        return objective, states, state_width, input_width, radial_width, deltas

    def solve(
        self, state: Array, parameter_center: Array, parameter_generators: Array
    ) -> PlanResult:
        start = time.perf_counter()
        cache_inputs: Array | None = None
        cache_result: tuple[float, Array, Array, Array, Array, Array] | None = None

        def evaluate(flat_inputs: Array):
            nonlocal cache_inputs, cache_result
            if cache_inputs is None or not np.array_equal(flat_inputs, cache_inputs):
                cache_inputs = np.asarray(flat_inputs).copy()
                cache_result = self._evaluate(
                    state, parameter_center, parameter_generators, flat_inputs
                )
            return cache_result

        def objective(flat_inputs: Array) -> float:
            return evaluate(flat_inputs)[0]

        def constraints(flat_inputs: Array) -> Array:
            _, states, state_width, input_width, radial_width, _ = evaluate(flat_inputs)
            inputs = flat_inputs.reshape(self.config.horizon, 2)
            values: list[float] = []
            for k in range(self.config.horizon + 1):
                for index in range(2, 6):
                    values.append(states[k, index] - state_width[k, index] - X_LOWER[index])
                    values.append(X_UPPER[index] - states[k, index] - state_width[k, index])
                for obstacle in self.config.solver_obstacles:
                    values.append(
                        np.linalg.norm(states[k, :2] - obstacle[:2])
                        - obstacle[2]
                        - radial_width[k]
                    )
                if k < self.config.horizon:
                    values.extend(inputs[k] - input_width[k] - U_LOWER)
                    values.extend(U_UPPER - inputs[k] - input_width[k])
            return np.asarray(values)

        solution = minimize(
            objective,
            self.inputs.ravel(),
            method="SLSQP",
            bounds=list(zip(np.tile(U_LOWER, self.config.horizon), np.tile(U_UPPER, self.config.horizon))),
            constraints={"type": "ineq", "fun": constraints},
            options={
                "maxiter": self.config.ccm_nlp_iterations,
                "ftol": 1e-5,
                "disp": False,
            },
        )
        elapsed = time.perf_counter() - start
        candidate = solution.x if np.all(np.isfinite(solution.x)) else self.inputs.ravel()
        objective_value, states, state_width, input_width, _, deltas = self._evaluate(
            state, parameter_center, parameter_generators, candidate
        )
        self.inputs = _shift_inputs(candidate.reshape(self.config.horizon, 2))
        self.first_solve = False
        minimum_constraint = float(np.min(constraints(candidate)))
        return PlanResult(
            control=candidate.reshape(self.config.horizon, 2)[0],
            states=states,
            inputs=candidate.reshape(self.config.horizon, 2),
            state_tube_widths=state_width,
            input_tube_widths=input_width,
            parameter_tube_widths=np.broadcast_to(
                np.sum(np.abs(parameter_generators), axis=1),
                (self.config.horizon + 1, 1),
            ).copy(),
            solve_time_seconds=elapsed,
            success=bool(np.all(np.isfinite(candidate)) and minimum_constraint >= -1e-4),
            status=str(solution.message),
            diagnostics={
                "objective": objective_value,
                "optimizer_success": bool(solution.success),
                "optimizer_iterations": int(solution.nit),
                "minimum_robust_constraint_margin": minimum_constraint,
                "tube_scaling": deltas.tolist(),
                "contraction_rate_used": self.rate,
                "original_sos_design_rate": self.ccm.original_rho,
                "ccm_source": self.ccm.source,
                "original_rate_sampled_audit_passed": self.original_rate_audit.passed,
                "parameter_lipschitz_constants": self.parameter_lipschitz.tolist(),
                "ccm_audit_passed": self.audit.passed,
                "ccm_guarantee": self.audit.guarantee,
            },
        )


class PavoneController:
    """Receding-horizon affine-disturbance-feedback ARMPC baseline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        # The Pavone objective is written around zero input. Optimize thrust
        # deviations and add hover thrust inside the plant map.
        self.inputs = np.zeros((config.horizon, 2))
        self.states: Array | None = None
        self.first_solve = True

    def solve(
        self, state: Array, parameter_center: Array, parameter_generators: Array
    ) -> PlanResult:
        from baselines.pavone_mpc.affine_disturbance_feedback import (
            AffineDisturbanceFeedbackSolver,
            BoxBounds,
            SolverConfig,
        )

        hover = np.full(2, self.config.hover_input)
        nominal_step = lambda x, u: rk4_step(
            x, u + hover, parameter_center, self.config
        )
        linearize = lambda x, u: finite_difference_linearization(
            x, u + hover, parameter_center, self.config
        )
        if self.states is None:
            self.states = rollout_nominal(
                state, self.inputs + hover, parameter_center, self.config
            )
        else:
            self.states[0] = state
            self.states = rollout_nominal(
                state, self.inputs + hover, parameter_center, self.config
            )

        half = np.sum(np.abs(parameter_generators), axis=1)
        sensitivities = [
            parameter_sensitivity(
                self.states[k], self.inputs[k] + hover, parameter_center, self.config
            )
            for k in range(self.config.horizon)
        ]
        parameter_width = np.max([np.abs(s) @ half for s in sensitivities], axis=0)
        disturbance_widths = [
            np.abs(
                disturbance_sensitivity(
                    self.states[k],
                    self.inputs[k] + hover,
                    parameter_center,
                    self.config,
                )[:, 0]
            )
            * self.config.disturbance_acceleration_half_width
            for k in range(self.config.horizon)
        ]
        disturbance_width = parameter_width + np.max(disturbance_widths, axis=0)
        solver = AffineDisturbanceFeedbackSolver(
            nominal_step=nominal_step,
            linearize=linearize,
            state_cost=Q,
            input_cost=R,
            terminal_cost=QF,
            bounds=BoxBounds(
                X_LOWER, X_UPPER, U_LOWER - hover, U_UPPER - hover
            ),
            disturbance_half_width=disturbance_width,
            config=SolverConfig(
                max_iterations=(6 if self.first_solve else self.config.pavone_scp_iterations),
                convergence_tolerance=2e-3,
                state_trust_region=1.0,
                input_trust_region=1.5,
                feedback_memory=None,
                solver="CLARABEL",
            ),
        )
        reference = np.broadcast_to(X_GOAL, (self.config.horizon + 1, 6)).copy()
        start = time.perf_counter()
        result = solver.solve(
            initial_state=state,
            reference=reference,
            initial_inputs=self.inputs,
            initial_states=self.states,
            obstacles=self.config.solver_obstacles,
            terminal_position_tolerance=None,
        )
        elapsed = time.perf_counter() - start
        self.inputs = _shift_inputs(result.inputs)
        self.states = result.states.copy()
        self.first_solve = False
        robust_margins = (
            float(result.diagnostics.get("minimum_robust_state_margin", -np.inf)),
            float(result.diagnostics.get("minimum_robust_input_margin", -np.inf)),
            float(result.diagnostics.get("minimum_radial_obstacle_margin", -np.inf)),
        )
        return PlanResult(
            control=result.inputs[0] + hover,
            states=result.states,
            inputs=result.inputs + hover,
            state_tube_widths=result.state_half_widths,
            input_tube_widths=result.input_half_widths,
            parameter_tube_widths=np.broadcast_to(
                half, (self.config.horizon + 1, 1)
            ).copy(),
            solve_time_seconds=elapsed,
            success=bool(
                np.all(np.isfinite(result.inputs)) and min(robust_margins) >= -1e-5
            ),
            status=str(result.diagnostics.get("solver_status", "unknown")),
            diagnostics={**result.diagnostics, "outer_scp_converged": result.converged},
        )


class AdaptiveSLSController:
    """Thin RTI adapter around the repository's adaptive Fast-SLS+LEB solver."""

    def __init__(self, config: ExperimentConfig) -> None:
        # The repository also contains a legacy top-level ``gpu_sls.py``.
        # Ensure the package under src/ wins module resolution.
        if str(SRC_ROOT) in sys.path:
            sys.path.remove(str(SRC_ROOT))
        sys.path.insert(0, str(SRC_ROOT))
        import jax
        import jax.numpy as jnp
        from gpu_sls.generic_mpc_wrapper import GenericMPCControllerWrapper
        from gpu_sls.gpu_admm import ADMMConfig
        from gpu_sls.gpu_sls import SLSConfig
        from gpu_sls.sqp import SQPConfig

        self.jax, self.jnp = jax, jnp
        self.config = config
        n_phys, n_theta, n_aug, n_u = 6, 1, 7, 2

        @dataclass(frozen=True)
        class MPCConfig:
            n: int
            nu: int
            N: int
            W: Any
            u_ref: Any
            dt: float

        def physical_derivative(x, u, theta):
            phi, v1, v2, omega = x[2], x[3], x[4], x[5]
            c, s = jnp.cos(phi), jnp.sin(phi)
            inverse_mass = config.nominal_inverse_mass + theta[0]
            return jnp.stack([
                v1 * c - v2 * s,
                v1 * s + v2 * c,
                omega,
                v2 * omega - config.gravity * s,
                -v1 * omega - config.gravity * c
                + (u[0] + u[1]) * inverse_mass,
                config.arm_length * (u[0] - u[1]) / config.inertia,
            ])

        def dynamics(x_aug, u, t, *, parameter):
            del t
            x, theta = x_aug[:n_phys], x_aug[n_phys:]
            h = parameter
            f = lambda state: physical_derivative(state, u, theta)
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x_next = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            return jnp.concatenate([x_next, theta])

        def disturbance(x_prefix):
            physical = x_prefix[..., :n_phys]
            phi = physical[:, 2]
            scale = config.dt * config.disturbance_acceleration_half_width
            zero = 0.0 * phi
            disturbance_vector = jnp.stack(
                [
                    zero,
                    zero,
                    zero,
                    scale * jnp.cos(phi),
                    -scale * jnp.sin(phi),
                    zero,
                ],
                axis=1,
            )
            first_generator = jnp.asarray(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=x_prefix.dtype
            )
            return disturbance_vector[:, :, None] * first_generator[None, None, :]

        q_diag = jnp.asarray(np.diag(Q))
        r_diag = jnp.asarray(np.diag(R))
        weights = jnp.concatenate([q_diag, jnp.zeros(n_theta), r_diag])
        hover = jnp.full((2,), config.hover_input)

        def cost(weight, reference, x, u, t):
            error = x - reference[t]
            return jnp.sum(weight[:n_aug] * error**2) + jnp.sum(
                weight[n_aug:] * (u - hover) ** 2
            )

        lower = jnp.asarray(X_LOWER)
        upper = jnp.asarray(X_UPPER)
        u_lower, u_upper = jnp.asarray(U_LOWER), jnp.asarray(U_UPPER)

        def constraints(x, u, t):
            del t
            physical = x[:n_phys]
            return jnp.concatenate([
                physical - upper,
                lower - physical,
                u - u_upper,
                u_lower - u,
            ])

        x0 = jnp.concatenate([jnp.zeros(6), jnp.zeros(1)])
        x_seed = jnp.broadcast_to(x0, (config.horizon + 1, n_aug))
        u_seed = jnp.broadcast_to(hover, (config.horizon, n_u))
        mpc_config = MPCConfig(n_aug, n_u, config.horizon, weights, hover, config.dt)
        self.wrapper = GenericMPCControllerWrapper(
            SLSConfig(
                max_sls_iterations=config.rti_sls_iterations,
                sls_primal_tol=config.rti_sls_primal_tolerance,
                enable_fastsls=True,
                warm_start=True,
                n_x=n_phys,
                n_w=n_phys,
                n_theta=n_theta,
                num_param=n_theta,
                q_max=config.zonotope_order,
                adaptive=True,
                enable_linearization_error=config.enable_linearization_error,
                enable_disturbance_variation_error=(
                    config.enable_disturbance_variation_error
                ),
                enable_posterior_gate=True,
            ),
            SQPConfig(
                max_sqp_iterations=config.rti_sqp_iterations,
                warm_start=True,
                feas_tol=config.rti_sqp_feasibility_tolerance,
                step_tol=1e-4,
                line_search=True,
            ),
            ADMMConfig(max_iterations=config.rti_admm_iterations),
            config=mpc_config,
            dynamics=dynamics,
            constraints=constraints,
            obstacles=jnp.asarray(config.solver_obstacles),
            cost=cost,
            num_constraints=2 * n_phys + 2 * n_u + len(config.solver_obstacles),
            disturbance=disturbance,
            measurement_matrix=jnp.diag(
                jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
            ),
            X_in=x_seed,
            U_in=u_seed,
            limited_memory=False,
            shift=1,
            Q_bar=jnp.diag(jnp.concatenate([q_diag, jnp.zeros(n_theta)])),
            R_bar=jnp.diag(r_diag),
            Q_f_bar=jnp.diag(jnp.concatenate([10.0 * q_diag, jnp.zeros(n_theta)])),
        )
        self.dynamics = dynamics

    def solve(
        self, state: Array, parameter_center: Array, parameter_generators: Array
    ) -> PlanResult:
        jnp = self.jnp
        from experiments.quadrotor.quadrotor_common import state_tube_widths

        x_aug = jnp.asarray(np.concatenate([state, parameter_center]))
        reference = jnp.asarray(
            np.broadcast_to(
                np.concatenate([X_GOAL, parameter_center]),
                (self.config.horizon + 1, 7),
            )
        )
        target_generator_count = (
            parameter_generators.shape[0] * self.config.zonotope_order
        )
        fixed_generators = np.pad(
            parameter_generators,
            (
                (0, 0),
                (
                    0,
                    max(
                        0,
                        target_generator_count - parameter_generators.shape[1],
                    ),
                ),
            ),
        )[:, :target_generator_count]
        start = time.perf_counter()
        output = self.wrapper.run(
            x0=x_aug,
            G_0=jnp.asarray(fixed_generators),
            reference=reference,
            nominal_param=jnp.asarray(parameter_center),
            parameter=self.config.dt,
        )
        output[0].block_until_ready()
        elapsed = time.perf_counter() - start
        u0, states, inputs, _, _, phi_x, phi_u, _, e_hat, _, g_sequence, _ = output
        state_width = np.asarray(state_tube_widths(phi_x, e_hat))[:, :6]
        response_u = jnp.einsum("kjun,jne->kjue", phi_u, e_hat)
        input_width = np.asarray(jnp.sum(jnp.sum(jnp.abs(response_u), axis=-1), axis=1))
        parameter_width = np.zeros((self.config.horizon + 1, 1))
        parameter_width[0] = np.sum(np.abs(fixed_generators), axis=1)
        parameter_width[1:] = np.sum(np.abs(np.asarray(g_sequence)), axis=-1)
        states_np = np.asarray(states)[:, :6]
        inputs_np = np.asarray(inputs)
        bounded_state_margin = np.minimum(
            (states_np[:, 2:] - state_width[:, 2:]) - X_LOWER[2:],
            X_UPPER[2:] - (states_np[:, 2:] + state_width[:, 2:]),
        )
        robust_input_margin = np.minimum(
            (inputs_np - input_width) - U_LOWER,
            U_UPPER - (inputs_np + input_width),
        )
        robust_physical_obstacle_margin = min(
            float(
                np.min(
                    np.linalg.norm(states_np[:, :2] - obstacle[:2], axis=1)
                    - obstacle[2]
                    - np.linalg.norm(state_width[:, :2], axis=1)
                )
            )
            for obstacle in self.config.obstacles
        )
        robust_obstacle_margin = min(
            float(
                np.min(
                    np.linalg.norm(states_np[:, :2] - obstacle[:2], axis=1)
                    - obstacle[2]
                    - np.linalg.norm(state_width[:, :2], axis=1)
                )
            )
            for obstacle in self.config.solver_obstacles
        )
        minimum_robust_margin = min(
            float(np.min(bounded_state_margin)),
            float(np.min(robust_input_margin)),
            robust_obstacle_margin,
        )
        finite = bool(
            np.all(np.isfinite(states_np)) and np.all(np.isfinite(inputs_np))
        )
        within_solver_tolerance = (
            minimum_robust_margin >= -self.config.rti_sqp_feasibility_tolerance
        )
        return PlanResult(
            control=np.asarray(u0),
            states=np.asarray(states)[:, :6],
            inputs=np.asarray(inputs),
            state_tube_widths=state_width,
            input_tube_widths=input_width,
            parameter_tube_widths=parameter_width,
            solve_time_seconds=elapsed,
            success=finite and within_solver_tolerance,
            status=(
                "finite_within_tolerance"
                if finite and within_solver_tolerance
                else "robust_constraint_violation"
                if finite
                else "nonfinite"
            ),
            diagnostics={
                "linearization_error_bound_enabled": (
                    self.config.enable_linearization_error
                ),
                "disturbance_variation_error_bound_enabled": (
                    self.config.enable_disturbance_variation_error
                ),
                "contraction_gate_enabled": True,
                "posterior_gate_enabled": True,
                "outer_inequality_dual_warm_start_enabled": True,
                "learning_channel_diagonal": [0, 0, 0, 1, 1, 0],
                "rti_sls_iterations": self.config.rti_sls_iterations,
                "rti_sqp_iterations": self.config.rti_sqp_iterations,
                "robust_feasibility_tolerance": (
                    self.config.rti_sqp_feasibility_tolerance
                ),
                "minimum_robust_state_margin": float(
                    np.min(bounded_state_margin)
                ),
                "minimum_robust_input_margin": float(np.min(robust_input_margin)),
                "minimum_robust_obstacle_margin": robust_obstacle_margin,
                "minimum_robust_physical_obstacle_margin": (
                    robust_physical_obstacle_margin
                ),
                "minimum_reconstructed_robust_margin": minimum_robust_margin,
                "jax_backend": self.jax.default_backend(),
                "jax_device": str(self.jax.devices()[0]),
            },
        )
