"""Evaluate saved CCM, Adaptive+LEB, and Pavone controllers on shared scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.ccm import FeedbackLinearizedCarCCM, ccm_feedback


METHOD_NAMES = np.asarray(["ccm", "adaptive_leb", "pavone"])
OBSTACLES = np.asarray([[-0.25, 0.20, 0.23], [0.25, -0.25, 0.23]])
GOAL = np.asarray([0.25, -1.0])


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def build_scenarios(
    horizon: int,
    *,
    parameter_bound: float,
    disturbance_scale: float,
    random_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Make the quadrotor-style axis/all-ones stress tests and random interiors."""
    parameter_corners = parameter_bound * np.asarray(
        [[sx, sy] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    )
    directions = np.concatenate(
        [np.eye(4), -np.eye(4), np.ones((1, 4)), -np.ones((1, 4))], axis=0
    )
    adversarial_count = len(directions)
    adversarial_parameters = parameter_corners[
        np.arange(adversarial_count) % len(parameter_corners)
    ]
    adversarial_coefficients = np.repeat(
        directions[:, None, :], horizon, axis=1
    )

    rng = np.random.default_rng(seed)
    random_parameters = rng.uniform(
        -parameter_bound, parameter_bound, size=(random_count, 2)
    )
    random_coefficients = rng.uniform(
        -1.0, 1.0, size=(random_count, horizon, 4)
    )
    count = adversarial_count + random_count
    return {
        "true_parameters": np.concatenate(
            [adversarial_parameters, random_parameters], axis=0
        ),
        "disturbance_coefficients": np.concatenate(
            [adversarial_coefficients, random_coefficients], axis=0
        ),
        "disturbances": disturbance_scale
        * np.concatenate([adversarial_coefficients, random_coefficients], axis=0),
        "scenario_types": np.asarray(
            ["adversarial"] * adversarial_count + ["random"] * random_count
        ),
        "rollout_labels": np.asarray(
            ["adversarial"] * adversarial_count
            + ["non_adversarial"] * random_count
        ),
        "is_adversarial": np.arange(count) < adversarial_count,
        "parameter_corner_indices": np.concatenate(
            [np.arange(adversarial_count) % 4, -np.ones(random_count, dtype=int)]
        ),
        "adversarial_direction_indices": np.concatenate(
            [np.arange(adversarial_count), -np.ones(random_count, dtype=int)]
        ),
        "adversarial_directions": directions,
        "parameter_corners": parameter_corners,
    }


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def car_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    parameter: np.ndarray,
    disturbance: np.ndarray,
) -> np.ndarray:
    heading, speed = state[2], state[3]
    return np.asarray(
        [
            state[0] + dt * (speed * np.cos(heading) + parameter[0]),
            state[1] + dt * (speed * np.sin(heading) + parameter[1]),
            state[2] + dt * control[0],
            state[3] + dt * control[1],
        ]
    ) + disturbance


def adaptive_jacobians(state: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    heading, speed = state[2], state[3]
    a_matrix = np.eye(6)
    a_matrix[0, 2] = -dt * speed * np.sin(heading)
    a_matrix[0, 3] = dt * np.cos(heading)
    a_matrix[0, 4] = dt
    a_matrix[1, 2] = dt * speed * np.cos(heading)
    a_matrix[1, 3] = dt * np.sin(heading)
    a_matrix[1, 5] = dt
    b_matrix = np.zeros((6, 2))
    b_matrix[2, 0] = dt
    b_matrix[3, 1] = dt
    return a_matrix, b_matrix


def simulate_adaptive(
    controller: dict[str, np.ndarray],
    parameter: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal_states = controller["nominal_states"]
    nominal_inputs = controller["nominal_inputs"]
    gains = controller["state_feedback_gains"]
    estimator_gains = controller["estimator_gains"]
    sensitivity_inverses = controller["estimator_sensitivity_inverses"]
    dt = float(controller["dt"])
    horizon = len(nominal_inputs)
    states = np.full((horizon + 1, 4), np.nan)
    controls = np.full((horizon, 2), np.nan)
    estimates = np.full((horizon + 1, 2), np.nan)
    augmented = nominal_states[0].copy()
    augmented[4:] = controller["initial_parameter_center"]
    states[0], estimates[0] = augmented[:4], augmented[4:]
    deviation_history: list[np.ndarray] = []
    for k in range(horizon):
        deviation = augmented - nominal_states[k]
        deviation[2] = wrap_to_pi(deviation[2])
        deviation_history.append(deviation)
        correction = sum(
            (gains[k, j] @ deviation_history[j] for j in range(k + 1)),
            start=np.zeros(2),
        )
        controls[k] = nominal_inputs[k] + correction
        physical_next = car_step(
            augmented[:4], controls[k], dt, parameter, disturbances[k]
        )
        a_matrix, b_matrix = adaptive_jacobians(nominal_states[k], dt)
        prediction = nominal_states[k + 1, :4] + (
            a_matrix @ deviation + b_matrix @ correction
        )[:4]
        innovation = sensitivity_inverses[k] @ (physical_next - prediction)
        estimate_next = augmented[4:] + estimator_gains[k] @ innovation
        augmented = np.concatenate([physical_next, estimate_next])
        states[k + 1], estimates[k + 1] = physical_next, estimate_next
        if not np.all(np.isfinite(augmented)):
            break
    return states, controls, estimates


def simulate_pavone(
    controller: dict[str, np.ndarray],
    parameter: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal_states = controller["nominal_states"]
    nominal_inputs = controller["nominal_inputs"]
    gains = controller["disturbance_gains"]
    a_matrices = controller["linearization_a"]
    b_matrices = controller["linearization_b"]
    dt = float(controller["dt"])
    horizon = len(nominal_inputs)
    states = np.full((horizon + 1, 4), np.nan)
    controls = np.full((horizon, 2), np.nan)
    reconstructed = np.zeros((horizon, 4))
    states[0] = nominal_states[0]
    for k in range(horizon):
        correction = sum(
            (gains[k, j] @ reconstructed[j] for j in range(k)),
            start=np.zeros(2),
        )
        controls[k] = nominal_inputs[k] + correction
        states[k + 1] = car_step(
            states[k], controls[k], dt, parameter, disturbances[k]
        )
        reconstructed[k] = (
            states[k + 1]
            - nominal_states[k + 1]
            - a_matrices[k] @ (states[k] - nominal_states[k])
            - b_matrices[k] @ correction
        )
        if not np.all(np.isfinite(states[k + 1])):
            break
    return states, controls, np.full((horizon + 1, 2), np.nan)


def nominal_state_inside_interval(
    nominal_state: np.ndarray, virtual_input: np.ndarray, tau: float
) -> np.ndarray:
    return np.concatenate(
        [
            nominal_state[:2]
            + tau * nominal_state[2:]
            + 0.5 * tau**2 * virtual_input,
            nominal_state[2:] + tau * virtual_input,
        ]
    )


def simulate_ccm(
    controller: dict[str, np.ndarray],
    parameter: np.ndarray,
    disturbances: np.ndarray,
    integration_substeps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = controller["nominal_linear_states"]
    virtual_inputs = controller["nominal_virtual_inputs"]
    certificate = FeedbackLinearizedCarCCM(
        metric=controller["metric"],
        differential_gain=controller["differential_gain"],
        contraction_rate=float(controller["contraction_rate"]),
        disturbance_bound=float(controller["disturbance_bound"]),
        position_support=float(controller["position_support"]),
        velocity_support=float(controller["velocity_support"]),
        virtual_input_support=float(controller["virtual_input_support"]),
        certificate_max_eigenvalue=float(controller["certificate_max_eigenvalue"]),
        maximum_disturbance_metric_norm=float(
            controller["maximum_disturbance_metric_norm"]
        ),
    )
    dt = float(controller["dt"])
    horizon = len(virtual_inputs)
    states = np.full((horizon + 1, 4), np.nan)
    controls = np.full((horizon, 2), np.nan)
    states[0] = controller["nominal_states"][0]
    substep = dt / integration_substeps
    for k in range(horizon):
        state = states[k].copy()
        controls[k] = ccm_feedback(state, nominal[k], virtual_inputs[k], certificate)
        continuous_disturbance = disturbances[k] / dt

        def derivative(point: np.ndarray, reference: np.ndarray) -> np.ndarray:
            control = ccm_feedback(point, reference, virtual_inputs[k], certificate)
            return np.asarray(
                [
                    point[3] * np.cos(point[2])
                    + parameter[0]
                    + continuous_disturbance[0],
                    point[3] * np.sin(point[2])
                    + parameter[1]
                    + continuous_disturbance[1],
                    control[0] + continuous_disturbance[2],
                    control[1] + continuous_disturbance[3],
                ]
            )

        for substep_index in range(integration_substeps):
            tau = substep_index * substep
            z1 = nominal_state_inside_interval(nominal[k], virtual_inputs[k], tau)
            z2 = nominal_state_inside_interval(
                nominal[k], virtual_inputs[k], tau + 0.5 * substep
            )
            z4 = nominal_state_inside_interval(
                nominal[k], virtual_inputs[k], tau + substep
            )
            k1 = derivative(state, z1)
            k2 = derivative(state + 0.5 * substep * k1, z2)
            k3 = derivative(state + 0.5 * substep * k2, z2)
            k4 = derivative(state + substep * k3, z4)
            state += substep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states[k + 1] = state
        if not np.all(np.isfinite(state)):
            break
    return states, controls, np.full((horizon + 1, 2), np.nan)


def rollout_metrics(states: np.ndarray) -> tuple[float, float]:
    finite_states = states[np.all(np.isfinite(states), axis=1)]
    if not len(finite_states):
        return np.nan, np.nan
    clearance = (
        np.linalg.norm(
            finite_states[:, None, :2] - OBSTACLES[None, :, :2], axis=-1
        )
        - OBSTACLES[None, :, 2]
    )
    return float(np.min(clearance)), float(np.linalg.norm(finite_states[-1, :2] - GOAL))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-dir",
        type=Path,
        default=Path(__file__).with_name("saved_controllers"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("controller_rollouts.npz"),
    )
    parser.add_argument("--random-rollouts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ccm-integration-substeps", type=int, default=10)
    args = parser.parse_args()
    if args.random_rollouts < 0 or args.ccm_integration_substeps < 1:
        parser.error("rollout count must be nonnegative and substeps must be positive")

    controllers = {
        "ccm": load_npz(args.controller_dir / "ccm_controller.npz"),
        "adaptive_leb": load_npz(args.controller_dir / "adaptive_leb_controller.npz"),
        "pavone": load_npz(args.controller_dir / "pavone_controller.npz"),
    }
    horizons = {name: len(data["nominal_states"]) - 1 for name, data in controllers.items()}
    dts = {name: float(data["dt"]) for name, data in controllers.items()}
    if len(set(horizons.values())) != 1 or not np.allclose(list(dts.values()), next(iter(dts.values()))):
        raise ValueError(f"controller time-grid mismatch: horizons={horizons}, dt={dts}")
    horizon = next(iter(horizons.values()))
    parameter_bound = float(controllers["ccm"]["parameter_uncertainty_bound"])
    disturbance_scale = float(controllers["ccm"]["exogenous_disturbance_scale"])
    scenarios = build_scenarios(
        horizon,
        parameter_bound=parameter_bound,
        disturbance_scale=disturbance_scale,
        random_count=args.random_rollouts,
        seed=args.seed,
    )
    count = len(scenarios["true_parameters"])
    states = np.full((len(METHOD_NAMES), count, horizon + 1, 4), np.nan)
    controls = np.full((len(METHOD_NAMES), count, horizon, 2), np.nan)
    estimates = np.full((len(METHOD_NAMES), count, horizon + 1, 2), np.nan)
    failed = np.zeros((len(METHOD_NAMES), count), dtype=bool)
    minimum_clearance = np.full((len(METHOD_NAMES), count), np.nan)
    distance_to_goal = np.full((len(METHOD_NAMES), count), np.nan)
    simulators = {
        "ccm": lambda data, theta, noise: simulate_ccm(
            data, theta, noise, args.ccm_integration_substeps
        ),
        "adaptive_leb": simulate_adaptive,
        "pavone": simulate_pavone,
    }
    for method_index, method in enumerate(METHOD_NAMES):
        for scenario_index in range(count):
            result = simulators[str(method)](
                controllers[str(method)],
                scenarios["true_parameters"][scenario_index],
                scenarios["disturbances"][scenario_index],
            )
            states[method_index, scenario_index] = result[0]
            controls[method_index, scenario_index] = result[1]
            estimates[method_index, scenario_index] = result[2]
            failed[method_index, scenario_index] = not np.all(np.isfinite(result[0]))
            minimum_clearance[method_index, scenario_index], distance_to_goal[
                method_index, scenario_index
            ] = rollout_metrics(result[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        method_names=METHOD_NAMES,
        states=states,
        controls=controls,
        parameter_estimates=estimates,
        failed=failed,
        minimum_obstacle_clearance=minimum_clearance,
        distance_to_goal=distance_to_goal,
        dt=np.asarray(next(iter(dts.values()))),
        seed=np.asarray(args.seed),
        **scenarios,
    )
    summary = {
        "output": str(args.output.resolve()),
        "methods": METHOD_NAMES.tolist(),
        "adversarial_rollouts": int(np.sum(scenarios["is_adversarial"])),
        "random_rollouts": args.random_rollouts,
        "failed_rollouts_by_method": {
            str(method): int(failed[index].sum())
            for index, method in enumerate(METHOD_NAMES)
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
