"""Evaluate saved car-ablation feedback policies without re-solving MPC.

For every saved controller, this script generates 30 stochastic rollouts and
10 corner rollouts.  Corner rollouts use independent disturbance primitives
from ``{-1, 1}^{n_w}`` at every step and a fixed parameter primitive from
``{-1, 1}^{n_theta}``, i.e., a parameter at a corner of its initial box.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_tube_deviations import plot_case


N_RANDOM_DEFAULT = 30
N_ADVERSARIAL_DEFAULT = 10


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def physical_step(x: np.ndarray, u: np.ndarray, parameter: np.ndarray, dt: float, matched: bool) -> np.ndarray:
    px, py, heading, speed = x
    omega, acceleration = u
    omega_bias, acceleration_bias = parameter
    if matched:
        return np.array(
            [
                px + dt * speed * np.cos(heading),
                py + dt * speed * np.sin(heading),
                heading + dt * (omega + omega_bias),
                speed + dt * (acceleration + acceleration_bias),
            ]
        )
    return np.array(
        [
            px + dt * (speed * np.cos(heading) + omega_bias),
            py + dt * (speed * np.sin(heading) + acceleration_bias),
            heading + dt * omega,
            speed + dt * acceleration,
        ]
    )


def linearized_dynamics(x: np.ndarray, dt: float, matched: bool) -> tuple[np.ndarray, np.ndarray]:
    """A/B matrices for the adaptive augmented state at a nominal state."""
    heading, speed = x[2], x[3]
    A = np.eye(6)
    A[0, 2] = -dt * speed * np.sin(heading)
    A[0, 3] = dt * np.cos(heading)
    A[1, 2] = dt * speed * np.cos(heading)
    A[1, 3] = dt * np.sin(heading)
    if matched:
        A[2, 4] = dt
        A[3, 5] = dt
    else:
        A[0, 4] = dt
        A[1, 5] = dt
    B = np.zeros((6, 2))
    B[2, 0] = dt
    B[3, 1] = dt
    return A, B


def rollout(
    controller: dict[str, np.ndarray], *, matched: bool, parameter_primitive: np.ndarray,
    disturbance_primitives: np.ndarray,
) -> np.ndarray:
    nominal = controller["nominal_states"]
    nominal_inputs = controller["nominal_inputs"]
    gains = controller["state_feedback_gains"]
    adaptive = bool(controller["adaptive"])
    dt = float(controller["dt"])
    parameter = controller["initial_parameter_center"] + float(controller["parameter_uncertainty_bound"]) * parameter_primitive
    x = np.array(controller["start_state"], dtype=float)
    if adaptive:
        x = np.concatenate([x, controller["initial_parameter_center"]])
    history = [x.copy()]
    deviations: list[np.ndarray] = []
    estimator_gains = controller["estimator_gains"]
    estimator_sensitivity = controller["estimator_sensitivity_inverses"]
    u_lower, u_upper = controller["input_lower"], controller["input_upper"]
    disturbance_scale = float(controller["exogenous_disturbance_scale"])

    for step in range(nominal_inputs.shape[0]):
        delta = x - nominal[step, : len(x)]
        delta[2] = wrap_to_pi(delta[2])
        deviations.append(delta)
        feedback = sum(gains[step, previous] @ deviations[previous] for previous in range(step + 1))
        u = np.clip(nominal_inputs[step] + feedback, u_lower, u_upper)
        physical_next = physical_step(x[:4], u, parameter, dt, matched)
        x_next_physical = physical_next + disturbance_scale * disturbance_primitives[step]
        if adaptive:
            A, B = linearized_dynamics(nominal[step, :6], dt, matched)
            delta_u = u - nominal_inputs[step]
            predicted_physical = nominal[step + 1, :4] + (A @ delta + B @ delta_u)[:4]
            innovation = estimator_sensitivity[step] @ (x_next_physical - predicted_physical)
            parameter_estimate = x[4:6] + estimator_gains[step] @ innovation
            x = np.concatenate([x_next_physical, parameter_estimate])
        else:
            x = x_next_physical
        history.append(x.copy())
    return np.asarray(history)


def load_controller(path: Path) -> dict[str, np.ndarray]:
    verification_path = path.with_name("verification_summary.json")
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if verification.get("saved_arrays_finite") is False:
            raise RuntimeError(
                f"{path.parent} has a failed solve; refusing to use its stale controller archive."
            )
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def sample_case(case_dir: Path, *, matched: bool, n_random: int, n_adversarial: int, seed: int) -> None:
    controller = load_controller(case_dir / "controller.npz")
    horizon = controller["nominal_inputs"].shape[0]
    n_w = 4  # The saved car policies use the four physical-state disturbance primitives.
    n_theta = len(controller["initial_parameter_center"])
    rng = np.random.default_rng(seed)
    parameter_primitives = np.empty((n_random + n_adversarial, n_theta))
    disturbance_primitives = np.empty((n_random + n_adversarial, horizon, n_w))
    kinds: list[str] = []
    for index in range(n_random + n_adversarial):
        adversarial = index >= n_random
        if adversarial:
            parameter_primitives[index] = rng.choice((-1.0, 1.0), size=n_theta)
            disturbance_primitives[index] = rng.choice((-1.0, 1.0), size=(horizon, n_w))
            kinds.append("adversarial_corner")
        else:
            parameter_primitives[index] = rng.uniform(-1.0, 1.0, size=n_theta)
            disturbance_primitives[index] = rng.uniform(-1.0, 1.0, size=(horizon, n_w))
            kinds.append("random")
    rollouts = np.asarray(
        [
            rollout(
                controller,
                matched=matched,
                parameter_primitive=parameter_primitives[index],
                disturbance_primitives=disturbance_primitives[index],
            )
            for index in range(len(kinds))
        ]
    )
    true_parameters = (
        controller["initial_parameter_center"][None, :]
        + float(controller["parameter_uncertainty_bound"]) * parameter_primitives
    )
    # Keep the table generator's established keys while adding the sampled
    # primitive sequences needed to reproduce every rollout exactly.
    np.savez_compressed(
        case_dir / "rollout_metrics.npz",
        rollout_states=rollouts,
        nominal_states=controller["nominal_states"],
        state_tube_half_widths=controller["state_tube_half_widths"],
        goal_state=controller["goal_state"],
        true_parameters=true_parameters,
        parameter_primitives=parameter_primitives,
        disturbance_primitives=disturbance_primitives,
        rollout_kinds=np.asarray(kinds),
        parameter_tube_half_widths=(
            np.sum(np.abs(controller["parameter_generators"]), axis=-1)
            if bool(controller["adaptive"]) and "parameter_generators" in controller
            else np.empty((0, n_theta))
        ),
    )
    with (case_dir / "rollout.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_s", "x_m", "y_m", "heading_rad", "speed_m_per_s"])
        for step, state in enumerate(rollouts[0, :, :4]):
            writer.writerow([step, step * float(controller["dt"]), *state])
    metadata = {
        "schema_version": 2,
        "num_random_rollouts": n_random,
        "num_adversarial_rollouts": n_adversarial,
        "adversarial_disturbance_primitives": "independent samples from {-1, 1}^{n_w} at every time step",
        "adversarial_parameter_primitives": "one sample from {-1, 1}^{n_theta}; parameter is at a box corner",
        "seed": seed,
    }
    (case_dir / "rollout_sampling_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Replace the internal-rollout diagnostic with the just-saved 30+10
    # policy-rollout archive.
    plot_case(case_dir, case_dir / "all_rollouts_deviation_vs_tube_width.png")
    print(f"Saved {n_random + n_adversarial} rollouts to {case_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--random-rollouts", type=int, default=N_RANDOM_DEFAULT)
    parser.add_argument("--adversarial-rollouts", type=int, default=N_ADVERSARIAL_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.random_rollouts < 1 or args.adversarial_rollouts < 1:
        parser.error("rollout counts must be positive")
    for model in ("unmatched", "matched"):
        for adaptive in (False, True):
            for leb in (False, True):
                case = args.results_dir / model / (
                    f"adaptive_{str(adaptive).lower()}__leb_{str(leb).lower()}"
                )
                sample_case(
                    case,
                    matched=(model == "matched"),
                    n_random=args.random_rollouts,
                    n_adversarial=args.adversarial_rollouts,
                    seed=args.seed + 10_000 * (model == "matched") + 100 * adaptive + leb,
                )


if __name__ == "__main__":
    main()
