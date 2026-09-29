"""Generate and persist common Monte-Carlo scenarios for every method."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ExperimentConfig


def generate_scenarios(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    parameters = rng.uniform(
        -config.parameter_half_width,
        config.parameter_half_width,
        size=(config.runs, 1),
    )
    disturbances = rng.uniform(
        -1.0,
        1.0,
        size=(config.runs, config.max_steps, 1),
    )
    return parameters, disturbances


def save_scenarios(path: str | Path, config: ExperimentConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters, disturbances = generate_scenarios(config)
    np.savez_compressed(
        path,
        true_inverse_mass_errors=parameters,
        disturbance_coefficients=disturbances,
        disturbance_acceleration_half_width=np.asarray(
            config.disturbance_acceleration_half_width
        ),
        seed=np.asarray(config.seed),
    )
    return path


def load_scenarios(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            data["true_inverse_mass_errors"].copy(),
            data["disturbance_coefficients"].copy(),
        )
