"""Load and audit the official RAMPC-CCM polynomial metric.

The vendored MAT files are from ``ics/RAMPC-CCM`` commit
``fbab02d8acd955e1fc18d40b651d747bbe431367``.  The original certificate was
certify the same inverse-mass and scalar-disturbance model used here.  A
separate sampled audit checks the serialized metric against the exact
trigonometric dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import json
import re

import numpy as np
from scipy.io import loadmat

from .config import ExperimentConfig, ROOT, U_LOWER, U_UPPER, X_LOWER, X_UPPER
from .model import continuous_dynamics


Array = np.ndarray
UPSTREAM_COMMIT = "fbab02d8acd955e1fc18d40b651d747bbe431367"
DATA_DIR = ROOT / "third_party" / "rampc_ccm" / "data"
CONTROLLER_FILE = DATA_DIR / "own_rccm_0.7_w_0.1_th_0.01_pd_3.14_sc_-2.5.mat"
SYNTHESIZED_FILE = ROOT / "generated" / "ccm_two_axis_wind.npz"


def _matlab_function_expressions(matlab_function: object) -> tuple[list[str], tuple[int, int]]:
    source = matlab_function.item().function_handle.function
    match = re.fullmatch(r"sf%0@\(in1\)reshape\(\[(.*)\],\[(\d+),(\d+)\]\)", source)
    if match is None:
        raise ValueError("unsupported serialized MATLAB anonymous function")
    # Replace MATLAB indexing before splitting: the only commas inside an
    # element expression are the ``in1(3,:)``/``in1(4,:)`` subscripts.
    body = match.group(1).replace("in1(3,:)", "a").replace("in1(4,:)", "b")
    expressions = body.split(",")
    return expressions, (int(match.group(2)), int(match.group(3)))


def _compile_polynomial(matlab_function: object):
    expressions, shape = _matlab_function_expressions(matlab_function)
    translated = []
    for expression in expressions:
        expression = expression.replace(".^", "**").replace(".*", "*").replace("./", "/")
        translated.append(compile(expression, "<official-ccm-polynomial>", "eval"))

    def evaluate(state: Array) -> Array:
        state = np.asarray(state, dtype=float)
        namespace = {"a": float(state[2]), "b": float(state[3])}
        values = [eval(code, {"__builtins__": {}}, namespace) for code in translated]
        return np.asarray(values, dtype=float).reshape(shape, order="F")

    return evaluate


class OfficialCCM:
    """Official dual metric W and differential-controller numerator Y."""

    def __init__(self, path: str | Path = CONTROLLER_FILE) -> None:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)["controller"]
        self.W = _compile_polynomial(data.W_fcn)
        self.Y = _compile_polynomial(data.Y_fcn)
        self.original_rho = float(data.rho_c)
        self.source = "official_certified_inverse_mass_scalar_disturbance"
        self.is_sos_resynthesized = False

    def metric(self, state: Array) -> Array:
        return np.linalg.inv(self.W(state))

    def differential_gain(self, state: Array) -> Array:
        return self.Y(state) @ self.metric(state)

    def metric_derivatives(self, state: Array, epsilon: float = 1e-5) -> Array:
        derivatives = np.zeros((6, 6, 6), dtype=float)
        for index in (2, 3):
            delta = np.zeros(6)
            delta[index] = epsilon
            derivatives[index] = (self.W(state + delta) - self.W(state - delta)) / (
                2.0 * epsilon
            )
        return derivatives


class SynthesizedCCM:
    """Polynomial CCM resynthesized for the two-axis inertial wind box."""

    def __init__(self, path: str | Path = SYNTHESIZED_FILE) -> None:
        data = np.load(path, allow_pickle=False)
        self._w_coefficients = np.asarray(data["W_coefficients"], dtype=float)
        self._y_coefficients = np.asarray(data["Y_coefficients"], dtype=float)
        self._basis = np.asarray(data["basis_exponents"], dtype=int)
        self.original_rho = float(data["rho"])
        self.source = "resynthesized_two_axis_wind_sos"
        self.is_sos_resynthesized = True

    def _monomials(self, state: Array) -> Array:
        q = float(state[2]) / (np.pi / 3.0)
        a = float(state[3]) / 2.0
        return np.asarray([q**i * a**j for i, j in self._basis])

    def W(self, state: Array) -> Array:
        value = np.einsum("ijk,k->ij", self._w_coefficients, self._monomials(state))
        return 0.5 * (value + value.T)

    def Y(self, state: Array) -> Array:
        return np.einsum("ijk,k->ij", self._y_coefficients, self._monomials(state))

    def metric(self, state: Array) -> Array:
        return np.linalg.inv(self.W(state))

    def differential_gain(self, state: Array) -> Array:
        return self.Y(state) @ self.metric(state)

    def metric_derivatives(self, state: Array, epsilon: float = 1e-5) -> Array:
        derivatives = np.zeros((6, 6, 6), dtype=float)
        for index in (2, 3):
            delta = np.zeros(6)
            delta[index] = epsilon
            derivatives[index] = (self.W(state + delta) - self.W(state - delta)) / (
                2.0 * epsilon
            )
        return derivatives


def load_preferred_ccm(*, allow_rejected_fallback: bool = False) -> OfficialCCM | SynthesizedCCM:
    """Use the changed-model SOS result when present, otherwise the vendor candidate."""
    if SYNTHESIZED_FILE.exists():
        data = np.load(SYNTHESIZED_FILE, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("certificate_valid", False):
            return SynthesizedCCM(SYNTHESIZED_FILE)
        if not allow_rejected_fallback:
            raise RuntimeError(
                "two-axis SOS result was rejected: "
                f"status={metadata.get('status')}, "
                "coefficient_violation="
                f"{metadata.get('maximum_coefficient_constraint_violation')}, "
                f"minimum_gram_eigenvalue={metadata.get('minimum_gram_eigenvalue')}"
            )
    return OfficialCCM()


def _continuous_jacobians(
    state: Array,
    control: Array,
    inverse_mass_error: float,
    disturbance: float,
    config: ExperimentConfig,
    epsilon: float = 1e-5,
) -> tuple[Array, Array]:
    a = np.empty((6, 6))
    b = np.empty((6, 2))
    for index in range(6):
        delta = np.zeros(6)
        delta[index] = epsilon
        a[:, index] = (
            continuous_dynamics(
                state + delta, control, inverse_mass_error, config, disturbance
            )
            - continuous_dynamics(
                state - delta, control, inverse_mass_error, config, disturbance
            )
        ) / (2 * epsilon)
    for index in range(2):
        delta = np.zeros(2)
        delta[index] = epsilon
        b[:, index] = (
            continuous_dynamics(
                state, control + delta, inverse_mass_error, config, disturbance
            )
            - continuous_dynamics(
                state, control - delta, inverse_mass_error, config, disturbance
            )
        ) / (2 * epsilon)
    return a, b


@dataclass(frozen=True)
class CCMAudit:
    passed: bool
    sampled_points: int
    minimum_dual_metric_eigenvalue: float
    maximum_contraction_residual_eigenvalue: float
    minimum_sampled_contraction_rate: float
    requested_rate: float
    metric_source: str
    upstream_commit: str = UPSTREAM_COMMIT
    guarantee: str = "sampled numerical audit; not an SOS certificate"

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def audit_candidate_metric(
    ccm: OfficialCCM | SynthesizedCCM,
    config: ExperimentConfig,
    *,
    samples: int = 512,
    seed: int = 19,
    requested_rate: float | None = None,
) -> CCMAudit:
    """Audit the certified uncertainty box using exact trigonometric dynamics."""
    requested_rate = ccm.original_rho if requested_rate is None else requested_rate
    rng = np.random.default_rng(seed)
    min_w_eigenvalue = np.inf
    max_residual_eigenvalue = -np.inf
    minimum_rate = np.inf
    audit_points: list[tuple[Array, Array, Array]] = []
    for state_signs in product((-1.0, 1.0), repeat=4):
        state = np.zeros(6)
        state[2:] = np.asarray(state_signs) * X_UPPER[2:]
        for disturbance_sign, parameter_sign in product((-1.0, 1.0), repeat=2):
            audit_points.append(
                (
                    state.copy(),
                    np.full(2, config.hover_input),
                    parameter_sign * config.parameter_half_width,
                    disturbance_sign * config.disturbance_acceleration_half_width,
                )
            )
    for _ in range(samples):
        state = np.zeros(6)
        state[2:] = rng.uniform(X_LOWER[2:], X_UPPER[2:])
        control = rng.uniform(U_LOWER, U_UPPER)
        inverse_mass_error = rng.uniform(
            -config.parameter_half_width, config.parameter_half_width
        )
        disturbance = rng.uniform(
            -config.disturbance_acceleration_half_width,
            config.disturbance_acceleration_half_width,
        )
        audit_points.append((state, control, inverse_mass_error, disturbance))
    for state, control, inverse_mass_error, disturbance in audit_points:
        w_dual = ccm.W(state)
        y = ccm.Y(state)
        a, b = _continuous_jacobians(
            state, control, inverse_mass_error, disturbance, config
        )
        w_dot = np.einsum(
            "k,kij->ij",
            continuous_dynamics(
                state, control, inverse_mass_error, config, disturbance
            ),
            ccm.metric_derivatives(state),
        )
        residual_without_rate = -w_dot + a @ w_dual + w_dual @ a.T + b @ y + y.T @ b.T
        residual = residual_without_rate + 2.0 * requested_rate * w_dual
        inverse_sqrt = _inverse_sqrt(w_dual)
        normalized = inverse_sqrt @ residual_without_rate @ inverse_sqrt
        local_rate = -0.5 * np.linalg.eigvalsh(0.5 * (normalized + normalized.T))[-1]
        min_w_eigenvalue = min(min_w_eigenvalue, np.linalg.eigvalsh(w_dual)[0])
        max_residual_eigenvalue = max(
            max_residual_eigenvalue, np.linalg.eigvalsh(0.5 * (residual + residual.T))[-1]
        )
        minimum_rate = min(minimum_rate, local_rate)
    passed = min_w_eigenvalue > 0.0 and max_residual_eigenvalue <= 1e-7
    guarantee = (
        "SOS certificate for Chebyshev model plus sampled exact-trigonometric audit"
        if ccm.is_sos_resynthesized
        else "authors' SOS-certified uncertainty model plus sampled exact audit"
    )
    return CCMAudit(
        passed=bool(passed),
        sampled_points=len(audit_points),
        minimum_dual_metric_eigenvalue=float(min_w_eigenvalue),
        maximum_contraction_residual_eigenvalue=float(max_residual_eigenvalue),
        minimum_sampled_contraction_rate=float(minimum_rate),
        requested_rate=float(requested_rate),
        metric_source=ccm.source,
        guarantee=guarantee,
    )


def _inverse_sqrt(matrix: Array) -> Array:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors / np.sqrt(np.maximum(values, 1e-12))[None, :]) @ vectors.T


def disturbance_map(state: Array) -> Array:
    phi = float(state[2])
    c, s = np.cos(phi), np.sin(phi)
    result = np.zeros((6, 1))
    result[3:5, 0] = np.array([c, -s])
    return result


def parameter_map(control: Array) -> Array:
    result = np.zeros((6, 1))
    result[4, 0] = float(np.sum(control))
    return result


def parameter_lipschitz_constants(
    ccm: OfficialCCM | SynthesizedCCM,
    config: ExperimentConfig,
    *,
    samples: int = 256,
) -> Array:
    """The inverse-mass vector field is state-independent, hence Lipschitz zero."""
    del ccm, config, samples
    return np.zeros(1)


def zonotope_vertices(center: Array, generators: Array) -> Array:
    active = generators[:, np.linalg.norm(generators, axis=0) > 1e-14]
    if active.shape[1] > 12:
        # Box over-approximation avoids exponential enumeration.
        half = np.sum(np.abs(active), axis=1)
        active = np.diag(half)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=active.shape[1])))
    return np.asarray(center)[None, :] + signs @ active.T


def tube_derivative(
    delta: float,
    state: Array,
    control: Array,
    parameter_center: Array,
    parameter_generators: Array,
    ccm: OfficialCCM | SynthesizedCCM,
    config: ExperimentConfig,
    contraction_rate: float,
    parameter_lipschitz: Array,
) -> float:
    """Paper homothetic tube equation for scalar ``[d, theta]`` uncertainty."""
    metric = ccm.metric(state)
    parameter_vertices = zonotope_vertices(parameter_center, parameter_generators)
    parameter_terms = (
        (parameter_vertices - parameter_center) @ parameter_map(control).T
    )
    disturbance_terms = np.array([-1.0, 1.0])[:, None] * (
        config.disturbance_acceleration_half_width * disturbance_map(state).T
    )
    mismatch = (
        parameter_terms[:, None, :] + disturbance_terms[None, :, :]
    ).reshape(-1, 6)
    squared_norms = np.einsum("bi,ij,bj->b", mismatch, metric, mismatch)
    maximum_norm = float(np.sqrt(max(float(np.max(squared_norms)), 0.0)))
    half_width = np.sum(np.abs(parameter_generators), axis=1)
    effective_rate = contraction_rate - float(parameter_lipschitz @ half_width)
    return -effective_rate * float(delta) + maximum_norm


def tube_widths(
    delta: float, state: Array, ccm: OfficialCCM | SynthesizedCCM
) -> tuple[Array, Array, float]:
    w_dual = ccm.W(state)
    gain = ccm.differential_gain(state)
    state_width = float(delta) * np.sqrt(np.maximum(np.diag(w_dual), 0.0))
    input_width = float(delta) * np.sqrt(
        np.maximum(np.diag(gain @ w_dual @ gain.T), 0.0)
    )
    radial_width = float(delta) * np.sqrt(
        max(float(np.linalg.eigvalsh(w_dual[:2, :2])[-1]), 0.0)
    )
    return state_width, input_width, radial_width
