"""SOS resynthesis of a planar-quadrotor CCM for two-axis inertial wind.

This is a Python Gram-matrix implementation of the authors' offline YALMIP
formulation.  It searches for polynomial dual metric ``W(phi,v1)`` and
differential numerator ``Y(phi,v1)`` such that

    dW/dt - He(A W + B Y) - 2 rho W >= 0

on the normalized paper domain, at all four vertices of the constant wind
box.  The exact trigonometric functions are replaced by the same Chebyshev
polynomials used by the official RAMPC-CCM code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import time
from typing import Iterable

import cvxpy as cp
import numpy as np
import sympy as sp

from .config import ExperimentConfig, ROOT


Exponent = tuple[int, ...]
PolyMatrix = dict[Exponent, object]
DEFAULT_OUTPUT = ROOT / "generated" / "ccm_two_axis_wind.npz"


def monomial_exponents(variables: int, degree: int) -> list[Exponent]:
    return [
        exponent
        for exponent in product(range(degree + 1), repeat=variables)
        if sum(exponent) <= degree
    ]


def add_exponents(left: Exponent, right: Exponent) -> Exponent:
    return tuple(a + b for a, b in zip(left, right))


def add_poly(*polynomials: PolyMatrix) -> PolyMatrix:
    result: PolyMatrix = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, 0) + coefficient
    return result


def scale_poly(polynomial: PolyMatrix, scalar: float) -> PolyMatrix:
    return {exponent: scalar * coefficient for exponent, coefficient in polynomial.items()}


def transpose_poly(polynomial: PolyMatrix) -> PolyMatrix:
    return {exponent: coefficient.T for exponent, coefficient in polynomial.items()}


def constant_left_matmul(left: PolyMatrix, right: PolyMatrix) -> PolyMatrix:
    result: PolyMatrix = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = add_exponents(left_exp, right_exp)
            value = left_coefficient @ right_coefficient
            result[exponent] = result.get(exponent, 0) + value
    return result


def scalar_times_matrix(scalar: dict[Exponent, float], matrix: PolyMatrix) -> PolyMatrix:
    result: PolyMatrix = {}
    for scalar_exp, scalar_coefficient in scalar.items():
        for matrix_exp, matrix_coefficient in matrix.items():
            exponent = add_exponents(scalar_exp, matrix_exp)
            result[exponent] = result.get(exponent, 0) + scalar_coefficient * matrix_coefficient
    return result


def gram_polynomial(
    variables: int,
    monomial_degree: int,
    matrix_dimension: int,
    name: str,
) -> tuple[PolyMatrix, cp.Variable]:
    """Return coefficients of ``(z kron I)' Q (z kron I)``."""
    monomials = monomial_exponents(variables, monomial_degree)
    block_size = matrix_dimension
    gram = cp.Variable(
        (len(monomials) * block_size, len(monomials) * block_size),
        PSD=True,
        name=name,
    )
    coefficients: PolyMatrix = {}
    for row, row_exp in enumerate(monomials):
        row_slice = slice(row * block_size, (row + 1) * block_size)
        for column, column_exp in enumerate(monomials):
            column_slice = slice(column * block_size, (column + 1) * block_size)
            exponent = add_exponents(row_exp, column_exp)
            coefficients[exponent] = coefficients.get(exponent, 0) + gram[
                row_slice, column_slice
            ]
    return coefficients, gram


def localized_matrix_sos(
    polynomial: PolyMatrix,
    *,
    variables: int,
    matrix_dimension: int,
    domain_polynomials: list[dict[Exponent, float]],
    name: str,
    sos_degree: int | None = None,
    scalar_multipliers: bool = False,
    multiplier_sos_degree: int = 2,
) -> tuple[list[cp.Constraint], list[cp.Variable]]:
    """Impose ``P=S0+sum(g_i S_i)`` with matrix-SOS multipliers."""
    polynomial_degree = max((sum(exponent) for exponent in polynomial), default=0)
    if sos_degree is None:
        sos_degree = 2 * ((polynomial_degree + 1) // 2)
    if sos_degree < polynomial_degree or sos_degree % 2:
        raise ValueError("sos_degree must be even and cover the polynomial degree")
    s0, q0 = gram_polynomial(
        variables, sos_degree // 2, matrix_dimension, f"{name}_q0"
    )
    rhs = s0
    grams = [q0]
    for index, domain in enumerate(domain_polynomials):
        multiplier_dimension = 1 if scalar_multipliers else matrix_dimension
        localized_multiplier_degree = min(multiplier_sos_degree, sos_degree - 2)
        multiplier, gram = gram_polynomial(
            variables,
            localized_multiplier_degree // 2,
            multiplier_dimension,
            f"{name}_q{index + 1}",
        )
        if scalar_multipliers:
            multiplier = {
                exponent: coefficient[0, 0] * np.eye(matrix_dimension)
                for exponent, coefficient in multiplier.items()
            }
        rhs = add_poly(rhs, scalar_times_matrix(domain, multiplier))
        grams.append(gram)
    zero = np.zeros((matrix_dimension, matrix_dimension))
    exponents = set(polynomial) | set(rhs)
    constraints = [
        polynomial.get(exponent, zero) == rhs.get(exponent, zero)
        for exponent in exponents
    ]
    return constraints, grams


def sympy_matrix_polynomial(matrix: sp.Matrix, variables: Iterable[sp.Symbol]) -> PolyMatrix:
    variables = tuple(variables)
    result: PolyMatrix = {}
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            polynomial = sp.Poly(sp.expand(matrix[row, column]), *variables)
            for exponent, coefficient in polynomial.terms():
                if exponent not in result:
                    result[exponent] = np.zeros((matrix.rows, matrix.cols))
                result[exponent][row, column] = float(coefficient)
    return result


def model_polynomials(
    config: ExperimentConfig,
    wind: tuple[float, float],
) -> tuple[PolyMatrix, PolyMatrix, dict[Exponent, float], dict[Exponent, float]]:
    """Return A, B, normalized q-dot, and normalized v1-dot polynomials."""
    q, a, b, r = sp.symbols("q a b r")
    phi, v1, v2, omega = sp.symbols("phi v1 v2 omega")
    u1, u2 = sp.symbols("u1 u2")
    phi_limit = sp.pi / 3

    # Identical Chebyshev approximations to offline/load_system.m.
    sin_phi = sp.Float("0.9101") * (phi / phi_limit) - sp.Float("0.04466") * (
        4 * (phi / phi_limit) ** 3 - 3 * (phi / phi_limit)
    )
    cos_phi = sp.Float("0.7441") - sp.Float("0.2499") * (
        2 * (phi / phi_limit) ** 2 - 1
    )
    fx, fy = map(sp.Float, wind)
    dynamics = sp.Matrix(
        [
            v1 * cos_phi - v2 * sin_phi,
            v1 * sin_phi + v2 * cos_phi,
            omega,
            v2 * omega - config.gravity * sin_phi
            + (cos_phi * fx + sin_phi * fy) / config.mass,
            -v1 * omega - config.gravity * cos_phi + (u1 + u2) / config.mass
            + (-sin_phi * fx + cos_phi * fy) / config.mass,
            config.arm_length * (u1 - u2) / config.inertia,
        ]
    )
    physical_state = sp.Matrix(sp.symbols("px py phi v1 v2 omega"))
    # Substitute the shared symbols into the full state used for differentiation.
    px_s, py_s, phi_s, v1_s, v2_s, omega_s = physical_state
    dynamics_for_jacobian = dynamics.subs(
        {phi: phi_s, v1: v1_s, v2: v2_s, omega: omega_s}
    )
    a_matrix = dynamics_for_jacobian.jacobian(physical_state)
    b_matrix = dynamics_for_jacobian.jacobian(sp.Matrix([u1, u2]))
    normalization = {
        phi_s: phi_limit * q,
        v1_s: 2 * a,
        v2_s: b,
        omega_s: sp.pi * r,
    }
    a_matrix = sp.simplify(a_matrix.subs(normalization))
    b_matrix = sp.simplify(b_matrix.subs(normalization))
    q_dot = sp.simplify((omega_s / phi_limit).subs(normalization))
    a_dot = sp.simplify((dynamics_for_jacobian[3] / 2).subs(normalization))
    variables = (q, a, b, r)

    def scalar_coefficients(expression: sp.Expr) -> dict[Exponent, float]:
        return {
            exponent: float(coefficient)
            for exponent, coefficient in sp.Poly(sp.expand(expression), *variables).terms()
        }

    return (
        sympy_matrix_polynomial(a_matrix, variables),
        sympy_matrix_polynomial(b_matrix, variables),
        scalar_coefficients(q_dot),
        scalar_coefficients(a_dot),
    )


def decision_polynomial(
    *,
    rows: int,
    columns: int,
    symmetric: bool,
    name: str,
    degree: int,
) -> tuple[PolyMatrix, list[Exponent], list[cp.Variable]]:
    basis_2d = monomial_exponents(2, degree)
    polynomial: PolyMatrix = {}
    variables: list[cp.Variable] = []
    for index, exponent_2d in enumerate(basis_2d):
        coefficient = cp.Variable(
            (rows, columns), symmetric=symmetric, name=f"{name}_{index}"
        )
        variables.append(coefficient)
        exponent_4d = (exponent_2d[0], exponent_2d[1], 0, 0)
        polynomial[exponent_4d] = coefficient
    return polynomial, basis_2d, variables


def derivative_along(
    polynomial: PolyMatrix,
    q_dot: dict[Exponent, float],
    a_dot: dict[Exponent, float],
) -> PolyMatrix:
    derivative_q: PolyMatrix = {}
    derivative_a: PolyMatrix = {}
    for exponent, coefficient in polynomial.items():
        if exponent[0]:
            reduced = (exponent[0] - 1, exponent[1], exponent[2], exponent[3])
            derivative_q[reduced] = exponent[0] * coefficient
        if exponent[1]:
            reduced = (exponent[0], exponent[1] - 1, exponent[2], exponent[3])
            derivative_a[reduced] = exponent[1] * coefficient
    return add_poly(
        scalar_times_matrix(q_dot, derivative_q),
        scalar_times_matrix(a_dot, derivative_a),
    )


@dataclass(frozen=True)
class SynthesisResult:
    status: str
    objective: float
    solve_time_seconds: float
    rho: float
    output: Path
    certificate_valid: bool


def synthesize(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    rho: float = 0.2,
    solver: str = "SCS",
    max_iterations: int = 100_000,
    verbose: bool = True,
    polynomial_degree: int = 4,
    metric_floor: float = 0.01,
    contraction_margin: float = 0.0,
    scalar_multipliers: bool = False,
) -> SynthesisResult:
    config = ExperimentConfig()
    output = Path(output)
    w_poly, basis, w_variables = decision_polynomial(
        rows=6, columns=6, symmetric=True, name="W", degree=polynomial_degree
    )
    y_poly, _, y_variables = decision_polynomial(
        rows=2, columns=6, symmetric=False, name="Y", degree=polynomial_degree
    )

    epsilon = metric_floor
    w_positive = w_poly.copy()
    zero_exp = (0, 0, 0, 0)
    w_positive[zero_exp] = w_positive[zero_exp] - epsilon * np.eye(6)
    domain = []
    for index in range(4):
        exponent = [0, 0, 0, 0]
        exponent[index] = 2
        domain.append({zero_exp: 1.0, tuple(exponent): -1.0})

    # Positivity is needed only on the synthesis box. Localizing it is also
    # essential for an affine ansatz: a globally positive affine matrix
    # polynomial would necessarily be constant.
    positivity_degree = 2 * ((polynomial_degree + 1) // 2)
    constraints, all_grams = localized_matrix_sos(
        w_positive,
        variables=4,
        matrix_dimension=6,
        domain_polynomials=domain,
        name="W_positive",
        sos_degree=positivity_degree,
        scalar_multipliers=scalar_multipliers,
    )
    # Fix metric scale at hover; otherwise the homogeneous CCM inequality has
    # a trivial scaling degree of freedom.
    constraints.append(cp.trace(w_poly[zero_exp]) == 6.0)

    wind_values = (-config.wind_half_width, config.wind_half_width)
    for vertex_index, wind in enumerate(product(wind_values, repeat=2)):
        a_poly, b_poly, q_dot, a_dot = model_polynomials(config, wind)
        dwdt = derivative_along(w_poly, q_dot, a_dot)
        aw = constant_left_matmul(a_poly, w_poly)
        by = constant_left_matmul(b_poly, y_poly)
        contraction = add_poly(
            dwdt,
            scale_poly(aw, -1.0),
            scale_poly(transpose_poly(aw), -1.0),
            scale_poly(by, -1.0),
            scale_poly(transpose_poly(by), -1.0),
            scale_poly(w_poly, -2.0 * rho),
        )
        contraction[zero_exp] = contraction.get(zero_exp, 0) - (
            contraction_margin * np.eye(6)
        )
        vertex_constraints, grams = localized_matrix_sos(
            contraction,
            variables=4,
            matrix_dimension=6,
            domain_polynomials=domain,
            name=f"contraction_v{vertex_index}",
            scalar_multipliers=scalar_multipliers,
        )
        constraints.extend(vertex_constraints)
        all_grams.extend(grams)

    regularizer = 1e-8 * sum(
        cp.sum_squares(variable) for variable in [*w_variables, *y_variables]
    )
    problem = cp.Problem(cp.Minimize(regularizer), constraints)
    start = time.perf_counter()
    solve_options: dict[str, object] = {"solver": solver, "verbose": verbose}
    if solver == "SCS":
        solve_options.update(
            max_iters=max_iterations,
            eps=2e-5,
            acceleration_lookback=10,
            normalize=True,
        )
    problem.solve(**solve_options)
    elapsed = time.perf_counter() - start
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SOS synthesis failed with status {problem.status}")
    if any(variable.value is None for variable in [*w_variables, *y_variables]):
        raise RuntimeError("SOS solver returned no polynomial coefficients")

    maximum_constraint_violation = max(
        float(np.max(np.abs(constraint.violation()))) for constraint in constraints
    )
    minimum_gram_eigenvalue = min(
        float(np.linalg.eigvalsh(0.5 * (gram.value + gram.value.T))[0])
        for gram in all_grams
    )
    certificate_valid = bool(
        problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        and maximum_constraint_violation <= 5e-6
        and minimum_gram_eigenvalue >= -5e-7
    )

    w_coefficients = np.stack([variable.value for variable in w_variables], axis=-1)
    y_coefficients = np.stack([variable.value for variable in y_variables], axis=-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": problem.status,
        "objective": float(problem.value),
        "solve_time_seconds": elapsed,
        "solver": solver,
        "rho": rho,
        "wind_half_width": config.wind_half_width,
        "mass_known": True,
        "wind_frame": "inertial",
        "wind_dimension": 2,
        "state_domain_normalized": ["phi/(pi/3)", "v1/2", "v2/1", "omega/pi"],
        "trigonometric_approximation": "official RAMPC-CCM Chebyshev polynomials",
        "certificate_type": "Putinar matrix SOS with Gram matrices",
        "polynomial_degree": polynomial_degree,
        "metric_floor": metric_floor,
        "contraction_margin": contraction_margin,
        "maximum_coefficient_constraint_violation": maximum_constraint_violation,
        "minimum_gram_eigenvalue": minimum_gram_eigenvalue,
        "certificate_validation_tolerance": {
            "maximum_coefficient_constraint_violation": 5e-6,
            "minimum_gram_eigenvalue": -5e-7,
        },
        "certificate_valid": certificate_valid,
        "localization_multipliers": (
            "scalar_SOS_times_identity" if scalar_multipliers else "matrix_SOS"
        ),
    }
    np.savez_compressed(
        output,
        W_coefficients=w_coefficients,
        Y_coefficients=y_coefficients,
        basis_exponents=np.asarray(basis, dtype=int),
        rho=np.asarray(rho),
        wind_half_width=np.asarray(config.wind_half_width),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return SynthesisResult(
        problem.status, float(problem.value), elapsed, rho, output, certificate_valid
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument(
        "--solver", choices=("SCS", "CVXOPT", "CLARABEL", "MOSEK"), default="SCS"
    )
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument(
        "--polynomial-degree", type=int, choices=(0, 1, 2, 3, 4), default=4
    )
    parser.add_argument("--metric-floor", type=float, default=0.01)
    parser.add_argument("--contraction-margin", type=float, default=0.0)
    parser.add_argument(
        "--scalar-multipliers",
        action="store_true",
        help="use a smaller, more restrictive scalar-times-I localization",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = synthesize(
        output=args.output,
        rho=args.rho,
        solver=args.solver,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        polynomial_degree=args.polynomial_degree,
        metric_floor=args.metric_floor,
        contraction_margin=args.contraction_margin,
        scalar_multipliers=args.scalar_multipliers,
    )
    print(json.dumps({**result.__dict__, "output": str(result.output)}, indent=2))
    if not result.certificate_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
