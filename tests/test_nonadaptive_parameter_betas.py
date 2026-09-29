import sys
from pathlib import Path

import jax.numpy as jnp


# The repository also contains a legacy top-level ``gpu_sls.py`` module.
# Put the regular package under ``src/gpu_sls`` first while retaining the
# repository root for tests collected later in the same pytest process.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gpu_sls.gpu_sls import (
    get_betas,
    get_constraint_tightenings,
    get_nonadaptive_betas_persistent_parameter,
)


def test_persistent_parameter_is_summed_before_norm():
    T, nx, nu, nc = 2, 1, 1, 1
    C = jnp.ones((T + 1, nc, nx))
    D = jnp.zeros((T + 1, nc, nu))

    Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
    Phi_x = Phi_x.at[0, 0, 0, 0].set(1.0)
    Phi_x = Phi_x.at[1, :2, 0, 0].set(1.0)
    Phi_x = Phi_x.at[2, :3, 0, 0].set(1.0)
    Phi_u = jnp.zeros((T, T + 1, nu, nx))

    # Columns are [independent process, shared parameter, independent residual].
    # The parameter response cancels at k=1 because the same primitive drives
    # the +1 and -1 parameter generators.
    E = jnp.array(
        [
            [[0.2, 1.0, 0.1]],
            [[0.2, -1.0, 0.1]],
            [[0.2, 1.0, 0.1]],
        ]
    )

    independent_beta = get_betas(C, D, Phi_x, Phi_u, E)
    persistent_beta = get_nonadaptive_betas_persistent_parameter(
        C,
        D,
        Phi_x,
        Phi_u,
        E,
        n_w=1,
        n_param_generators=1,
    )

    independent_radius = get_constraint_tightenings(
        independent_beta, eps_beta=0.0
    )[:, 0]
    persistent_radius = get_constraint_tightenings(
        persistent_beta, eps_beta=0.0
    )[:, 0]

    assert jnp.allclose(independent_radius, jnp.array([1.3, 2.6, 3.9]))
    assert jnp.allclose(persistent_radius, jnp.array([1.3, 0.6, 1.9]))
    assert jnp.all(persistent_radius <= independent_radius)


def test_independent_disturbance_columns_remain_stagewise():
    T, nx, nu, nc = 2, 1, 1, 1
    C = jnp.ones((T + 1, nc, nx))
    D = jnp.zeros((T + 1, nc, nu))

    Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
    Phi_x = Phi_x.at[0, 0, 0, 0].set(1.0)
    Phi_x = Phi_x.at[1, :2, 0, 0].set(1.0)
    Phi_x = Phi_x.at[2, :3, 0, 0].set(1.0)
    Phi_u = jnp.zeros((T, T + 1, nu, nx))

    # Zero parameter block leaves the independent process and residual
    # calculation exactly equal to the original implementation.
    E = jnp.array(
        [
            [[0.2, 0.0, 0.1]],
            [[0.2, 0.0, 0.1]],
            [[0.2, 0.0, 0.1]],
        ]
    )

    independent_beta = get_betas(C, D, Phi_x, Phi_u, E)
    persistent_beta = get_nonadaptive_betas_persistent_parameter(
        C,
        D,
        Phi_x,
        Phi_u,
        E,
        n_w=1,
        n_param_generators=1,
    )

    assert jnp.allclose(
        get_constraint_tightenings(independent_beta, eps_beta=0.0),
        get_constraint_tightenings(persistent_beta, eps_beta=0.0),
    )
