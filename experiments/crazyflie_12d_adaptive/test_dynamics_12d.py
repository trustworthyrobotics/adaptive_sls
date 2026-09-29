"""Fast model-level checks; these do not require MuJoCo or live hardware."""

import jax.numpy as jnp

from . import config_crazyflie_12d as config
from .dynamics_12d import discrete_jacobians, rk4_step, thrust_direction


def test_hover_and_jacobian_shapes():
    x = jnp.zeros(12, dtype=jnp.float32).at[2].set(0.5)
    beta = jnp.array([config.beta_hat_initial], dtype=jnp.float32)
    u = jnp.array([config.gravity / beta[0], 0.0, 0.0, 0.0], dtype=jnp.float32)
    next_x = rk4_step(x, u, beta, config.dt, **config.model_kwargs)
    A, B, E = discrete_jacobians(x, u, beta, config.dt, **config.model_kwargs)
    assert next_x.shape == (12,)
    assert A.shape == (12, 12)
    assert B.shape == (12, 4)
    assert E.shape == (12, 1)
    assert abs(float(next_x[5])) < 1.0e-5


def test_thrust_direction_and_beta_sensitivity():
    assert jnp.allclose(thrust_direction(0.0, 0.0, 0.0), jnp.array([0.0, 0.0, 1.0]))
    x = jnp.zeros(12, dtype=jnp.float32)
    u = jnp.array([0.3, 0.0, 0.0, 0.0], dtype=jnp.float32)
    low = rk4_step(x, u, jnp.array([20.0]), config.dt, **config.model_kwargs)
    high = rk4_step(x, u, jnp.array([30.0]), config.dt, **config.model_kwargs)
    assert high[5] > low[5]
