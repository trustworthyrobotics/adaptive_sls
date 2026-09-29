import jax.numpy as jnp
from typing import Dict, Tuple, List
from .polynomial import LinPoly
from .taylor_model import LinTM


Array = jnp.ndarray


def build_linear_tm(c: Array, S: Array, dtype=jnp.float32) -> LinTM:
    """
    Build a degree-1 polynomial P(z)=c + sum_i S_i * y_i (time is index 0, y axes start at 1).
    c,S: [B,D].  Returns LinTM in linear form.
    """
    B, D = c.shape
    V = D + 1
    P = LinPoly.zeros(B, D, V, dtype=dtype)
    P.c = c.astype(dtype)
    idx = jnp.arange(D)
    # place scales on spatial axes (1..D)
    P.L = P.L.at[:, idx, idx + 1].set(S.astype(dtype))
    return LinTM.from_poly(P)

def make_step_boxes(B: int, D: int, h: float, dtype=jnp.float32) -> Tuple[Array, Array, Array, Array]:
    """
    Build step-local boxes:
      step_lo/hi:  t ∈ [0,h], y ∈ [-1,1]
      eval_lo/hi:  t fixed at 0, y ∈ [-1,1]
    """
    zeros = jnp.zeros((B, 1), dtype=dtype)
    hcol  = jnp.full((B, 1), h, dtype=dtype)
    ones  = jnp.ones((B, D), dtype=dtype)
    step_lo = jnp.concatenate([zeros, -ones], axis=1)
    step_hi = jnp.concatenate([hcol,  +ones], axis=1)
    eval_lo = jnp.concatenate([zeros, -ones], axis=1)
    eval_hi = jnp.concatenate([zeros,  +ones], axis=1)
    return step_lo, step_hi, eval_lo, eval_hi


def _splits_dict_to_list(splits_dict: Dict[int,int], D_total: int) -> List[int]:
    """
    Convert a sparse dict like {0:8,1:8,2:8,3:2} into a dense list of length D_total.
    Unspecified dims default to 1. Non-positive values raise.
    """
    out = [1]*D_total
    for k, v in (splits_dict or {}).items():
        idx = int(k)
        if 0 <= idx < D_total:
            vv = int(v)
            if vv <= 0:
                raise ValueError(f"splits[{idx}] must be positive; got {vv}.")
            out[idx] = vv
    return out


def split_initial_box(x0_lo: jnp.ndarray,
                      x0_hi: jnp.ndarray,
                      splits_per_dim) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split an initial hyper-rectangle into a uniform grid of sub-boxes.

    Args:
      x0_lo: (B, D) lower bounds (float32).
      x0_hi: (B, D) upper bounds (float32).
      splits_per_dim: int or sequence[int] of length D (number of splits per dim).

    Returns:
      parts_lo, parts_hi: both (B*M, D) where M = prod(splits_per_dim).
    """
    # Build grid of integer indices of shape (M, D) using jnp.indices (static shape).
    idx_grid = jnp.indices(splits_per_dim, dtype=x0_lo.dtype)        # (D, s1, s2, ..., sD)
    idx = jnp.reshape(jnp.moveaxis(idx_grid, 0, -1), (-1, len(splits_per_dim)))  # (M, D)
    s = jnp.asarray(splits_per_dim, x0_lo.dtype)[None, :]            # (1, D)

    # Fractions for each sub-box along each dim
    frac_lo = idx / s                                        # (M, D)
    frac_hi = (idx + 1.0) / s                                # (M, D)

    # Broadcast over batch to get all sub-box lows/highs
    widths = (x0_hi - x0_lo)                                 # (B, D)
    parts_lo = x0_lo[:, None, :] + widths[:, None, :] * frac_lo[None, :, :]  # (B, M, D)
    parts_hi = x0_lo[:, None, :] + widths[:, None, :] * frac_hi[None, :, :]  # (B, M, D)

    # Flatten batch and grid into one list of boxes
    B = x0_lo.shape[0]
    D = x0_lo.shape[1]
    return parts_lo.reshape(-1, D), \
           parts_hi.reshape(-1, D)


def prepare_initial_set(x0_lo: jnp.ndarray,
                           x0_hi: jnp.ndarray,
                           splits_cfg: Dict[int,int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    x0_lo: (B, D) lower bounds (float32).
    x0_hi: (B, D) upper bounds (float32).
    splits_cfg: dict[int,int] -> per-dim split counts (unspecified -> 1).
    Returns:
      parts_lo, parts_hi: jnp arrays [B*M, D] after splitting (M=∏ splits)
    """
    D = x0_lo.shape[1]
    splits_per_dim = _splits_dict_to_list(splits_cfg, D)  # len=D
    parts_lo, parts_hi = split_initial_box(x0_lo, x0_hi, splits_per_dim)  # [B*M, D] each
    return parts_lo, parts_hi
