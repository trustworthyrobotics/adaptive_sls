from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable

import jax
import jax.numpy as jnp

from .interval import Interval

Array = jnp.ndarray


# ========================================
# Linear polynomial
#   P(t,x) = c + L·z, where z = [t, x1, ..., xD].
# ========================================
@dataclass
class LinPoly:
    """
    Shapes:
      c  : [B, D]
      L  : [B, D, V]   (coeffs for z_v, with v=0 for time)
    """
    c: Array
    L: Array

    # ---- Basic ----
    @property
    def B(self): return self.c.shape[0]

    @property
    def D(self): return self.c.shape[1]

    @property
    def V(self): return self.L.shape[2]

    def clone(self) -> "LinPoly":
        return LinPoly(jnp.array(self.c, copy=True), jnp.array(self.L, copy=True))

    # ---- Constructors ----
    @staticmethod
    def zeros(B: int, D: int, V: int, dtype=jnp.float32) -> "LinPoly":
        zc = jnp.zeros((B, D), dtype)
        zV = jnp.zeros((B, D, V), dtype)
        return LinPoly(zc, zV)

    @staticmethod
    def const(B: int, D: int, V: int, c: float, dtype=jnp.float32) -> "LinPoly":
        c0 = jnp.full((B, D), c, dtype)
        zV = jnp.zeros((B, D, V), dtype)
        return LinPoly(c0, zV)

    @staticmethod
    def var(B: int, D: int, V: int, idx: int, dtype=jnp.float32) -> "LinPoly":
        L = jnp.zeros((B, D, V), dtype).at[:, :, idx].set(1.0)
        zc = jnp.zeros((B, D), dtype)
        return LinPoly(zc, L)

    # ======================
    # Helpers & coercions
    # ======================

    @staticmethod
    def to_poly_like(x, ref: "LinPoly") -> "LinPoly":
        """Coerce numeric to LinPoly (shape Bx?xV) matching ref's batch & V."""
        if isinstance(x, LinPoly):
            return x
        B = ref.c.shape[0]
        V = ref.L.shape[2]
        c = jnp.asarray(x, ref.c.dtype)
        if c.ndim == 0:
            c = jnp.broadcast_to(c, (B, 1))
        elif c.ndim == 1:
            c = jnp.broadcast_to(c[None, :], (B, c.shape[0]))
        else:
            raise TypeError(f"Cannot coerce to LinPoly: shape {c.shape}, expected scalar or (D,)")
        L = jnp.zeros((B, c.shape[1], V), ref.L.dtype)
        return LinPoly(c, L)

    @staticmethod
    def slice(x: "LinPoly", start: int, limit: int) -> "LinPoly":
        """Select component slice along state dim; returns (B,limit-start,V)."""
        return LinPoly(x.c[:, start:limit], x.L[:, start:limit, :])

    @staticmethod
    def concat(xs: list["LinPoly"]) -> "LinPoly":
        c = jnp.concatenate([x.c for x in xs], axis=1)
        L = jnp.concatenate([x.L for x in xs], axis=1)
        return LinPoly(c, L)

    @staticmethod
    def repeat(x: "LinPoly", n: int) -> "LinPoly":
        """Repeat each component n times along state dim (e.g., for splitting)."""
        c_rep = jnp.repeat(x.c, n, axis=0)
        L_rep = jnp.repeat(x.L, n, axis=0)
        return LinPoly(c_rep, L_rep)

    def append_zero_var(self, n) -> "LinPoly":
        """Append a zero coefficient variable (increase D by n)."""
        B, V = self.B, self.V
        L_new = jnp.concatenate([self.L, jnp.zeros((B, n, V), dtype=self.L.dtype)], axis=1)
        c_new = jnp.concatenate([self.c, jnp.zeros((B, n), dtype=self.c.dtype)], axis=1)
        return LinPoly(c_new, L_new)

    # ---- Algebra ----
    def add(self, other: "LinPoly") -> "LinPoly":
        return LinPoly(self.c + other.c, self.L + other.L)

    def sub(self, other: "LinPoly") -> "LinPoly":
        return LinPoly(self.c - other.c, self.L - other.L)

    def scale(self, k: Array | float) -> "LinPoly":
        if isinstance(k, jnp.ndarray) and k.ndim == 2:
            return LinPoly(self.c * k, self.L * k[:, :, None])
        return LinPoly(self.c * k, self.L * k)

    # ---- Interval evaluation over box [lo,hi] in z-space ----
    @staticmethod
    def _lin_form_interval(a: Array, lo: Array, hi: Array) -> Interval:
        """
        Interval of a · z over z∈[lo,hi].  Shapes: a:[B,D,N], lo/hi:[B,N] → [B,D].
        """
        lo_e = lo[:, None, :]
        hi_e = hi[:, None, :]
        t_lo = jnp.minimum(a * lo_e, a * hi_e).sum(axis=-1)
        t_hi = jnp.maximum(a * lo_e, a * hi_e).sum(axis=-1)
        return Interval(t_lo, t_hi)

    def _eval_interval_affine(self, box_lo: Array, box_hi: Array) -> Interval:
        """Evaluate P(z) = c + L·z on z∈[box_lo, box_hi]."""
        I_lin = LinPoly._lin_form_interval(self.L, box_lo, box_hi)
        return Interval(self.c + I_lin.lo, self.c + I_lin.hi)

    def eval_interval(self, box_lo: Array, box_hi: Array) -> Interval:
        return self._eval_interval_affine(box_lo, box_hi)

    def _mul_trunc1(self, other: "LinPoly") -> "LinPoly":
        """
        Affine×Affine -> Affine polynomial product (truncate quadratic terms).

        If P1 = c1 + L1·z, P2 = c2 + L2·z, then
        keep:   c  = c1*c2
                L  = c1*L2 + c2*L1
        drop:   (L1·z)(L2·z)  (sent to overflow by mul_ctrunc).
        """
        c1, L1 = self.c, self.L
        c2, L2 = other.c, other.L

        c_out = c1 * c2
        L_out = L1 * c2[:, :, None] + L2 * c1[:, :, None]

        return LinPoly(c_out, L_out)

    def _mul_ctrunc1(self, other: "LinPoly", box_lo: Array, box_hi: Array) -> Tuple["LinPoly", Interval]:
        """
        Order-1 keep: keep affine; overflow = full quadratic of (L1·z)(L2·z).
        Diagonal z_i^2 via square bounds (nonnegative lower); off-diagonal z_i z_j via 4-corner bounds.
        """
        kept = self._mul_trunc1(other)

        a1 = self.L
        a2 = other.L
        Zlo, Zhi = box_lo, box_hi
        V = self.V

        def square_bounds(l, u):
            lo2 = jnp.minimum(l * l, u * u)
            hi2 = jnp.maximum(l * l, u * u)
            lo2 = jnp.where((l <= 0) & (u >= 0), jnp.zeros_like(lo2), lo2)
            return lo2, hi2

        Cdiag = a1 * a2
        Z2_lo, Z2_hi = square_bounds(Zlo, Zhi)
        lo_diag = jnp.minimum(Cdiag * Z2_lo[:, None, :], Cdiag * Z2_hi[:, None, :]).sum(axis=2)
        hi_diag = jnp.maximum(Cdiag * Z2_lo[:, None, :], Cdiag * Z2_hi[:, None, :]).sum(axis=2)

        C = a1[:, :, :, None] * a2[:, :, None, :]

        Zlo_i = Zlo[:, :, None]
        Zhi_i = Zhi[:, :, None]
        Zlo_j = Zlo[:, None, :]
        Zhi_j = Zhi[:, None, :]

        q1 = Zlo_i * Zlo_j
        q2 = Zlo_i * Zhi_j
        q3 = Zhi_i * Zlo_j
        q4 = Zhi_i * Zhi_j
        ZZ_lo = jnp.minimum(jnp.minimum(q1, q2), jnp.minimum(q3, q4))
        ZZ_hi = jnp.maximum(jnp.maximum(q1, q2), jnp.maximum(q3, q4))

        eye = jnp.eye(V, dtype=ZZ_lo.dtype)[None, :, :]
        mask_off = 1.0 - eye
        ZZ_lo_off = ZZ_lo * mask_off
        ZZ_hi_off = ZZ_hi * mask_off

        lo_off = jnp.minimum(C * ZZ_lo_off[:, None, :, :], C * ZZ_hi_off[:, None, :, :]).sum(axis=(2, 3))
        hi_off = jnp.maximum(C * ZZ_lo_off[:, None, :, :], C * ZZ_hi_off[:, None, :, :]).sum(axis=(2, 3))

        lo = lo_diag + lo_off
        hi = hi_diag + hi_off
        over = Interval(lo, hi)
        return kept, over

    def mul(self, other: "LinPoly") -> "LinPoly":
        return self._mul_trunc1(other)

    def mul_ctrunc(self, other: "LinPoly", box_lo: Array, box_hi: Array) -> Tuple["LinPoly", Interval]:
        return self._mul_ctrunc1(other, box_lo, box_hi)

    def _recip_trunc1(self) -> "LinPoly":
        """
        Affine reciprocal, polynomial part only (order-1).

        For P = c + L·z  →  (1/P)_{≤1} = (1/c) - (L·z)/c^2.
        """
        c = self.c
        L = self.L
        inv_c = 1.0 / c
        inv_c2 = inv_c * inv_c

        c_out = inv_c
        L_out = -L * inv_c2[:, :, None]
        return LinPoly(c_out, L_out)

    def _div_trunc1(self, other: "LinPoly") -> "LinPoly":
        """Affine division, polynomial part only (order-1): self * (other^{-1})_{≤1}."""
        recip_poly = other._recip_trunc1()
        return self._mul_trunc1(recip_poly)

    def recip(self) -> "LinPoly":
        return self._recip_trunc1()

    def div(self, other: "LinPoly") -> "LinPoly":
        return self._div_trunc1(other)

    # ---- Time substitution t := h, folding to degree-1 form ----
    def evaluate_time(self, h: float) -> "LinPoly":
        """
        Substitute t:=h.
        Returns a LinPoly with L[:, :, 0]=0 (time axis removed).
        """
        c_new = self.c + self.L[:, :, 0] * h
        L_new = jnp.zeros_like(self.L).at[:, :, 1:].set(self.L[:, :, 1:])
        return LinPoly(c_new, L_new)

    def is_zero(self) -> bool:
        """Check if polynomial is identically zero over all batches and dimensions."""
        return jnp.allclose(self.c, 0) & jnp.allclose(self.L, 0)

    def is_const_poly(self) -> Array:
        """Check if polynomial has no linear terms."""
        return jnp.allclose(self.L, 0)

    # ---- Affine composition z ↦ A z + b, with time at index 0 ----
    def compose_affine(self, other: "LinPoly") -> "LinPoly":
        """
        For P=c + L·z, under z -> Az + b and keeping t as first coord:
          L' = L @ A
          c' = c + L·b
        """
        A = other.L
        b = other.c
        t_id = jnp.zeros((self.B, 1, self.V), dtype=self.L.dtype).at[:, :, 0].set(1.0)
        A = jnp.concatenate([t_id, A], axis=1)
        b = jnp.concatenate([jnp.zeros((self.B, 1), dtype=self.c.dtype), b], axis=1)

        L_new = jnp.einsum("bdv,bvw->bdw", self.L, A)
        c_new = self.c + (self.L * b[:, None, :]).sum(axis=-1)
        return LinPoly(c_new, L_new)

    def log(self, prefix: str = "LinPoly", dim=None):
        if dim is None:
            jax.debug.print(f"{prefix}: c={self.c.tolist()}, L={self.L.tolist()}")
        else:
            jax.debug.print(f"{prefix}: c={self.c[:, dim].tolist()}, L={self.L[:, dim].tolist()}")
        return


# ======================
# Poly-level nonpoly helpers
# ======================
def poly_unary(
    P: LinPoly,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
) -> LinPoly:
    return _poly_unary1(P, f, df, fpp)


def _poly_unary1(P: LinPoly, f: Callable[[Array], Array], df: Callable[[Array], Array], *args) -> LinPoly:
    """Apply unary non-polynomial at poly level: 1st-order around constants."""
    c = f(P.c)
    s = df(P.c)
    L = P.L * s[:, :, None]
    return LinPoly(c, L)


# ---------- PyTree registrations ----------
def _linpoly_flatten(p: LinPoly):
    return ((p.c, p.L), None)


def _linpoly_unflatten(aux, children):
    c, L = children
    return LinPoly(c, L)


jax.tree_util.register_pytree_node(LinPoly, _linpoly_flatten, _linpoly_unflatten)
