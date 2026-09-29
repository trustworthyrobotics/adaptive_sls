from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .interval import Interval
from .polynomial import LinPoly, _poly_unary1

Array = jnp.ndarray


# =========================
# Linear TM (poly + remainder)
# =========================
@dataclass
class LinTM:
    """
    Shapes:
      P.c:[B,D], P.L:[B,D,V];  R:[B,D]
    """
    P: LinPoly
    R: Interval

    # ---- Basic ----
    @property
    def B(self): return self.P.B

    @property
    def D(self): return self.P.D

    @property
    def V(self): return self.P.V

    def clone(self) -> "LinTM":
        return LinTM(self.P.clone(), self.R.clone())

    # ---- Constructors ----
    @staticmethod
    def zeros(B: int, D: int, V: int, dtype=jnp.float32) -> "LinTM":
        return LinTM(LinPoly.zeros(B, D, V, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def const(B: int, D: int, V: int, c: float, dtype=jnp.float32) -> "LinTM":
        return LinTM(LinPoly.const(B, D, V, c, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def var(B: int, D: int, V: int, idx: int, dtype=jnp.float32) -> "LinTM":
        return LinTM(LinPoly.var(B, D, V, idx, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def from_poly(P: LinPoly) -> "LinTM":
        return LinTM(P, Interval.zeros_like(P.c))

    # ======================
    # Helpers & coercions
    # ======================
    @staticmethod
    def to_tm_like(x, ref: "LinTM") -> "LinTM":
        """Coerce numeric or LinPoly to LinTM (shape Bx?xV) matching ref's batch & V."""
        if isinstance(x, LinTM):
            return x
        if isinstance(x, LinPoly):
            return LinTM.from_poly(x)
        B = ref.P.c.shape[0]
        V = ref.P.L.shape[2]
        c = jnp.asarray(x, ref.P.c.dtype)
        if c.ndim == 0:
            c = jnp.broadcast_to(c, (B, 1))
        elif c.ndim == 1:
            c = jnp.broadcast_to(c[None, :], (B, c.shape[0]))
        else:
            raise TypeError(f"Cannot coerce to LinTM: shape {c.shape}, expected scalar or (D,)")
        L = jnp.zeros((B, c.shape[1], V), ref.P.L.dtype)
        return LinTM.from_poly(LinPoly(c, L))

    @staticmethod
    def slice(x: "LinTM", start: int, limit: int) -> "LinTM":
        P = LinPoly.slice(x.P, start, limit)
        lo = x.R.lo[:, start:limit]
        hi = x.R.hi[:, start:limit]
        return LinTM(P, Interval(lo, hi))

    @staticmethod
    def concat(xs: list["LinTM"]) -> "LinTM":
        P = LinPoly.concat([x.P for x in xs])
        R = Interval.concat([x.R for x in xs])
        return LinTM(P, R)

    @staticmethod
    def repeat(x: "LinTM", n: int) -> "LinTM":
        """Repeat each component n times along state dim (e.g., for splitting)."""
        P_rep = LinPoly.repeat(x.P, n)
        R_rep = Interval.repeat(x.R, n)
        return LinTM(P_rep, R_rep)

    def append_zero_var(self, n: int) -> "LinTM":
        return LinTM(self.P.append_zero_var(n), self.R.append_zero_var(n))

    # ---- Algebra ----
    def add(self, other: "LinTM") -> "LinTM":
        return LinTM(self.P.add(other.P), Interval(self.R.lo + other.R.lo, self.R.hi + other.R.hi))

    def sub(self, other: "LinTM") -> "LinTM":
        return LinTM(self.P.sub(other.P), Interval(self.R.lo - other.R.hi, self.R.hi - other.R.lo))

    def scale(self, k: Array | float) -> "LinTM":
        return LinTM(self.P.scale(k), self.R.scale(k))

    # ---- Interval evaluation over a box ----
    def _eval_interval_affine(self, box_lo: Array, box_hi: Array) -> Interval:
        """Evaluate TM(z) = P(z) + R with affine polynomial part."""
        I_poly = self.P._eval_interval_affine(box_lo, box_hi)
        return I_poly.add(self.R)

    def eval_interval(self, box_lo: Array, box_hi: Array) -> Interval:
        return self._eval_interval_affine(box_lo, box_hi)

    def _mul_ctrunc1(self, other: "LinTM", box_lo: Array, box_hi: Array) -> "LinTM":
        """
        TM×TM product with affine truncation on the polynomial part.

        If A = (P1 + I1), B = (P2 + I2), then
        poly_out = affine part of (P1 P2)
        R_out    = I1*I2 + rng(P2)*I1 + rng(P1)*I2 + trunc_overflow(P1,P2)
        """
        P1, R1 = self.P, self.R
        P2, R2 = other.P, other.R

        P_out, I_trunc = P1._mul_ctrunc1(P2, box_lo, box_hi)

        RP1 = P1._eval_interval_affine(box_lo, box_hi)
        RP2 = P2._eval_interval_affine(box_lo, box_hi)

        I1xI2 = R1.mul(R2)
        P2xI1 = RP2.mul(R1)
        P1xI2 = RP1.mul(R2)

        R_out = Interval(
            I1xI2.lo + P2xI1.lo + P1xI2.lo + I_trunc.lo,
            I1xI2.hi + P2xI1.hi + P1xI2.hi + I_trunc.hi,
        )

        return LinTM(P_out, R_out)

    def mul(self, other: "LinTM", box_lo: Array, box_hi: Array) -> "LinTM":
        return self._mul_ctrunc1(other, box_lo, box_hi)

    def _recip_ctrunc1(self, box_lo: Array, box_hi: Array) -> "LinTM":
        """
        Order-1 reciprocal of TM (Flow* style) under affine assumption.
        Keeps poly: (1/c) - (L·z)/c^2
        Remainder: Horner multiply overflow (order-1) + Lagrange tail (p=2).
        """
        P, R = self.P, self.R
        c = P.c
        inv_c = 1.0 / c
        is_const_poly = self.P.is_const_poly()

        def _branch_const_poly():
            is_zero_R = self.R.is_zero()

            def _branch_zero_R():
                tm_out = self.clone()
                tm_out.P.c = inv_c
                return tm_out

            def _branch_nonzero_R():
                I_in = R.add(c)
                I_out = I_in.recip()
                m = I_out.midpoint()
                tm_out = self.clone()
                tm_out.P.c = m
                tm_out.R = I_out.sub(m)
                return tm_out

            return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R)

        def _branch_general_poly():
            return self._recip_ctrunc1_general(box_lo, box_hi)

        return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general_poly)

    def _recip_ctrunc1_general(self, box_lo: Array, box_hi: Array) -> "LinTM":
        """
        Order-1 reciprocal of TM (Flow* style) under affine assumption.
        Keeps poly: (1/c) - (L·z)/c^2
        Remainder: Horner multiply overflow (order-1) + Lagrange tail (p=2).
        """
        P, R = self.P, self.R
        c = P.c
        inv_c = 1.0 / c

        P_out = LinPoly(c=inv_c, L=-P.L * (inv_c[:, :, None] * inv_c[:, :, None]))

        G_poly = LinPoly(c=jnp.zeros_like(c), L=P.L * inv_c[:, :, None])
        G = LinTM(G_poly, R.scale(inv_c))

        H = LinTM.zeros(P.B, P.D, P.V)
        H.P.c = H.P.c - 1
        H = H._mul_ctrunc1(G, box_lo, box_hi)
        H.P.c = H.P.c + 1
        H = H.scale(inv_c)

        I_P = P._eval_interval_affine(box_lo, box_hi)
        I_F = I_P.sub(c).add(R)
        Dom = I_F.add(c)
        ratio = I_F.div(Dom)
        Tail = ratio.pow(2).div(Dom).scale(inv_c)

        R_out = H.R.add(Tail)

        return LinTM(P_out, R_out)

    def _div_ctrunc1(self, other: "LinTM", box_lo: Array, box_hi: Array) -> "LinTM":
        """TM division with affine truncation (order-1)."""
        recip_tm = other._recip_ctrunc1(box_lo, box_hi)
        return self._mul_ctrunc1(recip_tm, box_lo, box_hi)

    def recip(self, box_lo: Array, box_hi: Array) -> "LinTM":
        return self._recip_ctrunc1(box_lo, box_hi)

    def div(self, other: "LinTM", box_lo: Array, box_hi: Array) -> "LinTM":
        return self._div_ctrunc1(other, box_lo, box_hi)

    def evaluate_time(self, h: float) -> "LinTM":
        return LinTM(self.P.evaluate_time(h), self.R)

    def is_zero(self) -> Array:
        """Check if TM is identically zero over all batches and dimensions."""
        return self.P.is_zero() & self.R.is_zero()

    def compose_affine(self, other: "LinTM", h: float = 0.0) -> "LinTM":
        """
        other = A z + b + R_other
        self = c + L·z + R

        self.compose_affine(other) =
            c + L·(Az + b) + L·R_other + R
        """
        _ = h
        poly = self.P.compose_affine(other.P)
        R_total = self.R.add(other.R.affine(self.P.L[:, :, 1:]))
        return LinTM(poly, R_total)

    def log(self, prefix: str = "LinTM", dim=None):
        jax.debug.print(f"------ {prefix} ------")
        self.P.log(dim=dim)
        self.R.log(dim=dim)
        return


def tm_unary_lagrange(
    A: LinTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_f: Callable[[Interval], Interval],
    iv_fpp: Callable[[Interval], Interval],
    iv_fppp: Callable[[Interval], Interval],
) -> LinTM:
    _ = fpp, iv_fppp
    return _tm_unary_lagrange1(A, f, df, box_lo, box_hi, iv_f, iv_fpp)


def _tm_unary_lagrange1(
    A: LinTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_f: Callable[[Interval], Interval],
    iv_fpp: Callable[[Interval], Interval],
) -> LinTM:
    """
    Flow*-mirroring order-1 keep:
    P_keep = f(c) + f'(c) * g,  g = P - c.
    R_out  ⊇ f'(c) * R_in  ⊕  (1/2) * T^2 * hull(f''(c+T)),  T = range(g) ⊕ R_in.
    """

    c = A.P.c

    is_const_poly = A.P.is_const_poly()

    def _branch_const_poly(A: LinTM):
        is_zero_R = A.R.is_zero()

        def _branch_zero_R(A: LinTM):
            tm_out = A.clone()
            tm_out.P.c = f(c)
            return tm_out

        def _branch_nonzero_R(A: LinTM):
            I_in = A.R.add(c)
            I_out = iv_f(I_in)
            m = I_out.midpoint()
            tm_out = A.clone()
            tm_out.P.c = m
            tm_out.R = I_out.sub(m)
            return tm_out

        return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R, A)

    def _branch_general(A: LinTM):
        return _tm_unary_lagrange1_general(A, f, df, box_lo, box_hi, iv_fpp)

    return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general, A)


def _tm_unary_lagrange1_general(
    A: LinTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_fpp: Callable[[Interval], Interval],
) -> LinTM:
    """
    Flow*-mirroring order-1 keep:
    P_keep = f(c) + f'(c) * g,  g = P - c.
    R_out  ⊇ f'(c) * R_in  ⊕  (1/2) * T^2 * hull(f''(c+T)),  T = range(g) ⊕ R_in.
    """

    c = A.P.c
    Pkeep = _poly_unary1(A.P, f, df)

    s1 = df(A.P.c)
    R_lin = A.R.scale(s1)

    I_poly = A.P._eval_interval_affine(box_lo, box_hi)
    G = I_poly.sub(c)

    T = G.add(A.R)

    CplusT = T.add(c)
    H = iv_fpp(CplusT)

    T2 = T.square()
    half = Interval(jnp.asarray(0.5, A.P.c.dtype), jnp.asarray(0.5, A.P.c.dtype))
    R2 = T2.mul(H).mul(half)

    R = R_lin.add(R2)
    return LinTM(Pkeep, R)


# ---------- PyTree registrations ----------
def _lintm_flatten(tm: LinTM):
    return ((tm.P, tm.R), None)


def _lintm_unflatten(aux, children):
    P, R = children
    return LinTM(P, R)


jax.tree_util.register_pytree_node(LinTM, _lintm_flatten, _lintm_unflatten)
