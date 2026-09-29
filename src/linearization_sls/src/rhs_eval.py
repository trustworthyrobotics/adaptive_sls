from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import jax.lax as jlax


from .interval import *
from .polynomial import LinPoly
from .taylor_model import LinTM, tm_unary_lagrange


Array = jnp.ndarray


# ======================
# JAXPR tracing
# ======================
@dataclass
class ClosedProg:
    jaxpr: Any
    consts: Any
    D: int

def _trace_prog(f: Callable, D: int) -> ClosedProg:
    x0 = jnp.zeros((D,), jnp.float32)
    closed = jax.make_jaxpr(f)(x0)  # ClosedJaxpr
    return ClosedProg(closed.jaxpr, closed.consts, D)


# ----------------------
# TM interpreter
# ----------------------
def build_rhs_tm_from_prog(prog: ClosedProg, V: int) -> Callable[[LinTM, Array, Array], LinTM]:
    """Evaluate RHS into a LinTM with reduced-order-2 keep for unary ops and Lagrange remainders."""
    jaxpr, consts, D = prog.jaxpr, prog.consts, prog.D

    def eval_tm(x_tm: LinTM, box_lo: Array, box_hi: Array) -> LinTM:
        # x_tm.log("Entering TM eval")
        env: Dict[Any, Any] = {}

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        x_var = jaxpr.invars[0]
        env[x_var] = x_tm  # enforce degree-1 on input TM

        for eqn in jaxpr.eqns:
            p = eqn.primitive
            pname = getattr(p, "name", "")
            ins = [env[i] if not hasattr(i, "val") else i.val for i in eqn.invars]

            if (p in (jlax.squeeze_p, jlax.reshape_p, jlax.convert_element_type_p)
                or pname in ("stop_gradient", "tie_in")):
                out = LinTM.to_tm_like(ins[0], x_tm)
            elif p is jlax.broadcast_in_dim_p:
                broadcast_shape = eqn.params["shape"]
                if broadcast_shape == (1,):
                    out = LinTM.to_tm_like(ins[0], x_tm)
                elif not isinstance(ins[0], LinTM):
                    out = LinTM.to_tm_like(jnp.broadcast_to(ins[0], eqn.params["shape"]), x_tm)
                else:
                    raise NotImplementedError("Broadcasting LinTM not implemented")
            elif p is jlax.slice_p:
                starts  = tuple(eqn.params.get("start_indices", ()))
                limits  = tuple(eqn.params.get("limit_indices", ()))
                strides = eqn.params.get("strides", None) or (1,)
                assert len(starts)==len(limits)==len(strides)==1, "only 1D slice supported"
                x = ins[0]
                if isinstance(x, LinTM):
                    out = LinTM.slice(x, int(starts[0]), int(limits[0]))
                else:
                    out = p.bind(*ins, **eqn.params)

            elif p is jlax.add_p:
                a, b = ins
                if isinstance(a, LinTM) or isinstance(b, LinTM):
                    out = LinTM.to_tm_like(a, x_tm).add(LinTM.to_tm_like(b, x_tm))
                else:
                    out = a + b

            elif p is jlax.sub_p:
                a, b = ins
                if isinstance(a, LinTM) or isinstance(b, LinTM):
                    out = LinTM.to_tm_like(a, x_tm).sub(LinTM.to_tm_like(b, x_tm))
                else:
                    out = a - b

            elif p is jlax.neg_p:
                a = ins[0]
                if isinstance(a, LinTM):
                    out = LinTM(a.P.scale(-1.0), Interval(-a.R.hi, -a.R.lo))
                elif isinstance(a, LinPoly):
                    out = LinTM.from_poly(a.scale(-1.0))
                else:
                    out = -a

            elif p is jlax.mul_p:
                a, b = ins
                if isinstance(a, (LinTM, LinPoly)) or isinstance(b, (LinTM, LinPoly)):
                    A = LinTM.to_tm_like(a, x_tm)
                    B = LinTM.to_tm_like(b, x_tm)
                    out = A._mul_ctrunc1(B, box_lo, box_hi)
                else:
                    out = a * b

            # ---- Non-polynomial unary (reduced-order-2 keep) ----
            elif p is jlax.sin_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.sin, jnp.cos, lambda x: -jnp.sin(x), box_lo, box_hi, iv_sin, iv_sin_fpp, iv_sin_fppp)
            elif p is jlax.cos_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.cos, lambda x: -jnp.sin(x), lambda x: -jnp.cos(x), box_lo, box_hi, iv_cos, iv_cos_fpp, iv_cos_fppp)
            elif p is jlax.tanh_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.tanh, lambda x: 1.0 - jnp.tanh(x)**2,
                                          lambda x: -2.0*jnp.tanh(x)*(1.0 - jnp.tanh(x)**2),
                                          box_lo, box_hi, iv_tanh, iv_tanh_fpp, iv_tanh_fppp)
            elif p is jlax.tan_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.tan, lambda x: 1.0 / (jnp.cos(x) ** 2),
                                          lambda x: 2.0*jnp.tan(x) / (jnp.cos(x) ** 2),
                                          box_lo, box_hi, iv_tan, iv_tan_fpp, iv_tan_fppp)

            elif p is jlax.exp_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.exp, jnp.exp, jnp.exp, box_lo, box_hi, iv_exp, iv_exp_fpp, iv_exp_fppp)
            elif p is jlax.log_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.log, lambda x: 1.0 / x, lambda x: -1.0 / (x * x), box_lo, box_hi, iv_log, iv_log_fpp, iv_log_fppp)

            elif p is jlax.sqrt_p:
                A = LinTM.to_tm_like(ins[0], x_tm)
                out = tm_unary_lagrange(A, jnp.sqrt, lambda x: 0.5 / jnp.sqrt(x), lambda x: -0.25 / (x ** 1.5), box_lo, box_hi, iv_sqrt, iv_sqrt_fpp, iv_sqrt_fppp)
            elif p is jlax.div_p:
                a, b = ins
                if isinstance(b, (LinTM, LinPoly)):
                    A = LinTM.to_tm_like(a, x_tm)
                    B = LinTM.to_tm_like(b, x_tm)
                    out = A.div(B, box_lo, box_hi)
                elif isinstance(a, (LinTM, LinPoly)):
                    out = LinTM.to_tm_like(a, x_tm).scale(1.0 / b)
                else:
                    out = a / b

            # elif p is jlax.pow_p:
            #     a, b = ins
            #     if isinstance(b, (int, float)) or (isinstance(b, jnp.ndarray) and b.ndim == 0):
            #         pexp = float(b)
            #     else:
            #         raise TypeError("Non-scalar exponent not supported for LinTM power")
            #     A = LinTM.to_tm_like(a, x_tm)
            #     out = _tm_pow(A, pexp, box_lo, box_hi)

            # elif getattr(jlax, "integer_pow_p", None) is not None and p is jlax.integer_pow_p:
            #     a = ins[0]
            #     n = None
            #     for k in ("y", "exponent", "n", "pow", "power"):
            #         if k in eqn.params:
            #             n = int(eqn.params[k])
            #             break
            #     if n is None:
            #         raise TypeError("integer_pow missing integer exponent parameter")
            #     A = LinTM.to_tm_like(a, x_tm)
            #     out = _tm_pow(A, float(n), box_lo, box_hi)

            elif p is jlax.concatenate_p:
                tms = [u if isinstance(u, LinTM) else LinTM.to_tm_like(u, x_tm) for u in ins]
                out = LinTM.concat(tms)

            else:
                if any(isinstance(u, (LinPoly, LinTM)) for u in ins):
                    raise TypeError(
                        f"Unsupported primitive in linear TM eval: {pname} with operands {[type(u) for u in ins]}"
                    )
                else:
                    out = p.bind(*ins, **eqn.params)

            if eqn.primitive.multiple_results:
                raise NotImplementedError("Multiple results not expected in current RHS forms")
            else:
                env[eqn.outvars[0]] = out

        out_tm = env[jaxpr.outvars[0]]
        # out_tm = out_tm.append_zero_var(n = D - out_tm.P.c.shape[1])

        return out_tm

    return eval_tm

# ----------------------
# Interval interpreter
# ----------------------
def build_rhs_interval_from_prog(prog: ClosedProg) -> Callable[[Interval], Interval]:
    """
    Evaluate RHS into an Interval using interval arithmetic in src.interval.

    Assumes:
      - input Interval has shape lo,hi: [B, D] (batch-parallel)
      - traced program takes a single vector input x:[D] and returns a vector (or scalar) output
    """
    jaxpr, consts, D = prog.jaxpr, prog.consts, prog.D

    def _as_interval(x, ref: Interval) -> Interval:
        if isinstance(x, Interval):
            return x
        arr = jnp.asarray(x)
        # Broadcast constants/arrays to the reference shape if needed.
        if arr.shape != ref.lo.shape:
            arr = jnp.broadcast_to(arr, ref.lo.shape)
        return Interval(arr, arr)

    def _lift_shape_op(p, I: Interval, params: dict) -> Interval:
        # Apply shape-only primitives (reshape/squeeze/convert_element_type/...) to lo/hi.
        lo = p.bind(I.lo, **params)
        hi = p.bind(I.hi, **params)
        return Interval(lo, hi)

    def eval_iv(x_iv: Interval) -> Interval:
        env: Dict[Any, Any] = {}

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        x_var = jaxpr.invars[0]
        env[x_var] = x_iv

        for eqn in jaxpr.eqns:
            p = eqn.primitive
            pname = getattr(p, "name", "")
            ins = [env[i] if not hasattr(i, "val") else i.val for i in eqn.invars]

            # ------------------------------------------------------------------
            # 1. UNIFIED SHAPE PRIMITIVES BLOCK
            # This completely replaces the old broadcast, slice, reshape, and squeeze blocks!
            # ------------------------------------------------------------------
            if p in (jlax.convert_element_type_p, jlax.broadcast_in_dim_p, jlax.slice_p, jlax.reshape_p, jlax.squeeze_p) or pname in ("stop_gradient", "tie_in"):
                a = ins[0]
                if isinstance(a, Interval):
                    # p.bind safely delegates all tensor transformations to vmap natively
                    out = Interval(p.bind(a.lo, **eqn.params), p.bind(a.hi, **eqn.params))
                else:
                    out = p.bind(*ins, **eqn.params)

            # ------------------------------------------------------------------
            # 2. STANDARD MATH PRIMITIVES
            # ------------------------------------------------------------------
            elif p is jlax.add_p:
                a, b = ins
                if isinstance(a, Interval):
                    out = a.add(b)
                elif isinstance(b, Interval):
                    out = b.add(a)
                else:
                    out = a + b

            elif p is jlax.sub_p:
                a, b = ins
                if isinstance(a, Interval):
                    out = a.sub(b)
                elif isinstance(b, Interval):
                    out = _as_interval(a, b).sub(b)
                else:
                    out = a - b

            elif p is jlax.neg_p:
                a = ins[0]
                if isinstance(a, Interval):
                    out = iv_neg(a)
                else:
                    out = -a

            elif p is jlax.mul_p:
                a, b = ins
                if isinstance(a, Interval):
                    out = a.mul(b)
                elif isinstance(b, Interval):
                    out = b.mul(a)
                else:
                    out = a * b

            elif p is jlax.div_p:
                a, b = ins
                if isinstance(a, Interval) and isinstance(b, Interval):
                    out = a.div(b)
                elif isinstance(a, Interval):
                    out = a.scale(1.0 / b)
                elif isinstance(b, Interval):
                    out = _as_interval(a, b).div(b)
                else:
                    out = a / b

            # ---- Non-polynomial unary ----
            elif p is jlax.sin_p:
                a = ins[0]
                out = iv_sin(a) if isinstance(a, Interval) else jnp.sin(a)
            elif p is jlax.cos_p:
                a = ins[0]
                out = iv_cos(a) if isinstance(a, Interval) else jnp.cos(a)
            elif p is jlax.tanh_p:
                a = ins[0]
                out = iv_tanh(a) if isinstance(a, Interval) else jnp.tanh(a)
            elif p is jlax.tan_p:
                a = ins[0]
                out = iv_tan(a) if isinstance(a, Interval) else jnp.tan(a)

            elif p is jlax.exp_p:
                a = ins[0]
                out = iv_exp(a) if isinstance(a, Interval) else jnp.exp(a)
            elif p is jlax.log_p:
                a = ins[0]
                out = iv_log(a) if isinstance(a, Interval) else jnp.log(a)
            elif p is jlax.sqrt_p:
                a = ins[0]
                out = iv_sqrt(a) if isinstance(a, Interval) else jnp.sqrt(a)

            # ---- Added Operations (Power & MatMul) ----
            elif p is jlax.integer_pow_p:
                a = ins[0]
                y = eqn.params['y']
                if isinstance(a, Interval):
                    if y == 0:
                        out = Interval(jnp.ones_like(a.lo), jnp.ones_like(a.hi))
                    elif y == 1:
                        out = a
                    elif y % 2 != 0:
                        out = Interval(jnp.power(a.lo, y), jnp.power(a.hi, y))
                    else:
                        lo_p = jnp.power(a.lo, y)
                        hi_p = jnp.power(a.hi, y)
                        max_p = jnp.maximum(lo_p, hi_p)
                        min_p = jnp.minimum(lo_p, hi_p)
                        contains_zero = jnp.logical_and(a.lo <= 0.0, a.hi >= 0.0)
                        min_p = jnp.where(contains_zero, 0.0, min_p)
                        out = Interval(min_p, max_p)
                else:
                    out = p.bind(*ins, **eqn.params)

            elif p is jlax.dot_general_p:
                lhs, rhs = ins
                if isinstance(lhs, Interval) and not isinstance(rhs, Interval):
                    pos_rhs = jnp.maximum(rhs, 0.0)
                    neg_rhs = jnp.minimum(rhs, 0.0)
                    lo = p.bind(lhs.lo, pos_rhs, **eqn.params) + p.bind(lhs.hi, neg_rhs, **eqn.params)
                    hi = p.bind(lhs.hi, pos_rhs, **eqn.params) + p.bind(lhs.lo, neg_rhs, **eqn.params)
                    out = Interval(lo, hi)
                elif not isinstance(lhs, Interval) and isinstance(rhs, Interval):
                    pos_lhs = jnp.maximum(lhs, 0.0)
                    neg_lhs = jnp.minimum(lhs, 0.0)
                    lo = p.bind(pos_lhs, rhs.lo, **eqn.params) + p.bind(neg_lhs, rhs.hi, **eqn.params)
                    hi = p.bind(pos_lhs, rhs.hi, **eqn.params) + p.bind(neg_lhs, rhs.lo, **eqn.params)
                    out = Interval(lo, hi)
                elif isinstance(lhs, Interval) and isinstance(rhs, Interval):
                    raise NotImplementedError("Interval @ Interval dot_general is not supported.")
                else:
                    out = p.bind(*ins, **eqn.params)

            elif p is jlax.concatenate_p:
                if any(isinstance(u, Interval) for u in ins):
                    ivs = [u if isinstance(u, Interval) else _as_interval(u, next(v for v in ins if isinstance(v, Interval))) for u in ins]
                    # Let JAX's primitive bind handle the concatenation natively
                    lo_out = p.bind(*[u.lo for u in ivs], **eqn.params)
                    hi_out = p.bind(*[u.hi for u in ivs], **eqn.params)
                    out = Interval(lo_out, hi_out)
                else:
                    out = p.bind(*ins, **eqn.params)

            else:
                if any(isinstance(u, Interval) for u in ins):
                    raise TypeError(
                        f"Unsupported primitive in Interval eval: {pname} with operands {[type(u) for u in ins]}"
                    )
                out = p.bind(*ins, **eqn.params)

            if eqn.primitive.multiple_results:
                for o, v in zip(out, eqn.outvars):
                    env[v] = o
            else:
                env[eqn.outvars[0]] = out

        out_iv = env[jaxpr.outvars[0]]
        # If output is a scalar, lift to [B,1] Interval.
        if not isinstance(out_iv, Interval):
            out_iv = _as_interval(out_iv, x_iv)

        return out_iv

    return eval_iv


# ======================
# Convenience wrapper
# ======================
def build_auto_rhs_analytic(f: Callable, D: int, V: int) -> Tuple[
    Callable[[LinTM], LinPoly],
    Callable[[LinTM, Array, Array], LinTM]
]:
    prog = _trace_prog(f, D)
    return build_rhs_tm_from_prog(prog, V)

def build_auto_rhs_analytic_int(f: Callable, D: int, V: int) -> Tuple[
    Callable[[LinTM], LinPoly],
    Callable[[LinTM, Array, Array], LinTM]
]:
    prog = _trace_prog(f, D)
    return build_rhs_interval_from_prog(prog)
