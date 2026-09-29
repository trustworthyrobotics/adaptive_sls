# gpu_sls.py
from __future__ import annotations
from typing import Union
from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
from jax import lax, vmap
import jax.scipy as jsp
from gpu_sls.gpu_admm import constrained_solve
from dataclasses import dataclass
from linearization_sls.src.helper import make_step_boxes, build_linear_tm, prepare_initial_set
from linearization_sls.src.taylor_model import LinTM

@dataclass(frozen=True)
class SLSConfig:
    max_sls_iterations: int = 2
    sls_primal_tol: float = 1e-2
    enable_fastsls: bool = True
    warm_start: bool = True
    n_x: int = None 
    n_w: int = None
    n_theta: int = None 
    num_param: int = None
    q_max: int = None
    adaptive: bool = True
    enable_linearization_error: bool = True
    enable_disturbance_variation_error: bool = True
    enable_posterior_gate: bool = True
    enable_information_cost: bool = False
    information_cost_weight: float = 1.0
    information_cost_discount: float = 1.0
    information_gain_regularization: float = 1e-7

@jax.jit
def controller_pas(Q, R, M, A, B):
    T = Q.shape[0] - 1
    n = Q.shape[1]
    I = jnp.eye(n, dtype=Q.dtype)

    def op(next_elem, prev_elem):
        def decompose(elem):
            A_blk = elem[:n, :]
            C_blk = elem[n:2*n, :]
            P_blk = elem[2*n:3*n, :]
            return A_blk, C_blk, P_blk

        A_l, C_l, P_l = decompose(prev_elem)
        A_r, C_r, P_r = decompose(next_elem)

        X1 = jnp.linalg.solve(I + C_l @ P_r, I)  
        X2 = jnp.linalg.solve(I + P_r @ C_l, I)  

        ArIClPr = A_r @ X1
        AlTIPrCl = A_l.T @ X2

        A_new = ArIClPr @ A_l
        C_new = ArIClPr @ C_l @ A_r.T + C_r
        P_new = AlTIPrCl @ P_r @ A_l + P_l

        return jnp.concatenate([A_new, C_new, P_new], axis=0)

    def chol_inv(mat):
        f = jsp.linalg.cho_factor(mat, lower=True)
        m = mat.shape[0]
        return jsp.linalg.cho_solve(f, jnp.eye(m, dtype=mat.dtype))

    Rinv = vmap(chol_inv)(R)                          
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))  
    MRinv = vmap(lambda t: M[t] @ Rinv[t])(jnp.arange(T))  

    A_bar = A - vmap(lambda t: BRinv[t] @ M[t].T)(jnp.arange(T))          
    C_bar = vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T))              
    P_bar = Q[:T] - vmap(lambda t: MRinv[t] @ M[t].T)(jnp.arange(T))      
    P_T   = Q[T]                                                          

    elems = jnp.concatenate(
        [
            jnp.concatenate([A_bar, jnp.zeros((1, n, n), dtype=Q.dtype)], axis=0),
            jnp.concatenate([C_bar, jnp.zeros((1, n, n), dtype=Q.dtype)], axis=0),
            jnp.concatenate([P_bar, P_T[None, :, :]], axis=0),
        ],
        axis=1,  
    )

    result = lax.associative_scan(lambda r, l: vmap(op)(r, l), elems, reverse=True)
    P = result[:, 2*n:3*n, :]  

    def gain_at(t):
        BtP = B[t].T @ P[t + 1]                 
        G = R[t] + BtP @ B[t]                   
        H = BtP @ A[t] + M[t].T                 
        return -jsp.linalg.solve(G, H, assume_a="pos")

    K = vmap(gain_at)(jnp.arange(T))            
    return K

@jax.jit
def calculate_phis(A, B, Cx, Cxu, Cu, E):
    T = Cu.shape[0]                 
    nx = A.shape[1]
    nu = B.shape[-1]
    Tp1 = T + 1
    nw = E.shape[-1]

    A = A[:T]
    B = B[:T]

    def solve_one_j(j):
        Qj = Cx[:, j, :, :]     
        Rj = Cu[:, j, :, :]     
        Mj = Cxu[:, j, :, :]    
        K = controller_pas(Qj, Rj, Mj, A, B)
        return K                

    K_all = jax.vmap(solve_one_j)(jnp.arange(T))      
    K_kj_core = jnp.swapaxes(K_all, 0, 1)             
    K_lastcol = jnp.zeros((T, 1, nu, nx), dtype=A.dtype)
    K_kj = jnp.concatenate([K_kj_core, K_lastcol], axis=1)  

    BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)        
    F  = A[:, None, :, :] + BK                        

    I = jnp.eye(nx, dtype=A.dtype)
    F = F.at[:, T].set(I)

    t_idx = jnp.arange(T)[:, None]       
    j_idx = jnp.arange(Tp1)[None, :]     
    use_F = (t_idx >= j_idx)             
    elems = jnp.where(use_F[:, :, None, None], F, I)  

    def compose(l, r):
        return jnp.einsum("...ab,...bc->...ac", r, l)

    P = lax.associative_scan(compose, elems, axis=0)  

    Phix_1toT = jnp.einsum("tjab,jbn->tjan", P, E)     
    Phi_x = jnp.concatenate(
        [jnp.zeros((1, Tp1, nx, nw), dtype=A.dtype), Phix_1toT],
        axis=0
    )  

    Phi_x = Phi_x.at[jnp.arange(Tp1), jnp.arange(Tp1)].set(E)

    k_idx_full = jnp.arange(Tp1)[:, None]             
    valid_x = (k_idx_full >= j_idx)                   
    Phi_x = Phi_x * valid_x[:, :, None, None]

    Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])    
    k_idx = jnp.arange(T)[:, None]                             
    valid_u = (k_idx >= j_idx)                                 
    Phi_u = Phi_u * valid_u[:, :, None, None]

    return Phi_x, Phi_u, K_kj

def calculate_cost(Q_bar, R_bar, C, D, eta):
    eta = jnp.asarray(eta).reshape(-1)
    eta = jnp.maximum(eta, 0.0)

    s = jnp.sqrt(eta)
    Cs = C * s[:, None]
    Ds = D * s[:, None]

    Cx  = Cs.T @ Cs + Q_bar
    Cxu = Cs.T @ Ds
    Cu  = Ds.T @ Ds + R_bar

    return Cx, Cxu, Cu

@jax.jit
def get_controller(Q, R, Q_f, A, B, C, D, E, eta_stage, eta_f):
    T, nx, _ = A.shape
    nc = C.shape[1]

    js = jnp.arange(T)
    ks = jnp.arange(T)

    def blocks_for_k(k):
        def blocks_for_j(j):
            return calculate_cost(Q[k], R[k], C[k], D[k], eta_stage[k, j])
        return vmap(blocks_for_j)(js)

    Cx_kj, Cxu_kj, Cu_kj = vmap(blocks_for_k)(ks)  

    Cterm = C[-1]  
    def terminal_Cx_for_j(j):
        w = eta_f[j]
        return (Cterm.T * w[None, :]) @ Cterm + Q_f

    Cx_Nj = vmap(terminal_Cx_for_j)(jnp.arange(T))  
    Cx = jnp.concatenate([Cx_kj, Cx_Nj[None, ...]], axis=0)      

    I = jnp.broadcast_to(jnp.eye(nx), (T + 1, nx, nx))
    Phi_x, Phi_u, K_kj = calculate_phis(A, B, Cx, Cxu_kj, Cu_kj, I)
    return Phi_x, Phi_u, K_kj

@jax.jit
def get_betas(C, D, Phi_x, Phi_u, E):
    T = Phi_u.shape[0]
    Tp1 = T + 1
    nc = C.shape[1]

    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E)
    Phi_u_E = jnp.einsum("kjun,jne->kjue", Phi_u, E)

    term_x = jnp.einsum("kix,kjxe->kjie", C[:-1], Phi_x_E[:-1])
    term_u = jnp.einsum("kiu,kjue->kjie", D[:-1], Phi_u_E)
    gPhi = term_x + term_u

    beta_stage = jnp.sum(jnp.abs(gPhi), axis=-1) ** 2

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    mask = (j_idx <= k_idx)
    beta_stage = beta_stage * mask[:, :, None]

    gPhi_term = jnp.einsum("ix,jxe->jie", C[-1], Phi_x_E[-1])
    beta_term = jnp.sum(jnp.abs(gPhi_term), axis=-1) ** 2

    beta = jnp.zeros((Tp1, Tp1, nc), dtype=Phi_x.dtype)
    beta = beta.at[:-1].set(beta_stage)
    beta = beta.at[-1].set(beta_term)
    return beta


@partial(jax.jit, static_argnames=['n_w', 'n_param_generators'])
def get_nonadaptive_betas_persistent_parameter(
    C,
    D,
    Phi_x,
    Phi_u,
    E,
    *,
    n_w,
    n_param_generators,
):
    """Compute non-adaptive backoffs for one time-invariant parameter.

    ``E`` is ordered as ``[process, parameter, residual]``.  The parameter
    block at every time multiplies the same generator coordinate, so its
    signed closed-loop responses are summed over time before taking the
    1-norm.  Process and residual disturbances remain independent at each
    time and retain the usual sum-of-norms treatment.

    The aggregated parameter contribution is stored in beta column zero so
    that ``get_constraint_tightenings`` returns its norm exactly once.
    """
    T = Phi_u.shape[0]
    Tp1 = T + 1

    response_stage = (
        jnp.einsum("kix,kjxn->kjin", C[:-1], Phi_x[:-1])
        + jnp.einsum("kiu,kjun->kjin", D[:-1], Phi_u)
    )
    response_term = jnp.einsum("ix,jxn->jin", C[-1], Phi_x[-1])

    param_end = n_w + n_param_generators
    E_param = E[:Tp1, :, n_w:param_end]
    E_independent = jnp.concatenate(
        [E[:Tp1, :, :n_w], E[:Tp1, :, param_end:]],
        axis=-1,
    )

    independent_stage = jnp.linalg.norm(
        jnp.einsum("kjin,jne->kjie", response_stage, E_independent),
        ord=1,
        axis=-1,
    )
    independent_term = jnp.linalg.norm(
        jnp.einsum("jin,jne->jie", response_term, E_independent),
        ord=1,
        axis=-1,
    )

    # Phi_x and Phi_u are causal, so this contraction sums only j <= k.
    shared_stage = jnp.linalg.norm(
        jnp.einsum("kjin,jng->kig", response_stage, E_param),
        ord=1,
        axis=-1,
    )
    shared_term = jnp.linalg.norm(
        jnp.einsum("jin,jng->ig", response_term, E_param),
        ord=1,
        axis=-1,
    )

    radius_stage = independent_stage.at[:, 0, :].add(shared_stage)
    radius_term = independent_term.at[0, :].add(shared_term)

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    radius_stage = radius_stage * (j_idx <= k_idx)[:, :, None]

    beta = jnp.zeros((Tp1, Tp1, C.shape[1]), dtype=Phi_x.dtype)
    beta = beta.at[:-1].set(radius_stage ** 2)
    beta = beta.at[-1].set(radius_term ** 2)
    return beta

@partial(jax.jit, static_argnames=['nx', 'n_theta'])
def get_adaptive_betas(C, D, Phi_x, Phi_u, F_orig, E_sens, K_gains, L_gains, W, G_0, nx, n_theta):
    """Compute gated observable-subspace tube contributions.

    ``K_gains`` contains the parameter-error maps ``L_t Pi_t`` and ``W``
    contains ``(C_meas E_t)^dagger C_meas F_t``.  Thus every ``I-K_gains`` below is
    exactly the PDF's ``I-L_t Pi_t``, while process noise enters through
    ``L_gains @ W``.
    """
    T_plus_1 = C.shape[0]
    nc, T = C.shape[1], T_plus_1 - 1
    
    D_padded = jnp.concatenate([D, jnp.zeros((1, nc, D.shape[2]), dtype=D.dtype)], axis=0) if D.shape[0] == T else D[:T_plus_1]
    Phi_u_padded = jnp.concatenate([Phi_u, jnp.zeros((1, T_plus_1, Phi_u.shape[2], Phi_u.shape[3]), dtype=Phi_u.dtype)], axis=0) if Phi_u.shape[0] == T else Phi_u[:T_plus_1]
    
    gPhi = jnp.einsum('kcx,kjxy->kjcy', C[:T_plus_1], Phi_x[:T_plus_1, :T_plus_1]) + \
           jnp.einsum('kcu,kjuy->kjcy', D_padded, Phi_u_padded[:, :T_plus_1])
           
    # SHIFT: Slice to j+1 to properly map the physics
    gPhi_tilde = gPhi[:, 1:, :, :nx] 
    gPhi_bar = gPhi[:, 1:, :, nx:]   
    
    F_orig = F_orig[:T]
    E_sens = E_sens[:T]
    K_gains = K_gains[:T]
    L_gains = L_gains[:T]
    W = W[:T]
    
    I_nt = jnp.eye(n_theta, dtype=C.dtype)
    def compute_psi_j(j):
        def scan_fn(carry, t):
            next_carry = jnp.where(t <= j, I_nt, (I_nt - K_gains[t-1]) @ carry)
            return next_carry, next_carry
        _, psi_j = jax.lax.scan(scan_fn, I_nt, jnp.arange(T))
        return psi_j
    Psi = jnp.swapaxes(jax.vmap(compute_psi_j)(jnp.arange(T)), 0, 1) 
    
    Psi_j_0 = Psi[:, 0]
    J_term = jnp.einsum('jxn,jnm->jxm', E_sens, Psi_j_0)
    K_term = jnp.einsum('jln,jnm->jlm', K_gains, Psi_j_0)
    
    k_idx, j_idx = jnp.arange(T_plus_1)[:, None], jnp.arange(T)[None, :]
    mask_k_gt_j = (k_idx > j_idx)
    
    M_k_xi = jnp.einsum('kj,kjcx,jxm->kcm', mask_k_gt_j, gPhi_tilde, J_term) + \
             jnp.einsum('kj,kjcl,jlm->kcm', mask_k_gt_j, gPhi_bar, K_term)
    tube_xi = jnp.linalg.norm(M_k_xi @ G_0, ord=1, axis=-1) 
    
    Z = jnp.einsum('ktcx,txn->ktcn', gPhi_tilde, E_sens) + \
        jnp.einsum('ktcl,tln->ktcn', gPhi_bar, K_gains)
    
    t_idx = jnp.arange(T)[:, None]
    Psi_padded = jnp.concatenate([Psi, jnp.zeros((T, 1, n_theta, n_theta), dtype=C.dtype)], axis=1)
    Psi_masked = Psi_padded[:, 1:] * (t_idx > j_idx)[:, :, None, None]
    
    Z_masked = Z * (k_idx > jnp.arange(T)[None, :])[:, :, None, None]
    S = jnp.einsum('ktcn,tjnm->kjcm', Z_masked, Psi_masked)
    
    # In the observable-subspace estimator, K = L P_obs E propagates the
    # parameter error, but the process-noise innovation channel is L W_obs.
    LW = jnp.einsum('jnm,jmw->jnw', L_gains, W)
    Delta_M = jnp.einsum('kjcn,jnw->kjcw', gPhi_bar - S, LW)
    M_base = jnp.einsum('kjcy,jyw->kjcw', gPhi_tilde, F_orig)
    
    M_k_j = (M_base + Delta_M) * mask_k_gt_j[:, :, None, None]
    tube_w = jnp.linalg.norm(M_k_j, ord=1, axis=-1) 
    
    beta = jnp.zeros((T_plus_1, T_plus_1, nc), dtype=C.dtype)
    beta = beta.at[:, 1:, :].set(tube_w ** 2)
    beta = beta.at[:, 0, :].set(tube_xi ** 2)
    return beta

@jax.jit
def get_constraint_tightenings(betas, eps_beta=1e-6):
    T1, _, _ = betas.shape
    s = jnp.sqrt(jnp.maximum(betas, 0.0))  

    k_idx = jnp.arange(T1)[:, None]        
    j_idx = jnp.arange(T1)[None, :]        
    valid = (j_idx <= k_idx)                
    s = s * valid[:, :, None]

    h_ct = jnp.sum(s, axis=1)              
    h_ct = h_ct + eps_beta
    return h_ct

@jax.jit
def get_etas(mus, betas, eps=1e-12):
    Tp1 = mus.shape[0]
    T = Tp1 - 1

    mu_k = mus[:-1]                    
    beta_kj = betas[:-1, :-1, :]       
    eta = (mu_k[:, None, :] /
           (2.0 * jnp.sqrt(jnp.maximum(beta_kj, eps))))

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(T)[None, :]
    eta = eta * (k_idx >= j_idx)[:, :, None]
    eta = jnp.maximum(eta, 0.0)

    mu_f = mus[-1]                     
    beta_f = betas[-1, :, :]
    eta_f = (mu_f[None, :] /
             (2.0 * jnp.sqrt(jnp.maximum(beta_f, eps))))
    eta_f = jnp.maximum(eta_f, 0.0)    

    return eta, eta_f

@jax.jit
def _scaled_primal_diff(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(1.0, jnp.max(jnp.abs(b)))
    return num / (den + eps)

@jax.jit
def primal_convergence_metric(
    X_new: jnp.ndarray, U_new: jnp.ndarray,
    X_old: jnp.ndarray, U_old: jnp.ndarray
) -> jnp.ndarray:
    mX = _scaled_primal_diff(X_new, X_old)
    mU = _scaled_primal_diff(U_new, U_old)
    return jnp.maximum(mX, mU)

@jax.jit
def add_obstacle_tightenings(
    obstacles: jnp.ndarray,        
    primal_pos: jnp.ndarray,        
    h_ct: jnp.ndarray,              
    tightened_constraints: jnp.ndarray,
    idx_px: int = 0,
    idx_py: int = 1,
    eps: float = 1e-6,
):
    pos = primal_pos[:, :2]              
    centers = obstacles[:, :2]           
    radii = obstacles[:, 2]              

    diff = pos[:, None, :] - centers[None, :, :]     
    dist = jnp.linalg.norm(diff, axis=-1) + eps      
    n = diff / dist[..., None]                        

    hx = jnp.abs(h_ct[:, idx_px])                     
    hy = jnp.abs(h_ct[:, idx_py])                     

    over = jnp.abs(n[..., 0]) * hx[:, None] + jnp.abs(n[..., 1]) * hy[:, None]  

    tightened = dist - radii[None, :] - over          
    return jnp.concatenate([tightened_constraints, tightened], axis=1)

def estimate_remainder_TM(tm_fn, x_lo, x_up, state_dim, splits_cfg={}):
    B, D = x_lo.shape
    step_lo, step_hi, _, _ = make_step_boxes(B, D, h=1.0)   

    c = 0.5 * (x_lo + x_up)          
    S = 0.5 * (x_up - x_lo)          
    x_tm0 = build_linear_tm(c, S)

    x_tmnext_nom = tm_fn(x_tm0, step_lo, step_hi)

    c_tm = x_tmnext_nom.P.c
    J_tm = x_tmnext_nom.P.L[:, :, 1:] / S[:, None, :]   

    A_tm = J_tm[:, :state_dim, :state_dim]
    B_tm = J_tm[:, :state_dim, state_dim:]

    step_lo_split, step_hi_split = prepare_initial_set(step_lo[:, 1:], step_hi[:, 1:], splits_cfg)
    step_lo_split = jnp.concatenate([jnp.zeros((step_lo_split.shape[0], 1), dtype=step_lo_split.dtype),
                                     step_lo_split], axis=-1)
    step_hi_split = jnp.concatenate([jnp.zeros((step_hi_split.shape[0], 1), dtype=step_hi_split.dtype),
                                     step_hi_split], axis=-1)

    BM = step_lo_split.shape[0]
    assert BM % B == 0
    M = BM // B

    x_tm0_split = LinTM.repeat(x_tm0, M)
    x_tmnext_split: LinTM = tm_fn(x_tm0_split, step_lo_split, step_hi_split)

    remainder = x_tmnext_split.R
    out_dim = remainder.lo.shape[-1]

    agg_r_lb = jnp.min(remainder.lo.reshape((B, M, out_dim)), axis=1)
    agg_r_ub = jnp.max(remainder.hi.reshape((B, M, out_dim)), axis=1)
    r_bound = jnp.maximum(jnp.abs(agg_r_lb), jnp.abs(agg_r_ub))   

    output_interval = x_tmnext_split.eval_interval(step_lo_split, step_hi_split)
    out_dim2 = output_interval.lo.shape[-1]

    agg_output_lb = jnp.min(output_interval.lo.reshape((B, M, out_dim2)), axis=1)
    agg_output_ub = jnp.max(output_interval.hi.reshape((B, M, out_dim2)), axis=1)

    return_dict = {
        "c_tm": c_tm,
        "A_tm": A_tm,
        "B_tm": B_tm,
        "output_lb": agg_output_lb[:, :state_dim],
        "output_ub": agg_output_ub[:, :state_dim],
    }

    return r_bound[:, :state_dim], return_dict

def get_tube_width(Phi_x, Phi_u, E):
    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E)   
    Phi_u_E = jnp.einsum("kjun,jne->kjue", Phi_u, E)   

    x_width = jnp.linalg.norm(Phi_x_E, ord=1, axis=-1).sum(axis=1)
    u_width = jnp.linalg.norm(Phi_u_E, ord=1, axis=-1).sum(axis=1)

    return x_width, u_width

# def get_combined_disturbance(
#     E, alpha_1, alpha_2,
#     X, U, Phi_x, Phi_u,
#     remainder_func, splits_cfg, E_prev
# ):
#     T = U.shape[0]
#     n_physical = E.shape[1]  # Extract physical state dimension (e.g., 4)
#     nu = U.shape[1]

#     x_tube_widths, u_tube_widths = get_tube_width(Phi_x, Phi_u, E_prev)

#     U_pad = jnp.concatenate([U, U[-1:]], axis=0)
#     u_width_pad = jnp.concatenate(
#         [u_tube_widths, jnp.zeros((1, nu), dtype=u_tube_widths.dtype)],
#         axis=0
#     )

#     t = jnp.arange(T + 1, dtype=X.dtype)[:, None]
#     t_width = jnp.zeros_like(t)

#     z_center = jnp.concatenate([X, U_pad, t], axis=-1)
#     z_width  = jnp.concatenate([x_tube_widths, u_width_pad, t_width], axis=-1)

#     z_lo = z_center - z_width
#     z_up = z_center + z_width

#     # r_bound has shape (T+1, 6)
#     r_bound = jax.vmap(remainder_func, in_axes=(0, 0))(z_lo, z_up)   
    
#     # Slice to keep only the physical dimensions (T+1, 4), then diagonalize
#     diag_r = jax.vmap(jnp.diag)(r_bound[:, :n_physical])                              

#     E_combined = jnp.concatenate([E, diag_r], axis=-1)                
#     return E_combined

# def get_combined_disturbance(
#     E, alpha_1, alpha_2,
#     X, U, x_tube_widths, u_tube_widths,
#     remainder_func, splits_cfg
# ):
#     T = U.shape[0]
#     n_physical = E.shape[1]  
#     nu = U.shape[1]

#     U_pad = jnp.concatenate([U, U[-1:]], axis=0)
    
#     t = jnp.arange(T + 1, dtype=X.dtype)[:, None]
#     t_width = jnp.zeros_like(t)

#     # Use the exact adaptive widths to construct the bounding box
#     z_center = jnp.concatenate([X, U_pad, t], axis=-1)
#     z_width  = jnp.concatenate([x_tube_widths, u_tube_widths, t_width], axis=-1)

#     z_lo = z_center - z_width
#     z_up = z_center + z_width

#     # The remainder_func will now evaluate the Taylor expansion over the massive adaptive box
#     r_bound = jax.vmap(remainder_func, in_axes=(0, 0))(z_lo, z_up)   
    
#     diag_r = jax.vmap(jnp.diag)(r_bound[:, :n_physical])                              

#     E_combined = jnp.concatenate([E, diag_r], axis=-1)                
#     return E_combined

# 1. Replace get_combined_disturbance in gpu_sls.py
# def get_combined_disturbance(
#     E, alpha_1, alpha_2,
#     X, U, x_tube_widths, u_tube_widths,
#     remainder_func, splits_cfg, G_0, nom_param
# ):
#     # T = U.shape[0]
#     n_physical = E.shape[1]  
#     # nu = U.shape[1]

#     # U_pad = jnp.concatenate([U, U[-1:]], axis=0)
    
#     # t = jnp.arange(T + 1, dtype=X.dtype)[:, None]
#     # t_width = jnp.zeros_like(t)

#     # z_center = jnp.concatenate([X, U_pad, t], axis=-1)
#     # z_width  = jnp.concatenate([x_tube_widths, u_tube_widths, t_width], axis=-1)

#     # z_lo = z_center - z_width
#     # z_up = z_center + z_width

#     # r_bound = jax.vmap(remainder_func, in_axes=(0, 0))(z_lo, z_up) * 0.0
#     u_tube_widths_pad = jnp.concatenate([u_tube_widths, u_tube_widths[-1:]], axis=0)
#     r_bound = remainder_func(X, U, x_tube_widths, u_tube_widths_pad, G_0, nom_param)
#     diag_r = jax.vmap(jnp.diag)(r_bound[:, :n_physical])                              

#     E_combined = jnp.concatenate([E, diag_r], axis=-1)                
#     return E_combined

def get_combined_disturbance(
    E, alpha_1, alpha_2,
    X, U, x_tube_widths, u_tube_widths,
    remainder_func, splits_cfg, G_0, nom_param
):
    n_physical = E.shape[1]  
    
    # u_tube_widths is extracted from h_ct, which ALREADY has shape (T+1, n_u).
    # We DO NOT need to pad it again!
    r_bound = remainder_func(X, U, x_tube_widths, u_tube_widths, G_0, nom_param)
    
    # Extract only the physical dimensions and diagonalize
    diag_r = jax.vmap(jnp.diag)(r_bound[:, :n_physical])                              

    E_combined = jnp.concatenate([E, diag_r], axis=-1)                
    return E_combined

def girard_reduction(G, q_max):
    n_theta, n_g = G.shape
    target_size = int(n_theta * q_max)
    p_keep = int(n_theta * (q_max - 1))

    gen_norms = jnp.linalg.norm(G, axis=0)
    idx = jnp.argsort(gen_norms)[::-1]
    G_sorted = G[:, idx]

    G_keep = G_sorted[:, :p_keep]
    G_tail = G_sorted[:, p_keep:]
    h = jnp.sum(jnp.abs(G_tail), axis=1)
    D = jnp.diag(h)
    G_reduced = jnp.concatenate([G_keep, D], axis=1)

    current_n_g = G_reduced.shape[1]
    padding_needed = max(0, target_size - current_n_g)
    G_padded = jnp.pad(G_reduced, ((0, 0), (0, padding_needed)))
    return G_padded[:, :target_size]

@partial(jax.jit, static_argnames=['q_max'])
def girard_reduction_opt(G, q_max):
    n_theta = G.shape[0]
    n_g = G.shape[1]
    target_size = int(n_theta * q_max)
    p_keep = int(n_theta * (q_max - 1))

    pad_len = max(0, target_size - n_g)
    G_padded = jnp.pad(G, ((0, 0), (0, pad_len)))
    n_g_padded = G_padded.shape[1]

    gen_norms = jnp.linalg.norm(G_padded, axis=0)
    idx = jnp.argsort(gen_norms)[::-1]
    G_sorted = G_padded[:, idx]

    indices = jnp.arange(n_g_padded)
    tail_mask = indices >= p_keep
    
    G_tail_abs = jnp.where(tail_mask[None, :], jnp.abs(G_sorted), 0.0)
    h = jnp.sum(G_tail_abs, axis=1)
    D = jnp.diag(h)

    return jnp.concatenate([G_sorted[:, :p_keep], D], axis=1)

@jax.jit
def observable_subspace_factorizer(E_t, F_t, measurement_matrix):
    """Factor one observable parameter measurement after applying gate C.

    The returned residual map acts on the original full-state residual:
    ``P_eff = (C E_t)^dagger C``.  Consequently ``Pi = P_eff E_t`` and
    ``W_obs = P_eff F_t`` are exactly the projector and noise map from the
    subset-state observable-subspace derivation.
    """
    H_t = measurement_matrix @ E_t
    U, s, Vt = jnp.linalg.svd(H_t, full_matrices=False)
    eps = jnp.maximum(
        jnp.asarray(1e-10, dtype=E_t.dtype),
        jnp.asarray(1e-6, dtype=E_t.dtype) * jnp.max(s),
    )
    s_inv = jnp.where(s >= eps, 1.0 / s, 0.0)
    H_pinv = (Vt.T * s_inv[None, :]) @ U.T
    P_eff = H_pinv @ measurement_matrix
    Pi = P_eff @ E_t
    W_obs = P_eff @ F_t
    return P_eff, Pi, W_obs


def adaptive_posterior_factors(
    G_t,
    Pi_t,
    W_t,
    gain_regularization=1e-7,
    enable_posterior_gate=True,
):
    """Return the one-step observable-subspace estimator factors.

    ``G_pre`` is the unreduced posterior generator

        [(I - L Pi) G, L W],

    while ``delta_G`` is the correlated generator block used by the adaptive
    SLS tube construction.  Keeping this calculation in one place ensures
    that the optional information cost predicts the same gated update used by
    the uncertainty propagation.
    """
    gg = G_t @ G_t.T
    reg = gain_regularization * jnp.eye(G_t.shape[0], dtype=gg.dtype)
    innovation_gram = Pi_t @ gg @ Pi_t.T + W_t @ W_t.T + reg
    L = gg @ Pi_t.T @ jnp.linalg.inv(innovation_gram)
    K = L @ Pi_t

    # Reject parameter rows that do not contract even before accounting for
    # the newly injected observable disturbance.
    scaled_G = K @ G_t
    contraction_gate = (
        jnp.abs(G_t - scaled_G).sum(axis=1)
        < jnp.abs(G_t).sum(axis=1)
    )
    L = jnp.diag(contraction_gate.astype(G_t.dtype)) @ L
    K = L @ Pi_t

    scaled_innovation = L @ W_t
    scaled_G = K @ G_t
    G_pre = jnp.hstack((G_t - scaled_G, scaled_innovation))

    if enable_posterior_gate:
        posterior_gate = (
            jnp.abs(G_pre).sum(axis=1)
            < jnp.abs(G_t).sum(axis=1)
        )
        L = jnp.diag(posterior_gate.astype(G_t.dtype)) @ L
        K = L @ Pi_t

    scaled_innovation = L @ W_t
    scaled_G = K @ G_t
    G_pre = jnp.hstack((G_t - scaled_G, scaled_innovation))
    delta_G = jnp.hstack((scaled_G, scaled_innovation))
    return K, L, G_pre, delta_G


def posterior_trace_information_gain(
    E_t,
    F_t,
    G_t,
    measurement_matrix,
    *,
    gain_regularization=1e-7,
    enable_posterior_gate=True,
):
    """One-step absolute reduction in the parameter uncertainty trace.

    The sensitivity and exogenous disturbance are first mapped into the
    SVD-thresholded observable parameter subspace. The returned scalar is

        trace(G G.T - G_plus G_plus.T).

    It is zero for a completely unobservable transition and positive when the
    gated one-step update contracts the lagged parameter uncertainty. Its
    magnitude naturally vanishes as the frozen parameter tube shrinks.
    """
    H_t = measurement_matrix @ E_t
    # Use JAX's custom pseudoinverse derivative while keeping the discrete
    # rank decision out of the SQP derivatives.
    singular_values = jax.lax.stop_gradient(
        jnp.linalg.svd(H_t, compute_uv=False)
    )
    sigma_max = jnp.max(singular_values)
    H_pinv_relative = jnp.linalg.pinv(H_t, rtol=1e-6)
    H_pinv = jnp.where(
        sigma_max >= jnp.asarray(1e-10, dtype=E_t.dtype),
        H_pinv_relative,
        jnp.zeros_like(H_pinv_relative),
    )
    P_eff = H_pinv @ measurement_matrix
    Pi_t = P_eff @ E_t
    W_t = P_eff @ F_t
    _, _, G_plus, _ = adaptive_posterior_factors(
        G_t,
        Pi_t,
        W_t,
        gain_regularization=gain_regularization,
        enable_posterior_gate=enable_posterior_gate,
    )

    prior_gram = G_t @ G_t.T
    posterior_gram = G_plus @ G_plus.T
    return jnp.trace(prior_gram - posterior_gram)


def transition_measurement_matrices(measurement_matrix, horizon):
    """Return one channel matrix for each of ``horizon`` transitions.

    A single ``(n_x, n_x)`` matrix is time invariant. A sequence with
    ``horizon`` entries already contains transition gates. A sequence with
    ``horizon + 1`` entries contains state-endpoint channel masks, so a
    channel is retained on transition ``t`` only when it is present at both
    endpoints. The latter is the elementwise intersection
    ``C[t] * C[t + 1]`` used by diagonal channel masks.
    """
    measurement_matrix = jnp.asarray(measurement_matrix)
    if measurement_matrix.ndim == 2:
        return jnp.broadcast_to(
            measurement_matrix,
            (horizon,) + measurement_matrix.shape,
        )
    if measurement_matrix.ndim != 3:
        raise ValueError(
            "measurement_matrix must have shape (n_x, n_x), "
            "(T, n_x, n_x), or (T + 1, n_x, n_x)"
        )
    if measurement_matrix.shape[0] == horizon:
        return measurement_matrix
    if measurement_matrix.shape[0] == horizon + 1:
        return measurement_matrix[:-1] * measurement_matrix[1:]
    raise ValueError(
        "time-varying measurement_matrix must have T or T + 1 entries; "
        f"got {measurement_matrix.shape[0]} for T={horizon}"
    )


@partial(
    jit,
    static_argnames=[
        'n_x',
        'n_w',
        'n_theta',
        'q_max',
        'ada',
        'enable_posterior_gate',
    ],
)
def get_adaptive_disturbance_opt(
    A,
    F,
    G_0,
    n_x,
    n_w,
    n_theta,
    q_max,
    ada,
    sens,
    measurement_matrix,
    enable_posterior_gate=True,
):
    T = A.shape[0]
    target_gen_size = int(n_theta * q_max)
    
    # exog = F[1:] if ada else F[1:,...,:-n_theta*q_max]

    # E = A[:, :n_x, n_x:] if ada else sens[:-1] 
    # E_inv = jnp.linalg.pinv(E) 
    # W_list = jnp.einsum("tij,tjk->tik", E_inv, exog) 
    exog = F[:-1] if ada else F[:-1,...,:-n_theta*q_max]

    E = A[:, :n_x, n_x:] if ada else sens[:-1]
    measurement_matrices = transition_measurement_matrices(
        measurement_matrix,
        T,
    )
    P_obs, Pi_list, W_list = jax.vmap(
        observable_subspace_factorizer
    )(E, exog, measurement_matrices)

    G_0_reduced = girard_reduction_opt(G_0, q_max)

    def scan_body(G_t, args):
        W_t, Pi_t = args
        K, L, G_pre, delta_G = adaptive_posterior_factors(
            G_t,
            Pi_t,
            W_t,
            gain_regularization=1e-7,
            enable_posterior_gate=enable_posterior_gate,
        )

        G_t_plus_1 = girard_reduction_opt(G_pre, q_max)
        delta_G_reduced = girard_reduction_opt(delta_G, q_max)
        
        return G_t_plus_1, (K, L, delta_G_reduced, G_t)

    _, (K_hat, L_hat, Delta_G_list, G_list) = lax.scan(scan_body, G_0_reduced, (W_list, Pi_list))

    total_aug_cols = n_w + 2 * target_gen_size
    
    if ada:
        F_hat_0 = jnp.hstack((F[0], jnp.zeros((n_x, 2 * target_gen_size))))
        F_hat_0 = jnp.vstack((F_hat_0, jnp.zeros((n_theta, total_aug_cols))))

        def construct_F_hat_t(F_t, E_t_m1, G_t_m1, dG_t_m1):
            upper = jnp.hstack((F_t, E_t_m1 @ G_t_m1, jnp.zeros((n_x, target_gen_size))))
            lower = jnp.hstack((jnp.zeros((n_theta, n_w)), jnp.zeros((n_theta, target_gen_size)), dG_t_m1))
            return jnp.vstack((upper, lower))

        F_hat_rest = vmap(construct_F_hat_t)(F[1:T+1], E, G_list, Delta_G_list)
        F_hat = jnp.concatenate([jnp.expand_dims(F_hat_0, axis=0), F_hat_rest], axis=0)
    else:
        F_hat = F

    return F_hat, K_hat, L_hat, G_list, P_obs

@partial(jit, static_argnums=(0, 1, 16, 17))
def sls_solve_gpu(cfg, remainder_func, Q: jnp.ndarray, q: jnp.ndarray,
                       R: jnp.ndarray, r: jnp.ndarray,
                       M: jnp.ndarray,
                       A: jnp.ndarray, B: jnp.ndarray, c: jnp.ndarray,
                       C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                       w: jnp.ndarray, y: jnp.ndarray, rho: jnp.ndarray, 
                       sls_config: SLSConfig, splits_cfg, E: jnp.ndarray, E_prev: jnp.ndarray,
                       Q_bar: jnp.ndarray, R_bar: jnp.ndarray, Q_f_bar: jnp.ndarray,
                       obstacles: jnp.ndarray, primal_pos: jnp.ndarray, h_ct_ws: jnp.ndarray,
                       beta_ws: jnp.ndarray, mu_ws: jnp.ndarray, Phi_x_ws: jnp.ndarray, Phi_u_ws: jnp.ndarray, X: jnp.ndarray, U: jnp.ndarray, K_prev: jnp.ndarray,G_prev: jnp.ndarray,
                       G_0: jnp.ndarray, P_inv_prev: jnp.ndarray, sens: Union[jnp.ndarray, None], nom_param: Union[jnp.ndarray, None],
                       measurement_matrix: jnp.ndarray):
    
    Tp1 = Q.shape[0]
    nx  = Q.shape[1]
    nu  = R.shape[1]
    nc  = w.shape[1]
    num_obstacles = obstacles.shape[0]
    # Only the original stage constraints receive robust tube tightenings.
    # Optional terminal-only nominal constraints are placed after these rows
    # and before obstacle rows; their count is therefore excluded here.
    nominal_nc = h_ct_ws.shape[1]
    non_obstacle_nc = nc - num_obstacles
    T   = Tp1 - 1

    x0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)
    u0 = jnp.zeros((T, nu),  dtype=Q.dtype)
    v0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)
    K_0 = jnp.zeros((T, Tp1, nu, nx), dtype=Q.dtype)
    L_0 = jnp.zeros_like(K_prev)

    i0 = jnp.array(0, dtype=rho.dtype)
    converged0 = jnp.array(False)
    admm_iterations0 = jnp.array(0, dtype=jnp.int32)
    admm_worst_index0 = jnp.array(0, dtype=jnp.int32)
    residual0 = jnp.asarray(jnp.inf, dtype=Q.dtype)

    max_iter = jnp.array(sls_config.max_sls_iterations, dtype=jnp.int32)
    tol = jnp.array(sls_config.sls_primal_tol, dtype=Q.dtype)

    h_ct0 = h_ct_ws
    carry0 = (
        i0, beta_ws, x0, u0, v0, w, y, rho, converged0, converged0,
        h_ct0, Phi_x_ws, Phi_u_ws, K_0, mu_ws, E_prev, K_prev, L_0,
        G_prev, P_inv_prev, residual0, admm_iterations0, residual0,
        residual0, residual0, residual0, admm_worst_index0, residual0,
        residual0,
    )

    def cond_fn(carry):
        i = carry[0]
        converged = carry[8]
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))
    
    def body_fn(carry):
        (
            i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _,
            h_ct, Phi_x_prev, Phi_u_prev, _, mu, E_prev, _, _, _, _,
            _, _, _, _, _, _, _, _, _,
        ) = carry
        
        # 2. Inside sls_solve_gpu -> body_fn in gpu_sls.py
        # ...
        # ...
        alpha_1 = 0.75
        alpha_2 = 0.25

        n_x = sls_config.n_x
        n_theta = sls_config.n_theta if sls_config.adaptive else 0
        n_full = n_x + n_theta
        
        # Extract physical widths from constraint tightenings
        physical_widths = h_ct[:, :n_x]
        
        if sls_config.adaptive:
            # FIX: Explicitly inject G_0 bounds for the parameter tube widths!
            # The maximum deviation for each parameter is the row-sum of G_0
            theta_widths = jnp.sum(jnp.abs(G_0), axis=-1)
            theta_widths_tiled = jnp.tile(theta_widths, (h_ct.shape[0], 1))
            x_tube_widths = jnp.concatenate([physical_widths, theta_widths_tiled], axis=-1)
        else:
            theta_widths = jnp.sum(jnp.abs(G_0), axis=-1)
            theta_widths_tiled = jnp.tile(theta_widths, (h_ct.shape[0], 1))
            x_tube_widths = jnp.concatenate([physical_widths, theta_widths_tiled], axis=-1)

        n_u = u_curr.shape[1]
        # Base constraints are [state upper/lower, control upper/lower].
        # Derive the control offset from the actual constraint layout so
        # adaptive parameter states do not need artificial box rows.
        state_box_rows = nominal_nc - 2 * n_u
        u_tube_widths = h_ct[:, state_box_rows : state_box_rows + n_u]
        # physical_widths = jnp.maximum(physical_widths, jnp.max(physical_widths, axis=0, keepdims=True))
        # u_tube_widths = jnp.maximum(u_tube_widths, jnp.max(u_tube_widths, axis=0, keepdims=True))
        if not sls_config.adaptive:
            # tile and concatenate the nominal param to X:
            nom_param_tiled = jnp.tile(nom_param, (h_ct.shape[0], 1))
            X_temp = jnp.concatenate([X, nom_param_tiled], axis=-1)
        else:
            X_temp = X

        E_aug = get_combined_disturbance(E, alpha_1, alpha_2, X_temp, U, x_tube_widths, u_tube_widths, remainder_func, splits_cfg, G_0, nom_param)
        # ...
        n_theta = sls_config.n_theta if sls_config.adaptive else sls_config.num_param
        E_aug_full, K_hat, L_hat, G_hat, P_inv = get_adaptive_disturbance_opt(
            A=A,
            F=E_aug,
            n_x=sls_config.n_x,
            n_w=E_aug.shape[-1],
            n_theta=n_theta,
            q_max=sls_config.q_max,
            G_0=G_0,
            ada=sls_config.adaptive,
            sens=sens,
            measurement_matrix=measurement_matrix,
            enable_posterior_gate=sls_config.enable_posterior_gate,
        )
        # ...
        # ... (rest of the function remains the same)
        # prev_rho = rho
        x_prev = x_curr
        u_prev = u_curr
        mu_nominal = mu[:, :nominal_nc]
        eta_stage, eta_f = get_etas(mu_nominal, beta)
        C_box = C[:, :nominal_nc, :]
        D_box = D[:, :nominal_nc, :]

        Phi_x, Phi_u, K_kj = get_controller(
            Q_bar,
            R_bar,
            Q_f_bar,
            A,
            B,
            C_box,
            D_box,
            E_aug_full,
            eta_stage,
            eta_f,
        )
        if not sls_config.adaptive:
            # Keep the separable beta surrogate for the Riccati/PAS update,
            # but certify the tube using the shared, time-invariant parameter
            # generator only once.
            beta = get_betas(C_box, D_box, Phi_x, Phi_u, E_aug_full)
            beta_backoff = get_nonadaptive_betas_persistent_parameter(
                C_box,
                D_box,
                Phi_x,
                Phi_u,
                E_aug_full,
                n_w=sls_config.n_w,
                n_param_generators=G_0.shape[-1],
            )
        else:
            nx = sls_config.n_x
            F_orig = E_aug 
            E_sens = A[:, :nx, nx:] if sls_config.adaptive else sens[:-1]
            
            E_sens_pad = jnp.concatenate([E_sens, jnp.zeros((1, nx, n_theta), dtype=E.dtype)], axis=0)
            P_inv_pad = jnp.concatenate([P_inv, jnp.zeros((1, n_theta, nx), dtype=P_inv.dtype)], axis=0)
            
            # P_inv is the effective full-residual map (C E)^dagger C,
            # so this is the gated W = (C E)^dagger C F used by both the
            # estimator and the observable-subspace tube recursion.
            W_exact = jnp.einsum("tij,tjk->tik", P_inv_pad, F_orig)
            
            K_gains = K_hat
            L_gains = L_hat
            beta = get_adaptive_betas(C_box, D_box, Phi_x, Phi_u, F_orig, E_sens_pad, K_gains, L_gains, W_exact, G_0, nx, n_theta)
            beta_backoff = beta

        h_ct = get_constraint_tightenings(beta_backoff)
        tightened_constraints = f[:, :nominal_nc] - h_ct
        terminal_constraints = f[:, nominal_nc:non_obstacle_nc]
        tightened_and_terminal = jnp.concatenate(
            [tightened_constraints, terminal_constraints],
            axis=1,
        )
        tightened_constraints_all = add_obstacle_tightenings(
            obstacles,
            primal_pos,
            h_ct,
            tightened_and_terminal,
        )
        warm_flag = jnp.array(bool(sls_config.warm_start))

        w   = lax.select(warm_flag, w, jnp.zeros_like(w))
        y   = lax.select(warm_flag, y, jnp.zeros_like(y))
        prev_rho = rho

        rho = lax.select(
            warm_flag,
            rho,
            jnp.asarray(cfg.initial_rho, dtype=rho.dtype),
        )
        (
            x_curr, u_curr, v_curr, w, y, rho, mu, converged_admm,
            admm_iterations, admm_primal_residual, admm_dual_residual,
            admm_primal_tolerance, admm_dual_tolerance,
            admm_primal_worst_flat_index, admm_primal_worst_z,
            admm_primal_worst_w,
        ) = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, tightened_constraints_all, w, y, rho
        )

        metric = primal_convergence_metric(x_curr, u_curr, x_prev, u_prev)
        # rho = jnp.maximum(jnp.minimum(rho, 1e4) * 0.9, 0.1)
        rho = jnp.asarray(rho, dtype=prev_rho.dtype)
        w   = jnp.asarray(w,   dtype=w.dtype)
        y   = jnp.asarray(y,   dtype=y.dtype)
        converged_now = metric <= tol
        converged = jnp.logical_or(converged, converged_now)

        return (i + jnp.array(1, dtype=jnp.int32),
                beta, x_curr, u_curr, v_curr, w, y, rho, converged, converged_admm, h_ct, Phi_x, Phi_u, K_kj, mu, E_aug_full, K_hat, L_hat, G_hat, P_inv,
                metric, admm_iterations, admm_primal_residual,
                admm_dual_residual, admm_primal_tolerance,
                admm_dual_tolerance, admm_primal_worst_flat_index,
                admm_primal_worst_z, admm_primal_worst_w)

    carryN = jax.lax.while_loop(cond_fn, body_fn, carry0)
    (
        _, betaN, xN, uN, vN, wN, yN, rhoN, convergedN,
        converged_admm, h_ct, Phi_x, Phi_u, K_kj, muN, EN, KN, LN,
        GN, P_inv, sls_residual, admm_iterations, admm_primal_residual,
        admm_dual_residual, admm_primal_tolerance, admm_dual_tolerance,
        admm_primal_worst_flat_index, admm_primal_worst_z,
        admm_primal_worst_w,
    ) = carryN
    return (
        xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct,
        Phi_x, Phi_u, K_kj, betaN, muN, EN, KN, LN, GN, P_inv,
        sls_residual, admm_iterations, admm_primal_residual,
        admm_dual_residual, admm_primal_tolerance, admm_dual_tolerance,
        admm_primal_worst_flat_index, admm_primal_worst_z,
        admm_primal_worst_w,
    )
