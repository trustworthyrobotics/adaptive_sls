# sls_visual.py
import jax.numpy as jnp
import jax
import numpy as np

def get_trajectory_tubes(Phi_x, E_prev):
    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E_prev)   
    return jnp.linalg.norm(Phi_x_E, ord=1, axis=-1).sum(axis=1)

@jax.jit
def get_adaptive_trajectory_tubes_exact(
    Phi_aug, F, E_sens, K_gains, W, G_0, L_gains=None
):
    """Visualize the observable-subspace tube, including a subset-state gate.

    ``K_gains`` must be the parameter-error map ``L_t Pi_t`` and ``W`` the
    gated noise map ``(C_meas E_t)^dagger C_meas F_t``.  Pass ``L_gains`` whenever the
    projector is not identity; omitting it preserves the legacy full-rank
    behavior in which ``K_gains == L_gains``.
    """
    T_plus_1 = min(Phi_aug.shape[0], F.shape[0], E_sens.shape[0], K_gains.shape[0], W.shape[0])
    T = T_plus_1 - 1
    n_theta, nx = G_0.shape[0], Phi_aug.shape[2] - G_0.shape[0]
    
    # SHIFT: Slice Phi to map disturbance from j to state at j+1
    Phi_tilde = Phi_aug[:T_plus_1, 1:T_plus_1, :nx, :nx]  
    Phi_bar   = Phi_aug[:T_plus_1, 1:T_plus_1, :nx, nx:]
    # In sls_visual.py -> get_adaptive_trajectory_tubes_exact    
    
    F, E_sens, K_gains, W = F[:T], E_sens[:T], K_gains[:T], W[:T]
    if L_gains is None:
        L_gains = K_gains
    else:
        L_gains = L_gains[:T]
    
    I_nt = jnp.eye(n_theta)
    def compute_psi_j(j):
        def scan_fn(carry, t):
            # FIX: Psi product is up to t-1, so we must use K_gains[t-1]
            next_carry = jnp.where(t <= j, I_nt, (I_nt - K_gains[t-1]) @ carry)
            return next_carry, next_carry
        _, psi_j = jax.lax.scan(scan_fn, I_nt, jnp.arange(T))
        return psi_j
    Psi = jnp.swapaxes(jax.vmap(compute_psi_j)(jnp.arange(T)), 0, 1) 
    
    # Eq 29: Initial Uncertainty M_{k, xi}
    Psi_j_0 = Psi[:, 0]
    J_term = jnp.einsum('jxn,jnm->jxm', E_sens, Psi_j_0, precision=jax.lax.Precision.HIGHEST)
    K_term = jnp.einsum('jln,jnm->jlm', K_gains, Psi_j_0, precision=jax.lax.Precision.HIGHEST)
    
    k_idx, j_idx = jnp.arange(T_plus_1)[:, None], jnp.arange(T)[None, :]
    mask_k_gt_j = (k_idx > j_idx) 
    
    M_k_xi = jnp.einsum('kj,kjxy,jym->kxm', mask_k_gt_j, Phi_tilde, J_term, precision=jax.lax.Precision.HIGHEST) + \
             jnp.einsum('kj,kjxl,jlm->kxm', mask_k_gt_j, Phi_bar, K_term, precision=jax.lax.Precision.HIGHEST)
    tube_xi = jnp.linalg.norm(M_k_xi @ G_0, ord=1, axis=-1)
    
    # Eq 30: Historical Noise M_{k, j}
    Z = jnp.einsum('ktxy,tyn->ktxn', Phi_tilde, E_sens, precision=jax.lax.Precision.HIGHEST) + \
        jnp.einsum('ktxl,tln->ktxn', Phi_bar, K_gains, precision=jax.lax.Precision.HIGHEST) 
    
    t_idx = jnp.arange(T)[:, None]
    Psi_padded = jnp.concatenate([Psi, jnp.zeros((T, 1, n_theta, n_theta))], axis=1)
    Psi_masked = Psi_padded[:, 1:] * (t_idx > j_idx)[:, :, None, None]
    
    Z_masked = Z * (k_idx > jnp.arange(T)[None, :])[:, :, None, None]
    S = jnp.einsum('ktxn,tjnm->kjxm', Z_masked, Psi_masked, precision=jax.lax.Precision.HIGHEST) 
    
    LW = jnp.einsum('jnm,jmw->jnw', L_gains, W, precision=jax.lax.Precision.HIGHEST)
    Delta_M = jnp.einsum('kjxn,jnw->kjxw', Phi_bar - S, LW, precision=jax.lax.Precision.HIGHEST)
    M_base = jnp.einsum('kjxy,jyw->kjxw', Phi_tilde, F, precision=jax.lax.Precision.HIGHEST)
    
    M_k_j = (M_base + Delta_M) * mask_k_gt_j[:, :, None, None]
    tube_w = jnp.linalg.norm(M_k_j, ord=1, axis=-1).sum(axis=1)
    
    return jnp.where((jnp.arange(T_plus_1) == 0)[:, None], 0.0, tube_xi + tube_w)
