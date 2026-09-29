# Pavone affine-disturbance-feedback MPC baseline

This folder implements the robust MPC policy class used in equations (5)--(6)
of `../pavone_rampc/rampc_pavone.pdf`. The nominal trajectory and a strictly
causal, time-varying affine disturbance-feedback policy are optimized once:

```text
u_k = v_k + sum_{j < k} K[k,j] d_j.
```

The car experiment uses a configurable 12-step disturbance-memory window by
default (older gain blocks are fixed to zero). Passing `--feedback-memory 0`
optimizes the full causal history.

The resulting gains are used for the complete finite-horizon rollout without
re-solving the optimization. Past disturbances are reconstructed from the
measured state-transition residuals, so the deployed policy is causal.

For nonlinear systems, the response and robust constraint tightening use the
LTV model along the nominal trajectory. In particular, this baseline does
**not** add a linearization-error bound (LEB). Its predicted tube is therefore
robust for the optimized LTV disturbance model, not a certified reachable set
for the original nonlinear dynamics. The nonlinear rollout is checked against
the predicted tube and reported separately.
