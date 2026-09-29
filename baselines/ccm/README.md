# Robust TVLQR tube (trajectory-local CCM approximation)

This baseline preserves the finite-horizon architecture of the RAMPC-CCM
paper: it plans a nominal trajectory and scalar ellipsoidal tube once, then
executes feedback around that frozen trajectory for the complete rollout.

For nominal states `z[k]`, inputs `v[k]`, Riccati metrics `P[k]`, and gains
`K[k]`, the implemented policy is

```text
u[k] = v[k] + K[k] (x[k] - z[k]).
```

The sampled tube is

```text
T[k] = {x : (x - z[k])' P[k] (x - z[k]) <= delta[k]^2}.
```

At each transition, boundary directions of `T[k]` and every vertex of the
parameter/disturbance box are propagated through the nonlinear dynamics. The
smallest `P[k+1]` ball containing those samples defines `delta[k+1]`. State,
input, and circular-obstacle constraints are tightened using support-function
bounds of that ellipsoid. Successive convexification alternates between the
nominal plan and these tube backoffs.

## Important limitation

This is not the paper's globally certified CCM. The experiment's dynamic
Dubins car begins at zero speed, where the lateral mode is not instantaneously
controllable, so a smooth global positive-rate CCM does not exist on the stated
domain. The Riccati metric is trajectory-local, and nonlinear propagation is
sampled. Output metadata states this explicitly.

The full experiment is launched with:

```bash
experiments/car/run_trajectory_local_ccm.sh
```
