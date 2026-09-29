# Planar quadrotor receding-horizon comparison

This experiment evaluates four controllers on the same 40 parameter and
disturbance realizations:

1. Adaptive SLS + LEB with the gated gain update.
2. Adaptive SLS + LEB with an exact scalar set-membership interval followed
   by a containing zonotope.
3. The authors' adaptive homothetic CCM tube MPC.
4. Pavone-style adaptive robust MPC with causal affine disturbance feedback.

All arms use horizon 10, `dt=0.05`, at most 100 MPC steps, and stop within
0.2 m of `(2,0)`. The initial position is `(0,0)` and the obstacle is centered
at `(-0.1,1.0)` with radius `0.5`.

## Certified uncertainty model

The plant now matches the uncertainty model used to synthesize the supplied
CCM. The nominal mass is `0.486 kg`, and the learned parameter is the additive
inverse-mass error

```text
theta = 1/m_true - 1/m_nominal,
theta in [-0.01/m_nominal, 0.01/m_nominal]
      = [-0.0205761, 0.0205761] kg^-1.
```

The exogenous disturbance is the authors' scalar inertial-horizontal
acceleration `d in [-0.1,0.1] m/s^2`. In body-velocity coordinates its
continuous-time channel is

```text
E(phi) d = [0, 0, 0, cos(phi)d, -sin(phi)d, 0].
```

This scalar channel—not an independent two-axis disturbance—is used by the
plant and every robust prediction model.

The parameter-learning residual is separately gated by

```text
C = diag([0, 0, 0, 1, 1, 0]).
```

Thus only the `v1` and `v2` transition residuals update `theta`. `C` is passed
to the Adaptive SLS solver as its `measurement_matrix` and is also applied by
the gain and SME estimators. It does not replace the physical disturbance
channel `E(phi)`.

## CCM certificate

The `W,Y` data under `third_party/rampc_ccm/` came from the authors' public
RAMPC-CCM repository at commit
`fbab02d8acd955e1fc18d40b651d747bbe431367`, with its MIT license. Unlike the
earlier two-axis-wind experiment, the current inverse-mass/scalar-disturbance
model is the model for which these polynomials were SOS synthesized.

Run the independent exact-trigonometric audit with:

```bash
conda run -n adaptive_sls python -m experiments.planar_quadrotor.audit_ccm
```

The stored rate is approximately `0.69905`. The audit includes every
state/parameter/disturbance corner plus random interior samples and records its
result in CCM rollout diagnostics. The old two-axis-wind resynthesis script
and rejected artifact are retained only as provenance and are not loaded.

## Run

Launch the complete 40-scenario, four-controller comparison with:

```bash
./experiments/planar_quadrotor/run_full_comparison.sh
```

Launch the second comparison with the `0.35 m` obstacle centered at
`(1.0,-0.1)` into its own results directory with:

```bash
./experiments/planar_quadrotor/run_center_obstacle_comparison.sh
```

This stress-test launcher keeps strict RTI with one outer SQP and one inner
SLS iteration and restores the original `1e-2` tolerances. The physical
obstacle radius is `0.35 m`, while every controller optimizes against a
`0.36 m` radius: the obstacle is inflated by the `0.01 m` solver tolerance.
Realized clearance is measured against the physical obstacle, and Adaptive
SLS records robust margins against both radii. Failed/rejected rollouts do not
abort the remaining Monte Carlo batches.

The strict one-iteration Pavone arm can fail to converge on this obstacle.
After retaining those failures as the RTI baseline, rerun Pavone with six
outer SCP iterations using:

```bash
./experiments/planar_quadrotor/run_pavone_scp6_center_obstacle.sh
```

This uses the same 40 scenarios, performs the exact directional disturbance
support check used by its optimization constraints, and writes to a separate
`*_pavone_scp6` results directory.

The launcher defaults to `JAX_PLATFORMS=cuda`, verifies and logs the selected
JAX backend/devices before starting, and writes progress to
`experiments/planar_quadrotor/results/full_comparison.log`.
The launcher exits nonzero after preserving all partial artifacts if any
rollout fails, so a failed Monte Carlo job is not reported as complete.
It runs two scenarios per Python process by default to release accumulated
JAX/XLA native state between batches, and resumes matching successful rollout
artifacts automatically. Override the batch size with `BATCH_SIZE` if needed.
Its conda environment, output directory, seed, and JAX platform can be
overridden, for example for a CPU run:

```bash
OUTPUT_DIR=/scratch/planar_results JAX_PLATFORMS=cpu \
  ./experiments/planar_quadrotor/run_full_comparison.sh
```

The equivalent direct Python command is:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-planar \
conda run -n adaptive_sls python -m experiments.planar_quadrotor.run_experiment
```

A short smoke test is:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-planar \
conda run -n adaptive_sls python -m experiments.planar_quadrotor.run_experiment \
  --runs 1 --max-steps 2 --output-dir /tmp/planar_quadrotor_smoke
```

The Adaptive SLS adapter uses one SLS and one SQP iteration per MPC call. Its
parameter generators are Girard-reduced/padded to exactly
`n_theta * q_max` columns, matching the solver's cached `G_prev`
representation while the set contracts.

All RTI warm starts remain enabled, including the nominal trajectory, tube
responses, adaptive `E/K/G/P` histories, and ADMM primal/dual state. The
common benchmark cost is `Q=diag(10,10,2,1,1,0.5)`, `Qf=10Q`, and `R=I`.
This balanced scaling avoids the singular shifted inequality-dual weights
produced by the earlier overly aggressive `Q_xy=30`, `R=0.1` choice. The LEB
and contraction/posterior gates remain enabled.

The CCM baseline solves its constrained nonlinear MPC problem to the configured
SLSQP convergence limit at every receding-horizon step; it does not replace the
authors' robust CCM optimization with a single optimizer iteration.

## Outputs

`results/` contains the configuration, exact shared scenarios, aggregate CSV
and JSON summaries, compressed rollout archives, and per-solve CSV files.
Each rollout stores trajectories, controls, horizon forecasts, state/input
tube widths, inverse-mass interval widths and errors, solver timing/status,
plant propagation time, separately measured estimator-update time, constraint
margins, obstacle clearance, goal error, SME endpoints, zonotope containment,
contraction/posterior gates, JAX backend/device, and scenario seeds. The SME
and gated-gain estimators are small NumPy routines rather than JIT-compiled
JAX kernels; their recorded update time includes residual construction,
measurement-channel gating, set/gain update, and zonotope construction.

Generate aggregate plots with:

```bash
conda run -n adaptive_sls python -m experiments.planar_quadrotor.plot_results \
  experiments/planar_quadrotor/results
```

Add `--rollout-run N` to also visualize selected receding-horizon forecasts
and terminal XY tube cross-sections for scenario `N`, for example:

```bash
conda run -n adaptive_sls python -m experiments.planar_quadrotor.plot_results \
  experiments/planar_quadrotor/results --rollout-run 0
```
