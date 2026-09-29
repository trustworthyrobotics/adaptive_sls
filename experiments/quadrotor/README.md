# Adaptive quadrotor experiment

This folder contains two adaptive wind-force experiments:

- `quadrotor_adaptive_sls.py`: one full-horizon SLS solve followed by feedback rollouts.
- `quadrotor_adaptive_mpc.py`: a receding-horizon MPC rollout that resolves after every transition.
- `quadrotor_common.py`: the shared 15-state model, observable-subspace estimator, disturbance simulation, constraints, and plotting utilities.

The augmented state is

```text
[px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r, Fx, Fy, Fz]
```

The estimator uses the SVD-thresholded map

```text
P_obs = (C E_force)^dagger C
```

with `C = I_12`. The nominal force is `[0, 0, 0]` N, the simulated force is
`[0.20, -0.15, 0.10]` N, and the initial force uncertainty is
`[-0.125, 0.125]` N per axis.

Run from the repository root with the project environment active:

```bash
python experiments/quadrotor/quadrotor_adaptive_sls.py
python experiments/quadrotor/quadrotor_adaptive_mpc.py
```

The full-horizon SLS experiment runs 200 reproducible validation rollouts by
default:

- 80 corner/adversarial cases: all eight corners of the force-parameter box,
  with 10 distinct boundary disturbance directions per corner.
- 120 cases with a uniformly sampled true force inside the parameter box and
  random disturbances.

Change those counts with:

```bash
python experiments/quadrotor/quadrotor_adaptive_sls.py \
  --corner-rollouts-per-corner 10 \
  --num-random-rollouts 120
```

Both versions disable the interval linearization-error bound by default. To
enable it:

```bash
python experiments/quadrotor/quadrotor_adaptive_sls.py --enable-linearization-error
python experiments/quadrotor/quadrotor_adaptive_mpc.py --enable-linearization-error
```

For the full-horizon experiment, enabling this flag (also available as
`--leb`) includes both the nonlinear dynamics linearization error and the
variation of a state-dependent disturbance map across the state tube.

Run the two adaptive formulations with LEB, with and without the E-dagger-F
cost, using:

```bash
experiments/quadrotor/run_adaptive_leb_comparison.sh
```

For a quick smoke test with one rollout per solver case, use:

```bash
experiments/quadrotor/run_adaptive_leb_comparison.sh --quick-test
```

The full-horizon script chooses a mode-specific output directory beside the
script unless `--output-dir` is supplied:

- adaptive: `quadrotor_adaptive_sls_results`
- adaptive with linearization error:
  `quadrotor_adaptive_sls_results_with_lin_err`
- non-adaptive: `quadrotor_non-adaptive_sls_results`
- non-adaptive with linearization error:
  `quadrotor_non-adaptive_sls_results_with_lin_err`

Run the fixed-uncertainty, non-learning baseline with:

```bash
python experiments/quadrotor/quadrotor_adaptive_sls.py --non-adaptive
```

Run the adaptive controller with the basic stage-wise
`||E^dagger F||_F^2` information-gathering cost using the default information
altitude `z=0.7` m:

```bash
python experiments/quadrotor/quadrotor_adaptive_sls.py \
  --edagger-f-cost
```

The default full-horizon length is 110 steps, and the default E-dagger-F
weight is 1000.

Override the information altitude with `--information-center-z`. This mode writes to
`edagger_f_adaptive_sls_quadrotor_results` by default.

The full-horizon experiment uses two solves. The first runs the same GPUSLS
controller with `enable_fastsls=False` and produces the nominal reference
trajectory. The robust FastSLS solve then uses that trajectory as its reference
and state/control initialization. When E-dagger-F is enabled, its stage cost is
also active in the first nominal/reference solve.

Both solves enforce the terminal nominal XYZ position within `0.01 m` of the
goal by default. Disable this with `--no-terminal-position-constraint`, or set
the box tolerance with `--terminal-position-tolerance`.

This mode uses a 12-state physical nominal model, keeps the force estimate
fixed, propagates the initial force zonotope at every stage, and uses the
original independent-response `get_betas` backoffs rather than the adaptive
estimator/M-matrix construction.

The baseline planned exogenous disturbance is `0.06 * dt * I_12`. In
E-dagger-F mode uses an information altitude (default `z=0.7` m), and the
map becomes `0.06 * dt * g(z) * I_12`, where
`g(z) = (1 - tanh(k (z - z_mid))) / 2`. The default midpoint is `z_mid=0.5`
m with `k=15`, making the disturbance approximately 5% of its low-altitude
value at `z=0.6` m. Configure the altitude and gate with `--information-center-z`,
`--disturbance-z-off`, and `--disturbance-z-sharpness`. The run also writes an
`*_xz.png` plot to show the altitude passage through the information center
before reaching the goal.
