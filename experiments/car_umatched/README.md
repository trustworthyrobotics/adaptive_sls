# Positive-speed car comparison

This experiment compares the existing adaptive+LEB controller with two
solve-once robust baselines on one shared car problem:

- the feedback-linearized CCM tube controller; and
- Pavone-style robust MPC with an optimized, time-varying causal affine
  disturbance-feedback policy.

The common defaults are:

- initial speed: `0.38 m/s`
- goal speed: `0.60 m/s`
- certified speed domain: `v >= 0.10 m/s`
- maximum speed: `2.0 m/s`
- certified velocity projection: at least `0.10 m/s` along the start-to-goal
  displacement heading `atan2(-2.0, 0.25)`
- implied heading sector: `atan2(-2.0, 0.25) +/- acos(0.10/2)`
- goal position: `(0.25, -1.0)`
- parameter box: `[-0.075, 0.075]^2`, with true parameter `[-0.05, 0.05]`
- discrete disturbance scale: `0.00030` per 0.05-second step
- horizon: 75 steps at 0.05 seconds
- upper x-position limit: `1.5 m`
- nominal terminal-position tolerance: `0.01 m` per coordinate
- adaptive SLS tube-response weights: `Q_bar` scaled by `10x`, `R_bar = I`

The two lowest obstacles, formerly centered at `(0.14, -1.45)` and
`(-0.26, -1.30)`, are removed. The full-horizon adaptive+LEB robust ADMM solve previously produced
nonfinite iterates for the larger `[-0.10, 0.10]^2` box. Both methods use the
revised `[-0.075, 0.075]^2` box.

Both remaining obstacle radii are `0.23 m`. Their centers are
`(-0.25, 0.20)` and `(0.25, -0.25)`.

## Run

```bash
./experiments/car_umatched/run_comparison.sh
```

Results are placed under `experiments/car_umatched/comparison_results`, with
separate `certified_ccm`, `pavone_affine_df`, and
`adaptive_true__leb_true` subdirectories. The combined
`ccm_vs_adaptive_leb.png` overlays all available rollouts and nominal paths;
adaptive+LEB and Pavone tubes are rectangles and CCM tubes are circles.

Common arguments can be passed to the comparison script, for example:

```bash
./experiments/car_umatched/run_comparison.sh --output-dir /tmp/car_comparison
```

## Scope of the CCM guarantee

For positive speed, the map
`(px, py, theta, v) -> (px, py, v cos(theta), v sin(theta))` is nonsingular on
the chosen non-wrapping heading sector. The implemented input transformation
turns the nominal system into two double integrators. A constant contraction
metric is found by SDP in those coordinates and pulled back analytically to a
state-dependent CCM for the car.

The homothetic tube radius is propagated from the contraction comparison
equation, and the disturbance box maximum is checked at all of its vertices;
there is no state-direction sampling in that computation. Robust constraints
are imposed at the planning sample times. Thus the contraction/tube statement
is continuous-time under the stated model and disturbance assumptions, while
the obstacle and input constraints are certified at the discrete plan nodes.

The adaptive formulation retains its original inactive upper speed bound of
`10 m/s`, because imposing the CCM planner's `2 m/s` limit caused its ADMM
iterations to fail convergence. Its generated rollout is checked separately
against the CCM domain and remains below `2 m/s`.

## Scope of the Pavone baseline

The Pavone baseline implements the causal affine disturbance-feedback policy
class in equations (5)--(6) of `baselines/pavone_rampc/rampc_pavone.pdf`:

```text
u_k = v_k + sum_{j < k} K[k,j] d_j.
```

It jointly optimizes the nominal trajectory and the active strictly causal
gain blocks once at the beginning, then reconstructs past disturbances from
measured transition residuals and executes that saved policy for all 75
steps. By default each input uses a 12-step disturbance-history window;
`--feedback-memory 0` enables the full causal history. The disturbance box includes the full `[-0.075, 0.075]^2` unknown
velocity-bias range and the same exogenous disturbance used by the adaptive
experiment. It does not learn or shrink that box during rollout.

The robust-policy SCP now runs for up to 12 iterations by default (override
with `--robust-policy-iterations`). The summary reports the separate state and
input iterate changes, obstacle slack, virtual control, tolerance ratios, and
the primary condition preventing convergence. The same per-iteration values
are written to `convergence_history.csv` and embedded in `summary.json`.
The CVXPY problem is built once per run as a parameterized problem. Later SCP
refinements update the linearization, lifted response maps, trust-region
centers, and obstacle hyperplanes in place and warm-start the previous solver
solution. CVXPY's DPP canonicalization cache is deliberately disabled: at the
75-step horizon its quadratic route requests about 139 GiB and its conic route
requests about 286 GiB. Constraints are therefore canonicalized again after
parameter updates, while the problem object and warm-start values are reused.

When a CCM `nominal_plan.csv` is available in the selected output directory,
or in the default `comparison_results/certified_ccm` directory, Pavone uses it
as the nominal SCP warm start. The state nodes are interpolated to the Pavone
time grid and nominal turn-rate/acceleration inputs are reconstructed from
heading and speed differences. Use `--ccm-warmstart PATH` to select a specific
plan or `--no-ccm-warmstart` to retain the geometric initialization.

For the nonlinear car, robust response propagation uses the LTV linearization
along the nominal trajectory. This baseline intentionally does **not** account
for linearization-error bounds (LEBs). Consequently, its tube is a robust tube
for the optimized LTV prediction model, not a certified nonlinear reachable
set. The experiment reports both nonlinear tube containment and the size of
the reconstructed disturbance relative to the modeled box, so this limitation
is visible in the results.

### Tighter-clearance Pavone run

The standard run retains the original conservative S-shaped seed and a
`0.12 m` nominal preprocessing reserve. To remove that artificial gap while
keeping the final robust LTV tube constraint active, run:

```bash
./experiments/car_umatched/run_pavone_mpc_tight.sh
```

This uses `--route-offset -0.45 --nominal-obstacle-reserve 0.0` and writes to
`experiments/car_umatched/pavone_tight_results/pavone_affine_df`. Both values
can be overridden by passing them after the script name.

## Save controllers and sample common rollouts

After producing the comparison runs, export each complete feedback law with:

```bash
./experiments/car_umatched/save_feedback_controllers.sh
```

The default Pavone source is the valid no-CCM-warm-start run at
`pavone_tight_no_ccm_results/pavone_affine_df`. CCM and Pavone can be exported
from their existing artifacts. Older Adaptive+LEB runs did not save their
causal state-feedback and estimator gains, so the first invocation reruns only
Adaptive+LEB if `comparison_results/adaptive_true__leb_true/controller.npz` is
missing. Pass `--no-rerun-adaptive` to fail instead, or use
`--adaptive-results-dir`, `--ccm-dir`, `--pavone-dir`, and `--output-dir` to
select other locations.

Then sample the saved controllers on identical uncertainty realizations:

```bash
conda run -n adaptive_sls python \
  experiments/car_umatched/sample_saved_controller_rollouts.py
```

This writes `controller_rollouts.npz`. Its first 10 scenarios are labeled
`adversarial`: the four parameter-box corners are cycled across the disturbance
coefficient directions `+e_i`, `-e_i`, all `+1`, and all `-1`. The following
30 are labeled `random`, with independently sampled parameter and per-step
disturbance coefficients inside their zonotopes. The archive includes the
shared parameters and disturbances, method-indexed states and controls,
Adaptive+LEB parameter estimates, failure flags, obstacle clearance and goal
distance, plus both `scenario_types` and the Boolean `is_adversarial` label.
The `rollout_labels` field spells these out as `adversarial` and
`non_adversarial`. Use `--random-rollouts`, `--seed`, or `--output` to override
those defaults.

Plot all sampled trajectories and the three saved nominal plans with:

```bash
conda run -n adaptive_sls python \
  experiments/car_umatched/plot_controller_rollouts.py
```

The default output is `all_controller_rollouts.png`. Adaptive+LEB is blue, CCM
is orange, and Pavone is green. Each nominal plan is a dark dashed line, while
all 40 corresponding rollouts use a lighter, thinner line. The input archive,
controller directory, output path, alpha, and line widths can be changed with
the script's command-line options. Transparent tube patches are drawn every
two time steps by default; tune their density and opacity with `--tube-stride`
and `--tube-alpha`. The planar trajectory occupies both rows of the left
column; the right column compares every rollout's absolute $p_x$ and $p_y$
deviation against the corresponding tube half-width. The figure uses LaTeX
text rendering with a serif font at 300 DPI. Its compact single-column default
is 3 by 2 inches; override this with `--figure-width` and `--figure-height`.
The compact styling follows `planar_visual.pdf`: in-panel `(a)`--`(c)` labels,
prominent gray grids, boxed color legend, and horizon-indexed tube panels.
Panel labels are placed in the upper-left corners; pass `--no-panel-labels` to
omit them. The planar legend contains a dashed controller entry and a matching
colored tube patch for each method. It is a single-column legend in the middle
of the original two-column subplot layout, floating between the trajectory and
tube-plot columns without occupying a separate grid column.
The compact legend names Adaptive+LEB as `A-SLS` and Pavone's affine
disturbance-feedback controller as `DF`.
