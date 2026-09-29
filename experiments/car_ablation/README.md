# Car ablation

This experiment compares non-adaptive SLS and adaptive SLS, each with and
without the linearization-error bound (LEB), for both unmatched and matched
parameter dynamics.

Defaults are a 70-step, `dt=0.05` full-horizon rollout from `(1, 1)` to
`(1, -1)`, with heading initialized along the straight-line path. The
obstacle is centered at `(0.95, 0.0)` with radius `0.35`.
The initial speed is `0.0 m/s`, with speed limits `[-1.0, 3.0] m/s`.
The quadratic costs match `experiments/car_umatched/car_adaptive_leb.py`:
nominal state weights are `(0.5, 0.5, 0.1, 0.1)`, nominal input weights are
`(1.0, 10.0)`, and the SLS response weights use
`Q_bar = 10*diag(10, 10, 1, ...)` and `R_bar = I`.

Run all eight cases from the repository root with:

```bash
python experiments/car_ablation/run_car_ablation.py
```

Results are written under `experiments/car_ablation/results/{unmatched,matched}`.
Each configuration contains a portable `controller.npz`; each model directory
also contains a `manifest.json` enumerating the four saved controllers and the
experiment geometry.
The matched model puts the two constant parameter errors into the heading-rate
and acceleration channels; the unmatched model uses the existing car runner,
where they enter the x/y position channels.

To evaluate the saved feedback policies without re-solving MPC, generate the
standard 30 random and 10 adversarial rollouts per controller:

```bash
conda run -n adaptive_sls python experiments/car_ablation/sample_saved_controller_rollouts.py
```

For adversarial rollouts, each of the four physical-state disturbance
primitives is independently sampled from `{-1, 1}` at every time step, and the
two parameter primitives are sampled from `{-1, 1}`, placing the true
parameter at a corner of its initial uncertainty box. The sampled primitives
and seed are saved in `rollout_metrics.npz`, so every trajectory is replayable.

Then generate the paper-ready table and a machine-readable metric CSV with:

```bash
conda run -n adaptive_sls python experiments/car_ablation/generate_results_table.py
```

The generator writes `results/car_ablation_results_table.tex` using the
requested `booktabs`/`resizebox` style, plus `results/car_ablation_metrics.csv`.
It reports matched and unmatched dynamics in separate row blocks, comparing
Adaptive SLS with and without LEB to SLS with and without LEB. Containment is
the percentage of physical-state rollout coordinates within their certified
half-width; goal distance and tracking RMSE are calculated across all 40
rollouts. Tracking errors are pooled over every valid rollout/timestep pair.
The table reports the timestep-level mean and population standard deviation
of the natural-log position and parameter tube volumes, computed as the log
of the product of the saved coordinate half-widths. The tube sequence is
shared by the sampled rollouts, so it is counted once per timestep rather
than duplicated 40 times.

To inspect the source of any containment failures, plot the x/y tube
half-widths with all rollout deviations (random in blue, corner cases in
orange, violations marked in red):

```bash
conda run -n adaptive_sls python experiments/car_ablation/plot_tube_deviations.py
```
