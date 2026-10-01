# Planning to Learn: A-SLS

This repository contains the shareable implementation and experiment code for **Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning**.

**Links:** [Website](https://trustworthyrobotics.github.io/adaptive_sls/) | [Paper](https://trustworthyrobotics.github.io/adaptive_sls/paper/root.pdf) | [Video](https://trustworthyrobotics.github.io/adaptive_sls/#crazyflie-video)

A-SLS is a robust adaptive model-predictive-control framework built on system-level synthesis. It augments the physical state with online parameter estimates, propagates correlated disturbance and estimation uncertainty, and accounts for guaranteed future learning while planning. The GPU-parallel solver is designed for long-horizon nonlinear robotic systems and was evaluated on Dubins-car, quadrotor, planar-quadrotor, quadruped, and Crazyflie problems, including systems with up to 61 states, 8 uncertain parameters, and 12 control inputs.

## Access and installation

Clone the repository and enter it:

```bash
git clone https://github.com/trustworthyrobotics/adaptive_sls.git
cd adaptive_sls
```

If this directory was supplied as an archive instead, extract it and run the remaining commands from the extracted repository root.

The linearization-bound implementation is vendored at `src/linearization_sls`; it is regular source code in this distribution, not a Git submodule. Do **not** run `git submodule update --init --recursive`.

Create and activate the Conda environment, install the repository in editable mode, and install the linearization dependency:

```bash
conda env create -f environment.yaml
conda activate adaptive_sls
pip install -e .
pip install linrax --no-deps
```

Python 3.11 is required. The default package configuration installs the CUDA 12 build of JAX and is intended for NVIDIA GPUs. CPU-only users should install the appropriate JAX build for their platform and may need to replace the `jax[cuda12]` dependency in `pyproject.toml` before installation.

Verify the installation with:

```bash
python -c "import jax, gpu_sls; print(jax.default_backend(), jax.devices())"
pytest -q
```

## Repository layout

- `src/gpu_sls/`: GPU-parallel adaptive SLS, ADMM, SQP, and MPC wrappers.
- `src/linearization_sls/`: vendored interval and Taylor-model linearization-error bounds; no submodule checkout is needed.
- `src/utils/`: shared constraint and tube-visualization utilities.
- `baselines/`: contraction-metric and affine disturbance-feedback baselines.
- `experiments/car_ablation/`: matched and unmatched Dubins-car ablations.
- `experiments/car_umatched/`: unmatched Dubins-car comparison against CCM and disturbance-feedback RAMPC.
- `experiments/quadrotor/`: 12-state quadrotor active-information experiment.
- `experiments/planar_quadrotor/`: receding-horizon planar-quadrotor comparison. This package was moved under `experiments/` for a consistent layout.
- `experiments/quadruped/`: final adaptive and non-adaptive Go2 damping experiments with the required MPX snapshot and model assets vendored locally.
- `experiments/crazyflie_12d_adaptive/`: self-contained 12D Crazyflie effective-gain simulation and timing-pipeline code.
- `tests/`: source-level regression tests for the retained code.

## Reproducing the main experiments

Run commands from the repository root with the Conda environment active. Full experiments can require substantial GPU memory and time; use the documented quick or reduced-run options first.

### Dubins-car ablation

```bash
python experiments/car_ablation/run_car_ablation.py
```

This runs adaptive/non-adaptive SLS with and without linearization-error bounds for matched and unmatched uncertainty. See `experiments/car_ablation/README.md` for sampling and aggregation commands.

### Dubins-car baselines

```bash
./experiments/car_umatched/run_comparison.sh
```

This compares A-SLS with the feedback-linearized CCM and causal affine disturbance-feedback baselines. See `experiments/car_umatched/README.md` for individual baseline runs and common-rollout evaluation.

### Quadrotor active information gathering

Start with the one-rollout smoke comparison:

```bash
./experiments/quadrotor/run_adaptive_leb_lower_info_fixed_comparison.sh --quick-test
```

Remove `--quick-test` for the complete A-SLS versus A-SLS+ active-information comparison. Additional modes are documented in `experiments/quadrotor/README.md`.

### Planar quadrotor

A short CPU smoke run is:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-planar \
python -m experiments.planar_quadrotor.run_experiment \
  --runs 1 --max-steps 2 --output-dir /tmp/planar_quadrotor_smoke
```

Run the complete shared-scenario comparison with:

```bash
./experiments/planar_quadrotor/run_full_comparison.sh
```

See `experiments/planar_quadrotor/README.md` for the CCM audit, centered-obstacle stress test, and plotting commands.

### Quadruped scalability

A short headless adaptive run is:

```bash
python experiments/quadruped/quadruped.py \
  --headless --steps 10 --no-plots --output-dir /tmp/quadruped_smoke
```

The fixed-uncertainty baseline uses `experiments/quadruped/quadruped_nonadaptive.py`. Sweep launchers in the same directory reproduce the damping studies.

### Crazyflie 12D simulation

```bash
python -m experiments.crazyflie_12d_adaptive.crazyflie_12d \
  --steps 10 --no-plots --output-dir /tmp/crazyflie_smoke
```

The MuJoCo model and meshes required by this simulation are vendored inside the experiment directory.

## Generated outputs

Experiment scripts create their own result directories and regenerate plots, tables, JSON/CSV summaries, and compressed rollout arrays. Those generated artifacts are intentionally omitted from this source distribution. Cache directories, bytecode, animations, paper sources, and the superseded experiment folders are also omitted.

For a clean run, direct large outputs to scratch storage with each script `--output-dir` option or corresponding environment variable.

## Notes

Third-party assets retain their accompanying license and provenance files. The planar-quadrotor CCM certificate data is under `experiments/planar_quadrotor/third_party/rampc_ccm`, and the quadruped MPX snapshot is described in `experiments/quadruped/vendor/README.md`.
