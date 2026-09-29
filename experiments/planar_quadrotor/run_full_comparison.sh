#!/usr/bin/env bash
set -euo pipefail

# Full planar-quadrotor benchmark: 40 shared scenarios x 4 controller arms.
#
# Optional environment overrides:
#   CONDA_ENV=adaptive_sls
#   OUTPUT_DIR=/path/to/results
#   JAX_PLATFORMS=cuda
#   SEED=0
#   BATCH_SIZE=2
#   OBSTACLE_X=-0.1
#   OBSTACLE_Y=1.0
#   OBSTACLE_RADIUS=0.5
#   OBSTACLE_INFLATION=0.0
#   ALLOW_ROLLOUT_FAILURES=0
#   SLS_PRIMAL_TOLERANCE=1e-2
#   SQP_FEASIBILITY_TOLERANCE=1e-2
#   SLS_ITERATIONS=1
#   SQP_ITERATIONS=1
#   PAVONE_SCP_ITERATIONS=1

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"

conda_env="${CONDA_ENV:-adaptive_sls}"
output_dir="${OUTPUT_DIR:-${script_dir}/results}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-2}"
obstacle_x="${OBSTACLE_X:--0.1}"
obstacle_y="${OBSTACLE_Y:-1.0}"
obstacle_radius="${OBSTACLE_RADIUS:-0.5}"
obstacle_inflation="${OBSTACLE_INFLATION:-0.0}"
allow_rollout_failures="${ALLOW_ROLLOUT_FAILURES:-0}"
sls_primal_tolerance="${SLS_PRIMAL_TOLERANCE:-1e-2}"
sqp_feasibility_tolerance="${SQP_FEASIBILITY_TOLERANCE:-1e-2}"
sls_iterations="${SLS_ITERATIONS:-1}"
sqp_iterations="${SQP_ITERATIONS:-1}"
pavone_scp_iterations="${PAVONE_SCP_ITERATIONS:-1}"
if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]] || (( batch_size > 40 )); then
  echo "BATCH_SIZE must be an integer from 1 through 40" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}:${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-planar-quadrotor}"

mkdir -p "${output_dir}" "${MPLCONFIGDIR}"
log_file="${output_dir}/full_comparison.log"
touch "${log_file}"

cd "${repo_dir}"
echo "Launching 40-run planar-quadrotor comparison"
echo "  conda environment: ${conda_env}"
echo "  output directory:  ${output_dir}"
echo "  JAX platform:      ${JAX_PLATFORMS}"
echo "  seed:              ${seed}"
echo "  scenario batch:    ${batch_size}"
echo "  obstacle:          (${obstacle_x}, ${obstacle_y}), r=${obstacle_radius}"
echo "  obstacle inflation:${obstacle_inflation}"
echo "  allow failures:    ${allow_rollout_failures}"
echo "  SLS primal tol:    ${sls_primal_tolerance}"
echo "  SQP feasible tol:  ${sqp_feasibility_tolerance}"
echo "  SLS iterations:    ${sls_iterations}"
echo "  SQP iterations:    ${sqp_iterations}"
echo "  Pavone SCP iters:  ${pavone_scp_iterations}"
echo "  log:               ${log_file}"

# Fail before starting the Monte Carlo job if the requested JAX backend is not
# available, and leave an explicit device record in the experiment log.
conda run --no-capture-output -n "${conda_env}" python -c \
  "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())" \
  2>&1 | tee -a "${log_file}"

for ((run_start = 0; run_start < 40; run_start += batch_size)); do
  run_stop=$((run_start + batch_size))
  if (( run_stop > 40 )); then
    run_stop=40
  fi
  echo "Starting scenario batch [${run_start}, ${run_stop})" | tee -a "${log_file}"
  failure_args=(--fail-on-rollout-error)
  if [[ "${allow_rollout_failures}" == "1" ]]; then
    failure_args=()
  fi
  conda run --no-capture-output -n "${conda_env}" \
    python -m experiments.planar_quadrotor.run_experiment \
    --runs 40 \
    --seed "${seed}" \
    --max-steps 100 \
    --obstacle-x "${obstacle_x}" \
    --obstacle-y "${obstacle_y}" \
    --obstacle-radius "${obstacle_radius}" \
    --obstacle-inflation "${obstacle_inflation}" \
    --sls-primal-tolerance "${sls_primal_tolerance}" \
    --sqp-feasibility-tolerance "${sqp_feasibility_tolerance}" \
    --sls-iterations "${sls_iterations}" \
    --sqp-iterations "${sqp_iterations}" \
    --pavone-scp-iterations "${pavone_scp_iterations}" \
    --run-start "${run_start}" \
    --run-stop "${run_stop}" \
    --resume \
    --methods \
      adaptive_sls_gain \
      adaptive_sls_sme \
      ccm_rampc \
      pavone_rampc \
    "${failure_args[@]}" \
    --output-dir "${output_dir}" \
    2>&1 | tee -a "${log_file}"
done

echo "Comparison complete: ${output_dir}"
