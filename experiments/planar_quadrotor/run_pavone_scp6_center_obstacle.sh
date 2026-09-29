#!/usr/bin/env bash
set -euo pipefail

# Rerun only Pavone RAMPC with enough outer SCP iterations to converge for the
# centered-obstacle benchmark. It uses the same scenarios and a new directory.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"

conda_env="${CONDA_ENV:-adaptive_sls}"
output_dir="${OUTPUT_DIR:-${script_dir}/results_obstacle_x1_y-0.1_r0.35_tol1e-2_inflated_pavone_scp6}"
batch_size="${BATCH_SIZE:-5}"
seed="${SEED:-0}"

if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]] || (( batch_size > 40 )); then
  echo "BATCH_SIZE must be an integer from 1 through 40" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}:${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-planar-quadrotor}"

mkdir -p "${output_dir}" "${MPLCONFIGDIR}"
log_file="${output_dir}/pavone_scp6.log"
touch "${log_file}"

cd "${repo_dir}"
echo "Launching 40-run Pavone SCP-6 centered-obstacle comparison" | tee -a "${log_file}"
echo "  output directory: ${output_dir}" | tee -a "${log_file}"
echo "  scenario batch:   ${batch_size}" | tee -a "${log_file}"

for ((run_start = 0; run_start < 40; run_start += batch_size)); do
  run_stop=$((run_start + batch_size))
  if (( run_stop > 40 )); then
    run_stop=40
  fi
  echo "Starting scenario batch [${run_start}, ${run_stop})" | tee -a "${log_file}"
  conda run --no-capture-output -n "${conda_env}" \
    python -m experiments.planar_quadrotor.run_experiment \
    --runs 40 \
    --seed "${seed}" \
    --max-steps 100 \
    --obstacle-x 1.0 \
    --obstacle-y -0.1 \
    --obstacle-radius 0.35 \
    --obstacle-inflation 0.01 \
    --sls-primal-tolerance 1e-2 \
    --sqp-feasibility-tolerance 1e-2 \
    --sls-iterations 1 \
    --sqp-iterations 1 \
    --pavone-scp-iterations 6 \
    --run-start "${run_start}" \
    --run-stop "${run_stop}" \
    --resume \
    --methods pavone_rampc \
    --fail-on-rollout-error \
    --output-dir "${output_dir}" \
    2>&1 | tee -a "${log_file}"
done

echo "Pavone SCP-6 comparison complete: ${output_dir}" | tee -a "${log_file}"
