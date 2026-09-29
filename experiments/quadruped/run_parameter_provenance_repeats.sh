#!/usr/bin/env bash

# Repeat one adaptive quadruped scenario to capture GPU-run variability and
# estimator decision provenance for the parameter figure.

set -uo pipefail

if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
fi

conda_setup="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
if [[ ! -f "${conda_setup}" ]]; then
    printf 'Conda initialization script not found: %s\n' "${conda_setup}" >&2
    exit 1
fi
source "${conda_setup}"
conda activate adaptive_sls

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
seed="${1:-33}"
repeat_count="${2:-5}"
steps="${3:-500}"
output_root="${4:-${script_dir}/figure_data/parameter_provenance_seed_$(printf '%04d' "${seed}")}" 
theta_half_width="${5:-1.6}"
admm_max_iterations="${6:-1000}"
rho_update_frequency="${7:-25}"
python_bin="${PYTHON_BIN:-python}"
manifest="${output_root}/manifest.tsv"

if (( repeat_count < 1 || steps < 1 )); then
    printf 'repeat_count and steps must be positive\n' >&2
    exit 2
fi

mkdir -p "${output_root}"
if [[ ! -f "${manifest}" ]]; then
    printf 'repeat\tseed\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tparameter_figure\tlog\n' > "${manifest}"
fi

for ((repeat = 1; repeat <= repeat_count; repeat++)); do
    run_dir="${output_root}/run_$(printf '%02d' "${repeat}")"
    checkpoint="${run_dir}/quadruped_adaptive_damping_mjx_rollout.npz"
    parameter_figure="${run_dir}/quadruped_adaptive_damping_parameters.png"
    log_file="${run_dir}/run.log"
    status_file="${run_dir}/status.tsv"
    mkdir -p "${run_dir}"

    if [[ -f "${status_file}" ]]; then
        printf 'Skipping completed/attempted repeat %s; remove %s to rerun it.\n' \
            "${repeat}" "${status_file}"
        continue
    fi

    started_at="$(date --iso-8601=seconds)"
    printf 'Starting repeat %s/%s with seed=%s at %s\n' \
        "${repeat}" "${repeat_count}" "${seed}" "${started_at}"

    "${python_bin}" -u "${script_dir}/quadruped.py" \
        --steps "${steps}" \
        --headless \
        --randomize-damping \
        --initial-theta-half-width "${theta_half_width}" \
        --admm-max-iterations "${admm_max_iterations}" \
        --rho-update-frequency "${rho_update_frequency}" \
        --seed "${seed}" \
        --output-dir "${run_dir}" \
        2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}
    finished_at="$(date --iso-8601=seconds)"

    if [[ ${exit_code} -eq 0 && -f "${checkpoint}" && -f "${parameter_figure}" ]]; then
        status="completed"
    elif [[ -f "${checkpoint}" ]]; then
        status="failed_partial_saved"
    else
        status="failed_no_checkpoint"
    fi

    printf 'repeat\tseed\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tparameter_figure\tlog\n' > "${status_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${repeat}" "${seed}" "${status}" "${exit_code}" "${started_at}" \
        "${finished_at}" "${checkpoint}" "${parameter_figure}" "${log_file}" \
        >> "${status_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${repeat}" "${seed}" "${status}" "${exit_code}" "${started_at}" \
        "${finished_at}" "${checkpoint}" "${parameter_figure}" "${log_file}" \
        >> "${manifest}"
done

printf 'Repeats finished. Figure data and manifest: %s\n' "${output_root}"
