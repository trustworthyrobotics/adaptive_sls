#!/usr/bin/env bash

if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
fi

set -uo pipefail

conda_setup="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
if [[ ! -f "${conda_setup}" ]]; then
    printf 'Conda initialization script not found: %s\n' "${conda_setup}" >&2
    exit 1
fi
source "${conda_setup}"
conda activate adaptive_sls

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sweep_root="${1:-${script_dir}/sweeps/damping_40}"
number_of_seeds="${2:-40}"
steps="${3:-500}"
seed_start="${4:-0}"
theta_half_width="${5:-1.6}"
admm_max_iterations="${6:-1000}"
rho_update_frequency="${7:-25}"
python_bin="${PYTHON_BIN:-python}"
max_run_attempts="${MAX_RUN_ATTEMPTS:-3}"
manifest="${sweep_root}/manifest.tsv"

mkdir -p "${sweep_root}"
if [[ ! -f "${manifest}" ]]; then
    printf 'seed\tmode\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tlog\n' > "${manifest}"
fi

run_one() {
    local seed="$1"
    local mode="$2"
    local script_name="$3"
    local run_dir="${sweep_root}/seed_$(printf '%04d' "${seed}")/${mode}"
    local log_file="${run_dir}/run.log"
    local status_file="${run_dir}/status.tsv"
    local checkpoint

    if [[ "${mode}" == "adaptive" ]]; then
        checkpoint="${run_dir}/quadruped_adaptive_damping_mjx_rollout.npz"
    else
        checkpoint="${run_dir}/quadruped_nonadaptive_damping_mjx_rollout.npz"
    fi

    mkdir -p "${run_dir}"
    if [[ -f "${status_file}" ]]; then
        printf 'Skipping seed=%s mode=%s; status file already exists.\n' "${seed}" "${mode}"
        return
    fi

    local started_at
    local finished_at
    local exit_code
    local status
    local attempt
    local attempt_log
    local retryable
    started_at="$(date --iso-8601=seconds)"

    printf 'Starting seed=%s mode=%s steps=%s half_width=%s admm_max=%s rho_update_frequency=%s at %s\n' \
        "${seed}" "${mode}" "${steps}" "${theta_half_width}" \
        "${admm_max_iterations}" "${rho_update_frequency}" "${started_at}"

    : > "${log_file}"
    exit_code=1
    for ((attempt = 1; attempt <= max_run_attempts; attempt++)); do
        attempt_log="${run_dir}/run_attempt_$(printf '%02d' "${attempt}").log"
        printf 'Attempt %s/%s for seed=%s mode=%s\n' \
            "${attempt}" "${max_run_attempts}" "${seed}" "${mode}" \
            | tee -a "${log_file}"

        "${python_bin}" -u "${script_dir}/${script_name}" \
            --steps "${steps}" \
            --headless \
            --no-plots \
            --randomize-damping \
            --initial-theta-half-width "${theta_half_width}" \
            --admm-max-iterations "${admm_max_iterations}" \
            --rho-update-frequency "${rho_update_frequency}" \
            --seed "${seed}" \
            --output-dir "${run_dir}" \
            2>&1 | tee "${attempt_log}" | tee -a "${log_file}"
        exit_code=${PIPESTATUS[0]}

        if [[ ${exit_code} -eq 0 ]]; then
            break
        fi

        retryable=false
        # Shell exit codes 134 and 139 are the usual abort/SIGSEGV results.
        if [[ ${exit_code} -eq 134 || ${exit_code} -eq 139 ]]; then
            retryable=true
        # A failure before any checkpoint is normally an import, JAX startup,
        # or process-loading failure and is also worth retrying.
        elif [[ ! -f "${checkpoint}" ]]; then
            retryable=true
        fi

        if [[ "${retryable}" != true || ${attempt} -ge ${max_run_attempts} ]]; then
            break
        fi

        printf 'Retrying seed=%s mode=%s after exit_code=%s; partial logs are preserved.\n' \
            "${seed}" "${mode}" "${exit_code}" | tee -a "${log_file}"
    done

    finished_at="$(date --iso-8601=seconds)"
    if [[ ${exit_code} -eq 0 ]]; then
        status="completed"
    elif [[ -f "${checkpoint}" ]]; then
        status="failed_partial_saved"
    else
        status="failed_no_checkpoint"
    fi

    printf 'seed\tmode\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tlog\n' > "${status_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${seed}" "${mode}" "${status}" "${exit_code}" \
        "${started_at}" "${finished_at}" "${checkpoint}" "${log_file}" \
        >> "${status_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${seed}" "${mode}" "${status}" "${exit_code}" \
        "${started_at}" "${finished_at}" "${checkpoint}" "${log_file}" \
        >> "${manifest}"

    printf 'Finished seed=%s mode=%s status=%s at %s\n' \
        "${seed}" "${mode}" "${status}" "${finished_at}"
}

for ((offset = 0; offset < number_of_seeds; offset++)); do
    seed=$((seed_start + offset))
    run_one "${seed}" adaptive quadruped.py
    run_one "${seed}" nonadaptive quadruped_nonadaptive.py
done

printf 'Sweep finished. Manifest: %s\n' "${manifest}"
