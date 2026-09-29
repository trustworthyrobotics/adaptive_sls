#!/usr/bin/env bash

# Re-run the two adaptive failures without modifying the original sweep data.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

set -uo pipefail

conda_setup="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
if [[ ! -f "${conda_setup}" ]]; then
    printf 'Conda initialization script not found: %s\n' "${conda_setup}" >&2
    exit 1
fi
source "${conda_setup}"
conda activate adaptive_sls

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${1:-${script_dir}/sweeps/adaptive_failure_reruns}"
steps="${2:-500}"
python_bin="${PYTHON_BIN:-python}"
max_run_attempts="${MAX_RUN_ATTEMPTS:-3}"

for seed in 11 18; do
    run_dir="${output_root}/seed_$(printf '%04d' "${seed}")/adaptive"
    checkpoint="${run_dir}/quadruped_adaptive_damping_mjx_rollout.npz"
    status_file="${run_dir}/status.tsv"
    log_file="${run_dir}/run.log"
    mkdir -p "${run_dir}"

    if [[ -f "${status_file}" ]]; then
        printf 'Skipping seed=%s; status file already exists at %s\n' \
            "${seed}" "${status_file}"
        continue
    fi

    started_at="$(date --iso-8601=seconds)"
    : > "${log_file}"
    exit_code=1

    for ((attempt = 1; attempt <= max_run_attempts; attempt++)); do
        attempt_log="${run_dir}/run_attempt_$(printf '%02d' "${attempt}").log"
        printf 'Attempt %s/%s for adaptive seed=%s\n' \
            "${attempt}" "${max_run_attempts}" "${seed}" | tee -a "${log_file}"

        "${python_bin}" -u "${script_dir}/quadruped.py" \
            --steps "${steps}" \
            --headless \
            --no-plots \
            --randomize-damping \
            --seed "${seed}" \
            --output-dir "${run_dir}" \
            2>&1 | tee "${attempt_log}" | tee -a "${log_file}"
        exit_code=${PIPESTATUS[0]}

        if [[ ${exit_code} -eq 0 ]]; then
            break
        fi
        if [[ ${exit_code} -ne 134 && ${exit_code} -ne 139 && -f "${checkpoint}" ]]; then
            break
        fi
        if [[ ${attempt} -lt ${max_run_attempts} ]]; then
            printf 'Retrying adaptive seed=%s after exit_code=%s.\n' \
                "${seed}" "${exit_code}" | tee -a "${log_file}"
        fi
    done

    finished_at="$(date --iso-8601=seconds)"
    if [[ ${exit_code} -eq 0 ]]; then
        status="completed"
    elif [[ -f "${checkpoint}" ]]; then
        status="failed_partial_saved"
    else
        status="failed_no_checkpoint"
    fi

    printf 'seed\tmode\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tlog\n' \
        > "${status_file}"
    printf '%s\tadaptive\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${seed}" "${status}" "${exit_code}" "${started_at}" "${finished_at}" \
        "${checkpoint}" "${log_file}" >> "${status_file}"
    printf 'Finished adaptive seed=%s status=%s\n' "${seed}" "${status}"
done

printf 'Adaptive failure reruns finished under %s\n' "${output_root}"
