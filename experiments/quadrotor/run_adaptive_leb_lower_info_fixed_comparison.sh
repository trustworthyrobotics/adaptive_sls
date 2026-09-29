#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
EXPERIMENT="${SCRIPT_DIR}/quadrotor_adaptive_sls.py"
OUTPUT_ROOT="${SCRIPT_DIR}/quadrotor_adaptive_leb_lower_info_fixed_comparison_results"

HORIZON=110
EDAGGER_F_COST_WEIGHT=600
INFORMATION_CENTER_Z=0.55
DISTURBANCE_Z_OFF=0.45
DISTURBANCE_Z_SHARPNESS=15.0

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate adaptive_sls

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${OUTPUT_ROOT}"
FAILURE_LOG="${OUTPUT_ROOT}/comparison_failures.log"

cd "${REPO_ROOT}"

run_case() {
    local label="$1"
    shift
    echo "Running ${label}..."
    python "${EXPERIMENT}" "$@"
}

COMMON_ARGS=(
    --horizon "${HORIZON}"
    --information-center-z "${INFORMATION_CENTER_Z}"
    --disturbance-z-off "${DISTURBANCE_Z_OFF}"
    --disturbance-z-sharpness "${DISTURBANCE_Z_SHARPNESS}"
    --leb
)

if [[ "${1:-}" == "--quick-test" ]]; then
    COMMON_ARGS+=(
        --corner-rollouts-per-corner 0
        --num-random-rollouts 1
    )
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--quick-test]" >&2
    exit 2
fi

if ! run_case "adaptive SLS + LEB without E-dagger-F" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/adaptive__leb_true__edagger_f_false"; then
    failure_message="adaptive SLS + LEB without E-dagger-F failed"
    echo "WARNING: ${failure_message}" >&2
    echo "$(date --iso-8601=seconds) ${failure_message}" >> "${FAILURE_LOG}"
fi

if ! run_case "adaptive SLS + LEB with E-dagger-F (weight=${EDAGGER_F_COST_WEIGHT})" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/adaptive__leb_true__edagger_f_true" \
        --edagger-f-cost \
        --edagger-f-cost-weight "${EDAGGER_F_COST_WEIGHT}"; then
    failure_message="adaptive SLS + LEB with E-dagger-F (weight=${EDAGGER_F_COST_WEIGHT}) failed"
    echo "WARNING: ${failure_message}" >&2
    echo "$(date --iso-8601=seconds) ${failure_message}" >> "${FAILURE_LOG}"
fi

echo "Adaptive LEB fixed comparison completed."
