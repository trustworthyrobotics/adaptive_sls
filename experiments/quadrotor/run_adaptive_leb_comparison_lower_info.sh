#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
EXPERIMENT="${SCRIPT_DIR}/quadrotor_adaptive_sls.py"
OUTPUT_ROOT="${SCRIPT_DIR}/quadrotor_adaptive_leb_comparison_lower_info_results"

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
FAILURE_LOG="${OUTPUT_ROOT}/sweep_failures.log"

cd "${REPO_ROOT}"

run_case() {
    local label="$1"
    shift
    echo "Running ${label}..."
    python "${EXPERIMENT}" "$@"
}

# The information center is at z=0.55 m. With the disturbance transition at
# z=0.45 m and sharpness 15, the tanh gate is approximately 0.047 (95.3% off)
# at the information-center altitude.
INFORMATION_CENTER_Z=0.55
DISTURBANCE_Z_OFF=0.45
DISTURBANCE_Z_SHARPNESS=15.0

# Arguments supplied here (for example --horizon 110 or rollout counts) are
# forwarded to every experiment. The E-dagger-F weight is owned by the sweep.
QUICK_TEST=false
COMMON_ARGS=()
SKIP_NEXT=false
for arg in "$@"; do
    if [[ "${SKIP_NEXT}" == true ]]; then
        SKIP_NEXT=false
        continue
    fi

    if [[ "${arg}" == "--quick-test" ]]; then
        QUICK_TEST=true
    elif [[ "${arg}" == "--edagger-f-cost-weight" ]]; then
        SKIP_NEXT=true
    elif [[ "${arg}" == --edagger-f-cost-weight=* ]]; then
        continue
    else
        COMMON_ARGS+=("${arg}")
    fi
done

if [[ "${QUICK_TEST}" == true ]]; then
    COMMON_ARGS+=(
        --corner-rollouts-per-corner 0
        --num-random-rollouts 1
    )
fi

for weight in $(seq 100 100 2000); do
    WEIGHT_ROOT="${OUTPUT_ROOT}/weight_${weight}"

    if ! run_case "adaptive SLS with LEB, with E-dagger-F (weight=${weight}, info_z=${INFORMATION_CENTER_Z})" \
            "${COMMON_ARGS[@]}" \
            --output-dir "${WEIGHT_ROOT}/adaptive__leb_true__edagger_f_true" \
            --edagger-f-cost \
            --edagger-f-cost-weight "${weight}" \
            --information-center-z "${INFORMATION_CENTER_Z}" \
            --disturbance-z-off "${DISTURBANCE_Z_OFF}" \
            --disturbance-z-sharpness "${DISTURBANCE_Z_SHARPNESS}" \
            --leb; then
        failure_message="weight=${weight} failed; continuing with the next weight"
        echo "WARNING: ${failure_message}" >&2
        echo "$(date --iso-8601=seconds) ${failure_message}" >> "${FAILURE_LOG}"
        continue
    fi
done

echo "All lower-information-altitude adaptive LEB quadrotor SLS weight sweeps completed."
