#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
EXPERIMENT="${SCRIPT_DIR}/quadrotor_adaptive_sls.py"
OUTPUT_ROOT="${SCRIPT_DIR}/quadrotor_nonadaptive_combination_results"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate gpusls_lin

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${OUTPUT_ROOT}"

cd "${REPO_ROOT}"

run_case() {
    local label="$1"
    shift
    echo "Running ${label}..."
    python "${EXPERIMENT}" "$@"
}

# Any arguments supplied to this script (for example --horizon 120 or rollout
# counts) are forwarded to both experiments.
COMMON_ARGS=("$@")

run_case "non-adaptive without LEB" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/non_adaptive__leb_false" \
    --non-adaptive

run_case "non-adaptive with LEB" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/non_adaptive__leb_true" \
    --non-adaptive \
    --leb

echo "Both non-adaptive quadrotor SLS combinations completed."
