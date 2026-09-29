#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
OUTPUT_DIR="${SCRIPT_DIR}/edagger_f_car_results"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate adaptive_sls

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

echo "Running baseline adaptive-SLS experiment..."
python "${SCRIPT_DIR}/car_adaptive_unmatched_v2.py" \
    --output-dir "${OUTPUT_DIR}" \
    "$@" \
    --adaptive \
    --no-information-cost \
    --no-edagger-f-cost

echo "Running E-dagger-F adaptive-SLS experiment..."
python "${SCRIPT_DIR}/car_adaptive_unmatched_v2.py" \
    --output-dir "${OUTPUT_DIR}" \
    "$@" \
    --adaptive \
    --no-information-cost \
    --edagger-f-cost

echo "Both E-dagger-F comparison experiments completed."
