#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate gpusls_lin

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

echo "Running inactive adaptive-SLS comparison..."
python "${SCRIPT_DIR}/car_adaptive_unmatched_v2.py" \
    --adaptive \
    --no-information-cost \
    "$@"

echo "Running active adaptive-SLS comparison..."
python "${SCRIPT_DIR}/car_adaptive_unmatched_v2.py" \
    --adaptive \
    --information-cost \
    "$@"

echo "Both car experiments completed."
