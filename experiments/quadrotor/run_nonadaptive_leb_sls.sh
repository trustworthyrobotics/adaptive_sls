#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
EXPERIMENT="${SCRIPT_DIR}/quadrotor_adaptive_sls.py"
OUTPUT_DIR="${SCRIPT_DIR}/quadrotor_nonadaptive_combination_results/non_adaptive__leb_true"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate gpusls_lin

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

# Any arguments supplied to this script are forwarded to the experiment.
python "${EXPERIMENT}" \
    "$@" \
    --output-dir "${OUTPUT_DIR}" \
    --non-adaptive \
    --leb

echo "Non-adaptive quadrotor SLS with LEB completed."
