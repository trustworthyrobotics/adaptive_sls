#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
OUTPUT_DIR="${SCRIPT_DIR}/adaptive_nonadaptive_car_results"

if [[ ! -f "${CONDA_SETUP}" ]]; then
    echo "Conda setup script not found: ${CONDA_SETUP}" >&2
    exit 1
fi

source "${CONDA_SETUP}"
conda activate adaptive_sls

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

echo "Running adaptive/non-adaptive Dubins car with and without LEBs (information costs disabled)..."
python "${SCRIPT_DIR}/car_adaptive_unmatched_v2.py" \
    "$@" \
    --output-dir "${OUTPUT_DIR}" \
    --horizon 100 \
    --dt 0.05 \
    --exogenous-disturbance-scale 0.00030 \
    --parameter-uncertainty-bound 0.10 \
    --goal-x -0.75 \
    --goal-y -2.25 \
    --nonadaptive-terminal-x-weight 10.0 \
    --nonadaptive-terminal-y-weight 10.0 \
    --x-min -1.5 \
    --x-max 0.6 \
    --obstacle-x -0.25 \
    --obstacle-y 0.20 \
    --obstacle-radius 0.23 \
    --second-obstacle-x 0.25 \
    --second-obstacle-y -0.35 \
    --second-obstacle-radius 0.23 \
    --third-obstacle-x -0.26 \
    --third-obstacle-y -0.90 \
    --third-obstacle-radius 0.23 \
    --extra-obstacle 0.14 -1.45 0.23 \
    --no-terminal-position-constraint \
    --no-information-cost \
    --no-edagger-f-cost \
    --all-combinations

echo "Comparison completed. Results (including tube_widths_all_combinations.csv) are in ${OUTPUT_DIR}."
