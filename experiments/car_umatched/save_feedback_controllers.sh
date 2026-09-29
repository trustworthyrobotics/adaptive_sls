#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"

CCM_DIR="${SCRIPT_DIR}/comparison_results/certified_ccm"
ADAPTIVE_ROOT="${SCRIPT_DIR}/comparison_results"
ADAPTIVE_DIR="${ADAPTIVE_ROOT}/adaptive_true__leb_true"
PAVONE_DIR="${SCRIPT_DIR}/pavone_tight_no_ccm_results/pavone_affine_df"
OUTPUT_DIR="${SCRIPT_DIR}/saved_controllers"
RERUN_ADAPTIVE=1

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "  --ccm-dir DIR" \
    "  --adaptive-results-dir DIR  Parent containing adaptive_true__leb_true" \
    "  --pavone-dir DIR" \
    "  --output-dir DIR" \
    "  --no-rerun-adaptive         Fail if the adaptive controller artifact is absent"
}

while (($#)); do
  case "$1" in
    --ccm-dir) CCM_DIR="$2"; shift 2 ;;
    --adaptive-results-dir)
      ADAPTIVE_ROOT="$2"
      ADAPTIVE_DIR="${ADAPTIVE_ROOT}/adaptive_true__leb_true"
      shift 2
      ;;
    --pavone-dir) PAVONE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --no-rerun-adaptive) RERUN_ADAPTIVE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

source "${CONDA_SETUP}"
conda activate adaptive_sls
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaptive_sls_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

if [[ ! -f "${ADAPTIVE_DIR}/controller.npz" ]]; then
  if ((RERUN_ADAPTIVE)); then
    printf 'Adaptive feedback gains were not saved by the earlier run; regenerating them once.\n'
    "${SCRIPT_DIR}/run_adaptive_leb.sh" --output-dir "${ADAPTIVE_ROOT}"
  else
    printf 'Missing adaptive controller: %s\n' "${ADAPTIVE_DIR}/controller.npz" >&2
    exit 1
  fi
fi

python "${SCRIPT_DIR}/export_feedback_controllers.py" \
  --ccm-dir "${CCM_DIR}" \
  --adaptive-dir "${ADAPTIVE_DIR}" \
  --pavone-dir "${PAVONE_DIR}" \
  --output-dir "${OUTPUT_DIR}"
