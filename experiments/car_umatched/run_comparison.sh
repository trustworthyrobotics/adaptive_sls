#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/comparison_results"
ARGUMENTS=("$@")
for ((INDEX = 0; INDEX < ${#ARGUMENTS[@]}; INDEX++)); do
    case "${ARGUMENTS[INDEX]}" in
        --output-dir)
            if ((INDEX + 1 < ${#ARGUMENTS[@]})); then
                OUTPUT_DIR="${ARGUMENTS[INDEX + 1]}"
            fi
            ;;
        --output-dir=*)
            OUTPUT_DIR="${ARGUMENTS[INDEX]#--output-dir=}"
            ;;
    esac
done

echo "Running certified solve-once CCM baseline..."
"${SCRIPT_DIR}/run_ccm.sh" --output-dir "${OUTPUT_DIR}" "$@"

echo "Running solve-once Pavone affine disturbance-feedback baseline (no LEB)..."
"${SCRIPT_DIR}/run_pavone_mpc.sh" --output-dir "${OUTPUT_DIR}" "$@"

echo "Running adaptive+LEB on the same positive-speed domain..."
"${SCRIPT_DIR}/run_adaptive_leb.sh" --output-dir "${OUTPUT_DIR}" "$@"

echo "Plotting both methods on shared axes..."
"${SCRIPT_DIR}/run_plot_comparison.sh" --results-dir "${OUTPUT_DIR}"

echo "Positive-speed CCM/Pavone/adaptive+LEB comparison completed."
