#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Remove the artificial nominal reserve and initialize the sequential obstacle
# approximation closer to the obstacles. Robust LTV tube tightening remains
# active in the final Pavone policy solve.
"${SCRIPT_DIR}/run_pavone_mpc.sh" \
  --output-dir "${SCRIPT_DIR}/pavone_tight_results" \
  --route-offset -0.45 \
  --nominal-obstacle-reserve 0.0 \
  "$@"
