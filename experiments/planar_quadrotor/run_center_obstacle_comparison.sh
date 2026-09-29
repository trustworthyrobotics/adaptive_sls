#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${script_dir}/results_obstacle_x1_y-0.1_r0.35_tol1e-2_inflated}"
export OBSTACLE_X="1.0"
export OBSTACLE_Y="-0.1"
export OBSTACLE_RADIUS="0.35"
export OBSTACLE_INFLATION="0.01"
export ALLOW_ROLLOUT_FAILURES="1"
export SLS_PRIMAL_TOLERANCE="1e-2"
export SQP_FEASIBILITY_TOLERANCE="1e-2"
export SLS_ITERATIONS="1"
export SQP_ITERATIONS="1"

exec "${script_dir}/run_full_comparison.sh"
