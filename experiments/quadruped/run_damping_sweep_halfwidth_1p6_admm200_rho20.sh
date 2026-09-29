#!/usr/bin/env bash

# Paired 40-seed sweep: theta half-width 1.6, ADMM cap 200, rho update every 20.
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_CLIENT_MEM_FRACTION=0.825

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sweep_root="${1:-${script_dir}/sweeps/damping_40_halfwidth_1p6_admm200_rho20}"
number_of_seeds="${2:-40}"
steps="${3:-500}"
seed_start="${4:-0}"

exec "${script_dir}/run_damping_sweep.sh" \
    "${sweep_root}" \
    "${number_of_seeds}" \
    "${steps}" \
    "${seed_start}" \
    1.6 \
    200 \
    20
