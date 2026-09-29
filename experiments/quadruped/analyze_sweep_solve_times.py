#!/usr/bin/env python3
"""Summarize saved MPC solve times from a quadruped damping sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MODES = ("adaptive", "nonadaptive")
CHECKPOINT_NAMES = {
    "adaptive": "quadruped_adaptive_damping_mjx_rollout.npz",
    "nonadaptive": "quadruped_nonadaptive_damping_mjx_rollout.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool the transition_times saved by each sweep checkpoint and report "
            "solve-time statistics separately for adaptive and non-adaptive MPC."
        )
    )
    parser.add_argument(
        "sweep_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "sweeps" / "damping_40",
        help="Sweep directory containing seed_NNNN folders.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed to include (default: 0).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=40,
        help="Number of consecutive seeds to include (default: 40).",
    )
    parser.add_argument(
        "--warmup-solves",
        type=int,
        default=4,
        help="Number of initial solves to discard from every run (default: 4).",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Exclude partial checkpoints from failed runs.",
    )
    return parser.parse_args()


def read_status(status_path: Path) -> str | None:
    if not status_path.exists():
        return None
    with status_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows[-1]["status"] if rows else None


def load_times(
    sweep_root: Path,
    mode: str,
    seeds: range,
    warmup_solves: int,
    completed_only: bool,
) -> tuple[np.ndarray, int, int, list[str]]:
    runs_used = 0
    failed_runs_used = 0
    arrays: list[np.ndarray] = []
    warnings: list[str] = []

    for seed in seeds:
        run_dir = sweep_root / f"seed_{seed:04d}" / mode
        status = read_status(run_dir / "status.tsv")
        if status is None:
            warnings.append(f"seed {seed} {mode}: missing status.tsv")
            continue
        if completed_only and status != "completed":
            continue

        checkpoint = run_dir / CHECKPOINT_NAMES[mode]
        if not checkpoint.exists():
            warnings.append(f"seed {seed} {mode}: missing checkpoint")
            continue

        with np.load(checkpoint) as data:
            if "transition_times" not in data:
                warnings.append(f"seed {seed} {mode}: transition_times not saved")
                continue
            times = np.asarray(data["transition_times"], dtype=np.float64).reshape(-1)

        times = times[np.isfinite(times)]
        if times.size <= warmup_solves:
            warnings.append(
                f"seed {seed} {mode}: only {times.size} finite solve times; "
                f"cannot discard {warmup_solves} warm-up solves"
            )
            continue

        arrays.append(times[warmup_solves:])
        runs_used += 1
        failed_runs_used += status != "completed"

    if not arrays:
        return np.empty(0, dtype=np.float64), runs_used, failed_runs_used, warnings
    return np.concatenate(arrays), runs_used, failed_runs_used, warnings


def main() -> None:
    args = parse_args()
    if args.seed_start < 0 or args.num_seeds <= 0 or args.warmup_solves < 0:
        raise SystemExit("seed-start and warmup-solves must be nonnegative; num-seeds must be positive")

    seeds = range(args.seed_start, args.seed_start + args.num_seeds)
    print(
        f"Sweep: {args.sweep_root.resolve()}\n"
        f"Seeds: {seeds.start}-{seeds.stop - 1}; "
        f"discarding first {args.warmup_solves} solves per run; "
        f"partial runs: {'excluded' if args.completed_only else 'included'}"
    )
    print("\nmode          runs  partial  solves    avg_ms     q1_ms  median_ms     q3_ms    std_ms")

    all_warnings: list[str] = []
    for mode in MODES:
        times, runs_used, failed_runs_used, warnings = load_times(
            args.sweep_root,
            mode,
            seeds,
            args.warmup_solves,
            args.completed_only,
        )
        all_warnings.extend(warnings)
        if not times.size:
            print(f"{mode:<12} {runs_used:5d} {failed_runs_used:8d} {0:7d}  no data")
            continue

        times_ms = 1.0e3 * times
        q1, median, q3 = np.percentile(times_ms, [25.0, 50.0, 75.0])
        print(
            f"{mode:<12} {runs_used:5d} {failed_runs_used:8d} {times_ms.size:7d} "
            f"{times_ms.mean():9.2f} {q1:9.2f} {median:10.2f} "
            f"{q3:9.2f} {times_ms.std(ddof=0):9.2f}"
        )

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
