"""Off-vehicle checks for the one-step-ahead hardware timing contract."""

from __future__ import annotations

import time

import numpy as np
import pytest

from experiments.crazyflie_12d_adaptive.hardware_pipeline import OneStepAheadPipeline, SolveDeadlineMissed


def test_beta_update_is_delayed_one_solve() -> None:
    solve_requests = []

    def predict(x, u, beta):
        return x + u + beta

    def solve(request):
        solve_requests.append(request)
        return request.predicted_state + 10.0

    def estimate(previous, actual, command, beta, generator):
        np.testing.assert_allclose(previous, [0.0])
        np.testing.assert_allclose(actual, [3.0])
        np.testing.assert_allclose(command, [1.0])
        return beta + 1.0, generator + 1

    pipeline = OneStepAheadPipeline(predict=predict, solve=solve, estimate=estimate)
    try:
        np.testing.assert_allclose(pipeline.start(np.array([0.0]), bootstrap_command=np.array([1.0]), beta=2.0, generator=0), [1.0])
        # Let the first background task complete.  Its request must use beta=2.
        for _ in range(100):
            if solve_requests:
                break
            time.sleep(.001)
        assert solve_requests[0].beta == 2.0
        result = pipeline.advance(np.array([3.0]), timeout_s=.1)
        np.testing.assert_allclose(result.command, [13.0])  # (0 + 1 + 2) + 10
        assert result.beta == 3.0
        # The newly launched solve is the first one allowed to use beta=3.
        for _ in range(100):
            if len(solve_requests) == 2:
                break
            time.sleep(.001)
        assert solve_requests[1].beta == 3.0
        np.testing.assert_allclose(solve_requests[1].predicted_state, [19.0])  # 3 + 13 + 3
    finally:
        pipeline.close()


def test_deadline_miss_is_not_silently_applied() -> None:
    pipeline = OneStepAheadPipeline(
        predict=lambda x, u, beta: x,
        solve=lambda request: (time.sleep(.05), np.array([2.0]))[1],
        estimate=lambda previous, actual, command, beta, generator: (beta, generator),
    )
    try:
        pipeline.start(np.array([0.0]), bootstrap_command=np.array([1.0]), beta=1.0, generator=None)
        with pytest.raises(SolveDeadlineMissed):
            pipeline.advance(np.array([0.0]))
    finally:
        pipeline.close()
