"""Deadline-driven MPC pipeline for the physical Crazyflie.

The hardware cannot pause while an RTI iteration runs.  This module owns the
one-sample-ahead schedule used by the hardware runner:

* command ``u_k`` is held by the commander while solve ``S_{k+1}`` runs;
* ``S_{k+1}`` is initialized at ``f(x_k, u_k, beta_k)``;
* at the next boundary, the measured transition updates beta, but that update
  is used by ``S_{k+2}``, never retroactively by ``S_{k+1}``.

It intentionally has no cflib or JAX dependency.  The radio and MPC adapters
are injected so its timing semantics can be unit-tested off vehicle.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Generic, TypeVar

import numpy as np


Parameter = TypeVar("Parameter")
Generator = TypeVar("Generator")


class SolveDeadlineMissed(RuntimeError):
    """Raised when the next command was not ready by its command boundary."""


@dataclass(frozen=True)
class SolveRequest(Generic[Parameter, Generator]):
    """Immutable input captured when a solve is launched."""

    predicted_state: np.ndarray
    beta: Parameter
    generator: Generator
    launched_at: float


@dataclass(frozen=True)
class BoundaryResult(Generic[Parameter, Generator]):
    """Information produced at a 30 ms command boundary."""

    command: np.ndarray
    prediction: np.ndarray
    beta: Parameter
    generator: Generator
    solve_seconds: float


class OneStepAheadPipeline(Generic[Parameter, Generator]):
    """Run exactly one RTI solve ahead of the command currently being held.

    ``solve`` must return the first *high-level* MPC action.  It is called by
    one worker only, which is important because the MPC wrapper owns mutable
    warm-start state.  ``predict`` and ``estimate`` run at the command
    boundary in the caller thread.
    """

    def __init__(
        self,
        *,
        predict: Callable[[np.ndarray, np.ndarray, Parameter], np.ndarray],
        solve: Callable[[SolveRequest[Parameter, Generator]], np.ndarray],
        estimate: Callable[[np.ndarray, np.ndarray, np.ndarray, Parameter, Generator], tuple[Parameter, Generator]],
    ) -> None:
        self._predict = predict
        self._solve = solve
        self._estimate = estimate
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crazyflie-rti")
        self._future: Future[np.ndarray] | None = None
        self._request: SolveRequest[Parameter, Generator] | None = None
        self._state: np.ndarray | None = None
        self._held_command: np.ndarray | None = None
        self._beta: Parameter | None = None
        self._generator: Generator | None = None

    @property
    def held_command(self) -> np.ndarray:
        if self._held_command is None:
            raise RuntimeError("pipeline has not been started")
        return self._held_command.copy()

    def start(
        self,
        measured_state: np.ndarray,
        *,
        bootstrap_command: np.ndarray,
        beta: Parameter,
        generator: Generator,
    ) -> np.ndarray:
        """Begin holding a known-safe command and solve for the next boundary.

        Call this only after the high-level handoff has settled.  The returned
        bootstrap command is the command that must be published immediately.
        """
        if self._future is not None:
            raise RuntimeError("pipeline is already running")
        self._state = np.asarray(measured_state, dtype=float).copy()
        self._held_command = np.asarray(bootstrap_command, dtype=float).copy()
        self._beta, self._generator = beta, generator
        self._launch_next(self._state, self._held_command, beta, generator)
        return self.held_command

    def advance(self, measured_state: np.ndarray, *, timeout_s: float = 0.0) -> BoundaryResult[Parameter, Generator]:
        """Consume the finished solve, update beta, and launch the next solve.

        ``timeout_s`` should normally be zero: the solve deadline is the
        boundary itself.  A nonzero value is useful only for a controlled
        startup test.  On a missed deadline the caller must retain/replace the
        command with its configured safety fallback; silently using a late
        command would destroy the timing contract.
        """
        if self._future is None or self._request is None or self._state is None or self._held_command is None:
            raise RuntimeError("pipeline has not been started")
        try:
            next_command = np.asarray(self._future.result(timeout=timeout_s), dtype=float)
        except TimeoutError as error:
            raise SolveDeadlineMissed("RTI solve missed the command boundary") from error
        if not np.all(np.isfinite(next_command)):
            raise FloatingPointError("RTI returned a non-finite command")

        actual = np.asarray(measured_state, dtype=float).copy()
        beta_next, generator_next = self._estimate(
            self._state, actual, self._held_command, self._beta, self._generator
        )
        solve_seconds = monotonic() - self._request.launched_at

        # The just-completed command was solved using the previous beta and a
        # predicted state.  Only the *following* solve sees this beta update.
        self._state = actual
        self._held_command = next_command.copy()
        self._beta, self._generator = beta_next, generator_next
        prediction = self._predict(actual, next_command, beta_next)
        self._launch_request(prediction, beta_next, generator_next)
        return BoundaryResult(next_command, prediction, beta_next, generator_next, solve_seconds)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _launch_next(self, state: np.ndarray, command: np.ndarray, beta: Parameter, generator: Generator) -> None:
        self._launch_request(self._predict(state, command, beta), beta, generator)

    def _launch_request(self, prediction: np.ndarray, beta: Parameter, generator: Generator) -> None:
        request = SolveRequest(np.asarray(prediction, dtype=float).copy(), beta, generator, monotonic())
        self._request = request
        self._future = self._executor.submit(self._solve, request)
