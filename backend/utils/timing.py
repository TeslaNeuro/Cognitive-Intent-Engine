"""Small timing helpers used across pipelines."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class RateLimiter:
    """Sleep just enough to keep a loop at a target rate."""

    def __init__(self, hz: float):
        self.period = 1.0 / max(hz, 1e-6)
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self.period
        now = time.perf_counter()
        delay = self._next - now
        if delay > 0:
            time.sleep(delay)
        else:
            # We're behind — resync so we don't snowball.
            self._next = now


class Stopwatch:
    """Cumulative timer with millisecond resolution."""

    def __init__(self) -> None:
        self._total = 0.0
        self._n = 0

    @contextmanager
    def lap(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._total += time.perf_counter() - t0
            self._n += 1

    @property
    def mean_ms(self) -> float:
        return (self._total / self._n) * 1000.0 if self._n else 0.0
