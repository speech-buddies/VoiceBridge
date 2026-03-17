"""
utils/timing.py

Lightweight timing helpers. Drop into any class that already has a logger.

"""

import time
import functools
from typing import Optional


class Timer:
    """Context manager and manual timer. Reports elapsed time in milliseconds."""

    def __init__(self, label: str = ""):
        self.label = label
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        return self

    @property
    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __str__(self) -> str:
        return f"{self.label}: {self.elapsed_ms:.1f} ms"


def log_timing(logger, level: str = "debug"):
    """
    Decorator that logs how long a method takes.

        @log_timing(logger)
        def my_method(self, ...):
            ...

    Logs: "my_method took 312.4 ms"
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with Timer(fn.__name__) as t:
                result = fn(*args, **kwargs)
            getattr(logger, level)(f"{fn.__name__} took {t.elapsed_ms:.1f} ms")
            return result
        return wrapper
    return decorator