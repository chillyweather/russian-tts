from __future__ import annotations

import re
import resource
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

UNAVAILABLE = "unavailable"
_SWAP_USED = re.compile(r"used\s*=\s*([\d.]+)([MG])")
_FREE_PERCENT = re.compile(r"System-wide memory free percentage:\s*(\d+)%")


@dataclass
class ResourceSample:
    elapsed_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _mlx_peak_bytes() -> int | str:
    try:
        import mlx.core as mx
    except ImportError:
        return UNAVAILABLE
    return int(mx.get_peak_memory())


def _reset_mlx_peak() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is not None:
        reset()


def _swap_used_bytes() -> int | str:
    output = _run(["sysctl", "vm.swapusage"])
    match = _SWAP_USED.search(output or "")
    if match is None:
        return UNAVAILABLE
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * scale)


def _memory_free_percent() -> int | str:
    output = _run(["memory_pressure"])
    match = _FREE_PERCENT.search(output or "")
    return int(match.group(1)) if match else UNAVAILABLE


def _snapshot() -> dict[str, Any]:
    return {
        "peak_rss_bytes": _peak_rss_bytes(),
        "mlx_peak_memory_bytes": _mlx_peak_bytes(),
        "swap_used_bytes": _swap_used_bytes(),
        "memory_free_percent": _memory_free_percent(),
    }


def sample() -> dict[str, Any]:
    """Current process and machine measurements; any metric that cannot be read is UNAVAILABLE."""
    return _snapshot()


@contextmanager
def measure() -> Iterator[ResourceSample]:
    """Measure one render. Metrics are sampled when the block exits."""
    sample = ResourceSample()
    _reset_mlx_peak()
    start = time.perf_counter()
    try:
        yield sample
    finally:
        sample.elapsed_seconds = time.perf_counter() - start
        sample.metrics = _snapshot()
