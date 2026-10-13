"""Signal-level checks that need no model. Each flag carries its time location and measured values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRAME_SECONDS = 0.02
SILENT_RMS = 0.01
INTERNAL_SILENCE_SECONDS = 2.0
EDGE_SILENCE_SECONDS = 1.5
MIN_CHARS_PER_SECOND = 8.0
MAX_CHARS_PER_SECOND = 25.0
CLIP_LEVEL = 0.999


@dataclass(frozen=True)
class AudioFlag:
    kind: str
    severity: str
    start: float
    end: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "severity": self.severity,
            "location": {"audio_seconds": [round(self.start, 3), round(self.end, 3)]},
            "evidence": self.evidence,
        }


def _silent_runs(samples: np.ndarray, rate: int) -> list[tuple[int, int]]:
    """(first frame, end frame) of every run of quiet frames."""
    frame = max(1, int(rate * FRAME_SECONDS))
    count = samples.size // frame
    if count == 0:
        return []
    blocks = samples[: count * frame].reshape(count, frame).astype(np.float64)
    quiet = np.sqrt(np.mean(blocks * blocks, axis=1)) < SILENT_RMS
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(quiet):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, count))
    return runs


def suspicious_silence(
    samples: np.ndarray,
    rate: int,
    *,
    internal_seconds: float = INTERNAL_SILENCE_SECONDS,
    edge_seconds: float = EDGE_SILENCE_SECONDS,
) -> list[AudioFlag]:
    total = samples.size / rate
    flags: list[AudioFlag] = []
    for first, last in _silent_runs(samples, rate):
        start, end = first * FRAME_SECONDS, min(last * FRAME_SECONDS, total)
        length = end - start
        if first == 0:
            kind, limit = "leading_silence", edge_seconds
        elif end >= total - 1e-9:
            kind, limit = "trailing_silence", edge_seconds
        else:
            kind, limit = "internal_silence", internal_seconds
        if length > limit:
            flags.append(AudioFlag(kind, "minor", start, end, {"seconds": round(length, 3), "threshold_seconds": limit}))
    return flags


def suspicious_duration(
    duration_seconds: float,
    characters: int,
    *,
    min_rate: float = MIN_CHARS_PER_SECOND,
    max_rate: float = MAX_CHARS_PER_SECOND,
) -> AudioFlag | None:
    rate = characters / duration_seconds if duration_seconds > 0 else float("inf")
    if min_rate <= rate <= max_rate:
        return None
    return AudioFlag(
        "suspicious_duration",
        "minor",
        0.0,
        duration_seconds,
        {
            "duration_seconds": round(duration_seconds, 3),
            "characters": characters,
            "chars_per_second": round(rate, 2) if np.isfinite(rate) else None,
            "plausible_chars_per_second": [min_rate, max_rate],
        },
    )


def clipping(samples: np.ndarray, rate: int, *, level: float = CLIP_LEVEL) -> AudioFlag | None:
    hits = np.flatnonzero(np.abs(samples) >= level)
    if hits.size == 0:
        return None
    first, last = int(hits[0]), int(hits[-1])
    return AudioFlag(
        "clipping",
        "major",
        first / rate,
        (last + 1) / rate,
        {"samples": int(hits.size), "peak": round(float(np.max(np.abs(samples))), 4), "level": level},
    )
