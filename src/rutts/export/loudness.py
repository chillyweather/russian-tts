from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

METHOD = "gated_rms_dbfs"
BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
ABSOLUTE_GATE_DBFS = -70.0
RELATIVE_GATE_DB = -10.0
FLOOR_DBFS = -120.0
PEAK_CEILING_DBFS = -1.0


@dataclass(frozen=True)
class Measurement:
    gated_rms_dbfs: float
    sample_peak_dbfs: float


@dataclass(frozen=True)
class Gain:
    chapter_id: str
    measurement: Measurement
    corrective: bool
    requested_db: float
    applied_db: float

    @property
    def reduced_for_peak(self) -> bool:
        return self.applied_db < self.requested_db

    @property
    def reduction_db(self) -> float:
        return self.requested_db - self.applied_db if self.reduced_for_peak else 0.0


def _power_db(power: float) -> float:
    return max(10.0 * math.log10(power), FLOOR_DBFS) if power > 0 else FLOOR_DBFS


def _amplitude_db(amplitude: float) -> float:
    return max(20.0 * math.log10(amplitude), FLOOR_DBFS) if amplitude > 0 else FLOOR_DBFS


def measure(samples: np.ndarray, rate: int) -> Measurement:
    """Gated mean-square level in dBFS (BS.1770-style gating without K-weighting; not LUFS)."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    peak = _amplitude_db(float(np.max(np.abs(x)))) if x.size else FLOOR_DBFS
    if x.size == 0:
        return Measurement(FLOOR_DBFS, peak)
    squared = x * x
    block = max(1, round(BLOCK_SECONDS * rate))
    step = max(1, round(STEP_SECONDS * rate))
    if squared.size < block:
        powers = np.array([float(np.mean(squared))])
    else:
        powers = np.lib.stride_tricks.sliding_window_view(squared, block)[::step].mean(axis=1)
    levels = np.array([_power_db(float(power)) for power in powers])
    audible = levels > ABSOLUTE_GATE_DBFS
    if not audible.any():
        return Measurement(FLOOR_DBFS, peak)
    relative = _power_db(float(powers[audible].mean())) + RELATIVE_GATE_DB
    gated = audible & (levels > relative)
    return Measurement(_power_db(float(powers[gated].mean())), peak)


def plan(measurements: dict[str, Measurement], threshold_db: float, target_dbfs: float) -> tuple[float, list[Gain]]:
    """One book-wide gain; a chapter further than the threshold from the book median gets its own corrective gain."""
    median = statistics.median(item.gated_rms_dbfs for item in measurements.values())
    common = target_dbfs - median
    gains: list[Gain] = []
    for chapter_id, measurement in measurements.items():
        corrective = abs(measurement.gated_rms_dbfs - median) > threshold_db
        requested = target_dbfs - measurement.gated_rms_dbfs if corrective else common
        ceiling = PEAK_CEILING_DBFS - measurement.sample_peak_dbfs
        gains.append(Gain(chapter_id, measurement, corrective, requested, min(requested, ceiling)))
    return median, gains


def apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    """Pure gain: one multiplication per sample, no compression, limiting, trimming or stretching."""
    factor = np.float32(10.0 ** (gain_db / 20.0))
    return np.asarray(samples, dtype=np.float32) * factor


def sample_peak_dbfs(samples: np.ndarray) -> float:
    return _amplitude_db(float(np.max(np.abs(samples)))) if samples.size else FLOOR_DBFS
