from __future__ import annotations

import numpy as np
import pytest

from rutts.verification.audiochecks import clipping, suspicious_duration, suspicious_silence

RATE = 16000


def tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_internal_silence_longer_than_threshold_is_flagged_with_time() -> None:
    samples = np.concatenate([tone(1.0), np.zeros(int(2.5 * RATE), dtype=np.float32), tone(1.0)])
    flags = suspicious_silence(samples, RATE)
    assert [(f.kind, round(f.start, 1), round(f.end, 1)) for f in flags] == [("internal_silence", 1.0, 3.5)]
    assert round(flags[0].evidence["seconds"], 1) == 2.5


def test_short_internal_pause_is_not_flagged() -> None:
    samples = np.concatenate([tone(1.0), np.zeros(int(1.5 * RATE), dtype=np.float32), tone(1.0)])
    assert suspicious_silence(samples, RATE) == []


def test_long_leading_and_trailing_silence_are_flagged() -> None:
    samples = np.concatenate([np.zeros(int(2.0 * RATE), dtype=np.float32), tone(1.0), np.zeros(int(1.8 * RATE), dtype=np.float32)])
    kinds = [flag.kind for flag in suspicious_silence(samples, RATE)]
    assert kinds == ["leading_silence", "trailing_silence"]


def test_short_edges_are_not_flagged() -> None:
    samples = np.concatenate([np.zeros(int(1.0 * RATE), dtype=np.float32), tone(1.0), np.zeros(int(1.0 * RATE), dtype=np.float32)])
    assert suspicious_silence(samples, RATE) == []


def test_duration_outside_plausible_rate_is_flagged() -> None:
    flag = suspicious_duration(duration_seconds=2.0, characters=100)
    assert flag is not None
    assert flag.kind == "suspicious_duration"
    assert flag.evidence["chars_per_second"] == 50.0


def test_duration_inside_plausible_rate_is_not_flagged() -> None:
    assert suspicious_duration(duration_seconds=10.0, characters=120) is None


def test_clipping_reports_samples_and_time_span() -> None:
    samples = tone(1.0, amplitude=0.2)
    samples[8000:8010] = 1.0
    flag = clipping(samples, RATE)
    assert flag is not None
    assert flag.kind == "clipping"
    assert flag.evidence["samples"] == 10
    assert round(flag.start, 3) == 0.5
    assert flag.end == pytest.approx(0.5 + 10 / RATE)


def test_clean_audio_has_no_clipping() -> None:
    assert clipping(tone(1.0), RATE) is None
