from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rutts.engines import base
from rutts.engines.base import EngineConfig, SynthesisResult
from rutts.narrator import NarratorProfile

FAKE_SAMPLE_RATE = 24000


class FakeEngine:
    """Deterministic engine: returns a short sine wave, optionally failing or interrupting per call.

    ``raise_on`` and ``fail_texts`` raise ``ValueError`` (a recoverable chunk failure);
    ``runtime_error_on`` raises ``RuntimeError`` (a failed attempt); ``memory_error_on`` raises a
    ``RuntimeError`` reporting memory exhaustion (fatal for the run); ``interrupt_on`` raises
    ``KeyboardInterrupt``; ``fail_attempts`` fails the first K attempts for a text;
    ``silent_texts`` returns silent audio for a text on every attempt.
    """

    def __init__(
        self,
        precision: str = "bf16",
        *,
        raise_on: frozenset[int] = frozenset(),
        runtime_error_on: frozenset[int] = frozenset(),
        memory_error_on: frozenset[int] = frozenset(),
        interrupt_on: frozenset[int] = frozenset(),
        silent_on: frozenset[int] = frozenset(),
        silent_texts: frozenset[str] = frozenset(),
        fail_load: bool = False,
        fail_texts: frozenset[str] = frozenset(),
        fail_attempts: dict[str, int] | None = None,
        amplitude: float = 0.2,
    ) -> None:
        self.precision = precision
        self.amplitude = amplitude
        self.raise_on = raise_on
        self.runtime_error_on = runtime_error_on
        self.memory_error_on = memory_error_on
        self.interrupt_on = interrupt_on
        self.silent_on = silent_on
        self.silent_texts = silent_texts
        self.fail_load = fail_load
        self.fail_texts = fail_texts
        self.fail_attempts = fail_attempts or {}
        self.received: list[str] = []
        self.calls = 0
        self.loads = 0
        self.loaded = False
        self.unloads = 0

    @property
    def config(self) -> EngineConfig:
        return EngineConfig(
            engine_id="fake",
            model_repo="fake/model",
            model_revision="0" * 40,
            precision=self.precision,
            runtime_name="fake-runtime",
            runtime_version="1.0",
            license="MIT",
            settings={"tone_hz": 440},
        )

    def load(self) -> None:
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("fake failure while loading")
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1

    def synthesize(self, text: str, narrator: NarratorProfile) -> SynthesisResult:
        self.calls += 1
        self.received.append(text)
        if self.calls in self.interrupt_on:
            raise KeyboardInterrupt
        if self.calls in self.runtime_error_on:
            raise RuntimeError(f"engine returned no audio segments on call {self.calls}")
        if self.calls in self.memory_error_on:
            raise RuntimeError(f"Metal: out of memory on call {self.calls}")
        attempts = self.received.count(text)
        if text in self.fail_texts or self.calls in self.raise_on:
            raise ValueError(f"fake failure on call {self.calls}")
        if attempts <= self.fail_attempts.get(text, 0):
            raise ValueError(f"fake transient failure on attempt {attempts}")
        duration = 0.5
        t = np.arange(int(duration * FAKE_SAMPLE_RATE)) / FAKE_SAMPLE_RATE
        samples = self.amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        if self.calls in self.silent_on or text in self.silent_texts:
            samples = np.zeros_like(samples)
        return SynthesisResult(samples=samples, sample_rate=FAKE_SAMPLE_RATE)


@pytest.fixture
def fake_engine_factory(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    """Registers ``fake`` as an engine id; returns the list of engines created."""
    created: list[FakeEngine] = []

    def factory(**options: object) -> FakeEngine:
        engine = FakeEngine(**options)  # type: ignore[arg-type]
        created.append(engine)
        return engine

    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", factory)
    return created


def write_reference_wav(path: Path, seconds: float = 5.0, silent: bool = False) -> Path:
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    samples = np.zeros_like(t) if silent else 0.3 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(path), samples.astype(np.float32), rate, subtype="PCM_16")
    return path


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    return write_reference_wav(tmp_path / "source.wav")


@pytest.fixture
def transcript_file(tmp_path: Path) -> Path:
    path = tmp_path / "transcript.txt"
    path.write_text("Это тестовая фраза для опорной записи.\n", encoding="utf-8")
    return path
