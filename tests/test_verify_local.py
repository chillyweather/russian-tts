"""Real Whisper and macOS voice fixtures. Run with: pytest -m local_model tests/test_verify_local.py -o addopts=""."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rutts.verification.asr import Word, WhisperMlxTranscriber
from rutts.verification.audiochecks import suspicious_silence
from rutts.verification.compare import compare

pytestmark = pytest.mark.local_model

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FIRST = "В тысяча восемьсот двенадцатом году армия ушла из Москвы."
SECOND = "Старый замок стоял на высоком холме."
EXPECTED = f"{FIRST} {SECOND}"
RATE = 16000


def _say(text: str, directory: Path, name: str) -> Path:
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    subprocess.run(["say", "-v", "Milena", "-o", str(aiff), text], check=True)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(aiff), "-ar", str(RATE), "-ac", "1", str(wav)], check=True)
    return wav


def _silence(directory: Path, seconds: float) -> Path:
    path = directory / f"silence-{seconds}.wav"
    sf.write(str(path), np.zeros(int(seconds * RATE), dtype=np.float32), RATE, subtype="PCM_16")
    return path


def _concat(directory: Path, name: str, parts: list[Path]) -> Path:
    listing = directory / f"{name}.txt"
    listing.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
    out = directory / f"{name}.wav"
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-ar", str(RATE), "-ac", "1", str(out)],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("local-asr")
    first = _say(FIRST, directory, "first")
    second = _say(SECOND, directory, "second")
    return {
        "clean": _concat(directory, "clean", [first, second]),
        "missing": _say(FIRST + " Старый стоял на высоком холме.", directory, "missing"),
        "repeated": _say(f"{FIRST} {SECOND} {SECOND}", directory, "repeated"),
        "silence": _concat(directory, "silence", [first, _silence(directory, 4.0), second]),
    }


@pytest.fixture(scope="module")
def whisper() -> Iterator[WhisperMlxTranscriber]:
    transcriber = WhisperMlxTranscriber()
    started = time.perf_counter()
    transcriber.load()
    print(f"\nwhisper load: {time.perf_counter() - started:.2f} s")
    yield transcriber
    transcriber.unload()


def _transcribe(whisper: WhisperMlxTranscriber, path: Path) -> list[Word]:
    started = time.perf_counter()
    transcript = whisper.transcribe(path)
    print(f"\n{path.name}: transcribe {time.perf_counter() - started:.2f} s: {transcript.text!r}")
    return transcript.words


def test_clean_reading_has_no_word_level_findings(fixtures: dict[str, Path], whisper: WhisperMlxTranscriber) -> None:
    result = compare(EXPECTED, _transcribe(whisper, fixtures["clean"]))
    print("clean findings:", [(f.type, f.expected, f.heard) for f in result.findings])
    assert [f for f in result.findings if f.type != "uncertain_number"] == []


def test_missing_word_is_reported(fixtures: dict[str, Path], whisper: WhisperMlxTranscriber) -> None:
    result = compare(EXPECTED, _transcribe(whisper, fixtures["missing"]))
    print("missing findings:", [(f.type, f.expected, f.heard) for f in result.findings])
    assert any(f.type == "omission" and "замок" in f.expected for f in result.findings)


def test_repeated_phrase_is_reported(fixtures: dict[str, Path], whisper: WhisperMlxTranscriber) -> None:
    result = compare(EXPECTED, _transcribe(whisper, fixtures["repeated"]))
    print("repeated findings:", [(f.type, f.expected, f.heard) for f in result.findings])
    assert any(f.type == "addition" or f.type == "repetition" for f in result.findings)


def test_inserted_silence_is_reported_with_its_location(fixtures: dict[str, Path]) -> None:
    samples, rate = sf.read(str(fixtures["silence"]), dtype="float32")
    flags = suspicious_silence(samples, rate)
    print("silence flags:", [(f.kind, round(f.start, 2), round(f.end, 2)) for f in flags])
    internal = [flag for flag in flags if flag.kind == "internal_silence"]
    assert len(internal) == 1
    assert internal[0].evidence["seconds"] >= 3.5
