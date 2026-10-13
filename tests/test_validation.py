from __future__ import annotations

from pathlib import Path

import pytest

from rutts.validation import (
    ValidationError,
    reference_audio_problems,
    text_problems,
    transcript_problems,
)
from tests.conftest import write_reference_wav


def test_russian_text_is_accepted() -> None:
    assert text_problems("Это обычный русский текст.") == []


def test_english_text_is_rejected_with_reason() -> None:
    problems = text_problems("This is an English sentence.")
    assert len(problems) == 1
    assert "Russian" in problems[0]


def test_empty_text_is_rejected() -> None:
    assert text_problems("   \n") == ["text is empty"]


def test_transcript_must_be_cyrillic_and_non_empty() -> None:
    assert transcript_problems("") == ["transcript is empty"]
    assert transcript_problems("Hello there") != []
    assert transcript_problems("Привет, мир.") == []


def test_missing_reference_audio_is_reported(tmp_path: Path) -> None:
    problems = reference_audio_problems(tmp_path / "missing.wav")
    assert problems == [f"reference audio not found: {tmp_path / 'missing.wav'}"]


def test_too_short_reference_audio_is_reported(tmp_path: Path) -> None:
    path = write_reference_wav(tmp_path / "short.wav", seconds=1.0)
    problems = reference_audio_problems(path)
    assert len(problems) == 1
    assert "1.00 s" in problems[0]
    assert "at least 3 s" in problems[0]


def test_silent_reference_audio_is_reported(tmp_path: Path) -> None:
    path = write_reference_wav(tmp_path / "silent.wav", seconds=4.0, silent=True)
    assert reference_audio_problems(path) == [f"reference audio is silent: {path}"]


def test_undecodable_reference_audio_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio at all")
    problems = reference_audio_problems(path)
    assert len(problems) == 1
    assert "cannot be decoded" in problems[0]


def test_valid_reference_audio_has_no_problems(tmp_path: Path) -> None:
    path = write_reference_wav(tmp_path / "ok.wav", seconds=5.0)
    assert reference_audio_problems(path) == []


def test_validation_error_carries_messages() -> None:
    error = ValidationError(["first", "second"])
    assert error.messages == ["first", "second"]
    with pytest.raises(ValidationError):
        raise error
