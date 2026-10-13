from __future__ import annotations

from pathlib import Path

import pytest

from rutts.narrator import create_profile, load_profile
from rutts.validation import ValidationError
from tests.conftest import write_reference_wav

TRANSCRIPT = "Это тестовая фраза для опорной записи."


def _make_profile(tmp_path: Path, audio: Path) -> Path:
    directory = tmp_path / "narrator"
    create_profile(
        directory,
        name="Test narrator",
        audio=audio,
        transcript=TRANSCRIPT,
        provenance="Recorded by the test suite",
    )
    return directory


def test_profile_round_trip_keeps_reference_and_metadata(tmp_path: Path, reference_wav: Path) -> None:
    directory = _make_profile(tmp_path, reference_wav)
    profile = load_profile(directory)
    assert profile.name == "Test narrator"
    assert profile.language == "ru"
    assert profile.provenance == "Recorded by the test suite"
    assert profile.transcript == TRANSCRIPT
    assert profile.reference_audio.read_bytes() == reference_wav.read_bytes()


def test_profile_records_reference_hash_of_original_audio(tmp_path: Path, reference_wav: Path) -> None:
    import hashlib

    directory = _make_profile(tmp_path, reference_wav)
    profile = load_profile(directory)
    assert profile.reference_sha256 == hashlib.sha256(reference_wav.read_bytes()).hexdigest()


def test_tampered_reference_audio_fails_clearly(tmp_path: Path, reference_wav: Path) -> None:
    directory = _make_profile(tmp_path, reference_wav)
    profile = load_profile(directory)
    profile.reference_audio.write_bytes(profile.reference_audio.read_bytes() + b"\x00")
    with pytest.raises(ValidationError) as excinfo:
        load_profile(directory)
    assert "reference audio changed since the profile was created" in excinfo.value.messages[0]


def test_tampered_transcript_fails_clearly(tmp_path: Path, reference_wav: Path) -> None:
    directory = _make_profile(tmp_path, reference_wav)
    (directory / "transcript.txt").write_text("Совсем другой текст.", encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        load_profile(directory)
    assert "transcript changed" in excinfo.value.messages[0]


def test_creating_profile_rejects_invalid_inputs(tmp_path: Path) -> None:
    silent = write_reference_wav(tmp_path / "silent.wav", seconds=4.0, silent=True)
    with pytest.raises(ValidationError) as excinfo:
        create_profile(tmp_path / "p", name="", audio=silent, transcript="", provenance=" ")
    messages = excinfo.value.messages
    assert "narrator name is empty" in messages
    assert "transcript is empty" in messages
    assert any("silent" in message for message in messages)
    assert any("provenance" in message for message in messages)


def test_loading_missing_profile_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_profile(tmp_path)
