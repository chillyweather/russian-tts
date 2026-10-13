from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rutts.cli import main
from rutts.narrator import create_profile
from tests.conftest import FakeEngine, write_reference_wav

TRANSCRIPT = "Это тестовая фраза для опорной записи."


@pytest.fixture
def profile_dir(tmp_path: Path, reference_wav: Path) -> Path:
    directory = tmp_path / "narrator"
    create_profile(
        directory,
        name="Fixture narrator",
        audio=reference_wav,
        transcript=TRANSCRIPT,
        provenance="Test recording, public domain",
    )
    return directory


def _create_narrator(tmp_path: Path, audio: Path, transcript: Path, out: Path) -> int:
    return main(
        [
            "narrator",
            "create",
            "--name",
            "CLI narrator",
            "--audio",
            str(audio),
            "--transcript-file",
            str(transcript),
            "--provenance",
            "Recorded by the test suite",
            "--out",
            str(out),
        ]
    )


def test_narrator_create_writes_profile(tmp_path: Path, reference_wav: Path, transcript_file: Path) -> None:
    out = tmp_path / "profile"
    assert _create_narrator(tmp_path, reference_wav, transcript_file, out) == 0
    document = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert document["name"] == "CLI narrator"
    assert document["provenance"] == "Recorded by the test suite"


def test_narrator_create_with_invalid_transcript_exits_2(
    tmp_path: Path, reference_wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    assert _create_narrator(tmp_path, reference_wav, empty, tmp_path / "p") == 2
    assert "error: transcript is empty" in capsys.readouterr().err


def test_narrator_create_with_missing_audio_exits_2(
    tmp_path: Path, transcript_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.wav"
    assert _create_narrator(tmp_path, missing, transcript_file, tmp_path / "p") == 2
    assert f"error: reference audio not found: {missing}" in capsys.readouterr().err


def test_narrator_create_with_too_short_audio_exits_2(
    tmp_path: Path, transcript_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = write_reference_wav(tmp_path / "short.wav", seconds=1.5)
    assert _create_narrator(tmp_path, short, transcript_file, tmp_path / "p") == 2
    assert "must be at least 3 s" in capsys.readouterr().err


def test_preview_rejects_english_text_exits_2(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "preview", "--engine", "fake", "--narrator", str(profile_dir),
            "--out", str(tmp_path / "out"), "--text", "Hello, this is English.",
        ]
    )
    assert code == 2
    assert "must be predominantly Russian" in capsys.readouterr().err
    assert fake_engine_factory == []


def test_preview_rejects_empty_text_exits_2(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(tmp_path / "o"), "--text", "  "]
    )
    assert code == 2
    assert "text is empty" in capsys.readouterr().err


def test_preview_rejects_tampered_narrator_exits_2(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    (profile_dir / "reference.wav").write_bytes(b"changed")
    code = main(
        ["preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(tmp_path / "o"), "--text", "Привет, мир."]
    )
    assert code == 2
    assert "changed since the profile was created" in capsys.readouterr().err


def test_successful_preview_writes_wav_and_results(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine]
) -> None:
    out = tmp_path / "preview"
    code = main(
        [
            "preview", "--engine", "fake", "--precision", "8bit", "--narrator", str(profile_dir),
            "--out", str(out), "--text", "Привет, мир. Это проверка.",
        ]
    )
    assert code == 0

    wav = out / "text" / "attempt-1.wav"
    info = sf.info(str(wav))
    assert info.subtype == "FLOAT"
    assert info.samplerate == 24000
    assert info.channels == 1

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    engine = results["engine"]
    assert engine["config"]["engine_id"] == "fake"
    assert engine["config"]["precision"] == "8bit"
    assert engine["config"]["license"] == "MIT"
    assert len(engine["fingerprint"]) == 64
    assert results["narrator"]["provenance"] == "Test recording, public domain"
    assert len(results["narrator"]["reference_sha256"]) == 64
    excerpt = results["excerpts"][0]
    assert excerpt["text"] == "Привет, мир. Это проверка."
    assert excerpt["status"] == "ok"
    attempt = excerpt["attempts"][0]
    assert attempt["status"] == "ok"
    assert attempt["file"] == "text/attempt-1.wav"
    assert attempt["audio_duration_seconds"] == pytest.approx(0.5)
    assert attempt["render_seconds"] >= 0
    assert "peak_rss_bytes" in attempt["resources"]
    assert (out / "results.md").read_text(encoding="utf-8").startswith("# Preview results")


def test_first_attempt_is_preserved_when_retry_succeeds(
    profile_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rutts.engines import base

    created: list[FakeEngine] = []

    def factory(**options: object) -> FakeEngine:
        engine = FakeEngine(silent_on=frozenset({1}))
        created.append(engine)
        return engine

    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", factory)
    out = tmp_path / "retry"
    code = main(
        [
            "preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(out),
            "--text", "Это проверка повтора.", "--retries", "2",
        ]
    )
    assert code == 0
    first = out / "text" / "attempt-1.wav"
    second = out / "text" / "attempt-2.wav"
    assert first.exists() and second.exists()
    assert np.max(np.abs(sf.read(str(first))[0])) == 0.0

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    attempts = results["excerpts"][0]["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "ok"]
    assert attempts[0]["error"] == "engine returned silent audio"
    assert attempts[1]["file"] == "text/attempt-2.wav"


def test_failure_is_recorded_after_retry_budget(
    profile_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rutts.engines import base

    def factory(**options: object) -> FakeEngine:
        return FakeEngine(raise_on=frozenset({1, 2}))

    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", factory)
    out = tmp_path / "failed"
    code = main(
        [
            "preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(out),
            "--text", "Это проверка ошибки.", "--retries", "1",
        ]
    )
    assert code == 1
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    excerpt = results["excerpts"][0]
    assert excerpt["status"] == "failed"
    assert [a["error"] for a in excerpt["attempts"]] == [
        "RuntimeError: fake failure on call 1",
        "RuntimeError: fake failure on call 2",
    ]
    assert all(a["file"] is None for a in excerpt["attempts"])
    assert results["summary"] == {"excerpts": 1, "ok": 0, "failed": 1}


def test_preview_refuses_to_overwrite_existing_attempts(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine]
) -> None:
    out = tmp_path / "again"
    args = [
        "preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(out),
        "--text", "Привет, мир.",
    ]
    assert main(args) == 0
    assert main(args) == 2


def test_suite_preview_renders_every_excerpt(
    profile_dir: Path, tmp_path: Path, fake_engine_factory: list[FakeEngine]
) -> None:
    out = tmp_path / "suite"
    assert main(["preview", "--engine", "fake", "--narrator", str(profile_dir), "--out", str(out), "--suite"]) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["summary"]["excerpts"] == 16
    assert (out / "prose_morning" / "attempt-1.wav").exists()
