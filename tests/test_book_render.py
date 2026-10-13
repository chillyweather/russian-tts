from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from rutts.book.chunking import ChunkLimits, plan_chunks
from rutts.book.project import BookProject, install_narrator, load_project
from rutts.book.render import render_chapter
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.narrator import create_profile
from rutts.validation import ValidationError
from tests.conftest import FakeEngine, write_reference_wav

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"
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


@pytest.fixture
def book(tmp_path: Path, profile_dir: Path) -> BookProject:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title="Книга", author="", chapter_regex=None)
    project = load_project(out)
    install_narrator(project, profile_dir)
    return project


def _manifest(project: BookProject, chapter_id: str, target: str) -> Any:
    path = project.target_dir(chapter_id, target) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_engine_receives_chunks_in_source_order(book: BookProject) -> None:
    engine = FakeEngine()
    render_chapter(book, "c001", engine, "short", limits=ChunkLimits(), force_new_run=False)
    plan = json.loads(
        (book.target_dir("c001", "short") / "plan.json").read_text(encoding="utf-8")
    )
    assert engine.received == [chunk["text"] for chunk in plan["chunks"]]
    assert engine.received[0].startswith("Утро началось рано.")
    assert all("* * *" not in text for text in engine.received)
    assert engine.unloads == 1
    assert engine.loaded is False


def test_manifest_records_fingerprints_and_measured_audio(book: BookProject) -> None:
    engine = FakeEngine()
    render_chapter(book, "c001", engine, "short", limits=ChunkLimits(), force_new_run=False)
    manifest = _manifest(book, "c001", "short")
    narrator_sha = hashlib.sha256((book.narrator_dir / "profile.json").read_bytes()).hexdigest()

    assert manifest["engine_fingerprint"] == engine.config.fingerprint()
    assert manifest["narrator_sha256"] == narrator_sha
    assert manifest["engine"]["engine_id"] == "fake"
    assert manifest["narrator"]["name"] == "Fixture narrator"

    chunks = manifest["chunks"]
    assert chunks
    for record in chunks:
        assert record["status"] == "ok"
        assert [(a["attempt"], a["status"], a["error"]) for a in record["attempts"]] == [(1, "ok", None)]
        audio = book.target_dir("c001", "short") / record["audio_file"]
        assert audio.read_bytes()[:4] == b"RIFF"
        assert hashlib.sha256(audio.read_bytes()).hexdigest() == record["audio_sha256"]
        assert record["sample_rate"] == 24000
        assert record["actual_duration_seconds"] == 0.5
        assert record["render_seconds"] is not None
        assert record["engine_fingerprint"] == engine.config.fingerprint()
        assert record["narrator_sha256"] == narrator_sha


def test_saved_chunk_audio_is_lossless(book: BookProject) -> None:
    engine = FakeEngine()
    render_chapter(book, "c001", engine, "short", limits=ChunkLimits(), force_new_run=False)
    first = _manifest(book, "c001", "short")["chunks"][0]
    samples, rate = sf.read(
        str(book.target_dir("c001", "short") / first["audio_file"]), dtype="float32"
    )
    expected = FakeEngine().synthesize("проверка", book.narrator()).samples
    assert rate == 24000
    assert np.array_equal(samples, expected)


def test_failed_chunk_is_recorded_and_others_continue(book: BookProject) -> None:
    plan_texts = [c.text for c in plan_chunks(book.planning_segments("c001"), "short", ChunkLimits())]
    failing = plan_texts[0]
    engine = FakeEngine(fail_texts=frozenset({failing}))
    manifest = render_chapter(book, "c001", engine, "short", limits=ChunkLimits(), force_new_run=False)
    records = manifest["chunks"]
    assert [r["status"] for r in records] == ["failed", "ok"]
    failed = records[0]
    assert failed["audio_file"] is None
    assert "fake failure" in failed["attempts"][0]["error"]
    assert not (book.target_dir("c001", "short") / f"{failed['id']}.wav").exists()
    assert engine.unloads == 1


def test_changed_engine_config_is_refused_without_force(book: BookProject) -> None:
    render_chapter(book, "c001", FakeEngine(precision="bf16"), "short", limits=ChunkLimits(), force_new_run=False)
    with pytest.raises(ValidationError, match="--accept-config-change"):
        render_chapter(
            book, "c001", FakeEngine(precision="fp16"), "short", limits=ChunkLimits(), force_new_run=False
        )


def test_force_new_run_archives_previous_chunks(book: BookProject) -> None:
    render_chapter(book, "c001", FakeEngine(precision="bf16"), "short", limits=ChunkLimits(), force_new_run=False)
    render_chapter(book, "c001", FakeEngine(precision="fp16"), "short", limits=ChunkLimits(), force_new_run=True)
    archive = book.directory / "chunks" / "c001.prev-1"
    assert (archive / "short" / "manifest.json").is_file()
    current = _manifest(book, "c001", "short")
    assert current["engine"]["precision"] == "fp16"


def test_changed_narrator_is_refused_without_force(
    book: BookProject, tmp_path: Path, reference_wav: Path
) -> None:
    render_chapter(book, "c001", FakeEngine(), "short", limits=ChunkLimits(), force_new_run=False)
    other = create_profile(
        tmp_path / "other-narrator",
        name="Other",
        audio=write_reference_wav(tmp_path / "other.wav", seconds=4.0),
        transcript="Другая опорная фраза для записи.",
        provenance="Second recording",
    )
    install_narrator(book, other.directory)
    with pytest.raises(ValidationError, match="narrator"):
        render_chapter(book, "c001", FakeEngine(), "short", limits=ChunkLimits(), force_new_run=False)


def test_render_requires_a_narrator(tmp_path: Path) -> None:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title=None, author="", chapter_regex=None)
    with pytest.raises(ValidationError, match="no narrator profile"):
        render_chapter(load_project(out), "c001", FakeEngine(), "short", limits=ChunkLimits(), force_new_run=False)


def test_cli_book_narrator_and_render(
    book: BookProject, profile_dir: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["book", "narrator", str(book.directory), "--profile", str(profile_dir)]) == 0
    code = main(
        ["book", "render", str(book.directory), "--chapter", "c001", "--engine", "fake", "--target", "short"]
    )
    assert code == 0
    assert "rendered c001" in capsys.readouterr().out
    assert len(fake_engine_factory) == 1


def test_cli_render_with_failed_chunk_exits_1(
    book: BookProject, profile_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rutts.engines import base

    main(["book", "narrator", str(book.directory), "--profile", str(profile_dir)])
    monkeypatch.setitem(
        base.ENGINE_FACTORIES,
        "fake",
        lambda **options: FakeEngine(raise_on=frozenset({1, 2, 3}), **options),
    )
    code = main(["book", "render", str(book.directory), "--chapter", "c001", "--engine", "fake", "--retries", "0"])
    assert code == 1
