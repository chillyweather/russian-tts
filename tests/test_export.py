from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from rutts.book.assemble import master_is_stale
from rutts.book.project import BookProject, install_narrator, load_project, read_json, sha256_file, write_json
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from tests.conftest import FakeEngine, FakeTranscriber
from tests.test_book_render import profile_dir  # noqa: F401  (shared fixture)
from tests.test_book_job import BOOK_TEXT, calm_machine  # noqa: F401  (shared fixtures)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
EXPORT = "audiobook"
CEILING = 10 ** (-1 / 20)


def _d(project: BookProject) -> str:
    return str(project.directory)


def _export(project: BookProject, *extra: str) -> int:
    return main(["book", "export", _d(project), *extra])


def _export_dir(project: BookProject) -> Path:
    return project.directory / "exports" / EXPORT


def _record(project: BookProject) -> dict[str, Any]:
    return read_json(_export_dir(project) / "export.json")


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_chapters", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    document: dict[str, Any] = json.loads(result.stdout)
    return document


def _master_samples(project: BookProject, chapter_id: str) -> tuple[np.ndarray, int]:
    report = read_json(project.directory / "chapters" / f"{chapter_id}.json")
    samples, rate = sf.read(str(project.directory / report["master_file"]), dtype="float32")
    return samples, rate


def _chunk_ids(project: BookProject, chapter_id: str) -> list[str]:
    manifest = read_json(project.target_dir(chapter_id, "short") / "manifest.json")
    return [str(record["id"]) for record in manifest["chunks"]]


def _sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def book(tmp_path: Path, profile_dir: Path) -> BookProject:  # noqa: F811
    text = BOOK_TEXT.split("Глава 4.")[0]
    source = tmp_path / "book.txt"
    source.write_text(text, encoding="utf-8")
    out = tmp_path / "book"
    import_text(source, out, title="Книга", author="Автор", chapter_regex=None)
    project = load_project(out)
    install_narrator(project, profile_dir)
    return project


@pytest.fixture
def install_engine(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeEngine], None]:
    def install(engine: FakeEngine) -> None:
        monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: engine)

    return install


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeTranscriber]:
    from rutts.verification import asr

    def install(heard: dict[str, str] | None = None) -> FakeTranscriber:
        instance = FakeTranscriber(heard or {})
        monkeypatch.setitem(asr.TRANSCRIBER_FACTORIES, "fake-asr", lambda: instance)
        return instance

    return install


@pytest.fixture
def rendered(book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]) -> BookProject:
    """Three chapters generated, assembled and verified by the scripted engine and transcriber."""
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 0
    for chapter_id in ("c001", "c002", "c003"):
        assert main(["book", "verify", _d(book), "--chapter", chapter_id, "--transcriber", "fake-asr"]) == 0
        assert main(["book", "assemble", _d(book), "--chapter", chapter_id, "--target", "short"]) == 0
    return load_project(book.directory)


def _assemble_all(project: BookProject) -> None:
    for chapter_id in ("c001", "c002", "c003"):
        assert main(["book", "assemble", _d(project), "--chapter", chapter_id, "--target", "short"]) == 0


@requires_ffmpeg
def test_reviewed_export_is_a_decodable_m4b_with_matching_chapters_and_tags(rendered: BookProject) -> None:
    assert _export(rendered) == 0
    record = _record(rendered)
    assert record["status"] == "ready"
    assert record["kind"] == "reviewed"
    output = _export_dir(rendered) / "audiobook.m4b"
    assert output.is_file()
    document = _probe(output)
    chapters = document["chapters"]
    assert [chapter["tags"]["title"] for chapter in chapters] == ["Глава 1. Утро", "Глава 2. Дорога", "Глава 3. Пристань"]

    samples = [_master_samples(rendered, chapter_id) for chapter_id in ("c001", "c002", "c003")]
    rate = samples[0][1]
    expected_starts: list[float] = []
    position = 0
    for audio, _ in samples:
        expected_starts.append(position / rate)
        position += len(audio)
    for chapter, expected in zip(chapters, expected_starts, strict=True):
        assert abs(float(chapter["start_time"]) - expected) <= 0.05
    assert document["format"]["tags"]["title"] == "Книга"
    assert document["format"]["tags"]["artist"] == "Автор"
    assert float(document["format"]["duration"]) == pytest.approx(position / rate, abs=0.25)


@requires_ffmpeg
def test_chapter_order_follows_the_source_after_a_chapter_is_excluded(rendered: BookProject) -> None:
    assert main(["book", "exclude", _d(rendered), "--chapter", "c002", "--reason", "notes"]) == 0
    assert _export(rendered) == 0
    chapters = _probe(_export_dir(rendered) / "audiobook.m4b")["chapters"]
    assert [chapter["tags"]["title"] for chapter in chapters] == ["Глава 1. Утро", "Глава 3. Пристань"]


@requires_ffmpeg
def test_loudness_is_a_single_pure_gain_that_never_clips_and_records_the_reduction(rendered: BookProject) -> None:
    assert _export(rendered) == 0
    record = _record(rendered)
    assert record["processing"]["loudness"]["method"] == "gated_rms_dbfs"
    gains = {row["id"]: row for row in record["processing"]["loudness"]["chapters"]}
    assert any(row["reduced_for_peak"] for row in gains.values())
    for chapter_id in ("c001", "c002", "c003"):
        original, _ = _master_samples(rendered, chapter_id)
        processed, _ = sf.read(str(_export_dir(rendered) / "work" / f"{chapter_id}.wav"), dtype="float32")
        assert len(processed) == len(original)
        assert float(np.max(np.abs(processed))) <= CEILING + 1e-6
        applied = 10 ** (gains[chapter_id]["applied_gain_db"] / 20)
        audible = np.abs(original) > 1e-3
        ratio = processed[audible] / original[audible]
        assert np.allclose(ratio, applied, rtol=1e-4)


@requires_ffmpeg
def test_chunk_and_master_files_are_byte_identical_after_export(rendered: BookProject) -> None:
    files = [rendered.directory / "chapters" / f"{cid}.wav" for cid in ("c001", "c002", "c003")]
    files += [
        rendered.target_dir(cid, "short") / record["audio_file"]
        for cid in ("c001", "c002", "c003")
        for record in read_json(rendered.target_dir(cid, "short") / "manifest.json")["chunks"]
    ]
    before = {path: _sha_bytes(path) for path in files}
    assert _export(rendered) == 0
    assert {path: _sha_bytes(path) for path in files} == before
    assert _record(rendered)["lossless"]["retained"] is True


def _refused(project: BookProject, capsys: pytest.CaptureFixture[str], reason: str) -> None:
    capsys.readouterr()
    assert _export(project) == 1
    assert reason in capsys.readouterr().err
    assert not (_export_dir(project) / "audiobook.m4b").exists()


def test_reviewed_export_is_refused_with_a_missing_chunk(rendered: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    chunk = _chunk_ids(rendered, "c001")[0]
    manifest = read_json(rendered.target_dir("c001", "short") / "manifest.json")
    record = next(item for item in manifest["chunks"] if item["id"] == chunk)
    (rendered.target_dir("c001", "short") / record["audio_file"]).unlink()
    _refused(rendered, capsys, "missing chunk")


def test_reviewed_export_is_refused_with_an_unresolved_content_finding(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_engine(FakeEngine(amplitude=0.2))
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short"]) == 0
    chunk = _chunk_ids(book, "c001")[0]
    transcriber({chunk: "совсем другие слова"})
    for chapter_id in ("c001", "c002", "c003"):
        assert main(["book", "verify", _d(book), "--chapter", chapter_id, "--transcriber", "fake-asr"]) == 0
        assert main(["book", "assemble", _d(book), "--chapter", chapter_id, "--target", "short"]) == 0
    _refused(book, capsys, "open")


def test_reviewed_export_is_refused_with_a_stale_master(rendered: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    chunk = _chunk_ids(rendered, "c002")[0]
    manifest_path = rendered.target_dir("c002", "short") / "manifest.json"
    manifest = read_json(manifest_path)
    record = next(item for item in manifest["chunks"] if item["id"] == chunk)
    path = rendered.target_dir("c002", "short") / record["audio_file"]
    samples, rate = sf.read(str(path), dtype="float32")
    sf.write(str(path), samples * 0.5, rate, subtype="FLOAT")
    record["audio_sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    assert master_is_stale(rendered, "c002")
    _refused(rendered, capsys, "master is stale")


def test_reviewed_export_is_refused_with_unverified_audio(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(amplitude=0.2))
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 0
    _refused(book, capsys, "not verified")


@requires_ffmpeg
@pytest.mark.parametrize("case", ["unverified", "stale", "missing"])
def test_draft_export_is_allowed_labelled_and_lists_the_gaps(
    rendered: BookProject, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if case == "unverified":
        chunk = _chunk_ids(rendered, "c001")[0]
        (rendered.directory / "review" / "findings" / "c001" / "short" / f"{chunk}.json").unlink()
    elif case == "stale":
        chunk = _chunk_ids(rendered, "c002")[0]
        manifest_path = rendered.target_dir("c002", "short") / "manifest.json"
        manifest = read_json(manifest_path)
        record = next(item for item in manifest["chunks"] if item["id"] == chunk)
        path = rendered.target_dir("c002", "short") / record["audio_file"]
        samples, rate = sf.read(str(path), dtype="float32")
        sf.write(str(path), samples * 0.5, rate, subtype="FLOAT")
        record["audio_sha256"] = sha256_file(path)
        write_json(manifest_path, manifest)
    else:
        chunk = _chunk_ids(rendered, "c003")[0]
        manifest = read_json(rendered.target_dir("c003", "short") / "manifest.json")
        record = next(item for item in manifest["chunks"] if item["id"] == chunk)
        (rendered.target_dir("c003", "short") / record["audio_file"]).unlink()
    assert _export(rendered, "--draft") == 0
    output = _export_dir(rendered) / "audiobook.draft.m4b"
    assert output.is_file()
    record = _record(rendered)
    assert record["kind"] == "draft"
    assert record["metadata"]["title"] == "Книга (ЧЕРНОВИК)"
    assert _probe(output)["format"]["tags"]["title"] == "Книга (ЧЕРНОВИК)"
    assert record["gaps"]


@requires_ffmpeg
def test_mp3_output_is_one_tagged_file_per_chapter_with_a_playlist(rendered: BookProject) -> None:
    assert _export(rendered, "--format", "mp3") == 0
    titles = ["Глава 1. Утро", "Глава 2. Дорога", "Глава 3. Пристань"]
    for number, title in enumerate(titles, start=1):
        path = _export_dir(rendered) / f"{number:02d} - {title}.mp3"
        assert path.is_file()
        tags = _probe(path)["format"]["tags"]
        assert tags["title"] == title
        assert tags["album"] == "Книга"
        assert tags["artist"] == "Автор"
        assert tags["track"].startswith(f"{number}/")
    playlist = _export_dir(rendered) / "audiobook.m3u8"
    assert playlist.is_file()
    assert "01 - Глава 1. Утро.mp3" in playlist.read_text(encoding="utf-8")


def _status_text(project: BookProject, *extra: str) -> str:
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main(["book", "export-status", _d(project), *extra]) == 0
    return buffer.getvalue()


@requires_ffmpeg
def test_export_becomes_stale_after_a_chunk_is_regenerated_and_a_rebuild_is_current(
    rendered: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    assert _export(rendered) == 0
    assert "ready" in _status_text(rendered)
    install_engine(FakeEngine(amplitude=0.1))
    assert main(["book", "render", _d(rendered), "--chapter", "c002", "--engine", "fake", "--target", "short", "--force-new-run"]) == 0
    assert "stale" in _status_text(rendered)
    assert main(["book", "verify", _d(rendered), "--chapter", "c002", "--transcriber", "fake-asr"]) == 0
    assert main(["book", "assemble", _d(rendered), "--chapter", "c002", "--target", "short"]) == 0
    assert _export(rendered) == 0
    assert "ready" in _status_text(rendered)


@requires_ffmpeg
def test_export_becomes_stale_after_a_chapter_is_renamed(rendered: BookProject) -> None:
    assert _export(rendered) == 0
    assert main(["book", "rename", _d(rendered), "--chapter", "c002", "--title", "Новое название"]) == 0
    assert "chapter order or titles" in _status_text(rendered)


@requires_ffmpeg
def test_export_becomes_stale_after_the_book_title_changes(rendered: BookProject) -> None:
    assert _export(rendered) == 0
    meta = read_json(rendered.directory / "project.json")
    meta["title"] = "Другая книга"
    write_json(rendered.directory / "project.json", meta)
    assert "book metadata" in _status_text(rendered)


@requires_ffmpeg
def test_export_becomes_stale_after_the_bitrate_setting_changes(rendered: BookProject) -> None:
    assert _export(rendered) == 0
    assert "stale" not in _status_text(rendered, "--bitrate", "96k")
    assert "processing settings" in _status_text(rendered, "--bitrate", "128k")


@requires_ffmpeg
def test_unchanged_project_is_a_no_op_and_force_rebuilds(rendered: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    assert _export(rendered) == 0
    before = (_export_dir(rendered) / "export.json").read_bytes()
    capsys.readouterr()
    assert _export(rendered) == 0
    assert "current" in capsys.readouterr().out
    assert (_export_dir(rendered) / "export.json").read_bytes() == before
    capsys.readouterr()
    assert _export(rendered, "--force") == 0
    assert "ready" in capsys.readouterr().out


@requires_ffmpeg
def test_corrupted_output_fails_export_check(rendered: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    assert _export(rendered) == 0
    assert main(["book", "export-check", _d(rendered)]) == 0
    (_export_dir(rendered) / "audiobook.m4b").write_bytes(b"not an audio file at all" * 100)
    capsys.readouterr()
    assert main(["book", "export-check", _d(rendered)]) == 1


@requires_ffmpeg
def test_export_open_uses_the_injected_runner(
    rendered: BookProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rutts.export import cli as export_cli

    assert _export(rendered) == 0
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(export_cli, "RUNNER", runner)
    assert main(["book", "export-open", _d(rendered)]) == 0
    assert calls == [["open", str(_export_dir(rendered) / "audiobook.m4b")]]

