from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

from rutts.book.project import BookProject, install_narrator, load_project, sha256_file, write_json
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.verification.verify import MANUAL_REVIEW_REQUIRED
from tests.conftest import FakeEngine, FakeTranscriber
from tests.test_book_render import profile_dir  # noqa: F401  (shared fixture)

PROSE = (
    "В 1812 году армия ушла из Москвы. Старый зам+ок стоял на холме.\n\n"
    "Анна открыла окно, и ёлка во дворе горела огнями. Всё было тихо.\n"
)


@pytest.fixture
def book(tmp_path: Path, profile_dir: Path, monkeypatch: pytest.MonkeyPatch) -> BookProject:  # noqa: F811
    source = tmp_path / "source.txt"
    source.write_text(PROSE, encoding="utf-8")
    project = import_text(source, tmp_path / "book", title="Книга", author="", chapter_regex=None)
    install_narrator(project, profile_dir)
    engines: list[FakeEngine] = []

    def factory(**options: Any) -> FakeEngine:
        engine = FakeEngine(speech_rate=14.0)
        engines.append(engine)
        return engine

    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", factory)
    assert main(["book", "render", str(project.directory), "--chapter", "c001", "--engine", "fake", "--target", "short"]) == 0
    return load_project(project.directory)


def _plan_chunks(project: BookProject, target: str = "short") -> list[dict[str, Any]]:
    document: dict[str, Any] = json.loads((project.target_dir("c001", target) / "plan.json").read_text(encoding="utf-8"))
    return list(document["chunks"])


def _findings_document(project: BookProject, chunk_id: str) -> dict[str, Any]:
    path = project.directory / "review" / "findings" / "c001" / "short" / f"{chunk_id}.json"
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _verify(project: BookProject, *extra: str) -> int:
    return main(["book", "verify", str(project.directory), "--chapter", "c001", "--transcriber", "fake-asr", *extra])


def test_clean_chapter_gets_no_automatic_findings(book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]) -> None:
    transcriber = fake_transcriber({})
    assert _verify(book) == 0
    chunk_id = _plan_chunks(book)[0]["id"]
    document = _findings_document(book, chunk_id)
    assert document["status"] == "no_automatic_findings"
    assert document["findings"] == []
    assert document["manual_review_required"] == MANUAL_REVIEW_REQUIRED
    assert transcriber.calls == [chunk_id]
    report = (book.directory / "review" / "findings" / "c001" / "short" / "report.md").read_text(encoding="utf-8")
    assert "not verified as correct" in report
    assert "lexical stress" in report.lower()
    assert "no_automatic_findings" in report


def test_numbers_yo_and_stress_hint_produce_no_findings(book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]) -> None:
    chunk = _plan_chunks(book)[0]
    heard = {chunk["id"]: "в тысяча восемьсот двенадцатом году армия ушла из москвы старый замок стоял на холме анна открыла окно и елка во дворе горела огнями всё было тихо"}
    fake_transcriber(heard)
    assert _verify(book) == 0
    assert _findings_document(book, chunk["id"])["findings"] == []


@pytest.mark.parametrize(
    ("kind", "old", "new", "expected", "heard"),
    [
        ("omission", "Анна открыла", "открыла", "Анна", ""),
        ("addition", "Старый зам+ок", "Старый очень зам+ок", "", "очень"),
        ("substitution", "открыла окно", "закрыла окно", "открыла", "закрыла"),
        ("repetition", "открыла окно", "открыла открыла окно", "открыла", "открыла"),
    ],
)
def test_word_level_differences_are_reported_with_span(
    book: BookProject,
    fake_transcriber: Callable[..., FakeTranscriber],
    kind: str,
    old: str,
    new: str,
    expected: str,
    heard: str,
) -> None:
    chunk = _plan_chunks(book)[0]
    text = chunk["text"]
    assert text.count(old) == 1
    fake_transcriber({chunk["id"]: text.replace(old, new, 1)})
    _verify(book)
    findings = _findings_document(book, chunk["id"])["findings"]
    assert [finding["type"] for finding in findings] == [kind]
    finding = findings[0]
    assert finding["expected"] == expected
    assert finding["heard"] == heard
    start, end = finding["location"]["chars"]
    if expected:
        assert text[start:end] == expected
    assert finding["location"]["audio_seconds"][0] >= 0.0


def test_stored_findings_are_printed(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    chunk = _plan_chunks(book)[0]
    fake_transcriber({chunk["id"]: "Старый замок стоял на холме."})
    _verify(book)
    capsys.readouterr()
    assert main(["book", "findings", str(book.directory), "--chapter", "c001", "--chunk", chunk["id"]]) == 0
    out = capsys.readouterr().out
    assert "omission" in out and "Анна" in out


def test_current_findings_are_reused_and_changed_audio_is_rechecked(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]
) -> None:
    transcriber = fake_transcriber({})
    _verify(book)
    first_calls = len(transcriber.calls)
    _verify(book)
    assert len(transcriber.calls) == first_calls

    manifest_path = book.target_dir("c001", "short") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["chunks"][0]
    audio_path = book.target_dir("c001", "short") / record["audio_file"]
    samples, rate = sf.read(str(audio_path), dtype="float32")
    sf.write(str(audio_path), samples * 0.5, rate, subtype="FLOAT")
    record["audio_sha256"] = sha256_file(audio_path)
    write_json(manifest_path, manifest)

    _verify(book)
    assert transcriber.calls[first_calls:] == [record["id"]]


def test_missing_audio_is_reported_not_skipped(book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]) -> None:
    transcriber = fake_transcriber({})
    manifest_path = book.target_dir("c001", "short") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["chunks"][0]
    (book.target_dir("c001", "short") / record["audio_file"]).unlink()

    assert _verify(book) == 0
    document = _findings_document(book, record["id"])
    assert [finding["type"] for finding in document["findings"]] == ["missing_audio"]
    assert record["id"] not in transcriber.calls


def test_transcriber_is_unloaded_after_a_recoverable_failure(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]
) -> None:
    chunk_id = _plan_chunks(book)[0]["id"]
    transcriber = fake_transcriber({}, fail_stems=frozenset({chunk_id}))
    assert _verify(book) == 0
    assert transcriber.unloads == 1
    assert [f["type"] for f in _findings_document(book, chunk_id)["findings"]] == ["transcription_failed"]


def test_transcriber_is_unloaded_when_verification_aborts(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]
) -> None:
    chunk_id = _plan_chunks(book)[0]["id"]
    transcriber = fake_transcriber({}, fatal_stems=frozenset({chunk_id}))
    with pytest.raises(MemoryError):
        _verify(book)
    assert transcriber.loads == 1
    assert transcriber.unloads == 1


def test_failed_transcriber_load_exits_1_and_unloads(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    transcriber = fake_transcriber({}, fail_load=True)
    assert _verify(book) == 1
    assert transcriber.unloads == 1
    assert "could not load" in capsys.readouterr().err


def test_verification_never_creates_a_synthesis_engine(
    book: BookProject, fake_transcriber: Callable[..., FakeTranscriber], monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    def factory(**options: Any) -> FakeEngine:
        created.append("engine")
        return FakeEngine()

    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", factory)
    fake_transcriber({})
    assert _verify(book) == 0
    assert created == []


def test_verify_without_a_render_exits_1(tmp_path: Path, fake_transcriber: Callable[..., FakeTranscriber]) -> None:
    source = tmp_path / "source.txt"
    source.write_text(PROSE, encoding="utf-8")
    project = import_text(source, tmp_path / "book", title="Книга", author="", chapter_regex=None)
    fake_transcriber({})
    assert _verify(project) == 1


def test_findings_without_verification_exits_1(book: BookProject, fake_transcriber: Callable[..., FakeTranscriber]) -> None:
    fake_transcriber({})
    assert main(["book", "findings", str(book.directory), "--chapter", "c001"]) == 1
