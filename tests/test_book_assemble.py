from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from rutts.book.boundaries import set_segment_included
from rutts.book.chunking import ChunkLimits
from rutts.book.project import BookProject, install_narrator, load_project
from rutts.book.render import render_chapter
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.narrator import create_profile
from rutts.validation import ValidationError
from tests.conftest import FakeEngine

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"
RATE = 24000


@pytest.fixture
def narrated_book(tmp_path: Path, reference_wav: Path) -> BookProject:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title="Книга", author="", chapter_regex=None)
    project = load_project(out)
    profile = create_profile(
        tmp_path / "narrator",
        name="Fixture narrator",
        audio=reference_wav,
        transcript="Это тестовая фраза для опорной записи.",
        provenance="Test recording",
    )
    install_narrator(project, profile.directory)
    return project


def _chapter_report(project: BookProject, chapter_id: str) -> Any:
    path = project.directory / "chapters" / f"{chapter_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _render(project: BookProject, engine: FakeEngine, target: str = "short") -> None:
    render_chapter(project, "c001", engine, target, limits=ChunkLimits(), force_new_run=False)


def test_assembly_orders_chunks_and_inserts_natural_pauses(narrated_book: BookProject) -> None:
    from rutts.book.assemble import assemble_chapter

    _render(narrated_book, FakeEngine())
    report = assemble_chapter(narrated_book, "c001", target="short")

    assert report["complete"] is True
    assert report["master_file"] == "chapters/c001.wav"
    chunks = report["chunks"]
    assert [c["boundary_after"] for c in chunks] == ["scene", "chapter_end"]
    # two 0.5 s chunks, then a 1.5 s scene pause and a 1.0 s chapter-end pause
    assert [(c["start_seconds"], c["end_seconds"]) for c in chunks] == [(0.0, 0.5), (2.0, 2.5)]
    master, rate = sf.read(str(narrated_book.directory / "chapters" / "c001.wav"), dtype="float32")
    assert rate == RATE
    assert len(master) == int(3.5 * RATE)
    assert np.all(master[12000:36000] == 0)


def test_coverage_accounts_for_every_included_segment(narrated_book: BookProject) -> None:
    from rutts.book.assemble import assemble_chapter

    _render(narrated_book, FakeEngine())
    report = assemble_chapter(narrated_book, "c001", target="short")
    coverage = {entry["segment_id"]: entry for entry in report["coverage"]}
    assert list(coverage) == ["c001-s0001", "c001-s0002", "c001-s0003", "c001-s0004"]
    assert coverage["c001-s0001"]["chunks"] == []  # heading is structural
    assert coverage["c001-s0002"]["chunks"] == ["c001-k0001"]
    assert coverage["c001-s0004"]["chunks"] == ["c001-k0002"]
    assert report["uncovered_segments"] == []
    assert report["excluded_segments"] == []


def test_exclusions_are_listed_with_reasons(narrated_book: BookProject) -> None:
    from rutts.book.assemble import assemble_chapter

    set_segment_included(narrated_book, "c001-s0002", False, reason="epigraph")
    _render(narrated_book, FakeEngine())
    report = assemble_chapter(narrated_book, "c001", target="short")
    assert report["excluded_segments"] == [{"segment_id": "c001-s0002", "reason": "epigraph"}]
    coverage_ids = [entry["segment_id"] for entry in report["coverage"]]
    assert "c001-s0002" not in coverage_ids
    assert report["complete"] is True


def test_failed_chunk_marks_assembly_incomplete_and_exits_1(
    narrated_book: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    from rutts.book.chunking import plan_chunks

    first_text = plan_chunks(narrated_book.planning_segments("c001"), "short", ChunkLimits())[0].text
    render_chapter(
        narrated_book,
        "c001",
        FakeEngine(fail_texts=frozenset({first_text})),
        "short",
        limits=ChunkLimits(),
        force_new_run=False,
    )
    code = main(["book", "assemble", str(narrated_book.directory), "--chapter", "c001", "--target", "short"])
    assert code == 1
    assert "failed chunk: c001-k0001" in capsys.readouterr().out
    assert (narrated_book.directory / "chapters" / "c001.incomplete.wav").is_file()
    assert not (narrated_book.directory / "chapters" / "c001.wav").exists()
    report = _chapter_report(narrated_book, "c001")
    assert report["complete"] is False
    assert report["failed_chunks"] == ["c001-k0001"]
    assert report["missing_chunks"] == []


def test_missing_chunk_audio_is_reported(
    narrated_book: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    _render(narrated_book, FakeEngine())
    (narrated_book.directory / "chunks" / "c001" / "short" / "c001-k0002.wav").unlink()
    code = main(["book", "assemble", str(narrated_book.directory), "--chapter", "c001", "--target", "short"])
    assert code == 1
    assert "missing chunk: c001-k0002" in capsys.readouterr().out
    report = _chapter_report(narrated_book, "c001")
    assert report["missing_chunks"] == ["c001-k0002"]
    assert report["complete"] is False


def test_assembly_without_render_is_refused(narrated_book: BookProject) -> None:
    from rutts.book.assemble import assemble_chapter

    with pytest.raises(ValidationError, match="has not been rendered"):
        assemble_chapter(narrated_book, "c001", target=None)


def test_ambiguous_target_requires_explicit_choice(
    narrated_book: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    _render(narrated_book, FakeEngine(), target="short")
    _render(narrated_book, FakeEngine(), target="long")
    code = main(["book", "assemble", str(narrated_book.directory), "--chapter", "c001"])
    assert code == 2
    assert "pass --target" in capsys.readouterr().err


def test_compare_targets_reports_measured_durations(
    narrated_book: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    _render(narrated_book, FakeEngine(), target="short")
    _render(narrated_book, FakeEngine(), target="long")
    code = main(["book", "compare-targets", str(narrated_book.directory), "--chapter", "c001"])
    out = capsys.readouterr().out
    assert code == 0
    assert "short: 2 chunk(s), 2 ok" in out
    assert "long: 2 chunk(s), 2 ok" in out
    assert "measured min 0.50 s, mean 0.50 s, max 0.50 s" in out
    assert "estimate" not in out.lower()
