from __future__ import annotations

import json
from pathlib import Path

import pytest

from rutts.book.boundaries import (
    merge_chapter,
    rename_chapter,
    set_chapter_included,
    set_segment_included,
    split_chapter,
)
from rutts.book.project import BookProject, load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.validation import ValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"


@pytest.fixture
def book(tmp_path: Path) -> BookProject:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title="Книга", author="", chapter_regex=None)
    return load_project(out)


def _chapters(project: BookProject) -> list[tuple[str, str, list[str]]]:
    return [
        (c.id, c.title, [s.id for s in c.segments])
        for c in project.read_source().chapters
    ]


def test_split_creates_new_chapter_and_rederives_ids(book: BookProject) -> None:
    split_chapter(book, "c001-s0004", "Вечер")
    assert _chapters(book) == [
        ("c001", "Глава 1. Утро", ["c001-s0001", "c001-s0002", "c001-s0003"]),
        ("c002", "Вечер", ["c002-s0001"]),
        ("c003", "Глава 2. Дорога", [f"c003-s{i:04d}" for i in range(1, 6)]),
        ("c004", "Третья глава", ["c004-s0001", "c004-s0002"]),
    ]


def test_split_keeps_narration_linked_to_source(book: BookProject) -> None:
    split_chapter(book, "c001-s0004", "Вечер")
    narration = book.read_narration().segments
    assert [n.id for n in narration] == [n.source_segment_id for n in narration]
    assert len(narration) == 11
    assert narration[3].chapter_id == "c002"


def test_split_at_first_segment_is_rejected(book: BookProject) -> None:
    with pytest.raises(ValidationError, match="already starts"):
        split_chapter(book, "c002-s0001", "Новая")


def test_merge_joins_chapter_into_previous_and_renumbers(book: BookProject) -> None:
    merge_chapter(book, "c002")
    chapters = _chapters(book)
    assert [c[0] for c in chapters] == ["c001", "c002"]
    assert chapters[0][2][-1] == "c001-s0009"
    assert chapters[1][1] == "Третья глава"


def test_merge_first_chapter_is_rejected(book: BookProject) -> None:
    with pytest.raises(ValidationError, match="no previous chapter"):
        merge_chapter(book, "c001")


def test_rename_chapter_updates_title(book: BookProject) -> None:
    rename_chapter(book, "c003", "Финал")
    assert book.read_source().chapter("c003").title == "Финал"


def test_blank_title_is_rejected(book: BookProject) -> None:
    with pytest.raises(ValidationError, match="title"):
        rename_chapter(book, "c001", "   ")


def test_exclude_and_include_segment_records_reason(book: BookProject) -> None:
    set_segment_included(book, "c002-s0003", False, reason="footnote text")
    segment = book.read_source().chapter("c002").segments[2]
    assert (segment.included, segment.exclusion_reason) == (False, "footnote text")
    assert book.planning_segments("c002")[2].id != "c002-s0003"

    set_segment_included(book, "c002-s0003", True)
    segment = book.read_source().chapter("c002").segments[2]
    assert (segment.included, segment.exclusion_reason) == (True, None)


def test_excluding_chapter_removes_it_from_planning(book: BookProject) -> None:
    set_chapter_included(book, "c002", False)
    assert book.planning_segments("c002") == []
    assert book.read_source().chapter("c002").included is False


def test_boundary_edits_leave_original_file_untouched(book: BookProject) -> None:
    before = (book.directory / "source" / "original.txt").read_bytes()
    split_chapter(book, "c001-s0004", "Вечер")
    merge_chapter(book, "c003")
    set_segment_included(book, "c001-s0002", False, reason="epigraph")
    assert (book.directory / "source" / "original.txt").read_bytes() == before
    assert before == FIXTURE.read_bytes()


def test_cli_split_merge_rename_exclude_include(book: BookProject) -> None:
    directory = str(book.directory)
    assert main(["book", "split", directory, "--at", "c001-s0004", "--title", "Вечер"]) == 0
    assert main(["book", "rename", directory, "--chapter", "c002", "--title", "Вечер дома"]) == 0
    assert main(["book", "exclude", directory, "--segment", "c002-s0001", "--reason", "тест"]) == 0
    assert main(["book", "include", directory, "--segment", "c002-s0001"]) == 0
    assert main(["book", "exclude", directory, "--chapter", "c003"]) == 0
    assert main(["book", "include", directory, "--chapter", "c003"]) == 0
    assert main(["book", "merge", directory, "--chapter", "c003"]) == 0
    chapters = json.loads((book.directory / "source" / "source.json").read_text(encoding="utf-8"))
    assert [c["title"] for c in chapters["chapters"]] == ["Глава 1. Утро", "Вечер дома", "Третья глава"]


def test_cli_boundary_error_exits_2(book: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["book", "merge", str(book.directory), "--chapter", "c001"])
    assert code == 2
    assert "no previous chapter" in capsys.readouterr().err
