from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.validation import ValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "book_sample.txt"


@pytest.fixture
def book(tmp_path: Path) -> Path:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title="Пробная книга", author="Автор", chapter_regex=None)
    return out


def _source(book: Path) -> Any:
    document: Any = json.loads((book / "source" / "source.json").read_text(encoding="utf-8"))
    return document


def test_import_writes_chapters_in_reading_order(book: Path) -> None:
    chapters = _source(book)["chapters"]
    assert [c["id"] for c in chapters] == ["c001", "c002", "c003"]
    assert [c["title"] for c in chapters] == [
        "Глава 1. Утро",
        "Глава 2. Дорога",
        "Третья глава",
    ]


def test_import_splits_paragraphs_scene_breaks_and_headings(book: Path) -> None:
    chapters = _source(book)["chapters"]
    first = chapters[0]["segments"]
    assert [s["kind"] for s in first] == ["heading", "paragraph", "scene_break", "paragraph"]
    assert first[0]["text"] == "Глава 1. Утро"
    assert first[2]["text"] == "* * *"
    assert [s["id"] for s in first] == ["c001-s0001", "c001-s0002", "c001-s0003", "c001-s0004"]


def test_import_records_line_locations(book: Path) -> None:
    first = _source(book)["chapters"][0]["segments"]
    assert first[1]["location"] == {"line_start": 3, "line_end": 5}
    assert first[3]["location"] == {"line_start": 9, "line_end": 11}


def test_import_preserves_dialogue_dashes_and_yo(book: Path) -> None:
    paragraph = _source(book)["chapters"][0]["segments"][1]["text"]
    assert paragraph.startswith("Утро началось рано.")
    assert "\n— Пора, — сказала она сама себе. — Сегодня нельзя опаздывать!\n" in paragraph
    assert "Всё ещё спало. Её мать" in paragraph


def test_import_keeps_every_segment_open_for_inclusion(book: Path) -> None:
    chapters = _source(book)["chapters"]
    for chapter in chapters:
        assert chapter["included"] is True
        for segment in chapter["segments"]:
            assert segment["included"] is True
            assert segment["exclusion_reason"] is None


def test_import_retains_byte_identical_original(tmp_path: Path) -> None:
    out = tmp_path / "book"
    import_text(FIXTURE, out, title=None, author="", chapter_regex=None)
    assert (out / "source" / "original.txt").read_bytes() == FIXTURE.read_bytes()


def test_project_metadata_records_source_hash(book: Path) -> None:
    project = load_project(book)
    import hashlib

    assert project.title == "Пробная книга"
    assert project.meta["original_filename"] == "book_sample.txt"
    assert project.meta["original_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert project.meta["source_format"] == "text"


def test_narration_links_back_to_source_segments(book: Path) -> None:
    narration = json.loads((book / "narration" / "narration.json").read_text(encoding="utf-8"))
    source_segments = [s for c in _source(book)["chapters"] for s in c["segments"]]
    assert [n["source_segment_id"] for n in narration["segments"]] == [s["id"] for s in source_segments]
    assert [n["text"] for n in narration["segments"]] == [s["text"] for s in source_segments]


def test_custom_chapter_regex_adds_markers(tmp_path: Path) -> None:
    text = tmp_path / "custom.txt"
    text.write_text("Пролог\n\nПервый абзац текста.\n\nСЦЕНА: второй абзац.\n", encoding="utf-8")
    out = tmp_path / "book"
    import_text(text, out, title=None, author="", chapter_regex=r"^СЦЕНА:")
    chapters = _source(out)["chapters"]
    assert [c["title"] for c in chapters] == ["", "СЦЕНА: второй абзац."]


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes("Привет, мир.".encode("cp1251"))
    with pytest.raises(ValidationError, match="not valid UTF-8"):
        import_text(path, tmp_path / "book", title=None, author="", chapter_regex=None)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("  \n\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="empty"):
        import_text(path, tmp_path / "book", title=None, author="", chapter_regex=None)


def test_file_without_russian_prose_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "english.txt"
    path.write_text("This is an English chapter about nothing.\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="predominantly Russian"):
        import_text(path, tmp_path / "book", title=None, author="", chapter_regex=None)


def test_cli_import_reports_invalid_input_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "english.txt"
    path.write_text("Only English here.\n", encoding="utf-8")
    code = main(["book", "import", str(path), "--out", str(tmp_path / "book")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_import_rejects_invalid_chapter_regex_with_exit_2(tmp_path: Path) -> None:
    code = main(
        [
            "book",
            "import",
            str(FIXTURE),
            "--out",
            str(tmp_path / "book"),
            "--chapter-regex",
            "([",
        ]
    )
    assert code == 2


def test_unwrapped_text_treats_each_line_as_a_paragraph(tmp_path: Path) -> None:
    long_line = "Он шёл по берегу и думал о том, что случилось вчера. " * 5
    source = tmp_path / "book.txt"
    source.write_text(f"Глава 1\nПервый абзац.\n{long_line.strip()}\nТретий абзац.\n", encoding="utf-8")
    book = tmp_path / "book"

    assert main(["book", "import", str(source), "--out", str(book)]) == 0

    document = json.loads((book / "source" / "source.json").read_text(encoding="utf-8"))
    paragraphs = [s["text"] for s in document["chapters"][0]["segments"] if s["kind"] == "paragraph"]
    assert paragraphs == ["Первый абзац.", long_line.strip(), "Третий абзац."]
