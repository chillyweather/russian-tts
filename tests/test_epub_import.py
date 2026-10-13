from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import ORIGINAL_EPUB_FILE, load_project
from rutts.cli import main
from rutts.narrator import create_profile
from tests.conftest import FakeEngine, write_reference_wav
from tests.epub_support import CONTAINER, write_epub, write_minimal_epub

TRANSCRIPT = "Это тестовая фраза для опорной записи."
NOTE_ONE = "Имеется в виду старая мельница."
NOTE_TWO = "Река называлась Смородина."


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
def epub(tmp_path: Path) -> Path:
    return write_epub(tmp_path / "book.epub")


def _import(epub_path: Path, out: Path, policy: str = "omit") -> int:
    return main(["book", "import", str(epub_path), "--out", str(out), "--footnotes", policy])


def _chapters(out: Path) -> list[dict[str, Any]]:
    document: list[dict[str, Any]] = json.loads((out / "source" / "source.json").read_text(encoding="utf-8"))["chapters"]
    return document


def _chapter(out: Path, title: str) -> dict[str, Any]:
    return next(chapter for chapter in _chapters(out) if chapter["title"] == title)


def _narration(out: Path, chapter_id: str) -> list[str]:
    """Narration text of the segments that would be synthesized (included ones, in order)."""
    included = {
        segment["id"]
        for chapter in _chapters(out)
        if chapter["id"] == chapter_id
        for segment in chapter["segments"]
        if segment["included"]
    }
    data = json.loads((out / "narration" / "narration.json").read_text(encoding="utf-8"))
    return [segment["text"] for segment in data["segments"] if segment["source_segment_id"] in included]


def _rendered(
    out: Path, chapter_id: str, profile_dir: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> str:
    assert main(["book", "narrator", str(out), "--profile", str(profile_dir)]) == 0
    code = main(["book", "render", str(out), "--chapter", chapter_id, "--engine", "fake", "--target", "short"])
    assert code == 0, capsys.readouterr()
    return "\n".join(fake_engine_factory[-1].received)


def test_reading_order_follows_spine_not_manifest(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    assert _import(epub, out) == 0
    titles = [chapter["title"] for chapter in _chapters(out)]
    assert titles == [
        "cover",
        "Тестовая книга",
        "Содержание",
        "Глава 2. Дорога",
        "Глава 1. Утро",
        "Часть первая",
        "Часть вторая",
        "Глава 3. Город",
        "Дополнение",
        "Примечания",
    ]


def test_metadata_and_original_are_retained(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    project = load_project(out)
    assert (project.title, project.author) == ("Тестовая книга", "Тестовый Автор")
    assert project.meta["language"] == "ru"
    assert project.meta["source_format"] == "epub"
    assert (out / ORIGINAL_EPUB_FILE).read_bytes() == epub.read_bytes()


def test_titles_come_from_nav_or_ncx(epub: Path, tmp_path: Path) -> None:
    out_nav = tmp_path / "nav"
    _import(epub, out_nav)
    ncx_only = write_epub(tmp_path / "ncx.epub", nav=False)
    out_ncx = tmp_path / "ncx"
    assert _import(ncx_only, out_ncx) == 0
    for out in (out_nav, out_ncx):
        titles = [chapter["title"] for chapter in _chapters(out)]
        assert "Глава 1. Утро" in titles
        assert "Часть вторая" in titles
    assert _chapter(out_ncx, "Глава 1. Утро")["included"] is True


def test_contents_cover_and_front_matter_are_excluded_with_reasons(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    reasons = {chapter["title"]: chapter["exclusion_reason"] for chapter in _chapters(out) if not chapter["included"]}
    assert reasons == {
        "cover": "cover",
        "Тестовая книга": "front_matter",
        "Содержание": "navigation",
        "Дополнение": "non_linear_spine_item",
    }


def test_excluded_contents_are_not_sent_to_engine(
    epub: Path, tmp_path: Path, profile_dir: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    chapter_id = _chapter(out, "Глава 2. Дорога")["id"]
    spoken = _rendered(out, chapter_id, profile_dir, fake_engine_factory, capsys)
    assert "Дорога шла через лес." in spoken
    assert "Глава 1. Утро" not in spoken
    assert "Тестовый Автор" not in spoken
    assert main(["book", "render", str(out), "--chapter", _chapter(out, "Содержание")["id"], "--engine", "fake"]) == 2


def test_illustrations_are_recorded_as_excluded_with_alt_text(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    chapter = _chapter(out, "Глава 2. Дорога")
    images = [segment for segment in chapter["segments"] if segment["kind"] == "auxiliary"]
    assert [(s["included"], s["exclusion_reason"], s["text"]) for s in images] == [(False, "illustration", "Карта дороги")]
    assert images[0]["location"]["href"] == "OEBPS/chapter2.xhtml"


def test_scene_breaks_from_hr_and_asterisks(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    chapter = _chapter(out, "Глава 1. Утро")
    kinds = [(segment["kind"], segment["text"]) for segment in chapter["segments"]]
    assert kinds == [
        ("heading", "Глава 1. Утро"),
        ("paragraph", "Утро началось рано. Всё ещё спало. Её мать1 уже ходила по кухне."),
        ("paragraph", NOTE_ONE),
        ("scene_break", ""),
        ("paragraph", "— Пора, — сказала она сама себе."),
        ("scene_break", "***"),
        ("paragraph", "Вечер пришёл неожиданно."),
    ]


def test_text_is_extracted_exactly(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    chapter = _chapter(out, "Глава 2. Дорога")
    texts = [segment["text"] for segment in chapter["segments"] if segment["kind"] == "paragraph"]
    assert texts == ["Дорога шла через лес.", "Поезд опоздал."]


def test_anchor_split_creates_two_chapters(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    first = _chapter(out, "Часть первая")
    second = _chapter(out, "Часть вторая")
    assert [s["text"] for s in first["segments"]] == ["Часть первая", "Первая часть книги начинается здесь."]
    assert [s["text"] for s in second["segments"]] == ["Часть вторая", "Вторая часть книги продолжается."]


def test_non_linear_item_is_imported_but_excluded(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    extra = _chapter(out, "Дополнение")
    assert extra["included"] is False
    assert extra["exclusion_reason"] == "non_linear_spine_item"
    assert extra["segments"][1]["text"] == "Дополнительный текст вне порядка чтения."


def test_omit_policy_removes_marker_and_excludes_note(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "omit")
    chapter = _chapter(out, "Глава 1. Утро")
    note = [s for s in chapter["segments"] if s["text"] == NOTE_ONE][0]
    assert (note["included"], note["exclusion_reason"]) == (False, "footnote_omitted")
    spoken = _narration(out, chapter["id"])
    assert spoken[1] == "Утро началось рано. Всё ещё спало. Её мать уже ходила по кухне."
    assert NOTE_ONE not in "\n".join(spoken)
    assert "мать1" in chapter["segments"][1]["text"]


def test_chapter_end_policy_reads_note_once_at_chapter_end(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "chapter-end")
    chapter = _chapter(out, "Глава 1. Утро")
    assert _narration(out, chapter["id"])[-1] == f"Примечание 1. {NOTE_ONE}"
    assert sum(NOTE_ONE in text for text in _narration(out, chapter["id"])) == 1


def test_inline_policy_reads_note_after_its_paragraph(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "inline")
    chapter = _chapter(out, "Глава 1. Утро")
    spoken = _narration(out, chapter["id"])
    assert spoken[2] == f"Примечание 1. {NOTE_ONE}"


def test_superscript_note_document_is_relocated_and_excluded(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "chapter-end")
    assert _chapter(out, "Примечания")["exclusion_reason"] == "footnotes_relocated"
    assert _narration(out, _chapter(out, "Глава 3. Город")["id"])[-1] == f"Примечание 1. {NOTE_TWO}"


def test_superscript_note_marker_is_removed_from_narration_only(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "omit")
    chapter = _chapter(out, "Глава 3. Город")
    assert chapter["segments"][1]["text"] == "Город спал1, а река шумела."
    assert _narration(out, chapter["id"])[1] == "Город спал, а река шумела."


def test_chapter_end_note_is_spoken_exactly_once(
    epub: Path, tmp_path: Path, profile_dir: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "book"
    _import(epub, out, "chapter-end")
    spoken = _rendered(out, _chapter(out, "Глава 1. Утро")["id"], profile_dir, fake_engine_factory, capsys)
    assert spoken.count(f"Примечание 1. {NOTE_ONE}") == 1
    assert "Её мать уже ходила по кухне." in spoken


def test_scene_break_policy_marks_hr_and_star_lines(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    scene_breaks = [s for c in _chapters(out) for s in c["segments"] if s["kind"] == "scene_break"]
    assert len(scene_breaks) == 4


def test_original_epub_is_byte_identical(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    assert (out / ORIGINAL_EPUB_FILE).read_bytes() == epub.read_bytes()


def test_preview_import_writes_nothing(epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["book", "preview-import", str(epub), "--footnotes", "inline"]) == 0
    output = capsys.readouterr().out
    assert "Глава 2. Дорога' -> narrated" in output
    assert "Содержание" in output and "navigation" in output
    assert "illustrations and auxiliary items (excluded): 2" in output
    assert "footnote references: 2" in output
    assert sorted(path.name for path in tmp_path.iterdir()) == ["book.epub"]


def test_split_merge_and_exclude_work_on_epub_projects(epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    chapter = _chapter(out, "Глава 1. Утро")
    at = chapter["segments"][4]["id"]
    assert main(["book", "split", str(out), "--at", at, "--title", "Вторая сцена"]) == 0
    titles = [c["title"] for c in _chapters(out)]
    assert "Вторая сцена" in titles
    new_chapter = _chapter(out, "Вторая сцена")
    assert new_chapter["segments"][0]["text"] == "— Пора, — сказала она сама себе."
    assert main(["book", "merge", str(out), "--chapter", new_chapter["id"]]) == 0
    assert _chapter(out, "Глава 1. Утро")["segments"][-1]["text"] == "Вечер пришёл неожиданно."
    assert main(["book", "exclude", str(out), "--chapter", _chapter(out, "Глава 3. Город")["id"]]) == 0
    assert main(["book", "show", str(out)]) == 0
    assert "Глава 3. Город" in capsys.readouterr().out


def test_show_lists_excluded_chapters_with_reasons(epub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "book"
    _import(epub, out)
    assert main(["book", "show", str(out)]) == 0
    output = capsys.readouterr().out
    assert "[excluded: cover]" in output
    assert "[excluded: navigation]" in output
    assert "[excluded: non_linear_spine_item]" in output


def test_malformed_document_is_reported_with_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_epub(tmp_path / "broken.epub", broken=True)
    out = tmp_path / "book"
    assert _import(path, out) == 1
    assert "unreadable document OEBPS/broken.xhtml" in capsys.readouterr().err
    assert _chapter(out, "broken")["exclusion_reason"] == "unreadable_document"
    report = json.loads((out / "source" / "import-report.json").read_text(encoding="utf-8"))
    assert any("broken.xhtml" in warning for warning in report["warnings"])


def test_drm_archive_is_rejected_with_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_epub(tmp_path / "drm.epub", drm=True)
    assert _import(path, tmp_path / "book") == 2
    assert "DRM" in capsys.readouterr().err


def test_non_zip_input_is_rejected_with_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "fake.epub"
    path.write_text("Это не EPUB.\n", encoding="utf-8")
    assert _import(path, tmp_path / "book") == 2
    assert "not a zip archive" in capsys.readouterr().err


def test_missing_package_document_is_rejected_with_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "nopkg.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTAINER)
    assert _import(path, tmp_path / "book") == 2
    assert "package document" in capsys.readouterr().err


def test_no_russian_prose_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_minimal_epub(tmp_path / "english.epub", "<p>Only English text here.</p>")
    assert _import(path, tmp_path / "book") == 2
    assert "Russian" in capsys.readouterr().err


def test_rendered_epub_chapter_is_assembled_completely(
    epub: Path, tmp_path: Path, profile_dir: Path, fake_engine_factory: list[FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "book"
    _import(epub, out, "chapter-end")
    chapter = _chapter(out, "Глава 1. Утро")
    _rendered(out, chapter["id"], profile_dir, fake_engine_factory, capsys)
    assert main(["book", "assemble", str(out), "--chapter", chapter["id"], "--target", "short"]) == 0
    report = json.loads((out / "chapters" / f"{chapter['id']}.json").read_text(encoding="utf-8"))
    assert report["complete"] is True
    paragraphs = [s["id"] for s in chapter["segments"] if s["included"] and s["kind"] == "paragraph"]
    covered = {entry["segment_id"] for entry in report["coverage"] if entry["chunks"]}
    assert set(paragraphs) <= covered
    assert report["uncovered_segments"] == []
    assert [(entry["segment_id"], entry["reason"]) for entry in report["excluded_segments"]] == [
        ("c005-s0003", "footnotes_relocated")
    ]


def test_footnote_policy_can_change_after_import(epub: Path, tmp_path: Path) -> None:
    out = tmp_path / "book"
    _import(epub, out, "omit")
    assert main(["book", "footnotes", str(out), "--policy", "chapter-end"]) == 0
    chapter = _chapter(out, "Глава 1. Утро")
    assert _narration(out, chapter["id"])[-1] == f"Примечание 1. {NOTE_ONE}"
    assert load_project(out).meta["footnote_policy"] == "chapter-end"
    report = json.loads((out / "source" / "import-report.json").read_text(encoding="utf-8"))
    assert report["footnote_policy"] == "chapter-end"
