from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

from rutts.book.boundaries import split_chapter
from rutts.book.project import ORIGINAL_FILE, SOURCE_FILE, BookProject, install_narrator, load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.pronunciation.store import DICTIONARY_FILE, HISTORY_FILE, OVERRIDES_FILE
from tests.conftest import FakeEngine
from tests.test_book_render import book, profile_dir  # noqa: F401  (shared fixtures)

PROSE = (
    "Глава 1. Утро\n\n"
    "Старый замок стоял на холме. Анна заперла замок на ключ. Всё было тихо, все спали, всего хватало.\n"
)
ZAMOK_LINE = "Старый замок стоял на холме. Анна заперла замок на ключ. Всё было тихо, все спали, всего хватало."
EVENING_ORIGINAL = (
    "Вечером на ёлке во дворе горели огни. Дети шумели у подъезда.\n"
    "— Ты придёшь? — спросил Пётр.\n"
    "— Приду, — ответила Анна. — Если всё будет хорошо."
)
EVENING_CORRECTED = EVENING_ORIGINAL.replace("Вечером", "Утром")
EVENING_HINT = EVENING_ORIGINAL.replace("Вечером", "Веч+ером")
CHUNK_ONE = (
    "Утро началось рано. Анна открыла окно, и в комнату ворвался холодный воздух.\n"
    "— Пора, — сказала она сама себе. — Сегодня нельзя опаздывать!\n"
    "Всё ещё спало. Её мать уже стояла на кухне, и на плите жарилась яичница."
)


@pytest.fixture
def prose(tmp_path: Path, profile_dir: Path) -> BookProject:  # noqa: F811
    text = tmp_path / "prose.txt"
    text.write_text(PROSE, encoding="utf-8")
    out = tmp_path / "prose-book"
    import_text(text, out, title="Проза", author="", chapter_regex=None)
    project = load_project(out)
    install_narrator(project, profile_dir)
    return project


@pytest.fixture
def install_engine(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeEngine], None]:
    def install(engine: FakeEngine) -> None:
        monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: engine)

    return install


def _run(*args: str) -> int:
    return main(["book", *args])


def _d(project: BookProject) -> str:
    return str(project.directory)


def _narration(project: BookProject) -> dict[str, str]:
    return {segment.id: segment.text for segment in project.read_narration().segments}


def _render(project: BookProject, *extra: str) -> int:
    return _run("render", _d(project), "--chapter", "c001", "--engine", "fake", "--target", "short", *extra)


def _history(project: BookProject) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads((project.directory / HISTORY_FILE).read_text(encoding="utf-8"))["entries"]
    return entries


def test_corrections_never_change_the_authors_source(prose: BookProject) -> None:
    source_bytes = (prose.directory / SOURCE_FILE).read_bytes()
    original_bytes = (prose.directory / ORIGINAL_FILE).read_bytes()
    assert _run("dict", "add", _d(prose), "--pattern", "все", "--replacement", "вс+е", "--yes") == 0
    assert _run(
        "override", "add", _d(prose), "--segment", "c001-s0002", "--find", "замок", "--occurrence", "2",
        "--replacement", "зам+ок", "--yes",
    ) == 0
    assert "зам+ок" in _narration(prose)["c001-s0002"]
    assert (prose.directory / SOURCE_FILE).read_bytes() == source_bytes
    assert (prose.directory / ORIGINAL_FILE).read_bytes() == original_bytes


def test_dictionary_word_match_respects_word_boundaries_and_yo(prose: BookProject) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "все", "--replacement", "вс+е", "--yes") == 0
    assert _narration(prose)["c001-s0002"] == (
        "Старый замок стоял на холме. Анна заперла замок на ключ. Всё было тихо, вс+е спали, всего хватало."
    )


def test_contextual_override_changes_one_homograph_and_dictionary_handles_the_rest(prose: BookProject) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    assert _run(
        "override", "add", _d(prose), "--segment", "c001-s0002", "--find", "замок", "--occurrence", "2",
        "--replacement", "зам+ок", "--yes",
    ) == 0
    assert _narration(prose)["c001-s0002"] == (
        "Старый з+амок стоял на холме. Анна заперла зам+ок на ключ. Всё было тихо, все спали, всего хватало."
    )


def test_override_takes_precedence_over_dictionary_on_the_same_span(
    prose: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    assert _run(
        "override", "add", _d(prose), "--segment", "c001-s0002", "--find", "замок", "--occurrence", "1",
        "--replacement", "зам+ок", "--yes",
    ) == 0
    assert _narration(prose)["c001-s0002"].startswith("Старый зам+ок стоял на холме. Анна заперла з+амок")
    capsys.readouterr()
    assert _run("diff", _d(prose)) == 0
    assert "(override o001, other)" in capsys.readouterr().out


def test_dictionary_add_shows_the_diff_and_requires_yes(prose: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок") == 2
    out = capsys.readouterr()
    assert "+ c001-s0002:" in out.out
    assert "[замок -> з+амок]" in out.out
    assert "would invalidate short: c001-k0001" in out.out
    assert "--yes" in out.err
    assert _narration(prose)["c001-s0002"] == ZAMOK_LINE
    assert not (prose.directory / DICTIONARY_FILE).exists()

    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    assert _narration(prose)["c001-s0002"].count("з+амок") == 2


def test_unsupported_markup_is_rejected_when_the_correction_is_added(
    prose: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "<phoneme>зам+ок</phoneme>", "--yes") == 2
    assert "angle brackets" in capsys.readouterr().err
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "зам+", "--yes") == 2
    assert "must be directly before a Russian vowel" in capsys.readouterr().err
    assert not (prose.directory / DICTIONARY_FILE).exists()
    assert _narration(prose)["c001-s0002"] == ZAMOK_LINE


def test_hints_are_stripped_for_an_engine_without_a_verified_convention(
    prose: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    engine = FakeEngine()
    install_engine(engine)
    assert _render(prose) == 0
    assert engine.received == [ZAMOK_LINE]
    assert "stress hints were not applied for fake" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("plus", ZAMOK_LINE.replace("замок", "з+амок")),
        ("acute", ZAMOK_LINE.replace("замок", "за\u0301мок")),
    ],
)
def test_hints_are_translated_for_an_engine_with_a_verified_convention(
    prose: BookProject, install_engine: Callable[[FakeEngine], None], convention: str, expected: str
) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    assert _run("markup-set", _d(prose), "--engine", "fake", "--convention", convention) == 2
    assert _run("markup-set", _d(prose), "--engine", "fake", "--convention", convention, "--verified-by-listening") == 0
    engine = FakeEngine()
    install_engine(engine)
    assert _render(prose) == 0
    assert engine.received == [expected]


def test_targeted_edit_regenerates_only_its_chunk_and_marks_the_master_stale(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine())
    assert _render(book) == 0
    assert _run("assemble", _d(book), "--chapter", "c001", "--target", "short") == 0
    capsys.readouterr()
    assert _run("status", _d(book), "--chapter", "c001") == 0
    assert "STALE" not in capsys.readouterr().out

    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Утром", "--yes") == 0
    engine = FakeEngine()
    install_engine(engine)
    assert _render(book) == 0
    assert engine.received == [EVENING_CORRECTED]
    capsys.readouterr()
    assert _run("status", _d(book), "--chapter", "c001") == 0
    assert "c001 chapter master: STALE" in capsys.readouterr().out

    assert _run("assemble", _d(book), "--chapter", "c001", "--target", "short") == 0
    capsys.readouterr()
    assert _run("status", _d(book), "--chapter", "c001") == 0
    assert "STALE" not in capsys.readouterr().out


def test_revert_restores_the_previous_narration_and_regenerates_only_that_chunk(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    assert _render(book) == 0
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Утром", "--yes") == 0
    install_engine(FakeEngine())
    assert _render(book) == 0

    assert _run("revert", _d(book), "1") == 0
    engine = FakeEngine()
    install_engine(engine)
    assert _render(book) == 0
    assert _narration(book)["c001-s0004"] == EVENING_ORIGINAL
    assert engine.received == [EVENING_ORIGINAL]


def test_history_is_append_only_and_revert_is_recorded(book: BookProject) -> None:
    assert _run("dict", "add", _d(book), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Утром", "--yes") == 0
    assert _run("dict", "remove", _d(book), "d001") == 0
    before = _history(book)
    assert _run("revert", _d(book), "2") == 0
    after = _history(book)

    assert after[:3] == before
    assert [record["action"] for record in after] == ["add", "add", "remove", "revert"]
    assert after[3]["reverts_seq"] == 2
    assert after[3]["seq"] == 4
    assert json.loads((book.directory / OVERRIDES_FILE).read_text(encoding="utf-8"))["entries"] == []


def test_configuration_change_invalidates_every_chunk_even_with_corrections(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(precision="bf16"))
    assert _render(book) == 0
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Утром", "--yes") == 0
    capsys.readouterr()
    refused = FakeEngine(precision="fp16")
    install_engine(refused)
    assert _render(book) == 2
    assert "2 chunk(s) will be invalidated" in capsys.readouterr().err
    assert refused.received == []

    accepted = FakeEngine(precision="fp16")
    install_engine(accepted)
    assert _render(book, "--accept-config-change") == 0
    assert accepted.received == [CHUNK_ONE, EVENING_CORRECTED]


def test_corrections_follow_the_text_across_a_chapter_split(book: BookProject) -> None:
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Веч+ером", "--yes") == 0
    split_chapter(book, "c001-s0004", "Вечер")
    narration = _narration(book)
    assert narration["c002-s0001"] == EVENING_HINT
    overrides = json.loads((book.directory / OVERRIDES_FILE).read_text(encoding="utf-8"))["entries"]
    assert [item["segment_id"] for item in overrides] == ["c002-s0001"]


def test_stale_override_is_reported_and_not_applied(book: BookProject, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Веч+ером", "--yes") == 0
    source_path = book.directory / SOURCE_FILE
    document = json.loads(source_path.read_text(encoding="utf-8"))
    document["chapters"][0]["segments"][3]["text"] = document["chapters"][0]["segments"][3]["text"].replace("Вечером", "Утром")
    source_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    capsys.readouterr()
    assert _run("diff", _d(book)) == 0
    out = capsys.readouterr().out
    assert "stale override o001 (c001-s0004): source text at this span is no longer 'Вечером'" in out
    assert "Веч+ером" not in out


def test_regenerate_writes_a_join_preview_with_neighbours_and_pauses(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine())
    assert _render(book) == 0
    assert _run("override", "add", _d(book), "--segment", "c001-s0004", "--find", "Вечером",
                "--replacement", "Утром", "--yes") == 0
    engine = FakeEngine()
    install_engine(engine)
    capsys.readouterr()
    assert _run("regenerate", _d(book), "--chapter", "c001", "--engine", "fake", "--target", "short") == 0
    assert engine.received == [EVENING_CORRECTED]
    preview = book.directory / "review" / "joins" / "c001-k0002.wav"
    assert f"join preview: {preview}" in capsys.readouterr().out
    info = sf.info(str(preview))
    # previous chunk (0.5 s) + scene pause (1.5 s) + regenerated chunk (0.5 s); no following chunk
    assert info.duration == pytest.approx(0.5 + 1.5 + 0.5, abs=1e-3)


def test_markup_test_renders_plain_plus_and_acute_variants(
    book: BookProject, profile_dir: Path, install_engine: Callable[[FakeEngine], None], tmp_path: Path
) -> None:
    engine = FakeEngine()
    install_engine(engine)
    out = tmp_path / "markup"
    assert _run("markup-test", _d(book), "--engine", "fake", "--narrator", str(profile_dir), "--out", str(out)) == 0
    assert engine.received == [
        "Старый замок стоял на высоком холме.",
        "Старый зам+ок стоял на высоком холме.",
        "Старый замо́к стоял на высоком холме.",
    ]
    assert sorted(path.name for path in out.iterdir()) == ["markup-acute.wav", "markup-plain.wav", "markup-plus.wav"]
