from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.book.project import BookProject, install_narrator, load_project
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.pronunciation.store import HISTORY_FILE, OVERRIDES_FILE
from tests.conftest import FakeAssistant, FakeEngine
from tests.test_book_render import book, profile_dir  # noqa: F401  (shared fixtures)

PARA = "Старый замок стоял на холме. Анна заперла замок на ключ. Всё было тихо, все спали, всего хватало."
PROSE = f"Глава 1. Утро\n\n{PARA}\n"
EVENING_ORIGINAL = (
    "Вечером на ёлке во дворе горели огни. Дети шумели у подъезда.\n"
    "— Ты придёшь? — спросил Пётр.\n"
    "— Приду, — ответила Анна. — Если всё будет хорошо."
)
EVENING_SENTENCE = "Вечером на ёлке во дворе горели огни."
EVENING_SENTENCE_HINT = "Веч+ером на ёлке во дворе горели огни."
EVENING_HINT = EVENING_ORIGINAL.replace("Вечером", "Веч+ером")


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


def _suggestions(project: BookProject, chapter: str = "c001") -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (project.directory / "narration" / "suggestions" / f"{chapter}.json").read_text(encoding="utf-8")
    )
    return document


def _brief(items: list[dict[str, Any]]) -> list[tuple[str, str, str, int, int, str, str]]:
    return [
        (item["id"], item["kind"], item["segment_id"], item["start"], item["end"], item["original"], item["replacement"])
        for item in items
    ]


def _render(project: BookProject, *extra: str) -> int:
    return _run("render", _d(project), "--chapter", "c001", "--engine", "fake", "--target", "short", *extra)


def _history(project: BookProject) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads((project.directory / HISTORY_FILE).read_text(encoding="utf-8"))["entries"]
    return entries


def test_suggestions_are_proposals_with_exact_spans_and_context(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_assistant({PARA: PARA.replace("замок", "з+амок").replace("все спали", "всё спали")})
    narration_before = (prose.directory / "narration" / "narration.json").read_bytes()
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    assert (prose.directory / "narration" / "narration.json").read_bytes() == narration_before
    assert not (prose.directory / OVERRIDES_FILE).exists()
    document = _suggestions(prose)
    assert _brief(document["suggestions"]) == [
        ("c001-g001", "stress", "c001-s0002", 7, 12, "замок", "з+амок"),
        ("c001-g002", "stress", "c001-s0002", 42, 47, "замок", "з+амок"),
        ("c001-g003", "yo", "c001-s0002", 72, 75, "все", "всё"),
    ]
    assert document["suggestions"][1]["context_before"] == "… стоял на холме. Анна заперла "
    assert document["suggestions"][1]["context_after"] == " на ключ. Всё было тихо, все с…"
    assert all(item["state"] == "proposed" for item in document["suggestions"])
    assert document["assistant"]["assistant_id"] == "fake-assist"
    capsys.readouterr()
    assert _run("suggestions", _d(prose), "--chapter", "c001") == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0] == "c001-g001 stress c001-s0002 [proposed] Старый [замок → з+амок] стоял на холме. Анна заперла …"


def test_stress_scope_homographs_filters_non_homographs(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    fake_assistant({PARA: PARA.replace("холме", "х+олме")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    assert _suggestions(prose)["suggestions"] == []
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist", "--stress-scope", "all") == 0
    assert [item["original"] for item in _suggestions(prose)["suggestions"]] == ["холме"]


def test_unaligned_paragraph_produces_no_suggestions_and_is_reported(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_assistant({PARA: PARA.replace("Анна", "Анна Петровна").replace("замок", "з+амок")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    document = _suggestions(prose)
    assert document["suggestions"] == []
    assert document["unaligned"] == ["c001-s0002"]
    assert "unaligned (assistant output changed the text; no suggestions): c001-s0002" in capsys.readouterr().out


def test_user_dictionary_entry_takes_precedence_over_a_suggestion(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("dict", "add", _d(prose), "--pattern", "замок", "--replacement", "з+амок", "--yes") == 0
    fake_assistant({PARA: PARA.replace("замок", "з+амок").replace("все спали", "всё спали")})
    capsys.readouterr()
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    document = _suggestions(prose)
    assert _brief(document["suggestions"]) == [("c001-g001", "yo", "c001-s0002", 72, 75, "все", "всё")]
    assert [(item["original"], item["reason"]) for item in document["skipped_user_override"]] == [
        ("замок", "user dictionary entry"),
        ("замок", "user dictionary entry"),
    ]


def test_user_occurrence_override_takes_precedence_over_a_suggestion(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    assert _run(
        "override", "add", _d(prose), "--segment", "c001-s0002", "--find", "замок", "--occurrence", "1",
        "--replacement", "зам+ок", "--yes",
    ) == 0
    fake_assistant({PARA: PARA.replace("замок", "з+амок")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    document = _suggestions(prose)
    assert _brief(document["suggestions"]) == [("c001-g001", "stress", "c001-s0002", 42, 47, "замок", "з+амок")]
    assert document["skipped_user_override"][0]["reason"] == "user occurrence override"
    assert document["skipped_user_override"][0]["start"] == 7


def test_accept_creates_an_override_and_regenerates_only_its_chunk(
    book: BookProject, install_engine: Callable[[FakeEngine], None], fake_assistant: Callable[..., FakeAssistant],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run("markup-set", _d(book), "--engine", "fake", "--convention", "plus", "--verified-by-listening") == 0
    install_engine(FakeEngine())
    assert _render(book) == 0
    fake_assistant({EVENING_ORIGINAL: EVENING_HINT})
    assert _run("suggest", _d(book), "--chapter", "c001", "--assistant", "fake-assist", "--stress-scope", "all") == 0
    assert _brief(_suggestions(book)["suggestions"]) == [
        ("c001-g001", "stress", "c001-s0004", 0, 7, "Вечером", "Веч+ером"),
    ]
    assert _narration(book)["c001-s0004"] == EVENING_ORIGINAL
    capsys.readouterr()
    assert _run("suggestion", "accept", _d(book), "c001-g001") == 2
    assert "--yes" in capsys.readouterr().err
    assert _run("suggestion", "accept", _d(book), "c001-g001", "--yes") == 0
    record = _history(book)[-1]
    assert (record["action"], record["target"], record["entry"]["kind"]) == ("add", "override", "stress")
    assert record["entry"]["note"] == "assistant fake-assist; suggestion c001-g001"
    assert _narration(book)["c001-s0004"] == EVENING_HINT
    assert _suggestions(book)["suggestions"][0]["state"] == "accepted"
    engine = FakeEngine()
    install_engine(engine)
    assert _render(book) == 0
    assert engine.received == [EVENING_HINT]


def test_reject_records_the_decision_only(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    fake_assistant({PARA: PARA.replace("все спали", "всё спали")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    narration_before = _narration(prose)
    assert _run("suggestion", "reject", _d(prose), "c001-g001") == 0
    assert _suggestions(prose)["suggestions"][0]["state"] == "rejected"
    assert _narration(prose) == narration_before
    assert not (prose.directory / OVERRIDES_FILE).exists()


def test_revert_removes_the_override_and_restores_the_narration(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    fake_assistant({PARA: PARA.replace("все спали", "всё спали")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    original = _narration(prose)
    assert _run("suggestion", "accept", _d(prose), "c001-g001", "--yes") == 0
    assert _narration(prose)["c001-s0002"] != original["c001-s0002"]
    assert _run("suggestion", "revert", _d(prose), "c001-g001") == 0
    assert _narration(prose) == original
    assert _suggestions(prose)["suggestions"][0]["state"] == "proposed"
    assert _suggestions(prose)["suggestions"][0]["override_id"] is None
    assert [record["action"] for record in _history(prose)] == ["add", "revert"]


def test_accept_all_yo_accepts_every_proposed_yo_suggestion(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    fake_assistant({PARA: PARA.replace("все спали", "всё спали")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    assert _run("suggestion", "accept", _d(prose), "--all-yo", "--yes") == 0
    assert _narration(prose)["c001-s0002"].count("всё спали") == 1
    assert [item["state"] for item in _suggestions(prose)["suggestions"]] == ["accepted"]


def test_stale_suggestions_are_reported_and_cannot_be_accepted(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_assistant({PARA: PARA.replace("все спали", "всё спали")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    assert _run("rename", _d(prose), "--chapter", "c001", "--title", "Новое утро") == 0
    capsys.readouterr()
    assert _run("suggestions", _d(prose), "--chapter", "c001") == 0
    assert "stale" in capsys.readouterr().out
    assert _run("suggestion", "accept", _d(prose), "c001-g001", "--yes") == 2
    assert "stale" in capsys.readouterr().err
    assert _suggestions(prose)["suggestions"][0]["state"] == "proposed"


def test_stress_accept_warns_and_hint_is_stripped_for_an_engine_without_a_verified_convention(
    book: BookProject, install_engine: Callable[[FakeEngine], None], fake_assistant: Callable[..., FakeAssistant],
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_engine(FakeEngine())
    assert _render(book) == 0
    fake_assistant({EVENING_ORIGINAL: EVENING_HINT})
    assert _run("suggest", _d(book), "--chapter", "c001", "--assistant", "fake-assist", "--stress-scope", "all") == 0
    capsys.readouterr()
    assert _run("suggestion", "accept", _d(book), "c001-g001", "--yes") == 0
    output = capsys.readouterr().out
    assert "removed before synthesis" in output
    assert "markup-test" in output and "markup-set" in output
    engine = FakeEngine()
    install_engine(engine)
    assert _render(book) == 0
    assert engine.received == []


def test_listen_writes_two_clips_and_leaves_the_manifest_untouched(
    book: BookProject, install_engine: Callable[[FakeEngine], None], fake_assistant: Callable[..., FakeAssistant],
    tmp_path: Path,
) -> None:
    assert _run("markup-set", _d(book), "--engine", "fake", "--convention", "plus", "--verified-by-listening") == 0
    install_engine(FakeEngine())
    assert _render(book) == 0
    manifest = (book.target_dir("c001", "short") / "manifest.json").read_bytes()
    fake_assistant({EVENING_ORIGINAL: EVENING_HINT})
    assert _run("suggest", _d(book), "--chapter", "c001", "--assistant", "fake-assist", "--stress-scope", "all") == 0
    engine = FakeEngine()
    install_engine(engine)
    out = tmp_path / "listen"
    assert _run("suggestion", "listen", _d(book), "c001-g001", "--engine", "fake", "--out", str(out)) == 0
    assert (out / "c001-g001-without.wav").is_file()
    assert (out / "c001-g001-with.wav").is_file()
    assert engine.received == [EVENING_SENTENCE, EVENING_SENTENCE_HINT]
    assert (book.target_dir("c001", "short") / "manifest.json").read_bytes() == manifest


def test_generation_works_without_the_assistant_and_never_imports_ruaccent(
    book: BookProject, install_engine: Callable[[FakeEngine], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []

    class Recorder:
        def find_spec(self, name: str, path: object = None, target: object = None) -> None:
            if name.split(".")[0] == "ruaccent":
                attempts.append(name)
            return None

    monkeypatch.delitem(sys.modules, "ruaccent", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Recorder(), *sys.meta_path])
    engine = FakeEngine()
    install_engine(engine)
    assert _render(book) == 0
    assert engine.received
    assert attempts == []


def test_assistant_is_unloaded_when_annotation_fails(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    assistant = fake_assistant(fail_on="холме")
    with pytest.raises(RuntimeError, match="crashed"):
        _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist")
    assert (assistant.loads, assistant.unloads, assistant.loaded) == (1, 1, False)


def test_missing_ruaccent_package_gives_an_actionable_error(
    prose: BookProject, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "ruaccent", None)
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "ruaccent") == 2
    assert "pip install ruaccent" in capsys.readouterr().err
    assert not (prose.directory / "narration" / "suggestions").exists()


def test_rerunning_suggest_keeps_decisions_for_unchanged_text(
    prose: BookProject, fake_assistant: Callable[..., FakeAssistant]
) -> None:
    fake_assistant({PARA: PARA.replace("все спали", "всё спали")})
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    assert _run("suggestion", "accept", _d(prose), "c001-g001", "--yes") == 0
    assert _run("suggest", _d(prose), "--chapter", "c001", "--assistant", "fake-assist") == 0
    item = _suggestions(prose)["suggestions"][0]
    assert (item["id"], item["state"], item["override_id"]) == ("c001-g001", "accepted", "o001")
