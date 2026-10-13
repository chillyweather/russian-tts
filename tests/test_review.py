from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

from rutts.book.project import BookProject, install_narrator, load_project, sha256_file, write_json
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.verification import asr
from rutts.verification.asr import Transcript, TranscriberConfig
from tests.conftest import FakeEngine, FakeTranscriber
from tests.test_book_render import profile_dir  # noqa: F401  (shared fixture)

SENTENCE = "Анна открыла окно, и ёлка во дворе горела огнями. Всё было тихо."
PROSE = SENTENCE + "\n"
HEARD_WITHOUT_ANNA = "открыла окно, и ёлка во дворе горела огнями. Всё было тихо."


class ScriptedTranscriber(FakeTranscriber):
    """Reports the scripted transcript for each successive transcription of a chunk, then echoes the chunk text."""

    def __init__(self, scripts: list[str], revision: str) -> None:
        super().__init__({})
        self.scripts = list(scripts)
        self.revision = revision

    @property
    def config(self) -> TranscriberConfig:
        return replace(super().config, model_revision=self.revision)

    def transcribe(self, path: Path) -> Transcript:
        self.heard = {path.stem: self.scripts.pop(0)} if self.scripts else {}
        return super().transcribe(path)


@pytest.fixture
def book(tmp_path: Path, profile_dir: Path, monkeypatch: pytest.MonkeyPatch) -> BookProject:  # noqa: F811
    source = tmp_path / "source.txt"
    source.write_text(PROSE, encoding="utf-8")
    project = import_text(source, tmp_path / "book", title="Книга", author="", chapter_regex=None)
    install_narrator(project, profile_dir)
    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: FakeEngine(speech_rate=14.0))
    assert main(["book", "render", str(project.directory), "--chapter", "c001", "--engine", "fake", "--target", "short"]) == 0
    return load_project(project.directory)


@pytest.fixture
def transcribe(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ScriptedTranscriber]:
    def install(scripts: list[str], revision: str = "1" * 40) -> ScriptedTranscriber:
        transcriber = ScriptedTranscriber(scripts, revision)
        monkeypatch.setitem(asr.TRANSCRIBER_FACTORIES, "scripted-asr", lambda **_: transcriber)
        return transcriber

    return install


def _d(project: BookProject) -> str:
    return str(project.directory)


def _review(*args: str) -> int:
    return main(["book", "review", *args])


def _verify(project: BookProject) -> int:
    return main(["book", "verify", _d(project), "--chapter", "c001", "--transcriber", "scripted-asr"])


def _chunk_id(project: BookProject) -> str:
    plan: dict[str, Any] = json.loads((project.target_dir("c001", "short") / "plan.json").read_text(encoding="utf-8"))
    return str(plan["chunks"][0]["id"])


def _list_json(project: BookProject, capsys: pytest.CaptureFixture[str], *extra: str) -> list[dict[str, Any]]:
    capsys.readouterr()
    assert _review("list", _d(project), "--json", *extra) == 0
    items: list[dict[str, Any]] = json.loads(capsys.readouterr().out)
    return items


def test_list_reports_source_narration_audio_reason_and_history(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    chunk_id = _chunk_id(book)
    transcribe([HEARD_WITHOUT_ANNA])
    assert _verify(book) == 0

    items = _list_json(book, capsys)

    assert len(items) == 1
    item = items[0]
    assert item["type"] == "omission"
    assert item["status"] == "open"
    assert item["chunk_id"] == chunk_id
    assert item["expected"] == "Анна"
    assert item["heard"] == ""
    assert item["source_text"] == SENTENCE
    assert item["narration_text"] == SENTENCE
    assert item["audio_file"] == f"{chunk_id}.wav"
    assert item["history"]["attempts"][0]["status"] == "ok"
    assert "Анна" in item["reason"]

    assert _review("list", _d(book)) == 0
    text = capsys.readouterr().out
    for label in ("source:", "narration:", "audio:", "reason:", "history:"):
        assert label in text


def _item_ids(project: BookProject, capsys: pytest.CaptureFixture[str], *extra: str) -> list[str]:
    return [str(item["id"]) for item in _list_json(project, capsys, *extra)]


def _only_item(project: BookProject, capsys: pytest.CaptureFixture[str], *extra: str) -> dict[str, Any]:
    items = _list_json(project, capsys, *extra)
    assert len(items) == 1, items
    return items[0]


def _flag_omission(project: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]) -> str:
    transcribe([HEARD_WITHOUT_ANNA])
    assert _verify(project) == 0
    return str(_only_item(project, capsys)["id"])


def _override_yo(project: BookProject) -> int:
    return main(["book", "override", "add", _d(project), "--segment", "c001-s0001", "--find", "ёлка",
                 "--replacement", "ёлочка", "--yes"])


def _regenerate(project: BookProject) -> int:
    return main(["book", "regenerate", _d(project), "--chapter", "c001", "--engine", "fake", "--target", "short"])


def _touch_audio(project: BookProject, chunk_id: str) -> None:
    """Change one chunk's audio the way a re-render would, and record the new hash in the manifest."""
    directory = project.target_dir("c001", "short")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["chunks"] if item["id"] == chunk_id)
    path = directory / record["audio_file"]
    samples, rate = sf.read(str(path), dtype="float32")
    sf.write(str(path), samples * 0.5, rate, subtype="FLOAT")
    record["audio_sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)


def test_show_prints_follow_up_commands_for_the_chunk(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    capsys.readouterr()
    assert _review("show", _d(book), item_id) == 0
    out = capsys.readouterr().out
    assert "reason: omission" in out
    assert f"rutts book regenerate {_d(book)} --chapter c001 --engine fake --target short" in out
    assert f"rutts book review recheck {_d(book)} --chapter c001 --target short" in out
    assert "--segment c001-s0001" in out


def test_play_writes_a_clip_with_context_and_never_starts_a_real_player(
    book: BookProject,
    transcribe: Callable[..., ScriptedTranscriber],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    played: list[list[str]] = []
    monkeypatch.setattr("rutts.review.playback.PLAYER", lambda command: played.append(command))
    item_id = _flag_omission(book, transcribe, capsys)
    capsys.readouterr()
    manual = _review("flag", _d(book), "--chunk", _chunk_id(book), "--kind", "prosody", "--note", "rushed",
                     "--time", "1.0")
    assert manual == 0
    capsys.readouterr()

    assert _review("play", _d(book), "m001", "--no-play") == 0
    clip = book.directory / "review" / "clips" / "m001.wav"
    assert str(clip) in capsys.readouterr().out
    info = sf.info(str(clip))
    assert info.duration == pytest.approx(2.5, abs=0.001)  # 1.0 s point, 1.5 s context each side
    assert played == []

    assert _review("play", _d(book), item_id, "--join", "--no-play") == 0
    join = book.directory / "review" / "clips" / f"{item_id}-join.wav"
    assert join.is_file()
    assert played == []

    assert _review("play", _d(book), "m001") == 0
    assert played == [["afplay", str(clip)]]


def test_resolve_and_dismiss_require_a_non_empty_reason(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _review("resolve", _d(book), item_id, "--reason", "   ") == 2
    assert _review("dismiss", _d(book), item_id, "--reason", "") == 2
    assert capsys.readouterr().err.count("non-empty --reason") == 2
    assert _item_ids(book, capsys) == [item_id]

    assert _review("resolve", _d(book), item_id, "--reason", "Heard correctly by ear on the full chapter") == 0
    assert _item_ids(book, capsys) == []
    listed = _list_json(book, capsys, "--all")
    assert [(item["id"], item["status"]) for item in listed] == [(item_id, "resolved")]
    log = json.loads((book.directory / "review" / "decisions-log.json").read_text(encoding="utf-8"))["entries"]
    assert [entry["action"] for entry in log] == ["resolved"]


def test_flag_correction_regenerate_recheck_removes_the_item(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _override_yo(book) == 0
    assert _regenerate(book) == 0

    waiting = _only_item(book, capsys)
    assert waiting["id"] == item_id
    assert waiting["current"] is False

    capsys.readouterr()
    assert _review("recheck", _d(book), "--chapter", "c001", "--transcriber", "scripted-asr") == 0
    out = capsys.readouterr().out
    assert f"disappeared: {item_id}" in out
    assert "appeared: none" in out
    assert _item_ids(book, capsys) == []


HEARD_WITH_YOCHKA_WITHOUT_ANNA = "открыла окно, и ёлочка во дворе горела огнями. Всё было тихо."


def _resolve(project: BookProject, item_id: str) -> int:
    return _review("resolve", _d(project), item_id, "--reason", "Checked by ear; the name is clear in the audio")


def test_resolved_item_becomes_stale_when_the_chunk_text_changes(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _resolve(book, item_id) == 0
    assert _override_yo(book) == 0
    assert _regenerate(book) == 0
    transcribe([HEARD_WITH_YOCHKA_WITHOUT_ANNA])
    assert _review("recheck", _d(book), "--chapter", "c001", "--transcriber", "scripted-asr") == 0

    item = _only_item(book, capsys)
    assert item["id"] != item_id
    assert item["status"] == "stale"
    assert item["decision"]["state"] == "resolved"
    assert item["decision"]["reason"] == "Checked by ear; the name is clear in the audio"


def test_resolved_item_becomes_stale_when_its_audio_is_regenerated(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _resolve(book, item_id) == 0
    chunk_id = _chunk_id(book)
    _touch_audio(book, chunk_id)
    transcribe([HEARD_WITHOUT_ANNA])
    assert _review("recheck", _d(book), "--chapter", "c001", "--transcriber", "scripted-asr") == 0

    item = _only_item(book, capsys, "--all")
    assert item["status"] == "stale"
    assert item["decision"]["audio_sha256"] != item["audio_sha256"]


def test_decision_stays_valid_until_the_verifier_configuration_changes(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _resolve(book, item_id) == 0
    transcribe([])
    assert _review("recheck", _d(book), "--chapter", "c001", "--transcriber", "scripted-asr") == 0
    assert _only_item(book, capsys, "--all")["status"] == "resolved"

    transcribe([HEARD_WITHOUT_ANNA], revision="2" * 40)
    assert _review("recheck", _d(book), "--chapter", "c001", "--transcriber", "scripted-asr") == 0
    item = _only_item(book, capsys, "--all")
    assert item["status"] == "stale"
    assert item["decision"]["verifier_fingerprint"] != item["verifier_fingerprint"]


def test_manual_finding_stays_open_until_closed_and_flags_changed_audio(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    transcribe([])
    assert _verify(book) == 0
    chunk_id = _chunk_id(book)
    assert _review("flag", _d(book), "--chunk", chunk_id, "--kind", "stress", "--word", "открыла",
                   "--note", "stress on the verb") == 0
    item = _only_item(book, capsys)
    assert (item["id"], item["origin"], item["type"], item["status"]) == ("m001", "manual", "stress", "open")
    assert item["expected"] == "открыла"
    assert item["needs_relisten"] is False

    _touch_audio(book, chunk_id)
    item = _only_item(book, capsys)
    assert item["status"] == "open"
    assert item["needs_relisten"] is True

    assert _review("resolve", _d(book), "m001", "--reason", "Relistened after the re-render") == 0
    assert _item_ids(book, capsys) == []
    assert [(i["id"], i["status"]) for i in _list_json(book, capsys, "--all")] == [("m001", "resolved")]
    assert _review("reopen", _d(book), "m001") == 0
    assert _only_item(book, capsys)["status"] == "open"


def test_manual_word_must_be_in_the_chunk(book: BookProject, transcribe: Callable[..., ScriptedTranscriber]) -> None:
    transcribe([])
    assert _verify(book) == 0
    assert _review("flag", _d(book), "--chunk", _chunk_id(book), "--kind", "pronunciation", "--word", "нет",
                   "--note", "x") == 2


def test_failed_chunk_is_listed_blocks_export_and_cannot_be_hidden(
    book: BookProject, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _override_yo(book) == 0
    monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: FakeEngine(speech_rate=14.0, raise_on=frozenset({1, 2, 3})))
    assert _regenerate(book) == 1

    item = _only_item(book, capsys)
    assert item["type"] == "content_failure"
    assert item["history"]["attempts"][-1]["status"] == "failed"
    assert "failed after 3 attempt(s)" in item["reason"]

    assert _review("dismiss", _d(book), item["id"], "--reason", "Will regenerate later") == 0
    assert _item_ids(book, capsys) == []
    assert _review("status", _d(book), "--target", "short") == 0
    out = capsys.readouterr().out
    assert "reviewed: no" in out
    assert f"{_chunk_id(book)}: failed chunk" in out
    assert "draft: no" in out


def test_text_changed_without_regeneration_is_a_missing_chunk(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    transcribe([])
    assert _verify(book) == 0
    assert _override_yo(book) == 0
    item = _only_item(book, capsys)
    assert item["type"] == "missing_chunk"
    assert item["chunk_id"] == _chunk_id(book)


def _assemble(project: BookProject) -> int:
    return main(["book", "assemble", _d(project), "--chapter", "c001", "--target", "short"])


def _eligibility(project: BookProject) -> Any:
    from rutts.review.eligibility import export_eligibility

    return export_eligibility(project, ["c001"], "short")


def test_reviewed_is_false_with_an_open_omission_and_draft_lists_it(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber]
) -> None:
    transcribe([HEARD_WITHOUT_ANNA])
    assert _verify(book) == 0
    assert _assemble(book) == 0
    result = _eligibility(book)
    assert result.reviewed.eligible is False
    assert any("open omission" in reason for reason in result.reviewed.reasons)
    assert result.draft.eligible is True
    assert any("open omission" in reason for reason in result.draft.reasons)


def test_reviewed_is_false_with_a_missing_chunk_even_when_dismissed(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    transcribe([])
    assert _verify(book) == 0
    assert _override_yo(book) == 0
    item = _only_item(book, capsys)
    assert _review("dismiss", _d(book), item["id"], "--reason", "not needed") == 0
    result = _eligibility(book)
    assert result.reviewed.eligible is False
    assert any("missing chunk" in reason for reason in result.reviewed.reasons)


def test_reviewed_is_false_with_unverified_audio(book: BookProject) -> None:
    result = _eligibility(book)
    assert result.reviewed.eligible is False
    assert any("not verified" in reason for reason in result.reviewed.reasons)
    assert result.draft.eligible is True


def test_reviewed_is_false_with_a_stale_master(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber]
) -> None:
    transcribe([])
    assert _verify(book) == 0
    assert _assemble(book) == 0
    assert _eligibility(book).reviewed.eligible is True
    _touch_audio(book, _chunk_id(book))
    assert _verify(book) == 0
    result = _eligibility(book)
    assert result.reviewed.eligible is False
    assert any("master is stale" in reason for reason in result.reviewed.reasons)


def test_reviewed_is_true_when_every_finding_is_resolved(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    item_id = _flag_omission(book, transcribe, capsys)
    assert _resolve(book, item_id) == 0
    assert _assemble(book) == 0
    result = _eligibility(book)
    assert result.reviewed.eligible is True
    assert result.reviewed.reasons == ()


def test_draft_lists_unresolved_items_and_status_prints_both_states(
    book: BookProject, transcribe: Callable[..., ScriptedTranscriber], capsys: pytest.CaptureFixture[str]
) -> None:
    _flag_omission(book, transcribe, capsys)
    capsys.readouterr()
    assert _review("status", _d(book)) == 0
    out = capsys.readouterr().out
    assert "c001 (short): open 1, stale 0, resolved 0, dismissed 0" in out
    assert "reviewed: no" in out
    assert "draft: yes" in out
