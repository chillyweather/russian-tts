from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.acceptance.network import network_state
from rutts.book.chunking import limits_for_target, plan_chunks
from rutts.book.project import BookProject, install_narrator, load_project, read_json, write_json
from rutts.book.text_import import import_text
from rutts.cli import main
from rutts.engines import base
from rutts.verification import asr
from tests.conftest import FakeEngine, FakeTranscriber
from tests.test_book_job import BOOK_TEXT, calm_machine  # noqa: F401  (shared fixtures)
from tests.test_book_render import profile_dir  # noqa: F401  (shared fixture)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
SECTIONS = ("configuration", "offline", "coverage", "listening", "exercises", "resources", "output")


def _d(project: BookProject) -> str:
    return str(project.directory)


def _record(project: BookProject) -> dict[str, Any]:
    """The record as evaluated from the disk now (status refreshes and saves it)."""
    main(["book", "accept", "status", _d(project)])
    return read_json(project.directory / "acceptance" / "record.json")


@pytest.fixture
def book(tmp_path: Path, profile_dir: Path) -> BookProject:  # noqa: F811
    source = tmp_path / "book.txt"
    source.write_text(BOOK_TEXT.split("Глава 4.")[0], encoding="utf-8")
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
    def install(heard: dict[str, str] | None = None) -> FakeTranscriber:
        instance = FakeTranscriber(heard or {})
        monkeypatch.setitem(asr.TRANSCRIBER_FACTORIES, "fake-asr", lambda: instance)
        return instance

    return install


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def set_state(state: str) -> None:
        monkeypatch.setattr("rutts.acceptance.cli.NETWORK_STATE", lambda: state)

    set_state("offline")
    return set_state


def _generate_and_verify(book: BookProject, engine: FakeEngine, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber], chapters: tuple[str, ...] = ("c001", "c002", "c003")) -> None:
    install_engine(engine)
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 0
    for chapter_id in chapters:
        assert main(["book", "verify", _d(book), "--chapter", chapter_id, "--transcriber", "fake-asr"]) == 0
        assert main(["book", "assemble", _d(book), "--chapter", chapter_id, "--target", "short"]) == 0


def _first_chunk_text(project: BookProject, chapter_id: str) -> str:
    return plan_chunks(project.planning_segments(chapter_id), "short", limits_for_target("short"))[0].text


def _section(project: BookProject, name: str) -> dict[str, Any]:
    return _record(project)["sections"][name]  # type: ignore[no-any-return]


SCORES = {"naturalness": 4, "sustained_comfort": 4, "pronunciation": 4, "completeness": 4, "voice_continuity": 4}


def test_network_state_is_offline_without_a_route_and_online_when_the_probe_connects() -> None:
    attempts: list[tuple[str, int, float]] = []

    def connect(host: str, port: int, timeout: float) -> None:
        attempts.append((host, port, timeout))

    assert network_state(route=lambda: False, connect=connect) == "offline"
    assert attempts == []
    assert network_state(route=lambda: True, connect=connect) == "online"
    assert attempts == [("1.1.1.1", 443, 2.0)]
    assert network_state(route=lambda: None, connect=connect) == "unknown"


def test_network_state_is_offline_when_the_connect_attempt_fails() -> None:
    def refused(host: str, port: int, timeout: float) -> None:
        raise ConnectionRefusedError("refused")

    assert network_state(route=lambda: True, connect=refused) == "offline"


def test_fresh_project_reports_every_section_not_evidenced_and_overall_incomplete(
    book: BookProject, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["book", "accept", "status", _d(book)]) == 1
    record = _record(book)
    assert {name: record["sections"][name]["status"] for name in SECTIONS} == {name: "not_evidenced" for name in SECTIONS}
    assert record["overall"] == "incomplete"
    out = capsys.readouterr().out
    for name in SECTIONS:
        assert f"{name}: not_evidenced" in out
    assert "overall: incomplete" in out


def test_step_recorded_while_online_is_kept_but_does_not_count_as_offline_evidence(
    book: BookProject, network: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
    network("online")
    assert main(["book", "accept", "step", _d(book), "import"]) == 0
    assert "does not count as offline evidence" in capsys.readouterr().out
    record = _record(book)
    assert record["inputs"]["offline"][0]["network"] == "online"
    assert _section(book, "offline")["status"] == "not_evidenced"
    assert record["sections"]["offline"]["evidence"]["import"]["evidenced"] is False


def test_step_counts_as_offline_evidence_only_when_the_probe_said_offline(
    book: BookProject, network: Callable[[str], None]
) -> None:
    assert main(["book", "accept", "step", _d(book), "import"]) == 0
    evidence = _section(book, "offline")["evidence"]
    assert evidence["import"]["evidenced"] is True
    assert evidence["import"]["offline_observations"] == 1


def test_step_is_refused_when_the_stage_artifacts_are_missing(book: BookProject, network: Callable[[str], None]) -> None:
    assert main(["book", "accept", "step", _d(book), "generation"]) == 2
    assert not (book.directory / "acceptance" / "record.json").is_file()


def test_step_is_refused_when_the_stage_is_no_longer_current(
    book: BookProject,
    network: Callable[[str], None],
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert main(["book", "accept", "step", _d(book), "generation"]) == 0
    assert main(["book", "dict", "add", _d(book), "--pattern", "улице", "--replacement", "улицы", "--yes"]) == 0
    assert main(["book", "accept", "step", _d(book), "generation"]) == 2


def test_configuration_fails_with_an_unresolved_revision(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    job = read_json(book.directory / "job" / "job.json")
    job["engine"]["model_revision"] = "main"
    write_json(book.directory / "job" / "job.json", job)
    assert main(["book", "accept", "status", _d(book)]) == 1
    section = _section(book, "configuration")
    assert section["status"] == "failed"
    assert any("model revision is unresolved" in item for item in section["missing"])


def test_configuration_fails_with_empty_narrator_provenance(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    job = read_json(book.directory / "job" / "job.json")
    job["narrator"]["provenance"] = ""
    write_json(book.directory / "job" / "job.json", job)
    assert main(["book", "accept", "status", _d(book)]) == 1
    assert _section(book, "configuration")["status"] == "failed"


def test_configuration_is_evidenced_with_revision_licences_and_provenance(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert main(["book", "accept", "status", _d(book)]) == 1
    evidence = _section(book, "configuration")["evidence"]
    assert _section(book, "configuration")["status"] == "evidenced"
    assert evidence["engine"]["model_revision"] == "0" * 40
    assert evidence["engine"]["license"] == "MIT"
    assert evidence["verifier"][0]["license"] == "MIT"
    assert evidence["narrator"]["provenance"] == "Test recording, public domain"


def test_coverage_lists_exclusions_and_names_unreviewed_chapters(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    assert main(["book", "exclude", _d(book), "--chapter", "c003", "--reason", "Appendix not narrated"]) == 0
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber, ("c001", "c002"))
    assert main(["book", "accept", "status", _d(book)]) == 1
    evidence = _section(book, "coverage")["evidence"]
    assert {"chapter_id": "c003", "reason": "Appendix not narrated"} in evidence["excluded_chapters"]
    assert any(item["chapter_id"] == "c003" and item["reason"] == "Appendix not narrated" for item in evidence["excluded_segments"])
    assert evidence["unreviewed_chapters"] == ["c001", "c002"]
    assert evidence["fully_listened_chapters"] == []
    assert "Passing a sampled chapter does not prove later chapters are error-free." in evidence["disclosure"]


def test_coverage_is_not_evidenced_and_names_uncovered_segments(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    install_engine(FakeEngine(amplitude=0.02, spike=0.95, fail_texts=frozenset({_first_chunk_text(book, "c002")})))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short"]) == 1
    assert main(["book", "accept", "status", _d(book)]) == 1
    section = _section(book, "coverage")
    assert section["status"] == "not_evidenced"
    assert section["evidence"]["uncovered_segments"]
    assert any(item.startswith("uncovered segment") for item in section["missing"])


def test_listening_scores_are_typed_validated_and_never_defaulted(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001"]) == 2
    assert not (book.directory / "acceptance" / "record.json").is_file()
    monkeypatch.setattr("sys.stdin", io.StringIO("6\n5\n4\n4\n5\n4\ny\nно хорошо\n"))
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001"]) == 0
    listening = _record(book)["inputs"]["listening"]
    assert listening["scores"] == {"naturalness": 5, "sustained_comfort": 4, "pronunciation": 4, "completeness": 5, "voice_continuity": 4}
    assert listening["notes"] == "но хорошо"
    assert "1 to 5" in capsys.readouterr().out


def test_listening_gates_fail_below_thresholds_and_scores_file_is_hashed(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    tmp_path: Path,
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    low = tmp_path / "scores-low.json"
    low.write_text(json.dumps({"scores": {**SCORES, "naturalness": 3}, "listened_whole_chapter": True, "notes": ""}), encoding="utf-8")
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001", "--scores-file", str(low)]) == 0
    assert _section(book, "listening")["status"] == "failed"
    assert "naturalness 3 is below 4" in _section(book, "listening")["missing"]

    passing = tmp_path / "scores.json"
    passing.write_text(json.dumps({"scores": SCORES, "listened_whole_chapter": True, "notes": "ok"}), encoding="utf-8")
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001", "--scores-file", str(passing)]) == 0
    section = _section(book, "listening")
    assert section["status"] == "evidenced"
    copied = section["evidence"]["copied_scores"]
    assert copied["sha256"] == hashlib.sha256(passing.read_bytes()).hexdigest()
    assert copied["source_name"] == "scores.json"


def test_listening_scores_file_rejects_out_of_range_values(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    tmp_path: Path,
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scores": {**SCORES, "naturalness": 9}, "listened_whole_chapter": True}), encoding="utf-8")
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001", "--scores-file", str(bad)]) == 2
    assert not (book.directory / "acceptance" / "record.json").is_file()


def test_listening_without_whole_chapter_confirmation_is_not_evidenced(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    tmp_path: Path,
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"scores": SCORES, "listened_whole_chapter": False}), encoding="utf-8")
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001", "--scores-file", str(partial)]) == 0
    assert _section(book, "listening")["status"] == "not_evidenced"


def _exercise(project: BookProject, name: str) -> dict[str, Any]:
    return _record(project)["sections"]["exercises"]["evidence"][name]  # type: ignore[no-any-return]


def _interrupted_generation(book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]) -> None:
    install_engine(FakeEngine(amplitude=0.02, spike=0.95, interrupt_on=frozenset({2})))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 130


def test_correction_exercise_is_evidenced_only_after_targeted_regeneration(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert _exercise(book, "pronunciation_correction")["evidenced"] is False
    assert main(["book", "dict", "add", _d(book), "--pattern", "улице", "--replacement", "улицы", "--yes"]) == 0
    assert _exercise(book, "pronunciation_correction")["evidenced"] is False
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    assert main(["book", "regenerate", _d(book), "--chapter", "c002", "--engine", "fake", "--target", "short"]) == 0
    assert main(["book", "verify", _d(book), "--chapter", "c002", "--transcriber", "fake-asr"]) == 0
    evidence = _exercise(book, "pronunciation_correction")
    assert evidence["evidenced"] is True
    assert evidence["regenerated_chunks"] and evidence["regenerated_chunks"][0].startswith("c002-")
    assert evidence["regenerated_chunks"] and len(evidence["regenerated_chunks"]) < evidence["total_chunks"]


def test_interruption_exercise_is_evidenced_only_after_a_restart_keeps_valid_chunks(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    _interrupted_generation(book, install_engine, transcriber)
    assert main(["book", "accept", "status", _d(book)]) == 1
    assert _exercise(book, "interruption_and_restart")["evidenced"] is False
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 0
    pairs = _exercise(book, "interruption_and_restart")["pairs"]
    assert pairs[0]["interrupted_status"] == "interrupted"
    assert pairs[0]["valid_chunks_at_restart"] >= pairs[0]["valid_chunks_at_interruption"] == 1
    assert _exercise(book, "interruption_and_restart")["evidenced"] is True


def test_export_rebuild_exercise_is_evidenced_only_when_a_later_export_supersedes_an_earlier_one(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert main(["book", "export", _d(book)]) == 0
    assert _exercise(book, "export_rebuilt_after_change")["evidenced"] is False
    assert main(["book", "dict", "add", _d(book), "--pattern", "улице", "--replacement", "улицы", "--yes"]) == 0
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    assert main(["book", "regenerate", _d(book), "--chapter", "c002", "--engine", "fake", "--target", "short"]) == 0
    assert main(["book", "verify", _d(book), "--chapter", "c002", "--transcriber", "fake-asr"]) == 0
    assert main(["book", "assemble", _d(book), "--chapter", "c002", "--target", "short"]) == 0
    assert main(["book", "export", _d(book)]) == 0
    assert _exercise(book, "export_rebuilt_after_change")["evidenced"] is True


def test_a_memory_stop_that_was_not_recovered_fails_the_resources_section(
    book: BookProject, install_engine: Callable[[FakeEngine], None], transcriber: Callable[..., FakeTranscriber]
) -> None:
    install_engine(FakeEngine(amplitude=0.02, spike=0.95, memory_error_on=frozenset({1})))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short"]) == 3
    assert main(["book", "accept", "status", _d(book)]) == 1
    section = _section(book, "resources")
    assert section["status"] == "failed"
    assert section["evidence"]["unrecovered_stop_invocations"] == [1]
    assert section["evidence"]["resource_stops"][0]["reasons"][0].startswith("engine memory exhaustion")


def _full_flow(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
    network: Callable[[str], None],
) -> None:
    """Interrupt and restart a generation, correct one pronunciation, rebuild the export, listen, then confirm playback."""
    _interrupted_generation(book, install_engine, transcriber)
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    transcriber()
    assert main(["book", "generate", _d(book), "--engine", "fake", "--target", "short", "--assemble"]) == 0
    for chapter_id in ("c001", "c002", "c003"):
        assert main(["book", "verify", _d(book), "--chapter", chapter_id, "--transcriber", "fake-asr"]) == 0
        assert main(["book", "assemble", _d(book), "--chapter", chapter_id, "--target", "short"]) == 0
    assert main(["book", "export", _d(book)]) == 0
    assert main(["book", "dict", "add", _d(book), "--pattern", "улице", "--replacement", "улицы", "--yes"]) == 0
    install_engine(FakeEngine(amplitude=0.02, spike=0.95))
    assert main(["book", "regenerate", _d(book), "--chapter", "c002", "--engine", "fake", "--target", "short"]) == 0
    assert main(["book", "verify", _d(book), "--chapter", "c002", "--transcriber", "fake-asr"]) == 0
    assert main(["book", "assemble", _d(book), "--chapter", "c002", "--target", "short"]) == 0
    assert main(["book", "export", _d(book)]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("5\n4\n4\n5\n4\ny\nсветло и ровно\n"))
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001"]) == 0
    network("offline")
    for stage in ("import", "preparation", "generation", "verification", "correction", "export"):
        assert main(["book", "accept", "step", _d(book), stage]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("1.0\ny\ny\ny\ny\n"))
    assert main(["book", "accept", "playback", _d(book), "--player", "IINA"]) == 0


@requires_ffmpeg
def test_fully_exercised_flow_reaches_accepted_and_the_report_states_the_sampling_limit(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
    network: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_flow(book, install_engine, transcriber, monkeypatch, network)
    assert main(["book", "accept", "status", _d(book)]) == 0
    assert "overall: accepted" in capsys.readouterr().out
    record = _record(book)
    assert record["overall"] == "accepted"
    assert {name: record["sections"][name]["status"] for name in SECTIONS} == {name: "evidenced" for name in SECTIONS}
    assert main(["book", "accept", "report", _d(book)]) == 0
    report = (book.directory / "acceptance" / "acceptance.md").read_text(encoding="utf-8")
    assert "Overall result: **accepted**" in report
    assert "Passing a sampled chapter does not prove later chapters are error-free." in report
    assert "Unreviewed chapters: c002, c003" in report
    assert main(["book", "check-portable", _d(book)]) == 0


@requires_ffmpeg
def test_playback_must_be_confirmed_before_output_is_evidenced(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
    network: Callable[[str], None],
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert main(["book", "export", _d(book)]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("1.0\ny\ny\nn\ny\n"))
    assert main(["book", "accept", "playback", _d(book), "--player", "IINA"]) == 0
    assert _section(book, "output")["status"] == "not_evidenced"
    assert _record(book)["sections"]["output"]["evidence"]["validation"]["ok"] is True
    monkeypatch.setattr("sys.stdin", io.StringIO("1.0\ny\ny\ny\ny\n"))
    assert main(["book", "accept", "playback", _d(book), "--player", "IINA"]) == 0
    assert _section(book, "output")["status"] == "evidenced"


def test_export_step_and_playback_are_refused_without_a_ready_export(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    assert main(["book", "accept", "step", _d(book), "export"]) == 2
    monkeypatch.setattr("sys.stdin", io.StringIO("1.0\ny\ny\ny\ny\n"))
    assert main(["book", "accept", "playback", _d(book), "--player", "IINA"]) == 2
    assert not (book.directory / "acceptance" / "record.json").is_file()


def test_listening_review_goes_stale_when_the_chapter_audio_changes(
    book: BookProject,
    install_engine: Callable[[FakeEngine], None],
    transcriber: Callable[..., FakeTranscriber],
    tmp_path: Path,
) -> None:
    _generate_and_verify(book, FakeEngine(amplitude=0.02, spike=0.95), install_engine, transcriber)
    passing = tmp_path / "scores.json"
    passing.write_text(json.dumps({"scores": SCORES, "listened_whole_chapter": True}), encoding="utf-8")
    assert main(["book", "accept", "listen", _d(book), "--chapter", "c001", "--scores-file", str(passing)]) == 0
    assert _section(book, "listening")["status"] == "evidenced"
    install_engine(FakeEngine(amplitude=0.03, spike=0.95))
    assert main(["book", "render", _d(book), "--chapter", "c001", "--engine", "fake", "--target", "short", "--force-new-run"]) == 0
    assert _section(book, "listening")["status"] == "not_evidenced"
    assert "listen again" in " ".join(_section(book, "listening")["missing"])
