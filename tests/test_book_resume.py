from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rutts.book.chunking import ChunkLimits, plan_chunks
from rutts.book.project import NARRATION_FILE, BookProject, write_json
from rutts.cli import main
from rutts.engines import base
from tests.conftest import FakeEngine
from tests.test_book_render import book, profile_dir  # noqa: F401  (shared fixtures)

FIXTURE_TEXT = "Утро началось рано."


@pytest.fixture
def install_engine(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeEngine], None]:
    def install(engine: FakeEngine) -> None:
        monkeypatch.setitem(base.ENGINE_FACTORIES, "fake", lambda **options: engine)

    return install


def _render_cli(book: BookProject, *extra: str) -> int:
    return main(
        ["book", "render", str(book.directory), "--chapter", "c001", "--engine", "fake", "--target", "short", *extra]
    )


def _target(book: BookProject) -> Path:
    return book.target_dir("c001", "short")


def _records(book: BookProject) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads((_target(book) / "manifest.json").read_text(encoding="utf-8"))["chunks"]
    return records


def _plan_texts(book: BookProject) -> list[str]:
    return [chunk.text for chunk in plan_chunks(book.planning_segments("c001"), "short", ChunkLimits())]


def _assert_no_unverified_ok(book: BookProject) -> None:
    from rutts.book.project import sha256_file

    for record in _records(book):
        if record["status"] == "ok":
            audio = _target(book) / record["audio_file"]
            assert audio.is_file()
            assert sha256_file(audio) == record["audio_sha256"]


def test_interrupted_chunk_is_never_marked_ok(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    engine = FakeEngine(interrupt_on=frozenset({2}))
    install_engine(engine)
    assert _render_cli(book) == 130
    records = _records(book)
    assert [record["status"] for record in records] == ["ok", "missing"]
    _assert_no_unverified_ok(book)
    assert not list(_target(book).glob(".*.tmp*"))
    assert engine.unloads == 1
    run = json.loads((_target(book) / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "interrupted"
    assert "resume" in capsys.readouterr().err


def test_restart_renders_only_the_remaining_chunks(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine(interrupt_on=frozenset({2})))
    _render_cli(book)
    texts = _plan_texts(book)
    resumed = FakeEngine()
    install_engine(resumed)
    assert _render_cli(book) == 0
    assert resumed.received == texts[1:]
    assert all(record["status"] == "ok" for record in _records(book))


def test_complete_chapter_does_not_load_the_engine(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    assert _render_cli(book) == 0
    again = FakeEngine()
    install_engine(again)
    assert _render_cli(book) == 0
    assert again.received == []
    assert again.loads == 0
    assert again.unloads == 0


def test_deleted_chunk_audio_is_regenerated_alone(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    _render_cli(book)
    texts = _plan_texts(book)
    (_target(book) / _records(book)[0]["audio_file"]).unlink()
    engine = FakeEngine()
    install_engine(engine)
    assert _render_cli(book) == 0
    assert engine.received == [texts[0]]
    assert _records(book)[0]["history"] == [{"event": "invalidated", "reason": "missing_audio"}]


def test_corrupt_chunk_audio_is_regenerated_alone(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    _render_cli(book)
    texts = _plan_texts(book)
    (_target(book) / _records(book)[1]["audio_file"]).write_bytes(b"RIFF not really audio")
    engine = FakeEngine()
    install_engine(engine)
    assert _render_cli(book) == 0
    assert engine.received == [texts[1]]
    assert _records(book)[1]["history"] == [{"event": "invalidated", "reason": "corrupt_audio"}]
    assert _records(book)[1]["status"] == "ok"


def test_edited_narration_regenerates_only_chunks_containing_it(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine())
    _render_cli(book)
    narration = book.read_narration()
    edited = next(segment for segment in narration.segments if segment.id == "c001-s0004")
    edited.text = "Вечером пошёл дождь."
    write_json(book.directory / NARRATION_FILE, narration.to_dict())
    engine = FakeEngine()
    install_engine(engine)
    assert _render_cli(book) == 0
    assert engine.received == ["Вечером пошёл дождь."]
    assert _records(book)[1]["history"] == [{"event": "invalidated", "reason": "text_changed"}]
    assert _records(book)[0]["history"] == []
    assert len(plan_chunks(book.planning_segments("c001"), "short", ChunkLimits())) == 2


def test_changed_engine_config_needs_the_flag_then_regenerates_everything(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(precision="bf16"))
    _render_cli(book)
    texts = _plan_texts(book)
    refused = FakeEngine(precision="fp16")
    install_engine(refused)
    assert _render_cli(book) == 2
    assert "2 chunk(s) will be invalidated" in capsys.readouterr().err
    assert refused.received == [] and refused.loads == 0

    accepted = FakeEngine(precision="fp16")
    install_engine(accepted)
    assert _render_cli(book, "--accept-config-change") == 0
    assert accepted.received == texts
    assert all(record["status"] == "ok" for record in _records(book))
    assert _records(book)[0]["history"] == [{"event": "invalidated", "reason": "config_changed"}]
    superseded = sorted(path.name for path in (_target(book) / "superseded").iterdir())
    assert superseded == ["c001-k0001.wav", "c001-k0002.wav"]


def test_retry_succeeds_within_budget_and_records_every_attempt(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    texts = _plan_texts(book)
    install_engine(FakeEngine(fail_attempts={texts[0]: 1}))
    assert _render_cli(book) == 0
    attempts = _records(book)[0]["attempts"]
    assert [(a["attempt"], a["status"]) for a in attempts] == [(1, "failed"), (2, "ok")]
    assert attempts[0]["error"] == "ValueError: fake transient failure on attempt 1"
    assert attempts[0]["exception_type"] == "ValueError"
    assert attempts[0]["render_seconds"] is not None
    assert isinstance(attempts[1]["resources"], dict)
    assert _records(book)[0]["needs_review"] is False


def test_retry_limit_flags_the_chunk_and_later_chunks_still_render(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    texts = _plan_texts(book)
    engine = FakeEngine(fail_texts=frozenset({texts[0]}))
    install_engine(engine)
    assert _render_cli(book, "--retries", "2") == 1
    assert engine.received == [texts[0], texts[0], texts[0], texts[1]]
    first, second = _records(book)
    assert first["status"] == "failed"
    assert first["needs_review"] is True
    assert len(first["attempts"]) == 3
    assert second["status"] == "ok"
    assert "failed chunk (needs review): c001-k0001" in capsys.readouterr().out


def test_silent_attempts_are_kept_for_diagnosis(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    texts = _plan_texts(book)
    install_engine(FakeEngine(silent_texts=frozenset({texts[0]})))
    assert _render_cli(book, "--retries", "1") == 1
    first = _records(book)[0]
    assert [a["failed_audio_file"] for a in first["attempts"]] == [
        "c001-k0001.attempt-1.failed.wav",
        "c001-k0001.attempt-2.failed.wav",
    ]
    assert first["attempts"][0]["error"] == "engine returned silent audio"
    assert (_target(book) / "c001-k0001.attempt-2.failed.wav").is_file()
    assert not (_target(book) / "c001-k0001.wav").exists()


def test_retry_failed_is_only_used_when_asked(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    texts = _plan_texts(book)
    install_engine(FakeEngine(fail_texts=frozenset({texts[0]})))
    assert _render_cli(book, "--retries", "0") == 1

    untouched = FakeEngine()
    install_engine(untouched)
    assert _render_cli(book) == 1
    assert untouched.received == []
    assert "failed chunk (needs review): c001-k0001" in capsys.readouterr().out

    retried = FakeEngine()
    install_engine(retried)
    assert _render_cli(book, "--retry-failed") == 0
    assert retried.received == [texts[0]]
    assert _records(book)[0]["status"] == "ok"
    assert _records(book)[0]["needs_review"] is False


def test_status_shows_progress_failures_and_review_flags(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    texts = _plan_texts(book)
    install_engine(FakeEngine(fail_texts=frozenset({texts[0]})))
    _render_cli(book, "--retries", "0")
    capsys.readouterr()
    assert main(["book", "status", str(book.directory), "--chapter", "c001", "--target", "short"]) == 0
    out = capsys.readouterr().out
    assert "c001 short: failed" in out
    assert "completed 1 of 2 chunk(s); remaining 0; failed 1" in out
    assert "needs review: c001-k0001" in out
    assert "c001-k0001: ValueError: fake failure on call 1" in out


def test_progress_lines_count_reused_and_failed_chunks(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    install_engine(FakeEngine(interrupt_on=frozenset({2})))
    _render_cli(book)
    capsys.readouterr()
    install_engine(FakeEngine())
    _render_cli(book)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("c001-k0002")]
    assert len(lines) == 1
    assert lines[0].startswith("c001-k0002 2/2 ok ")
    assert lines[0].endswith("| reused 1 | failed 0 | elapsed 0:00:00")


def test_memory_exhaustion_is_fatal_but_resumable(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    texts = _plan_texts(book)
    engine = FakeEngine(memory_error_on=frozenset({2}))
    install_engine(engine)
    assert _render_cli(book) == 1
    assert [record["status"] for record in _records(book)] == ["ok", "missing"]
    assert engine.unloads == 1
    run = json.loads((_target(book) / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"

    resumed = FakeEngine()
    install_engine(resumed)
    assert _render_cli(book) == 0
    assert resumed.received == texts[1:]


def test_ordinary_runtime_error_is_retried_within_budget(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine(runtime_error_on=frozenset({1})))
    assert _render_cli(book) == 0
    first = _records(book)[0]
    assert [a["status"] for a in first["attempts"]] == ["failed", "ok"]
    assert first["attempts"][0]["error"] == "RuntimeError: engine returned no audio segments on call 1"
    assert first["status"] == "ok"


def test_interrupted_then_resumed_chapter_assembles_with_full_coverage(
    book: BookProject, install_engine: Callable[[FakeEngine], None], capsys: pytest.CaptureFixture[str]
) -> None:
    from rutts.book.assemble import assemble_chapter

    interrupted = FakeEngine(interrupt_on=frozenset({2}))
    install_engine(interrupted)
    assert _render_cli(book) == 130
    resumed = FakeEngine()
    install_engine(resumed)
    assert _render_cli(book) == 0

    report = assemble_chapter(book, "c001", target="short")
    assert report["complete"] is True
    assert report["uncovered_segments"] == []
    covered = [entry["segment_id"] for entry in report["coverage"] if entry["chunks"]]
    assert covered == ["c001-s0002", "c001-s0004"]
    # the interrupted call produced no audio; every chunk was synthesized exactly once
    assert (interrupted.calls - 1) + resumed.calls == len(_records(book))


def test_leftover_temp_files_are_removed_at_the_start_of_a_run(
    book: BookProject, install_engine: Callable[[FakeEngine], None]
) -> None:
    install_engine(FakeEngine(interrupt_on=frozenset({2})))
    _render_cli(book)
    leftover = _target(book) / f".{'0' * 32}.tmp.wav"
    leftover.write_bytes(b"partial")
    install_engine(FakeEngine())
    assert _render_cli(book) == 0
    assert not leftover.exists()


@pytest.mark.parametrize(
    ("exc", "fatal"),
    [
        (MemoryError(), True),
        (RuntimeError("Metal: Out of memory while loading"), True),
        (RuntimeError("Insufficient memory for the KV cache"), True),
        (RuntimeError("buffer OutOfMemory"), True),
        (RuntimeError("failed to allocate 512 MB"), True),
        (RuntimeError("engine returned no audio segments"), False),
        (ValueError("bad input"), False),
        (KeyboardInterrupt(), True),
    ],
)
def test_only_memory_exhaustion_is_fatal(exc: BaseException, fatal: bool) -> None:
    from rutts.book.render import is_fatal

    assert is_fatal(exc) is fatal
