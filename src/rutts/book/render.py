from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path, remove_temp_files
from rutts.book.chunking import TARGET_SECONDS, Chunk, ChunkLimits, plan_chunks, plan_document
from rutts.book.project import (
    SCHEMA_VERSION,
    BookProject,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from rutts.book.runstate import RunTracker, format_elapsed, read_run_state
from rutts.narrator import NarratorProfile
from rutts.preview import audio_problems
from rutts.pronunciation.markup import effective_convention, hint_chunk_ids, read_markup, translate_chunks
from rutts.resources import measure
from rutts.validation import ValidationError

if TYPE_CHECKING:
    from rutts.engines.base import SynthesisEngine

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.json"
SUPERSEDED_DIR = "superseded"
MEMORY_EXHAUSTION_MARKERS = ("out of memory", "insufficient memory", "outofmemory", "failed to allocate")


class ConfigChangeRequired(ValidationError):
    """The engine or narrator differs from the one that rendered the chapter."""


def is_fatal(exc: BaseException) -> bool:
    """Only memory exhaustion (or an interrupt) stops the run; every other error is a failed attempt."""
    if isinstance(exc, (KeyboardInterrupt, MemoryError)):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in MEMORY_EXHAUSTION_MARKERS)


def record_fingerprint(text_sha256: str, engine_fingerprint: str, narrator_sha256: str) -> str:
    canonical = json.dumps(
        {
            "text_sha256": text_sha256,
            "engine_fingerprint": engine_fingerprint,
            "narrator_sha256": narrator_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def _pending_record(
    chunk: Chunk,
    engine_fingerprint: str,
    narrator_sha256: str,
    *,
    attempts: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text_sha256 = sha256_text(chunk.text)
    return {
        "id": chunk.id,
        "status": "missing",
        "text_sha256": text_sha256,
        "engine_fingerprint": engine_fingerprint,
        "narrator_sha256": narrator_sha256,
        "fingerprint": record_fingerprint(text_sha256, engine_fingerprint, narrator_sha256),
        "audio_file": None,
        "audio_sha256": None,
        "sample_rate": None,
        "actual_duration_seconds": None,
        "render_seconds": None,
        "resources": None,
        "needs_review": False,
        "attempts": list(attempts or []),
        "history": list(history or []),
    }


def _stale_reason(chunk: Chunk, record: dict[str, Any], engine_fingerprint: str, narrator_sha256: str) -> str | None:
    """Why a recorded chunk no longer matches its inputs, or None when the fingerprint still matches."""
    fresh = record_fingerprint(sha256_text(chunk.text), engine_fingerprint, narrator_sha256)
    if record.get("fingerprint") == fresh:
        return None
    if record.get("engine_fingerprint") != engine_fingerprint or record.get("narrator_sha256") != narrator_sha256:
        return "config_changed"
    return "text_changed"


def _audio_problem(record: dict[str, Any], directory: Path) -> str | None:
    """Check that an ok record's audio is present, unchanged and decodable."""
    if record["status"] != "ok":
        return None
    if not record.get("audio_file") or not (directory / str(record["audio_file"])).is_file():
        return "missing_audio"
    path = directory / str(record["audio_file"])
    if sha256_file(path) != record.get("audio_sha256"):
        return "corrupt_audio"
    try:
        samples, rate = sf.read(str(path), dtype="float32")
    except Exception:
        return "corrupt_audio"
    if rate != record.get("sample_rate") or samples.size == 0:
        return "corrupt_audio"
    return None


def _supersede(directory: Path, record: dict[str, Any]) -> None:
    name = record.get("audio_file")
    if not name or not (directory / str(name)).is_file():
        return
    superseded = directory / SUPERSEDED_DIR
    superseded.mkdir(exist_ok=True)
    os.replace(directory / str(name), superseded / str(name))


def _reconcile(
    chunks: list[Chunk],
    previous: dict[str, dict[str, Any]],
    engine_fingerprint: str,
    narrator_sha256: str,
    directory: Path,
    retry_failed: bool,
) -> list[dict[str, Any]]:
    """Return one record per planned chunk, keeping only the records whose work is still valid."""
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        old = previous.get(chunk.id)
        if old is None:
            records.append(_pending_record(chunk, engine_fingerprint, narrator_sha256))
            continue
        reason = _stale_reason(chunk, old, engine_fingerprint, narrator_sha256)
        if reason is None:
            reason = _audio_problem(old, directory)
        if reason is None and old["status"] == "ok":
            records.append(old)
            continue
        if reason is None and old["status"] == "failed" and not retry_failed:
            records.append(old)
            continue
        history = list(old.get("history", []))
        if reason is not None:
            history.append({"event": "invalidated", "reason": reason})
            _supersede(directory, old)
            attempts: list[dict[str, Any]] = []
        else:
            attempts = list(old.get("attempts", []))
            if old["status"] == "failed":
                history.append({"event": "retry_failed"})
        records.append(_pending_record(chunk, engine_fingerprint, narrator_sha256, attempts=attempts, history=history))
    return records


def _write_audio(path: Path, samples: np.ndarray, rate: int) -> None:
    with atomic_path(path) as temp:
        sf.write(str(temp), samples, rate, subtype="FLOAT")


def _commit_audio(
    directory: Path, chunk_id: str, samples: np.ndarray, rate: int, record: dict[str, Any]
) -> None:
    """Persist audio atomically, then mark the record ok. Only this order keeps ok records valid."""
    name = f"{chunk_id}.wav"
    _write_audio(directory / name, samples, rate)
    info = sf.info(str(directory / name))
    record.update(
        status="ok",
        audio_file=name,
        audio_sha256=sha256_file(directory / name),
        sample_rate=info.samplerate,
        actual_duration_seconds=round(info.frames / info.samplerate, 3),
        needs_review=False,
    )


def _synthesize_once(
    engine: SynthesisEngine,
    profile: NarratorProfile,
    chunk: Chunk,
    record: dict[str, Any],
    directory: Path,
) -> bool:
    """Run one attempt, append its diagnostics to the record and commit audio when valid."""
    attempt: dict[str, Any] = {
        "attempt": len(record["attempts"]) + 1,
        "status": "failed",
        "error": None,
        "exception_type": None,
        "render_seconds": None,
        "resources": None,
        "failed_audio_file": None,
    }
    record["attempts"].append(attempt)
    samples: np.ndarray | None = None
    rate = 0
    problem: str | None = None
    try:
        with measure() as sample:
            try:
                result = engine.synthesize(chunk.text, profile)
                rate = result.sample_rate
                samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
                problem = audio_problems(samples)
            except BaseException as exc:
                if not isinstance(exc, Exception) or is_fatal(exc):
                    attempt.update(status="interrupted", exception_type=type(exc).__name__, error=str(exc))
                    raise
                attempt["exception_type"] = type(exc).__name__
                problem = f"{type(exc).__name__}: {exc}"
    finally:
        attempt.update(render_seconds=round(sample.elapsed_seconds, 3), resources=sample.metrics)

    if problem is not None:
        attempt["error"] = problem
        if samples is not None and samples.size > 0:
            name = f"{chunk.id}.attempt-{attempt['attempt']}.failed.wav"
            _write_audio(directory / name, samples, rate)
            attempt["failed_audio_file"] = name
        return False
    assert samples is not None
    _commit_audio(directory, chunk.id, samples, rate, record)
    record["render_seconds"] = attempt["render_seconds"]
    record["resources"] = attempt["resources"]
    attempt["status"] = "ok"
    return True


def _attempt_chunk(
    engine: SynthesisEngine,
    profile: NarratorProfile,
    chunk: Chunk,
    record: dict[str, Any],
    directory: Path,
    retries: int,
) -> None:
    """Try a chunk up to retries + 1 times; leave it failed and flagged for review when none succeed."""
    budget_end = len(record["attempts"]) + retries + 1
    while len(record["attempts"]) < budget_end:
        if _synthesize_once(engine, profile, chunk, record, directory):
            return
    record["status"] = "failed"
    record["needs_review"] = True


def _render_pending(
    engine: SynthesisEngine,
    profile: NarratorProfile,
    chunks: list[Chunk],
    records: list[dict[str, Any]],
    pending: list[int],
    directory: Path,
    manifest: dict[str, Any],
    run: RunTracker,
    retries: int,
    progress: Callable[[str], None] | None,
) -> None:
    manifest_path = directory / MANIFEST_FILE
    for index in pending:
        chunk, record = chunks[index], records[index]
        run.current_chunk = chunk.id
        started = run.elapsed()
        _attempt_chunk(engine, profile, chunk, record, directory, retries)
        write_json(manifest_path, manifest)
        if record["status"] == "failed":
            run.record_failure(chunk.id, str(record["attempts"][-1]["error"]))
        document = run.save(records, "running")
        if progress is not None:
            seconds = run.elapsed() - started
            progress(
                f"{chunk.id} {index + 1}/{len(chunks)} {record['status']} {seconds:.1f}s  "
                f"| reused {document['reused']} | failed {document['failed']} "
                f"| elapsed {format_elapsed(run.elapsed())}"
            )


def _archive_path(chapter_dir: Path) -> Path:
    number = 1
    while (chapter_dir.parent / f"{chapter_dir.name}.prev-{number}").exists():
        number += 1
    return chapter_dir.parent / f"{chapter_dir.name}.prev-{number}"


def _check_config(
    previous: dict[str, Any] | None,
    engine_fingerprint: str,
    narrator_sha256: str,
    accept_config_change: bool,
    chapter_id: str,
) -> None:
    if previous is None:
        return
    if previous.get("engine_fingerprint") == engine_fingerprint and previous.get("narrator_sha256") == narrator_sha256:
        return
    count = len(previous["chunks"])
    if not accept_config_change:
        raise ConfigChangeRequired(
            [
                f"chapter {chapter_id} was rendered with a different engine configuration or narrator; "
                f"{count} chunk(s) will be invalidated as config_changed; pass --accept-config-change "
                "to regenerate them (previous audio moves to superseded/)"
            ]
        )


def _manifest_document(
    chapter_id: str,
    target: str,
    limits: ChunkLimits,
    engine: SynthesisEngine,
    profile: NarratorProfile,
    narrator_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    config = engine.config
    return {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "target": target,
        "target_seconds": {"min": TARGET_SECONDS[target][0], "max": TARGET_SECONDS[target][1]},
        "limits": limits.to_dict(),
        "engine": config.to_dict(),
        "engine_fingerprint": config.fingerprint(),
        "narrator": {
            "name": profile.name,
            "language": profile.language,
            "provenance": profile.provenance,
            "reference_sha256": profile.reference_sha256,
        },
        "narrator_sha256": narrator_sha256,
        "chunks": records,
    }


def render_chapter(
    project: BookProject,
    chapter_id: str,
    engine: SynthesisEngine,
    target: str,
    *,
    limits: ChunkLimits = ChunkLimits(),
    force_new_run: bool = False,
    accept_config_change: bool = False,
    retries: int = 2,
    retry_failed: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Synthesize the planned chunks of a chapter, reusing valid work and saving each success as it completes."""
    chapter = project.read_source().chapter(chapter_id)
    if not chapter.included:
        raise ValidationError([f"chapter {chapter_id} is excluded from the book"])
    if retries < 0:
        raise ValidationError(["--retries must be zero or positive"])
    profile = project.narrator()
    narrator_sha256 = project.narrator_sha256()
    planned = plan_chunks(project.planning_segments(chapter_id), target, limits)
    if not planned:
        raise ValidationError([f"chapter {chapter_id} has no text to render"])
    convention = effective_convention(engine.config.engine_id, read_markup(project.directory))
    chunks = translate_chunks(planned, convention)

    config = engine.config
    engine_fingerprint = config.fingerprint()
    directory = project.target_dir(chapter_id, target)
    if force_new_run:
        chapter_dir = project.chapter_dir(chapter_id)
        if chapter_dir.exists():
            shutil.move(str(chapter_dir), str(_archive_path(chapter_dir)))
        previous_manifest = None
    else:
        previous_manifest = read_json(directory / MANIFEST_FILE) if (directory / MANIFEST_FILE).is_file() else None
        _check_config(previous_manifest, engine_fingerprint, narrator_sha256, accept_config_change, chapter_id)
    previous = {record["id"]: record for record in previous_manifest["chunks"]} if previous_manifest else {}

    remove_temp_files(directory)
    write_json(directory / PLAN_FILE, plan_document(chapter_id, target, limits, planned))
    records = _reconcile(chunks, previous, engine_fingerprint, narrator_sha256, directory, retry_failed)
    manifest = _manifest_document(chapter_id, target, limits, engine, profile, narrator_sha256, records)
    manifest["pronunciation"] = {
        "engine_id": engine.config.engine_id,
        "convention": convention,
        "hints_not_applied": hint_chunk_ids(planned, convention),
    }
    write_json(directory / MANIFEST_FILE, manifest)
    pending = [index for index, record in enumerate(records) if record["status"] == "missing"]

    reused = sum(1 for record in records if record["status"] == "ok")
    run = RunTracker(directory, chapter_id, target, len(chunks), reused, read_run_state(directory))
    run.save(records, "running")
    loaded = False
    try:
        if pending:
            engine.load()
            loaded = True
        _render_pending(engine, profile, chunks, records, pending, directory, manifest, run, retries, progress)
    except KeyboardInterrupt:
        write_json(directory / MANIFEST_FILE, manifest)
        run.save(records, "interrupted")
        raise
    except BaseException:
        write_json(directory / MANIFEST_FILE, manifest)
        run.save(records, "failed")
        raise
    finally:
        if loaded:
            engine.unload()
    failed = any(record["status"] == "failed" for record in records)
    run.save(records, "failed" if failed else "completed")
    return manifest


def resolve_target(project: BookProject, chapter_id: str, target: str | None) -> str:
    rendered = sorted(
        path.parent.name for path in project.chapter_dir(chapter_id).glob(f"*/{MANIFEST_FILE}")
    )
    if target is not None:
        if target not in rendered:
            raise ValidationError([f"chapter {chapter_id} has no render for target '{target}'"])
        return target
    if not rendered:
        raise ValidationError([f"chapter {chapter_id} has not been rendered; run book render first"])
    if len(rendered) > 1:
        raise ValidationError(
            [f"chapter {chapter_id} was rendered for several targets ({', '.join(rendered)}); pass --target"]
        )
    return rendered[0]


def summarize_targets(project: BookProject, chapter_id: str) -> list[dict[str, Any]]:
    """Counts and measured (not estimated) audio durations per rendered target."""
    rows: list[dict[str, Any]] = []
    for target in TARGET_SECONDS:
        path = project.target_dir(chapter_id, target) / MANIFEST_FILE
        if not path.is_file():
            rows.append({"target": target, "rendered": False})
            continue
        records = read_json(path)["chunks"]
        durations = [
            float(record["actual_duration_seconds"]) for record in records if record["status"] == "ok"
        ]
        row: dict[str, Any] = {
            "target": target,
            "rendered": True,
            "chunks": len(records),
            "ok": sum(1 for record in records if record["status"] == "ok"),
            "failed": sum(1 for record in records if record["status"] == "failed"),
            "missing": sum(1 for record in records if record["status"] == "missing"),
            "measured_seconds": None,
        }
        if durations:
            row["measured_seconds"] = {
                "min": min(durations),
                "mean": sum(durations) / len(durations),
                "max": max(durations),
            }
        rows.append(row)
    return rows
