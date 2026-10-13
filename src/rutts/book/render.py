from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path
from rutts.book.chunking import TARGET_SECONDS, Chunk, ChunkLimits, plan_chunks, plan_document
from rutts.book.project import (
    SCHEMA_VERSION,
    BookProject,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from rutts.narrator import NarratorProfile
from rutts.preview import audio_problems
from rutts.resources import measure
from rutts.validation import ValidationError

if TYPE_CHECKING:
    from rutts.engines.base import SynthesisEngine

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.json"


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


def _pending_record(chunk: Chunk, engine_fingerprint: str, narrator_sha256: str) -> dict[str, Any]:
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
        "attempts": [],
    }


def _render_chunk(
    engine: SynthesisEngine,
    profile: NarratorProfile,
    chunk: Chunk,
    directory: Path,
    engine_fingerprint: str,
    narrator_sha256: str,
) -> dict[str, Any]:
    record = _pending_record(chunk, engine_fingerprint, narrator_sha256)
    audio_name = f"{chunk.id}.wav"
    error: str | None = None
    with measure() as sample:
        try:
            result = engine.synthesize(chunk.text, profile)
            samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
            error = audio_problems(samples)
            if error is None:
                with atomic_path(directory / audio_name) as temp:
                    sf.write(str(temp), samples, result.sample_rate, subtype="FLOAT")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    record["render_seconds"] = round(sample.elapsed_seconds, 3)
    record["resources"] = sample.metrics
    if error is not None:
        record["status"] = "failed"
        record["attempts"] = [{"attempt": 1, "status": "failed", "error": error}]
        return record

    audio_path = directory / audio_name
    info = sf.info(str(audio_path))
    record.update(
        status="ok",
        audio_file=audio_name,
        audio_sha256=sha256_file(audio_path),
        sample_rate=info.samplerate,
        actual_duration_seconds=round(info.frames / info.samplerate, 3),
        attempts=[{"attempt": 1, "status": "ok", "error": None}],
    )
    return record


def _archive_path(chapter_dir: Path) -> Path:
    number = 1
    while (chapter_dir.parent / f"{chapter_dir.name}.prev-{number}").exists():
        number += 1
    return chapter_dir.parent / f"{chapter_dir.name}.prev-{number}"


def _archive_or_refuse(
    project: BookProject,
    chapter_id: str,
    engine_fingerprint: str,
    narrator_sha256: str,
    force_new_run: bool,
) -> None:
    chapter_dir = project.chapter_dir(chapter_id)
    stale: list[str] = []
    for manifest_path in sorted(chapter_dir.glob(f"*/{MANIFEST_FILE}")):
        manifest = read_json(manifest_path)
        if (
            manifest.get("engine_fingerprint") != engine_fingerprint
            or manifest.get("narrator_sha256") != narrator_sha256
        ):
            stale.append(manifest_path.parent.name)
    if stale and not force_new_run:
        raise ValidationError(
            [
                f"chapter {chapter_id} was rendered with a different engine configuration or "
                f"narrator ({', '.join(stale)}); pass --force-new-run to archive the previous "
                "chunks and start a new run"
            ]
        )
    if force_new_run and chapter_dir.exists():
        shutil.move(str(chapter_dir), str(_archive_path(chapter_dir)))


def render_chapter(
    project: BookProject,
    chapter_id: str,
    engine: SynthesisEngine,
    target: str,
    *,
    limits: ChunkLimits = ChunkLimits(),
    force_new_run: bool = False,
) -> dict[str, Any]:
    """Synthesize every planned chunk of a chapter, saving each success as it completes."""
    chapter = project.read_source().chapter(chapter_id)
    if not chapter.included:
        raise ValidationError([f"chapter {chapter_id} is excluded from the book"])
    profile = project.narrator()
    narrator_sha256 = project.narrator_sha256()
    chunks = plan_chunks(project.planning_segments(chapter_id), target, limits)
    if not chunks:
        raise ValidationError([f"chapter {chapter_id} has no text to render"])

    config = engine.config
    engine_fingerprint = config.fingerprint()
    _archive_or_refuse(project, chapter_id, engine_fingerprint, narrator_sha256, force_new_run)

    directory = project.target_dir(chapter_id, target)
    write_json(directory / PLAN_FILE, plan_document(chapter_id, target, limits, chunks))
    records = [_pending_record(chunk, engine_fingerprint, narrator_sha256) for chunk in chunks]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "target": target,
        "target_seconds": {"min": TARGET_SECONDS[target][0], "max": TARGET_SECONDS[target][1]},
        "limits": limits.to_dict(),
        "engine": config.to_dict(),
        "engine_fingerprint": engine_fingerprint,
        "narrator": {
            "name": profile.name,
            "language": profile.language,
            "provenance": profile.provenance,
            "reference_sha256": profile.reference_sha256,
        },
        "narrator_sha256": narrator_sha256,
        "chunks": records,
    }
    manifest_path = directory / MANIFEST_FILE
    write_json(manifest_path, manifest)
    try:
        engine.load()
        for index, chunk in enumerate(chunks):
            records[index] = _render_chunk(
                engine, profile, chunk, directory, engine_fingerprint, narrator_sha256
            )
            write_json(manifest_path, manifest)
    finally:
        engine.unload()
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
