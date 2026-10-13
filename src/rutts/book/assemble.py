from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path
from rutts.book.chunking import Chunk, ChunkLimits, plan_chunks
from rutts.book.project import CHAPTERS_DIR, SCHEMA_VERSION, BookProject, read_json, sha256_file, sha256_text, write_json
from rutts.book.render import MANIFEST_FILE, resolve_target
from rutts.validation import ValidationError


@dataclass(frozen=True)
class Pauses:
    sentence: float = 0.35
    clause: float = 0.15
    paragraph: float = 0.7
    scene: float = 1.5
    chapter_end: float = 1.0

    def seconds(self, boundary: str) -> float:
        return {
            "sentence": self.sentence,
            "clause": self.clause,
            "paragraph": self.paragraph,
            "scene": self.scene,
            "chapter_end": self.chapter_end,
        }[boundary]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _chunk_audio_path(chunk: Chunk, record: dict[str, Any] | None, directory: Path) -> Path | None:
    """Return the verified audio path for an ok chunk, or None when it is failed or missing."""
    if record is None or record["text_sha256"] != sha256_text(chunk.text):
        return None
    if record["status"] != "ok":
        return None
    path = directory / str(record["audio_file"])
    if not path.is_file() or sha256_file(path) != record["audio_sha256"]:
        return None
    return path


def _chunk_status(chunk: Chunk, record: dict[str, Any] | None, directory: Path) -> str:
    if record is None or record["text_sha256"] != sha256_text(chunk.text):
        return "missing"
    if record["status"] == "failed":
        return "failed"
    if _chunk_audio_path(chunk, record, directory) is None:
        return "missing"
    return "ok"


def assemble_chapter(
    project: BookProject,
    chapter_id: str,
    target: str | None,
    pauses: Pauses = Pauses(),
) -> dict[str, Any]:
    """Concatenate chunk audio in plan order into a chapter master and write its coverage report."""
    chapter = project.read_source().chapter(chapter_id)
    if not chapter.included:
        raise ValidationError([f"chapter {chapter_id} is excluded from the book"])
    target = resolve_target(project, chapter_id, target)
    directory = project.target_dir(chapter_id, target)
    manifest = read_json(directory / MANIFEST_FILE)
    limits = ChunkLimits.from_dict(manifest["limits"])
    chunks = plan_chunks(project.planning_segments(chapter_id), target, limits)
    records = {record["id"]: record for record in manifest["chunks"]}

    statuses = {chunk.id: _chunk_status(chunk, records.get(chunk.id), directory) for chunk in chunks}
    loaded: dict[str, tuple[np.ndarray, int]] = {}
    for chunk in chunks:
        if statuses[chunk.id] == "ok":
            path = _chunk_audio_path(chunk, records.get(chunk.id), directory)
            assert path is not None
            samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
            loaded[chunk.id] = (np.asarray(samples, dtype=np.float32).reshape(-1), rate)
    if not loaded:
        raise ValidationError([f"no rendered audio is available for {chapter_id} ({target})"])
    rates = {rate for _, rate in loaded.values()}
    if len(rates) > 1:
        raise ValidationError([f"chunk sample rates differ ({sorted(rates)}); resampling is not supported"])
    rate = next(iter(rates))

    pieces: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    position = 0
    for chunk in chunks:
        status = statuses[chunk.id]
        entry: dict[str, Any] = {
            "id": chunk.id,
            "status": status,
            "boundary_after": chunk.boundary_after,
            "start_seconds": None,
            "end_seconds": None,
            "pause_after_seconds": 0.0,
        }
        if status == "ok":
            samples = loaded[chunk.id][0]
            entry["start_seconds"] = round(position / rate, 3)
            pieces.append(samples)
            position += len(samples)
            entry["end_seconds"] = round(position / rate, 3)
            pause = pauses.seconds(chunk.boundary_after)
            silence = np.zeros(round(pause * rate), dtype=np.float32)
            pieces.append(silence)
            position += len(silence)
            entry["pause_after_seconds"] = pause
        entries.append(entry)

    covering: dict[str, list[str]] = {
        segment.id: [] for segment in chapter.segments if segment.included
    }
    for chunk in chunks:
        for segment_id in chunk.narration_segment_ids:
            if segment_id in covering:
                covering[segment_id].append(chunk.id)
    coverage = [
        {"segment_id": segment.id, "kind": segment.kind, "chunks": covering[segment.id]}
        for segment in chapter.segments
        if segment.included
    ]
    uncovered = [
        segment.id
        for segment in chapter.segments
        if segment.included and segment.kind == "paragraph" and not covering[segment.id]
    ]
    missing = [chunk.id for chunk in chunks if statuses[chunk.id] == "missing"]
    failed = [chunk.id for chunk in chunks if statuses[chunk.id] == "failed"]
    excluded = [
        {"segment_id": segment.id, "reason": segment.exclusion_reason}
        for segment in chapter.segments
        if not segment.included
    ]
    complete = not missing and not failed and not uncovered

    master_name = f"{chapter_id}.wav" if complete else f"{chapter_id}.incomplete.wav"
    chapters_dir = project.directory / CHAPTERS_DIR
    audio = np.concatenate(pieces)
    with atomic_path(chapters_dir / master_name) as temp:
        sf.write(str(temp), audio, rate, subtype="FLOAT")
    other = chapters_dir / (f"{chapter_id}.incomplete.wav" if complete else f"{chapter_id}.wav")
    other.unlink(missing_ok=True)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "title": chapter.title,
        "target": target,
        "master_file": f"{CHAPTERS_DIR}/{master_name}",
        "sample_rate": rate,
        "duration_seconds": round(len(audio) / rate, 3),
        "pauses": pauses.to_dict(),
        "complete": complete,
        "chunks": entries,
        "coverage": coverage,
        "missing_chunks": missing,
        "failed_chunks": failed,
        "uncovered_segments": uncovered,
        "excluded_segments": excluded,
    }
    write_json(chapters_dir / f"{chapter_id}.json", report)
    return report
