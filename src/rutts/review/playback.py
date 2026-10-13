from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path
from rutts.book.assemble import Pauses
from rutts.book.project import BookProject, read_json
from rutts.book.render import MANIFEST_FILE, PLAN_FILE
from rutts.review.queue import QueueItem
from rutts.validation import ValidationError

CONTEXT_SECONDS = 1.5
CLIPS_DIR = "review/clips"
Runner = Callable[[list[str]], None]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


# Tests replace these to avoid playing audio; the CLI calls them only when asked to play or open.
PLAYER: Runner = _run
OPENER: Runner = _run


def _read(path: Path) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(str(path), dtype="float32")
    return np.asarray(samples, dtype=np.float32).reshape(-1), int(rate)


def _cut(samples: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    first = max(0, round(start * rate))
    last = min(samples.size, round(end * rate))
    return samples[first:last]


def _join(project: BookProject, item: QueueItem) -> tuple[np.ndarray, int]:
    """Previous chunk, pause, this chunk, pause, next chunk; neighbours without valid audio are skipped."""
    directory = project.target_dir(item.chapter_id, item.target)
    records: list[dict[str, Any]] = read_json(directory / MANIFEST_FILE)["chunks"]
    plan: list[dict[str, Any]] = read_json(directory / PLAN_FILE)["chunks"]
    index = next(i for i, record in enumerate(records) if record["id"] == item.chunk_id)
    pauses = Pauses()
    rate = int(records[index]["sample_rate"])
    own, _ = _read(directory / str(records[index]["audio_file"]))
    pieces: list[np.ndarray] = []
    if index > 0 and records[index - 1]["status"] == "ok":
        before, _ = _read(directory / str(records[index - 1]["audio_file"]))
        silence = np.zeros(round(pauses.seconds(str(plan[index - 1]["boundary_after"])) * rate), dtype=np.float32)
        pieces += [before, silence]
    pieces.append(own)
    if index + 1 < len(records) and records[index + 1]["status"] == "ok":
        silence = np.zeros(round(pauses.seconds(str(plan[index]["boundary_after"])) * rate), dtype=np.float32)
        after, _ = _read(directory / str(records[index + 1]["audio_file"]))
        pieces += [silence, after]
    return np.concatenate(pieces), rate


def write_clip(project: BookProject, item: QueueItem, *, join: bool = False) -> Path:
    """Cut the flagged span (with context) or the whole chunk; ``join`` writes the surrounding join instead."""
    if item.audio_file is None:
        raise ValidationError([f"item {item.id} has no audio to play"])
    name = f"{item.id}-join.wav" if join else f"{item.id}.wav"
    destination = project.directory / CLIPS_DIR / name
    if join:
        samples, rate = _join(project, item)
    else:
        samples, rate = _read(project.target_dir(item.chapter_id, item.target) / item.audio_file)
        if item.audio_seconds is not None:
            start, end = item.audio_seconds
            samples = _cut(samples, rate, start - CONTEXT_SECONDS, end + CONTEXT_SECONDS)
    with atomic_path(destination) as temp:
        sf.write(str(temp), samples, rate, subtype="FLOAT")
    return destination
