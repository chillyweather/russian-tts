from __future__ import annotations

import os
import random
import string
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path
from rutts.evaluation.common import (
    BLIND_FILE,
    KEY_FILE,
    RESULTS_FILE,
    SAMPLES_DIR,
    SCHEMA_VERSION,
    read_json,
    write_json,
)
from rutts.validation import ValidationError

ATTEMPT_MODES = ("first", "final")
ACTIVE_FRAME_SECONDS = 0.02
ACTIVE_FLOOR_DBFS = -40.0
PEAK_CEILING = 1.0
MAX_CANDIDATES = len(string.ascii_uppercase)


@dataclass(frozen=True)
class Candidate:
    index: int
    directory: Path
    results: dict[str, Any]


@dataclass(frozen=True)
class Pending:
    excerpt_id: str
    category: str
    text: str
    candidate: Candidate
    attempt: dict[str, Any]
    samples: np.ndarray
    rate: int


def load_candidate(index: int, directory: Path) -> Candidate:
    results = read_json(directory / RESULTS_FILE)
    return Candidate(index=index, directory=directory.resolve(), results=results)


def chosen_attempt(excerpt: dict[str, Any], mode: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the attempt to compare, or the reason no usable attempt exists."""
    attempts: list[dict[str, Any]] = excerpt["attempts"]
    if mode == "first":
        first = attempts[0]
        if first["status"] == "ok":
            return first, None
        return None, f"first attempt failed: {first['error']}"
    successful = [attempt for attempt in attempts if attempt["status"] == "ok"]
    if successful:
        return successful[-1], None
    return None, f"no successful attempt within the retry budget; last error: {attempts[-1]['error']}"


def _active_rms(samples: np.ndarray, rate: int) -> float | None:
    """RMS over the frames that are not silent; None when nothing is audible."""
    frame = max(1, round(ACTIVE_FRAME_SECONDS * rate))
    count = samples.size // frame
    if count == 0:
        return None
    frames = samples[: count * frame].reshape(count, frame).astype(np.float64)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active = frames[frame_rms >= 10 ** (ACTIVE_FLOOR_DBFS / 20)]
    if active.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(active))))


def _common_target(pending: list[Pending]) -> tuple[float, bool]:
    """One RMS target for the whole set, lowered when raising to the mean would clip any file."""
    rms_values = [_active_rms(item.samples, item.rate) for item in pending]
    levels = [value for value in rms_values if value is not None]
    mean_db = float(np.mean([20 * np.log10(level) for level in levels]))
    target = 10 ** (mean_db / 20)
    ceilings = [
        rms * PEAK_CEILING / float(np.max(np.abs(item.samples)))
        for item, rms in zip(pending, rms_values, strict=True)
        if rms is not None and float(np.max(np.abs(item.samples))) > 0
    ]
    lowered = min(ceilings) < target
    return min([target, *ceilings]), lowered


def _check_inputs(candidates: list[Candidate]) -> None:
    problems: list[str] = []
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        problems.append(f"give between 1 and {MAX_CANDIDATES} preview directories")
    for candidate in candidates:
        if not candidate.results.get("excerpts"):
            problems.append(f"preview has no excerpts: {candidate.directory}")
    if problems:
        raise ValidationError(problems)


def _shared_excerpt_ids(lookups: list[dict[str, dict[str, Any]]]) -> list[str]:
    shared_ids = sorted(set(lookups[0]).intersection(*lookups[1:]))
    problems = [
        f"excerpt '{excerpt_id}' has different text across previews"
        for excerpt_id in shared_ids
        if len({lookup[excerpt_id]["text"] for lookup in lookups}) > 1
    ]
    if problems:
        raise ValidationError(problems)
    return shared_ids


def _gather(candidates: list[Candidate], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (usable excerpt groups, unavailable records)."""
    lookups = [{item["id"]: item for item in candidate.results["excerpts"]} for candidate in candidates]
    usable: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for excerpt_id in _shared_excerpt_ids(lookups):
        choices = [chosen_attempt(lookup[excerpt_id], mode) for lookup in lookups]
        if any(attempt is None for attempt, _ in choices):
            reasons = [reason for _, reason in choices if reason is not None]
            unavailable.append(
                {
                    "excerpt_id": excerpt_id,
                    "reasons": reasons,
                    "candidates": [
                        {"candidate": candidate.index, "reason": reason}
                        for candidate, (attempt, reason) in zip(candidates, choices, strict=True)
                        if attempt is None
                    ],
                }
            )
            continue
        usable.append({"excerpt": lookups[0][excerpt_id], "choices": [attempt for attempt, _ in choices]})
    return usable, unavailable


def _load_sources(candidates: list[Candidate], group: dict[str, Any]) -> list[Pending]:
    pending: list[Pending] = []
    for candidate, attempt in zip(candidates, group["choices"], strict=True):
        source = candidate.directory / str(attempt["file"])
        if not source.is_file():
            raise ValidationError([f"preview audio is missing: {source}"])
        samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
        flat = np.asarray(samples, dtype=np.float32).reshape(-1)
        if _active_rms(flat, int(rate)) is None:
            raise ValidationError([f"preview audio has no audible frames: {source}"])
        pending.append(
            Pending(
                excerpt_id=group["excerpt"]["id"],
                category=group["excerpt"]["category"],
                text=group["excerpt"]["text"],
                candidate=candidate,
                attempt=attempt,
                samples=flat,
                rate=int(rate),
            )
        )
    return pending


def prepare(
    candidate_dirs: list[Path],
    out: Path,
    *,
    seed: int,
    attempt_mode: str = "first",
) -> dict[str, Any]:
    if attempt_mode not in ATTEMPT_MODES:
        raise ValidationError([f"attempt mode must be one of {', '.join(ATTEMPT_MODES)}"])
    if out.exists():
        raise FileExistsError(f"evaluation directory already exists: {out}")
    candidates = [load_candidate(index, directory) for index, directory in enumerate(candidate_dirs)]
    _check_inputs(candidates)

    groups, unavailable = _gather(candidates, attempt_mode)
    rng = random.Random(seed)
    letters = list(string.ascii_uppercase[: len(candidates)])
    assigned: list[tuple[dict[str, Any], list[Pending], list[str]]] = []
    for group in groups:
        labels = list(letters)
        rng.shuffle(labels)
        assigned.append((group, _load_sources(candidates, group), labels))

    every = [item for _, pending, _ in assigned for item in pending]
    target_rms, lowered = _common_target(every) if every else (None, False)

    staging = out.parent / f".{out.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    try:
        samples_out: list[dict[str, Any]] = []
        blind_excerpts: list[dict[str, Any]] = []
        blind_pairs: list[dict[str, Any]] = []
        for group, pending, labels in assigned:
            excerpt_id = group["excerpt"]["id"]
            for item, label in zip(pending, labels, strict=True):
                rms = _active_rms(item.samples, item.rate)
                assert rms is not None and target_rms is not None
                gain = target_rms / rms
                relative = f"{SAMPLES_DIR}/{excerpt_id}/{label}.wav"
                with atomic_path(staging / relative) as temp:
                    sf.write(str(temp), (item.samples * gain).astype(np.float32), item.rate, subtype="FLOAT")
                samples_out.append(
                    {
                        "excerpt_id": excerpt_id,
                        "label": label,
                        "candidate": item.candidate.index,
                        "candidate_dir": str(item.candidate.directory),
                        "engine_fingerprint": item.candidate.results["engine"]["fingerprint"],
                        "attempt": item.attempt["attempt"],
                        "source_file": item.attempt["file"],
                        "gain_db": round(20 * float(np.log10(gain)), 4),
                        "file": relative,
                    }
                )
            sorted_labels = sorted(labels)
            blind_excerpts.append(
                {
                    "id": excerpt_id,
                    "category": group["excerpt"]["category"],
                    "text": group["excerpt"]["text"],
                    "labels": sorted_labels,
                }
            )
            blind_pairs += [
                {"excerpt_id": excerpt_id, "labels": list(pair)} for pair in combinations(sorted_labels, 2)
            ]

        blind = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "attempt_mode": attempt_mode,
            "labels": letters,
            "excerpts": blind_excerpts,
            "pairs": blind_pairs,
            "unavailable": [
                {"excerpt_id": item["excerpt_id"], "reason": "; ".join(item["reasons"])}
                for item in unavailable
            ],
        }
        key = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "attempt_mode": attempt_mode,
            "loudness": {
                "method": "one constant gain per file to a common RMS over non-silent frames",
                "frame_seconds": ACTIVE_FRAME_SECONDS,
                "active_floor_dbfs": ACTIVE_FLOOR_DBFS,
                "peak_ceiling": PEAK_CEILING,
                "target_rms_dbfs": None if target_rms is None else round(20 * float(np.log10(target_rms)), 4),
                "target_lowered": lowered,
            },
            "candidates": [
                {
                    "candidate": candidate.index,
                    "directory": str(candidate.directory),
                    "engine_fingerprint": candidate.results["engine"]["fingerprint"],
                    "engine": candidate.results["engine"]["config"],
                    "narrator": candidate.results["narrator"],
                    "retry_budget": candidate.results["retry_budget"],
                }
                for candidate in candidates
            ],
            "samples": samples_out,
            "unavailable": unavailable,
        }
        write_json(staging / BLIND_FILE, blind)
        write_json(staging / KEY_FILE, key)
        os.replace(staging, out)
    except BaseException:
        _remove_tree(staging)
        raise
    return blind


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    if path.exists():
        path.rmdir()
