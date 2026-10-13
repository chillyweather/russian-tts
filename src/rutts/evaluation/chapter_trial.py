from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rutts.book.project import CHAPTERS_DIR, SCHEMA_VERSION, load_project, read_json, write_json
from rutts.book.render import MANIFEST_FILE
from rutts.evaluation.common import NOTES_DIR, TRIALS_DIR, UNAVAILABLE
from rutts.validation import ValidationError

LABEL_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
MEMORY_KEYS = ("peak_rss_bytes", "mlx_peak_memory_bytes", "swap_used_bytes", "memory_free_percent")
SECONDS_PER_HOUR = 3600.0


def _check_label(label: str) -> None:
    if not re.match(LABEL_PATTERN, label):
        raise ValidationError([f"label '{label}' must be lowercase letters, digits, - or _"])


def _memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [record["resources"] for record in records if isinstance(record.get("resources"), dict)]
    memory: dict[str, Any] = {}
    for key in MEMORY_KEYS:
        values = [item[key] for item in observations if isinstance(item.get(key), int)]
        if not values:
            memory[key] = UNAVAILABLE
        elif key == "memory_free_percent":
            memory[key] = min(values)
        else:
            memory[key] = max(values)
    return memory


def _retries(records: list[dict[str, Any]]) -> int | str:
    if not all(isinstance(record.get("attempts"), list) for record in records):
        return UNAVAILABLE
    return sum(max(len(record["attempts"]) - 1, 0) for record in records)


def record_trial(book_dir: Path, chapter_id: str, target: str, label: str, out: Path) -> dict[str, Any]:
    _check_label(label)
    project = load_project(book_dir)
    manifest_path = project.target_dir(chapter_id, target) / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValidationError([f"chapter {chapter_id} has no render for target '{target}'"])
    manifest = read_json(manifest_path)
    records: list[dict[str, Any]] = manifest["chunks"]

    assembly_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    assembly = read_json(assembly_path) if assembly_path.is_file() else None
    if assembly is not None and assembly.get("target") != target:
        raise ValidationError(
            [f"the assembly of {chapter_id} is for target '{assembly.get('target')}'; assemble '{target}' first"]
        )

    trial_path = out / TRIALS_DIR / f"{label}.json"
    if trial_path.exists():
        raise FileExistsError(f"chapter trial '{label}' already exists: {trial_path}")

    ok = [record for record in records if record["status"] == "ok"]
    failed = [record["id"] for record in records if record["status"] == "failed"]
    finished = round(sum(float(record["actual_duration_seconds"]) for record in ok), 3)
    rendered = [record["render_seconds"] for record in records if record["render_seconds"] is not None]
    render_seconds: float | str = round(sum(rendered), 3) if rendered else UNAVAILABLE
    retries = _retries(records)
    assembly_complete: bool | str = assembly["complete"] if assembly is not None else UNAVAILABLE
    completed = assembly_complete is True and not failed and len(ok) == len(records)

    render_per_hour: float | str = UNAVAILABLE
    retries_per_hour: float | str = UNAVAILABLE
    if finished > 0 and isinstance(render_seconds, float):
        render_per_hour = round(render_seconds / finished, 4)
    if finished > 0 and isinstance(retries, int):
        retries_per_hour = round(retries / (finished / SECONDS_PER_HOUR), 4)

    trial: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "chapter_id": chapter_id,
        "target": target,
        "engine_fingerprint": manifest["engine_fingerprint"],
        "engine": manifest["engine"],
        "narrator_sha256": manifest["narrator_sha256"],
        "chunks": len(records),
        "ok_chunks": len(ok),
        "failed_chunks": failed,
        "finished_audio_seconds": finished,
        "render_seconds": render_seconds,
        "retries": retries,
        "assembly_complete": assembly_complete,
        "completed": completed,
        "memory": _memory(records),
        "render_hours_per_audio_hour": render_per_hour,
        "retries_per_audio_hour": retries_per_hour,
    }
    write_json(trial_path, trial)
    return trial


def record_notes(
    eval_dir: Path,
    label: str,
    *,
    stress_errors: int,
    omissions: int,
    distracting_joins: int,
    notes: str,
) -> dict[str, Any]:
    _check_label(label)
    problems = [
        f"{name} must be zero or positive"
        for name, value in (
            ("stress errors", stress_errors),
            ("omissions", omissions),
            ("distracting joins", distracting_joins),
        )
        if value < 0
    ]
    if not (eval_dir / TRIALS_DIR / f"{label}.json").is_file():
        problems.append(f"no chapter trial named '{label}'; run eval chapter-trial first")
    if problems:
        raise ValidationError(problems)
    path = eval_dir / NOTES_DIR / f"{label}.json"
    if path.exists():
        raise FileExistsError(f"chapter notes for '{label}' already exist: {path}")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "stress_errors": stress_errors,
        "omissions": omissions,
        "distracting_joins": distracting_joins,
        "notes": notes,
    }
    write_json(path, document)
    return document
