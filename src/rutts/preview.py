from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources as package_resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path, atomic_write_text
from rutts.engines.base import SynthesisEngine
from rutts.narrator import NarratorProfile
from rutts.resources import measure
from rutts.validation import ValidationError, raise_for, text_problems

SILENCE_THRESHOLD = 1e-4
SUITE_FILE = "excerpts_ru.json"
ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


@dataclass(frozen=True)
class Excerpt:
    id: str
    category: str
    text: str


def load_excerpts(path: Path) -> list[Excerpt]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError([f"excerpts file not found: {path}"]) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError([f"excerpts file is not valid UTF-8 JSON: {path} ({exc})"]) from exc
    return _parse_excerpts(document)


def load_suite() -> list[Excerpt]:
    text = package_resources.files("rutts").joinpath("data", SUITE_FILE).read_text(encoding="utf-8")
    return _parse_excerpts(json.loads(text))


def _parse_excerpts(document: dict[str, Any]) -> list[Excerpt]:
    excerpts = [
        Excerpt(id=str(item["id"]), category=str(item["category"]), text=str(item["text"]))
        for item in document["excerpts"]
    ]
    problems: list[str] = []
    for excerpt in excerpts:
        if not re.match(ID_PATTERN, excerpt.id):
            problems.append(f"excerpt id '{excerpt.id}' must be lowercase letters, digits, - or _")
        problems += text_problems(excerpt.text, f"excerpt '{excerpt.id}'")
    raise_for(problems)
    return excerpts


def audio_problems(samples: np.ndarray) -> str | None:
    if samples.size == 0:
        return "engine returned empty audio"
    if not np.all(np.isfinite(samples)):
        return "engine returned non-finite audio samples"
    if float(np.max(np.abs(samples))) < SILENCE_THRESHOLD:
        return "engine returned silent audio"
    return None


def _write_json(path: Path, document: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def _write_summary(path: Path, document: dict[str, Any]) -> None:
    lines = [
        "# Preview results",
        "",
        f"- Engine: {document['engine']['config']['engine_id']} "
        f"({document['engine']['config']['precision']}, "
        f"{document['engine']['config']['model_repo']} @ "
        f"{document['engine']['config']['model_revision']})",
        f"- Fingerprint: {document['engine']['fingerprint']}",
        f"- Narrator: {document['narrator']['name']} (reference sha256 "
        f"{document['narrator']['reference_sha256']})",
        f"- Retry budget: {document['retry_budget']}",
        "",
        "| Excerpt | Category | Status | Attempts | Last render (s) | Audio (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in document["excerpts"]:
        last = item["attempts"][-1]
        duration = last["audio_duration_seconds"]
        audio = f"{duration:.2f}" if duration is not None else "-"
        lines.append(
            f"| {item['id']} | {item['category']} | {item['status']} | "
            f"{len(item['attempts'])} | {last['render_seconds']:.2f} | {audio} |"
        )
    failures = [
        (item["id"], attempt)
        for item in document["excerpts"]
        for attempt in item["attempts"]
        if attempt["status"] == "failed"
    ]
    if failures:
        lines += ["", "## Failed attempts", ""]
        lines += [f"- {excerpt_id} attempt {a['attempt']}: {a['error']}" for excerpt_id, a in failures]
    atomic_write_text(path, "\n".join(lines) + "\n")


def _check_free(out_dir: Path, excerpts: list[Excerpt]) -> None:
    existing = [
        str(path)
        for excerpt in excerpts
        for path in (out_dir / excerpt.id).glob("attempt-*.wav")
    ]
    if existing:
        raise FileExistsError(
            "preview output already contains attempts; choose a new --out directory: "
            + ", ".join(existing[:3])
        )


def _render_attempt(
    engine: SynthesisEngine,
    narrator: NarratorProfile,
    excerpt: Excerpt,
    out_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt": attempt,
        "status": "failed",
        "error": None,
        "file": None,
        "render_seconds": 0.0,
        "audio_duration_seconds": None,
        "resources": {},
    }
    samples = np.zeros(0, dtype=np.float32)
    sample_rate = 0
    with measure() as sample:
        try:
            result = engine.synthesize(excerpt.text, narrator)
            samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
            sample_rate = result.sample_rate
            problem = audio_problems(samples)
            if problem is None:
                record["status"] = "ok"
            else:
                record["error"] = problem
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    record["render_seconds"] = sample.elapsed_seconds
    record["resources"] = sample.metrics
    if samples.size > 0:
        relative = f"{excerpt.id}/attempt-{attempt}.wav"
        with atomic_path(out_dir / relative) as temp:
            sf.write(str(temp), samples, sample_rate, format="WAV", subtype="FLOAT")
        record["file"] = relative
        record["audio_duration_seconds"] = samples.size / sample_rate
    return record


def run_preview(
    engine: SynthesisEngine,
    narrator: NarratorProfile,
    excerpts: list[Excerpt],
    out_dir: Path,
    retry_budget: int,
) -> dict[str, Any]:
    if retry_budget < 0:
        raise ValueError("retry budget must be zero or positive")
    _check_free(out_dir, excerpts)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine.load()
    try:
        config = engine.config
        document: dict[str, Any] = {
            "engine": {
                "config": config.to_dict(),
                "fingerprint": config.fingerprint(),
            },
            "narrator": {
                "name": narrator.name,
                "language": narrator.language,
                "provenance": narrator.provenance,
                "reference_sha256": narrator.reference_sha256,
                "reference_file": narrator.reference_audio.name,
            },
            "retry_budget": retry_budget,
            "excerpts": [],
            "summary": {},
        }
        for excerpt in excerpts:
            attempts: list[dict[str, Any]] = []
            status = "failed"
            for attempt in range(1, retry_budget + 2):
                record = _render_attempt(engine, narrator, excerpt, out_dir, attempt)
                attempts.append(record)
                if record["status"] == "ok":
                    status = "ok"
                    break
            document["excerpts"].append(
                {
                    "id": excerpt.id,
                    "category": excerpt.category,
                    "text": excerpt.text,
                    "status": status,
                    "attempts": attempts,
                }
            )
            _persist(out_dir, document)
    finally:
        engine.unload()
    return document


def _persist(out_dir: Path, document: dict[str, Any]) -> None:
    ok = sum(1 for item in document["excerpts"] if item["status"] == "ok")
    document["summary"] = {
        "excerpts": len(document["excerpts"]),
        "ok": ok,
        "failed": len(document["excerpts"]) - ok,
    }
    _write_json(out_dir / "results.json", document)
    _write_summary(out_dir / "results.md", document)
