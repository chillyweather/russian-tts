from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_write_text
from rutts.book.project import BookProject, read_json, sha256_file, sha256_text, write_json
from rutts.book.render import MANIFEST_FILE, PLAN_FILE
from rutts.validation import ValidationError
from rutts.verification import audiochecks
from rutts.verification.asr import Transcriber, Transcript
from rutts.verification.compare import Comparison, compare

VERIFIER_VERSION = 1
NORMALISATION_VERSION = 1
FINDINGS_DIR = "review/findings"
REPORT_FILE = "report.md"
NO_FINDINGS = "no_automatic_findings"
FLAGGED = "findings"
MANUAL_REVIEW_REQUIRED = [
    "Lexical stress: ASR does not judge whether stressed syllables are placed correctly; listen to stressed words, especially homographs.",
    "е/ё pronunciation: comparison folds ё to е, so ASR cannot confirm which vowel was spoken; listen to words with ё.",
    "Prosody: intonation, emphasis and phrasing are not measured; only silence and speaking-rate checks are automatic.",
    "Listening comfort: pacing, voice quality and distracting joins need a listening pass.",
]


@dataclass(frozen=True)
class VerifyResult:
    chapter_id: str
    target: str
    checked: int
    reused: int
    report: Path


def findings_dir(project: BookProject, chapter_id: str, target: str) -> Path:
    return project.directory / FINDINGS_DIR / chapter_id / target


def _settings(transcriber: Transcriber) -> dict[str, Any]:
    try:
        config = transcriber.config
    except Exception as exc:
        raise ValidationError([f"transcriber is not usable offline: {exc}"]) from exc
    return {
        "verifier_version": VERIFIER_VERSION,
        "normalisation_version": NORMALISATION_VERSION,
        "transcriber": config.to_dict(),
        "transcriber_fingerprint": config.fingerprint(),
    }


def _fingerprint(settings: dict[str, Any]) -> str:
    return sha256_text(json.dumps(settings, sort_keys=True, separators=(",", ":")))


def _simple_finding(kind: str, severity: str, expected: str, heard: str, chars: int, audio: float, reason: str) -> dict[str, Any]:
    return {
        "type": kind,
        "severity": severity,
        "expected": expected,
        "heard": heard,
        "location": {"chars": [0, chars], "audio_seconds": [0.0, audio]},
        "evidence": {"reason": reason},
    }


def _audio_path(target_dir: Path, record: dict[str, Any]) -> Path | None:
    if record.get("status") != "ok" or not record.get("audio_file"):
        return None
    return target_dir / str(record["audio_file"])


def _audio_problem(path: Path | None, record: dict[str, Any]) -> str | None:
    if path is None:
        return f"chunk has no ok audio (status: {record.get('status')})"
    if not path.is_file():
        return "audio file is missing"
    if sha256_file(path) != record.get("audio_sha256"):
        return "audio file does not match its recorded sha256"
    return None


def _read_samples(path: Path) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(rate)


def _check_audio(path: Path, expected: str) -> list[dict[str, Any]]:
    samples, rate = _read_samples(path)
    duration = samples.size / rate
    flags = [flag.to_dict() for flag in audiochecks.suspicious_silence(samples, rate)]
    flags += [flag.to_dict() for flag in [audiochecks.clipping(samples, rate)] if flag is not None]
    rate_flag = audiochecks.suspicious_duration(duration, len(expected.strip()))
    if rate_flag is not None:
        flags.append(rate_flag.to_dict())
    return flags


def _comparison_findings(comparison: Comparison) -> list[dict[str, Any]]:
    return [finding.to_dict() for finding in comparison.findings]


def _chunk_document(
    chapter_id: str,
    target: str,
    record: dict[str, Any],
    settings: dict[str, Any],
    fingerprint: str,
    expected: str,
    findings: list[dict[str, Any]],
    transcript: Transcript | None,
    comparison: Comparison | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "chapter_id": chapter_id,
        "target": target,
        "chunk_id": record["id"],
        "status": FLAGGED if findings else NO_FINDINGS,
        "audio_file": record.get("audio_file"),
        "audio_sha256": record.get("audio_sha256"),
        "chunk_fingerprint": record.get("fingerprint"),
        "verifier": settings,
        "verifier_fingerprint": fingerprint,
        "expected_text": expected,
        "transcript": transcript.to_dict() if transcript is not None else None,
        "comparison": comparison.to_dict() if comparison is not None else None,
        "findings": findings,
        "manual_review_required": MANUAL_REVIEW_REQUIRED,
    }


def _is_current(path: Path, record: dict[str, Any], fingerprint: str) -> bool:
    if not path.is_file():
        return False
    previous = read_json(path)
    return (
        previous.get("audio_sha256") == record.get("audio_sha256")
        and previous.get("chunk_fingerprint") == record.get("fingerprint")
        and previous.get("verifier_fingerprint") == fingerprint
    )


def _ensure_loaded(transcriber: Transcriber, loaded: list[bool]) -> None:
    if loaded:
        return
    loaded.append(True)
    try:
        transcriber.load()
    except Exception as exc:
        raise ValidationError([f"transcriber could not load: {exc}"]) from exc


def _check_chunk(
    target_dir: Path,
    record: dict[str, Any],
    expected: str,
    transcriber: Transcriber,
    loaded: list[bool],
) -> tuple[list[dict[str, Any]], Transcript | None, Comparison | None]:
    path = _audio_path(target_dir, record)
    problem = _audio_problem(path, record)
    if path is None or problem is not None:
        reason = problem or "no audio"
        return [_simple_finding("missing_audio", "major", expected, "", len(expected), 0.0, reason)], None, None

    _ensure_loaded(transcriber, loaded)
    try:
        transcript = transcriber.transcribe(path)
    except (KeyboardInterrupt, MemoryError):
        raise
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return [_simple_finding("transcription_failed", "major", expected, "", len(expected), 0.0, reason)], None, None

    comparison = compare(expected, transcript.words)
    return _comparison_findings(comparison) + _check_audio(path, expected), transcript, comparison


def _expected_texts(project: BookProject, chapter_id: str, target: str) -> dict[str, str]:
    """Chunk texts as planned. Stress marks are dropped during comparison, so these offsets match what the reviewer reads."""
    plan_path = project.target_dir(chapter_id, target) / PLAN_FILE
    if not plan_path.is_file():
        raise ValidationError([f"plan is missing for chapter {chapter_id} target {target}; render again"])
    return {str(chunk["id"]): str(chunk["text"]) for chunk in read_json(plan_path)["chunks"]}


def verify_chapter(
    project: BookProject,
    chapter_id: str,
    target: str,
    transcriber: Transcriber,
    *,
    recheck: bool = False,
) -> VerifyResult:
    """Transcribe and check each rendered chunk, reusing findings whose audio and config are unchanged."""
    target_dir = project.target_dir(chapter_id, target)
    manifest = read_json(target_dir / MANIFEST_FILE)
    expected_by_id = _expected_texts(project, chapter_id, target)
    settings = _settings(transcriber)
    fingerprint = _fingerprint(settings)
    out_dir = findings_dir(project, chapter_id, target)
    out_dir.mkdir(parents=True, exist_ok=True)

    checked = 0
    reused = 0
    loaded: list[bool] = []
    try:
        for record in manifest["chunks"]:
            chunk_id = str(record["id"])
            expected = expected_by_id.get(chunk_id)
            if expected is None:
                raise ValidationError([f"chunk {chunk_id} is in the manifest but not in plan.json"])
            chunk_path = out_dir / f"{chunk_id}.json"
            if not recheck and _is_current(chunk_path, record, fingerprint):
                reused += 1
                continue
            findings, transcript, comparison = _check_chunk(target_dir, record, expected, transcriber, loaded)
            document = _chunk_document(
                chapter_id, target, record, settings, fingerprint, expected, findings, transcript, comparison
            )
            write_json(chunk_path, document)
            checked += 1
    finally:
        if loaded:
            transcriber.unload()

    report = write_report(project, chapter_id, target, manifest)
    return VerifyResult(chapter_id=chapter_id, target=target, checked=checked, reused=reused, report=report)


def load_findings(project: BookProject, chapter_id: str, target: str, chunk_id: str | None = None) -> list[dict[str, Any]]:
    directory = findings_dir(project, chapter_id, target)
    if chunk_id is not None:
        path = directory / f"{chunk_id}.json"
        if not path.is_file():
            raise ValidationError([f"no stored findings for {chunk_id}; run book verify first"])
        return [read_json(path)]
    if not directory.is_dir():
        raise ValidationError([f"no stored findings for chapter {chapter_id} target {target}; run book verify first"])
    return [read_json(path) for path in sorted(directory.glob("*.json"))]


def write_report(project: BookProject, chapter_id: str, target: str, manifest: dict[str, Any]) -> Path:
    documents = {document["chunk_id"]: document for document in load_findings(project, chapter_id, target)}
    ordered = [documents[str(record["id"])] for record in manifest["chunks"] if str(record["id"]) in documents]
    counts: dict[str, int] = {}
    for document in ordered:
        for finding in document["findings"]:
            counts[finding["type"]] = counts.get(finding["type"], 0) + 1
    flagged = sum(1 for document in ordered if document["status"] == FLAGGED)
    lines = [
        f"# Verification report: {chapter_id} ({target})",
        "",
        "Automatic checks are a review aid. A chunk marked `no_automatic_findings` has no automatic findings; "
        "it is not verified as correct.",
        "",
        f"- Chunks: {len(ordered)}; with findings: {flagged}; `no_automatic_findings`: {len(ordered) - flagged}",
        "- Finding types: " + (", ".join(f"{kind} {count}" for kind, count in sorted(counts.items())) or "none"),
    ]
    if ordered:
        transcriber = ordered[0]["verifier"]["transcriber"]
        lines.append(
            f"- Transcriber: {transcriber['transcriber_id']} ({transcriber['model_repo']} @ "
            f"{transcriber['model_revision']}, language {transcriber['language']})"
        )
    lines += ["", "## Manual review required", ""]
    lines += [f"- {item}" for item in MANUAL_REVIEW_REQUIRED]
    lines += ["", "## Chunks", ""]
    for document in ordered:
        lines.append(f"### {document['chunk_id']}: {document['status']}")
        lines.append("")
        if not document["findings"]:
            lines += ["No automatic findings.", ""]
            continue
        lines += ["| Type | Severity | Expected | Heard | Chars | Audio (s) |", "| --- | --- | --- | --- | --- | --- |"]
        for finding in document["findings"]:
            chars = finding["location"].get("chars", [])
            audio = finding["location"].get("audio_seconds", [])
            lines.append(
                f"| {finding['type']} | {finding['severity']} | {finding.get('expected', '')} | "
                f"{finding.get('heard', '')} | {chars[0] if chars else '-'}-{chars[1] if chars else '-'} | "
                f"{audio[0] if audio else '-'}-{audio[1] if audio else '-'} |"
            )
        lines.append("")
    path = findings_dir(project, chapter_id, target) / REPORT_FILE
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path
