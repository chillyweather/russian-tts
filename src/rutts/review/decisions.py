from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rutts.book.project import BookProject, read_json, write_json
from rutts.validation import ValidationError

if TYPE_CHECKING:
    from rutts.review.queue import QueueItem

DECISIONS_FILE = "review/decisions.json"
LOG_FILE = "review/decisions-log.json"
SCHEMA_VERSION = 1
STATES = ("resolved", "dismissed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_reason(reason: str | None) -> str:
    text = (reason or "").strip()
    if not text:
        raise ValidationError(["a non-empty --reason is required"])
    return text


def load_decisions(project: BookProject) -> dict[str, dict[str, Any]]:
    """Current decision per item lineage."""
    path = project.directory / DECISIONS_FILE
    if not path.is_file():
        return {}
    return {str(item["lineage"]): item for item in read_json(path)["decisions"]}


def _save(project: BookProject, decisions: dict[str, dict[str, Any]]) -> None:
    ordered = [decisions[key] for key in sorted(decisions)]
    write_json(project.directory / DECISIONS_FILE, {"schema_version": SCHEMA_VERSION, "decisions": ordered})


def _append_log(project: BookProject, entry: dict[str, Any]) -> None:
    path = project.directory / LOG_FILE
    entries: list[dict[str, Any]] = read_json(path)["entries"] if path.is_file() else []
    entries.append({"seq": len(entries) + 1, **entry})
    write_json(path, {"schema_version": SCHEMA_VERSION, "entries": entries})


def load_log(project: BookProject) -> list[dict[str, Any]]:
    path = project.directory / LOG_FILE
    return list(read_json(path)["entries"]) if path.is_file() else []


def decide(project: BookProject, item: QueueItem, state: str, reason: str | None) -> dict[str, Any]:
    """Record an explicit resolution or dismissal, bound to the item's current audio, text and verifier."""
    if state not in STATES:
        raise ValidationError([f"unknown decision state '{state}'"])
    text = require_reason(reason)
    decision = {
        "item_id": item.id,
        "lineage": item.lineage,
        "chapter_id": item.chapter_id,
        "target": item.target,
        "chunk_id": item.chunk_id,
        "type": item.type,
        "state": state,
        "reason": text,
        "timestamp": _now(),
        "audio_sha256": item.audio_sha256,
        "chunk_fingerprint": item.chunk_fingerprint,
        "verifier_fingerprint": item.verifier_fingerprint,
    }
    decisions = load_decisions(project)
    decisions[item.lineage] = decision
    _save(project, decisions)
    _append_log(project, {"action": state, "timestamp": decision["timestamp"], "item_id": item.id, "lineage": item.lineage, "reason": text})
    return decision


def reopen(project: BookProject, item: QueueItem) -> None:
    decisions = load_decisions(project)
    if item.lineage not in decisions:
        raise ValidationError([f"item {item.id} has no decision to reopen"])
    previous = decisions.pop(item.lineage)
    _save(project, decisions)
    _append_log(project, {"action": "reopen", "timestamp": _now(), "item_id": item.id, "lineage": item.lineage, "reverted": previous["state"]})


def decision_is_valid(decision: dict[str, Any], item: QueueItem) -> bool:
    return (
        decision.get("audio_sha256") == item.audio_sha256
        and decision.get("chunk_fingerprint") == item.chunk_fingerprint
        and decision.get("verifier_fingerprint") == item.verifier_fingerprint
    )


def stale_causes(decision: dict[str, Any], item: QueueItem) -> list[str]:
    causes: list[str] = []
    if decision.get("audio_sha256") != item.audio_sha256:
        causes.append("audio changed")
    if decision.get("chunk_fingerprint") != item.chunk_fingerprint:
        causes.append("chunk text or engine/narrator settings changed")
    if decision.get("verifier_fingerprint") != item.verifier_fingerprint:
        causes.append("verifier configuration changed")
    return causes
