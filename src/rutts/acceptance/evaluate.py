from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rutts.acceptance import record as rec
from rutts.acceptance.listening import GATED_TYPES, chapter_signature, gate_failures
from rutts.acceptance.record import EVIDENCED, FAILED, NOT_EVIDENCED, Section
from rutts.acceptance.stages import EXPORT_NAME, STAGES, included_chapters, job_target, stage_problems
from rutts.book.assemble import master_is_stale
from rutts.book.job import COVERAGE_FILE, DIAGNOSTICS_DIR, INVOCATIONS_FILE, JOB_FILE, RESOURCES_FILE, STATE_FILE
from rutts.book.project import CHAPTERS_DIR, BookProject, read_json
from rutts.book.render import MANIFEST_FILE, MEMORY_EXHAUSTION_MARKERS
from rutts.export.build import BUILDS_FILE, export_dir, record_path
from rutts.export.status import export_status
from rutts.pronunciation.store import load_corrections
from rutts.review import decisions as decision_store
from rutts.review import manual as manual_store
from rutts.review.queue import build_queue

HEX_REVISION = re.compile(r"[0-9a-f]{40}")
UNLICENSED = frozenset({"", "unverified", "unknown"})
INTERRUPTED = "interrupted"
STOPPED = "stopped_resource_pressure"
COMPLETED = "completed"


def _licensed(value: object) -> bool:
    return str(value or "").strip().lower() not in UNLICENSED


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        seen.setdefault(json.dumps(item, sort_keys=True), item)
    return list(seen.values())


def _verifier_configs(project: BookProject) -> list[dict[str, Any]]:
    root = project.directory / "review" / "findings"
    documents = [read_json(path) for path in sorted(root.glob("*/*/*.json"))] if root.is_dir() else []
    return _unique(
        [
            {key: document.get("verifier", {}).get("transcriber", {}).get(key) for key in ("transcriber_id", "model_repo", "model_revision", "runtime_name", "runtime_version", "license")}
            for document in documents
        ]
    )


def _assistant_configs(project: BookProject) -> list[dict[str, Any]]:
    root = project.directory / "narration" / "suggestions"
    documents = [read_json(path) for path in sorted(root.glob("*.json"))] if root.is_dir() else []
    return _unique(
        [
            {key: document["assistant"].get(key) for key in ("assistant_id", "package", "package_version", "model", "license")}
            for document in documents
            if "assistant" in document
        ]
    )


def configuration(project: BookProject) -> Section:
    job_path = project.directory / JOB_FILE
    if not job_path.is_file():
        return Section(NOT_EVIDENCED, {}, ["no frozen job configuration; run book generate"])
    job = read_json(job_path)
    engine = job["engine"]
    narrator = job["narrator"]
    verifiers = _verifier_configs(project)
    assistants = _assistant_configs(project)
    failures: list[str] = []
    revision = str(engine.get("model_revision") or "")
    if not HEX_REVISION.fullmatch(revision):
        failures.append(f"model revision is unresolved: {revision!r}")
    if not _licensed(engine.get("license")):
        failures.append("engine license is missing")
    if not str(narrator.get("provenance") or "").strip():
        failures.append("narrator provenance is empty")
    failures += [f"verifier license is missing for {item['transcriber_id']}; re-run book verify" for item in verifiers if not _licensed(item["license"])]
    failures += [f"stress assistant license is missing for {item['assistant_id']}" for item in assistants if not _licensed(item["license"])]
    evidence: dict[str, Any] = {
        "job": {"target": job["target"], "created_at": job["created_at"], "policy": job["policy"]},
        "engine": {key: engine.get(key) for key in ("engine_id", "model_repo", "model_revision", "precision", "runtime_name", "runtime_version", "license", "settings")}
        | {"fingerprint": job["engine_fingerprint"]},
        "narrator": {key: narrator.get(key) for key in ("name", "language", "provenance", "reference_sha256")},
        "verifier": verifiers,
        "stress_assistant": assistants or None,
    }
    if failures:
        return Section(FAILED, evidence, failures)
    missing = [] if verifiers else ["no verification run recorded, so the verifier licence is not frozen; run book verify"]
    return Section(NOT_EVIDENCED if missing else EVIDENCED, evidence, missing)


def offline(project: BookProject, inputs: dict[str, Any]) -> Section:
    evidence: dict[str, Any] = {}
    missing: list[str] = []
    for stage in STAGES:
        observed = [step for step in inputs["offline"] if step["stage"] == stage and step["network"] == "offline"]
        problems = stage_problems(project, stage)
        evidenced = bool(observed) and not problems
        evidence[stage] = {
            "evidenced": evidenced,
            "offline_observations": len(observed),
            "last_observed_at": observed[-1]["recorded_at"] if observed else None,
            "current": not problems,
        }
        if not observed:
            missing.append(f"{stage}: not observed complete with networking offline; run book accept step {project.directory} {stage}")
        elif problems:
            missing.append(f"{stage}: recorded offline, but no longer current: {problems[0]}")
    evidence["steps"] = inputs["offline"]
    status = EVIDENCED if not missing else NOT_EVIDENCED
    return Section(status, evidence, missing)


def _counts(items: list[Any]) -> dict[str, int]:
    return {state: sum(1 for item in items if item.status == state) for state in ("open", "stale", "resolved", "dismissed")}


def coverage(project: BookProject, inputs: dict[str, Any]) -> Section:
    path = project.directory / COVERAGE_FILE
    if not path.is_file():
        return Section(NOT_EVIDENCED, {}, ["no generation coverage report; run book generate"])
    report = read_json(path)
    source = project.read_source()
    segments = report["segments"]
    excluded = [
        {"segment_id": item["segment_id"], "chapter_id": item["chapter_id"], "kind": item["kind"], "reason": item["exclusion_reason"]}
        for item in segments
        if not item["included"]
    ]
    chapter_exclusions = [{"chapter_id": chapter.id, "reason": chapter.exclusion_reason} for chapter in source.chapters if not chapter.included]
    included = included_chapters(project)
    masters: dict[str, Any] = {}
    missing: list[str] = []
    for chapter_id in included:
        master_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
        if not master_path.is_file():
            masters[chapter_id] = {"assembled": False}
            missing.append(f"{chapter_id}: no assembled master; run book assemble")
            continue
        master = read_json(master_path)
        stale = master_is_stale(project, chapter_id)
        masters[chapter_id] = {
            "assembled": True,
            "complete": master["complete"],
            "stale": stale,
            "uncovered_segments": master["uncovered_segments"],
            "missing_chunks": master["missing_chunks"],
            "failed_chunks": master["failed_chunks"],
        }
        if not master["complete"] or stale:
            missing.append(f"{chapter_id}: master is {'stale' if stale else 'incomplete'}")
    missing = [f"uncovered segment {segment_id}" for segment_id in report["uncovered_segments"]] + missing
    queue = build_queue(project, None, str(report["target"]))
    listened = inputs["listening"]["chapter_id"] if inputs["listening"] else None
    fully_listened = [listened] if listened else []
    unreviewed = [chapter_id for chapter_id in included if chapter_id not in fully_listened]
    evidence = {
        "target": report["target"],
        "totals": report["totals"],
        "narratable_segments": sum(1 for item in segments if item["narratable"]),
        "narratable_covered": sum(1 for item in segments if item["narratable"] and item["covered"]),
        "segments_accounted": report["totals"]["included"] + report["totals"]["excluded"] == len(segments),
        "uncovered_segments": report["uncovered_segments"],
        "excluded_segments": excluded,
        "excluded_chapters": chapter_exclusions,
        "masters": masters,
        "review_items": _counts(queue),
        "fully_listened_chapters": fully_listened,
        "unreviewed_chapters": unreviewed,
        "disclosure": (
            f"Full listening review covered: {', '.join(fully_listened) or 'none'}. "
            f"Unreviewed chapters: {', '.join(unreviewed) or 'none'}. "
            "Passing a sampled chapter does not prove later chapters are error-free."
        ),
    }
    if not report["complete"] or missing:
        return Section(NOT_EVIDENCED, evidence, missing)
    return Section(EVIDENCED, evidence, [])


def _corrections_for(project: BookProject, chapter_id: str) -> dict[str, int]:
    corrections = load_corrections(project)
    prefix = f"{chapter_id}-"
    decisions = decision_store.load_decisions(project).values()
    manual = [finding for finding in manual_store.load_manual(project) if finding["chapter_id"] == chapter_id]
    return {
        "override_changes": sum(
            1 for item in corrections.history if item["target"] == "override" and str(item["entry"]["segment_id"]).startswith(prefix)
        ),
        "dictionary_changes_book_wide": sum(1 for item in corrections.history if item["target"] == "dictionary"),
        "markup_changes_book_wide": sum(1 for item in corrections.history if item["target"] == "markup"),
        "review_resolved": sum(1 for item in decisions if item["chapter_id"] == chapter_id and item["state"] == "resolved"),
        "review_dismissed": sum(1 for item in decisions if item["chapter_id"] == chapter_id and item["state"] == "dismissed"),
        "manual_resolved": sum(1 for item in manual if item["state"] == "resolved"),
        "manual_dismissed": sum(1 for item in manual if item["state"] == "dismissed"),
    }


def listening(project: BookProject, inputs: dict[str, Any]) -> Section:
    review = inputs["listening"]
    if review is None:
        return Section(NOT_EVIDENCED, {}, [f"no listening review recorded; run book accept listen {project.directory} --chapter <id>"])
    chapter_id = review["chapter_id"]
    target = review["target"]
    evidence: dict[str, Any] = {
        "chapter_id": chapter_id,
        "target": target,
        "recorded_at": review["recorded_at"],
        "source": review["source"],
        "copied_scores": review["copied_scores"],
        "listened_whole_chapter": review["listened_whole_chapter"],
        "scores": review["scores"],
        "notes": review["notes"],
        "corrections": _corrections_for(project, chapter_id),
    }
    if not review["listened_whole_chapter"]:
        return Section(NOT_EVIDENCED, evidence, ["the listener did not confirm listening to the whole chapter"])
    if chapter_signature(project, chapter_id, target) != review["chapter_signature"]:
        return Section(NOT_EVIDENCED, evidence, [f"{chapter_id} audio changed since the listening review; listen again"])
    failures = gate_failures(review["scores"])
    unresolved = [item for item in build_queue(project, [chapter_id], target) if item.is_open and item.type in GATED_TYPES]
    evidence["unresolved_findings"] = [{"id": item.id, "type": item.type, "status": item.status} for item in unresolved]
    failures += [f"unresolved {item.type} finding {item.id} in {chapter_id}" for item in unresolved]
    return Section(FAILED if failures else EVIDENCED, evidence, failures)


def _passed_pairs(invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, interrupted in enumerate(invocations):
        if interrupted["status"] not in (INTERRUPTED, STOPPED):
            continue
        for later in invocations[index + 1 :]:
            if later["status"] == COMPLETED and later["ok_at_start"] >= interrupted["ok_at_end"]:
                pairs.append(
                    {
                        "interrupted_invocation": interrupted["invocation"],
                        "interrupted_status": interrupted["status"],
                        "valid_chunks_at_interruption": interrupted["ok_at_end"],
                        "completed_invocation": later["invocation"],
                        "valid_chunks_at_restart": later["ok_at_start"],
                    }
                )
                break
    return pairs


def _rebuilt_pairs(builds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = [build for build in builds if build["status"] == "ready"]
    pairs: list[dict[str, Any]] = []
    for index, earlier in enumerate(ready):
        for later in ready[index + 1 :]:
            if later["signature"] != earlier["signature"]:
                pairs.append({"superseded_build": earlier["seq"], "rebuilt_build": later["seq"]})
                break
    return pairs


def _regenerated(project: BookProject, target: str | None) -> tuple[list[str], int]:
    if target is None:
        return [], 0
    regenerated: list[str] = []
    total = 0
    for chapter_id in included_chapters(project):
        path = project.target_dir(chapter_id, target) / MANIFEST_FILE
        if not path.is_file():
            continue
        records = read_json(path)["chunks"]
        total += len(records)
        regenerated += [
            record["id"]
            for record in records
            if record["status"] == "ok"
            and any(event.get("event") == "invalidated" and event.get("reason") == "text_changed" for event in record.get("history", []))
        ]
    return regenerated, total


def exercises(project: BookProject) -> Section:
    target = job_target(project)
    corrections = [item["seq"] for item in load_corrections(project).history if item["target"] in ("dictionary", "override")]
    regenerated, total = _regenerated(project, target)
    pronunciation = bool(corrections) and bool(regenerated) and len(regenerated) < total
    restarts = _passed_pairs(_jsonl(project.directory / INVOCATIONS_FILE))
    rebuilds = _rebuilt_pairs(_build_entries(export_dir(project, EXPORT_NAME) / BUILDS_FILE))
    evidence = {
        "pronunciation_correction": {
            "history_seq": corrections,
            "regenerated_chunks": regenerated,
            "total_chunks": total,
            "evidenced": pronunciation,
        },
        "interruption_and_restart": {"pairs": restarts, "evidenced": bool(restarts)},
        "export_rebuilt_after_change": {"pairs": rebuilds, "evidenced": bool(rebuilds)},
    }
    labels = {
        "pronunciation_correction": "a pronunciation correction followed by targeted regeneration (a text_changed chunk regenerated, fewer than all chunks)",
        "interruption_and_restart": "an interruption followed by a restart that kept the valid chunks",
        "export_rebuilt_after_change": "an export rebuilt after a change, superseding an earlier ready export",
    }
    missing = [f"not yet exercised: {labels[name]}" for name in labels if not evidence[name]["evidenced"]]
    return Section(NOT_EVIDENCED if missing else EVIDENCED, evidence, missing)


def _build_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = read_json(path)["entries"]
    return entries


def _memory_errors(project: BookProject, target: str | None, diagnostics: list[dict[str, Any]]) -> list[str]:
    errors = [
        f"diagnostics {item['file']}: {reason}"
        for item in diagnostics
        for reason in item["reasons"]
        if "memory exhaustion" in reason
    ]
    if target is None:
        return errors
    for chapter_id in included_chapters(project):
        path = project.target_dir(chapter_id, target) / MANIFEST_FILE
        if not path.is_file():
            continue
        for record in read_json(path)["chunks"]:
            for attempt in record.get("attempts", []):
                message = str(attempt.get("error") or "").lower()
                if any(marker in message for marker in MEMORY_EXHAUSTION_MARKERS):
                    errors.append(f"{record['id']} attempt {attempt['attempt']}: {attempt['error']}")
    return errors


def resources(project: BookProject) -> Section:
    state_path = project.directory / STATE_FILE
    if not state_path.is_file():
        return Section(NOT_EVIDENCED, {}, ["no generation job state; run book generate"])
    state = read_json(state_path)
    throughput = state["throughput"]
    measured = state["resources"]
    samples = [
        measurement
        for line in _jsonl(project.directory / RESOURCES_FILE)
        for measurement in (line.get("before"), line.get("after"))
        if measurement is not None
    ]
    diagnostics: list[dict[str, Any]] = []
    diagnostics_dir = project.directory / DIAGNOSTICS_DIR
    for path in sorted(diagnostics_dir.glob("*.json")) if diagnostics_dir.is_dir() else []:
        document = read_json(path)
        diagnostics.append(
            {"file": path.relative_to(project.directory).as_posix(), "status": document["status"], "reasons": document["reasons"], "chapter": document["chapter"], "chunk": document["chunk"]}
        )
    invocations = _jsonl(project.directory / INVOCATIONS_FILE)
    unrecovered = [
        item["invocation"]
        for index, item in enumerate(invocations)
        if item["status"] == STOPPED and not any(later["status"] == COMPLETED for later in invocations[index + 1 :])
    ]
    target = job_target(project)
    attempts = failed_attempts = retried = 0
    if target is not None:
        for chapter_id in included_chapters(project):
            path = project.target_dir(chapter_id, target) / MANIFEST_FILE
            if not path.is_file():
                continue
            for record in read_json(path)["chunks"]:
                count = len(record.get("attempts", []))
                attempts += count
                failed_attempts += sum(1 for attempt in record.get("attempts", []) if attempt["status"] == "failed")
                retried += 1 if count > 1 else 0
    mlx = [item["mlx_peak_memory_bytes"] for item in samples if isinstance(item.get("mlx_peak_memory_bytes"), int)]
    swap = [item["swap_used_bytes"] for item in samples if isinstance(item.get("swap_used_bytes"), int)]
    memory_errors = _memory_errors(project, target, diagnostics)
    evidence: dict[str, Any] = {
        "job_status": state["status"],
        "invocations": state["invocations"],
        "rendered_chunks": throughput["rendered_chunks"],
        "render_seconds": throughput["render_seconds"],
        "audio_seconds": throughput["audio_seconds"],
        "render_seconds_per_audio_second": throughput["render_seconds_per_audio_second"],
        "chunks_with_retries": retried,
        "attempts": attempts,
        "failed_attempts": failed_attempts,
        "peak_rss_bytes": measured.get("max_peak_rss_bytes"),
        "mlx_peak_memory_bytes": max(mlx) if mlx else None,
        "max_swap_growth_bytes": max(swap) - swap[0] if swap else None,
        "min_memory_free_percent": measured.get("min_memory_free_percent"),
        "samples": len(samples),
        "resource_stops": diagnostics,
        "memory_exhaustion_errors": memory_errors,
        "unrecovered_stop_invocations": unrecovered,
    }
    if memory_errors or unrecovered:
        reasons = [f"memory exhaustion: {item}" for item in memory_errors]
        reasons += [f"resource stop in invocation {number} was not recovered" for number in unrecovered]
        return Section(FAILED, evidence, reasons)
    missing: list[str] = []
    if state["status"] != COMPLETED:
        missing.append(f"generation job status is {state['status']}; complete the job")
    if throughput["rendered_chunks"] == 0 or measured.get("max_peak_rss_bytes") is None:
        missing.append("no measured render metrics")
    return Section(NOT_EVIDENCED if missing else EVIDENCED, evidence, missing)


def output(project: BookProject, inputs: dict[str, Any]) -> Section:
    path = record_path(project, EXPORT_NAME)
    if not path.is_file():
        return Section(NOT_EVIDENCED, {}, [f"no export {EXPORT_NAME}; run book export"])
    export = read_json(path)
    status = export_status(project, EXPORT_NAME, None)
    m4b = next((entry for entry in export["files"] if entry["format"] == "m4b"), None)
    playback = inputs["playback"]
    confirmed = (
        playback is not None
        and m4b is not None
        and playback["m4b_sha256"] == m4b["sha256"]
        and all(playback["checks"].values())
    )
    evidence: dict[str, Any] = {
        "export": EXPORT_NAME,
        "kind": export["kind"],
        "export_status": export["status"],
        "currency": status.state,
        "changes": list(status.changes),
        "validation": {"ok": export["validation"]["ok"], "checks": export["validation"]["checks"]},
        "chapters": export["chapters"],
        "metadata": export["metadata"],
        "lossless": export["lossless"],
        "files": export["files"],
        "playback": playback,
    }
    if export["status"] != "ready":
        return Section(FAILED, evidence, [f"export validation failed: {item['name']}" for item in export["validation"]["checks"] if not item["ok"]])
    missing: list[str] = []
    if status.state != "ready":
        missing.append(f"export is {status.state}: {'; '.join(status.changes) or 'rebuild it'}")
    if not export["lossless"]["retained"]:
        return Section(FAILED, evidence, ["lossless chunks or masters changed during export"])
    if not confirmed:
        missing.append(f"playback not confirmed; run book accept playback {project.directory} --player NAME")
    return Section(NOT_EVIDENCED if missing else EVIDENCED, evidence, missing)


def evaluate(project: BookProject, inputs: dict[str, Any]) -> dict[str, Section]:
    return {
        "configuration": configuration(project),
        "offline": offline(project, inputs),
        "coverage": coverage(project, inputs),
        "listening": listening(project, inputs),
        "exercises": exercises(project),
        "resources": resources(project),
        "output": output(project, inputs),
    }


def refresh(project: BookProject, inputs: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every section from the disk and the recorded inputs, then persist the record."""
    return rec.save(project, evaluate(project, inputs), inputs)
