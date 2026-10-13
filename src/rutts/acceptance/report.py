from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rutts.acceptance.record import REPORT_FILE, SECTIONS
from rutts.acceptance.stages import job_target
from rutts.atomic import atomic_write_text
from rutts.book.project import BookProject

LIMITATIONS = (
    "Automatic ASR findings are review aids; a chunk with no automatic findings is not verified as correct.",
    "Stress, е/ё, prosody and comfort are only checked by listening, and only the acceptance chapter was listened to in full.",
    "Stress hints are stripped for engines without a verified markup convention.",
    "Chapter headings are not narrated.",
    "The EPUB path and the Whisper transcriber are not verified on a real book in this repository.",
)


def _verification_summary(project: BookProject) -> dict[str, Any] | None:
    target = job_target(project)
    if target is None:
        return None
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(project.directory.joinpath("review", "findings").glob(f"*/{target}/*.json"))
    ]
    if not documents:
        return None
    kinds: Counter[str] = Counter(finding["type"] for document in documents for finding in document["findings"])
    flagged = sum(1 for document in documents if document["status"] == "findings")
    return {"target": target, "chunks": len(documents), "flagged": flagged, "without_findings": len(documents) - flagged, "finding_types": dict(sorted(kinds.items()))}


def _line(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _listed(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- none"]


def render(project: BookProject, document: dict[str, Any], inputs: dict[str, Any]) -> str:
    sections = document["sections"]
    config = sections["configuration"]["evidence"]
    engine = config.get("engine", {})
    narrator = config.get("narrator", {})
    lines = [
        f"# Acceptance record: {project.title}",
        "",
        f"Overall result: **{document['overall']}** (updated {document['updated_at']}).",
        "",
        "## Sections",
        "",
        "| Section | Status |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {sections[name]['status']} |" for name in SECTIONS]
    lines += ["", "## Configuration and licences", ""]
    if engine:
        lines += [
            f"- Engine: {engine['engine_id']}, {engine['model_repo']} @ {engine['model_revision']}, precision {engine['precision']}",
            f"- Runtime: {engine['runtime_name']} {engine['runtime_version']}; configuration fingerprint {str(engine['fingerprint'])[:12]}",
            f"- Engine licence: {engine['license']}",
        ]
    if narrator:
        lines += [
            f"- Narrator: {narrator['name']} (reference sha256 {str(narrator['reference_sha256'])[:12]})",
            f"- Reference provenance: {narrator['provenance']}",
        ]
    for verifier in config.get("verifier", []):
        lines.append(f"- Verifier: {verifier['transcriber_id']}, {verifier['model_repo']} @ {verifier['model_revision']}; licence {verifier['license']}")
    assistants = config.get("stress_assistant") or []
    lines.append(
        "- Stress assistant: "
        + ("; ".join(f"{item['assistant_id']} licence {item['license']}" for item in assistants) if assistants else "not used")
    )
    lines += ["", "## Evidenced and not evidenced", ""]
    for name in SECTIONS:
        section = sections[name]
        lines.append(f"- {name}: {section['status']}")
        lines += [f"    - missing: {item}" for item in section["missing"]]
    lines += ["", "## Offline workflow steps", ""]
    steps = inputs["offline"]
    if not steps:
        lines.append("- none recorded")
    for step in steps:
        counted = "counts as offline evidence" if step["network"] == "offline" else "does not count as offline evidence"
        lines.append(f"- {step['stage']} at {step['recorded_at']}: network {step['network']} ({counted})")
    coverage = sections["coverage"]["evidence"]
    lines += ["", "## Coverage", ""]
    if coverage:
        totals = coverage["totals"]
        lines += [
            f"- Segments: {totals['segments']} total, {totals['included']} included, {totals['excluded']} excluded; "
            f"narratable paragraphs {coverage['narratable_covered']} of {coverage['narratable_segments']} covered, "
            f"{totals['uncovered']} uncovered (headings and scene breaks are not narrated and are not counted as uncovered)",
            f"- Review items: {coverage['review_items']}",
            f"- {coverage['disclosure']}",
            "- Uncovered segments:",
        ]
        lines += [f"    - {item}" for item in coverage["uncovered_segments"]] or ["    - none"]
        lines.append("- Deliberate exclusions:")
        exclusions = [f"{item['chapter_id']}: {item['reason']}" for item in coverage["excluded_chapters"]]
        exclusions += [f"{item['segment_id']} ({item['chapter_id']}): {item['reason']}" for item in coverage["excluded_segments"]]
        lines += [f"    - {item}" for item in exclusions] or ["    - none"]
    else:
        lines.append("- not evidenced (no coverage report)")
    listening = sections["listening"]["evidence"]
    lines += ["", "## Listening review (scores as entered)", ""]
    if listening:
        lines += [
            f"- Chapter {listening['chapter_id']} ({listening['target']}), recorded {listening['recorded_at']}, source {listening['source']}",
            f"- Listened to the whole chapter: {'yes' if listening['listened_whole_chapter'] else 'no'}",
        ]
        lines += [f"- {key.replace('_', ' ')}: {value}" for key, value in listening["scores"].items()]
        lines.append(f"- Notes: {listening['notes'] or '(none)'}")
        corrections = listening["corrections"]
        lines += [f"- Corrections: {key.replace('_', ' ')} {value}" for key, value in corrections.items()]
    else:
        lines.append("- not recorded")
    lines += ["", "## Measured throughput and resources", ""]
    resources = sections["resources"]["evidence"]
    if resources:
        for key in (
            "rendered_chunks", "render_seconds", "audio_seconds", "render_seconds_per_audio_second", "attempts",
            "failed_attempts", "chunks_with_retries", "peak_rss_bytes", "mlx_peak_memory_bytes", "max_swap_growth_bytes",
            "min_memory_free_percent", "samples",
        ):
            lines.append(f"- {key.replace('_', ' ')}: {_line(resources[key])}")
        lines.append(f"- resource stops: {len(resources['resource_stops'])}; memory exhaustion errors: {len(resources['memory_exhaustion_errors'])}")
    else:
        lines.append("- not measured")
    automated = _verification_summary(project)
    lines += ["", "## Automated checks (reported alongside; not a substitute for listening)", ""]
    if automated:
        lines += [
            f"- Target {automated['target']}: {automated['chunks']} chunk(s) checked, {automated['flagged']} with findings, {automated['without_findings']} without automatic findings",
            f"- Finding types: {automated['finding_types'] or 'none'}",
        ]
    else:
        lines.append("- no verification run recorded")
    output = sections["output"]["evidence"]
    lines += ["", "## Output", ""]
    if output:
        lines.append(f"- Export {output['export']} ({output['export_status']}, currency {output['currency']})")
        lines.append(f"- Validation: {'passed' if output['validation']['ok'] else 'FAILED'}; lossless retained: {output['lossless']['retained']}")
        lines += [f"- Chapter {item['id']}: {item['title']} from {item['start_ms'] / 1000:.3f} s" for item in output["chapters"]]
        playback = output["playback"]
        lines.append(
            f"- Playback: {playback['player']} {playback['player_version']}, checks {playback['checks']}"
            if playback
            else "- Playback: not confirmed"
        )
    else:
        lines.append("- no export recorded")
    lines += ["", "## Known limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines += [
        "",
        "Passing a sampled chapter does not prove later chapters are error-free. Only the acceptance chapter received "
        "full listening review; every other chapter is unreviewed unless its own review is recorded.",
        "",
    ]
    return "\n".join(lines)


def write_report(project: BookProject, document: dict[str, Any], inputs: dict[str, Any]) -> Path:
    path = project.directory / REPORT_FILE
    atomic_write_text(path, render(project, document, inputs))
    return path
