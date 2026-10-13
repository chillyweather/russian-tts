from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rutts.book.assemble import master_is_stale
from rutts.book.project import CHAPTERS_DIR, BookProject, read_json
from rutts.book.render import MANIFEST_FILE
from rutts.review.queue import build_queue, chunk_states
from rutts.verification.verify import findings_dir


@dataclass(frozen=True)
class ExportState:
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Eligibility:
    reviewed: ExportState
    draft: ExportState


def _verified(project: BookProject, chapter_id: str, target: str, chunk_id: str, record: dict[str, object]) -> bool:
    path = findings_dir(project, chapter_id, target) / f"{chunk_id}.json"
    if not path.is_file():
        return False
    document = read_json(path)
    return document.get("audio_sha256") == record.get("audio_sha256") and document.get("chunk_fingerprint") == record.get("fingerprint")


def _master_problems(project: BookProject, chapter_id: str, target: str) -> list[str]:
    report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
    if not report_path.is_file():
        return [f"{chapter_id}: no assembled master; run book assemble"]
    report = read_json(report_path)
    if report["target"] != target:
        return [f"{chapter_id}: master was assembled for target {report['target']}, not {target}"]
    problems = [] if report["complete"] else [f"{chapter_id}: master coverage is incomplete"]
    if master_is_stale(project, chapter_id):
        problems.append(f"{chapter_id}: master is stale; run book assemble")
    return problems


def export_eligibility(project: BookProject, chapters: Sequence[str], target: str) -> Eligibility:
    """Reviewed (blocking reasons only) and draft (everything unreviewed) states for the given chapters."""
    blocking: list[str] = []
    unreviewed: list[str] = []
    audio_chunks = 0
    source = project.read_source()
    included = [chapter_id for chapter_id in chapters if source.chapter(chapter_id).included]
    if not included:
        blocking.append("no included chapters to export")
    for chapter_id in included:
        if not (project.target_dir(chapter_id, target) / MANIFEST_FILE).is_file():
            blocking.append(f"{chapter_id}: not rendered for target {target}")
            continue
        manifest = read_json(project.target_dir(chapter_id, target) / MANIFEST_FILE)
        states = chunk_states(project, chapter_id, target)
        for record in manifest["chunks"]:
            chunk_id = str(record["id"])
            state = states[chunk_id]
            if state is not None:
                blocking.append(f"{chunk_id}: {state} chunk")
                continue
            audio_chunks += 1
            if not _verified(project, chapter_id, target, chunk_id, record):
                blocking.append(f"{chunk_id}: audio not verified on its current audio; run book review recheck")
        blocking += _master_problems(project, chapter_id, target)
        for item in build_queue(project, [chapter_id], target):
            if not item.is_open or item.origin == "chunk":
                continue
            text = f"{item.chunk_id}: open {item.type} ({item.id})"
            if item.is_content_failure or item.origin == "manual":
                blocking.append(text)
            else:
                unreviewed.append(text)
    reviewed = ExportState(eligible=not blocking, reasons=tuple(blocking))
    draft = ExportState(eligible=audio_chunks > 0, reasons=tuple(blocking + unreviewed))
    return Eligibility(reviewed=reviewed, draft=draft)
