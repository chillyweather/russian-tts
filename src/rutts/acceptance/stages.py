from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from rutts.book.job import COVERAGE_FILE, JOB_FILE, STATE_FILE
from rutts.book.project import NARRATION_FILE, SOURCE_FILE, BookProject, read_json
from rutts.book.render import MANIFEST_FILE
from rutts.export.build import record_path
from rutts.export.status import export_status
from rutts.pronunciation import store
from rutts.pronunciation.store import DICTIONARY_FILE, HISTORY_FILE, OVERRIDES_FILE
from rutts.review import manual as manual_store
from rutts.review.queue import chunk_states
from rutts.validation import ValidationError
from rutts.verification.verify import findings_dir

STAGES = ("import", "preparation", "generation", "verification", "correction", "export")
EXPORT_NAME = "audiobook"
MARKUP_FILE = "narration/markup.json"
MANUAL_FILE = manual_store.MANUAL_FILE


NO_JOB = "no generation job has been run; run book generate"


def job_target(project: BookProject) -> str | None:
    path = project.directory / JOB_FILE
    return str(read_json(path)["target"]) if path.is_file() else None


def included_chapters(project: BookProject) -> list[str]:
    return [chapter.id for chapter in project.read_source().chapters if chapter.included]


def _import_problems(project: BookProject) -> list[str]:
    if not (project.directory / SOURCE_FILE).is_file():
        return ["source/source.json is missing; run book import"]
    return []


def _preparation_problems(project: BookProject) -> list[str]:
    if not (project.directory / NARRATION_FILE).is_file():
        return ["narration/narration.json is missing"]
    current = store.preview(project, store.load_corrections(project)).document.to_dict()
    if current != project.read_narration().to_dict():
        return ["narration is out of date with the source and corrections; run book diff or book regenerate"]
    return []


def _chunk_problems(project: BookProject, target: str) -> list[str]:
    problems: list[str] = []
    for chapter_id in included_chapters(project):
        if not (project.target_dir(chapter_id, target) / MANIFEST_FILE).is_file():
            problems.append(f"{chapter_id}: not rendered for target {target}")
            continue
        for chunk_id, state in chunk_states(project, chapter_id, target).items():
            if state is not None:
                problems.append(f"{chunk_id}: {state} chunk")
    return problems


def _generation_problems(project: BookProject) -> list[str]:
    state_path = project.directory / STATE_FILE
    if not state_path.is_file():
        return [NO_JOB]
    state = read_json(state_path)
    if state["status"] != "completed":
        return [f"generation job status is {state['status']}, not completed"]
    coverage_path = project.directory / COVERAGE_FILE
    if not coverage_path.is_file() or not read_json(coverage_path)["complete"]:
        return ["generation coverage is incomplete"]
    return _chunk_problems(project, str(state["target"]))


def _verification_problems(project: BookProject) -> list[str]:
    target = job_target(project)
    if target is None:
        return [NO_JOB]
    problems: list[str] = []
    for chapter_id in included_chapters(project):
        manifest_path = project.target_dir(chapter_id, target) / MANIFEST_FILE
        if not manifest_path.is_file():
            problems.append(f"{chapter_id}: not rendered for target {target}")
            continue
        for record in read_json(manifest_path)["chunks"]:
            if record["status"] != "ok":
                continue
            path = findings_dir(project, chapter_id, target) / f"{record['id']}.json"
            if not path.is_file():
                problems.append(f"{record['id']}: not verified; run book verify")
                continue
            document = read_json(path)
            if document.get("audio_sha256") != record.get("audio_sha256") or document.get("chunk_fingerprint") != record.get("fingerprint"):
                problems.append(f"{record['id']}: verification is out of date for its audio; run book verify")
    return problems


def _correction_problems(project: BookProject) -> list[str]:
    problems = _preparation_problems(project)
    target = job_target(project)
    problems += [NO_JOB] if target is None else _chunk_problems(project, target)
    open_manual = [finding["id"] for finding in manual_store.load_manual(project) if finding["state"] == "open"]
    if open_manual:
        problems.append(f"open manual findings: {', '.join(open_manual)}")
    return problems


def _export_problems(project: BookProject) -> list[str]:
    status = export_status(project, EXPORT_NAME, None)
    if status.state == "ready":
        return []
    detail = "; ".join(status.changes)
    return [f"export {EXPORT_NAME} is {status.state}" + (f": {detail}" if detail else "") + "; run book export"]


def stage_problems(project: BookProject, stage: str) -> list[str]:
    """Why a stage is not complete now; an empty list means its own artifacts exist and are current."""
    if stage == "import":
        return _import_problems(project)
    if stage == "preparation":
        return _preparation_problems(project)
    if stage == "generation":
        return _generation_problems(project)
    if stage == "verification":
        return _verification_problems(project)
    if stage == "correction":
        return _correction_problems(project)
    if stage == "export":
        return _export_problems(project)
    raise ValidationError([f"unknown stage '{stage}'"])


def stage_artifacts(project: BookProject, stage: str) -> dict[str, str]:
    """sha256 of the files a stage's completion rests on, keyed by project-relative path."""
    paths: list[Path] = []
    if stage == "import":
        paths = [project.directory / SOURCE_FILE]
    elif stage == "preparation":
        paths = [project.directory / NARRATION_FILE]
    elif stage == "generation":
        paths = [project.directory / JOB_FILE, project.directory / STATE_FILE, project.directory / COVERAGE_FILE]
    elif stage == "verification":
        target = job_target(project) or ""
        paths = [findings_dir(project, chapter_id, target) / "report.md" for chapter_id in included_chapters(project)]
    elif stage == "correction":
        paths = [project.directory / name for name in (DICTIONARY_FILE, OVERRIDES_FILE, MARKUP_FILE, HISTORY_FILE, MANUAL_FILE)]
    elif stage == "export":
        paths = [record_path(project, EXPORT_NAME)]
    artifacts: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            artifacts[path.relative_to(project.directory).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return artifacts
