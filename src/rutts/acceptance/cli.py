from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rutts.acceptance import listening as listening_store
from rutts.acceptance import record as record_store
from rutts.acceptance.evaluate import refresh
from rutts.acceptance.network import network_state
from rutts.acceptance.report import write_report
from rutts.acceptance.stages import EXPORT_NAME, STAGES, stage_artifacts, stage_problems
from rutts.book.project import load_project, read_json
from rutts.book.render import resolve_target
from rutts.export.build import record_path
from rutts.validation import ValidationError

NETWORK_STATE: Callable[[], str] = network_state
OFFLINE = "offline"


def add_acceptance_parser(book_commands: Any) -> None:
    accept = book_commands.add_parser("accept", help="acceptance record for the offline audiobook workflow")
    commands = accept.add_subparsers(dest="accept_command", required=True)

    step = commands.add_parser("step", help="record that a workflow stage is complete, with the network state")
    step.add_argument("directory", type=Path)
    step.add_argument("stage", choices=STAGES)
    step.set_defaults(book_handler=_step)

    listen = commands.add_parser("listen", help="record your own listening scores for the acceptance chapter")
    listen.add_argument("directory", type=Path)
    listen.add_argument("--chapter", required=True)
    listen.add_argument("--scores-file", type=Path, default=None, help="scores you wrote yourself, instead of prompts")
    listen.set_defaults(book_handler=_listen)

    playback = commands.add_parser("playback", help="record your playback confirmation for the exported M4B")
    playback.add_argument("directory", type=Path)
    playback.add_argument("--player", required=True)
    playback.set_defaults(book_handler=_playback)

    status = commands.add_parser("status", help="show each acceptance section and what is missing")
    status.add_argument("directory", type=Path)
    status.set_defaults(book_handler=_status)

    report = commands.add_parser("report", help="write acceptance/acceptance.md")
    report.add_argument("directory", type=Path)
    report.set_defaults(book_handler=_report)


def _print_sections(document: dict[str, Any]) -> None:
    for name, section in document["sections"].items():
        print(f"{name}: {section['status']}")
        for item in section["missing"]:
            print(f"  - {item}")
    print(f"overall: {document['overall']}")


def _step(args: Any) -> int:
    project = load_project(args.directory)
    problems = stage_problems(project, args.stage)
    if problems:
        raise ValidationError([f"stage {args.stage} is not complete: {problem}" for problem in problems])
    network = NETWORK_STATE()
    inputs = record_store.load_inputs(project)
    inputs["offline"].append(
        {
            "stage": args.stage,
            "recorded_at": record_store.now(),
            "network": network,
            "artifacts": stage_artifacts(project, args.stage),
        }
    )
    refresh(project, inputs)
    counted = "counts as offline evidence" if network == OFFLINE else "does not count as offline evidence"
    print(f"recorded {args.stage} complete with network {network}; {counted}")
    return 0


def _listen(args: Any) -> int:
    project = load_project(args.directory)
    chapter_id = args.chapter
    chapter = project.read_source().chapter(chapter_id)
    if not chapter.included:
        raise ValidationError([f"chapter {chapter_id} is excluded from the book"])
    inputs = record_store.load_inputs(project)
    target = resolve_target(project, chapter_id, None)
    if args.scores_file is not None:
        answers = listening_store.load_scores_file(args.scores_file)
        copied = listening_store.copy_scores_file(project, chapter_id, args.scores_file)
        source = "scores_file"
    else:
        answers = listening_store.ask_scores(sys.stdin, sys.stdout)
        copied = None
        source = "interactive"
    signature = listening_store.chapter_signature(project, chapter_id, target)
    inputs["listening"] = {
        "chapter_id": chapter_id,
        "target": target,
        "recorded_at": record_store.now(),
        "source": source,
        "copied_scores": copied,
        "chapter_signature": signature,
        **answers,
    }
    document = refresh(project, inputs)
    print(f"recorded listening review for {chapter_id} ({target})")
    _print_sections({"sections": {"listening": document["sections"]["listening"]}, "overall": document["overall"]})
    return 0


def _playback(args: Any) -> int:
    project = load_project(args.directory)
    problems = stage_problems(project, "export")
    if problems:
        raise ValidationError(problems)
    export = read_json(record_path(project, EXPORT_NAME))
    m4b = next((entry for entry in export["files"] if entry["format"] == "m4b"), None)
    if m4b is None:
        raise ValidationError([f"export {EXPORT_NAME} has no M4B to confirm; export with the m4b format"])
    answers = listening_store.ask_playback(sys.stdin, sys.stdout, args.player)
    inputs = record_store.load_inputs(project)
    inputs["playback"] = {**answers, "m4b_sha256": m4b["sha256"], "recorded_at": record_store.now()}
    document = refresh(project, inputs)
    _print_sections({"sections": {"output": document["sections"]["output"]}, "overall": document["overall"]})
    return 0


def _status(args: Any) -> int:
    project = load_project(args.directory)
    document = refresh(project, record_store.load_inputs(project))
    _print_sections(document)
    return 0 if document["overall"] == record_store.ACCEPTED else 1


def _report(args: Any) -> int:
    project = load_project(args.directory)
    inputs = record_store.load_inputs(project)
    document = refresh(project, inputs)
    path = write_report(project, document, inputs)
    print(f"acceptance report written to {path}; overall: {document['overall']}")
    return 0
