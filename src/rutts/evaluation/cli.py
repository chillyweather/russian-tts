from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path
from typing import Any

from rutts.book.chunking import TARGET_SECONDS
from rutts.evaluation.blind import ATTEMPT_MODES, prepare
from rutts.evaluation.chapter_trial import record_notes, record_trial
from rutts.evaluation.decision import decide
from rutts.evaluation.report import write_report
from rutts.evaluation.scoring import afplay, import_scores, run_session
from rutts.validation import ValidationError


def add_eval_parser(commands: Any) -> None:
    evaluation = commands.add_parser("eval", help="blind listening evaluation and decision record")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)

    blind = eval_commands.add_parser("prepare", help="build anonymous, loudness-matched samples")
    blind.add_argument("--candidates", type=Path, nargs="+", required=True, help="preview directories")
    blind.add_argument("--out", type=Path, required=True)
    blind.add_argument("--seed", type=int, default=None, help="labelling seed (random when omitted)")
    blind.add_argument("--attempt", choices=ATTEMPT_MODES, default="first")
    blind.set_defaults(eval_handler=_run_prepare)

    score = eval_commands.add_parser("score", help="interactive blind scoring session (resumable)")
    score.add_argument("eval_dir", type=Path)
    score.add_argument("--no-play", action="store_true", help="do not play samples with afplay")
    score.set_defaults(eval_handler=_run_score)

    imported = eval_commands.add_parser("import-scores", help="import scores written down elsewhere")
    imported.add_argument("eval_dir", type=Path)
    imported.add_argument("--file", type=Path, required=True)
    imported.set_defaults(eval_handler=_run_import_scores)

    trial = eval_commands.add_parser("chapter-trial", help="record a rendered chapter's cost and reliability")
    trial.add_argument("book_dir", type=Path)
    trial.add_argument("--chapter", required=True)
    trial.add_argument("--target", choices=sorted(TARGET_SECONDS), required=True)
    trial.add_argument("--label", required=True)
    trial.add_argument("--out", type=Path, required=True)
    trial.set_defaults(eval_handler=_run_chapter_trial)

    notes = eval_commands.add_parser("chapter-notes", help="record manual chapter listening findings")
    notes.add_argument("eval_dir", type=Path)
    notes.add_argument("--label", required=True)
    notes.add_argument("--stress-errors", type=int, required=True)
    notes.add_argument("--omissions", type=int, required=True)
    notes.add_argument("--distracting-joins", type=int, required=True)
    notes.add_argument("--notes", default="", help="free-text listening findings")
    notes.set_defaults(eval_handler=_run_chapter_notes)

    report = eval_commands.add_parser("report", help="un-blind and summarise scores, failures and trials")
    report.add_argument("eval_dir", type=Path)
    report.set_defaults(eval_handler=_run_report)

    decision = eval_commands.add_parser("decide", help="apply the PRD gates and write the decision record")
    decision.add_argument("eval_dir", type=Path)
    decision.add_argument(
        "--approve-target-revision",
        default=None,
        metavar="REASON",
        help="accept a documented revision of the quality targets, recorded verbatim",
    )
    decision.set_defaults(eval_handler=_run_decide)


def _report(messages: list[str]) -> int:
    for message in messages:
        print(f"error: {message}", file=sys.stderr)
    return 2


def _run_prepare(args: argparse.Namespace) -> int:
    seed = secrets.randbelow(2**31) if args.seed is None else args.seed
    try:
        blind = prepare(args.candidates, args.out, seed=seed, attempt_mode=args.attempt)
    except (ValidationError, FileExistsError) as exc:
        return _report([str(exc)] if isinstance(exc, FileExistsError) else exc.messages)
    print(
        f"prepared {len(blind['excerpts'])} excerpt(s) with {len(args.candidates)} candidate(s), "
        f"seed {blind['seed']}; {len(blind['unavailable'])} unavailable; samples in {args.out}"
    )
    return 0


def _run_score(args: argparse.Namespace) -> int:
    try:
        run_session(
            args.eval_dir,
            stdin=sys.stdin,
            stdout=sys.stdout,
            play=None if args.no_play else afplay,
        )
    except ValidationError as exc:
        return _report(exc.messages)
    return 0


def _run_import_scores(args: argparse.Namespace) -> int:
    try:
        counts = import_scores(args.eval_dir, args.file)
    except ValidationError as exc:
        return _report(exc.messages)
    print(f"imported {counts['samples']} sample score set(s) and {counts['preferences']} preference(s)")
    return 0


def _run_chapter_trial(args: argparse.Namespace) -> int:
    try:
        trial = record_trial(args.book_dir, args.chapter, args.target, args.label, args.out)
    except (ValidationError, FileExistsError) as exc:
        return _report([str(exc)] if isinstance(exc, FileExistsError) else exc.messages)
    state = "completed" if trial["completed"] else "not completed"
    print(f"chapter trial '{args.label}' {state}: {trial['finished_audio_seconds']} s of finished audio")
    return 0


def _run_chapter_notes(args: argparse.Namespace) -> int:
    try:
        record_notes(
            args.eval_dir,
            args.label,
            stress_errors=args.stress_errors,
            omissions=args.omissions,
            distracting_joins=args.distracting_joins,
            notes=args.notes,
        )
    except (ValidationError, FileExistsError) as exc:
        return _report([str(exc)] if isinstance(exc, FileExistsError) else exc.messages)
    print(f"chapter notes for '{args.label}' recorded")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    try:
        report = write_report(args.eval_dir)
    except ValidationError as exc:
        return _report(exc.messages)
    print(f"report {report['status']}: {report['samples_scored']} of {report['samples_total']} sample(s) scored")
    print(f"written to {args.eval_dir / 'report.md'} and {args.eval_dir / 'report.json'}")
    return 0


def _run_decide(args: argparse.Namespace) -> int:
    revision = args.approve_target_revision
    if revision is not None and not revision.strip():
        return _report(["--approve-target-revision needs a non-empty reason"])
    try:
        document = decide(args.eval_dir, target_revision=revision)
    except ValidationError as exc:
        return _report(exc.messages)
    if document["status"] != "decided":
        print("no decision:", file=sys.stderr)
        for gap in document["gaps"]:
            print(f"  - {gap}", file=sys.stderr)
        print(f"  {document['next_step']}", file=sys.stderr)
        return 1
    chosen = document["chosen"]
    print(
        f"decided: {chosen['model_repo']} @ {chosen['model_revision']} ({chosen['precision']}), "
        f"candidate {chosen['candidate']}; record in {args.eval_dir / 'decision.md'}"
    )
    return 0


def run_eval(args: argparse.Namespace) -> int:
    handler: Any = args.eval_handler
    return int(handler(args))
