from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from rutts.book.chunking import TARGET_SECONDS
from rutts.book.portable import find_portability_problems
from rutts.book.job import COVERAGE_FILE, STATE_FILE, JobRequest, JobResult, run_job
from rutts.book.project import load_project, read_json
from rutts.book.runstate import format_elapsed
from rutts.engines import base
from rutts.validation import ValidationError


def add_job_parser(book_commands: Any) -> None:
    generate = book_commands.add_parser("generate", help="generate a whole book as a resumable job")
    generate.add_argument("directory", type=Path)
    generate.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    generate.add_argument("--precision", help="engine precision (default depends on the engine)")
    generate.add_argument("--target", choices=sorted(TARGET_SECONDS), help="chunk target (default short for a new job)")
    scope = generate.add_mutually_exclusive_group()
    scope.add_argument("--chapters", help="comma-separated chapter ids; rendered in source order")
    scope.add_argument("--from", dest="from_chapter", help="first chapter id to generate, through the end of the book")
    generate.add_argument("--retries", type=int, help="extra attempts per chunk (default 2 for a new job)")
    generate.add_argument("--retry-failed", action="store_true", help="retry chunks previously marked failed")
    generate.add_argument("--accept-config-change", action="store_true", help="supersede a different job configuration")
    generate.add_argument("--offline", action="store_true")
    generate.add_argument("--assemble", action="store_true", help="assemble each chapter once it is complete")
    generate.add_argument("--max-swap-gb", type=float, help="stop when swap used exceeds this (default 4)")
    generate.add_argument("--min-free-percent", type=float, help="stop when free memory is below this on two samples (default 5)")
    generate.add_argument("--max-rss-gb", type=float, help="stop when process peak RSS exceeds this (default 13)")
    generate.set_defaults(book_handler=_generate)

    job_status = book_commands.add_parser("job-status", help="show the generation job's state and estimate")
    job_status.add_argument("directory", type=Path)
    job_status.set_defaults(book_handler=_job_status)

    portable = book_commands.add_parser("check-portable", help="verify the project stores no absolute or outside paths")
    portable.add_argument("directory", type=Path)
    portable.set_defaults(book_handler=_check_portable)


def _engine(engine_id: str, precision: str | None) -> base.SynthesisEngine:
    options: dict[str, str] = {} if precision is None else {"precision": precision}
    try:
        return base.get_engine(engine_id, **options)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc


def _generate(args: argparse.Namespace) -> int:
    if args.retries is not None and args.retries < 0:
        return _errors(["--retries must be zero or positive"])
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    project = load_project(args.directory)
    engine = _engine(args.engine, args.precision)
    chapters = tuple(part.strip() for part in args.chapters.split(",") if part.strip()) if args.chapters else None
    request = JobRequest(
        target=args.target,
        chapters=chapters,
        from_chapter=args.from_chapter,
        retries=args.retries,
        retry_failed=args.retry_failed,
        accept_config_change=args.accept_config_change,
        assemble=args.assemble,
        max_swap_gb=args.max_swap_gb,
        min_free_percent=args.min_free_percent,
        max_rss_gb=args.max_rss_gb,
    )
    result = run_job(project, engine, request, progress=print)
    _summarize(result, args.directory)
    return result.exit_code


def _summarize(result: JobResult, directory: Path) -> None:
    state = result.state
    if result.status == "completed":
        print(f"book complete: every included narratable segment is covered ({state['chunks']['ok']} chunk(s))")
    elif result.status == "incomplete":
        print("book is not complete:")
        for chapter_id, row in state["chapters"].items():
            if row["status"] == "incomplete":
                failed = ", ".join(row["failed_chunks"]) or "none"
                detail = f"; {row['reason']}" if row["reason"] else ""
                print(f"  incomplete {chapter_id} {row['title']}: failed chunks {failed}{detail}")
        print("  run the same command with --retry-failed to retry failed chunks")
    elif result.status == "interrupted":
        print("interrupted: the job stopped; saved chunks are kept. Run the same command again to resume.", file=sys.stderr)
    elif result.status == "stopped_resource_pressure":
        reasons = "; ".join(state["stop"]["reasons"]) if state.get("stop") else "unknown"
        print(
            f"stopped for resource pressure: {reasons}. Diagnostics: {result.diagnostics}. "
            "Saved chunks are kept; free memory and run the same command again to resume.",
            file=sys.stderr,
        )
    excluded = [
        f"{chapter_id} ({row['reason'] or 'no reason recorded'})"
        for chapter_id, row in state["chapters"].items()
        if row["status"] == "excluded"
    ]
    if excluded:
        print("excluded chapters: " + "; ".join(excluded))
    print(f"job state: {directory / STATE_FILE}")


def _job_status(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    path = project.directory / STATE_FILE
    if not path.is_file():
        raise ValidationError([f"no generation job has been started in {args.directory}; run book generate first"])
    state = read_json(path)
    print(f"job {state['status']}: target {state['target']}, engine {state['engine']}, invocations {state['invocations']}")
    chunks = state["chunks"]
    print(
        f"chunks: {chunks['ok']} ok, {chunks['failed']} failed, {chunks['remaining']} remaining of {chunks['total']}"
    )
    print(f"current chapter {state['current_chapter'] or 'none'}, chunk {state['current_chunk'] or 'none'}")
    print(
        f"elapsed (last invocation) {format_elapsed(state['elapsed_seconds'])}; "
        f"cumulative {format_elapsed(state['cumulative_seconds'])}"
    )
    for chapter_id, row in state["chapters"].items():
        print(f"  {chapter_id} {row['status']}: {row['ok']}/{row['total']} ok, failed {row['failed']}, {row['title']}")
    estimate = state["estimate"]
    if estimate["available"]:
        print(f"estimate: about {format_elapsed(estimate['remaining_seconds'])} remaining ({estimate['basis']})")
    if state["recent_failures"]:
        print("recent failures:")
        for item in state["recent_failures"]:
            print(f"  {item['chunk']}: {item['error']}")
    coverage_path = project.directory / COVERAGE_FILE
    if coverage_path.is_file():
        coverage = read_json(coverage_path)
        totals = coverage["totals"]
        print(
            f"coverage: {'complete' if coverage['complete'] else 'INCOMPLETE'}; included {totals['included']}, "
            f"covered {totals['covered']}, uncovered {totals['uncovered']}, excluded {totals['excluded']}"
        )
    return 0


def _check_portable(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    problems = find_portability_problems(project.directory)
    if not problems:
        print(f"portable: {args.directory} stores no absolute or outside paths")
        return 0
    for problem in problems:
        print(f"not portable: {problem}")
    return 1


def _errors(messages: list[str]) -> int:
    for message in messages:
        print(f"error: {message}", file=sys.stderr)
    return 2

