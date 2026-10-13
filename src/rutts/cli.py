from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from collections import Counter
from typing import Any

from rutts.book.assemble import assemble_chapter, master_is_stale
from rutts.book.boundaries import (
    merge_chapter,
    rename_chapter,
    set_chapter_included,
    set_segment_included,
    split_chapter,
)
from rutts.book.chunking import TARGET_SECONDS, limits_for_target, plan_chunks
from rutts.book.project import install_narrator, load_project, read_json
from rutts.book.render import MANIFEST_FILE, ConfigChangeRequired, render_chapter, summarize_targets
from rutts.book.runstate import read_run_state, status_lines
from rutts.book.text_import import import_text
from rutts.engines import base
from rutts.evaluation.cli import add_eval_parser, run_eval
from rutts.pronunciation.cli import add_pronunciation_parser
from rutts.narrator import create_profile, load_profile
from rutts.preview import Excerpt, load_excerpts, load_suite, run_preview
from rutts.smoke import run_smoke
from rutts.verification.cli import add_verification_parser
from rutts.validation import ValidationError, read_text_file, text_problems

PREVIEW_CHARS = 70


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rutts")
    commands = parser.add_subparsers(dest="command", required=True)

    narrator = commands.add_parser("narrator", help="manage narrator profiles")
    narrator_commands = narrator.add_subparsers(dest="narrator_command", required=True)
    create = narrator_commands.add_parser("create", help="create a narrator profile")
    create.add_argument("--name", required=True)
    create.add_argument("--audio", type=Path, required=True)
    create.add_argument("--transcript-file", type=Path, required=True)
    create.add_argument("--provenance", required=True)
    create.add_argument("--out", type=Path, required=True)

    preview = commands.add_parser("preview", help="render narration previews")
    preview.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    preview.add_argument("--precision", help="engine precision (default depends on the engine)")
    preview.add_argument("--narrator", type=Path, required=True)
    preview.add_argument("--out", type=Path, required=True)
    sources = preview.add_mutually_exclusive_group(required=True)
    sources.add_argument("--text")
    sources.add_argument("--text-file", type=Path)
    sources.add_argument("--excerpts", type=Path)
    sources.add_argument("--suite", action="store_true")
    preview.add_argument("--retries", type=int, default=1)
    preview.add_argument("--offline", action="store_true")

    smoke = commands.add_parser("smoke", help="one-sentence memory smoke test for one engine")
    smoke.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    smoke.add_argument("--precision", help="engine precision (default depends on the engine)")
    smoke.add_argument("--narrator", type=Path, required=True)
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--offline", action="store_true")

    book = commands.add_parser("book", help="book projects: import, correct, render and assemble")
    book_commands = book.add_subparsers(dest="book_command", required=True)

    book_import = book_commands.add_parser("import", help="create a book project from UTF-8 text")
    book_import.add_argument("source", type=Path)
    book_import.add_argument("--out", type=Path, required=True)
    book_import.add_argument("--title")
    book_import.add_argument("--author", default="")
    book_import.add_argument("--chapter-regex", help="extra chapter marker pattern (re.match)")

    show = book_commands.add_parser("show", help="list chapters, exclusions or one chapter's segments")
    show.add_argument("directory", type=Path)
    show.add_argument("--chapter")

    split = book_commands.add_parser("split", help="start a new chapter at a segment")
    split.add_argument("directory", type=Path)
    split.add_argument("--at", required=True, help="segment id that starts the new chapter")
    split.add_argument("--title", required=True)

    merge = book_commands.add_parser("merge", help="merge a chapter into the previous one")
    merge.add_argument("directory", type=Path)
    merge.add_argument("--chapter", required=True)

    rename = book_commands.add_parser("rename", help="rename a chapter")
    rename.add_argument("directory", type=Path)
    rename.add_argument("--chapter", required=True)
    rename.add_argument("--title", required=True)

    for name, help_text in (("exclude", "exclude a chapter or segment"), ("include", "include a chapter or segment")):
        command = book_commands.add_parser(name, help=help_text)
        command.add_argument("directory", type=Path)
        scope = command.add_mutually_exclusive_group(required=True)
        scope.add_argument("--chapter")
        scope.add_argument("--segment")
        if name == "exclude":
            command.add_argument("--reason", help="why the segment is excluded")

    narrator_copy = book_commands.add_parser("narrator", help="copy a narrator profile into the project")
    narrator_copy.add_argument("directory", type=Path)
    narrator_copy.add_argument("--profile", type=Path, required=True)

    plan = book_commands.add_parser("plan", help="show chunk count and estimated duration range")
    plan.add_argument("directory", type=Path)
    plan.add_argument("--chapter", required=True)
    plan.add_argument("--target", choices=sorted(TARGET_SECONDS), required=True)
    plan.add_argument("--max-chars", type=int, default=None)

    render = book_commands.add_parser("render", help="render a chapter's chunks with one engine")
    render.add_argument("directory", type=Path)
    render.add_argument("--chapter", required=True)
    render.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    render.add_argument("--precision", help="engine precision (default depends on the engine)")
    render.add_argument("--target", choices=sorted(TARGET_SECONDS), default="short")
    render.add_argument("--max-chars", type=int, default=None)
    render.add_argument("--offline", action="store_true")
    render.add_argument(
        "--force-new-run",
        action="store_true",
        help="archive the whole chapter's chunks and start a new run",
    )
    render.add_argument(
        "--accept-config-change",
        action="store_true",
        help="regenerate every chunk rendered with a different engine or narrator",
    )
    render.add_argument("--retries", type=int, default=2, help="extra attempts per chunk (default 2)")
    render.add_argument("--retry-failed", action="store_true", help="retry chunks previously marked failed")

    status = book_commands.add_parser("status", help="show render progress from the project files")
    status.add_argument("directory", type=Path)
    status.add_argument("--chapter")
    status.add_argument("--target", choices=sorted(TARGET_SECONDS))

    assemble = book_commands.add_parser("assemble", help="assemble a chapter master from its chunks")
    assemble.add_argument("directory", type=Path)
    assemble.add_argument("--chapter", required=True)
    assemble.add_argument("--target", choices=sorted(TARGET_SECONDS))

    compare = book_commands.add_parser("compare-targets", help="compare rendered short and long targets")
    compare.add_argument("directory", type=Path)
    compare.add_argument("--chapter", required=True)
    add_pronunciation_parser(book_commands)
    add_verification_parser(book_commands)
    add_eval_parser(commands)
    return parser


def _report(messages: list[str]) -> int:
    for message in messages:
        print(f"error: {message}", file=sys.stderr)
    return 2


def _run_narrator_create(args: argparse.Namespace) -> int:
    try:
        transcript = read_text_file(args.transcript_file)
        profile = create_profile(
            args.out,
            name=args.name,
            audio=args.audio,
            transcript=transcript,
            provenance=args.provenance,
        )
    except ValidationError as exc:
        return _report(exc.messages)
    print(f"narrator profile written to {profile.directory}")
    return 0


def _excerpts_for(args: argparse.Namespace) -> list[Excerpt]:
    if args.suite:
        return load_suite()
    if args.excerpts is not None:
        return load_excerpts(args.excerpts)
    text = args.text if args.text is not None else read_text_file(args.text_file)
    problems = text_problems(text)
    if problems:
        raise ValidationError(problems)
    return [Excerpt(id="text", category="custom", text=text)]


def _build_engine(engine_id: str, precision: str | None) -> base.SynthesisEngine:
    options: dict[str, str] = {} if precision is None else {"precision": precision}
    try:
        return base.get_engine(engine_id, **options)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc


def _run_preview(args: argparse.Namespace) -> int:
    if args.retries < 0:
        return _report(["--retries must be zero or positive"])
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        narrator = load_profile(args.narrator)
        excerpts = _excerpts_for(args)
        engine = _build_engine(args.engine, args.precision)
    except ValidationError as exc:
        return _report(exc.messages)

    try:
        document = run_preview(engine, narrator, excerpts, args.out, args.retries)
    except FileExistsError as exc:
        return _report([str(exc)])
    summary = document["summary"]
    print(
        f"rendered {summary['excerpts']} excerpt(s): {summary['ok']} ok, "
        f"{summary['failed']} failed; results in {args.out}"
    )
    return 0 if summary["failed"] == 0 else 1


def _run_smoke(args: argparse.Namespace) -> int:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        narrator = load_profile(args.narrator)
        engine = _build_engine(args.engine, args.precision)
    except ValidationError as exc:
        return _report(exc.messages)
    try:
        record = run_smoke(engine, narrator, args.out)
    except FileExistsError as exc:
        return _report([str(exc)])
    print(f"smoke {record['status']} for {args.engine}; record in {args.out / 'smoke.json'}")
    return 0 if record["feasible"] else 1


def _preview(text: str) -> str:
    flat = text.replace("\n", " / ")
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "..."


def _book_import(args: argparse.Namespace) -> int:
    project = import_text(
        args.source,
        args.out,
        title=args.title,
        author=args.author,
        chapter_regex=args.chapter_regex,
    )
    chapters = project.read_source().chapters
    print(f"book project written to {args.out}: {len(chapters)} chapter(s)")
    return 0


def _book_show(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    source = project.read_source()
    if args.chapter is not None:
        chapter = source.chapter(args.chapter)
        state = "included" if chapter.included else "excluded"
        print(f"{chapter.id} {chapter.title!r} [{state}]")
        for segment in chapter.segments:
            reason = "" if segment.included else f" [excluded: {segment.exclusion_reason}]"
            print(f"  {segment.id} {segment.kind}{reason}: {_preview(segment.text)}")
        return 0
    title = project.title if not project.author else f"{project.title} ({project.author})"
    print(title)
    for chapter in source.chapters:
        kinds = Counter(segment.kind for segment in chapter.segments)
        excluded = sum(1 for segment in chapter.segments if not segment.included)
        state = "included" if chapter.included else "excluded"
        print(
            f"{chapter.id} {chapter.title!r} [{state}] paragraphs={kinds['paragraph']} "
            f"scene_breaks={kinds['scene_break']} headings={kinds['heading']} "
            f"excluded_segments={excluded}"
        )
    exclusions = [
        (segment.id, segment.exclusion_reason)
        for chapter in source.chapters
        for segment in chapter.segments
        if not segment.included
    ]
    if exclusions:
        print("exclusions:")
        for excluded_id, excluded_reason in exclusions:
            print(f"  {excluded_id}: {excluded_reason}")
    return 0


def _book_boundary(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    if args.book_command == "split":
        split_chapter(project, args.at, args.title)
        print(f"chapter split at {args.at}")
    elif args.book_command == "merge":
        merge_chapter(project, args.chapter)
        print(f"{args.chapter} merged into the previous chapter")
    elif args.book_command == "rename":
        rename_chapter(project, args.chapter, args.title)
        print(f"{args.chapter} renamed to {args.title!r}")
    else:
        included = args.book_command == "include"
        if args.chapter is not None:
            set_chapter_included(project, args.chapter, included)
            print(f"{args.chapter} {'included' if included else 'excluded'}")
        else:
            set_segment_included(project, args.segment, included, reason=getattr(args, "reason", None))
            print(f"{args.segment} {'included' if included else 'excluded'}")
    return 0


def _book_narrator(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    install_narrator(project, args.profile)
    print(f"narrator copied into {project.narrator_dir}")
    return 0


def _book_plan(args: argparse.Namespace) -> int:
    if args.max_chars is not None and args.max_chars <= 0:
        return _report(["--max-chars must be positive"])
    project = load_project(args.directory)
    limits = limits_for_target(args.target, args.max_chars)
    chunks = plan_chunks(project.planning_segments(args.chapter), args.target, limits)
    if not chunks:
        return _report([f"chapter {args.chapter} has no text to narrate"])
    estimates = [chunk.estimated_seconds for chunk in chunks]
    print(f"{args.chapter} target={args.target}: {len(chunks)} chunk(s)")
    print(
        f"estimated duration per chunk {min(estimates):.1f}-{max(estimates):.1f} s; "
        f"about {sum(estimates):.0f} s in total (estimate at {limits.chars_per_second:g} chars/s, "
        "not measured)"
    )
    return 0


def _book_render(args: argparse.Namespace) -> int:
    if args.max_chars is not None and args.max_chars <= 0:
        return _report(["--max-chars must be positive"])
    if args.retries < 0:
        return _report(["--retries must be zero or positive"])
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    project = load_project(args.directory)
    engine = _build_engine(args.engine, args.precision)
    try:
        manifest = render_chapter(
            project,
            args.chapter,
            engine,
            args.target,
            limits=limits_for_target(args.target, args.max_chars),
            force_new_run=args.force_new_run,
            accept_config_change=args.accept_config_change,
            retries=args.retries,
            retry_failed=args.retry_failed,
            progress=print,
        )
    except ConfigChangeRequired as exc:
        return _report(exc.messages)
    except KeyboardInterrupt:
        print(
            f"interrupted: rendering {args.chapter} ({args.target}) stopped; saved chunks are kept. "
            "Run the same render command again to resume.",
            file=sys.stderr,
        )
        return 130
    except ValidationError:
        raise
    except Exception as exc:
        print(
            f"error: render stopped: {type(exc).__name__}: {exc}; saved chunks are kept, "
            "run the same render command again to resume",
            file=sys.stderr,
        )
        return 1
    records: list[dict[str, Any]] = manifest["chunks"]
    failed = [record["id"] for record in records if record["status"] == "failed"]
    ok = sum(1 for record in records if record["status"] == "ok")
    print(
        f"rendered {args.chapter} ({args.target}): {len(records)} chunk(s), {ok} ok, "
        f"{len(failed)} failed; chunks in {project.target_dir(args.chapter, args.target)}"
    )
    hints = manifest["pronunciation"]["hints_not_applied"]
    if hints:
        print(
            f"warning: stress hints were not applied for {manifest['engine']['engine_id']} "
            f"(no verified convention; hints stripped) in: {', '.join(hints)}"
        )
    for chunk_id in failed:
        print(f"failed chunk (needs review): {chunk_id}")
    return 0 if not failed else 1


def _book_status(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    chapters = [args.chapter] if args.chapter else [chapter.id for chapter in project.read_source().chapters]
    if args.chapter:
        project.read_source().chapter(args.chapter)
    for chapter_id in chapters:
        targets = [args.target] if args.target else sorted(TARGET_SECONDS)
        rendered = [target for target in targets if (project.target_dir(chapter_id, target) / MANIFEST_FILE).is_file()]
        if not rendered:
            print(f"{chapter_id}: not rendered")
        if master_is_stale(project, chapter_id):
            print(f"{chapter_id} chapter master: STALE (narration or chunk audio changed); run book assemble")
        for target in rendered:
            directory = project.target_dir(chapter_id, target)
            records: list[dict[str, Any]] = read_json(directory / MANIFEST_FILE)["chunks"]
            for line in status_lines(chapter_id, target, records, read_run_state(directory)):
                print(line)
    return 0


def _book_assemble(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    was_stale = master_is_stale(project, args.chapter)
    report = assemble_chapter(project, args.chapter, args.target)
    if was_stale:
        print(f"previous {args.chapter} master was stale and has been rebuilt")
    for chunk_id in report["missing_chunks"]:
        print(f"missing chunk: {chunk_id}")
    for chunk_id in report["failed_chunks"]:
        print(f"failed chunk: {chunk_id}")
    for segment_id in report["uncovered_segments"]:
        print(f"uncovered segment: {segment_id}")
    state = "complete" if report["complete"] else "INCOMPLETE"
    print(f"{args.chapter} master {report['master_file']} ({state}); report chapters/{args.chapter}.json")
    return 0 if report["complete"] else 1


def _book_compare(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    for row in summarize_targets(project, args.chapter):
        if not row["rendered"]:
            print(f"{row['target']}: not rendered")
            continue
        line = (
            f"{row['target']}: {row['chunks']} chunk(s), {row['ok']} ok, "
            f"{row['failed']} failed, {row['missing']} missing"
        )
        measured = row["measured_seconds"]
        if measured is None:
            line += "; no measured audio"
        else:
            line += (
                f"; measured min {measured['min']:.2f} s, mean {measured['mean']:.2f} s, "
                f"max {measured['max']:.2f} s"
            )
        print(line)
    return 0


BOOK_COMMANDS = {
    "import": _book_import,
    "show": _book_show,
    "split": _book_boundary,
    "merge": _book_boundary,
    "rename": _book_boundary,
    "exclude": _book_boundary,
    "include": _book_boundary,
    "narrator": _book_narrator,
    "plan": _book_plan,
    "render": _book_render,
    "status": _book_status,
    "assemble": _book_assemble,
    "compare-targets": _book_compare,
}


def _run_book(args: argparse.Namespace) -> int:
    handler = getattr(args, "book_handler", None) or BOOK_COMMANDS[args.book_command]
    try:
        return handler(args)
    except ValidationError as exc:
        return _report(exc.messages)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "narrator":
        return _run_narrator_create(args)
    if args.command == "smoke":
        return _run_smoke(args)
    if args.command == "book":
        return _run_book(args)
    if args.command == "eval":
        return run_eval(args)
    return _run_preview(args)


if __name__ == "__main__":
    raise SystemExit(main())
