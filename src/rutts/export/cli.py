from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rutts.book.assemble import assemble_chapter, master_is_stale
from rutts.book.chunking import TARGET_SECONDS
from rutts.book.project import CHAPTERS_DIR, load_project, read_json
from rutts.book.render import resolve_target
from rutts.export.build import DEFAULT_NAME, ExportSettings, build_export, included_ids, record_path, settings_document
from rutts.export.status import export_status, recheck_export
from rutts.validation import ValidationError

NAME_PATTERN = re.compile(r"[\w-]+")


def _run_command(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


RUNNER: Callable[[list[str]], int] = _run_command


def add_export_parser(book_commands: Any) -> None:
    export = book_commands.add_parser("export", help="build a chaptered M4B (and optional MP3) audiobook")
    export.add_argument("directory", type=Path)
    export.add_argument("--target", choices=sorted(TARGET_SECONDS))
    export.add_argument("--draft", action="store_true", help="export an explicitly labelled draft of unfinished review")
    export.add_argument("--format", dest="formats", action="append", choices=("m4b", "mp3"), help="output format (default m4b)")
    export.add_argument("--bitrate", default="96k")
    export.add_argument("--name", default=DEFAULT_NAME)
    export.add_argument("--no-loudness", action="store_true", help="skip the book-wide loudness gain")
    export.add_argument("--force", action="store_true", help="rebuild even when the export is current")
    export.add_argument("--assemble", action="store_true", help="assemble chapters whose master is missing or stale first")
    export.set_defaults(book_handler=_export)

    status = book_commands.add_parser("export-status", help="report whether an export is ready, stale or missing")
    status.add_argument("directory", type=Path)
    status.add_argument("--name", default=DEFAULT_NAME)
    status.add_argument("--target", choices=sorted(TARGET_SECONDS))
    status.add_argument("--bitrate")
    status.add_argument("--draft", action="store_true")
    status.set_defaults(book_handler=_export_status)

    check = book_commands.add_parser("export-check", help="re-run validation on an existing export and print its chapters")
    check.add_argument("directory", type=Path)
    check.add_argument("--name", default=DEFAULT_NAME)
    check.set_defaults(book_handler=_export_check)

    open_cmd = book_commands.add_parser("export-open", help="open the export's M4B with the default player")
    open_cmd.add_argument("directory", type=Path)
    open_cmd.add_argument("--name", default=DEFAULT_NAME)
    open_cmd.set_defaults(book_handler=_export_open)


def _name(value: str) -> str:
    if not NAME_PATTERN.fullmatch(value):
        raise ValidationError([f"export name may use letters, digits, '_' and '-' only: {value!r}"])
    return value


def _resolved_target(project: Any, requested: str | None) -> str:
    ids = included_ids(project)
    if not ids:
        raise ValidationError(["no included chapters to export"])
    return resolve_target(project, ids[0], requested)


def _export(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    name = _name(args.name)
    if args.assemble:
        for chapter_id in included_ids(project):
            report_path = project.directory / CHAPTERS_DIR / f"{chapter_id}.json"
            if not report_path.is_file() or master_is_stale(project, chapter_id):
                report = assemble_chapter(project, chapter_id, args.target)
                print(f"assembled {chapter_id}: {'complete' if report['complete'] else 'INCOMPLETE'}")
    kind = "draft" if args.draft else "reviewed"
    formats = tuple(dict.fromkeys(args.formats or ["m4b"]))
    target = _resolved_target(project, args.target)
    settings = ExportSettings(name=name, bitrate=args.bitrate, loudness=not args.no_loudness)
    current = export_status(project, name, settings_document(kind, target, formats, settings))
    if current.state == "ready" and not args.force:
        print(f"export {name} is current ({kind}); nothing to rebuild (use --force to rebuild)")
        return 0
    for change in current.changes:
        print(f"rebuilding: {change}")
    outcome = build_export(project, target, kind, formats, settings)
    if outcome.status == "refused":
        for message in outcome.messages:
            print(f"error: {message}", file=sys.stderr)
        print("export refused: nothing was written", file=sys.stderr)
        return 1
    if outcome.status == "invalid":
        for message in outcome.messages:
            print(f"validation failed: {message}", file=sys.stderr)
        print(f"export {name} is INVALID; see exports/{name}/export.json", file=sys.stderr)
        return 1
    for gap in outcome.messages:
        print(f"gap: {gap}")
    for entry in outcome.record["files"]:
        print(f"wrote {entry['path']} ({entry['size']} bytes, sha256 {entry['sha256'][:12]})")
    print(f"export {name} ready ({kind}); record exports/{name}/export.json")
    return 0


def _export_status(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    name = _name(args.name)
    requested = None
    if args.bitrate is not None or args.target is not None or args.draft:
        target = _resolved_target(project, args.target)
        settings = ExportSettings(name=name, bitrate=args.bitrate or "96k")
        requested = settings_document("draft" if args.draft else "reviewed", target, ("m4b",), settings)
    status = export_status(project, name, requested)
    print(f"export {name}: {status.state}")
    for change in status.changes:
        print(f"  {change}")
    if status.state == "stale":
        print(f"  rebuild with: rutts book export {args.directory}")
    if status.state == "missing":
        print(f"  no export yet; run: rutts book export {args.directory}")
    return 0


def _export_check(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    path = record_path(project, _name(args.name))
    if not path.is_file():
        raise ValidationError([f"no export named {args.name}; run book export first"])
    record = read_json(path)
    if record["status"] != "ready":
        for item in record["validation"]["checks"]:
            if not item["ok"]:
                print(f"FAIL {item['name']}: {item['detail']}")
        return 1
    print(f"{'#':>3}  {'start':>10}  {'end':>10}  title")
    for number, item in enumerate(record["chapters"], start=1):
        print(f"{number:>3}  {item['start_ms'] / 1000:>10.3f}  {item['end_ms'] / 1000:>10.3f}  {item['title']}")
    checks = recheck_export(project, record)
    for item in checks:
        print(f"{'ok  ' if item['ok'] else 'FAIL'} {item['name']}{': ' + item['detail'] if item['detail'] and not item['ok'] else ''}")
    return 0 if all(item["ok"] for item in checks) else 1


def _export_open(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    name = _name(args.name)
    status = export_status(project, name)
    if status.state != "ready" or status.record is None:
        raise ValidationError([f"export {name} is {status.state}; only a ready export can be opened"])
    if status.changes:
        print("warning: the export is stale: " + "; ".join(status.changes), file=sys.stderr)
    m4b = next(entry for entry in status.record["files"] if entry["format"] == "m4b")
    return RUNNER(["open", str(project.directory / m4b["path"])])
