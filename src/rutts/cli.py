from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rutts.engines import base
from rutts.narrator import create_profile, load_profile
from rutts.preview import Excerpt, load_excerpts, load_suite, run_preview
from rutts.smoke import run_smoke
from rutts.validation import ValidationError, read_text_file, text_problems


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "narrator":
        return _run_narrator_create(args)
    if args.command == "smoke":
        return _run_smoke(args)
    return _run_preview(args)


if __name__ == "__main__":
    raise SystemExit(main())
