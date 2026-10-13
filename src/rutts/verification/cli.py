from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from rutts.book.chunking import TARGET_SECONDS
from rutts.book.project import load_project
from rutts.book.render import resolve_target
from rutts.validation import ValidationError
from rutts.verification import asr
from rutts.verification.verify import load_findings, verify_chapter


def add_verification_parser(book_commands: Any) -> None:
    verify = book_commands.add_parser("verify", help="check rendered chunks with a local ASR (no synthesis model)")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--chapter", required=True)
    verify.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    verify.add_argument("--transcriber", choices=sorted(asr.TRANSCRIBER_FACTORIES), default="whisper-mlx")
    verify.add_argument("--offline", action="store_true")
    verify.add_argument("--recheck", action="store_true", help="re-verify every chunk even if findings are current")
    verify.set_defaults(book_handler=_verify)

    findings = book_commands.add_parser("findings", help="print stored verification findings")
    findings.add_argument("directory", type=Path)
    findings.add_argument("--chapter", required=True)
    findings.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    findings.add_argument("--chunk", default=None)
    findings.set_defaults(book_handler=_findings)


def _verify(args: Namespace) -> int:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        project = load_project(args.directory)
        target = resolve_target(project, args.chapter, args.target)
        transcriber = asr.get_transcriber(args.transcriber)
        result = verify_chapter(project, args.chapter, target, transcriber, recheck=args.recheck)
    except ValidationError as exc:
        for message in exc.messages:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(
        f"{result.chapter_id} target={result.target}: checked {result.checked} chunk(s), "
        f"reused {result.reused} current result(s)"
    )
    print(f"report: {result.report}")
    print("Automatic findings are review aids. Stress, е/ё, prosody and comfort need manual listening.")
    return 0


def _findings(args: Namespace) -> int:
    try:
        project = load_project(args.directory)
        target = resolve_target(project, args.chapter, args.target)
        documents = load_findings(project, args.chapter, target, args.chunk)
    except ValidationError as exc:
        for message in exc.messages:
            print(f"error: {message}", file=sys.stderr)
        return 1
    for document in documents:
        print(f"{document['chunk_id']}: {document['status']}")
        for finding in document["findings"]:
            audio = finding["location"].get("audio_seconds", [0.0, 0.0])
            print(
                f"  [{finding['severity']}] {finding['type']} at {audio[0]:.2f}-{audio[1]:.2f} s: "
                f"expected {finding.get('expected', '')!r}, heard {finding.get('heard', '')!r}"
            )
    return 0
