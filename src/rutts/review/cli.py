from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from rutts.book.chunking import TARGET_SECONDS
from rutts.book.project import BookProject, load_project, read_json
from rutts.book.render import MANIFEST_FILE, PLAN_FILE, resolve_target
from rutts.review import decisions as decision_store
from rutts.review import manual as manual_store
from rutts.review import playback
from rutts.review.eligibility import export_eligibility
from rutts.review.queue import QueueItem, build_queue, find_item, scopes
from rutts.validation import ValidationError
from rutts.verification import asr
from rutts.verification.verify import verify_chapter

DEFAULT_TRANSCRIBER = "whisper-mlx"


def add_review_parser(book_commands: Any) -> None:
    review = book_commands.add_parser("review", help="review flagged passages, corrections and export eligibility")
    commands = review.add_subparsers(dest="review_command", required=True)

    listed = commands.add_parser("list", help="list review items (open by default)")
    listed.add_argument("directory", type=Path)
    listed.add_argument("--chapter", default=None)
    listed.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    listed.add_argument("--all", action="store_true", help="include resolved, dismissed and stale items")
    listed.add_argument("--type", default=None)
    listed.add_argument("--json", action="store_true")
    listed.set_defaults(book_handler=_list)

    show = commands.add_parser("show", help="show one item with its follow-up commands")
    show.add_argument("directory", type=Path)
    show.add_argument("item_id")
    show.set_defaults(book_handler=_show)

    play = commands.add_parser("play", help="write a clip of the item's passage and play or open it")
    play.add_argument("directory", type=Path)
    play.add_argument("item_id")
    play.add_argument("--join", action="store_true", help="clip the join with the neighbouring chunks")
    play.add_argument("--open", action="store_true", help="open the clip instead of playing it")
    play.add_argument("--no-play", action="store_true", help="only write the clip and print its path")
    play.set_defaults(book_handler=_play)

    flag = commands.add_parser("flag", help="record a manual finding ASR cannot detect")
    flag.add_argument("directory", type=Path)
    flag.add_argument("--chunk", required=True)
    flag.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    flag.add_argument("--kind", choices=manual_store.KINDS, required=True)
    flag.add_argument("--note", required=True)
    location = flag.add_mutually_exclusive_group()
    location.add_argument("--word", default=None)
    location.add_argument("--time", type=float, default=None, help="seconds into the chunk audio")
    flag.set_defaults(book_handler=_flag)

    for name, help_text in (("resolve", "mark an item resolved"), ("dismiss", "dismiss an item as not a problem")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("directory", type=Path)
        command.add_argument("item_id")
        command.add_argument("--reason", required=True)
        command.set_defaults(book_handler=_resolve if name == "resolve" else _dismiss)

    reopen = commands.add_parser("reopen", help="reopen a resolved or dismissed item")
    reopen.add_argument("directory", type=Path)
    reopen.add_argument("item_id")
    reopen.set_defaults(book_handler=_reopen)

    recheck = commands.add_parser("recheck", help="re-verify chunks whose audio changed and report item changes")
    recheck.add_argument("directory", type=Path)
    recheck.add_argument("--chapter", required=True)
    recheck.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    recheck.add_argument("--transcriber", choices=sorted(asr.TRANSCRIBER_FACTORIES) or None, default=DEFAULT_TRANSCRIBER)
    recheck.set_defaults(book_handler=_recheck)

    status = commands.add_parser("status", help="counts and export eligibility per chapter")
    status.add_argument("directory", type=Path)
    status.add_argument("--target", choices=sorted(TARGET_SECONDS), default=None)
    status.set_defaults(book_handler=_status)


def _line(item: QueueItem) -> str:
    where = f" chars {item.chars[0]}-{item.chars[1]}" if item.chars else ""
    if item.audio_seconds:
        where += f" at {item.audio_seconds[0]:.2f}-{item.audio_seconds[1]:.2f} s"
    flags = " [needs relisten]" if item.needs_relisten else ""
    return f"{item.id} [{item.status}] {item.chapter_id} {item.target} {item.chunk_id} {item.type}{where}{flags}"


def _format(item: QueueItem) -> list[str]:
    lines = [_line(item), f"  reason: {item.reason}"]
    if item.note:
        lines.append(f"  note: {item.note}")
    lines.append(f"  source: {item.source_text}")
    lines.append(f"  narration: {item.narration_text}")
    audio = item.audio_file or "none"
    if item.audio_seconds:
        audio += f" at {item.audio_seconds[0]:.2f}-{item.audio_seconds[1]:.2f} s"
    if item.master_offset_seconds is not None:
        audio += f" (chapter master offset {item.master_offset_seconds:.2f} s)"
    lines.append(f"  audio: {audio}")
    history = item.history
    attempts = ", ".join(str(attempt.get("status")) for attempt in history.get("attempts", [])) or "none"
    events = ", ".join(str(event.get("event") or event) for event in history.get("events", [])) or "none"
    superseded = ", ".join(history.get("superseded_audio", [])) or "none"
    lines.append(f"  history: attempts [{attempts}]; events [{events}]; superseded audio [{superseded}]")
    if item.decision is not None:
        label = "stale decision" if item.status == "stale" else "decision"
        lines.append(f"  {label}: {item.decision['state']} {item.decision['timestamp']} - {item.decision['reason']}")
    return lines


def _filter(project: BookProject, args: Namespace) -> list[QueueItem]:
    chapters = [args.chapter] if args.chapter else None
    items = build_queue(project, chapters, args.target)
    if not args.all:
        items = [item for item in items if item.is_open]
    if args.type:
        items = [item for item in items if item.type == args.type]
    return items


def _list(args: Namespace) -> int:
    project = load_project(args.directory)
    items = _filter(project, args)
    if args.json:
        print(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2))
        return 0
    if not items:
        print("no review items")
    for item in items:
        print("\n".join(_format(item)))
    return 0


def _followups(project: BookProject, item: QueueItem) -> list[str]:
    directory = project.target_dir(item.chapter_id, item.target)
    engine = item.engine_id
    lines = [
        "next steps:",
        f"  correct text: rutts book override add {project.directory} --segment {item.segment_ids[0] if item.segment_ids else '<segment>'} "
        "--find '<exact text>' --replacement '<corrected text>' --yes",
        f"  correct pronunciation: rutts book dict add {project.directory} --pattern '<word>' --replacement '<respelling>' --yes",
        f"  regenerate: rutts book regenerate {project.directory} --chapter {item.chapter_id} --engine {engine} --target {item.target}",
        f"  recheck: rutts book review recheck {project.directory} --chapter {item.chapter_id} --target {item.target}",
        f"  listen: rutts book review play {project.directory} {item.id} --join",
    ]
    if not (directory / MANIFEST_FILE).is_file():
        lines.append("  (chunk has no manifest yet; render the chapter first)")
    return lines


def _show(args: Namespace) -> int:
    project = load_project(args.directory)
    item = find_item(project, args.item_id)
    print("\n".join(_format(item)))
    print("\n".join(_followups(project, item)))
    return 0


def _play(args: Namespace) -> int:
    project = load_project(args.directory)
    item = find_item(project, args.item_id)
    path = playback.write_clip(project, item, join=args.join)
    if not args.no_play:
        if args.open:
            playback.OPENER(["open", str(path)])
        else:
            playback.PLAYER(["afplay", str(path)])
    print(f"clip: {path}")
    return 0


def _chunk_target(project: BookProject, chunk_id: str, target: str | None) -> tuple[str, dict[str, Any], str]:
    """Locate a chunk's render: its target, manifest record and planned text."""
    chapter_id = chunk_id.split("-")[0]
    candidates = []
    for name in ([target] if target else sorted(TARGET_SECONDS)):
        path = project.target_dir(chapter_id, name) / MANIFEST_FILE
        if path.is_file():
            for record in read_json(path)["chunks"]:
                if record["id"] == chunk_id:
                    candidates.append((name, record))
    if not candidates:
        raise ValidationError([f"no rendered chunk {chunk_id}"])
    if len(candidates) > 1:
        raise ValidationError([f"chunk {chunk_id} is rendered for several targets; pass --target"])
    name, record = candidates[0]
    plan = read_json(project.target_dir(chapter_id, name) / PLAN_FILE)["chunks"]
    text = next(str(chunk["text"]) for chunk in plan if chunk["id"] == chunk_id)
    return name, record, text


def _flag(args: Namespace) -> int:
    project = load_project(args.directory)
    chapter_id = args.chunk.split("-")[0]
    name, record, text = _chunk_target(project, args.chunk, args.target)
    finding = manual_store.flag(
        project,
        chapter_id=chapter_id,
        target=name,
        chunk_id=args.chunk,
        kind=args.kind,
        note=args.note,
        audio_sha256=record.get("audio_sha256"),
        chunk_fingerprint=record.get("fingerprint"),
        chunk_text=text,
        word=args.word,
        time=args.time,
    )
    print(f"flagged {finding['id']} ({finding['kind']}) on {args.chunk}; run book review show {args.directory} {finding['id']}")
    return 0


def _manual_ids(project: BookProject) -> set[str]:
    return {str(finding["id"]) for finding in manual_store.load_manual(project)}


def _decide(args: Namespace, state: str) -> int:
    project = load_project(args.directory)
    reason = decision_store.require_reason(args.reason)
    if args.item_id in _manual_ids(project):
        manual_store.set_state(project, args.item_id, state, reason)
    else:
        item = find_item(project, args.item_id)
        if not item.current:
            raise ValidationError(["verification is out of date for this item; run book review recheck first"])
        decision_store.decide(project, item, state, reason)
    print(f"{args.item_id}: {state}")
    return 0


def _resolve(args: Namespace) -> int:
    return _decide(args, "resolved")


def _dismiss(args: Namespace) -> int:
    return _decide(args, "dismissed")


def _reopen(args: Namespace) -> int:
    project = load_project(args.directory)
    if args.item_id in _manual_ids(project):
        manual_store.reopen(project, args.item_id)
    else:
        decision_store.reopen(project, find_item(project, args.item_id))
    print(f"{args.item_id}: open")
    return 0


def _recheck(args: Namespace) -> int:
    project = load_project(args.directory)
    target = resolve_target(project, args.chapter, args.target)
    before = {item.id for item in build_queue(project, [args.chapter], target)}
    try:
        transcriber = asr.get_transcriber(args.transcriber)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc
    result = verify_chapter(project, args.chapter, target, transcriber)
    after = {item.id for item in build_queue(project, [args.chapter], target)}
    print(f"{args.chapter} target={target}: checked {result.checked} chunk(s), reused {result.reused} current result(s)")
    for label, ids in (
        ("disappeared", sorted(before - after)),
        ("remained", sorted(before & after)),
        ("appeared", sorted(after - before)),
    ):
        print(f"{label}: {', '.join(ids) or 'none'}")
    return 0


def _status(args: Namespace) -> int:
    project = load_project(args.directory)
    for chapter_id, name in scopes(project, None, args.target):
        items = build_queue(project, [chapter_id], name)
        counts = {state: sum(1 for item in items if item.status == state) for state in ("open", "stale", "resolved", "dismissed")}
        print(
            f"{chapter_id} ({name}): open {counts['open']}, stale {counts['stale']}, "
            f"resolved {counts['resolved']}, dismissed {counts['dismissed']}"
        )
        eligibility = export_eligibility(project, [chapter_id], name)
        for label, state in (("reviewed", eligibility.reviewed), ("draft", eligibility.draft)):
            verdict = "yes" if state.eligible else "no"
            print(f"  {label}: {verdict}")
            for reason in state.reasons:
                print(f"    - {reason}")
    return 0
