from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.assist import base
from rutts.assist import ruaccent as _ruaccent  # noqa: F401  (registers the ruaccent assistant)
from rutts.assist import suggest
from rutts.atomic import atomic_path
from rutts.book.project import BookProject, SourceDocument, load_project
from rutts.engines import base as engines
from rutts.pronunciation import store
from rutts.pronunciation.apply import OverrideEntry, remap_overrides
from rutts.pronunciation.cli import review_and_commit
from rutts.pronunciation.markup import effective_convention, read_markup, translate
from rutts.pronunciation.store import Corrections, load_corrections
from rutts.validation import ValidationError

STRESS_NOTE = (
    "note: stress hints are stored as '+' and are removed before synthesis for any engine whose convention is 'none' "
    "(the default for every engine). Run 'rutts book markup-test' and enable a convention with 'rutts book markup-set' "
    "only after listening to the result."
)
SENTENCE_BOUNDARY = re.compile(r"[.!?…]+\s+")
LISTEN_DIR = "narration/suggestions/listen"
Target = tuple[dict[str, Any], dict[str, Any]]


def add_assist_parser(book_commands: Any) -> None:
    suggest_parser = book_commands.add_parser("suggest", help="propose stress and ё changes with a local assistant")
    suggest_parser.add_argument("directory", type=Path)
    suggest_parser.add_argument("--chapter", required=True)
    suggest_parser.add_argument("--assistant", choices=sorted(base.ASSISTANT_FACTORIES), default="ruaccent")
    suggest_parser.add_argument("--stress-scope", choices=("homographs", "all"), default="homographs")
    suggest_parser.add_argument("--yo", choices=("on", "off"), default="on")
    suggest_parser.set_defaults(book_handler=_suggest)

    listed = book_commands.add_parser("suggestions", help="list suggestions for a chapter in context")
    listed.add_argument("directory", type=Path)
    listed.add_argument("--chapter", required=True)
    listed.add_argument("--state", choices=(*suggest.STATES, "all"), default="all")
    listed.add_argument("--kind", choices=suggest.KINDS, default=None)
    listed.set_defaults(book_handler=_suggestions)

    single = book_commands.add_parser("suggestion", help="accept, reject, revert or listen to suggestions")
    actions = single.add_subparsers(dest="suggestion_command", required=True)
    accept = actions.add_parser("accept", help="accept suggestions as occurrence overrides (shows the diff first)")
    accept.add_argument("directory", type=Path)
    accept.add_argument("ids", nargs="*")
    accept.add_argument("--all-yo", action="store_true", help="accept every proposed ё suggestion in the book")
    accept.add_argument("--yes", action="store_true", help="apply without interactive confirmation")
    accept.set_defaults(book_handler=_accept)
    reject = actions.add_parser("reject", help="record that suggestions are rejected (narration is not changed)")
    reject.add_argument("directory", type=Path)
    reject.add_argument("ids", nargs="+")
    reject.set_defaults(book_handler=_reject)
    revert = actions.add_parser("revert", help="remove an accepted suggestion's override and mark it proposed")
    revert.add_argument("directory", type=Path)
    revert.add_argument("id")
    revert.set_defaults(book_handler=_revert)
    listen = actions.add_parser("listen", help="write A/B clips of the sentence without and with the change")
    listen.add_argument("directory", type=Path)
    listen.add_argument("id")
    listen.add_argument("--engine", choices=sorted(engines.ENGINE_FACTORIES), required=True)
    listen.add_argument("--precision")
    listen.add_argument("--out", type=Path, default=None)
    listen.set_defaults(book_handler=_listen)


def _load_assistant(assistant_id: str) -> base.StressAssistant:
    try:
        return base.get_assistant(assistant_id)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc


def _own_overrides(project: BookProject, chapter_id: str) -> frozenset[str]:
    previous = suggest.read_suggestions(project, chapter_id)
    if previous is None or suggest.is_stale(project, previous):
        return frozenset()
    return frozenset(
        str(item["override_id"]) for item in previous["suggestions"] if item["state"] == "accepted" and item["override_id"]
    )


def _suggest(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    if not project.read_source().chapter(args.chapter).included:
        raise ValidationError([f"chapter {args.chapter} is excluded; include it before suggesting"])
    assistant = _load_assistant(args.assistant)
    assistant.load()
    try:
        result = suggest.compute(
            project,
            args.chapter,
            assistant,
            stress_scope=args.stress_scope,
            yo_on=args.yo == "on",
            own_overrides=_own_overrides(project, args.chapter),
        )
        document = {
            "schema_version": 1,
            "chapter_id": args.chapter,
            "assistant": assistant.config.to_dict(),
            "stress_scope": args.stress_scope,
            "yo": args.yo,
            "source_sha256": suggest.chapter_sha256(project.read_source().chapter(args.chapter)),
            "unaligned": result.unaligned,
            "skipped_user_override": result.skipped_user_override,
            "suggestions": result.suggestions,
        }
    finally:
        assistant.unload()
    suggest.carry_forward(suggest.read_suggestions(project, args.chapter), document, document["source_sha256"])
    suggest.write_suggestions(project, args.chapter, document)
    counts = {kind: sum(1 for item in result.suggestions if item["kind"] == kind) for kind in suggest.KINDS}
    print(f"{args.chapter}: {counts['stress']} stress and {counts['yo']} yo suggestion(s) proposed")
    for segment_id in result.unaligned:
        print(f"unaligned (assistant output changed the text; no suggestions): {segment_id}")
    print(f"skipped (user dictionary or override takes precedence): {len(result.skipped_user_override)}")
    return 0


def _suggestions(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    document = suggest.read_suggestions(project, args.chapter)
    if document is None:
        raise ValidationError([f"no suggestions for {args.chapter}; run rutts book suggest first"])
    if suggest.is_stale(project, document):
        print("stale: the chapter text or boundaries changed since these were computed; suggest again (accept is blocked)")
    for item in document["suggestions"]:
        if args.state not in ("all", item["state"]) or args.kind not in (None, item["kind"]):
            continue
        print(
            f"{item['id']} {item['kind']} {item['segment_id']} [{item['state']}] "
            f"{item['context_before']}[{item['original']} → {item['replacement']}]{item['context_after']}"
        )
    for segment_id in document["unaligned"]:
        print(f"unaligned (no suggestions): {segment_id}")
    return 0


def _proposed_yo(project: BookProject, excluded: set[str]) -> list[Target]:
    found: list[Target] = []
    directory = project.directory / suggest.SUGGESTIONS_DIR
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        document = suggest.read_suggestions(project, path.stem)
        if document is None or suggest.is_stale(project, document):
            continue
        for item in document["suggestions"]:
            if item["kind"] == "yo" and item["state"] == "proposed" and item["id"] not in excluded:
                found.append((document, item))
    return found


def _refuse_if_covered(project: BookProject, item: dict[str, Any], corrections: Corrections) -> None:
    source = project.read_source()
    for override in remap_overrides(source, [OverrideEntry.from_dict(raw) for raw in corrections.overrides]):
        if override.segment_id == item["segment_id"] and override.start < int(item["end"]) and int(item["start"]) < override.end:
            raise ValidationError([f"suggestion {item['id']} overlaps an existing override {override.id}; not accepted"])


def _segment_text(source: SourceDocument, segment_id: str) -> str:
    chapter_index, segment_index = source.segment_location(segment_id)
    return source.chapters[chapter_index].segments[segment_index].text


def _accept(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    if not args.ids and not args.all_yo:
        raise ValidationError(["name the suggestion ids to accept, or pass --all-yo"])
    targets = [suggest.find_suggestion(project, item_id) for item_id in args.ids]
    if args.all_yo:
        targets += _proposed_yo(project, {item["id"] for _, item in targets})
    for document, item in targets:
        if suggest.is_stale(project, document):
            raise ValidationError(
                [f"suggestions for {document['chapter_id']} are stale; run rutts book suggest again before accepting"]
            )
        if item["state"] == "accepted":
            raise ValidationError([f"suggestion {item['id']} is already accepted"])
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    source = project.read_source()
    for document, item in targets:
        _refuse_if_covered(project, item, corrections)
        entry = store.add_override(
            corrections,
            source,
            segment_id=item["segment_id"],
            find=item["original"],
            occurrence=suggest.occurrence_of(_segment_text(source, item["segment_id"]), int(item["start"]), item["original"]),
            replacement=item["replacement"],
            kind=item["kind"],
            note=f"assistant {document['assistant']['assistant_id']}; suggestion {item['id']}",
        )
        item["state"] = "accepted"
        item["override_id"] = entry.id
    if any(item["kind"] == "stress" for _, item in targets):
        print(STRESS_NOTE)
    code = review_and_commit(project, corrections, before, args.yes, f"accepted {len(targets)} suggestion(s)")
    if code == 0:
        for chapter_id in sorted({document["chapter_id"] for document, _ in targets}):
            suggest.write_suggestions(project, chapter_id, _document_for(targets, chapter_id))
    return code


def _document_for(targets: list[Target], chapter_id: str) -> dict[str, Any]:
    return next(document for document, _ in targets if document["chapter_id"] == chapter_id)


def _reject(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    targets = [suggest.find_suggestion(project, item_id) for item_id in args.ids]
    for _, item in targets:
        if item["state"] == "accepted":
            raise ValidationError([f"suggestion {item['id']} is accepted; revert it before rejecting"])
    for document, item in targets:
        item["state"] = "rejected"
        suggest.write_suggestions(project, document["chapter_id"], document)
    print(f"rejected {len(targets)} suggestion(s); narration unchanged")
    return 0


def _revert(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    document, item = suggest.find_suggestion(project, args.id)
    if item["state"] != "accepted" or not item["override_id"]:
        raise ValidationError([f"suggestion {args.id} is not accepted"])
    corrections = load_corrections(project)
    seqs = [
        int(record["seq"])
        for record in corrections.history
        if record["target"] == "override" and record["action"] == "add" and record["entry"]["id"] == item["override_id"]
    ]
    if not seqs:
        raise ValidationError([f"no history entry adds override {item['override_id']}"])
    before = store.preview(project, corrections)
    store.revert(corrections, max(seqs))
    code = review_and_commit(project, corrections, before, True, f"reverted suggestion {args.id}")
    if code == 0:
        item["state"] = "proposed"
        item["override_id"] = None
        suggest.write_suggestions(project, document["chapter_id"], document)
    return code


def _sentence_pair(text: str, start: int, end: int, replacement: str) -> tuple[str, str]:
    """The sentence containing a span, without and with the replacement applied."""
    bounds = [0, *(match.end() for match in SENTENCE_BOUNDARY.finditer(text)), len(text)]
    for low, high in zip(bounds, bounds[1:]):
        if low <= start < high:
            raw = text[low:high]
            lo = low + len(raw) - len(raw.lstrip())
            hi = lo + len(raw.strip())
            return text[lo:hi], text[lo:start] + replacement + text[end:hi]
    raise ValidationError(["suggestion span is outside its paragraph"])


def _listen(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    _, item = suggest.find_suggestion(project, args.id)
    text = _segment_text(project.read_source(), item["segment_id"])
    without, with_change = _sentence_pair(text, int(item["start"]), int(item["end"]), item["replacement"])
    convention = effective_convention(args.engine, read_markup(project.directory))
    options: dict[str, str] = {} if args.precision is None else {"precision": args.precision}
    try:
        engine = engines.get_engine(args.engine, **options)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc
    narrator = project.narrator()
    out = args.out if args.out is not None else project.directory / LISTEN_DIR / args.id
    out.mkdir(parents=True, exist_ok=True)
    engine.load()
    try:
        for label, sentence in (("without", without), ("with", with_change)):
            result = engine.synthesize(translate(sentence, convention), narrator)
            samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
            path = out / f"{args.id}-{label}.wav"
            with atomic_path(path) as temp:
                sf.write(str(temp), samples, result.sample_rate, subtype="FLOAT")
            print(f"{label}: {path}")
    finally:
        engine.unload()
    if item["kind"] == "stress" and convention == "none":
        print(f"note: {args.engine} has no verified stress convention, so both clips are rendered without the hint")
    return 0
