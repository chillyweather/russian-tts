from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rutts.atomic import atomic_path
from rutts.book.assemble import Pauses
from rutts.book.chunking import TARGET_SECONDS
from rutts.book.project import BookProject, load_project, read_json
from rutts.book.render import MANIFEST_FILE, render_chapter
from rutts.engines import base
from rutts.narrator import load_profile
from rutts.pronunciation import store
from rutts.pronunciation.apply import KINDS, MATCH_MODES, NarrationBuild, format_changes
from rutts.pronunciation.markup import ACUTE, CONVENTIONS, STRESS_MARK
from rutts.pronunciation.store import Corrections, invalidated_chunks, load_corrections
from rutts.validation import ValidationError

MARKUP_SENTENCE = "Старый замок стоял на высоком холме."
MARKUP_VARIANTS = {
    "plain": MARKUP_SENTENCE,
    "plus": MARKUP_SENTENCE.replace("замок", f"зам{STRESS_MARK}ок"),
    "acute": MARKUP_SENTENCE.replace("замок", f"замо{ACUTE}к"),
}
JOIN_DIR = "review/joins"


def add_pronunciation_parser(book_commands: Any) -> None:
    dictionary = book_commands.add_parser("dict", help="per-book pronunciation dictionary")
    dict_commands = dictionary.add_subparsers(dest="dict_command", required=True)
    add = dict_commands.add_parser("add", help="add a dictionary entry (shows the diff first)")
    add.add_argument("directory", type=Path)
    add.add_argument("--pattern", required=True)
    add.add_argument("--replacement", required=True)
    add.add_argument("--match", choices=MATCH_MODES, default="word")
    add.add_argument("--kind", choices=KINDS, default="other")
    add.add_argument("--note", default="")
    add.add_argument("--case-sensitive", action="store_true")
    add.add_argument("--yes", action="store_true", help="apply without interactive confirmation")
    add.set_defaults(book_handler=_dict_add)
    listed = dict_commands.add_parser("list", help="list dictionary entries")
    listed.add_argument("directory", type=Path)
    listed.set_defaults(book_handler=_dict_list)
    removed = dict_commands.add_parser("remove", help="remove a dictionary entry")
    removed.add_argument("directory", type=Path)
    removed.add_argument("id")
    removed.set_defaults(book_handler=_dict_remove)

    override = book_commands.add_parser("override", help="occurrence-specific pronunciation overrides")
    override_commands = override.add_subparsers(dest="override_command", required=True)
    add_override = override_commands.add_parser("add", help="override one occurrence (shows the diff first)")
    add_override.add_argument("directory", type=Path)
    add_override.add_argument("--segment", required=True, help="source segment id, e.g. c001-s0004")
    add_override.add_argument("--find", required=True, help="exact text inside the segment")
    add_override.add_argument("--occurrence", type=int, default=None, help="1-based occurrence when --find repeats")
    add_override.add_argument("--replacement", required=True)
    add_override.add_argument("--kind", choices=KINDS, default="other")
    add_override.add_argument("--note", default="")
    add_override.add_argument("--yes", action="store_true", help="apply without interactive confirmation")
    add_override.set_defaults(book_handler=_override_add)
    listed_overrides = override_commands.add_parser("list", help="list overrides and stale ones")
    listed_overrides.add_argument("directory", type=Path)
    listed_overrides.set_defaults(book_handler=_override_list)
    removed_override = override_commands.add_parser("remove", help="remove an override")
    removed_override.add_argument("directory", type=Path)
    removed_override.add_argument("id")
    removed_override.set_defaults(book_handler=_override_remove)

    diff = book_commands.add_parser("diff", help="review source-to-narration changes and invalidated chunks")
    diff.add_argument("directory", type=Path)
    diff.add_argument("--chapter")
    diff.add_argument("--pending-file", type=Path, help="JSON with proposed dictionary/overrides (not applied)")
    diff.set_defaults(book_handler=_diff)

    history = book_commands.add_parser("history", help="list every correction change, oldest first")
    history.add_argument("directory", type=Path)
    history.set_defaults(book_handler=_history)

    revert = book_commands.add_parser("revert", help="undo a history entry by appending a new one")
    revert.add_argument("directory", type=Path)
    revert.add_argument("seq", type=int)
    revert.set_defaults(book_handler=_revert)

    regenerate = book_commands.add_parser("regenerate", help="render only invalidated chunks and write join previews")
    regenerate.add_argument("directory", type=Path)
    regenerate.add_argument("--chapter", required=True)
    regenerate.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    regenerate.add_argument("--precision")
    regenerate.add_argument("--target", choices=sorted(TARGET_SECONDS), default="short")
    regenerate.add_argument("--offline", action="store_true")
    regenerate.add_argument("--retries", type=int, default=2)
    regenerate.add_argument("--accept-config-change", action="store_true")
    regenerate.set_defaults(book_handler=_regenerate)

    markup_test = book_commands.add_parser("markup-test", help="render plain, '+' and acute variants for listening")
    markup_test.add_argument("directory", type=Path)
    markup_test.add_argument("--engine", choices=sorted(base.ENGINE_FACTORIES), required=True)
    markup_test.add_argument("--precision")
    markup_test.add_argument("--narrator", type=Path, required=True)
    markup_test.add_argument("--out", type=Path, required=True)
    markup_test.add_argument("--offline", action="store_true")
    markup_test.set_defaults(book_handler=_markup_test)

    markup_set = book_commands.add_parser("markup-set", help="enable a stress-hint convention for one engine")
    markup_set.add_argument("directory", type=Path)
    markup_set.add_argument("--engine", required=True)
    markup_set.add_argument("--convention", choices=CONVENTIONS, required=True)
    markup_set.add_argument("--verified-by-listening", action="store_true")
    markup_set.set_defaults(book_handler=_markup_set)


def _engine(engine_id: str, precision: str | None) -> base.SynthesisEngine:
    options: dict[str, str] = {} if precision is None else {"precision": precision}
    try:
        return base.get_engine(engine_id, **options)
    except ValueError as exc:
        raise ValidationError([str(exc)]) from exc


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ValidationError(["review the diff above, then pass --yes to apply the change"])
    return input("apply these changes? [y/N] ").strip().lower() in ("y", "yes")


def _print_diff(before: NarrationBuild, after: NarrationBuild, chapter: str | None = None) -> None:
    before_keys = {change.key for change in before.changes}
    after_keys = {change.key for change in after.changes}
    in_chapter = (lambda segment_id: segment_id.startswith(f"{chapter}-")) if chapter else (lambda _: True)
    for line in format_changes([change for change in after.changes if change.key not in before_keys and in_chapter(change.segment_id)]):
        print(f"+ {line}")
    for line in format_changes([change for change in before.changes if change.key not in after_keys and in_chapter(change.segment_id)]):
        print(f"- {line}")
    for stale in after.stale:
        print(stale.describe())


def _print_invalidation(project: BookProject, before: NarrationBuild, after: NarrationBuild) -> None:
    for target, chunk_ids in invalidated_chunks(project, before.document, after.document).items():
        print(f"would invalidate {target}: {', '.join(chunk_ids) if chunk_ids else 'none'}")


def _review_and_commit(
    project: BookProject, corrections: Corrections, before: NarrationBuild, yes: bool, message: str
) -> int:
    after = store.preview(project, corrections)
    _print_diff(before, after)
    _print_invalidation(project, before, after)
    if not _confirm(yes):
        print("not applied")
        return 1
    store.commit(project, corrections)
    print(f"{message}; narration updated")
    return 0


def _dict_add(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    entry = store.add_dictionary(
        corrections,
        pattern=args.pattern,
        replacement=args.replacement,
        match=args.match,
        kind=args.kind,
        note=args.note,
        case_sensitive=args.case_sensitive,
    )
    return _review_and_commit(project, corrections, before, args.yes, f"added dictionary entry {entry.id}")


def _dict_list(args: argparse.Namespace) -> int:
    corrections = load_corrections(load_project(args.directory))
    if not corrections.dictionary:
        print("no dictionary entries")
    for item in corrections.dictionary:
        case = "case-sensitive" if item["case_sensitive"] else "case-insensitive"
        print(f"{item['id']} {item['match']} {item['pattern']!r} -> {item['replacement']!r} [{item['kind']}, {case}] {item['note']}".rstrip())
    return 0


def _dict_remove(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    store.remove_entry(corrections, "dictionary", args.id)
    return _review_and_commit(project, corrections, before, True, f"removed dictionary entry {args.id}")


def _override_add(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    entry = store.add_override(
        corrections,
        project.read_source(),
        segment_id=args.segment,
        find=args.find,
        occurrence=args.occurrence,
        replacement=args.replacement,
        kind=args.kind,
        note=args.note,
    )
    return _review_and_commit(project, corrections, before, args.yes, f"added override {entry.id}")


def _override_list(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    build = store.preview(project, corrections)
    if not corrections.overrides:
        print("no overrides")
    for item in corrections.overrides:
        print(
            f"{item['id']} {item['segment_id']} [{item['start']}:{item['end']}] "
            f"{item['original']!r} -> {item['replacement']!r} ({item['kind']}) {item['note']}".rstrip()
        )
    for stale in build.stale:
        print(stale.describe())
    return 0


def _override_remove(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    store.remove_entry(corrections, "override", args.id)
    return _review_and_commit(project, corrections, before, True, f"removed override {args.id}")


def _load_pending(corrections: Corrections, project: BookProject, path: Path) -> None:
    """Add the proposed entries from a JSON file to an in-memory copy; nothing is written."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        for item in document.get("dictionary", []):
            store.add_dictionary(
                corrections,
                pattern=item["pattern"],
                replacement=item["replacement"],
                match=item.get("match", "word"),
                kind=item.get("kind", "other"),
                note=item.get("note", ""),
                case_sensitive=item.get("case_sensitive", False),
            )
        for item in document.get("overrides", []):
            store.add_override(
                corrections,
                project.read_source(),
                segment_id=item["segment_id"],
                find=item["find"],
                occurrence=item.get("occurrence"),
                replacement=item["replacement"],
                kind=item.get("kind", "other"),
                note=item.get("note", ""),
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError([f"pending file {path} is not usable: {exc}"]) from exc


def _diff(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    current = load_corrections(project)
    current_keys = {change.key for change in store.preview(project, current).changes}
    proposed = current.snapshot()
    if args.pending_file is not None:
        _load_pending(proposed, project, args.pending_file)
    build = store.preview(project, proposed)
    in_chapter = (lambda segment_id: segment_id.startswith(f"{args.chapter}-")) if args.chapter else (lambda _: True)
    changes = [change for change in build.changes if in_chapter(change.segment_id)]
    if not changes:
        print("no source-to-narration changes")
    for change in changes:
        marker = " " if change.key in current_keys else "+"
        print(f"{marker} {change.describe()}")
    for stale in build.stale:
        print(stale.describe())
    _print_invalidation(project, NarrationBuild(project.read_narration(), [], []), build)
    return 0


def _summary(item: dict[str, Any], target: str) -> str:
    if target == "markup":
        return f"{item['id']} {item['convention']}"
    if target == "dictionary":
        return f"{item['pattern']!r} -> {item['replacement']!r}"
    return f"{item['segment_id']} {item['original']!r} -> {item['replacement']!r}"


def _history(args: argparse.Namespace) -> int:
    corrections = load_corrections(load_project(args.directory))
    if not corrections.history:
        print("no history")
    for record in corrections.history:
        reverted = f" (reverts #{record['reverts_seq']})" if record["reverts_seq"] else ""
        entry = record["entry"]
        print(
            f"#{record['seq']} {record['timestamp']} {record['action']} {record['target']} "
            f"{entry['id']}: {_summary(entry, record['target'])}{reverted}"
        )
    return 0


def _revert(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    before = store.preview(project, corrections)
    store.revert(corrections, args.seq)
    return _review_and_commit(project, corrections, before, True, f"reverted #{args.seq}")


def _fingerprints(directory: Path) -> dict[str, str]:
    """Fingerprints of the chunks that currently have valid audio."""
    if not (directory / MANIFEST_FILE).is_file():
        return {}
    return {
        record["id"]: record["fingerprint"]
        for record in read_json(directory / MANIFEST_FILE)["chunks"]
        if record["status"] == "ok"
    }


def _silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(round(seconds * rate), dtype=np.float32)


def write_join_preview(project: BookProject, chapter_id: str, target: str, chunk_id: str) -> Path:
    """Previous chunk, pause, regenerated chunk, pause, next chunk; only neighbours with audio are included."""
    directory = project.target_dir(chapter_id, target)
    manifest = read_json(directory / MANIFEST_FILE)
    records: list[dict[str, Any]] = manifest["chunks"]
    plan: list[dict[str, Any]] = read_json(directory / "plan.json")["chunks"]
    pauses = Pauses()
    position = next(index for index, record in enumerate(records) if record["id"] == chunk_id)

    def audio(index: int) -> np.ndarray:
        samples, _ = sf.read(str(directory / str(records[index]["audio_file"])), dtype="float32")
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    rate = int(records[position]["sample_rate"])
    pieces: list[np.ndarray] = []
    if position > 0 and records[position - 1]["status"] == "ok":
        pieces += [audio(position - 1), _silence(pauses.seconds(plan[position - 1]["boundary_after"]), rate)]
    pieces.append(audio(position))
    if position + 1 < len(records) and records[position + 1]["status"] == "ok":
        pieces += [_silence(pauses.seconds(plan[position]["boundary_after"]), rate), audio(position + 1)]
    destination = project.directory / JOIN_DIR / f"{chunk_id}.wav"
    with atomic_path(destination) as temp:
        sf.write(str(temp), np.concatenate(pieces), rate, subtype="FLOAT")
    return destination


def _regenerate(args: argparse.Namespace) -> int:
    if args.retries < 0:
        raise ValidationError(["--retries must be zero or positive"])
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    project = load_project(args.directory)
    engine = _engine(args.engine, args.precision)
    before = _fingerprints(project.target_dir(args.chapter, args.target))
    try:
        manifest = render_chapter(
            project,
            args.chapter,
            engine,
            args.target,
            accept_config_change=args.accept_config_change,
            retries=args.retries,
            progress=print,
        )
    except KeyboardInterrupt:
        print("interrupted: saved chunks are kept; run regenerate again to resume", file=sys.stderr)
        return 130
    records: list[dict[str, Any]] = manifest["chunks"]
    regenerated = [
        record["id"]
        for record in records
        if record["status"] == "ok" and before.get(record["id"]) != record["fingerprint"]
    ]
    print(f"regenerated {len(regenerated)} chunk(s): {', '.join(regenerated) or 'none'}")
    for chunk_id in regenerated:
        print(f"join preview: {write_join_preview(project, args.chapter, args.target, chunk_id)}")
    failed = [record["id"] for record in records if record["status"] == "failed"]
    for chunk_id in failed:
        print(f"failed chunk (needs review): {chunk_id}")
    return 0 if not failed else 1


def _markup_test(args: argparse.Namespace) -> int:
    load_project(args.directory)
    profile = load_profile(args.narrator)
    engine = _engine(args.engine, args.precision)
    args.out.mkdir(parents=True, exist_ok=True)
    engine.load()
    try:
        for name, text in MARKUP_VARIANTS.items():
            result = engine.synthesize(text, profile)
            samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
            path = args.out / f"markup-{name}.wav"
            with atomic_path(path) as temp:
                sf.write(str(temp), samples, result.sample_rate, subtype="FLOAT")
            print(f"{name}: {text} -> {path}")
    finally:
        engine.unload()
    return 0


def _markup_set(args: argparse.Namespace) -> int:
    project = load_project(args.directory)
    corrections = load_corrections(project)
    store.set_markup(corrections, args.engine, args.convention, verified_by_listening=args.verified_by_listening)
    store.commit(project, corrections)
    print(f"{args.engine}: pronunciation convention set to {args.convention}; chunks with hints regenerate on the next render")
    return 0
