from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rutts.book.epub.archive import ZIP_SIGNATURE, EpubArchive, open_epub
from rutts.book.epub.content import Block, LinkMark, UnreadableDocument, parse_document
from rutts.book.epub.package import NavEntry, Package, SpineItem, check_drm, read_package
from rutts.book.project import (
    IMPORT_REPORT_FILE,
    ORIGINAL_EPUB_FILE,
    PROJECT_FILE,
    SCHEMA_VERSION,
    BookProject,
    SourceChapter,
    SourceDocument,
    SourceSegment,
    create_project,
    load_project,
    renumber,
    sha256_bytes,
    write_json,
)
from rutts.pronunciation.store import rebuild_narration
from rutts.validation import ValidationError, raise_for, text_problems

POLICIES = ("omit", "chapter-end", "inline")
NAV_TYPES = frozenset({"toc", "landmarks", "page-list"})
FRONT_TYPES = frozenset({"titlepage", "copyright-page", "frontmatter", "colophon", "dedication", "imprint"})
FRONT_GUIDE = frozenset({"title-page", "copyright-page", "colophon", "dedication"})
FRONT_NAME = re.compile(r"title-?page|title_page|copyright|colophon|dedication|imprint", re.IGNORECASE)
PROSE_KINDS = ("paragraph", "heading")


@dataclass
class _Piece:
    spine_index: int
    href: str
    title: str | None
    blocks: list[Block]
    reason: str | None


@dataclass
class EpubBuild:
    package: Package
    source: SourceDocument
    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def check_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValidationError([f"unknown footnote policy {policy!r}; choose one of {', '.join(POLICIES)}"])
    return policy


def is_epub(path: Path) -> bool:
    if path.suffix.lower() == ".epub":
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) == ZIP_SIGNATURE
    except OSError:
        return False


def _stem(href: str) -> str:
    return Path(href).stem or href


def _doc_reason(item: SpineItem, package: Package, blocks: list[Block], epub_types: set[str], nav_types: set[str]) -> str | None:
    guide = package.guide_types.get(item.item.path, "")
    types = epub_types | nav_types
    name = Path(item.item.path).name
    if item.item.path == package.nav_document or types & NAV_TYPES or guide == "toc":
        return "navigation"
    if "cover-image" in item.item.properties or "cover" in types or guide == "cover" or name.lower().startswith("cover"):
        return "cover"
    if types & FRONT_TYPES or guide in FRONT_GUIDE or FRONT_NAME.search(name):
        return "front_matter"
    prose = [block for block in blocks if block.kind in PROSE_KINDS]
    linked = [block for block in prose if block.links]
    if prose and len(linked) >= 3 and len(linked) >= 0.8 * len(prose):
        return "navigation"
    if not item.linear:
        return "non_linear_spine_item"
    return None


def _pieces(archive: EpubArchive, package: Package, item: SpineItem, issues: list[str]) -> list[_Piece]:
    href = item.item.path
    try:
        document = parse_document(href, archive.read(href))
    except UnreadableDocument as exc:
        issues.append(f"unreadable document {href} (excluded): {exc}")
        return [_Piece(item.index, href, None, [], "unreadable_document")]
    issues.extend(f"unsupported content: {entry}" for entry in document.unsupported)
    ids: dict[str, int] = {}
    for index, block in enumerate(document.blocks):
        for ident in sorted(block.ids):
            ids.setdefault(ident, index)
    starts: dict[int, str] = {}
    for entry in package.nav_entries:
        if entry.path == href and entry.fragment and entry.fragment in ids:
            starts.setdefault(ids[entry.fragment], entry.title)
    plain = next((entry.title for entry in package.nav_entries if entry.path == href and not entry.fragment), None)
    doc_reason = _doc_reason(item, package, document.blocks, document.epub_types, document.nav_types)
    cuts = sorted({0, *(index for index in starts if index > 0)})
    ends = [*cuts[1:], len(document.blocks)]
    pieces: list[_Piece] = []
    for number, (begin, end) in enumerate(zip(cuts, ends, strict=True)):
        blocks = document.blocks[begin:end]
        title = starts.get(begin, plain if number == 0 else None)
        if number == 0 and 0 in starts:
            title = starts[0]
        reason = doc_reason
        if reason is None and not any(block.kind in PROSE_KINDS for block in blocks):
            reason = "no_prose"
        pieces.append(_Piece(item.index, href, title, blocks, reason))
    return pieces


def _location(piece: _Piece, block_index: int, block: Block) -> dict[str, Any]:
    anchor = min(block.ids) if block.ids else None
    return {"href": piece.href, "spine_index": piece.spine_index, "block_index": block_index, "anchor": anchor}


def _body_refs(
    pieces: list[_Piece], containers: dict[str, list[tuple[int, int]]], ids: dict[str, list[tuple[int, int]]]
) -> tuple[dict[tuple[int, int], list[LinkMark]], set[str]]:
    """Valid footnote references from included chapters: block -> links, and the referenced targets."""
    valid: dict[tuple[int, int], list[LinkMark]] = {}
    targets: set[str] = set()
    for pi, piece in enumerate(pieces):
        if piece.reason is not None:
            continue
        for bi, block in enumerate(piece.blocks):
            for link in block.links:
                if not link.target:
                    continue
                if link.noteref:
                    ok = link.target in containers or link.target in ids
                else:
                    ok = link.symbolic and not block.note_containers and link.target in ids
                if ok:
                    valid.setdefault((pi, bi), []).append(link)
                    targets.add(link.target)
    return valid, targets


def _build_chapters(pieces: list[_Piece], policy: str) -> tuple[list[SourceChapter], dict[str, Any]]:
    containers: dict[str, list[tuple[int, int]]] = {}
    ids: dict[str, list[tuple[int, int]]] = {}
    for pi, piece in enumerate(pieces):
        for bi, block in enumerate(piece.blocks):
            for key in block.note_containers:
                containers.setdefault(key, []).append((pi, bi))
            for ident in block.ids:
                ids.setdefault(f"{piece.href}#{ident}", []).append((pi, bi))
    valid, targets = _body_refs(pieces, containers, ids)

    def bodies(key: str) -> list[tuple[int, int]]:
        found = {
            (pi, bi)
            for pi, bi in containers.get(key, []) + ids.get(key, [])
            if pieces[pi].blocks[bi].kind in PROSE_KINDS
        }
        return sorted(found)

    note_bodies: dict[tuple[int, int], str] = {}
    for key in targets:
        for body_position in bodies(key):
            note_bodies.setdefault(body_position, key)

    relocations: list[dict[str, Any]] = []
    chapters: list[SourceChapter] = []
    relocated: set[str] = set()
    for pi, piece in enumerate(pieces):
        segments: list[SourceSegment] = []
        tail: list[SourceSegment] = []
        note_number = 0
        for bi, block in enumerate(piece.blocks):
            location = _location(piece, bi, block)
            if block.kind == "auxiliary":
                segment = SourceSegment("", "", "auxiliary", block.alt or block.text, location, False, block.reason)
            else:
                reason = None
                if (pi, bi) in note_bodies:
                    reason = "footnote_omitted" if policy == "omit" else "footnotes_relocated"
                cuts = [(link.start, link.end) for link in valid.get((pi, bi), [])]
                segment = SourceSegment(
                    "", "", block.kind, block.text, location, reason is None, reason, narration_cuts=cuts
                )
            segments.append(segment)
            if piece.reason is not None or policy == "omit":
                continue
            copies: list[SourceSegment] = []
            for link in valid.get((pi, bi), []):
                if link.target in relocated:
                    continue
                relocated.add(link.target)
                note_number += 1
                for position, (qi, qj) in enumerate(bodies(link.target)):
                    body = pieces[qi].blocks[qj]
                    body_location = {**_location(pieces[qi], qj, body), "relocated": True}
                    prefix = f"Примечание {note_number}. " if position == 0 else ""
                    copies.append(SourceSegment("", "", "paragraph", body.text, body_location, narration_prefix=prefix))
                    relocations.append(
                        {"chapter": len(chapters), "number": note_number, "note": _location(pieces[qi], qj, body)}
                    )
            if policy == "inline":
                segments.extend(copies)
            else:
                tail.extend(copies)
        segments.extend(tail)
        reason = piece.reason
        prose_included = any(seg.included and seg.kind == "paragraph" for seg in segments)
        if reason is None and not prose_included and any(seg.exclusion_reason == "footnotes_relocated" for seg in segments):
            reason = "footnotes_relocated"
        title = piece.title or next((b.text for b in piece.blocks if b.kind == "heading"), None)
        chapters.append(
            SourceChapter(
                id="",
                title=title or "",
                order=0,
                included=reason is None,
                segments=segments,
                exclusion_reason=reason,
            )
        )
    for order, chapter in enumerate(chapters, start=1):
        if not chapter.title:
            chapter.title = f"Глава {order}" if chapter.included else _stem(pieces[order - 1].href)
    renumber(chapters)
    footnotes = {
        "policy": policy,
        "references": sum(len(links) for links in valid.values()),
        "bodies": len(note_bodies),
        "relocated": [
            {**relocation, "chapter": f"c{relocation['chapter'] + 1:03d}"} for relocation in relocations
        ],
    }
    return chapters, footnotes


def _report(package: Package, pieces: list[_Piece], chapters: list[SourceChapter], footnotes: dict[str, Any], policy: str, warnings: list[str]) -> dict[str, Any]:
    spine = [
        {
            "spine_index": piece.spine_index,
            "href": piece.href,
            "chapter_id": chapter.id,
            "title": chapter.title,
            "included": chapter.included,
            "reason": chapter.exclusion_reason,
        }
        for piece, chapter in zip(pieces, chapters, strict=True)
    ]
    exclusions = [
        {"chapter_id": chapter.id, "segment_id": None, "reason": chapter.exclusion_reason}
        for chapter in chapters
        if not chapter.included
    ] + [
        {"chapter_id": chapter.id, "segment_id": segment.id, "reason": segment.exclusion_reason, "location": segment.location}
        for chapter in chapters
        for segment in chapter.segments
        if not segment.included
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "footnote_policy": policy,
        "metadata": {"title": package.title, "author": package.author, "language": package.language},
        "spine": spine,
        "footnotes": footnotes,
        "exclusions": exclusions,
        "warnings": warnings,
    }


def build_epub(data: bytes, policy: str) -> EpubBuild:
    check_policy(policy)
    archive = open_epub(data)
    package = read_package(archive)
    check_drm(archive, [item.item.path for item in package.spine])
    warnings: list[str] = []
    if package.fixed_layout:
        warnings.append("fixed-layout EPUB: text is imported as reflowed prose; check the result")
    pieces: list[_Piece] = []
    for item in package.spine:
        pieces.extend(_pieces(archive, package, item, warnings))
    chapters, footnotes = _build_chapters(pieces, policy)
    prose = " ".join(
        segment.text
        for chapter in chapters
        if chapter.included
        for segment in chapter.segments
        if segment.included and segment.kind in PROSE_KINDS
    )
    if not prose.strip():
        raise ValidationError(["no Russian prose found in EPUB"])
    raise_for(text_problems(prose, "EPUB prose"))
    source = SourceDocument(chapters=chapters)
    report = _report(package, pieces, chapters, footnotes, policy, warnings)
    return EpubBuild(package=package, source=source, report=report, warnings=warnings)


def import_epub(
    source_file: Path,
    out: Path,
    *,
    title: str | None,
    author: str,
    footnotes: str,
) -> tuple[BookProject, list[str]]:
    check_policy(footnotes)
    if not source_file.is_file():
        raise ValidationError([f"input file not found: {source_file}"])
    if (out / PROJECT_FILE).exists():
        raise ValidationError([f"a book project already exists in {out}"])
    data = source_file.read_bytes()
    build = build_epub(data, footnotes)
    original = out / ORIGINAL_EPUB_FILE
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(data)
    metadata = build.package
    meta = {
        "title": title or metadata.title or source_file.stem,
        "author": author or metadata.author,
        "language": metadata.language,
        "source_format": "epub",
        "footnote_policy": footnotes,
        "original_filename": source_file.name,
        "original_sha256": sha256_bytes(data),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    project = create_project(out, meta, build.source)
    write_json(out / IMPORT_REPORT_FILE, build.report)
    return project, build.warnings


def set_footnote_policy(project: BookProject, policy: str) -> list[str]:
    """Re-derive the EPUB source from the retained original with another footnote policy.

    Boundary corrections made after import are not carried over; re-apply them afterwards.
    """
    check_policy(policy)
    if project.meta.get("source_format") != "epub":
        raise ValidationError(["footnote policies apply only to EPUB projects"])
    build = build_epub((project.directory / ORIGINAL_EPUB_FILE).read_bytes(), policy)
    project.save_source(build.source)
    write_json(project.directory / PROJECT_FILE, {"schema_version": SCHEMA_VERSION, **{**project.meta, "footnote_policy": policy}})
    write_json(project.directory / IMPORT_REPORT_FILE, build.report)
    rebuild_narration(load_project(project.directory))
    return build.warnings
