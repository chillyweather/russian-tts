from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from rutts.book.epub.archive import resolve

BLOCK_TAGS = frozenset(
    "p div blockquote li dt dd h1 h2 h3 h4 h5 h6 section article aside header footer nav figure "
    "figcaption ul ol dl body html pre address hgroup".split()
)
HEADING_TAGS = frozenset("h1 h2 h3 h4 h5 h6".split())
UNSUPPORTED_TAGS = {
    "math": "mathml",
    "audio": "audio",
    "video": "video",
    "object": "embedded_object",
    "iframe": "embedded_frame",
}
NOTE_TYPES = frozenset("footnote endnote rearnote footnotes endnotes rearnotes note".split())
SCENE_CLASSES = ("scene", "break")
SCENE_TEXT = re.compile(r"[*⁂⁕❦❧✦✧~•](?:\s*[*⁂⁕❦❧✦✧~•])*")
SYMBOLIC_MARK = re.compile(r"[\d*†‡]{1,4}")
SOFT_HYPHEN = "­"


@dataclass
class LinkMark:
    start: int
    end: int
    target: str
    noteref: bool
    symbolic: bool


@dataclass
class Block:
    kind: str  # paragraph | heading | scene_break | auxiliary
    text: str
    ids: frozenset[str] = frozenset()
    note_containers: frozenset[str] = frozenset()
    links: list[LinkMark] = field(default_factory=list)
    reason: str | None = None
    alt: str = ""


@dataclass
class Document:
    path: str
    blocks: list[Block]
    nav_types: set[str]
    epub_types: set[str]
    unsupported: list[str]


class UnreadableDocument(Exception):
    pass


def normalize(raw: str, spans: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace (NBSP included), drop soft hyphens and edge spaces; remap spans."""
    out: list[str] = []
    mapping = [0] * (len(raw) + 1)
    previous_space = True
    for index, char in enumerate(raw):
        mapping[index] = len(out)
        if char == SOFT_HYPHEN:
            continue
        if char.isspace():
            if previous_space:
                continue
            out.append(" ")
            previous_space = True
        else:
            out.append(char)
            previous_space = False
    mapping[len(raw)] = len(out)
    text = "".join(out).rstrip(" ")
    limit = len(text)
    return text, [(min(mapping[start], limit), min(mapping[end], limit)) for start, end in spans]


@dataclass
class _Open:
    ids: set[str]
    containers: frozenset[str]
    css: str
    tag: str
    parts: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    link_meta: list[tuple[str, bool, bool]] = field(default_factory=list)
    link_open: tuple[int, str, bool] | None = None

    def length(self) -> int:
        return sum(len(part) for part in self.parts)


class _Collector(HTMLParser):
    def __init__(self, path: str) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.base = posixpath.dirname(path)
        self.blocks: list[Block] = []
        self.current: _Open | None = None
        self.pending_ids: set[str] = set()
        self.containers: list[tuple[str, str]] = []
        self.block_tag = "div"
        self.block_class = ""
        self.skip_tag: str | None = None
        self.skip_parts: list[str] = []
        self.skip_alt = ""
        self.table_depth = 0
        self.table_parts: list[str] = []
        self.unsupported: list[str] = []
        self.nav_types: set[str] = set()
        self.epub_types: set[str] = set()

    # Block bookkeeping
    def _open(self) -> _Open:
        if self.current is None:
            self.current = _Open(
                ids=set(self.pending_ids),
                containers=frozenset(key for _, key in self.containers),
                css=self.block_class,
                tag=self.block_tag,
            )
            self.pending_ids = set()
        return self.current

    def _add(self, block: Block) -> None:
        self.blocks.append(block)

    def _flush(self) -> None:
        if self.current is None:
            return
        opened, self.current = self.current, None
        text, spans = normalize("".join(opened.parts), opened.spans)
        if not text:
            self.pending_ids |= opened.ids
            return
        if SCENE_TEXT.fullmatch(text):
            kind = "scene_break"
        elif opened.tag in HEADING_TAGS:
            kind = "heading"
        else:
            kind = "paragraph"
        links = [
            LinkMark(start, end, target, noteref, symbolic)
            for (start, end), (target, noteref, symbolic) in zip(spans, opened.link_meta, strict=True)
        ]
        self._add(Block(kind, text, frozenset(opened.ids), frozenset(opened.containers), links))

    def _auxiliary(self, reason: str, text: str = "", alt: str = "") -> None:
        self._flush()
        self._add(Block("auxiliary", text, frozenset(self.pending_ids), reason=reason, alt=alt))
        self.pending_ids = set()

    def _is_scene_marker(self, css: str) -> bool:
        return any(name in css for name in SCENE_CLASSES)

    # HTMLParser hooks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if self.skip_tag is not None:
            return
        if tag == "head":
            self._flush()
            self.skip_tag = tag
            return
        if tag in ("script", "style"):
            self._auxiliary(tag)
            self.skip_tag = tag
            return
        if tag == "img":
            self._auxiliary("illustration", alt=values.get("alt", ""))
            return
        if tag == "svg":
            self._flush()
            self.skip_tag = tag
            self.skip_alt = values.get("aria-label", "")
            self.skip_parts = []
            return
        if tag in UNSUPPORTED_TAGS:
            self.unsupported.append(f"{UNSUPPORTED_TAGS[tag]} in {self.path}")
            self._auxiliary(UNSUPPORTED_TAGS[tag])
            self.skip_tag = tag
            return
        if tag == "table":
            self._flush()
            self.table_depth += 1
            if self.table_depth == 1:
                self.table_parts = []
            return
        if tag == "hr":
            self._auxiliary_scene()
            return
        epub_type = values.get("epub:type", "")
        self.epub_types.update(epub_type.split())
        if tag == "nav":
            self.nav_types.update(epub_type.split())
        if tag == "br":
            if self.current is not None:
                self.current.parts.append(" ")
            return
        if tag == "a" and self.table_depth == 0:
            opened = self._open()
            noteref = "noteref" in epub_type.split()
            opened.link_open = (opened.length(), values.get("href", ""), noteref)
        if values.get("id"):
            if tag in BLOCK_TAGS:
                self._flush()
                self.pending_ids.add(values["id"])
            elif self.current is not None:
                self.current.ids.add(values["id"])
            else:
                self.pending_ids.add(values["id"])
        if tag in BLOCK_TAGS:
            self._flush()
            self.block_tag = tag
            self.block_class = values.get("class", "")
            if tag == "aside" or set(epub_type.split()) & NOTE_TYPES:
                key = f"{self.path}#{values['id']}" if values.get("id") else f"{self.path}#note{len(self.blocks)}"
                self.containers.append((tag, key))

    def _auxiliary_scene(self) -> None:
        self._flush()
        self._add(Block("scene_break", "", frozenset(self.pending_ids)))
        self.pending_ids = set()

    def handle_endtag(self, tag: str) -> None:
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                if tag == "svg":
                    alt = self.skip_alt or " ".join("".join(self.skip_parts).split())
                    self._auxiliary("illustration", alt=alt)
                self.skip_tag = None
            return
        if tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0:
                self._auxiliary("table", text=" ".join("".join(self.table_parts).split()))
            return
        if tag == "a" and self.current is not None and self.current.link_open is not None:
            start, href, noteref = self.current.link_open
            self.current.link_open = None
            raw = "".join(self.current.parts)
            visible = " ".join(raw[start:].split())
            self.current.spans.append((start, len(raw)))
            self.current.link_meta.append(self._link_target(href, noteref, bool(SYMBOLIC_MARK.fullmatch(visible))))
            return
        if tag in BLOCK_TAGS:
            was_scene = self._is_scene_marker(self.block_class) and self.current is None
            self._flush()
            if was_scene and tag != "body":
                self._add(Block("scene_break", "", frozenset(self.pending_ids)))
                self.pending_ids = set()
            if self.containers and self.containers[-1][0] == tag:
                self.containers.pop()
            self.block_tag = "div"
            self.block_class = ""

    def _link_target(self, href: str, noteref: bool, symbolic: bool) -> tuple[str, bool, bool]:
        if not href or "://" in href or href.startswith(("mailto:", "data:")):
            return ("", noteref, symbolic)
        path_part, _, fragment = href.partition("#")
        target = resolve(self.base, path_part) if path_part else self.path
        return (f"{target}#{fragment}" if fragment else target, noteref, symbolic)

    def handle_data(self, data: str) -> None:
        if self.skip_tag is not None:
            if self.skip_tag == "svg":
                self.skip_parts.append(data)
            return
        if self.table_depth:
            self.table_parts.append(data)
            return
        if data.strip() or self.current is not None:
            self._open().parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def parse_document(path: str, data: bytes) -> Document:
    try:
        markup = data.decode("utf-8").removeprefix("﻿")
        collector = _Collector(path)
        collector.feed(markup)
        collector.close()
    except (UnicodeDecodeError, ValueError, AssertionError) as exc:
        raise UnreadableDocument(f"{path}: {exc}") from exc
    return Document(
        path=path,
        blocks=collector.blocks,
        nav_types=collector.nav_types,
        epub_types=collector.epub_types,
        unsupported=collector.unsupported,
    )


def nav_links(markup: str) -> list[tuple[str, str, str]]:
    """Return (href, title, nav epub:type) for each link inside a nav element."""
    parser = _NavCollector()
    parser.feed(markup)
    parser.close()
    return parser.links


class _NavCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str, str]] = []
        self.nav_type = ""
        self.nav_depth = 0
        self.href: str | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "nav":
            self.nav_depth += 1
            self.nav_type = values.get("epub:type", "")
        elif tag == "a" and self.nav_depth:
            self.href = values.get("href", "")
            self.parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav" and self.nav_depth:
            self.nav_depth -= 1
        elif tag == "a" and self.href is not None:
            title = " ".join("".join(self.parts).split())
            if self.href:
                self.links.append((self.href, title, self.nav_type))
            self.href = None

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.parts.append(data)
