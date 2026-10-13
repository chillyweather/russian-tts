from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from rutts.book.epub.archive import EpubArchive, resolve
from rutts.validation import ValidationError

DC = "{http://purl.org/dc/elements/1.1/}"
OPF = "{http://www.idpf.org/2007/opf}"
NCX = "{http://www.daisy.org/z3986/2005/ncx/}"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"


@dataclass(frozen=True)
class ManifestItem:
    id: str
    path: str
    media_type: str
    properties: frozenset[str]


@dataclass(frozen=True)
class SpineItem:
    index: int
    item: ManifestItem
    linear: bool


@dataclass(frozen=True)
class NavEntry:
    path: str
    fragment: str | None
    title: str


@dataclass
class Package:
    opf_path: str
    title: str
    author: str
    language: str
    spine: list[SpineItem]
    manifest: dict[str, ManifestItem]
    nav_entries: list[NavEntry] = field(default_factory=list)
    nav_document: str | None = None
    guide_types: dict[str, str] = field(default_factory=dict)
    fixed_layout: bool = False


def _parse_xml(data: bytes, label: str) -> ET.Element:
    try:
        # ElementTree never fetches DTDs or external entities; the document is parsed locally only.
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValidationError([f"{label} is not well-formed XML: {exc}"]) from exc


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _container_opf(archive: EpubArchive) -> str:
    if not archive.has("META-INF/container.xml"):
        raise ValidationError(["EPUB container is missing META-INF/container.xml"])
    root = _parse_xml(archive.read("META-INF/container.xml"), "META-INF/container.xml")
    for rootfile in root.iter(f"{CONTAINER}rootfile"):
        if rootfile.get("media-type", "application/oebps-package+xml") == "application/oebps-package+xml":
            path = rootfile.get("full-path")
            if path:
                return posixpath.normpath(path)
    raise ValidationError(["EPUB container does not name a package document (OPF)"])


def check_drm(archive: EpubArchive, content_paths: list[str]) -> None:
    if not archive.has("META-INF/encryption.xml"):
        return
    root = _parse_xml(archive.read("META-INF/encryption.xml"), "META-INF/encryption.xml")
    encrypted = {
        posixpath.normpath(reference.get("URI", ""))
        for reference in root.iter()
        if reference.tag.endswith("CipherReference")
    }
    if any(path in encrypted for path in content_paths):
        raise ValidationError(
            ["EPUB is DRM-protected (its content documents are encrypted); import an unprotected copy"]
        )


def _manifest(opf: ET.Element, base: str) -> dict[str, ManifestItem]:
    manifest: dict[str, ManifestItem] = {}
    manifest_element = opf.find(f"{OPF}manifest")
    if manifest_element is None:
        raise ValidationError(["EPUB package has no manifest"])
    for item in manifest_element.findall(f"{OPF}item"):
        item_id = item.get("id", "")
        href = item.get("href", "")
        if not item_id or not href:
            continue
        manifest[item_id] = ManifestItem(
            id=item_id,
            path=resolve(base, href),
            media_type=item.get("media-type", ""),
            properties=frozenset((item.get("properties") or "").split()),
        )
    return manifest


def _nav_entries(archive: EpubArchive, item: ManifestItem) -> tuple[list[NavEntry], list[str]]:
    from rutts.book.epub.content import nav_links

    entries: list[NavEntry] = []
    types: list[str] = []
    for href, title, nav_type in nav_links(archive.read(item.path).decode("utf-8", "replace")):
        target = resolve(posixpath.dirname(item.path), href)
        fragment = href.split("#", 1)[1] if "#" in href else None
        if target:
            entries.append(NavEntry(path=target, fragment=fragment, title=title))
        types.append(nav_type)
    return entries, types


def _ncx_entries(archive: EpubArchive, path: str) -> list[NavEntry]:
    root = _parse_xml(archive.read(path), path)
    base = posixpath.dirname(path)
    entries: list[NavEntry] = []
    for point in root.iter(f"{NCX}navPoint"):
        label = _text(point.find(f"{NCX}navLabel/{NCX}text"))
        content = point.find(f"{NCX}content")
        src = content.get("src", "") if content is not None else ""
        target = resolve(base, src)
        if target and label:
            fragment = src.split("#", 1)[1] if "#" in src else None
            entries.append(NavEntry(path=target, fragment=fragment, title=label))
    return entries


def read_package(archive: EpubArchive) -> Package:
    opf_path = _container_opf(archive)
    if not archive.has(opf_path):
        raise ValidationError([f"EPUB package document is missing from the archive: {opf_path}"])
    opf = _parse_xml(archive.read(opf_path), opf_path)
    base = posixpath.dirname(opf_path)
    manifest = _manifest(opf, base)
    metadata = opf.find(f"{OPF}metadata")
    title = _text(metadata.find(f"{DC}title")) if metadata is not None else ""
    author = _text(metadata.find(f"{DC}creator")) if metadata is not None else ""
    language = _text(metadata.find(f"{DC}language")) if metadata is not None else ""
    fixed_layout = any(
        (meta.get("property") or "") == "rendition:layout" and _text(meta) == "pre-paginated"
        for meta in (metadata.findall(f"{OPF}meta") if metadata is not None else [])
    )

    spine_element = opf.find(f"{OPF}spine")
    if spine_element is None:
        raise ValidationError(["EPUB package has no spine"])
    spine: list[SpineItem] = []
    for index, ref in enumerate(spine_element.findall(f"{OPF}itemref")):
        item = manifest.get(ref.get("idref", ""))
        if item is None:
            raise ValidationError([f"spine references an unknown manifest item: {ref.get('idref')}"])
        if not archive.has(item.path):
            raise ValidationError([f"spine item is missing from the archive: {item.path}"])
        spine.append(SpineItem(index=index, item=item, linear=ref.get("linear", "yes") != "no"))
    if not spine:
        raise ValidationError(["EPUB spine is empty"])

    nav_entries: list[NavEntry] = []
    nav_document: str | None = None
    nav_item = next((item for item in manifest.values() if "nav" in item.properties), None)
    if nav_item is not None and archive.has(nav_item.path):
        nav_document = nav_item.path
        nav_entries, _ = _nav_entries(archive, nav_item)
    if not nav_entries:
        ncx_id = spine_element.get("toc")
        ncx_item = manifest.get(ncx_id or "") or next(
            (item for item in manifest.values() if item.media_type == "application/x-dtd-ncx"), None
        )
        if ncx_item is not None and archive.has(ncx_item.path):
            nav_entries = _ncx_entries(archive, ncx_item.path)

    guide_types: dict[str, str] = {}
    guide = opf.find(f"{OPF}guide")
    if guide is not None:
        for reference in guide.findall(f"{OPF}reference"):
            href = reference.get("href", "")
            if href:
                guide_types.setdefault(resolve(base, href), reference.get("type", ""))

    return Package(
        opf_path=opf_path,
        title=title,
        author=author,
        language=language,
        spine=spine,
        manifest=manifest,
        nav_entries=nav_entries,
        nav_document=nav_document,
        guide_types=guide_types,
        fixed_layout=fixed_layout,
    )
