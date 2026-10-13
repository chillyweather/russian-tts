from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import unquote

from rutts.validation import ValidationError

ZIP_SIGNATURE = b"PK\x03\x04"
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_MEMBERS = 5000


def is_epub_signature(data: bytes) -> bool:
    return data.startswith(ZIP_SIGNATURE)


@dataclass
class EpubArchive:
    members: dict[str, bytes]

    def has(self, name: str) -> bool:
        return name in self.members

    def read(self, name: str) -> bytes:
        if name not in self.members:
            raise ValidationError([f"EPUB is missing {name}"])
        return self.members[name]


def _safe_name(name: str) -> str:
    if name.startswith(("/", "\\")) or "\\" in name or ":" in name.split("/")[0]:
        raise ValidationError([f"EPUB contains an unsafe member path: {name!r}"])
    if any(part == ".." for part in name.split("/")):
        raise ValidationError([f"EPUB contains a path that escapes the archive: {name!r}"])
    return posixpath.normpath(name)


def open_epub(data: bytes) -> EpubArchive:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValidationError([f"input is not an EPUB (not a zip archive): {exc}"]) from exc
    members: dict[str, bytes] = {}
    total = 0
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValidationError([f"EPUB has too many members ({len(infos)})"])
        for info in infos:
            if info.is_dir():
                continue
            name = _safe_name(info.filename)
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValidationError([f"EPUB member {name} is too large ({info.file_size} bytes)"])
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValidationError(["EPUB is too large when unpacked"])
            members[name] = archive.read(info)
    return EpubArchive(members=members)


def resolve(base_dir: str, href: str) -> str:
    """Resolve a package- or document-relative href to an archive member name (no network access)."""
    path = unquote(href.split("#", 1)[0])
    return posixpath.normpath(posixpath.join(base_dir, path)) if path else ""
