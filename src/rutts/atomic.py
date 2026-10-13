from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temporary sibling path; fsync it and move it onto ``target`` only on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{uuid.uuid4().hex}.tmp{target.suffix}"
    try:
        yield temp
        _fsync_file(temp)
        os.replace(temp, target)
        _fsync_directory(target.parent)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_text(target: Path, text: str) -> None:
    with atomic_path(target) as temp:
        temp.write_text(text, encoding="utf-8")


def remove_temp_files(directory: Path) -> int:
    """Delete temporary siblings left behind by an interrupted atomic write."""
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob(".*.tmp*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
