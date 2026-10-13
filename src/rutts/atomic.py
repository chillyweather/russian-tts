from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temporary sibling path; move it onto ``target`` only on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{uuid.uuid4().hex}.tmp{target.suffix}"
    try:
        yield temp
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_text(target: Path, text: str) -> None:
    with atomic_path(target) as temp:
        temp.write_text(text, encoding="utf-8")
