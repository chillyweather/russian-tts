from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ABSOLUTE = re.compile(r"^(/|~/|[A-Za-z]:[\\/]|file://)")
PATH_KEY = re.compile(r"(file|path|dir)$|_file$")


def _strings(value: Any, key: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _strings(item, str(name))
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, key)
    elif isinstance(value, str):
        yield key, value


def _documents(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def find_portability_problems(root: Path) -> list[str]:
    """Report JSON values in the project that are absolute paths, or relative paths leaving the project.

    Model weights are referenced by repository id and revision only, so no model path may appear.
    """
    base = root.resolve()
    problems: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            documents = _documents(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{relative}: unreadable ({exc})")
            continue
        for document in documents:
            for key, value in _strings(document):
                if " " not in value and ABSOLUTE.match(value):
                    if "model" in key.lower():
                        problems.append(f"{relative}: {key} records a local model path; record repo id and revision")
                    else:
                        problems.append(f"{relative}: {key} stores an absolute path")
                elif PATH_KEY.search(key) and value and " " not in value and not ABSOLUTE.match(value):
                    target = (path.parent / value).resolve()
                    if target != base and base not in target.parents:
                        problems.append(f"{relative}: {key} points outside the project")
    return problems
