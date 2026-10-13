from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from rutts.engines.base import SynthesisResult

UNRESOLVED_REVISION = "unresolved"
_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def resolve_revision(repo: str) -> str:
    # mlx-audio downloads only the files it needs, so a "complete snapshot" check would fail;
    # the commit is the snapshot directory that holds the cached config.
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(repo, "config.json")
    if not isinstance(cached, str):
        raise RuntimeError(f"{repo} is not in the local Hugging Face cache")
    commit = cached.rstrip("/").split("/")[-2]
    if not _COMMIT_HASH.match(commit):
        raise RuntimeError(f"could not resolve a commit hash for {repo}: {cached}")
    return commit


def cached_revision(repo: str) -> str:
    """Revision of the locally cached snapshot, known without loading the model."""
    try:
        return resolve_revision(repo)
    except Exception:
        return UNRESOLVED_REVISION


def concat_segments(segments: Sequence[Any]) -> SynthesisResult:
    if not segments:
        raise RuntimeError("engine returned no audio segments")
    rates = {int(segment.sample_rate) for segment in segments}
    if len(rates) != 1:
        raise RuntimeError(f"engine returned mixed sample rates: {sorted(rates)}")
    samples = np.concatenate(
        [np.asarray(segment.audio, dtype=np.float32).reshape(-1) for segment in segments]
    )
    return SynthesisResult(samples=samples, sample_rate=rates.pop())


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()
