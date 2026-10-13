from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rutts.verification.asr import Word
from rutts.verification.normalize import Token, near_identical, same_word, tokenize


@dataclass(frozen=True)
class HeardToken:
    token: Token
    start: float
    end: float


@dataclass(frozen=True)
class Finding:
    type: str
    severity: str
    expected: str
    heard: str
    char_start: int
    char_end: int
    audio_start: float
    audio_end: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "expected": self.expected,
            "heard": self.heard,
            "location": {
                "chars": [self.char_start, self.char_end],
                "audio_seconds": [self.audio_start, self.audio_end],
            },
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Comparison:
    findings: list[Finding]
    expected_tokens: list[Token]
    heard_tokens: list[HeardToken]
    matched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalised_expected": [token.norm for token in self.expected_tokens],
            "normalised_heard": [item.token.norm for item in self.heard_tokens],
            "matched_tokens": self.matched,
        }


@dataclass(frozen=True)
class _Op:
    kind: str  # match, sub, del (expected only) or ins (heard only)
    expected_at: int  # expected tokens consumed before this operation
    heard_at: int  # heard tokens consumed before this operation


@dataclass(frozen=True)
class _Block:
    expected: list[int]
    heard: list[int]
    expected_at: int
    heard_at: int


def _heard_tokens(words: list[Word]) -> list[HeardToken]:
    return [HeardToken(token=token, start=word.start, end=word.end) for word in words for token in tokenize(word.text)]


def _step_cost(expected: Token, heard: Token) -> int:
    return 0 if same_word(expected, heard) else 1


def _operations(expected: list[Token], heard: list[HeardToken]) -> list[_Op]:
    n, m = len(expected), len(heard)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = min(
                table[i - 1][j - 1] + _step_cost(expected[i - 1], heard[j - 1].token),
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
            )
    ops: list[_Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = _step_cost(expected[i - 1], heard[j - 1].token)
            if table[i][j] == table[i - 1][j - 1] + cost:
                ops.append(_Op("match" if cost == 0 else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append(_Op("del", i - 1, j))
            i -= 1
        else:
            ops.append(_Op("ins", i, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _blocks(ops: list[_Op]) -> list[_Block]:
    """Merge consecutive non-matching operations into one edit."""
    blocks: list[_Block] = []
    current: list[_Op] = []
    for op in [*ops, _Op("match", 0, 0)]:
        if op.kind == "match":
            if current:
                blocks.append(
                    _Block(
                        expected=[item.expected_at for item in current if item.kind in ("sub", "del")],
                        heard=[item.heard_at for item in current if item.kind in ("sub", "ins")],
                        expected_at=current[0].expected_at,
                        heard_at=current[0].heard_at,
                    )
                )
                current = []
            continue
        current.append(op)
    return blocks


def _repeated_run(expected: list[Token], heard_norms: list[str], at: int) -> tuple[int, int] | None:
    """The expected run that an inserted run duplicates, immediately before or after the gap at ``at``."""
    length = len(heard_norms)
    before = at - length
    if before >= 0 and [token.norm for token in expected[before:at]] == heard_norms:
        return before, at
    after = at + length
    if after <= len(expected) and [token.norm for token in expected[at:after]] == heard_norms:
        return at, after
    return None


def _audio_span(heard: list[HeardToken], block: _Block) -> tuple[float, float]:
    if block.heard:
        return heard[block.heard[0]].start, heard[block.heard[-1]].end
    if not heard:
        return 0.0, 0.0
    point = heard[block.heard_at].start if block.heard_at < len(heard) else heard[-1].end
    return point, point


def _classify(text: str, expected: list[Token], heard: list[HeardToken], block: _Block) -> Finding:
    heard_tokens = [heard[index] for index in block.heard]
    heard_text = " ".join(item.token.surface for item in heard_tokens)
    audio_start, audio_end = _audio_span(heard, block)
    evidence: dict[str, Any] = {
        "expected_normalised": [expected[index].norm for index in block.expected],
        "heard_normalised": [item.token.norm for item in heard_tokens],
    }

    if not block.expected:
        repeated = _repeated_run(expected, evidence["heard_normalised"], block.expected_at)
        if repeated is not None:
            first, last = repeated
            char_start, char_end = expected[first].start, expected[last - 1].end
            return Finding(
                "repetition", "major", text[char_start:char_end], heard_text,
                char_start, char_end, audio_start, audio_end, evidence,
            )
        if block.expected_at < len(expected):
            char_start = char_end = expected[block.expected_at].start
        else:
            char_start = char_end = expected[-1].end if expected else 0
        return Finding("addition", "major", "", heard_text, char_start, char_end, audio_start, audio_end, evidence)

    char_start = expected[block.expected[0]].start
    char_end = expected[block.expected[-1]].end
    expected_text = text[char_start:char_end]
    if any(expected[index].numeric for index in block.expected) or any(item.token.numeric for item in heard_tokens):
        kind, severity = "uncertain_number", "low"
    elif not block.heard:
        kind, severity = "omission", "major"
    elif len(block.heard) == len(block.expected) and all(
        near_identical(expected[i], heard[j].token) for i, j in zip(block.expected, block.heard)
    ):
        kind, severity = "substitution", "minor"
    else:
        kind, severity = "substitution", "major"
    return Finding(kind, severity, expected_text, heard_text, char_start, char_end, audio_start, audio_end, evidence)


def compare(expected_text: str, words: list[Word]) -> Comparison:
    """Align normalised expected tokens with timed heard words and report the differences."""
    expected = tokenize(expected_text)
    heard = _heard_tokens(words)
    ops = _operations(expected, heard)
    findings = [_classify(expected_text, expected, heard, block) for block in _blocks(ops)]
    matched = sum(1 for op in ops if op.kind == "match")
    return Comparison(findings=findings, expected_tokens=expected, heard_tokens=heard, matched=matched)
