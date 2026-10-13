"""Comparison-only normalisation. It never changes narration text."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

STRESS_MARKS = ("+", "́")
TOKEN = re.compile(r"(?:[^\W_]|[+́])+")
MIN_NUMERIC_PREFIX = 3

ONES = ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
ONES_FEMININE = ("", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")


@dataclass(frozen=True)
class Token:
    """A comparison unit. ``surface`` and the offsets point back into the text it came from."""

    norm: str
    surface: str
    start: int
    end: int
    numeric: bool = False


def _strip_stress(text: str) -> str:
    for mark in STRESS_MARKS:
        text = text.replace(mark, "")
    return text


def fold(text: str) -> str:
    """Lowercase, drop stress marks and fold ё to е."""
    decomposed = unicodedata.normalize("NFD", _strip_stress(text).lower())
    return unicodedata.normalize("NFC", decomposed).replace("ё", "е")


def _below_thousand(number: int, feminine: bool) -> list[str]:
    hundreds, rest = divmod(number, 100)
    words = [HUNDREDS[hundreds]] if hundreds else []
    if 10 <= rest < 20:
        words.append(TEENS[rest - 10])
    else:
        tens, ones = divmod(rest, 10)
        if tens:
            words.append(TENS[tens])
        if ones:
            words.append((ONES_FEMININE if feminine else ONES)[ones])
    return words


def cardinal_words(number: int) -> list[str] | None:
    """Nominative Russian cardinal numeral for 0..999999, or None outside that range."""
    if not 0 <= number <= 999999:
        return None
    if number == 0:
        return ["ноль"]
    thousands, rest = divmod(number, 1000)
    words = [*(_below_thousand(thousands, feminine=True) if thousands > 1 else []), "тысяча"] if thousands else []
    return words + _below_thousand(rest, feminine=False)


def _expand(surface: str, start: int, end: int) -> list[Token]:
    if not surface.isdecimal():
        return [Token(fold(surface), surface, start, end)]
    words = cardinal_words(int(surface))
    if words is None:
        return [Token(surface, surface, start, end, numeric=True)]
    return [Token(word, surface, start, end, numeric=True) for word in words]


def tokenize(text: str) -> list[Token]:
    """Split text on whitespace, punctuation and hyphens; spell out digit tokens."""
    tokens: list[Token] = []
    for match in TOKEN.finditer(text):
        word = _strip_stress(match.group(0))
        if word:
            tokens.extend(_expand(word, match.start(), match.end()))
    return tokens


def normalise(text: str) -> str:
    return " ".join(token.norm for token in tokenize(text))


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def same_word(left: Token, right: Token) -> bool:
    """Equal after normalisation, or a numeral and a word that differ only by case or ordinal endings."""
    if left.norm == right.norm:
        return True
    if not (left.numeric or right.numeric):
        return False
    prefix = len(os.path.commonprefix([left.norm, right.norm]))
    return prefix >= MIN_NUMERIC_PREFIX and prefix >= min(len(left.norm), len(right.norm)) - 2


def near_identical(left: Token, right: Token) -> bool:
    """One-letter spelling noise in a longer word, which ASR produces often."""
    return (
        min(len(left.norm), len(right.norm)) > 5
        and levenshtein(left.norm, right.norm) == 1
    )
