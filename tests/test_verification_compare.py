from __future__ import annotations

from rutts.verification.asr import Word
from rutts.verification.compare import compare
from rutts.verification.normalize import normalise


def words(text: str, step: float = 0.5) -> list[Word]:
    """Evenly timed words from whitespace-separated text."""
    return [Word(text=item, start=round(index * step, 3), end=round(index * step + step, 3)) for index, item in enumerate(text.split())]


def only(expected: str, heard: str):  # type: ignore[no-untyped-def]
    result = compare(expected, words(heard))
    return result.findings


def test_clean_reading_has_no_findings() -> None:
    assert only("Утро началось рано.", "утро началось рано") == []


def test_omission_reports_expected_words_and_span() -> None:
    expected = "Утро началось рано. Анна открыла окно."
    findings = only(expected, "Утро началось. Анна открыла окно.")
    assert len(findings) == 1
    finding = findings[0]
    assert (finding.type, finding.severity) == ("omission", "major")
    assert finding.expected == "рано"
    assert finding.heard == ""
    assert expected[finding.char_start : finding.char_end] == "рано"


def test_addition_reports_heard_word_and_time() -> None:
    findings = compare("Утро началось рано.", words("Утро внезапно началось рано")).findings
    assert [(f.type, f.heard, f.expected) for f in findings] == [("addition", "внезапно", "")]
    assert (findings[0].audio_start, findings[0].audio_end) == (0.5, 1.0)


def test_substitution_reports_both_sides() -> None:
    expected = "Анна открыла окно."
    findings = only(expected, "Анна открыла дверь.")
    assert [(f.type, f.severity, f.expected, f.heard) for f in findings] == [
        ("substitution", "major", "окно", "дверь")
    ]
    finding = findings[0]
    assert expected[finding.char_start : finding.char_end] == "окно"
    assert (finding.audio_start, finding.audio_end) == (1.0, 1.5)


def test_repeated_phrase_is_a_repetition() -> None:
    expected = "Он вошёл в дом."
    findings = only(expected, "Он вошёл в в дом")
    assert [(f.type, f.expected, f.heard) for f in findings] == [("repetition", "в", "в")]
    finding = findings[0]
    assert expected[finding.char_start : finding.char_end] == "в"


def test_repeated_multiword_phrase_is_a_repetition() -> None:
    findings = only("Иди прямо домой.", "Иди прямо прямо домой")
    assert [f.type for f in findings] == ["repetition"]
    assert findings[0].heard == "прямо"


def test_yo_and_punctuation_differences_are_not_findings() -> None:
    assert only("Всё ещё спало, её мать.", "все еще спало ее мать") == []


def test_hyphen_and_stress_marks_are_not_findings() -> None:
    assert only("Что-то зам+ок.", "что то замок") == []
    assert only("Старый зам́ок.", "старый замок") == []


def test_stress_hint_in_chunk_text_produces_no_finding() -> None:
    assert only("Старый зам+ок стоял.", "Старый замок стоял.") == []


def test_spelled_numeral_matches_digits() -> None:
    assert only("в 1812 году", "в тысяча восемьсот двенадцатом году") == []


def test_spelled_numeral_with_case_ending_matches_digits() -> None:
    assert only("25 рублей", "двадцать пять рублей") == []


def test_numeral_mismatch_is_uncertain_not_substitution() -> None:
    findings = only("В 1812 году", "В тысяча восемьсот пятом году")
    assert [(f.type, f.severity) for f in findings] == [("uncertain_number", "low")]


def test_near_identical_spelling_is_minor_substitution() -> None:
    findings = only("Старинная библиотека.", "Старинная библиатека.")
    assert [(f.type, f.severity) for f in findings] == [("substitution", "minor")]


def test_adjacent_edits_merge_into_one_finding() -> None:
    findings = only("Один два три четыре.", "Один пять шесть четыре.")
    assert [(f.type, f.expected, f.heard) for f in findings] == [("substitution", "два три", "пять шесть")]


def test_normalise_drops_punctuation_and_folds() -> None:
    assert normalise("Ёж, «что-то» +зам+ок!") == "еж что то замок"
