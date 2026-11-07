# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-light IPA contract for the local DiffSinger worker.

This module intentionally imports no ONNX, NumPy, or model runtime. Unknown
phones and missing language IDs are errors; they never fall back to schwa or
language ID zero. Composite source phones preserve order and divide their
natural duration across mapped voicebank phones.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

IPA_TO_ARPABET: dict[str, tuple[str, ...]] = {
    "aɪ": ("ay",), "aʊ": ("aw",), "eɪ": ("ey",), "oʊ": ("ow",), "ɔɪ": ("oy",),
    "iː": ("iy",), "uː": ("uw",), "ɑː": ("aa",), "ɔː": ("ao",), "ɜː": ("er",),
    "æ": ("ae",), "ɑ": ("aa",), "ɒ": ("aa",), "ɔ": ("ao",), "ə": ("ax",),
    "ɛ": ("eh",), "ɪ": ("ih",), "ʊ": ("uh",), "ʌ": ("ah",), "i": ("iy",),
    "u": ("uw",), "a": ("ah",),
    "tʃ": ("ch",), "dʒ": ("jh",), "ð": ("dh",), "ŋ": ("ng",), "ɹ": ("r",),
    "ʃ": ("sh",), "ʒ": ("zh",), "θ": ("th",),
    "b": ("b",), "d": ("d",), "f": ("f",), "ɡ": ("g",), "g": ("g",), "h": ("hh",),
    "j": ("y",), "k": ("k",), "l": ("l",), "m": ("m",), "n": ("n",), "p": ("p",),
    "s": ("s",), "t": ("t",), "v": ("v",), "w": ("w",), "z": ("z",),
    "ks": ("k", "s"),
}


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Return the explicit ordered mapping or reject an unsupported phone."""
    if not isinstance(ipa, str) or not ipa.strip():
        raise ValueError(f"IPA phoneme must be a non-empty string: {ipa!r}")
    try:
        return list(IPA_TO_ARPABET[ipa])
    except KeyError as exc:
        raise ValueError(
            f"unsupported IPA phoneme {ipa!r}; refusing silent schwa substitution"
        ) from exc


def map_syllable_phones(
    phonemes: object,
    syllable_index: int,
) -> tuple[list[list[str]], str | None]:
    """Map source phones and return the final explicitly marked vowel token."""
    if not isinstance(phonemes, list) or not phonemes:
        raise ValueError(
            f"syllable {syllable_index} must contain at least one phoneme; "
            "refusing implicit schwa"
        )

    mapped_groups: list[list[str]] = []
    last_vowel: str | None = None
    for phone_index, phoneme in enumerate(phonemes):
        if not isinstance(phoneme, Mapping):
            raise ValueError(
                f"phoneme {phone_index} in syllable {syllable_index} must be an object"
            )
        ipa = phoneme.get("ipa")
        if not isinstance(ipa, str) or not ipa.strip():
            raise ValueError(
                f"phoneme {phone_index} in syllable {syllable_index} requires non-empty IPA"
            )
        is_vowel = phoneme.get("is_vowel")
        if type(is_vowel) is not bool:
            raise ValueError(
                f"phoneme {phone_index} in syllable {syllable_index} "
                "requires boolean is_vowel"
            )
        natural_duration_s = phoneme.get("natural_duration_s")
        if (
            type(natural_duration_s) not in (int, float)
            or not math.isfinite(natural_duration_s)
            or natural_duration_s <= 0.0
        ):
            raise ValueError(
                f"phoneme {phone_index} in syllable {syllable_index} "
                "requires finite positive natural_duration_s"
            )
        mapped = ipa_to_arpabet(ipa)
        mapped_groups.append(mapped)
        if is_vowel:
            last_vowel = mapped[-1]
    return mapped_groups, last_vowel


def require_melisma_vowel(last_vowel: str | None, syllable_index: int) -> str:
    """Return a vowel for melisma, never a final consonant or invented schwa."""
    if not isinstance(last_vowel, str) or not last_vowel:
        raise ValueError(
            f"syllable {syllable_index} has melisma notes but no mapped vowel; "
            "refusing to sustain a consonant or synthesize a schwa"
        )
    return last_vowel


def duration_per_mapped_phone(duration_ms: float, mapped_phone_count: int) -> float:
    """Divide one source phone's time among composite voicebank phone tokens."""
    if (
        type(duration_ms) not in (int, float)
        or not math.isfinite(duration_ms)
        or duration_ms <= 0.0
    ):
        raise ValueError(f"source phone duration must be finite and positive: {duration_ms!r}")
    if type(mapped_phone_count) is not int or mapped_phone_count <= 0:
        raise ValueError(
            f"mapped phone count must be a positive integer: {mapped_phone_count!r}"
        )
    return duration_ms / mapped_phone_count


def note_timing(note: object, context: str) -> tuple[float, float, float, float]:
    """Validate a score note and return (start_s, duration_s, frequency_hz, end_s)."""
    if not isinstance(note, Mapping):
        raise ValueError(f"{context} must be an object")
    values: dict[str, float] = {}
    for field in ("start_time", "duration", "frequency"):
        value = note.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{context}.{field} must be a finite number")
        values[field] = float(value)
    start = values["start_time"]
    duration = values["duration"]
    frequency = values["frequency"]
    if start < 0.0 or duration <= 0.0 or frequency <= 0.0:
        raise ValueError(f"{context} requires non-negative start and positive duration/frequency")
    end = start + duration
    if not math.isfinite(end) or end <= start:
        raise ValueError(f"{context} has an invalid end time")
    return start, duration, frequency, end


def timeline_gap_ms(start_s: float, previous_end_s: float, context: str) -> float:
    """Return silence between adjacent events; reject overlaps instead of compressing them."""
    if (
        type(start_s) not in (int, float)
        or type(previous_end_s) not in (int, float)
        or not math.isfinite(start_s)
        or not math.isfinite(previous_end_s)
        or start_s < 0.0
        or previous_end_s < 0.0
    ):
        raise ValueError(f"{context} timeline values must be finite and non-negative")
    gap_s = float(start_s) - float(previous_end_s)
    if gap_s < -1e-6:
        raise ValueError(f"{context} overlaps the preceding score event by {-gap_s * 1000.0:.3f} ms")
    return max(gap_s, 0.0) * 1000.0


def flatten_performance_phones(
    performance: object,
    minimum_silence_ms: float,
) -> list[tuple[str, float, float]]:
    """Flatten the validated English v2 score into (ARPAbet/SP, duration_ms, Hz).

    Absolute note start times are retained as explicit SP events. The minimum
    silence threshold is supplied by the renderer's frame resolution.
    """
    if not isinstance(performance, Mapping):
        raise ValueError("performance must be an object")
    if performance.get("language") != "en":
        raise ValueError(
            f"DiffSinger phone flattening currently supports language='en' only; "
            f"got {performance.get('language')!r}"
        )
    if (
        type(minimum_silence_ms) not in (int, float)
        or not math.isfinite(minimum_silence_ms)
        or minimum_silence_ms < 0.0
    ):
        raise ValueError("minimum_silence_ms must be finite and non-negative")
    syllables = performance.get("syllables")
    if not isinstance(syllables, list) or not syllables:
        raise ValueError("performance requires a non-empty syllables array")

    events: list[tuple[str, float, float]] = []
    previous_end_s = 0.0
    for syllable_index, syllable in enumerate(syllables):
        if not isinstance(syllable, Mapping):
            raise ValueError(f"syllable {syllable_index} must be an object")
        phonemes = syllable.get("phonemes")
        mapped_groups, last_vowel = map_syllable_phones(phonemes, syllable_index)
        note_start_s, note_duration_s, note_hz, note_end_s = note_timing(
            syllable.get("note"),
            f"syllable {syllable_index} note",
        )
        gap_ms = timeline_gap_ms(
            note_start_s,
            previous_end_s,
            f"syllable {syllable_index} onset",
        )
        if gap_ms >= minimum_silence_ms and gap_ms > 0.0:
            events.append(("SP", gap_ms, note_hz))

        total_natural_ms = sum(
            float(phone["natural_duration_s"]) * 1000.0 for phone in phonemes
        )
        if not math.isfinite(total_natural_ms) or total_natural_ms <= 0.0:
            raise ValueError(f"syllable {syllable_index} has invalid natural phone duration")
        note_ms = max(note_duration_s * 1000.0, 1.0)
        for phoneme, mapped_phones in zip(phonemes, mapped_groups, strict=True):
            per_token_ms = duration_per_mapped_phone(
                float(phoneme["natural_duration_s"]) * 1000.0,
                len(mapped_phones),
            )
            for arpabet in mapped_phones:
                share = per_token_ms / total_natural_ms
                events.append((arpabet, max(share * note_ms, 1.0), note_hz))

        previous_end_s = note_end_s
        melisma_notes = syllable.get("melisma_notes")
        if not isinstance(melisma_notes, list):
            raise ValueError(f"syllable {syllable_index}.melisma_notes must be an array")
        for melisma_index, melisma_note in enumerate(melisma_notes):
            melisma_start_s, melisma_duration_s, melisma_hz, melisma_end_s = note_timing(
                melisma_note,
                f"syllable {syllable_index} melisma note {melisma_index}",
            )
            melisma_gap_ms = timeline_gap_ms(
                melisma_start_s,
                previous_end_s,
                f"syllable {syllable_index} melisma note {melisma_index}",
            )
            if melisma_gap_ms >= minimum_silence_ms and melisma_gap_ms > 0.0:
                events.append(("SP", melisma_gap_ms, melisma_hz))
            events.append(
                (
                    require_melisma_vowel(last_vowel, syllable_index),
                    max(melisma_duration_s * 1000.0, 1.0),
                    melisma_hz,
                )
            )
            previous_end_s = melisma_end_s
    return events


def resolve_language_id(language_ids: Mapping[str, Any], language: str) -> int:
    """Require an explicit valid language entry; never default to language zero."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"language must be an explicit non-empty tag: {language!r}")
    try:
        token = language_ids[language]
    except KeyError as exc:
        raise ValueError(f"voicebank language map has no entry for {language!r}") from exc
    if type(token) is not int or token < 0:
        raise ValueError(f"voicebank language ID for {language!r} is invalid: {token!r}")
    return token
