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


def _append_weighted_group(
    events: list[tuple[str, float, float]],
    phonemes: list[Mapping[str, Any]],
    mapped_groups: list[list[str]],
    indices: range,
    target_duration_ms: float,
    frequency_hz: float,
    context: str,
) -> int | None:
    """Append a phone group with conserved total duration; return its last vowel event."""
    selected = list(indices)
    if not selected:
        return None
    if not math.isfinite(target_duration_ms) or target_duration_ms <= 0.0:
        raise ValueError(f"{context} requires positive finite target duration")

    total_weight_ms = sum(
        float(phonemes[index]["natural_duration_s"]) * 1000.0
        for index in selected
    )
    if not math.isfinite(total_weight_ms) or total_weight_ms <= 0.0:
        raise ValueError(f"{context} has invalid natural phone duration")

    last_vowel_event: int | None = None
    for index in selected:
        phoneme = phonemes[index]
        phone_weight_ms = float(phoneme["natural_duration_s"]) * 1000.0
        source_budget_ms = target_duration_ms * phone_weight_ms / total_weight_ms
        mapped = mapped_groups[index]
        per_token_ms = source_budget_ms / len(mapped)
        if per_token_ms < 1.0:
            raise ValueError(
                f"{context} maps {phoneme['ipa']!r} to {len(mapped)} voicebank tokens, "
                f"but each would receive only {per_token_ms:.6f} ms; refusing to stretch "
                "the score event to the 1 ms minimum"
            )
        for arpabet in mapped:
            event_index = len(events)
            events.append((arpabet, per_token_ms, frequency_hz))
            if phoneme["is_vowel"]:
                last_vowel_event = event_index
    return last_vowel_event


def flatten_performance_phones(
    performance: object,
) -> list[tuple[str, float, float]]:
    """Flatten an English v2 performance into (ARPAbet/SP, duration_ms, Hz).

    The builder preserves absolute note gaps, places onset consonants before
    the vowel-note onset where possible, and borrows a small amount from the
    preceding vowel for legato consonant anticipation. Coda consonants remain
    after the sustained vowel/melisma. Overlap without a preceding vowel to
    borrow is rejected rather than compressed or discarded.
    """
    minimum_vowel_ms = 40.0
    if not isinstance(performance, Mapping):
        raise ValueError("performance must be an object")
    if performance.get("language") != "en":
        raise ValueError(
            f"DiffSinger phone flattening currently supports language='en' only; "
            f"got {performance.get('language')!r}"
        )
    syllables = performance.get("syllables")
    if not isinstance(syllables, list) or not syllables:
        raise ValueError("performance requires a non-empty syllables array")

    events: list[tuple[str, float, float]] = []
    previous_end_s = 0.0
    previous_vowel_event: int | None = None

    for syllable_index, syllable in enumerate(syllables):
        if not isinstance(syllable, Mapping):
            raise ValueError(f"syllable {syllable_index} must be an object")
        phonemes = syllable.get("phonemes")
        if not isinstance(phonemes, list) or not phonemes:
            raise ValueError(f"syllable {syllable_index} requires phonemes")
        mapped_groups, last_vowel = map_syllable_phones(phonemes, syllable_index)
        note_start_s, note_duration_s, note_hz, note_end_s = note_timing(
            syllable.get("note"),
            f"syllable {syllable_index} note",
        )
        melisma_notes = syllable.get("melisma_notes")
        if not isinstance(melisma_notes, list):
            raise ValueError(f"syllable {syllable_index}.melisma_notes must be an array")
        melisma_timing = [
            note_timing(note, f"syllable {syllable_index} melisma note {index}")
            for index, note in enumerate(melisma_notes)
        ]

        requested_advance_s = syllable.get("consonant_advance_s", 0.0)
        if (
            type(requested_advance_s) not in (int, float)
            or not math.isfinite(requested_advance_s)
            or not 0.0 <= requested_advance_s <= 0.15
        ):
            raise ValueError(
                f"syllable {syllable_index}.consonant_advance_s must be finite and in [0, 0.15]"
            )

        vowel_indices = [
            index for index, phone in enumerate(phonemes) if phone["is_vowel"]
        ]
        if vowel_indices:
            first_vowel = vowel_indices[0]
            last_vowel_index = vowel_indices[-1]
            onset_indices = range(0, first_vowel)
            core_indices = range(first_vowel, last_vowel_index + 1)
            coda_indices = range(last_vowel_index + 1, len(phonemes))
        else:
            first_vowel = len(phonemes)
            last_vowel_index = -1
            onset_indices = range(0)
            core_indices = range(0, len(phonemes))
            coda_indices = range(0)

        onset_natural_ms = sum(
            float(phonemes[index]["natural_duration_s"]) * 1000.0
            for index in onset_indices
        )
        advance_ms = min(
            float(requested_advance_s) * 1000.0,
            onset_natural_ms,
            note_start_s * 1000.0,
        )
        attack_start_s = note_start_s - advance_ms / 1000.0

        coda_natural_ms = sum(
            float(phonemes[index]["natural_duration_s"]) * 1000.0
            for index in coda_indices
        )
        has_melisma = bool(melisma_timing)
        coda_limit_ms = (
            melisma_timing[-1][1] * 1000.0
            if has_melisma
            else note_duration_s * 1000.0
        )
        coda_budget_ms = min(
            coda_natural_ms,
            max(coda_limit_ms - minimum_vowel_ms, 0.0),
        )
        if coda_indices and coda_budget_ms <= 0.0:
            raise ValueError(
                f"syllable {syllable_index} has coda consonants but no time remains "
                "for a sustained vowel"
            )

        main_note_ms = note_duration_s * 1000.0
        core_note_ms = main_note_ms if has_melisma else main_note_ms - coda_budget_ms
        if vowel_indices and core_note_ms < minimum_vowel_ms:
            raise ValueError(
                f"syllable {syllable_index} leaves less than {minimum_vowel_ms:.0f} ms "
                "for its vowel nucleus"
            )

        onset_inside_ms = 0.0
        if onset_indices and advance_ms <= 0.0:
            onset_inside_ms = min(
                onset_natural_ms,
                max(core_note_ms - minimum_vowel_ms, 0.0),
            )
            if onset_inside_ms <= 0.0:
                raise ValueError(
                    f"syllable {syllable_index} cannot fit onset consonants and a vowel "
                    "inside the note"
                )
        core_budget_ms = core_note_ms - onset_inside_ms

        # If the requested pre-onset overlaps the previous score event, take
        # the overlap from its last vowel. This yields a sequential phone
        # stream without silently moving either note or duplicating a phone.
        overlap_ms = max((previous_end_s - attack_start_s) * 1000.0, 0.0)
        if overlap_ms > 0.0:
            if advance_ms <= 0.0:
                raise ValueError(
                    f"syllable {syllable_index} overlaps the preceding event without "
                    "a positive consonant_advance_s"
                )
            if previous_vowel_event is None:
                raise ValueError(
                    f"syllable {syllable_index} consonant anticipation overlaps the "
                    "preceding event, but there is no preceding vowel to borrow from"
                )
            symbol, duration_ms, frequency_hz = events[previous_vowel_event]
            if duration_ms - overlap_ms < minimum_vowel_ms:
                raise ValueError(
                    f"syllable {syllable_index} consonant anticipation would shorten "
                    "the preceding vowel below the minimum duration"
                )
            events[previous_vowel_event] = (
                symbol,
                duration_ms - overlap_ms,
                frequency_hz,
            )
            previous_end_s -= overlap_ms / 1000.0

        gap_ms = timeline_gap_ms(
            attack_start_s,
            previous_end_s,
            f"syllable {syllable_index} onset",
        )
        if gap_ms > 0.0:
            events.append(("SP", gap_ms, note_hz))
        # Sub-frame gaps are below the renderer's timing resolution; preserve
        # the event order and accept less than half a frame of onset error.
        previous_end_s = attack_start_s

        if onset_indices:
            actual_onset_ms = advance_ms if advance_ms > 0.0 else onset_inside_ms
            previous_vowel_event = _append_weighted_group(
                events,
                phonemes,
                mapped_groups,
                onset_indices,
                actual_onset_ms,
                note_hz,
                f"syllable {syllable_index} onset",
            )
            previous_end_s += actual_onset_ms / 1000.0
        else:
            previous_vowel_event = None

        core_vowel_event = _append_weighted_group(
            events,
            phonemes,
            mapped_groups,
            core_indices,
            core_budget_ms,
            note_hz,
            f"syllable {syllable_index} nucleus",
        )
        if core_vowel_event is not None:
            previous_vowel_event = core_vowel_event
        previous_end_s += core_budget_ms / 1000.0

        if has_melisma:
            for melisma_index, (
                melisma_start_s,
                melisma_duration_s,
                melisma_hz,
                melisma_end_s,
            ) in enumerate(melisma_timing):
                melisma_gap_ms = timeline_gap_ms(
                    melisma_start_s,
                    previous_end_s,
                    f"syllable {syllable_index} melisma note {melisma_index}",
                )
                if melisma_gap_ms > 0.0:
                    events.append(("SP", melisma_gap_ms, melisma_hz))
                if last_vowel is None:
                    raise ValueError(
                        f"syllable {syllable_index} has melisma notes but no mapped vowel"
                    )
                vowel_duration_ms = melisma_duration_s * 1000.0
                if melisma_index + 1 == len(melisma_timing):
                    vowel_duration_ms -= coda_budget_ms
                if vowel_duration_ms < minimum_vowel_ms:
                    raise ValueError(
                        f"syllable {syllable_index} melisma note {melisma_index} "
                        "leaves insufficient time for the sustained vowel"
                    )
                previous_vowel_event = len(events)
                events.append((last_vowel, vowel_duration_ms, melisma_hz))
                previous_end_s = melisma_end_s
        else:
            previous_end_s = note_end_s

        if coda_indices:
            _append_weighted_group(
                events,
                phonemes,
                mapped_groups,
                coda_indices,
                coda_budget_ms,
                melisma_timing[-1][2] if has_melisma else note_hz,
                f"syllable {syllable_index} coda",
            )
        if coda_indices and has_melisma:
            previous_end_s = melisma_timing[-1][3]

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
