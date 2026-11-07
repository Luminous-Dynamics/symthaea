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
