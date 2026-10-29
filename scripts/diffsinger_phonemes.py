# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict English IPA-to-ARPAbet mapping for the local DiffSinger bridge.

This module intentionally has no ONNX/runtime dependency so its input contract
can be tested on any CI runner. Missing phonemes never silently become AX.
"""
from collections.abc import Mapping
import math

IPA_TO_ARPABET: dict[str, list[str]] = {
    "aɪ": ["ay"], "aʊ": ["aw"], "eɪ": ["ey"], "oʊ": ["ow"], "ɔɪ": ["oy"],
    "iː": ["iy"], "uː": ["uw"], "ɑː": ["aa"], "ɔː": ["ao"], "ɜː": ["er"],
    "æ": ["ae"], "ɑ": ["aa"], "ɒ": ["aa"], "ɔ": ["ao"], "ə": ["ax"],
    "ɛ": ["eh"], "ɪ": ["ih"], "ʊ": ["uh"], "ʌ": ["ah"], "i": ["iy"],
    "u": ["uw"], "a": ["ah"],
    "tʃ": ["ch"], "dʒ": ["jh"], "ð": ["dh"], "ŋ": ["ng"], "ɹ": ["r"],
    "ʃ": ["sh"], "ʒ": ["zh"], "θ": ["th"],
    "b": ["b"], "d": ["d"], "f": ["f"], "ɡ": ["g"], "g": ["g"], "h": ["hh"],
    "j": ["y"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"], "p": ["p"],
    "s": ["s"], "t": ["t"], "v": ["v"], "w": ["w"], "z": ["z"],
    "ks": ["k", "s"],
}


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Return an explicit mapping or fail instead of substituting a schwa."""
    try:
        # Copy the list so callers cannot mutate the shared inventory.
        return list(IPA_TO_ARPABET[ipa])
    except KeyError as exc:
        raise ValueError(
            f"unsupported IPA phoneme {ipa!r}; refusing silent schwa substitution"
        ) from exc


def map_syllable_phones(phonemes: object, syllable_index: int) -> tuple[list[list[str]], str | None]:
    """Map each source phone and identify the last vowel token for melisma.

    Fail closed for empty/malformed syllables instead of inventing a schwa or
    allowing malformed vowel metadata to silently change the rendered word.
    """
    if not isinstance(phonemes, list) or not phonemes:
        raise ValueError(
            f"syllable {syllable_index} must contain at least one phoneme; refusing implicit schwa"
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
                f"phoneme {phone_index} in syllable {syllable_index} requires boolean is_vowel"
            )
        mapped = ipa_to_arpabet(ipa)
        mapped_groups.append(mapped)
        if is_vowel:
            last_vowel = mapped[-1]
    return mapped_groups, last_vowel


def require_melisma_vowel(last_vowel: str | None, syllable_index: int) -> str:
    """Return an explicit vowel token to sustain, rejecting consonant-only syllables."""
    if not isinstance(last_vowel, str) or not last_vowel:
        raise ValueError(
            f"syllable {syllable_index} has melisma notes but no mapped vowel; "
            "refusing to sustain a consonant or synthesize a schwa"
        )
    return last_vowel


def duration_per_mapped_phone(duration_ms: float, mapped_phone_count: int) -> float:
    """Distribute one source phone's duration across its mapped output tokens."""
    if (
        type(duration_ms) not in (int, float)
        or not math.isfinite(duration_ms)
        or duration_ms <= 0.0
    ):
        raise ValueError(f"source phone duration must be finite and positive: {duration_ms!r}")
    if type(mapped_phone_count) is not int or mapped_phone_count <= 0:
        raise ValueError(f"mapped phone count must be a positive integer: {mapped_phone_count!r}")
    return duration_ms / mapped_phone_count

def resolve_language_id(language_ids: Mapping[str, int], language: str) -> int:
    """Resolve the only implemented language path, with no default-to-zero."""
    if not isinstance(language, str) or language.strip().lower() != "en":
        raise ValueError(
            f"DiffSinger IPA-to-ARPAbet bridge currently supports language='en' only; got {language!r}"
        )
    try:
        token = language_ids["en"]
    except KeyError as exc:
        raise ValueError("voicebank language map has no explicit 'en' entry") from exc
    if type(token) is not int or token < 0:
        raise ValueError(f"voicebank language ID for 'en' is invalid: {token!r}")
    return token
