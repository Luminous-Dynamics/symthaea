# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict English IPA-to-ARPAbet mapping for the local DiffSinger bridge.

This module intentionally has no ONNX/runtime dependency so its input contract
can be tested on any CI runner. Missing phonemes never silently become AX.
"""
from collections.abc import Mapping

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
