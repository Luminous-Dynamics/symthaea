#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free regressions for the DiffSinger phoneme contract."""
from __future__ import annotations

import unittest

from diffsinger_phonemes import (
    duration_per_mapped_phone,
    ipa_to_arpabet,
    map_syllable_phones,
    require_melisma_vowel,
    resolve_language_id,
)


class DiffSingerPhonemeContractTests(unittest.TestCase):
    def test_known_english_phones_have_explicit_mappings(self) -> None:
        self.assertEqual(ipa_to_arpabet("ə"), ["ax"])
        self.assertEqual(ipa_to_arpabet("ɹ"), ["r"])
        self.assertEqual(ipa_to_arpabet("tʃ"), ["ch"])
        self.assertEqual(ipa_to_arpabet("aɪ"), ["ay"])

    def test_composite_phone_preserves_order_and_divides_duration(self) -> None:
        self.assertEqual(ipa_to_arpabet("ks"), ["k", "s"])
        self.assertEqual(duration_per_mapped_phone(80.0, 2), 40.0)

    def test_unknown_ipa_and_clicks_are_rejected_not_mapped_to_schwa(self) -> None:
        for ipa in ("ǃ", "ʘ", "ǀ", "ǂ", "ǁ", "ɬ", "ɣ", "not-a-phone"):
            with self.subTest(ipa=ipa):
                with self.assertRaisesRegex(ValueError, "unsupported IPA"):
                    ipa_to_arpabet(ipa)

    def test_empty_and_malformed_phone_inputs_are_rejected(self) -> None:
        for value in ("", " ", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ipa_to_arpabet(value)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "at least one phoneme"):
            map_syllable_phones([], 2)
        with self.assertRaisesRegex(ValueError, "must be an object"):
            map_syllable_phones(["a"], 2)
        with self.assertRaisesRegex(ValueError, "requires boolean is_vowel"):
            map_syllable_phones([{"ipa": "a", "is_vowel": 1}], 2)

    def test_melisma_sustains_last_vowel_not_final_consonant(self) -> None:
        phones = [
            {"ipa": "m", "is_vowel": False},
            {"ipa": "a", "is_vowel": True},
            {"ipa": "t", "is_vowel": False},
        ]
        mapped, last_vowel = map_syllable_phones(phones, 0)
        self.assertEqual(mapped, [["m"], ["ah"], ["t"]])
        self.assertEqual(require_melisma_vowel(last_vowel, 0), "ah")

    def test_consonant_only_syllable_cannot_gain_implicit_schwa_for_melisma(self) -> None:
        _, last_vowel = map_syllable_phones(
            [{"ipa": "t", "is_vowel": False}, {"ipa": "s", "is_vowel": False}],
            3,
        )
        with self.assertRaisesRegex(ValueError, "no mapped vowel"):
            require_melisma_vowel(last_vowel, 3)

    def test_duration_and_language_contracts_reject_invalid_values(self) -> None:
        for duration, count in ((0.0, 1), (float("nan"), 1), (40.0, 0)):
            with self.subTest(duration=duration, count=count):
                with self.assertRaises(ValueError):
                    duration_per_mapped_phone(duration, count)
        self.assertEqual(resolve_language_id({"en": 1}, "en"), 1)
        with self.assertRaisesRegex(ValueError, "no entry"):
            resolve_language_id({}, "en")
        with self.assertRaisesRegex(ValueError, "invalid"):
            resolve_language_id({"en": True}, "en")


if __name__ == "__main__":
    unittest.main()
