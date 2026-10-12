#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-light regressions for DiffSinger pronunciation/language handling."""
import unittest

from diffsinger_phonemes import (
    duration_per_mapped_phone,
    ipa_to_arpabet,
    map_syllable_phones,
    require_melisma_vowel,
    resolve_language_id,
)


class DiffSingerPhonemeContractTests(unittest.TestCase):
    def test_common_english_phones_have_explicit_mappings(self):
        self.assertEqual(ipa_to_arpabet("aɪ"), ["ay"])
        self.assertEqual(ipa_to_arpabet("tʃ"), ["ch"])
        self.assertEqual(ipa_to_arpabet("ks"), ["k", "s"])
        self.assertEqual(ipa_to_arpabet("ə"), ["ax"])


    def test_known_symthaea_g2p_inventory_has_explicit_diff_singer_mappings(self):
        # Includes dictionary phones and the letter-fallback outputs (notably
        # x -> ks). Rust tests query the G2P implementation directly.
        source_phones = (
            "aɪ", "eɪ", "æ", "b", "tʃ", "k", "d", "ɛ", "f", "ɡ",
            "h", "ɪ", "dʒ", "l", "m", "ŋ", "uː", "ɒ", "p", "ks",
            "ɹ", "ʃ", "θ", "ʌ", "v", "w", "z", "ə", "ð", "iː",
            "oʊ", "ɜː", "ɑ", "n", "s", "t", "j",
        )
        for ipa in source_phones:
            with self.subTest(ipa=ipa):
                self.assertTrue(ipa_to_arpabet(ipa), f"no DiffSinger mapping for {ipa!r}")


    def test_composite_phone_duration_is_shared_across_output_tokens(self):
        mapped = ipa_to_arpabet("ks")
        self.assertEqual(mapped, ["k", "s"])
        per_token_ms = duration_per_mapped_phone(300.0, len(mapped))
        self.assertEqual(per_token_ms, 150.0)
        self.assertEqual(per_token_ms * len(mapped), 300.0)

    def test_melisma_holds_last_vowel_not_final_consonant(self):
        groups, last_vowel = map_syllable_phones(
            [
                {"ipa": "l", "is_vowel": False},
                {"ipa": "aɪ", "is_vowel": True},
                {"ipa": "t", "is_vowel": False},
            ],
            4,
        )
        self.assertEqual(groups, [["l"], ["ay"], ["t"]])
        self.assertEqual(require_melisma_vowel(last_vowel, 4), "ay")

    def test_empty_or_consonant_only_syllable_never_invents_schwa_for_melisma(self):
        with self.assertRaisesRegex(ValueError, "must contain at least one phoneme"):
            map_syllable_phones([], 2)

        groups, last_vowel = map_syllable_phones(
            [{"ipa": "ks", "is_vowel": False}],
            2,
        )
        self.assertEqual(groups, [["k", "s"]])
        with self.assertRaisesRegex(ValueError, "no mapped vowel"):
            require_melisma_vowel(last_vowel, 2)

    def test_syllable_phone_contract_requires_boolean_vowel_metadata(self):
        with self.assertRaisesRegex(ValueError, "requires boolean is_vowel"):
            map_syllable_phones([{"ipa": "aɪ", "is_vowel": "true"}], 0)

    def test_non_finite_or_non_positive_phone_duration_is_rejected(self):
        for duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    duration_per_mapped_phone(duration, 2)

    def test_click_consonants_are_rejected_until_a_click_specific_phone_inventory_exists(self):
        # Clicks are non-pulmonic articulations. They must not be approximated
        # by ordinary ARPAbet consonants without an explicit acoustic model.
        for click in ("ʘ", "ǀ", "ǃ", "ǂ", "ǁ"):
            with self.subTest(click=click):
                with self.assertRaisesRegex(ValueError, "unsupported IPA phoneme"):
                    ipa_to_arpabet(click)

    def test_unknown_ipa_is_rejected_not_replaced_with_schwa(self):
        with self.assertRaisesRegex(ValueError, "unsupported IPA phoneme.*ʎ"):
            ipa_to_arpabet("ʎ")

    def test_only_explicit_english_language_id_is_accepted(self):
        self.assertEqual(resolve_language_id({"en": 7}, "en"), 7)
        with self.assertRaisesRegex(ValueError, "supports language='en' only"):
            resolve_language_id({"en": 7, "fr": 2}, "fr")

    def test_missing_or_invalid_english_language_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no explicit 'en' entry"):
            resolve_language_id({"fr": 2}, "en")
        with self.assertRaisesRegex(ValueError, "language ID.*invalid"):
            resolve_language_id({"en": -1}, "en")


if __name__ == "__main__":
    unittest.main()
