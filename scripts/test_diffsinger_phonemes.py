#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-light regressions for DiffSinger pronunciation/language handling."""
import unittest

from diffsinger_phonemes import ipa_to_arpabet, resolve_language_id


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
