#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free regressions for the DiffSinger phoneme contract."""
from __future__ import annotations

import unittest

from diffsinger_phonemes import (
    duration_per_mapped_phone,
    flatten_performance_phones,
    ipa_to_arpabet,
    map_syllable_phones,
    note_timing,
    require_melisma_vowel,
    resolve_language_id,
    timeline_gap_ms,
)


class DiffSingerPhonemeContractTests(unittest.TestCase):
    def test_known_english_phones_have_explicit_mappings(self) -> None:
        self.assertEqual(ipa_to_arpabet("ə"), ["ax"])
        self.assertEqual(ipa_to_arpabet("ɹ"), ["r"])
        self.assertEqual(ipa_to_arpabet("tʃ"), ["ch"])
        self.assertEqual(ipa_to_arpabet("aɪ"), ["ay"])
        self.assertEqual(ipa_to_arpabet("h"), ["hh"])
        self.assertEqual(ipa_to_arpabet("eɪ"), ["ey"])
        self.assertEqual(ipa_to_arpabet("ɜː"), ["er"])

    def test_current_muse_english_g2p_inventory_is_fully_mapped(self) -> None:
        # Symbols emitted by the current speech.rs dictionary and ASCII-letter
        # fallback. Keep this fixture synchronized with that source inventory;
        # an added G2P phone must not silently become an AX or a missing token.
        muse_g2p_inventory = (
            "aɪ", "ð", "ə", "ɪ", "z", "æ", "m", "f", "iː", "l",
            "h", "ɛ", "oʊ", "w", "ɜː", "d", "j", "p", "k", "ɒ",
            "n", "ʃ", "s", "ɡ", "t", "v", "ʌ", "θ", "ŋ", "ɑ",
            "eɪ", "uː", "ɹ", "b", "g", "i", "u", "a", "ɔ", "ʊ",
            "ʒ", "g", "ks",
        )
        for ipa in muse_g2p_inventory:
            with self.subTest(ipa=ipa):
                self.assertTrue(ipa_to_arpabet(ipa), f"missing mapping for {ipa!r}")

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

    def test_note_timing_requires_finite_positive_fields(self) -> None:
        note = {"start_time": 0.25, "duration": 0.5, "frequency": 220.0}
        self.assertEqual(note_timing(note, "note"), (0.25, 0.5, 220.0, 0.75))
        invalid_notes = (
            {"start_time": float("nan"), "duration": 0.5, "frequency": 220.0},
            {"start_time": 0.0, "duration": 0.0, "frequency": 220.0},
            {"start_time": 0.0, "duration": 0.5, "frequency": float("inf")},
            {"start_time": -0.1, "duration": 0.5, "frequency": 220.0},
            {"start_time": 0.0, "duration": 0.5, "frequency": True},
        )
        for invalid in invalid_notes:
            with self.subTest(note=invalid):
                with self.assertRaises(ValueError):
                    note_timing(invalid, "fixture")

    def test_timeline_gaps_are_preserved_and_overlaps_rejected(self) -> None:
        self.assertAlmostEqual(timeline_gap_ms(0.75, 0.5, "note"), 250.0)
        self.assertEqual(timeline_gap_ms(0.5, 0.5, "note"), 0.0)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            timeline_gap_ms(0.49, 0.5, "note")

    def test_syllable_mapping_rejects_missing_natural_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "natural_duration_s"):
            map_syllable_phones(
                [{"ipa": "a", "is_vowel": True}],
                0,
            )

    def test_flattened_score_preserves_silence_gaps_and_vowel_only_melisma(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [
                        {"ipa": "m", "is_vowel": False, "natural_duration_s": 0.1},
                        {"ipa": "a", "is_vowel": True, "natural_duration_s": 0.2},
                        {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.1},
                    ],
                    "note": {"start_time": 0.0, "duration": 0.4, "frequency": 220.0},
                    "melisma_notes": [
                        {"start_time": 0.4, "duration": 0.3, "frequency": 246.94},
                        {"start_time": 0.9, "duration": 0.2, "frequency": 261.63},
                    ],
                },
                {
                    "phonemes": [
                        {"ipa": "k", "is_vowel": False, "natural_duration_s": 0.1},
                        {"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.2},
                    ],
                    "note": {"start_time": 1.2, "duration": 0.5, "frequency": 293.66},
                    "melisma_notes": [],
                },
            ],
        }
        events = flatten_performance_phones(performance, minimum_silence_ms=5.0)
        self.assertEqual([event[0] for event in events], [
            "m", "ah", "t", "ah", "SP", "ah", "SP", "k", "eh"
        ])
        self.assertAlmostEqual(events[4][1], 200.0)
        self.assertAlmostEqual(events[6][1], 100.0)
        self.assertEqual(events[4][2], 261.63)
        self.assertEqual(events[6][2], 293.66)

    def test_flattened_score_rejects_overlap_and_consonant_only_melisma(self) -> None:
        overlapping = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [{"ipa": "a", "is_vowel": True, "natural_duration_s": 0.2}],
                    "note": {"start_time": 0.0, "duration": 0.5, "frequency": 220.0},
                    "melisma_notes": [],
                },
                {
                    "phonemes": [{"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.2}],
                    "note": {"start_time": 0.4, "duration": 0.5, "frequency": 246.94},
                    "melisma_notes": [],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "overlaps"):
            flatten_performance_phones(overlapping, minimum_silence_ms=5.0)

        consonant_only = {
            "language": "en",
            "syllables": [{
                "phonemes": [
                    {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.1},
                    {"ipa": "s", "is_vowel": False, "natural_duration_s": 0.1},
                ],
                "note": {"start_time": 0.0, "duration": 0.2, "frequency": 220.0},
                "melisma_notes": [
                    {"start_time": 0.2, "duration": 0.3, "frequency": 246.94}
                ],
            }],
        }
        with self.assertRaisesRegex(ValueError, "no mapped vowel"):
            flatten_performance_phones(consonant_only, minimum_silence_ms=5.0)

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
