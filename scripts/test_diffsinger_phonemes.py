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
            {"ipa": "m", "is_vowel": False, "natural_duration_s": 0.05},
            {"ipa": "a", "is_vowel": True, "natural_duration_s": 0.1},
            {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.05},
        ]
        mapped, last_vowel = map_syllable_phones(phones, 0)
        self.assertEqual(mapped, [["m"], ["ah"], ["t"]])
        self.assertEqual(require_melisma_vowel(last_vowel, 0), "ah")

    def test_consonant_only_syllable_cannot_gain_implicit_schwa_for_melisma(self) -> None:
        _, last_vowel = map_syllable_phones(
            [
                {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.05},
                {"ipa": "s", "is_vowel": False, "natural_duration_s": 0.05},
            ],
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
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], [
            "m", "ah", "ah", "SP", "ah", "t", "SP", "k", "eh"
        ])
        self.assertAlmostEqual(events[3][1], 200.0)
        self.assertAlmostEqual(events[6][1], 100.0)
        self.assertEqual(events[3][2], 261.63)
        self.assertEqual(events[6][2], 293.66)

    def test_non_melismatic_coda_duration_is_conserved_before_next_note(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [
                        {"ipa": "m", "is_vowel": False, "natural_duration_s": 0.1},
                        {"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.3},
                        {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.1},
                    ],
                    "note": {"start_time": 0.0, "duration": 0.5, "frequency": 220.0},
                    "melisma_notes": [],
                },
                {
                    "phonemes": [
                        {"ipa": "k", "is_vowel": False, "natural_duration_s": 0.1},
                        {"ipa": "æ", "is_vowel": True, "natural_duration_s": 0.3},
                    ],
                    "note": {"start_time": 0.5, "duration": 0.5, "frequency": 246.94},
                    "melisma_notes": [],
                },
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["m", "eh", "t", "k", "ae"])
        self.assertAlmostEqual(sum(event[1] for event in events), 1_000.0)
        self.assertNotIn("SP", [event[0] for event in events])

    def test_subframe_score_gaps_are_preserved_before_cumulative_frame_quantization(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [{"ipa": "a", "is_vowel": True, "natural_duration_s": 0.2}],
                    "note": {"start_time": 0.0, "duration": 0.5, "frequency": 220.0},
                    "melisma_notes": [],
                },
                {
                    "phonemes": [{"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.2}],
                    "note": {"start_time": 0.503, "duration": 0.5, "frequency": 246.94},
                    "melisma_notes": [],
                },
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["ah", "SP", "eh"])
        self.assertAlmostEqual(events[1][1], 3.0)

    def test_consonant_anticipation_borrows_from_the_preceding_vowel(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [
                        {"ipa": "m", "is_vowel": False, "natural_duration_s": 0.1},
                        {"ipa": "a", "is_vowel": True, "natural_duration_s": 0.3},
                        {"ipa": "t", "is_vowel": False, "natural_duration_s": 0.1},
                    ],
                    "note": {"start_time": 0.0, "duration": 0.5, "frequency": 220.0},
                    "melisma_notes": [],
                    "consonant_advance_s": 0.0,
                },
                {
                    "phonemes": [
                        {"ipa": "k", "is_vowel": False, "natural_duration_s": 0.05},
                        {"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.3},
                    ],
                    "note": {"start_time": 0.5, "duration": 0.5, "frequency": 246.94},
                    "melisma_notes": [],
                    "consonant_advance_s": 0.05,
                },
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["m", "ah", "t", "k", "eh"])
        # The next /k/ starts at 450 ms, borrowing 50 ms from the first vowel.
        self.assertAlmostEqual(events[1][1], 250.0)
        self.assertAlmostEqual(events[3][1], 50.0)
        self.assertAlmostEqual(sum(event[1] for event in events), 1_000.0)

    def test_consonant_anticipation_requires_a_preceding_vowel_to_borrow(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                {
                    "phonemes": [{"ipa": "m", "is_vowel": False, "natural_duration_s": 0.1}],
                    "note": {"start_time": 0.0, "duration": 0.1, "frequency": 220.0},
                    "melisma_notes": [],
                },
                {
                    "phonemes": [
                        {"ipa": "k", "is_vowel": False, "natural_duration_s": 0.05},
                        {"ipa": "ɛ", "is_vowel": True, "natural_duration_s": 0.3},
                    ],
                    "note": {"start_time": 0.1, "duration": 0.5, "frequency": 246.94},
                    "melisma_notes": [],
                    "consonant_advance_s": 0.05,
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "no preceding vowel"):
            flatten_performance_phones(performance)

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
            flatten_performance_phones(overlapping)

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
            flatten_performance_phones(consonant_only)

    @staticmethod
    def _syllable(start: float, duration: float, phones, melisma=(), advance: float = 0.0):
        return {
            "note": {"start_time": start, "duration": duration, "frequency": 220.0, "velocity": 0.7},
            "phonemes": [
                {
                    "ipa": ipa,
                    "is_vowel": is_vowel,
                    "natural_duration_s": natural_duration,
                }
                for ipa, is_vowel, natural_duration in phones
            ],
            "melisma_notes": [
                {"start_time": note_start, "duration": note_duration, "frequency": frequency, "velocity": 0.7}
                for note_start, note_duration, frequency in melisma
            ],
            "consonant_advance_s": advance,
        }

    @staticmethod
    def _event_total_ms(events):
        return sum(duration_ms for _phone, duration_ms, _frequency in events)

    def test_diff_singer_flattening_conserves_note_and_coda_timeline(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                self._syllable(
                    0.0,
                    0.5,
                    [("l", False, 0.04), ("aɪ", True, 0.14), ("t", False, 0.04)],
                )
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["l", "ay", "t"])
        self.assertAlmostEqual(self._event_total_ms(events), 500.0, places=3)

    def test_diff_singer_absolute_gaps_are_explicit_sp_events(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                self._syllable(0.0, 0.5, [("a", True, 0.12)]),
                self._syllable(0.6, 0.5, [("k", False, 0.04), ("a", True, 0.12)]),
            ],
        }
        events = flatten_performance_phones(performance)
        silence = [duration for phone, duration, _frequency in events if phone == "SP"]
        self.assertEqual(len(silence), 1)
        self.assertAlmostEqual(silence[0], 100.0, places=3)
        self.assertAlmostEqual(self._event_total_ms(events), 1_100.0, places=3)

    def test_diff_singer_consonant_anticipation_borrows_from_prior_vowel(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                self._syllable(0.0, 0.5, [("a", True, 0.12)]),
                self._syllable(
                    0.5,
                    0.5,
                    [("k", False, 0.04), ("a", True, 0.12)],
                    advance=0.02,
                ),
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["a", "k", "ah"])
        self.assertAlmostEqual(events[0][1], 480.0, places=3)
        self.assertAlmostEqual(events[1][1], 20.0, places=3)
        self.assertAlmostEqual(self._event_total_ms(events), 1_000.0, places=3)

    def test_diff_singer_melisma_is_a_sustained_vowel_with_coda_after_final_note(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                self._syllable(
                    0.0,
                    0.5,
                    [("l", False, 0.04), ("aɪ", True, 0.14), ("t", False, 0.04)],
                    melisma=((0.5, 0.5, 246.94),),
                )
            ],
        }
        events = flatten_performance_phones(performance)
        self.assertEqual([event[0] for event in events], ["l", "ay", "ay", "t"])
        self.assertAlmostEqual(events[-1][1], 40.0, places=3)
        self.assertAlmostEqual(self._event_total_ms(events), 1_000.0, places=3)

    def test_diff_singer_rejects_overlaps_without_safe_vowel_borrow(self) -> None:
        performance = {
            "language": "en",
            "syllables": [
                self._syllable(0.0, 0.05, [("a", True, 0.05)]),
                self._syllable(
                    0.05,
                    0.5,
                    [("k", False, 0.04), ("a", True, 0.12)],
                    advance=0.05,
                ),
            ],
        }
        with self.assertRaisesRegex(ValueError, "shorten the preceding vowel"):
            flatten_performance_phones(performance)

    def test_diff_singer_rejects_implicit_gaps_inside_melisma_and_non_english(self) -> None:
        invalid_gap = {
            "language": "en",
            "syllables": [
                self._syllable(
                    0.0,
                    0.5,
                    [("a", True, 0.12)],
                    melisma=((0.55, 0.5, 246.94),),
                )
            ],
        }
        with self.assertRaisesRegex(ValueError, "overlaps|timeline"):
            flatten_performance_phones(invalid_gap)

        invalid_language = {"language": "zu", "syllables": []}
        with self.assertRaisesRegex(ValueError, "supports language='en' only"):
            flatten_performance_phones(invalid_language)

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
