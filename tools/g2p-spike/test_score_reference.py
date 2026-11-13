#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
import csv
import tempfile
import unittest
from pathlib import Path

from score_reference import (
    edit_distance,
    evaluate,
    normalize_ipa,
    split_piper_words,
)


class G2PReferenceScorerTests(unittest.TestCase):
    def test_normalizer_flattens_token_boundaries_and_attaches_length(self):
        self.assertEqual(
            normalize_ipa(["ˈ", "o", "ʊ", "ː", "tʃ"]),
            ["o", "ʊː", "t", "ʃ"],
        )

    def test_piper_word_split_uses_explicit_space_tokens(self):
        self.assertEqual(
            split_piper_words(["k", "w", "ɪ", "k", " ", "k", "w", "i", "ː", "n"]),
            [["k", "w", "ɪ", "k"], ["k", "w", "i", "ː", "n"]],
        )

    def test_piper_word_split_preserves_empty_intermediate_word_slots(self):
        self.assertEqual(
            split_piper_words(["h", "ə", " ", " ", "n", "aʊ"]),
            [["h", "ə"], [], ["n", "aʊ"]],
        )

    def test_edit_distance_counts_phone_edits(self):
        self.assertEqual(edit_distance(list("quick"), list("kwik")), 3)
        self.assertEqual(edit_distance(list("read"), list("read")), 0)

    def test_scoring_requires_adjudication_and_flags_guessed_oov(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            candidates.write_text(
                'case_id\ttext\tpiper_ipa_tokens\tmuse_word_ipa_tokens\tprosody_debug\n'
                'ordinary\tread\t["ɹ", "ɛ", "d"]\t[["ɹ", "ɛ", "d"]]\t[]\n'
                'oov\txqzpl\t["ɛ", "k", "s", "k"]\t[["k", "s", "z", "p", "l"]]\t[]\n',
                encoding="utf-8",
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id", "word_index", "accepted_ipa_variants",
                        "reference_status", "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "ordinary", "word_index": 0,
                    "accepted_ipa_variants": "ɹ ɛ d", "reference_status": "reviewed",
                    "reviewer_evidence": "review:001",
                })
                writer.writerow({
                    "case_id": "oov", "word_index": 0,
                    "accepted_ipa_variants": "", "reference_status": "out_of_vocabulary",
                    "reviewer_evidence": "review:002",
                })

            result = evaluate(candidates, references)
            piper, muse = result["systems"]
            self.assertEqual(piper["exact_normalized_ipa_sequence_accuracy"], 1.0)
            self.assertEqual(muse["exact_normalized_ipa_sequence_accuracy"], 1.0)
            self.assertEqual(piper["oov_nonempty_guess_rate"], 1.0)
            self.assertEqual(muse["oov_nonempty_guess_rate"], 1.0)


    def test_candidate_reference_case_mismatch_fails_instead_of_dropping_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            candidates.write_text(
                'case_id\ttext\tpiper_ipa_tokens\tmuse_word_ipa_tokens\tprosody_debug\n'
                'unexpected\tword\t["w"]\t[["w"]]\t[]\n',
                encoding="utf-8",
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id", "word_index", "accepted_ipa_variants",
                        "reference_status", "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "word", "word_index": 0,
                    "accepted_ipa_variants": "w ɜ d", "reference_status": "reviewed",
                    "reviewer_evidence": "review:003",
                })
            with self.assertRaisesRegex(ValueError, "candidate/reference case mismatch"):
                evaluate(candidates, references)

    def test_candidate_word_count_mismatch_fails_to_prevent_word_index_shifts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            candidates.write_text(
                'case_id\ttext\tpiper_ipa_tokens\tmuse_word_ipa_tokens\tprosody_debug\n'
                'phrase\thello world\t["h", "ɛ", "l", "oʊ"]\t[["h", "ɛ", "l", "oʊ"]]\t[]\n',
                encoding="utf-8",
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id", "word_index", "accepted_ipa_variants",
                        "reference_status", "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "phrase", "word_index": 0,
                    "accepted_ipa_variants": "h ɛ l oʊ", "reference_status": "reviewed",
                    "reviewer_evidence": "review:004",
                })
                writer.writerow({
                    "case_id": "phrase", "word_index": 1,
                    "accepted_ipa_variants": "w ɜː l d", "reference_status": "reviewed",
                    "reviewer_evidence": "review:005",
                })
            with self.assertRaisesRegex(ValueError, "emitted 1 word slots"):
                evaluate(candidates, references)

    def test_pending_rows_preserve_slots_but_are_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            candidates.write_text(
                'case_id\ttext\tpiper_ipa_tokens\tmuse_word_ipa_tokens\tprosody_debug\n'
                'phrase\thello world\t["h", "ə", "l", "oʊ", " ", "w", "ɜː", "l", "d"]\t[["h", "ə", "l", "oʊ"], ["w", "ɜː", "l", "d"]]\t[]\n',
                encoding="utf-8",
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id", "word_index", "accepted_ipa_variants",
                        "reference_status", "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "phrase", "word_index": 0,
                    "accepted_ipa_variants": "h ə l oʊ", "reference_status": "reviewed",
                    "reviewer_evidence": "review:006",
                })
                writer.writerow({
                    "case_id": "phrase", "word_index": 1,
                    "accepted_ipa_variants": "", "reference_status": "pending",
                    "reviewer_evidence": "",
                })
            result = evaluate(candidates, references)
            self.assertEqual(result["adjudicated_rows"], 1)
            self.assertEqual(result["systems"][0]["evaluated_words"], 1)
            self.assertEqual(result["systems"][0]["exact_word_pronunciation_accuracy"], 1.0)



if __name__ == "__main__":
    unittest.main()
