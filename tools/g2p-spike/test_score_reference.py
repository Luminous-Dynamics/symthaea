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
            self.assertEqual(piper["exact_word_pronunciation_accuracy"], 1.0)
            self.assertEqual(muse["exact_word_pronunciation_accuracy"], 1.0)
            self.assertEqual(piper["oov_false_positive_rate"], 1.0)
            self.assertEqual(muse["oov_false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
