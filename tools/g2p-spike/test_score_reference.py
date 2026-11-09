#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
import csv
import json
import tempfile
import unittest
from pathlib import Path

from score_reference import (
    edit_distance,
    evaluate,
    normalize_ipa,
    split_normalized_piper_words,
    split_piper_words,
)


def word_hypothesis(index: int, raw_ipa: str) -> dict:
    tokens = list(raw_ipa)
    return {
        "word_index": index,
        "raw_ipa": raw_ipa,
        "ipa_character_tokens": tokens,
        "stress_evidence": [],
        "character_prosody": [None] * len(tokens),
        "syllable_boundaries": None,
        "oov_status": "not_assessed",
        "resolution_status": "needs_review" if tokens else "empty_hypothesis",
    }


def normalized_hypothesis(
    words: list[str],
    punctuation: list[dict] | None = None,
    raw_tokens: list[str] | None = None,
) -> str:
    if raw_tokens is None:
        raw_tokens = []
        for index, word in enumerate(words):
            if index:
                raw_tokens.append(" ")
            raw_tokens.extend(list(word))
    return json.dumps(
        {
            "language": "en",
            "provider": "piper-plus-g2p",
            "provider_version": "0.5.3",
            "raw_tokens": raw_tokens,
            "raw_prosody": [None] * len(raw_tokens),
            "punctuation": punctuation or [],
            "words": [word_hypothesis(index, word) for index, word in enumerate(words)],
        },
        ensure_ascii=False,
    )


def write_candidates(path: Path, rows: list[tuple[str, str, list[str], list[str], list[list[str]], list[dict] | None]]) -> None:
    fieldnames = [
        "case_id",
        "text",
        "piper_ipa_tokens",
        "normalized_word_hypotheses_json",
        "muse_word_ipa_tokens",
        "prosody_debug",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for case_id, text, raw_tokens, piper_words, muse_words, punctuation in rows:
            writer.writerow(
                {
                    "case_id": case_id,
                    "text": text,
                    "piper_ipa_tokens": repr(raw_tokens),
                    "normalized_word_hypotheses_json": normalized_hypothesis(
                        piper_words, punctuation, raw_tokens
                    ),
                    "muse_word_ipa_tokens": repr(muse_words),
                    "prosody_debug": "[]",
                }
            )


class G2PReferenceScorerTests(unittest.TestCase):
    def test_codepoint_normalizer_does_not_merge_length_marks_or_affricates(self):
        self.assertEqual(
            normalize_ipa(["ˈ", "o", "ʊ", "ː", "tʃ"]),
            ["o", "ʊ", "ː", "t", "ʃ"],
        )
        self.assertEqual(
            normalize_ipa(["ɜː"]),
            ["ɜ", "ː"],
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

    def test_structured_word_hypotheses_preserve_empty_slots_and_keep_punctuation_separate(self):
        raw_tokens = [
            "“", "h", "ə", "l", "ˈ", "o", "ʊ", ",", " ",
            " ", "w", "ˈ", "ɜ", "ː", "l", "d", "!", "”",
        ]
        hypothesis = normalized_hypothesis(
            ["həloʊ", "", "wɜːld"],
            [
                {"raw_token_index": 0, "word_index": 0, "after_ipa_character_index": 0, "token": "“"},
                {"raw_token_index": 7, "word_index": 0, "after_ipa_character_index": 5, "token": ","},
                {"raw_token_index": 16, "word_index": 2, "after_ipa_character_index": 5, "token": "!"},
                {"raw_token_index": 17, "word_index": 2, "after_ipa_character_index": 5, "token": "”"},
            ],
            raw_tokens,
        )
        words = split_normalized_piper_words(hypothesis)
        self.assertEqual(words, [list("həloʊ"), [], list("wɜːld")])

    def test_structured_word_hypothesis_rejects_noncontiguous_indices(self):
        value = json.loads(normalized_hypothesis(["hello"]))
        value["words"][0]["word_index"] = 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            split_normalized_piper_words(json.dumps(value))

    def test_structured_word_hypothesis_rejects_raw_token_stream_mismatch(self):
        hypothesis = json.loads(normalized_hypothesis(["hello"]))
        hypothesis["raw_tokens"][0] = "j"
        with self.assertRaisesRegex(ValueError, "differs from the TSV source"):
            split_normalized_piper_words(
                json.dumps(hypothesis),
                raw_tokens=["h", "e", "l", "l", "o"],
            )

    def test_structured_word_hypothesis_rejects_punctuation_offset_drift(self):
        raw_tokens = ["“", "h", "ə", "l", "oʊ", ",", " "]
        hypothesis = json.loads(
            normalized_hypothesis(
                ["həloʊ"],
                [
                    {
                        "raw_token_index": 5,
                        "word_index": 0,
                        "after_ipa_character_index": 4,
                        "token": ",",
                    }
                ],
                raw_tokens,
            )
        )
        with self.assertRaisesRegex(ValueError, "punctuation offsets"):
            split_normalized_piper_words(
                json.dumps(hypothesis),
                raw_tokens=raw_tokens,
            )

    def test_structured_word_hypothesis_rejects_character_prosody_mismatch(self):
        value = json.loads(normalized_hypothesis(["hello"]))
        value["words"][0]["character_prosody"].pop()
        with self.assertRaisesRegex(ValueError, "misaligned"):
            split_normalized_piper_words(json.dumps(value))

    def test_edit_distance_counts_unicode_codepoint_edits(self):
        self.assertEqual(edit_distance(list("quick"), list("kwik")), 3)
        self.assertEqual(edit_distance(list("read"), list("read")), 0)

    def test_scoring_requires_adjudication_and_flags_guessed_oov(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            write_candidates(
                candidates,
                [
                    (
                        "ordinary",
                        "read",
                        ["ɹ", "ɛ", "d"],
                        ["ɹɛd"],
                        [["ɹ", "ɛ", "d"]],
                        None,
                    ),
                    (
                        "oov",
                        "xqzpl",
                        ["ɛ", "k", "s", "k"],
                        ["ɛksk"],
                        [["k", "s", "z", "p", "l"]],
                        None,
                    ),
                ],
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "word_index",
                        "accepted_ipa_variants",
                        "reference_status",
                        "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "ordinary",
                        "word_index": 0,
                        "accepted_ipa_variants": "ɹ ɛ d",
                        "reference_status": "reviewed",
                        "reviewer_evidence": "review:001",
                    }
                )
                writer.writerow(
                    {
                        "case_id": "oov",
                        "word_index": 0,
                        "accepted_ipa_variants": "",
                        "reference_status": "out_of_vocabulary",
                        "reviewer_evidence": "review:002",
                    }
                )

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
            write_candidates(
                candidates,
                [("unexpected", "word", ["w"], ["w"], [["w"]], None)],
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "word_index",
                        "accepted_ipa_variants",
                        "reference_status",
                        "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "word",
                        "word_index": 0,
                        "accepted_ipa_variants": "w ɜ d",
                        "reference_status": "reviewed",
                        "reviewer_evidence": "review:003",
                    }
                )
            with self.assertRaisesRegex(ValueError, "candidate/reference case mismatch"):
                evaluate(candidates, references)

    def test_candidate_word_count_mismatch_fails_to_prevent_word_index_shifts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            write_candidates(
                candidates,
                [("phrase", "hello world", ["h", "ɛ", "l", "oʊ"], ["hɛloʊ"], [["h", "ɛ", "l", "oʊ"]], None)],
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "word_index",
                        "accepted_ipa_variants",
                        "reference_status",
                        "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "phrase",
                        "word_index": 0,
                        "accepted_ipa_variants": "h ɛ l oʊ",
                        "reference_status": "reviewed",
                        "reviewer_evidence": "review:004",
                    }
                )
                writer.writerow(
                    {
                        "case_id": "phrase",
                        "word_index": 1,
                        "accepted_ipa_variants": "w ɜː l d",
                        "reference_status": "reviewed",
                        "reviewer_evidence": "review:005",
                    }
                )
            with self.assertRaisesRegex(ValueError, "emitted 1 word slots"):
                evaluate(candidates, references)

    def test_pending_rows_preserve_slots_but_are_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.tsv"
            write_candidates(
                candidates,
                [
                    (
                        "phrase",
                        "hello world",
                        ["h", "ə", "l", "o", "ʊ", " ", "w", "ɜ", "ː", "l", "d"],
                        ["həloʊ", "wɜːld"],
                        [["h", "ə", "l", "oʊ"], ["w", "ɜː", "l", "d"]],
                        None,
                    )
                ],
            )
            references = root / "references.csv"
            with references.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "word_index",
                        "accepted_ipa_variants",
                        "reference_status",
                        "reviewer_evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "phrase",
                        "word_index": 0,
                        "accepted_ipa_variants": "h ə l oʊ",
                        "reference_status": "reviewed",
                        "reviewer_evidence": "review:006",
                    }
                )
                writer.writerow(
                    {
                        "case_id": "phrase",
                        "word_index": 1,
                        "accepted_ipa_variants": "",
                        "reference_status": "pending",
                        "reviewer_evidence": "",
                    }
                )
            result = evaluate(candidates, references)
            self.assertEqual(result["adjudicated_rows"], 1)
            self.assertEqual(result["systems"][0]["evaluated_words"], 1)
            self.assertEqual(result["systems"][0]["exact_normalized_ipa_sequence_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
