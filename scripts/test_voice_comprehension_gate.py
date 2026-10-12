#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free tests for the blinded spoken comprehension pack."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from voice_comprehension_gate import (
    DEFAULT_CASES,
    build_documents,
    load_case_bank,
    prepare,
    score_responses,
)


class VoiceComprehensionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_case_bank(DEFAULT_CASES)
        cls.public, cls.key, cls.render = build_documents(cls.bank)
        cls.correct_responses = [
            {"clip_id": row["clip_id"], "question_id": row["question_id"], "answer": row["answer"]}
            for row in cls.key["answers"]
        ]

    def test_fixture_has_required_real_world_categories_and_question_count(self):
        categories = {case["category"] for case in self.bank["cases"]}
        self.assertGreaterEqual(len(self.bank["cases"]), 16)
        self.assertTrue({
            "negation", "quantities", "amounts", "dates", "time", "names",
            "units", "sequence", "conditions", "safety"
        }.issubset(categories))
        self.assertGreaterEqual(len(self.key["answers"]), 24)

    def test_public_questions_do_not_expose_source_text_category_or_answer_key(self):
        self.assertEqual(set(self.public), {"format_version", "cases"})
        for case in self.public["cases"]:
            self.assertEqual(set(case), {"clip_id", "questions"})
            for question in case["questions"]:
                self.assertEqual(set(question), {"question_id", "question", "options"})
                self.assertNotIn("answer", question)
                self.assertEqual([o["id"] for o in question["options"]], ["A", "B", "C", "D"])
        encoded = json.dumps(self.public, ensure_ascii=False).lower()
        self.assertNotIn("text_sha256", encoded)
        self.assertNotIn("category", encoded)

    def test_correct_complete_response_pack_scores_per_question_and_case(self):
        result = score_responses(self.public, self.key, self.correct_responses)
        self.assertEqual(result["correct_questions"], result["total_questions"])
        self.assertEqual(result["question_accuracy"], 1.0)
        self.assertEqual(result["all_correct_cases"], result["total_cases"])
        self.assertEqual(result["case_all_questions_correct_rate"], 1.0)
        self.assertIsNone(result["release_threshold"])
        self.assertIn("not_set", result["release_decision"])

    def test_wrong_answer_reduces_accuracy_and_fails_its_case(self):
        responses = copy.deepcopy(self.correct_responses)
        responses[0]["answer"] = next(
            option for option in ("A", "B", "C", "D") if option != responses[0]["answer"]
        )
        result = score_responses(self.public, self.key, responses)
        self.assertEqual(result["correct_questions"], result["total_questions"] - 1)
        self.assertEqual(result["case_scores"][responses[0]["clip_id"]]["all_correct"], False)

    def test_missing_response_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing .* response"):
            score_responses(self.public, self.key, self.correct_responses[:-1])

    def test_duplicate_response_fails_closed(self):
        rows = copy.deepcopy(self.correct_responses)
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate response"):
            score_responses(self.public, self.key, rows)

    def test_unknown_question_fails_closed(self):
        rows = copy.deepcopy(self.correct_responses)
        rows[0]["question_id"] = "not-a-real-question"
        with self.assertRaisesRegex(ValueError, "unknown question"):
            score_responses(self.public, self.key, rows)

    def test_invalid_choice_fails_closed(self):
        rows = copy.deepcopy(self.correct_responses)
        rows[0]["answer"] = "E"
        with self.assertRaisesRegex(ValueError, "A, B, C, or D"):
            score_responses(self.public, self.key, rows)

    def test_answer_key_drift_against_public_pack_fails_closed(self):
        key = copy.deepcopy(self.key)
        key["answers"].pop()
        with self.assertRaisesRegex(ValueError, "coverage .* do not match"):
            score_responses(self.public, key, self.correct_responses)


    def test_malformed_unhashable_response_fields_fail_closed(self):
        rows = copy.deepcopy(self.correct_responses)
        rows[0]["answer"] = ["A"]
        with self.assertRaisesRegex(ValueError, "must select A, B, C, or D"):
            score_responses(self.public, self.key, rows)

        rows = copy.deepcopy(self.correct_responses)
        rows[0]["clip_id"] = ["VC-001"]
        with self.assertRaisesRegex(ValueError, "non-empty string clip/question IDs"):
            score_responses(self.public, self.key, rows)

    def test_duplicate_public_clip_identity_fails_closed(self):
        public = copy.deepcopy(self.public)
        public["cases"].append(copy.deepcopy(public["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate public clip ID"):
            score_responses(public, self.key, self.correct_responses)

    def test_prepare_separates_public_questions_and_private_answer_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "listening-pack"
            prepare(output, DEFAULT_CASES)
            public_file = output / "public_questions.json"
            private = Path(str(output) + ".private")
            key_file = private / "answer_key.json"
            render_file = private / "render_manifest.json"
            self.assertTrue(public_file.is_file())
            self.assertTrue(key_file.is_file())
            self.assertTrue(render_file.is_file())
            self.assertNotIn("answer_key", public_file.read_text(encoding="utf-8"))
            self.assertEqual(len(json.loads(public_file.read_text())["cases"]), len(self.bank["cases"]))
            if os.name == "posix":
                self.assertEqual(private.stat().st_mode & 0o777, 0o700)
                self.assertEqual(key_file.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                prepare(output, DEFAULT_CASES)


if __name__ == "__main__":
    unittest.main()
