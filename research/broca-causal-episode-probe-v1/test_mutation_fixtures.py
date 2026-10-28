"""Integrity tests for the mutation/currentness synthetic fixture pack."""
from __future__ import annotations

import unittest
from pathlib import Path

from run_probe import read_jsonl, resolve_pointer, validate_fixtures

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "mutations"


class MutationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectories = read_jsonl(FIXTURE_DIR / "trajectories.jsonl")
        cls.questions = read_jsonl(FIXTURE_DIR / "questions.jsonl")
        validate_fixtures(cls.trajectories, cls.questions)
        cls.trajectory_by_id = {row["id"]: row for row in cls.trajectories}
        cls.question_by_id = {row["question_id"]: row for row in cls.questions}

    def test_fixture_size_and_ids_are_stable(self) -> None:
        self.assertEqual(len(self.trajectories), 9)
        self.assertEqual(len(self.questions), 12)
        self.assertEqual(len(self.trajectory_by_id), len(self.trajectories))
        self.assertEqual(len(self.question_by_id), len(self.questions))

    def test_every_question_has_gold_support_and_explicit_category(self) -> None:
        for row in self.questions:
            with self.subTest(question_id=row["question_id"]):
                self.assertTrue(row["category"])
                self.assertTrue(row["query"].strip())
                self.assertTrue(row["required_support"])

    def test_current_value_questions_forbid_superseded_evidence(self) -> None:
        expected = {
            "MCB-001", "MCB-003", "MCB-005", "MCB-006",
            "MCB-007", "MCB-010", "MCB-011",
        }
        for question_id in expected:
            question = self.question_by_id[question_id]
            with self.subTest(question_id=question_id):
                self.assertTrue(question["forbidden_support"])
                required = {(x["trajectory_id"], x["source_pointer"]) for x in question["required_support"]}
                forbidden = {(x["trajectory_id"], x["source_pointer"]) for x in question["forbidden_support"]}
                self.assertTrue(required.isdisjoint(forbidden))

    def test_aba_reversion_requires_new_provenance_not_string_match(self) -> None:
        question = self.question_by_id["MCB-012"]
        trajectory = self.trajectory_by_id["mutation-return-to-prior-value"]
        required_pointers = {x["source_pointer"] for x in question["required_support"]}
        forbidden_pointers = {x["source_pointer"] for x in question["forbidden_support"]}
        self.assertIn("/states/2/text", required_pointers)
        self.assertIn("/states/0/text", forbidden_pointers)
        self.assertIn("/states/1/text", forbidden_pointers)
        old_value = resolve_pointer(trajectory, "/states/0/text")
        current_value = resolve_pointer(trajectory, "/states/2/text")
        self.assertIn("10 Old Mill Road", old_value)
        self.assertIn("10 Old Mill Road", current_value)
        self.assertNotEqual("/states/0/text", "/states/2/text")

    def test_revocation_and_tentative_state_have_distinct_negative_controls(self) -> None:
        revoked = self.question_by_id["MCB-006"]
        self.assertIn("/states/1/text", {x["source_pointer"] for x in revoked["required_support"]})
        self.assertIn("/states/0/text", {x["source_pointer"] for x in revoked["forbidden_support"]})

        current_region = self.question_by_id["MCB-007"]
        proposed_region = self.question_by_id["MCB-008"]
        self.assertIn("/states/0/text", {x["source_pointer"] for x in current_region["forbidden_support"]})
        self.assertIn("/states/0/text", {x["source_pointer"] for x in proposed_region["required_support"]})

    def test_causal_chain_labels_exclude_the_failed_intervention_from_required_support(self) -> None:
        question = self.question_by_id["MCB-009"]
        required = {x["source_pointer"] for x in question["required_support"]}
        distractor = {x["source_pointer"] for x in question["distractor_support"]}
        self.assertIn("/states/0/text", required)
        self.assertIn("/states/2/text", required)
        self.assertIn("/states/1/text", distractor)
        self.assertNotIn("/states/1/text", required)


if __name__ == "__main__":
    unittest.main()
