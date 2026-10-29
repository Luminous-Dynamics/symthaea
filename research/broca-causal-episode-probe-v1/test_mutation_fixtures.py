"""Integrity tests for the mutation/currentness synthetic fixture pack."""
from __future__ import annotations

import unittest
from pathlib import Path

from run_probe import read_jsonl, resolve_pointer, validate_fixtures
from score_probe import score_dataset

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "mutations"


class MutationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectories = read_jsonl(FIXTURE_DIR / "trajectories.jsonl")
        cls.questions = read_jsonl(FIXTURE_DIR / "questions.jsonl")
        cls.ledger = read_jsonl(FIXTURE_DIR / "eligibility-ledger.jsonl")
        validate_fixtures(cls.trajectories, cls.questions)
        cls.trajectory_by_id = {row["id"]: row for row in cls.trajectories}
        cls.question_by_id = {row["question_id"]: row for row in cls.questions}

    def test_fixture_size_and_ids_are_stable(self) -> None:
        self.assertEqual(len(self.trajectories), 9)
        self.assertEqual(len(self.questions), 12)
        self.assertEqual(len(self.trajectory_by_id), len(self.trajectories))
        self.assertEqual(len(self.question_by_id), len(self.questions))

    def test_every_question_has_a_supported_retrieval_intent(self) -> None:
        allowed = {
            "current_state", "historical_state", "revocation_status",
            "approved_state", "proposal_history", "causal_chain",
        }
        for row in self.questions:
            with self.subTest(question_id=row["question_id"]):
                self.assertIn(row.get("retrieval_intent"), allowed)

    def test_policy_ledger_covers_every_string_source_pointer_exactly_once(self) -> None:
        def string_pointers(value, prefix=""):
            out = []
            if isinstance(value, str):
                out.append(prefix)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    out.extend(string_pointers(child, f"{prefix}/{index}"))
            elif isinstance(value, dict):
                for key, child in value.items():
                    escaped = str(key).replace("~", "~0").replace("/", "~1")
                    out.extend(string_pointers(child, f"{prefix}/{escaped}"))
            return out

        expected = {
            (trajectory["id"], pointer)
            for trajectory in self.trajectories
            for pointer in string_pointers(trajectory)
        }
        actual = {
            (row["trajectory_id"], row["source_pointer"])
            for row in self.ledger
        }
        self.assertEqual(len(actual), len(self.ledger), "ledger keys must be unique")
        self.assertEqual(actual, expected, "all string leaves must be policy-labelled")

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

    def test_stale_evidence_breaks_safe_chain_even_when_ordinary_chain_is_complete(self) -> None:
        predictions = []
        for question in self.questions:
            audit = [
                {"trajectory_id": ref["trajectory_id"], "source_pointer": ref["source_pointer"]}
                for ref in question["required_support"]
            ]
            # Adversarial negative control: return the obsolete address alongside
            # all current support. Ordinary completeness must not hide the violation.
            if question["question_id"] == "MCB-001":
                audit.extend(
                    {"trajectory_id": ref["trajectory_id"], "source_pointer": ref["source_pointer"]}
                    for ref in question["forbidden_support"]
                )
            predictions.append({"question_id": question["question_id"], "audit": audit})

        report = score_dataset(self.questions, predictions, 5)
        self.assertEqual(report["overall"]["complete_chain_rate_at_k"], 1.0)
        self.assertAlmostEqual(report["overall"]["safe_complete_chain_rate_at_k"], 11 / 12, places=5)
        bad = next(row for row in report["per_question"] if row["question_id"] == "MCB-001")
        self.assertTrue(bad["complete_chain_at_k"])
        self.assertFalse(bad["safe_complete_chain_at_k"])
        self.assertEqual(bad["forbidden_hits_at_k"], 1)

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
