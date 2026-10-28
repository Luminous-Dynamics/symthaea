"""Standard-library tests for the causal-episode probe evaluator."""
from __future__ import annotations

import unittest

from score_probe import score_dataset


def ref(trajectory_id: str, pointer: str) -> dict[str, str]:
    return {"trajectory_id": trajectory_id, "source_pointer": pointer}


class ScoreProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = [
            {
                "question_id": "Q1",
                "category": "causal_chain",
                "required_support": [ref("t1", "/states/0/action"), ref("t1", "/states/1/text")],
                "forbidden_support": [ref("t1", "/states/0/text")],
                "distractor_support": [ref("t2", "/states/0/text")],
            }
        ]

    def test_complete_chain_and_ranked_metrics(self) -> None:
        predictions = [{
            "question_id": "Q1",
            "audit": [
                {"trajectory_id": "t1", "source_pointer": "/states/1/text"},
                {"trajectory_id": "t1", "source_pointer": "/states/0/action"},
                {"trajectory_id": "t1", "source_pointer": "/states/0/text"},
            ],
        }]
        report = score_dataset(self.questions, predictions, 3)
        self.assertEqual(report["overall"]["complete_chain_rate_at_k"], 1.0)
        self.assertEqual(report["overall"]["required_support_recall_at_k"], 1.0)
        self.assertEqual(report["overall"]["forbidden_support_hit_rate_at_k"], 1.0)
        self.assertEqual(report["overall"]["safe_complete_chain_rate_at_k"], 0.0)
        self.assertFalse(report["per_question"][0]["safe_complete_chain_at_k"])
        self.assertEqual(report["overall"]["distractor_retrieval_rate_at_k"], 0.0)

    def test_complete_chain_without_forbidden_evidence_is_safe(self) -> None:
        predictions = [{
            "question_id": "Q1",
            "audit": [
                {"trajectory_id": "t1", "source_pointer": "/states/0/action"},
                {"trajectory_id": "t1", "source_pointer": "/states/1/text"},
            ],
        }]
        report = score_dataset(self.questions, predictions, 2)
        self.assertEqual(report["overall"]["complete_chain_rate_at_k"], 1.0)
        self.assertEqual(report["overall"]["safe_complete_chain_rate_at_k"], 1.0)
        self.assertTrue(report["per_question"][0]["safe_complete_chain_at_k"])

    def test_incomplete_chain_is_not_hidden_by_one_hit(self) -> None:
        predictions = [{
            "question_id": "Q1",
            "audit": [{"trajectory_id": "t1", "source_pointer": "/states/0/action"}],
        }]
        report = score_dataset(self.questions, predictions, 3)
        self.assertEqual(report["overall"]["required_support_recall_at_k"], 0.5)
        self.assertEqual(report["overall"]["complete_chain_rate_at_k"], 0.0)

    def test_missing_question_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            score_dataset(self.questions, [], 5)

    def test_duplicate_prediction_id_fails_closed(self) -> None:
        prediction = {"question_id": "Q1", "audit": []}
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            score_dataset(self.questions, [prediction, prediction], 5)

    def test_duplicate_pointer_in_ranked_audit_fails_closed(self) -> None:
        item = {"trajectory_id": "t1", "source_pointer": "/states/0/action"}
        with self.assertRaisesRegex(ValueError, "duplicate ranked evidence"):
            score_dataset(self.questions, [{"question_id": "Q1", "audit": [item, item]}], 5)

    def test_out_of_range_k_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            score_dataset(self.questions, [{"question_id": "Q1", "audit": []}], 0)

    def test_empty_question_set_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "questions must not be empty"):
            score_dataset([], [], 5)


if __name__ == "__main__":
    unittest.main()
