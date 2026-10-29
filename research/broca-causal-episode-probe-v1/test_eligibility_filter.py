"""Dependency-free tests for the oracle-intent eligibility-first ablation."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eligibility_filter import (
    LEDGER_SCHEMA,
    RUNNER_SCHEMA,
    filter_dataset,
    sha256_bytes,
    verify_input_manifest,
    infer_retrieval_intent,
    validate_ledger_coverage,
)


def question(qid: str = "Q1", intent: str = "current_state") -> dict:
    return {
        "question_id": qid,
        "query": "what is current?",
        "retrieval_intent": intent,
        # Deliberately present gold labels: the filter must not use them.
        "required_support": [{"trajectory_id": "t1", "source_pointer": "/gold"}],
        "forbidden_support": [{"trajectory_id": "t1", "source_pointer": "/gold-stale"}],
    }


def candidate(pointer: str, trajectory_id: str = "t1", rank: int | None = None) -> dict:
    item = {"trajectory_id": trajectory_id, "source_pointer": pointer, "content": pointer}
    if rank is not None:
        item["rank"] = rank
    return item


def ledger(pointer: str, status: str, trajectory_id: str = "t1") -> dict:
    return {
        "schema": LEDGER_SCHEMA,
        "trajectory_id": trajectory_id,
        "source_pointer": pointer,
        "status": status,
        "reason": "controlled test policy",
    }


class EligibilityFilterTests(unittest.TestCase):
    def test_current_intent_removes_historical_before_final_top_k(self) -> None:
        q = [question()]
        predictions = [{
            "question_id": "Q1",
            "audit": [
                candidate("/states/0/text", rank=1),
                candidate("/states/1/action", rank=2),
                candidate("/states/1/text", rank=3),
            ],
        }]
        policy = [
            ledger("/states/0/text", "historical"),
            ledger("/states/1/action", "current"),
            ledger("/states/1/text", "current"),
        ]
        result, metrics = filter_dataset(q, predictions, policy, 2)
        self.assertEqual(
            [row["source_pointer"] for row in result[0]["audit"]],
            ["/states/1/action", "/states/1/text"],
        )
        self.assertEqual(metrics["removed_by_status"]["historical"], 1)
        self.assertEqual(metrics["output_count"], 2)

    def test_historical_intent_keeps_old_state_and_removes_current(self) -> None:
        q = [question(intent="historical_state")]
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/current"), candidate("/old")],
        }]
        policy = [ledger("/current", "current"), ledger("/old", "historical")]
        result, _ = filter_dataset(q, predictions, policy, 2)
        self.assertEqual([row["source_pointer"] for row in result[0]["audit"]], ["/old"])

    def test_revocation_query_keeps_revocation_event_not_old_active_claim(self) -> None:
        q = [question(intent="revocation_status")]
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/active-before"), candidate("/revocation-event")],
        }]
        policy = [
            ledger("/active-before", "historical"),
            ledger("/revocation-event", "revoked"),
        ]
        result, _ = filter_dataset(q, predictions, policy, 2)
        self.assertEqual(
            [row["source_pointer"] for row in result[0]["audit"]],
            ["/revocation-event"],
        )

    def test_proposal_history_keeps_tentative_and_not_applied_records(self) -> None:
        q = [question(intent="proposal_history")]
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/approved"), candidate("/proposal"), candidate("/not-applied")],
        }]
        policy = [
            ledger("/approved", "current"),
            ledger("/proposal", "tentative"),
            ledger("/not-applied", "not_applied"),
        ]
        result, _ = filter_dataset(q, predictions, policy, 3)
        self.assertEqual(
            [row["source_pointer"] for row in result[0]["audit"]],
            ["/proposal", "/not-applied"],
        )

    def test_causal_chain_can_retain_failed_intervention_and_outcome_events(self) -> None:
        q = [question(intent="causal_chain")]
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/failed-cache"), candidate("/old-release"), candidate("/rollback")],
        }]
        policy = [
            ledger("/failed-cache", "causal"),
            ledger("/old-release", "historical"),
            ledger("/rollback", "causal"),
        ]
        result, _ = filter_dataset(q, predictions, policy, 3)
        self.assertEqual(len(result[0]["audit"]), 3)

    def test_unannotated_candidates_fail_closed(self) -> None:
        q = [question()]
        predictions = [{"question_id": "Q1", "audit": [candidate("/unmapped"), candidate("/known")]}]
        policy = [ledger("/known", "current")]
        result, metrics = filter_dataset(q, predictions, policy, 2)
        self.assertEqual([row["source_pointer"] for row in result[0]["audit"]], ["/known"])
        self.assertEqual(metrics["removed_unannotated_count"], 1)

    def test_candidates_are_filtered_before_final_budget(self) -> None:
        q = [question()]
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/old-1"), candidate("/new-1"), candidate("/old-2"), candidate("/new-2")],
        }]
        policy = [
            ledger("/old-1", "historical"),
            ledger("/new-1", "current"),
            ledger("/old-2", "historical"),
            ledger("/new-2", "current"),
        ]
        result, _ = filter_dataset(q, predictions, policy, 1)
        self.assertEqual([row["source_pointer"] for row in result[0]["audit"]], ["/new-1"])

    def test_unknown_intent_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown or missing retrieval_intent"):
            filter_dataset([question(intent="guess")], [{"question_id": "Q1", "audit": []}], [], 1)

    def test_duplicate_ledger_key_fails_closed(self) -> None:
        repeated = ledger("/x", "current")
        with self.assertRaisesRegex(ValueError, "duplicate ledger key"):
            filter_dataset(
                [question()],
                [{"question_id": "Q1", "audit": [candidate("/x")]}],
                [repeated, repeated],
                1,
            )

    def test_unversioned_ledger_fails_closed(self) -> None:
        bad = ledger("/x", "current")
        bad.pop("schema")
        with self.assertRaisesRegex(ValueError, "wrong or missing schema"):
            filter_dataset(
                [question()],
                [{"question_id": "Q1", "audit": [candidate("/x")]}],
                [bad],
                1,
            )

    def test_duplicate_candidate_key_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            filter_dataset(
                [question()],
                [{"question_id": "Q1", "audit": [candidate("/x"), candidate("/x")]}],
                [ledger("/x", "current")],
                1,
            )

    def test_prediction_set_must_match_question_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            filter_dataset([question()], [], [], 1)
        with self.assertRaisesRegex(ValueError, "unexpected prediction"):
            filter_dataset([question()], [{"question_id": "OTHER", "audit": []}], [], 1)

    def test_invalid_budget_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            filter_dataset([question()], [{"question_id": "Q1", "audit": []}], [], True)

    def test_keyword_classifier_gives_proposal_rule_priority_over_approved_words(self) -> None:
        self.assertEqual(
            infer_retrieval_intent("Which region was proposed but never approved?")[0],
            "proposal_history",
        )
        self.assertEqual(
            infer_retrieval_intent("Which region is currently approved and deployed?")[0],
            "approved_state",
        )

    def test_keyword_classifier_distinguishes_revocation_from_generic_currentness(self) -> None:
        self.assertEqual(
            infer_retrieval_intent("Can credential stage-key-17 still be used?")[0],
            "revocation_status",
        )
        self.assertEqual(
            infer_retrieval_intent("What is the current support office address?")[0],
            "current_state",
        )

    def test_keyword_classifier_detects_historical_and_causal_queries(self) -> None:
        self.assertEqual(
            infer_retrieval_intent("What address did the team use before the update?")[0],
            "historical_state",
        )
        self.assertEqual(
            infer_retrieval_intent("Which release preceded the spike, and which action preceded recovery?")[0],
            "causal_chain",
        )

    def test_keyword_classifier_abstains_when_no_rule_matches(self) -> None:
        self.assertEqual(infer_retrieval_intent("Tell me something interesting.")[0], None)
        self.assertEqual(infer_retrieval_intent("   ")[0], None)

    def test_heuristic_mode_uses_prediction_not_annotated_intent(self) -> None:
        q = question(intent="current_state")
        q["query"] = "What address did the team use before the update?"
        predictions = [{
            "question_id": "Q1",
            "audit": [candidate("/current"), candidate("/old")],
        }]
        policy = [ledger("/current", "current"), ledger("/old", "historical")]
        result, metrics = filter_dataset([q], predictions, policy, 2, intent_source="heuristic")
        self.assertEqual([row["source_pointer"] for row in result[0]["audit"]], ["/old"])
        self.assertEqual(metrics["intent_source"], "heuristic")
        self.assertEqual(metrics["intent_evaluation"]["accuracy"], 0.0)
        self.assertEqual(metrics["intent_evaluation"]["rows"][0]["predicted_intent"], "historical_state")

    def test_heuristic_unknown_intent_abstains_with_empty_context(self) -> None:
        q = question(intent="current_state")
        q["query"] = "Tell me something interesting."
        result, metrics = filter_dataset(
            [q],
            [{"question_id": "Q1", "audit": [candidate("/current")]}],
            [ledger("/current", "current")],
            1,
            intent_source="heuristic",
        )
        self.assertEqual(result[0]["audit"], [])
        self.assertEqual(metrics["intent_evaluation"]["unknown"], 1)
        self.assertEqual(metrics["per_intent"]["abstain"]["output"], 0)

    def test_unknown_intent_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "intent_source must be"):
            filter_dataset(
                [question()],
                [{"question_id": "Q1", "audit": []}],
                [],
                1,
                intent_source="neural-oracle",
            )

    def test_source_policy_coverage_accepts_exact_string_leaf_map(self) -> None:
        trajectories = [{
            "id": "t1",
            "states": [{"action": "run", "text": "completed"}],
        }]
        policy = [
            ledger("/id", "contextual"),
            ledger("/states/0/action", "causal"),
            ledger("/states/0/text", "causal"),
        ]
        validate_ledger_coverage(trajectories, policy)

    def test_source_policy_coverage_rejects_missing_or_dangling_pointer(self) -> None:
        trajectories = [{
            "id": "t1",
            "states": [{"action": "run", "text": "completed"}],
        }]
        policy = [
            ledger("/id", "contextual"),
            ledger("/states/0/action", "causal"),
            ledger("/states/9/text", "causal"),
        ]
        with self.assertRaisesRegex(ValueError, "eligibility ledger omits source string pointers"):
            validate_ledger_coverage(trajectories, policy)

    def _write_manifest_fixture(self, directory: Path) -> tuple[Path, Path, Path, Path]:
        questions_path = directory / "questions.jsonl"
        trajectories_path = directory / "trajectories.jsonl"
        predictions_path = directory / "candidates.jsonl"
        questions_bytes = (json.dumps(question(), sort_keys=True) + "\n").encode()
        trajectories_bytes = (json.dumps({"id": "t1", "states": []}, sort_keys=True) + "\n").encode()
        predictions_bytes = (
            json.dumps({"question_id": "Q1", "audit": []}, sort_keys=True) + "\n"
        ).encode()
        questions_path.write_bytes(questions_bytes)
        trajectories_path.write_bytes(trajectories_bytes)
        predictions_path.write_bytes(predictions_bytes)
        manifest = {
            "schema": RUNNER_SCHEMA,
            "worker_sha256": "a" * 64,
            "bm25": {"k1": 1.2, "b": 0.75, "max_items": 3},
            "question_fixture_sha256": sha256_bytes(questions_bytes),
            "trajectory_fixture_sha256": sha256_bytes(trajectories_bytes),
            "prediction_sha256": sha256_bytes(predictions_bytes),
            "question_count": 1,
            "trajectory_count": 1,
        }
        predictions_path.with_suffix(".jsonl.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return predictions_path, questions_path, trajectories_path, predictions_path.with_suffix(".jsonl.manifest.json")

    def test_candidate_receipt_and_fixture_digests_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pred, questions, trajectories, _ = self._write_manifest_fixture(Path(temp))
            manifest, *_ = verify_input_manifest(pred, questions, trajectories, 2)
            self.assertEqual(manifest["schema"], RUNNER_SCHEMA)
            self.assertEqual(manifest["bm25"]["max_items"], 3)

    def test_candidate_receipt_rejects_prediction_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pred, questions, trajectories, _ = self._write_manifest_fixture(Path(temp))
            pred.write_text(json.dumps({"question_id": "Q1", "audit": [candidate("/mutated")]}) + "\n")
            with self.assertRaisesRegex(ValueError, "prediction digest"):
                verify_input_manifest(pred, questions, trajectories, 2)

    def test_final_budget_cannot_exceed_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pred, questions, trajectories, _ = self._write_manifest_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "candidate pool"):
                verify_input_manifest(pred, questions, trajectories, 4)


if __name__ == "__main__":
    unittest.main()
