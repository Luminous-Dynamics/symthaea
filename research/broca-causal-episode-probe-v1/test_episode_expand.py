"""Offline tests for fixed-budget episode-neighborhood expansion."""
from __future__ import annotations

import unittest

from episode_expand import extract_indexed_refs, expand_predictions, read_jsonl, FIXTURE_DIR


TRAJECTORIES = read_jsonl(FIXTURE_DIR / "trajectories.jsonl")


def prediction(question_id: str, audit: list[dict[str, object]]) -> dict[str, object]:
    return {"question_id": question_id, "audit": audit}


def hit(trajectory_id: str, pointer: str, step_index: int | None) -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "source_pointer": pointer,
        "step_index": step_index,
    }


class EpisodeExpandTests(unittest.TestCase):
    def test_expansion_stays_inside_budget_and_adds_adjacent_support(self) -> None:
        base = prediction("Q1", [
            hit("episode-release-rollback", "/states/0/text", 0),
            hit("episode-release-rollback", "/states/1/text", 1),
        ])
        output = expand_predictions(TRAJECTORIES, [base], k=5)
        audit = output[0]["audit"]
        self.assertEqual(len(audit), 5)
        keys = {(item["trajectory_id"], item["source_pointer"]) for item in audit}
        self.assertIn(("episode-release-rollback", "/states/0/action"), keys)
        self.assertIn(("episode-release-rollback", "/states/1/action"), keys)

    def test_no_expansion_item_can_be_created_from_an_unindexed_field(self) -> None:
        indexed = extract_indexed_refs(TRAJECTORIES)
        allowed = set(indexed)
        expanded = expand_predictions(
            TRAJECTORIES,
            [prediction("Q1", [hit("episode-release-rollback", "/states/0/text", 0)])],
            k=10,
        )[0]["audit"]
        for item in expanded:
            self.assertIn((item["trajectory_id"], item["source_pointer"]), allowed)

    def test_root_level_outcome_and_screenshot_are_not_expansion_candidates(self) -> None:
        indexed = extract_indexed_refs(TRAJECTORIES)
        self.assertFalse(any(pointer == "/outcome" for _trajectory, pointer in indexed))
        self.assertFalse(any("screenshot" in pointer for _trajectory, pointer in indexed))

    def test_step_index_mismatch_fails_closed(self) -> None:
        wrong = prediction("Q1", [
            hit("episode-release-rollback", "/states/0/text", 1),
        ])
        with self.assertRaisesRegex(ValueError, "step_index disagrees"):
            expand_predictions(TRAJECTORIES, [wrong], k=5)

    def test_pointer_outside_source_allowlist_fails_closed(self) -> None:
        invalid = prediction("Q1", [
            hit("episode-release-rollback", "/outcome", None),
        ])
        with self.assertRaisesRegex(ValueError, "outside indexed source allowlist"):
            expand_predictions(TRAJECTORIES, [invalid], k=5)

    def test_duplicate_seed_pointer_fails_closed(self) -> None:
        duplicate = hit("episode-release-rollback", "/states/0/text", 0)
        with self.assertRaisesRegex(ValueError, "duplicate BM25 evidence"):
            expand_predictions(TRAJECTORIES, [prediction("Q1", [duplicate, duplicate])], k=5)

    def test_expansion_order_is_deterministic(self) -> None:
        row = prediction("Q1", [hit("episode-release-rollback", "/states/0/text", 0)])
        first = expand_predictions(TRAJECTORIES, [row], k=5)
        second = expand_predictions(TRAJECTORIES, [row], k=5)
        self.assertEqual(first, second)

    def test_neighbor_expansion_does_not_claim_to_suppress_stale_state(self) -> None:
        # Deliberate adversarial result: adjacency alone can surface a retired fact.
        # Currentness needs explicit version/supersession semantics, not proximity.
        row = prediction("Q2", [
            hit("episode-office-address", "/states/1/text", 1),
        ])
        audit = expand_predictions(TRAJECTORIES, [row], k=5)[0]["audit"]
        keys = {(item["trajectory_id"], item["source_pointer"]) for item in audit}
        self.assertIn(("episode-office-address", "/states/0/text"), keys)

    def test_duplicate_question_ids_fail_closed(self) -> None:
        row = prediction("Q1", [hit("episode-release-rollback", "/states/0/text", 0)])
        with self.assertRaisesRegex(ValueError, "duplicate prediction question_id"):
            expand_predictions(TRAJECTORIES, [row, row], k=5)


if __name__ == "__main__":
    unittest.main()
