"""Unit tests for the independent, test-only Python task replay oracle.

These tests validate the reference oracle's own invariants. The workflow then
compares it against the separately implemented Rust generator.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_replay as rr  # noqa: E402


HERE = Path(__file__).resolve().parent
CONFIG_BYTES = (HERE.parent / "fixtures" / "generator-config.example.json").read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
SEED_A = bytes([0x41]) * 32
SEED_B = bytes([0x42]) * 32


def manifest_for(seed: bytes = SEED_A) -> dict:
    del seed  # The task generator needs the frozen family plan, not its commitment.
    return {
        "schema": "symthaea.science_bench_refresh_manifest.v1",
        "generation_id": "python-oracle-unit-test-v1",
        "purpose": "qualified_run",
        "lane": "sealed_refresh",
        "generator": {
            "repository": "fixture-only",
            "revision": "b" * 40,
            "configuration_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
        },
        "evaluator": {
            "version": "fixture-only",
            "source_sha256": "c" * 64,
            "oracle_separation": "independent_evaluator_boundary_verified",
        },
        "expected_task_count": 8,
        "task_family_mixture": [
            {"family_id": recipe["family_id"], "weight": 0.25, "planned_task_count": 2}
            for recipe in CONFIG["recipes"]
        ],
    }


class ReferenceReplayTests(unittest.TestCase):
    def test_same_seed_reproduces_identical_structure(self) -> None:
        manifest = manifest_for()
        first = rr.independently_generate(manifest, SEED_A, CONFIG)
        second = rr.independently_generate(manifest, SEED_A, CONFIG)
        self.assertEqual(first, second)

    def test_distinct_seeds_change_task_ids_and_observations(self) -> None:
        manifest = manifest_for()
        first = rr.independently_generate(manifest, SEED_A, CONFIG)
        second = rr.independently_generate(manifest, SEED_B, CONFIG)
        self.assertNotEqual(first, second)
        first_ids = {task["task_id"] for task in first["tasks"]}
        second_ids = {task["task_id"] for task in second["tasks"]}
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_tasks_are_sorted_and_ids_do_not_reveal_family(self) -> None:
        manifest = manifest_for()
        generated = rr.independently_generate(manifest, SEED_A, CONFIG)
        tasks = generated["tasks"]
        ids = [task["task_id"] for task in tasks]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        for task in tasks:
            self.assertTrue(task["task_id"].startswith("task-"))
            self.assertNotIn(task["family_id"], task["task_id"])
            policy_input = task["policy_input"]
            for forbidden in (
                "family_id",
                "task_family",
                "recipe_kind",
                "constructor_kind",
                "evaluator_only",
                "expected_law",
                "ground_truth",
            ):
                self.assertNotIn(forbidden, policy_input)

    def test_task_count_and_per_family_counts_match_frozen_plan(self) -> None:
        manifest = manifest_for()
        generated = rr.independently_generate(manifest, SEED_A, CONFIG)
        tasks = generated["tasks"]
        counts = Counter(task["family_id"] for task in tasks)
        expected = {
            family["family_id"]: family["planned_task_count"]
            for family in manifest["task_family_mixture"]
        }
        self.assertEqual(len(tasks), manifest["expected_task_count"])
        self.assertEqual(counts, Counter(expected))

    def test_each_constructor_keeps_evaluator_targets_separate(self) -> None:
        manifest = manifest_for()
        generated = rr.independently_generate(manifest, SEED_A, CONFIG)
        by_family = {task["family_id"]: task for task in generated["tasks"]}
        self.assertEqual(
            set(by_family),
            {recipe["family_id"] for recipe in CONFIG["recipes"]},
        )
        for task in generated["tasks"]:
            self.assertIn("instruction", task["policy_input"])
            self.assertIn("observations", task["policy_input"])
            self.assertIsInstance(task["evaluator_only"], dict)
            self.assertNotIn("evaluator_only", task["policy_input"])
        self.assertTrue(
            all(set(row) >= {"u", "y"} for row in by_family["semantic_invariance"]["policy_input"]["observations"])
        )
        self.assertTrue(
            all(set(row) >= {"x", "y"} for row in by_family["sampling_noise_shift"]["policy_input"]["observations"])
        )
        self.assertTrue(
            all(set(row) >= {"x", "y", "z"} for row in by_family["confounding_null"]["policy_input"]["observations"])
        )
        self.assertEqual(
            by_family["identifiability_abstention"]["policy_input"]["required_response_mode"],
            "report_identifiability_or_abstain",
        )

    def test_task_generation_domain_and_field_separation(self) -> None:
        first = rr.seed_word(SEED_A, "g", "f", 0, "slope")
        second = rr.seed_word(SEED_A, "g", "f", 0, "intercept")
        other_family = rr.seed_word(SEED_A, "g", "another-family", 0, "slope")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, other_family)


if __name__ == "__main__":
    unittest.main()
