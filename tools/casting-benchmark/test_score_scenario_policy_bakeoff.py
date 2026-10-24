"""Tests for independent replay of the multi-fidelity scenario policy ledger."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from multifidelity_scenario_policy_bakeoff import run_policy_sweep
from score_scenario_policy_bakeoff import (
    hypervolume_2d,
    loads_no_duplicate_keys,
    score_document,
)

ROOT = Path(__file__).resolve().parents[2]


class ScenarioPolicyReplayScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario_manifest = json.loads(
            (ROOT / "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.policy_manifest = json.loads(
            (ROOT / "docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.result = run_policy_sweep(
            scenario_ids=("smooth_stationary",),
            dimensions=(2,),
            severities=(0.0,),
            seeds=(17, 23),
            design_count=75,
            bootstrap_samples=100,
        )
        cls.source_hashes = {"test/source.py": "a" * 64}
        cls.input_hash = "b" * 64

    def _score(self, result: dict | None = None):
        return score_document(
            copy.deepcopy(self.result if result is None else result),
            self.scenario_manifest,
            self.policy_manifest,
            self.source_hashes,
            self.input_hash,
        )

    def test_known_answer_hypervolume(self) -> None:
        self.assertAlmostEqual(
            hypervolume_2d([(2, 8), (4, 4), (7, 2)], (10, 10)),
            46.0,
            places=12,
        )

    def test_valid_result_replays_all_policy_runs(self) -> None:
        receipt, errors = self._score()
        self.assertEqual(errors, [])
        self.assertEqual(receipt["status"], "SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE")
        self.assertEqual(receipt["run_count"], 2 * 5)
        self.assertEqual(receipt["summary_count_recomputed"], 5)
        self.assertEqual(receipt["paired_comparison_count_recomputed"], 4)
        self.assertEqual(receipt["input_sha256"], self.input_hash)
        self.assertEqual(receipt["source_sha256"], self.source_hashes)
        self.assertIn("not evidence of external SOTA", receipt["warning"])

    def test_declared_hypervolume_tampering_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][0]["normalized_hypervolume_fraction"] += 0.05
        receipt, errors = self._score(result)
        self.assertTrue(any("normalized_hypervolume_fraction mismatch" in error for error in errors))
        self.assertEqual(receipt["status"], "REJECTED_TRACE_OR_CONTRACT_MISMATCH")

    def test_trace_objective_tampering_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][0]["evaluations"][0]["objectives"][0] += 0.1
        _, errors = self._score(result)
        self.assertTrue(any("differs from scenario replay" in error for error in errors))

    def test_cost_ledger_tampering_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][0]["fidelity_cost_units_by_level"]["low"] += 1
        _, errors = self._score(result)
        self.assertTrue(any("fidelity_cost_units_by_level mismatch" in error for error in errors))

    def test_promotion_margin_tampering_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        run = next(row for row in result["paired_runs"] if row["promotion_records"])
        promotion = run["promotion_records"][0]
        promotion["observed_high_margin"] += 0.1
        _, errors = self._score(result)
        self.assertTrue(any("observed_high_margin differs" in error for error in errors))

    def test_duplicate_run_identity_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][1] = copy.deepcopy(result["paired_runs"][0])
        _, errors = self._score(result)
        self.assertTrue(any("duplicate policy-run key" in error for error in errors))
        self.assertTrue(any("run key coverage mismatch" in error for error in errors))

    def test_missing_raw_run_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"].pop()
        _, errors = self._score(result)
        self.assertTrue(any("paired_runs length" in error for error in errors))

    def test_untrusted_aggregate_values_are_recomputed(self) -> None:
        result = copy.deepcopy(self.result)
        result["summaries"][0]["median_normalized_hypervolume_fraction"] = 999.0
        receipt, errors = self._score(result)
        self.assertEqual(errors, [])
        self.assertNotEqual(
            receipt["recomputed_summaries"][0]["median_normalized_hypervolume_fraction"],
            999.0,
        )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            loads_no_duplicate_keys('{"trace":{"seed":17,"seed":23}}')

    def test_malformed_cost_type_is_rejected_without_crashing(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][0]["evaluations"][0]["cost_units"] = "one"
        _, errors = self._score(result)
        self.assertTrue(any("cost_units does not match" in error for error in errors))

    def test_unhashable_scenario_identifier_is_rejected_without_crashing(self) -> None:
        result = copy.deepcopy(self.result)
        result["scenario_ids"] = [["smooth_stationary"]]
        _, errors = self._score(result)
        self.assertTrue(any("unique list of strings" in error for error in errors))

    def test_invalid_top_level_budget_is_rejected_without_crashing(self) -> None:
        result = copy.deepcopy(self.result)
        result["cost_budget"] = ["200"]
        _, errors = self._score(result)
        self.assertTrue(any("top-level cost_budget must be a positive integer" in error for error in errors))

    def test_unknown_run_fields_fail_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["paired_runs"][0]["trust_me"] = True
        _, errors = self._score(result)
        self.assertTrue(any("has unknown keys" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
