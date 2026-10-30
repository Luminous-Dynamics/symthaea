"""Test-only checks for the deterministic rubric exploitability corpus."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SCRIPT = TOOLS / "rubric_attack_vectors.py"
SCHEMA = TOOLS.parent / "schema" / "rubric-robustness-plan.schema.json"
SPEC = importlib.util.spec_from_file_location("rubric_attack_vectors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RubricAttackVectorTests(unittest.TestCase):
    def test_schema_and_generated_plan_shape(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        plan, corpus = MODULE.build_artifacts()
        self.assertEqual(plan["schema"], schema["properties"]["schema"]["const"])
        self.assertEqual(plan["purpose"], "example_only")
        self.assertEqual(plan["planned_case_count"], len(plan["cases"]))
        self.assertEqual(len(plan["cases"]), 11)
        self.assertEqual(len({item["case_id"] for item in plan["cases"]}), 10)
        self.assertEqual(len(corpus["artifacts"]), len(plan["cases"]))
        self.assertFalse(any(case["expected_judge_behavior"] == "accept" for case in plan["cases"]))

    def test_plan_commitment_is_deterministic_and_bound(self) -> None:
        first, _ = MODULE.build_artifacts()
        second, _ = MODULE.build_artifacts()
        self.assertEqual(first, second)
        fields = {key: value for key, value in first.items() if key != "plan_commitment_sha256"}
        expected = MODULE.framed_hash(MODULE.DOMAIN, MODULE.canonical_bytes(fields))
        self.assertEqual(first["plan_commitment_sha256"], expected)

    def test_rubric_paraphrase_changes_bytes_not_construct(self) -> None:
        plan, corpus = MODULE.build_artifacts()
        case = next(row for row in plan["cases"] if row["case_id"] == "rubric-paraphrase-001")
        artifact = next(row for row in corpus["artifacts"] if row["case_id"] == case["case_id"])
        self.assertNotEqual(case["baseline_rubric_sha256"], case["variant_rubric_sha256"])
        self.assertEqual(case["baseline_candidate_sha256"], case["variant_candidate_sha256"])
        self.assertEqual(artifact["baseline_candidate_utf8"], artifact["variant_candidate_utf8"])
        self.assertEqual(case["expected_judge_behavior"], "score_invariant_within_tolerance")

    def test_surface_attacks_keep_rubric_and_construct_fixed(self) -> None:
        plan, corpus = MODULE.build_artifacts()
        artifacts = {row["case_id"]: row for row in corpus["artifacts"]}
        attacks = [row for row in plan["cases"] if row["family"] == "surface_cue_attack"]
        self.assertEqual(len(attacks), 5)
        for case in attacks:
            self.assertEqual(case["baseline_rubric_sha256"], case["variant_rubric_sha256"])
            self.assertEqual(case["expected_relation"], "candidate_construct_invariant")
            self.assertEqual(case["expected_judge_behavior"], "no_unearned_score_increase")
            self.assertEqual(case["maximum_score_uplift_without_construct_gain_millionths"], 0)
            self.assertNotEqual(
                artifacts[case["case_id"]]["baseline_candidate_utf8"],
                artifacts[case["case_id"]]["variant_candidate_utf8"],
            )

    def test_fabricated_authority_cue_is_its_own_adversarial_family(self) -> None:
        plan, corpus = MODULE.build_artifacts()
        cases = [row for row in plan["cases"] if row["family"] == "authority_cue_attack"]
        self.assertEqual(len(cases), 1)
        case = cases[0]
        artifact = next(row for row in corpus["artifacts"] if row["case_id"] == case["case_id"])
        self.assertEqual(case["expected_relation"], "unsupported_authority_claim_added")
        self.assertEqual(case["expected_judge_behavior"], "no_unearned_score_increase")
        self.assertEqual(case["baseline_rubric_sha256"], case["variant_rubric_sha256"])
        self.assertNotEqual(artifact["baseline_candidate_utf8"], artifact["variant_candidate_utf8"])
    def test_meaning_changing_rubric_controls_change_rubric_not_candidate(self) -> None:
        plan, _ = MODULE.build_artifacts()
        cases = [row for row in plan["cases"] if row["family"] == "rubric_meaning_changing"]
        self.assertEqual(len(cases), 2)
        for case in cases:
            self.assertEqual(case["baseline_candidate_sha256"], case["variant_candidate_sha256"])
            self.assertNotEqual(case["baseline_rubric_sha256"], case["variant_rubric_sha256"])
            self.assertEqual(case["expected_relation"], "rubric_construct_changes")

    def test_evidence_withdrawal_changes_candidate_not_rubric(self) -> None:
        plan, _ = MODULE.build_artifacts()
        case = next(row for row in plan["cases"] if row["family"] == "evidence_semantic_change")
        self.assertNotEqual(case["baseline_candidate_sha256"], case["variant_candidate_sha256"])
        self.assertEqual(case["baseline_rubric_sha256"], case["variant_rubric_sha256"])
        self.assertEqual(case["expected_judge_behavior"], "lower_score_or_abstain")

    def test_example_plan_is_non_scoring_and_declares_no_judge_budget(self) -> None:
        plan, corpus = MODULE.build_artifacts()
        self.assertEqual(plan["purpose"], "example_only")
        self.assertEqual(corpus["purpose"], "example_only")
        self.assertEqual(plan["resource_budget"]["max_judge_calls"], 0)
        self.assertEqual(plan["resource_budget"]["max_tokens"], 0)
        self.assertEqual(plan["repeats_per_condition"], 3)
        self.assertEqual(plan["randomization_commitment_sha256"], "0" * 64)
        self.assertTrue(all(row["maximum_score_uplift_without_construct_gain_millionths"] == 0 for row in plan["cases"]))


if __name__ == "__main__":
    unittest.main()
