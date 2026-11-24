#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("ci_scope_shadow.py")
spec = importlib.util.spec_from_file_location("ci_scope_shadow", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def actual(*jobs: dict, complete: bool = True) -> dict:
    return {
        "schema": mod.ACTUAL_SCHEMA,
        "run_id": 123,
        "run_attempt": 1,
        "head_sha": "abc123",
        "complete": complete,
        "jobs": list(jobs),
    }


class ScopeRecommendationTests(unittest.TestCase):
    def test_docs_only_does_not_request_full_matrix(self) -> None:
        result = mod.recommend(["docs/architecture/FIELD.md", "README.md"])
        self.assertEqual(result["recommendation_class"], "docs_only")
        self.assertFalse(result["full_matrix_recommended"])
        self.assertFalse(result["enforcement_allowed"])

    def test_ci_evidence_infra_is_explicitly_modeled(self) -> None:
        result = mod.recommend([
            "scripts/ci_scope_shadow.py",
            "tests/fixtures/ci_scope_shadow/docs_only.txt",
            ".github/workflows/ci-scope-shadow.yml",
        ])
        self.assertEqual(result["recommendation_class"], "ci_evidence_infra_only")
        self.assertFalse(result["full_matrix_recommended"])

    def test_unmodeled_workflow_escalates_to_full_matrix(self) -> None:
        result = mod.recommend([".github/workflows/hal-safety.yml"])
        self.assertTrue(result["full_matrix_recommended"])

    def test_isolated_muse_maps_to_dedicated_job(self) -> None:
        result = mod.recommend([
            "crates/domains/symthaea-muse/src/lib.rs",
            "crates/domains/symthaea-music-theory/src/lib.rs",
            "docs/muse-note.md",
        ])
        self.assertEqual(result["recommendation_class"], "muse_isolated")
        self.assertIn("Muse (tests, studio, wasm UI)", result["predicted_job_patterns"])

    def test_shared_core_escalates_to_full_matrix(self) -> None:
        result = mod.recommend(["crates/core/symthaea-core/src/hdc_encoding.rs"])
        self.assertTrue(result["full_matrix_recommended"])
        self.assertEqual(result["predicted_job_patterns"], ["*"])

    def test_unknown_domain_escalates_to_full_matrix(self) -> None:
        result = mod.recommend(["crates/domains/symthaea-optics/src/lib.rs"])
        self.assertTrue(result["full_matrix_recommended"])
        self.assertTrue(any("not yet modeled" in reason for reason in result["reasons"]))

    def test_mixed_muse_and_core_escalates(self) -> None:
        result = mod.recommend([
            "crates/domains/symthaea-muse/src/lib.rs",
            "crates/core/symthaea-core/src/lib.rs",
        ])
        self.assertTrue(result["full_matrix_recommended"])

    def test_digest_is_order_independent_and_deduplicated(self) -> None:
        a = mod.recommend(["README.md", "docs/a.md", "README.md"])
        b = mod.recommend(["docs/a.md", "README.md"])
        self.assertEqual(a["changed_files_sha256"], b["changed_files_sha256"])
        self.assertEqual(a["changed_file_count"], 2)

    def test_escape_path_fails_closed(self) -> None:
        with self.assertRaises(mod.ShadowInputError):
            mod.recommend(["../outside"])


class ScopeEvaluationTests(unittest.TestCase):
    def test_unpredicted_failure_is_candidate_false_negative(self) -> None:
        prediction = mod.recommend(["docs/design.md"])
        result = mod.evaluate(prediction, actual(
            {"name": "Format Check", "conclusion": "success"},
            {"name": "Clippy", "conclusion": "failure"},
        ))
        self.assertEqual(result["candidate_false_negative_count"], 1)
        self.assertFalse(result["observation_complete_without_candidate_miss"])

    def test_predicted_failure_is_not_false_negative(self) -> None:
        prediction = mod.recommend(["crates/domains/symthaea-muse/src/lib.rs"])
        result = mod.evaluate(prediction, actual(
            {"name": "Muse (tests, studio, wasm UI)", "conclusion": "failure"}
        ))
        self.assertEqual(result["candidate_false_negative_count"], 0)
        self.assertTrue(result["observation_complete_without_candidate_miss"])

    def test_full_matrix_prediction_matches_any_failure(self) -> None:
        prediction = mod.recommend(["Cargo.lock"])
        result = mod.evaluate(prediction, actual(
            {"name": "Test CI-safe (science-ai-core)", "conclusion": "failure"},
            {"name": "Clippy", "conclusion": "failure"},
        ))
        self.assertEqual(result["candidate_false_negative_count"], 0)

    def test_cancelled_job_makes_observation_inconclusive(self) -> None:
        prediction = mod.recommend(["docs/design.md"])
        result = mod.evaluate(prediction, actual(
            {"name": "Clippy", "conclusion": "cancelled"}
        ))
        self.assertEqual(result["inconclusive_job_count"], 1)
        self.assertFalse(result["observation_complete_without_candidate_miss"])

    def test_declared_incomplete_job_set_never_counts_complete(self) -> None:
        prediction = mod.recommend(["docs/design.md"])
        result = mod.evaluate(prediction, actual(
            {"name": "Format Check", "conclusion": "success"},
            complete=False,
        ))
        self.assertEqual(result["candidate_false_negative_count"], 0)
        self.assertFalse(result["observation_complete_without_candidate_miss"])

    def test_actual_provenance_is_required(self) -> None:
        prediction = mod.recommend(["docs/design.md"])
        with self.assertRaises(mod.ShadowInputError):
            mod.evaluate(prediction, {"jobs": [{"name": "Format Check", "conclusion": "success"}]})

    def test_zero_misses_is_still_not_an_enforcement_verdict(self) -> None:
        prediction = mod.recommend(["docs/design.md"])
        result = mod.evaluate(prediction, actual(
            {"name": "Format Check", "conclusion": "success"}
        ))
        self.assertIn("not an enforcement verdict", result["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
