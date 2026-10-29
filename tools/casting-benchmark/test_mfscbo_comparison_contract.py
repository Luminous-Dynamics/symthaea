#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from mfscbo_comparison_contract import (
    PREFLIGHT_LABEL,
    main,
    margin_to_mfscbo_constraint,
    scalarize_minimization_objectives,
    validate_contract,
)

UPSTREAM_SHA = "ca42fd6c5ec8c841e3f1ff9642509e12d104b581"
HASH = "a" * 64


def valid_contract():
    return {
        "schema_version": "1.0.0",
        "contract_id": "casting-mfscbo-control-v1",
        "upstream_repository": "Nemo-Virtual-Lab/MF-SCBO",
        "upstream_commit_sha": UPSTREAM_SHA,
        "task_suite_sha256": "b" * 64,
        "evidence_class": "public_analytic",
        "objective_directions": ["minimize", "minimize"],
        "normalization_offset": [0.1, 0.2],
        "normalization_scale": [0.4, 0.5],
        "normalization_source": "predeclared_bounds",
        "normalization_frozen_before_unseal": True,
        "scalarization_weights": [
            [1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]
        ],
        "scalarization_frozen_before_unseal": True,
        "constraint_fidelity_invariant": True,
        "constraint_definition_sha256": "c" * 64,
        "constraint_margin_sign": "positive_means_feasible",
        "upstream_constraint_sign": "c(x) <= 0",
        "target_hf_queries_allowed_in_search": True,
        "full_pool_audit_truth_used_in_search": False,
        "policy_frozen_before_outcome_unseal": True,
        "holdout_split": "independent_held_out",
        "fidelity_cost_units": {"low": 1, "medium": 5, "high": 25},
        "initial_points_per_fidelity": {"low": 8, "medium": 4, "high": 2},
        "cost_budget_units": 200,
        "seed": 17,
        "stop_rule": "Stop after cumulative evaluation cost reaches 200 units; audit all candidate truth after the run.",
    }


class MfScboComparisonContractTests(unittest.TestCase):
    def test_known_answer_contract_is_compatible_and_costed(self):
        value = validate_contract(valid_contract())
        self.assertEqual(value["initialization_cost_units"], 78)
        self.assertLessEqual(value["initialization_cost_units"], value["cost_budget_units"])
        self.assertEqual(value["upstream_commit_sha"], UPSTREAM_SHA)
        self.assertEqual(len(value["scalarization_weights"]), 5)

    def test_scalarization_maps_minimization_to_maximization(self):
        score = scalarize_minimization_objectives(
            values=[0.5, 0.4],
            offsets=[0.1, 0.2],
            scales=[0.4, 0.5],
            weights=[0.75, 0.25],
        )
        self.assertAlmostEqual(score, -1.0)
        self.assertGreater(
            scalarize_minimization_objectives([0.2, 0.3], [0.1, 0.2], [0.4, 0.5], [0.75, 0.25]),
            score,
        )

    def test_constraint_margin_sign_is_converted_for_upstream_api(self):
        self.assertLess(margin_to_mfscbo_constraint(0.25), 0.0)
        self.assertEqual(margin_to_mfscbo_constraint(0.0), 0.0)
        self.assertGreater(margin_to_mfscbo_constraint(-0.25), 0.0)

    def test_rejects_fidelity_dependent_constraint_margin(self):
        value = valid_contract()
        value["constraint_fidelity_invariant"] = False
        with self.assertRaisesRegex(ValueError, "one shared constraint callable"):
            validate_contract(value)

    def test_rejects_using_full_pool_audit_truth_in_search(self):
        value = valid_contract()
        value["full_pool_audit_truth_used_in_search"] = True
        with self.assertRaisesRegex(ValueError, "full-pool audit truth"):
            validate_contract(value)

    def test_rejects_normalization_or_weights_tuned_after_unseal(self):
        value = valid_contract()
        value["normalization_frozen_before_unseal"] = False
        with self.assertRaisesRegex(ValueError, "normalization_frozen_before_unseal"):
            validate_contract(value)
        value = valid_contract()
        value["scalarization_frozen_before_unseal"] = False
        with self.assertRaisesRegex(ValueError, "scalarization_frozen_before_unseal"):
            validate_contract(value)

    def test_rejects_invalid_scalarization_roster(self):
        value = valid_contract()
        value["scalarization_weights"][2] = [0.6, 0.6]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_contract(value)
        value = valid_contract()
        value["scalarization_weights"][-1] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_contract(value)
        value = valid_contract()
        value["scalarization_weights"] = [[0.5, 0.5]] * 5
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_contract(value)

    def test_rejects_budget_that_does_not_cover_initialization(self):
        value = valid_contract()
        value["cost_budget_units"] = 77
        with self.assertRaisesRegex(ValueError, "initialization cost 78 exceeds"):
            validate_contract(value)

    def test_rejects_changed_upstream_commit_and_wrong_holdout(self):
        value = valid_contract()
        value["upstream_commit_sha"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            validate_contract(value)
        value = valid_contract()
        value["holdout_split"] = "training"
        with self.assertRaisesRegex(ValueError, "independent_held_out"):
            validate_contract(value)

    def test_rejects_strict_schema_extra_field_and_bad_digest(self):
        value = valid_contract()
        value["nice_to_have"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_contract(value)
        value = valid_contract()
        value["task_suite_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_contract(value)

    def test_cli_binds_contract_lock_and_evaluator_digests(self):
        value = valid_contract()
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path, output_path = root / "contract.json", root / "receipt.json"
            contract_path.write_bytes(raw)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(contract_path), "--output", str(output_path)])
            self.assertEqual(code, 0)
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["receipt_label"], PREFLIGHT_LABEL)
            self.assertEqual(receipt["decision"], "compatible_contract_validated_not_run")
            self.assertEqual(receipt["source_artifacts"]["raw_contract_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(receipt["source_artifacts"]["mf_scbo_lock_sha256"]), 64)
            self.assertEqual(len(receipt["source_artifacts"]["preflight_evaluator_sha256"]), 64)
            self.assertIn("not an MF-SCBO execution", receipt["claim_ceiling"])

    def test_cli_rejects_duplicate_json_keys_and_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "contract.json"
            contract_path.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(contract_path)]), 2)
            contract_path.write_text(json.dumps(valid_contract()), encoding="utf-8")
            original = contract_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(contract_path), "--output", str(contract_path)]), 2)
            self.assertEqual(contract_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
