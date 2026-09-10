#!/usr/bin/env python3
"""Regression tests for lineage-aware trusted-main P0 version binding V2."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_lineage as enforcement_lineage  # noqa: E402
import trusted_main_ruleset_version_binding as binding_v1  # noqa: E402
import trusted_main_ruleset_version_binding_v2 as binding_v2  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402
import test_trusted_main_enforcement_lineage as linfx  # noqa: E402
import test_trusted_main_ruleset_version_binding as v1fx  # noqa: E402


def evidence_bundle() -> tuple[dict, dict, dict, dict]:
    v2_evidence, v3_evidence = linfx.evidence_pair()
    lineage_receipt = enforcement_lineage.derive_enforcement_lineage(
        policy=fx.policy(),
        enforcement_evidence_v2=v2_evidence,
        enforcement_evidence_v3=v3_evidence,
        expected_v2_evidence_id=v2_evidence["evidence_id"],
        expected_v3_evidence_id=v3_evidence["evidence_id"],
    )
    observations = {
        "schema": binding_v1.OBSERVATIONS_SCHEMA,
        "observations": copy.deepcopy(v2_evidence["operation_observations"]),
    }
    return v2_evidence, v3_evidence, lineage_receipt, observations


def derive(
    *,
    history=None,
    state=None,
    v2_evidence=None,
    v3_evidence=None,
    lineage_receipt=None,
    observations=None,
    **selectors,
):
    default_v2, default_v3, default_lineage, default_observations = evidence_bundle()
    v2_evidence = default_v2 if v2_evidence is None else v2_evidence
    v3_evidence = default_v3 if v3_evidence is None else v3_evidence
    lineage_receipt = default_lineage if lineage_receipt is None else lineage_receipt
    observations = default_observations if observations is None else observations
    history = v1fx.verified_history() if history is None else history
    state = v1fx.exact_state() if state is None else state
    values = {
        "policy": fx.policy(),
        "history_verification": history,
        "selected_version_state": state,
        "enforcement_evidence_v2": v2_evidence,
        "enforcement_evidence_v3": v3_evidence,
        "enforcement_lineage_receipt": lineage_receipt,
        "selected_observations": observations,
        "expected_history_id": history["history_id"],
        "expected_version_state_id": state["version_state_id"],
        "expected_v2_evidence_id": v2_evidence["evidence_id"],
        "expected_v3_evidence_id": v3_evidence["evidence_id"],
        "expected_lineage_id": lineage_receipt["lineage_id"],
    }
    values.update(selectors)
    return binding_v2.derive_version_binding_v2(**values)


class RulesetVersionBindingV2Tests(unittest.TestCase):
    def test_exact_lineage_and_closed_interval_bind_narrowly(self):
        result = derive()
        self.assertEqual(result["schema"], binding_v2.SCHEMA)
        self.assertEqual(
            result["disposition"],
            "P0RulesetVersionBindingWithEnforcementLineageOnly",
        )
        self.assertEqual(
            result["lineage_basis"],
            "exact-v2-v3-enforcement-lineage-plus-v1-closed-version-interval",
        )
        self.assertRegex(result["v1_binding_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(result["enforcement_lineage_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(result["binding_id"], r"^sha256:[0-9a-f]{64}$")

    def test_binding_identity_is_deterministic(self):
        self.assertEqual(derive()["binding_id"], derive()["binding_id"])

    def test_output_preserves_non_authorizing_ceiling(self):
        result = derive()
        self.assertEqual(result["provider_order_authority"], "github-provider-valid-utc-instants-only")
        self.assertEqual(result["capture_authentication"], "none")
        self.assertEqual(result["operator_authentication"], "none")
        self.assertEqual(result["receipt_attestation"], "none")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["scientific_authority"], "none")

    def test_lineage_selector_is_independent(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "enforcement_lineage: bytes do not match independently selected identity",
        ):
            derive(expected_lineage_id="sha256:" + "0" * 64)

    def test_supplied_lineage_bytes_must_equal_exact_rederivation(self):
        _, _, lineage_receipt, _ = evidence_bundle()
        changed = copy.deepcopy(lineage_receipt)
        changed["current_admission"] = "admitted"
        payload = dict(changed)
        payload.pop("lineage_id", None)
        changed["lineage_id"] = enforcement_lineage._content_id(payload)
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "supplied bytes differ from exact V2/V3 re-derived lineage",
        ):
            derive(lineage_receipt=changed, expected_lineage_id=changed["lineage_id"])

    def test_v3_cannot_repoint_to_different_v2_ancestry(self):
        v2_evidence, v3_evidence, _, observations = evidence_bundle()
        changed = copy.deepcopy(v3_evidence)
        changed["v2_evidence_id"] = "sha256:" + "1" * 64
        linfx.rederive_v3(changed)
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "enforcement_lineage: revalidation failed.*does not bind supplied exact V2",
        ):
            derive(
                v2_evidence=v2_evidence,
                v3_evidence=changed,
                observations=observations,
                expected_v3_evidence_id=changed["evidence_id"],
            )

    def test_v2_selector_is_independent(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "enforcement_lineage: revalidation failed.*enforcement_v2: bytes do not match independently selected identity",
        ):
            derive(expected_v2_evidence_id="sha256:" + "0" * 64)

    def test_v3_selector_is_independent(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "enforcement_lineage: revalidation failed.*enforcement_v3: bytes do not match independently selected identity",
        ):
            derive(expected_v3_evidence_id="sha256:" + "0" * 64)

    def test_history_selector_failure_remains_fail_closed(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "binding_v1: revalidation failed.*history_verification.*independently selected",
        ):
            derive(expected_history_id="sha256:" + "0" * 64)

    def test_version_state_selector_failure_remains_fail_closed(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "binding_v1: revalidation failed.*version_state_v2.*independently selected",
        ):
            derive(expected_version_state_id="sha256:" + "0" * 64)

    def test_missing_successor_boundary_remains_fail_closed(self):
        history = v1fx.verified_history([
            (6, "2026-09-09T20:00:00Z", v1fx.ACTOR_ID),
            (7, "2026-09-09T20:30:00Z", v1fx.ACTOR_ID),
        ])
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "binding_v1: revalidation failed.*no successor boundary",
        ):
            derive(history=history)

    def test_missing_predecessor_boundary_remains_fail_closed(self):
        history = v1fx.verified_history([
            (7, "2026-09-09T20:30:00Z", v1fx.ACTOR_ID),
            (8, "2026-09-09T22:00:00Z", v1fx.ACTOR_ID),
        ])
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "binding_v1: revalidation failed.*no predecessor boundary",
        ):
            derive(history=history)

    def test_selected_observation_drift_still_rejects(self):
        _, _, _, observations = evidence_bundle()
        changed = copy.deepcopy(observations)
        changed["observations"]["force_push"]["pushed_at"] = "2026-09-09T20:46:00Z"
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "binding_v1: revalidation failed.*force_push.*V3-selected observation identity",
        ):
            derive(observations=changed)

    def test_invalid_hash_shape_rejects_before_composition(self):
        with self.assertRaisesRegex(
            binding_v2.RulesetVersionBindingV2Error,
            "expected_v2_evidence_id: sha256 identity required",
        ):
            derive(expected_v2_evidence_id=True)

    def test_unexpected_programming_error_from_v1_remains_loud(self):
        original = binding_v1.derive_version_binding
        try:
            def explode(**_kwargs):
                raise RuntimeError("synthetic programming defect")

            binding_v1.derive_version_binding = explode
            with self.assertRaisesRegex(RuntimeError, "synthetic programming defect"):
                derive()
        finally:
            binding_v1.derive_version_binding = original

    def test_changed_exact_lineage_changes_v2_binding_identity(self):
        first = derive()
        v2_evidence, v3_evidence, _, observations = evidence_bundle()
        changed_v2 = copy.deepcopy(v2_evidence)
        changed_v3 = copy.deepcopy(v3_evidence)
        changed_v2["operation_observations"]["ordinary_direct_update"]["actor_name"] = "Actor Two"
        linfx.rehash_v2(changed_v2)
        new_observation_id = linfx.v3.rule_suite_observation_id(
            changed_v2["operation_observations"]["ordinary_direct_update"]
        )
        changed_v3["v2_evidence_id"] = changed_v2["evidence_id"]
        changed_v3["selected_rule_suite_observation_ids"]["ordinary_direct_update"] = new_observation_id
        linfx.rederive_v3(changed_v3)
        changed_lineage = enforcement_lineage.derive_enforcement_lineage(
            policy=fx.policy(),
            enforcement_evidence_v2=changed_v2,
            enforcement_evidence_v3=changed_v3,
            expected_v2_evidence_id=changed_v2["evidence_id"],
            expected_v3_evidence_id=changed_v3["evidence_id"],
        )
        changed_observations = {
            "schema": binding_v1.OBSERVATIONS_SCHEMA,
            "observations": copy.deepcopy(changed_v2["operation_observations"]),
        }
        second = derive(
            v2_evidence=changed_v2,
            v3_evidence=changed_v3,
            lineage_receipt=changed_lineage,
            observations=changed_observations,
        )
        self.assertNotEqual(first["enforcement_lineage_id"], second["enforcement_lineage_id"])
        self.assertNotEqual(first["binding_id"], second["binding_id"])


if __name__ == "__main__":
    unittest.main()
