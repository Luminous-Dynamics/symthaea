#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial self-tests for the RSK reference schema-registry state verifier."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
import unittest
from pathlib import Path

import rsk_schema_registry as registry
import rsk_semantic_schema as semantic


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SCHEMA_REGISTRY_GOLDEN_V0_1.json"
)
SEMANTIC_V1 = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json"
)


def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def signer_records(data: dict) -> list[registry.AuthenticatedSignerEvidence]:
    return [registry.AuthenticatedSignerEvidence(**record) for record in data["authenticated_signer_evidence"]]


def interval(data: dict) -> registry.TrustedInterval:
    return registry.TrustedInterval(**data["trusted_interval"])


def rebound_signers(
    records: list[registry.AuthenticatedSignerEvidence], digest_value: str
) -> list[registry.AuthenticatedSignerEvidence]:
    return [replace(record, bound_snapshot_digest=digest_value) for record in records]


def evaluate(
    data: dict,
    snapshot: dict | None = None,
    *,
    prior: registry.AntiRollbackState | None = None,
    records: list[registry.AuthenticatedSignerEvidence] | None = None,
    trusted: registry.TrustedInterval | None = None,
    claimed_digest: str | None = None,
    policy: dict | None = None,
) -> registry.RegistryDecision:
    snapshot = copy.deepcopy(snapshot if snapshot is not None else data["snapshot"])
    policy = copy.deepcopy(policy if policy is not None else data["policy"])
    actual = registry.snapshot_digest(snapshot)
    records = records if records is not None else signer_records(data)
    if actual != data["snapshot_sha256"] and records == signer_records(data):
        records = rebound_signers(records, actual)
    return registry.evaluate_registry_snapshot(
        snapshot=snapshot,
        claimed_snapshot_digest=claimed_digest or actual,
        authenticated_signers=records,
        policy=policy,
        trust_snapshot_digest=data["trust_snapshot_sha256"],
        trusted_interval=trusted or interval(data),
        prior_state=prior or registry.AntiRollbackState.genesis(policy["registry_id"]),
    )


def sequence_two_from(data: dict, first: registry.RegistryDecision) -> dict:
    snapshot = copy.deepcopy(data["snapshot"])
    snapshot["sequence"] = 2
    snapshot["issued_at"] = 1300
    snapshot["expires_at"] = 2300
    snapshot["previous_snapshot_digest"] = first.state.accepted_digest
    return snapshot


class SchemaRegistryTests(unittest.TestCase):
    def test_committed_genesis_golden_accepts_and_resolves_exact_schemas(self) -> None:
        data = golden()
        self.assertEqual(registry.policy_digest(data["policy"]), data["policy_sha256"])
        self.assertEqual(registry.snapshot_digest(data["snapshot"]), data["snapshot_sha256"])

        decision = evaluate(data)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.status, "accepted")
        self.assertFalse(decision.state.forked)
        expected = data["expected_state"]
        self.assertEqual(decision.state.registry_id, expected["registry_id"])
        self.assertEqual(decision.state.highest_sequence, expected["highest_sequence"])
        self.assertEqual(decision.state.accepted_digest, expected["accepted_digest"])
        self.assertEqual(decision.state.latest_issued_at, expected["latest_issued_at"])
        self.assertEqual(decision.state.policy_digest, expected["policy_digest"])

        capability = decision.verified.resolve(
            "capability",
            "rsk.test.capabilities",
            1,
            "da004c77da0df512ef772aa167fd386b61d6a581a3be40ded93d056e36dbc856",
        )
        resource = decision.verified.resolve(
            "resource",
            "rsk.test.resources",
            2,
            "8386b56b11818273612cc9f15e6c6fa8cd19bbf9aa2c7a3476022d2f7ef0f1e1",
        )
        self.assertEqual(
            semantic.validate_capability_schema(capability),
            data["snapshot"]["entries"][0]["schema_id"],
        )
        self.assertEqual(
            semantic.require_runtime_bound_resource_schema(resource),
            data["snapshot"]["entries"][1]["schema_id"],
        )
        self.assertFalse(hasattr(decision.verified, "grant_replication_authority"))

    def test_duplicate_signer_identity_does_not_inflate_quorum(self) -> None:
        data = golden()
        records = signer_records(data)
        duplicate_identity = [
            records[0],
            replace(records[0], key_id="key.alpha.second"),
        ]
        decision = evaluate(data, records=duplicate_identity)
        self.assertEqual(decision.status, "denied")
        self.assertIn("signer identities", decision.reason)

    def test_failure_domain_collapse_does_not_satisfy_independence(self) -> None:
        data = golden()
        records = signer_records(data)
        collapsed = [records[0], replace(records[1], failure_domain="domain.alpha")]
        decision = evaluate(data, records=collapsed)
        self.assertEqual(decision.status, "denied")
        self.assertIn("failure domains", decision.reason)

    def test_signer_must_bind_exact_snapshot_and_be_current(self) -> None:
        data = golden()
        records = signer_records(data)
        wrong_digest = [replace(records[0], bound_snapshot_digest="2" * 64), records[1]]
        self.assertEqual(evaluate(data, records=wrong_digest).status, "denied")

        wrong_trust = [replace(records[0], trust_snapshot_digest="2" * 64), records[1]]
        self.assertEqual(evaluate(data, records=wrong_trust).status, "denied")

        revoked = [replace(records[0], lifecycle="revoked"), records[1]]
        self.assertEqual(evaluate(data, records=revoked).status, "denied")

        wrong_role = [replace(records[0], role="other-role"), records[1]]
        self.assertEqual(evaluate(data, records=wrong_role).status, "denied")

        not_yet_valid_at_issuance = [replace(records[0], valid_from=1050), records[1]]
        decision = evaluate(data, records=not_yet_valid_at_issuance)
        self.assertEqual(decision.status, "denied")
        self.assertIn("signer identities", decision.reason)

    def test_one_key_cannot_authenticate_two_signer_identities(self) -> None:
        data = golden()
        records = signer_records(data)
        conflicting = [records[0], replace(records[1], key_id=records[0].key_id)]
        decision = evaluate(data, records=conflicting)
        self.assertEqual(decision.status, "denied")
        self.assertIn("key identity maps to multiple signer identities", decision.reason)

    def test_extra_ineligible_signature_does_not_poison_valid_quorum(self) -> None:
        data = golden()
        records = signer_records(data)
        extra = registry.AuthenticatedSignerEvidence(
            signer_id="operator.gamma",
            key_id="key.gamma",
            role="schema-registry-signer",
            failure_domain="domain.gamma",
            signature_profile="ed25519-test",
            valid_from=900,
            valid_until=2100,
            lifecycle="revoked",
            bound_snapshot_digest=data["snapshot_sha256"],
            trust_snapshot_digest=data["trust_snapshot_sha256"],
        )
        decision = evaluate(data, records=[*records, extra])
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.verified.signer_ids, ("operator.alpha", "operator.beta"))

    def test_stale_snapshot_does_not_gain_time_from_replay(self) -> None:
        data = golden()
        first = evaluate(data)
        self.assertTrue(first.accepted)
        stale = registry.TrustedInterval(start=2001, end=2002)
        decision = evaluate(data, prior=first.state, trusted=stale)
        self.assertEqual(decision.status, "denied")
        self.assertEqual(decision.state, first.state)

    def test_claimed_snapshot_digest_and_embedded_schema_bytes_are_recomputed(self) -> None:
        data = golden()
        wrong_claim = evaluate(data, claimed_digest="3" * 64)
        self.assertEqual(wrong_claim.status, "denied")
        self.assertIn("digest mismatch", wrong_claim.reason)

        snapshot = copy.deepcopy(data["snapshot"])
        cap_entry = snapshot["entries"][0]
        cap = json.loads(cap_entry["canonical_schema_json"])
        cap["entries"][0]["description"] = "tampered meaning"
        cap_entry["canonical_schema_json"] = semantic.canonical_bytes(cap).decode("utf-8")
        decision = evaluate(data, snapshot)
        self.assertEqual(decision.status, "denied")
        self.assertIn("schema digest mismatch", decision.reason)

        noncanonical = copy.deepcopy(data["snapshot"])
        cap_entry = noncanonical["entries"][0]
        cap = json.loads(cap_entry["canonical_schema_json"])
        cap_entry["canonical_schema_json"] = json.dumps(cap, indent=2)
        decision = evaluate(data, noncanonical)
        self.assertEqual(decision.status, "denied")
        self.assertIn("canonical byte form", decision.reason)

    def test_duplicate_schema_keys_and_active_resource_v1_fail_closed(self) -> None:
        data = golden()
        duplicate = copy.deepcopy(data["snapshot"])
        duplicate["entries"].insert(1, copy.deepcopy(duplicate["entries"][0]))
        self.assertEqual(evaluate(data, duplicate).status, "denied")

        v1_data = json.loads(SEMANTIC_V1.read_text())
        historical = copy.deepcopy(data["snapshot"])
        resource = v1_data["resource_schema"]
        historical["entries"][1] = {
            "schema_kind": "resource",
            "family": resource["family"],
            "version": resource["version"],
            "canonical_encoding_version": "rsk.semantic-json.v1",
            "canonical_schema_json": semantic.canonical_bytes(resource).decode("utf-8"),
            "schema_id": v1_data["resource_schema_sha256"],
            "lifecycle_state": "active",
            "supersedes": [],
        }
        decision = evaluate(data, historical)
        self.assertEqual(decision.status, "denied")
        self.assertIn("runtime numeric dimension IDs", decision.reason)

    def test_exact_replay_is_idempotent_but_sequence_rollback_is_denied(self) -> None:
        data = golden()
        first = evaluate(data)
        replay = evaluate(data, prior=first.state)
        self.assertEqual(replay.status, "replay")
        self.assertEqual(replay.state, first.state)
        self.assertTrue(replay.accepted)

        seq2 = sequence_two_from(data, first)
        digest2 = registry.snapshot_digest(seq2)
        records2 = rebound_signers(signer_records(data), digest2)
        accepted2 = evaluate(
            data,
            seq2,
            prior=first.state,
            records=records2,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(accepted2.status, "accepted")

        rollback = evaluate(data, prior=accepted2.state)
        self.assertEqual(rollback.status, "denied")
        self.assertIn("sequence rollback", rollback.reason)
        self.assertEqual(rollback.state, accepted2.state)

    def test_same_sequence_collision_marks_tracker_forked_and_fork_is_sticky(self) -> None:
        data = golden()
        first = evaluate(data)
        collision = copy.deepcopy(data["snapshot"])
        collision["expires_at"] = 1999
        collision_digest = registry.snapshot_digest(collision)
        records = rebound_signers(signer_records(data), collision_digest)
        forked = evaluate(data, collision, prior=first.state, records=records)
        self.assertEqual(forked.status, "forked")
        self.assertTrue(forked.state.forked)
        self.assertIsNone(forked.verified)

        later = sequence_two_from(data, first)
        later_digest = registry.snapshot_digest(later)
        later_records = rebound_signers(signer_records(data), later_digest)
        still_forked = evaluate(
            data,
            later,
            prior=forked.state,
            records=later_records,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(still_forked.status, "forked")
        self.assertTrue(still_forked.state.forked)

    def test_predecessor_mismatch_is_fork_but_issued_at_regression_is_denial(self) -> None:
        data = golden()
        first = evaluate(data)

        bad_parent = sequence_two_from(data, first)
        bad_parent["previous_snapshot_digest"] = "4" * 64
        bad_digest = registry.snapshot_digest(bad_parent)
        bad_records = rebound_signers(signer_records(data), bad_digest)
        forked = evaluate(
            data,
            bad_parent,
            prior=first.state,
            records=bad_records,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(forked.status, "forked")
        self.assertTrue(forked.state.forked)

        regressed = sequence_two_from(data, first)
        regressed["issued_at"] = 999
        regressed["expires_at"] = 1800
        regressed_digest = registry.snapshot_digest(regressed)
        regressed_records = rebound_signers(signer_records(data), regressed_digest)
        denied = evaluate(
            data,
            regressed,
            prior=first.state,
            records=regressed_records,
            trusted=registry.TrustedInterval(1100, 1200),
        )
        self.assertEqual(denied.status, "denied")
        self.assertIn("issued_at regression", denied.reason)
        self.assertFalse(denied.state.forked)

    def test_schema_bytes_are_immutable_under_existing_family_version_key(self) -> None:
        data = golden()
        first = evaluate(data)
        seq2 = sequence_two_from(data, first)
        entry = seq2["entries"][0]
        cap = json.loads(entry["canonical_schema_json"])
        cap["entries"][0]["description"] = "new meaning under old version"
        entry["canonical_schema_json"] = semantic.canonical_bytes(cap).decode("utf-8")
        entry["schema_id"] = semantic.validate_capability_schema(cap)
        digest2 = registry.snapshot_digest(seq2)
        records2 = rebound_signers(signer_records(data), digest2)
        decision = evaluate(
            data,
            seq2,
            prior=first.state,
            records=records2,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(decision.status, "denied")
        self.assertIn("changed under existing key", decision.reason)

    def test_valid_explicit_supersession_advances_active_version(self) -> None:
        data = golden()
        first = evaluate(data)
        seq2 = sequence_two_from(data, first)
        old = seq2["entries"][0]
        old["lifecycle_state"] = "superseded"

        cap2 = json.loads(old["canonical_schema_json"])
        cap2["version"] = 2
        cap2_id = semantic.validate_capability_schema(cap2)
        new_entry = {
            "schema_kind": "capability",
            "family": cap2["family"],
            "version": 2,
            "canonical_encoding_version": "rsk.semantic-json.v1",
            "canonical_schema_json": semantic.canonical_bytes(cap2).decode("utf-8"),
            "schema_id": cap2_id,
            "lifecycle_state": "active",
            "supersedes": [old["schema_id"]],
        }
        seq2["entries"].insert(1, new_entry)
        digest2 = registry.snapshot_digest(seq2)
        records2 = rebound_signers(signer_records(data), digest2)
        second = evaluate(
            data,
            seq2,
            prior=first.state,
            records=records2,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(second.status, "accepted")
        self.assertEqual(second.state.highest_sequence, 2)

        with self.assertRaises(registry.RegistryError):
            second.verified.resolve(
                "capability",
                "rsk.test.capabilities",
                1,
                old["schema_id"],
            )
        resolved = second.verified.resolve(
            "capability", "rsk.test.capabilities", 2, cap2_id
        )
        self.assertEqual(resolved["version"], 2)

        seq3 = copy.deepcopy(seq2)
        seq3["sequence"] = 3
        seq3["issued_at"] = 1600
        seq3["expires_at"] = 2600
        seq3["previous_snapshot_digest"] = second.state.accepted_digest
        seq3["entries"][0]["lifecycle_state"] = "active"
        seq3["entries"][1]["lifecycle_state"] = "superseded"
        digest3 = registry.snapshot_digest(seq3)
        records3 = rebound_signers(signer_records(data), digest3)
        rollback = evaluate(
            data,
            seq3,
            prior=second.state,
            records=records3,
            trusted=registry.TrustedInterval(1700, 1800),
        )
        self.assertEqual(rollback.status, "denied")
        self.assertIn("more permissive", rollback.reason)

    def test_known_schema_entry_cannot_disappear_from_full_snapshot(self) -> None:
        data = golden()
        first = evaluate(data)
        seq2 = sequence_two_from(data, first)
        seq2["entries"] = seq2["entries"][:1]
        digest2 = registry.snapshot_digest(seq2)
        records2 = rebound_signers(signer_records(data), digest2)
        decision = evaluate(
            data,
            seq2,
            prior=first.state,
            records=records2,
            trusted=registry.TrustedInterval(1400, 1500),
        )
        self.assertEqual(decision.status, "denied")
        self.assertIn("disappeared", decision.reason)


if __name__ == "__main__":
    unittest.main()
