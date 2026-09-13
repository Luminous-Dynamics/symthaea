#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial self-tests for the RSK monotonic-anchor reference composition."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest
from pathlib import Path

import rsk_monotonic_anchor as anchor
import rsk_schema_registry as registry
import rsk_semantic_schema as semantic


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SCHEMA_REGISTRY_GOLDEN_V0_1.json"
)
ANCHOR_GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_MONOTONIC_ANCHOR_GOLDEN_V0_1.json"
)
XENIA_GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_XENIA_STATE_WITNESS_GOLDEN_V0_1.json"
)

XENIA_TARGET_DOMAIN = b"symthaea.rsk.xenia-target.v1\0"
XENIA_EPOCH_DOMAIN = b"symthaea.rsk.xenia-epoch.v1\0"
XENIA_COMMITMENT_DOMAIN = b"xenia:state-commitment:v1"
XENIA_COMMITMENT_SCHEMA = b"xenia-state-commitment-v1"


def registry_golden() -> dict:
    return json.loads(REGISTRY_GOLDEN.read_text())


def anchor_golden() -> dict:
    return json.loads(ANCHOR_GOLDEN.read_text())


def xenia_golden() -> dict:
    return json.loads(XENIA_GOLDEN.read_text())


def _push_bytes(buffer: bytearray, value: bytes) -> None:
    buffer.extend(len(value).to_bytes(8, "big"))
    buffer.extend(value)


def xenia_commitment_message(namespace: str, vector: dict) -> bytes:
    message = bytearray()
    _push_bytes(message, XENIA_COMMITMENT_DOMAIN)
    _push_bytes(message, XENIA_COMMITMENT_SCHEMA)
    _push_bytes(message, namespace.encode("utf-8"))
    _push_bytes(message, bytes.fromhex(vector["target_id_hex"]))
    _push_bytes(message, bytes.fromhex(vector["epoch_id_hex"]))
    message.extend(vector["counter"].to_bytes(8, "big"))
    _push_bytes(message, bytes.fromhex(vector["previous_commitment_hex"]))
    _push_bytes(message, bytes.fromhex(vector["state_digest_hex"]))
    _push_bytes(message, bytes.fromhex(vector["trust_context_digest_hex"]))
    message.extend(vector["timestamp_unix_secs"].to_bytes(8, "big"))
    return bytes(message)


def accepted_registry_state() -> registry.AntiRollbackState:
    data = registry_golden()
    signers = [
        registry.AuthenticatedSignerEvidence(**record)
        for record in data["authenticated_signer_evidence"]
    ]
    interval = registry.TrustedInterval(**data["trusted_interval"])
    decision = registry.evaluate_registry_snapshot(
        snapshot=data["snapshot"],
        claimed_snapshot_digest=data["snapshot_sha256"],
        authenticated_signers=signers,
        policy=data["policy"],
        trust_snapshot_digest=data["trust_snapshot_sha256"],
        trusted_interval=interval,
        prior_state=registry.AntiRollbackState.genesis(data["policy"]["registry_id"]),
    )
    if not decision.accepted:
        raise AssertionError(f"registry golden did not accept: {decision.reason}")
    return decision.state


def policy(data: dict | None = None) -> anchor.AnchorPolicy:
    data = data or anchor_golden()
    raw = data["policy"]
    return anchor.AnchorPolicy(
        namespace=raw["namespace"],
        epoch=raw["epoch"],
        allowed_provider_profiles=tuple(raw["allowed_provider_profiles"]),
        allowed_provider_identities=tuple(raw["allowed_provider_identities"]),
        trust_snapshot_digest=raw["trust_snapshot_digest"],
    )


def evidence(data: dict | None = None) -> anchor.AuthenticatedMonotonicAnchorEvidence:
    data = data or anchor_golden()
    return anchor.AuthenticatedMonotonicAnchorEvidence(**data["authenticated_anchor_evidence"])


def trusted(data: dict | None = None) -> anchor.TrustedAnchorInterval:
    data = data or anchor_golden()
    return anchor.TrustedAnchorInterval(**data["trusted_interval"])


class MonotonicAnchorTests(unittest.TestCase):
    def test_committed_golden_binds_complete_registry_tracker(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        self.assertEqual(
            anchor.registry_tracker_digest(state),
            data["expected_registry_tracker_digest"],
        )
        decision = anchor.evaluate_registry_anchor(state, evidence(data), policy(data), trusted(data))
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.verified.counter, state.highest_sequence)
        self.assertEqual(decision.verified.state_digest, data["expected_registry_tracker_digest"])
        self.assertFalse(hasattr(decision.verified, "advance"))
        self.assertFalse(hasattr(decision.verified, "reset"))

    def test_unavailable_anchor_freezes_new_positive_authority(self) -> None:
        data = anchor_golden()
        decision = anchor.evaluate_registry_anchor(
            accepted_registry_state(), None, policy(data), trusted(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("unavailable", decision.reason)

    def test_local_ahead_and_anchor_ahead_both_freeze(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        base = evidence(data)

        anchor_behind = replace(base, counter=state.highest_sequence - 1)
        behind = anchor.evaluate_registry_anchor(state, anchor_behind, policy(data), trusted(data))
        self.assertEqual(behind.status, "frozen")
        self.assertIn("behind local", behind.reason)

        anchor_ahead = replace(base, counter=state.highest_sequence + 1)
        ahead = anchor.evaluate_registry_anchor(state, anchor_ahead, policy(data), trusted(data))
        self.assertEqual(ahead.status, "frozen")
        self.assertIn("local tracker is behind", ahead.reason)

    def test_same_counter_different_state_digest_is_not_reconciled(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        conflicting = replace(evidence(data), anchored_state_digest="f" * 64)
        decision = anchor.evaluate_registry_anchor(state, conflicting, policy(data), trusted(data))
        self.assertEqual(decision.status, "frozen")
        self.assertIn("different local state digest", decision.reason)

    def test_namespace_epoch_provider_and_trust_binding_are_exact(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        base = evidence(data)
        mutations = [
            replace(base, namespace="rsk.schema-registry.other"),
            replace(base, epoch=2),
            replace(base, provider_profile="rsk.test.other-provider.v1"),
            replace(base, provider_identity="anchor.other"),
            replace(base, trust_snapshot_digest="3" * 64),
        ]
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                self.assertEqual(
                    anchor.evaluate_registry_anchor(state, mutated, policy(data), trusted(data)).status,
                    "frozen",
                )

    def test_stale_or_ambiguous_time_freezes(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        stale = anchor.TrustedAnchorInterval(start=1901, end=1902)
        self.assertEqual(
            anchor.evaluate_registry_anchor(state, evidence(data), policy(data), stale).status,
            "frozen",
        )

        malformed = anchor.TrustedAnchorInterval(start=1200, end=1100)
        self.assertEqual(
            anchor.evaluate_registry_anchor(state, evidence(data), policy(data), malformed).status,
            "frozen",
        )

    def test_local_forked_tracker_cannot_be_rescued_by_matching_anchor(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        forked_state = replace(state, forked=True)
        matching = replace(
            evidence(data),
            counter=forked_state.highest_sequence,
            anchored_state_digest=anchor.registry_tracker_digest(forked_state),
        )
        decision = anchor.evaluate_registry_anchor(
            forked_state, matching, policy(data), trusted(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("already forked", decision.reason)

    def test_complete_state_digest_changes_when_authority_relevant_tracker_state_changes(self) -> None:
        state = accepted_registry_state()
        original = anchor.registry_tracker_digest(state)
        self.assertNotEqual(
            original,
            anchor.registry_tracker_digest(replace(state, latest_issued_at=state.latest_issued_at + 1)),
        )
        self.assertNotEqual(
            original,
            anchor.registry_tracker_digest(replace(state, policy_digest="4" * 64)),
        )
        changed_entries = list(state.known_entries)
        changed_entries[0] = replace(changed_entries[0], lifecycle_state="superseded")
        self.assertNotEqual(
            original,
            anchor.registry_tracker_digest(replace(state, known_entries=tuple(changed_entries))),
        )

    def test_reference_fixture_is_deterministic_but_not_provider_proof(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        generated = anchor.reference_anchor_evidence(
            state,
            policy(data),
            provider_profile="rsk.test.monotonic-anchor.v1",
            provider_identity="anchor.alpha",
            issued_at=1050,
            expires_at=1900,
        )
        self.assertEqual(generated, evidence(data))
        self.assertFalse(hasattr(anchor, "write_tpm_counter"))
        self.assertFalse(hasattr(anchor, "reset_hardware_anchor"))

    def test_crash_between_local_and_anchor_updates_is_fail_closed_in_both_orders(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        base = evidence(data)

        # Local durable state advanced but external anchor did not.
        local_advanced = replace(
            state,
            highest_sequence=state.highest_sequence + 1,
            accepted_digest="5" * 64,
        )
        local_first = anchor.evaluate_registry_anchor(
            local_advanced, base, policy(data), trusted(data)
        )
        self.assertEqual(local_first.status, "frozen")

        # External anchor advanced but local durable state did not.
        anchor_first = replace(
            base,
            counter=base.counter + 1,
            anchored_state_digest="6" * 64,
        )
        external_first = anchor.evaluate_registry_anchor(
            state, anchor_first, policy(data), trusted(data)
        )
        self.assertEqual(external_first.status, "frozen")

    def test_xenia_target_and_epoch_derivation_are_frozen(self) -> None:
        data = xenia_golden()
        target = data["target_derivation"]
        canonical = semantic.canonical_bytes(
            {
                "deployment_identity": target["deployment_identity"],
                "registry_id": target["registry_id"],
            }
        )
        self.assertEqual(canonical.decode("utf-8"), target["canonical_json_utf8"])
        target_id = hashlib.sha256(XENIA_TARGET_DOMAIN + canonical).digest()
        self.assertEqual(target_id.hex(), target["expected_target_id_hex"])

        epoch = data["epoch_derivation"]
        epoch_id = hashlib.sha256(
            XENIA_EPOCH_DOMAIN
            + target_id
            + epoch["rsk_recovery_epoch"].to_bytes(8, "big")
        ).digest()
        self.assertEqual(epoch_id.hex(), epoch["expected_epoch_id_hex"])

    def test_xenia_commitment_wire_vectors_match_published_bytes(self) -> None:
        data = xenia_golden()
        for vector in data["wire_vectors"]:
            with self.subTest(vector=vector["name"]):
                message = xenia_commitment_message(data["namespace"], vector)
                self.assertEqual(len(message), 321)
                self.assertEqual(message.hex(), vector["expected_message_hex"])

    def test_sequence_one_binds_published_xenia_genesis_fingerprint(self) -> None:
        vectors = {item["name"]: item for item in xenia_golden()["wire_vectors"]}
        genesis = vectors["counter_zero_genesis"]
        sequence_one = vectors["counter_one_binds_genesis"]
        self.assertEqual(genesis["counter"], 0)
        self.assertEqual(genesis["previous_commitment_hex"], "0" * 64)
        self.assertEqual(sequence_one["counter"], 1)
        self.assertEqual(
            sequence_one["previous_commitment_hex"],
            genesis["xenia_blake3_fingerprint_hex"],
        )

    def test_rsk_state_digest_mapping_is_decode_not_rehash(self) -> None:
        digest_hex = anchor.registry_tracker_digest(accepted_registry_state())
        raw = bytes.fromhex(digest_hex)
        self.assertEqual(len(raw), 32)
        self.assertEqual(raw.hex(), digest_hex)
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), digest_hex)

    def test_epoch_change_is_not_ordinary_progress(self) -> None:
        data = anchor_golden()
        state = accepted_registry_state()
        next_epoch = replace(evidence(data), epoch=2)
        decision = anchor.evaluate_registry_anchor(
            state, next_epoch, policy(data), trusted(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("epoch mismatch", decision.reason)


if __name__ == "__main__":
    unittest.main()
