#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial tests for the RSK Xenia witness trust-context reference boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rsk_monotonic_anchor as anchor
import rsk_xenia_witness_trust as trust

GOLDEN = HERE.parent / "golden" / "RSK_XENIA_WITNESS_TRUST_CONTEXT_GOLDEN_V0_1.json"


def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def anchor_policy(data: dict | None = None) -> anchor.AnchorPolicy:
    data = data or golden()
    raw = data["anchor_policy"]
    return anchor.AnchorPolicy(
        namespace=raw["namespace"],
        epoch=raw["epoch"],
        allowed_provider_profiles=tuple(raw["allowed_provider_profiles"]),
        allowed_provider_identities=tuple(raw["allowed_provider_identities"]),
        trust_snapshot_digest=raw["trust_snapshot_digest"],
    )


def principal(raw: dict) -> trust.XeniaWitnessPrincipal:
    return trust.XeniaWitnessPrincipal(**raw)


def policy(data: dict | None = None) -> trust.XeniaWitnessTrustContextPolicy:
    data = data or golden()
    raw = data["trust_context"]
    return trust.XeniaWitnessTrustContextPolicy(
        anchor_policy=anchor_policy(data),
        xenia_profile=raw["xenia_profile"],
        minimum_key_quorum=raw["minimum_key_quorum"],
        minimum_signer_identities=raw["minimum_signer_identities"],
        minimum_failure_domains=raw["minimum_failure_domains"],
        allowed_signature_profiles=tuple(raw["allowed_signature_profiles"]),
        signer_lifecycle_policy_digest=raw["signer_lifecycle_policy_digest"],
        failure_domain_policy_digest=raw["failure_domain_policy_digest"],
        trust_snapshot_digest=raw["trust_snapshot_digest"],
        principals=tuple(principal(item) for item in raw["principals"]),
    )


def verified_keys(data: dict | None = None) -> tuple[trust.VerifiedXeniaWitnessKeyReference, ...]:
    data = data or golden()
    return tuple(trust.VerifiedXeniaWitnessKeyReference(**item) for item in data["verified_xenia_keys"])


def trusted_interval(data: dict | None = None) -> anchor.TrustedAnchorInterval:
    data = data or golden()
    return anchor.TrustedAnchorInterval(**data["trusted_interval"])


class XeniaWitnessTrustContextTests(unittest.TestCase):
    def test_golden_policy_and_three_quorum_result(self) -> None:
        data = golden()
        configured = policy(data)
        self.assertEqual(
            trust.anchor_policy_digest(configured.anchor_policy),
            data["expected_anchor_policy_digest"],
        )
        self.assertEqual(configured.digest(), data["expected_trust_context_digest"])

        decision = trust.evaluate_xenia_witness_quorum_quality(
            verified_keys(data), configured, trusted_interval(data)
        )
        self.assertTrue(decision.accepted, decision.reason)
        self.assertEqual(decision.verified.trust_context_digest, data["expected_trust_context_digest"])
        self.assertEqual(list(decision.verified.key_ids), data["expected_verified_key_ids"])
        self.assertEqual(list(decision.verified.signer_ids), data["expected_verified_signer_ids"])
        self.assertEqual(
            list(decision.verified.failure_domains),
            data["expected_verified_failure_domains"],
        )

    def test_multiple_keys_from_one_signer_do_not_inflate_signer_quorum(self) -> None:
        data = golden()
        configured = replace(
            policy(data),
            minimum_key_quorum=2,
            minimum_signer_identities=2,
            minimum_failure_domains=1,
        )
        alpha_keys = verified_keys(data)[:2]
        decision = trust.evaluate_xenia_witness_quorum_quality(
            alpha_keys, configured, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("signer identities", decision.reason)

    def test_multiple_signers_in_one_failure_domain_do_not_inflate_domain_quorum(self) -> None:
        data = golden()
        configured = policy(data)
        principals = list(configured.principals)
        principals[-1] = replace(principals[-1], failure_domain="operator.alpha")
        collapsed = replace(configured, principals=tuple(principals))
        decision = trust.evaluate_xenia_witness_quorum_quality(
            verified_keys(data), collapsed, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("failure-domain quorum", decision.reason)

    def test_unknown_duplicate_and_wrong_profile_keys_freeze(self) -> None:
        data = golden()
        configured = policy(data)
        keys = list(verified_keys(data))

        unknown = replace(keys[0], public_key_fingerprint="9" * 64)
        decision = trust.evaluate_xenia_witness_quorum_quality(
            (unknown, keys[1], keys[2]), configured, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("absent from trust snapshot", decision.reason)

        decision = trust.evaluate_xenia_witness_quorum_quality(
            (keys[0], keys[0], keys[2]), configured, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("duplicate verified", decision.reason)

        wrong_profile = replace(keys[0], signature_profile="ml-dsa-87")
        decision = trust.evaluate_xenia_witness_quorum_quality(
            (wrong_profile, keys[1], keys[2]), configured, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("signature profile", decision.reason)

    def test_revoked_or_interval_invalid_key_freezes(self) -> None:
        data = golden()
        configured = policy(data)
        principals = list(configured.principals)

        revoked_principals = principals.copy()
        revoked_principals[-1] = replace(revoked_principals[-1], lifecycle="revoked")
        revoked = replace(configured, principals=tuple(revoked_principals))
        decision = trust.evaluate_xenia_witness_quorum_quality(
            verified_keys(data), revoked, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")

        expired_principals = principals.copy()
        expired_principals[-1] = replace(expired_principals[-1], valid_until=1150)
        expired = replace(configured, principals=tuple(expired_principals))
        decision = trust.evaluate_xenia_witness_quorum_quality(
            verified_keys(data), expired, trusted_interval(data)
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("full trusted interval", decision.reason)

    def test_fingerprint_alias_and_conflicting_signer_metadata_are_structurally_rejected(self) -> None:
        data = golden()
        configured = policy(data)
        principals = list(configured.principals)

        aliased = principals.copy()
        aliased[1] = replace(
            aliased[1],
            public_key_fingerprint=aliased[0].public_key_fingerprint,
        )
        with self.assertRaisesRegex(trust.WitnessTrustError, "raw-key fingerprint"):
            replace(configured, principals=tuple(aliased)).validate()

        conflicted = principals.copy()
        conflicted[1] = replace(conflicted[1], failure_domain="operator.other")
        with self.assertRaisesRegex(trust.WitnessTrustError, "conflicting role/failure-domain"):
            replace(configured, principals=tuple(conflicted)).validate()

    def test_trust_snapshot_substitution_changes_digest_and_is_not_anchor_policy_compatible(self) -> None:
        data = golden()
        configured = policy(data)
        substituted = replace(configured, trust_snapshot_digest="8" * 64)
        with self.assertRaisesRegex(trust.WitnessTrustError, "differs from anchor policy"):
            substituted.validate()

        principals = tuple(
            replace(item, valid_until=item.valid_until + 1)
            for item in configured.principals
        )
        changed = replace(configured, principals=principals)
        self.assertNotEqual(changed.digest(), configured.digest())

    def test_verified_quorum_type_cannot_be_constructed_without_internal_marker(self) -> None:
        with self.assertRaises(trust.WitnessTrustError):
            trust.VerifiedXeniaWitnessQuorumReference(
                trust_context_digest="0" * 64,
                key_ids=("key.one",),
                signer_ids=("signer.one",),
                failure_domains=("domain.one",),
                _marker=object(),
            )


if __name__ == "__main__":
    unittest.main()
