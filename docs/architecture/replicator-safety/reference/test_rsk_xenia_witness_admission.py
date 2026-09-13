#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial tests for exact Xenia commitment trust-context admission."""

from __future__ import annotations

from dataclasses import replace
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

import rsk_xenia_witness_admission as admission
import rsk_xenia_witness_trust as trust
import test_rsk_xenia_witness_trust as fixtures


class XeniaWitnessContextAdmissionTests(unittest.TestCase):
    def test_exact_bound_context_is_accepted(self) -> None:
        data = fixtures.golden()
        configured = fixtures.policy(data)
        decision = admission.evaluate_xenia_witness_context(
            commitment_trust_context_digest=data["expected_trust_context_digest"],
            verified_keys=fixtures.verified_keys(data),
            policy=configured,
            trusted_interval=fixtures.trusted_interval(data),
        )
        self.assertTrue(decision.accepted, decision.reason)
        self.assertEqual(
            decision.verified.trust_context_digest,
            data["expected_trust_context_digest"],
        )
        self.assertEqual(list(decision.verified.key_ids), data["expected_verified_key_ids"])
        self.assertEqual(list(decision.verified.signer_ids), data["expected_verified_signer_ids"])
        self.assertEqual(
            list(decision.verified.failure_domains),
            data["expected_verified_failure_domains"],
        )

    def test_commitment_bound_to_different_context_freezes(self) -> None:
        data = fixtures.golden()
        decision = admission.evaluate_xenia_witness_context(
            commitment_trust_context_digest="f" * 64,
            verified_keys=fixtures.verified_keys(data),
            policy=fixtures.policy(data),
            trusted_interval=fixtures.trusted_interval(data),
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("different witness trust context", decision.reason)

    def test_digest_match_cannot_bypass_signer_or_domain_quality(self) -> None:
        data = fixtures.golden()
        configured = fixtures.policy(data)
        alpha_only = fixtures.verified_keys(data)[:2]
        relaxed_keys = replace(
            configured,
            minimum_key_quorum=2,
            minimum_signer_identities=2,
            minimum_failure_domains=1,
        )
        decision = admission.evaluate_xenia_witness_context(
            commitment_trust_context_digest=relaxed_keys.digest(),
            verified_keys=alpha_only,
            policy=relaxed_keys,
            trusted_interval=fixtures.trusted_interval(data),
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("signer identities", decision.reason)

        principals = list(configured.principals)
        principals[-1] = replace(principals[-1], failure_domain="operator.alpha")
        collapsed_domains = replace(configured, principals=tuple(principals))
        decision = admission.evaluate_xenia_witness_context(
            commitment_trust_context_digest=data["expected_trust_context_digest"],
            verified_keys=fixtures.verified_keys(data),
            policy=collapsed_domains,
            trusted_interval=fixtures.trusted_interval(data),
        )
        self.assertEqual(decision.status, "frozen")
        self.assertIn("failure-domain quorum", decision.reason)

    def test_malformed_digest_and_direct_construction_fail_closed(self) -> None:
        data = fixtures.golden()
        malformed = admission.evaluate_xenia_witness_context(
            commitment_trust_context_digest="not-a-digest",
            verified_keys=fixtures.verified_keys(data),
            policy=fixtures.policy(data),
            trusted_interval=fixtures.trusted_interval(data),
        )
        self.assertEqual(malformed.status, "frozen")

        with self.assertRaises(admission.XeniaWitnessAdmissionError):
            admission.VerifiedXeniaWitnessContextReference(
                trust_context_digest="0" * 64,
                key_ids=("key.one",),
                signer_ids=("signer.one",),
                failure_domains=("domain.one",),
                _marker=object(),
            )


if __name__ == "__main__":
    unittest.main()
