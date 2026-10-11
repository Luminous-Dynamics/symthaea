"""Adversarial contract tests for Cyber Crucible verifier-decision receipts."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.verify_cyber_crucible_decision_receipt import (
    DecisionVerificationError,
    POLICY_VERSION,
    SCHEMA_VERSION,
    ReplayLedger,
    canonical_json_digest,
    canonical_signed_bytes,
    load_json,
    verify_decision_receipt,
)

NOW = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SEED = bytes(range(32))


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class DecisionReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "replay.sqlite3"
        self.private = Ed25519PrivateKey.from_private_bytes(SEED)
        self.public = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.policy = {
            "schema_version": POLICY_VERSION,
            "policy_id": "fixture-decision-policy-v1",
            "keys": [{
                "key_id": "verifier-key-1",
                "verifier_id": "independent-verifier-a",
                "public_key": b64url(self.public),
                "not_before": "2029-01-01T00:00:00Z",
                "not_after": "2031-01-01T00:00:00Z",
                "revoked_at": None,
            }],
        }
        self.context = {
            "campaign_id": "campaign-fixture-1",
            "subject_id": "symthaea-test-build",
            "attempt_id": "attempt-0001",
            "scenario_id": "scenario-12",
            "scenario_revision": 3,
            "scenario_digest": "1" * 64,
            "input_receipt_digest": "2" * 64,
            "report_digest": "6" * 64,
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": "4" * 64,
            "decision_nonce": b64url(bytes(range(32, 64))),
        }

    def make_receipt(self, *, decision="accept", context=None, policy=None, nonce=None, decision_id=None):
        context = deepcopy(context if context is not None else self.context)
        policy = deepcopy(policy if policy is not None else self.policy)
        if nonce is not None:
            context["decision_nonce"] = b64url(nonce)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "Ed25519",
            "decision_id": decision_id or ("a" * 32),
            "decision_nonce": context["decision_nonce"],
            "signer_key_id": "verifier-key-1",
            "verifier_id": "independent-verifier-a",
            "decision": decision,
            "reason_code": "authenticated_under_policy" if decision == "accept" else "policy_rejected",
            **context,
            "context_digest": canonical_json_digest(context),
            "decision_trust_policy_digest": canonical_json_digest(policy),
            "issued_at": "2030-01-01T00:00:00Z",
            "expires_at": "2030-01-01T00:05:00Z",
            "non_authorizing": True,
        }
        signature = b64url(self.private.sign(canonical_signed_bytes(payload)))
        return {"payload": payload, "signature": signature}

    def verify(self, receipt, *, context=None, policy=None, path=None, now=NOW):
        return verify_decision_receipt(
            receipt,
            self.policy if policy is None else policy,
            self.context if context is None else context,
            now=now,
            replay_ledger=ReplayLedger(self.db_path if path is None else path),
        )

    def test_accept_decision_is_authenticated_but_non_authorizing(self):
        result = self.verify(self.make_receipt())
        self.assertTrue(result["decision_receipt_verified"])
        self.assertEqual(result["decision"], "accept")
        self.assertTrue(result["non_authorizing"])
        self.assertTrue(result["replay_checked"])

    def test_signed_reject_is_a_valid_decision_not_a_pass(self):
        result = self.verify(self.make_receipt(decision="reject"))
        self.assertTrue(result["decision_receipt_verified"])
        self.assertEqual(result["decision"], "reject")

    def test_non_string_decision_is_rejected_without_type_error(self):
        receipt = self.make_receipt()
        receipt["payload"]["decision"] = []
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_signature_mutation_is_rejected(self):
        receipt = self.make_receipt()
        original = receipt["signature"]
        receipt["signature"] = ("A" if original[0] != "A" else "B") + original[1:]
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_payload_mutation_invalidates_signature(self):
        receipt = self.make_receipt()
        context = deepcopy(self.context)
        context["campaign_id"] = "different-campaign"
        receipt["payload"]["campaign_id"] = "different-campaign"
        receipt["payload"]["context_digest"] = canonical_json_digest(context)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, context=context)

    def test_wrong_attempt_context_is_rejected(self):
        receipt = self.make_receipt()
        context = deepcopy(self.context)
        context["attempt_id"] = "other-attempt"
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, context=context)

    def test_wrong_input_receipt_digest_is_rejected(self):
        context = deepcopy(self.context)
        context["input_receipt_digest"] = "f" * 64
        receipt = self.make_receipt(context=context)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_policy_digest_substitution_is_rejected(self):
        receipt = self.make_receipt()
        altered = deepcopy(self.policy)
        altered["policy_id"] = "different-policy"
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, policy=altered)

    def test_untrusted_signer_is_rejected(self):
        receipt = self.make_receipt()
        policy = deepcopy(self.policy)
        policy["keys"][0]["public_key"] = b64url(bytes(reversed(range(32))))
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, policy=policy)

    def test_signer_identity_mismatch_is_rejected(self):
        receipt = self.make_receipt()
        receipt["payload"]["verifier_id"] = "subject-controlled"
        receipt["payload"]["context_digest"] = canonical_json_digest(self.context)
        # The signature is now invalid as well; rejection must still fail closed.
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_expired_decision_is_rejected(self):
        with self.assertRaises(DecisionVerificationError):
            self.verify(self.make_receipt(), now=datetime(2030, 1, 1, 0, 6, tzinfo=timezone.utc))

    def test_lifetime_over_one_hour_is_rejected(self):
        receipt = self.make_receipt()
        receipt["payload"]["expires_at"] = "2030-01-01T02:00:00Z"
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_revoked_signer_is_rejected(self):
        policy = deepcopy(self.policy)
        policy["keys"][0]["revoked_at"] = "2029-12-31T00:00:00Z"
        receipt = self.make_receipt(policy=policy)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, policy=policy)

    def test_expired_key_is_rejected(self):
        policy = deepcopy(self.policy)
        policy["keys"][0]["not_after"] = "2029-12-31T00:00:00Z"
        receipt = self.make_receipt(policy=policy)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, policy=policy)

    def test_duplicate_public_key_aliases_are_rejected(self):
        policy = deepcopy(self.policy)
        alias = deepcopy(policy["keys"][0])
        alias["key_id"] = "second-key-id"
        policy["keys"].append(alias)
        receipt = self.make_receipt(policy=policy)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt, policy=policy)

    def test_duplicate_json_members_are_rejected(self):
        path = Path(self.temp.name) / "duplicate.json"
        path.write_text('{"decision":"accept","decision":"reject"}', encoding="utf-8")
        with self.assertRaises(DecisionVerificationError):
            load_json(path)

    def test_unknown_payload_fields_are_rejected(self):
        receipt = self.make_receipt()
        receipt["payload"]["extra"] = "ignored?"
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_missing_replay_ledger_is_not_permitted(self):
        with self.assertRaises(DecisionVerificationError):
            verify_decision_receipt(self.make_receipt(), self.policy, self.context)

    def test_replay_of_same_decision_is_rejected(self):
        receipt = self.make_receipt()
        self.verify(receipt)
        with self.assertRaises(DecisionVerificationError):
            self.verify(receipt)

    def test_one_nonce_cannot_be_consumed_twice_under_distinct_decision_ids(self):
        first = self.make_receipt(decision_id="a" * 32)
        self.verify(first)
        second = self.make_receipt(decision_id="b" * 32)
        with self.assertRaises(DecisionVerificationError):
            self.verify(second)

    def test_independent_decisions_with_fresh_nonces_are_allowed(self):
        first = self.make_receipt(decision_id="a" * 32)
        self.verify(first)
        context = deepcopy(self.context)
        context["decision_nonce"] = b64url(bytes(range(64, 96)))
        second = self.make_receipt(context=context, decision_id="b" * 32)
        self.assertEqual(self.verify(second, context=context)["decision"], "accept")

    def test_concurrent_replay_has_exactly_one_winner(self):
        receipt = self.make_receipt()
        def consume(_):
            try:
                self.verify(receipt)
                return "accepted"
            except DecisionVerificationError:
                return "rejected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(consume, range(2)))
        self.assertCountEqual(outcomes, ["accepted", "rejected"])

    def test_symlink_replay_ledger_is_rejected(self):
        real = Path(self.temp.name) / "real.sqlite3"
        real.touch()
        link = Path(self.temp.name) / "ledger-link.sqlite3"
        link.symlink_to(real)
        with self.assertRaises(DecisionVerificationError):
            self.verify(self.make_receipt(), path=link)


if __name__ == "__main__":
    unittest.main()
