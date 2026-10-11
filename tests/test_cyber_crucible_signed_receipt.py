"""Adversarial tests for the Cyber Crucible authenticated-receipt boundary."""
from __future__ import annotations

import base64
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_cyber_crucible_signed_receipt as verifier


NOW = datetime(2026, 10, 11, 0, 30, 0, tzinfo=timezone.utc)
TEST_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
)
PRIVATE = Ed25519PrivateKey.from_private_bytes(TEST_SEED)
PUBLIC_BYTES = PRIVATE.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def scenario_digest(char: str = "a") -> str:
    return char * 64


def signed_receipt(
    *,
    role: str = "security_verdict",
    status: str = "pass",
    nonce: str = "A" * 43,
    receipt_id: str = "00112233445566778899aabbccddeeff",
    issued_at: str = "2026-10-11T00:00:00Z",
    expires_at: str = "2026-10-11T01:00:00Z",
    scenario_id: str = "NET-DNS-TLS-001",
    digest: str | None = None,
    proposal_id: str | None = None,
    proposal_digest: str | None = None,
) -> dict:
    payload = {
        "schema_version": verifier.SCHEMA_VERSION,
        "algorithm": "Ed25519",
        "receipt_id": receipt_id,
        "key_id": "test-key-1",
        "evaluator_id": "independent-security-evaluator",
        "role": role,
        "campaign_id": "campaign-2026-10-A",
        "subject_id": "symthaea-candidate-A",
        "attempt_id": "attempt-2026-10-11-0001",
        "run_nonce": nonce,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "scenario_id": scenario_id,
        "scenario_revision": 1,
        "scenario_digest": scenario_digest(),
        "evidence_digest": "b" * 64,
        "status": status,
        "proposal_id": proposal_id,
        "proposal_digest": proposal_digest,
    }
    payload_bytes = verifier.canonical_signed_bytes(payload)
    signature = PRIVATE.sign(payload_bytes)
    return {"payload": payload, "signature": b64url(signature)}


def policy(roles: list[str] | None = None, *, revoked_at=None) -> dict:
    return {
        "schema_version": verifier.POLICY_VERSION,
        "policy_id": "test-policy-v1",
        "keys": [
            {
                "key_id": "test-key-1",
                "evaluator_id": "independent-security-evaluator",
                "public_key": b64url(PUBLIC_BYTES),
                "roles": roles or ["security_verdict"],
                "not_before": "2026-01-01T00:00:00Z",
                "not_after": "2027-01-01T00:00:00Z",
                "revoked_at": revoked_at,
            }
        ],
    }


def context(receipt: dict) -> dict:
    p = receipt["payload"]
    return {
        "campaign_id": p["campaign_id"],
        "subject_id": p["subject_id"],
        "attempt_id": p["attempt_id"],
        "run_nonce": p["run_nonce"],
        "scenario_id": p["scenario_id"],
        "scenario_revision": p["scenario_revision"],
        "scenario_digest": p["scenario_digest"],
        "evidence_digest": p["evidence_digest"],
        "role": p["role"],
        **(
            {"proposal_id": p["proposal_id"], "proposal_digest": p["proposal_digest"]}
            if p["role"] == "authorization_decision"
            else {}
        ),
    }


class SignedReceiptVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.replay_db = Path(self.temp.name) / "replay.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def verify(self, receipt=None, trust=None, expected=None, **kwargs):
        receipt = receipt or signed_receipt()
        return verifier.verify_receipt(
            receipt, trust or policy(), expected or context(receipt),
            now=kwargs.pop("now", NOW),
            replay_ledger=kwargs.pop("replay_ledger", None),
        )

    def test_rfc8032_ed25519_known_answer_vector(self):
        # RFC 8032, section 7.1, test vector 1: empty message.
        public = bytes.fromhex(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        )
        signature = bytes.fromhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        )
        verifier_key = __import__(
            "cryptography.hazmat.primitives.asymmetric.ed25519",
            fromlist=["Ed25519PublicKey"],
        ).Ed25519PublicKey.from_public_bytes(public)
        verifier_key.verify(signature, b"")
        with self.assertRaises(InvalidSignature):
            verifier_key.verify(signature, b"changed")

    def test_rfc8785_canonicalization_known_answer(self):
        # RFC 8785 section 3.2.2 number-serialization example.
        value = {"numbers": [333333333.3333333, 1e30, 4.5, 0.002, 1e-27]}
        self.assertEqual(
            __import__("rfc8785").dumps(value),
            b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}',
        )

    def test_unhashable_role_and_status_values_fail_closed(self):
        receipt = signed_receipt()
        receipt["payload"]["role"] = ["security_verdict"]
        with self.assertRaisesRegex(verifier.VerificationError, "payload.role is unsupported"):
            self.verify(receipt)
        receipt = signed_receipt()
        receipt["payload"]["status"] = ["pass"]
        with self.assertRaisesRegex(verifier.VerificationError, "payload.status is unsupported"):
            self.verify(receipt)

    def test_key_not_after_boundary_is_exclusive(self):
        trust = policy()
        trust["keys"][0]["not_after"] = "2026-10-11T00:30:00Z"
        with self.assertRaisesRegex(verifier.VerificationError, "not currently within its validity interval"):
            self.verify(trust=trust)

    def test_schema_accepts_valid_receipt_envelope(self):
        from jsonschema import Draft202012Validator
        schema_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_signed_receipt_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = list(Draft202012Validator(schema).iter_errors(signed_receipt()))
        self.assertEqual(errors, [])

    def test_schema_rejects_unknown_signed_payload_field(self):
        from jsonschema import Draft202012Validator
        schema_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_signed_receipt_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        receipt = signed_receipt()
        receipt["payload"]["unbound_security_claim"] = True
        errors = list(Draft202012Validator(schema).iter_errors(receipt))
        self.assertTrue(any("Additional properties are not allowed" in error.message for error in errors), errors)

    def test_valid_receipt_is_signature_verified_but_non_authorizing(self):
        result = self.verify(replay_ledger=verifier.ReplayLedger(self.replay_db))
        self.assertTrue(result["signature_verified"])
        self.assertTrue(result["replay_checked"])
        self.assertTrue(result["non_authorizing"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["subject_id"], "symthaea-candidate-A")
        self.assertEqual(result["attempt_id"], "attempt-2026-10-11-0001")
        self.assertRegex(result["trust_policy_digest"], r"^[0-9a-f]{64}$")

    def test_receipt_or_signed_payload_mutation_breaks_signature(self):
        receipt = signed_receipt()
        receipt["payload"]["status"] = "fail"
        with self.assertRaisesRegex(verifier.VerificationError, "signature verification failed"):
            self.verify(receipt)

    def test_modified_signature_is_rejected(self):
        receipt = signed_receipt()
        sig = bytearray(base64.urlsafe_b64decode(receipt["signature"] + "=="))
        sig[0] ^= 1
        receipt["signature"] = b64url(bytes(sig))
        with self.assertRaises(verifier.VerificationError):
            self.verify(receipt)

    def test_expected_subject_and_attempt_are_bound(self):
        receipt = signed_receipt()
        expected = context(receipt)
        expected["subject_id"] = "different-candidate"
        with self.assertRaisesRegex(verifier.VerificationError, "expected context field: subject_id"):
            self.verify(receipt, expected=expected)
        expected = context(receipt)
        expected["attempt_id"] = "different-attempt"
        with self.assertRaisesRegex(verifier.VerificationError, "expected context field: attempt_id"):
            self.verify(receipt, expected=expected)

    def test_expected_scenario_identity_mismatch_is_rejected(self):
        receipt = signed_receipt()
        expected = context(receipt)
        expected["scenario_digest"] = "c" * 64
        with self.assertRaisesRegex(verifier.VerificationError, "expected context field: scenario_digest"):
            self.verify(receipt, expected=expected)

    def test_trusted_key_cannot_claim_an_unpermitted_role(self):
        receipt = signed_receipt(role="functional_verdict")
        with self.assertRaisesRegex(verifier.VerificationError, "not authorized for receipt role"):
            self.verify(receipt, trust=policy(["security_verdict", "authorization_decision"]))

    def test_key_identity_must_match_signed_evaluator_identity(self):
        receipt = signed_receipt()
        trust = policy()
        trust["keys"][0]["evaluator_id"] = "different-evaluator"
        with self.assertRaisesRegex(verifier.VerificationError, "does not match trusted key identity"):
            self.verify(receipt, trust=trust)

    def test_expired_receipt_is_rejected(self):
        receipt = signed_receipt(expires_at="2026-10-11T00:29:59Z")
        with self.assertRaisesRegex(verifier.VerificationError, "has expired"):
            self.verify(receipt)

    def test_expiry_boundary_is_exclusive(self):
        receipt = signed_receipt(expires_at="2026-10-11T00:30:00Z")
        with self.assertRaisesRegex(verifier.VerificationError, "has expired"):
            self.verify(receipt)

    def test_future_issued_receipt_is_rejected(self):
        receipt = signed_receipt(issued_at="2026-10-11T00:31:00Z")
        with self.assertRaisesRegex(verifier.VerificationError, "issue time is in the future"):
            self.verify(receipt)

    def test_revoked_key_is_rejected_even_for_valid_signature(self):
        with self.assertRaisesRegex(verifier.VerificationError, "key is revoked"):
            self.verify(trust=policy(revoked_at="2026-10-11T00:20:00Z"))

    def test_replayed_receipt_id_or_nonce_is_rejected_atomically(self):
        receipt = signed_receipt()
        ledger = verifier.ReplayLedger(self.replay_db)
        self.verify(receipt, replay_ledger=ledger)
        with self.assertRaisesRegex(verifier.VerificationError, "replay rejected"):
            self.verify(receipt, replay_ledger=ledger)

    def test_invalid_signature_does_not_burn_nonce(self):
        receipt = signed_receipt()
        ledger = verifier.ReplayLedger(self.replay_db)
        bad = copy.deepcopy(receipt)
        bad["payload"]["status"] = "fail"
        with self.assertRaises(verifier.VerificationError):
            self.verify(bad, replay_ledger=ledger)
        self.verify(receipt, replay_ledger=ledger)

    def test_same_nonce_in_different_receipt_is_rejected(self):
        ledger = verifier.ReplayLedger(self.replay_db)
        first = signed_receipt()
        second = signed_receipt(receipt_id="11112222333344445555666677778888")
        self.verify(first, replay_ledger=ledger)
        with self.assertRaisesRegex(verifier.VerificationError, "replay rejected"):
            self.verify(second, replay_ledger=ledger)

    def test_signature_shape_and_padding_are_strict(self):
        receipt = signed_receipt()
        receipt["signature"] += "="
        with self.assertRaisesRegex(verifier.VerificationError, "unpadded base64url"):
            self.verify(receipt)

    def test_duplicate_json_member_is_rejected_before_parsing(self):
        path = Path(self.temp.name) / "duplicate.json"
        path.write_text('{"payload":{},"payload":{},"signature":"x"}', encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate JSON member"):
            verifier.load_json(path)

    def test_authorization_decision_binds_exact_proposal_digest(self):
        receipt = signed_receipt(
            role="authorization_decision",
            proposal_id="containment-1",
            proposal_digest="d" * 64,
        )
        trust = policy(["authorization_decision"])
        expected = context(receipt)
        expected["proposal_digest"] = "e" * 64
        with self.assertRaisesRegex(verifier.VerificationError, "expected context field: proposal_digest"):
            self.verify(receipt, trust=trust, expected=expected)

    def test_authorization_decision_requires_trusted_role_and_proposal_binding(self):
        receipt = signed_receipt(
            role="authorization_decision",
            proposal_id="containment-1",
            proposal_digest="d" * 64,
        )
        with self.assertRaisesRegex(verifier.VerificationError, "not authorized for receipt role"):
            self.verify(receipt, trust=policy(["security_verdict"]))

    def test_invalid_receipt_does_not_write_replay_row(self):
        receipt = signed_receipt()
        receipt["payload"]["status"] = "fail"
        ledger = verifier.ReplayLedger(self.replay_db)
        with self.assertRaises(verifier.VerificationError):
            self.verify(receipt, replay_ledger=ledger)
        self.assertFalse(self.replay_db.exists())

    def test_policy_duplicate_key_id_is_rejected(self):
        trust = policy()
        trust["keys"].append(copy.deepcopy(trust["keys"][0]))
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate key_id"):
            self.verify(trust=trust)

    def test_bool_is_not_accepted_as_scenario_revision(self):
        receipt = signed_receipt()
        receipt["payload"]["scenario_revision"] = True
        with self.assertRaises(verifier.VerificationError):
            self.verify(receipt)

    def test_noncanonical_context_nonce_is_rejected(self):
        receipt = signed_receipt()
        expected = context(receipt)
        expected["run_nonce"] += "="
        with self.assertRaisesRegex(verifier.VerificationError, "unpadded base64url"):
            self.verify(receipt, expected=expected)


if __name__ == "__main__":
    unittest.main()
