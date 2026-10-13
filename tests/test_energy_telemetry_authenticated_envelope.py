"""Tests for signed, encrypted satellite telemetry envelopes."""
from __future__ import annotations

import base64
from datetime import datetime
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.energy_telemetry.authenticated_envelope import (
    EnvelopeAuthenticationError,
    EnvelopeError,
    TelemetryMetadata,
    create_ack,
    open_envelope,
    seal_payload,
    verify_ack,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


class AuthenticatedEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.encryption_key = bytes(range(32))
        self.signing_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        self.encryption_keys = {"enc-epoch-7": self.encryption_key}
        self.verification_keys = {"signer-rotating-2": self.signing_key.public_key()}
        self.metadata = TelemetryMetadata(
            event_id="a" * 32,
            domain="energy",
            source_sequence=17,
            observed_at_utc=dt("2026-10-13T06:00:00Z"),
            expires_at_utc=dt("2026-10-13T06:15:00Z"),
            max_action_age_seconds=300,
            priority=2,
            quality="validated",
        )
        self.payload = b"cohort_energy_wh=123456;coverage=0.98"

    def seal(self, payload=None):
        return seal_payload(
            metadata=self.metadata,
            payload=self.payload if payload is None else payload,
            encryption_key_id="enc-epoch-7",
            encryption_key=self.encryption_key,
            signer_key_id="signer-rotating-2",
            signing_key=self.signing_key,
        )

    def test_round_trip_preserves_payload_and_authenticated_metadata(self):
        sealed = self.seal()
        opened = open_envelope(
            sealed.envelope_bytes,
            encryption_keys=self.encryption_keys,
            verification_keys=self.verification_keys,
        )
        self.assertEqual(opened.payload, self.payload)
        self.assertEqual(opened.event.event_id, self.metadata.event_id)
        self.assertEqual(opened.event.domain, "energy")
        self.assertEqual(opened.event.source_sequence, 17)
        self.assertEqual(opened.event.payload_sha256, sealed.event.payload_sha256)
        self.assertEqual(opened.event.wire_size_bytes, len(sealed.envelope_bytes))
        self.assertEqual(opened.encryption_key_id, "enc-epoch-7")
        self.assertEqual(opened.signer_key_id, "signer-rotating-2")

    def test_event_metadata_and_payload_are_not_plaintext_in_outer_envelope(self):
        sealed = self.seal()
        self.assertNotIn(b"energy", sealed.envelope_bytes)
        self.assertNotIn(self.payload, sealed.envelope_bytes)
        self.assertNotIn(b"source_sequence", sealed.envelope_bytes)
        self.assertNotIn(b"2026-10-13T06:00:00Z", sealed.envelope_bytes)

    def test_sealed_event_digest_binds_exact_outer_wire_bytes(self):
        sealed = self.seal()
        self.assertEqual(sealed.event.payload_sha256, __import__("hashlib").sha256(sealed.envelope_bytes).hexdigest())
        sealed.validate_binding()

    def test_payload_changes_change_ciphertext_and_wire_digest(self):
        first = self.seal(b"reading-A")
        second = self.seal(b"reading-B")
        self.assertNotEqual(first.envelope_bytes, second.envelope_bytes)
        self.assertNotEqual(first.event.payload_sha256, second.event.payload_sha256)

    def test_signature_mutation_is_rejected_before_decryption(self):
        sealed = self.seal()
        value = json.loads(sealed.envelope_bytes)
        value["nonce"] = base64.b64encode(b"Z" * 12).decode("ascii")
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "signature verification failed"):
            open_envelope(canonical(value), encryption_keys=self.encryption_keys, verification_keys=self.verification_keys)

    def test_valid_signature_with_tampered_ciphertext_fails_aead(self):
        sealed = self.seal()
        value = json.loads(sealed.envelope_bytes)
        ciphertext = bytearray(base64.b64decode(value["ciphertext"], validate=True))
        ciphertext[-1] ^= 1
        value["ciphertext"] = base64.b64encode(bytes(ciphertext)).decode("ascii")
        unsigned = dict(value)
        unsigned.pop("signature")
        value["signature"] = base64.b64encode(self.signing_key.sign(canonical(unsigned))).decode("ascii")
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "AEAD authentication failed"):
            open_envelope(canonical(value), encryption_keys=self.encryption_keys, verification_keys=self.verification_keys)

    def test_wrong_encryption_key_fails_aead(self):
        sealed = self.seal()
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "AEAD authentication failed"):
            open_envelope(
                sealed.envelope_bytes,
                encryption_keys={"enc-epoch-7": b"x" * 32},
                verification_keys=self.verification_keys,
            )

    def test_missing_trusted_signer_fails_closed(self):
        sealed = self.seal()
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "no trusted Ed25519"):
            open_envelope(sealed.envelope_bytes, encryption_keys=self.encryption_keys, verification_keys={})

    def test_missing_decryption_key_fails_closed(self):
        sealed = self.seal()
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "no valid decryption key"):
            open_envelope(sealed.envelope_bytes, encryption_keys={}, verification_keys=self.verification_keys)

    def test_unknown_fields_are_rejected(self):
        sealed = self.seal()
        value = json.loads(sealed.envelope_bytes)
        value["recipient"] = "should-not-be-here"
        unsigned = dict(value)
        unsigned.pop("signature")
        value["signature"] = base64.b64encode(self.signing_key.sign(canonical(unsigned))).decode("ascii")
        with self.assertRaisesRegex(EnvelopeError, "unknown or missing fields"):
            open_envelope(canonical(value), encryption_keys=self.encryption_keys, verification_keys=self.verification_keys)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(EnvelopeError, "duplicate JSON object key"):
            open_envelope(
                b'{"version":"telemetry-protected/v1","version":"telemetry-protected/v1"}',
                encryption_keys=self.encryption_keys,
                verification_keys=self.verification_keys,
            )

    def test_noncanonical_outer_json_is_rejected(self):
        sealed = self.seal()
        with self.assertRaisesRegex(EnvelopeError, "not canonical JSON"):
            open_envelope(sealed.envelope_bytes + b" ", encryption_keys=self.encryption_keys, verification_keys=self.verification_keys)

    def test_malformed_base64_and_nonce_lengths_are_rejected(self):
        sealed = self.seal()
        for change in (
            lambda value: value.update(nonce="%%%"),
            lambda value: value.update(nonce=base64.b64encode(b"short").decode("ascii")),
        ):
            value = json.loads(sealed.envelope_bytes)
            change(value)
            unsigned = dict(value)
            unsigned.pop("signature")
            value["signature"] = base64.b64encode(self.signing_key.sign(canonical(unsigned))).decode("ascii")
            with self.subTest(value=value["nonce"]):
                with self.assertRaises(EnvelopeError):
                    open_envelope(canonical(value), encryption_keys=self.encryption_keys, verification_keys=self.verification_keys)

    def test_invalid_metadata_is_rejected_before_encryption(self):
        bad = TelemetryMetadata(
            event_id="private-id",
            domain="energy",
            source_sequence=0,
            observed_at_utc=dt("2026-10-13T06:00:00Z"),
            expires_at_utc=dt("2026-10-13T06:15:00Z"),
            max_action_age_seconds=300,
            priority=1,
            quality="validated",
        )
        with self.assertRaises(EnvelopeError):
            seal_payload(
                metadata=bad,
                payload=self.payload,
                encryption_key_id="enc-epoch-7",
                encryption_key=self.encryption_key,
                signer_key_id="signer-rotating-2",
                signing_key=self.signing_key,
            )

    def test_key_sizes_and_key_ids_are_fail_closed(self):
        args = dict(
            metadata=self.metadata, payload=self.payload,
            encryption_key_id="enc-epoch-7", encryption_key=self.encryption_key,
            signer_key_id="signer-rotating-2", signing_key=self.signing_key,
        )
        with self.assertRaises(EnvelopeError):
            seal_payload(**{**args, "encryption_key": b"small"})
        with self.assertRaises(EnvelopeError):
            seal_payload(**{**args, "signer_key_id": "household alice"})
        with self.assertRaises(EnvelopeError):
            seal_payload(**{**args, "encryption_key_id": ""})

    def test_envelope_size_limit_fails_closed(self):
        with self.assertRaisesRegex(EnvelopeError, "maximum wire size"):
            self.seal(b"p" * 70_000)

    def test_empty_payload_round_trips(self):
        sealed = self.seal(b"")
        opened = open_envelope(
            sealed.envelope_bytes,
            encryption_keys=self.encryption_keys,
            verification_keys=self.verification_keys,
        )
        self.assertEqual(opened.payload, b"")

    def make_ack(self):
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack = create_ack(
            receiver_key_id="receiver-epoch-4",
            event_id=self.metadata.event_id,
            payload_sha256="d" * 64,
            acknowledged_at_utc=dt("2026-10-13T06:03:00Z"),
            signing_key=receiver_key,
        )
        return ack, receiver_key

    def test_signed_receiver_ack_round_trips(self):
        ack_bytes, receiver_key = self.make_ack()
        ack = verify_ack(ack_bytes, verification_keys={"receiver-epoch-4": receiver_key.public_key()})
        self.assertEqual(ack.receiver_key_id, "receiver-epoch-4")
        self.assertEqual(ack.event_id, self.metadata.event_id)
        self.assertEqual(ack.payload_sha256, "d" * 64)
        self.assertEqual(ack.disposition, "accepted")
        self.assertEqual(ack.acknowledged_at_utc, dt("2026-10-13T06:03:00Z"))

    def test_unknown_receiver_key_cannot_acknowledge(self):
        ack_bytes, _receiver_key = self.make_ack()
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "no trusted Ed25519 key"):
            verify_ack(ack_bytes, verification_keys={})

    def test_modified_ack_digest_is_rejected_by_signature(self):
        import json

        ack_bytes, receiver_key = self.make_ack()
        value = json.loads(ack_bytes)
        value["payload_sha256"] = "e" * 64
        forged = canonical(value)
        with self.assertRaisesRegex(EnvelopeAuthenticationError, "signature verification failed"):
            verify_ack(forged, verification_keys={"receiver-epoch-4": receiver_key.public_key()})

    def test_ack_is_canonical_and_rejects_unknown_fields(self):
        import json

        ack_bytes, receiver_key = self.make_ack()
        with self.assertRaisesRegex(EnvelopeError, "not canonical JSON"):
            verify_ack(ack_bytes + b" ", verification_keys={"receiver-epoch-4": receiver_key.public_key()})
        value = json.loads(ack_bytes)
        value["free_form_identity"] = "private"
        with self.assertRaisesRegex(EnvelopeError, "unknown or missing fields"):
            verify_ack(canonical(value), verification_keys={"receiver-epoch-4": receiver_key.public_key()})

    def test_ack_creation_rejects_invalid_digest(self):
        with self.assertRaises(EnvelopeError):
            create_ack(
                receiver_key_id="receiver-epoch-4",
                event_id=self.metadata.event_id,
                payload_sha256="not-a-digest",
                acknowledged_at_utc=dt("2026-10-13T06:03:00Z"),
                signing_key=self.signing_key,
            )


if __name__ == "__main__":
    unittest.main()
