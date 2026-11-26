"""End-to-end synthetic sender/receiver telemetry delivery contract."""
from __future__ import annotations

from datetime import datetime
import base64
import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.energy_telemetry.authenticated_envelope import TelemetryMetadata, seal_payload
from tools.energy_telemetry.durable_queue import DurableTelemetryQueue
from tools.energy_telemetry.receiver_inbox import ReceiverInbox
from tools.energy_telemetry.store_forward import TelemetryQueueError


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


class TelemetryPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.sender_db = str(root / "sender.sqlite3")
        self.receiver_db = str(root / "receiver.sqlite3")
        self.now = dt("2026-10-13T06:00:00Z")
        self.encryption_key = bytes(range(32))
        self.sender_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        self.receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        self.encryption_keys = {"enc-integration-epoch": self.encryption_key}
        self.sender_verification_keys = {"sender-integration": self.sender_key.public_key()}
        self.receiver_verification_keys = {"receiver-integration": self.receiver_key.public_key()}
        self.sender = DurableTelemetryQueue(self.sender_db)
        self.receiver = ReceiverInbox(self.receiver_db)

    def tearDown(self):
        self.temp.cleanup()

    def seal(self, event_id="a", payload=b"synthetic cohort total=4100 Wh"):
        metadata = TelemetryMetadata(
            event_id=event_id * 32,
            domain="energy",
            source_sequence=9,
            observed_at_utc=dt("2026-10-13T05:59:00Z"),
            expires_at_utc=dt("2026-10-13T06:15:00Z"),
            max_action_age_seconds=300,
            priority=2,
            quality="validated",
        )
        return seal_payload(
            metadata=metadata,
            payload=payload,
            encryption_key_id="enc-integration-epoch",
            encryption_key=self.encryption_key,
            signer_key_id="sender-integration",
            signing_key=self.sender_key,
        )

    def receiver_accept(self, envelope_bytes):
        return self.receiver.accept_and_ack(
            envelope_bytes,
            encryption_keys=self.encryption_keys,
            verification_keys=self.sender_verification_keys,
            receiver_key_id="receiver-integration",
            receiver_signing_key=self.receiver_key,
            now_utc=self.now,
        )

    def test_end_to_end_seal_persist_deliver_commit_ack_and_clear_pending(self):
        sealed = self.seal()
        self.assertEqual(self.sender.enqueue_protected(sealed, now_utc=self.now), "queued")
        selected = self.sender.select_protected_batch(now_utc=self.now, max_bytes=4096)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].envelope_bytes, sealed.envelope_bytes)

        accepted = self.receiver_accept(selected[0].envelope_bytes)
        self.assertFalse(accepted.duplicate)
        self.assertEqual(accepted.payload, b"synthetic cohort total=4100 Wh")
        self.assertEqual(self.receiver.stored_envelope(sealed.event.event_id), sealed.envelope_bytes)

        self.assertEqual(
            self.sender.acknowledge_protected(
                accepted.ack_bytes,
                verification_keys=self.receiver_verification_keys,
                now_utc=self.now,
            ),
            "acknowledged",
        )
        self.assertEqual(self.sender.pending_count, 0)
        self.assertEqual(self.receiver.record_count, 1)
        self.assertEqual(self.sender.integrity_check(), "ok")
        self.assertEqual(self.receiver.integrity_check(), "ok")

    def test_lost_ack_retries_exact_ciphertext_and_receiver_deduplicates(self):
        sealed = self.seal()
        self.sender.enqueue_protected(sealed, now_utc=self.now)

        first = self.sender.select_protected_batch(now_utc=self.now, max_bytes=4096)[0]
        first_acceptance = self.receiver_accept(first.envelope_bytes)
        # Simulate loss: do not forward first_acceptance.ack_bytes to the sender.
        self.assertEqual(self.sender.pending_count, 1)

        second = self.sender.select_protected_batch(now_utc=self.now, max_bytes=4096)[0]
        self.assertEqual(second.envelope_bytes, first.envelope_bytes)
        self.assertEqual(second.attempt_count, 2)
        second_acceptance = self.receiver_accept(second.envelope_bytes)
        self.assertTrue(second_acceptance.duplicate)
        self.assertEqual(self.receiver.record_count, 1)

        self.assertEqual(
            self.sender.acknowledge_protected(
                second_acceptance.ack_bytes,
                verification_keys=self.receiver_verification_keys,
                now_utc=self.now,
            ),
            "acknowledged",
        )
        self.assertEqual(self.sender.pending_count, 0)
        # The earlier ACK may arrive late after the successful retry; exact
        # signed ACK retransmission is idempotent when its signed time is valid.
        self.assertEqual(
            self.sender.acknowledge_protected(
                first_acceptance.ack_bytes,
                verification_keys=self.receiver_verification_keys,
                now_utc=self.now,
            ),
            "idempotent_ack",
        )

    def test_tampered_wire_message_never_reaches_receiver_commit_or_sender_ack(self):
        sealed = self.seal()
        self.sender.enqueue_protected(sealed, now_utc=self.now)
        selected = self.sender.select_protected_batch(now_utc=self.now, max_bytes=4096)[0]
        value = json.loads(selected.envelope_bytes)
        signature = bytearray(base64.b64decode(value["signature"], validate=True))
        signature[-1] ^= 1
        value["signature"] = base64.b64encode(bytes(signature)).decode("ascii")
        tampered = canonical(value)

        with self.assertRaisesRegex(TelemetryQueueError, "authentication failed"):
            self.receiver_accept(tampered)
        self.assertEqual(self.receiver.record_count, 0)
        self.assertEqual(self.sender.pending_count, 1)
        # No ACK exists to present; legitimate bytes remain available for retry.
        retry = self.sender.select_protected_batch(now_utc=self.now, max_bytes=4096)[0]
        self.assertEqual(retry.envelope_bytes, sealed.envelope_bytes)
        self.assertEqual(retry.attempt_count, 2)


if __name__ == "__main__":
    unittest.main()
