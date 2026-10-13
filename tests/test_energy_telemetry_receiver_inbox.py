"""Receiver inbox tests for commit-before-ACK and replay-safe acceptance."""
from __future__ import annotations

from datetime import datetime
import base64
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
import stat

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlite3 import IntegrityError

from tools.energy_telemetry.authenticated_envelope import (
    EnvelopeAuthenticationError,
    TelemetryMetadata,
    open_envelope,
    seal_payload,
    verify_ack,
)
from tools.energy_telemetry.receiver_inbox import ReceiverInbox
from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    TelemetryQueueError,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sealed(event_id="a", *, payload=b"synthetic-energy-aggregate",
           observed="2026-10-12T10:00:00Z", expires="2026-10-12T11:00:00Z",
           max_age=900, quality="validated"):
    metadata = TelemetryMetadata(
        event_id=event_id * 32,
        domain="energy",
        source_sequence=5,
        observed_at_utc=dt(observed),
        expires_at_utc=dt(expires),
        max_action_age_seconds=max_age,
        priority=2,
        quality=quality,
    )
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    return seal_payload(
        metadata=metadata,
        payload=payload,
        encryption_key_id="enc-receiver-test",
        encryption_key=bytes(range(32)),
        signer_key_id="sender-test",
        signing_key=key,
    )


class ReceiverInboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp.name) / "receiver.sqlite3")
        self.now = dt("2026-10-12T10:05:00Z")
        self.sender_signing_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        self.receiver_signing_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        self.encryption_keys = {"enc-receiver-test": bytes(range(32))}
        self.sender_verification_keys = {"sender-test": self.sender_signing_key.public_key()}
        self.receiver_verification_keys = {"receiver-test": self.receiver_signing_key.public_key()}
        self.inbox = ReceiverInbox(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def reopen(self, **kwargs):
        kwargs.setdefault("max_records", 100_000)
        kwargs.setdefault("max_envelope_bytes", 65_536)
        kwargs.setdefault("max_future_skew_seconds", 300)
        return ReceiverInbox(self.database, **kwargs)

    def accept(self, packet, *, now=None, inbox=None):
        return (self.inbox if inbox is None else inbox).accept_and_ack(
            packet.envelope_bytes,
            encryption_keys=self.encryption_keys,
            verification_keys=self.sender_verification_keys,
            receiver_key_id="receiver-test",
            receiver_signing_key=self.receiver_signing_key,
            now_utc=self.now if now is None else now,
        )

    def test_first_accept_commits_ciphertext_then_returns_signed_ack(self):
        packet = sealed(payload=b"private-but-synthetic-meter-aggregate")
        result = self.accept(packet)
        self.assertFalse(result.duplicate)
        self.assertEqual(result.payload, b"private-but-synthetic-meter-aggregate")
        self.assertEqual(self.inbox.record_count, 1)
        self.assertEqual(self.inbox.stored_envelope(result.event_id), packet.envelope_bytes)
        ack = verify_ack(result.ack_bytes, verification_keys=self.receiver_verification_keys)
        self.assertEqual(ack.event_id, packet.event.event_id)
        self.assertEqual(ack.payload_sha256, packet.event.payload_sha256)
        self.assertEqual(ack.acknowledged_at_utc, self.now)
        self.assertEqual(ack.disposition, "accepted")

    def test_receiver_persists_ciphertext_not_plaintext_payload(self):
        private_payload = b"synthetic-household-secret-value"
        packet = sealed(payload=private_payload)
        self.accept(packet)
        self.assertNotIn(private_payload, self.inbox.stored_envelope(packet.event.event_id))
        for path in (self.database, self.database + "-wal", self.database + "-shm"):
            if Path(path).exists():
                self.assertNotIn(private_payload, Path(path).read_bytes())
        self.assertEqual(self.inbox.integrity_check(), "ok")

    def test_duplicate_exact_bytes_after_restart_is_idempotently_acked(self):
        packet = sealed()
        first = self.accept(packet)
        second_inbox = self.reopen()
        second = self.accept(packet, inbox=second_inbox)
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(second.payload, b"synthetic-energy-aggregate")
        self.assertEqual(second_inbox.record_count, 1)

    def test_replayed_event_id_with_different_ciphertext_is_rejected(self):
        first_packet = sealed(payload=b"same event content")
        second_packet = sealed(payload=b"same event content")
        self.assertEqual(first_packet.event.event_id, second_packet.event.event_id)
        self.assertNotEqual(first_packet.envelope_bytes, second_packet.envelope_bytes)
        self.accept(first_packet)
        with self.assertRaises(EventIdentityConflict):
            self.accept(second_packet)
        self.assertEqual(self.inbox.record_count, 1)

    def test_unknown_sender_key_is_rejected_without_commit(self):
        packet = sealed()
        with self.assertRaisesRegex(TelemetryQueueError, "authentication failed"):
            self.inbox.accept_and_ack(
                packet.envelope_bytes,
                encryption_keys=self.encryption_keys,
                verification_keys={},
                receiver_key_id="receiver-test",
                receiver_signing_key=self.receiver_signing_key,
                now_utc=self.now,
            )
        self.assertEqual(self.inbox.record_count, 0)

    def test_wrong_encryption_key_is_rejected_without_commit(self):
        packet = sealed()
        with self.assertRaisesRegex(TelemetryQueueError, "authentication failed"):
            self.inbox.accept_and_ack(
                packet.envelope_bytes,
                encryption_keys={"enc-receiver-test": b"x" * 32},
                verification_keys=self.sender_verification_keys,
                receiver_key_id="receiver-test",
                receiver_signing_key=self.receiver_signing_key,
                now_utc=self.now,
            )
        self.assertEqual(self.inbox.record_count, 0)

    def test_signed_but_expired_envelope_is_rejected(self):
        packet = sealed(expires="2026-10-12T10:04:59Z")
        with self.assertRaisesRegex(TelemetryQueueError, "expired envelope"):
            self.accept(packet)
        self.assertEqual(self.inbox.record_count, 0)

    def test_future_dated_envelope_is_rejected(self):
        packet = sealed(observed="2026-10-12T10:20:00Z")
        with self.assertRaisesRegex(TelemetryQueueError, "future clock skew"):
            self.accept(packet)
        self.assertEqual(self.inbox.record_count, 0)

    def test_stale_but_unexpired_envelope_is_stored_but_not_actionable(self):
        packet = sealed(max_age=30)
        result = self.accept(packet)
        self.assertEqual(result.evidence_state, "stale")
        self.assertFalse(result.actionable)
        self.assertEqual(self.inbox.record_count, 1)
        self.assertEqual(
            verify_ack(result.ack_bytes, verification_keys=self.receiver_verification_keys).payload_sha256,
            packet.event.payload_sha256,
        )

    def test_record_limit_fails_closed(self):
        path = str(Path(self.temp.name) / "limited.sqlite3")
        limited = ReceiverInbox(path, max_records=1)
        first = sealed(event_id="a")
        limited.accept_and_ack(
            first.envelope_bytes,
            encryption_keys=self.encryption_keys,
            verification_keys=self.sender_verification_keys,
            receiver_key_id="receiver-test",
            receiver_signing_key=self.receiver_signing_key,
            now_utc=self.now,
        )
        second = sealed(event_id="b")
        with self.assertRaises(QueueCapacityError):
            limited.accept_and_ack(
                second.envelope_bytes,
                encryption_keys=self.encryption_keys,
                verification_keys=self.sender_verification_keys,
                receiver_key_id="receiver-test",
                receiver_signing_key=self.receiver_signing_key,
                now_utc=self.now,
            )
        self.assertEqual(limited.record_count, 1)

    def test_configuration_mismatch_fails_closed(self):
        with self.assertRaisesRegex(TelemetryQueueError, "configuration mismatch"):
            self.reopen(max_records=999)
        with self.assertRaisesRegex(TelemetryQueueError, "configuration mismatch"):
            self.reopen(max_future_skew_seconds=299)

    def test_unowned_sqlite_database_with_unrelated_table_is_not_claimed(self):
        path = str(Path(self.temp.name) / "other.sqlite3")
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE private_business_data (id INTEGER PRIMARY KEY)")
        with self.assertRaisesRegex(TelemetryQueueError, "unowned SQLite database"):
            ReceiverInbox(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0].lower(), "delete")
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )}
        self.assertEqual(tables, {"private_business_data"})

    def test_foreign_application_id_is_rejected(self):
        with sqlite3.connect(self.database) as db:
            db.execute("PRAGMA application_id = 12345")
        with self.assertRaisesRegex(TelemetryQueueError, "different application"):
            self.reopen()

    def test_file_permissions_are_private_on_posix(self):
        packet = sealed()
        self.accept(packet)
        if os.name == "posix":
            for path in (self.database, self.database + "-wal", self.database + "-shm"):
                if Path(path).exists():
                    self.assertEqual(stat.S_IMODE(os.stat(path).st_mode) & 0o077, 0)

    def test_concurrent_duplicate_accepts_create_one_durable_record(self):
        packet = sealed()
        workers = 8
        barrier = threading.Barrier(workers)
        results = []
        errors = []
        lock = threading.Lock()

        def submit():
            try:
                local_inbox = self.reopen()
                barrier.wait(timeout=5)
                result = self.accept(packet, inbox=local_inbox)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(self.inbox.record_count, 1)
        self.assertEqual(sum(not r.duplicate for r in results), 1)
        self.assertEqual(sum(r.duplicate for r in results), workers - 1)
        for result in results:
            ack = verify_ack(result.ack_bytes, verification_keys=self.receiver_verification_keys)
            self.assertEqual(ack.payload_sha256, packet.event.payload_sha256)

    def test_bad_outer_signature_is_rejected_before_insert(self):
        packet = sealed()
        value = json.loads(packet.envelope_bytes)
        signature = bytearray(base64.b64decode(value["signature"], validate=True))
        signature[0] ^= 1
        value["signature"] = base64.b64encode(bytes(signature)).decode("ascii")
        forged = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        with self.assertRaisesRegex(TelemetryQueueError, "authentication failed"):
            self.accept(type(packet)(event=packet.event, envelope_bytes=forged))
        self.assertEqual(self.inbox.record_count, 0)

    def test_receiver_ack_is_not_created_if_commit_fails(self):
        packet = sealed()
        # A deterministic abort trigger simulates failure at the durable
        # acceptance write. No positive ACK may be returned for this input.
        with sqlite3.connect(self.database) as db:
            db.execute(
                "CREATE TRIGGER reject_receiver_insert BEFORE INSERT ON received_envelopes "
                "BEGIN SELECT RAISE(ABORT, 'forced receiver commit failure'); END"
            )
        with self.assertRaises(IntegrityError):
            self.accept(packet)
        self.assertEqual(self.inbox.record_count, 0)


if __name__ == "__main__":
    unittest.main()
