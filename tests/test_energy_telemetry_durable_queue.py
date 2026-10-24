"""Restart, transaction and capacity tests for the SQLite telemetry ledger."""
from __future__ import annotations

from datetime import datetime
import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
import stat

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.energy_telemetry.authenticated_envelope import TelemetryMetadata, create_ack, open_envelope, seal_payload
from tools.energy_telemetry.durable_queue import DurableTelemetryQueue
from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    TelemetryEvent,
    TelemetryQueueError,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event(event_id="a", *, domain="energy", observed="2026-10-12T10:00:00Z",
          expires="2026-10-12T11:00:00Z", age=900, priority=1, quality="validated",
          digest="b", size=100, sequence=1):
    return TelemetryEvent(
        event_id=event_id * 32,
        domain=domain,
        source_sequence=sequence,
        observed_at_utc=dt(observed),
        expires_at_utc=dt(expires),
        max_action_age_seconds=age,
        priority=priority,
        quality=quality,
        payload_sha256=digest * 64,
        wire_size_bytes=size,
    )


def protected(event_id="a", *, payload=b"test energy payload", observed="2026-10-12T10:00:00Z", expires="2026-10-12T11:00:00Z", priority=1):
    metadata = TelemetryMetadata(
        event_id=event_id * 32,
        domain="energy",
        source_sequence=3,
        observed_at_utc=dt(observed),
        expires_at_utc=dt(expires),
        max_action_age_seconds=900,
        priority=priority,
        quality="validated",
    )
    signing_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    return seal_payload(
        metadata=metadata,
        payload=payload,
        encryption_key_id="enc-test-epoch",
        encryption_key=bytes(range(32)),
        signer_key_id="signer-test-epoch",
        signing_key=signing_key,
    )


class DurableQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp.name) / "telemetry.sqlite3")
        self.now = dt("2026-10-12T10:05:00Z")
        self.queue = DurableTelemetryQueue(self.database, max_queue_bytes=1000)

    def tearDown(self):
        self.temp.cleanup()

    def reopen(self, **kwargs):
        kwargs.setdefault("max_queue_bytes", 1000)
        return DurableTelemetryQueue(self.database, **kwargs)

    def test_event_survives_restart_and_pending_bytes_are_rebuilt(self):
        e = event()
        self.assertEqual(self.queue.enqueue(e, now_utc=self.now), "queued")
        other = self.reopen()
        self.assertEqual(other.pending_count, 1)
        self.assertEqual(other.pending_bytes, 100)
        self.assertEqual(other.enqueue(e, now_utc=self.now), "duplicate_pending")

    def test_selected_event_is_resent_after_restart_until_ack(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        first = self.queue.select_batch(now_utc=self.now, max_bytes=100)
        self.assertEqual(self.queue.attempt_count(e.event_id), 1)
        other = self.reopen()
        second = other.select_batch(now_utc=self.now, max_bytes=100)
        self.assertEqual(first, second)
        self.assertEqual(other.attempt_count(e.event_id), 2)
        self.assertEqual(
            other.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now),
            "acknowledged",
        )
        third = self.reopen()
        self.assertEqual(third.pending_count, 0)
        self.assertEqual(
            third.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now),
            "idempotent_ack",
        )

    def test_changed_metadata_on_replayed_event_id_fails_closed_after_restart(self):
        self.queue.enqueue(event(), now_utc=self.now)
        other = self.reopen()
        with self.assertRaises(EventIdentityConflict):
            other.enqueue(event(priority=3), now_utc=self.now)

    def test_terminal_tombstone_survives_restart(self):
        e = event(expires="2026-10-12T10:04:00Z")
        self.assertEqual(self.queue.enqueue(e, now_utc=self.now), "expired_not_queued")
        other = self.reopen()
        self.assertEqual(other.enqueue(e, now_utc=self.now), "already_expired")
        with self.assertRaises(EventIdentityConflict):
            other.enqueue(event(digest="c", expires="2026-10-12T11:00:00Z"), now_utc=self.now)

    def test_expiry_is_transactionally_persisted(self):
        e = event(expires="2026-10-12T10:06:00Z")
        self.queue.enqueue(e, now_utc=self.now)
        ids = self.queue.expire(now_utc=dt("2026-10-12T10:06:00Z"))
        self.assertEqual(ids, (e.event_id,))
        other = self.reopen()
        self.assertEqual(other.pending_count, 0)
        self.assertEqual(other.enqueue(e, now_utc=self.now), "already_expired")

    def test_expired_event_cannot_be_acked_after_restart(self):
        e = event(expires="2026-10-12T10:06:00Z")
        self.queue.enqueue(e, now_utc=self.now)
        self.queue.select_batch(now_utc=self.now, max_bytes=100)
        other = self.reopen()
        with self.assertRaisesRegex(TelemetryQueueError, "expired before ACK"):
            other.acknowledge(
                event_id=e.event_id, payload_sha256=e.payload_sha256,
                now_utc=dt("2026-10-12T10:06:00Z"),
            )
        self.assertEqual(other.expire(now_utc=dt("2026-10-12T10:06:00Z")), (e.event_id,))

    def test_ack_before_selection_and_wrong_digest_do_not_remove_record(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        with self.assertRaisesRegex(TelemetryQueueError, "not selected"):
            self.queue.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now)
        self.queue.select_batch(now_utc=self.now, max_bytes=100)
        with self.assertRaises(EventIdentityConflict):
            self.queue.acknowledge(event_id=e.event_id, payload_sha256="c" * 64, now_utc=self.now)
        self.assertEqual(self.queue.pending_count, 1)

    def test_byte_limit_fails_without_silent_eviction(self):
        limited = DurableTelemetryQueue(str(Path(self.temp.name) / "limited.sqlite3"), max_queue_bytes=100)
        limited.enqueue(event(size=80), now_utc=self.now)
        with self.assertRaises(QueueCapacityError):
            limited.enqueue(event("c", size=30), now_utc=self.now)
        self.assertEqual(limited.pending_count, 1)
        self.assertEqual(limited.pending_bytes, 80)

    def test_terminal_record_limit_fails_closed_instead_of_pruning_tombstones(self):
        tiny = DurableTelemetryQueue(str(Path(self.temp.name) / "tiny.sqlite3"), max_queue_bytes=1000, max_records=1)
        tiny.enqueue(event(expires="2026-10-12T10:04:00Z"), now_utc=self.now)
        with self.assertRaisesRegex(QueueCapacityError, "record limit"):
            tiny.enqueue(event("c"), now_utc=self.now)
        self.assertEqual(tiny.record_count, 1)

    def test_invalid_event_never_commits_a_row(self):
        with self.assertRaises(TelemetryQueueError):
            self.queue.enqueue(event(domain=[]), now_utc=self.now)
        self.assertEqual(self.queue.record_count, 0)

    def test_integrity_check_reports_ok_after_state_transitions(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        self.queue.select_batch(now_utc=self.now, max_bytes=100)
        self.queue.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now)
        self.assertEqual(self.queue.integrity_check(), "ok")
        self.assertEqual(self.reopen().integrity_check(), "ok")

    def test_concurrent_duplicate_enqueues_commit_exactly_one_event(self):
        workers = 12
        barrier = threading.Barrier(workers)
        results = []
        errors = []
        lock = threading.Lock()
        e = event()

        def submit():
            try:
                local_queue = self.reopen()
                barrier.wait(timeout=5)
                result = local_queue.enqueue(e, now_utc=self.now)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(results.count("queued"), 1)
        self.assertEqual(results.count("duplicate_pending"), workers - 1)
        self.assertEqual(self.queue.record_count, 1)
        self.assertEqual(self.queue.pending_bytes, e.wire_size_bytes)

    def test_concurrent_unique_enqueues_respect_byte_limit_atomically(self):
        workers = 8
        barrier = threading.Barrier(workers)
        results = []
        errors = []
        lock = threading.Lock()
        limited = DurableTelemetryQueue(str(Path(self.temp.name) / "concurrent-limited.sqlite3"), max_queue_bytes=250, max_records=100)

        def submit(i):
            try:
                local_queue = limited
                barrier.wait(timeout=5)
                result = local_queue.enqueue(event("01234567"[i], size=100), now_utc=self.now)
                with lock:
                    results.append(result)
            except QueueCapacityError:
                with lock:
                    results.append("capacity_rejected")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(results.count("queued"), 2)
        self.assertEqual(results.count("capacity_rejected"), workers - 2)
        self.assertEqual(limited.pending_bytes, 200)
        self.assertEqual(limited.pending_count, 2)

    def test_concurrent_exact_acknowledgements_are_idempotent(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        self.queue.select_batch(now_utc=self.now, max_bytes=100)
        workers = 8
        barrier = threading.Barrier(workers)
        results = []
        errors = []
        lock = threading.Lock()

        def ack():
            try:
                local_queue = self.reopen()
                barrier.wait(timeout=5)
                result = local_queue.acknowledge(
                    event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now
                )
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=ack) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(results.count("acknowledged"), 1)
        self.assertEqual(results.count("idempotent_ack"), workers - 1)
        self.assertEqual(self.queue.pending_count, 0)
        self.assertEqual(self.queue.record_count, 1)

    def test_ephemeral_database_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "persistent filesystem path"):
            DurableTelemetryQueue(":memory:")
        with self.assertRaisesRegex(ValueError, "persistent filesystem path"):
            DurableTelemetryQueue("bad\x00path")

    def test_policy_configuration_mismatch_fails_closed(self):
        with self.assertRaisesRegex(TelemetryQueueError, "configuration mismatch"):
            DurableTelemetryQueue(self.database, max_queue_bytes=999)
        with self.assertRaisesRegex(TelemetryQueueError, "configuration mismatch"):
            DurableTelemetryQueue(self.database, max_records=99_999)
        with self.assertRaisesRegex(TelemetryQueueError, "configuration mismatch"):
            DurableTelemetryQueue(self.database, max_future_skew_seconds=299)

    def test_schema_v1_migrates_transactionally_to_v3(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DROP TABLE telemetry_queue_config")
            db.execute("ALTER TABLE telemetry_events DROP COLUMN envelope_bytes")
            db.execute("PRAGMA user_version = 1")
        migrated = self.reopen()
        self.assertEqual(migrated.integrity_check(), "ok")
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)
            columns = {row[1] for row in db.execute("PRAGMA table_info(telemetry_events)")}
            config = db.execute(
                "SELECT max_queue_bytes, max_records, max_future_skew_seconds "
                "FROM telemetry_queue_config WHERE singleton = 1"
            ).fetchone()
        self.assertIn("envelope_bytes", columns)
        self.assertEqual(config, (1000, 100_000, 300))

    def test_unknown_schema_version_fails_closed(self):
        with sqlite3.connect(self.database) as db:
            db.execute("PRAGMA user_version = 999")
        with self.assertRaisesRegex(TelemetryQueueError, "unsupported telemetry queue schema"):
            self.reopen()

    def test_protected_envelope_bytes_survive_restart_and_replay(self):
        sealed = protected()
        self.assertEqual(sealed.event.wire_size_bytes, len(sealed.envelope_bytes))
        self.assertEqual(sealed.event.payload_sha256, hashlib.sha256(sealed.envelope_bytes).hexdigest())
        self.assertEqual(self.queue.enqueue_protected(sealed, now_utc=self.now), "queued")
        # Metadata-only selection is deliberately not allowed to select ciphertext rows.
        self.assertEqual(self.queue.select_batch(now_utc=self.now, max_bytes=1000), ())
        reopened = self.reopen()
        batch = reopened.select_protected_batch(now_utc=self.now, max_bytes=1000)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].envelope_bytes, sealed.envelope_bytes)
        self.assertEqual(batch[0].event, sealed.event)
        opened = open_envelope(
            batch[0].envelope_bytes,
            encryption_keys={"enc-test-epoch": bytes(range(32))},
            verification_keys={"signer-test-epoch": Ed25519PrivateKey.from_private_bytes(bytes(range(32))).public_key()},
        )
        self.assertEqual(opened.payload, b"test energy payload")
        self.assertEqual(batch[0].attempt_count, 1)
        batch_again = self.reopen().select_protected_batch(now_utc=self.now, max_bytes=1000)
        self.assertEqual(batch_again[0].envelope_bytes, sealed.envelope_bytes)
        self.assertEqual(batch_again[0].attempt_count, 2)
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack_bytes = create_ack(
            receiver_key_id="receiver-test-epoch",
            event_id=sealed.event.event_id,
            payload_sha256=sealed.event.payload_sha256,
            acknowledged_at_utc=self.now,
            signing_key=receiver_key,
        )
        with self.assertRaisesRegex(TelemetryQueueError, "requires an authenticated receiver ACK"):
            self.reopen().acknowledge(
                event_id=sealed.event.event_id,
                payload_sha256=sealed.event.payload_sha256,
                now_utc=self.now,
            )
        self.assertEqual(
            self.reopen().acknowledge_protected(
                ack_bytes,
                verification_keys={"receiver-test-epoch": receiver_key.public_key()},
                now_utc=self.now,
            ),
            "acknowledged",
        )
        self.assertEqual(self.reopen().select_protected_batch(now_utc=self.now, max_bytes=1000), ())

    def test_protected_envelope_digest_or_size_tampering_is_rejected(self):
        sealed = protected()
        damaged = type(sealed)(event=sealed.event, envelope_bytes=sealed.envelope_bytes + b"x")
        with self.assertRaisesRegex(TelemetryQueueError, "binding is invalid"):
            self.queue.enqueue_protected(damaged, now_utc=self.now)
        self.assertEqual(self.queue.record_count, 0)

    def test_metadata_only_event_cannot_be_replayed_as_protected_payload(self):
        sealed = protected()
        self.queue.enqueue(sealed.event, now_utc=self.now)
        with self.assertRaises(EventIdentityConflict):
            self.queue.enqueue_protected(sealed, now_utc=self.now)

    def test_protected_enqueue_respects_actual_wire_byte_budget(self):
        sealed = protected()
        limited_path = str(Path(self.temp.name) / "protected-limited.sqlite3")
        limited = DurableTelemetryQueue(limited_path, max_queue_bytes=sealed.event.wire_size_bytes - 1)
        with self.assertRaises(QueueCapacityError):
            limited.enqueue_protected(sealed, now_utc=self.now)
        self.assertEqual(limited.pending_count, 0)

    def test_protected_event_rejects_unknown_or_forged_receiver_ack(self):
        sealed = protected()
        self.queue.enqueue_protected(sealed, now_utc=self.now)
        self.queue.select_protected_batch(now_utc=self.now, max_bytes=1000)
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack = create_ack(
            receiver_key_id="receiver-test-epoch",
            event_id=sealed.event.event_id,
            payload_sha256=sealed.event.payload_sha256,
            acknowledged_at_utc=self.now,
            signing_key=receiver_key,
        )
        with self.assertRaisesRegex(ValueError, "no trusted Ed25519 key"):
            self.queue.acknowledge_protected(ack, verification_keys={}, now_utc=self.now)
        self.assertEqual(self.queue.pending_count, 1)

        ack_object = json.loads(ack)
        signature = bytearray(base64.b64decode(ack_object["signature"], validate=True))
        signature[0] ^= 1
        ack_object["signature"] = base64.b64encode(bytes(signature)).decode("ascii")
        forged_ack = json.dumps(
            ack_object, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            self.queue.acknowledge_protected(
                forged_ack,
                verification_keys={"receiver-test-epoch": receiver_key.public_key()},
                now_utc=self.now,
            )
        self.assertEqual(self.queue.pending_count, 1)

    def test_signed_ack_for_wrong_digest_does_not_remove_protected_event(self):
        sealed = protected()
        self.queue.enqueue_protected(sealed, now_utc=self.now)
        self.queue.select_protected_batch(now_utc=self.now, max_bytes=1000)
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack = create_ack(
            receiver_key_id="receiver-test-epoch",
            event_id=sealed.event.event_id,
            payload_sha256="c" * 64,
            acknowledged_at_utc=self.now,
            signing_key=receiver_key,
        )
        with self.assertRaises(EventIdentityConflict):
            self.queue.acknowledge_protected(
                ack,
                verification_keys={"receiver-test-epoch": receiver_key.public_key()},
                now_utc=self.now,
            )
        self.assertEqual(self.queue.pending_count, 1)

    def test_future_dated_signed_ack_is_rejected(self):
        sealed = protected()
        self.queue.enqueue_protected(sealed, now_utc=self.now)
        self.queue.select_protected_batch(now_utc=self.now, max_bytes=1000)
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack = create_ack(
            receiver_key_id="receiver-test-epoch",
            event_id=sealed.event.event_id,
            payload_sha256=sealed.event.payload_sha256,
            acknowledged_at_utc=dt("2026-10-12T12:00:00Z"),
            signing_key=receiver_key,
        )
        with self.assertRaisesRegex(TelemetryQueueError, "ACK timestamp exceeds"):
            self.queue.acknowledge_protected(
                ack,
                verification_keys={"receiver-test-epoch": receiver_key.public_key()},
                now_utc=self.now,
            )
        self.assertEqual(self.queue.pending_count, 1)

    def test_signed_ack_claiming_acceptance_after_expiry_is_rejected(self):
        expired_soon = protected(expires="2026-10-12T10:24:00Z")
        self.queue.enqueue_protected(expired_soon, now_utc=self.now)
        self.queue.select_protected_batch(now_utc=self.now, max_bytes=1000)
        receiver_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        ack = create_ack(
            receiver_key_id="receiver-test-epoch",
            event_id=expired_soon.event.event_id,
            payload_sha256=expired_soon.event.payload_sha256,
            acknowledged_at_utc=dt("2026-10-12T10:24:30Z"),
            signing_key=receiver_key,
        )
        # Set the local receive/verification clock just after the signed ACK
        # timestamp so the test reaches the earlier-than-expiry guard rather
        # than failing the independent future-clock-skew guard first.
        with self.assertRaisesRegex(TelemetryQueueError, "not before event expiry"):
            self.queue.acknowledge_protected(
                ack,
                verification_keys={"receiver-test-epoch": receiver_key.public_key()},
                now_utc=dt("2026-10-12T10:24:31Z"),
            )
        self.assertEqual(self.queue.pending_count, 1)

    def test_legacy_metadata_queue_does_not_expose_fake_payload_bytes(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        self.assertEqual(self.queue.select_protected_batch(now_utc=self.now, max_bytes=1000), ())
        self.assertEqual(len(self.queue.select_batch(now_utc=self.now, max_bytes=1000)), 1)

    def test_application_id_is_set_and_preserved_across_restart(self):
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0x53594D54)
        self.assertEqual(self.reopen().integrity_check(), "ok")
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0x53594D54)

    @unittest.skipUnless(os.name == "posix", "POSIX file mode assertion")
    def test_database_and_existing_wal_sidecars_are_private(self):
        self.queue.enqueue(event(), now_utc=self.now)
        for candidate in (self.database, self.database + "-wal", self.database + "-shm"):
            if Path(candidate).exists():
                mode = stat.S_IMODE(os.stat(candidate).st_mode)
                self.assertEqual(mode & 0o077, 0, f"{candidate} exposes group/other permissions")

    def test_foreign_application_id_is_rejected(self):
        with sqlite3.connect(self.database) as db:
            db.execute("PRAGMA application_id = 123456")
        with self.assertRaisesRegex(TelemetryQueueError, "different application"):
            self.reopen()

    def test_unowned_database_with_other_tables_is_not_modified(self):
        other = str(Path(self.temp.name) / "other.sqlite3")
        with sqlite3.connect(other) as db:
            db.execute("CREATE TABLE customer_records (id INTEGER PRIMARY KEY)")
        with self.assertRaisesRegex(TelemetryQueueError, "unowned SQLite database"):
            DurableTelemetryQueue(other)
        with sqlite3.connect(other) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0].lower(), "delete")
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )}
        self.assertEqual(tables, {"customer_records"})

    def test_invalid_busy_timeout_and_numeric_overflow_are_rejected(self):
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1, 301, 1e300, 10 ** 1000):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "finite and in"):
                    DurableTelemetryQueue(self.database, busy_timeout_seconds=timeout)
        for kwargs in (
            {"max_queue_bytes": 1 << 100},
            {"max_records": 1 << 100},
            {"max_future_skew_seconds": 1 << 100},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DurableTelemetryQueue(self.database, **kwargs)


if __name__ == "__main__":
    unittest.main()
