"""Restart, transaction and capacity tests for the SQLite telemetry ledger."""
from __future__ import annotations

from datetime import datetime
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

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


class DurableQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp.name) / "telemetry.sqlite3")
        self.now = dt("2026-10-12T10:05:00Z")
        self.queue = DurableTelemetryQueue(self.database, max_queue_bytes=1000)

    def tearDown(self):
        self.temp.cleanup()

    def reopen(self, **kwargs):
        return DurableTelemetryQueue(self.database, max_queue_bytes=1000, **kwargs)

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

    def test_schema_v1_migrates_transactionally_to_v2(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DROP TABLE telemetry_queue_config")
            db.execute("PRAGMA user_version = 1")
        migrated = self.reopen()
        self.assertEqual(migrated.integrity_check(), "ok")
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
            config = db.execute(
                "SELECT max_queue_bytes, max_records, max_future_skew_seconds "
                "FROM telemetry_queue_config WHERE singleton = 1"
            ).fetchone()
        self.assertEqual(config, (1000, 100_000, 300))

    def test_unknown_schema_version_fails_closed(self):
        with sqlite3.connect(self.database) as db:
            db.execute("PRAGMA user_version = 999")
        with self.assertRaisesRegex(TelemetryQueueError, "unsupported telemetry queue schema"):
            self.reopen()

    def test_application_id_is_set_and_preserved_across_restart(self):
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0x53594D54)
        self.assertEqual(self.reopen().integrity_check(), "ok")
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("PRAGMA application_id").fetchone()[0], 0x53594D54)

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
