"""Restart, transaction and capacity tests for the SQLite telemetry ledger."""
from __future__ import annotations

from datetime import datetime
import tempfile
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
        limited = DurableTelemetryQueue(self.database, max_queue_bytes=100)
        limited.enqueue(event(size=80), now_utc=self.now)
        with self.assertRaises(QueueCapacityError):
            limited.enqueue(event("c", size=30), now_utc=self.now)
        self.assertEqual(limited.pending_count, 1)
        self.assertEqual(limited.pending_bytes, 80)

    def test_terminal_record_limit_fails_closed_instead_of_pruning_tombstones(self):
        tiny = DurableTelemetryQueue(self.database, max_queue_bytes=1000, max_records=1)
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

    def test_ephemeral_database_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "persistent filesystem path"):
            DurableTelemetryQueue(":memory:")

    def test_nonfinite_busy_timeout_is_rejected(self):
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    DurableTelemetryQueue(self.database, busy_timeout_seconds=timeout)


if __name__ == "__main__":
    unittest.main()
