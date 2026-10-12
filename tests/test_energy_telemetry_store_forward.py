"""Deterministic tests for delay-tolerant satellite telemetry policy."""
from __future__ import annotations

from datetime import datetime
import unittest

from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    StoreForwardQueue,
    TelemetryEvent,
    TelemetryQueueError,
    make_transmission,
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


class StoreForwardTests(unittest.TestCase):
    def setUp(self):
        self.now = dt("2026-10-12T10:05:00Z")
        self.queue = StoreForwardQueue(max_queue_bytes=1000)

    def test_accepts_multiple_domain_profiles_without_changing_transport_policy(self):
        for idx, domain in enumerate(("energy", "water", "agriculture", "infrastructure", "spacecraft"), 1):
            self.assertEqual(self.queue.enqueue(event(chr(96 + idx), domain=domain), now_utc=self.now), "queued")
        self.assertEqual(self.queue.pending_count, 5)

    def test_duplicate_pending_event_is_idempotent(self):
        e = event()
        self.assertEqual(self.queue.enqueue(e, now_utc=self.now), "queued")
        self.assertEqual(self.queue.enqueue(e, now_utc=self.now), "duplicate_pending")
        self.assertEqual(self.queue.pending_count, 1)

    def test_reused_event_id_with_different_metadata_fails_closed(self):
        self.queue.enqueue(event(), now_utc=self.now)
        with self.assertRaises(EventIdentityConflict):
            self.queue.enqueue(event(priority=3), now_utc=self.now)

    def test_reused_terminal_event_id_with_different_digest_fails_closed(self):
        self.assertEqual(self.queue.enqueue(event(expires="2026-10-12T10:04:00Z"), now_utc=self.now), "expired_not_queued")
        with self.assertRaises(EventIdentityConflict):
            self.queue.enqueue(event(digest="c", expires="2026-10-12T11:00:00Z"), now_utc=self.now)
        with self.assertRaises(EventIdentityConflict):
            self.queue.enqueue(event(priority=3, expires="2026-10-12T10:04:00Z"), now_utc=self.now)

    def test_future_timestamp_beyond_clock_skew_is_rejected(self):
        with self.assertRaisesRegex(TelemetryQueueError, "future clock skew"):
            self.queue.enqueue(event(observed="2026-10-12T10:20:00Z"), now_utc=self.now)

    def test_stale_measurement_is_preserved_as_non_actionable(self):
        e = event(observed="2026-10-12T08:00:00Z", age=300)
        self.queue.enqueue(e, now_utc=self.now)
        selected = self.queue.select_batch(now_utc=self.now, max_bytes=100)
        self.assertEqual(selected[0].evidence_state, "stale")
        self.assertFalse(selected[0].actionable)

    def test_nonvalidated_measurement_cannot_drive_immediate_action(self):
        for q in ("partial", "estimated", "unknown"):
            assessment = make_transmission(event(quality=q), now_utc=self.now)
            self.assertFalse(assessment.actionable)
            self.assertEqual(assessment.evidence_state, f"quality_{q}")

    def test_only_fresh_validated_measurement_is_actionable(self):
        result = make_transmission(event(), now_utc=self.now)
        self.assertTrue(result.actionable)
        self.assertEqual(result.evidence_state, "fresh_validated")

    def test_mild_future_clock_skew_is_not_actionable(self):
        result = make_transmission(event(observed="2026-10-12T10:06:00Z"), now_utc=self.now)
        self.assertFalse(result.actionable)
        self.assertEqual(result.evidence_state, "future_dated")

    def test_priority_then_observation_time_deterministically_orders_batch(self):
        self.queue.enqueue(event("a", priority=1, observed="2026-10-12T10:02:00Z"), now_utc=self.now)
        self.queue.enqueue(event("b", priority=3, observed="2026-10-12T10:04:00Z"), now_utc=self.now)
        self.queue.enqueue(event("c", priority=1, observed="2026-10-12T10:01:00Z"), now_utc=self.now)
        selected = self.queue.select_batch(now_utc=self.now, max_bytes=1000)
        self.assertEqual([x.event.event_id[0] for x in selected], ["b", "c", "a"])

    def test_batch_respects_byte_budget_without_dropping_nonfitting_events(self):
        self.queue.enqueue(event("a", size=80), now_utc=self.now)
        self.queue.enqueue(event("b", size=60, priority=2), now_utc=self.now)
        selected = self.queue.select_batch(now_utc=self.now, max_bytes=60)
        self.assertEqual([x.event.event_id[0] for x in selected], ["b"])
        self.assertEqual(self.queue.pending_count, 2)

    def test_queue_capacity_error_does_not_silently_evict(self):
        small = StoreForwardQueue(max_queue_bytes=100)
        small.enqueue(event(size=80), now_utc=self.now)
        with self.assertRaises(QueueCapacityError):
            small.enqueue(event("c", size=30), now_utc=self.now)
        self.assertEqual(small.pending_count, 1)
        self.assertEqual(small.pending_bytes, 80)

    def test_expiry_returns_audit_ids_and_frees_capacity(self):
        e = event(expires="2026-10-12T10:04:00Z")
        self.queue.enqueue(e, now_utc=dt("2026-10-12T10:03:00Z"))
        self.assertEqual(self.queue.expire(now_utc=self.now), ("a" * 32,))
        self.assertEqual(self.queue.pending_count, 0)
        self.assertEqual(self.queue.pending_bytes, 0)
        self.assertEqual(self.queue.enqueue(e, now_utc=self.now), "already_expired")

    def test_batch_is_at_least_once_until_digest_checked_ack(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        first = self.queue.select_batch(now_utc=self.now, max_bytes=100)
        second = self.queue.select_batch(now_utc=self.now, max_bytes=100)
        self.assertEqual(first, second)
        with self.assertRaises(EventIdentityConflict):
            self.queue.acknowledge(event_id=e.event_id, payload_sha256="c" * 64, now_utc=self.now)
        self.assertEqual(self.queue.pending_count, 1)
        self.assertEqual(self.queue.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now), "acknowledged")
        self.assertEqual(self.queue.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now), "idempotent_ack")
        self.assertEqual(self.queue.pending_count, 0)

    def test_ack_before_selection_is_rejected(self):
        e = event()
        self.queue.enqueue(e, now_utc=self.now)
        with self.assertRaisesRegex(TelemetryQueueError, "not selected"):
            self.queue.acknowledge(event_id=e.event_id, payload_sha256=e.payload_sha256, now_utc=self.now)

    def test_expired_event_cannot_be_acked_as_delivered(self):
        e = event(expires="2026-10-12T10:06:00Z")
        self.queue.enqueue(e, now_utc=self.now)
        self.queue.select_batch(now_utc=self.now, max_bytes=100)
        with self.assertRaisesRegex(TelemetryQueueError, "expired before ACK"):
            self.queue.acknowledge(
                event_id=e.event_id, payload_sha256=e.payload_sha256,
                now_utc=dt("2026-10-12T10:06:00Z"),
            )
        self.assertEqual(self.queue.expire(now_utc=dt("2026-10-12T10:06:00Z")), (e.event_id,))

    def test_wrong_timezone_and_invalid_domain_fail_closed(self):
        with self.assertRaisesRegex(TelemetryQueueError, "must be UTC"):
            self.queue.enqueue(event(observed="2026-10-12T12:00:00+02:00"), now_utc=self.now)
        with self.assertRaisesRegex(TelemetryQueueError, "unsupported telemetry domain"):
            self.queue.enqueue(event(domain="other"), now_utc=self.now)
        with self.assertRaisesRegex(TelemetryQueueError, "unsupported telemetry domain"):
            self.queue.enqueue(event(domain=[]), now_utc=self.now)
        with self.assertRaisesRegex(TelemetryQueueError, "unsupported measurement quality"):
            self.queue.enqueue(event(quality=[]), now_utc=self.now)

    def test_invalid_wire_size_and_priority_types_fail_closed(self):
        for bad in (0, -1, True, 65_537):
            with self.assertRaises(TelemetryQueueError):
                self.queue.enqueue(event(size=bad), now_utc=self.now)
        with self.assertRaises(TelemetryQueueError):
            self.queue.enqueue(event(priority=True), now_utc=self.now)


if __name__ == "__main__":
    unittest.main()
