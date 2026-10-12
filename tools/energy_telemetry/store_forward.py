"""Deterministic, in-memory store-and-forward policy prototype for telemetry.

The queue carries metadata and a payload digest, not sensor payloads. It models
transport policy for energy, water, agriculture, infrastructure and spacecraft
health telemetry. It is not a durable queue, transport, signature verifier, or
production security boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

EVENT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
DOMAINS = frozenset({"energy", "water", "agriculture", "infrastructure", "spacecraft"})
QUALITIES = frozenset({"validated", "partial", "estimated", "unknown"})


class TelemetryQueueError(ValueError):
    """A telemetry event or queue operation violates local policy."""


class EventIdentityConflict(TelemetryQueueError):
    """An event identifier was reused for different content or metadata."""


class QueueCapacityError(TelemetryQueueError):
    """The queue is full; an event was not silently dropped."""


def _utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TelemetryQueueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise TelemetryQueueError(f"{field} must be UTC")


@dataclass(frozen=True)
class TelemetryEvent:
    """Wire-safe event metadata; payload remains outside this prototype.

    source_sequence is scoped to a local-only source key and need not identify
    the source on the wire. payload_sha256 is a content digest, not privacy:
    never hash low-entropy private measurements and publish the digest alone.
    """

    event_id: str
    domain: str
    source_sequence: int
    observed_at_utc: datetime
    expires_at_utc: datetime
    max_action_age_seconds: int
    priority: int
    quality: str
    payload_sha256: str
    wire_size_bytes: int

    def validate(self) -> None:
        if not isinstance(self.event_id, str) or not EVENT_ID_RE.fullmatch(self.event_id):
            raise TelemetryQueueError("event_id must be a 32-character lowercase hexadecimal ID")
        if not isinstance(self.domain, str) or self.domain not in DOMAINS:
            raise TelemetryQueueError("unsupported telemetry domain")
        if type(self.source_sequence) is not int or self.source_sequence < 0:
            raise TelemetryQueueError("source_sequence must be a non-negative integer")
        _utc(self.observed_at_utc, "observed_at_utc")
        _utc(self.expires_at_utc, "expires_at_utc")
        if self.expires_at_utc <= self.observed_at_utc:
            raise TelemetryQueueError("expires_at_utc must be later than observed_at_utc")
        if (
            type(self.max_action_age_seconds) is not int
            or not 1 <= self.max_action_age_seconds <= 31_536_000
        ):
            raise TelemetryQueueError("max_action_age_seconds must be in [1, 31536000]")
        if type(self.priority) is not int or not 0 <= self.priority <= 3:
            raise TelemetryQueueError("priority must be an integer in [0, 3]")
        if not isinstance(self.quality, str) or self.quality not in QUALITIES:
            raise TelemetryQueueError("unsupported measurement quality")
        if not isinstance(self.payload_sha256, str) or not SHA256_RE.fullmatch(self.payload_sha256):
            raise TelemetryQueueError("payload_sha256 must be a lowercase SHA-256 digest")
        if type(self.wire_size_bytes) is not int or not 1 <= self.wire_size_bytes <= 65_536:
            raise TelemetryQueueError("wire_size_bytes must be in [1, 65536]")


@dataclass(frozen=True)
class Transmission:
    """A candidate delivery with explicit freshness/quality disposition."""

    event: TelemetryEvent
    evidence_state: str
    actionable: bool


class StoreForwardQueue:
    """In-memory queue with explicit expiry, batching, idempotency and ACK checks.

    Only metadata/digests are held. The caller must separately authenticate and
    encrypt transport, persist queue state, manage keys, and verify source
    provenance. now_utc is injected to make every behavior deterministic.
    """

    def __init__(self, *, max_queue_bytes: int = 1_048_576, max_future_skew_seconds: int = 300) -> None:
        if type(max_queue_bytes) is not int or max_queue_bytes < 1:
            raise ValueError("max_queue_bytes must be a positive integer")
        if type(max_future_skew_seconds) is not int or max_future_skew_seconds < 0:
            raise ValueError("max_future_skew_seconds must be a non-negative integer")
        self.max_queue_bytes = max_queue_bytes
        self.max_future_skew = timedelta(seconds=max_future_skew_seconds)
        self._pending: dict[str, TelemetryEvent] = {}
        self._in_flight: set[str] = set()
        self._terminal: dict[str, tuple[TelemetryEvent, str]] = {}
        self._pending_bytes = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    def enqueue(self, event: TelemetryEvent, *, now_utc: datetime) -> str:
        _utc(now_utc, "now_utc")
        event.validate()
        if event.observed_at_utc > now_utc + self.max_future_skew:
            raise TelemetryQueueError("observation time exceeds permitted future clock skew")

        if event.event_id in self._pending:
            if self._pending[event.event_id] == event:
                return "duplicate_pending"
            raise EventIdentityConflict("event_id was reused with different event contents")
        if event.event_id in self._terminal:
            prior_event, prior_state = self._terminal[event.event_id]
            if prior_event != event:
                raise EventIdentityConflict("terminal event_id was reused with different event contents")
            return f"already_{prior_state}"
        if now_utc >= event.expires_at_utc:
            self._terminal[event.event_id] = (event, "expired")
            return "expired_not_queued"
        if self._pending_bytes + event.wire_size_bytes > self.max_queue_bytes:
            raise QueueCapacityError("queue capacity exceeded; event was not queued or dropped")
        self._pending[event.event_id] = event
        self._pending_bytes += event.wire_size_bytes
        return "queued"

    def expire(self, *, now_utc: datetime) -> tuple[str, ...]:
        """Remove expired events and return their IDs for an explicit audit receipt."""
        _utc(now_utc, "now_utc")
        expired = tuple(sorted(
            event_id for event_id, event in self._pending.items()
            if now_utc >= event.expires_at_utc
        ))
        for event_id in expired:
            event = self._pending.pop(event_id)
            self._in_flight.discard(event_id)
            self._pending_bytes -= event.wire_size_bytes
            self._terminal[event_id] = (event, "expired")
        return expired

    def select_batch(self, *, now_utc: datetime, max_bytes: int) -> tuple[Transmission, ...]:
        """Select a deterministic priority batch without removing unacknowledged data.

        Expiry is deliberately explicit: call expire first and record returned
        IDs. Stale but unexpired events can be transmitted for historical analysis,
        but are always marked non-actionable. Only fresh, validated events are
        eligible for immediate action in this baseline policy.
        """
        _utc(now_utc, "now_utc")
        if type(max_bytes) is not int or max_bytes < 1:
            raise TelemetryQueueError("max_bytes must be a positive integer")
        ordered = sorted(
            self._pending.values(),
            key=lambda e: (-e.priority, e.observed_at_utc, e.event_id),
        )
        result: list[Transmission] = []
        used = 0
        for event in ordered:
            if now_utc >= event.expires_at_utc:
                continue
            if used + event.wire_size_bytes > max_bytes:
                continue
            age = (now_utc - event.observed_at_utc).total_seconds()
            if age < 0:
                state, actionable = "future_dated", False
            elif age > event.max_action_age_seconds:
                state, actionable = "stale", False
            elif event.quality != "validated":
                state, actionable = f"quality_{event.quality}", False
            else:
                state, actionable = "fresh_validated", True
            result.append(Transmission(event=event, evidence_state=state, actionable=actionable))
            used += event.wire_size_bytes
            self._in_flight.add(event.event_id)
        return tuple(result)

    def acknowledge(self, *, event_id: str, payload_sha256: str, now_utc: datetime) -> str:
        """Remove only a selected, unexpired event for its exact digest."""
        _utc(now_utc, "now_utc")
        if not isinstance(event_id, str) or not EVENT_ID_RE.fullmatch(event_id):
            raise TelemetryQueueError("ack event_id must be a 32-character lowercase hexadecimal ID")
        if not isinstance(payload_sha256, str) or not SHA256_RE.fullmatch(payload_sha256):
            raise TelemetryQueueError("ack payload digest must be a lowercase SHA-256 digest")
        event = self._pending.get(event_id)
        if event is None:
            terminal = self._terminal.get(event_id)
            if terminal is not None and terminal[0].payload_sha256 == payload_sha256 and terminal[1] == "delivered":
                return "idempotent_ack"
            raise TelemetryQueueError("ack references no pending event")
        if now_utc >= event.expires_at_utc:
            raise TelemetryQueueError("event expired before ACK; call expire to record it")
        if event_id not in self._in_flight:
            raise TelemetryQueueError("cannot acknowledge an event not selected for delivery")
        if event.payload_sha256 != payload_sha256:
            raise EventIdentityConflict("ack digest does not match queued event")
        self._pending.pop(event_id)
        self._in_flight.discard(event_id)
        self._pending_bytes -= event.wire_size_bytes
        self._terminal[event_id] = (event, "delivered")
        return "acknowledged"


def make_transmission(event: TelemetryEvent, *, now_utc: datetime) -> Transmission:
    """Pure assessment helper usable by receivers before current-state updates."""
    _utc(now_utc, "now_utc")
    event.validate()
    if now_utc >= event.expires_at_utc:
        return Transmission(event, "expired", False)
    age = (now_utc - event.observed_at_utc).total_seconds()
    if age < 0:
        return Transmission(event, "future_dated", False)
    if age > event.max_action_age_seconds:
        return Transmission(event, "stale", False)
    if event.quality != "validated":
        return Transmission(event, f"quality_{event.quality}", False)
    return Transmission(event, "fresh_validated", True)
