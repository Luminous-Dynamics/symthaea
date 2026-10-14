"""SQLite-backed store-and-forward ledger for metadata and sealed envelopes.

The metadata-only API stores no application payload. The protected API stores
canonical AEAD-encrypted envelope bytes produced by authenticated_envelope.py.
This module provides transactional retry/ACK state and bounded, fail-closed
growth; it is not an authenticated ACK protocol, key manager, database
encryption layer, network transport, or production telemetry service.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import os
import sqlite3
from typing import Iterator

from tools.energy_telemetry.authenticated_envelope import SealedTelemetry, verify_ack
from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    TelemetryEvent,
    TelemetryQueueError,
    Transmission,
    _utc,
    make_transmission,
)

_SCHEMA_VERSION = 3
_APPLICATION_ID = 0x53594D54  # ASCII 'SYMT'; identifies this SQLite file format


@dataclass(frozen=True)
class QueuedProtectedTransmission:
    """Exact persisted protected bytes selected for at-least-once delivery."""

    event: TelemetryEvent
    envelope_bytes: bytes
    evidence_state: str
    actionable: bool
    attempt_count: int


class DurableTelemetryQueue:
    """SQLite queue with persistent retries, terminal records and idempotent ACKs.

    Every public operation opens its own connection. A new instance can reopen
    the same database after a process restart. Expired/delivered rows are kept
    as replay tombstones; they are never silently pruned by this prototype.
    Once max_records is reached, insertion fails closed until an explicit,
    reviewed archival/retention policy is implemented.
    """

    def __init__(
        self,
        database_path: str,
        *,
        max_queue_bytes: int = 1_048_576,
        max_records: int = 100_000,
        max_future_skew_seconds: int = 300,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if (
            not isinstance(database_path, str)
            or not database_path.strip()
            or "\x00" in database_path
            or database_path == ":memory:"
        ):
            raise ValueError("database_path must be a persistent filesystem path, not :memory:")
        sqlite_int_max = (1 << 63) - 1
        if type(max_queue_bytes) is not int or not 1 <= max_queue_bytes <= sqlite_int_max:
            raise ValueError("max_queue_bytes must be a positive SQLite integer")
        if type(max_records) is not int or not 1 <= max_records <= sqlite_int_max:
            raise ValueError("max_records must be a positive SQLite integer")
        if (
            type(max_future_skew_seconds) is not int
            or not 0 <= max_future_skew_seconds <= 31_536_000
        ):
            raise ValueError("max_future_skew_seconds must be in [0, 31536000]")
        if (
            type(busy_timeout_seconds) not in (int, float)
            or not 0 < busy_timeout_seconds <= 300
        ):
            raise ValueError("busy_timeout_seconds must be finite and in (0, 300]")
        self.database_path = database_path
        self.max_queue_bytes = max_queue_bytes
        self.max_records = max_records
        self.max_future_skew = timedelta(seconds=max_future_skew_seconds)
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._ownership_verified = False
        self._initialize()
        self._ownership_verified = True
        self._secure_file_modes()

    def _secure_file_modes(self) -> None:
        """Restrict database and existing WAL sidecars to the current user."""
        for candidate in (
            self.database_path,
            self.database_path + "-wal",
            self.database_path + "-shm",
        ):
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise TelemetryQueueError(
                    "unable to enforce private SQLite file permissions"
                ) from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
            # journal_mode is persistent database state and is established once
            # during initialization. Reissuing the mode-changing pragma for
            # every operation needlessly contends with other writers.
            connection.execute("PRAGMA synchronous = FULL")
            yield connection
        finally:
            try:
                if self._ownership_verified:
                    self._secure_file_modes()
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, _SCHEMA_VERSION):
                raise TelemetryQueueError(f"unsupported telemetry queue schema version: {version}")
            application_id = db.execute("PRAGMA application_id").fetchone()[0]
            if application_id not in (0, _APPLICATION_ID):
                raise TelemetryQueueError("SQLite database belongs to a different application")
            if application_id == 0:
                existing_tables = {
                    row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                if existing_tables - {"telemetry_events", "telemetry_queue_config"}:
                    raise TelemetryQueueError(
                        "refusing to claim an unowned SQLite database with unrelated tables"
                    )
            journal_mode = db.execute("PRAGMA journal_mode").fetchone()[0].lower()
            if journal_mode != "wal":
                journal_mode = db.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower()
            if journal_mode != "wal":
                raise TelemetryQueueError(
                    "SQLite WAL mode is required but unavailable on this database/filesystem"
                )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    event_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0),
                    observed_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    max_action_age_seconds INTEGER NOT NULL CHECK(max_action_age_seconds > 0),
                    priority INTEGER NOT NULL CHECK(priority BETWEEN 0 AND 3),
                    quality TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    wire_size_bytes INTEGER NOT NULL CHECK(wire_size_bytes BETWEEN 1 AND 65536),
                    envelope_bytes BLOB,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'delivered', 'expired')),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
                    accepted_at_utc TEXT NOT NULL,
                    terminal_at_utc TEXT,
                    CHECK (
                        (status = 'pending' AND terminal_at_utc IS NULL)
                        OR (status IN ('delivered', 'expired') AND terminal_at_utc IS NOT NULL)
                    )
                )
                """
            )
            # Additive migration for databases created before ciphertext
            # persistence. If a process stops after ALTER TABLE but before
            # user_version advances, reopening sees the column and safely resumes.
            event_columns = {
                row[1] for row in db.execute("PRAGMA table_info(telemetry_events)").fetchall()
            }
            if "envelope_bytes" not in event_columns:
                db.execute("ALTER TABLE telemetry_events ADD COLUMN envelope_bytes BLOB")
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS telemetry_pending_order
                ON telemetry_events(status, priority DESC, observed_at_utc, event_id)
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_queue_config (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    max_queue_bytes INTEGER NOT NULL CHECK(max_queue_bytes > 0),
                    max_records INTEGER NOT NULL CHECK(max_records > 0),
                    max_future_skew_seconds INTEGER NOT NULL CHECK(max_future_skew_seconds >= 0)
                )
                """
            )
            # Policy limits must be common to every process opening this DB.
            # Otherwise one worker could instantiate a larger limit and bypass
            # another worker's capacity/clock-skew policy.
            db.execute("BEGIN IMMEDIATE")
            try:
                # Re-check ownership under the write lock to close the race
                # between preflight and initialization by another process.
                application_id = db.execute("PRAGMA application_id").fetchone()[0]
                if application_id not in (0, _APPLICATION_ID):
                    raise TelemetryQueueError("SQLite database belongs to a different application")
                if application_id == 0:
                    existing_tables = {
                        row[0] for row in db.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                        ).fetchall()
                    }
                    if existing_tables - {"telemetry_events", "telemetry_queue_config"}:
                        raise TelemetryQueueError(
                            "refusing to claim an unowned SQLite database with unrelated tables"
                        )
                    db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                config = db.execute(
                    "SELECT max_queue_bytes, max_records, max_future_skew_seconds "
                    "FROM telemetry_queue_config WHERE singleton = 1"
                ).fetchone()
                expected = (
                    self.max_queue_bytes, self.max_records,
                    int(self.max_future_skew.total_seconds()),
                )
                if config is None:
                    db.execute(
                        "INSERT INTO telemetry_queue_config "
                        "(singleton, max_queue_bytes, max_records, max_future_skew_seconds) "
                        "VALUES (1, ?, ?, ?)",
                        expected,
                    )
                elif tuple(config) != expected:
                    raise TelemetryQueueError(
                        "queue policy configuration mismatch with existing database"
                    )
                db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    @staticmethod
    def _timestamp(value: datetime) -> str:
        _utc(value, "timestamp")
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _row_event(row: tuple) -> TelemetryEvent:
        return TelemetryEvent(
            event_id=row[0],
            domain=row[1],
            source_sequence=row[2],
            observed_at_utc=DurableTelemetryQueue._parse_timestamp(row[3]),
            expires_at_utc=DurableTelemetryQueue._parse_timestamp(row[4]),
            max_action_age_seconds=row[5],
            priority=row[6],
            quality=row[7],
            payload_sha256=row[8],
            wire_size_bytes=row[9],
        )

    @staticmethod
    def _row_select() -> str:
        return (
            "event_id, domain, source_sequence, observed_at_utc, expires_at_utc, "
            "max_action_age_seconds, priority, quality, payload_sha256, wire_size_bytes, "
            "status, attempt_count, accepted_at_utc, terminal_at_utc, envelope_bytes"
        )

    @staticmethod
    def _same_event(row: tuple, event: TelemetryEvent) -> bool:
        return DurableTelemetryQueue._row_event(row) == event

    @property
    def pending_count(self) -> int:
        with self._connection() as db:
            return db.execute(
                "SELECT COUNT(*) FROM telemetry_events WHERE status = 'pending'"
            ).fetchone()[0]

    @property
    def pending_bytes(self) -> int:
        with self._connection() as db:
            return db.execute(
                "SELECT COALESCE(SUM(wire_size_bytes), 0) "
                "FROM telemetry_events WHERE status = 'pending'"
            ).fetchone()[0]

    @property
    def record_count(self) -> int:
        with self._connection() as db:
            return db.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]

    def enqueue(self, event: TelemetryEvent, *, now_utc: datetime) -> str:
        """Enqueue metadata-only event; it cannot be selected by protected delivery."""
        return self._enqueue(event, envelope_bytes=None, now_utc=now_utc)

    def enqueue_protected(self, sealed: SealedTelemetry, *, now_utc: datetime) -> str:
        """Persist exactly the canonical signed ciphertext created by seal_payload."""
        if not isinstance(sealed, SealedTelemetry):
            raise TelemetryQueueError("enqueue_protected requires a SealedTelemetry result")
        try:
            sealed.validate_binding()
        except ValueError as exc:
            raise TelemetryQueueError(f"protected envelope binding is invalid: {exc}") from exc
        if hashlib.sha256(sealed.envelope_bytes).hexdigest() != sealed.event.payload_sha256:
            raise TelemetryQueueError("protected envelope digest mismatch")
        if len(sealed.envelope_bytes) != sealed.event.wire_size_bytes:
            raise TelemetryQueueError("protected envelope wire size mismatch")
        return self._enqueue(sealed.event, envelope_bytes=sealed.envelope_bytes, now_utc=now_utc)

    def _enqueue(
        self, event: TelemetryEvent, *, envelope_bytes: bytes | None, now_utc: datetime
    ) -> str:
        now_text = self._timestamp(now_utc)
        event.validate()
        if event.observed_at_utc > now_utc + self.max_future_skew:
            raise TelemetryQueueError("observation time exceeds permitted future clock skew")
        if now_utc >= event.expires_at_utc:
            initial_status = "expired"
            terminal_text = now_text
            initial_result = "expired_not_queued"
        else:
            initial_status = "pending"
            terminal_text = None
            initial_result = "queued"

        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    f"SELECT {self._row_select()} FROM telemetry_events WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    if not self._same_event(existing, event):
                        raise EventIdentityConflict(
                            "event_id was reused with different event contents"
                        )
                    if existing[14] != envelope_bytes:
                        raise EventIdentityConflict(
                            "event_id replay changed or omitted the protected envelope bytes"
                        )
                    if existing[10] == "pending":
                        result = "duplicate_pending"
                    else:
                        result = f"already_{existing[10]}"
                    db.execute("COMMIT")
                    return result

                record_count = db.execute(
                    "SELECT COUNT(*) FROM telemetry_events"
                ).fetchone()[0]
                if record_count >= self.max_records:
                    raise QueueCapacityError(
                        "durable record limit reached; archive/retention review required"
                    )
                pending_bytes = db.execute(
                    "SELECT COALESCE(SUM(wire_size_bytes), 0) "
                    "FROM telemetry_events WHERE status = 'pending'"
                ).fetchone()[0]
                if initial_status == "pending" and pending_bytes + event.wire_size_bytes > self.max_queue_bytes:
                    raise QueueCapacityError(
                        "durable queue byte limit exceeded; no event was silently evicted"
                    )
                db.execute(
                    """
                    INSERT INTO telemetry_events (
                        event_id, domain, source_sequence, observed_at_utc, expires_at_utc,
                        max_action_age_seconds, priority, quality, payload_sha256, wire_size_bytes,
                        status, attempt_count, accepted_at_utc, terminal_at_utc, envelope_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.domain, event.source_sequence,
                        self._timestamp(event.observed_at_utc), self._timestamp(event.expires_at_utc),
                        event.max_action_age_seconds, event.priority, event.quality,
                        event.payload_sha256, event.wire_size_bytes, initial_status,
                        now_text, terminal_text, envelope_bytes,
                    ),
                )
                db.execute("COMMIT")
                return initial_result
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def select_batch(self, *, now_utc: datetime, max_bytes: int) -> tuple[Transmission, ...]:
        self._timestamp(now_utc)
        if type(max_bytes) is not int or max_bytes < 1:
            raise TelemetryQueueError("max_bytes must be a positive integer")
        batch: list[tuple[TelemetryEvent, int]] = []
        used = 0
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(
                    f"SELECT {self._row_select()} FROM telemetry_events "
                    "WHERE status = 'pending' AND envelope_bytes IS NULL "
                    "ORDER BY priority DESC, observed_at_utc ASC, event_id ASC"
                ).fetchall()
                for row in rows:
                    event = self._row_event(row)
                    if now_utc >= event.expires_at_utc:
                        continue
                    if used + event.wire_size_bytes > max_bytes:
                        continue
                    used += event.wire_size_bytes
                    attempt_count = row[11] + 1
                    db.execute(
                        "UPDATE telemetry_events SET attempt_count = ? WHERE event_id = ? AND status = 'pending'",
                        (attempt_count, event.event_id),
                    )
                    batch.append((event, attempt_count))
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        # The queue state remains pending until authenticated delivery ACK.
        # A new process will resend it, yielding at-least-once delivery.
        return tuple(
            Transmission(event, make_transmission(event, now_utc=now_utc).evidence_state,
                         make_transmission(event, now_utc=now_utc).actionable)
            for event, _attempt_count in batch
        )

    def select_protected_batch(
        self, *, now_utc: datetime, max_bytes: int
    ) -> tuple[QueuedProtectedTransmission, ...]:
        """Select persisted ciphertext only; bytes stay pending until an ACK."""
        self._timestamp(now_utc)
        if type(max_bytes) is not int or max_bytes < 1:
            raise TelemetryQueueError("max_bytes must be a positive integer")
        batch: list[tuple[TelemetryEvent, bytes, int]] = []
        used = 0
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(
                    f"SELECT {self._row_select()} FROM telemetry_events "
                    "WHERE status = 'pending' AND envelope_bytes IS NOT NULL "
                    "ORDER BY priority DESC, observed_at_utc ASC, event_id ASC"
                ).fetchall()
                for row in rows:
                    event = self._row_event(row)
                    envelope_bytes = row[14]
                    if not isinstance(envelope_bytes, bytes):
                        raise TelemetryQueueError("protected event has invalid stored envelope bytes")
                    if hashlib.sha256(envelope_bytes).hexdigest() != event.payload_sha256:
                        raise TelemetryQueueError("stored protected envelope digest mismatch")
                    if len(envelope_bytes) != event.wire_size_bytes:
                        raise TelemetryQueueError("stored protected envelope size mismatch")
                    if now_utc >= event.expires_at_utc:
                        continue
                    if used + len(envelope_bytes) > max_bytes:
                        continue
                    used += len(envelope_bytes)
                    attempt_count = row[11] + 1
                    db.execute(
                        "UPDATE telemetry_events SET attempt_count = ? "
                        "WHERE event_id = ? AND status = 'pending'",
                        (attempt_count, event.event_id),
                    )
                    batch.append((event, envelope_bytes, attempt_count))
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        results: list[QueuedProtectedTransmission] = []
        for event, envelope_bytes, attempt_count in batch:
            assessed = make_transmission(event, now_utc=now_utc)
            results.append(QueuedProtectedTransmission(
                event=event,
                envelope_bytes=envelope_bytes,
                evidence_state=assessed.evidence_state,
                actionable=assessed.actionable,
                attempt_count=attempt_count,
            ))
        return tuple(results)

    def expire(self, *, now_utc: datetime) -> tuple[str, ...]:
        now_text = self._timestamp(now_utc)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                ids = tuple(row[0] for row in db.execute(
                    "SELECT event_id FROM telemetry_events "
                    "WHERE status = 'pending' AND expires_at_utc <= ? ORDER BY event_id",
                    (now_text,),
                ).fetchall())
                db.execute(
                    "UPDATE telemetry_events SET status = 'expired', terminal_at_utc = ? "
                    "WHERE status = 'pending' AND expires_at_utc <= ?",
                    (now_text, now_text),
                )
                db.execute("COMMIT")
                return ids
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def acknowledge(
        self, *, event_id: str, payload_sha256: str, now_utc: datetime
    ) -> str:
        """Acknowledge metadata-only events; protected envelopes require signed ACKs."""
        return self._acknowledge(
            event_id=event_id,
            payload_sha256=payload_sha256,
            now_utc=now_utc,
            protected_ack=False,
        )

    def acknowledge_protected(
        self,
        ack_bytes: bytes,
        *,
        verification_keys: dict[str, object],
        now_utc: datetime,
    ) -> str:
        """Verify a receiver-signed ACK before committing protected delivery state."""
        self._timestamp(now_utc)
        ack = verify_ack(ack_bytes, verification_keys=verification_keys)
        if ack.acknowledged_at_utc > now_utc + self.max_future_skew:
            raise TelemetryQueueError("ACK timestamp exceeds permitted future clock skew")
        return self._acknowledge(
            event_id=ack.event_id,
            payload_sha256=ack.payload_sha256,
            now_utc=now_utc,
            protected_ack=True,
            acknowledged_at_utc=ack.acknowledged_at_utc,
        )

    def _acknowledge(
        self,
        *,
        event_id: str,
        payload_sha256: str,
        now_utc: datetime,
        protected_ack: bool,
        acknowledged_at_utc: datetime | None = None,
    ) -> str:
        now_text = self._timestamp(now_utc)
        if not isinstance(event_id, str) or len(event_id) != 32 or any(ch not in "0123456789abcdef" for ch in event_id):
            raise TelemetryQueueError("ack event_id must be a 32-character lowercase hexadecimal ID")
        if not isinstance(payload_sha256, str) or len(payload_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in payload_sha256):
            raise TelemetryQueueError("ack payload digest must be a lowercase SHA-256 digest")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    f"SELECT {self._row_select()} FROM telemetry_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    raise TelemetryQueueError("ack references no pending event")
                event = self._row_event(row)
                status, attempts = row[10], row[11]
                has_protected_bytes = row[14] is not None
                if has_protected_bytes and not protected_ack:
                    raise TelemetryQueueError(
                        "protected envelope requires an authenticated receiver ACK"
                    )
                if protected_ack and not has_protected_bytes:
                    raise TelemetryQueueError(
                        "authenticated receiver ACK cannot acknowledge a metadata-only event"
                    )
                if protected_ack:
                    if acknowledged_at_utc is None:
                        raise TelemetryQueueError("signed ACK timestamp is required")
                    if acknowledged_at_utc >= event.expires_at_utc:
                        raise TelemetryQueueError("ACK timestamp is not before event expiry")
                if status == "delivered" and event.payload_sha256 == payload_sha256:
                    db.execute("COMMIT")
                    return "idempotent_ack"
                if status != "pending":
                    raise TelemetryQueueError(f"cannot acknowledge terminal {status} event")
                if now_utc >= event.expires_at_utc:
                    raise TelemetryQueueError("event expired before ACK; call expire to record it")
                if attempts < 1:
                    raise TelemetryQueueError("cannot acknowledge an event not selected for delivery")
                if event.payload_sha256 != payload_sha256:
                    raise EventIdentityConflict("ack digest does not match queued event")
                db.execute(
                    "UPDATE telemetry_events SET status = 'delivered', terminal_at_utc = ? "
                    "WHERE event_id = ? AND status = 'pending'",
                    (now_text, event_id),
                )
                db.execute("COMMIT")
                return "acknowledged"
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def integrity_check(self) -> str:
        with self._connection() as db:
            result = db.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise TelemetryQueueError(f"SQLite integrity check failed: {result}")
            return result

    def attempt_count(self, event_id: str) -> int:
        with self._connection() as db:
            row = db.execute(
                "SELECT attempt_count FROM telemetry_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise TelemetryQueueError("unknown event_id")
            return row[0]
