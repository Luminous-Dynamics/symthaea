"""Durable receiver inbox: verify, persist ciphertext, then emit signed ACK.

The inbox never stores decrypted payload bytes. It authenticates/decrypts a
protected envelope in memory, commits the exact ciphertext and replay tombstone
in SQLite, and only then creates the receiver-signed acceptance ACK. The caller
may process the returned plaintext in memory, but must still make any domain
action decision using its own policy and safety boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import os
import sqlite3
from typing import Iterator, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from tools.energy_telemetry.authenticated_envelope import (
    EnvelopeError,
    OpenedTelemetry,
    create_ack,
    open_envelope,
)
from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    TelemetryQueueError,
    _utc,
    make_transmission,
)

_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x54454C52  # ASCII 'TELR'


@dataclass(frozen=True)
class ReceiverAcceptance:
    """Durable receipt; ACK means committed storage, not truth or action approval."""

    event_id: str
    payload_sha256: str
    ack_bytes: bytes
    duplicate: bool
    accepted_at_utc: datetime
    evidence_state: str
    actionable: bool
    payload: bytes


class ReceiverInbox:
    """SQLite-backed inbox with replay dedupe and commit-before-ACK semantics."""

    def __init__(
        self,
        database_path: str,
        *,
        max_records: int = 100_000,
        max_envelope_bytes: int = 65_536,
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
        if type(max_records) is not int or not 1 <= max_records <= sqlite_int_max:
            raise ValueError("max_records must be a positive SQLite integer")
        if type(max_envelope_bytes) is not int or not 256 <= max_envelope_bytes <= 65_536:
            raise ValueError("max_envelope_bytes must be in [256, 65536]")
        if (
            type(max_future_skew_seconds) is not int
            or not 0 <= max_future_skew_seconds <= 31_536_000
        ):
            raise ValueError("max_future_skew_seconds must be in [0, 31536000]")
        if type(busy_timeout_seconds) not in (int, float) or not 0 < busy_timeout_seconds <= 300:
            raise ValueError("busy_timeout_seconds must be finite and in (0, 300]")
        self.database_path = database_path
        self.max_records = max_records
        self.max_envelope_bytes = max_envelope_bytes
        self.max_future_skew = timedelta(seconds=max_future_skew_seconds)
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._ownership_verified = False
        self._initialize()
        self._ownership_verified = True
        self._secure_file_modes()

    @staticmethod
    def _timestamp(value: datetime) -> str:
        _utc(value, "timestamp")
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(
            self.database_path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        try:
            db.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
            db.execute("PRAGMA synchronous = FULL")
            yield db
        finally:
            try:
                if self._ownership_verified:
                    self._secure_file_modes()
            finally:
                db.close()

    def _secure_file_modes(self) -> None:
        for path in (self.database_path, self.database_path + "-wal", self.database_path + "-shm"):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise TelemetryQueueError("unable to enforce private receiver inbox file permissions") from exc

    def _initialize(self) -> None:
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA_VERSION):
                raise TelemetryQueueError(f"unsupported receiver inbox schema version: {version}")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            if app_id not in (0, _APPLICATION_ID):
                raise TelemetryQueueError("SQLite database belongs to a different application")
            if app_id == 0:
                tables = {
                    row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                if tables - {"received_envelopes", "receiver_inbox_config"}:
                    raise TelemetryQueueError("refusing to claim unowned SQLite database with unrelated tables")
            mode = db.execute("PRAGMA journal_mode").fetchone()[0].lower()
            if mode != "wal":
                mode = db.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower()
            if mode != "wal":
                raise TelemetryQueueError("SQLite WAL mode is required but unavailable")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS received_envelopes (
                    event_id TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    envelope_bytes BLOB NOT NULL,
                    accepted_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    CHECK(length(envelope_bytes) BETWEEN 1 AND 65536)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS receiver_inbox_config (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    max_records INTEGER NOT NULL CHECK(max_records > 0),
                    max_envelope_bytes INTEGER NOT NULL CHECK(max_envelope_bytes BETWEEN 256 AND 65536),
                    max_future_skew_seconds INTEGER NOT NULL CHECK(max_future_skew_seconds >= 0)
                )
                """
            )
            db.execute("BEGIN IMMEDIATE")
            try:
                app_id = db.execute("PRAGMA application_id").fetchone()[0]
                if app_id not in (0, _APPLICATION_ID):
                    raise TelemetryQueueError("SQLite database belongs to a different application")
                if app_id == 0:
                    tables = {
                        row[0] for row in db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                        ).fetchall()
                    }
                    if tables - {"received_envelopes", "receiver_inbox_config"}:
                        raise TelemetryQueueError("refusing to claim unowned SQLite database with unrelated tables")
                    db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                expected = (
                    self.max_records,
                    self.max_envelope_bytes,
                    int(self.max_future_skew.total_seconds()),
                )
                current = db.execute(
                    "SELECT max_records, max_envelope_bytes, max_future_skew_seconds "
                    "FROM receiver_inbox_config WHERE singleton = 1"
                ).fetchone()
                if current is None:
                    db.execute(
                        "INSERT INTO receiver_inbox_config "
                        "(singleton,max_records,max_envelope_bytes,max_future_skew_seconds) "
                        "VALUES (1,?,?,?)",
                        expected,
                    )
                elif tuple(current) != expected:
                    raise TelemetryQueueError("receiver inbox policy configuration mismatch")
                db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    @property
    def record_count(self) -> int:
        with self._connection() as db:
            return db.execute("SELECT COUNT(*) FROM received_envelopes").fetchone()[0]

    def stored_envelope(self, event_id: str) -> bytes:
        """Return the persisted ciphertext for local replay/processing, never plaintext at rest."""
        with self._connection() as db:
            row = db.execute(
                "SELECT envelope_bytes FROM received_envelopes WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise TelemetryQueueError("unknown receiver event ID")
            if not isinstance(row[0], bytes):
                raise TelemetryQueueError("stored receiver envelope is invalid")
            return row[0]

    def integrity_check(self) -> str:
        with self._connection() as db:
            result = db.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise TelemetryQueueError(f"receiver inbox integrity check failed: {result}")
            return result

    def accept_and_ack(
        self,
        envelope_bytes: bytes,
        *,
        encryption_keys: Mapping[str, bytes],
        verification_keys: Mapping[str, Ed25519PublicKey],
        receiver_key_id: str,
        receiver_signing_key: Ed25519PrivateKey,
        now_utc: datetime,
    ) -> ReceiverAcceptance:
        """Authenticate, persist, COMMIT, then sign acceptance.

        If signing fails after commit, a retry sees the same durable event and
        can safely sign a fresh ACK. Event-ID reuse with different envelope
        bytes/digest is rejected. Expired and excessively future-dated events
        are not accepted into the actionable stream.
        """
        now_text = self._timestamp(now_utc)
        if not isinstance(envelope_bytes, bytes) or not 1 <= len(envelope_bytes) <= self.max_envelope_bytes:
            raise TelemetryQueueError("received envelope is outside the configured byte limit")

        try:
            opened: OpenedTelemetry = open_envelope(
                envelope_bytes,
                encryption_keys=encryption_keys,
                verification_keys=verification_keys,
            )
        except EnvelopeError as exc:
            raise TelemetryQueueError(f"received envelope authentication failed: {exc}") from exc

        event = opened.event
        if event.observed_at_utc > now_utc + self.max_future_skew:
            raise TelemetryQueueError("observation time exceeds receiver future clock skew")
        if now_utc >= event.expires_at_utc:
            raise TelemetryQueueError("expired envelope cannot be accepted")

        digest = hashlib.sha256(envelope_bytes).hexdigest()
        if digest != event.payload_sha256 or len(envelope_bytes) != event.wire_size_bytes:
            raise TelemetryQueueError("verified envelope digest/size binding mismatch")

        duplicate = False
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    "SELECT payload_sha256, envelope_bytes FROM received_envelopes WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != digest or existing[1] != envelope_bytes:
                        raise EventIdentityConflict(
                            "receiver event ID was replayed with different protected bytes"
                        )
                    duplicate = True
                else:
                    count = db.execute("SELECT COUNT(*) FROM received_envelopes").fetchone()[0]
                    if count >= self.max_records:
                        raise QueueCapacityError(
                            "receiver inbox record limit reached; retention review required"
                        )
                    db.execute(
                        "INSERT INTO received_envelopes "
                        "(event_id,payload_sha256,envelope_bytes,accepted_at_utc,expires_at_utc) "
                        "VALUES (?,?,?,?,?)",
                        (
                            event.event_id,
                            digest,
                            envelope_bytes,
                            now_text,
                            self._timestamp(event.expires_at_utc),
                        ),
                    )
                # The commit is deliberately before create_ack: a positive ACK
                # means exact ciphertext is durably recorded, not merely parsed.
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

        ack_bytes = create_ack(
            receiver_key_id=receiver_key_id,
            event_id=event.event_id,
            payload_sha256=digest,
            acknowledged_at_utc=now_utc,
            signing_key=receiver_signing_key,
        )
        assessed = make_transmission(event, now_utc=now_utc)
        return ReceiverAcceptance(
            event_id=event.event_id,
            payload_sha256=digest,
            ack_bytes=ack_bytes,
            duplicate=duplicate,
            accepted_at_utc=now_utc,
            evidence_state=assessed.evidence_state,
            actionable=assessed.actionable,
            payload=opened.payload,
        )
