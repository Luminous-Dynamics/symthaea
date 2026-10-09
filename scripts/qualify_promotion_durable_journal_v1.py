#!/usr/bin/env python3
"""Adversarial qualification for a disk-backed promotion journal reference model.

This is a reference transaction model, not a production promotion adapter.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from validate_promotion_reservation_v1 import PromotionTemporalAttemptIdentityV1

GENESIS_HASH = hashlib.sha256(b"").hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    head TEXT NOT NULL,
    active_lease TEXT NOT NULL,
    fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
    trust_root_generation INTEGER NOT NULL CHECK (trust_root_generation >= 0),
    governance_generation INTEGER NOT NULL CHECK (governance_generation >= 0),
    revision INTEGER NOT NULL CHECK (revision >= 0)
);

CREATE TRIGGER IF NOT EXISTS ledger_state_fence_monotonicity_guard
BEFORE UPDATE ON ledger_state
WHEN NEW.singleton <> OLD.singleton
  OR NEW.revision <> OLD.revision + 1
  OR NEW.fencing_token < OLD.fencing_token
  OR NEW.fencing_token > OLD.fencing_token + 1
  OR NEW.trust_root_generation < OLD.trust_root_generation
  OR NEW.governance_generation < OLD.governance_generation
  OR (
    NEW.fencing_token = OLD.fencing_token
    AND (
      NEW.head <> OLD.head
      OR NEW.active_lease <> OLD.active_lease
      OR NEW.trust_root_generation <> OLD.trust_root_generation
      OR NEW.governance_generation <> OLD.governance_generation
    )
  )
BEGIN
    SELECT RAISE(ABORT, 'ledger authority state monotonicity violated');
END;

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL UNIQUE,
    lease_id TEXT NOT NULL,
    predecessor_head TEXT NOT NULL,
    reservation_head TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'PromotionReserved', 'PromotionDispatchPrepared', 'PromotionSuperseded'
    )),
    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
    trust_root_generation INTEGER NOT NULL CHECK (trust_root_generation >= 0),
    governance_generation INTEGER NOT NULL CHECK (governance_generation >= 0),
    operation_identity_digest TEXT NOT NULL CHECK (length(operation_identity_digest) = 64),
    clock_id TEXT NOT NULL CHECK (length(trim(clock_id)) > 0),
    reserved_wall_time_ms INTEGER NOT NULL CHECK (reserved_wall_time_ms >= 0),
    reserved_monotonic_ns INTEGER NOT NULL CHECK (reserved_monotonic_ns >= 0),
    created_revision INTEGER NOT NULL CHECK (created_revision > 0),
    dispatch_attempt_id TEXT UNIQUE,
    dispatch_attempt_sequence INTEGER,
    dispatch_wall_time_ms INTEGER,
    dispatch_monotonic_ns INTEGER,
    dispatch_clock_id TEXT,
    attempt_identity_digest TEXT,
    superseded_by_fence INTEGER,
    CHECK (
      (state = 'PromotionReserved'
       AND dispatch_attempt_id IS NULL
       AND dispatch_attempt_sequence IS NULL
       AND dispatch_wall_time_ms IS NULL
       AND dispatch_monotonic_ns IS NULL
       AND dispatch_clock_id IS NULL
       AND attempt_identity_digest IS NULL
       AND superseded_by_fence IS NULL)
      OR
      (state = 'PromotionDispatchPrepared'
       AND dispatch_attempt_id IS NOT NULL
       AND length(trim(dispatch_attempt_id)) > 0
       AND dispatch_attempt_sequence IS NOT NULL
       AND dispatch_attempt_sequence > 0
       AND dispatch_wall_time_ms IS NOT NULL
       AND dispatch_wall_time_ms >= reserved_wall_time_ms
       AND dispatch_monotonic_ns IS NOT NULL
       AND dispatch_monotonic_ns >= reserved_monotonic_ns
       AND dispatch_clock_id = clock_id
       AND attempt_identity_digest IS NOT NULL
       AND length(attempt_identity_digest) = 64
       AND superseded_by_fence IS NULL)
      OR
      (state = 'PromotionSuperseded'
       AND dispatch_attempt_id IS NULL
       AND dispatch_attempt_sequence IS NULL
       AND dispatch_wall_time_ms IS NULL
       AND dispatch_monotonic_ns IS NULL
       AND dispatch_clock_id IS NULL
       AND attempt_identity_digest IS NULL
       AND superseded_by_fence IS NOT NULL
       AND superseded_by_fence > fencing_token)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_promotion_reservation
ON reservations((1))
WHERE state IN ('PromotionReserved', 'PromotionDispatchPrepared');

CREATE UNIQUE INDEX IF NOT EXISTS one_sequence_per_reservation
ON reservations(reservation_id, dispatch_attempt_sequence)
WHERE dispatch_attempt_sequence IS NOT NULL;

CREATE TABLE IF NOT EXISTS journal_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    reservation_id TEXT REFERENCES reservations(reservation_id),
    fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL CHECK (length(previous_hash) = 64),
    event_hash TEXT NOT NULL UNIQUE CHECK (length(event_hash) = 64)
);

CREATE TRIGGER IF NOT EXISTS journal_events_no_update
BEFORE UPDATE ON journal_events
BEGIN
    SELECT RAISE(ABORT, 'promotion journal is append-only');
END;

CREATE TRIGGER IF NOT EXISTS journal_events_no_delete
BEFORE DELETE ON journal_events
BEGIN
    SELECT RAISE(ABORT, 'promotion journal is append-only');
END;

CREATE TRIGGER IF NOT EXISTS reservation_insert_authority_guard
BEFORE INSERT ON reservations
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM ledger_state l
        WHERE l.singleton = 1
          AND l.head = NEW.reservation_head
          AND l.active_lease = NEW.lease_id
          AND l.fencing_token = NEW.fencing_token
          AND l.trust_root_generation = NEW.trust_root_generation
          AND l.governance_generation = NEW.governance_generation
          AND l.revision = NEW.created_revision
    ) THEN RAISE(ABORT, 'reservation storage fence rejected') END;
END;

CREATE TRIGGER IF NOT EXISTS reservation_identity_immutable
BEFORE UPDATE OF
    reservation_id, operation_id, lease_id, predecessor_head, reservation_head,
    fencing_token, trust_root_generation, governance_generation,
    operation_identity_digest, clock_id, reserved_wall_time_ms,
    reserved_monotonic_ns, created_revision
ON reservations
BEGIN
    SELECT RAISE(ABORT, 'reservation identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS reservation_state_transition_guard
BEFORE UPDATE OF state ON reservations
WHEN OLD.state <> NEW.state
 AND NOT (
    OLD.state = 'PromotionReserved'
    AND NEW.state IN ('PromotionDispatchPrepared', 'PromotionSuperseded')
 )
BEGIN
    SELECT RAISE(ABORT, 'invalid reservation state transition');
END;

CREATE TRIGGER IF NOT EXISTS reservation_prepare_storage_fence_guard
BEFORE UPDATE OF state ON reservations
WHEN NEW.state = 'PromotionDispatchPrepared'
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM ledger_state l
        WHERE l.singleton = 1
          AND l.head = NEW.reservation_head
          AND l.active_lease = NEW.lease_id
          AND l.fencing_token = NEW.fencing_token
          AND l.trust_root_generation = NEW.trust_root_generation
          AND l.governance_generation = NEW.governance_generation
          AND l.revision = NEW.created_revision + 1
    ) THEN RAISE(ABORT, 'storage-enforced promotion fence rejected') END;
    SELECT CASE WHEN NEW.dispatch_attempt_id IS NULL
                      OR length(trim(NEW.dispatch_attempt_id)) = 0
                      OR NEW.dispatch_attempt_sequence IS NULL
                      OR NEW.dispatch_attempt_sequence <= 0
                      OR NEW.dispatch_wall_time_ms < NEW.reserved_wall_time_ms
                      OR NEW.dispatch_monotonic_ns < NEW.reserved_monotonic_ns
                      OR NEW.dispatch_clock_id <> NEW.clock_id
                      OR NEW.attempt_identity_digest IS NULL
        THEN RAISE(ABORT, 'invalid prepared dispatch evidence') END;
END;

CREATE TRIGGER IF NOT EXISTS reservation_supersede_storage_fence_guard
BEFORE UPDATE OF state ON reservations
WHEN NEW.state = 'PromotionSuperseded'
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM ledger_state l
        WHERE l.singleton = 1
          AND l.fencing_token = NEW.superseded_by_fence
          AND NEW.superseded_by_fence > OLD.fencing_token
    ) THEN RAISE(ABORT, 'supersession requires a newer storage fence') END;
END;
"""


class DurablePromotionJournalV1:
    """SQLite reference journal; every mutating API uses BEGIN IMMEDIATE."""

    REQUIRED_TRIGGERS = {
        "journal_events_no_update",
        "journal_events_no_delete",
        "ledger_state_fence_monotonicity_guard",
        "reservation_insert_authority_guard",
        "reservation_identity_immutable",
        "reservation_state_transition_guard",
        "reservation_prepare_storage_fence_guard",
        "reservation_supersede_storage_fence_guard",
    }

    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            raise ValueError("durable journal requires a file-backed SQLite database")
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

        with self._write() as connection:
            state = connection.execute(
                "SELECT singleton FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            if state is None:
                initial = {
                    "head": "L0",
                    "active_lease": "LEASE-1",
                    "fencing_token": 0,
                    "trust_root_generation": 1,
                    "governance_generation": 1,
                    "revision": 0,
                }
                connection.execute(
                    """INSERT INTO ledger_state
                       (singleton, head, active_lease, fencing_token,
                        trust_root_generation, governance_generation, revision)
                       VALUES (1, ?, ?, ?, ?, ?, ?)""",
                    (
                        initial["head"], initial["active_lease"], initial["fencing_token"],
                        initial["trust_root_generation"], initial["governance_generation"],
                        initial["revision"],
                    ),
                )
                self._append_event(
                    connection,
                    event_id="ledger-initialized:v1",
                    event_type="LedgerInitialized",
                    reservation_id=None,
                    fencing_token=0,
                    payload=initial,
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        event_id: str,
        event_type: str,
        reservation_id: str | None,
        fencing_token: int,
        payload: dict[str, Any],
    ) -> None:
        last = connection.execute(
            "SELECT seq, event_hash FROM journal_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if last is None else int(last["seq"]) + 1
        previous_hash = GENESIS_HASH if last is None else str(last["event_hash"])
        payload_json = canonical_json(payload)
        unsigned = {
            "seq": seq,
            "event_id": event_id,
            "event_type": event_type,
            "reservation_id": reservation_id,
            "fencing_token": fencing_token,
            "payload_json": payload_json,
            "previous_hash": previous_hash,
        }
        event_hash = sha256_json(unsigned)
        connection.execute(
            """INSERT INTO journal_events
               (seq, event_id, event_type, reservation_id, fencing_token,
                payload_json, previous_hash, event_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seq, event_id, event_type, reservation_id, fencing_token,
                payload_json, previous_hash, event_hash,
            ),
        )

    def current_state(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("journal has no ledger state")
            return dict(row)
        finally:
            connection.close()

    def get_reservation(self, reservation_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return None if row is None else dict(row)
        finally:
            connection.close()

    def event_count(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM journal_events").fetchone()[0])
        finally:
            connection.close()

    def reserve(
        self,
        *,
        observed_head: str,
        lease_id: str,
        candidate: str,
        reservation_id: str,
        operation_id: str,
        trust_root_generation: int,
        governance_generation: int,
        operation_identity_digest: str,
        clock_id: str,
        reserved_wall_time_ms: int,
        reserved_monotonic_ns: int,
    ) -> bool:
        if (
            not all((observed_head, lease_id, candidate, reservation_id, operation_id, clock_id))
            or not is_digest(operation_identity_digest)
            or trust_root_generation < 0
            or governance_generation < 0
            or reserved_wall_time_ms < 0
            or reserved_monotonic_ns < 0
        ):
            return False

        with self._write() as connection:
            state_row = connection.execute(
                "SELECT * FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            state = dict(state_row)
            active = connection.execute(
                """SELECT 1 FROM reservations
                   WHERE state IN ('PromotionReserved', 'PromotionDispatchPrepared')
                   LIMIT 1"""
            ).fetchone()
            if (
                observed_head != state["head"]
                or lease_id != state["active_lease"]
                or trust_root_generation != state["trust_root_generation"]
                or governance_generation != state["governance_generation"]
                or active is not None
            ):
                return False

            new_fence = int(state["fencing_token"]) + 1
            new_revision = int(state["revision"]) + 1
            updated = connection.execute(
                """UPDATE ledger_state
                   SET head = ?, fencing_token = ?, revision = ?
                   WHERE singleton = 1 AND head = ? AND active_lease = ?
                     AND fencing_token = ? AND trust_root_generation = ?
                     AND governance_generation = ? AND revision = ?""",
                (
                    candidate, new_fence, new_revision, observed_head, lease_id,
                    state["fencing_token"], trust_root_generation,
                    governance_generation, state["revision"],
                ),
            )
            if updated.rowcount != 1:
                return False

            payload = {
                "reservation_id": reservation_id,
                "operation_id": operation_id,
                "lease_id": lease_id,
                "predecessor_head": observed_head,
                "reservation_head": candidate,
                "fencing_token": new_fence,
                "trust_root_generation": trust_root_generation,
                "governance_generation": governance_generation,
                "operation_identity_digest": operation_identity_digest,
                "clock_id": clock_id,
                "reserved_wall_time_ms": reserved_wall_time_ms,
                "reserved_monotonic_ns": reserved_monotonic_ns,
                "created_revision": new_revision,
                "previous_fencing_token": int(state["fencing_token"]),
                "previous_revision": int(state["revision"]),
            }
            connection.execute(
                """INSERT INTO reservations
                   (reservation_id, operation_id, lease_id, predecessor_head,
                    reservation_head, state, fencing_token, trust_root_generation,
                    governance_generation, operation_identity_digest, clock_id,
                    reserved_wall_time_ms, reserved_monotonic_ns, created_revision)
                   VALUES (?, ?, ?, ?, ?, 'PromotionReserved', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reservation_id, operation_id, lease_id, observed_head, candidate,
                    new_fence, trust_root_generation, governance_generation,
                    operation_identity_digest, clock_id, reserved_wall_time_ms,
                    reserved_monotonic_ns, new_revision,
                ),
            )
            self._append_event(
                connection,
                event_id=f"reserve:{reservation_id}",
                event_type="PromotionReserved",
                reservation_id=reservation_id,
                fencing_token=new_fence,
                payload=payload,
            )
            return True

    def advance_fence(
        self,
        *,
        active_lease: str,
        trust_root_generation: int,
        governance_generation: int,
    ) -> bool:
        if not active_lease or trust_root_generation < 0 or governance_generation < 0:
            return False
        with self._write() as connection:
            state = dict(connection.execute(
                "SELECT * FROM ledger_state WHERE singleton = 1"
            ).fetchone())
            prepared = connection.execute(
                "SELECT 1 FROM reservations WHERE state = 'PromotionDispatchPrepared' LIMIT 1"
            ).fetchone()
            if prepared is not None:
                return False
            reserved = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM reservations WHERE state = 'PromotionReserved' ORDER BY reservation_id"
                ).fetchall()
            ]
            new_fence = int(state["fencing_token"]) + 1
            new_revision = int(state["revision"]) + 1
            updated = connection.execute(
                """UPDATE ledger_state
                   SET active_lease = ?, fencing_token = ?, trust_root_generation = ?,
                       governance_generation = ?, revision = ?
                   WHERE singleton = 1 AND head = ? AND active_lease = ?
                     AND fencing_token = ? AND trust_root_generation = ?
                     AND governance_generation = ? AND revision = ?""",
                (
                    active_lease, new_fence, trust_root_generation, governance_generation,
                    new_revision, state["head"], state["active_lease"],
                    state["fencing_token"], state["trust_root_generation"],
                    state["governance_generation"], state["revision"],
                ),
            )
            if updated.rowcount != 1:
                return False

            for reservation in reserved:
                connection.execute(
                    """UPDATE reservations SET state = 'PromotionSuperseded',
                       superseded_by_fence = ?
                       WHERE reservation_id = ? AND state = 'PromotionReserved'""",
                    (new_fence, reservation["reservation_id"]),
                )

            rotation_payload = {
                "head": state["head"],
                "previous_lease_id": state["active_lease"],
                "active_lease": active_lease,
                "previous_fencing_token": int(state["fencing_token"]),
                "fencing_token": new_fence,
                "previous_trust_root_generation": int(state["trust_root_generation"]),
                "trust_root_generation": trust_root_generation,
                "previous_governance_generation": int(state["governance_generation"]),
                "governance_generation": governance_generation,
                "previous_revision": int(state["revision"]),
                "revision": new_revision,
            }
            self._append_event(
                connection,
                event_id=f"authority-fence-advanced:{new_fence}",
                event_type="AuthorityFenceAdvanced",
                reservation_id=None,
                fencing_token=new_fence,
                payload=rotation_payload,
            )
            for reservation in reserved:
                self._append_event(
                    connection,
                    event_id=f"supersede:{reservation['reservation_id']}:{new_fence}",
                    event_type="ReservationSuperseded",
                    reservation_id=reservation["reservation_id"],
                    fencing_token=new_fence,
                    payload={
                        "reservation_id": reservation["reservation_id"],
                        "previous_fencing_token": reservation["fencing_token"],
                        "superseded_by_fence": new_fence,
                    },
                )
            return True

    @staticmethod
    def _identity_from(
        reservation: dict[str, Any],
        *,
        dispatch_attempt_id: str,
        dispatch_attempt_sequence: int,
        dispatch_wall_time_ms: int,
        dispatch_monotonic_ns: int,
    ) -> PromotionTemporalAttemptIdentityV1:
        return PromotionTemporalAttemptIdentityV1(
            operation_identity_digest=reservation["operation_identity_digest"],
            reservation_id=reservation["reservation_id"],
            promotion_operation_id=reservation["operation_id"],
            reservation_head=reservation["reservation_head"],
            dispatch_attempt_id=dispatch_attempt_id,
            dispatch_attempt_sequence=dispatch_attempt_sequence,
            fencing_token=reservation["fencing_token"],
            trust_root_generation=reservation["trust_root_generation"],
            governance_generation=reservation["governance_generation"],
            local_monotonic_clock_id=reservation["clock_id"],
            reservation_time_ms=reservation["reserved_wall_time_ms"],
            dispatch_time_ms=dispatch_wall_time_ms,
            reservation_monotonic_ns=reservation["reserved_monotonic_ns"],
            dispatch_monotonic_ns=dispatch_monotonic_ns,
        )

    def prepare_dispatch(
        self,
        *,
        reservation_id: str,
        observed_head: str,
        observed_trust_root_generation: int,
        observed_governance_generation: int,
        observed_fencing_token: int,
        attempt_id: str,
        attempt_sequence: int,
        dispatch_wall_time_ms: int,
        dispatch_monotonic_ns: int,
        dispatch_clock_id: str,
    ) -> bool:
        if (
            attempt_sequence <= 0
            or not attempt_id.strip()
            or dispatch_wall_time_ms < 0
            or dispatch_monotonic_ns < 0
            or not dispatch_clock_id
        ):
            return False
        with self._write() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            state_row = connection.execute(
                "SELECT * FROM ledger_state WHERE singleton = 1"
            ).fetchone()
            if row is None or state_row is None:
                return False
            reservation = dict(row)
            state = dict(state_row)
            if (
                reservation["state"] != "PromotionReserved"
                or observed_head != state["head"]
                or observed_head != reservation["reservation_head"]
                or observed_trust_root_generation != state["trust_root_generation"]
                or observed_trust_root_generation != reservation["trust_root_generation"]
                or observed_governance_generation != state["governance_generation"]
                or observed_governance_generation != reservation["governance_generation"]
                or observed_fencing_token != state["fencing_token"]
                or observed_fencing_token != reservation["fencing_token"]
                or state["active_lease"] != reservation["lease_id"]
                or dispatch_clock_id != reservation["clock_id"]
                or dispatch_wall_time_ms < reservation["reserved_wall_time_ms"]
                or dispatch_monotonic_ns < reservation["reserved_monotonic_ns"]
            ):
                return False

            identity = self._identity_from(
                reservation,
                dispatch_attempt_id=attempt_id,
                dispatch_attempt_sequence=attempt_sequence,
                dispatch_wall_time_ms=dispatch_wall_time_ms,
                dispatch_monotonic_ns=dispatch_monotonic_ns,
            )
            if not identity.structurally_valid():
                return False
            identity_digest = identity.identity_digest()
            new_revision = int(state["revision"]) + 1
            updated = connection.execute(
                """UPDATE ledger_state SET revision = ?
                   WHERE singleton = 1 AND head = ? AND active_lease = ?
                     AND fencing_token = ? AND trust_root_generation = ?
                     AND governance_generation = ? AND revision = ?""",
                (
                    new_revision, state["head"], state["active_lease"],
                    state["fencing_token"], state["trust_root_generation"],
                    state["governance_generation"], state["revision"],
                ),
            )
            if updated.rowcount != 1:
                return False

            changed = connection.execute(
                """UPDATE reservations
                   SET state = 'PromotionDispatchPrepared',
                       dispatch_attempt_id = ?, dispatch_attempt_sequence = ?,
                       dispatch_wall_time_ms = ?, dispatch_monotonic_ns = ?,
                       dispatch_clock_id = ?, attempt_identity_digest = ?
                   WHERE reservation_id = ? AND state = 'PromotionReserved'
                     AND fencing_token = ? AND trust_root_generation = ?
                     AND governance_generation = ?
                     AND EXISTS (
                       SELECT 1 FROM ledger_state l
                       WHERE l.singleton = 1
                         AND l.head = reservations.reservation_head
                         AND l.active_lease = reservations.lease_id
                         AND l.fencing_token = reservations.fencing_token
                         AND l.trust_root_generation = reservations.trust_root_generation
                         AND l.governance_generation = reservations.governance_generation
                         AND l.revision = reservations.created_revision + 1
                     )""",
                (
                    attempt_id, attempt_sequence, dispatch_wall_time_ms, dispatch_monotonic_ns,
                    dispatch_clock_id, identity_digest, reservation_id,
                    observed_fencing_token, observed_trust_root_generation,
                    observed_governance_generation,
                ),
            )
            if changed.rowcount != 1:
                # The ledger revision was already changed in this transaction.
                # A silent compare-and-swap miss must abort, never commit a partial prepare.
                raise RuntimeError(
                    "reservation compare-and-swap failed after ledger revision update"
                )

            payload = {
                "reservation_id": reservation_id,
                "operation_id": reservation["operation_id"],
                "reservation_head": reservation["reservation_head"],
                "operation_identity_digest": reservation["operation_identity_digest"],
                "dispatch_attempt_id": attempt_id,
                "dispatch_attempt_sequence": attempt_sequence,
                "fencing_token": reservation["fencing_token"],
                "trust_root_generation": reservation["trust_root_generation"],
                "governance_generation": reservation["governance_generation"],
                "clock_id": reservation["clock_id"],
                "reserved_wall_time_ms": reservation["reserved_wall_time_ms"],
                "reserved_monotonic_ns": reservation["reserved_monotonic_ns"],
                "dispatch_wall_time_ms": dispatch_wall_time_ms,
                "dispatch_monotonic_ns": dispatch_monotonic_ns,
                "attempt_identity_digest": identity_digest,
                "previous_revision": int(state["revision"]),
                "revision": new_revision,
            }
            self._append_event(
                connection,
                event_id=f"dispatch-prepared:{reservation_id}:{attempt_sequence}",
                event_type="PromotionDispatchPrepared",
                reservation_id=reservation_id,
                fencing_token=observed_fencing_token,
                payload=payload,
            )
            return True

    def derive_attempt_identity(
        self, reservation_id: str
    ) -> PromotionTemporalAttemptIdentityV1 | None:
        if not self.verify_journal():
            return None
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            event = connection.execute(
                """SELECT payload_json FROM journal_events
                   WHERE event_type = 'PromotionDispatchPrepared' AND reservation_id = ?
                   ORDER BY seq DESC LIMIT 1""",
                (reservation_id,),
            ).fetchone()
            if row is None or event is None:
                return None
            reservation = dict(row)
            if reservation["state"] != "PromotionDispatchPrepared":
                return None
            identity = self._identity_from(
                reservation,
                dispatch_attempt_id=reservation["dispatch_attempt_id"],
                dispatch_attempt_sequence=reservation["dispatch_attempt_sequence"],
                dispatch_wall_time_ms=reservation["dispatch_wall_time_ms"],
                dispatch_monotonic_ns=reservation["dispatch_monotonic_ns"],
            )
            payload = json.loads(event["payload_json"])
            digest = identity.identity_digest()
            if (
                not identity.structurally_valid()
                or digest != reservation["attempt_identity_digest"]
                or digest != payload.get("attempt_identity_digest")
            ):
                return None
            return identity
        finally:
            connection.close()

    def verify_journal(self) -> bool:
        connection = self._connect()
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                return False
            triggers = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
            }
            if not self.REQUIRED_TRIGGERS <= triggers:
                return False

            events = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM journal_events ORDER BY seq"
                ).fetchall()
            ]
            if not events or len({event["event_id"] for event in events}) != len(events):
                return False
            previous_hash = GENESIS_HASH
            for expected_seq, event in enumerate(events, start=1):
                if event["seq"] != expected_seq or event["previous_hash"] != previous_hash:
                    return False
                payload = json.loads(event["payload_json"])
                if canonical_json(payload) != event["payload_json"]:
                    return False
                unsigned = {
                    "seq": event["seq"],
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "reservation_id": event["reservation_id"],
                    "fencing_token": event["fencing_token"],
                    "payload_json": event["payload_json"],
                    "previous_hash": event["previous_hash"],
                }
                if sha256_json(unsigned) != event["event_hash"]:
                    return False
                previous_hash = event["event_hash"]

            replay_state: dict[str, Any] | None = None
            replay_reservations: dict[str, dict[str, Any]] = {}
            for event in events:
                payload = json.loads(event["payload_json"])
                kind = event["event_type"]
                if kind == "LedgerInitialized":
                    if replay_state is not None or event["seq"] != 1 or event["reservation_id"] is not None:
                        return False
                    replay_state = dict(payload)
                    continue
                if replay_state is None:
                    return False

                if kind == "PromotionReserved":
                    reservation_id = payload["reservation_id"]
                    if (
                        reservation_id in replay_reservations
                        or payload["previous_revision"] != replay_state["revision"]
                        or payload["previous_fencing_token"] != replay_state["fencing_token"]
                        or payload["predecessor_head"] != replay_state["head"]
                        or payload["lease_id"] != replay_state["active_lease"]
                        or payload["trust_root_generation"] != replay_state["trust_root_generation"]
                        or payload["governance_generation"] != replay_state["governance_generation"]
                        or payload["fencing_token"] != replay_state["fencing_token"] + 1
                        or payload["created_revision"] != replay_state["revision"] + 1
                        or payload["reservation_head"] == ""
                        or event["reservation_id"] != reservation_id
                        or event["fencing_token"] != payload["fencing_token"]
                    ):
                        return False
                    replay_state["head"] = payload["reservation_head"]
                    replay_state["fencing_token"] = payload["fencing_token"]
                    replay_state["revision"] = payload["created_revision"]
                    replay_reservations[reservation_id] = {
                        **payload,
                        "state": "PromotionReserved",
                        "dispatch_attempt_id": None,
                        "dispatch_attempt_sequence": None,
                        "dispatch_wall_time_ms": None,
                        "dispatch_monotonic_ns": None,
                        "dispatch_clock_id": None,
                        "attempt_identity_digest": None,
                        "superseded_by_fence": None,
                    }
                elif kind == "AuthorityFenceAdvanced":
                    if (
                        event["reservation_id"] is not None
                        or payload["previous_revision"] != replay_state["revision"]
                        or payload["previous_fencing_token"] != replay_state["fencing_token"]
                        or payload["previous_lease_id"] != replay_state["active_lease"]
                        or payload["previous_trust_root_generation"] != replay_state["trust_root_generation"]
                        or payload["previous_governance_generation"] != replay_state["governance_generation"]
                        or payload["head"] != replay_state["head"]
                        or payload["fencing_token"] != replay_state["fencing_token"] + 1
                        or payload["revision"] != replay_state["revision"] + 1
                        or not payload["active_lease"]
                        or payload["trust_root_generation"] < 0
                        or payload["governance_generation"] < 0
                        or event["fencing_token"] != payload["fencing_token"]
                    ):
                        return False
                    replay_state.update({
                        "active_lease": payload["active_lease"],
                        "fencing_token": payload["fencing_token"],
                        "trust_root_generation": payload["trust_root_generation"],
                        "governance_generation": payload["governance_generation"],
                        "revision": payload["revision"],
                    })
                elif kind == "ReservationSuperseded":
                    reservation_id = payload["reservation_id"]
                    reservation = replay_reservations.get(reservation_id)
                    if (
                        reservation is None
                        or reservation["state"] != "PromotionReserved"
                        or payload["previous_fencing_token"] != reservation["fencing_token"]
                        or payload["superseded_by_fence"] != replay_state["fencing_token"]
                        or event["reservation_id"] != reservation_id
                        or event["fencing_token"] != replay_state["fencing_token"]
                    ):
                        return False
                    reservation["state"] = "PromotionSuperseded"
                    reservation["superseded_by_fence"] = payload["superseded_by_fence"]
                elif kind == "PromotionDispatchPrepared":
                    reservation_id = payload["reservation_id"]
                    reservation = replay_reservations.get(reservation_id)
                    if (
                        reservation is None
                        or reservation["state"] != "PromotionReserved"
                        or payload["previous_revision"] != replay_state["revision"]
                        or payload["revision"] != replay_state["revision"] + 1
                        or payload["operation_id"] != reservation["operation_id"]
                        or payload["reservation_head"] != replay_state["head"]
                        or payload["reservation_head"] != reservation["reservation_head"]
                        or payload["operation_identity_digest"] != reservation["operation_identity_digest"]
                        or payload["fencing_token"] != replay_state["fencing_token"]
                        or payload["fencing_token"] != reservation["fencing_token"]
                        or payload["trust_root_generation"] != replay_state["trust_root_generation"]
                        or payload["trust_root_generation"] != reservation["trust_root_generation"]
                        or payload["governance_generation"] != replay_state["governance_generation"]
                        or payload["governance_generation"] != reservation["governance_generation"]
                        or payload["clock_id"] != reservation["clock_id"]
                        or payload["reserved_wall_time_ms"] != reservation["reserved_wall_time_ms"]
                        or payload["reserved_monotonic_ns"] != reservation["reserved_monotonic_ns"]
                        or payload["dispatch_wall_time_ms"] < payload["reserved_wall_time_ms"]
                        or payload["dispatch_monotonic_ns"] < payload["reserved_monotonic_ns"]
                        or event["reservation_id"] != reservation_id
                        or event["fencing_token"] != payload["fencing_token"]
                    ):
                        return False
                    identity = self._identity_from(
                        reservation,
                        dispatch_attempt_id=payload["dispatch_attempt_id"],
                        dispatch_attempt_sequence=payload["dispatch_attempt_sequence"],
                        dispatch_wall_time_ms=payload["dispatch_wall_time_ms"],
                        dispatch_monotonic_ns=payload["dispatch_monotonic_ns"],
                    )
                    if (
                        not identity.structurally_valid()
                        or identity.identity_digest() != payload["attempt_identity_digest"]
                    ):
                        return False
                    replay_state["revision"] = payload["revision"]
                    reservation.update({
                        "state": "PromotionDispatchPrepared",
                        "dispatch_attempt_id": payload["dispatch_attempt_id"],
                        "dispatch_attempt_sequence": payload["dispatch_attempt_sequence"],
                        "dispatch_wall_time_ms": payload["dispatch_wall_time_ms"],
                        "dispatch_monotonic_ns": payload["dispatch_monotonic_ns"],
                        "dispatch_clock_id": payload["clock_id"],
                        "attempt_identity_digest": payload["attempt_identity_digest"],
                    })
                else:
                    return False

            if replay_state is None:
                return False
            materialized_state = dict(connection.execute(
                "SELECT * FROM ledger_state WHERE singleton = 1"
            ).fetchone())
            if materialized_state.pop("singleton") != 1:
                return False
            if replay_state != materialized_state:
                return False

            materialized_reservations = {
                row["reservation_id"]: dict(row)
                for row in connection.execute("SELECT * FROM reservations").fetchall()
            }
            if set(materialized_reservations) != set(replay_reservations):
                return False
            compared_fields = (
                "reservation_id", "operation_id", "lease_id", "predecessor_head",
                "reservation_head", "state", "fencing_token", "trust_root_generation",
                "governance_generation", "operation_identity_digest", "clock_id",
                "reserved_wall_time_ms", "reserved_monotonic_ns", "created_revision",
                "dispatch_attempt_id", "dispatch_attempt_sequence", "dispatch_wall_time_ms",
                "dispatch_monotonic_ns", "dispatch_clock_id", "attempt_identity_digest",
                "superseded_by_fence",
            )
            for reservation_id, actual in materialized_reservations.items():
                expected = replay_reservations[reservation_id]
                if any(actual.get(field) != expected.get(field) for field in compared_fields):
                    return False
            return True
        except (sqlite3.Error, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        finally:
            connection.close()


class DurablePromotionJournalTests(unittest.TestCase):
    def make_journal(self) -> DurablePromotionJournalV1:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return DurablePromotionJournalV1(Path(temporary.name) / "promotion.sqlite3")

    @staticmethod
    def reserve_one(journal: DurablePromotionJournalV1) -> None:
        self_digest = hashlib.sha256(b"operation-1").hexdigest()
        assert journal.reserve(
            observed_head="L0",
            lease_id="LEASE-1",
            candidate="L1",
            reservation_id="RES-1",
            operation_id="OP-1",
            trust_root_generation=1,
            governance_generation=1,
            operation_identity_digest=self_digest,
            clock_id="clock-A",
            reserved_wall_time_ms=1000,
            reserved_monotonic_ns=100000,
        )

    @staticmethod
    def prepare_one(
        journal: DurablePromotionJournalV1,
        *,
        attempt_id: str = "ATTEMPT-1",
        sequence: int = 9,
        wall_ms: int = 1010,
        monotonic_ns: int = 101000,
        clock_id: str = "clock-A",
    ) -> bool:
        return journal.prepare_dispatch(
            reservation_id="RES-1",
            observed_head="L1",
            observed_trust_root_generation=1,
            observed_governance_generation=1,
            observed_fencing_token=1,
            attempt_id=attempt_id,
            attempt_sequence=sequence,
            dispatch_wall_time_ms=wall_ms,
            dispatch_monotonic_ns=monotonic_ns,
            dispatch_clock_id=clock_id,
        )

    def test_prepared_identity_is_derived_from_committed_journal_after_reopen(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        self.assertTrue(self.prepare_one(journal))
        identity = journal.derive_attempt_identity("RES-1")
        self.assertIsNotNone(identity)
        self.assertEqual(identity.dispatch_attempt_sequence, 9)
        self.assertEqual(identity.local_monotonic_clock_id, "clock-A")
        self.assertTrue(journal.verify_journal())

        reopened = DurablePromotionJournalV1(journal.path)
        reopened_identity = reopened.derive_attempt_identity("RES-1")
        self.assertIsNotNone(reopened_identity)
        self.assertEqual(reopened_identity.identity_digest(), identity.identity_digest())
        self.assertEqual(reopened.current_state()["revision"], 2)
        self.assertTrue(reopened.verify_journal())

    def test_dispatch_sequence_is_single_use_and_positive(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        self.assertFalse(self.prepare_one(journal, sequence=0))
        self.assertEqual(journal.get_reservation("RES-1")["state"], "PromotionReserved")
        self.assertTrue(self.prepare_one(journal, sequence=9))
        self.assertFalse(self.prepare_one(journal, attempt_id="ATTEMPT-2", sequence=10))
        self.assertFalse(self.prepare_one(journal, attempt_id="ATTEMPT-3", sequence=9))
        self.assertEqual(journal.get_reservation("RES-1")["dispatch_attempt_sequence"], 9)
        self.assertTrue(journal.verify_journal())

    def test_clock_identity_and_both_time_orders_fail_closed(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        self.assertFalse(self.prepare_one(journal, clock_id="another-runtime"))
        self.assertFalse(self.prepare_one(journal, wall_ms=999, monotonic_ns=101001))
        self.assertFalse(self.prepare_one(journal, wall_ms=1010, monotonic_ns=99999))
        self.assertTrue(self.prepare_one(journal, wall_ms=1010, monotonic_ns=101000))
        self.assertTrue(journal.verify_journal())

    def test_failed_journal_append_rolls_back_prepared_state_and_revision(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        before_revision = journal.current_state()["revision"]
        connection = sqlite3.connect(journal.path)
        try:
            connection.execute(
                """CREATE TRIGGER fail_dispatch_journal_insert
                   BEFORE INSERT ON journal_events
                   WHEN NEW.event_type = 'PromotionDispatchPrepared'
                   BEGIN SELECT RAISE(ABORT, 'injected journal append failure'); END"""
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(sqlite3.IntegrityError):
            self.prepare_one(journal)
        self.assertEqual(journal.current_state()["revision"], before_revision)
        self.assertEqual(journal.get_reservation("RES-1")["state"], "PromotionReserved")
        self.assertEqual(journal.event_count(), 2)
        self.assertTrue(journal.verify_journal())

    def test_silent_reservation_compare_and_swap_miss_rolls_back_revision(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        before_state = journal.current_state()
        before_events = journal.event_count()
        connection = sqlite3.connect(journal.path)
        try:
            connection.execute(
                """CREATE TRIGGER ignore_dispatch_transition
                   BEFORE UPDATE OF state ON reservations
                   WHEN OLD.reservation_id = 'RES-1'
                    AND NEW.state = 'PromotionDispatchPrepared'
                   BEGIN SELECT RAISE(IGNORE); END"""
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(RuntimeError, "compare-and-swap failed"):
            self.prepare_one(journal)
        self.assertEqual(journal.current_state(), before_state)
        self.assertEqual(journal.get_reservation("RES-1")["state"], "PromotionReserved")
        self.assertEqual(journal.event_count(), before_events)
        self.assertTrue(journal.verify_journal())

    def test_database_trigger_rejects_stale_writer_even_if_api_guard_is_bypassed(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        connection = sqlite3.connect(journal.path, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE ledger_state SET fencing_token = 2, revision = 2 WHERE singleton = 1"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """UPDATE reservations
                       SET state = 'PromotionDispatchPrepared',
                           dispatch_attempt_id = 'DIRECT-ATTEMPT',
                           dispatch_attempt_sequence = 9,
                           dispatch_wall_time_ms = 1010,
                           dispatch_monotonic_ns = 101000,
                           dispatch_clock_id = 'clock-A',
                           attempt_identity_digest = ?
                       WHERE reservation_id = 'RES-1'""",
                    ("a" * 64,),
                )
        finally:
            connection.rollback()
            connection.close()
        self.assertTrue(journal.verify_journal())
        self.assertEqual(journal.get_reservation("RES-1")["state"], "PromotionReserved")

    def test_stale_connection_cannot_dispatch_after_fence_rotation(self) -> None:
        first = self.make_journal()
        second = DurablePromotionJournalV1(first.path)
        self.reserve_one(first)
        self.assertTrue(second.advance_fence(
            active_lease="LEASE-2",
            trust_root_generation=2,
            governance_generation=2,
        ))
        self.assertEqual(second.get_reservation("RES-1")["state"], "PromotionSuperseded")
        self.assertTrue(second.reserve(
            observed_head="L1",
            lease_id="LEASE-2",
            candidate="L2",
            reservation_id="RES-2",
            operation_id="OP-2",
            trust_root_generation=2,
            governance_generation=2,
            operation_identity_digest=hashlib.sha256(b"operation-2").hexdigest(),
            clock_id="clock-A",
            reserved_wall_time_ms=2000,
            reserved_monotonic_ns=200000,
        ))
        self.assertFalse(self.prepare_one(first))
        self.assertEqual(first.get_reservation("RES-1")["state"], "PromotionSuperseded")
        self.assertTrue(first.verify_journal())

    def test_concurrent_connections_have_exactly_one_dispatch_winner(self) -> None:
        first = self.make_journal()
        second = DurablePromotionJournalV1(first.path)
        self.reserve_one(first)
        barrier = threading.Barrier(2)

        def attempt(journal: DurablePromotionJournalV1, sequence: int) -> bool:
            barrier.wait(timeout=5)
            return self.prepare_one(
                journal,
                attempt_id=f"ATTEMPT-{sequence}",
                sequence=sequence,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda args: attempt(*args),
                [(first, 9), (second, 10)],
            ))
        self.assertEqual(sum(results), 1, results)
        self.assertEqual(
            first.get_reservation("RES-1")["state"],
            "PromotionDispatchPrepared",
        )
        self.assertTrue(first.verify_journal())

    def test_journal_is_append_only_through_sqlite_triggers(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        connection = sqlite3.connect(journal.path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE journal_events SET payload_json = payload_json WHERE seq = 1")
            connection.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM journal_events WHERE seq = 1")
            connection.rollback()
        finally:
            connection.close()
        self.assertTrue(journal.verify_journal())

    def test_chain_audit_detects_mutated_event_payload(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        connection = sqlite3.connect(journal.path)
        try:
            connection.execute("DROP TRIGGER journal_events_no_update")
            connection.execute(
                "UPDATE journal_events SET payload_json = '{}' WHERE event_type = 'PromotionReserved'"
            )
            connection.commit()
        finally:
            connection.close()
        self.assertFalse(journal.verify_journal())
        self.assertIsNone(journal.derive_attempt_identity("RES-1"))

    def test_storage_guard_rejects_fence_regression_and_head_rewrite(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        before = journal.current_state()
        connection = sqlite3.connect(journal.path, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """UPDATE ledger_state
                       SET fencing_token = 0, revision = revision + 1
                       WHERE singleton = 1"""
                )
            connection.rollback()

            connection.execute("BEGIN IMMEDIATE")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """UPDATE ledger_state
                       SET head = 'L0', revision = revision + 1
                       WHERE singleton = 1"""
                )
            connection.rollback()
        finally:
            connection.close()
        self.assertEqual(journal.current_state(), before)
        self.assertTrue(journal.verify_journal())

    def test_materialized_authority_cannot_drift_from_replayed_journal(self) -> None:
        journal = self.make_journal()
        self.reserve_one(journal)
        connection = sqlite3.connect(journal.path)
        try:
            connection.execute("DROP TRIGGER ledger_state_fence_monotonicity_guard")
            connection.execute(
                "UPDATE ledger_state SET trust_root_generation = 22 WHERE singleton = 1"
            )
            connection.commit()
        finally:
            connection.close()
        self.assertFalse(journal.verify_journal())

    def test_wrong_reservation_head_cannot_consume_a_fence_or_event(self) -> None:
        journal = self.make_journal()
        before_state = journal.current_state()
        before_count = journal.event_count()
        self.assertFalse(journal.reserve(
            observed_head="stale",
            lease_id="LEASE-1",
            candidate="L1",
            reservation_id="RES-1",
            operation_id="OP-1",
            trust_root_generation=1,
            governance_generation=1,
            operation_identity_digest=hashlib.sha256(b"op").hexdigest(),
            clock_id="clock-A",
            reserved_wall_time_ms=1000,
            reserved_monotonic_ns=100000,
        ))
        self.assertEqual(journal.current_state(), before_state)
        self.assertEqual(journal.event_count(), before_count)
        self.assertTrue(journal.verify_journal())


if __name__ == "__main__":
    unittest.main(verbosity=2)
