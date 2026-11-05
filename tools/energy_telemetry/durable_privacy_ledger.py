"""Durable per-participant differential-privacy budget accounting prototype.

This ledger reserves cumulative epsilon/delta atomically and prevents overlapping
window releases for participants appearing in multiple outputs. Raw participant
tokens are never persisted; a stable caller-managed HMAC key maps them to
database-local tags. It does NOT generate DP noise, verify contribution clipping,
or prove the caller's query/mechanism is differentially private.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
from typing import Iterable, Iterator

from tools.energy_telemetry.store_forward import (
    EventIdentityConflict,
    QueueCapacityError,
    TelemetryQueueError,
    _utc,
)

_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x4450524C  # ASCII 'DPRL'
MINIMUM_COHORT_SIZE = 20
MAX_COHORT_PARTICIPANTS = 10_000  # hard resource ceiling; review before changing
EPSILON_SCALE = 1_000_000_000
DELTA_SCALE = 1_000_000_000_000
MAX_RELEASE_EPSILON_NANO = EPSILON_SCALE
MAX_RELEASE_DELTA_PICO = 1_000_000
MAX_DP_OUTPUT_BYTES = 1_048_576
RELEASE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
POLICY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MECHANISMS = frozenset({"discrete_laplace", "gaussian"})
ADJACENCIES = frozenset({"add_remove_one_participant", "replace_one_participant"})
RECIPIENTS = frozenset({"approved_research", "public_reporting"})


class DurablePrivacyPolicyError(ValueError):
    """A privacy-release reservation violates durable policy."""


@dataclass(frozen=True)
class PrivacyReservation:
    release_id: str
    receipt_sha256: str
    cohort_size: int
    epsilon: float
    delta: float
    status: str
    output_sha256: str | None


@dataclass(frozen=True)
class CommittedPrivacyOutput:
    release_id: str
    receipt_sha256: str
    output_sha256: str
    output_bytes: bytes
    duplicate: bool


@dataclass(frozen=True)
class PrivacyReservationContext:
    """Non-sensitive policy context needed to verify a release attestation."""

    release_id: str
    receipt_sha256: str
    query_sha256: str
    window_start_utc: str
    window_end_utc: str
    recipient_class: str
    mechanism: str
    adjacency: str
    contribution_limit_wh: int
    contribution_policy_id: str
    privacy_policy_id: str
    epsilon_nano: int
    delta_pico: int
    cohort_size: int


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _scaled(value: int | float, *, scale: int, allow_zero: bool, field: str, maximum: int) -> int:
    if type(value) not in (int, float):
        raise DurablePrivacyPolicyError(f"{field} must be a finite numeric value")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise DurablePrivacyPolicyError(f"{field} must be a finite numeric value") from exc
    if not number.is_finite():
        raise DurablePrivacyPolicyError(f"{field} must be a finite numeric value")
    lower_ok = number >= 0 if allow_zero else number > 0
    if not lower_ok:
        comparator = "non-negative" if allow_zero else "positive"
        raise DurablePrivacyPolicyError(f"{field} must be {comparator}")
    if number > Decimal(maximum) / Decimal(scale):
        raise DurablePrivacyPolicyError(f"{field} exceeds the configured release limit")
    scaled = int((number * scale).to_integral_value(rounding=ROUND_CEILING))
    if scaled > maximum:
        raise DurablePrivacyPolicyError(f"{field} exceeds the configured release limit")
    return scaled


def _is_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DurablePrivacyPolicyError(f"{field} must be a lowercase SHA-256 digest")
    return value


class DurablePrivacyLedger:
    """SQLite budget ledger with HMAC participant tags and immutable release output.

    Reservation spends budget permanently, including if a caller crashes before
    generating the output. After generation, commit_output binds one exact byte
    string to that reservation. Callers must publish/use only the bytes returned
    by commit_output (including on retries) and must not publish an uncommitted
    freshly-generated output.
    """

    def __init__(
        self,
        database_path: str,
        *,
        participant_hmac_key: bytes,
        max_epsilon_per_participant: int | float = 2.0,
        max_delta_per_participant: int | float = 2e-6,
        max_reservations: int = 100_000,
        minimum_cohort_size: int = MINIMUM_COHORT_SIZE,
        busy_timeout_seconds: int | float = 5.0,
    ) -> None:
        if (
            not isinstance(database_path, str)
            or not database_path.strip()
            or "\x00" in database_path
            or database_path == ":memory:"
        ):
            raise ValueError("database_path must be a persistent filesystem path, not :memory:")
        if not isinstance(participant_hmac_key, bytes) or len(participant_hmac_key) < 32:
            raise ValueError("participant_hmac_key must be at least 32 bytes")
        sqlite_int_max = (1 << 63) - 1
        if type(max_reservations) is not int or not 1 <= max_reservations <= sqlite_int_max:
            raise ValueError("max_reservations must be a positive SQLite integer")
        if type(minimum_cohort_size) is not int or minimum_cohort_size < MINIMUM_COHORT_SIZE:
            raise ValueError(f"minimum_cohort_size must be at least {MINIMUM_COHORT_SIZE}")
        if (
            type(busy_timeout_seconds) not in (int, float)
            or not 0 < busy_timeout_seconds <= 300
        ):
            raise ValueError("busy_timeout_seconds must be finite and in (0, 300]")
        self.database_path = database_path
        self._participant_hmac_key = participant_hmac_key
        self.max_epsilon_nano = _scaled(
            max_epsilon_per_participant, scale=EPSILON_SCALE, allow_zero=False,
            field="max_epsilon_per_participant", maximum=sqlite_int_max,
        )
        self.max_delta_pico = _scaled(
            max_delta_per_participant, scale=DELTA_SCALE, allow_zero=True,
            field="max_delta_per_participant", maximum=sqlite_int_max,
        )
        self.max_reservations = max_reservations
        self.minimum_cohort_size = minimum_cohort_size
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._key_check = hmac.new(
            participant_hmac_key, b"durable-privacy-ledger-key-check:v1", hashlib.sha256
        ).hexdigest()
        self._ownership_verified = False
        self._initialize()
        self._ownership_verified = True
        self._secure_file_modes()

    @staticmethod
    def _timestamp(value: datetime, field: str = "timestamp") -> str:
        _utc(value, field)
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _participant_tag(self, token: str) -> str:
        return hmac.new(
            self._participant_hmac_key,
            b"telemetry-dp-participant:v1\x00" + token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(
            self.database_path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        try:
            db.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
            db.execute("PRAGMA foreign_keys = ON")
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
                raise TelemetryQueueError("unable to enforce private privacy-ledger file permissions") from exc

    def _initialize(self) -> None:
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA_VERSION):
                raise TelemetryQueueError(f"unsupported privacy ledger schema version: {version}")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            if app_id not in (0, _APPLICATION_ID):
                raise TelemetryQueueError("SQLite database belongs to a different application")
            allowed = {
                "privacy_releases", "privacy_release_members",
                "privacy_participant_spend", "privacy_ledger_config",
                "privacy_release_attestations",
            }
            if app_id == 0:
                tables = {
                    row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                if tables - allowed:
                    raise TelemetryQueueError("refusing to claim unowned SQLite database with unrelated tables")
            mode = db.execute("PRAGMA journal_mode").fetchone()[0].lower()
            if mode != "wal":
                mode = db.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower()
            if mode != "wal":
                raise TelemetryQueueError("SQLite WAL mode is required but unavailable")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_releases (
                    release_id TEXT PRIMARY KEY,
                    query_sha256 TEXT NOT NULL,
                    window_start_utc TEXT NOT NULL,
                    window_end_utc TEXT NOT NULL,
                    recipient_class TEXT NOT NULL CHECK(recipient_class IN ('approved_research','public_reporting')),
                    mechanism TEXT NOT NULL CHECK(mechanism IN ('discrete_laplace','gaussian')),
                    adjacency TEXT NOT NULL CHECK(adjacency IN ('add_remove_one_participant','replace_one_participant')),
                    contribution_limit_wh INTEGER NOT NULL CHECK(contribution_limit_wh > 0),
                    contribution_policy_id TEXT NOT NULL,
                    privacy_policy_id TEXT NOT NULL,
                    epsilon_nano INTEGER NOT NULL CHECK(epsilon_nano BETWEEN 1 AND 1000000000),
                    delta_pico INTEGER NOT NULL CHECK(delta_pico BETWEEN 0 AND 1000000),
                    minimum_cohort_size INTEGER NOT NULL CHECK(minimum_cohort_size >= 20),
                    cohort_size INTEGER NOT NULL CHECK(cohort_size >= minimum_cohort_size),
                    participants_sha256 TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('reserved','output_committed')),
                    output_sha256 TEXT,
                    output_bytes BLOB,
                    reserved_at_utc TEXT NOT NULL,
                    committed_at_utc TEXT,
                    CHECK(
                        (status='reserved' AND output_sha256 IS NULL AND output_bytes IS NULL AND committed_at_utc IS NULL)
                        OR
                        (status='output_committed' AND output_sha256 IS NOT NULL AND output_bytes IS NOT NULL AND committed_at_utc IS NOT NULL)
                    )
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_release_members (
                    release_id TEXT NOT NULL,
                    participant_tag TEXT NOT NULL,
                    PRIMARY KEY(release_id, participant_tag),
                    FOREIGN KEY(release_id) REFERENCES privacy_releases(release_id) ON DELETE RESTRICT
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_participant_spend (
                    participant_tag TEXT PRIMARY KEY,
                    epsilon_nano_spent INTEGER NOT NULL CHECK(epsilon_nano_spent >= 0),
                    delta_pico_spent INTEGER NOT NULL CHECK(delta_pico_spent >= 0),
                    release_count INTEGER NOT NULL CHECK(release_count >= 0)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_ledger_config (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    max_epsilon_nano INTEGER NOT NULL CHECK(max_epsilon_nano > 0),
                    max_delta_pico INTEGER NOT NULL CHECK(max_delta_pico >= 0),
                    max_reservations INTEGER NOT NULL CHECK(max_reservations > 0),
                    minimum_cohort_size INTEGER NOT NULL CHECK(minimum_cohort_size >= 20),
                    participant_key_check TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_release_attestations (
                    release_id TEXT PRIMARY KEY,
                    attestation_sha256 TEXT NOT NULL CHECK(length(attestation_sha256) = 64),
                    attestation_bytes BLOB NOT NULL CHECK(length(attestation_bytes) BETWEEN 1 AND 16384),
                    signature BLOB NOT NULL CHECK(length(signature) = 64),
                    signer_key_sha256 TEXT NOT NULL CHECK(length(signer_key_sha256) = 64),
                    FOREIGN KEY(release_id) REFERENCES privacy_releases(release_id) ON DELETE RESTRICT
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
                    if tables - allowed:
                        raise TelemetryQueueError("refusing to claim unowned SQLite database with unrelated tables")
                    db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                expected = (
                    self.max_epsilon_nano, self.max_delta_pico, self.max_reservations,
                    self.minimum_cohort_size, self._key_check,
                )
                current = db.execute(
                    "SELECT max_epsilon_nano, max_delta_pico, max_reservations, "
                    "minimum_cohort_size, participant_key_check "
                    "FROM privacy_ledger_config WHERE singleton = 1"
                ).fetchone()
                if current is None:
                    db.execute(
                        "INSERT INTO privacy_ledger_config "
                        "(singleton,max_epsilon_nano,max_delta_pico,max_reservations,minimum_cohort_size,participant_key_check) "
                        "VALUES (1,?,?,?,?,?)",
                        expected,
                    )
                elif tuple(current) != expected:
                    raise TelemetryQueueError(
                        "privacy ledger budget, cohort or participant-key configuration mismatch"
                    )
                db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                db.execute("COMMIT")
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    @property
    def reservation_count(self) -> int:
        with self._connection() as db:
            return db.execute("SELECT COUNT(*) FROM privacy_releases").fetchone()[0]

    def participant_spend(self, participant_token: str) -> tuple[float, float, int]:
        """Inspect one participant's local cumulative budget; never returns their stored tag."""
        if not isinstance(participant_token, str) or not participant_token:
            raise DurablePrivacyPolicyError("participant token must be a non-empty local string")
        tag = self._participant_tag(participant_token)
        with self._connection() as db:
            row = db.execute(
                "SELECT epsilon_nano_spent, delta_pico_spent, release_count "
                "FROM privacy_participant_spend WHERE participant_tag = ?",
                (tag,),
            ).fetchone()
        if row is None:
            return 0.0, 0.0, 0
        return row[0] / EPSILON_SCALE, row[1] / DELTA_SCALE, row[2]

    def reserve_release(
        self,
        *,
        release_id: str,
        query_sha256: str,
        participant_tokens: Iterable[str],
        window_start_utc: datetime,
        window_end_utc: datetime,
        reserved_at_utc: datetime,
        recipient_class: str,
        mechanism: str,
        epsilon: int | float,
        delta: int | float,
        adjacency: str,
        contribution_limit_wh: int,
        contribution_policy_id: str,
        privacy_policy_id: str,
        minimum_cohort_size: int | None = None,
    ) -> PrivacyReservation:
        """Spend budget durably before a DP output may be published.

        Participant tokens are HMACed locally and never written as raw tokens.
        The ledger requires a named contribution-bound policy but cannot prove
        that the upstream aggregator actually applies clipping/bounds.
        """
        if not isinstance(release_id, str) or not RELEASE_ID_RE.fullmatch(release_id):
            raise DurablePrivacyPolicyError("release_id must be a random 32-character lowercase hex ID")
        query_sha256 = _is_sha256(query_sha256, "query_sha256")
        start_text = self._timestamp(window_start_utc, "window_start_utc")
        end_text = self._timestamp(window_end_utc, "window_end_utc")
        reserved_text = self._timestamp(reserved_at_utc, "reserved_at_utc")
        if window_end_utc <= window_start_utc:
            raise DurablePrivacyPolicyError("window_end_utc must be later than window_start_utc")
        if window_end_utc > reserved_at_utc + self.max_future_skew_for_ledger():
            raise DurablePrivacyPolicyError("release window end exceeds allowed future clock skew")
        if not isinstance(recipient_class, str) or recipient_class not in RECIPIENTS:
            raise DurablePrivacyPolicyError("durable DP ledger only reserves research/public releases")
        if not isinstance(mechanism, str) or mechanism not in MECHANISMS:
            raise DurablePrivacyPolicyError("unsupported differential-privacy mechanism")
        if not isinstance(adjacency, str) or adjacency not in ADJACENCIES:
            raise DurablePrivacyPolicyError("unsupported neighboring-dataset adjacency")
        if type(contribution_limit_wh) is not int or not 1 <= contribution_limit_wh <= 10**15:
            raise DurablePrivacyPolicyError("contribution_limit_wh must be a positive bounded integer")
        for value, name in ((contribution_policy_id, "contribution_policy_id"), (privacy_policy_id, "privacy_policy_id")):
            if not isinstance(value, str) or not POLICY_RE.fullmatch(value):
                raise DurablePrivacyPolicyError(f"{name} must be a 1-128 character policy identifier")
        epsilon_nano = _scaled(
            epsilon, scale=EPSILON_SCALE, allow_zero=False,
            field="epsilon", maximum=MAX_RELEASE_EPSILON_NANO,
        )
        delta_pico = _scaled(
            delta, scale=DELTA_SCALE, allow_zero=True,
            field="delta", maximum=MAX_RELEASE_DELTA_PICO,
        )
        # Mechanism labels must agree with their privacy parameters.
        if mechanism == "discrete_laplace" and delta_pico != 0:
            raise DurablePrivacyPolicyError("discrete_laplace requires delta=0")
        if mechanism == "gaussian" and delta_pico == 0:
            raise DurablePrivacyPolicyError("gaussian mechanism requires positive delta")

        floor = self.minimum_cohort_size if minimum_cohort_size is None else minimum_cohort_size
        if type(floor) is not int or floor < self.minimum_cohort_size:
            raise DurablePrivacyPolicyError(
                f"minimum_cohort_size cannot be below policy floor {self.minimum_cohort_size}"
            )
        if isinstance(participant_tokens, (str, bytes)):
            raise DurablePrivacyPolicyError(
                "participant_tokens must be an iterable of local tokens, not a single string"
            )
        # Bound materialization before hashing or opening the SQLite write transaction.
        # Otherwise an unbounded iterable can exhaust memory before policy rejects it.
        tokens = []
        try:
            for token in participant_tokens:
                if len(tokens) >= MAX_COHORT_PARTICIPANTS:
                    raise DurablePrivacyPolicyError(
                        f"participant cohort exceeds hard maximum of {MAX_COHORT_PARTICIPANTS}"
                    )
                tokens.append(token)
        except DurablePrivacyPolicyError:
            raise
        except (TypeError, ValueError) as exc:
            raise DurablePrivacyPolicyError("participant_tokens must be an iterable of local tokens") from exc
        if any(not isinstance(token, str) or not token.strip() or len(token) > 1024 for token in tokens):
            raise DurablePrivacyPolicyError("participant tokens must be non-empty bounded local strings")
        if len(set(tokens)) != len(tokens):
            raise DurablePrivacyPolicyError("participant_tokens must be unique; duplicates are not silently collapsed")
        participant_tags = sorted({self._participant_tag(token) for token in tokens})
        if len(participant_tags) != len(tokens):
            raise DurablePrivacyPolicyError("participant token tag collision detected")
        if len(participant_tags) < floor:
            raise DurablePrivacyPolicyError("participant cohort is below the required minimum size")

        participants_sha256 = hashlib.sha256(_canonical(participant_tags)).hexdigest()
        receipt_fields = {
            "adjacency": adjacency,
            "contribution_limit_wh": contribution_limit_wh,
            "contribution_policy_id": contribution_policy_id,
            "delta_pico": delta_pico,
            "epsilon_nano": epsilon_nano,
            "mechanism": mechanism,
            "minimum_cohort_size": floor,
            "participants_sha256": participants_sha256,
            "privacy_policy_id": privacy_policy_id,
            "query_sha256": query_sha256,
            "recipient_class": recipient_class,
            "release_id": release_id,
            "window_end_utc": end_text,
            "window_start_utc": start_text,
        }
        receipt_sha256 = hashlib.sha256(_canonical(receipt_fields)).hexdigest()

        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    "SELECT query_sha256, window_start_utc, window_end_utc, recipient_class, "
                    "mechanism, adjacency, contribution_limit_wh, contribution_policy_id, privacy_policy_id, "
                    "epsilon_nano, delta_pico, minimum_cohort_size, cohort_size, participants_sha256, "
                    "receipt_sha256, status, output_sha256 "
                    "FROM privacy_releases WHERE release_id = ?",
                    (release_id,),
                ).fetchone()
                if existing is not None:
                    expected = (
                        query_sha256, start_text, end_text, recipient_class,
                        mechanism, adjacency, contribution_limit_wh, contribution_policy_id,
                        privacy_policy_id, epsilon_nano, delta_pico, floor,
                        len(participant_tags), participants_sha256, receipt_sha256,
                    )
                    if tuple(existing[:15]) != expected:
                        raise EventIdentityConflict("release_id was reused with different query/privacy policy")
                    stored_members = sorted(row[0] for row in db.execute(
                        "SELECT participant_tag FROM privacy_release_members WHERE release_id = ?",
                        (release_id,),
                    ).fetchall())
                    if stored_members != participant_tags:
                        raise EventIdentityConflict("release_id participant set changed")
                    db.execute("COMMIT")
                    return PrivacyReservation(
                        release_id=release_id,
                        receipt_sha256=existing[14],
                        cohort_size=existing[12],
                        epsilon=existing[9] / EPSILON_SCALE,
                        delta=existing[10] / DELTA_SCALE,
                        status=existing[15],
                        output_sha256=existing[16],
                    )

                count = db.execute("SELECT COUNT(*) FROM privacy_releases").fetchone()[0]
                if count >= self.max_reservations:
                    raise QueueCapacityError(
                        "privacy reservation limit reached; budget history will not be pruned implicitly"
                    )
                # Conservative query policy: any overlapping windows for a
                # shared participant are rejected to block simple differencing.
                for tag in participant_tags:
                    overlap = db.execute(
                        """
                        SELECT 1 FROM privacy_release_members m
                        JOIN privacy_releases r ON r.release_id = m.release_id
                        WHERE m.participant_tag = ?
                          AND r.window_start_utc < ?
                          AND r.window_end_utc > ?
                        LIMIT 1
                        """,
                        (tag, end_text, start_text),
                    ).fetchone()
                    if overlap is not None:
                        raise DurablePrivacyPolicyError(
                            "overlapping release window for a shared participant is not allowed"
                        )

                for tag in participant_tags:
                    spent = db.execute(
                        "SELECT epsilon_nano_spent, delta_pico_spent, release_count "
                        "FROM privacy_participant_spend WHERE participant_tag = ?",
                        (tag,),
                    ).fetchone()
                    epsilon_spent, delta_spent, release_count = (0, 0, 0) if spent is None else spent
                    if epsilon_spent + epsilon_nano > self.max_epsilon_nano:
                        raise DurablePrivacyPolicyError("cumulative epsilon budget exceeded")
                    if delta_spent + delta_pico > self.max_delta_pico:
                        raise DurablePrivacyPolicyError("cumulative delta budget exceeded")

                db.execute(
                    """
                    INSERT INTO privacy_releases (
                        release_id, query_sha256, window_start_utc, window_end_utc,
                        recipient_class, mechanism, adjacency, contribution_limit_wh,
                        contribution_policy_id, privacy_policy_id, epsilon_nano, delta_pico,
                        minimum_cohort_size, cohort_size, participants_sha256, receipt_sha256,
                        status, output_sha256, output_bytes, reserved_at_utc, committed_at_utc
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'reserved', NULL, NULL, ?, NULL)
                    """,
                    (
                        release_id, query_sha256, start_text, end_text,
                        recipient_class, mechanism, adjacency, contribution_limit_wh,
                        contribution_policy_id, privacy_policy_id, epsilon_nano, delta_pico,
                        floor, len(participant_tags), participants_sha256, receipt_sha256,
                        reserved_text,
                    ),
                )
                for tag in participant_tags:
                    db.execute(
                        "INSERT INTO privacy_release_members (release_id, participant_tag) VALUES (?,?)",
                        (release_id, tag),
                    )
                    spent = db.execute(
                        "SELECT epsilon_nano_spent, delta_pico_spent, release_count "
                        "FROM privacy_participant_spend WHERE participant_tag = ?",
                        (tag,),
                    ).fetchone()
                    if spent is None:
                        db.execute(
                            "INSERT INTO privacy_participant_spend "
                            "(participant_tag,epsilon_nano_spent,delta_pico_spent,release_count) VALUES (?,?,?,1)",
                            (tag, epsilon_nano, delta_pico),
                        )
                    else:
                        db.execute(
                            "UPDATE privacy_participant_spend SET epsilon_nano_spent = ?, "
                            "delta_pico_spent = ?, release_count = ? WHERE participant_tag = ?",
                            (spent[0] + epsilon_nano, spent[1] + delta_pico, spent[2] + 1, tag),
                        )
                db.execute("COMMIT")
                return PrivacyReservation(
                    release_id=release_id,
                    receipt_sha256=receipt_sha256,
                    cohort_size=len(participant_tags),
                    epsilon=epsilon_nano / EPSILON_SCALE,
                    delta=delta_pico / DELTA_SCALE,
                    status="reserved",
                    output_sha256=None,
                )
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def max_future_skew_for_ledger(self) -> timedelta:
        # The privacy ledger's timestamp window may not extend far past the
        # reservation time. Keep a small constant consistent with telemetry.
        return timedelta(seconds=300)

    def reservation_context(self, release_id: str) -> PrivacyReservationContext:
        """Return the exact stored policy fields to which a signed attestation must bind.

        This is an audit/read interface. It does not assert that the claimed
        mechanism ran; that claim must be checked by a separately configured
        trusted-attestation verifier.
        """
        if not isinstance(release_id, str) or not RELEASE_ID_RE.fullmatch(release_id):
            raise DurablePrivacyPolicyError("release_id must be a random 32-character lowercase hex ID")
        with self._connection() as db:
            row = db.execute(
                "SELECT receipt_sha256, query_sha256, window_start_utc, window_end_utc, "
                "recipient_class, mechanism, adjacency, contribution_limit_wh, "
                "contribution_policy_id, privacy_policy_id, epsilon_nano, delta_pico, cohort_size "
                "FROM privacy_releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None:
            raise DurablePrivacyPolicyError("cannot attest a release without a reservation")
        return PrivacyReservationContext(
            release_id=release_id,
            receipt_sha256=row[0],
            query_sha256=row[1],
            window_start_utc=row[2],
            window_end_utc=row[3],
            recipient_class=row[4],
            mechanism=row[5],
            adjacency=row[6],
            contribution_limit_wh=row[7],
            contribution_policy_id=row[8],
            privacy_policy_id=row[9],
            epsilon_nano=row[10],
            delta_pico=row[11],
            cohort_size=row[12],
        )

    def _commit_output_storage(
        self,
        *,
        release_id: str,
        output_bytes: bytes,
        committed_at_utc: datetime,
        attestation_bytes: bytes | None = None,
        signature: bytes | None = None,
        signer_key_sha256: str | None = None,
    ) -> CommittedPrivacyOutput:
        """Internal storage primitive; not a release-authorization method.

        It enforces reservation timing and output immutability and can atomically
        retain attestation bytes supplied by the trusted verifier. It does NOT
        authenticate the receipt itself. Production callers must use
        QualifiedOutputVerifier.verify_and_commit / load_committed_output.
        """
        committed_text = self._timestamp(committed_at_utc, "committed_at_utc")
        if not isinstance(release_id, str) or not RELEASE_ID_RE.fullmatch(release_id):
            raise DurablePrivacyPolicyError("release_id must be a random 32-character lowercase hex ID")
        if not isinstance(output_bytes, bytes) or not 1 <= len(output_bytes) <= MAX_DP_OUTPUT_BYTES:
            raise DurablePrivacyPolicyError("output_bytes must be non-empty and at most 1 MiB")
        output_sha256 = hashlib.sha256(output_bytes).hexdigest()
        attestation_fields = (attestation_bytes, signature, signer_key_sha256)
        if any(value is not None for value in attestation_fields) and not all(
            value is not None for value in attestation_fields
        ):
            raise DurablePrivacyPolicyError("signed attestation bytes, signature and signer digest must be supplied together")
        attestation_sha256 = None
        if attestation_bytes is not None:
            if not isinstance(attestation_bytes, bytes) or not 1 <= len(attestation_bytes) <= 16_384:
                raise DurablePrivacyPolicyError("attestation bytes must be non-empty and at most 16 KiB")
            if not isinstance(signature, bytes) or len(signature) != 64:
                raise DurablePrivacyPolicyError("attestation signature must be exactly 64 bytes")
            if not isinstance(signer_key_sha256, str) or not SHA256_RE.fullmatch(signer_key_sha256):
                raise DurablePrivacyPolicyError("signer_key_sha256 must be a lowercase SHA-256 digest")
            attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT receipt_sha256, status, output_sha256, output_bytes, reserved_at_utc, window_end_utc "
                    "FROM privacy_releases WHERE release_id = ?",
                    (release_id,),
                ).fetchone()
                if row is None:
                    raise DurablePrivacyPolicyError("cannot commit output without a reservation")
                receipt_sha256, status, old_digest, old_bytes, reserved_at_text, window_end_text = row
                if status == "output_committed":
                    if old_digest != output_sha256 or old_bytes != output_bytes:
                        raise EventIdentityConflict(
                            "release_id is already bound to different output bytes"
                        )
                    stored_attestation = db.execute(
                        "SELECT attestation_sha256, attestation_bytes, signature, signer_key_sha256 "
                        "FROM privacy_release_attestations WHERE release_id = ?",
                        (release_id,),
                    ).fetchone()
                    if attestation_bytes is None:
                        if stored_attestation is not None:
                            raise DurablePrivacyPolicyError(
                                "attested output retries must use the signed-attestation verifier"
                            )
                    else:
                        expected_attestation = (
                            attestation_sha256, attestation_bytes, signature, signer_key_sha256
                        )
                        if stored_attestation is None or tuple(stored_attestation) != expected_attestation:
                            raise EventIdentityConflict(
                                "release_id is already bound to different or missing attestation evidence"
                            )
                    db.execute("COMMIT")
                    return CommittedPrivacyOutput(
                        release_id, receipt_sha256, old_digest, old_bytes, True
                    )
                if committed_text < reserved_at_text:
                    raise DurablePrivacyPolicyError("output commit time cannot precede its reservation")
                if committed_text < window_end_text:
                    raise DurablePrivacyPolicyError("output cannot be committed before its release window ends")
                db.execute(
                    "UPDATE privacy_releases SET status='output_committed', output_sha256=?, "
                    "output_bytes=?, committed_at_utc=? WHERE release_id=? AND status='reserved'",
                    (output_sha256, output_bytes, committed_text, release_id),
                )
                if attestation_bytes is not None:
                    db.execute(
                        "INSERT INTO privacy_release_attestations "
                        "(release_id, attestation_sha256, attestation_bytes, signature, signer_key_sha256) "
                        "VALUES (?,?,?,?,?)",
                        (release_id, attestation_sha256, attestation_bytes, signature, signer_key_sha256),
                    )
                db.execute("COMMIT")
                return CommittedPrivacyOutput(
                    release_id, receipt_sha256, output_sha256, output_bytes, False
                )
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def _committed_output_storage(self, release_id: str) -> CommittedPrivacyOutput:
        if not isinstance(release_id, str) or not RELEASE_ID_RE.fullmatch(release_id):
            raise DurablePrivacyPolicyError("release_id must be a random 32-character lowercase hex ID")
        with self._connection() as db:
            row = db.execute(
                "SELECT receipt_sha256, status, output_sha256, output_bytes "
                "FROM privacy_releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None or row[1] != "output_committed":
            raise DurablePrivacyPolicyError("release has no committed output")
        if not isinstance(row[3], bytes) or hashlib.sha256(row[3]).hexdigest() != row[2]:
            raise TelemetryQueueError("committed privacy output digest mismatch")
        return CommittedPrivacyOutput(release_id, row[0], row[2], row[3], True)

    def _stored_attestation(
        self, release_id: str
    ) -> tuple[bytes, bytes, str, datetime]:
        """Load retained signature evidence only after checking its stored digest."""
        if not isinstance(release_id, str) or not RELEASE_ID_RE.fullmatch(release_id):
            raise DurablePrivacyPolicyError("release_id must be a random 32-character lowercase hex ID")
        with self._connection() as db:
            row = db.execute(
                "SELECT a.attestation_sha256, a.attestation_bytes, a.signature, "
                "a.signer_key_sha256, r.status, r.committed_at_utc "
                "FROM privacy_release_attestations a "
                "JOIN privacy_releases r ON r.release_id = a.release_id "
                "WHERE a.release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None or row[4] != "output_committed":
            raise DurablePrivacyPolicyError("release has no retained signed attestation")
        if (
            not isinstance(row[1], bytes)
            or hashlib.sha256(row[1]).hexdigest() != row[0]
            or not isinstance(row[2], bytes)
            or len(row[2]) != 64
            or not isinstance(row[3], str)
            or not SHA256_RE.fullmatch(row[3])
            or not isinstance(row[5], str)
        ):
            raise TelemetryQueueError("stored signed-attestation evidence failed integrity checks")
        return row[1], row[2], row[3], self._parse_timestamp(row[5])

    def integrity_check(self) -> str:
        with self._connection() as db:
            result = db.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise TelemetryQueueError(f"privacy ledger integrity check failed: {result}")
            return result
