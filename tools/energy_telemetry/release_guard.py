"""In-memory research guard for aggregate telemetry privacy releases.

Participant tokens and release membership are local to the trusted aggregation
boundary and must never be serialized to the satellite envelope. This prototype
rejects overlapping windows for any participant and uses basic sequential
composition of epsilon/delta. It is NOT a durable production accountant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Iterable

BATCH_RE = re.compile(r"^[a-f0-9]{32}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
MINIMUM_COHORT_SIZE = 20
MAX_RELEASE_EPSILON = 1.0
MAX_RELEASE_DELTA = 1e-6


def _is_finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


class ReleasePolicyError(ValueError):
    """A proposed aggregate release violates privacy policy."""


@dataclass(frozen=True)
class ReleaseRecord:
    batch_id: str
    payload_sha256: str
    participant_tokens: frozenset[str] = field(repr=False)
    start_utc: datetime
    end_utc: datetime
    recipient_class: str
    epsilon: float | None
    delta: float | None


class ReleasePolicyGuard:
    """Check cohort eligibility, interval differencing and basic DP composition.

    The member tokens stay in-process. Callers must supply opaque, local-only
    tokens rather than names, account IDs or serialized meter identifiers.
    """

    def __init__(
        self,
        *,
        max_epsilon_per_participant: float = 2.0,
        max_delta_per_participant: float = 2e-6,
        minimum_cohort_size: int = MINIMUM_COHORT_SIZE,
    ) -> None:
        if (
            not _is_finite_number(max_epsilon_per_participant)
            or max_epsilon_per_participant <= 0
        ):
            raise ValueError("max_epsilon_per_participant must be finite and positive")
        if (
            not _is_finite_number(max_delta_per_participant)
            or max_delta_per_participant < 0
        ):
            raise ValueError("max_delta_per_participant must be finite and non-negative")
        if type(minimum_cohort_size) is not int or minimum_cohort_size < MINIMUM_COHORT_SIZE:
            raise ValueError(f"minimum_cohort_size must be at least {MINIMUM_COHORT_SIZE}")
        self.max_epsilon = float(max_epsilon_per_participant)
        self.max_delta = float(max_delta_per_participant)
        self.minimum_cohort_size = minimum_cohort_size
        self._records: list[ReleaseRecord] = []
        self._by_batch: dict[str, ReleaseRecord] = {}
        self._epsilon_spent: dict[str, float] = {}
        self._delta_spent: dict[str, float] = {}

    @property
    def release_count(self) -> int:
        return len(self._records)

    @staticmethod
    def _validate_window(start_utc: datetime, end_utc: datetime) -> None:
        for name, value in (("start_utc", start_utc), ("end_utc", end_utc)):
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ReleasePolicyError(f"{name} must be timezone-aware")
            if value.utcoffset() != timezone.utc.utcoffset(value):
                raise ReleasePolicyError(f"{name} must be UTC")
        if end_utc <= start_utc:
            raise ReleasePolicyError("end_utc must be later than start_utc")

    @staticmethod
    def _validate_dp(epsilon: float | None, delta: float | None) -> None:
        if (epsilon is None) != (delta is None):
            raise ReleasePolicyError("epsilon and delta must either both be supplied or both be omitted")
        if epsilon is None and delta is None:
            return
        if (
            not _is_finite_number(epsilon)
            or not 0 < epsilon <= MAX_RELEASE_EPSILON
        ):
            raise ReleasePolicyError(f"epsilon must be finite and in (0,{MAX_RELEASE_EPSILON}]")
        if (
            not _is_finite_number(delta)
            or not 0 <= delta <= MAX_RELEASE_DELTA
        ):
            raise ReleasePolicyError(f"delta must be finite and in [0,{MAX_RELEASE_DELTA}]")

    def submit(
        self,
        *,
        batch_id: str,
        payload_sha256: str,
        participant_tokens: Iterable[str],
        start_utc: datetime,
        end_utc: datetime,
        recipient_class: str,
        epsilon: float | None = None,
        delta: float | None = None,
        minimum_cohort_size: int | None = None,
    ) -> str:
        """Reserve one local release. Returns 'accepted' or 'idempotent_retransmission'."""
        if not isinstance(batch_id, str) or not BATCH_RE.fullmatch(batch_id):
            raise ReleasePolicyError("batch_id must be a random 32-character lowercase hex ID")
        if not isinstance(payload_sha256, str) or not SHA256_RE.fullmatch(payload_sha256):
            raise ReleasePolicyError("payload_sha256 must be a lowercase SHA-256 digest")
        if recipient_class not in ("grid_operations", "approved_research", "public_reporting"):
            raise ReleasePolicyError("unsupported recipient class")
        self._validate_window(start_utc, end_utc)
        self._validate_dp(epsilon, delta)

        try:
            participants = frozenset(participant_tokens)
        except (TypeError, ValueError) as exc:
            raise ReleasePolicyError("participant_tokens must be iterable of opaque local tokens") from exc
        if not participants or any(not isinstance(token, str) or not token for token in participants):
            raise ReleasePolicyError("participant_tokens must be non-empty opaque local strings")
        floor = self.minimum_cohort_size if minimum_cohort_size is None else minimum_cohort_size
        if type(floor) is not int or floor < self.minimum_cohort_size:
            raise ReleasePolicyError(
                f"minimum_cohort_size cannot be below policy floor {self.minimum_cohort_size}"
            )
        if len(participants) < floor:
            raise ReleasePolicyError("cohort is below the required minimum size")

        if recipient_class in ("approved_research", "public_reporting") and epsilon is None:
            raise ReleasePolicyError("research/public release requires differential privacy")
        if recipient_class in ("approved_research", "public_reporting") and delta is None:
            raise ReleasePolicyError("research/public release requires differential privacy parameters")

        existing = self._by_batch.get(batch_id)
        if existing is not None:
            same = (
                existing.payload_sha256 == payload_sha256
                and existing.participant_tokens == participants
                and existing.start_utc == start_utc
                and existing.end_utc == end_utc
                and existing.recipient_class == recipient_class
                and existing.epsilon == epsilon
                and existing.delta == delta
            )
            if same:
                return "idempotent_retransmission"
            raise ReleasePolicyError("batch_id was reused with different release contents")

        # Fail closed on overlapping windows for any shared participant. This
        # intentionally suppresses rolling-window analytics until a more formal
        # authorised post-processing/query policy has been designed.
        for prior in self._records:
            if participants.isdisjoint(prior.participant_tokens):
                continue
            if start_utc < prior.end_utc and prior.start_utc < end_utc:
                raise ReleasePolicyError(
                    "overlapping release windows for shared participants are blocked to prevent differencing"
                )

        if epsilon is not None and delta is not None:
            for token in participants:
                spent_epsilon = self._epsilon_spent.get(token, 0.0) + float(epsilon)
                spent_delta = self._delta_spent.get(token, 0.0) + float(delta)
                if spent_epsilon > self.max_epsilon:
                    raise ReleasePolicyError("cumulative epsilon budget exceeded for a participant")
                if spent_delta > self.max_delta:
                    raise ReleasePolicyError("cumulative delta budget exceeded for a participant")

        record = ReleaseRecord(
            batch_id=batch_id,
            payload_sha256=payload_sha256,
            participant_tokens=participants,
            start_utc=start_utc,
            end_utc=end_utc,
            recipient_class=recipient_class,
            epsilon=None if epsilon is None else float(epsilon),
            delta=None if delta is None else float(delta),
        )
        self._records.append(record)
        self._by_batch[batch_id] = record
        if epsilon is not None and delta is not None:
            for token in participants:
                self._epsilon_spent[token] = self._epsilon_spent.get(token, 0.0) + float(epsilon)
                self._delta_spent[token] = self._delta_spent.get(token, 0.0) + float(delta)
        return "accepted"
