#!/usr/bin/env python3
"""Strict provider-observed UTC instant parsing for trusted-main evidence.

This module deliberately proves only that a GitHub-provided timestamp is a
calendar-valid instant in a canonical UTC subset and gives that instant a
stable integer ordering coordinate. It does not authenticate GitHub, establish
external chronology, or prove freshness/currentness.

Accepted syntax:

    YYYY-MM-DDTHH:MM:SS[.1-9 digits]Z

Offsets, leap-second ``:60``, lowercase ``z``, whitespace, and more than nine
fractional digits are rejected. Ordering is represented as signed nanoseconds
from the Unix epoch using integer arithmetic only; no floating-point timestamp
conversion participates in evidence semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Final

SCHEMA: Final = "symthaea.github-provider-utc-instant.v1"
RFC3339_UTC_RE: Final = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)
EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
NANOS_PER_SECOND: Final = 1_000_000_000


class ProviderTimeError(ValueError):
    """Fail-closed GitHub provider-time validation error."""


@dataclass(frozen=True, order=True)
class ProviderInstant:
    """Validated provider instant with an integer-only ordering coordinate."""

    unix_nanos: int
    canonical_text: str

    @property
    def schema(self) -> str:
        return SCHEMA


def parse_github_utc_instant(value: object, *, where: str = "provider_time") -> ProviderInstant:
    """Parse one canonical, calendar-valid GitHub UTC timestamp.

    The returned coordinate is suitable for deterministic provider-order
    comparisons. It is not an externally anchored chronology witness.
    """

    if not isinstance(value, str) or not value:
        raise ProviderTimeError(f"{where}: non-empty string required")
    if value != value.strip():
        raise ProviderTimeError(f"{where}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ProviderTimeError(f"{where}: control characters are forbidden")

    match = RFC3339_UTC_RE.fullmatch(value)
    if match is None:
        raise ProviderTimeError(
            f"{where}: canonical UTC RFC3339 subset required "
            "(YYYY-MM-DDTHH:MM:SS[.1-9 digits]Z)"
        )

    parts = {name: int(match.group(name)) for name in (
        "year", "month", "day", "hour", "minute", "second"
    )}
    # Python's datetime deliberately rejects invalid calendar dates, hour 24,
    # minute 60, and leap-second :60. That gives us calendar validity rather
    # than regex-shaped text validity.
    try:
        whole = datetime(
            parts["year"],
            parts["month"],
            parts["day"],
            parts["hour"],
            parts["minute"],
            parts["second"],
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise ProviderTimeError(f"{where}: invalid UTC calendar instant: {exc}") from exc

    fraction = match.group("fraction") or ""
    fractional_nanos = int(fraction.ljust(9, "0")) if fraction else 0

    delta = whole - EPOCH
    whole_seconds = delta.days * 86_400 + delta.seconds
    unix_nanos = whole_seconds * NANOS_PER_SECOND + fractional_nanos
    return ProviderInstant(unix_nanos=unix_nanos, canonical_text=value)


def compare_provider_instants(left: object, right: object) -> int:
    """Return -1/0/1 under provider-observed ordering only."""

    lhs = parse_github_utc_instant(left, where="left")
    rhs = parse_github_utc_instant(right, where="right")
    return (lhs.unix_nanos > rhs.unix_nanos) - (lhs.unix_nanos < rhs.unix_nanos)
