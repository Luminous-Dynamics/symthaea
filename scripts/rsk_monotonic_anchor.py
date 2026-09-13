#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference composition verifier for externally anchored RSK anti-rollback state.

This module deliberately does NOT implement TPM/HSM/remote-witness operations and
cannot advance or reset an external anchor. It consumes evidence that an external
cryptographic/provider boundary has already authenticated, then checks exact
namespace/epoch/counter/state-digest/freshness agreement with local RSK state.

Reference tooling only. Objects here are not production authority capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import rsk_schema_registry as registry
import rsk_semantic_schema as semantic

ANCHOR_STATE_SCHEMA = "symthaea.rsk.monotonic-anchor-state.v1"
ANCHOR_STATE_DOMAIN = b"symthaea.rsk.monotonic-anchor-state.v1\0"


class AnchorError(ValueError):
    """Raised for malformed or inconsistent monotonic-anchor evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnchorError(message)


def _require_id(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and semantic.ID_RE.fullmatch(value) is not None,
        f"{field}: invalid canonical identifier",
    )
    return value


def _require_digest(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and semantic.HEX64.fullmatch(value) is not None,
        f"{field}: must be lowercase SHA-256 hex",
    )
    return value


def canonical_registry_tracker(state: registry.AntiRollbackState) -> dict[str, Any]:
    """Return the exact canonical state object protected by the external anchor."""

    _require(isinstance(state, registry.AntiRollbackState), "invalid registry tracker state")
    known_entries = [
        {
            "schema_kind": entry.schema_kind,
            "family": entry.family,
            "version": entry.version,
            "schema_id": entry.schema_id,
            "lifecycle_state": entry.lifecycle_state,
        }
        for entry in state.known_entries
    ]
    known_entries.sort(key=lambda entry: (entry["schema_kind"], entry["family"], entry["version"]))

    return {
        "schema": ANCHOR_STATE_SCHEMA,
        "tracker_kind": "schema-registry",
        "registry_id": state.registry_id,
        "highest_sequence": state.highest_sequence,
        "accepted_digest": state.accepted_digest,
        "latest_issued_at": state.latest_issued_at,
        "policy_digest": state.policy_digest,
        "forked": state.forked,
        "known_entries": known_entries,
    }


def registry_tracker_digest(state: registry.AntiRollbackState) -> str:
    """Domain-separated SHA-256 over the complete canonical registry tracker state."""

    payload = canonical_registry_tracker(state)
    return hashlib.sha256(ANCHOR_STATE_DOMAIN + semantic.canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class TrustedAnchorInterval:
    start: int
    end: int

    def validate(self) -> None:
        _require(
            isinstance(self.start, int)
            and not isinstance(self.start, bool)
            and isinstance(self.end, int)
            and not isinstance(self.end, bool)
            and self.start >= 0
            and self.end >= self.start,
            "trusted anchor interval invalid",
        )


@dataclass(frozen=True)
class AuthenticatedMonotonicAnchorEvidence:
    """Metadata produced by an external provider/trust verifier.

    There is intentionally no `verified` boolean. Production must construct an
    opaque equivalent only after authenticating the provider/attestation itself.
    """

    namespace: str
    epoch: int
    counter: int
    anchored_state_digest: str
    provider_profile: str
    provider_identity: str
    trust_snapshot_digest: str
    issued_at: int
    expires_at: int


@dataclass(frozen=True)
class AnchorPolicy:
    namespace: str
    epoch: int
    allowed_provider_profiles: tuple[str, ...]
    allowed_provider_identities: tuple[str, ...]
    trust_snapshot_digest: str

    def validate(self) -> None:
        _require_id(self.namespace, "anchor policy namespace")
        _require(
            isinstance(self.epoch, int) and not isinstance(self.epoch, bool) and self.epoch >= 1,
            "anchor policy epoch must be positive integer",
        )
        _require_digest(self.trust_snapshot_digest, "anchor policy trust_snapshot_digest")
        for field, values in (
            ("allowed_provider_profiles", self.allowed_provider_profiles),
            ("allowed_provider_identities", self.allowed_provider_identities),
        ):
            _require(isinstance(values, tuple) and values, f"{field} must be nonempty tuple")
            _require(tuple(sorted(values)) == values, f"{field} must be sorted")
            _require(len(set(values)) == len(values), f"{field} contains duplicates")
            for value in values:
                _require_id(value, field)


class _VerifiedMarker:
    pass


_VERIFIED_MARKER = _VerifiedMarker()


class VerifiedMonotonicAnchorObservationReference:
    """Reference-only process capability after exact anchor/local-state agreement."""

    __slots__ = (
        "_namespace",
        "_epoch",
        "_counter",
        "_state_digest",
        "_provider_profile",
        "_provider_identity",
        "_trust_snapshot_digest",
    )

    def __init__(
        self,
        evidence: AuthenticatedMonotonicAnchorEvidence,
        *,
        _marker: _VerifiedMarker,
    ) -> None:
        if _marker is not _VERIFIED_MARKER:
            raise AnchorError("verified anchor observation cannot be constructed directly")
        self._namespace = evidence.namespace
        self._epoch = evidence.epoch
        self._counter = evidence.counter
        self._state_digest = evidence.anchored_state_digest
        self._provider_profile = evidence.provider_profile
        self._provider_identity = evidence.provider_identity
        self._trust_snapshot_digest = evidence.trust_snapshot_digest

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def state_digest(self) -> str:
        return self._state_digest


@dataclass(frozen=True)
class AnchorDecision:
    status: str
    reason: str
    verified: VerifiedMonotonicAnchorObservationReference | None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.verified is not None


def _validate_evidence_shape(
    evidence: AuthenticatedMonotonicAnchorEvidence,
    policy: AnchorPolicy,
    trusted_interval: TrustedAnchorInterval,
) -> None:
    _require(isinstance(evidence, AuthenticatedMonotonicAnchorEvidence), "invalid anchor evidence type")
    _require_id(evidence.namespace, "anchor namespace")
    _require_id(evidence.provider_profile, "anchor provider_profile")
    _require_id(evidence.provider_identity, "anchor provider_identity")
    _require_digest(evidence.anchored_state_digest, "anchored_state_digest")
    _require_digest(evidence.trust_snapshot_digest, "anchor trust_snapshot_digest")
    for field in ("epoch", "counter", "issued_at", "expires_at"):
        value = getattr(evidence, field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"anchor {field} must be nonnegative integer",
        )
    _require(evidence.epoch >= 1, "anchor epoch must be positive")
    _require(evidence.expires_at > evidence.issued_at, "anchor validity window invalid")
    _require(evidence.namespace == policy.namespace, "anchor namespace mismatch")
    _require(evidence.epoch == policy.epoch, "anchor epoch mismatch")
    _require(
        evidence.provider_profile in policy.allowed_provider_profiles,
        "anchor provider profile not policy-allowed",
    )
    _require(
        evidence.provider_identity in policy.allowed_provider_identities,
        "anchor provider identity not policy-allowed",
    )
    _require(
        evidence.trust_snapshot_digest == policy.trust_snapshot_digest,
        "anchor evidence binds wrong trust snapshot",
    )
    _require(
        evidence.issued_at <= trusted_interval.start
        and trusted_interval.end <= evidence.expires_at,
        "anchor observation not fresh for full trusted interval",
    )


def evaluate_registry_anchor(
    state: registry.AntiRollbackState,
    evidence: AuthenticatedMonotonicAnchorEvidence | None,
    policy: AnchorPolicy,
    trusted_interval: TrustedAnchorInterval,
) -> AnchorDecision:
    """Require exact external-anchor agreement with one local registry tracker.

    Any uncertainty or disagreement freezes new positive authority. This helper
    never advances/reset the anchor and never guesses which side is correct.
    """

    try:
        policy.validate()
        trusted_interval.validate()
        _require(isinstance(state, registry.AntiRollbackState), "invalid registry tracker state")
        _require(not state.forked, "local registry tracker is already forked/non-operational")
        _require(evidence is not None, "monotonic anchor observation unavailable")
        _validate_evidence_shape(evidence, policy, trusted_interval)

        expected_counter = state.highest_sequence
        expected_digest = registry_tracker_digest(state)

        if evidence.counter < expected_counter:
            raise AnchorError("anchor counter is behind local tracker")
        if evidence.counter > expected_counter:
            raise AnchorError("local tracker is behind monotonic anchor")
        if evidence.anchored_state_digest != expected_digest:
            raise AnchorError("same anchor counter binds different local state digest")

        verified = VerifiedMonotonicAnchorObservationReference(
            evidence,
            _marker=_VERIFIED_MARKER,
        )
        return AnchorDecision(
            status="accepted",
            reason="external monotonic anchor exactly matches local registry tracker",
            verified=verified,
        )
    except AnchorError as exc:
        return AnchorDecision(status="frozen", reason=str(exc), verified=None)


def reference_anchor_evidence(
    state: registry.AntiRollbackState,
    policy: AnchorPolicy,
    *,
    provider_profile: str,
    provider_identity: str,
    issued_at: int,
    expires_at: int,
) -> AuthenticatedMonotonicAnchorEvidence:
    """Create deterministic *test-fixture* evidence for reference vectors only.

    This function is intentionally named `reference_*`; it does not authenticate
    or write any hardware/provider anchor and MUST NOT be used as production proof.
    """

    return AuthenticatedMonotonicAnchorEvidence(
        namespace=policy.namespace,
        epoch=policy.epoch,
        counter=state.highest_sequence,
        anchored_state_digest=registry_tracker_digest(state),
        provider_profile=provider_profile,
        provider_identity=provider_identity,
        trust_snapshot_digest=policy.trust_snapshot_digest,
        issued_at=issued_at,
        expires_at=expires_at,
    )
