#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference verifier for RSK's Xenia witness trust-context boundary.

This module consumes cryptographic key fingerprints that a separate Xenia
`VerifiedStateWitnessAdmission` boundary has already verified. It maps those exact
keys through an independently authenticated RSK trust snapshot and requires
separate key, signer-identity, and failure-domain quorums.

It does not verify Xenia signatures, compute Xenia BLAKE3 fingerprints, persist an
anchor, provide trusted time, or grant replication authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import rsk_monotonic_anchor as anchor
import rsk_semantic_schema as semantic

TRUST_CONTEXT_SCHEMA = "symthaea.rsk.xenia-witness-trust-context.v1"
TRUST_CONTEXT_DOMAIN = b"symthaea.rsk.xenia-witness-trust-context.v1\0"
ANCHOR_POLICY_DOMAIN = b"symthaea.rsk.anchor-policy.v1\0"
XENIA_KEY_FINGERPRINT_ALGORITHM = "blake3-256"
WITNESS_LIFECYCLES = {"active", "suspended", "revoked", "retired"}


class WitnessTrustError(ValueError):
    """Raised for malformed or inconsistent Xenia witness trust evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WitnessTrustError(message)


def _require_id(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and semantic.ID_RE.fullmatch(value) is not None,
        f"{field}: invalid canonical identifier",
    )
    return value


def _require_digest(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and semantic.HEX64.fullmatch(value) is not None,
        f"{field}: must be lowercase 32-byte hex digest",
    )
    return value


def canonical_anchor_policy(policy: anchor.AnchorPolicy) -> dict[str, Any]:
    """Return the canonical object whose identity is bound into witness trust."""

    _require(isinstance(policy, anchor.AnchorPolicy), "invalid anchor policy type")
    try:
        policy.validate()
    except anchor.AnchorError as exc:
        raise WitnessTrustError(str(exc)) from exc
    return {
        "namespace": policy.namespace,
        "epoch": policy.epoch,
        "allowed_provider_profiles": list(policy.allowed_provider_profiles),
        "allowed_provider_identities": list(policy.allowed_provider_identities),
        "trust_snapshot_digest": policy.trust_snapshot_digest,
    }


def anchor_policy_digest(policy: anchor.AnchorPolicy) -> str:
    """Domain-separated identity of the exact RSK monotonic-anchor policy."""

    return hashlib.sha256(
        ANCHOR_POLICY_DOMAIN + semantic.canonical_bytes(canonical_anchor_policy(policy))
    ).hexdigest()


@dataclass(frozen=True)
class XeniaWitnessPrincipal:
    """One trust-snapshot mapping from cryptographic key to governed identity."""

    signer_id: str
    key_id: str
    role: str
    failure_domain: str
    signature_profile: str
    public_key_fingerprint: str
    lifecycle: str
    valid_from: int
    valid_until: int

    def canonical_object(self) -> dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "key_id": self.key_id,
            "role": self.role,
            "failure_domain": self.failure_domain,
            "signature_profile": self.signature_profile,
            "public_key_fingerprint": self.public_key_fingerprint,
            "lifecycle": self.lifecycle,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
        }


@dataclass(frozen=True)
class VerifiedXeniaWitnessKeyReference:
    """Reference stand-in for one key binding returned by Xenia after verification.

    Production code must construct an opaque equivalent from Xenia's verified
    admission type rather than from unverified bundle metadata.
    """

    signature_profile: str
    public_key_fingerprint: str

    def validate(self) -> None:
        _require_id(self.signature_profile, "verified key signature_profile")
        _require_digest(self.public_key_fingerprint, "verified key fingerprint")


@dataclass(frozen=True)
class XeniaWitnessTrustContextPolicy:
    """Exact policy/trust context that a Xenia state commitment must bind."""

    anchor_policy: anchor.AnchorPolicy
    xenia_profile: str
    minimum_key_quorum: int
    minimum_signer_identities: int
    minimum_failure_domains: int
    allowed_signature_profiles: tuple[str, ...]
    signer_lifecycle_policy_digest: str
    failure_domain_policy_digest: str
    trust_snapshot_digest: str
    principals: tuple[XeniaWitnessPrincipal, ...]

    def validate(self) -> None:
        canonical_anchor_policy(self.anchor_policy)
        _require_id(self.xenia_profile, "xenia_profile")
        _require_digest(self.signer_lifecycle_policy_digest, "signer_lifecycle_policy_digest")
        _require_digest(self.failure_domain_policy_digest, "failure_domain_policy_digest")
        _require_digest(self.trust_snapshot_digest, "trust_snapshot_digest")
        _require(
            self.trust_snapshot_digest == self.anchor_policy.trust_snapshot_digest,
            "witness trust snapshot differs from anchor policy trust snapshot",
        )
        for field in ("minimum_key_quorum", "minimum_signer_identities", "minimum_failure_domains"):
            value = getattr(self, field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{field} must be positive integer",
            )
        _require(
            self.minimum_failure_domains <= self.minimum_signer_identities,
            "failure-domain quorum cannot exceed signer-identity quorum",
        )
        _require(
            self.minimum_signer_identities <= self.minimum_key_quorum,
            "signer-identity quorum cannot exceed key quorum",
        )
        _require(
            isinstance(self.allowed_signature_profiles, tuple) and self.allowed_signature_profiles,
            "allowed_signature_profiles must be nonempty tuple",
        )
        _require(
            self.allowed_signature_profiles == tuple(sorted(self.allowed_signature_profiles)),
            "allowed_signature_profiles must be sorted",
        )
        _require(
            len(set(self.allowed_signature_profiles)) == len(self.allowed_signature_profiles),
            "allowed_signature_profiles contains duplicates",
        )
        for profile in self.allowed_signature_profiles:
            _require_id(profile, "allowed_signature_profiles")

        _require(isinstance(self.principals, tuple) and self.principals, "principals must be nonempty")
        ordering = tuple(
            (principal.signer_id, principal.key_id, principal.public_key_fingerprint)
            for principal in self.principals
        )
        _require(ordering == tuple(sorted(ordering)), "principals must be sorted")

        key_owners: dict[str, str] = {}
        fingerprint_keys: dict[str, str] = {}
        signer_metadata: dict[str, tuple[str, str]] = {}
        active_keys: set[str] = set()
        active_signers: set[str] = set()
        active_domains: set[str] = set()

        for principal in self.principals:
            _require(isinstance(principal, XeniaWitnessPrincipal), "invalid principal type")
            _require_id(principal.signer_id, "principal.signer_id")
            _require_id(principal.key_id, "principal.key_id")
            _require_id(principal.role, "principal.role")
            _require_id(principal.failure_domain, "principal.failure_domain")
            _require_id(principal.signature_profile, "principal.signature_profile")
            _require_digest(principal.public_key_fingerprint, "principal.public_key_fingerprint")
            _require(
                principal.signature_profile in self.allowed_signature_profiles,
                "principal signature profile is not policy-allowed",
            )
            _require(principal.lifecycle in WITNESS_LIFECYCLES, "principal lifecycle is unknown")
            _require(
                isinstance(principal.valid_from, int)
                and not isinstance(principal.valid_from, bool)
                and isinstance(principal.valid_until, int)
                and not isinstance(principal.valid_until, bool)
                and principal.valid_from >= 0
                and principal.valid_until >= principal.valid_from,
                "principal validity interval invalid",
            )
            previous_owner = key_owners.get(principal.key_id)
            _require(
                previous_owner is None or previous_owner == principal.signer_id,
                "one key identity maps to multiple signer identities",
            )
            key_owners[principal.key_id] = principal.signer_id
            previous_key = fingerprint_keys.get(principal.public_key_fingerprint)
            _require(
                previous_key is None or previous_key == principal.key_id,
                "one raw-key fingerprint maps to multiple key identities",
            )
            fingerprint_keys[principal.public_key_fingerprint] = principal.key_id
            metadata = (principal.role, principal.failure_domain)
            previous_metadata = signer_metadata.get(principal.signer_id)
            _require(
                previous_metadata is None or previous_metadata == metadata,
                "one signer identity has conflicting role/failure-domain metadata",
            )
            signer_metadata[principal.signer_id] = metadata
            if principal.lifecycle == "active":
                active_keys.add(principal.key_id)
                active_signers.add(principal.signer_id)
                active_domains.add(principal.failure_domain)

        _require(len(active_keys) >= self.minimum_key_quorum, "trust context cannot structurally satisfy key quorum")
        _require(
            len(active_signers) >= self.minimum_signer_identities,
            "trust context cannot structurally satisfy signer quorum",
        )
        _require(
            len(active_domains) >= self.minimum_failure_domains,
            "trust context cannot structurally satisfy failure-domain quorum",
        )

    def canonical_object(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": TRUST_CONTEXT_SCHEMA,
            "anchor_policy_digest": anchor_policy_digest(self.anchor_policy),
            "xenia_profile": self.xenia_profile,
            "key_fingerprint_algorithm": XENIA_KEY_FINGERPRINT_ALGORITHM,
            "minimum_key_quorum": self.minimum_key_quorum,
            "minimum_signer_identities": self.minimum_signer_identities,
            "minimum_failure_domains": self.minimum_failure_domains,
            "allowed_signature_profiles": list(self.allowed_signature_profiles),
            "signer_lifecycle_policy_digest": self.signer_lifecycle_policy_digest,
            "failure_domain_policy_digest": self.failure_domain_policy_digest,
            "trust_snapshot_digest": self.trust_snapshot_digest,
            "principals": [principal.canonical_object() for principal in self.principals],
        }

    def digest(self) -> str:
        return hashlib.sha256(
            TRUST_CONTEXT_DOMAIN + semantic.canonical_bytes(self.canonical_object())
        ).hexdigest()


class _VerifiedWitnessQualityMarker:
    pass


_VERIFIED_WITNESS_QUALITY_MARKER = _VerifiedWitnessQualityMarker()


class VerifiedXeniaWitnessQuorumReference:
    """Reference-only type state after all three quorum dimensions succeed."""

    __slots__ = ("_trust_context_digest", "_key_ids", "_signer_ids", "_failure_domains")

    def __init__(
        self,
        *,
        trust_context_digest: str,
        key_ids: tuple[str, ...],
        signer_ids: tuple[str, ...],
        failure_domains: tuple[str, ...],
        _marker: _VerifiedWitnessQualityMarker,
    ) -> None:
        if _marker is not _VERIFIED_WITNESS_QUALITY_MARKER:
            raise WitnessTrustError("verified witness quorum cannot be constructed directly")
        self._trust_context_digest = trust_context_digest
        self._key_ids = key_ids
        self._signer_ids = signer_ids
        self._failure_domains = failure_domains

    @property
    def trust_context_digest(self) -> str:
        return self._trust_context_digest

    @property
    def key_ids(self) -> tuple[str, ...]:
        return self._key_ids

    @property
    def signer_ids(self) -> tuple[str, ...]:
        return self._signer_ids

    @property
    def failure_domains(self) -> tuple[str, ...]:
        return self._failure_domains


@dataclass(frozen=True)
class WitnessQualityDecision:
    status: str
    reason: str
    verified: VerifiedXeniaWitnessQuorumReference | None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.verified is not None


def evaluate_xenia_witness_quorum_quality(
    verified_keys: Iterable[VerifiedXeniaWitnessKeyReference],
    policy: XeniaWitnessTrustContextPolicy,
    trusted_interval: anchor.TrustedAnchorInterval,
) -> WitnessQualityDecision:
    """Map verified Xenia keys through RSK trust policy and require three quorums."""

    try:
        _require(isinstance(policy, XeniaWitnessTrustContextPolicy), "invalid trust-context policy")
        policy.validate()
        try:
            trusted_interval.validate()
        except anchor.AnchorError as exc:
            raise WitnessTrustError(str(exc)) from exc
        keys = tuple(verified_keys)
        _require(keys, "no verified Xenia witness keys supplied")
        by_fingerprint = {principal.public_key_fingerprint: principal for principal in policy.principals}
        observed_fingerprints: set[str] = set()
        key_ids: set[str] = set()
        signer_ids: set[str] = set()
        failure_domains: set[str] = set()

        for key in keys:
            _require(isinstance(key, VerifiedXeniaWitnessKeyReference), "invalid verified Xenia witness-key type")
            key.validate()
            _require(
                key.public_key_fingerprint not in observed_fingerprints,
                "duplicate verified Xenia witness key",
            )
            observed_fingerprints.add(key.public_key_fingerprint)
            principal = by_fingerprint.get(key.public_key_fingerprint)
            _require(principal is not None, "verified Xenia witness key is absent from trust snapshot")
            _require(
                key.signature_profile == principal.signature_profile,
                "verified Xenia witness signature profile does not match trusted key mapping",
            )
            _require(principal.lifecycle == "active", "verified Xenia witness key is not active")
            _require(
                principal.valid_from <= trusted_interval.start
                and trusted_interval.end <= principal.valid_until,
                "verified Xenia witness key is not valid for full trusted interval",
            )
            key_ids.add(principal.key_id)
            signer_ids.add(principal.signer_id)
            failure_domains.add(principal.failure_domain)

        _require(len(key_ids) >= policy.minimum_key_quorum, "insufficient distinct verified witness keys")
        _require(
            len(signer_ids) >= policy.minimum_signer_identities,
            "insufficient distinct verified witness signer identities",
        )
        _require(
            len(failure_domains) >= policy.minimum_failure_domains,
            "insufficient independent verified witness failure domains",
        )
        verified = VerifiedXeniaWitnessQuorumReference(
            trust_context_digest=policy.digest(),
            key_ids=tuple(sorted(key_ids)),
            signer_ids=tuple(sorted(signer_ids)),
            failure_domains=tuple(sorted(failure_domains)),
            _marker=_VERIFIED_WITNESS_QUALITY_MARKER,
        )
        return WitnessQualityDecision(
            status="accepted",
            reason="verified Xenia keys satisfy key, signer, and failure-domain quorums",
            verified=verified,
        )
    except WitnessTrustError as exc:
        return WitnessQualityDecision(status="frozen", reason=str(exc), verified=None)
