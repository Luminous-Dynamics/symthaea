#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference composition boundary for Xenia witness trust-context admission.

This layer proves only that an already-verified Xenia witness-key set satisfies
RSK trust policy AND that the exact Xenia state commitment binds the independently
reconstructed RSK TrustContextDigest. It does not verify signatures, state
digests, monotonic continuity, trusted time, or replication authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import rsk_monotonic_anchor as anchor
import rsk_semantic_schema as semantic
import rsk_xenia_witness_trust as trust


class XeniaWitnessAdmissionError(ValueError):
    """Raised for inconsistent trust-context admission evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XeniaWitnessAdmissionError(message)


def _require_digest(value: object, field: str) -> str:
    _require(
        isinstance(value, str) and semantic.HEX64.fullmatch(value) is not None,
        f"{field}: must be lowercase 32-byte hex digest",
    )
    return value


class _VerifiedContextMarker:
    pass


_VERIFIED_CONTEXT_MARKER = _VerifiedContextMarker()


class VerifiedXeniaWitnessContextReference:
    """Reference-only type state after quorum quality and digest equality pass."""

    __slots__ = (
        "_trust_context_digest",
        "_key_ids",
        "_signer_ids",
        "_failure_domains",
    )

    def __init__(
        self,
        *,
        trust_context_digest: str,
        key_ids: tuple[str, ...],
        signer_ids: tuple[str, ...],
        failure_domains: tuple[str, ...],
        _marker: _VerifiedContextMarker,
    ) -> None:
        if _marker is not _VERIFIED_CONTEXT_MARKER:
            raise XeniaWitnessAdmissionError(
                "verified Xenia witness context cannot be constructed directly"
            )
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
class XeniaWitnessContextDecision:
    status: str
    reason: str
    verified: VerifiedXeniaWitnessContextReference | None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.verified is not None


def evaluate_xenia_witness_context(
    *,
    commitment_trust_context_digest: str,
    verified_keys: Iterable[trust.VerifiedXeniaWitnessKeyReference],
    policy: trust.XeniaWitnessTrustContextPolicy,
    trusted_interval: anchor.TrustedAnchorInterval,
) -> XeniaWitnessContextDecision:
    """Require exact trust-context equality after independent quorum verification.

    The digest supplied here is the value cryptographically bound inside the
    Xenia state commitment by the lower-level witness verifier. This function
    never accepts an unverified bundle field as authority on its own.
    """

    try:
        bound_digest = _require_digest(
            commitment_trust_context_digest,
            "commitment_trust_context_digest",
        )
        quality = trust.evaluate_xenia_witness_quorum_quality(
            verified_keys,
            policy,
            trusted_interval,
        )
        _require(quality.accepted, quality.reason)
        expected_digest = quality.verified.trust_context_digest
        _require(
            bound_digest == expected_digest,
            "Xenia commitment binds a different witness trust context",
        )
        verified = VerifiedXeniaWitnessContextReference(
            trust_context_digest=expected_digest,
            key_ids=quality.verified.key_ids,
            signer_ids=quality.verified.signer_ids,
            failure_domains=quality.verified.failure_domains,
            _marker=_VERIFIED_CONTEXT_MARKER,
        )
        return XeniaWitnessContextDecision(
            status="accepted",
            reason="Xenia commitment binds the independently reconstructed RSK witness trust context",
            verified=verified,
        )
    except (XeniaWitnessAdmissionError, trust.WitnessTrustError) as exc:
        return XeniaWitnessContextDecision(status="frozen", reason=str(exc), verified=None)
