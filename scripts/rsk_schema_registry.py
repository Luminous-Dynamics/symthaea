#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference semantic-state verifier for the RSK schema registry.

This module deliberately does NOT verify digital signatures or trust roots.
It consumes signer evidence that an external cryptographic/trust boundary has
already authenticated and evaluates the registry semantics that RSK itself must
fail closed on: exact schema identity, quorum/independence metadata, freshness,
rollback/fork handling, lifecycle monotonicity, and exact resolution.

Reference tooling only. Python objects here are not production opaque authority
capabilities and cannot grant replication authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import rsk_semantic_schema as semantic

REGISTRY_SCHEMA_VERSION = "symthaea.rsk.schema-registry-snapshot.v1"
REGISTRY_SNAPSHOT_DOMAIN = b"symthaea.rsk.schema-registry-snapshot.v1\0"
ZERO_DIGEST = "0" * 64
LIFECYCLE_RANK = {
    "active": 0,
    "superseded": 1,
    "revoked": 2,
    "tombstoned": 3,
}
SCHEMA_KINDS = {"capability", "resource"}


class RegistryError(ValueError):
    """Raised for malformed reference registry inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


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


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """Domain-separated SHA-256 over exact canonical snapshot bytes."""

    return hashlib.sha256(
        REGISTRY_SNAPSHOT_DOMAIN + semantic.canonical_bytes(snapshot)
    ).hexdigest()


def policy_digest(policy: dict[str, Any]) -> str:
    """Canonical identity of one externally governed registry policy."""

    validate_policy(policy)
    return semantic.digest(policy)


def validate_policy(policy: dict[str, Any]) -> None:
    _require(isinstance(policy, dict), "registry policy must be object")
    expected = {
        "registry_id",
        "min_signer_identities",
        "min_failure_domains",
        "allowed_roles",
        "allowed_signature_profiles",
        "allowed_canonical_encoding_versions",
        "max_entries",
        "max_snapshot_bytes",
        "max_schema_bytes",
        "max_validity_seconds",
        "require_predecessor_chain",
        "single_active_per_family",
        "require_known_entries_retained",
        "require_supersedes_on_version_transition",
    }
    _require(set(policy) == expected, "registry policy fields mismatch")
    _require_id(policy["registry_id"], "policy.registry_id")

    for field in (
        "min_signer_identities",
        "min_failure_domains",
        "max_entries",
        "max_snapshot_bytes",
        "max_schema_bytes",
        "max_validity_seconds",
    ):
        value = policy[field]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"policy.{field} must be positive integer",
        )

    _require(
        policy["min_failure_domains"] <= policy["min_signer_identities"],
        "failure-domain floor cannot exceed signer-identity floor",
    )
    for field in (
        "require_predecessor_chain",
        "single_active_per_family",
        "require_known_entries_retained",
        "require_supersedes_on_version_transition",
    ):
        _require(isinstance(policy[field], bool), f"policy.{field} must be bool")

    for field in (
        "allowed_roles",
        "allowed_signature_profiles",
        "allowed_canonical_encoding_versions",
    ):
        values = policy[field]
        _require(isinstance(values, list) and values, f"policy.{field} must be nonempty list")
        for value in values:
            _require_id(value, f"policy.{field}")
        _require(values == sorted(values), f"policy.{field} must be sorted")
        _require(len(values) == len(set(values)), f"policy.{field} contains duplicates")


@dataclass(frozen=True)
class TrustedInterval:
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
            "trusted interval is invalid",
        )


@dataclass(frozen=True)
class AuthenticatedSignerEvidence:
    """Result metadata from an external signature/trust verifier.

    There is intentionally no `signature_verified` boolean here. This reference
    layer assumes a separate cryptographic boundary produced these records and
    then checks that they bind the exact registry digest under current trust
    metadata. Production code must use an actually opaque authenticated type.
    """

    signer_id: str
    key_id: str
    role: str
    failure_domain: str
    signature_profile: str
    valid_from: int
    valid_until: int
    lifecycle: str
    bound_snapshot_digest: str


@dataclass(frozen=True)
class KnownSchemaState:
    schema_kind: str
    family: str
    version: int
    schema_id: str
    lifecycle_state: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.schema_kind, self.family, self.version)


@dataclass(frozen=True)
class AntiRollbackState:
    registry_id: str
    highest_sequence: int
    accepted_digest: str
    latest_issued_at: int
    policy_digest: str
    forked: bool
    known_entries: tuple[KnownSchemaState, ...]

    @classmethod
    def genesis(cls, registry_id: str) -> "AntiRollbackState":
        _require_id(registry_id, "registry_id")
        return cls(
            registry_id=registry_id,
            highest_sequence=0,
            accepted_digest=ZERO_DIGEST,
            latest_issued_at=0,
            policy_digest=ZERO_DIGEST,
            forked=False,
            known_entries=(),
        )


class VerifiedSchemaRegistrySnapshotReference:
    """Reference type-state result, not a production opaque capability."""

    __slots__ = (
        "_snapshot",
        "snapshot_digest",
        "signer_ids",
        "failure_domains",
        "trust_snapshot_digest",
        "registry_policy_digest",
        "trusted_interval",
    )

    def __init__(
        self,
        snapshot: dict[str, Any],
        digest_value: str,
        signer_ids: tuple[str, ...],
        failure_domains: tuple[str, ...],
        trust_snapshot_digest: str,
        registry_policy_digest: str,
        trusted_interval: TrustedInterval,
        *,
        _verification_marker: object,
    ) -> None:
        _require(
            _verification_marker is _VERIFICATION_MARKER,
            "verified registry reference must come from verifier",
        )
        # JSON round-trip produces a private immutable-by-convention copy.
        self._snapshot = json.loads(json.dumps(snapshot))
        self.snapshot_digest = digest_value
        self.signer_ids = signer_ids
        self.failure_domains = failure_domains
        self.trust_snapshot_digest = trust_snapshot_digest
        self.registry_policy_digest = registry_policy_digest
        self.trusted_interval = trusted_interval

    @property
    def sequence(self) -> int:
        return self._snapshot["sequence"]

    @property
    def registry_id(self) -> str:
        return self._snapshot["registry_id"]

    def resolve(
        self,
        schema_kind: str,
        family: str,
        version: int,
        expected_schema_id: str,
    ) -> dict[str, Any]:
        """Resolve one exact active schema from the already verified snapshot."""

        _require(schema_kind in SCHEMA_KINDS, "unknown schema kind")
        _require_id(family, "family")
        _require(
            isinstance(version, int) and not isinstance(version, bool) and version >= 1,
            "version must be positive integer",
        )
        _require_digest(expected_schema_id, "expected_schema_id")

        matches = [
            entry
            for entry in self._snapshot["entries"]
            if entry["schema_kind"] == schema_kind
            and entry["family"] == family
            and entry["version"] == version
        ]
        _require(len(matches) == 1, "exact schema resolution is ambiguous or missing")
        entry = matches[0]
        _require(entry["schema_id"] == expected_schema_id, "resolved schema ID mismatch")
        _require(entry["lifecycle_state"] == "active", "schema is not active")
        return _parse_canonical_schema(entry)


_VERIFICATION_MARKER = object()


@dataclass(frozen=True)
class RegistryDecision:
    status: str
    reason: str
    state: AntiRollbackState
    verified: VerifiedSchemaRegistrySnapshotReference | None

    @property
    def accepted(self) -> bool:
        return self.status in {"accepted", "replay"} and self.verified is not None


def _parse_canonical_schema(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry["canonical_schema_json"]
    _require(isinstance(raw, str), "canonical_schema_json must be string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError("canonical schema JSON is invalid") from exc
    _require(isinstance(value, dict), "canonical schema must decode to object")
    _require(
        raw.encode("utf-8") == semantic.canonical_bytes(value),
        "embedded schema JSON is not in canonical byte form",
    )
    return value


def _validate_and_normalize_entries(
    snapshot: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    entries = snapshot["entries"]
    _require(isinstance(entries, list), "snapshot.entries must be list")
    _require(len(entries) <= policy["max_entries"], "snapshot entry limit exceeded")

    expected_fields = {
        "schema_kind",
        "family",
        "version",
        "canonical_encoding_version",
        "canonical_schema_json",
        "schema_id",
        "lifecycle_state",
        "supersedes",
    }
    normalized: list[dict[str, Any]] = []
    keys: list[tuple[str, str, int]] = []
    active_by_family: dict[tuple[str, str], int] = {}

    for entry in entries:
        _require(isinstance(entry, dict), "registry entry must be object")
        _require(set(entry) == expected_fields, "registry entry fields mismatch")
        kind = entry["schema_kind"]
        _require(kind in SCHEMA_KINDS, "unknown registry schema kind")
        family = _require_id(entry["family"], "entry.family")
        version = entry["version"]
        _require(
            isinstance(version, int) and not isinstance(version, bool) and version >= 1,
            "entry.version must be positive integer",
        )
        encoding = _require_id(
            entry["canonical_encoding_version"], "entry.canonical_encoding_version"
        )
        _require(
            encoding in policy["allowed_canonical_encoding_versions"],
            "entry canonical encoding is not policy-allowed",
        )
        lifecycle = entry["lifecycle_state"]
        _require(lifecycle in LIFECYCLE_RANK, "invalid schema lifecycle state")
        claimed_schema_id = _require_digest(entry["schema_id"], "entry.schema_id")

        supersedes = entry["supersedes"]
        _require(isinstance(supersedes, list), "entry.supersedes must be list")
        for digest_value in supersedes:
            _require_digest(digest_value, "entry.supersedes")
        _require(supersedes == sorted(supersedes), "entry.supersedes must be sorted")
        _require(len(supersedes) == len(set(supersedes)), "entry.supersedes contains duplicates")

        schema_value = _parse_canonical_schema(entry)
        raw_schema = entry["canonical_schema_json"].encode("utf-8")
        _require(len(raw_schema) <= policy["max_schema_bytes"], "schema byte limit exceeded")

        if kind == "capability":
            actual_schema_id = semantic.validate_capability_schema(schema_value)
            _require(
                schema_value.get("schema") == semantic.CAPABILITY_SCHEMA_TAG,
                "capability entry contains wrong schema class",
            )
        else:
            actual_schema_id = semantic.validate_resource_schema(schema_value)
            _require(
                schema_value.get("schema")
                in {semantic.RESOURCE_SCHEMA_TAG, semantic.RESOURCE_SCHEMA_TAG_V2},
                "resource entry contains wrong schema class",
            )
            if lifecycle == "active":
                # New positive authority under the current registry profile may
                # only use resource schemas that commit runtime numeric IDs.
                semantic.require_runtime_bound_resource_schema(schema_value)

        _require(actual_schema_id == claimed_schema_id, "entry schema digest mismatch")
        _require(schema_value.get("family") == family, "entry/schema family mismatch")
        _require(schema_value.get("version") == version, "entry/schema version mismatch")

        key = (kind, family, version)
        keys.append(key)
        if lifecycle == "active":
            family_key = (kind, family)
            if family_key in active_by_family and policy["single_active_per_family"]:
                raise RegistryError("multiple active schema versions in one family")
            active_by_family[family_key] = version
        normalized.append(entry)

    _require(keys == sorted(keys), "registry entries must be sorted by exact key")
    _require(len(keys) == len(set(keys)), "duplicate registry schema key")

    all_ids = {entry["schema_id"] for entry in normalized}
    for entry in normalized:
        _require(
            all(value in all_ids for value in entry["supersedes"]),
            "supersedes references schema absent from full snapshot",
        )
    return normalized


def _validate_snapshot_shape(snapshot: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    _require(isinstance(snapshot, dict), "registry snapshot must be object")
    expected = {
        "registry_schema_version",
        "registry_id",
        "sequence",
        "issued_at",
        "expires_at",
        "previous_snapshot_digest",
        "registry_policy_id",
        "entries",
    }
    _require(set(snapshot) == expected, "registry snapshot fields mismatch")
    _require(
        snapshot["registry_schema_version"] == REGISTRY_SCHEMA_VERSION,
        "unsupported registry snapshot schema",
    )
    registry_id = _require_id(snapshot["registry_id"], "snapshot.registry_id")
    _require(registry_id == policy["registry_id"], "registry ID not allowed by policy")
    _require_digest(snapshot["previous_snapshot_digest"], "previous_snapshot_digest")
    _require_digest(snapshot["registry_policy_id"], "registry_policy_id")

    for field in ("sequence", "issued_at", "expires_at"):
        value = snapshot[field]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"snapshot.{field} must be nonnegative integer",
        )
    _require(snapshot["sequence"] >= 1, "snapshot sequence must start at one")
    _require(snapshot["expires_at"] > snapshot["issued_at"], "snapshot validity window invalid")
    _require(
        snapshot["expires_at"] - snapshot["issued_at"] <= policy["max_validity_seconds"],
        "snapshot validity window exceeds policy maximum",
    )
    return _validate_and_normalize_entries(snapshot, policy)


def _validate_signers(
    records: Iterable[AuthenticatedSignerEvidence],
    policy: dict[str, Any],
    digest_value: str,
    trusted_interval: TrustedInterval,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    records = tuple(records)
    _require(records, "no authenticated signer evidence supplied")
    by_signer: dict[str, AuthenticatedSignerEvidence] = {}

    for record in records:
        _require(isinstance(record, AuthenticatedSignerEvidence), "invalid signer evidence type")
        _require_id(record.signer_id, "signer_id")
        _require_id(record.key_id, "key_id")
        _require_id(record.role, "signer.role")
        _require_id(record.failure_domain, "signer.failure_domain")
        _require_id(record.signature_profile, "signer.signature_profile")
        _require_digest(record.bound_snapshot_digest, "signer.bound_snapshot_digest")
        _require(record.bound_snapshot_digest == digest_value, "signer evidence binds wrong snapshot")
        _require(record.role in policy["allowed_roles"], "signer role not allowed")
        _require(
            record.signature_profile in policy["allowed_signature_profiles"],
            "signature profile not allowed",
        )
        _require(record.lifecycle == "active", "signer/key lifecycle is not active")
        _require(
            isinstance(record.valid_from, int)
            and not isinstance(record.valid_from, bool)
            and isinstance(record.valid_until, int)
            and not isinstance(record.valid_until, bool)
            and record.valid_from >= 0
            and record.valid_until >= record.valid_from,
            "signer validity interval invalid",
        )
        _require(
            record.valid_from <= trusted_interval.start
            and record.valid_until >= trusted_interval.end,
            "signer evidence not valid for full trusted interval",
        )

        prior = by_signer.get(record.signer_id)
        if prior is not None:
            # Multiple keys do not create multiple identities. Contradictory
            # trusted metadata for one identity is rejected rather than guessed.
            _require(
                prior.role == record.role
                and prior.failure_domain == record.failure_domain,
                "one signer identity has conflicting trusted metadata",
            )
        else:
            by_signer[record.signer_id] = record

    signer_ids = tuple(sorted(by_signer))
    failure_domains = tuple(sorted({record.failure_domain for record in by_signer.values()}))
    _require(
        len(signer_ids) >= policy["min_signer_identities"],
        "insufficient distinct signer identities",
    )
    _require(
        len(failure_domains) >= policy["min_failure_domains"],
        "insufficient independent failure domains",
    )
    return signer_ids, failure_domains


def _forked_state(prior: AntiRollbackState) -> AntiRollbackState:
    return AntiRollbackState(
        registry_id=prior.registry_id,
        highest_sequence=prior.highest_sequence,
        accepted_digest=prior.accepted_digest,
        latest_issued_at=prior.latest_issued_at,
        policy_digest=prior.policy_digest,
        forked=True,
        known_entries=prior.known_entries,
    )


def _state_known_map(state: AntiRollbackState) -> dict[tuple[str, str, int], KnownSchemaState]:
    return {entry.key: entry for entry in state.known_entries}


def _current_active_versions(
    entries: Iterable[KnownSchemaState],
) -> dict[tuple[str, str], KnownSchemaState]:
    result: dict[tuple[str, str], KnownSchemaState] = {}
    for entry in entries:
        if entry.lifecycle_state == "active":
            result[(entry.schema_kind, entry.family)] = entry
    return result


def _validate_transition(
    prior: AntiRollbackState,
    snapshot: dict[str, Any],
    entries: list[dict[str, Any]],
    policy: dict[str, Any],
    digest_value: str,
    policy_id: str,
) -> tuple[str, AntiRollbackState] | None:
    """Return replay status/state, fork status/state, or None for forward transition."""

    if prior.forked:
        return ("forked", prior)
    _require(prior.registry_id == snapshot["registry_id"], "anti-rollback registry mismatch")

    if prior.highest_sequence == 0:
        _require(snapshot["sequence"] == 1, "genesis acceptance requires sequence 1")
        if policy["require_predecessor_chain"]:
            _require(
                snapshot["previous_snapshot_digest"] == ZERO_DIGEST,
                "genesis predecessor digest must be zero",
            )
        return None

    if prior.policy_digest != policy_id:
        raise RegistryError("registry policy changed without explicit epoch transition")

    if snapshot["sequence"] < prior.highest_sequence:
        raise RegistryError("registry sequence rollback")
    if snapshot["sequence"] == prior.highest_sequence:
        if digest_value != prior.accepted_digest:
            return ("forked", _forked_state(prior))
        return ("replay", prior)

    if policy["require_predecessor_chain"] and snapshot["previous_snapshot_digest"] != prior.accepted_digest:
        return ("forked", _forked_state(prior))
    if snapshot["issued_at"] < prior.latest_issued_at:
        raise RegistryError("registry issued_at regression")

    prior_known = _state_known_map(prior)
    current_by_key = {
        (entry["schema_kind"], entry["family"], entry["version"]): entry
        for entry in entries
    }
    if policy["require_known_entries_retained"]:
        missing = set(prior_known) - set(current_by_key)
        _require(not missing, "previously known schema key disappeared from full snapshot")

    for key, old in prior_known.items():
        if key not in current_by_key:
            continue
        new = current_by_key[key]
        _require(new["schema_id"] == old.schema_id, "schema bytes changed under existing key")
        _require(
            LIFECYCLE_RANK[new["lifecycle_state"]] >= LIFECYCLE_RANK[old.lifecycle_state],
            "schema lifecycle moved to a more permissive state",
        )

    prior_active = _current_active_versions(prior.known_entries)
    current_states = [
        KnownSchemaState(
            schema_kind=entry["schema_kind"],
            family=entry["family"],
            version=entry["version"],
            schema_id=entry["schema_id"],
            lifecycle_state=entry["lifecycle_state"],
        )
        for entry in entries
    ]
    current_active = _current_active_versions(current_states)

    for family_key, old_active in prior_active.items():
        new_active = current_active.get(family_key)
        if new_active is None:
            continue
        _require(
            new_active.version >= old_active.version,
            "active schema version rollback",
        )
        if new_active.version > old_active.version and policy["require_supersedes_on_version_transition"]:
            new_entry = current_by_key[new_active.key]
            _require(
                old_active.schema_id in new_entry["supersedes"],
                "new active schema does not explicitly supersede prior active schema",
            )

    return None


def evaluate_registry_snapshot(
    snapshot: dict[str, Any],
    claimed_snapshot_digest: str,
    authenticated_signers: Iterable[AuthenticatedSignerEvidence],
    policy: dict[str, Any],
    trust_snapshot_digest: str,
    trusted_interval: TrustedInterval,
    prior_state: AntiRollbackState,
) -> RegistryDecision:
    """Evaluate one full registry snapshot and return monotonic reference state.

    Any malformed/ordinary denial returns status `denied` with prior state intact.
    Same-sequence collisions or predecessor mismatches return `forked` and a
    persistently forked state. Accepted/replayed results include a reference
    verified type-state object that is valid only for this process evaluation.
    """

    try:
        validate_policy(policy)
        policy_id = semantic.digest(policy)
        _require_digest(trust_snapshot_digest, "trust_snapshot_digest")
        trusted_interval.validate()
        entries = _validate_snapshot_shape(snapshot, policy)
        canonical = semantic.canonical_bytes(snapshot)
        _require(len(canonical) <= policy["max_snapshot_bytes"], "snapshot byte limit exceeded")
        actual_digest = snapshot_digest(snapshot)
        _require_digest(claimed_snapshot_digest, "claimed_snapshot_digest")
        _require(actual_digest == claimed_snapshot_digest, "snapshot digest mismatch")
        _require(snapshot["registry_policy_id"] == policy_id, "snapshot registry policy mismatch")
        _require(
            snapshot["issued_at"] <= trusted_interval.start
            and trusted_interval.end <= snapshot["expires_at"],
            "snapshot is not fresh for full trusted interval",
        )
        signer_ids, failure_domains = _validate_signers(
            authenticated_signers, policy, actual_digest, trusted_interval
        )
        transition = _validate_transition(
            prior_state, snapshot, entries, policy, actual_digest, policy_id
        )
        if transition is not None and transition[0] == "forked":
            return RegistryDecision(
                status="forked",
                reason="registry collision/fork or prior forked state",
                state=transition[1],
                verified=None,
            )

        verified = VerifiedSchemaRegistrySnapshotReference(
            snapshot,
            actual_digest,
            signer_ids,
            failure_domains,
            trust_snapshot_digest,
            policy_id,
            trusted_interval,
            _verification_marker=_VERIFICATION_MARKER,
        )
        if transition is not None and transition[0] == "replay":
            return RegistryDecision(
                status="replay",
                reason="exact already-accepted snapshot replay",
                state=prior_state,
                verified=verified,
            )

        known_entries = tuple(
            KnownSchemaState(
                schema_kind=entry["schema_kind"],
                family=entry["family"],
                version=entry["version"],
                schema_id=entry["schema_id"],
                lifecycle_state=entry["lifecycle_state"],
            )
            for entry in entries
        )
        new_state = AntiRollbackState(
            registry_id=snapshot["registry_id"],
            highest_sequence=snapshot["sequence"],
            accepted_digest=actual_digest,
            latest_issued_at=snapshot["issued_at"],
            policy_digest=policy_id,
            forked=False,
            known_entries=known_entries,
        )
        return RegistryDecision(
            status="accepted",
            reason="registry snapshot accepted by reference semantic verifier",
            state=new_state,
            verified=verified,
        )
    except (RegistryError, semantic.SchemaError) as exc:
        return RegistryDecision(
            status="denied",
            reason=str(exc),
            state=prior_state,
            verified=None,
        )
