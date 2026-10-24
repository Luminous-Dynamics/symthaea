"""Verification gate for signed differential-privacy mechanism attestations.

This module verifies that a separately trusted attester signed a canonical receipt
whose claims exactly match a durable reservation and the output bytes. A valid
signature authenticates the attester's assertion; it is NOT, by itself, a proof
that the implementation is correct or that the output satisfies differential
privacy. Production use requires an independently managed trust policy and a
qualified mechanism/clipper behind the trusted attester.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tools.energy_telemetry.durable_privacy_ledger import (
    MAX_DP_OUTPUT_BYTES,
    CommittedPrivacyOutput,
    DurablePrivacyLedger,
    DurablePrivacyPolicyError,
)

ATTESTATION_VERSION = 1
MAX_ATTESTATION_BYTES = 16_384
EXECUTION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_ATTESTATION_DOMAIN = b"symthaea-dp-mechanism-attestation:v1\x00"

_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "release_id",
    "reservation_receipt_sha256",
    "query_sha256",
    "window_start_utc",
    "window_end_utc",
    "recipient_class",
    "mechanism",
    "library_name",
    "library_version",
    "mechanism_id",
    "feature_flags",
    "privacy_unit",
    "adjacency",
    "contribution_limit_wh",
    "sensitivity_wh",
    "contribution_policy_id",
    "privacy_policy_id",
    "epsilon_nano",
    "delta_pico",
    "cohort_size",
    "qualification_reference",
    "execution_id",
    "signer_key_sha256",
    "output_sha256",
})


class MechanismAttestationError(ValueError):
    """A signed mechanism attestation is malformed, untrusted or mismatched."""


@dataclass(frozen=True)
class MechanismAttestationPolicy:
    """Deployment-controlled trust policy; never construct from receipt fields.

    The trusted public key and every approved identity/version/reference must
    arrive from configuration controlled separately from the requesting caller.
    """

    trusted_public_key: bytes
    qualification_reference: str
    library_name: str
    library_version: str
    mechanism: str
    mechanism_id: str
    privacy_unit: str
    adjacency: str
    contribution_policy_id: str
    privacy_policy_id: str
    max_contribution_limit_wh: int
    expected_feature_flags: tuple[str, ...] = ()


def canonical_attestation_bytes(fields: Mapping[str, object]) -> bytes:
    """Canonical JSON used by the attester and verifier."""
    try:
        return json.dumps(
            dict(fields),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError, OverflowError) as exc:
        raise MechanismAttestationError("attestation fields are not canonical JSON values") from exc


def attestation_signing_message(canonical_bytes: bytes) -> bytes:
    """Domain-separate this signature from envelopes and unrelated receipts."""
    if not isinstance(canonical_bytes, bytes):
        raise MechanismAttestationError("canonical attestation must be bytes")
    return _ATTESTATION_DOMAIN + canonical_bytes


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MechanismAttestationError("attestation contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise MechanismAttestationError(f"attestation contains non-standard JSON constant: {value}")


def _require_text(fields: dict[str, object], name: str, maximum: int = 256) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise MechanismAttestationError(f"{name} must be a non-empty bounded string")
    return value


def _require_int(fields: dict[str, object], name: str, *, minimum: int = 0) -> int:
    value = fields.get(name)
    if type(value) is not int or value < minimum:
        raise MechanismAttestationError(f"{name} must be an integer >= {minimum}")
    return value


class QualifiedOutputVerifier:
    """Verify a pinned attester's signed claim before binding output to a reservation.

    The public key and qualification policy are fixed when this verifier is
    instantiated. Request data cannot nominate its own trusted key or library.
    """

    def __init__(self, policy: MechanismAttestationPolicy) -> None:
        if not isinstance(policy, MechanismAttestationPolicy):
            raise MechanismAttestationError("a deployment-controlled attestation policy is required")
        if not isinstance(policy.trusted_public_key, bytes) or len(policy.trusted_public_key) != 32:
            raise MechanismAttestationError("trusted Ed25519 public key must be exactly 32 raw bytes")
        for name in (
            "qualification_reference", "library_name", "library_version",
            "mechanism_id", "privacy_unit", "adjacency",
            "contribution_policy_id", "privacy_policy_id",
        ):
            value = getattr(policy, name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise MechanismAttestationError(f"policy {name} must be non-empty and bounded")
        if not isinstance(policy.mechanism, str) or policy.mechanism not in {"discrete_laplace", "gaussian"}:
            raise MechanismAttestationError("policy mechanism must be a supported ledger mechanism")
        if policy.privacy_unit not in {"household", "site"}:
            raise MechanismAttestationError("policy privacy_unit must be household or site")
        if policy.adjacency not in {"add_remove_one_participant", "replace_one_participant"}:
            raise MechanismAttestationError("policy adjacency must be a supported neighboring-dataset relation")
        if (
            type(policy.max_contribution_limit_wh) is not int
            or not 1 <= policy.max_contribution_limit_wh <= 10**15
        ):
            raise MechanismAttestationError("policy max_contribution_limit_wh must be a bounded positive integer")
        flags = policy.expected_feature_flags
        if (
            not isinstance(flags, tuple)
            or any(not isinstance(flag, str) or not flag or len(flag) > 128 for flag in flags)
            or tuple(sorted(set(flags))) != flags
        ):
            raise MechanismAttestationError("approved feature flags must be a sorted unique tuple")
        try:
            self._public_key = Ed25519PublicKey.from_public_bytes(policy.trusted_public_key)
        except (TypeError, ValueError) as exc:
            raise MechanismAttestationError("invalid trusted Ed25519 public key") from exc
        self.policy = policy
        self.signer_key_sha256 = hashlib.sha256(policy.trusted_public_key).hexdigest()

    def verify_and_commit(
        self,
        ledger: DurablePrivacyLedger,
        *,
        release_id: str,
        output_bytes: bytes,
        attestation_bytes: bytes,
        signature: bytes,
        committed_at_utc: datetime,
    ) -> CommittedPrivacyOutput:
        """Verify the exact receipt/output binding, then durably commit those bytes.

        This authenticates a trusted attester's signed claim. It does not prove
        that the claimed library was honestly run; the signing authority must
        independently verify its mechanism, clipping, sensitivity and execution.
        """
        if not isinstance(attestation_bytes, bytes) or not 1 <= len(attestation_bytes) <= MAX_ATTESTATION_BYTES:
            raise MechanismAttestationError("attestation must be non-empty and at most 16 KiB")
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise MechanismAttestationError("Ed25519 signature must be exactly 64 bytes")
        if not isinstance(output_bytes, bytes) or not 1 <= len(output_bytes) <= MAX_DP_OUTPUT_BYTES:
            raise MechanismAttestationError("output must be non-empty bytes and at most 1 MiB")

        try:
            fields = json.loads(
                attestation_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except MechanismAttestationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise MechanismAttestationError("attestation is not valid UTF-8 JSON") from exc
        if not isinstance(fields, dict):
            raise MechanismAttestationError("attestation must be a JSON object")
        if frozenset(fields) != _REQUIRED_FIELDS:
            missing = sorted(_REQUIRED_FIELDS - frozenset(fields))
            extra = sorted(frozenset(fields) - _REQUIRED_FIELDS)
            raise MechanismAttestationError(f"attestation field set mismatch; missing={missing}, extra={extra}")
        canonical = canonical_attestation_bytes(fields)
        if canonical != attestation_bytes:
            raise MechanismAttestationError("attestation JSON is not in canonical form")
        try:
            self._public_key.verify(signature, attestation_signing_message(canonical))
        except InvalidSignature as exc:
            raise MechanismAttestationError("attestation signature is invalid for the pinned key") from exc

        if type(fields["schema_version"]) is not int or fields["schema_version"] != ATTESTATION_VERSION:
            raise MechanismAttestationError("unsupported attestation schema_version")
        if fields["signer_key_sha256"] != self.signer_key_sha256:
            raise MechanismAttestationError("attestation signer fingerprint differs from pinned key")
        for field_name, expected in (
            ("qualification_reference", self.policy.qualification_reference),
            ("library_name", self.policy.library_name),
            ("library_version", self.policy.library_version),
            ("mechanism", self.policy.mechanism),
            ("mechanism_id", self.policy.mechanism_id),
            ("privacy_unit", self.policy.privacy_unit),
            ("adjacency", self.policy.adjacency),
            ("contribution_policy_id", self.policy.contribution_policy_id),
            ("privacy_policy_id", self.policy.privacy_policy_id),
        ):
            if _require_text(fields, field_name) != expected:
                raise MechanismAttestationError(f"{field_name} differs from deployment-approved policy")

        feature_flags = fields["feature_flags"]
        if (
            not isinstance(feature_flags, list)
            or any(not isinstance(flag, str) or not flag or len(flag) > 128 for flag in feature_flags)
            or feature_flags != list(self.policy.expected_feature_flags)
        ):
            raise MechanismAttestationError("feature_flags differ from deployment-approved policy")
        if not isinstance(fields["execution_id"], str) or not EXECUTION_ID_RE.fullmatch(fields["execution_id"]):
            raise MechanismAttestationError("execution_id must be a 32-character lowercase hexadecimal ID")
        for digest_field in ("reservation_receipt_sha256", "query_sha256", "output_sha256"):
            value = fields[digest_field]
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise MechanismAttestationError(f"{digest_field} must be a lowercase SHA-256 digest")
        expected_output_digest = hashlib.sha256(output_bytes).hexdigest()
        if fields["output_sha256"] != expected_output_digest:
            raise MechanismAttestationError("attested output digest does not match exact output bytes")

        context = ledger.reservation_context(release_id)
        if context.contribution_limit_wh > self.policy.max_contribution_limit_wh:
            raise MechanismAttestationError(
                "reservation contribution limit exceeds deployment-qualified maximum"
            )
        expected_context = {
            "release_id": context.release_id,
            "reservation_receipt_sha256": context.receipt_sha256,
            "query_sha256": context.query_sha256,
            "window_start_utc": context.window_start_utc,
            "window_end_utc": context.window_end_utc,
            "recipient_class": context.recipient_class,
            "mechanism": context.mechanism,
            "adjacency": context.adjacency,
            "contribution_limit_wh": context.contribution_limit_wh,
            "contribution_policy_id": context.contribution_policy_id,
            "privacy_policy_id": context.privacy_policy_id,
            "epsilon_nano": context.epsilon_nano,
            "delta_pico": context.delta_pico,
            "cohort_size": context.cohort_size,
        }
        for name, expected in expected_context.items():
            actual = fields.get(name)
            if actual != expected or (type(expected) is int and type(actual) is not int):
                raise MechanismAttestationError(f"{name} does not match durable reservation")
        sensitivity = _require_int(fields, "sensitivity_wh", minimum=1)
        if sensitivity != context.contribution_limit_wh:
            raise MechanismAttestationError(
                "sensitivity_wh must equal the configured [0, L] sum bound for the declared adjacency profile"
            )
        if context.mechanism == "discrete_laplace" and context.delta_pico != 0:
            raise MechanismAttestationError("discrete_laplace reservation must have delta=0")
        if context.mechanism == "gaussian" and context.delta_pico <= 0:
            raise MechanismAttestationError("gaussian reservation must have positive delta")

        return ledger._commit_output_storage(
            release_id=release_id,
            output_bytes=output_bytes,
            committed_at_utc=committed_at_utc,
            attestation_bytes=attestation_bytes,
            signature=signature,
            signer_key_sha256=self.signer_key_sha256,
        )

    def load_committed_output(
        self, ledger: DurablePrivacyLedger, *, release_id: str
    ) -> CommittedPrivacyOutput:
        """Reload a prior result only after verifying its retained signed attestation."""
        stored = ledger._committed_output_storage(release_id)
        attestation_bytes, signature, signer_key_sha256, committed_at_utc = ledger._stored_attestation(
            release_id
        )
        if signer_key_sha256 != self.signer_key_sha256:
            raise MechanismAttestationError(
                "stored attestation signer fingerprint differs from deployment-pinned key"
            )
        return self.verify_and_commit(
            ledger,
            release_id=release_id,
            output_bytes=stored.output_bytes,
            attestation_bytes=attestation_bytes,
            signature=signature,
            committed_at_utc=committed_at_utc,
        )
