#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

import base64
import binascii
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path

MAX_PROVIDER_TOPOLOGY_ATTESTATION_BYTES = 1_048_576
ED25519_SIGNATURE_BYTES = 64
ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


@dataclass
class PromotionEffectReceipt:
    # Local reconciliation record only. Provider provenance/causal attribution
    # is intentionally not represented here; see #7101.
    promotion_operation_id: str
    expected_pr_head_sha: str
    observed_merge_commit: str


@dataclass(frozen=True)
class StackEntryV1:
    pr_number: int
    head_sha: str
    base_ref: str
    base_head_sha: str


@dataclass(frozen=True)
class PromotionOperationIdentityV1:
    repository: str
    provider_stack_number: int
    requested_pr_number: int
    requested_pr_head_sha: str
    base_ref: str
    base_tip_sha: str
    ordered_stack: tuple[StackEntryV1, ...]
    merge_method: str
    merge_action: str
    trust_root_generation: int
    governance_generation: int
    bypass_rules: bool = False

    def canonical_bytes(self) -> bytes:
        payload = {
            "base_ref": self.base_ref,
            "provider_stack_number": self.provider_stack_number,
            "base_tip_sha": self.base_tip_sha,
            "governance_generation": self.governance_generation,
            "bypass_rules": self.bypass_rules,
            "merge_action": self.merge_action,
            "merge_method": self.merge_method,
            "ordered_stack": [
                {
                    "base_head_sha": entry.base_head_sha,
                    "base_ref": entry.base_ref,
                    "head_sha": entry.head_sha,
                    "pr_number": entry.pr_number,
                }
                for entry in self.ordered_stack
            ],
            "repository": self.repository,
            "requested_pr_head_sha": self.requested_pr_head_sha,
            "requested_pr_number": self.requested_pr_number,
            "trust_root_generation": self.trust_root_generation,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def provider_constraints_valid(self) -> bool:
        # GitHub stacked merges do not support bypass_rules for a whole
        # multi-PR stack; bypass is only available for the bottom PR.
        return not (len(self.ordered_stack) > 1 and self.bypass_rules)


@dataclass(frozen=True)
class PromotionStackEffectV1:
    pr_number: int
    expected_head_sha: str
    observed_merge_commit: str


@dataclass(frozen=True)
class PromotionStackEffectSetV1:
    operation_identity_digest: str
    effects: tuple[PromotionStackEffectV1, ...]

    def validates_complete(self, identity: PromotionOperationIdentityV1) -> bool:
        if self.operation_identity_digest != identity.digest():
            return False
        expected = identity.ordered_stack
        if len(self.effects) != len(expected):
            return False
        observed_prs = [effect.pr_number for effect in self.effects]
        if len(observed_prs) != len(set(observed_prs)):
            return False

        return all(
            observed.pr_number == entry.pr_number
            and observed.expected_head_sha == entry.head_sha
            and bool(observed.observed_merge_commit)
            for entry, observed in zip(expected, self.effects)
        )


@dataclass(frozen=True)
class ProviderStackObservationV1:
    observation_source: str
    stack_number: int
    stack_size: int
    stack_position: int
    base_ref: str
    base_tip_sha: str
    ordered_stack: tuple[StackEntryV1, ...]
    observation_id: str | None = None

    def internally_consistent(self, requested_pr_number: int) -> bool:
        if not self.observation_source:
            return False
        if self.stack_number <= 0 or self.stack_size <= 0:
            return False
        if self.stack_size != len(self.ordered_stack):
            return False
        if not 1 <= self.stack_position <= self.stack_size:
            return False
        if self.ordered_stack[self.stack_position - 1].pr_number != requested_pr_number:
            return False
        bottom = self.ordered_stack[0]
        if bottom.base_ref != self.base_ref or bottom.base_head_sha != self.base_tip_sha:
            return False
        return True

    def matches_reserved(self, identity: PromotionOperationIdentityV1) -> bool:
        if not self.internally_consistent(identity.requested_pr_number):
            return False
        operation_depth = len(identity.ordered_stack)
        if self.stack_number != identity.provider_stack_number:
            return False
        if self.stack_position != operation_depth:
            return False
        if tuple(self.ordered_stack[:operation_depth]) != identity.ordered_stack:
            return False
        requested = self.ordered_stack[self.stack_position - 1]
        return (
            self.base_ref == identity.base_ref
            and self.base_tip_sha == identity.base_tip_sha
            and requested.pr_number == identity.requested_pr_number
            and requested.head_sha == identity.requested_pr_head_sha
        )

    def digest(self) -> str:
        payload = {
            "base_ref": self.base_ref,
            "base_tip_sha": self.base_tip_sha,
            "observation_id": self.observation_id,
            "observation_source": self.observation_source,
            "ordered_stack": [
                {
                    "base_head_sha": entry.base_head_sha,
                    "base_ref": entry.base_ref,
                    "head_sha": entry.head_sha,
                    "pr_number": entry.pr_number,
                }
                for entry in self.ordered_stack
            ],
            "stack_number": self.stack_number,
            "stack_position": self.stack_position,
            "stack_size": self.stack_size,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()


def provider_stack_observation_matches_reserved(
    observation: ProviderStackObservationV1 | None,
    identity: PromotionOperationIdentityV1,
) -> bool:
    return observation is not None and observation.matches_reserved(identity)


@dataclass(frozen=True)
class ProviderTopologyCasPredicateV1:
    """Explicit conditional predicate a provider would have to enforce."""
    operation_identity_digest: str
    observation_digest: str
    pre_submit_sequence: int

    def canonical_bytes(self) -> bytes:
        payload = {
            "observation_digest": self.observation_digest,
            "operation_identity_digest": self.operation_identity_digest,
            "pre_submit_sequence": self.pre_submit_sequence,
            "predicate": "provider-topology-cas-v1",
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_binding(
        cls,
        identity: PromotionOperationIdentityV1,
        observation: ProviderStackObservationV1,
        pre_submit_sequence: int,
    ) -> "ProviderTopologyCasPredicateV1":
        return cls(
            operation_identity_digest=identity.digest(),
            observation_digest=observation.digest(),
            pre_submit_sequence=pre_submit_sequence,
        )


@dataclass(frozen=True)
class ProviderTopologyCasRequestV1:
    """Canonical provider request that carries the conditional topology predicate."""
    requested_pr_number: int
    expected_head_sha: str
    merge_method: str
    merge_action: str
    bypass_rules: bool
    predicate_digest: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "bypass_rules": self.bypass_rules,
            "expected_head_sha": self.expected_head_sha,
            "merge_action": self.merge_action,
            "merge_method": self.merge_method,
            "predicate_digest": self.predicate_digest,
            "requested_pr_number": self.requested_pr_number,
            "request": "provider-topology-cas-v1",
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_identity_predicate(
        cls,
        identity: PromotionOperationIdentityV1,
        predicate: ProviderTopologyCasPredicateV1,
    ) -> "ProviderTopologyCasRequestV1":
        return cls(
            requested_pr_number=identity.requested_pr_number,
            expected_head_sha=identity.requested_pr_head_sha,
            merge_method=identity.merge_method,
            merge_action=identity.merge_action,
            bypass_rules=identity.bypass_rules,
            predicate_digest=predicate.digest(),
        )


@dataclass(frozen=True)
class ProviderTopologyCasSubmissionV1:
    """Canonical acceptance record binding a provider operation handle to one request."""
    request_digest: str
    provider_operation_id: str
    submission_source: str
    submission_result: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "provider_operation_id": self.provider_operation_id,
            "request_digest": self.request_digest,
            "submission": "provider-topology-cas-v1",
            "submission_result": self.submission_result,
            "submission_source": self.submission_source,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validates(self, request: ProviderTopologyCasRequestV1) -> bool:
        return (
            self.submission_source == "provider-submission-response"
            and self.submission_result == "accepted"
            and bool(self.provider_operation_id)
            and self.request_digest == request.digest()
        )


@dataclass(frozen=True)
class ProviderTopologyCasExecutionV1:
    """Canonical provider execution witness stating that the predicate was enforced."""
    execution_source: str
    provider_operation_id: str
    request_digest: str
    submission_digest: str
    predicate_digest: str
    enforcement_result: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "enforcement_result": self.enforcement_result,
            "execution_source": self.execution_source,
            "predicate_digest": self.predicate_digest,
            "provider_operation_id": self.provider_operation_id,
            "request_digest": self.request_digest,
            "submission_digest": self.submission_digest,
            "execution": "provider-topology-cas-v1",
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validates(
        self,
        request: ProviderTopologyCasRequestV1,
        submission: ProviderTopologyCasSubmissionV1,
        predicate: ProviderTopologyCasPredicateV1,
    ) -> bool:
        return (
            self.execution_source == "provider-conditional-execution"
            and self.enforcement_result == "enforced"
            and bool(self.provider_operation_id)
            and self.request_digest == request.digest()
            and self.submission_digest == submission.digest()
            and self.provider_operation_id == submission.provider_operation_id
            and self.predicate_digest == predicate.digest()
        )


@dataclass(frozen=True)
class ProviderTopologyCasDsseEnvelopeV1:
    """Single-signature DSSE envelope for the exact provider topology statement."""
    payload_type: str
    payload_base64: str
    key_id: str
    signature_base64: str

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "payloadType": self.payload_type,
                "payload": self.payload_base64,
                "signatures": [{"keyid": self.key_id, "sig": self.signature_base64}],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def decoded_payload(self) -> bytes:
        return base64.b64decode(self.payload_base64, validate=True)

    def pae(self) -> bytes:
        payload_type = self.payload_type.encode("utf-8")
        payload = self.decoded_payload()
        # DSSEv1 PAE: "DSSEv1 " + LEN(type) + " " + type + " " + LEN(payload) + " " + payload
        return (
            b"DSSEv1 "
            + str(len(payload_type)).encode("ascii")
            + b" "
            + payload_type
            + b" "
            + str(len(payload)).encode("ascii")
            + b" "
            + payload
        )


@dataclass(frozen=True)
class ProviderTopologyCasTrustRootV1:
    """Out-of-band verifier policy; never accepted from the attestation itself."""
    trust_root_id: str
    generation: int
    repository: str
    signer_identity: str
    key_id: str
    public_key_pem: str


@dataclass(frozen=True)
class ProviderTopologyCasTrustPolicyV1:
    """Versioned signer authorization policy whose digest is pinned out of band."""
    policy_id: str
    generation: int
    repository: str
    authorized_roots: tuple[ProviderTopologyCasTrustRootV1, ...]
    revoked_key_ids: tuple[str, ...] = ()
    revoked_signer_identities: tuple[str, ...] = ()

    def canonical_bytes(self) -> bytes:
        payload = {
            "policy": "provider-topology-cas-trust-policy-v1",
            "policy_id": self.policy_id,
            "generation": self.generation,
            "repository": self.repository,
            "authorized_roots": [
                {
                    "trust_root_id": root.trust_root_id,
                    "generation": root.generation,
                    "repository": root.repository,
                    "signer_identity": root.signer_identity,
                    "key_id": root.key_id,
                    "public_key_pem": root.public_key_pem,
                }
                for root in sorted(
                    self.authorized_roots,
                    key=lambda candidate: (candidate.key_id, candidate.trust_root_id),
                )
            ],
            "revoked_key_ids": sorted(self.revoked_key_ids),
            "revoked_signer_identities": sorted(self.revoked_signer_identities),
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def structurally_valid(self) -> bool:
        if not self.policy_id or self.generation <= 0 or not self.repository:
            return False
        keys = [root.key_id for root in self.authorized_roots]
        root_ids = [root.trust_root_id for root in self.authorized_roots]
        if len(keys) != len(set(keys)) or len(root_ids) != len(set(root_ids)):
            return False
        if len(self.revoked_key_ids) != len(set(self.revoked_key_ids)):
            return False
        if len(self.revoked_signer_identities) != len(set(self.revoked_signer_identities)):
            return False
        return all(
            root.repository == self.repository
            and root.generation > 0
            and bool(root.signer_identity)
            and bool(root.key_id)
            and bool(root.public_key_pem)
            for root in self.authorized_roots
        )

    def authorizes(
        self,
        root: ProviderTopologyCasTrustRootV1,
        identity: PromotionOperationIdentityV1,
        expected_policy_digest: str | None,
        expected_policy_generation: int | None,
    ) -> bool:
        if not self.structurally_valid():
            return False
        if not expected_policy_digest or self.digest() != expected_policy_digest:
            return False
        if expected_policy_generation is None or expected_policy_generation <= 0:
            return False
        if self.generation != expected_policy_generation:
            return False
        if self.repository != identity.repository or root.repository != self.repository:
            return False
        if root.generation != identity.trust_root_generation:
            return False
        if root.key_id in self.revoked_key_ids or root.signer_identity in self.revoked_signer_identities:
            return False
        if root not in self.authorized_roots:
            return False
        return _public_key_fingerprint(root.public_key_pem) == root.key_id


@dataclass(frozen=True)
class ProviderTopologyCasAttestationV1:
    """DSSE envelope containing a signed in-toto statement for topology enforcement."""
    envelope: ProviderTopologyCasDsseEnvelopeV1

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "attestation": "provider-topology-cas-v1",
                "envelope_digest": self.envelope.digest(),
                "envelope": json.loads(self.envelope.canonical_bytes()),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class ProviderTopologyCasVerificationV1:
    """Receipt from actual DSSE Ed25519 verification against an out-of-band trust root."""
    verifier_source: str
    verification_method: str
    attestation_digest: str
    verification_result: str
    trust_root_id: str
    trust_root_generation: int
    signer_identity: str
    subject_repository: str
    operation_identity_digest: str
    key_id: str
    trust_policy_id: str
    trust_policy_generation: int
    trust_policy_digest: str

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "verification": "provider-topology-cas-v1",
                "verifier_source": self.verifier_source,
                "verification_method": self.verification_method,
                "attestation_digest": self.attestation_digest,
                "verification_result": self.verification_result,
                "trust_root_id": self.trust_root_id,
                "trust_root_generation": self.trust_root_generation,
                "signer_identity": self.signer_identity,
                "subject_repository": self.subject_repository,
                "operation_identity_digest": self.operation_identity_digest,
                "key_id": self.key_id,
                "trust_policy_id": self.trust_policy_id,
                "trust_policy_generation": self.trust_policy_generation,
                "trust_policy_digest": self.trust_policy_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validates(
        self,
        attestation: ProviderTopologyCasAttestationV1,
        trust_root: ProviderTopologyCasTrustRootV1,
        identity: PromotionOperationIdentityV1,
        observation: ProviderStackObservationV1,
        pre_submit_sequence: int,
        request: ProviderTopologyCasRequestV1,
        submission: ProviderTopologyCasSubmissionV1,
        execution: ProviderTopologyCasExecutionV1,
        trust_policy: ProviderTopologyCasTrustPolicyV1 | None,
        expected_trust_policy_digest: str | None,
        expected_trust_policy_generation: int | None,
    ) -> bool:
        verified = verify_provider_topology_cas_attestation(
            attestation,
            trust_root,
            identity,
            observation,
            pre_submit_sequence,
            request,
            submission,
            execution,
            trust_policy,
            expected_trust_policy_digest,
            expected_trust_policy_generation,
        )
        return verified is not None and self == verified


@dataclass(frozen=True)
class ProviderTopologyCasProviderResultV1:
    """Canonical provider result separating admission, enforcement, and authenticated statement."""
    result_source: str
    provider_operation_id: str
    request_digest: str
    submission_digest: str
    predicate_admission: str
    execution: ProviderTopologyCasExecutionV1 | None
    execution_digest: str | None
    attestation: ProviderTopologyCasAttestationV1 | None
    attestation_digest: str | None
    verification: ProviderTopologyCasVerificationV1 | None
    verification_digest: str | None

    def canonical_bytes(self) -> bytes:
        payload = {
            "predicate_admission": self.predicate_admission,
            "provider_operation_id": self.provider_operation_id,
            "request_digest": self.request_digest,
            "result_source": self.result_source,
            "submission_digest": self.submission_digest,
            "execution_digest": self.execution_digest,
            "attestation_digest": self.attestation_digest,
            "verification_digest": self.verification_digest,
            "result": "provider-topology-cas-v1",
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validates(
        self,
        request: ProviderTopologyCasRequestV1,
        submission: ProviderTopologyCasSubmissionV1,
        predicate: ProviderTopologyCasPredicateV1,
        identity: PromotionOperationIdentityV1,
        observation: ProviderStackObservationV1,
        pre_submit_sequence: int,
        trust_root: ProviderTopologyCasTrustRootV1,
        trust_policy: ProviderTopologyCasTrustPolicyV1 | None,
        expected_trust_policy_digest: str | None,
        expected_trust_policy_generation: int | None,
    ) -> bool:
        if (
            self.result_source != "provider-operation-result"
            or not bool(self.provider_operation_id)
            or self.predicate_admission != "accepted"
            or self.request_digest != request.digest()
            or self.submission_digest != submission.digest()
            or submission.provider_operation_id != self.provider_operation_id
            or request.predicate_digest != predicate.digest()
        ):
            return False
        execution = self.execution
        if execution is None or self.execution_digest != execution.digest():
            return False
        if not execution.validates(request, submission, predicate):
            return False
        attestation = self.attestation
        if attestation is None or self.attestation_digest != attestation.digest():
            return False
        verification = self.verification
        if verification is None or self.verification_digest != verification.digest():
            return False
        return verification.validates(
            attestation,
            trust_root,
            identity,
            observation,
            pre_submit_sequence,
            request,
            submission,
            execution,
            trust_policy,
            expected_trust_policy_digest,
            expected_trust_policy_generation,
        )


def _strict_json_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON object key")
        out[key] = value
    return out


def _provider_topology_statement(
    identity: PromotionOperationIdentityV1,
    observation: ProviderStackObservationV1,
    pre_submit_sequence: int,
    request: ProviderTopologyCasRequestV1,
    submission: ProviderTopologyCasSubmissionV1,
    execution: ProviderTopologyCasExecutionV1,
    trust_root: ProviderTopologyCasTrustRootV1,
    trust_policy: ProviderTopologyCasTrustPolicyV1,
) -> dict:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": identity.repository,
                "digest": {"sha256": identity.digest()},
            }
        ],
        "predicateType": "https://luminousdynamics.org/attestations/provider-topology-cas/v1",
        "predicate": {
            "attestation_id": "attestation-1",
            "statement": "predicate-enforced",
            "signer_identity": trust_root.signer_identity,
            "repository": identity.repository,
            "operation_identity_digest": identity.digest(),
            "request_digest": request.digest(),
            "submission_digest": submission.digest(),
            "provider_operation_id": execution.provider_operation_id,
            "execution_digest": execution.digest(),
            "predicate_digest": request.predicate_digest,
            "observation_digest": observation.digest(),
            "pre_submit_sequence": pre_submit_sequence,
            "trust_root_id": trust_root.trust_root_id,
            "trust_root_generation": trust_root.generation,
            "trust_policy_id": trust_policy.policy_id,
            "trust_policy_generation": trust_policy.generation,
            "trust_policy_digest": trust_policy.digest(),
            "governance_generation": identity.governance_generation,
        },
    }


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _public_key_fingerprint(public_key_pem: str) -> str | None:
    try:
        with tempfile.TemporaryDirectory(prefix="topology-cas-key-") as temp:
            key_path = Path(temp) / "public.pem"
            key_path.write_text(public_key_pem, encoding="utf-8")
            result = subprocess.run(
                ["openssl", "pkey", "-pubin", "-in", str(key_path), "-outform", "DER"],
                check=False,
                capture_output=True,
                timeout=10,
            )
            if (
                result.returncode != 0
                or len(result.stdout) != 44
                or not result.stdout.startswith(ED25519_SPKI_PREFIX)
            ):
                return None
            return "sha256:" + hashlib.sha256(result.stdout).hexdigest()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def verify_provider_topology_cas_attestation(
    attestation: ProviderTopologyCasAttestationV1,
    trust_root: ProviderTopologyCasTrustRootV1,
    identity: PromotionOperationIdentityV1,
    observation: ProviderStackObservationV1,
    pre_submit_sequence: int,
    request: ProviderTopologyCasRequestV1,
    submission: ProviderTopologyCasSubmissionV1,
    execution: ProviderTopologyCasExecutionV1,
    trust_policy: ProviderTopologyCasTrustPolicyV1 | None,
    expected_trust_policy_digest: str | None,
    expected_trust_policy_generation: int | None,
) -> ProviderTopologyCasVerificationV1 | None:
    """Cryptographically verify DSSE PAE with Ed25519 and then enforce exact claims.

    The trust policy digest and generation are supplied as independent pins.
    The policy cannot authenticate itself and is not loaded from the attestation.
    This is not a GitHub-topology-CAS adapter; it only verifies evidence.
    """
    if trust_policy is None or not trust_policy.authorizes(
        trust_root,
        identity,
        expected_trust_policy_digest,
        expected_trust_policy_generation,
    ):
        return None
    if (
        trust_root.generation != identity.trust_root_generation
        or trust_root.repository != identity.repository
        or pre_submit_sequence <= 0
        or attestation.envelope.payload_type != "application/vnd.in-toto+json"
    ):
        return None
    envelope = attestation.envelope
    if envelope.key_id != trust_root.key_id:
        return None
    if len(envelope.payload_base64) > 4 * ((MAX_PROVIDER_TOPOLOGY_ATTESTATION_BYTES + 2) // 3):
        return None
    if len(envelope.signature_base64) > 4 * ((ED25519_SIGNATURE_BYTES + 2) // 3):
        return None
    if _public_key_fingerprint(trust_root.public_key_pem) != trust_root.key_id:
        return None
    try:
        payload = envelope.decoded_payload()
        signature = base64.b64decode(envelope.signature_base64, validate=True)
        if not payload or len(payload) > MAX_PROVIDER_TOPOLOGY_ATTESTATION_BYTES:
            return None
        if len(signature) != ED25519_SIGNATURE_BYTES:
            return None
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_json_object)
        if not isinstance(parsed, dict) or _canonical_json_bytes(parsed) != payload:
            return None
        expected = _provider_topology_statement(
            identity,
            observation,
            pre_submit_sequence,
            request,
            submission,
            execution,
            trust_root,
            trust_policy,
        )
        if parsed != expected:
            return None
        with tempfile.TemporaryDirectory(prefix="topology-cas-verify-") as temp:
            key_path = Path(temp) / "public.pem"
            message_path = Path(temp) / "pae.bin"
            signature_path = Path(temp) / "signature.bin"
            key_path.write_text(trust_root.public_key_pem, encoding="utf-8")
            message_path.write_bytes(envelope.pae())
            signature_path.write_bytes(signature)
            result = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(key_path),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ],
                check=False,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ):
        return None
    return ProviderTopologyCasVerificationV1(
        verifier_source="openssl-ed25519-dsse-verifier",
        verification_method="dsse-v1-pae+in-toto-statement-v1",
        attestation_digest=attestation.digest(),
        verification_result="verified",
        trust_root_id=trust_root.trust_root_id,
        trust_root_generation=trust_root.generation,
        signer_identity=trust_root.signer_identity,
        subject_repository=identity.repository,
        operation_identity_digest=identity.digest(),
        key_id=trust_root.key_id,
        trust_policy_id=trust_policy.policy_id,
        trust_policy_generation=trust_policy.generation,
        trust_policy_digest=trust_policy.digest(),
    )


@dataclass(frozen=True)
class ProviderTopologyCasEvidenceV1:
    submission: ProviderTopologyCasSubmissionV1
    submission_digest: str
    provider_result: ProviderTopologyCasProviderResultV1
    provider_result_digest: str
    evidence_source: str

    def validates(
        self,
        identity: PromotionOperationIdentityV1,
        observation: ProviderStackObservationV1,
        pre_submit_sequence: int,
        trust_root: ProviderTopologyCasTrustRootV1 | None,
        trust_policy: ProviderTopologyCasTrustPolicyV1 | None,
        expected_trust_policy_digest: str | None,
        expected_trust_policy_generation: int | None,
    ) -> bool:
        if trust_root is None or trust_policy is None:
            return False
        if self.evidence_source != "provider-result-capture":
            return False
        if self.submission_digest != self.submission.digest():
            return False
        if self.provider_result_digest != self.provider_result.digest():
            return False
        if pre_submit_sequence <= 0:
            return False
        predicate = ProviderTopologyCasPredicateV1.from_binding(
            identity,
            observation,
            pre_submit_sequence,
        )
        request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
        return (
            self.submission.validates(request)
            and self.provider_result.validates(
                request,
                self.submission,
                predicate,
                identity,
                observation,
                pre_submit_sequence,
                trust_root,
                trust_policy,
                expected_trust_policy_digest,
                expected_trust_policy_generation,
            )
        )


@dataclass(frozen=True)
class ProviderTopologyBindingV1:
    initial_observation: ProviderStackObservationV1 | None
    pre_submit_observation: ProviderStackObservationV1 | None
    initial_sequence: int
    pre_submit_sequence: int | None
    provider_topology_cas_evidence: ProviderTopologyCasEvidenceV1 | None = None
    attestation_trust_root: ProviderTopologyCasTrustRootV1 | None = None
    attestation_trust_policy: ProviderTopologyCasTrustPolicyV1 | None = None
    expected_trust_policy_digest: str | None = None
    expected_trust_policy_generation: int | None = None

    def classify(self, identity: PromotionOperationIdentityV1) -> str:
        if not identity.provider_constraints_valid():
            return "invalid-provider-operation-options"
        if self.initial_sequence <= 0:
            return "invalid-observation-sequence"
        if self.initial_observation is None:
            return "unobserved"
        if not self.initial_observation.matches_reserved(identity):
            return "initial-mismatch"
        if self.pre_submit_observation is None:
            return "unrevalidated"
        if self.pre_submit_sequence is None or self.pre_submit_sequence <= self.initial_sequence:
            return "invalid-observation-order"
        if not self.pre_submit_observation.matches_reserved(identity):
            return "stale-before-submit"
        if self.provider_topology_cas_evidence is None:
            return "observed-not-cas"
        if not self.provider_topology_cas_evidence.validates(
            identity,
            self.pre_submit_observation,
            self.pre_submit_sequence,
            self.attestation_trust_root,
            self.attestation_trust_policy,
            self.expected_trust_policy_digest,
            self.expected_trust_policy_generation,
        ):
            return "observed-not-cas"
        return "provider-topology-cas"


@dataclass
class Reservation:
    reservation_id: str
    operation_id: str
    lease_id: str
    predecessor: str
    state: str = "PromotionReserved"
    dispatch_intent: bool = False
    reservation_head: str = ""
    fencing_token: int = 0
    expected_pr_head_sha: str = "H1"
    trust_root_generation: int = 1
    effect_receipt: PromotionEffectReceipt | None = None


class Ledger:
    def __init__(self):
        self.head = "L0"
        self.active_lease = "LEASE-1"
        self.reservation: Reservation | None = None
        self.fencing_token = 0
        self.trust_root_generation = 1
        self.transitions = []
        self._reservation_n = 0

    def reserve(
        self,
        observed_head: str,
        lease_id: str,
        candidate: str,
        trust_root_generation: int = 1,
        expected_pr_head_sha: str = "H1",
    ) -> bool:
        if observed_head != self.head or lease_id != self.active_lease:
            return False
        if trust_root_generation != self.trust_root_generation:
            return False
        if self.reservation is not None:
            return False
        self._reservation_n += 1
        self.fencing_token += 1
        self.reservation = Reservation(
            f"RES-{self._reservation_n}",
            f"OP-{self._reservation_n}",
            lease_id,
            observed_head,
        )
        self.reservation.reservation_head = candidate
        self.reservation.fencing_token = self.fencing_token
        self.reservation.expected_pr_head_sha = expected_pr_head_sha
        self.reservation.trust_root_generation = trust_root_generation
        self.active_lease = None
        self.head = candidate
        self.transitions.append(("reservation", observed_head, candidate))
        return True

    def invalidate(self, observed_head: str, candidate: str, trust_root_generation: int) -> bool:
        if observed_head != self.head:
            return False
        self.fencing_token += 1
        self.trust_root_generation = trust_root_generation
        self.head = candidate
        self.transitions.append(("invalidation", observed_head, candidate))
        if self.reservation is not None and self.reservation.state == "PromotionReserved":
            self.reservation.state = "PromotionSuperseded"
        return True

    def prepare_dispatch(
        self,
        observed_head: str,
        observed_trust_root_generation: int,
        observed_fencing_token: int,
        attempt_sequence: int,
    ) -> bool:
        del attempt_sequence  # Reserved for durable dispatch-intent sequencing.
        r = self.reservation
        if r is None or r.dispatch_intent or r.state != "PromotionReserved":
            return False
        if observed_head != self.head:
            return False
        if observed_trust_root_generation != self.trust_root_generation:
            return False
        if observed_fencing_token != self.fencing_token:
            return False
        if r.reservation_head != self.head or r.fencing_token != self.fencing_token:
            return False
        if r.trust_root_generation != self.trust_root_generation:
            return False
        r.dispatch_intent = True
        r.state = "PromotionDispatchPrepared"
        return True

    def record_unknown(self, operation_id: str) -> bool:
        r = self.reservation
        if r is None or r.operation_id != operation_id:
            return False
        if not r.dispatch_intent or r.state != "PromotionDispatchPrepared":
            return False
        r.state = "PromotionReconciliationRequired"
        return True

    def reconcile_complete(
        self,
        operation_id: str,
        receipt: PromotionEffectReceipt,
    ) -> bool:
        r = self.reservation
        if r is None or r.operation_id != operation_id:
            return False
        if r.state != "PromotionReconciliationRequired":
            return False
        if (
            receipt.promotion_operation_id != r.operation_id
            or receipt.expected_pr_head_sha != r.expected_pr_head_sha
            or not receipt.observed_merge_commit
        ):
            return False
        self.reservation.effect_receipt = receipt
        self.reservation.state = "PromotionCompleted"
        return True


class ProviderOutcome:
    def __init__(
        self,
        http: int,
        kind: str,
        uuid: str | None = None,
        merge_method: str | None = None,
        merge_action: str | None = None,
        bypass_rules: bool = False,
    ):
        self.http = http
        self.kind = kind
        self.uuid = uuid
        self.merge_method = merge_method
        self.merge_action = merge_action
        self.bypass_rules = bypass_rules


@dataclass(frozen=True)
class EffectReconciliation:
    effect_observed: bool
    observation_source: str
    causal_attribution: str
    observed_merge_commit: str | None = None


class GitHubAsyncModel:
    def __init__(self):
        self.pr_head = "H1"
        self.pending_uuid: str | None = None
        self.pending_merge_method: str | None = None
        self.pending_merge_action: str | None = None
        self.pending_bypass_rules: bool = False
        self.async_status: str | None = None
        self.merge_sha: str | None = None
        self.expired: set[str] = set()
        self.calls = 0

    def submit(
        self,
        expected_head: str,
        merge_method: str = "squash",
        merge_action: str = "direct_merge",
        bypass_rules: bool = False,
        timeout_after_accept: bool = False,
        stack_size: int = 1,
    ) -> ProviderOutcome:
        self.calls += 1
        if stack_size <= 0:
            return ProviderOutcome(400, "invalid-stack-size")
        if bypass_rules and stack_size > 1:
            return ProviderOutcome(400, "bypass-not-supported-for-stack")
        if expected_head != self.pr_head:
            return ProviderOutcome(409, "rejected")
        if self.merge_sha is not None:
            return ProviderOutcome(200, "merged")
        if self.pending_uuid is not None:
            if (
                merge_method == self.pending_merge_method
                and merge_action == self.pending_merge_action
                and bypass_rules == self.pending_bypass_rules
            ):
                return ProviderOutcome(
                    409,
                    "duplicate",
                    self.pending_uuid,
                    self.pending_merge_method,
                    self.pending_merge_action,
                    self.pending_bypass_rules,
                )
            return ProviderOutcome(
                409,
                "duplicate-parameter-mismatch",
                self.pending_uuid,
                self.pending_merge_method,
                self.pending_merge_action,
                self.pending_bypass_rules,
            )
        self.pending_uuid = f"uuid-{self.calls}"
        self.pending_merge_method = merge_method
        self.pending_merge_action = merge_action
        self.pending_bypass_rules = bypass_rules
        self.async_status = "pending"
        if timeout_after_accept:
            return ProviderOutcome(
                599,
                "timeout-after-accept",
                self.pending_uuid,
                merge_method,
                merge_action,
                bypass_rules,
            )
        return ProviderOutcome(
            202,
            "accepted",
            self.pending_uuid,
            merge_method,
            merge_action,
            bypass_rules,
        )

    def get_async_result(self, uuid: str) -> ProviderOutcome:
        if uuid in self.expired:
            return ProviderOutcome(404, "not-found")
        if self.async_status == "enqueued" and self.pending_uuid == uuid:
            return ProviderOutcome(
                200,
                "enqueued",
                uuid,
                self.pending_merge_method,
                self.pending_merge_action,
                self.pending_bypass_rules,
            )
        if self.async_status == "merged" and self.pending_uuid == uuid:
            return ProviderOutcome(
                200,
                "merged",
                uuid,
                self.pending_merge_method,
                self.pending_merge_action,
                self.pending_bypass_rules,
            )
        if self.pending_uuid == uuid:
            return ProviderOutcome(
                200,
                "pending",
                uuid,
                self.pending_merge_method,
                self.pending_merge_action,
                self.pending_bypass_rules,
            )
        return ProviderOutcome(404, "not-found")

    def complete(self) -> None:
        assert self.pending_uuid is not None
        self.merge_sha = "M1"
        # An enqueued merge-queue result is final and remains enqueued;
        # durable PR merged state is the separate reconciliation surface.
        self.async_status = "enqueued"

    def merge_directly(self) -> None:
        assert self.pending_uuid is not None
        self.merge_sha = "M1"
        self.async_status = "merged"

    def reconcile(
        self,
        uuid: str,
        expected_head: str,
        merge_method: str = "squash",
        merge_action: str = "direct_merge",
        bypass_rules: bool = False,
    ) -> EffectReconciliation:
        result = self.get_async_result(uuid)

        if result.kind == "merged":
            if (
                result.uuid == uuid
                and result.merge_method == merge_method
                and result.merge_action == merge_action
                and result.bypass_rules == bypass_rules
                and expected_head == self.pr_head
            ):
                return EffectReconciliation(
                    effect_observed=True,
                    observation_source="provider-operation-result",
                    causal_attribution="established",
                    observed_merge_commit=self.merge_sha,
                )

        if result.kind == "enqueued" and self.merge_sha is not None:
            return EffectReconciliation(
                effect_observed=True,
                observation_source="durable-pr-state",
                causal_attribution="unestablished",
                observed_merge_commit=self.merge_sha,
            )

        if result.kind == "not-found" and self.merge_sha is not None:
            return EffectReconciliation(
                effect_observed=True,
                observation_source="durable-pr-state-after-uuid-expiry",
                causal_attribution="unestablished",
                observed_merge_commit=self.merge_sha,
            )

        return EffectReconciliation(
            effect_observed=False,
            observation_source="no-durable-effect",
            causal_attribution="unestablished",
        )


def legal_interleavings():
    out = []
    events = ("E-read", "E-construct", "E-commit", "I-read", "I-construct", "I-commit")
    for schedule in permutations(events):
        pos = {event: i for i, event in enumerate(schedule)}
        if (
            pos["E-read"] < pos["E-construct"] < pos["E-commit"]
            and pos["I-read"] < pos["I-construct"] < pos["I-commit"]
        ):
            out.append(schedule)
    assert len(out) == 20
    return out


def run_race_schedule(schedule):
    ledger = Ledger()
    snapshots = {}
    candidates = {}
    results = {}

    for event in schedule:
        writer = event[0]
        if event.endswith("-read"):
            snapshots[writer] = ledger.head
        elif event.endswith("-construct"):
            candidates[writer] = f"{writer}-SUCCESSOR"
        else:
            ok = (
                ledger.reserve(snapshots[writer], "LEASE-1", candidates[writer])
                if writer == "E"
                else ledger.invalidate(snapshots[writer], candidates[writer], 2)
            )
            results[event] = ok

    return ledger, results


def stack_identity_fixture() -> PromotionOperationIdentityV1:
    return PromotionOperationIdentityV1(
        repository="Luminous-Dynamics/symthaea",
        provider_stack_number=41,
        requested_pr_number=7087,
        requested_pr_head_sha="H3",
        base_ref="main",
        base_tip_sha="T1",
        ordered_stack=(
            StackEntryV1(7085, "H1", "main", "T1"),
            StackEntryV1(7087, "H3", "stack/7085", "H1"),
        ),
        merge_method="squash",
        merge_action="direct_merge",
        trust_root_generation=7,
        governance_generation=11,
    )


def stack_effect_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
) -> PromotionStackEffectSetV1:
    identity = identity or stack_identity_fixture()
    return PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(
            PromotionStackEffectV1(7085, "H1", "M1"),
            PromotionStackEffectV1(7087, "H3", "M2"),
        ),
    )


def provider_stack_observation_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
) -> ProviderStackObservationV1:
    identity = identity or stack_identity_fixture()
    return ProviderStackObservationV1(
        observation_source="rest-pull-request-stack",
        stack_number=identity.provider_stack_number,
        stack_size=3,
        stack_position=2,
        base_ref=identity.base_ref,
        base_tip_sha=identity.base_tip_sha,
        ordered_stack=(
            identity.ordered_stack[0],
            identity.ordered_stack[1],
            StackEntryV1(7090, "H9", "qual/promotion-stack-identity-v1", "H3"),
        ),
        observation_id="obs-1",
    )




def topology_binding_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    pre_submit_observation: ProviderStackObservationV1 | None = None,
    pre_submit_sequence: int | None = 2,
    provider_topology_cas_evidence: ProviderTopologyCasEvidenceV1 | None = None,
) -> ProviderTopologyBindingV1:
    identity = identity or stack_identity_fixture()
    initial = provider_stack_observation_fixture(identity)
    pre_submit = pre_submit_observation if pre_submit_observation is not None else initial
    return ProviderTopologyBindingV1(
        initial_observation=initial,
        pre_submit_observation=pre_submit,
        initial_sequence=1,
        pre_submit_sequence=pre_submit_sequence,
        provider_topology_cas_evidence=provider_topology_cas_evidence,
        attestation_trust_root=(
            _test_trust_root(identity)
            if provider_topology_cas_evidence is not None
            else None
        ),
        attestation_trust_policy=(
            _test_trust_policy(identity)
            if provider_topology_cas_evidence is not None
            else None
        ),
        expected_trust_policy_digest=(
            _test_trust_policy(identity).digest()
            if provider_topology_cas_evidence is not None
            else None
        ),
        expected_trust_policy_generation=(
            _test_trust_policy(identity).generation
            if provider_topology_cas_evidence is not None
            else None
        ),
    )


def provider_topology_cas_evidence_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    observation: ProviderStackObservationV1 | None = None,
    pre_submit_sequence: int = 2,
    *,
    provider_operation_id: str = "provider-op-1",
    result_source: str = "provider-operation-result",
    predicate_result: str = "accepted",
    execution_source: str = "provider-conditional-execution",
    enforcement_result: str = "enforced",
    evidence_source: str = "provider-result-capture",
) -> ProviderTopologyCasEvidenceV1:
    identity = identity or stack_identity_fixture()
    observation = observation or provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(
        identity,
        observation,
        pre_submit_sequence,
    )
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    submission = ProviderTopologyCasSubmissionV1(
        request_digest=request.digest(),
        provider_operation_id=provider_operation_id,
        submission_source="provider-submission-response",
        submission_result="accepted",
    )
    execution = ProviderTopologyCasExecutionV1(
        execution_source=execution_source,
        provider_operation_id=provider_operation_id,
        request_digest=request.digest(),
        submission_digest=submission.digest(),
        predicate_digest=predicate.digest(),
        enforcement_result=enforcement_result,
    )
    trust_root = _test_trust_root(identity)
    trust_policy = _test_trust_policy(identity, trust_root)
    attestation = _test_sign_topology_statement(
        identity,
        observation,
        pre_submit_sequence,
        request,
        submission,
        execution,
        trust_root,
        trust_policy,
    )
    verification = verify_provider_topology_cas_attestation(
        attestation,
        trust_root,
        identity,
        observation,
        pre_submit_sequence,
        request,
        submission,
        execution,
        trust_policy,
        trust_policy.digest(),
        trust_policy.generation,
    )
    if verification is None:
        raise RuntimeError("test DSSE attestation did not verify")
    provider_result = ProviderTopologyCasProviderResultV1(
        result_source=result_source,
        provider_operation_id=provider_operation_id,
        request_digest=request.digest(),
        submission_digest=submission.digest(),
        predicate_admission=predicate_result,
        execution=execution,
        execution_digest=execution.digest(),
        attestation=attestation,
        attestation_digest=attestation.digest(),
        verification=verification,
        verification_digest=verification.digest(),
    )
    return ProviderTopologyCasEvidenceV1(
        submission=submission,
        submission_digest=submission.digest(),
        provider_result=provider_result,
        provider_result_digest=provider_result.digest(),
        evidence_source=evidence_source,
    )


@lru_cache(maxsize=4)
def _test_signer_material_named(name: str) -> tuple[str, str, str]:
    """Create ephemeral Ed25519 test keys; never use them as production trust roots."""
    with tempfile.TemporaryDirectory(prefix=f"topology-cas-test-key-{name}-") as temp:
        private_path = Path(temp) / "private.pem"
        public_path = Path(temp) / "public.pem"
        generated = subprocess.run(
            ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_path)],
            check=False,
            capture_output=True,
            timeout=10,
        )
        if generated.returncode != 0:
            raise RuntimeError("OpenSSL Ed25519 test-key generation failed")
        published = subprocess.run(
            ["openssl", "pkey", "-in", str(private_path), "-pubout", "-out", str(public_path)],
            check=False,
            capture_output=True,
            timeout=10,
        )
        if published.returncode != 0:
            raise RuntimeError("OpenSSL public-key extraction failed")
        private_pem = private_path.read_text(encoding="utf-8")
        public_pem = public_path.read_text(encoding="utf-8")
        fingerprint = _public_key_fingerprint(public_pem)
        if fingerprint is None:
            raise RuntimeError("OpenSSL public-key fingerprint failed")
        return private_pem, public_pem, fingerprint


def _test_signer_material() -> tuple[str, str, str]:
    return _test_signer_material_named("primary")


def _test_rotated_signer_material() -> tuple[str, str, str]:
    return _test_signer_material_named("rotated")


def _test_trust_root(identity: PromotionOperationIdentityV1) -> ProviderTopologyCasTrustRootV1:
    _, public_pem, key_id = _test_signer_material()
    return ProviderTopologyCasTrustRootV1(
        trust_root_id="test-only-ephemeral-ed25519-root-v1",
        generation=identity.trust_root_generation,
        repository=identity.repository,
        signer_identity=(
            "https://github.com/Luminous-Dynamics/symthaea/"
            ".github/workflows/qual-promotion-reservation-v1.yml@refs/heads/main"
        ),
        key_id=key_id,
        public_key_pem=public_pem,
    )


def _test_trust_policy(
    identity: PromotionOperationIdentityV1,
    trust_root: ProviderTopologyCasTrustRootV1 | None = None,
    *,
    generation: int = 1,
    repository: str | None = None,
    revoked_key_ids: tuple[str, ...] = (),
    revoked_signer_identities: tuple[str, ...] = (),
    authorized_roots: tuple[ProviderTopologyCasTrustRootV1, ...] | None = None,
) -> ProviderTopologyCasTrustPolicyV1:
    root = trust_root or _test_trust_root(identity)
    return ProviderTopologyCasTrustPolicyV1(
        policy_id="test-only-topology-trust-policy-v1",
        generation=generation,
        repository=repository or identity.repository,
        authorized_roots=authorized_roots if authorized_roots is not None else (root,),
        revoked_key_ids=revoked_key_ids,
        revoked_signer_identities=revoked_signer_identities,
    )


def _test_sign_dsse_payload(
    payload: bytes,
    payload_type: str = "application/vnd.in-toto+json",
    *,
    signer_private_pem: str | None = None,
    signer_key_id: str | None = None,
) -> ProviderTopologyCasAttestationV1:
    default_private_pem, _, default_key_id = _test_signer_material()
    private_pem = signer_private_pem or default_private_pem
    key_id = signer_key_id or default_key_id
    unsigned = ProviderTopologyCasDsseEnvelopeV1(
        payload_type=payload_type,
        payload_base64=base64.b64encode(payload).decode("ascii"),
        key_id=key_id,
        signature_base64="",
    )
    with tempfile.TemporaryDirectory(prefix="topology-cas-test-sign-") as temp:
        private_path = Path(temp) / "private.pem"
        pae_path = Path(temp) / "pae.bin"
        signature_path = Path(temp) / "signature.bin"
        private_path.write_text(private_pem, encoding="utf-8")
        pae_path.write_bytes(unsigned.pae())
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private_path),
                "-rawin",
                "-in",
                str(pae_path),
                "-out",
                str(signature_path),
            ],
            check=False,
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("OpenSSL DSSE test signing failed")
        signature_base64 = base64.b64encode(signature_path.read_bytes()).decode("ascii")
    return ProviderTopologyCasAttestationV1(
        envelope=ProviderTopologyCasDsseEnvelopeV1(
            payload_type=unsigned.payload_type,
            payload_base64=unsigned.payload_base64,
            key_id=key_id,
            signature_base64=signature_base64,
        )
    )


def _test_sign_topology_statement(
    identity: PromotionOperationIdentityV1,
    observation: ProviderStackObservationV1,
    pre_submit_sequence: int,
    request: ProviderTopologyCasRequestV1,
    submission: ProviderTopologyCasSubmissionV1,
    execution: ProviderTopologyCasExecutionV1,
    trust_root: ProviderTopologyCasTrustRootV1,
    trust_policy: ProviderTopologyCasTrustPolicyV1,
    signer_private_pem: str | None = None,
) -> ProviderTopologyCasAttestationV1:
    payload = _canonical_json_bytes(
        _provider_topology_statement(
            identity,
            observation,
            pre_submit_sequence,
            request,
            submission,
            execution,
            trust_root,
            trust_policy,
        )
    )
    return _test_sign_dsse_payload(
        payload,
        signer_private_pem=signer_private_pem,
        signer_key_id=trust_root.key_id,
    )


def test_provider_topology_binding_requires_an_initial_observation():
    identity = stack_identity_fixture()
    binding = ProviderTopologyBindingV1(None, None, 1, None)
    assert binding.classify(identity) == "unobserved"


def test_provider_topology_binding_rejects_initial_topology_mismatch():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "base_tip_sha": "T2"}
    )
    binding = ProviderTopologyBindingV1(changed, observation, 1, 2)
    assert binding.classify(identity) == "initial-mismatch"


def test_provider_topology_binding_requires_pre_submit_revalidation():
    identity = stack_identity_fixture()
    initial = provider_stack_observation_fixture(identity)
    binding = ProviderTopologyBindingV1(initial, None, 1, None)
    assert binding.classify(identity) == "unrevalidated"


def test_provider_topology_binding_matching_revalidation_without_cas_is_observed_only():
    identity = stack_identity_fixture()
    binding = topology_binding_fixture(identity)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_binding_detects_stale_pre_submit_topology():
    identity = stack_identity_fixture()
    initial = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**initial.__dict__, "stack_number": 42}
    )
    binding = topology_binding_fixture(identity, changed, 2)
    assert binding.classify(identity) == "stale-before-submit"


def test_provider_topology_binding_detects_invalid_observation_order():
    identity = stack_identity_fixture()
    binding = topology_binding_fixture(identity, pre_submit_sequence=1)
    assert binding.classify(identity) == "invalid-observation-order"


def test_provider_topology_binding_requires_positive_initial_sequence():
    identity = stack_identity_fixture()
    initial = provider_stack_observation_fixture(identity)
    binding = ProviderTopologyBindingV1(initial, initial, 0, 2)
    assert binding.classify(identity) == "invalid-observation-sequence"


def test_provider_topology_binding_requires_cas_evidence_for_strong_class():
    identity = stack_identity_fixture()
    binding = topology_binding_fixture(identity)
    assert binding.classify(identity) == "observed-not-cas"

    admitted_only = provider_topology_cas_evidence_fixture(
        identity,
        enforcement_result="accepted",
    )
    assert topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=admitted_only,
    ).classify(identity) == "observed-not-cas"

    evidence = provider_topology_cas_evidence_fixture(identity)
    bound = topology_binding_fixture(identity, provider_topology_cas_evidence=evidence)
    assert bound.classify(identity) == "provider-topology-cas"


def test_provider_topology_binding_rejects_unbound_cas_evidence():
    identity = stack_identity_fixture()
    other = PromotionOperationIdentityV1(**{**identity.__dict__, "base_tip_sha": "T2"})
    evidence = provider_topology_cas_evidence_fixture(other)
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=evidence)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_binding_rejects_cas_evidence_sequence_drift():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity, pre_submit_sequence=3)
    binding = topology_binding_fixture(
        identity,
        pre_submit_sequence=2,
        provider_topology_cas_evidence=evidence,
    )
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_predicate_digest_binds_observation():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    changed = ProviderStackObservationV1(**{**observation.__dict__, "stack_number": 42})
    changed_predicate = ProviderTopologyCasPredicateV1.from_binding(identity, changed, 2)
    assert predicate.digest() != changed_predicate.digest()


def test_provider_topology_cas_predicate_digest_binds_sequence():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    first = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    second = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 3)
    assert first.digest() != second.digest()


def test_provider_topology_cas_predicate_digest_binds_bypass_rules():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    normal = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    bypass_identity = PromotionOperationIdentityV1(
        **{**identity.__dict__, "bypass_rules": True},
    )
    bypass = ProviderTopologyCasPredicateV1.from_binding(bypass_identity, observation, 2)
    assert normal.digest() != bypass.digest()


def test_provider_topology_cas_request_binds_requested_operation_parameters():
    identity = stack_identity_fixture()
    predicate = ProviderTopologyCasPredicateV1.from_binding(
        identity,
        provider_stack_observation_fixture(identity),
        2,
    )
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    changed = ProviderTopologyCasRequestV1(
        **{**request.__dict__, "expected_head_sha": "H0"},
    )
    assert request.digest() != changed.digest()


def test_provider_topology_cas_provider_result_binds_request_digest():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(
        identity,
        ProviderTopologyCasPredicateV1.from_binding(
            identity,
            provider_stack_observation_fixture(identity),
            2,
        ),
    )
    assert evidence.provider_result.request_digest == request.digest()


def test_provider_topology_cas_submission_digest_binds_operation_id():
    identity = stack_identity_fixture()
    predicate = ProviderTopologyCasPredicateV1.from_binding(
        identity,
        provider_stack_observation_fixture(identity),
        2,
    )
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    first = ProviderTopologyCasSubmissionV1(
        request_digest=request.digest(),
        provider_operation_id="provider-op-1",
        submission_source="provider-submission-response",
        submission_result="accepted",
    )
    second = ProviderTopologyCasSubmissionV1(
        **{**first.__dict__, "provider_operation_id": "provider-op-2"},
    )
    assert first.digest() != second.digest()


def test_provider_topology_cas_attestation_factory_is_deterministic():
    identity = stack_identity_fixture()
    first = provider_topology_cas_evidence_fixture(identity)
    second = provider_topology_cas_evidence_fixture(identity)
    assert first.provider_result.attestation == second.provider_result.attestation


def test_provider_topology_cas_verification_factory_is_deterministic():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.verification is not None
    verified = verify_provider_topology_cas_attestation(
        evidence.provider_result.attestation,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    )
    assert verified == evidence.provider_result.verification


def test_provider_topology_cas_attestation_digest_binds_execution():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    envelope = evidence.provider_result.attestation.envelope
    original_payload = envelope.decoded_payload()
    statement = json.loads(original_payload.decode("utf-8"))
    statement["predicate"]["execution_digest"] = "wrong-execution-digest"
    changed_payload = _canonical_json_bytes(statement)
    changed_envelope = ProviderTopologyCasDsseEnvelopeV1(
        **{
            **envelope.__dict__,
            "payload_base64": base64.b64encode(changed_payload).decode("ascii"),
        }
    )
    changed = ProviderTopologyCasAttestationV1(envelope=changed_envelope)
    assert changed.digest() != evidence.provider_result.attestation.digest()
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert verify_provider_topology_cas_attestation(
        changed,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_verification_binds_attestation():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.verification is not None
    changed = ProviderTopologyCasVerificationV1(
        **{
            **evidence.provider_result.verification.__dict__,
            "attestation_digest": "wrong-attestation-digest",
        }
    )
    assert evidence.provider_result.verification.digest() != changed.digest()


def test_provider_topology_cas_provider_result_rejects_attestation_digest_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    bad = ProviderTopologyCasProviderResultV1(
        **{
            **evidence.provider_result.__dict__,
            "attestation_digest": "wrong-attestation-digest",
        }
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=bad,
        provider_result_digest=bad.digest(),
        evidence_source="provider-result-capture",
    )
    assert topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=spliced,
    ).classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_rejects_verification_digest_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    bad = ProviderTopologyCasProviderResultV1(
        **{
            **evidence.provider_result.__dict__,
            "verification_digest": "wrong-verification-digest",
        }
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=bad,
        provider_result_digest=bad.digest(),
        evidence_source="provider-result-capture",
    )
    assert topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=spliced,
    ).classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_requires_attestation_verification():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    admitted_and_enforced = ProviderTopologyCasProviderResultV1(
        **{
            **evidence.provider_result.__dict__,
            "attestation": None,
            "attestation_digest": None,
            "verification": None,
            "verification_digest": None,
        }
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=admitted_and_enforced,
        provider_result_digest=admitted_and_enforced.digest(),
        evidence_source="provider-result-capture",
    )
    assert topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=spliced,
    ).classify(identity) == "observed-not-cas"


def test_provider_topology_cas_attestation_rejects_wrong_source():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    envelope = evidence.provider_result.attestation.envelope
    changed = ProviderTopologyCasAttestationV1(
        envelope=ProviderTopologyCasDsseEnvelopeV1(
            **{**envelope.__dict__, "payload_type": "application/json"}
        )
    )
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert verify_provider_topology_cas_attestation(
        changed,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_verification_rejects_wrong_source():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.verification is not None
    changed = ProviderTopologyCasVerificationV1(
        **{
            **evidence.provider_result.verification.__dict__,
            "verifier_source": "local-assertion",
        }
    )
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert not changed.validates(
        evidence.provider_result.attestation,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    )


def test_provider_topology_cas_signature_tampering_rejects():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    envelope = evidence.provider_result.attestation.envelope
    signature = bytearray(base64.b64decode(envelope.signature_base64, validate=True))
    signature[0] ^= 0x01
    changed = ProviderTopologyCasAttestationV1(
        envelope=ProviderTopologyCasDsseEnvelopeV1(
            **{
                **envelope.__dict__,
                "signature_base64": base64.b64encode(bytes(signature)).decode("ascii"),
            }
        )
    )
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert verify_provider_topology_cas_attestation(
        changed,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_payload_size_limit_fails_closed():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.execution is not None
    envelope = evidence.provider_result.attestation.envelope
    oversized = ProviderTopologyCasAttestationV1(
        envelope=ProviderTopologyCasDsseEnvelopeV1(
            **{
                **envelope.__dict__,
                "payload_base64": base64.b64encode(
                    b"x" * (MAX_PROVIDER_TOPOLOGY_ATTESTATION_BYTES + 1)
                ).decode("ascii"),
            }
        )
    )
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert verify_provider_topology_cas_attestation(
        oversized,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_signature_length_mismatch_fails_closed():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.execution is not None
    envelope = evidence.provider_result.attestation.envelope
    malformed = ProviderTopologyCasAttestationV1(
        envelope=ProviderTopologyCasDsseEnvelopeV1(
            **{
                **envelope.__dict__,
                "signature_base64": base64.b64encode(b"short").decode("ascii"),
            }
        )
    )
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert verify_provider_topology_cas_attestation(
        malformed,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_trust_root_key_id_mismatch_rejects():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.execution is not None
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    root = ProviderTopologyCasTrustRootV1(
        **{**_test_trust_root(identity).__dict__, "key_id": "sha256:" + ("0" * 64)}
    )
    assert verify_provider_topology_cas_attestation(
        evidence.provider_result.attestation,
        root,
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_missing_trust_root_fails_closed():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    binding = topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=evidence,
    )
    binding = ProviderTopologyBindingV1(
        initial_observation=binding.initial_observation,
        pre_submit_observation=binding.pre_submit_observation,
        initial_sequence=binding.initial_sequence,
        pre_submit_sequence=binding.pre_submit_sequence,
        provider_topology_cas_evidence=evidence,
        attestation_trust_root=None,
    )
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_trust_root_generation_mismatch_rejects():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.execution is not None
    wrong_root = ProviderTopologyCasTrustRootV1(
        **{
            **_test_trust_root(identity).__dict__,
            "generation": identity.trust_root_generation + 1,
        }
    )
    assert verify_provider_topology_cas_attestation(
        evidence.provider_result.attestation,
        wrong_root,
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_trust_root_repository_scope_rejects():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.attestation is not None
    assert evidence.provider_result.execution is not None
    wrong_root = ProviderTopologyCasTrustRootV1(
        **{**_test_trust_root(identity).__dict__, "repository": "other/repository"}
    )
    assert verify_provider_topology_cas_attestation(
        evidence.provider_result.attestation,
        wrong_root,
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_signed_duplicate_json_keys_rejects():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.attestation is not None
    payload = evidence.provider_result.attestation.envelope.decoded_payload()
    needle = b'"statement":"predicate-enforced"'
    assert needle in payload
    duplicated = payload.replace(
        needle,
        b'"statement":"ignored","statement":"predicate-enforced"',
        1,
    )
    assert duplicated != payload
    changed = _test_sign_dsse_payload(duplicated)
    observation = provider_stack_observation_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(identity, observation, 2)
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    assert evidence.provider_result.execution is not None
    assert verify_provider_topology_cas_attestation(
        changed,
        _test_trust_root(identity),
        identity,
        observation,
        2,
        request,
        evidence.submission,
        evidence.provider_result.execution,
        _test_trust_policy(identity),
        _test_trust_policy(identity).digest(),
        _test_trust_policy(identity).generation,
    ) is None


def test_provider_topology_cas_execution_digest_binds_predicate():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.execution is not None
    changed = ProviderTopologyCasExecutionV1(
        **{
            **evidence.provider_result.execution.__dict__,
            "predicate_digest": "wrong-predicate-digest",
        }
    )
    assert evidence.provider_result.execution.digest() != changed.digest()


def test_provider_topology_cas_provider_result_requires_enforcement_witness():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    admitted_only = ProviderTopologyCasProviderResultV1(
        **{
            **evidence.provider_result.__dict__,
            "execution": None,
            "execution_digest": None,
        }
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=admitted_only,
        provider_result_digest=admitted_only.digest(),
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_rejects_execution_digest_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    bad = ProviderTopologyCasProviderResultV1(
        **{**evidence.provider_result.__dict__, "execution_digest": "wrong-execution-digest"},
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=bad,
        provider_result_digest=bad.digest(),
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_rejects_execution_field_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    assert evidence.provider_result.execution is not None
    changed = ProviderTopologyCasExecutionV1(
        **{
            **evidence.provider_result.execution.__dict__,
            "provider_operation_id": "provider-op-2",
        }
    )
    bad = ProviderTopologyCasProviderResultV1(
        **{**evidence.provider_result.__dict__, "execution": changed},
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=bad,
        provider_result_digest=bad.digest(),
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_evidence_rejects_unenforced_execution():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(
        identity,
        enforcement_result="accepted",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=evidence)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_binds_submission():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    predicate = ProviderTopologyCasPredicateV1.from_binding(
        identity,
        provider_stack_observation_fixture(identity),
        2,
    )
    request = ProviderTopologyCasRequestV1.from_identity_predicate(identity, predicate)
    submission = ProviderTopologyCasSubmissionV1(
        request_digest=request.digest(),
        provider_operation_id=evidence.provider_result.provider_operation_id,
        submission_source="provider-submission-response",
        submission_result="accepted",
    )
    assert evidence.provider_result.submission_digest == submission.digest()


def test_provider_topology_cas_evidence_rejects_submission_digest_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    other = ProviderTopologyCasSubmissionV1(
        request_digest=evidence.submission.request_digest,
        provider_operation_id="provider-op-other",
        submission_source="provider-submission-response",
        submission_result="accepted",
    )
    spliced_result = ProviderTopologyCasProviderResultV1(
        **{
            **evidence.provider_result.__dict__,
            "submission_digest": other.digest(),
        }
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=evidence.submission,
        submission_digest=evidence.submission_digest,
        provider_result=spliced_result,
        provider_result_digest=spliced_result.digest(),
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_evidence_rejects_submission_field_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    changed_submission = ProviderTopologyCasSubmissionV1(
        **{**evidence.submission.__dict__, "submission_result": "accepted-with-warning"},
    )
    spliced = ProviderTopologyCasEvidenceV1(
        submission=changed_submission,
        submission_digest=evidence.submission_digest,
        provider_result=evidence.provider_result,
        provider_result_digest=evidence.provider_result_digest,
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_provider_result_digest_binds_operation_id():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    changed = ProviderTopologyCasProviderResultV1(
        **{**evidence.provider_result.__dict__, "provider_operation_id": "provider-op-2"},
    )
    assert evidence.provider_result.digest() != changed.digest()


def test_provider_topology_cas_evidence_rejects_provider_result_digest_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    other = provider_topology_cas_evidence_fixture(identity, provider_operation_id="provider-op-2")
    spliced = ProviderTopologyCasEvidenceV1(
        provider_result=evidence.provider_result,
        provider_result_digest=other.provider_result_digest,
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_cas_evidence_rejects_result_field_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    changed = ProviderTopologyCasProviderResultV1(
        **{**evidence.provider_result.__dict__, "predicate_result": "observed-only"},
    )
    spliced = ProviderTopologyCasEvidenceV1(
        provider_result=changed,
        provider_result_digest=evidence.provider_result_digest,
        evidence_source="provider-result-capture",
    )
    binding = topology_binding_fixture(identity, provider_topology_cas_evidence=spliced)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_binding_rejects_bypass_mode_splice():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(identity)
    bypass_identity = PromotionOperationIdentityV1(
        **{**identity.__dict__, "bypass_rules": True},
    )
    binding = topology_binding_fixture(
        bypass_identity,
        provider_topology_cas_evidence=evidence,
    )
    assert binding.classify(bypass_identity) == "invalid-provider-operation-options"


def test_provider_topology_binding_rejects_non_provider_cas_evidence_source():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(
        identity,
        result_source="local-receipt",
    )
    binding = topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=evidence,
    )
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_binding_rejects_empty_provider_operation_id():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(
        identity,
        provider_operation_id="",
    )
    binding = topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=evidence,
    )
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_topology_binding_rejects_unaccepted_cas_predicate_result():
    identity = stack_identity_fixture()
    evidence = provider_topology_cas_evidence_fixture(
        identity,
        predicate_result="observed-only",
    )
    binding = topology_binding_fixture(
        identity,
        provider_topology_cas_evidence=evidence,
    )
    assert binding.classify(identity) == "observed-not-cas"


def test_matching_revalidation_does_not_claim_post_submit_freshness():
    identity = stack_identity_fixture()
    binding = topology_binding_fixture(identity)
    assert binding.classify(identity) == "observed-not-cas"


def test_provider_stack_observation_exact_selected_prefix_matches():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    assert observation.matches_reserved(identity)


def test_provider_stack_observation_missing_value_fails_closed():
    identity = stack_identity_fixture()
    assert not provider_stack_observation_matches_reserved(None, identity)


def test_provider_stack_observation_rejects_stack_number_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_number": 42}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_base_ref_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "base_ref": "release"}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_base_tip_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "base_tip_sha": "T2"}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_position_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_position": 3}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_lower_head_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed_stack = (
        StackEntryV1(7085, "H1-CHANGED", "main", "T1"),
        *observation.ordered_stack[1:],
    )
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "ordered_stack": changed_stack}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_inserted_lower_entry():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    inserted = (
        observation.ordered_stack[0],
        StackEntryV1(7086, "H2", "main", "T1"),
        observation.ordered_stack[1],
        observation.ordered_stack[2],
    )
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_size": 4, "stack_position": 3, "ordered_stack": inserted}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_removed_lower_entry():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    shortened = (observation.ordered_stack[1], observation.ordered_stack[2])
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_size": 2, "stack_position": 1, "ordered_stack": shortened}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_requested_head_drift():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed_stack = (
        observation.ordered_stack[0],
        StackEntryV1(7087, "H3-CHANGED", "stack/7085", "H1"),
        observation.ordered_stack[2],
    )
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "ordered_stack": changed_stack}
    )
    assert not changed.matches_reserved(identity)


def test_provider_stack_observation_allows_unrelated_upper_stack_growth():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    grown = (
        *observation.ordered_stack,
        StackEntryV1(7091, "H10", "stack/7087", "H3"),
    )
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_size": 4, "ordered_stack": grown}
    )
    assert changed.matches_reserved(identity)


def test_provider_stack_observation_rejects_malformed_size_or_position():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    bad_size = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_size": 99}
    )
    bad_position = ProviderStackObservationV1(
        **{**observation.__dict__, "stack_position": 0}
    )
    assert not bad_size.matches_reserved(identity)
    assert not bad_position.matches_reserved(identity)


def test_provider_stack_observation_allows_missing_optional_observation_id():
    identity = stack_identity_fixture()
    observation = ProviderStackObservationV1(
        **{**provider_stack_observation_fixture(identity).__dict__, "observation_id": None}
    )
    assert observation.matches_reserved(identity)


def test_stack_effect_set_exact_match_is_complete():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    assert effects.validates_complete(identity)


def test_stack_effect_set_missing_bottom_entry_is_incomplete():
    identity = stack_identity_fixture()
    effects = PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(PromotionStackEffectV1(7087, "H3", "M2"),),
    )
    assert not effects.validates_complete(identity)


def test_stack_effect_set_missing_requested_entry_is_incomplete():
    identity = stack_identity_fixture()
    effects = PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(PromotionStackEffectV1(7085, "H1", "M1"),),
    )
    assert not effects.validates_complete(identity)


def test_stack_effect_set_rejects_extra_entry():
    identity = stack_identity_fixture()
    effects = PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(
            PromotionStackEffectV1(7085, "H1", "M1"),
            PromotionStackEffectV1(7087, "H3", "M2"),
            PromotionStackEffectV1(7090, "H9", "M3"),
        ),
    )
    assert not effects.validates_complete(identity)


def test_stack_effect_set_rejects_lower_stack_head_mismatch():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    changed = (
        PromotionStackEffectV1(7085, "H1-CHANGED", "M1"),
        effects.effects[1],
    )
    assert not PromotionStackEffectSetV1(identity.digest(), changed).validates_complete(identity)


def test_stack_effect_set_rejects_requested_head_mismatch():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    changed = (
        effects.effects[0],
        PromotionStackEffectV1(7087, "H3-CHANGED", "M2"),
    )
    assert not PromotionStackEffectSetV1(identity.digest(), changed).validates_complete(identity)


def test_stack_effect_set_rejects_empty_merge_commit():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    changed = (
        effects.effects[0],
        PromotionStackEffectV1(7087, "H3", ""),
    )
    assert not PromotionStackEffectSetV1(identity.digest(), changed).validates_complete(identity)


def test_stack_effect_set_rejects_reordered_effects():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    reversed_effects = tuple(reversed(effects.effects))
    assert not PromotionStackEffectSetV1(
        identity.digest(), reversed_effects
    ).validates_complete(identity)


def test_stack_effect_set_rejects_duplicate_pr():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    duplicated = (
        effects.effects[0],
        PromotionStackEffectV1(7085, "H1", "M2"),
    )
    assert not PromotionStackEffectSetV1(identity.digest(), duplicated).validates_complete(identity)


def test_stack_effect_set_rejects_operation_identity_mismatch():
    identity = stack_identity_fixture()
    effects = stack_effect_fixture(identity)
    other_identity = PromotionOperationIdentityV1(
        **{**identity.__dict__, "base_tip_sha": "T2"},
    )
    assert not PromotionStackEffectSetV1(
        other_identity.digest(), effects.effects
    ).validates_complete(identity)


def test_stack_identity_is_deterministic():
    a = stack_identity_fixture()
    b = stack_identity_fixture()
    assert a.canonical_bytes() == b.canonical_bytes()
    assert a.digest() == b.digest()


def test_stack_identity_binds_ordered_stack_topology():
    original = stack_identity_fixture()
    reversed_stack = PromotionOperationIdentityV1(
        **{
            **original.__dict__,
            "ordered_stack": tuple(reversed(original.ordered_stack)),
        }
    )
    assert original.digest() != reversed_stack.digest()


def test_stack_identity_binds_lower_stack_head_sha():
    original = stack_identity_fixture()
    changed = PromotionOperationIdentityV1(
        **{
            **original.__dict__,
            "ordered_stack": (
                StackEntryV1(7085, "H1-CHANGED", "main", "T1"),
                original.ordered_stack[1],
            ),
        }
    )
    assert original.digest() != changed.digest()


def test_stack_identity_binds_base_tip():
    original = stack_identity_fixture()
    changed = PromotionOperationIdentityV1(
        **{**original.__dict__, "base_tip_sha": "T2"},
    )
    assert original.digest() != changed.digest()


def test_stack_identity_binds_merge_parameters():
    original = stack_identity_fixture()
    changed_method = PromotionOperationIdentityV1(
        **{**original.__dict__, "merge_method": "merge"},
    )
    changed_action = PromotionOperationIdentityV1(
        **{**original.__dict__, "merge_action": "default"},
    )
    assert original.digest() != changed_method.digest()
    assert original.digest() != changed_action.digest()


def test_stack_identity_binds_authority_generations():
    original = stack_identity_fixture()
    changed_root = PromotionOperationIdentityV1(
        **{**original.__dict__, "trust_root_generation": 8},
    )
    changed_governance = PromotionOperationIdentityV1(
        **{**original.__dict__, "governance_generation": 12},
    )
    assert original.digest() != changed_root.digest()
    assert original.digest() != changed_governance.digest()


def test_stack_identity_binds_provider_stack_number():
    original = stack_identity_fixture()
    changed = PromotionOperationIdentityV1(
        **{**original.__dict__, "provider_stack_number": 42},
    )
    assert original.digest() != changed.digest()


def test_stack_identity_binds_bypass_rules():
    original = stack_identity_fixture()
    changed = PromotionOperationIdentityV1(
        **{**original.__dict__, "bypass_rules": True},
    )
    assert original.digest() != changed.digest()


def test_stack_identity_rejects_bypass_for_multi_pr_stack():
    identity = PromotionOperationIdentityV1(
        **{**stack_identity_fixture().__dict__, "bypass_rules": True},
    )
    assert not identity.provider_constraints_valid()


def test_stack_identity_binds_requested_subject_identity():
    original = stack_identity_fixture()
    changed_pr = PromotionOperationIdentityV1(
        **{**original.__dict__, "requested_pr_number": 7090},
    )
    changed_head = PromotionOperationIdentityV1(
        **{**original.__dict__, "requested_pr_head_sha": "H4"},
    )
    assert original.digest() != changed_pr.digest()
    assert original.digest() != changed_head.digest()


def test_exact_20_schedules_are_executed():
    sequential = 0
    concurrent = 0
    for schedule in legal_interleavings():
        ledger, results = run_race_schedule(schedule)
        winners = [event for event, ok in results.items() if ok]
        assert len(winners) in (1, 2)
        if len(winners) == 2:
            sequential += 1
            assert (
                schedule.index("E-commit") < schedule.index("I-read")
                or schedule.index("I-commit") < schedule.index("E-read")
            )
        else:
            concurrent += 1
        assert ledger.head in {"E-SUCCESSOR", "I-SUCCESSOR"}
    assert sequential == 2
    assert concurrent == 18


def test_single_use_reservation():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert not ledger.reserve("L1", "LEASE-1", "L2")
    assert not ledger.reserve("L0", "LEASE-1", "L3")
    assert ledger.active_lease is None


def test_dispatch_intent_fences_crash():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1, 1)
    assert ledger.reservation.dispatch_intent
    assert ledger.reservation.state == "PromotionDispatchPrepared"
    ledger.record_unknown(ledger.reservation.operation_id)
    assert ledger.reservation.state == "PromotionReconciliationRequired"
    assert not ledger.prepare_dispatch("L1", 1, 1, 2)


def test_timeout_after_acceptance_is_unknown():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1, 1)
    outcome = provider.submit("H1", timeout_after_accept=True)
    assert outcome.kind == "timeout-after-accept"
    assert ledger.record_unknown(ledger.reservation.operation_id)
    assert provider.pending_uuid is not None
    assert ledger.reservation.state == "PromotionReconciliationRequired"


def test_duplicate_async_request_reuses_provider_handle():
    provider = GitHubAsyncModel()
    first = provider.submit("H1", "squash", "direct_merge")
    second = provider.submit("H1", "squash", "direct_merge")
    assert first.http == 202 and first.uuid
    assert second.http == 409 and second.kind == "duplicate"
    assert second.uuid == first.uuid
    assert second.merge_method == "squash"
    assert second.merge_action == "direct_merge"


def test_duplicate_async_request_option_mismatch_is_not_idempotent():
    provider = GitHubAsyncModel()
    first = provider.submit("H1", "squash", "direct_merge")
    second = provider.submit("H1", "merge", "default")
    assert first.uuid is not None
    assert second.http == 409
    assert second.kind == "duplicate-parameter-mismatch"
    assert second.uuid == first.uuid
    assert second.merge_method == "squash"
    assert second.merge_action == "direct_merge"


def test_duplicate_async_request_bypass_rules_mismatch_is_not_idempotent():
    provider = GitHubAsyncModel()
    first = provider.submit("H1", "squash", "direct_merge", False)
    second = provider.submit("H1", "squash", "direct_merge", True)
    assert first.uuid is not None
    assert second.http == 409
    assert second.kind == "duplicate-parameter-mismatch"
    assert second.uuid == first.uuid
    assert second.bypass_rules is False


def test_provider_model_rejects_bypass_for_multi_pr_stack():
    provider = GitHubAsyncModel()
    result = provider.submit(
        "H1",
        "squash",
        "direct_merge",
        True,
        stack_size=2,
    )
    assert result.http == 400
    assert result.kind == "bypass-not-supported-for-stack"
    assert provider.pending_uuid is None


def test_async_provider_result_preserves_bypass_rules_for_reconciliation():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1", "squash", "direct_merge", True)
    assert accepted.uuid is not None
    assert accepted.bypass_rules is True
    pending = provider.get_async_result(accepted.uuid)
    assert pending.bypass_rules is True
    provider.merge_directly()
    merged = provider.get_async_result(accepted.uuid)
    assert merged.bypass_rules is True
    reconciliation = provider.reconcile(
        accepted.uuid,
        "H1",
        "squash",
        "direct_merge",
        True,
    )
    assert reconciliation.effect_observed
    assert reconciliation.causal_attribution == "established"


def test_async_provider_result_bypass_rules_mismatch_is_not_causal():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1", "squash", "direct_merge", True)
    assert accepted.uuid is not None
    provider.merge_directly()
    reconciliation = provider.reconcile(
        accepted.uuid,
        "H1",
        "squash",
        "direct_merge",
        False,
    )
    assert not reconciliation.effect_observed
    assert reconciliation.causal_attribution == "unestablished"


def test_direct_provider_merge_establishes_causal_attribution():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1", "squash", "direct_merge")
    assert accepted.uuid is not None
    provider.merge_directly()
    reconciliation = provider.reconcile(
        accepted.uuid,
        "H1",
        "squash",
        "direct_merge",
    )
    assert reconciliation.effect_observed
    assert reconciliation.observation_source == "provider-operation-result"
    assert reconciliation.causal_attribution == "established"
    assert reconciliation.observed_merge_commit == "M1"


def test_enqueued_then_durable_merge_observes_effect_without_causality():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1", "squash", "direct_merge")
    assert accepted.uuid is not None
    queued = provider.get_async_result(accepted.uuid)
    assert queued.kind == "pending"
    provider.async_status = "enqueued"
    provider.complete()

    reconciliation = provider.reconcile(
        accepted.uuid,
        "H1",
        "squash",
        "direct_merge",
    )
    assert reconciliation.effect_observed
    assert reconciliation.observation_source == "durable-pr-state"
    assert reconciliation.causal_attribution == "unestablished"
    assert reconciliation.observed_merge_commit == "M1"


def test_expired_uuid_then_durable_merge_observes_effect_without_causality():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1")
    assert accepted.uuid is not None
    provider.complete()
    provider.expired.add(accepted.uuid)

    reconciliation = provider.reconcile(accepted.uuid, "H1")
    assert reconciliation.effect_observed
    assert reconciliation.observation_source == "durable-pr-state-after-uuid-expiry"
    assert reconciliation.causal_attribution == "unestablished"


def test_already_merged_retry_observes_effect_without_causality():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1")
    assert accepted.uuid is not None
    provider.merge_directly()

    retry = provider.submit("H1")
    assert retry.kind == "merged"
    assert retry.uuid is None

    reconciliation = EffectReconciliation(
        effect_observed=True,
        observation_source="already-merged-pr-state",
        causal_attribution="unestablished",
        observed_merge_commit=provider.merge_sha,
    )
    assert reconciliation.effect_observed
    assert reconciliation.causal_attribution == "unestablished"


def test_local_operation_id_does_not_mint_causal_attribution():
    receipt = PromotionEffectReceipt("OP-1", "H1", "M1")
    assert receipt.promotion_operation_id == "OP-1"
    assert not hasattr(receipt, "causal_attribution")


def test_enqueued_is_not_completion():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1")
    assert accepted.uuid is not None
    provider.async_status = "enqueued"
    queued_before = provider.get_async_result(accepted.uuid)
    assert queued_before.http == 200 and queued_before.kind == "enqueued"
    assert provider.merge_sha is None

    provider.complete()
    queued_after = provider.get_async_result(accepted.uuid)
    assert queued_after.http == 200 and queued_after.kind == "enqueued"
    assert provider.merge_sha == "M1"


def test_already_merged_is_durable_completion():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1")
    provider.complete()
    outcome = provider.submit("H1")
    assert accepted.uuid is not None
    assert outcome.http == 200 and outcome.kind == "merged"
    assert outcome.uuid is None
    assert provider.merge_sha == "M1"


def test_expired_uuid_with_merged_pr_uses_pr_state():
    provider = GitHubAsyncModel()
    first = provider.submit("H1")
    provider.complete()
    assert first.uuid is not None
    provider.expired.add(first.uuid)
    lookup = provider.get_async_result(first.uuid)
    assert lookup.http == 404
    assert provider.merge_sha == "M1"


def test_expired_uuid_without_effect_stays_unknown():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1, 1)
    first = provider.submit("H1")
    assert first.uuid is not None
    provider.expired.add(first.uuid)
    assert ledger.record_unknown(ledger.reservation.operation_id)
    lookup = provider.get_async_result(first.uuid)
    assert lookup.http == 404
    assert provider.merge_sha is None
    assert ledger.reservation.state == "PromotionReconciliationRequired"


def test_exact_subject_mismatch_rejects():
    provider = GitHubAsyncModel()
    outcome = provider.submit("H-old")
    assert outcome.http == 409 and outcome.kind == "rejected"
    assert provider.merge_sha is None


def test_root_change_before_dispatch_blocks_effect():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.invalidate("L1", "I1", 2)
    assert ledger.reservation.state == "PromotionSuperseded"
    assert not ledger.prepare_dispatch("L1", 1, 1, 1)
    assert provider.calls == 0


def test_root_change_after_dispatch_is_not_retroactive():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1, 1)
    outcome = provider.submit("H1")
    assert outcome.http == 202
    ledger.record_unknown(ledger.reservation.operation_id)
    assert ledger.reservation.state == "PromotionReconciliationRequired"
    assert provider.pending_uuid is not None


def test_stale_coordinator_cannot_reserve_after_new_head():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert not ledger.reserve("L0", "LEASE-1", "L-stale")
    assert ledger.head == "L1"


def test_stale_trust_root_rejects_reservation():
    ledger = Ledger()
    assert not ledger.reserve("L0", "LEASE-1", "L1", 2)
    assert ledger.head == "L0"


def test_completion_requires_effect_receipt():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1, 1)
    assert ledger.record_unknown(ledger.reservation.operation_id)
    op = ledger.reservation.operation_id
    assert not ledger.reconcile_complete(
        "wrong", PromotionEffectReceipt("wrong", "H1", "M1")
    )
    assert ledger.reconcile_complete(op, PromotionEffectReceipt(op, "H1", "M1"))
    assert ledger.reservation.state == "PromotionCompleted"


def test_unrelated_ledger_transition_rejects_stale_dispatch_fence():
    ledger = Ledger()
    ledger.reserve("L0", "LEASE-1", "L1", 1)
    ledger.fencing_token += 1
    ledger.head = "L2"
    assert not ledger.prepare_dispatch("L1", 1, 1, 1)


TESTS = [
    test_provider_topology_binding_requires_an_initial_observation,
    test_provider_topology_binding_rejects_initial_topology_mismatch,
    test_provider_topology_binding_requires_pre_submit_revalidation,
    test_provider_topology_binding_matching_revalidation_without_cas_is_observed_only,
    test_provider_topology_binding_detects_stale_pre_submit_topology,
    test_provider_topology_binding_detects_invalid_observation_order,
    test_provider_topology_binding_requires_positive_initial_sequence,
    test_provider_topology_binding_requires_cas_evidence_for_strong_class,
    test_provider_topology_binding_rejects_unbound_cas_evidence,
    test_provider_topology_binding_rejects_cas_evidence_sequence_drift,
    test_provider_topology_cas_predicate_digest_binds_observation,
    test_provider_topology_cas_predicate_digest_binds_sequence,
    test_provider_topology_cas_predicate_digest_binds_bypass_rules,
    test_provider_topology_cas_request_binds_requested_operation_parameters,
    test_provider_topology_cas_provider_result_binds_request_digest,
    test_provider_topology_cas_submission_digest_binds_operation_id,
    test_provider_topology_cas_attestation_factory_is_deterministic,
    test_provider_topology_cas_verification_factory_is_deterministic,
    test_provider_topology_cas_attestation_digest_binds_execution,
    test_provider_topology_cas_verification_binds_attestation,
    test_provider_topology_cas_provider_result_rejects_attestation_digest_splice,
    test_provider_topology_cas_provider_result_rejects_verification_digest_splice,
    test_provider_topology_cas_provider_result_requires_attestation_verification,
    test_provider_topology_cas_attestation_rejects_wrong_source,
    test_provider_topology_cas_verification_rejects_wrong_source,
    test_provider_topology_cas_signature_tampering_rejects,
    test_provider_topology_cas_payload_size_limit_fails_closed,
    test_provider_topology_cas_signature_length_mismatch_fails_closed,
    test_provider_topology_cas_trust_root_key_id_mismatch_rejects,
    test_provider_topology_cas_missing_trust_root_fails_closed,
    test_provider_topology_cas_trust_root_generation_mismatch_rejects,
    test_provider_topology_cas_trust_root_repository_scope_rejects,
    test_provider_topology_cas_signed_duplicate_json_keys_rejects,
    test_provider_topology_cas_execution_digest_binds_predicate,
    test_provider_topology_cas_provider_result_requires_enforcement_witness,
    test_provider_topology_cas_provider_result_rejects_execution_digest_splice,
    test_provider_topology_cas_provider_result_rejects_execution_field_splice,
    test_provider_topology_cas_evidence_rejects_unenforced_execution,
    test_provider_topology_cas_provider_result_binds_submission,
    test_provider_topology_cas_evidence_rejects_submission_digest_splice,
    test_provider_topology_cas_evidence_rejects_submission_field_splice,
    test_provider_topology_cas_provider_result_digest_binds_operation_id,
    test_provider_topology_cas_evidence_rejects_provider_result_digest_splice,
    test_provider_topology_cas_evidence_rejects_result_field_splice,
    test_provider_topology_binding_rejects_bypass_mode_splice,
    test_provider_topology_binding_rejects_non_provider_cas_evidence_source,
    test_provider_topology_binding_rejects_empty_provider_operation_id,
    test_provider_topology_binding_rejects_unaccepted_cas_predicate_result,
    test_matching_revalidation_does_not_claim_post_submit_freshness,
    test_provider_stack_observation_exact_selected_prefix_matches,
    test_provider_stack_observation_missing_value_fails_closed,
    test_provider_stack_observation_rejects_stack_number_drift,
    test_provider_stack_observation_rejects_base_ref_drift,
    test_provider_stack_observation_rejects_base_tip_drift,
    test_provider_stack_observation_rejects_position_drift,
    test_provider_stack_observation_rejects_lower_head_drift,
    test_provider_stack_observation_rejects_inserted_lower_entry,
    test_provider_stack_observation_rejects_removed_lower_entry,
    test_provider_stack_observation_rejects_requested_head_drift,
    test_provider_stack_observation_allows_unrelated_upper_stack_growth,
    test_provider_stack_observation_rejects_malformed_size_or_position,
    test_provider_stack_observation_allows_missing_optional_observation_id,
    test_stack_effect_set_exact_match_is_complete,
    test_stack_effect_set_missing_bottom_entry_is_incomplete,
    test_stack_effect_set_missing_requested_entry_is_incomplete,
    test_stack_effect_set_rejects_extra_entry,
    test_stack_effect_set_rejects_lower_stack_head_mismatch,
    test_stack_effect_set_rejects_requested_head_mismatch,
    test_stack_effect_set_rejects_empty_merge_commit,
    test_stack_effect_set_rejects_reordered_effects,
    test_stack_effect_set_rejects_duplicate_pr,
    test_stack_effect_set_rejects_operation_identity_mismatch,
    test_stack_identity_is_deterministic,
    test_stack_identity_binds_ordered_stack_topology,
    test_stack_identity_binds_lower_stack_head_sha,
    test_stack_identity_binds_base_tip,
    test_stack_identity_binds_merge_parameters,
    test_stack_identity_binds_authority_generations,
    test_stack_identity_binds_provider_stack_number,
    test_stack_identity_binds_bypass_rules,
    test_stack_identity_rejects_bypass_for_multi_pr_stack,
    test_stack_identity_binds_requested_subject_identity,
    test_exact_20_schedules_are_executed,
    test_single_use_reservation,
    test_dispatch_intent_fences_crash,
    test_timeout_after_acceptance_is_unknown,
    test_duplicate_async_request_reuses_provider_handle,
    test_duplicate_async_request_option_mismatch_is_not_idempotent,
    test_duplicate_async_request_bypass_rules_mismatch_is_not_idempotent,
    test_provider_model_rejects_bypass_for_multi_pr_stack,
    test_async_provider_result_preserves_bypass_rules_for_reconciliation,
    test_async_provider_result_bypass_rules_mismatch_is_not_causal,
    test_direct_provider_merge_establishes_causal_attribution,
    test_enqueued_then_durable_merge_observes_effect_without_causality,
    test_expired_uuid_then_durable_merge_observes_effect_without_causality,
    test_already_merged_retry_observes_effect_without_causality,
    test_local_operation_id_does_not_mint_causal_attribution,
    test_enqueued_is_not_completion,
    test_already_merged_is_durable_completion,
    test_expired_uuid_with_merged_pr_uses_pr_state,
    test_expired_uuid_without_effect_stays_unknown,
    test_exact_subject_mismatch_rejects,
    test_root_change_before_dispatch_blocks_effect,
    test_root_change_after_dispatch_is_not_retroactive,
    test_stale_coordinator_cannot_reserve_after_new_head,
    test_stale_trust_root_rejects_reservation,
    test_completion_requires_effect_receipt,
    test_unrelated_ledger_transition_rejects_stale_dispatch_fence,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print("PASS", test.__name__)
    print("PromotionReservationV1 model: PASS")
    print("claim_ceiling=deterministic local transaction/reconciliation model only")
    print("promotion_authority=false")
