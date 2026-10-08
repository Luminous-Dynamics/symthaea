#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from itertools import permutations


_UNSET = object()


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

    def canonical_bytes(self) -> bytes:
        payload = {
            "base_ref": self.base_ref,
            "provider_stack_number": self.provider_stack_number,
            "base_tip_sha": self.base_tip_sha,
            "governance_generation": self.governance_generation,
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


def provider_stack_observation_matches_reserved(
    observation: ProviderStackObservationV1 | None,
    identity: PromotionOperationIdentityV1,
) -> bool:
    return observation is not None and observation.matches_reserved(identity)


@dataclass(frozen=True)
class ProviderTopologyBindingV1:
    initial_observation: ProviderStackObservationV1 | None
    pre_submit_observation: ProviderStackObservationV1 | None
    initial_sequence: int
    pre_submit_sequence: int | None
    provider_topology_cas: bool = False

    def classify(self, identity: PromotionOperationIdentityV1) -> str:
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
        if self.provider_topology_cas:
            return "provider-topology-cas"
        return "observed-not-cas"


@dataclass(frozen=True)
class ProviderCaptureIntegrityV1:
    raw_bytes_digest: str
    storage_id: str
    capture_sequence: int
    durable: bool = True

    @classmethod
    def capture(
        cls,
        raw_bytes: bytes,
        storage_id: str,
        capture_sequence: int,
        durable: bool = True,
    ) -> "ProviderCaptureIntegrityV1":
        return cls(
            raw_bytes_digest=hashlib.sha256(raw_bytes).hexdigest(),
            storage_id=storage_id,
            capture_sequence=capture_sequence,
            durable=durable,
        )

    def is_valid(self) -> bool:
        return (
            bool(self.raw_bytes_digest)
            and bool(self.storage_id)
            and self.capture_sequence > 0
            and self.durable
        )


@dataclass(frozen=True)
class ProviderSourceAuthenticationV1:
    method: str
    verified: bool
    provider_identity: str | None = None

    def is_accepted_transport_auth(self) -> bool:
        return (
            self.verified
            and bool(self.provider_identity)
            and self.method in {
                "authenticated-api-channel",
                "webhook-hmac-verified",
            }
        )


@dataclass(frozen=True)
class ProviderAttestationV1:
    scheme: str | None = None
    verified: bool = False

    def is_verified(self) -> bool:
        return self.verified and bool(self.scheme)


@dataclass(frozen=True)
class ProviderEvidenceEnvelopeV1:
    capture: ProviderCaptureIntegrityV1
    source_authentication: ProviderSourceAuthenticationV1
    provider_attestation: ProviderAttestationV1 = ProviderAttestationV1()

    def is_preserved_provider_evidence(self) -> bool:
        return (
            self.capture.is_valid()
            and self.source_authentication.is_accepted_transport_auth()
        )

    def has_provider_attestation(self) -> bool:
        return self.provider_attestation.is_verified()


@dataclass(frozen=True)
class ProviderWebhookReceiptV1:
    delivery_id: str
    payload_bytes_digest: str
    signature: str
    signature_algorithm: str = "HMAC-SHA256"

    @classmethod
    def from_delivery(
        cls,
        delivery_id: str,
        payload: bytes,
        secret: bytes,
    ) -> "ProviderWebhookReceiptV1":
        digest = hashlib.sha256(payload).hexdigest()
        mac = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return cls(
            delivery_id=delivery_id,
            payload_bytes_digest=digest,
            signature=f"sha256={mac}",
        )

    def verify(self, payload: bytes, secret: bytes) -> bool:
        expected = (
            "sha256="
            + hmac.new(secret, payload, hashlib.sha256).hexdigest()
        )
        return (
            self.signature_algorithm == "HMAC-SHA256"
            and bool(self.delivery_id)
            and hmac.compare_digest(self.signature, expected)
            and self.payload_bytes_digest == hashlib.sha256(payload).hexdigest()
        )


@dataclass(frozen=True)
class ProviderDeliveryRegistryV1:
    deliveries: tuple[tuple[str, str], ...] = ()

    def observe(self, receipt: ProviderWebhookReceiptV1) -> str:
        for delivery_id, payload_digest in self.deliveries:
            if delivery_id != receipt.delivery_id:
                continue
            if payload_digest == receipt.payload_bytes_digest:
                return "duplicate-identical"
            return "delivery-id-conflict"
        return "new-delivery"

    def record(self, receipt: ProviderWebhookReceiptV1) -> "ProviderDeliveryRegistryV1":
        classification = self.observe(receipt)
        if classification == "new-delivery":
            return ProviderDeliveryRegistryV1(
                self.deliveries + ((receipt.delivery_id, receipt.payload_bytes_digest),)
            )
        return self


@dataclass(frozen=True)
class ProviderMergeResultV1:
    result_source: str
    status: str
    provider_uuid: str | None
    requested_pr_number: int
    expected_head_sha: str
    merge_method: str
    merge_action: str
    observed_merge_commit: str | None = None

    def directly_binds_requested_effect(
        self,
        identity: PromotionOperationIdentityV1,
        evidence: ProviderEvidenceEnvelopeV1 | None = None,
    ) -> bool:
        return (
            evidence is not None
            and evidence.is_preserved_provider_evidence()
            and self.result_source == "provider-async-result"
            and self.status == "merged"
            and bool(self.provider_uuid)
            and self.requested_pr_number == identity.requested_pr_number
            and self.expected_head_sha == identity.requested_pr_head_sha
            and self.merge_method == identity.merge_method
            and self.merge_action == identity.merge_action
            and bool(self.observed_merge_commit)
        )


@dataclass(frozen=True)
class ProviderResultRetentionV1:
    retention_hours: int = 24
    age_hours: int = 0
    captured_locally: bool = False
    provider_available: bool = True

    def classify(self) -> str:
        if self.retention_hours <= 0 or self.age_hours < 0:
            return "invalid-retention-state"
        if self.captured_locally:
            return "provider-result-captured-locally"
        if not self.provider_available:
            return "provider-result-unavailable"
        if self.age_hours >= self.retention_hours:
            return "provider-result-expired"
        return "provider-result-provider-recoverable"

    def direct_evidence_recoverable(self) -> bool:
        return self.classify() in {
            "provider-result-captured-locally",
            "provider-result-provider-recoverable",
        }


def provider_result_retention_fixture(
    *,
    age_hours: int = 1,
    captured_locally: bool = False,
    provider_available: bool = True,
) -> ProviderResultRetentionV1:
    return ProviderResultRetentionV1(
        retention_hours=24,
        age_hours=age_hours,
        captured_locally=captured_locally,
        provider_available=provider_available,
    )


@dataclass(frozen=True)
class PromotionCausalResolutionV1:
    outcome: str
    requested_effect_causal: bool
    stack_effect_causal: bool

    @classmethod
    def resolve(
        cls,
        identity: PromotionOperationIdentityV1,
        provider_result: ProviderMergeResultV1 | None,
        effect_set: PromotionStackEffectSetV1 | None,
        topology_binding: ProviderTopologyBindingV1 | None,
        provider_evidence: ProviderEvidenceEnvelopeV1 | None = None,
    ) -> "PromotionCausalResolutionV1":
        effect_observed = (
            effect_set is not None and effect_set.validates_complete(identity)
        )
        requested_causal = (
            provider_result is not None
            and provider_result.directly_binds_requested_effect(
                identity,
                provider_evidence,
            )
        )

        if requested_causal and effect_observed:
            requested_entry = effect_set.effects[-1]
            requested_causal = (
                requested_entry.pr_number == identity.requested_pr_number
                and requested_entry.expected_head_sha == identity.requested_pr_head_sha
                and requested_entry.observed_merge_commit
                == provider_result.observed_merge_commit
            )

        topology_cas = (
            topology_binding is not None
            and topology_binding.classify(identity) == "provider-topology-cas"
        )
        stack_causal = requested_causal and effect_observed and topology_cas

        if stack_causal:
            return cls("stack-effect-causal", True, True)
        if requested_causal:
            return cls("requested-effect-causal", True, False)
        if effect_observed:
            return cls("effect-observed-only", False, False)
        return cls("causality-unestablished", False, False)


def provider_evidence_fixture(
    *,
    method: str = "authenticated-api-channel",
    verified: bool = True,
    durable: bool = True,
    sequence: int = 1,
) -> ProviderEvidenceEnvelopeV1:
    capture = ProviderCaptureIntegrityV1(
        raw_bytes_digest="raw-digest",
        storage_id="capture-1",
        capture_sequence=sequence,
        durable=durable,
    )
    source = ProviderSourceAuthenticationV1(
        method=method,
        verified=verified,
        provider_identity="github",
    )
    return ProviderEvidenceEnvelopeV1(capture, source)


def provider_merge_result_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    *,
    status: str = "merged",
    provider_uuid: str | None = "uuid-1",
    observed_merge_commit: str | None = "M2",
) -> ProviderMergeResultV1:
    identity = identity or stack_identity_fixture()
    return ProviderMergeResultV1(
        result_source="provider-async-result",
        status=status,
        provider_uuid=provider_uuid,
        requested_pr_number=identity.requested_pr_number,
        expected_head_sha=identity.requested_pr_head_sha,
        merge_method=identity.merge_method,
        merge_action=identity.merge_action,
        observed_merge_commit=observed_merge_commit,
    )


def causal_resolution_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    *,
    provider_result: ProviderMergeResultV1 | None = None,
    effect_set: PromotionStackEffectSetV1 | None = None,
    topology_binding: ProviderTopologyBindingV1 | None = None,
    provider_evidence: ProviderEvidenceEnvelopeV1 | None = None,
) -> PromotionCausalResolutionV1:dent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from itertools import permutations


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

    def canonical_bytes(self) -> bytes:
        payload = {
            "base_ref": self.base_ref,
            "provider_stack_number": self.provider_stack_number,
            "base_tip_sha": self.base_tip_sha,
            "governance_generation": self.governance_generation,
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


def provider_stack_observation_matches_reserved(
    observation: ProviderStackObservationV1 | None,
    identity: PromotionOperationIdentityV1,
) -> bool:
    return observation is not None and observation.matches_reserved(identity)


@dataclass(frozen=True)
class ProviderTopologyBindingV1:
    initial_observation: ProviderStackObservationV1 | None
    pre_submit_observation: ProviderStackObservationV1 | None
    initial_sequence: int
    pre_submit_sequence: int | None
    provider_topology_cas: bool = False

    def classify(self, identity: PromotionOperationIdentityV1) -> str:
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
        if self.provider_topology_cas:
            return "provider-topology-cas"
        return "observed-not-cas"


@dataclass(frozen=True)
class ProviderCaptureIntegrityV1:
    raw_bytes_digest: str
    storage_id: str
    capture_sequence: int
    durable: bool = True

    @classmethod
    def capture(
        cls,
        raw_bytes: bytes,
        storage_id: str,
        capture_sequence: int,
        durable: bool = True,
    ) -> "ProviderCaptureIntegrityV1":
        return cls(
            raw_bytes_digest=hashlib.sha256(raw_bytes).hexdigest(),
            storage_id=storage_id,
            capture_sequence=capture_sequence,
            durable=durable,
        )

    def is_valid(self) -> bool:
        return (
            bool(self.raw_bytes_digest)
            and bool(self.storage_id)
            and self.capture_sequence > 0
            and self.durable
        )


@dataclass(frozen=True)
class ProviderSourceAuthenticationV1:
    method: str
    verified: bool
    provider_identity: str | None = None

    def is_accepted_transport_auth(self) -> bool:
        return self.verified and self.method in {
            "authenticated-api-channel",
            "webhook-hmac-verified",
        }


@dataclass(frozen=True)
class ProviderAttestationV1:
    scheme: str | None = None
    verified: bool = False

    def is_verified(self) -> bool:
        return self.verified and bool(self.scheme)


@dataclass(frozen=True)
class ProviderEvidenceEnvelopeV1:
    capture: ProviderCaptureIntegrityV1
    source_authentication: ProviderSourceAuthenticationV1
    provider_attestation: ProviderAttestationV1 = ProviderAttestationV1()

    def is_preserved_provider_evidence(self) -> bool:
        return (
            self.capture.is_valid()
            and self.source_authentication.is_accepted_transport_auth()
        )

    def has_provider_attestation(self) -> bool:
        return self.provider_attestation.is_verified()


@dataclass(frozen=True)
class ProviderWebhookReceiptV1:
    delivery_id: str
    payload_bytes_digest: str
    signature: str
    signature_algorithm: str = "HMAC-SHA256"

    @classmethod
    def from_delivery(
        cls,
        delivery_id: str,
        payload: bytes,
        secret: bytes,
    ) -> "ProviderWebhookReceiptV1":
        digest = hashlib.sha256(payload).hexdigest()
        mac = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return cls(
            delivery_id=delivery_id,
            payload_bytes_digest=digest,
            signature=f"sha256={mac}",
        )

    def verify(self, payload: bytes, secret: bytes) -> bool:
        expected = (
            "sha256="
            + hmac.new(secret, payload, hashlib.sha256).hexdigest()
        )
        return (
            self.signature_algorithm == "HMAC-SHA256"
            and bool(self.delivery_id)
            and hmac.compare_digest(self.signature, expected)
            and self.payload_bytes_digest == hashlib.sha256(payload).hexdigest()
        )


@dataclass(frozen=True)
class ProviderDeliveryRegistryV1:
    deliveries: tuple[tuple[str, str], ...] = ()

    def observe(self, receipt: ProviderWebhookReceiptV1) -> str:
        for delivery_id, payload_digest in self.deliveries:
            if delivery_id != receipt.delivery_id:
                continue
            if payload_digest == receipt.payload_bytes_digest:
                return "duplicate-identical"
            return "delivery-id-conflict"
        return "new-delivery"

    def record(self, receipt: ProviderWebhookReceiptV1) -> "ProviderDeliveryRegistryV1":
        classification = self.observe(receipt)
        if classification == "new-delivery":
            return ProviderDeliveryRegistryV1(
                self.deliveries + ((receipt.delivery_id, receipt.payload_bytes_digest),)
            )
        return self


@dataclass(frozen=True)
class ProviderMergeResultV1:
    result_source: str
    status: str
    provider_uuid: str | None
    requested_pr_number: int
    expected_head_sha: str
    merge_method: str
    merge_action: str
    observed_merge_commit: str | None = None

    def directly_binds_requested_effect(
        self,
        identity: PromotionOperationIdentityV1,
        evidence: ProviderEvidenceEnvelopeV1 | None = None,
    ) -> bool:
        return (
            evidence is not None
            and evidence.is_preserved_provider_evidence()
            and self.result_source == "provider-async-result"
            and self.status == "merged"
            and bool(self.provider_uuid)
            and self.requested_pr_number == identity.requested_pr_number
            and self.expected_head_sha == identity.requested_pr_head_sha
            and self.merge_method == identity.merge_method
            and self.merge_action == identity.merge_action
            and bool(self.observed_merge_commit)
        )


@dataclass(frozen=True)
class ProviderResultRetentionV1:
    retention_hours: int = 24
    age_hours: int = 0
    captured_locally: bool = False
    provider_available: bool = True

    def classify(self) -> str:
        if self.retention_hours <= 0 or self.age_hours < 0:
            return "invalid-retention-state"
        if self.captured_locally:
            return "provider-result-captured-locally"
        if not self.provider_available:
            return "provider-result-unavailable"
        if self.age_hours >= self.retention_hours:
            return "provider-result-expired"
        return "provider-result-provider-recoverable"

    def direct_evidence_recoverable(self) -> bool:
        return self.classify() in {
            "provider-result-captured-locally",
            "provider-result-provider-recoverable",
        }


def provider_result_retention_fixture(
    *,
    age_hours: int = 1,
    captured_locally: bool = False,
    provider_available: bool = True,
) -> ProviderResultRetentionV1:
    return ProviderResultRetentionV1(
        retention_hours=24,
        age_hours=age_hours,
        captured_locally=captured_locally,
        provider_available=provider_available,
    )


@dataclass(frozen=True)
class PromotionCausalResolutionV1:
    outcome: str
    requested_effect_causal: bool
    stack_effect_causal: bool

    @classmethod
    def resolve(
        cls,
        identity: PromotionOperationIdentityV1,
        provider_result: ProviderMergeResultV1 | None,
        effect_set: PromotionStackEffectSetV1 | None,
        topology_binding: ProviderTopologyBindingV1 | None,
        provider_evidence: ProviderEvidenceEnvelopeV1 | None = None,
    ) -> "PromotionCausalResolutionV1":
        effect_observed = (
            effect_set is not None and effect_set.validates_complete(identity)
        )
        requested_causal = (
            provider_result is not None
            and provider_result.directly_binds_requested_effect(
                identity,
                provider_evidence,
            )
        )

        if requested_causal and effect_observed:
            requested_entry = effect_set.effects[-1]
            requested_causal = (
                requested_entry.pr_number == identity.requested_pr_number
                and requested_entry.expected_head_sha == identity.requested_pr_head_sha
                and requested_entry.observed_merge_commit
                == provider_result.observed_merge_commit
            )

        topology_cas = (
            topology_binding is not None
            and topology_binding.classify(identity) == "provider-topology-cas"
        )
        stack_causal = requested_causal and effect_observed and topology_cas

        if stack_causal:
            return cls("stack-effect-causal", True, True)
        if requested_causal:
            return cls("requested-effect-causal", True, False)
        if effect_observed:
            return cls("effect-observed-only", False, False)
        return cls("causality-unestablished", False, False)


def provider_evidence_fixture(
    *,
    method: str = "authenticated-api-channel",
    verified: bool = True,
    durable: bool = True,
    sequence: int = 1,
) -> ProviderEvidenceEnvelopeV1:
    capture = ProviderCaptureIntegrityV1(
        raw_bytes_digest="raw-digest",
        storage_id="capture-1",
        capture_sequence=sequence,
        durable=durable,
    )
    source = ProviderSourceAuthenticationV1(
        method=method,
        verified=verified,
        provider_identity="github",
    )
    return ProviderEvidenceEnvelopeV1(capture, source)


def provider_merge_result_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    *,
    status: str = "merged",
    provider_uuid: str | None = "uuid-1",
    observed_merge_commit: str | None = "M2",
) -> ProviderMergeResultV1:
    identity = identity or stack_identity_fixture()
    return ProviderMergeResultV1(
        result_source="provider-async-result",
        status=status,
        provider_uuid=provider_uuid,
        requested_pr_number=identity.requested_pr_number,
        expected_head_sha=identity.requested_pr_head_sha,
        merge_method=identity.merge_method,
        merge_action=identity.merge_action,
        observed_merge_commit=observed_merge_commit,
    )


def causal_resolution_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
    *,
    provider_result: ProviderMergeResultV1 | None | object = _UNSET,
    effect_set: PromotionStackEffectSetV1 | None | object = _UNSET,
    topology_binding: ProviderTopologyBindingV1 | None | object = _UNSET,
    provider_evidence: ProviderEvidenceEnvelopeV1 | None | object = _UNSET,
) -> PromotionCausalResolutionV1:
    identity = identity or stack_identity_fixture()
    if provider_result is _UNSET:
        provider_result = provider_merge_result_fixture(identity)
    if effect_set is _UNSET:
        effect_set = stack_effect_fixture(identity)
    if topology_binding is _UNSET:
        topology_binding = topology_binding_fixture(identity)
    if provider_evidence is _UNSET:
        provider_evidence = provider_evidence_fixture()
    return PromotionCausalResolutionV1.resolve(
        identity,
        provider_result,
        effect_set,
        topology_binding,
        provider_evidence,
    )


def test_capture_integrity_hashes_exact_raw_bytes():
    capture = ProviderCaptureIntegrityV1.capture(
        b"provider-result",
        "store-1",
        1,
    )
    assert capture.raw_bytes_digest == hashlib.sha256(b"provider-result").hexdigest()
    assert capture.is_valid()


def test_capture_integrity_rejects_empty_storage_identity():
    capture = ProviderCaptureIntegrityV1(
        "digest",
        "",
        1,
        durable=True,
    )
    assert not capture.is_valid()


def test_capture_integrity_requires_durable_storage():
    capture = ProviderCaptureIntegrityV1("D", "store-1", 1, durable=True)
    assert capture.is_valid()


def test_capture_integrity_rejects_nonpositive_sequence():
    capture = ProviderCaptureIntegrityV1("D", "store-1", 0, durable=True)
    assert not capture.is_valid()


def test_source_authentication_api_channel_is_distinct_from_attestation():
    source = ProviderSourceAuthenticationV1(
        "authenticated-api-channel",
        True,
        "github",
    )
    evidence = ProviderEvidenceEnvelopeV1(
        ProviderCaptureIntegrityV1("D", "store-1", 1),
        source,
    )
    assert evidence.is_preserved_provider_evidence()
    assert not evidence.has_provider_attestation()


def test_source_authentication_rejects_missing_provider_identity():
    evidence = ProviderEvidenceEnvelopeV1(
        ProviderCaptureIntegrityV1("D", "store-1", 1),
        ProviderSourceAuthenticationV1(
            "authenticated-api-channel",
            True,
            None,
        ),
    )
    assert not evidence.is_preserved_provider_evidence()


def test_source_authentication_rejects_unverified_api_channel():
    evidence = provider_evidence_fixture(verified=False)
    assert not evidence.is_preserved_provider_evidence()


def test_webhook_hmac_verification_accepts_exact_payload():
    payload = b'{"action":"closed","pull_request":{"number":7087}}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-1",
        payload,
        b"secret",
    )
    assert receipt.verify(payload, b"secret")


def test_webhook_hmac_verification_rejects_tampered_payload():
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-2",
        payload,
        b"secret",
    )
    assert not receipt.verify(b'{"action":"opened"}', b"secret")


def test_webhook_hmac_verification_rejects_wrong_secret():
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-3",
        payload,
        b"secret",
    )
    assert not receipt.verify(payload, b"wrong")


def test_webhook_delivery_registry_accepts_new_delivery():
    payload = b"{}"
    receipt = ProviderWebhookReceiptV1.from_delivery("delivery-4", payload, b"secret")
    registry = ProviderDeliveryRegistryV1()
    assert registry.observe(receipt) == "new-delivery"
    registry = registry.record(receipt)
    assert registry.observe(receipt) == "duplicate-identical"


def test_webhook_delivery_registry_rejects_same_id_with_different_payload():
    first = ProviderWebhookReceiptV1.from_delivery("delivery-5", b"{}", b"secret")
    second = ProviderWebhookReceiptV1.from_delivery(
        "delivery-5",
        b'{"action":"different"}',
        b"secret",
    )
    registry = ProviderDeliveryRegistryV1().record(first)
    assert registry.observe(second) == "delivery-id-conflict"


def test_webhook_authentication_does_not_prove_merge_result_causality():
    identity = stack_identity_fixture()
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-6",
        payload,
        b"secret",
    )
    assert receipt.verify(payload, b"secret")
    webhook_evidence = ProviderEvidenceEnvelopeV1(
        ProviderCaptureIntegrityV1(
            hashlib.sha256(payload).hexdigest(),
            "webhook-1",
            1,
        ),
        ProviderSourceAuthenticationV1(
            "webhook-hmac-verified",
            True,
            "github",
        ),
    )
    result = provider_merge_result_fixture(identity)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=webhook_evidence,
    )
    assert resolution.outcome == "causality-unestablished"


def test_fabricated_local_capture_cannot_establish_requested_causality():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    local = provider_evidence_fixture(method="local-untrusted", verified=True)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=local,
    )
    assert resolution.outcome == "causality-unestablished"


def test_authenticated_api_capture_can_support_requested_causality():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    evidence = provider_evidence_fixture()
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=evidence,
    )
    assert resolution.outcome == "requested-effect-causal"


def test_invalid_capture_cannot_support_requested_causality():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    evidence = provider_evidence_fixture(durable=False)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=evidence,
    )
    assert resolution.outcome == "causality-unestablished"


def test_provider_result_retention_captured_locally_survives_expiry():
    retention = provider_result_retention_fixture(
        age_hours=72,
        captured_locally=True,
    )
    assert retention.classify() == "provider-result-captured-locally"
    assert retention.direct_evidence_recoverable()


def test_provider_result_retention_is_recoverable_before_expiry():
    retention = provider_result_retention_fixture(age_hours=12)
    assert retention.classify() == "provider-result-provider-recoverable"
    assert retention.direct_evidence_recoverable()


def test_provider_result_retention_expires_at_window_boundary():
    retention = provider_result_retention_fixture(age_hours=24)
    assert retention.classify() == "provider-result-expired"
    assert not retention.direct_evidence_recoverable()


def test_provider_result_retention_expired_after_window():
    retention = provider_result_retention_fixture(age_hours=48)
    assert retention.classify() == "provider-result-expired"
    assert not retention.direct_evidence_recoverable()


def test_provider_result_retention_provider_unavailable_before_expiry_is_distinct():
    retention = provider_result_retention_fixture(
        age_hours=6,
        provider_available=False,
    )
    assert retention.classify() == "provider-result-unavailable"
    assert not retention.direct_evidence_recoverable()


def test_provider_result_retention_invalid_negative_age_fails_closed():
    retention = provider_result_retention_fixture(age_hours=-1)
    assert retention.classify() == "invalid-retention-state"
    assert not retention.direct_evidence_recoverable()


def test_provider_result_retention_invalid_nonpositive_window_fails_closed():
    retention = ProviderResultRetentionV1(retention_hours=0, age_hours=1)
    assert retention.classify() == "invalid-retention-state"
    assert not retention.direct_evidence_recoverable()


def test_expiry_does_not_change_later_effect_observation_class():
    identity = stack_identity_fixture()
    retention = provider_result_retention_fixture(age_hours=48)
    assert retention.classify() == "provider-result-expired"
    resolution = causal_resolution_fixture(
        identity,
        provider_result=None,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"


def test_requested_effect_causality_from_direct_provider_result():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "requested-effect-causal"
    assert resolution.requested_effect_causal
    assert not resolution.stack_effect_causal


def test_direct_result_plus_exact_effect_set_stays_requested_causal_without_provider_cas():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    topology = topology_binding_fixture(identity)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology,
    )
    assert resolution.outcome == "requested-effect-causal"


def test_direct_result_plus_exact_effect_set_becomes_stack_causal_only_with_explicit_provider_cas():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    topology = topology_binding_fixture(identity, provider_topology_cas=True)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology,
    )
    assert resolution.outcome == "stack-effect-causal"
    assert resolution.stack_effect_causal


def test_direct_result_plus_stale_topology_remains_requested_causal_only():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    initial = provider_stack_observation_fixture(identity)
    stale = ProviderStackObservationV1(**{**initial.__dict__, "stack_number": 42})
    topology = topology_binding_fixture(identity, stale)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology,
    )
    assert resolution.outcome == "requested-effect-causal"


def test_direct_result_plus_matching_revalidation_without_cas_is_not_stack_causal():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    topology = topology_binding_fixture(identity)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology,
    )
    assert resolution.outcome == "requested-effect-causal"
    assert not resolution.stack_effect_causal


def test_enqueued_effect_set_is_observed_only():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(
        identity,
        status="enqueued",
        provider_uuid="uuid-1",
        observed_merge_commit=None,
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"
    assert not resolution.requested_effect_causal


def test_already_merged_retry_with_no_uuid_is_observed_only():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(
        identity,
        status="merged",
        provider_uuid=None,
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"


def test_expired_or_missing_async_result_with_exact_effect_is_observed_only():
    identity = stack_identity_fixture()
    resolution = causal_resolution_fixture(
        identity,
        provider_result=None,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"


def test_local_receipt_without_provider_result_cannot_establish_causality():
    identity = stack_identity_fixture()
    resolution = causal_resolution_fixture(
        identity,
        provider_result=None,
        effect_set=None,
        topology_binding=None,
    )
    assert resolution.outcome == "causality-unestablished"


def test_provider_result_wrong_requested_head_cannot_establish_causality():
    identity = stack_identity_fixture()
    wrong = ProviderMergeResultV1(
        **{
            **provider_merge_result_fixture(identity).__dict__,
            "expected_head_sha": "H0",
        }
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=wrong,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"


def test_provider_result_conflicting_requested_merge_commit_is_not_causal():
    identity = stack_identity_fixture()
    wrong = ProviderMergeResultV1(
        **{
            **provider_merge_result_fixture(identity).__dict__,
            "observed_merge_commit": "M9",
        }
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=wrong,
        effect_set=stack_effect_fixture(identity),
        topology_binding=topology_binding_fixture(identity),
    )
    assert resolution.outcome == "effect-observed-only"


def test_stack_causality_requires_complete_effect_set():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    incomplete = PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(PromotionStackEffectV1(7085, "H1", "M1"),),
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=incomplete,
        topology_binding=topology_binding_fixture(identity, provider_topology_cas=True),
    )
    assert resolution.outcome == "requested-effect-causal"
    assert not resolution.stack_effect_causal


def test_provider_operation_result_without_result_source_is_not_causal():
    identity = stack_identity_fixture()
    result = ProviderMergeResultV1(
        **{
            **provider_merge_result_fixture(identity).__dict__,
            "result_source": "local-receipt",
        }
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        topology_binding=None,
    )
    assert resolution.outcome == "causality-unestablished"


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
    ):
        self.http = http
        self.kind = kind
        self.uuid = uuid
        self.merge_method = merge_method
        self.merge_action = merge_action


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
        self.async_status: str | None = None
        self.merge_sha: str | None = None
        self.expired: set[str] = set()
        self.calls = 0

    def submit(
        self,
        expected_head: str,
        merge_method: str = "squash",
        merge_action: str = "direct_merge",
        timeout_after_accept: bool = False,
    ) -> ProviderOutcome:
        self.calls += 1
        if expected_head != self.pr_head:
            return ProviderOutcome(409, "rejected")
        if self.merge_sha is not None:
            return ProviderOutcome(200, "merged")
        if self.pending_uuid is not None:
            if (
                merge_method == self.pending_merge_method
                and merge_action == self.pending_merge_action
            ):
                return ProviderOutcome(
                    409,
                    "duplicate",
                    self.pending_uuid,
                    self.pending_merge_method,
                    self.pending_merge_action,
                )
            return ProviderOutcome(
                409,
                "duplicate-parameter-mismatch",
                self.pending_uuid,
                self.pending_merge_method,
                self.pending_merge_action,
            )
        self.pending_uuid = f"uuid-{self.calls}"
        self.pending_merge_method = merge_method
        self.pending_merge_action = merge_action
        self.async_status = "pending"
        if timeout_after_accept:
            return ProviderOutcome(
                599,
                "timeout-after-accept",
                self.pending_uuid,
                merge_method,
                merge_action,
            )
        return ProviderOutcome(
            202,
            "accepted",
            self.pending_uuid,
            merge_method,
            merge_action,
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
            )
        if self.async_status == "merged" and self.pending_uuid == uuid:
            return ProviderOutcome(
                200,
                "merged",
                uuid,
                self.pending_merge_method,
                self.pending_merge_action,
            )
        if self.pending_uuid == uuid:
            return ProviderOutcome(
                200,
                "pending",
                uuid,
                self.pending_merge_method,
                self.pending_merge_action,
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
    ) -> EffectReconciliation:
        result = self.get_async_result(uuid)

        if result.kind == "merged":
            if (
                result.uuid == uuid
                and result.merge_method == merge_method
                and result.merge_action == merge_action
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
    provider_topology_cas: bool = False,
) -> ProviderTopologyBindingV1:
    identity = identity or stack_identity_fixture()
    initial = provider_stack_observation_fixture(identity)
    return ProviderTopologyBindingV1(
        initial_observation=initial,
        pre_submit_observation=(
            pre_submit_observation if pre_submit_observation is not None else initial
        ),
        initial_sequence=1,
        pre_submit_sequence=pre_submit_sequence,
        provider_topology_cas=provider_topology_cas,
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


def test_provider_topology_binding_synthetic_cas_is_explicit():
    identity = stack_identity_fixture()
    binding = topology_binding_fixture(identity, provider_topology_cas=True)
    assert binding.classify(identity) == "provider-topology-cas"


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
    test_capture_integrity_hashes_exact_raw_bytes,
    test_capture_integrity_rejects_empty_storage_identity,
    test_capture_integrity_requires_durable_storage,
    test_capture_integrity_rejects_nonpositive_sequence,
    test_source_authentication_api_channel_is_distinct_from_attestation,
    test_source_authentication_rejects_missing_provider_identity,
    test_source_authentication_rejects_unverified_api_channel,
    test_webhook_hmac_verification_accepts_exact_payload,
    test_webhook_hmac_verification_rejects_tampered_payload,
    test_webhook_hmac_verification_rejects_wrong_secret,
    test_webhook_delivery_registry_accepts_new_delivery,
    test_webhook_delivery_registry_rejects_same_id_with_different_payload,
    test_webhook_authentication_does_not_prove_merge_result_causality,
    test_fabricated_local_capture_cannot_establish_requested_causality,
    test_authenticated_api_capture_can_support_requested_causality,
    test_invalid_capture_cannot_support_requested_causality,
    test_expiry_does_not_change_later_effect_observation_class,
    test_provider_result_retention_captured_locally_survives_expiry,
    test_provider_result_retention_is_recoverable_before_expiry,
    test_provider_result_retention_expires_at_window_boundary,
    test_provider_result_retention_expired_after_window,
    test_provider_result_retention_provider_unavailable_before_expiry_is_distinct,
    test_provider_result_retention_invalid_negative_age_fails_closed,
    test_provider_result_retention_invalid_nonpositive_window_fails_closed,
    test_expiry_does_not_change_later_effect_observation_class,
    test_requested_effect_causality_from_direct_provider_result,
    test_direct_result_plus_exact_effect_set_stays_requested_causal_without_provider_cas,
    test_direct_result_plus_exact_effect_set_becomes_stack_causal_only_with_explicit_provider_cas,
    test_direct_result_plus_stale_topology_remains_requested_causal_only,
    test_direct_result_plus_matching_revalidation_without_cas_is_not_stack_causal,
    test_enqueued_effect_set_is_observed_only,
    test_already_merged_retry_with_no_uuid_is_observed_only,
    test_expired_or_missing_async_result_with_exact_effect_is_observed_only,
    test_local_receipt_without_provider_result_cannot_establish_causality,
    test_provider_result_wrong_requested_head_cannot_establish_causality,
    test_provider_result_conflicting_requested_merge_commit_is_not_causal,
    test_stack_causality_requires_complete_effect_set,
    test_provider_operation_result_without_result_source_is_not_causal,
    test_provider_topology_binding_requires_an_initial_observation,
    test_provider_topology_binding_rejects_initial_topology_mismatch,
    test_provider_topology_binding_requires_pre_submit_revalidation,
    test_provider_topology_binding_matching_revalidation_without_cas_is_observed_only,
    test_provider_topology_binding_detects_stale_pre_submit_topology,
    test_provider_topology_binding_detects_invalid_observation_order,
    test_provider_topology_binding_requires_positive_initial_sequence,
    test_provider_topology_binding_synthetic_cas_is_explicit,
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
    test_stack_identity_binds_requested_subject_identity,
    test_exact_20_schedules_are_executed,
    test_single_use_reservation,
    test_dispatch_intent_fences_crash,
    test_timeout_after_acceptance_is_unknown,
    test_duplicate_async_request_reuses_provider_handle,
    test_duplicate_async_request_option_mismatch_is_not_idempotent,
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
