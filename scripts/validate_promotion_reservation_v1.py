#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

from datetime import datetime

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

    @classmethod
    def from_webhook_observations(
        cls,
        identity: PromotionOperationIdentityV1,
        observations: tuple["ProviderPullRequestMergeObservationV1", ...],
    ) -> "PromotionStackEffectSetV1 | None":
        expected = identity.ordered_stack
        expected_prs = {entry.pr_number for entry in expected}
        by_pr: dict[int, "ProviderPullRequestMergeObservationV1"] = {}

        for observation in observations:
            if observation.repository != identity.repository:
                return None
            if observation.pr_number in by_pr:
                return None
            by_pr[observation.pr_number] = observation

        if set(by_pr) != expected_prs:
            return None

        effects: list[PromotionStackEffectV1] = []
        for entry in expected:
            observation = by_pr.get(entry.pr_number)
            if observation is None:
                return None
            effect = observation.to_effect_for_stack_entry(entry)
            if effect is None:
                return None
            effects.append(effect)

        return cls(
            operation_identity_digest=identity.digest(),
            effects=tuple(effects),
        )

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
    hook_id: str
    event_type: str
    repository: str
    payload_bytes_digest: str
    signature: str
    signature_algorithm: str = "HMAC-SHA256"

    @classmethod
    def from_delivery(
        cls,
        delivery_id: str,
        hook_id: str,
        event_type: str,
        repository: str,
        payload: bytes,
        secret: bytes,
    ) -> "ProviderWebhookReceiptV1":
        digest = hashlib.sha256(payload).hexdigest()
        mac = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return cls(
            delivery_id=delivery_id,
            hook_id=hook_id,
            event_type=event_type,
            repository=repository,
            payload_bytes_digest=digest,
            signature=f"sha256={mac}",
        )

    def verify(
        self,
        payload: bytes,
        secret: bytes,
        received_context: "ProviderWebhookRequestContextV1",
        expected_hook_id: str,
        expected_event_type: str,
        expected_repository: str,
    ) -> bool:
        expected = "sha256=" + hmac.new(
            secret,
            payload,
            hashlib.sha256,
        ).hexdigest()
        return (
            self.signature_algorithm == "HMAC-SHA256"
            and bool(self.delivery_id)
            and bool(self.hook_id)
            and bool(self.event_type)
            and bool(self.repository)
            and received_context.matches_receipt(self)
            and received_context.hook_id == expected_hook_id
            and received_context.event_type == expected_event_type
            and received_context.repository == expected_repository
            and hmac.compare_digest(self.signature, expected)
            and self.payload_bytes_digest == hashlib.sha256(payload).hexdigest()
        )


@dataclass(frozen=True)
class ProviderWebhookRequestContextV1:
    delivery_id: str
    hook_id: str
    event_type: str
    repository: str
    received_at_ms: int | None = None
    received_monotonic_ns: int | None = None

    def matches_receipt(self, receipt: ProviderWebhookReceiptV1) -> bool:
        return (
            self.delivery_id == receipt.delivery_id
            and self.hook_id == receipt.hook_id
            and self.event_type == receipt.event_type
            and self.repository == receipt.repository
        )


@dataclass(frozen=True)
class ProviderWebhookStackMetadataV1:
    stack_number: int
    stack_size: int
    stack_position: int
    base_ref: str
    base_sha: str

    def internally_consistent(self) -> bool:
        return (
            self.stack_number > 0
            and self.stack_size > 0
            and 1 <= self.stack_position <= self.stack_size
            and bool(self.base_ref)
            and bool(self.base_sha)
        )

    def matches_operation(
        self,
        identity: PromotionOperationIdentityV1,
    ) -> bool:
        if not self.internally_consistent():
            return False
        expected_position = next(
            (
                index + 1
                for index, entry in enumerate(identity.ordered_stack)
                if entry.pr_number == identity.requested_pr_number
            ),
            None,
        )
        return (
            expected_position is not None
            and self.stack_number == identity.provider_stack_number
            and self.stack_size >= len(identity.ordered_stack)
            and self.stack_position == expected_position
            and self.base_ref == identity.base_ref
            and self.base_sha == identity.base_tip_sha
        )


@dataclass(frozen=True)
class ProviderPullRequestMergeObservationV1:
    delivery_id: str
    repository: str
    pr_number: int
    event_type: str
    action: str
    merged: bool
    head_sha: str
    merge_commit_sha: str
    payload_bytes_digest: str
    merged_at: str = ""
    provider_stack: ProviderWebhookStackMetadataV1 | None = None
    local_received_at_ms: int | None = None
    local_received_monotonic_ns: int | None = None
    hook_id: str = ""
    source_authentication: str = ""

    @classmethod
    def from_authenticated_delivery(
        cls,
        receipt: ProviderWebhookReceiptV1,
        payload: bytes,
        secret: bytes,
        received_context: ProviderWebhookRequestContextV1,
        expected_hook_id: str,
        expected_event_type: str,
        expected_repository: str,
    ) -> "ProviderPullRequestMergeObservationV1 | None":
        if not receipt.verify(
            payload,
            secret,
            received_context,
            expected_hook_id,
            expected_event_type,
            expected_repository,
        ):
            return None

        try:
            document = json.loads(payload.decode("utf-8"))
            pull_request = document["pull_request"]
            repository = document["repository"]
            head = pull_request["head"]
            merge_commit_sha = pull_request["merge_commit_sha"]
            merged_at = pull_request.get("merged_at") or ""
            stack = pull_request.get("stack")
            provider_stack = (
                ProviderWebhookStackMetadataV1(
                    stack_number=int(stack["number"]),
                    stack_size=int(stack["size"]),
                    stack_position=int(stack["position"]),
                    base_ref=str(stack["base"]["ref"]),
                    base_sha=str(stack["base"]["sha"]),
                )
                if stack is not None
                else None
            )
            pr_number = int(document["number"])
            nested_pr_number = int(pull_request["number"])
            event_type = receipt.event_type
            action = document["action"]
            merged = bool(pull_request["merged"])
            head_sha = str(head["sha"])
            repository_name = str(repository["full_name"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None

        if (
            event_type != "pull_request"
            or action != "closed"
            or not merged
            or not repository_name
            or repository_name != receipt.repository
            or pr_number != nested_pr_number
            or not head_sha
            or not merge_commit_sha
        ):
            return None

        return cls(
            delivery_id=receipt.delivery_id,
            repository=repository_name,
            pr_number=pr_number,
            event_type=event_type,
            action=action,
            merged=merged,
            head_sha=head_sha,
            merge_commit_sha=str(merge_commit_sha),
            payload_bytes_digest=hashlib.sha256(payload).hexdigest(),
            merged_at=str(merged_at),
            provider_stack=provider_stack,
            local_received_at_ms=received_context.received_at_ms,
            local_received_monotonic_ns=received_context.received_monotonic_ns,
            hook_id=receipt.hook_id,
            source_authentication="webhook-hmac-verified",
        )

    def to_effect_for_stack_entry(
        self,
        entry: StackEntryV1,
    ) -> PromotionStackEffectV1 | None:
        if not (
            self.event_type == "pull_request"
            and self.action == "closed"
            and self.merged
            and self.pr_number == entry.pr_number
            and self.head_sha == entry.head_sha
            and bool(self.merge_commit_sha)
        ):
            return None
        return PromotionStackEffectV1(
            pr_number=self.pr_number,
            expected_head_sha=self.head_sha,
            observed_merge_commit=self.merge_commit_sha,
        )

    def has_authenticated_source_provenance(self) -> bool:
        return (
            bool(self.delivery_id)
            and bool(self.hook_id)
            and self.event_type == "pull_request"
            and bool(self.repository)
            and self.source_authentication == "webhook-hmac-verified"
            and bool(self.payload_bytes_digest)
        )

    def to_provenance_effect(
        self,
        entry: StackEntryV1,
    ) -> "PromotionStackEffectEvidenceV1 | None":
        effect = self.to_effect_for_stack_entry(entry)
        if effect is None or not self.has_authenticated_source_provenance():
            return None
        return PromotionStackEffectEvidenceV1(
            effect=effect,
            source_delivery_id=self.delivery_id,
            source_payload_digest=self.payload_bytes_digest,
            source_hook_id=self.hook_id,
            source_event_type=self.event_type,
            source_repository=self.repository,
            source_authentication=self.source_authentication,
        )

    def validates_provider_stack_metadata(
        self,
        identity: PromotionOperationIdentityV1,
    ) -> bool:
        return (
            self.provider_stack is not None
            and self.provider_stack.matches_operation(identity)
        )

    def validates_requested_effect(self, identity: PromotionOperationIdentityV1) -> bool:
        return (
            self.event_type == "pull_request"
            and self.action == "closed"
            and self.merged
            and self.repository == identity.repository
            and self.pr_number == identity.requested_pr_number
            and self.head_sha == identity.requested_pr_head_sha
            and bool(self.merge_commit_sha)
        )

    def to_stack_effect(
        self,
        identity: PromotionOperationIdentityV1,
    ) -> PromotionStackEffectV1 | None:
        if not self.validates_requested_effect(identity):
            return None
        return PromotionStackEffectV1(
            pr_number=self.pr_number,
            expected_head_sha=self.head_sha,
            observed_merge_commit=self.merge_commit_sha,
        )


@dataclass(frozen=True)
class ProviderDeliveryIdentityV1:
    delivery_id: str
    payload_bytes_digest: str
    hook_id: str
    event_type: str
    repository: str

    @classmethod
    def from_observation(
        cls,
        observation: ProviderPullRequestMergeObservationV1,
    ) -> "ProviderDeliveryIdentityV1":
        return cls(
            delivery_id=observation.delivery_id,
            payload_bytes_digest=observation.payload_bytes_digest,
            hook_id=observation.hook_id,
            event_type=observation.event_type,
            repository=observation.repository,
        )

    def matches(self, other: "ProviderDeliveryIdentityV1") -> bool:
        return self == other


@dataclass(frozen=True)
class PromotionPrEffectStateV1:
    repository: str
    expected_entry: StackEntryV1
    operation_identity_digest: str
    state: str = "Unobserved"
    effect: PromotionStackEffectV1 | None = None
    source_delivery_ids: tuple[str, ...] = ()
    source_delivery_identities: tuple[ProviderDeliveryIdentityV1, ...] = ()

    def validates_operation_identity(
        self,
        identity: PromotionOperationIdentityV1,
    ) -> bool:
        return (
            self.operation_identity_digest == identity.digest()
            and self.repository == identity.repository
            and self.expected_entry in identity.ordered_stack
        )

    def ingest(
        self,
        observation: ProviderPullRequestMergeObservationV1 | None,
    ) -> tuple["PromotionPrEffectStateV1", str]:
        if observation is None:
            return self, "no-eligible-effect"

        if not observation.has_authenticated_source_provenance():
            return self, "rejected-untrusted"

        if observation.repository != self.repository:
            return self, "ignored-unrelated"

        if observation.pr_number != self.expected_entry.pr_number:
            return self, "ignored-unrelated"

        candidate = observation.to_effect_for_stack_entry(self.expected_entry)
        if candidate is None:
            return self, "rejected-non-effect"

        delivery_identity = ProviderDeliveryIdentityV1.from_observation(observation)

        if self.state == "Unobserved":
            return (
                PromotionPrEffectStateV1(
                    repository=self.repository,
                    expected_entry=self.expected_entry,
                    operation_identity_digest=self.operation_identity_digest,
                    state="EffectObserved",
                    effect=candidate,
                    source_delivery_ids=(observation.delivery_id,),
                    source_delivery_identities=(delivery_identity,),
                ),
                "admitted",
            )

        if self.state != "EffectObserved" or self.effect is None:
            return self, "rejected-invalid-state"

        for prior_identity in self.source_delivery_identities:
            if prior_identity.delivery_id != delivery_identity.delivery_id:
                continue
            if prior_identity.matches(delivery_identity):
                return self, "duplicate-delivery"
            return (
                PromotionPrEffectStateV1(
                    repository=self.repository,
                    expected_entry=self.expected_entry,
                    operation_identity_digest=self.operation_identity_digest,
                    state="Conflict",
                    effect=None,
                    source_delivery_ids=self.source_delivery_ids,
                    source_delivery_identities=self.source_delivery_identities,
                ),
                "delivery-identity-conflict",
            )

        if candidate != self.effect:
            return (
                PromotionPrEffectStateV1(
                    repository=self.repository,
                    expected_entry=self.expected_entry,
                    operation_identity_digest=self.operation_identity_digest,
                    state="Conflict",
                    effect=None,
                    source_delivery_ids=self.source_delivery_ids,
                    source_delivery_identities=self.source_delivery_identities,
                ),
                "conflict",
            )

        return (
            PromotionPrEffectStateV1(
                repository=self.repository,
                expected_entry=self.expected_entry,
                operation_identity_digest=self.operation_identity_digest,
                state=self.state,
                effect=self.effect,
                source_delivery_ids=self.source_delivery_ids + (observation.delivery_id,),
                source_delivery_identities=self.source_delivery_identities + (delivery_identity,),
            ),
            "compatible-repeat",
        )

    def validates_source_delivery_bindings(self) -> bool:
        identities = self.source_delivery_identities
        delivery_ids = self.source_delivery_ids
        return (
            len(delivery_ids) == len(identities)
            and delivery_ids == tuple(identity.delivery_id for identity in identities)
            and len(delivery_ids) == len(set(delivery_ids))
            and all(
                bool(identity.delivery_id)
                and bool(identity.payload_bytes_digest)
                and bool(identity.hook_id)
                and bool(identity.event_type)
                and bool(identity.repository)
                for identity in identities
            )
        )

    def is_terminally_observed(self) -> bool:
        return (
            self.state == "EffectObserved"
            and self.effect is not None
            and self.validates_source_delivery_bindings()
        )

def parse_provider_timestamp_ms(value: str) -> int | None:
    interval = parse_provider_timestamp_interval_ms(
        value,
        occurrence_semantics="truncated",
    )
    return interval[0] if interval is not None else None


def _timestamp_reported_resolution_ms(value: str) -> int | None:
    if not value:
        return None
    raw = value
    if raw.endswith("Z"):
        raw = raw[:-1]
    try:
        date_part, time_part = raw.split("T", 1)
        if "+" in time_part[1:] or time_part.count("-") > 0:
            for separator in ("+", "-"):
                idx = time_part.find(separator, 1)
                if idx >= 0:
                    time_part = time_part[:idx]
                    break
        fraction = time_part.split(".", 1)[1] if "." in time_part else ""
    except ValueError:
        return None
    if not fraction:
        return 1000
    if not fraction.isdigit() or len(fraction) > 3:
        return None
    return 10 ** (3 - len(fraction))


def parse_provider_timestamp_interval_ms(
    value: str,
    *,
    occurrence_semantics: str,
) -> tuple[int, int] | None:
    if occurrence_semantics not in {"truncated", "exact"}:
        return None
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    resolution_ms = _timestamp_reported_resolution_ms(value)
    if resolution_ms is None:
        return None
    lower = int(parsed.timestamp() * 1000)
    if occurrence_semantics == "exact":
        if resolution_ms != 1:
            return None
        return lower, lower
    return lower, lower + resolution_ms - 1


@dataclass(frozen=True)
class ProviderTimestampPolicyV1:
    provider_identity: str
    source_field: str
    occurrence_semantics: str
    max_reported_resolution_ms: int
    policy_generation: int

    def usable(self) -> bool:
        return (
            bool(self.provider_identity)
            and self.source_field == "pull_request.merged_at"
            and self.occurrence_semantics in {"truncated", "exact"}
            and self.max_reported_resolution_ms > 0
            and self.policy_generation > 0
        )

    def identity_digest(self) -> str:
        payload = {
            "max_reported_resolution_ms": self.max_reported_resolution_ms,
            "occurrence_semantics": self.occurrence_semantics,
            "policy_generation": self.policy_generation,
            "provider_identity": self.provider_identity,
            "source_field": self.source_field,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ClockRelationV1:
    provider_clock_domain: str
    local_clock_domain: str
    max_skew_ms: int
    verified: bool = False
    verified_at_local_time_ms: int | None = None
    valid_until_local_time_ms: int | None = None
    max_drift_ppm: int = 0

    def usable(self) -> bool:
        return (
            bool(self.provider_clock_domain)
            and bool(self.local_clock_domain)
            and self.provider_clock_domain != self.local_clock_domain
            and self.max_skew_ms >= 0
            and self.max_drift_ppm >= 0
            and self.verified
            and self.verified_at_local_time_ms is not None
            and self.valid_until_local_time_ms is not None
            and self.verified_at_local_time_ms >= 0
            and self.valid_until_local_time_ms >= self.verified_at_local_time_ms
        )

    def effective_skew_ms(self, at_local_time_ms: int) -> int | None:
        if not self.usable():
            return None
        if at_local_time_ms < self.verified_at_local_time_ms:
            return None
        if at_local_time_ms > self.valid_until_local_time_ms:
            return None
        elapsed_ms = at_local_time_ms - self.verified_at_local_time_ms
        drift_ms = (
            self.max_drift_ppm * elapsed_ms + 999_999
        ) // 1_000_000
        return self.max_skew_ms + drift_ms


@dataclass(frozen=True)
class ProviderWebhookEffectTimingV1:
    provider_event_time_ms: int | None
    provider_event_time_upper_ms: int | None
    provider_timestamp_policy: ProviderTimestampPolicyV1 | None
    provider_timestamp_policy_digest: str | None
    provider_delivery_time_ms: int | None
    local_reservation_time_ms: int | None
    local_dispatch_time_ms: int | None
    local_observation_time_ms: int | None
    local_reservation_monotonic_ns: int | None
    local_dispatch_monotonic_ns: int | None
    local_observation_monotonic_ns: int | None
    clock_relation: ClockRelationV1 | None

    @classmethod
    def from_observation(
        cls,
        observation: ProviderPullRequestMergeObservationV1,
        local_reservation_time_ms: int | None,
        local_dispatch_time_ms: int | None,
        clock_relation: ClockRelationV1 | None,
        provider_delivery_time_ms: int | None = None,
        timestamp_policy: ProviderTimestampPolicyV1 | None = None,
        local_reservation_monotonic_ns: int | None = None,
        local_dispatch_monotonic_ns: int | None = None,
    ) -> "ProviderWebhookEffectTimingV1":
        interval = (
            parse_provider_timestamp_interval_ms(
                observation.merged_at,
                occurrence_semantics=timestamp_policy.occurrence_semantics,
            )
            if timestamp_policy is not None and timestamp_policy.usable()
            else None
        )
        if (
            interval is not None
            and timestamp_policy is not None
            and interval[1] - interval[0] + 1
            > timestamp_policy.max_reported_resolution_ms
        ):
            interval = None
        return cls(
            provider_event_time_ms=interval[0] if interval is not None else None,
            provider_event_time_upper_ms=interval[1] if interval is not None else None,
            provider_timestamp_policy=timestamp_policy,
            provider_timestamp_policy_digest=(
                timestamp_policy.identity_digest()
                if timestamp_policy is not None
                else None
            ),
            provider_delivery_time_ms=provider_delivery_time_ms,
            local_reservation_time_ms=local_reservation_time_ms,
            local_dispatch_time_ms=local_dispatch_time_ms,
            local_observation_time_ms=observation.local_received_at_ms,
            local_reservation_monotonic_ns=local_reservation_monotonic_ns,
            local_dispatch_monotonic_ns=local_dispatch_monotonic_ns,
            local_observation_monotonic_ns=observation.local_received_monotonic_ns,
            clock_relation=clock_relation,
        )

    def classify(self) -> str:
        values = (
            self.provider_event_time_ms,
            self.provider_event_time_upper_ms,
            self.provider_delivery_time_ms,
            self.local_reservation_time_ms,
            self.local_dispatch_time_ms,
            self.local_observation_time_ms,
            self.local_reservation_monotonic_ns,
            self.local_dispatch_monotonic_ns,
            self.local_observation_monotonic_ns,
        )
        if any(value is not None and value < 0 for value in values):
            return "invalid-negative-time"
        if self.provider_event_time_ms is None:
            return "provider-event-time-missing"
        if self.provider_event_time_upper_ms is None:
            return "provider-event-time-upper-missing"
        if self.provider_event_time_upper_ms < self.provider_event_time_ms:
            return "invalid-provider-event-interval"
        if (
            self.provider_timestamp_policy is None
            or not self.provider_timestamp_policy.usable()
        ):
            return "provider-timestamp-policy-unusable"
        if (
            self.provider_timestamp_policy_digest
            != self.provider_timestamp_policy.identity_digest()
        ):
            return "provider-timestamp-policy-integrity-invalid"
        if self.local_reservation_time_ms is None:
            return "local-reservation-time-missing"
        if self.local_dispatch_time_ms is None:
            return "local-dispatch-time-missing"
        if self.local_observation_time_ms is None:
            return "local-observation-time-missing"
        if self.local_reservation_monotonic_ns is None:
            return "local-reservation-monotonic-time-missing"
        if self.local_dispatch_monotonic_ns is None:
            return "local-dispatch-monotonic-time-missing"
        if self.local_observation_monotonic_ns is None:
            return "local-observation-monotonic-time-missing"
        if self.local_dispatch_time_ms < self.local_reservation_time_ms:
            return "invalid-local-time-order"
        if self.local_observation_time_ms < self.local_dispatch_time_ms:
            return "invalid-local-time-order"
        if self.local_dispatch_monotonic_ns < self.local_reservation_monotonic_ns:
            return "invalid-local-monotonic-order"
        if self.local_observation_monotonic_ns < self.local_dispatch_monotonic_ns:
            return "invalid-local-monotonic-order"
        if self.provider_delivery_time_ms is not None:
            if self.provider_delivery_time_ms < self.provider_event_time_ms:
                return "invalid-provider-time-order"
        if self.clock_relation is None or not self.clock_relation.usable():
            return "cross-domain-time-unbounded"

        skew = self.clock_relation.effective_skew_ms(
            self.local_observation_time_ms
        )
        if skew is None:
            return "clock-relation-invalid-at-observation"

        event_lower = self.provider_event_time_ms
        event_upper = self.provider_event_time_upper_ms
        dispatch = self.local_dispatch_time_ms
        observed = self.local_observation_time_ms

        if event_upper + skew < dispatch:
            return "provider-event-before-dispatch"
        if event_lower - skew > observed:
            return "provider-event-after-observation"
        if event_lower - skew < dispatch or event_upper + skew > observed:
            return "cross-domain-time-uncertain"
        return "temporally-admissible"

    def temporally_admissible(self) -> bool:
        return self.classify() == "temporally-admissible"


@dataclass(frozen=True)
class PromotionStackEffectTimingV1:
    pr_number: int
    source_effect_evidence_identity_digest: str
    timing: ProviderWebhookEffectTimingV1


@dataclass(frozen=True)
class PromotionStackEffectTimingSetV1:
    operation_identity_digest: str
    timestamp_policy_identity_digest: str
    clock_relation_identity_digest: str
    timings: tuple[PromotionStackEffectTimingV1, ...]

    def validates_complete(
        self,
        identity: PromotionOperationIdentityV1,
        effect_evidence: "PromotionStackEffectEvidenceSetV1 | None",
    ) -> bool:
        if self.operation_identity_digest != identity.digest():
            return False
        if not self.timestamp_policy_identity_digest:
            return False
        if not self.clock_relation_identity_digest:
            return False
        if effect_evidence is None or not effect_evidence.validates_complete(identity):
            return False
        expected = identity.ordered_stack
        if len(self.timings) != len(expected):
            return False
        if len(effect_evidence.effects) != len(expected):
            return False
        observed_prs = [item.pr_number for item in self.timings]
        if len(observed_prs) != len(set(observed_prs)):
            return False
        return all(
            item.pr_number == entry.pr_number
            and item.source_effect_evidence_identity_digest == evidence.identity_digest()
            and item.timing.provider_timestamp_policy_digest
            == self.timestamp_policy_identity_digest
            and item.timing.clock_relation is not None
            and item.timing.clock_relation.identity_digest()
            == self.clock_relation_identity_digest
            and item.timing.temporally_admissible()
            for entry, item, evidence in zip(
                expected,
                self.timings,
                effect_evidence.effects,
            )
        )


@dataclass(frozen=True)
class PromotionStackEffectEvidenceV1:
    effect: PromotionStackEffectV1
    source_delivery_id: str
    source_payload_digest: str
    source_hook_id: str
    source_event_type: str
    source_repository: str
    source_authentication: str

    def identity_digest(self) -> str:
        payload = {
            "effect": {
                "observed_merge_commit": self.effect.observed_merge_commit,
                "pr_number": self.effect.pr_number,
                "expected_head_sha": self.effect.expected_head_sha,
            },
            "source_authentication": self.source_authentication,
            "source_delivery_id": self.source_delivery_id,
            "source_event_type": self.source_event_type,
            "source_hook_id": self.source_hook_id,
            "source_payload_digest": self.source_payload_digest,
            "source_repository": self.source_repository,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    def validates(self, identity: PromotionOperationIdentityV1) -> bool:
        expected_prs = {entry.pr_number for entry in identity.ordered_stack}
        return (
            self.effect.pr_number in expected_prs
            and bool(self.effect.expected_head_sha)
            and bool(self.effect.observed_merge_commit)
            and bool(self.source_delivery_id)
            and bool(self.source_payload_digest)
            and bool(self.source_hook_id)
            and self.source_event_type == "pull_request"
            and self.source_repository == identity.repository
            and self.source_authentication == "webhook-hmac-verified"
        )


@dataclass(frozen=True)
class PromotionStackEffectEvidenceSetV1:
    operation_identity_digest: str
    effects: tuple[PromotionStackEffectEvidenceV1, ...]

    @classmethod
    def from_observations(
        cls,
        identity: PromotionOperationIdentityV1,
        observations: tuple[ProviderPullRequestMergeObservationV1, ...],
    ) -> "PromotionStackEffectEvidenceSetV1 | None":
        expected = identity.ordered_stack
        expected_prs = {entry.pr_number for entry in expected}
        by_pr: dict[int, ProviderPullRequestMergeObservationV1] = {}
        by_delivery: dict[str, ProviderPullRequestMergeObservationV1] = {}

        for observation in observations:
            if observation.repository != identity.repository:
                return None
            if observation.pr_number in by_pr:
                return None
            if not observation.delivery_id or observation.delivery_id in by_delivery:
                return None
            if not observation.has_authenticated_source_provenance():
                return None
            by_pr[observation.pr_number] = observation
            by_delivery[observation.delivery_id] = observation

        if set(by_pr) != expected_prs:
            return None

        evidence: list[PromotionStackEffectEvidenceV1] = []
        for entry in expected:
            observation = by_pr.get(entry.pr_number)
            if observation is None:
                return None
            item = observation.to_provenance_effect(entry)
            if item is None or not item.validates(identity):
                return None
            evidence.append(item)

        return cls(
            operation_identity_digest=identity.digest(),
            effects=tuple(evidence),
        )

    def validates_complete(self, identity: PromotionOperationIdentityV1) -> bool:
        if self.operation_identity_digest != identity.digest():
            return False
        expected = identity.ordered_stack
        if len(self.effects) != len(expected):
            return False

        prs = [item.effect.pr_number for item in self.effects]
        deliveries = [item.source_delivery_id for item in self.effects]
        if len(prs) != len(set(prs)):
            return False
        if len(deliveries) != len(set(deliveries)):
            return False

        return all(
            item.validates(identity)
            and item.effect.pr_number == entry.pr_number
            and item.effect.expected_head_sha == entry.head_sha
            for entry, item in zip(expected, self.effects)
        )

    def to_compact_effect_set(
        self,
        identity: PromotionOperationIdentityV1,
    ) -> PromotionStackEffectSetV1 | None:
        if not self.validates_complete(identity):
            return None
        return PromotionStackEffectSetV1(
            operation_identity_digest=self.operation_identity_digest,
            effects=tuple(item.effect for item in self.effects),
        )


@dataclass(frozen=True)
class ProviderDeliveryRegistryV1:
    deliveries: tuple[ProviderDeliveryIdentityV1, ...] = ()

    def _identity(self, receipt: ProviderWebhookReceiptV1) -> ProviderDeliveryIdentityV1:
        return ProviderDeliveryIdentityV1(
            delivery_id=receipt.delivery_id,
            payload_bytes_digest=receipt.payload_bytes_digest,
            hook_id=receipt.hook_id,
            event_type=receipt.event_type,
            repository=receipt.repository,
        )

    def observe(self, receipt: ProviderWebhookReceiptV1) -> str:
        identity = self._identity(receipt)
        for prior in self.deliveries:
            if prior.delivery_id != identity.delivery_id:
                continue
            if prior.matches(identity):
                return "duplicate-identical"
            return "delivery-id-conflict"
        return "new-delivery"

    def record(self, receipt: ProviderWebhookReceiptV1) -> "ProviderDeliveryRegistryV1":
        classification = self.observe(receipt)
        if classification == "new-delivery":
            return ProviderDeliveryRegistryV1(
                self.deliveries + (self._identity(receipt),)
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
    assert capture.raw_bytes_digest == hashlib.sha256(
        b"provider-result"
    ).hexdigest()
    assert capture.is_valid()


def test_capture_integrity_rejects_empty_storage_identity():
    capture = ProviderCaptureIntegrityV1("digest", "", 1, durable=True)
    assert not capture.is_valid()


def test_capture_integrity_rejects_nonpositive_sequence():
    capture = ProviderCaptureIntegrityV1("D", "store-1", 0, durable=True)
    assert not capture.is_valid()


def test_source_authentication_api_channel_is_distinct_from_attestation():
    evidence = provider_evidence_fixture()
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
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    context = ProviderWebhookRequestContextV1(
        delivery_id="delivery-1",
        hook_id="hook-1",
        event_type="pull_request",
        repository="Luminous-Dynamics/symthaea",
    )
    assert receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )


def test_webhook_hmac_verification_rejects_tampered_payload():
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-2",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    context = ProviderWebhookRequestContextV1(
        "delivery-2",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        b'{"action":"opened"}',
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )


def test_webhook_hmac_verification_rejects_wrong_secret():
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-3",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    context = ProviderWebhookRequestContextV1(
        "delivery-3",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"wrong",
        context,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )


def webhook_received_context(
    receipt: ProviderWebhookReceiptV1,
    received_at_ms: int | None = None,
    received_monotonic_ns: int | None = None,
) -> ProviderWebhookRequestContextV1:
    return ProviderWebhookRequestContextV1(
        delivery_id=receipt.delivery_id,
        hook_id=receipt.hook_id,
        event_type=receipt.event_type,
        repository=receipt.repository,
        received_at_ms=received_at_ms,
        received_monotonic_ns=received_monotonic_ns,
    )


def webhook_merge_payload(
    *,
    repository: str = "Luminous-Dynamics/symthaea",
    pr_number: int = 7087,
    head_sha: str = "H3",
    merged: bool = True,
    action: str = "closed",
    merge_commit: str | None = None,
    extra_field: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "action": action,
            "number": pr_number,
            "pull_request": {
                "number": pr_number,
                "merged": merged,
                "head": {"sha": head_sha},
                "merge_commit_sha": (
                    merge_commit
                    if merge_commit is not None
                    else ("M2" if merged else None)
                ),
                "merged_at": (
                    "2026-10-08T16:00:00Z"
                    if merged
                    else None
                ),
                "stack": {
                    "number": 41,
                    "size": 3,
                    "position": 2,
                    "base": {
                        "ref": "main",
                        "sha": "T1",
                    },
                },
            },
            "repository": {"full_name": repository},
            **(
                {"qualification_extra": extra_field}
                if extra_field is not None
                else {}
            ),
        },
        separators=(",", ":"),
    ).encode("utf-8")

def webhook_merge_receipt(
    payload: bytes,
    *,
    delivery_id: str = "delivery-merge",
    repository: str = "Luminous-Dynamics/symthaea",
    event_type: str = "pull_request",
) -> ProviderWebhookReceiptV1:
    return ProviderWebhookReceiptV1.from_delivery(
        delivery_id,
        "hook-1",
        event_type,
        repository,
        payload,
        b"secret",
    )



def stack_webhook_observation(
    identity: PromotionOperationIdentityV1 | None = None,
    *,
    pr_number: int = 7087,
    head_sha: str = "H3",
    merge_commit: str = "M2",
    delivery_id: str = "delivery-merge",
    repository: str = "Luminous-Dynamics/symthaea",
    payload_extra: str | None = None,
    received_at_ms: int = 1791475205000,
    received_monotonic_ns: int = 9000000000000,
) -> ProviderPullRequestMergeObservationV1:
    identity = identity or stack_identity_fixture()
    payload = webhook_merge_payload(
        repository=repository,
        pr_number=pr_number,
        head_sha=head_sha,
        merged=True,
        action="closed",
        merge_commit=merge_commit,
        extra_field=payload_extra,
    )
    receipt = webhook_merge_receipt(
        payload,
        delivery_id=delivery_id,
        repository=repository,
    )
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt, received_at_ms, received_monotonic_ns),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    return observation


def timestamp_policy_fixture(
    *,
    provider_identity: str = "github",
    source_field: str = "pull_request.merged_at",
    occurrence_semantics: str = "truncated",
    max_reported_resolution_ms: int = 1000,
    policy_generation: int = 1,
) -> ProviderTimestampPolicyV1:
    return ProviderTimestampPolicyV1(
        provider_identity=provider_identity,
        source_field=source_field,
        occurrence_semantics=occurrence_semantics,
        max_reported_resolution_ms=max_reported_resolution_ms,
        policy_generation=policy_generation,
    )


def clock_relation_fixture(
    *,
    max_skew_ms: int = 1000,
    verified: bool = True,
    verified_at_local_time_ms: int = 1791475190000,
    valid_until_local_time_ms: int = 1791478800000,
    max_drift_ppm: int = 0,
) -> ClockRelationV1:
    return ClockRelationV1(
        provider_clock_domain="github",
        local_clock_domain="local",
        max_skew_ms=max_skew_ms,
        verified=verified,
        verified_at_local_time_ms=verified_at_local_time_ms,
        valid_until_local_time_ms=valid_until_local_time_ms,
        max_drift_ppm=max_drift_ppm,
    )


def effect_timing_fixture(
    *,
    event_time_ms: int | None = 1791475200000,
    event_upper_time_ms: int | None = 1791475200999,
    timestamp_policy: ProviderTimestampPolicyV1 | None = None,
    delivery_time_ms: int | None = None,
    reservation_time_ms: int | None = 1791475190000,
    dispatch_time_ms: int | None = 1791475195000,
    observation_time_ms: int | None = 1791475205000,
    reservation_monotonic_ns: int | None = 8999999000000,
    dispatch_monotonic_ns: int | None = 8999999500000,
    observation_monotonic_ns: int | None = 9000000000000,
    clock_relation: ClockRelationV1 | None = None,
) -> ProviderWebhookEffectTimingV1:
    return ProviderWebhookEffectTimingV1(
        provider_event_time_ms=event_time_ms,
        provider_event_time_upper_ms=event_upper_time_ms,
        provider_timestamp_policy=(
            timestamp_policy
            if timestamp_policy is not None
            else timestamp_policy_fixture()
        ),
        provider_timestamp_policy_digest=(
            (
                timestamp_policy
                if timestamp_policy is not None
                else timestamp_policy_fixture()
            ).identity_digest()
        ),
        provider_delivery_time_ms=delivery_time_ms,
        local_reservation_time_ms=reservation_time_ms,
        local_dispatch_time_ms=dispatch_time_ms,
        local_observation_time_ms=observation_time_ms,
        local_reservation_monotonic_ns=reservation_monotonic_ns,
        local_dispatch_monotonic_ns=dispatch_monotonic_ns,
        local_observation_monotonic_ns=observation_monotonic_ns,
        clock_relation=(
            clock_relation if clock_relation is not None else clock_relation_fixture()
        ),
    )


def webhook_effect_timing_from_observation(
    identity: PromotionOperationIdentityV1 | None = None,
) -> ProviderWebhookEffectTimingV1:
    identity = identity or stack_identity_fixture()
    observation = stack_webhook_observation(
        identity,
        received_at_ms=1791475205000,
    )
    return ProviderWebhookEffectTimingV1.from_observation(
        observation,
        local_reservation_time_ms=1791475190000,
        local_dispatch_time_ms=1791475195000,
        local_reservation_monotonic_ns=8999999000000,
        local_dispatch_monotonic_ns=8999999500000,
        clock_relation=clock_relation_fixture(),
        timestamp_policy=timestamp_policy_fixture(),
    )


def effect_state_fixture(
    identity: PromotionOperationIdentityV1 | None = None,
) -> PromotionPrEffectStateV1:
    identity = identity or stack_identity_fixture()
    entry = identity.ordered_stack[-1]
    return PromotionPrEffectStateV1(
        repository=identity.repository,
        expected_entry=entry,
        operation_identity_digest=identity.digest(),
    )


def test_webhook_stack_metadata_matches_exact_operation():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    assert observation.provider_stack is not None
    assert observation.validates_provider_stack_metadata(identity)


def test_webhook_stack_metadata_rejects_stack_number_drift():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    altered = ProviderPullRequestMergeObservationV1(
        **{
            **observation.__dict__,
            "provider_stack": ProviderWebhookStackMetadataV1(
                stack_number=99,
                stack_size=3,
                stack_position=2,
                base_ref="main",
                base_sha="T1",
            ),
        }
    )
    assert not altered.validates_provider_stack_metadata(identity)


def test_webhook_stack_metadata_rejects_base_sha_drift():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    altered = ProviderPullRequestMergeObservationV1(
        **{
            **observation.__dict__,
            "provider_stack": ProviderWebhookStackMetadataV1(
                stack_number=41,
                stack_size=3,
                stack_position=2,
                base_ref="main",
                base_sha="T-DRIFT",
            ),
        }
    )
    assert not altered.validates_provider_stack_metadata(identity)


def test_webhook_stack_metadata_missing_is_not_topology_proof():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    altered = ProviderPullRequestMergeObservationV1(
        **{
            **observation.__dict__,
            "provider_stack": None,
        }
    )
    assert not altered.validates_provider_stack_metadata(identity)


def test_provider_timestamp_parser_accepts_utc_and_offset():
    assert parse_provider_timestamp_ms("2026-10-08T16:00:00Z") == parse_provider_timestamp_ms(
        "2026-10-08T18:00:00+02:00"
    )


def test_provider_timestamp_interval_respects_reported_precision():
    assert parse_provider_timestamp_interval_ms(
        "2026-10-08T16:00:00Z",
        occurrence_semantics="truncated",
    ) == (
        parse_provider_timestamp_ms("2026-10-08T16:00:00Z"),
        parse_provider_timestamp_ms("2026-10-08T16:00:00Z") + 999,
    )
    assert parse_provider_timestamp_interval_ms(
        "2026-10-08T16:00:00.12Z",
        occurrence_semantics="truncated",
    )[1] == parse_provider_timestamp_ms("2026-10-08T16:00:00.12Z") + 9


def test_provider_timestamp_interval_rejects_unsupported_precision_or_semantics():
    assert parse_provider_timestamp_interval_ms(
        "2026-10-08T16:00:00.1234Z",
        occurrence_semantics="truncated",
    ) is None
    assert parse_provider_timestamp_interval_ms(
        "2026-10-08T16:00:00Z",
        occurrence_semantics="unknown",
    ) is None


def test_timestamp_policy_binds_provider_source_and_interpretation():
    policy = timestamp_policy_fixture()
    assert policy.usable()
    assert policy.identity_digest()
    assert not timestamp_policy_fixture(source_field="pull_request.closed_at").usable()
    assert not timestamp_policy_fixture(occurrence_semantics="unknown").usable()


def test_timestamp_policy_resolution_limit_is_enforced():
    timing = effect_timing_fixture(
        timestamp_policy=timestamp_policy_fixture(max_reported_resolution_ms=500),
    )
    assert timing.provider_event_time_ms is not None
    assert timing.classify() == "provider-event-time-missing"


def test_temporal_effect_rejects_tampered_timestamp_policy_identity():
    policy = timestamp_policy_fixture()
    timing = effect_timing_fixture(timestamp_policy=policy)
    tampered = ProviderWebhookEffectTimingV1(
        provider_event_time_ms=timing.provider_event_time_ms,
        provider_event_time_upper_ms=timing.provider_event_time_upper_ms,
        provider_timestamp_policy=timestamp_policy_fixture(policy_generation=2),
        provider_timestamp_policy_digest=timing.provider_timestamp_policy_digest,
        provider_delivery_time_ms=timing.provider_delivery_time_ms,
        local_reservation_time_ms=timing.local_reservation_time_ms,
        local_dispatch_time_ms=timing.local_dispatch_time_ms,
        local_observation_time_ms=timing.local_observation_time_ms,
        local_reservation_monotonic_ns=timing.local_reservation_monotonic_ns,
        local_dispatch_monotonic_ns=timing.local_dispatch_monotonic_ns,
        local_observation_monotonic_ns=timing.local_observation_monotonic_ns,
        clock_relation=timing.clock_relation,
    )
    assert tampered.classify() == "provider-timestamp-policy-integrity-invalid"


def test_stack_timing_rejects_mixed_clock_relation_identities():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-timing-clock-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-timing-clock-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    first_timing = effect_timing_fixture()
    second_timing = effect_timing_fixture(
        clock_relation=clock_relation_fixture(max_skew_ms=2000),
    )
    timings = PromotionStackEffectTimingSetV1(
        identity.digest(),
        first_timing.provider_timestamp_policy_digest,
        first_timing.clock_relation.identity_digest(),
        (
            PromotionStackEffectTimingV1(
                7085,
                evidence.effects[0].identity_digest(),
                first_timing,
            ),
            PromotionStackEffectTimingV1(
                7087,
                evidence.effects[1].identity_digest(),
                second_timing,
            ),
        ),
    )
    assert not timings.validates_complete(identity, evidence)


def test_stack_timing_rejects_mixed_timestamp_policy_identities():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-timing-policy-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-timing-policy-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    first_timing = effect_timing_fixture(
        timestamp_policy=timestamp_policy_fixture(policy_generation=1),
    )
    second_timing = effect_timing_fixture(
        timestamp_policy=timestamp_policy_fixture(policy_generation=2),
    )
    timings = PromotionStackEffectTimingSetV1(
        identity.digest(),
        first_timing.provider_timestamp_policy_digest,
        first_timing.clock_relation.identity_digest(),
        (
            PromotionStackEffectTimingV1(
                7085,
                evidence.effects[0].identity_digest(),
                first_timing,
            ),
            PromotionStackEffectTimingV1(
                7087,
                evidence.effects[1].identity_digest(),
                second_timing,
            ),
        ),
    )
    assert not timings.validates_complete(identity, evidence)


def test_temporal_effect_without_timestamp_policy_is_not_admissible():
    timing = effect_timing_fixture(timestamp_policy=None)
    timing = ProviderWebhookEffectTimingV1(
        provider_event_time_ms=timing.provider_event_time_ms,
        provider_event_time_upper_ms=timing.provider_event_time_upper_ms,
        provider_timestamp_policy=None,
        provider_timestamp_policy_digest=None,
        provider_delivery_time_ms=timing.provider_delivery_time_ms,
        local_reservation_time_ms=timing.local_reservation_time_ms,
        local_dispatch_time_ms=timing.local_dispatch_time_ms,
        local_observation_time_ms=timing.local_observation_time_ms,
        local_reservation_monotonic_ns=timing.local_reservation_monotonic_ns,
        local_dispatch_monotonic_ns=timing.local_dispatch_monotonic_ns,
        local_observation_monotonic_ns=timing.local_observation_monotonic_ns,
        clock_relation=timing.clock_relation,
    )
    assert timing.classify() == "provider-timestamp-policy-unusable"


def test_provider_timestamp_parser_rejects_malformed_timestamp():
    assert parse_provider_timestamp_ms("not-a-timestamp") is None
    assert parse_provider_timestamp_ms("2026-10-08T16:00:00") is None


def test_clock_relation_unverified_is_unusable():
    assert not clock_relation_fixture(verified=False).usable()


def test_clock_relation_negative_skew_is_unusable():
    assert not clock_relation_fixture(max_skew_ms=-1, verified=True).usable()


def test_clock_relation_requires_explicit_validity_window():
    relation = clock_relation_fixture(
        verified_at_local_time_ms=None,
        valid_until_local_time_ms=None,
    )
    assert not relation.usable()


def test_clock_relation_rejects_expiry_before_verification():
    relation = clock_relation_fixture(
        verified_at_local_time_ms=10,
        valid_until_local_time_ms=9,
    )
    assert not relation.usable()


def test_clock_relation_expiry_blocks_late_observation():
    relation = clock_relation_fixture(
        verified_at_local_time_ms=1791475190000,
        valid_until_local_time_ms=1791475200000,
    )
    assert relation.effective_skew_ms(1791475200000) == 1000
    assert relation.effective_skew_ms(1791475200001) is None


def test_clock_relation_drift_expands_uncertainty_monotonically():
    relation = clock_relation_fixture(
        max_skew_ms=100,
        max_drift_ppm=1000,
        valid_until_local_time_ms=1791475290000,
    )
    assert relation.effective_skew_ms(1791475200000) == 1100
    assert relation.effective_skew_ms(1791475210000) == 1110


def test_temporal_effect_with_valid_skew_is_admissible():
    timing = effect_timing_fixture()
    assert timing.classify() == "temporally-admissible"
    assert timing.temporally_admissible()




def test_temporal_effect_interval_overlap_is_not_admissible():
    timing = effect_timing_fixture(
        event_time_ms=1791475200000,
        event_upper_time_ms=1791475200999,
        dispatch_time_ms=1791475200500,
        observation_time_ms=1791475202000,
        clock_relation=clock_relation_fixture(max_skew_ms=600),
    )
    assert timing.classify() == "cross-domain-time-uncertain"
    assert not timing.temporally_admissible()


def test_temporal_effect_expired_clock_relation_is_not_admissible():
    timing = effect_timing_fixture(
        observation_time_ms=1791476000000,
        clock_relation=clock_relation_fixture(
            valid_until_local_time_ms=1791475999999,
        ),
    )
    assert timing.classify() == "clock-relation-invalid-at-observation"


def test_temporal_effect_drift_can_turn_boundary_into_uncertainty():
    timing = effect_timing_fixture(
        event_time_ms=1791475200000,
        event_upper_time_ms=1791475200999,
        dispatch_time_ms=1791475199000,
        observation_time_ms=1791475203000,
        clock_relation=clock_relation_fixture(
            max_skew_ms=1,
            max_drift_ppm=1000,
            valid_until_local_time_ms=1791475203000,
        ),
    )
    assert timing.classify() == "cross-domain-time-uncertain"


def test_temporal_timing_rejects_missing_local_monotonic_time():
    timing = effect_timing_fixture(
        observation_monotonic_ns=None,
    )
    assert timing.classify() == "local-observation-monotonic-time-missing"


def test_temporal_timing_rejects_local_monotonic_rollback():
    timing = effect_timing_fixture(
        observation_monotonic_ns=8999999400000,
        dispatch_monotonic_ns=8999999500000,
    )
    assert timing.classify() == "invalid-local-monotonic-order"


def test_temporal_timing_rejects_wall_clock_valid_but_monotonic_invalid_order():
    timing = effect_timing_fixture(
        reservation_time_ms=1791475190000,
        dispatch_time_ms=1791475195000,
        observation_time_ms=1791475205000,
        reservation_monotonic_ns=8999999000000,
        dispatch_monotonic_ns=9000000500000,
        observation_monotonic_ns=9000000000000,
    )
    assert timing.classify() == "invalid-local-monotonic-order"


def test_temporal_effect_without_clock_relation_is_unbounded():
    timing = effect_timing_fixture(clock_relation=None)
    assert timing.classify() == "cross-domain-time-unbounded"


def test_temporal_effect_with_event_before_dispatch_is_rejected():
    timing = effect_timing_fixture(
        event_time_ms=1728400000000,
        reservation_time_ms=1728402000000,
        dispatch_time_ms=1728402500000,
    )
    assert timing.classify() == "provider-event-before-dispatch"


def test_temporal_effect_with_event_after_observation_is_rejected():
    timing = effect_timing_fixture(
        event_time_ms=1728405000000,
        observation_time_ms=1728403200000,
    )
    assert timing.classify() == "provider-event-after-observation"


def test_temporal_effect_with_uncertain_clock_overlap_is_not_admissible():
    timing = effect_timing_fixture(
        event_time_ms=1728402949500,
        dispatch_time_ms=1728402950000,
        clock_relation=clock_relation_fixture(max_skew_ms=1000),
    )
    assert timing.classify() == "cross-domain-time-uncertain"
    assert not timing.temporally_admissible()


def test_temporal_effect_accepts_exact_skew_boundaries():
    timing = effect_timing_fixture(
        event_time_ms=1791475200000,
        dispatch_time_ms=1791475199000,
        observation_time_ms=1791475201000,
        clock_relation=clock_relation_fixture(max_skew_ms=1000),
    )
    assert timing.classify() == "temporally-admissible"


def test_provider_delivery_time_before_event_is_invalid():
    timing = effect_timing_fixture(
        delivery_time_ms=1728402959000,
        event_time_ms=1728402960000,
    )
    assert timing.classify() == "invalid-provider-time-order"


def test_temporal_timing_can_be_derived_from_authenticated_observation():
    timing = webhook_effect_timing_from_observation()
    assert timing.provider_event_time_ms == parse_provider_timestamp_ms(
        "2026-10-08T16:00:00Z"
    )
    assert timing.local_observation_time_ms == 1791475205000
    assert timing.temporally_admissible()


def test_historical_merge_delivered_after_new_reservation_is_inadmissible():
    timing = effect_timing_fixture(
        event_time_ms=1791475000000,
        reservation_time_ms=1791475300000,
        dispatch_time_ms=1791475350000,
        observation_time_ms=1791475500000,
        clock_relation=clock_relation_fixture(max_skew_ms=1000),
    )
    assert timing.classify() == "provider-event-before-dispatch"


def test_temporal_timing_rejects_missing_event_time():
    timing = effect_timing_fixture(event_time_ms=None)
    assert timing.classify() == "provider-event-time-missing"


def test_temporal_timing_rejects_invalid_local_order():
    timing = effect_timing_fixture(
        reservation_time_ms=1728403100000,
        dispatch_time_ms=1728403000000,
    )
    assert timing.classify() == "invalid-local-time-order"


def test_temporal_timing_rejects_local_observation_rollback():
    timing = effect_timing_fixture(
        observation_time_ms=1728403000000,
        dispatch_time_ms=1728403100000,
    )
    assert timing.classify() == "invalid-local-time-order"


def test_complete_stack_timing_requires_every_member_admissible():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-timing-complete-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-timing-complete-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    good = PromotionStackEffectTimingV1(
        7085,
        evidence.effects[0].identity_digest(),
        effect_timing_fixture(),
    )
    bad = PromotionStackEffectTimingV1(
        7087,
        evidence.effects[1].identity_digest(),
        effect_timing_fixture(event_time_ms=1791475000000),
    )
    timings = PromotionStackEffectTimingSetV1(
        identity.digest(),
        timestamp_policy_fixture().identity_digest(),
        clock_relation_fixture().identity_digest(),
        (good, bad),
    )
    assert not timings.validates_complete(identity, evidence)


def test_stack_timing_rejects_crosswired_effect_evidence_identity():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-timing-crosswire-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-timing-crosswire-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    crosswired = PromotionStackEffectTimingSetV1(
        identity.digest(),
        timestamp_policy_fixture().identity_digest(),
        clock_relation_fixture().identity_digest(),
        (
            PromotionStackEffectTimingV1(
                7085,
                evidence.effects[1].identity_digest(),
                effect_timing_fixture(),
            ),
            PromotionStackEffectTimingV1(
                7087,
                evidence.effects[0].identity_digest(),
                effect_timing_fixture(),
            ),
        ),
    )
    assert not crosswired.validates_complete(identity, evidence)


def test_effect_state_binds_to_exact_operation_identity():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    assert state.validates_operation_identity(identity)

    changed_identity = PromotionOperationIdentityV1(
        repository=identity.repository,
        provider_stack_number=identity.provider_stack_number,
        requested_pr_number=identity.requested_pr_number,
        requested_pr_head_sha=identity.requested_pr_head_sha,
        base_ref=identity.base_ref,
        base_tip_sha="BASE-OTHER",
        ordered_stack=identity.ordered_stack,
        merge_method=identity.merge_method,
        merge_action=identity.merge_action,
        trust_root_generation=identity.trust_root_generation,
        governance_generation=identity.governance_generation,
    )
    assert not state.validates_operation_identity(changed_identity)

def test_effect_state_same_delivery_id_with_changed_payload_fails_closed():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(
        identity,
        delivery_id="delivery-same-id",
        payload_extra="first-payload",
    )
    conflicting_payload = stack_webhook_observation(
        identity,
        delivery_id="delivery-same-id",
        payload_extra="second-payload",
    )
    assert first.payload_bytes_digest != conflicting_payload.payload_bytes_digest
    assert first.to_stack_effect(identity) == conflicting_payload.to_stack_effect(identity)

    state, decision = state.ingest(first)
    assert decision == "admitted"
    state_after, decision = state.ingest(conflicting_payload)
    assert decision == "delivery-identity-conflict"
    assert state_after.state == "Conflict"
    assert state_after.effect is None

def test_effect_state_rejects_inconsistent_source_delivery_bindings():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    state = PromotionPrEffectStateV1(
        repository=identity.repository,
        expected_entry=identity.ordered_stack[-1],
        operation_identity_digest=identity.digest(),
        state="EffectObserved",
        effect=observation.to_stack_effect(identity),
        source_delivery_ids=(observation.delivery_id,),
    )
    assert not state.validates_source_delivery_bindings()
    assert not state.is_terminally_observed()

def test_effect_state_conflict_is_absorbing():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(
        identity,
        delivery_id="delivery-absorbing-first",
        merge_commit="M2",
    )
    conflicting = stack_webhook_observation(
        identity,
        delivery_id="delivery-absorbing-conflict",
        merge_commit="M9",
    )
    compatible = stack_webhook_observation(
        identity,
        delivery_id="delivery-absorbing-late",
        merge_commit="M2",
    )
    state, _ = state.ingest(first)
    state, decision = state.ingest(conflicting)
    assert decision == "conflict"
    assert state.state == "Conflict"
    state_after, decision = state.ingest(compatible)
    assert decision == "rejected-invalid-state"
    assert state_after.state == "Conflict"
    assert state_after.effect is None

def test_effect_state_conflict_never_reclassifies_as_terminal_observed():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(identity, delivery_id="delivery-reclass-first")
    conflicting = stack_webhook_observation(
        identity,
        delivery_id="delivery-reclass-conflict",
        head_sha="H0",
    )
    state, _ = state.ingest(first)
    state, decision = state.ingest(conflicting)
    assert decision == "conflict"
    assert not state.is_terminally_observed()
    assert state.effect is None

def test_webhook_delivery_registry_rejects_same_id_with_changed_context():
    payload = b"{}"
    first = ProviderWebhookReceiptV1.from_delivery(
        "delivery-context-reuse",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    second = ProviderWebhookReceiptV1.from_delivery(
        "delivery-context-reuse",
        "hook-2",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    registry = ProviderDeliveryRegistryV1().record(first)
    assert registry.observe(second) == "delivery-id-conflict"

def test_effect_state_admits_first_authenticated_merge():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    observation = stack_webhook_observation(identity)
    state, decision = state.ingest(observation)
    assert decision == "admitted"
    assert state.is_terminally_observed()
    assert state.effect is not None
    assert state.effect.observed_merge_commit == "M2"


def test_effect_state_merged_then_non_effect_does_not_downgrade():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    merged = stack_webhook_observation(identity)
    state, decision = state.ingest(merged)
    assert decision == "admitted"
    state, decision = state.ingest(None)
    assert decision == "no-eligible-effect"
    assert state.is_terminally_observed()
    assert state.effect == merged.to_stack_effect(identity)


def test_effect_state_equivalent_second_delivery_is_idempotent():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(
        identity,
        delivery_id="delivery-first",
    )
    second = stack_webhook_observation(
        identity,
        delivery_id="delivery-second",
    )
    state, _ = state.ingest(first)
    state, decision = state.ingest(second)
    assert decision == "compatible-repeat"
    assert state.is_terminally_observed()
    assert state.source_delivery_ids == ("delivery-first", "delivery-second")


def test_effect_state_repeated_same_delivery_is_idempotent():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    observation = stack_webhook_observation(
        identity,
        delivery_id="delivery-repeat",
    )
    state, _ = state.ingest(observation)
    state_after, decision = state.ingest(observation)
    assert decision == "duplicate-delivery"
    assert state_after == state


def test_effect_state_conflicting_merge_commit_fails_closed():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(
        identity,
        delivery_id="delivery-conflict-first",
        merge_commit="M2",
    )
    conflicting = stack_webhook_observation(
        identity,
        delivery_id="delivery-conflict-second",
        merge_commit="M9",
    )
    state, _ = state.ingest(first)
    state, decision = state.ingest(conflicting)
    assert decision == "conflict"
    assert state.state == "Conflict"
    assert state.effect is None


def test_effect_state_conflicting_head_fails_closed():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(identity, delivery_id="delivery-head-first")
    conflicting = stack_webhook_observation(
        identity,
        delivery_id="delivery-head-second",
        head_sha="H0",
    )
    state, _ = state.ingest(first)
    state_after, decision = state.ingest(conflicting)
    assert decision == "conflict"
    assert state_after.state == "Conflict"
    assert not state_after.is_terminally_observed()


def test_effect_state_untrusted_after_merge_does_not_downgrade():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(identity, delivery_id="delivery-trust-first")
    state, _ = state.ingest(first)
    untrusted = ProviderPullRequestMergeObservationV1(
        delivery_id="delivery-untrusted",
        repository=identity.repository,
        pr_number=identity.requested_pr_number,
        event_type="pull_request",
        action="closed",
        merged=True,
        head_sha=identity.requested_pr_head_sha,
        merge_commit_sha="M2",
        payload_bytes_digest="digest",
    )
    state_after, decision = state.ingest(untrusted)
    assert decision == "rejected-untrusted"
    assert state_after == state


def test_effect_state_unrelated_pr_does_not_downgrade():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(identity)
    state, _ = state.ingest(first)
    unrelated = stack_webhook_observation(
        identity,
        pr_number=7090,
        head_sha="H9",
        merge_commit="M9",
        delivery_id="delivery-unrelated",
    )
    state_after, decision = state.ingest(unrelated)
    assert decision == "ignored-unrelated"
    assert state_after == state


def test_effect_state_is_not_latest_delivery_wins():
    identity = stack_identity_fixture()
    state = effect_state_fixture(identity)
    first = stack_webhook_observation(
        identity,
        delivery_id="delivery-terminal",
    )
    state, _ = state.ingest(first)
    later_non_effect = None
    state_after, decision = state.ingest(later_non_effect)
    assert decision == "no-eligible-effect"
    assert state_after.effect == first.to_stack_effect(identity)
    assert state_after.state == "EffectObserved"


def test_webhook_effect_provenance_set_is_complete():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-prov-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    assert evidence.validates_complete(identity)


def test_webhook_effect_provenance_normalizes_input_order():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-order-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-prov-order-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (requested, bottom),
    )
    assert evidence is not None
    assert [item.effect.pr_number for item in evidence.effects] == [7085, 7087]


def test_webhook_effect_provenance_rejects_missing_source_provenance():
    identity = stack_identity_fixture()
    observation = stack_webhook_observation(identity)
    stripped = ProviderPullRequestMergeObservationV1(
        delivery_id=observation.delivery_id,
        repository=observation.repository,
        pr_number=observation.pr_number,
        event_type=observation.event_type,
        action=observation.action,
        merged=observation.merged,
        head_sha=observation.head_sha,
        merge_commit_sha=observation.merge_commit_sha,
        payload_bytes_digest=observation.payload_bytes_digest,
    )
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-missing-bottom",
    )
    assert PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, stripped),
    ) is None


def test_webhook_effect_provenance_rejects_duplicate_delivery_id():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-same",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-same",
    )
    assert PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    ) is None


def test_webhook_effect_provenance_rejects_mixed_repository():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-mixed-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        repository="other/repo",
        delivery_id="delivery-prov-mixed-requested",
    )
    assert PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    ) is None


def test_webhook_effect_provenance_rejects_effect_disagreement():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-effect-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-prov-effect-requested",
    )
    altered = PromotionStackEffectEvidenceV1(
        effect=PromotionStackEffectV1(7085, "H0", "M1"),
        source_delivery_id=bottom.delivery_id,
        source_payload_digest=bottom.payload_bytes_digest,
        source_hook_id=bottom.hook_id,
        source_event_type=bottom.event_type,
        source_repository=bottom.repository,
        source_authentication=bottom.source_authentication,
    )
    evidence = PromotionStackEffectEvidenceSetV1(
        operation_identity_digest=identity.digest(),
        effects=(altered, PromotionStackEffectEvidenceV1(
            effect=PromotionStackEffectV1(7087, "H3", "M2"),
            source_delivery_id=requested.delivery_id,
            source_payload_digest=requested.payload_bytes_digest,
            source_hook_id=requested.hook_id,
            source_event_type=requested.event_type,
            source_repository=requested.repository,
            source_authentication=requested.source_authentication,
        )),
    )
    assert not evidence.validates_complete(identity)


def test_webhook_effect_provenance_projects_to_compact_effect_set():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-prov-project-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-prov-project-requested",
    )
    evidence = PromotionStackEffectEvidenceSetV1.from_observations(
        identity,
        (bottom, requested),
    )
    assert evidence is not None
    compact = evidence.to_compact_effect_set(identity)
    assert compact is not None
    assert compact.validates_complete(identity)
    assert [item.pr_number for item in compact.effects] == [7085, 7087]


def test_webhook_derived_stack_effect_set_is_complete():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-bottom",
    )
    requested = stack_webhook_observation(identity)
    effects = PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested),
    )
    assert effects is not None
    assert effects.validates_complete(identity)


def test_webhook_derived_stack_effect_set_normalizes_input_order():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-bottom-order",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-requested-order",
    )
    effects = PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (requested, bottom),
    )
    assert effects is not None
    assert [effect.pr_number for effect in effects.effects] == [7085, 7087]


def test_webhook_derived_stack_effect_set_rejects_missing_member():
    identity = stack_identity_fixture()
    requested = stack_webhook_observation(identity)
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (requested,),
    ) is None


def test_webhook_derived_stack_effect_set_rejects_extra_member():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-extra-bottom",
    )
    requested = stack_webhook_observation(identity)
    extra = stack_webhook_observation(
        identity,
        pr_number=7090,
        head_sha="H9",
        merge_commit="M9",
        delivery_id="delivery-extra",
    )
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested, extra),
    ) is None


def test_webhook_derived_stack_effect_set_rejects_duplicate_pr():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-dup-bottom",
    )
    requested1 = stack_webhook_observation(
        identity,
        delivery_id="delivery-dup-requested-1",
    )
    requested2 = stack_webhook_observation(
        identity,
        delivery_id="delivery-dup-requested-2",
    )
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested1, requested2),
    ) is None


def test_webhook_derived_stack_effect_set_rejects_mixed_repository():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-mixed-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        repository="other/repo",
        delivery_id="delivery-mixed-requested",
    )
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested),
    ) is None


def test_webhook_derived_stack_effect_set_rejects_head_mismatch():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1-CHANGED",
        merge_commit="M1",
        delivery_id="delivery-head-mismatch",
    )
    requested = stack_webhook_observation(identity)
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested),
    ) is None


def test_webhook_derived_stack_effect_set_rejects_semantically_invalid_observation():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-invalid-bottom",
    )
    invalid = ProviderPullRequestMergeObservationV1(
        delivery_id="delivery-invalid-requested",
        repository=identity.repository,
        pr_number=7087,
        event_type="pull_request",
        action="closed",
        merged=False,
        head_sha="H3",
        merge_commit_sha="M2",
        payload_bytes_digest="digest",
    )
    assert PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, invalid),
    ) is None


def test_webhook_derived_effect_set_preserves_observation_only_semantics():
    identity = stack_identity_fixture()
    bottom = stack_webhook_observation(
        identity,
        pr_number=7085,
        head_sha="H1",
        merge_commit="M1",
        delivery_id="delivery-effect-bottom",
    )
    requested = stack_webhook_observation(
        identity,
        delivery_id="delivery-effect-requested",
    )
    effects = PromotionStackEffectSetV1.from_webhook_observations(
        identity,
        (bottom, requested),
    )
    assert effects is not None
    resolution = causal_resolution_fixture(
        identity,
        provider_result=None,
        effect_set=effects,
        provider_evidence=None,
    )
    assert resolution.outcome == "effect-observed-only"


def test_webhook_verify_rejects_empty_expected_context():
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    context = webhook_received_context(receipt)
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "",
    )


def test_webhook_parser_rejects_substituted_received_context():
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    context = ProviderWebhookRequestContextV1(
        receipt.delivery_id,
        "wrong-hook",
        receipt.event_type,
        receipt.repository,
    )
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_parses_authenticated_merged_pr():
    identity = stack_identity_fixture()
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    assert observation.validates_requested_effect(identity)
    effect = observation.to_stack_effect(identity)
    assert effect is not None
    assert effect.observed_merge_commit == "M2"


def test_webhook_merge_effect_rejects_invalid_hmac():
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    assert (
        ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
            receipt,
            payload,
            b"wrong",
            webhook_received_context(receipt),
            "hook-1",
            "pull_request",
            "Luminous-Dynamics/symthaea",
        )
        is None
    )


def test_webhook_merge_effect_rejects_non_pull_request_event():
    payload = webhook_merge_payload()
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-merge-event",
        "hook-1",
        "issues",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_rejects_non_closed_action():
    payload = webhook_merge_payload(action="opened")
    receipt = webhook_merge_receipt(payload)
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_rejects_closed_not_merged():
    payload = webhook_merge_payload(merged=False)
    receipt = webhook_merge_receipt(payload)
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_rejects_wrong_repository():
    payload = webhook_merge_payload(repository="other/repo")
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-wrong-repo",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_rejects_wrong_pr_number():
    identity = stack_identity_fixture()
    payload = webhook_merge_payload(pr_number=7090)
    receipt = webhook_merge_receipt(payload)
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    assert not observation.validates_requested_effect(identity)


def test_webhook_merge_effect_rejects_wrong_head():
    identity = stack_identity_fixture()
    payload = webhook_merge_payload(head_sha="H0")
    receipt = webhook_merge_receipt(payload)
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    assert not observation.validates_requested_effect(identity)


def test_webhook_merge_effect_rejects_missing_merge_commit():
    payload = json.dumps(
        {
            "action": "closed",
            "number": 7087,
            "pull_request": {
                "number": 7087,
                "merged": True,
                "head": {"sha": "H3"},
                "merge_commit_sha": "",
            },
            "repository": {"full_name": "Luminous-Dynamics/symthaea"},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    receipt = webhook_merge_receipt(payload)
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_rejects_tampered_payload_after_receipt():
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    tampered = payload.replace(b'"H3"', b'"H0"')
    assert ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        tampered,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    ) is None


def test_webhook_merge_effect_requires_exact_payload_digest():
    identity = stack_identity_fixture()
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    assert observation.payload_bytes_digest == receipt.payload_bytes_digest
    assert observation.validates_requested_effect(identity)


def test_webhook_merge_effect_does_not_establish_async_operation_causality():
    identity = stack_identity_fixture()
    payload = webhook_merge_payload()
    receipt = webhook_merge_receipt(payload)
    observation = ProviderPullRequestMergeObservationV1.from_authenticated_delivery(
        receipt,
        payload,
        b"secret",
        webhook_received_context(receipt),
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert observation is not None
    effect_set = PromotionStackEffectSetV1(
        operation_identity_digest=identity.digest(),
        effects=(
            PromotionStackEffectV1(7085, "H1", "M1"),
            PromotionStackEffectV1(7087, "H3", "M2"),
        ),
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=None,
        effect_set=effect_set,
        provider_evidence=None,
    )
    assert resolution.outcome == "effect-observed-only"


def test_webhook_delivery_registry_accepts_new_delivery():
    payload = b"{}"
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-4",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    registry = ProviderDeliveryRegistryV1()
    assert registry.observe(receipt) == "new-delivery"
    registry = registry.record(receipt)
    assert registry.observe(receipt) == "duplicate-identical"


def test_webhook_context_mismatch_rejects_even_with_valid_hmac():
    payload = b'{"action":"closed"}'
    receipt = ProviderWebhookReceiptV1.from_delivery(
        "delivery-context",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    context = ProviderWebhookRequestContextV1(
        "delivery-context",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "hook-2",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "issues",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "other/repo",
    )
    substituted = ProviderWebhookRequestContextV1(
        "delivery-context",
        "hook-2",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert not receipt.verify(
        payload,
        b"secret",
        substituted,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )


def test_webhook_delivery_registry_rejects_same_id_with_different_payload():
    first = ProviderWebhookReceiptV1.from_delivery(
        "delivery-5",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        b"{}",
        b"secret",
    )
    second = ProviderWebhookReceiptV1.from_delivery(
        "delivery-5",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
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
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
        payload,
        b"secret",
    )
    context = ProviderWebhookRequestContextV1(
        "delivery-6",
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
    assert receipt.verify(
        payload,
        b"secret",
        context,
        "hook-1",
        "pull_request",
        "Luminous-Dynamics/symthaea",
    )
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


def test_attestation_material_does_not_rescue_untrusted_capture():
    evidence = ProviderEvidenceEnvelopeV1(
        ProviderCaptureIntegrityV1("D", "store-1", 1),
        ProviderSourceAuthenticationV1(
            "local-untrusted",
            True,
            "github",
        ),
        ProviderAttestationV1(
            scheme="future-provider-attestation",
            verified=True,
        ),
    )
    assert evidence.has_provider_attestation()
    assert not evidence.is_preserved_provider_evidence()


def test_fabricated_local_capture_cannot_establish_requested_causality():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    local = provider_evidence_fixture(
        method="local-untrusted",
        verified=True,
    )
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=local,
    )
    assert resolution.outcome == "causality-unestablished"


def test_direct_provider_result_without_evidence_cannot_be_causal():
    identity = stack_identity_fixture()
    result = provider_merge_result_fixture(identity)
    resolution = causal_resolution_fixture(
        identity,
        provider_result=result,
        effect_set=None,
        provider_evidence=None,
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
    test_webhook_stack_metadata_matches_exact_operation,
    test_webhook_stack_metadata_rejects_stack_number_drift,
    test_webhook_stack_metadata_rejects_base_sha_drift,
    test_webhook_stack_metadata_missing_is_not_topology_proof,
    test_provider_timestamp_parser_accepts_utc_and_offset,
    test_provider_timestamp_interval_respects_reported_precision,
    test_provider_timestamp_interval_rejects_unsupported_precision_or_semantics,
    test_timestamp_policy_binds_provider_source_and_interpretation,
    test_timestamp_policy_resolution_limit_is_enforced,
    test_temporal_effect_rejects_tampered_timestamp_policy_identity,
    test_temporal_effect_without_timestamp_policy_is_not_admissible,
    test_provider_timestamp_parser_rejects_malformed_timestamp,
    test_clock_relation_unverified_is_unusable,
    test_clock_relation_negative_skew_is_unusable,
    test_clock_relation_requires_explicit_validity_window,
    test_clock_relation_rejects_expiry_before_verification,
    test_clock_relation_expiry_blocks_late_observation,
    test_clock_relation_drift_expands_uncertainty_monotonically,
    test_temporal_effect_with_valid_skew_is_admissible,
    test_stack_timing_rejects_mixed_timestamp_policy_identities,
    test_stack_timing_rejects_mixed_clock_relation_identities,
    test_temporal_effect_interval_overlap_is_not_admissible,
    test_temporal_effect_expired_clock_relation_is_not_admissible,
    test_temporal_effect_drift_can_turn_boundary_into_uncertainty,
    test_temporal_timing_rejects_missing_local_monotonic_time,
    test_temporal_timing_rejects_local_monotonic_rollback,
    test_temporal_timing_rejects_wall_clock_valid_but_monotonic_invalid_order,
    test_temporal_effect_without_clock_relation_is_unbounded,
    test_temporal_effect_with_event_before_dispatch_is_rejected,
    test_temporal_effect_with_event_after_observation_is_rejected,
    test_temporal_effect_with_uncertain_clock_overlap_is_not_admissible,
    test_temporal_effect_accepts_exact_skew_boundaries,
    test_provider_delivery_time_before_event_is_invalid,
    test_temporal_timing_can_be_derived_from_authenticated_observation,
    test_historical_merge_delivered_after_new_reservation_is_inadmissible,
    test_temporal_timing_rejects_missing_event_time,
    test_temporal_timing_rejects_invalid_local_order,
    test_temporal_timing_rejects_local_observation_rollback,
    test_complete_stack_timing_requires_every_member_admissible,
    test_stack_timing_rejects_crosswired_effect_evidence_identity,
    test_effect_state_binds_to_exact_operation_identity,
    test_effect_state_same_delivery_id_with_changed_payload_fails_closed,
    test_effect_state_rejects_inconsistent_source_delivery_bindings,
    test_effect_state_conflict_is_absorbing,
    test_effect_state_conflict_never_reclassifies_as_terminal_observed,
    test_effect_state_admits_first_authenticated_merge,
    test_effect_state_merged_then_non_effect_does_not_downgrade,
    test_effect_state_equivalent_second_delivery_is_idempotent,
    test_effect_state_repeated_same_delivery_is_idempotent,
    test_effect_state_conflicting_merge_commit_fails_closed,
    test_effect_state_conflicting_head_fails_closed,
    test_effect_state_untrusted_after_merge_does_not_downgrade,
    test_effect_state_unrelated_pr_does_not_downgrade,
    test_effect_state_is_not_latest_delivery_wins,
    test_webhook_effect_provenance_set_is_complete,
    test_webhook_effect_provenance_normalizes_input_order,
    test_webhook_effect_provenance_rejects_missing_source_provenance,
    test_webhook_effect_provenance_rejects_duplicate_delivery_id,
    test_webhook_effect_provenance_rejects_mixed_repository,
    test_webhook_effect_provenance_rejects_effect_disagreement,
    test_webhook_effect_provenance_projects_to_compact_effect_set,
    test_webhook_derived_stack_effect_set_is_complete,
    test_webhook_derived_stack_effect_set_normalizes_input_order,
    test_webhook_derived_stack_effect_set_rejects_missing_member,
    test_webhook_derived_stack_effect_set_rejects_extra_member,
    test_webhook_derived_stack_effect_set_rejects_duplicate_pr,
    test_webhook_derived_stack_effect_set_rejects_mixed_repository,
    test_webhook_derived_stack_effect_set_rejects_head_mismatch,
    test_webhook_derived_stack_effect_set_rejects_semantically_invalid_observation,
    test_webhook_derived_effect_set_preserves_observation_only_semantics,
    test_webhook_verify_rejects_empty_expected_context,
    test_webhook_parser_rejects_substituted_received_context,
    test_webhook_merge_effect_parses_authenticated_merged_pr,
    test_webhook_merge_effect_rejects_invalid_hmac,
    test_webhook_merge_effect_rejects_non_pull_request_event,
    test_webhook_merge_effect_rejects_non_closed_action,
    test_webhook_merge_effect_rejects_closed_not_merged,
    test_webhook_merge_effect_rejects_wrong_repository,
    test_webhook_merge_effect_rejects_wrong_pr_number,
    test_webhook_merge_effect_rejects_wrong_head,
    test_webhook_merge_effect_rejects_missing_merge_commit,
    test_webhook_merge_effect_rejects_tampered_payload_after_receipt,
    test_webhook_merge_effect_requires_exact_payload_digest,
    test_webhook_merge_effect_does_not_establish_async_operation_causality,
    test_capture_integrity_hashes_exact_raw_bytes,
    test_capture_integrity_rejects_empty_storage_identity,
    test_capture_integrity_rejects_nonpositive_sequence,
    test_source_authentication_api_channel_is_distinct_from_attestation,
    test_source_authentication_rejects_missing_provider_identity,
    test_source_authentication_rejects_unverified_api_channel,
    test_webhook_hmac_verification_accepts_exact_payload,
    test_webhook_hmac_verification_rejects_tampered_payload,
    test_webhook_hmac_verification_rejects_wrong_secret,
    test_webhook_context_mismatch_rejects_even_with_valid_hmac,
    test_webhook_delivery_registry_accepts_new_delivery,
    test_webhook_delivery_registry_rejects_same_id_with_different_payload,
    test_webhook_delivery_registry_rejects_same_id_with_changed_context,
    test_webhook_authentication_does_not_prove_merge_result_causality,
    test_attestation_material_does_not_rescue_untrusted_capture,
    test_fabricated_local_capture_cannot_establish_requested_causality,
    test_direct_provider_result_without_evidence_cannot_be_causal,
    test_authenticated_api_capture_can_support_requested_causality,
    test_invalid_capture_cannot_support_requested_causality,
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
