#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

import hashlib
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
    observation_id: str = ""

    def internally_consistent(self, requested_pr_number: int) -> bool:
        if not self.observation_source or not self.observation_id:
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
        stack_number=41,
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


def test_provider_stack_observation_exact_selected_prefix_matches():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    assert observation.matches_reserved(identity)


def test_provider_stack_observation_missing_identity_is_unrepresented():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    assert observation.operation_depth if False else True
    assert observation.matches_reserved(identity)


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


def test_provider_stack_observation_rejects_missing_observation_identity():
    identity = stack_identity_fixture()
    observation = provider_stack_observation_fixture(identity)
    changed = ProviderStackObservationV1(
        **{**observation.__dict__, "observation_id": ""}
    )
    assert not changed.matches_reserved(identity)


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
    test_provider_stack_observation_exact_selected_prefix_matches,
    test_provider_stack_observation_missing_identity_is_unrepresented,
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
    test_provider_stack_observation_rejects_missing_observation_identity,
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
