#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

from dataclasses import dataclass
from itertools import permutations


@dataclass
class PromotionEffectReceipt:
    promotion_operation_id: str
    expected_pr_head_sha: str
    observed_merge_commit: str


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

    def prepare_dispatch(self) -> bool:
        r = self.reservation
        if r is None or r.dispatch_intent or r.state != "PromotionReserved":
            return False
        r.dispatch_intent = True
        r.state = "PromotionDispatchPrepared"
        return True

    def record_unknown(self) -> None:
        assert self.reservation is not None
        self.reservation.state = "PromotionReconciliationRequired"


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
    assert ledger.prepare_dispatch("L1", 1, 1)
    assert ledger.reservation.dispatch_intent
    assert ledger.reservation.state == "PromotionDispatchPrepared"
    ledger.record_unknown()
    assert ledger.reservation.state == "PromotionReconciliationRequired"
    assert not ledger.prepare_dispatch("L1", 1, 1)


def test_timeout_after_acceptance_is_unknown():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    ledger.reserve("L0", "LEASE-1", "L1", 1)
    ledger.prepare_dispatch("L1", 1, 1)
    outcome = provider.submit("H1", timeout_after_accept=True)
    assert outcome.kind == "timeout-after-accept"
    ledger.record_unknown()
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
    ledger.prepare_dispatch("L1", 1, 1)
    first = provider.submit("H1")
    assert first.uuid is not None
    provider.expired.add(first.uuid)
    ledger.record_unknown()
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
    assert not ledger.prepare_dispatch("L1", 1, 1)
    assert provider.calls == 0


def test_root_change_after_dispatch_is_not_retroactive():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    assert ledger.reserve("L0", "LEASE-1", "L1", 1)
    assert ledger.prepare_dispatch("L1", 1, 1)
    outcome = provider.submit("H1")
    assert outcome.http == 202
    ledger.record_unknown()
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
    assert ledger.prepare_dispatch("L1", 1, 1)
    ledger.record_unknown()
    op = ledger.reservation.operation_id
    assert not ledger.reconcile_complete(PromotionEffectReceipt("wrong", "H1", "M1"))
    assert ledger.reconcile_complete(PromotionEffectReceipt(op, "H1", "M1"))
    assert ledger.reservation.state == "PromotionCompleted"


def test_unrelated_ledger_transition_rejects_stale_dispatch_fence():
    ledger = Ledger()
    ledger.reserve("L0", "LEASE-1", "L1", 1)
    ledger.fencing_token += 1
    ledger.head = "L2"
    assert not ledger.prepare_dispatch("L1", 1, 1)


TESTS = [
    test_exact_20_schedules_are_executed,
    test_single_use_reservation,
    test_dispatch_intent_fences_crash,
    test_timeout_after_acceptance_is_unknown,
    test_duplicate_async_request_reuses_provider_handle,
    test_duplicate_async_request_option_mismatch_is_not_idempotent,
    test_enqueued_is_not_completion,
    test_already_merged_is_durable_completion,
    test_expired_uuid_with_merged_pr_uses_pr_state,
    test_expired_uuid_without_effect_stays_unknown,
    test_exact_subject_mismatch_rejects,
    test_root_change_before_dispatch_blocks_effect,
    test_root_change_after_dispatch_is_not_retroactive,
    test_stale_coordinator_cannot_reserve_after_new_head,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print("PASS", test.__name__)
    print("PromotionReservationV1 model: PASS")
    print("claim_ceiling=deterministic local transaction/reconciliation model only")
    print("promotion_authority=false")
