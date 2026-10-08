#!/usr/bin/env python3
"""Independent, promotion-authority-free model for Promotion Reservation v1.

claim_ceiling=deterministic local transaction/reconciliation model only
promotion_authority=false
"""

from dataclasses import dataclass
from itertools import permutations


@dataclass
class Reservation:
    reservation_id: str
    operation_id: str
    lease_id: str
    predecessor: str
    state: str = "PromotionReserved"
    dispatch_intent: bool = False


class Ledger:
    def __init__(self):
        self.head = "L0"
        self.active_lease = "LEASE-1"
        self.reservation: Reservation | None = None
        self.transitions = []
        self._reservation_n = 0

    def reserve(self, observed_head: str, lease_id: str, candidate: str) -> bool:
        if observed_head != self.head or lease_id != self.active_lease:
            return False
        if self.reservation is not None:
            return False
        self._reservation_n += 1
        self.reservation = Reservation(
            f"RES-{self._reservation_n}",
            f"OP-{self._reservation_n}",
            lease_id,
            observed_head,
        )
        self.active_lease = None
        self.head = candidate
        self.transitions.append(("reservation", observed_head, candidate))
        return True

    def invalidate(self, observed_head: str, candidate: str) -> bool:
        if observed_head != self.head:
            return False
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
    def __init__(self, http: int, kind: str, uuid: str | None = None):
        self.http = http
        self.kind = kind
        self.uuid = uuid


class GitHubAsyncModel:
    def __init__(self):
        self.pr_head = "H1"
        self.pending_uuid: str | None = None
        self.merge_sha: str | None = None
        self.expired: set[str] = set()
        self.calls = 0

    def submit(self, expected_head: str, timeout_after_accept: bool = False) -> ProviderOutcome:
        self.calls += 1
        if expected_head != self.pr_head:
            return ProviderOutcome(409, "rejected")
        if self.merge_sha is not None:
            return ProviderOutcome(200, "merged")
        if self.pending_uuid is not None:
            return ProviderOutcome(409, "duplicate", self.pending_uuid)
        self.pending_uuid = f"uuid-{self.calls}"
        if timeout_after_accept:
            return ProviderOutcome(599, "timeout-after-accept")
        return ProviderOutcome(202, "accepted", self.pending_uuid)

    def get_async_result(self, uuid: str) -> ProviderOutcome:
        if uuid in self.expired:
            return ProviderOutcome(404, "not-found")
        if self.merge_sha is not None:
            return ProviderOutcome(200, "merged")
        if self.pending_uuid == uuid:
            return ProviderOutcome(200, "enqueued")
        return ProviderOutcome(404, "not-found")

    def complete(self) -> None:
        assert self.pending_uuid is not None
        self.merge_sha = "M1"


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
                else ledger.invalidate(snapshots[writer], candidates[writer])
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
    assert ledger.reserve("L0", "LEASE-1", "L1")
    assert not ledger.reserve("L1", "LEASE-1", "L2")
    assert not ledger.reserve("L0", "LEASE-1", "L3")
    assert ledger.active_lease is None


def test_dispatch_intent_fences_crash():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1")
    assert ledger.prepare_dispatch()
    assert ledger.reservation.dispatch_intent
    assert ledger.reservation.state == "PromotionDispatchPrepared"
    ledger.record_unknown()
    assert ledger.reservation.state == "PromotionReconciliationRequired"
    assert not ledger.prepare_dispatch()


def test_timeout_after_acceptance_is_unknown():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    ledger.reserve("L0", "LEASE-1", "L1")
    ledger.prepare_dispatch()
    outcome = provider.submit("H1", timeout_after_accept=True)
    assert outcome.kind == "timeout-after-accept"
    ledger.record_unknown()
    assert provider.pending_uuid is not None
    assert ledger.reservation.state == "PromotionReconciliationRequired"


def test_duplicate_async_request_reuses_provider_handle():
    provider = GitHubAsyncModel()
    first = provider.submit("H1")
    second = provider.submit("H1")
    assert first.http == 202 and first.uuid
    assert second.http == 409 and second.kind == "duplicate"
    assert second.uuid == first.uuid


def test_enqueued_is_not_completion():
    provider = GitHubAsyncModel()
    accepted = provider.submit("H1")
    assert accepted.uuid is not None
    queued = provider.get_async_result(accepted.uuid)
    assert queued.http == 200 and queued.kind == "enqueued"
    assert provider.merge_sha is None


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
    ledger.reserve("L0", "LEASE-1", "L1")
    ledger.prepare_dispatch()
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
    assert ledger.reserve("L0", "LEASE-1", "L1")
    assert ledger.invalidate("L1", "I1")
    assert ledger.reservation.state == "PromotionSuperseded"
    assert not ledger.prepare_dispatch()
    assert provider.calls == 0


def test_root_change_after_dispatch_is_not_retroactive():
    ledger = Ledger()
    provider = GitHubAsyncModel()
    assert ledger.reserve("L0", "LEASE-1", "L1")
    assert ledger.prepare_dispatch()
    outcome = provider.submit("H1")
    assert outcome.http == 202
    ledger.record_unknown()
    assert ledger.reservation.state == "PromotionReconciliationRequired"
    assert provider.pending_uuid is not None


def test_stale_coordinator_cannot_reserve_after_new_head():
    ledger = Ledger()
    assert ledger.reserve("L0", "LEASE-1", "L1")
    assert not ledger.reserve("L0", "LEASE-1", "L-stale")
    assert ledger.head == "L1"


TESTS = [
    test_exact_20_schedules_are_executed,
    test_single_use_reservation,
    test_dispatch_intent_fences_crash,
    test_timeout_after_acceptance_is_unknown,
    test_duplicate_async_request_reuses_provider_handle,
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
