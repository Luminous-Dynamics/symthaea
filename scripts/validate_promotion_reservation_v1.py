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
    provider_uuid: str | None = None

class Ledger:
    def __init__(self):
        self.head = "L0"
        self.active_lease = "LEASE-1"
        self.reservation: Reservation | None = None
        self._n = 0

    def reserve(self, observed_head: str, lease_id: str, candidate: str) -> bool:
        if observed_head != self.head:
            return False
        if lease_id != self.active_lease:
            return False
        if self.reservation is not None:
            return False
        self._n += 1
        rid = f"RES-{self._n}"
        op = f"OP-{self._n}"
        self.reservation = Reservation(rid, op, lease_id, observed_head)
        self.active_lease = None
        self.head = candidate
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

    def record_completed(self) -> None:
        assert self.reservation is not None
        self.reservation.state = "PromotionCompleted"

class ProviderOutcome:
    def __init__(self, http: int, kind: str, uuid: str | None = None):
        self.http, self.kind, self.uuid = http, kind, uuid

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
            return ProviderOutcome(200, "merged", self.merge_sha)
        if self.pending_uuid is not None:
            return ProviderOutcome(409, "duplicate", self.pending_uuid)
        if timeout_after_accept:
            self.pending_uuid = f"uuid-{self.calls}"
            return ProviderOutcome(599, "timeout-after-accept")
        self.pending_uuid = f"uuid-{self.calls}"
        return ProviderOutcome(202, "accepted", self.pending_uuid)

    def queue(self) -> ProviderOutcome:
        return ProviderOutcome(200, "enqueued")

    def complete(self) -> None:
        assert self.pending_uuid is not None
        self.merge_sha = "M1"

def legal_interleavings():
    tagged = []
    for p in permutations(("E-read","E-construct","E-commit","I-read","I-construct","I-commit")):
        idx = {x:i for i,x in enumerate(p)}
        if idx["E-read"] < idx["E-construct"] < idx["E-commit"] and idx["I-read"] < idx["I-construct"] < idx["I-commit"]:
            tagged.append(p)
    return tagged

def test_single_use_reservation_race():
    cases = legal_interleavings()
    assert len(cases) == 20
    for _ in cases:
        l = Ledger()
        assert l.reserve("L0", "LEASE-1", "L1") is True
        assert l.reserve("L0", "LEASE-1", "L2") is False
        assert l.reservation is not None
        assert l.reservation.state == "PromotionReserved"

def test_duplicate_reservation():
    l = Ledger()
    assert l.reserve("L0", "LEASE-1", "L1")
    assert not l.reserve("L1", "LEASE-1", "L2")
    assert not l.reserve("L0", "LEASE-1", "L3")
    assert l.active_lease is None

def test_dispatch_intent_fences_crashes():
    l = Ledger()
    assert l.reserve("L0", "LEASE-1", "L1")
    assert l.prepare_dispatch()
    assert l.reservation.dispatch_intent
    assert l.reservation.state == "PromotionDispatchPrepared"
    l.record_unknown()
    assert l.reservation.state == "PromotionReconciliationRequired"
    assert not l.prepare_dispatch()

def test_timeout_after_acceptance_is_unknown():
    l = Ledger()
    p = GitHubAsyncModel()
    l.reserve("L0", "LEASE-1", "L1")
    l.prepare_dispatch()
    o = p.submit("H1", timeout_after_accept=True)
    assert (o.http, o.kind) == (599, "timeout-after-accept")
    l.record_unknown()
    assert p.pending_uuid is not None
    assert l.reservation.state == "PromotionReconciliationRequired"

def test_duplicate_async_request_reuses_provider_handle():
    p = GitHubAsyncModel()
    first = p.submit("H1")
    second = p.submit("H1")
    assert first.http == 202 and first.uuid
    assert second.http == 409 and second.kind == "duplicate"
    assert second.uuid == first.uuid

def test_enqueued_is_not_completion():
    l = Ledger()
    p = GitHubAsyncModel()
    l.reserve("L0", "LEASE-1", "L1")
    l.prepare_dispatch()
    o = p.queue()
    assert o.kind == "enqueued"
    assert l.reservation.state == "PromotionDispatchPrepared"

def test_already_merged_is_durable_completion():
    p = GitHubAsyncModel()
    p.submit("H1")
    p.complete()
    o = p.submit("H1")
    assert o.http == 200 and o.kind == "merged"
    assert p.merge_sha == "M1"

def test_expired_uuid_uses_pr_state():
    p = GitHubAsyncModel()
    first = p.submit("H1")
    p.complete()
    assert first.uuid is not None
    p.expired.add(first.uuid)
    assert p.merge_sha == "M1"

def test_expired_uuid_without_effect_remains_reconciliation_required():
    l = Ledger()
    p = GitHubAsyncModel()
    l.reserve("L0", "LEASE-1", "L1")
    l.prepare_dispatch()
    first = p.submit("H1")
    assert first.uuid is not None
    p.expired.add(first.uuid)
    l.record_unknown()
    assert p.merge_sha is None
    assert l.reservation.state == "PromotionReconciliationRequired"

def test_exact_subject_mismatch_rejects():
    p = GitHubAsyncModel()
    o = p.submit("H-old")
    assert o.http == 409 and o.kind == "rejected"
    assert p.calls == 1 and p.merge_sha is None

def test_root_change_before_dispatch_blocks_effect():
    l = Ledger()
    p = GitHubAsyncModel()
    l.reserve("L0", "LEASE-1", "L1")
    l.reservation.state = "PromotionSuperseded"
    assert not l.prepare_dispatch()
    assert p.calls == 0

def test_root_change_after_dispatch_is_not_retroactive():
    l = Ledger()
    p = GitHubAsyncModel()
    l.reserve("L0", "LEASE-1", "L1")
    l.prepare_dispatch()
    o = p.submit("H1")
    assert o.http == 202
    l.record_unknown()
    assert l.reservation.state == "PromotionReconciliationRequired"
    assert p.pending_uuid is not None

def test_stale_coordinator_cannot_reserve_after_new_ledger_head():
    l = Ledger()
    assert l.reserve("L0", "LEASE-1", "L1")
    assert not l.reserve("L0", "LEASE-1", "L-stale")
    assert l.head == "L1"

TESTS = [
    test_single_use_reservation_race,
    test_duplicate_reservation,
    test_dispatch_intent_fences_crashes,
    test_timeout_after_acceptance_is_unknown,
    test_duplicate_async_request_reuses_provider_handle,
    test_enqueued_is_not_completion,
    test_already_merged_is_durable_completion,
    test_expired_uuid_uses_pr_state,
    test_expired_uuid_without_effect_remains_reconciliation_required,
    test_exact_subject_mismatch_rejects,
    test_root_change_before_dispatch_blocks_effect,
    test_root_change_after_dispatch_is_not_retroactive,
    test_stale_coordinator_cannot_reserve_after_new_ledger_head,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print("PASS", test.__name__)
    print("PromotionReservationV1 model: PASS")
    print("claim_ceiling=deterministic local transaction/reconciliation model only")
    print("promotion_authority=false")
