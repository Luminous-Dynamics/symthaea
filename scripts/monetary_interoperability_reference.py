#!/usr/bin/env python3
"""Deterministic reference semantics for monetary interoperability adapters.

Research oracle only: integer atomic units, explicit adapter state, no floats.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json

class Mode(str, Enum):
    REDEEM_REISSUE = "redeem_reissue"
    ATOMIC_SWAP = "escrowed_atomic_swap"
    NET_SETTLEMENT = "multilateral_net_settlement"
    MESSAGE_ONLY = "message_only"
    ABSENT = "absent"

class Status(str, Enum):
    SETTLED = "settled"
    QUEUED = "queued"
    STRANDED = "stranded"
    REJECTED = "rejected"
    MESSAGE_ONLY = "message_only"

@dataclass
class Ledger:
    name: str
    unit: str
    balances: dict[str, int] = field(default_factory=dict)
    locked: dict[str, int] = field(default_factory=dict)

    def balance(self, owner: str) -> int:
        return self.balances.get(owner, 0)

    def spendable(self, owner: str) -> int:
        return self.balance(owner) - self.locked.get(owner, 0)

    def credit(self, owner: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative credit")
        self.balances[owner] = self.balance(owner) + amount

    def debit(self, owner: str, amount: int) -> None:
        if amount < 0 or self.spendable(owner) < amount:
            raise ValueError("insufficient spendable balance")
        self.balances[owner] = self.balance(owner) - amount

    def lock(self, owner: str, amount: int) -> None:
        if amount < 0 or self.spendable(owner) < amount:
            raise ValueError("insufficient spendable balance")
        self.locked[owner] = self.locked.get(owner, 0) + amount

    def unlock(self, owner: str, amount: int) -> None:
        current = self.locked.get(owner, 0)
        if amount < 0 or current < amount:
            raise ValueError("invalid unlock")
        remaining = current - amount
        if remaining:
            self.locked[owner] = remaining
        else:
            self.locked.pop(owner, None)

@dataclass(frozen=True)
class Adapter:
    adapter_id: str
    mode: Mode
    rate_num: int = 1
    rate_den: int = 1
    fee_num: int = 0
    fee_den: int = 1
    liquidity_limit: int | None = None

    def convert(self, amount: int) -> int:
        if amount <= 0 or self.rate_num <= 0 or self.rate_den <= 0:
            raise ValueError("invalid conversion")
        gross = (amount * self.rate_num) // self.rate_den
        fee = (amount * self.fee_num) // self.fee_den if self.fee_num else 0
        if gross < fee:
            raise ValueError("fee exceeds gross")
        return gross - fee

@dataclass
class Receipt:
    event_id: str
    status: Status
    source: str
    target: str
    source_amount: int
    target_amount: int
    adapter_id: str
    reason: str | None
    trace_digest: str

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "status": self.status.value,
            "source": self.source,
            "target": self.target,
            "source_amount": self.source_amount,
            "target_amount": self.target_amount,
            "adapter_id": self.adapter_id,
            "reason": self.reason,
            "trace_digest": self.trace_digest,
        }

class ReferenceAdapter:
    def __init__(self, source: Ledger, target: Ledger) -> None:
        self.source = source
        self.target = target
        self.processed: set[str] = set()
        self.queue: list[tuple[str, str, str, int]] = []
        self.trace: list[dict] = []

    def _receipt(self, event_id: str, status: Status, src: str, dst: str,
                 source_amount: int, target_amount: int,
                 adapter: Adapter, reason: str | None = None) -> Receipt:
        state = {
            "event_id": event_id,
            "status": status.value,
            "source": src,
            "target": dst,
            "source_amount": source_amount,
            "target_amount": target_amount,
            "adapter_id": adapter.adapter_id,
            "reason": reason,
            "source_balances": sorted(self.source.balances.items()),
            "target_balances": sorted(self.target.balances.items()),
            "source_locked": sorted(self.source.locked.items()),
            "target_locked": sorted(self.target.locked.items()),
        }
        digest = sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        self.trace.append(state)
        return Receipt(event_id, status, src, dst, source_amount,
                       target_amount, adapter.adapter_id, reason, digest)

    def transfer(self, event_id: str, src: str, dst: str, amount: int,
                 adapter: Adapter, *, target_available: bool = True) -> Receipt:
        if event_id in self.processed:
            return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "replay")
        if amount <= 0:
            return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "non_positive_amount")
        target_amount = adapter.convert(amount)
        if adapter.mode is Mode.ABSENT:
            return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "no_interoperability_adapter")
        if adapter.mode is Mode.MESSAGE_ONLY:
            self.processed.add(event_id)
            return self._receipt(event_id, Status.MESSAGE_ONLY, src, dst, amount, target_amount, adapter, "no_settlement")
        if adapter.liquidity_limit is not None and target_amount > adapter.liquidity_limit:
            return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "adapter_liquidity_limit")
        if self.source.spendable(src) < amount:
            return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "source_insufficient")

        if adapter.mode is Mode.ATOMIC_SWAP:
            if not target_available:
                return self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter, "atomic_target_unavailable")
            self.source.lock(src, amount)
            self.source.debit(src, amount)
            self.source.unlock(src, amount)
            self.target.credit(dst, target_amount)
            self.processed.add(event_id)
            return self._receipt(event_id, Status.SETTLED, src, dst, amount, target_amount, adapter)

        if adapter.mode is Mode.REDEEM_REISSUE:
            self.source.debit(src, amount)
            if not target_available:
                self.processed.add(event_id)
                return self._receipt(event_id, Status.STRANDED, src, dst, amount, 0, adapter,
                                     "destination_unavailable_after_source_commit")
            self.target.credit(dst, target_amount)
            self.processed.add(event_id)
            return self._receipt(event_id, Status.SETTLED, src, dst, amount, target_amount, adapter)

        if adapter.mode is Mode.NET_SETTLEMENT:
            self.queue.append((event_id, src, dst, amount))
            self.processed.add(event_id)
            return self._receipt(event_id, Status.QUEUED, src, dst, amount, target_amount, adapter)

        raise AssertionError(f"unsupported mode: {adapter.mode}")

    def settle_net(self, adapter: Adapter) -> list[Receipt]:
        if adapter.mode is not Mode.NET_SETTLEMENT:
            raise ValueError("wrong adapter mode")
        queued, self.queue = self.queue, []
        out = []
        for event_id, src, dst, amount in queued:
            target_amount = adapter.convert(amount)
            if self.source.spendable(src) < amount:
                out.append(self._receipt(event_id, Status.REJECTED, src, dst, amount, 0, adapter,
                                         "source_insufficient_at_settlement"))
                continue
            self.source.debit(src, amount)
            self.target.credit(dst, target_amount)
            out.append(self._receipt(event_id, Status.SETTLED, src, dst, amount, target_amount, adapter))
        return out

def own_asset_quantity(ledger: Ledger) -> int:
    return sum(ledger.balances.values())

def demo() -> dict:
    source = Ledger("A", "A", {"alice": 100})
    target = Ledger("B", "B", {"bob": 0})
    model = ReferenceAdapter(source, target)
    atomic = Adapter("atomic-1", Mode.ATOMIC_SWAP)
    sequential = Adapter("seq-1", Mode.REDEEM_REISSUE)

    success = model.transfer("e1", "alice", "bob", 10, atomic)
    state_before_atomic_failure = (source.balance("alice"), target.balance("bob"))
    failure = model.transfer("e2", "alice", "bob", 10, atomic, target_available=False)
    state_after_atomic_failure = (source.balance("alice"), target.balance("bob"))
    stranded = model.transfer("e3", "alice", "bob", 10, sequential, target_available=False)
    replay = model.transfer("e1", "alice", "bob", 10, atomic)

    assert success.status is Status.SETTLED
    assert state_before_atomic_failure == (90, 10)
    assert failure.status is Status.REJECTED
    assert state_after_atomic_failure == state_before_atomic_failure
    assert stranded.status is Status.STRANDED
    assert (source.balance("alice"), target.balance("bob")) == (80, 10)
    assert replay.reason == "replay"

    net_source = Ledger("A", "A", {"a": 100, "c": 50})
    net_target = Ledger("B", "B", {"b": 0, "d": 0})
    net = ReferenceAdapter(net_source, net_target)
    net_adapter = Adapter("net-1", Mode.NET_SETTLEMENT)
    q1 = net.transfer("n1", "a", "b", 20, net_adapter)
    q2 = net.transfer("n2", "c", "d", 10, net_adapter)
    settled = net.settle_net(net_adapter)
    assert q1.status is Status.QUEUED and q2.status is Status.QUEUED
    assert all(r.status is Status.SETTLED for r in settled)

    return {
        "atomic_success": success.as_dict(),
        "atomic_failure_rollback": failure.as_dict(),
        "sequential_stranded": stranded.as_dict(),
        "replay_rejected": replay.as_dict(),
        "net_queued": [q1.as_dict(), q2.as_dict()],
        "net_settled": [r.as_dict() for r in settled],
        "asset_quantity_vector": {
            "A": own_asset_quantity(net_source),
            "B": own_asset_quantity(net_target),
        },
        "invariant_note": "Conservation is per asset identity. Cross-asset value preservation requires an explicit conversion or valuation rule."
    }

if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
