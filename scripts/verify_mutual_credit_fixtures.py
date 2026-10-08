#!/usr/bin/env python3
"""Independent semantic oracle for the frozen Mutual Credit fixture corpus.

This verifier intentionally does not import Rust code or call the production
kernel. It checks the declared fixture semantics independently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


LOWER = -100.0
UPPER = 100.0


class OracleError(Exception):
    pass


def apply(events):
    balances = {"alice": 0.0, "bob": 0.0, "carol": 0.0}
    defaulted = set()

    for event in events:
        op = event["op"]
        if op == "trade":
            buyer = event["buyer"]
            seller = event["seller"]
            amount = event["amount"]
            if buyer not in balances or seller not in balances:
                raise OracleError("UnknownMember")
            if buyer == seller:
                raise OracleError("SelfTrade")
            if not isinstance(amount, (int, float)) or amount <= 0:
                raise OracleError("NonPositiveAmount")
            if buyer in defaulted or seller in defaulted:
                raise OracleError("DefaultedMember")
            buyer_new = balances[buyer] - amount
            seller_new = balances[seller] + amount
            if buyer_new < LOWER or buyer_new > UPPER or seller_new < LOWER or seller_new > UPPER:
                raise OracleError("CreditLimitExceeded")
            balances[buyer] = buyer_new
            balances[seller] = seller_new
        elif op == "settle_external":
            member = event["member"]
            amount = event["amount"]
            if member not in balances:
                raise OracleError("UnknownMember")
            if not isinstance(amount, (int, float)) or amount <= 0:
                raise OracleError("NonPositiveAmount")
            if member in defaulted:
                raise OracleError("DefaultedMember")
            balance = balances[member]
            if balance <= 0 or amount > balance:
                raise OracleError("InsufficientPositiveBalance")
            balances[member] = balance - amount
        elif op == "mark_defaulted":
            member = event["member"]
            if member not in balances:
                raise OracleError("UnknownMember")
            defaulted.add(member)
        else:
            raise OracleError("UnknownOperation")

    return balances


def main() -> int:
    corpus = json.loads(Path(sys.argv[1]).read_text())
    failures = []

    for fixture in corpus["valid"]:
        try:
            balances = apply(fixture["events"])
        except OracleError as exc:
            failures.append(f"{fixture['id']}: rejected unexpectedly: {exc}")
            continue
        expected = fixture["expect"]["balances"]
        if any(abs(balances[k] - float(v)) > 1e-12 for k, v in expected.items()):
            failures.append(f"{fixture['id']}: balance mismatch")
        if abs(sum(balances.values()) - fixture["expect"]["total"]) > 1e-12:
            failures.append(f"{fixture['id']}: total mismatch")

    for fixture in corpus["rejected"]:
        events = fixture.get("events")
        if events is None:
            events = [fixture["event"]]
        try:
            apply(events)
        except OracleError as exc:
            if str(exc) != fixture["error"]:
                failures.append(f"{fixture['id']}: expected {fixture['error']}, got {exc}")
        else:
            failures.append(f"{fixture['id']}: accepted unexpectedly")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(
        f"independent semantic check: {len(corpus['valid'])}/{len(corpus['valid'])} valid; "
        f"{len(corpus['rejected'])}/{len(corpus['rejected'])} rejected; zero mismatches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
