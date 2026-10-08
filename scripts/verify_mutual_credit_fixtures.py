#!/usr/bin/env python3
"""Independent semantic oracle for the frozen Mutual Credit fixture corpus.

This verifier intentionally does not import Rust code or call the production
kernel. It validates the fixture contract and evaluates the mechanics from
scratch using Python's arbitrary-precision integers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


I128_MIN = -(2**127)
I128_MAX = 2**127 - 1


class OracleError(Exception):
    pass


def fail(message: str) -> None:
    raise OracleError(message)


def require_keys(obj: dict, required: set[str], allowed: set[str], context: str) -> None:
    keys = set(obj)
    if not required.issubset(keys):
        fail(f"{context}: missing required keys {sorted(required - keys)}")
    if not keys.issubset(allowed):
        fail(f"{context}: unexpected keys {sorted(keys - allowed)}")


def require_int(value, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{context}: expected integer")
    if value < I128_MIN or value > I128_MAX:
        fail(f"{context}: outside i128 range")
    if minimum is not None and value < minimum:
        fail(f"{context}: below minimum {minimum}")
    return value


def validate_corpus(corpus: dict) -> tuple[list[str], int, int]:
    if not isinstance(corpus, dict):
        fail("corpus: expected object")

    require_keys(
        corpus,
        {"schema_version", "member_limit", "initial_members", "valid", "rejected"},
        {"schema_version", "member_limit", "initial_members", "valid", "rejected"},
        "corpus",
    )
    if corpus["schema_version"] != "mutual-credit-fixtures-v0":
        fail("corpus: unsupported schema_version")

    limits = corpus["member_limit"]
    if not isinstance(limits, dict):
        fail("member_limit: expected object")
    require_keys(
        limits,
        {"debit_limit", "credit_limit"},
        {"debit_limit", "credit_limit"},
        "member_limit",
    )
    debit_limit = require_int(limits["debit_limit"], "debit_limit", 0)
    credit_limit = require_int(limits["credit_limit"], "credit_limit", 0)

    members = corpus["initial_members"]
    if not isinstance(members, list) or not members:
        fail("initial_members: expected non-empty array")
    if len(set(members)) != len(members):
        fail("initial_members: duplicate member")
    for member in members:
        if not isinstance(member, str) or not member:
            fail("initial_members: member must be non-empty string")

    if not isinstance(corpus["valid"], list) or not isinstance(corpus["rejected"], list):
        fail("valid/rejected: expected arrays")

    return members, debit_limit, credit_limit


def validate_event(event: dict, context: str) -> None:
    if not isinstance(event, dict):
        fail(f"{context}: expected object")
    if "op" not in event or "id" not in event:
        fail(f"{context}: missing id/op")

    event_id = event["id"]
    if not isinstance(event_id, str) or not event_id:
        fail(f"{context}: id must be non-empty string")

    op = event["op"]
    if op == "trade":
        require_keys(
            event,
            {"id", "op", "buyer", "seller", "amount"},
            {"id", "op", "buyer", "seller", "amount"},
            context,
        )
        if not isinstance(event["buyer"], str) or not event["buyer"]:
            fail(f"{context}: buyer must be non-empty string")
        if not isinstance(event["seller"], str) or not event["seller"]:
            fail(f"{context}: seller must be non-empty string")
        require_int(event["amount"], f"{context}.amount", 1)
    elif op == "settle_external":
        require_keys(
            event,
            {"id", "op", "member", "amount"},
            {"id", "op", "member", "amount"},
            context,
        )
        if not isinstance(event["member"], str) or not event["member"]:
            fail(f"{context}: member must be non-empty string")
        require_int(event["amount"], f"{context}.amount", 1)
    elif op == "mark_defaulted":
        require_keys(
            event,
            {"id", "op", "member"},
            {"id", "op", "member"},
            context,
        )
        if not isinstance(event["member"], str) or not event["member"]:
            fail(f"{context}: member must be non-empty string")
    else:
        # The production kernel has no dynamic operation fallback; unknown ops
        # are retained as rejected fixtures so the oracle can test fail-closed
        # dispatch while still recognizing that they are outside the valid
        # operation schema.
        require_keys(event, {"id", "op"}, {"id", "op"}, context)


def apply(
    events: list[dict],
    members: list[str],
    debit_limit: int,
    credit_limit: int,
) -> tuple[dict[str, int], int]:
    balances = {member: 0 for member in members}
    defaulted: set[str] = set()
    seen_event_ids: set[str] = set()
    external_settled_total = 0

    for index, event in enumerate(events):
        validate_event(event, f"event[{index}]")
        event_id = event["id"]

        if event_id in seen_event_ids:
            fail("DuplicateEventId")
        seen_event_ids.add(event_id)

        op = event["op"]
        if op == "trade":
            buyer = event["buyer"]
            seller = event["seller"]
            amount = require_int(event["amount"], f"event[{index}].amount", 1)

            if buyer not in balances or seller not in balances:
                fail("UnknownMember")
            if buyer == seller:
                fail("SelfTrade")
            if buyer in defaulted or seller in defaulted:
                fail("DefaultedMember")

            buyer_new = balances[buyer] - amount
            seller_new = balances[seller] + amount
            lower_buyer = -debit_limit
            lower_seller = -debit_limit

            if buyer_new < lower_buyer or buyer_new > credit_limit:
                fail("CreditLimitExceeded")
            if seller_new < lower_seller or seller_new > credit_limit:
                fail("CreditLimitExceeded")

            balances[buyer] = buyer_new
            balances[seller] = seller_new

        elif op == "settle_external":
            member = event["member"]
            amount = require_int(event["amount"], f"event[{index}].amount", 1)

            if member not in balances:
                fail("UnknownMember")
            if member in defaulted:
                fail("DefaultedMember")

            balance = balances[member]
            if balance <= 0 or amount > balance:
                fail("InsufficientPositiveBalance")

            balances[member] = balance - amount
            external_settled_total += amount

        elif op == "mark_defaulted":
            member = event["member"]
            if member not in balances:
                fail("UnknownMember")
            defaulted.add(member)

        else:
            fail("UnknownOperation")

    return balances, external_settled_total


def validate_valid_fixture(
    fixture: dict,
    index: int,
    members: list[str],
    debit_limit: int,
    credit_limit: int,
) -> None:
    context = f"valid[{index}]"
    if not isinstance(fixture, dict):
        fail(f"{context}: expected object")
    require_keys(fixture, {"id", "events", "expect"}, {"id", "events", "expect"}, context)

    fixture_id = fixture["id"]
    if not isinstance(fixture_id, str) or not fixture_id:
        fail(f"{context}: invalid id")
    events = fixture["events"]
    if not isinstance(events, list):
        fail(f"{context}.events: expected array")
    expect = fixture["expect"]
    if not isinstance(expect, dict):
        fail(f"{context}.expect: expected object")
    require_keys(
        expect,
        {"balances", "total", "external_settled_total"},
        {"balances", "total", "external_settled_total"},
        f"{context}.expect",
    )

    expected_balances = expect["balances"]
    if not isinstance(expected_balances, dict):
        fail(f"{context}.expect.balances: expected object")
    if set(expected_balances) != set(members):
        fail(f"{context}.expect.balances: member keys do not match initial_members")
    for member, value in expected_balances.items():
        require_int(value, f"{context}.expect.balances[{member!r}]")

    expected_total = require_int(expect["total"], f"{context}.expect.total")
    expected_external = require_int(
        expect["external_settled_total"],
        f"{context}.expect.external_settled_total",
        0,
    )

    balances, external = apply(events, members, debit_limit, credit_limit)
    if balances != expected_balances:
        fail(f"{fixture_id}: balance mismatch")
    if sum(balances.values()) != expected_total:
        fail(f"{fixture_id}: total mismatch")
    if external != expected_external:
        fail(f"{fixture_id}: external settlement mismatch")
    if sum(balances.values()) + external != 0:
        fail(f"{fixture_id}: external conservation mismatch")


def validate_rejected_fixture(
    fixture: dict,
    index: int,
    members: list[str],
    debit_limit: int,
    credit_limit: int,
) -> None:
    context = f"rejected[{index}]"
    if not isinstance(fixture, dict):
        fail(f"{context}: expected object")
    require_keys(fixture, {"id", "events", "error"}, {"id", "events", "error"}, context)

    fixture_id = fixture["id"]
    if not isinstance(fixture_id, str) or not fixture_id:
        fail(f"{context}: invalid id")
    error = fixture["error"]
    if not isinstance(error, str) or not error:
        fail(f"{context}: invalid error")
    events = fixture["events"]
    if not isinstance(events, list) or not events:
        fail(f"{context}.events: expected non-empty array")

    try:
        apply(events, members, debit_limit, credit_limit)
    except OracleError as exc:
        if str(exc) != error:
            fail(f"{fixture_id}: expected {error}, got {exc}")
    else:
        fail(f"{fixture_id}: accepted unexpectedly")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_mutual_credit_fixtures.py CORPUS.json", file=sys.stderr)
        return 2

    try:
        corpus = json.loads(Path(sys.argv[1]).read_text())
        members, debit_limit, credit_limit = validate_corpus(corpus)

        fixture_ids: set[str] = set()
        for index, fixture in enumerate(corpus["valid"]):
            validate_valid_fixture(fixture, index, members, debit_limit, credit_limit)
            fixture_id = fixture["id"]
            if fixture_id in fixture_ids:
                fail(f"duplicate fixture id: {fixture_id}")
            fixture_ids.add(fixture_id)

        for index, fixture in enumerate(corpus["rejected"]):
            validate_rejected_fixture(fixture, index, members, debit_limit, credit_limit)
            fixture_id = fixture["id"]
            if fixture_id in fixture_ids:
                fail(f"duplicate fixture id: {fixture_id}")
            fixture_ids.add(fixture_id)

    except (OSError, json.JSONDecodeError, OracleError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    valid_count = len(corpus["valid"])
    rejected_count = len(corpus["rejected"])
    print(
        f"independent semantic check: {valid_count}/{valid_count} valid; "
        f"{rejected_count}/{rejected_count} rejected; zero mismatches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
