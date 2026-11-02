#!/usr/bin/env python3
"""Exact-rational two-phase source-escrow event writer and independent verifier.

Reservations, assignments and credits are separate immutable events.
Assignments are chosen before current-round evidence is generated.
This is an accounting/protocol screen, not an e-process or FDR experiment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
from fractions import Fraction

N = 16
ROUNDS = 40
DEFAULT_TRACES = 250
GAMMA = [Fraction(1, 16) for _ in range(N)]
POLICY = hashlib.sha256(
    b"two-phase-source-escrow-v1|gamma=1/16|predictable-assign"
).hexdigest()


def reserve_target(weight: Fraction, evidence: Fraction) -> Fraction:
    return min(max(weight * evidence - 1, Fraction(0)), weight)


def build_trace(trace_id: int) -> tuple[list[dict], list[int]]:
    """Create a deterministic writer-side event stream."""
    rng = random.Random(20261102 + 17 * trace_id)
    evidence = [Fraction(0) for _ in range(N)]
    active = [False] * N
    activation_round = [min(ROUNDS - 1, (i // 4) * 5) for i in range(N)]
    outgoing = [Fraction(0) for _ in range(N)]
    incoming = [Fraction(0) for _ in range(N)]
    token_ordinal = [0] * N
    assignment_ordinal: dict[tuple[int, int], int] = {}
    tokens: dict[tuple[int, int], dict] = {}
    assignments: dict[tuple[int, int, int], dict] = {}
    events: list[dict] = []
    sequence = 0

    def emit(kind: str, round_no: int, **fields) -> None:
        nonlocal sequence
        sequence += 1
        events.append({
            "seq": sequence, "kind": kind, "round": round_no,
            "policy": POLICY, **fields
        })

    for round_no in range(ROUNDS):
        # Predictable pre-observation activation schedule.
        for candidate in range(N):
            if not active[candidate] and activation_round[candidate] == round_no:
                active[candidate] = True
                emit("Activate", round_no, candidate=candidate)

        # Phase 2a: assignment uses only previously committed state/tokens.
        for token_id in sorted(tokens):
            token = tokens[token_id]
            source, ordinal = token_id
            if (
                token["unassigned"] <= 0
                or round_no <= token["created_round"]
                or source >= N - 1
            ):
                continue
            if rng.random() < 0.58:
                destination = source + 1 + (
                    (source * 7 + ordinal * 3 + round_no * 5 + trace_id)
                    % (N - source - 1)
                )
                next_ordinal = assignment_ordinal.get(token_id, 0) + 1
                assignment_ordinal[token_id] = next_ordinal
                amount = (
                    token["unassigned"] if rng.random() < 0.45
                    else token["unassigned"] / 2
                )
                assignment_id = (source, ordinal, next_ordinal)
                token["unassigned"] -= amount
                assignments[assignment_id] = {
                    "token_id": token_id,
                    "source": source,
                    "destination": destination,
                    "amount": amount,
                    "remaining": amount,
                    "credited": Fraction(0),
                    "assignment_round": round_no,
                }
                emit(
                    "Assigned", round_no,
                    token_id=token_id,
                    assignment_id=assignment_id,
                    source=source,
                    destination=destination,
                    amount=amount,
                    assignment_ordinal=next_ordinal,
                )

        # Phase 2b: only assigned balances can be credited; partial credits
        # are allowed, and dormant recipients retain destination-specific escrow.
        for assignment_id in sorted(assignments):
            assignment = assignments[assignment_id]
            destination = assignment["destination"]
            if assignment["remaining"] <= 0 or not active[destination]:
                continue
            amount = (
                assignment["remaining"]
                if (round_no + sum(assignment_id)) % 3 == 0
                else assignment["remaining"] / 2
            )
            assignment["remaining"] -= amount
            assignment["credited"] += amount
            incoming[destination] += amount
            emit(
                "Credited", round_no,
                token_id=assignment["token_id"],
                assignment_id=assignment_id,
                source=assignment["source"],
                destination=destination,
                amount=amount,
            )

        # Phase 1: evidence follows routing. Evidence and reserve are atomic
        # for score monotonicity; LookCommit marks the observable state boundary.
        for candidate in range(N):
            if not active[candidate]:
                continue
            delta = Fraction(rng.randint(500, 2000), 1000)
            evidence[candidate] += delta
            emit(
                "Evidence", round_no,
                candidate=candidate,
                delta=delta,
                new_evidence=evidence[candidate],
            )
            target = reserve_target(GAMMA[candidate], evidence[candidate])
            amount = target - outgoing[candidate]
            assert amount >= 0
            assert amount <= GAMMA[candidate] * delta
            if amount:
                token_ordinal[candidate] += 1
                token_id = (candidate, token_ordinal[candidate])
                tokens[token_id] = {
                    "source": candidate,
                    "created_round": round_no,
                    "face_amount": amount,
                    "unassigned": amount,
                }
                outgoing[candidate] += amount
                emit(
                    "UnassignedReserved", round_no,
                    source=candidate,
                    token_id=token_id,
                    amount=amount,
                    face_amount=amount,
                    source_evidence=evidence[candidate],
                )

        emit("LookCommit", round_no)

    return events, activation_round


def verify_event_log(events: list[dict], activation_round: list[int]) -> bool:
    """Independent replay, separate from writer transitions."""
    evidence = [Fraction(0) for _ in range(N)]
    active = [False] * N
    outgoing = [Fraction(0) for _ in range(N)]
    incoming = [Fraction(0) for _ in range(N)]
    prefix_balance = [Fraction(0) for _ in range(N)]
    source_tokens: dict[tuple[int, int], dict] = {}
    assignments: dict[tuple[int, int, int], dict] = {}
    token_ordinal = [0] * N
    assignment_ordinal: dict[tuple[int, int], int] = {}
    pending_reserve: dict[int, Fraction] = {}
    prior_scores = [Fraction(0) for _ in range(N)]
    token_assigned_remaining: dict[tuple[int, int], Fraction] = {}
    token_credited: dict[tuple[int, int], Fraction] = {}
    unassigned_total = Fraction(0)
    assigned_total = Fraction(0)
    credited_total = Fraction(0)
    reserved_total = Fraction(0)
    last_sequence = 0
    last_round = -1

    try:
        for event in events:
            assert event["policy"] == POLICY
            assert event["seq"] == last_sequence + 1
            assert event["round"] >= last_round
            last_sequence = event["seq"]
            last_round = event["round"]
            kind = event["kind"]
            round_no = event["round"]
            touched_token = None
            touched_candidate = None

            due_sources = [i for i, amount in pending_reserve.items() if amount > 0]
            if due_sources:
                assert kind == "UnassignedReserved"
                assert event.get("source") in due_sources

            if kind == "Activate":
                candidate = event["candidate"]
                assert not active[candidate]
                assert activation_round[candidate] == round_no
                active[candidate] = True

            elif kind == "Evidence":
                candidate = event["candidate"]
                delta = event["delta"]
                assert active[candidate] and delta > 0
                assert event["new_evidence"] == evidence[candidate] + delta
                evidence[candidate] = event["new_evidence"]
                due = (
                    reserve_target(GAMMA[candidate], evidence[candidate])
                    - outgoing[candidate]
                )
                assert due >= 0
                pending_reserve[candidate] = due
                touched_candidate = candidate

            elif kind == "UnassignedReserved":
                source = event["source"]
                token_id = tuple(event["token_id"])
                amount = event["amount"]
                due = pending_reserve.get(source, Fraction(0))
                assert due > 0 and amount == due
                assert event["face_amount"] == amount
                assert event["source_evidence"] == evidence[source]
                token_ordinal[source] += 1
                assert token_id == (source, token_ordinal[source])
                assert token_id not in source_tokens
                source_tokens[token_id] = {
                    "source": source,
                    "creation_round": round_no,
                    "face_amount": amount,
                    "unassigned": amount,
                }
                token_assigned_remaining[token_id] = Fraction(0)
                token_credited[token_id] = Fraction(0)
                outgoing[source] += amount
                reserved_total += amount
                unassigned_total += amount
                pending_reserve[source] = Fraction(0)
                for prefix_index in range(source, N):
                    prefix_balance[prefix_index] -= amount
                assert outgoing[source] <= GAMMA[source] * min(
                    evidence[source], Fraction(1)
                )
                touched_candidate = source
                touched_token = token_id

            elif kind == "Assigned":
                token_id = tuple(event["token_id"])
                assignment_id = tuple(event["assignment_id"])
                source = event["source"]
                destination = event["destination"]
                amount = event["amount"]
                assert token_id in source_tokens
                assert source == token_id[0]
                assert source < destination < N
                assert round_no > source_tokens[token_id]["creation_round"]
                assert assignment_id == (
                    source, token_id[1], event["assignment_ordinal"]
                )
                assert assignment_id not in assignments
                expected_ordinal = assignment_ordinal.get(token_id, 0) + 1
                assert event["assignment_ordinal"] == expected_ordinal
                assignment_ordinal[token_id] = expected_ordinal
                assert amount > 0
                assert amount <= source_tokens[token_id]["unassigned"]
                source_tokens[token_id]["unassigned"] -= amount
                unassigned_total -= amount
                assigned_total += amount
                token_assigned_remaining[token_id] += amount
                assignments[assignment_id] = {
                    "token_id": token_id,
                    "source": source,
                    "destination": destination,
                    "face_amount": amount,
                    "remaining": amount,
                    "credited": Fraction(0),
                    "assignment_round": round_no,
                }
                touched_token = token_id

            elif kind == "Credited":
                assignment_id = tuple(event["assignment_id"])
                token_id = tuple(event["token_id"])
                amount = event["amount"]
                assert assignment_id in assignments
                assignment = assignments[assignment_id]
                assert token_id == assignment["token_id"]
                assert event["source"] == assignment["source"]
                assert event["destination"] == assignment["destination"]
                destination = assignment["destination"]
                assert active[destination]
                assert round_no >= assignment["assignment_round"]
                assert amount > 0 and amount <= assignment["remaining"]
                assignment["remaining"] -= amount
                assignment["credited"] += amount
                incoming[destination] += amount
                assigned_total -= amount
                credited_total += amount
                token_assigned_remaining[token_id] -= amount
                token_credited[token_id] += amount
                for prefix_index in range(destination, N):
                    prefix_balance[prefix_index] += amount
                touched_candidate = destination
                touched_token = token_id

            elif kind == "LookCommit":
                assert not any(amount > 0 for amount in pending_reserve.values())
                for candidate in range(N):
                    score = (
                        GAMMA[candidate] * evidence[candidate]
                        + incoming[candidate] - outgoing[candidate]
                    )
                    assert score >= 0
                    assert score >= prior_scores[candidate]
                    prior_scores[candidate] = score

            else:
                raise AssertionError("unknown event kind")

            assert max(prefix_balance, default=Fraction(0)) <= 0
            assert (
                unassigned_total + assigned_total + credited_total
                == reserved_total
            )
            if touched_token is not None:
                token = source_tokens[touched_token]
                assert token["face_amount"] == (
                    token["unassigned"]
                    + token_assigned_remaining[touched_token]
                    + token_credited[touched_token]
                )
            if touched_candidate is not None:
                candidate = touched_candidate
                assert outgoing[candidate] <= GAMMA[candidate] * min(
                    evidence[candidate], Fraction(1)
                )
                assert (
                    GAMMA[candidate] * evidence[candidate]
                    + incoming[candidate] - outgoing[candidate]
                    >= 0
                )

        assert not any(amount > 0 for amount in pending_reserve.values())
        return True
    except (AssertionError, KeyError, TypeError, ValueError, IndexError):
        return False


def negative_controls(events: list[dict], activation_round: list[int]) -> dict[str, bool]:
    results: dict[str, bool] = {}

    changed = copy.deepcopy(events)
    changed[0]["policy"] = "0" * 64
    results["policy_digest_tamper"] = not verify_event_log(changed, activation_round)

    changed = copy.deepcopy(events)
    index = next(i for i, row in enumerate(changed) if row["kind"] == "Assigned")
    changed[index]["destination"] = changed[index]["source"]
    results["backward_destination"] = not verify_event_log(changed, activation_round)

    changed = copy.deepcopy(events)
    index = next(i for i, row in enumerate(changed) if row["kind"] == "Assigned")
    changed[index]["amount"] += Fraction(1, 16)
    results["overassignment"] = not verify_event_log(changed, activation_round)

    changed = copy.deepcopy(events)
    index = next(i for i, row in enumerate(changed) if row["kind"] == "Assigned")
    duplicate = copy.deepcopy(changed[index])
    duplicate["seq"] = changed[-1]["seq"] + 1
    changed.append(duplicate)
    results["duplicate_assignment_id"] = not verify_event_log(changed, activation_round)

    changed = copy.deepcopy(events)
    index = next(i for i, row in enumerate(changed) if row["kind"] == "Credited")
    changed[index]["amount"] += Fraction(1, 16)
    results["credit_overdraw"] = not verify_event_log(changed, activation_round)

    changed = copy.deepcopy(events)
    index = next(i for i, row in enumerate(changed) if row["kind"] == "Credited")
    credit = changed[index]
    fake = {
        "seq": credit["seq"] + 1,
        "kind": "UnassignedReserved",
        "round": credit["round"],
        "policy": POLICY,
        "source": credit["destination"],
        "token_id": (credit["destination"], 999999),
        "amount": Fraction(1, 16),
        "face_amount": Fraction(1, 16),
        "source_evidence": Fraction(10),
    }
    for later in changed[index + 1:]:
        later["seq"] += 1
    changed.insert(index + 1, fake)
    results["incoming_credit_cannot_fund_reserve"] = not verify_event_log(
        changed, activation_round
    )
    return results


def run(traces: int = DEFAULT_TRACES) -> dict:
    event_counts = {
        "Activate": 0, "Assigned": 0, "Credited": 0,
        "Evidence": 0, "UnassignedReserved": 0, "LookCommit": 0
    }
    total_events = 0
    first_trace = None
    for trace_id in range(traces):
        events, activation = build_trace(trace_id)
        total_events += len(events)
        for event in events:
            event_counts[event["kind"]] += 1
        if not verify_event_log(events, activation):
            raise AssertionError(f"independent replay failed at trace {trace_id}")
        if trace_id == 0:
            first_trace = (events, activation)
    assert first_trace is not None
    controls = negative_controls(*first_trace)
    assert all(controls.values())
    return {
        "scope": "exact Fraction accounting/protocol screen only; not e-process validity, FDR/SupFDR, or Rust",
        "traces": traces,
        "hypotheses": N,
        "rounds": ROUNDS,
        "gamma": "1/16 per candidate; total mass 1",
        "activation": "deterministic predictable blocks of four candidates",
        "assignment": "past-only; destination fixed at assignment; assignment follows reservation",
        "arithmetic": "fractions.Fraction exact",
        "policy_digest": POLICY,
        "events_replayed": total_events,
        "event_counts": event_counts,
        "all_base_replays_passed": True,
        "negative_controls": controls,
        "all_negative_controls_rejected": True
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=DEFAULT_TRACES)
    args = parser.parse_args()
    if args.traces < 1:
        raise SystemExit("--traces must be positive")
    print(json.dumps(run(args.traces), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
