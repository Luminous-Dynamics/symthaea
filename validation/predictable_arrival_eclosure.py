#!/usr/bin/env python3
"""Predictable random-arrival ledger and bounded-envelope dynamic e-closure.

This is an exact finite-catalogue research oracle. It checks that every
activation is committed against the pre-round evidence state and before any
evidence increment in that round. It then replays the existing self-financing
projective portfolio and capacity selector.

The ledger audit does not authenticate wall-clock time, signatures, or the
truth of the upstream conditional e-value contract. Those remain external.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from typing import Any, Sequence

from self_financing_projective_eclosure import (
    as_fraction,
    fraction_pair,
    select_self_financing_capacity_actions,
    verify_self_financing_sequence,
)


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _catalogue(hypotheses: Sequence[dict[str, Any]]):
    if not hypotheses:
        raise ValueError("the deterministic finite candidate envelope is empty")
    rows = []
    seen = set()
    for row in hypotheses:
        if not isinstance(row, dict):
            raise TypeError("each hypothesis catalogue row must be a mapping")
        hypothesis_id = row.get("id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("hypothesis ids must be nonempty strings")
        if hypothesis_id in seen:
            raise ValueError("duplicate hypothesis id in deterministic envelope")
        seen.add(hypothesis_id)
        score = as_fraction(row.get("priority_score"))
        value = as_fraction(row.get("value"))
        cost = as_fraction(row.get("cost"))
        if score < 0 or value < 0 or cost < 0:
            raise ValueError("priority scores, values and costs must be nonnegative")
        rows.append({
            "id": hypothesis_id,
            "priority_score": score,
            "value": value,
            "cost": cost,
        })
    # Stable ID order is the coordinate order used by masks and receipts.
    rows.sort(key=lambda item: item["id"])
    payload = [
        {
            "id": row["id"],
            "priority_score": fraction_pair(row["priority_score"]),
            "value": fraction_pair(row["value"]),
            "cost": fraction_pair(row["cost"]),
        }
        for row in rows
    ]
    return rows, _canonical_sha256({
        "schema": "symthaea-finite-arrival-envelope-v1",
        "hypotheses": payload,
    })


def activation_state_digest(
    hypotheses: Sequence[dict[str, Any]],
    round_index: int,
    active_before: Sequence[str],
    activation_history: Sequence[Sequence[Any]],
    evidence_history: Sequence[Sequence[Any]],
    *,
    context_sha256: str | None = None,
) -> str:
    """Digest the *recorded* prefix state immediately before one round.

    History tuples are (round, hypothesis_id) for activations and
    (round, hypothesis_id, factor) for evidence increments. Every external
    covariate or other input consulted by the arrival policy must also be
    committed in the event log by the caller; this digest cannot discover
    omitted filtration information. It is a tamper-evident binding, not a
    digital signature or trusted timestamp.
    """
    if context_sha256 is not None and (
        not isinstance(context_sha256, str)
        or len(context_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in context_sha256)
    ):
        raise ValueError("context_sha256 must be a lowercase 64-character SHA-256 digest")
    rows, catalogue_sha256 = _catalogue(hypotheses)
    known = {row["id"] for row in rows}
    active = sorted(active_before)
    if len(active) != len(set(active)) or not set(active).issubset(known):
        raise ValueError("active_before must contain unique catalogue ids")
    activations = []
    for item in activation_history:
        if len(item) != 2:
            raise ValueError("activation history entries must have length two")
        r, hypothesis_id = item
        if hypothesis_id not in known:
            raise ValueError("activation history contains an unknown id")
        activations.append([int(r), hypothesis_id])
    evidence = []
    for item in evidence_history:
        if len(item) != 3:
            raise ValueError("evidence history entries must have length three")
        r, hypothesis_id, factor = item
        if hypothesis_id not in known:
            raise ValueError("evidence history contains an unknown id")
        evidence.append([int(r), hypothesis_id, fraction_pair(as_fraction(factor))])
    return _canonical_sha256({
        "schema": "symthaea-pre-evidence-recorded-state-v1",
        "catalogue_sha256": catalogue_sha256,
        "round": int(round_index),
        "active_before": active,
        "activation_history": activations,
        "evidence_history": evidence,
        "external_context_sha256": context_sha256,
    })


def _replay_arrival_ledger(
    hypotheses: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    round_count: int,
) -> dict[str, Any]:
    rows, catalogue_sha256 = _catalogue(hypotheses)
    if isinstance(round_count, bool) or not isinstance(round_count, int) or round_count < 1:
        raise ValueError("round_count must be a positive integer")
    if not events:
        raise ValueError("arrival ledger must contain at least one event")
    ids = [row["id"] for row in rows]
    index = {hypothesis_id: i for i, hypothesis_id in enumerate(ids)}
    groups: list[list[dict[str, Any]]] = [[] for _ in range(round_count)]

    previous_round = -1
    for expected_seq, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError("ledger events must be mappings")
        seq = event.get("seq")
        round_index = event.get("round")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != expected_seq:
            raise ValueError("ledger sequence numbers must be contiguous and ordered")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise ValueError("event round must be an integer")
        if not 0 <= round_index < round_count:
            raise ValueError("event round is outside the declared horizon")
        if round_index < previous_round or round_index > previous_round + 1:
            raise ValueError("ledger rounds must be contiguous and nondecreasing")
        previous_round = round_index
        groups[round_index].append(event)

    if any(not group for group in groups):
        raise ValueError("every round needs a complete active-coordinate evidence batch")

    active: set[str] = set()
    activation_history: list[list[Any]] = []
    evidence_history: list[list[Any]] = []
    activation_rounds: dict[str, int] = {}
    active_masks = []
    increment_paths = [[Fraction(1) for _ in range(round_count)] for _ in rows]
    activation_checks = 0
    evidence_checks = 0

    for round_index, group in enumerate(groups):
        phase = "activation"
        context_seen = False
        round_context_sha256 = None
        pending: list[str] = []
        evidence_for_round: dict[str, Fraction] = {}
        last_activation_index = -1
        last_evidence_index = -1

        for event in group:
            kind = event.get("kind")
            hypothesis_id = event.get("hypothesis_id")
            if kind == "activation_commit":
                if phase == "evidence":
                    raise ValueError("activation_after_evidence: activation must precede every current-round increment")
                if hypothesis_id not in index:
                    raise ValueError("activation is outside the deterministic finite envelope")
                if hypothesis_id in active or hypothesis_id in pending:
                    raise ValueError("hypothesis activation is duplicated")
                event_context = event.get("context_sha256")
                if event_context is not None and (
                    not isinstance(event_context, str)
                    or len(event_context) != 64
                    or any(ch not in "0123456789abcdef" for ch in event_context)
                ):
                    raise ValueError("activation context_sha256 must be a lowercase 64-character SHA-256 digest")
                if context_seen and event_context != round_context_sha256:
                    raise ValueError("same-round activation commitments must bind the same external context")
                if not context_seen:
                    context_seen = True
                    round_context_sha256 = event_context
                expected_state_sha256 = activation_state_digest(
                    hypotheses,
                    round_index,
                    sorted(active),
                    activation_history,
                    evidence_history,
                    context_sha256=event_context,
                )
                if event.get("pre_state_sha256") != expected_state_sha256:
                    raise ValueError("activation commitment does not bind to the pre-evidence recorded prefix and context digest")
                coordinate = index[hypothesis_id]
                if coordinate <= last_activation_index:
                    raise ValueError("activation commits must use canonical catalogue order")
                last_activation_index = coordinate
                pending.append(hypothesis_id)
                activation_checks += 1
            elif kind == "evidence_increment":
                phase = "evidence"
                if hypothesis_id not in active and hypothesis_id not in pending:
                    raise ValueError("evidence increment supplied for a not-yet-active hypothesis")
                if hypothesis_id in evidence_for_round:
                    raise ValueError("duplicate coordinate increment in one round")
                coordinate = index[hypothesis_id]
                if coordinate <= last_evidence_index:
                    raise ValueError("evidence increments must use canonical catalogue order")
                last_evidence_index = coordinate
                factor = as_fraction(event.get("factor"))
                if factor < 0:
                    raise ValueError("one-step e-value factors must be nonnegative")
                evidence_for_round[hypothesis_id] = factor
            else:
                raise ValueError("unknown ledger event kind")

        active.update(pending)
        if not active:
            raise ValueError("the active family must be nonempty after the first round")
        if set(evidence_for_round) != active:
            missing = sorted(active - set(evidence_for_round))
            extra = sorted(set(evidence_for_round) - active)
            raise ValueError(
                f"round evidence must cover every active coordinate exactly once; missing={missing}, extra={extra}"
            )

        mask = 0
        for hypothesis_id in active:
            mask |= 1 << index[hypothesis_id]
        active_masks.append(mask)

        for hypothesis_id in pending:
            activation_rounds[hypothesis_id] = round_index
            activation_history.append([round_index, hypothesis_id])
        for hypothesis_id in ids:
            if hypothesis_id in active:
                factor = evidence_for_round[hypothesis_id]
                increment_paths[index[hypothesis_id]][round_index] = factor
                evidence_history.append([round_index, hypothesis_id, factor])
                evidence_checks += 1

    canonical_events = []
    for event in events:
        item = {
            "seq": event["seq"],
            "round": event["round"],
            "kind": event["kind"],
            "hypothesis_id": event["hypothesis_id"],
        }
        if item["kind"] == "activation_commit":
            item["pre_state_sha256"] = event["pre_state_sha256"]
            if event.get("context_sha256") is not None:
                item["context_sha256"] = event["context_sha256"]
        else:
            item["factor"] = fraction_pair(as_fraction(event["factor"]))
        canonical_events.append(item)

    ledger_sha256 = _canonical_sha256({
        "schema": "symthaea-predictable-arrival-ledger-v1",
        "catalogue_sha256": catalogue_sha256,
        "round_count": round_count,
        "events": canonical_events,
    })
    return {
        "catalogue": rows,
        "catalogue_sha256": catalogue_sha256,
        "ledger_sha256": ledger_sha256,
        "active_masks": tuple(active_masks),
        "increment_paths": tuple(tuple(row) for row in increment_paths),
        "activation_rounds": activation_rounds,
        "activation_checks": activation_checks,
        "evidence_checks": evidence_checks,
        "canonical_events": canonical_events,
    }


def audit_predictable_arrival_ledger(
    hypotheses: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    round_count: int,
) -> dict:
    """Audit predictable activation ordering and deterministic-envelope bounds."""
    replay = _replay_arrival_ledger(hypotheses, events, round_count)
    return {
        "valid": True,
        "schema": "symthaea-predictable-arrival-audit-v1",
        "catalogue_ids": [row["id"] for row in replay["catalogue"]],
        "catalogue_sha256": replay["catalogue_sha256"],
        "ledger_sha256": replay["ledger_sha256"],
        "active_masks": list(replay["active_masks"]),
        "activation_rounds": replay["activation_rounds"],
        "activation_commitments_checked": replay["activation_checks"],
        "evidence_increments_checked": replay["evidence_checks"],
        "round_count": round_count,
        "scope": (
            "pathwise ordering and digest audit; it does not authenticate a "
            "trusted clock, signer, filtration, or conditional e-value law"
        ),
    }


def select_predictable_arrival_capacity_actions(
    hypotheses: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    alpha,
    cumulative_capacity: Sequence[Any],
    thresholds: Sequence[Any],
    adjuster_increments: Sequence[Any],
    *,
    share_policy: str = "static_priority",
    exhaustive_limit: int = 8,
) -> dict:
    """Replay a finite-envelope predictable arrival ledger and select actions."""
    replay = _replay_arrival_ledger(hypotheses, events, len(cumulative_capacity))
    rows = replay["catalogue"]
    selection = select_self_financing_capacity_actions(
        replay["increment_paths"],
        replay["active_masks"],
        [row["priority_score"] for row in rows],
        alpha,
        [row["value"] for row in rows],
        [row["cost"] for row in rows],
        cumulative_capacity,
        thresholds,
        adjuster_increments,
        share_policy=share_policy,
        exhaustive_limit=exhaustive_limit,
    )
    audit = {
        "valid": True,
        "schema": "symthaea-predictable-arrival-audit-v1",
        "catalogue_ids": [row["id"] for row in rows],
        "catalogue_sha256": replay["catalogue_sha256"],
        "ledger_sha256": replay["ledger_sha256"],
        "active_masks": list(replay["active_masks"]),
        "activation_rounds": replay["activation_rounds"],
        "activation_commitments_checked": replay["activation_checks"],
        "evidence_increments_checked": replay["evidence_checks"],
        "round_count": len(cumulative_capacity),
        "scope": (
            "pathwise ordering and digest audit; it does not authenticate a "
            "trusted clock, signer, filtration, or conditional e-value law"
        ),
    }
    return {
        "schema": "symthaea-predictable-arrival-capacity-v1",
        "status": "EXACT_FINITE_ENVELOPE_RESEARCH_ORACLE_NOT_PRODUCTION",
        "arrival_audit": audit,
        "share_policy": share_policy,
        "alpha": fraction_pair(as_fraction(alpha)),
        "selection": selection,
        "theorem_scope": (
            "Conditional on predictable nested activation, a deterministic finite "
            "candidate envelope at every finite horizon, valid one-step coordinate "
            "e-value factors in one common global filtration, and the common valid "
            "time-monotone adjuster, the finite-envelope localization argument "
            "supports simultaneous stopped-FDR and SupFDR. The unrestricted "
            "unbounded random-arrival case is not covered."
        ),
        "limitations": [
            "the candidate catalogue is a deterministic finite envelope fixed before the data stream",
            "activation commits must be F_(t-1)-measurable and precede all round-t evidence increments",
            "the ledger digest does not authenticate wall-clock ordering or signer identity",
            "the upstream conditional expectation contract E[X_i,t | F_(t-1)] <= 1 is assumed, not inferred from paths",
            "all coordinate processes must use one common global filtration",
            "capacity is cumulative; no per-period queue service or delay guarantee",
            "the wrapped selector is an exact exponential small-family oracle",
        ],
    }


def verify_predictable_arrival_capacity_result(
    result: dict,
    hypotheses: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    alpha,
    cumulative_capacity: Sequence[Any],
    thresholds: Sequence[Any],
    adjuster_increments: Sequence[Any],
    *,
    share_policy: str = "static_priority",
) -> dict:
    """Independently replay event receipts, closure, and exact action optimality."""
    failures = []
    try:
        replay = _replay_arrival_ledger(hypotheses, events, len(cumulative_capacity))
    except (TypeError, ValueError, ArithmeticError) as error:
        return {
            "valid": False,
            "failures": [("arrival_ledger_invalid", str(error))],
            "optimality_verified": False,
        }

    if result.get("schema") != "symthaea-predictable-arrival-capacity-v1":
        failures.append(("result_schema_mismatch",))
    if result.get("share_policy") != share_policy:
        failures.append(("share_policy_receipt_mismatch",))
    if result.get("alpha") != fraction_pair(as_fraction(alpha)):
        failures.append(("alpha_receipt_mismatch",))

    audit = result.get("arrival_audit", {})
    if audit.get("schema") != "symthaea-predictable-arrival-audit-v1" or audit.get("valid") is not True:
        failures.append(("arrival_audit_schema_or_validity_mismatch",))
    if audit.get("round_count") != len(cumulative_capacity):
        failures.append(("arrival_round_count_mismatch",))
    if audit.get("catalogue_ids") != [row["id"] for row in replay["catalogue"]]:
        failures.append(("catalogue_id_receipt_mismatch",))
    if audit.get("activation_commitments_checked") != replay["activation_checks"]:
        failures.append(("activation_count_receipt_mismatch",))
    if audit.get("evidence_increments_checked") != replay["evidence_checks"]:
        failures.append(("evidence_count_receipt_mismatch",))
    if audit.get("catalogue_sha256") != replay["catalogue_sha256"]:
        failures.append(("catalogue_digest_mismatch",))
    if audit.get("ledger_sha256") != replay["ledger_sha256"]:
        failures.append(("arrival_ledger_digest_mismatch",))
    if audit.get("active_masks") != list(replay["active_masks"]):
        failures.append(("active_mask_receipt_mismatch",))
    if audit.get("activation_rounds") != replay["activation_rounds"]:
        failures.append(("activation_round_receipt_mismatch",))

    rows = replay["catalogue"]
    selection = result.get("selection", {})
    if selection.get("schema") != "symthaea-self-financing-projective-eclosure-v1":
        failures.append(("selection_schema_mismatch",))
    if selection.get("share_policy") != share_policy:
        failures.append(("selection_share_policy_mismatch",))
    base_check = verify_self_financing_sequence(
        selection.get("decisions", []),
        replay["increment_paths"],
        replay["active_masks"],
        tuple(row["priority_score"] for row in rows),
        as_fraction(alpha),
        tuple(row["value"] for row in rows),
        tuple(row["cost"] for row in rows),
        tuple(as_fraction(v) for v in cumulative_capacity),
        thresholds,
        adjuster_increments,
        share_policy=share_policy,
    )
    if not base_check.get("valid"):
        failures.append(("selection_replay_failed", base_check.get("failures", [])))
    return {
        "valid": not failures,
        "failures": failures,
        "optimality_verified": bool(base_check.get("optimality_verified", False)) and not failures,
        "arrival_commitments_checked": replay["activation_checks"],
        "evidence_increments_checked": replay["evidence_checks"],
        "base_selection_verification": base_check,
        "scope": "finite-envelope event ordering plus exact deterministic selector replay",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_predictable_arrival_eclosure.py for exact fixtures.")
