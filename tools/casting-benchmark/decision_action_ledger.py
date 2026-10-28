#!/usr/bin/env python3
"""Hash-chained JSONL action/outcome ledger for casting decision experiments.

Single-writer only: the append API verifies existing bytes before appending,
and the verifier detects edits, truncation, reordering, duplicates and bad links.
This is tamper-evident, not a signature or an independently trusted timestamp.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
LEDGER_DOMAIN = b"CASTING-DECISION-ACTION-LEDGER-V1\0"
ZERO_SHA256 = "0" * 64
ACTION_DISPOSITIONS = frozenset({"evaluate", "escalate", "replicate"})
ALL_DISPOSITIONS = ACTION_DISPOSITIONS | {"stop", "abstain"}
OUTCOME_STATUSES = frozenset({"succeeded", "failed", "timeout", "invalid", "refused"})

EVENT_FIELDS = frozenset({
    "record_type",
    "task_id",
    "run_id",
    "action_id",
    "state_sha256",
    "policy_sha256",
    "disposition",
    "reason_code",
    "candidate_id",
    "fidelity",
    "estimated_cost_units",
    "decision_loss_reduction",
    "normalized_constraint_risk_reduction",
    "value_per_cost",
    "actual_cost_units",
    "outcome_status",
    "outcome_sha256",
    "resulting_state_sha256",
})
RECORD_FIELDS = EVENT_FIELDS | {
    "schema_version", "sequence", "previous_record_sha256", "record_sha256"
}


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically and reject NaN/Infinity."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without edge whitespace")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    if set(event) != EVENT_FIELDS:
        raise ValueError(
            f"event keys mismatch; missing={sorted(EVENT_FIELDS - set(event))}, "
            f"extra={sorted(set(event) - EVENT_FIELDS)}"
        )

    if not isinstance(event["record_type"], str) or event["record_type"] not in {"decision", "outcome"}:
        raise ValueError("record_type must be decision or outcome")
    for field in ("task_id", "run_id", "reason_code"):
        _text(event[field], field)
    _digest(event["state_sha256"], "state_sha256")
    _digest(event["policy_sha256"], "policy_sha256")

    disposition = event["disposition"]
    if not isinstance(disposition, str) or disposition not in ALL_DISPOSITIONS:
        raise ValueError(f"disposition must be one of {sorted(ALL_DISPOSITIONS)}")
    action_id = _optional_text(event["action_id"], "action_id")
    candidate_id = _optional_text(event["candidate_id"], "candidate_id")
    fidelity = _optional_text(event["fidelity"], "fidelity")
    estimated_cost = _number(event["estimated_cost_units"], "estimated_cost_units")
    if estimated_cost < 0.0:
        raise ValueError("estimated_cost_units must be non-negative")

    loss_value = event["decision_loss_reduction"]
    risk_value = event["normalized_constraint_risk_reduction"]
    score_value = event["value_per_cost"]

    if event["record_type"] == "decision":
        if event["actual_cost_units"] is not None:
            raise ValueError("decision record actual_cost_units must be null")
        if event["outcome_status"] is not None:
            raise ValueError("decision record outcome_status must be null")
        if event["outcome_sha256"] is not None:
            raise ValueError("decision record outcome_sha256 must be null")
        if event["resulting_state_sha256"] is not None:
            raise ValueError("decision record resulting_state_sha256 must be null")
        if disposition in ACTION_DISPOSITIONS:
            if action_id is None or candidate_id is None or fidelity is None:
                raise ValueError("evaluation decisions require action_id, candidate_id and fidelity")
            if estimated_cost <= 0.0:
                raise ValueError("action decisions require strictly positive estimated cost")
            loss = _number(loss_value, "decision_loss_reduction")
            risk = _number(risk_value, "normalized_constraint_risk_reduction")
            score = _number(score_value, "value_per_cost")
            if loss <= 0.0 or score <= 0.0:
                raise ValueError("action decisions require positive loss reduction and value_per_cost")
            if not 0.0 <= risk <= 1.0:
                raise ValueError("normalized_constraint_risk_reduction must be in [0, 1]")
            if not math.isclose(score, loss / estimated_cost, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("value_per_cost must equal decision_loss_reduction / estimated_cost_units")
        else:
            if action_id is not None or candidate_id is not None or fidelity is not None:
                raise ValueError("stop/abstain decisions must not name an evaluation action")
            if estimated_cost != 0.0:
                raise ValueError("stop/abstain decisions must have zero estimated cost")
            if loss_value is not None or risk_value is not None or score_value is not None:
                raise ValueError("stop/abstain decisions must not carry action-value scores")
    else:
        if disposition not in ACTION_DISPOSITIONS:
            raise ValueError("outcome records must reference evaluate/escalate/replicate decisions")
        if action_id is None or candidate_id is None or fidelity is None:
            raise ValueError("outcome records require action_id, candidate_id and fidelity")
        if estimated_cost <= 0.0:
            raise ValueError("outcome records require strictly positive estimated cost")
        loss = _number(loss_value, "decision_loss_reduction")
        risk = _number(risk_value, "normalized_constraint_risk_reduction")
        score = _number(score_value, "value_per_cost")
        if loss <= 0.0 or score <= 0.0:
            raise ValueError("outcome records require positive decision values")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("normalized_constraint_risk_reduction must be in [0, 1]")
        if not math.isclose(score, loss / estimated_cost, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("value_per_cost must equal decision_loss_reduction / estimated_cost_units")
        actual_cost = _number(event["actual_cost_units"], "actual_cost_units")
        if actual_cost < 0.0:
            raise ValueError("actual_cost_units must be non-negative")
        status = event["outcome_status"]
        if not isinstance(status, str) or status not in OUTCOME_STATUSES:
            raise ValueError(f"outcome_status must be one of {sorted(OUTCOME_STATUSES)}")
        _digest(event["outcome_sha256"], "outcome_sha256")
        _digest(event["resulting_state_sha256"], "resulting_state_sha256")

    return event


def decision_event(
    *,
    task_id: str,
    run_id: str,
    state_sha256: str,
    policy_sha256: str,
    disposition: str,
    reason_code: str,
    action_id: str | None = None,
    candidate_id: str | None = None,
    fidelity: str | None = None,
    estimated_cost_units: float = 0.0,
    decision_loss_reduction: float | None = None,
    normalized_constraint_risk_reduction: float | None = None,
    value_per_cost: float | None = None,
) -> dict[str, Any]:
    """Construct a decision event. Stopping/abstention uses zero estimated cost."""
    event = {
        "record_type": "decision",
        "task_id": task_id,
        "run_id": run_id,
        "action_id": action_id,
        "state_sha256": state_sha256,
        "policy_sha256": policy_sha256,
        "disposition": disposition,
        "reason_code": reason_code,
        "candidate_id": candidate_id,
        "fidelity": fidelity,
        "estimated_cost_units": estimated_cost_units,
        "decision_loss_reduction": decision_loss_reduction,
        "normalized_constraint_risk_reduction": normalized_constraint_risk_reduction,
        "value_per_cost": value_per_cost,
        "actual_cost_units": None,
        "outcome_status": None,
        "outcome_sha256": None,
        "resulting_state_sha256": None,
    }
    return _validate_event(event)


def outcome_event(
    decision: dict[str, Any],
    *,
    actual_cost_units: float,
    outcome_status: str,
    outcome_sha256: str,
    resulting_state_sha256: str,
    reason_code: str,
) -> dict[str, Any]:
    """Construct an outcome event bound to its original decision state and action."""
    if not isinstance(decision, dict) or decision.get("record_type") != "decision":
        raise ValueError("outcome must reference a decision record")
    event = {
        "record_type": "outcome",
        "task_id": decision.get("task_id"),
        "run_id": decision.get("run_id"),
        "action_id": decision.get("action_id"),
        "state_sha256": decision.get("state_sha256"),
        "policy_sha256": decision.get("policy_sha256"),
        "disposition": decision.get("disposition"),
        "reason_code": reason_code,
        "candidate_id": decision.get("candidate_id"),
        "fidelity": decision.get("fidelity"),
        "estimated_cost_units": decision.get("estimated_cost_units"),
        "decision_loss_reduction": decision.get("decision_loss_reduction"),
        "normalized_constraint_risk_reduction": decision.get("normalized_constraint_risk_reduction"),
        "value_per_cost": decision.get("value_per_cost"),
        "actual_cost_units": actual_cost_units,
        "outcome_status": outcome_status,
        "outcome_sha256": outcome_sha256,
        "resulting_state_sha256": resulting_state_sha256,
    }
    return _validate_event(event)


def _make_record(event: dict[str, Any], sequence: int, previous_sha256: str) -> dict[str, Any]:
    validated = _validate_event(event)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "previous_record_sha256": previous_sha256,
        **validated,
    }
    record["record_sha256"] = hashlib.sha256(
        LEDGER_DOMAIN + canonical_json(record)
    ).hexdigest()
    return record


def _validate_record_shape(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise ValueError("record shape or keys are invalid")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    sequence = record["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    _digest(record["previous_record_sha256"], "previous_record_sha256")
    _digest(record["record_sha256"], "record_sha256")
    event = {key: record[key] for key in EVENT_FIELDS}
    _validate_event(event)
    return record


def verify_ledger_bytes(raw: bytes) -> dict[str, Any]:
    """Verify byte-level JSONL format, hash chain and decision/outcome linkage."""
    if not isinstance(raw, bytes):
        raise ValueError("ledger input must be bytes")
    if raw == b"":
        return {
            "valid": True,
            "record_count": 0,
            "decision_count": 0,
            "outcome_count": 0,
            "unresolved_action_count": 0,
            "total_estimated_action_cost_units": 0.0,
            "total_actual_action_cost_units": 0.0,
            "actual_minus_estimated_cost_units": 0.0,
            "last_record_sha256": ZERO_SHA256,
            "records": [],
        }
    if not raw.endswith(b"\n"):
        raise ValueError("ledger must end with a newline; trailing partial record")
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        raise ValueError("blank ledger lines are forbidden")

    records: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    completed: set[tuple[str, str, str]] = set()
    previous = ZERO_SHA256
    total_estimated = 0.0
    total_actual = 0.0
    decision_count = 0
    outcome_count = 0

    for expected_sequence, line in enumerate(lines):
        try:
            record = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON at record {expected_sequence}") from exc
        _validate_record_shape(record)
        if record["sequence"] != expected_sequence:
            raise ValueError("ledger sequence is not contiguous")
        if record["previous_record_sha256"] != previous:
            raise ValueError("previous_record_sha256 does not link to prior record")

        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        calculated = hashlib.sha256(LEDGER_DOMAIN + canonical_json(unsigned)).hexdigest()
        if calculated != record["record_sha256"]:
            raise ValueError("record_sha256 does not match canonical record bytes")

        key_action = None
        if record["action_id"] is not None:
            key_action = (record["task_id"], record["run_id"], record["action_id"])

        if record["record_type"] == "decision":
            decision_count += 1
            if record["disposition"] in ACTION_DISPOSITIONS:
                if key_action is None:
                    raise ValueError("action decision is missing its action identity")
                if key_action in decisions:
                    raise ValueError("duplicate action_id decision in task/run")
                decisions[key_action] = record
                total_estimated += float(record["estimated_cost_units"])
        else:
            outcome_count += 1
            if key_action is None:
                raise ValueError("outcome is missing its action identity")
            if key_action not in decisions:
                raise ValueError("outcome does not reference a prior action decision")
            if key_action in completed:
                raise ValueError("duplicate outcome for action")
            decision = decisions[key_action]
            for field in (
                "task_id", "run_id", "action_id", "state_sha256", "policy_sha256",
                "disposition", "candidate_id", "fidelity", "estimated_cost_units",
                "decision_loss_reduction", "normalized_constraint_risk_reduction", "value_per_cost",
            ):
                if record[field] != decision[field]:
                    raise ValueError(f"outcome {field} differs from its decision")
            completed.add(key_action)
            total_actual += float(record["actual_cost_units"])

        records.append(record)
        previous = record["record_sha256"]

    return {
        "valid": True,
        "record_count": len(records),
        "decision_count": decision_count,
        "outcome_count": outcome_count,
        "unresolved_action_count": len(decisions) - len(completed),
        "total_estimated_action_cost_units": total_estimated,
        "total_actual_action_cost_units": total_actual,
        "actual_minus_estimated_cost_units": total_actual - total_estimated,
        "last_record_sha256": previous,
        "records": records,
    }


def append_record(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Verify then append one record (single writer; do not share concurrent writers)."""
    target = Path(path)
    existing = target.read_bytes() if target.exists() else b""
    status = verify_ledger_bytes(existing)
    records = status["records"]
    sequence = len(records)
    previous = records[-1]["record_sha256"] if records else ZERO_SHA256
    record = _make_record(event, sequence, previous)
    line = canonical_json(record) + b"\n"

    # Verify linkage and event pairing before mutating persistent bytes.
    verify_ledger_bytes(existing + line)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return record


if __name__ == "__main__":
    raise SystemExit(
        "Library module: call decision_event/outcome_event, append_record, and verify_ledger_bytes."
    )
