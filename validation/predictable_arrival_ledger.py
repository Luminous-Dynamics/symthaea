"""Event-ordered replay for predictable nested hypothesis arrivals.

This research ledger verifies a declared lagged-observation arrival policy.
It proves only that the recorded simulator/event stream follows that policy
before each round's evidence is exposed. Hashes do not prove source truth,
clock chronology, filtration completeness, or authenticity of external data.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

POLICY_ID = "lagged-last-active-mean-positive-v1"
SCHEMA = "predictable-arrival-ledger-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _score_hex(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid score")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("scores must be finite")
    return number.hex()


def _decode_score(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("score must use canonical float.hex encoding")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid hexadecimal score") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError("score encoding is non-finite or non-canonical")
    return number


def _validate_ids(values: Iterable[str], name: str) -> list[str]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def _history_payload(
    hypothesis_ids: Sequence[str],
    histories: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": hypothesis_id,
            "observed_scores_hex": [float(x).hex() for x in histories[hypothesis_id]],
        }
        for hypothesis_id in hypothesis_ids
    ]


def _expected_activation(
    hypothesis_ids: Sequence[str],
    active_before: Sequence[str],
    histories: Mapping[str, Sequence[float]],
    *,
    min_observations: int,
    mean_threshold: float,
) -> list[str]:
    """Return at most the next stable ID using strictly pre-round history."""
    if len(active_before) >= len(hypothesis_ids):
        return []
    last_id = active_before[-1]
    past = histories[last_id]
    if len(past) < min_observations:
        return []
    if sum(past) / len(past) > mean_threshold:
        return [hypothesis_ids[len(active_before)]]
    return []


def _build_ledger(
    scores_by_round: Sequence[Sequence[float]],
    hypothesis_ids: Sequence[str],
    *,
    initial_active_count: int,
    min_observations: int,
    mean_threshold: float,
    use_same_round_score: bool,
) -> dict[str, Any]:
    ids = _validate_ids(hypothesis_ids, "hypothesis_ids")
    if not scores_by_round:
        raise ValueError("at least one score round is required")
    if (
        isinstance(initial_active_count, bool)
        or not isinstance(initial_active_count, int)
        or not 1 <= initial_active_count <= len(ids)
    ):
        raise ValueError("initial_active_count must be within the universe")
    if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 1:
        raise ValueError("min_observations must be a positive integer")
    threshold = float(mean_threshold)
    if not math.isfinite(threshold):
        raise ValueError("mean_threshold must be finite")

    matrix: list[list[str]] = []
    for row in scores_by_round:
        if len(row) != len(ids):
            raise ValueError("every score row must match the hypothesis universe")
        matrix.append([_score_hex(value) for value in row])

    header = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "parameters": {
            "initial_active_count": initial_active_count,
            "min_observations": min_observations,
            "mean_threshold_hex": threshold.hex(),
        },
        "hypothesis_ids": ids,
        "decision_phase": "PRE_OBSERVATION",
        "observation_phase": "POST_DECISION",
    }
    header_hash = _digest(header)
    histories: dict[str, list[float]] = {hypothesis_id: [] for hypothesis_id in ids}
    active = ids[:initial_active_count]
    previous_hash = header_hash
    events: list[dict[str, Any]] = []

    for round_index, score_row in enumerate(matrix):
        active_before = list(active)
        pre_history_digest = _digest(_history_payload(ids, histories))
        decision_input_hash = _digest(
            {
                "policy_id": POLICY_ID,
                "round": round_index,
                "parameters": header["parameters"],
                "active_before": active_before,
                "pre_history_digest": pre_history_digest,
            }
        )

        activated = _expected_activation(
            ids,
            active_before,
            histories,
            min_observations=min_observations,
            mean_threshold=threshold,
        )
        if use_same_round_score and len(active_before) < len(ids):
            last_id = active_before[-1]
            last_values = list(histories[last_id])
            last_values.append(_decode_score(score_row[len(active_before) - 1]))
            if (
                len(last_values) >= min_observations
                and sum(last_values) / len(last_values) > threshold
            ):
                activated = [ids[len(active_before)]]

        active_after = active_before + activated
        decision_output_hash = _digest(
            {
                "activated": activated,
                "active_after": active_after,
            }
        )
        observed_scores = [
            [hypothesis_id, score_row[ids.index(hypothesis_id)]]
            for hypothesis_id in active_after
        ]
        event = {
            "round": round_index,
            "decision_phase": "PRE_OBSERVATION",
            "observation_phase": "POST_DECISION",
            "active_before": active_before,
            "activated": activated,
            "active_after": active_after,
            "pre_history_digest": pre_history_digest,
            "decision_input_hash": decision_input_hash,
            "decision_output_hash": decision_output_hash,
            "previous_event_hash": previous_hash,
            "observed_scores": observed_scores,
        }
        event_hash = _digest(event)
        events.append({**event, "event_hash": event_hash})

        # Only now do current-round scores enter the history used at t+1.
        for hypothesis_id, score_hex in observed_scores:
            histories[hypothesis_id].append(_decode_score(score_hex))
        active = active_after
        previous_hash = event_hash

    return {
        "header": header,
        "header_hash": header_hash,
        "events": events,
        "final_event_hash": previous_hash,
    }


def build_predictable_arrival_ledger(
    scores_by_round: Sequence[Sequence[float]],
    hypothesis_ids: Sequence[str],
    *,
    initial_active_count: int = 1,
    min_observations: int = 5,
    mean_threshold: float = 0.0,
) -> dict[str, Any]:
    """Build a valid lagged-mean arrival ledger.

    At the start of round t, activate the next ID iff the most recently active
    coordinate has at least min_observations strictly prior observations and
    its prior sample mean is above mean_threshold. Then, and only then, record
    current-round scores for the newly formed active set.
    """
    return _build_ledger(
        scores_by_round,
        hypothesis_ids,
        initial_active_count=initial_active_count,
        min_observations=min_observations,
        mean_threshold=mean_threshold,
        use_same_round_score=False,
    )


def build_same_round_lookahead_negative_control(
    scores_by_round: Sequence[Sequence[float]],
    hypothesis_ids: Sequence[str],
    *,
    initial_active_count: int = 1,
    min_observations: int = 5,
    mean_threshold: float = 0.0,
) -> dict[str, Any]:
    """Construct a hash-consistent but statistically invalid lookahead ledger.

    This is only a negative-control fixture. It intentionally lets the current
    round's score influence activation before that score is recorded.
    """
    return _build_ledger(
        scores_by_round,
        hypothesis_ids,
        initial_active_count=initial_active_count,
        min_observations=min_observations,
        mean_threshold=mean_threshold,
        use_same_round_score=True,
    )


def verify_predictable_arrival_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Replay and verify phase order, pre-state policy decisions, and hash chain."""
    header = ledger.get("header")
    events = ledger.get("events")
    if not isinstance(header, dict) or not isinstance(events, list) or not events:
        raise ValueError("ledger header and nonempty events are required")
    if header.get("schema") != SCHEMA or header.get("policy_id") != POLICY_ID:
        raise ValueError("unsupported arrival ledger schema or policy")
    if header.get("decision_phase") != "PRE_OBSERVATION":
        raise ValueError("arrival decision must precede current-round observation")
    if header.get("observation_phase") != "POST_DECISION":
        raise ValueError("observation phase must follow arrival decision")
    if _digest(header) != ledger.get("header_hash"):
        raise ValueError("ledger header hash mismatch")

    ids = _validate_ids(header.get("hypothesis_ids", []), "hypothesis_ids")
    params = header.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("policy parameters missing")
    initial = params.get("initial_active_count")
    min_obs = params.get("min_observations")
    threshold_hex = params.get("mean_threshold_hex")
    if isinstance(initial, bool) or not isinstance(initial, int) or not 1 <= initial <= len(ids):
        raise ValueError("invalid initial_active_count")
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 1:
        raise ValueError("invalid min_observations")
    threshold = _decode_score(threshold_hex)

    histories: dict[str, list[float]] = {hypothesis_id: [] for hypothesis_id in ids}
    active = ids[:initial]
    previous_hash = ledger["header_hash"]
    active_by_round: list[tuple[str, ...]] = []

    for round_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("event must be an object")
        event_core = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("round") != round_index:
            raise ValueError("round sequence is not contiguous")
        if event.get("decision_phase") != "PRE_OBSERVATION":
            raise ValueError("same-round lookahead: decision phase is not pre-observation")
        if event.get("observation_phase") != "POST_DECISION":
            raise ValueError("observation phase must follow decision")
        if event.get("previous_event_hash") != previous_hash:
            raise ValueError("event hash chain predecessor mismatch")
        if event.get("active_before") != active:
            raise ValueError("active_before does not match prior verified state")

        expected_pre_digest = _digest(_history_payload(ids, histories))
        if event.get("pre_history_digest") != expected_pre_digest:
            raise ValueError("pre-round evidence digest mismatch")
        expected_input_hash = _digest(
            {
                "policy_id": POLICY_ID,
                "round": round_index,
                "parameters": params,
                "active_before": active,
                "pre_history_digest": expected_pre_digest,
            }
        )
        if event.get("decision_input_hash") != expected_input_hash:
            raise ValueError("arrival decision input is not bound to prior evidence")

        expected_activated = _expected_activation(
            ids,
            active,
            histories,
            min_observations=min_obs,
            mean_threshold=threshold,
        )
        if event.get("activated") != expected_activated:
            raise ValueError("arrival policy replay mismatch: current or future evidence may have been used")
        expected_after = active + expected_activated
        if event.get("active_after") != expected_after:
            raise ValueError("active_after does not match replayed predictable decision")
        expected_output_hash = _digest(
            {
                "activated": expected_activated,
                "active_after": expected_after,
            }
        )
        if event.get("decision_output_hash") != expected_output_hash:
            raise ValueError("arrival decision output digest mismatch")

        observations = event.get("observed_scores")
        if not isinstance(observations, list):
            raise ValueError("observed_scores must be a list")
        expected_ids = expected_after
        observed_ids: list[str] = []
        decoded: list[tuple[str, float]] = []
        for item in observations:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("each observation must be [hypothesis_id, score_hex]")
            hypothesis_id, score_hex = item
            if not isinstance(hypothesis_id, str) or hypothesis_id not in ids:
                raise ValueError("observation references unknown hypothesis ID")
            observed_ids.append(hypothesis_id)
            decoded.append((hypothesis_id, _decode_score(score_hex)))
        if observed_ids != expected_ids:
            raise ValueError("observations must exactly match post-decision active IDs")

        if _digest(event_core) != event.get("event_hash"):
            raise ValueError("event hash mismatch")
        for hypothesis_id, score in decoded:
            histories[hypothesis_id].append(score)
        active = expected_after
        active_by_round.append(tuple(active))
        previous_hash = event["event_hash"]

    if ledger.get("final_event_hash") != previous_hash:
        raise ValueError("final event hash mismatch")
    return {
        "valid": True,
        "policy_id": POLICY_ID,
        "rounds": len(events),
        "universe_size": len(ids),
        "active_by_round": active_by_round,
        "final_active": tuple(active),
        "final_event_hash": previous_hash,
        "scope": (
            "deterministic event-order replay for the declared policy; "
            "not proof of source truth or real-world filtration completeness"
        ),
    }


if __name__ == "__main__":
    raise SystemExit("Run test_predictable_arrival_ledger.py.")
