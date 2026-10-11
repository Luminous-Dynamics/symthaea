#!/usr/bin/env python3
"""PIE-009W independent metrology quarantine/re-entry oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from math import isfinite

EPSILON = 1e-12


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    generation: int
    status: str
    quarantine_step: int | None = None


@dataclass(frozen=True)
class RemediationReceipt:
    reference_id: str
    prior_generation: int
    new_generation: int
    step: int
    action: str


@dataclass(frozen=True)
class ComparisonReceipt:
    left_id: str
    left_generation: int
    right_id: str
    right_generation: int
    observed_delta: float
    allowed_delta: float
    step: int
    qualified: bool
    failure_groups: tuple[str, ...]


def pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def relation(comparison: ComparisonReceipt) -> str:
    if not comparison.qualified:
        return "UNKNOWN"
    return (
        "AGREE"
        if abs(comparison.observed_delta) <= comparison.allowed_delta + EPSILON
        else "DISAGREE"
    )


def validate_reference(reference: ReferenceRecord) -> None:
    if (
        not reference.reference_id
        or reference.generation < 0
        or reference.status not in {"ACTIVE", "QUARANTINED", "RETIRED"}
    ):
        raise ValueError("invalid reference")
    if reference.status == "QUARANTINED" and reference.quarantine_step is None:
        raise ValueError("missing quarantine step")
    if reference.quarantine_step is not None and reference.quarantine_step < 0:
        raise ValueError("invalid quarantine step")


def validate_remediation(
    reference: ReferenceRecord, remediation: RemediationReceipt
) -> None:
    if remediation.reference_id != reference.reference_id:
        raise ValueError("remediation targets wrong reference")
    if remediation.prior_generation != reference.generation:
        raise ValueError("remediation binds wrong prior generation")
    if remediation.new_generation <= remediation.prior_generation:
        raise ValueError("remediation must advance generation")
    if remediation.step < 0 or not remediation.action:
        raise ValueError("invalid remediation receipt")
    if reference.status != "QUARANTINED":
        raise ValueError("reference is not quarantined")
    assert reference.quarantine_step is not None
    if remediation.step <= reference.quarantine_step:
        raise ValueError("remediation must occur after quarantine")


def expected_generation(
    reference: ReferenceRecord,
    remediation: RemediationReceipt,
    peer_map: dict[str, ReferenceRecord],
    reference_id: str,
) -> int:
    if reference_id == reference.reference_id:
        return remediation.new_generation
    return peer_map[reference_id].generation


def receipt_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate_reentry(
    reference: ReferenceRecord,
    remediation: RemediationReceipt,
    peers: list[ReferenceRecord],
    comparisons: list[ComparisonReceipt],
) -> dict[str, object]:
    validate_reference(reference)
    validate_remediation(reference, remediation)

    peer_map: dict[str, ReferenceRecord] = {}
    for peer in peers:
        validate_reference(peer)
        if peer.reference_id == reference.reference_id or peer.reference_id in peer_map:
            raise ValueError("duplicate peer reference")
        peer_map[peer.reference_id] = peer

    if len(peer_map) < 2:
        return {
            "status": "REENTRY_NOT_ESTABLISHED",
            "reason": "INSUFFICIENT_PEERS",
            "new_generation": remediation.new_generation,
            "scope": "eligibility-only",
        }

    known_ids = {reference.reference_id, *peer_map.keys()}
    comparison_map: dict[tuple[str, str], ComparisonReceipt] = {}

    for comparison in comparisons:
        if (
            comparison.left_id == comparison.right_id
            or comparison.left_id not in known_ids
            or comparison.right_id not in known_ids
        ):
            raise ValueError("invalid comparison endpoints")
        if (
            comparison.step < 0
            or not isfinite(comparison.observed_delta)
            or not isfinite(comparison.allowed_delta)
            or comparison.allowed_delta < 0.0
        ):
            raise ValueError("invalid comparison")
        if (
            not comparison.failure_groups
            or len(set(comparison.failure_groups)) != len(comparison.failure_groups)
            or any(not group for group in comparison.failure_groups)
        ):
            raise ValueError("invalid failure groups")

        key = pair(comparison.left_id, comparison.right_id)
        if key in comparison_map:
            raise ValueError("duplicate comparison pair")

        # Only post-remediation, exact-generation evidence can contribute.
        if comparison.step <= remediation.step:
            continue
        if comparison.left_generation != expected_generation(
            reference, remediation, peer_map, comparison.left_id
        ):
            continue
        if comparison.right_generation != expected_generation(
            reference, remediation, peer_map, comparison.right_id
        ):
            continue
        comparison_map[key] = comparison

    active_peers = sorted(
        peer.reference_id for peer in peers if peer.status == "ACTIVE"
    )

    witnesses: list[tuple[str, str]] = []
    for first, second in combinations(active_peers, 2):
        candidate_first = comparison_map.get(pair(reference.reference_id, first))
        candidate_second = comparison_map.get(pair(reference.reference_id, second))
        peers_pair = comparison_map.get(pair(first, second))

        if not candidate_first or not candidate_second or not peers_pair:
            continue
        if (
            relation(candidate_first) != "AGREE"
            or relation(candidate_second) != "AGREE"
            or relation(peers_pair) != "AGREE"
        ):
            continue

        groups = [
            set(candidate_first.failure_groups),
            set(candidate_second.failure_groups),
            set(peers_pair.failure_groups),
        ]
        if any(
            groups[left] & groups[right]
            for left, right in combinations(range(3), 2)
        ):
            continue

        witnesses.append((first, second))

    base_payload = {
        "reference": asdict(reference),
        "remediation": asdict(remediation),
        "active_peer_ids": active_peers,
        "post_remediation_comparisons": [
            {**asdict(comparison), "failure_groups": list(comparison.failure_groups)}
            for comparison in sorted(
                comparison_map.values(),
                key=lambda item: pair(item.left_id, item.right_id),
            )
        ],
    }

    if not witnesses:
        result: dict[str, object] = {
            "status": "REENTRY_NOT_ESTABLISHED",
            "reason": "NO_INDEPENDENT_POST_REMEDIATION_WITNESS",
            "new_generation": remediation.new_generation,
            "scope": "eligibility-only",
        }
    else:
        result = {
            "status": "REENTRY_ELIGIBLE",
            "new_generation": remediation.new_generation,
            "witness_peers": list(witnesses[0]),
            "scope": "eligibility-only",
        }

    result["receipt_digest"] = receipt_digest({**base_payload, **result})
    return result


def expect_value_error(callable_) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def fixture() -> tuple[
    ReferenceRecord,
    RemediationReceipt,
    list[ReferenceRecord],
    list[ComparisonReceipt],
]:
    candidate = ReferenceRecord("C", 0, "QUARANTINED", 10)
    remediation = RemediationReceipt("C", 0, 1, 11, "recalibrate")
    peers = [
        ReferenceRecord("A", 4, "ACTIVE"),
        ReferenceRecord("B", 7, "ACTIVE"),
    ]
    comparisons = [
        ComparisonReceipt("C", 1, "A", 4, 0.10, 0.20, 12, True, ("ca",)),
        ComparisonReceipt("C", 1, "B", 7, 0.10, 0.20, 12, True, ("cb",)),
        ComparisonReceipt("A", 4, "B", 7, 0.05, 0.20, 12, True, ("ab",)),
    ]
    return candidate, remediation, peers, comparisons


def self_test() -> None:
    candidate, remediation, peers, good = fixture()

    result = evaluate_reentry(candidate, remediation, peers, good)
    assert result["status"] == "REENTRY_ELIGIBLE"
    assert result["scope"] == "eligibility-only"

    pre_remediation = [
        ComparisonReceipt("C", 1, "A", 4, 0.10, 0.20, 11, True, ("ca",)),
        ComparisonReceipt("C", 1, "B", 7, 0.10, 0.20, 11, True, ("cb",)),
        ComparisonReceipt("A", 4, "B", 7, 0.05, 0.20, 11, True, ("ab",)),
    ]
    assert evaluate_reentry(candidate, remediation, peers, pre_remediation)["status"] == "REENTRY_NOT_ESTABLISHED"

    assert evaluate_reentry(candidate, remediation, peers, good[:1])["status"] == "REENTRY_NOT_ESTABLISHED"

    shared_common_mode = [
        ComparisonReceipt("C", 1, "A", 4, 0.10, 0.20, 12, True, ("fixture",)),
        ComparisonReceipt("C", 1, "B", 7, 0.10, 0.20, 12, True, ("fixture",)),
        ComparisonReceipt("A", 4, "B", 7, 0.05, 0.20, 12, True, ("ab",)),
    ]
    assert evaluate_reentry(candidate, remediation, peers, shared_common_mode)["status"] == "REENTRY_NOT_ESTABLISHED"

    disagreement = list(good)
    disagreement[1] = ComparisonReceipt("C", 1, "B", 7, 0.50, 0.20, 12, True, ("cb",))
    assert evaluate_reentry(candidate, remediation, peers, disagreement)["status"] == "REENTRY_NOT_ESTABLISHED"

    stale_generation = list(good)
    stale_generation[0] = ComparisonReceipt("C", 0, "A", 4, 0.10, 0.20, 12, True, ("ca",))
    assert evaluate_reentry(candidate, remediation, peers, stale_generation)["status"] == "REENTRY_NOT_ESTABLISHED"

    inactive_peers = [
        ReferenceRecord("A", 4, "ACTIVE"),
        ReferenceRecord("B", 7, "QUARANTINED", 9),
    ]
    assert evaluate_reentry(candidate, remediation, inactive_peers, good)["status"] == "REENTRY_NOT_ESTABLISHED"

    expect_value_error(
        lambda: evaluate_reentry(
            candidate,
            RemediationReceipt("C", 0, 0, 11, "noop"),
            peers,
            good,
        )
    )
    expect_value_error(
        lambda: evaluate_reentry(
            candidate,
            RemediationReceipt("C", 0, 1, 10, "recalibrate"),
            peers,
            good,
        )
    )

    repeated = evaluate_reentry(candidate, remediation, peers, good)
    assert result["receipt_digest"] == repeated["receipt_digest"]

    changed_remediation = RemediationReceipt(
        "C", 0, 1, 11, "repair-and-recalibrate"
    )
    changed = evaluate_reentry(candidate, changed_remediation, peers, good)
    assert result["receipt_digest"] != changed["receipt_digest"]

    print("ok")


if __name__ == "__main__":
    self_test()
