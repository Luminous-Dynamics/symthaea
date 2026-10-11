#!/usr/bin/env python3
"""PIE-009M independent proof-carrying recovery proposal verifier.

Synthetic deterministic reference only. The verifier proves structural
admissibility for an exact snapshot; it does not execute or authorize hardware.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Tuple
import hashlib
import json

VERIFIER_VERSION = "pie-009m-v1"
SCOPE = "admissibility-only"


@dataclass(frozen=True, order=True)
class HiddenWorld:
    generator_healthy: bool
    tie_closed: bool


@dataclass(frozen=True)
class Snapshot:
    topology_version: int
    operational: Tuple[str, ...]
    worlds: Tuple[HiddenWorld, ...]


@dataclass(frozen=True)
class Proposal:
    action: str
    expected_topology_version: int
    snapshot_digest: str
    proposal_digest: str


@dataclass(frozen=True)
class Receipt:
    accepted: bool
    reason: str
    snapshot_digest: str
    proposal_digest: str
    verifier_version: str
    scope: str


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def validate_snapshot(snapshot: Snapshot) -> None:
    if snapshot.topology_version < 0:
        raise ValueError("negative topology version")
    if not snapshot.worlds:
        raise ValueError("snapshot must contain at least one admissible world")
    if tuple(sorted(set(snapshot.operational))) != snapshot.operational:
        raise ValueError("operational set must be sorted and unique")
    if tuple(sorted(set(snapshot.worlds))) != snapshot.worlds:
        raise ValueError("world set must be sorted and unique")


def snapshot_digest(snapshot: Snapshot) -> str:
    validate_snapshot(snapshot)
    payload = {
        "topology_version": snapshot.topology_version,
        "operational": list(snapshot.operational),
        "worlds": [asdict(w) for w in snapshot.worlds],
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _proposal_body(action: str, expected_topology_version: int, snap_digest: str):
    return {
        "action": action,
        "expected_topology_version": expected_topology_version,
        "snapshot_digest": snap_digest,
    }


def make_proposal(snapshot: Snapshot, action: str) -> Proposal:
    sd = snapshot_digest(snapshot)
    body = _proposal_body(action, snapshot.topology_version, sd)
    pd = hashlib.sha256(_canonical(body)).hexdigest()
    return Proposal(action, snapshot.topology_version, sd, pd)


def _proposal_digest(proposal: Proposal) -> str:
    return hashlib.sha256(_canonical(_proposal_body(
        proposal.action,
        proposal.expected_topology_version,
        proposal.snapshot_digest,
    ))).hexdigest()


def _safe_in_world(action: str, world: HiddenWorld, operational: set[str]) -> bool:
    if action == "start_generator":
        return "controls" in operational and world.generator_healthy
    if action == "close_tie":
        return "controls" in operational and not world.tie_closed
    if action == "start_water":
        return "generator" in operational and world.tie_closed
    if action in ("probe_generator", "probe_tie", "wait"):
        return True
    return False


def verify(snapshot: Snapshot, proposal: Proposal) -> Receipt:
    try:
        sd = snapshot_digest(snapshot)
    except ValueError as exc:
        return Receipt(False, f"INVALID_SNAPSHOT:{exc}", "", proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    if proposal.snapshot_digest != sd:
        return Receipt(False, "SNAPSHOT_DIGEST_MISMATCH", sd, proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    expected_pd = _proposal_digest(proposal)
    if proposal.proposal_digest != expected_pd:
        return Receipt(False, "PROPOSAL_DIGEST_MISMATCH", sd, proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    if proposal.expected_topology_version != snapshot.topology_version:
        return Receipt(False, "STALE_TOPOLOGY_VERSION", sd, proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    if proposal.action not in {
        "start_generator", "close_tie", "start_water",
        "probe_generator", "probe_tie", "wait",
    }:
        return Receipt(False, "UNKNOWN_ACTION", sd, proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    operational = set(snapshot.operational)
    if not all(_safe_in_world(proposal.action, world, operational) for world in snapshot.worlds):
        return Receipt(False, "NOT_UNIVERSALLY_SAFE", sd, proposal.proposal_digest,
                       VERIFIER_VERSION, SCOPE)

    return Receipt(True, "ADMISSIBLE", sd, proposal.proposal_digest,
                   VERIFIER_VERSION, SCOPE)


def self_test():
    healthy = HiddenWorld(True, False)
    failed = HiddenWorld(False, False)
    uncertain = Snapshot(7, ("controls",), tuple(sorted((healthy, failed))))

    # 1. Properly formed but unsafe proposal is rejected by recomputation.
    unsafe = make_proposal(uncertain, "start_generator")
    r = verify(uncertain, unsafe)
    assert not r.accepted and r.reason == "NOT_UNIVERSALLY_SAFE"

    # 2. Diagnostic action over same uncertainty is accepted.
    probe = make_proposal(uncertain, "probe_generator")
    rp = verify(uncertain, probe)
    assert rp.accepted and rp.reason == "ADMISSIBLE"
    assert rp.scope == "admissibility-only"

    # 3. Tampering with action while retaining digest is detected.
    tampered = replace(probe, action="start_generator")
    assert verify(uncertain, tampered).reason == "PROPOSAL_DIGEST_MISMATCH"

    # 4. Proposal cannot be replayed against a changed snapshot.
    changed = Snapshot(7, ("controls", "generator"), tuple(sorted((healthy, failed))))
    assert verify(changed, probe).reason == "SNAPSHOT_DIGEST_MISMATCH"

    # 5. Correct digest with stale topology expectation is rejected.
    sd = snapshot_digest(uncertain)
    stale_body = _proposal_body("probe_generator", 6, sd)
    stale = Proposal(
        "probe_generator", 6, sd,
        hashlib.sha256(_canonical(stale_body)).hexdigest(),
    )
    assert verify(uncertain, stale).reason == "STALE_TOPOLOGY_VERSION"

    # 6. Exact known-healthy snapshot permits generator start.
    known = Snapshot(7, ("controls",), (healthy,))
    start = make_proposal(known, "start_generator")
    assert verify(known, start).accepted

    # 7. Same exact inputs yield byte-for-byte-equivalent receipts.
    assert verify(known, start) == verify(known, start)

    # 8. Malformed snapshot fails closed.
    malformed = Snapshot(7, ("controls", "controls"), (healthy,))
    assert verify(malformed, start).reason.startswith("INVALID_SNAPSHOT:")

    # 9. Unknown actions fail even with internally consistent digest.
    weird = make_proposal(known, "teleport_power")
    assert verify(known, weird).reason == "UNKNOWN_ACTION"

    print("ok")


if __name__ == "__main__":
    self_test()
