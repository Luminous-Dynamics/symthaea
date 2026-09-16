#!/usr/bin/env python3
"""Derive the PARADOX A0 representation-to-policy gap matrix.

Development-only. This script reads only the frozen A0 registry/reachability
manifests. It does not execute cognition, fixtures, or scoring.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DIRECT_LIVE = {"live_service_public"}
READOUT_REACHABLE = {
    "live_service_public",
    "live_subsystem_public",
    "partial_live_projection",
    "feature_gated_live_projection",
}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    here = Path(__file__).parent
    registry = json.loads((here / "registry.json").read_text(encoding="utf-8"))
    reachability = json.loads((here / "reachability.json").read_text(encoding="utf-8"))

    reach = reachability["candidates"]
    candidates = registry["candidates"]

    direct_live_atoms: set[str] = set()
    readout_atoms: set[str] = set()
    any_live_atoms: set[str] = set()
    repo_native_atoms: set[str] = set()

    for candidate in candidates:
        cid = candidate["id"]
        native_atoms = set(candidate["supported_atoms"])
        exposed_atoms = set(reach[cid]["live_exposed_atoms"])
        status = reach[cid]["status"]

        repo_native_atoms.update(native_atoms)
        if status in READOUT_REACHABLE:
            any_live_atoms.update(exposed_atoms)
            # A0-R may decode any already-exposed frozen production/subsystem state,
            # including direct signals; it does not inherit unexposed internal semantics.
            readout_atoms.update(exposed_atoms)
        if candidate["tier"] == "A0-D" and status in DIRECT_LIVE:
            direct_live_atoms.update(exposed_atoms)

    response_rows = {}
    for response, required_list in registry["response_requirements"].items():
        required = set(required_list)
        direct_missing = sorted(required - direct_live_atoms)
        readout_missing = sorted(required - readout_atoms)
        live_any_missing = sorted(required - any_live_atoms)
        repo_native_missing = sorted(required - repo_native_atoms)

        if not direct_missing:
            gap_class = "direct_live_atoms_complete"
        elif not readout_missing:
            gap_class = "readout_atoms_complete_direct_policy_gap"
        elif not live_any_missing:
            gap_class = "live_atoms_exist_but_policy_gap"
        elif not repo_native_missing:
            gap_class = "native_atoms_exist_but_runtime_integration_gap"
        else:
            gap_class = "audited_inventory_gap"

        response_rows[response] = {
            "required_atoms": sorted(required),
            "direct_live_missing": direct_missing,
            "a0_readout_missing": readout_missing,
            "any_live_missing": live_any_missing,
            "repo_native_missing": repo_native_missing,
            "gap_class": gap_class,
        }

    report = {
        "schema_version": "PARADOX-A0-GAP-REPORT-V1",
        "authority": "DevelopmentOnly / derived from frozen source inventory",
        "production_subject_sha": registry["production_subject_sha"],
        "g2b_subject_sha": registry["g2b_subject_sha"],
        "direct_live_atoms": sorted(direct_live_atoms),
        "a0_readout_atoms": sorted(readout_atoms),
        "any_live_atoms": sorted(any_live_atoms),
        "repo_native_atoms": sorted(repo_native_atoms),
        "responses": response_rows,
        "interpretation_rule": (
            "Runtime/readout layers may credit only reachability.live_exposed_atoms, never the richer internal "
            "supported_atoms of a partially projected subsystem. A missing atom localizes a gap in the audited "
            "inventory, not proof of cognitive impossibility. Future development-only evidence may promote an "
            "existing latent atom without rewriting prior receipts. No gap class authorizes confirmatory behavior "
            "or A1 implementation."
        ),
    }
    encoded = canonical_json(report)
    envelope = {
        "report": report,
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    print(canonical_json(envelope).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
