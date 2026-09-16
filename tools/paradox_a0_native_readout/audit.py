#!/usr/bin/env python3
"""Static audit for the PARADOX A0 native-readout development plane.

Development-only. This script does not execute Symthaea cognition or score fixtures.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXPECTED_SCHEMA = "PARADOX-A0-NATIVE-REGISTRY-V2"
EXPECTED_REACHABILITY_SCHEMA = "PARADOX-A0-RUNTIME-REACHABILITY-V1"
EXPECTED_PRODUCTION = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
EXPECTED_G2B = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
RESPONSES = {
    "Commit",
    "CommitConditionally",
    "AbstainPreservePlurality",
    "ReflexiveUpdate",
    "RequestRepresentationRevision",
}
TIERS = {"A0-D", "A0-L", "A0-R"}
ADMISSION_TIERS = {"A0-D", "A0-R"}
REACHABILITY = {
    "live_service_public",
    "live_subsystem_public",
    "partial_live_projection",
    "feature_gated_live_projection",
    "internal_unproven",
    "library_only",
}
DIRECT_LIVE = {"live_service_public"}
READOUT_REACHABLE = {
    "live_service_public",
    "live_subsystem_public",
    "partial_live_projection",
    "feature_gated_live_projection",
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> None:
    raise SystemExit(f"A0 development audit failed: {message}")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {path.name}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path.name} root must be an object")
    return value


def main() -> int:
    here = Path(__file__).parent
    data = load_json(here / "registry.json")
    reach = load_json(here / "reachability.json")

    if data.get("schema_version") != EXPECTED_SCHEMA:
        fail("unexpected registry schema version")
    if reach.get("schema_version") != EXPECTED_REACHABILITY_SCHEMA:
        fail("unexpected reachability schema version")
    if data.get("production_subject_sha") != EXPECTED_PRODUCTION:
        fail("registry production subject drift")
    if reach.get("production_subject_sha") != EXPECTED_PRODUCTION:
        fail("reachability production subject drift")
    if data.get("g2b_subject_sha") != EXPECTED_G2B:
        fail("G2b subject drift")
    if data.get("confirmatory_execution_allowed") is not False:
        fail("development registry must not authorize confirmatory execution")
    if set(data.get("response_classes", [])) != RESPONSES:
        fail("response vocabulary drift")
    if set(reach.get("status_vocabulary", [])) != REACHABILITY:
        fail("runtime reachability vocabulary drift")

    atoms = data.get("capability_atoms")
    if not isinstance(atoms, list) or not atoms or len(atoms) != len(set(atoms)):
        fail("capability_atoms must be a non-empty unique list")
    atom_set = set(atoms)

    requirements = data.get("response_requirements")
    if set(requirements or {}) != RESPONSES:
        fail("response_requirements must cover every response class exactly once")
    for response, required in requirements.items():
        if not isinstance(required, list) or not required:
            fail(f"{response}: must require at least one capability atom")
        unknown = set(required) - atom_set
        if unknown:
            fail(f"{response}: unknown required atoms {sorted(unknown)}")
        if len(required) != len(set(required)):
            fail(f"{response}: duplicate required capability atom")

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        fail("candidate registry must be non-empty")
    reach_candidates = reach.get("candidates")
    if not isinstance(reach_candidates, dict):
        fail("reachability candidates must be an object")

    ids: set[str] = set()
    candidate_by_id: dict[str, dict] = {}
    for candidate in candidates:
        cid = candidate.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            fail("candidate IDs must be unique non-empty strings")
        ids.add(cid)
        candidate_by_id[cid] = candidate
        tier = candidate.get("tier")
        if tier not in TIERS:
            fail(f"{cid}: invalid evidence tier")
        sha = candidate.get("source_blob_sha", "")
        if not SHA40.fullmatch(sha):
            fail(f"{cid}: invalid source blob SHA")

        supported = candidate.get("supported_atoms")
        if not isinstance(supported, list):
            fail(f"{cid}: supported_atoms must be a list")
        unknown_atoms = set(supported) - atom_set
        if unknown_atoms:
            fail(f"{cid}: unknown supported atoms {sorted(unknown_atoms)}")
        if len(supported) != len(set(supported)):
            fail(f"{cid}: duplicate supported atom")

        admitted = candidate.get("admitted_response_classes")
        if not isinstance(admitted, list):
            fail(f"{cid}: admitted_response_classes must be a list")
        unknown = set(admitted) - RESPONSES
        if unknown:
            fail(f"{cid}: unknown admitted response classes {sorted(unknown)}")
        for response in admitted:
            missing = set(requirements[response]) - set(supported)
            if missing:
                fail(f"{cid}: admits {response} without required atoms {sorted(missing)}")

        forbidden = candidate.get("forbidden_aliases", {})
        if not isinstance(forbidden, dict):
            fail(f"{cid}: forbidden_aliases must be an object")
        for native_variant, response_list in forbidden.items():
            if not isinstance(native_variant, str) or not native_variant:
                fail(f"{cid}: invalid forbidden alias key")
            if set(response_list) - RESPONSES:
                fail(f"{cid}: forbidden alias contains unknown response class")
            if set(response_list) & set(admitted):
                fail(f"{cid}: response cannot be both admitted and forbidden")

    if set(reach_candidates) != ids:
        missing = ids - set(reach_candidates)
        extra = set(reach_candidates) - ids
        fail(f"reachability candidate mismatch missing={sorted(missing)} extra={sorted(extra)}")

    live_atoms_by_id: dict[str, set[str]] = {}
    for cid, entry in reach_candidates.items():
        status = entry.get("status")
        if status not in REACHABILITY:
            fail(f"{cid}: invalid runtime reachability status")
        receiptable = entry.get("receiptable_without_production_mutation")
        if not isinstance(receiptable, bool):
            fail(f"{cid}: receiptability must be boolean")

        source = candidate_by_id[cid]
        supported = set(source["supported_atoms"])
        live_exposed = entry.get("live_exposed_atoms")
        if not isinstance(live_exposed, list):
            fail(f"{cid}: live_exposed_atoms must be a list")
        if len(live_exposed) != len(set(live_exposed)):
            fail(f"{cid}: duplicate live-exposed atom")
        live_set = set(live_exposed)
        if live_set - supported:
            fail(f"{cid}: live projection claims atoms absent from native supported_atoms")
        if status in {"library_only", "internal_unproven"} and live_set:
            fail(f"{cid}: unproven/library-only source cannot expose live atoms")
        if live_set and not receiptable:
            fail(f"{cid}: live-exposed atoms require a receiptable runtime surface")
        live_atoms_by_id[cid] = live_set

        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            fail(f"{cid}: reachability evidence must be non-empty")
        source_bound = False
        for proof in evidence:
            if not isinstance(proof, dict):
                fail(f"{cid}: malformed reachability evidence")
            blob = proof.get("blob_sha", "")
            path = proof.get("path", "")
            if not SHA40.fullmatch(blob) or not isinstance(path, str) or not path:
                fail(f"{cid}: invalid reachability path/blob binding")
            if path == source["source_path"] and blob == source["source_blob_sha"]:
                source_bound = True
        if not source_bound:
            fail(f"{cid}: reachability evidence does not bind registry source blob")
        if status in {"library_only", "internal_unproven"} and receiptable:
            fail(f"{cid}: unproven/library-only candidate cannot be marked receiptable")

    direct_atoms: set[str] = set()
    readout_atoms: set[str] = set()
    for cid, candidate in candidate_by_id.items():
        status = reach_candidates[cid]["status"]
        exposed = live_atoms_by_id[cid]
        if candidate["tier"] == "A0-D" and status in DIRECT_LIVE:
            direct_atoms.update(exposed)
        if status in READOUT_REACHABLE:
            readout_atoms.update(exposed)

    admissions = data.get("class_admission")
    if set(admissions or {}) != RESPONSES:
        fail("class_admission must cover every response class exactly once")
    for response, tiers in admissions.items():
        if set(tiers) != ADMISSION_TIERS:
            fail(f"{response}: admission must declare A0-D and A0-R")
        if any(value not in {"not_admitted_v1", "admitted_v1"} for value in tiers.values()):
            fail(f"{response}: invalid admission state")
        for tier, state in tiers.items():
            if state != "admitted_v1":
                continue
            eligible_atoms = direct_atoms if tier == "A0-D" else readout_atoms
            runtime_missing = set(requirements[response]) - eligible_atoms
            if runtime_missing:
                fail(
                    f"{response}/{tier}: admitted without live-exposed required atoms "
                    f"{sorted(runtime_missing)}"
                )

    # V2 remains the pre-wiring inventory. No complete response mapping is admitted yet.
    if any(
        state == "admitted_v1"
        for tiers in admissions.values()
        for state in tiers.values()
    ):
        fail("v2 inventory unexpectedly contains an admitted full response mapping")
    if any(candidate["admitted_response_classes"] for candidate in candidates):
        fail("v2 inventory unexpectedly claims a complete PARADOX response")

    print(
        "PARADOX-A0 development audit PASS "
        f"candidates={len(candidates)} atoms={len(atom_set)} "
        f"direct_live_atoms={len(direct_atoms)} readout_atoms={len(readout_atoms)} "
        f"production={EXPECTED_PRODUCTION} g2b={EXPECTED_G2B}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
