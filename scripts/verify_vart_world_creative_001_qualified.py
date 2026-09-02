#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from verify_vart_world_creative_001 import (
    PAIR_POLICIES,
    Reject,
    file_from_index,
    read_json,
    require,
    require_sha256,
    sha256_file,
    verify_bundle,
)

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
GENERALIZATION_SECRET_KEYS = {
    "fixture_label",
    "fixture_kind",
    "target_defect",
    "known_solution",
    "expected_optimum",
    "trap_kind",
}
CROSS_POLICY_OUTCOME_KEYS = {
    "other_policy_outcome",
    "prior_policy_outcome",
    "full_outcome",
    "random_valid_outcome",
    "heuristic_outcome",
    "baseline_outcome",
}


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def preflight_freeze(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    require_sha256(expected_freeze_sha256, "expected_freeze_sha256")
    freeze_path = root / "confirmatory_freeze.json"
    actual_freeze_sha = sha256_file(freeze_path)
    require(
        actual_freeze_sha == expected_freeze_sha256,
        "POST_FREEZE_CONTRACT_MUTATION",
        f"confirmatory_freeze.json {actual_freeze_sha} != externally anchored {expected_freeze_sha256}",
    )
    freeze = read_json(freeze_path)
    require(
        isinstance(freeze, dict)
        and freeze.get("experiment_id", EXPERIMENT_ID) == EXPERIMENT_ID,
        "PREREGISTRATION_INVALID",
        "freeze experiment identity",
    )
    require(
        freeze.get("frozen") is True,
        "PREREGISTRATION_INVALID",
        "confirmatory freeze is not marked frozen",
    )
    analysis_sha = sha256_file(root / "analysis_contract.json")
    metric_sha = sha256_file(root / "metric_definitions.json")
    require(
        freeze.get("analysis_contract_sha256") == analysis_sha,
        "ANALYSIS_CONTRACT_MISMATCH",
        "analysis contract differs from frozen digest",
    )
    require(
        freeze.get("metric_definition_set_sha256") == metric_sha,
        "ANALYSIS_CONTRACT_MISMATCH",
        "metric definitions differ from frozen digest",
    )
    return freeze


def expected_trial_ids(inventory: Any) -> list[str]:
    require(isinstance(inventory, dict), "PREREGISTRATION_INVALID", "trial inventory root")
    listed = inventory.get("trial_ids")
    rows = inventory.get("trials")
    derived: list[str] | None = None
    if isinstance(rows, list) and rows:
        derived = []
        for i, row in enumerate(rows):
            require(isinstance(row, dict) and isinstance(row.get("trial_id"), str) and row["trial_id"],
                    "PREREGISTRATION_INVALID", f"trial inventory row {i}")
            derived.append(row["trial_id"])
        require(len(derived) == len(set(derived)), "PREREGISTRATION_INVALID", "duplicate trial IDs in trials rows")

    if listed is not None:
        require(isinstance(listed, list) and listed and all(isinstance(x, str) and x for x in listed),
                "PREREGISTRATION_INVALID", "trial_ids")
        require(len(listed) == len(set(listed)), "PREREGISTRATION_INVALID", "duplicate trial_ids")
        if derived is not None:
            require(set(listed) == set(derived), "PREREGISTRATION_INVALID",
                    "trial_ids and trials rows disagree on preregistered membership")
        return list(listed)

    require(derived is not None and derived, "PREREGISTRATION_INVALID",
            "trial inventory requires trial_ids or trials rows")
    return derived


def preflight_trial_inventory(root: Path) -> tuple[list[str], list[Path]]:
    inventory = read_json(root / "trial_inventory.json")
    expected_ids = expected_trial_ids(inventory)
    manifests = sorted((root / "trials").glob("*/manifest.json"))
    actual_ids: list[str] = []
    for path in manifests:
        m = read_json(path)
        if isinstance(m, dict) and isinstance(m.get("trial_id"), str):
            actual_ids.append(m["trial_id"])
    require(len(actual_ids) == len(set(actual_ids)), "PREREGISTRATION_INVALID", "duplicate emitted trial manifests")
    present = set(actual_ids)
    expected = set(expected_ids)
    extras = sorted(present - expected)
    require(not extras, "PREREGISTRATION_INVALID", f"unexpected confirmatory trials: {extras}")
    if len(actual_ids) < len(expected_ids):
        prefix = set(expected_ids[: len(actual_ids)])
        if present == prefix:
            raise Reject(
                "UNAUTHORIZED_EARLY_STOP",
                f"only frozen prefix {len(actual_ids)}/{len(expected_ids)} trials is present",
            )
        missing = sorted(expected - present)
        raise Reject("PREREGISTERED_TRIAL_MISSING", ",".join(missing))
    require(present == expected, "PREREGISTRATION_INVALID", "complete trial membership mismatch")
    return expected_ids, manifests


def preflight_revisit(manifests: list[Path]) -> None:
    for path in manifests:
        manifest = read_json(path)
        if not isinstance(manifest, dict) or manifest.get("trial_state") != "complete":
            continue
        trial_dir = path.parent
        idx = read_json(trial_dir / "evidence_index.json")
        files = idx.get("files", {}) if isinstance(idx, dict) else {}
        revisit_rel = files.get("revisit_observation") if isinstance(files, dict) else None
        outcome_rel = files.get("revision_outcome") if isinstance(files, dict) else None
        if outcome_rel and (
            not isinstance(revisit_rel, str) or not (trial_dir / revisit_rel).is_file()
        ):
            raise Reject("OUTCOME_WITHOUT_REVISIT", manifest.get("trial_id", str(path)))


def verify_trial_surface(trial_dir: Path, manifest: dict[str, Any]) -> str:
    decision_sha = require_sha256(
        manifest.get("decision_input_sha256"), "decision_input_sha256"
    )
    generated = manifest.get("generated_candidate_count")
    admitted_count = manifest.get("admissible_candidate_count")
    require(
        isinstance(generated, int)
        and generated > 0
        and isinstance(admitted_count, int)
        and 0 < admitted_count <= generated,
        "MANIFEST_SCHEMA_INVALID",
        f"{manifest['trial_id']}: candidate counts",
    )
    idx = read_json(trial_dir / "evidence_index.json")
    decision_path = file_from_index(trial_dir, idx, "decision_input")
    require(
        sha256_file(decision_path) == decision_sha,
        "EVIDENCE_DIGEST_MISMATCH",
        f"{manifest['trial_id']}: decision input",
    )
    decision_input = read_json(decision_path)
    require(
        isinstance(decision_input, dict)
        and decision_input.get("experiment_id") == EXPERIMENT_ID
        and decision_input.get("paired_block_id") == manifest["paired_block_id"]
        and decision_input.get("seed") == manifest["seed"]
        and decision_input.get("revision_index") == manifest["revision_index"],
        "PAIRED_BLOCK_IDENTITY_MISMATCH",
        f"{manifest['trial_id']}: decision input identity",
    )
    decision_keys = set(walk_keys(decision_input))
    leaked_outcomes = sorted(CROSS_POLICY_OUTCOME_KEYS.intersection(decision_keys))
    require(
        not leaked_outcomes,
        "POLICY_ORDER_INFORMATION_LEAK",
        f"{manifest['trial_id']}: {leaked_outcomes}",
    )
    if manifest.get("campaign") == "confirmatory_generalization":
        leaked_fixture = sorted(GENERALIZATION_SECRET_KEYS.intersection(decision_keys))
        require(
            not leaked_fixture,
            "GENERALIZATION_FIXTURE_LEAK",
            f"{manifest['trial_id']}: {leaked_fixture}",
        )

    candidate_path = file_from_index(trial_dir, idx, "candidate_set")
    candidate_set = read_json(candidate_path)
    candidates = candidate_set.get("candidates") if isinstance(candidate_set, dict) else None
    require(
        isinstance(candidates, list) and len(candidates) == generated,
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{manifest['trial_id']}: generated/rejected candidates truncated",
    )
    require(
        all(
            isinstance(candidate, dict)
            and isinstance(candidate.get("physically_admitted"), bool)
            for candidate in candidates
        ),
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{manifest['trial_id']}: candidate admission state missing",
    )
    require(
        sum(1 for c in candidates if c["physically_admitted"]) == admitted_count,
        "INCOMPLETE_EVIDENCE_CLOSURE",
        f"{manifest['trial_id']}: admitted candidate count mismatch",
    )

    if manifest.get("trial_state") == "complete":
        receipt_path = file_from_index(trial_dir, idx, "applied_receipt")
        receipt = read_json(receipt_path)
        require(
            isinstance(receipt, dict)
            and receipt.get("decision_input_sha256") == decision_sha
            and receipt.get("candidate_set_sha256") == manifest["candidate_set_sha256"],
            "PROSPECTIVE_BINDING_MISMATCH",
            f"{manifest['trial_id']}: receipt decision/candidate binding",
        )
        require(
            receipt.get("revision_hypothesis_sha256")
            == manifest["revision_hypothesis_sha256"],
            "POST_HOC_HYPOTHESIS_MUTATION",
            manifest["trial_id"],
        )

    violations = manifest.get("integrity_violations") or []
    if manifest.get("trial_state") == "invalid_integrity" or violations:
        raise Reject(
            "INVALID_EXCLUSION_RECLASSIFICATION",
            f"{manifest['trial_id']}: confirmatory claim admission requires zero integrity-invalid trials",
        )
    return decision_sha


def verify_qualified(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    preflight_freeze(root, expected_freeze_sha256)
    _, manifests = preflight_trial_inventory(root)
    preflight_revisit(manifests)
    result = verify_bundle(root)

    by_block: dict[str, list[tuple[str, str]]] = {}
    for manifest_path in manifests:
        trial_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        require(
            isinstance(manifest, dict)
            and manifest.get("campaign") != "pilot"
            and manifest.get("included_in_confirmatory_analysis") is True,
            "PILOT_CONFIRMATORY_CONTAMINATION",
            str(manifest_path),
        )
        decision_sha = verify_trial_surface(trial_dir, manifest)
        if manifest.get("policy") in PAIR_POLICIES:
            by_block.setdefault(manifest["paired_block_id"], []).append(
                (manifest["policy"], decision_sha)
            )

    for block_id, entries in by_block.items():
        if len(entries) > 1:
            require(
                len({digest for _, digest in entries}) == 1,
                "PAIRED_DECISION_INPUT_MISMATCH",
                block_id,
            )

    qualified = dict(result)
    qualified.update(
        {
            "externally_anchored_freeze_sha256": expected_freeze_sha256,
            "prospective_receipt_binding": "PASS",
            "decision_input_pairing": "PASS",
            "candidate_retention": "PASS",
            "zero_integrity_invalid_trials": "PASS",
        }
    )
    return qualified


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Externally anchored confirmatory verifier for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--expected-freeze-sha256",
        required=True,
        help="SHA-256 of confirmatory_freeze.json committed before confirmatory outcomes",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_qualified(args.root, args.expected_freeze_sha256)
    except Reject as exc:
        payload = {"verdict": "REJECT", "reason_class": exc.code, "detail": exc.detail}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            f"ACCEPT: {result['trial_count']} confirmatory trials; "
            f"freeze {result['externally_anchored_freeze_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
