#!/usr/bin/env python3
"""Static verifier for the preregistered REL-005A full predicate contract.

Authority: PredicateContractOnly. This verifier never runs Cargo, production code,
or scientific observations. It only binds the machine-readable predicate census
to the immutable #3142 source blob and checks internal completeness/invariants.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

CONTRACT_PATH = pathlib.Path(".github/rel/rel-005a-full-predicate-contract.json")
TEST_PATH = pathlib.Path("crates/core/symthaea-fep/tests/rel_graft_frame.rs")
FROZEN_SUBJECT = "7f5826675b44dd0d3f702f62bc9818f7990e4a01"
FROZEN_TEST_BLOB = "787ea051ae0bf8e5667b6925481afd46c15d0bc4"
LEGACY_REPLAY_PARENT = "db4a76843b7a1538259a91443c07fbebfa9f3e3e"
EXPECTED_PREDICATE_COUNT = 41


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    source = TEST_PATH.read_text()

    require(contract["schema"] == "symthaea.rel.full-predicate-contract.v1", "schema mismatch")
    require(contract["authority"] == "PredicateContractOnly", "authority mismatch")
    require(contract["relation"] == "REL-005A", "relation mismatch")
    require(contract["frozen_scientific_subject"] == FROZEN_SUBJECT, "frozen subject mismatch")
    require(contract["frozen_test_blob"] == FROZEN_TEST_BLOB, "frozen test blob declaration mismatch")
    require(contract["legacy_replay_parent"] == LEGACY_REPLAY_PARENT, "legacy replay parent mismatch")
    require(git("rev-parse", f"HEAD:{TEST_PATH.as_posix()}") == FROZEN_TEST_BLOB, "committed frozen test blob mismatch")

    thresholds = contract["thresholds"]
    require(thresholds == {
        "POSITIVE_TOL": 1.0e-5,
        "FIXED_HADAMARD_MIN_DEFECT": 1.2,
        "RAW_COSINE_MIN_DRIFT": 0.25,
        "NEGATIVE_HELD_OUT_MIN_ERROR": 0.20,
        "NEGATIVE_DISPERSION_MIN": 0.10,
        "INVERSE_FLOOR": 1.0e-7,
    }, "threshold set mismatch")

    source_constants = [
        "const POSITIVE_TOL: f32 = 1.0e-5;",
        "const FIXED_HADAMARD_MIN_DEFECT: f32 = 1.2;",
        "const RAW_COSINE_MIN_DRIFT: f32 = 0.25;",
        "const NEGATIVE_HELD_OUT_MIN_ERROR: f32 = 0.20;",
        "const NEGATIVE_DISPERSION_MIN: f32 = 0.10;",
        "const INVERSE_FLOOR: f32 = 1.0e-7;",
    ]
    for fragment in source_constants:
        require(fragment in source, f"missing frozen threshold fragment: {fragment}")

    predicates = contract["predicates"]
    require(len(predicates) == EXPECTED_PREDICATE_COUNT, "predicate count mismatch")
    expected_ids = [f"REL005A-P{i:03d}" for i in range(1, EXPECTED_PREDICATE_COUNT + 1)]
    ids = [item["id"] for item in predicates]
    require(ids == expected_ids, "predicate ids must be complete and ordered")
    observations = [item["observation"] for item in predicates]
    require(len(observations) == len(set(observations)), "duplicate predicate observation key")

    legal_ops = {"eq", "le", "gt", "le_threshold", "gt_threshold"}
    for item in predicates:
        require(item["operator"] in legal_ops, f"illegal operator in {item['id']}")
        if item["operator"].endswith("_threshold"):
            require(item.get("threshold") in thresholds, f"unknown threshold in {item['id']}")
            require("value" not in item, f"threshold predicate carries literal value in {item['id']}")
        else:
            require("value" in item, f"literal predicate lacks value in {item['id']}")
            require("threshold" not in item, f"literal predicate carries threshold in {item['id']}")

    required_new = set(contract["required_raw_observation_schema"]["new_required_observables_not_complete_in_legacy_metrics"])
    derived_new = {item["observation"] for item in predicates if not item["legacy_serialized"]}
    require(required_new == derived_new, "new raw observation census mismatch")

    claims = contract["claims"]
    require(set(claims) == {
        "observation_sealed",
        "comparison_only_adjudicated",
        "rel_005a_qualified",
        "scientific_pass",
        "scientific_fail",
    }, "claims key set mismatch")
    require(not any(claims.values()), "PredicateContractOnly must not make scientific/observation claims")

    # Ground every normalized predicate in the immutable frozen source. P001/P002
    # intentionally split the source's compound finite/range assertion into two
    # logically conjunctive raw comparisons; P041 normalizes `.any(nonfinite)` to
    # an equivalent raw non-finite count > 0 for downstream ComparisonOnly use.
    source_needles = {
        "REL005A-P001": "value.is_finite() && value.abs() <= 1.0",
        "REL005A-P002": "value.is_finite() && value.abs() <= 1.0",
        "REL005A-P003": "assert_eq!(positive_inverse_floor_affected_count, 0);",
        "REL005A-P004": "shared-mask A->B fixture must produce a transform",
        "REL005A-P005": "assert!(mask_recovery_max <= POSITIVE_TOL);",
        "REL005A-P006": "assert!(shared_mask_dispersion_max <= POSITIVE_TOL);",
        "REL005A-P007": "assert!(construction_max <= POSITIVE_TOL);",
        "REL005A-P008": "assert!(held_out_max <= POSITIVE_TOL);",
        "REL005A-P009": "shared-mask B->C fixture must produce a transform",
        "REL005A-P010": "shared-mask A->C fixture must produce a transform",
        "REL005A-P011": "max_abs_error(&estimated_ac, &expected_ac) <= POSITIVE_TOL",
        "REL005A-P012": "assert!(direct_vs_composed_mask_error <= POSITIVE_TOL);",
        "REL005A-P013": "assert!(direct_vs_sequential_transport_error <= POSITIVE_TOL);",
        "REL005A-P014": "shared-mask C->A fixture must produce a transform",
        "REL005A-P015": "assert!(loop_closure_max <= POSITIVE_TOL);",
        "REL005A-P016": "assert!(bundle_covariance_defect <= POSITIVE_TOL);",
        "REL005A-P017": "assert!(transported_unit_defect <= POSITIVE_TOL);",
        "REL005A-P018": "assert!(fixed_hadamard_defect > FIXED_HADAMARD_MIN_DEFECT);",
        "REL005A-P019": "assert!(dressed_hadamard_defect <= POSITIVE_TOL);",
        "REL005A-P020": "assert!(dressed_unit_action_defect <= POSITIVE_TOL);",
        "REL005A-P021": "assert!(raw_cosine_drift > RAW_COSINE_MIN_DRIFT);",
        "REL005A-P022": "assert!(pulled_back_cosine_drift <= POSITIVE_TOL);",
        "REL005A-P023": "compute_xenobot_graft_transform(&[], &[]).is_none()",
        "REL005A-P024": "compute_xenobot_graft_transform(&anchors_b[..2], &anchors_a[..1])",
        "REL005A-P025": "assert!(dimension_mismatch_panics);",
        "REL005A-P026": "shuffled equal-size anchors still produce the current API estimate",
        "REL005A-P027": "assert!(shuffled_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);",
        "REL005A-P028": "assert!(shuffled_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);",
        "REL005A-P029": "mixed masks are currently averaged rather than rejected",
        "REL005A-P030": "assert!(mixed_mask_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);",
        "REL005A-P031": "assert!(mixed_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);",
        "REL005A-P032": "one corrupted anchor is currently averaged rather than rejected",
        "REL005A-P033": "assert!(corrupted_anchor_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);",
        "REL005A-P034": "assert!(corrupted_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);",
        "REL005A-P035": "near-floor input still produces the current API estimate",
        "REL005A-P036": "assert_eq!(inverse_floor_affected_count, 1);",
        "REL005A-P037": "assert!(near_floor_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);",
        "REL005A-P038": "coordinate reversal currently yields a best-fit mask estimate",
        "REL005A-P039": "assert!(permutation_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);",
        "REL005A-P040": "current API does not reject non-finite anchor content",
        "REL005A-P041": "assert!(nonfinite_input_returns_nonfinite_transform);",
    }
    require(set(source_needles) == set(expected_ids), "source grounding census mismatch")
    for predicate_id, fragment in source_needles.items():
        require(fragment in source, f"{predicate_id} is not grounded in frozen source")

    receipt = {
        "schema": "symthaea.rel.predicate-contract-static-receipt.v1",
        "authority": "PredicateContractOnly",
        "subject_head": git("rev-parse", "HEAD"),
        "parent_head": git("rev-parse", "HEAD^"),
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_TEST_BLOB,
        "contract_sha256": sha256(CONTRACT_PATH),
        "predicate_count": EXPECTED_PREDICATE_COUNT,
        "source_grounded": True,
        "raw_observation_gap_census_complete": True,
        "predicate_contract_static_valid": True,
        "claims": claims,
    }
    receipt_path = pathlib.Path(os.environ["REL005A_PREDICATE_RECEIPT_PATH"])
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
