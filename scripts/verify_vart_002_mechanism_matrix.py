#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.vart-002.mechanism-matrix.v1"
EXPERIMENT = "VART-002-EPISTEMIC-MECHANISMS"

MATCHING_KEYS = {
    "same_starting_world_snapshot",
    "same_subject_visible_observations",
    "same_candidate_generation_surface",
    "same_physical_admission_policy",
    "same_memory_capacity_budget",
    "same_retrieval_budget",
    "same_compute_budget",
    "same_subject_source",
    "same_action_authority",
    "same_scoring_contract",
}

FEATURE_KEYS = {
    "source_identity",
    "epistemic_domain",
    "temporal_order",
    "immutable_history",
    "retrieval_filtering",
    "counterfactual_taint",
}

EXPECTED_PRIMARY = {
    "FULL_TYPED_PROVENANCE": {
        "source_identity": True,
        "epistemic_domain": True,
        "temporal_order": True,
        "immutable_history": True,
        "retrieval_filtering": True,
        "counterfactual_taint": True,
    },
    "NO_PROVENANCE": {
        "source_identity": False,
        "epistemic_domain": False,
        "temporal_order": False,
        "immutable_history": False,
        "retrieval_filtering": False,
        "counterfactual_taint": False,
    },
    "SOURCE_LABEL_ONLY": {
        "source_identity": True,
        "epistemic_domain": False,
        "temporal_order": False,
        "immutable_history": False,
        "retrieval_filtering": False,
        "counterfactual_taint": False,
    },
    "DOMAIN_NAMESPACE_ONLY": {
        "source_identity": False,
        "epistemic_domain": True,
        "temporal_order": False,
        "immutable_history": False,
        "retrieval_filtering": False,
        "counterfactual_taint": True,
    },
    "IMMUTABLE_HISTORY_ONLY": {
        "source_identity": False,
        "epistemic_domain": False,
        "temporal_order": True,
        "immutable_history": True,
        "retrieval_filtering": False,
        "counterfactual_taint": False,
    },
    "NO_TEMPORAL_ORDER": {
        "source_identity": True,
        "epistemic_domain": True,
        "temporal_order": False,
        "immutable_history": True,
        "retrieval_filtering": True,
        "counterfactual_taint": True,
    },
    "NO_SOURCE_IDENTITY": {
        "source_identity": False,
        "epistemic_domain": True,
        "temporal_order": True,
        "immutable_history": True,
        "retrieval_filtering": True,
        "counterfactual_taint": True,
    },
    "RETRIEVAL_FILTER_REMOVED": {
        "source_identity": True,
        "epistemic_domain": True,
        "temporal_order": True,
        "immutable_history": True,
        "retrieval_filtering": False,
        "counterfactual_taint": True,
    },
}

REQUIRED_FORBIDDEN = {
    "reuse_of_vart001_confirmatory_fixtures_as_hidden_confirmatory_worlds",
    "reuse_of_vart001_exact_confirmatory_seeds",
    "scalar_world_quality",
    "post_unblinding_threshold_tuning",
    "memory_capacity_mismatch_masquerading_as_provenance_ablation",
    "compute_budget_mismatch_masquerading_as_architecture_advantage",
}


class Reject(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise Reject(code, detail)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Reject("MATRIX_INVALID_JSON", str(exc)) from exc
    require(isinstance(value, dict), "MATRIX_SCHEMA_INVALID", "root must be object")
    return value


def verify(matrix: dict[str, Any]) -> dict[str, Any]:
    require(matrix.get("schema") == SCHEMA, "MATRIX_SCHEMA_INVALID", "schema")
    require(matrix.get("experiment_id") == EXPERIMENT, "MATRIX_SCHEMA_INVALID", "experiment_id")

    # This branch is a design instrument, never execution authority.
    require(
        matrix.get("confirmatory_execution_authorized") is False,
        "PREMATURE_EXECUTION_AUTHORITY",
        "confirmatory execution must remain false before a separately frozen preregistration",
    )
    require(
        matrix.get("claim_authorized") is False,
        "PREMATURE_CLAIM_AUTHORITY",
        "claim authority must remain false",
    )

    matching = matrix.get("shared_matching_constraints")
    require(isinstance(matching, dict), "MATCHING_CONSTRAINTS_INVALID", "missing object")
    require(set(matching) == MATCHING_KEYS, "MATCHING_CONSTRAINTS_INVALID", "unexpected/missing matching keys")
    for key in sorted(MATCHING_KEYS):
        require(
            matching.get(key) is True,
            "MECHANISM_CONFOUND_NOT_MATCHED",
            key,
        )

    conditions = matrix.get("provenance_primary_conditions")
    require(isinstance(conditions, list), "PRIMARY_CONDITION_INVALID", "conditions must be list")
    seen: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        require(isinstance(condition, dict), "PRIMARY_CONDITION_INVALID", "condition must be object")
        cid = condition.get("id")
        require(isinstance(cid, str) and cid, "PRIMARY_CONDITION_INVALID", "missing id")
        require(cid not in seen, "PRIMARY_CONDITION_DUPLICATE", cid)
        require(
            set(condition) == FEATURE_KEYS | {"id"},
            "PRIMARY_CONDITION_INVALID",
            f"{cid}: feature surface changed",
        )
        for key in FEATURE_KEYS:
            require(isinstance(condition[key], bool), "PRIMARY_CONDITION_INVALID", f"{cid}.{key}")
        seen[cid] = condition

    require(
        set(seen) == set(EXPECTED_PRIMARY),
        "PRIMARY_CONDITION_SET_MISMATCH",
        "primary condition IDs changed",
    )
    for cid, expected in EXPECTED_PRIMARY.items():
        actual = {key: seen[cid][key] for key in FEATURE_KEYS}
        require(actual == expected, "PRIMARY_MECHANISM_DRIFT", cid)

    uncertainty = matrix.get("uncertainty_block")
    require(isinstance(uncertainty, dict), "UNCERTAINTY_BLOCK_INVALID", "missing")
    require(
        uncertainty.get("separate_claim_family") is True,
        "CLAIM_FAMILY_COLLAPSE",
        "uncertainty must remain separate from provenance mechanism inference",
    )

    endpoints = matrix.get("candidate_endpoint_families")
    require(isinstance(endpoints, list) and endpoints, "ENDPOINT_FAMILY_INVALID", "missing endpoints")
    endpoint_text = " ".join(str(x).lower() for x in endpoints)
    for forbidden_scalar in ("world_quality", "creative_score", "intelligence_score"):
        require(
            forbidden_scalar not in endpoint_text,
            "FORBIDDEN_SCALAR_AGGREGATE",
            forbidden_scalar,
        )

    forbidden = matrix.get("forbidden")
    require(isinstance(forbidden, list), "FORBIDDEN_SET_INVALID", "missing")
    require(
        REQUIRED_FORBIDDEN.issubset(set(forbidden)),
        "FORBIDDEN_SET_INVALID",
        "required contamination/confound protections missing",
    )

    unresolved = matrix.get("unresolved_before_preregistration")
    require(isinstance(unresolved, list) and unresolved, "PREREGISTRATION_INCOMPLETE", "unresolved list missing")

    return {
        "result": "VART_002_MECHANISM_MATRIX_PASS",
        "primary_condition_count": len(conditions),
        "matching_constraint_count": len(matching),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(load_json(args.matrix))
    except Reject as exc:
        if args.json:
            print(json.dumps({"result": "REJECT", "code": exc.code, "detail": exc.detail}, sort_keys=True))
        else:
            print(f"REJECT: {exc.code}: {exc.detail}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
