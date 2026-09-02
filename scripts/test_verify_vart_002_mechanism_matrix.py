#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

from verify_vart_002_mechanism_matrix import Reject, verify

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/research/VART_002_MECHANISM_MATRIX.template.json"


def load() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def expect_reject(value: dict, code: str) -> None:
    try:
        verify(value)
    except Reject as exc:
        assert exc.code == code, (exc.code, code, exc.detail)
    else:
        raise AssertionError(f"expected {code}")


def main() -> None:
    baseline = load()
    result = verify(copy.deepcopy(baseline))
    assert result["result"] == "VART_002_MECHANISM_MATRIX_PASS"

    # M1: memory-capacity mismatch cannot masquerade as provenance ablation.
    mutated = copy.deepcopy(baseline)
    mutated["shared_matching_constraints"]["same_memory_capacity_budget"] = False
    expect_reject(mutated, "MECHANISM_CONFOUND_NOT_MATCHED")

    # M2: compute mismatch cannot masquerade as architecture advantage.
    mutated = copy.deepcopy(baseline)
    mutated["shared_matching_constraints"]["same_compute_budget"] = False
    expect_reject(mutated, "MECHANISM_CONFOUND_NOT_MATCHED")

    # M3: duplicate/replace a primary condition.
    mutated = copy.deepcopy(baseline)
    mutated["provenance_primary_conditions"].append(
        copy.deepcopy(mutated["provenance_primary_conditions"][0])
    )
    expect_reject(mutated, "PRIMARY_CONDITION_DUPLICATE")

    # M4: quietly change the FULL mechanism surface.
    mutated = copy.deepcopy(baseline)
    mutated["provenance_primary_conditions"][0]["retrieval_filtering"] = False
    expect_reject(mutated, "PRIMARY_MECHANISM_DRIFT")

    # M5: collapse uncertainty into the provenance claim family.
    mutated = copy.deepcopy(baseline)
    mutated["uncertainty_block"]["separate_claim_family"] = False
    expect_reject(mutated, "CLAIM_FAMILY_COLLAPSE")

    # M6: insert an overall quality scalar.
    mutated = copy.deepcopy(baseline)
    mutated["candidate_endpoint_families"].append("world_quality")
    expect_reject(mutated, "FORBIDDEN_SCALAR_AGGREGATE")

    # M7: a design matrix cannot grant confirmatory execution authority.
    mutated = copy.deepcopy(baseline)
    mutated["confirmatory_execution_authorized"] = True
    expect_reject(mutated, "PREMATURE_EXECUTION_AUTHORITY")

    print("VART_002_MECHANISM_MATRIX_SUITE=PASS")


if __name__ == "__main__":
    main()
