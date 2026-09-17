#!/usr/bin/env python3
"""REL-005A QualificationOnly evaluator.

This layer never reads raw scientific observations or applies thresholds.
It verifies the authority chain and maps an already-valid ComparisonOnly
adjudication to the final REL-005A qualification status.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import Any

CONTRACT_SCHEMA = "symthaea.rel.qualification-only-contract.v1"
CONTRACT_AUTHORITY = "QualificationOnlyContractOnly"
OUTPUT_SCHEMA = "symthaea.rel.qualification-only.v1"
OUTPUT_AUTHORITY = "QualificationOnly"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def false_claims(value: dict[str, Any]) -> None:
    for key in ("rel_005a_qualified", "scientific_pass", "scientific_fail"):
        require(value.get(key) is False, f"upstream claim {key} must be false")


def qualify(
    contract: dict[str, Any],
    predicate_receipt: dict[str, Any],
    execution_receipt: dict[str, Any],
    seal: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    require(contract["schema"] == CONTRACT_SCHEMA, "qualification contract schema mismatch")
    require(contract["authority"] == CONTRACT_AUTHORITY, "qualification contract authority mismatch")

    required_count = contract["required_predicate_count"]
    predicate_head = contract["predicate_contract_head"]
    execution_head = contract["execution_subject_head"]
    frozen_subject = contract["frozen_scientific_subject"]
    frozen_blob = contract["frozen_test_blob"]

    require(predicate_receipt["authority"] == "PredicateContractOnly", "predicate receipt authority mismatch")
    require(predicate_receipt["predicate_count"] == required_count, "predicate receipt count mismatch")
    require(predicate_receipt["source_grounded"] is True, "predicate receipt not source-grounded")
    require(predicate_receipt["predicate_contract_static_valid"] is True, "predicate static contract invalid")
    require(predicate_receipt["frozen_scientific_subject"] == frozen_subject, "predicate frozen subject mismatch")
    require(predicate_receipt["frozen_test_blob"] == frozen_blob, "predicate frozen blob mismatch")
    false_claims(predicate_receipt["claims"])
    require(predicate_receipt["claims"].get("comparison_only_adjudicated") is False, "predicate receipt comparison claim must be false")

    require(execution_receipt["authority"] == "ExecutionOnly", "execution receipt authority mismatch")
    require(execution_receipt["subject_head"] == execution_head, "execution subject mismatch")
    require(execution_receipt["predicate_contract_parent"] == predicate_head, "execution predicate head mismatch")
    require(execution_receipt["frozen_scientific_subject"] == frozen_subject, "execution frozen subject mismatch")
    require(execution_receipt["frozen_test_blob"] == frozen_blob, "execution frozen blob mismatch")
    require(execution_receipt["result"] == "EXECUTION_OK", "execution is not EXECUTION_OK")
    require(str(execution_receipt["measurement_exit_code"]) == "0", "execution exit is not zero")
    require(execution_receipt["observation_present"] is True, "execution observation missing")
    require(isinstance(execution_receipt["observation_sha256"], str), "execution observation hash missing")
    false_claims(execution_receipt["claims"])
    require(execution_receipt["claims"].get("comparison_only_adjudicated") is False, "execution comparison claim must be false")

    require(seal["authority"] == "ObservationSeal", "seal authority mismatch")
    require(seal["observation_status"] == "sealed", "observation is not sealed")
    require(seal["adjudication"] == "not_run", "seal carries adjudication")
    require(seal["scientific_result"] == "not_run", "seal carries scientific result")
    require(seal["execution_subject_head"] == execution_head, "seal execution subject mismatch")
    require(seal["predicate_contract_head"] == predicate_head, "seal predicate head mismatch")
    require(seal["frozen_scientific_subject"] == frozen_subject, "seal frozen subject mismatch")
    require(seal["frozen_test_blob"] == frozen_blob, "seal frozen blob mismatch")
    require(seal["observation_sha256"] == execution_receipt["observation_sha256"], "seal/execution observation hash mismatch")
    false_claims(seal["claims"])
    require(seal["claims"].get("comparison_only_adjudicated") is False, "seal comparison claim must be false")

    require(comparison["authority"] == "ComparisonOnly", "comparison authority mismatch")
    require(comparison["execution_subject_head"] == execution_head, "comparison execution subject mismatch")
    require(comparison["predicate_contract_head"] == predicate_head, "comparison predicate head mismatch")
    require(comparison["frozen_scientific_subject"] == frozen_subject, "comparison frozen subject mismatch")
    require(comparison["frozen_test_blob"] == frozen_blob, "comparison frozen blob mismatch")
    require(comparison["observation_sha256"] == execution_receipt["observation_sha256"], "comparison/execution observation hash mismatch")
    require(
        comparison["seal_chain_commitment_sha256"] == seal["chain_commitment_sha256"],
        "comparison/seal commitment mismatch",
    )
    require(comparison["predicate_count"] == required_count, "comparison predicate count mismatch")
    require(
        comparison["passed_count"] + comparison["failed_count"] == required_count,
        "comparison counts inconsistent",
    )
    require(
        comparison["failed_count"] == len(comparison["failed_predicate_ids"]),
        "comparison failed-id count inconsistent",
    )
    require(comparison["claims"].get("comparison_only_adjudicated") is True, "comparison not adjudicated")
    false_claims(comparison["claims"])

    result = comparison["comparison_result"]
    require(result in contract["allowed_comparison_results"], "unsupported comparison result")
    if result == "ALL_PREDICATES_PASS":
        require(comparison["passed_count"] == required_count, "all-pass count mismatch")
        require(comparison["failed_count"] == 0, "all-pass has failures")
    else:
        require(comparison["failed_count"] > 0, "predicate-failure result has no failures")

    mapping = contract["qualification_mapping"][result]

    return {
        "schema": OUTPUT_SCHEMA,
        "authority": OUTPUT_AUTHORITY,
        "predicate_contract_head": predicate_head,
        "execution_subject_head": execution_head,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "predicate_count": required_count,
        "comparison_result": result,
        "passed_count": comparison["passed_count"],
        "failed_count": comparison["failed_count"],
        "failed_predicate_ids": comparison["failed_predicate_ids"],
        "qualification_completed": mapping["qualification_completed"],
        "rel_005a_qualified": mapping["rel_005a_qualified"],
        "scientific_pass": mapping["scientific_pass"],
        "scientific_fail": mapping["scientific_fail"],
    }


def synthetic_chain(contract: dict[str, Any], result: str) -> tuple[dict[str, Any], ...]:
    count = contract["required_predicate_count"]
    execution_head = contract["execution_subject_head"]
    predicate_head = contract["predicate_contract_head"]
    frozen_subject = contract["frozen_scientific_subject"]
    frozen_blob = contract["frozen_test_blob"]
    observation_hash = "33" * 32
    commitment = "44" * 32

    predicate_receipt = {
        "authority": "PredicateContractOnly",
        "predicate_count": count,
        "source_grounded": True,
        "predicate_contract_static_valid": True,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "claims": {
            "observation_sealed": False,
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    execution_receipt = {
        "authority": "ExecutionOnly",
        "subject_head": execution_head,
        "predicate_contract_parent": predicate_head,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "result": "EXECUTION_OK",
        "measurement_exit_code": "0",
        "observation_present": True,
        "observation_sha256": observation_hash,
        "claims": {
            "observation_sealed": False,
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    seal = {
        "authority": "ObservationSeal",
        "observation_status": "sealed",
        "adjudication": "not_run",
        "scientific_result": "not_run",
        "execution_subject_head": execution_head,
        "predicate_contract_head": predicate_head,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "observation_sha256": observation_hash,
        "chain_commitment_sha256": commitment,
        "claims": {
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    failed_ids = [] if result == "ALL_PREDICATES_PASS" else ["REL005A-P001"]
    failed_count = len(failed_ids)
    comparison = {
        "authority": "ComparisonOnly",
        "execution_subject_head": execution_head,
        "predicate_contract_head": predicate_head,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "observation_sha256": observation_hash,
        "seal_chain_commitment_sha256": commitment,
        "predicate_count": count,
        "passed_count": count - failed_count,
        "failed_count": failed_count,
        "failed_predicate_ids": failed_ids,
        "comparison_result": result,
        "claims": {
            "comparison_only_adjudicated": True,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    return predicate_receipt, execution_receipt, seal, comparison


def self_test(contract: dict[str, Any]) -> dict[str, Any]:
    chain = synthetic_chain(contract, "ALL_PREDICATES_PASS")
    passed = qualify(contract, *chain)
    require(passed["rel_005a_qualified"] is True, "pass self-test did not qualify")
    require(passed["scientific_pass"] is True, "pass self-test missing scientific pass")
    require(passed["scientific_fail"] is False, "pass self-test set scientific fail")

    chain = synthetic_chain(contract, "PREDICATE_FAILURES")
    failed = qualify(contract, *chain)
    require(failed["rel_005a_qualified"] is False, "failure self-test qualified unexpectedly")
    require(failed["scientific_pass"] is False, "failure self-test set scientific pass")
    require(failed["scientific_fail"] is True, "failure self-test missing scientific fail")

    chain = list(synthetic_chain(contract, "ALL_PREDICATES_PASS"))
    chain[1]["subject_head"] = "00" * 20
    try:
        qualify(contract, *chain)
    except ValueError as exc:
        require("execution subject mismatch" in str(exc), "identity self-test failed for wrong reason")
    else:
        raise ValueError("identity mismatch was accepted")

    return {
        "schema": "symthaea.rel.qualification-only-self-test.v1",
        "authority": "QualificationOnlyContractOnly",
        "pass_chain": "PASS",
        "predicate_failure_chain": "PASS",
        "identity_rejection": "PASS",
        "raw_observation_input_supported": False,
        "scientific_threshold_input_supported": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--predicate-receipt", type=pathlib.Path)
    parser.add_argument("--execution-receipt", type=pathlib.Path)
    parser.add_argument("--seal", type=pathlib.Path)
    parser.add_argument("--comparison", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load(args.contract)

    if args.self_test:
        print(json.dumps(self_test(contract), indent=2, sort_keys=True))
        return

    for name in ("predicate_receipt", "execution_receipt", "seal", "comparison", "output"):
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")

    result = qualify(
        contract,
        load(args.predicate_receipt),
        load(args.execution_receipt),
        load(args.seal),
        load(args.comparison),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
