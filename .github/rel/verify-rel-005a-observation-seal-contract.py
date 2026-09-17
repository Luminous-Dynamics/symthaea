#!/usr/bin/env python3
"""Static verifier for the preregistered REL-005A ObservationSeal contract."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

CONTRACT = pathlib.Path(".github/rel/rel-005a-observation-seal-contract.json")
EXECUTION_WORKFLOW = pathlib.Path(".github/workflows/rel-005a-execution-v3.yml")
EXECUTION_TEST = pathlib.Path("crates/core/symthaea-fep/tests/rel_graft_frame_execution_v3.rs")

EXPECTED_PARENT = "02563ef9196bba2a5a8e41f4e3a77aa60afe3321"
EXPECTED_EXECUTION_WORKFLOW_BLOB = "97ad6abe497e2bbb5bc1f1d8d14b4330996b3a95"
EXPECTED_EXECUTION_TEST_BLOB = "f1c31629b3b4924863fd460380e07ae14078a7c7"
EXPECTED_FROZEN_TEST_BLOB = "787ea051ae0bf8e5667b6925481afd46c15d0bc4"
EXPECTED_PREDICATE_HEAD = "43bf1d588447f602ce0f1549986bb558a839762c"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    workflow = EXECUTION_WORKFLOW.read_text()

    require(git("rev-parse", "HEAD^") == EXPECTED_PARENT, "seal-contract parent mismatch")
    require(
        git("rev-parse", f"HEAD^:{EXECUTION_WORKFLOW.as_posix()}")
        == EXPECTED_EXECUTION_WORKFLOW_BLOB,
        "ExecutionOnly workflow blob mismatch",
    )
    require(
        git("rev-parse", f"HEAD^:{EXECUTION_TEST.as_posix()}")
        == EXPECTED_EXECUTION_TEST_BLOB,
        "ExecutionOnly test blob mismatch",
    )
    require(
        git("rev-parse", "HEAD:crates/core/symthaea-fep/tests/rel_graft_frame.rs")
        == EXPECTED_FROZEN_TEST_BLOB,
        "frozen #3142 test blob mismatch",
    )

    require(contract["schema"] == "symthaea.rel.observation-seal-contract.v1", "schema mismatch")
    require(contract["authority"] == "ObservationSealContractOnly", "authority mismatch")
    require(contract["relation"] == "REL-005A", "relation mismatch")
    require(contract["execution_subject_head"] == EXPECTED_PARENT, "execution subject mismatch")
    require(contract["predicate_contract_head"] == EXPECTED_PREDICATE_HEAD, "predicate head mismatch")
    require(
        contract["expected_execution_observation_schema"]
        == "symthaea.rel.graft-frame-execution-observation.v3",
        "observation schema mismatch",
    )
    require(
        contract["expected_execution_receipt_schema"]
        == "symthaea.rel.execution-only-receipt.v3",
        "receipt schema mismatch",
    )
    require(contract["expected_execution_authority"] == "ExecutionOnly", "execution authority mismatch")
    require(contract["required_execution_result"] == "EXECUTION_OK", "execution result mismatch")
    require(contract["required_execution_exit_code"] == "0", "execution exit mismatch")

    files = contract["required_upstream_files"]
    require(len(files) == len(set(files)) == 10, "required upstream file census mismatch")
    for basename in files:
        require(basename in workflow, f"ExecutionOnly workflow does not stage {basename}")

    bindings = contract["required_identity_bindings"]
    require(
        bindings
        == [
            "subject_head",
            "predicate_contract_parent",
            "predicate_contract_blob",
            "frozen_scientific_subject",
            "frozen_test_blob",
        ],
        "identity binding set mismatch",
    )

    false_claims = contract["required_false_claims"]
    require(
        false_claims
        == [
            "observation_sealed",
            "comparison_only_adjudicated",
            "rel_005a_qualified",
            "scientific_pass",
            "scientific_fail",
        ],
        "false-claim set mismatch",
    )

    manifest = contract["seal_manifest"]
    require(manifest["hash_algorithm"] == "sha256", "manifest hash mismatch")
    require(manifest["chain_commitment_algorithm"] == "sha256", "chain hash mismatch")
    require(
        manifest["chain_commitment_input_fields"]
        == [
            "execution_subject_head",
            "execution_receipt_sha256",
            "observation_sha256",
            "artifact_manifest_sha256",
        ],
        "chain input mismatch",
    )

    output = contract["seal_output"]
    require(output["authority"] == "ObservationSeal", "seal output authority mismatch")
    require(output["observation_status"] == "sealed", "seal output status mismatch")
    require(output["adjudication"] == "not_run", "seal contract must not adjudicate")
    require(output["scientific_result"] == "not_run", "seal contract must not make scientific result")

    require(not any(contract["claims"].values()), "seal contract carries scientific claim")

    receipt = {
        "schema": "symthaea.rel.observation-seal-contract-static-receipt.v1",
        "authority": "ObservationSealContractOnly",
        "subject_head": git("rev-parse", "HEAD"),
        "execution_subject_head": EXPECTED_PARENT,
        "execution_workflow_blob": EXPECTED_EXECUTION_WORKFLOW_BLOB,
        "execution_test_blob": EXPECTED_EXECUTION_TEST_BLOB,
        "contract_sha256": sha256(CONTRACT),
        "required_upstream_file_count": len(files),
        "identity_binding_count": len(bindings),
        "static_contract_valid": True,
        "claims": {
            "observation_sealed": False,
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    out = pathlib.Path(os.environ["REL005A_SEAL_CONTRACT_RECEIPT_PATH"])
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
