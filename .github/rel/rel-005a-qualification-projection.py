#!/usr/bin/env python3
"""Project REL-005A authority evidence into a metric-free QualificationOnly bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
from typing import Any

CONTRACT_SCHEMA = "symthaea.rel.qualification-projection-contract.v1"
CONTRACT_AUTHORITY = "EvidenceProjectionContractOnly"
PROJECTED_COMPARISON_SCHEMA = "symthaea.rel.comparison-only-qualification-receipt.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def project(
    contract: dict[str, Any],
    predicate_path: pathlib.Path,
    execution_path: pathlib.Path,
    seal_path: pathlib.Path,
    comparison_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    require(contract["schema"] == CONTRACT_SCHEMA, "projection contract schema mismatch")
    require(contract["authority"] == CONTRACT_AUTHORITY, "projection contract authority mismatch")

    predicate = load(predicate_path)
    execution = load(execution_path)
    seal = load(seal_path)
    comparison = load(comparison_path)

    predicate_head = contract["predicate_contract_head"]
    execution_head = contract["execution_subject_head"]
    frozen_subject = contract["frozen_scientific_subject"]
    frozen_blob = contract["frozen_test_blob"]

    require(predicate["authority"] == "PredicateContractOnly", "predicate authority mismatch")
    require(predicate["subject_head"] == predicate_head, "predicate head mismatch")
    require(predicate["predicate_count"] == 41, "predicate count mismatch")
    require(predicate["source_grounded"] is True, "predicate receipt not source-grounded")
    require(predicate["predicate_contract_static_valid"] is True, "predicate contract not static-valid")
    require(predicate["frozen_scientific_subject"] == frozen_subject, "predicate frozen subject mismatch")
    require(predicate["frozen_test_blob"] == frozen_blob, "predicate frozen blob mismatch")
    require(not any(predicate["claims"].values()), "predicate receipt exceeds authority")

    require(execution["authority"] == "ExecutionOnly", "execution authority mismatch")
    require(execution["subject_head"] == execution_head, "execution head mismatch")
    require(execution["predicate_contract_parent"] == predicate_head, "execution predicate head mismatch")
    require(execution["frozen_scientific_subject"] == frozen_subject, "execution frozen subject mismatch")
    require(execution["frozen_test_blob"] == frozen_blob, "execution frozen blob mismatch")
    require(execution["result"] == "EXECUTION_OK", "execution is not EXECUTION_OK")
    require(str(execution["measurement_exit_code"]) == "0", "execution exit is not zero")
    require(execution["observation_present"] is True, "execution observation missing")
    require(not any(execution["claims"].values()), "execution receipt exceeds authority")

    require(seal["authority"] == "ObservationSeal", "seal authority mismatch")
    require(seal["observation_status"] == "sealed", "observation is not sealed")
    require(seal["adjudication"] == "not_run", "seal carries adjudication")
    require(seal["scientific_result"] == "not_run", "seal carries scientific result")
    require(seal["execution_subject_head"] == execution_head, "seal execution head mismatch")
    require(seal["predicate_contract_head"] == predicate_head, "seal predicate head mismatch")
    require(seal["frozen_scientific_subject"] == frozen_subject, "seal frozen subject mismatch")
    require(seal["frozen_test_blob"] == frozen_blob, "seal frozen blob mismatch")
    require(seal["observation_sha256"] == execution["observation_sha256"], "seal/execution observation mismatch")
    require(seal["claims"]["observation_sealed"] is True, "seal does not claim sealed observation")
    for key in ("comparison_only_adjudicated", "rel_005a_qualified", "scientific_pass", "scientific_fail"):
        require(seal["claims"][key] is False, f"seal claim {key} exceeds authority")

    require(comparison["authority"] == "ComparisonOnly", "comparison authority mismatch")
    require(comparison["predicate_contract_head"] == predicate_head, "comparison predicate head mismatch")
    require(comparison["execution_subject_head"] == execution_head, "comparison execution head mismatch")
    require(comparison["frozen_scientific_subject"] == frozen_subject, "comparison frozen subject mismatch")
    require(comparison["frozen_test_blob"] == frozen_blob, "comparison frozen blob mismatch")
    require(comparison["observation_sha256"] == execution["observation_sha256"], "comparison observation mismatch")
    require(comparison["seal_chain_commitment_sha256"] == seal["chain_commitment_sha256"], "comparison seal commitment mismatch")
    require(comparison["predicate_count"] == 41, "comparison predicate count mismatch")
    require(comparison["passed_count"] + comparison["failed_count"] == 41, "comparison counts inconsistent")
    require(comparison["failed_count"] == len(comparison["failed_predicate_ids"]), "failed-id count inconsistent")
    require(comparison["comparison_result"] in {"ALL_PREDICATES_PASS", "PREDICATE_FAILURES"}, "unsupported comparison result")
    if comparison["comparison_result"] == "ALL_PREDICATES_PASS":
        require(comparison["passed_count"] == 41 and comparison["failed_count"] == 0, "all-pass count mismatch")
    else:
        require(comparison["failed_count"] > 0, "predicate-failure result has no failures")
    require(comparison["claims"] == {
        "comparison_only_adjudicated": True,
        "rel_005a_qualified": False,
        "scientific_pass": False,
        "scientific_fail": False,
    }, "comparison claim boundary mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(predicate_path, output_dir / "predicate-contract-receipt.json")
    shutil.copy2(execution_path, output_dir / "execution-v3-receipt.json")
    shutil.copy2(seal_path, output_dir / "observation-seal-v3.json")

    projected = {
        "schema": PROJECTED_COMPARISON_SCHEMA,
        "authority": "ComparisonOnly",
        "predicate_contract_head": comparison["predicate_contract_head"],
        "execution_subject_head": comparison["execution_subject_head"],
        "frozen_scientific_subject": comparison["frozen_scientific_subject"],
        "frozen_test_blob": comparison["frozen_test_blob"],
        "observation_sha256": comparison["observation_sha256"],
        "seal_chain_commitment_sha256": comparison["seal_chain_commitment_sha256"],
        "predicate_count": comparison["predicate_count"],
        "passed_count": comparison["passed_count"],
        "failed_count": comparison["failed_count"],
        "failed_predicate_ids": comparison["failed_predicate_ids"],
        "comparison_result": comparison["comparison_result"],
        "claims": comparison["claims"],
        "full_comparison_sha256": sha256(comparison_path),
    }
    projected_path = output_dir / "comparison-only-qualification-receipt.json"
    write_json(projected_path, projected)

    authority_files = [
        "predicate-contract-receipt.json",
        "execution-v3-receipt.json",
        "observation-seal-v3.json",
        "comparison-only-qualification-receipt.json",
    ]
    files = []
    for name in authority_files:
        path = output_dir / name
        files.append({
            "basename": name,
            "byte_length": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema": "symthaea.rel.qualification-input-manifest.v2",
        "authority": "EvidenceProjectionOnly",
        "predicate_contract_head": predicate_head,
        "execution_subject_head": execution_head,
        "observation_sha256": execution["observation_sha256"],
        "seal_chain_commitment_sha256": seal["chain_commitment_sha256"],
        "comparison_result": comparison["comparison_result"],
        "raw_observation_included": False,
        "raw_execution_logs_included": False,
        "detailed_predicate_values_included": False,
        "files": files,
    }
    write_json(output_dir / "qualification-input-manifest.json", manifest)

    expected = set(contract["output_files"])
    actual = {p.name for p in output_dir.iterdir() if p.is_file()}
    require(actual == expected, f"output file census mismatch: {sorted(actual)}")
    return manifest


def synthetic_chain(contract: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    predicate_head = contract["predicate_contract_head"]
    execution_head = contract["execution_subject_head"]
    frozen_subject = contract["frozen_scientific_subject"]
    frozen_blob = contract["frozen_test_blob"]
    observation_hash = "33" * 32
    commitment = "44" * 32
    predicate = {
        "schema": "symthaea.rel.predicate-contract-static-receipt.v1",
        "authority": "PredicateContractOnly",
        "subject_head": predicate_head,
        "predicate_count": 41,
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
    execution = {
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
            "observation_sealed": True,
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    comparison = {
        "schema": "symthaea.rel.comparison-only.v1",
        "authority": "ComparisonOnly",
        "predicate_contract_head": predicate_head,
        "execution_subject_head": execution_head,
        "frozen_scientific_subject": frozen_subject,
        "frozen_test_blob": frozen_blob,
        "observation_sha256": observation_hash,
        "seal_chain_commitment_sha256": commitment,
        "predicate_count": 41,
        "passed_count": 40,
        "failed_count": 1,
        "failed_predicate_ids": ["REL005A-P041"],
        "comparison_result": "PREDICATE_FAILURES",
        "predicates": [{
            "id": "REL005A-P041",
            "observed": "SENTINEL_SCIENTIFIC_VALUE_918273645",
            "expected": "SENTINEL_THRESHOLD_VALUE_564738291",
            "passed": False,
        }],
        "threshold_table": {"SENTINEL": "SENTINEL_THRESHOLD_VALUE_564738291"},
        "claims": {
            "comparison_only_adjudicated": True,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    return predicate, execution, seal, comparison


def self_test(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        inputs = root / "inputs"
        output = root / "output"
        inputs.mkdir()
        predicate, execution, seal, comparison = synthetic_chain(contract)
        paths = []
        for name, value in (
            ("predicate.json", predicate),
            ("execution.json", execution),
            ("seal.json", seal),
            ("comparison.json", comparison),
        ):
            path = inputs / name
            write_json(path, value)
            paths.append(path)
        manifest = project(contract, *paths, output)
        require(manifest["detailed_predicate_values_included"] is False, "projection claims detailed values")
        combined = b"\n".join(p.read_bytes() for p in output.iterdir() if p.is_file())
        require(b"SENTINEL_SCIENTIFIC_VALUE_918273645" not in combined, "scientific sentinel leaked")
        require(b"SENTINEL_THRESHOLD_VALUE_564738291" not in combined, "threshold sentinel leaked")
        projected = load(output / "comparison-only-qualification-receipt.json")
        require("predicates" not in projected, "detailed predicates leaked")
        require("threshold_table" not in projected, "threshold table leaked")
        require("observed" not in projected, "observed values leaked")
        require("expected" not in projected, "expected values leaked")
        require(projected["failed_predicate_ids"] == ["REL005A-P041"], "failed predicate identity lost")
        require(projected["comparison_result"] == "PREDICATE_FAILURES", "comparison result lost")
    return {
        "schema": "symthaea.rel.qualification-projection-self-test.v1",
        "authority": "EvidenceProjectionContractOnly",
        "metric_sentinel_rejected": True,
        "threshold_sentinel_rejected": True,
        "detailed_predicates_removed": True,
        "failed_predicate_identity_preserved": True,
        "comparison_result_preserved": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--predicate-receipt", type=pathlib.Path)
    parser.add_argument("--execution-receipt", type=pathlib.Path)
    parser.add_argument("--seal", type=pathlib.Path)
    parser.add_argument("--comparison", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load(args.contract)
    if args.self_test:
        print(json.dumps(self_test(contract), indent=2, sort_keys=True))
        return
    for name in ("predicate_receipt", "execution_receipt", "seal", "comparison", "output_dir"):
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    manifest = project(
        contract,
        args.predicate_receipt,
        args.execution_receipt,
        args.seal,
        args.comparison,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
