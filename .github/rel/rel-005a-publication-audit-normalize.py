#!/usr/bin/env python3
"""Byte-preserving filename normalization for REL-005A publication audit receipts."""

from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
from typing import Any

CONTRACT_SCHEMA = "symthaea.rel.publication-audit-normalization-contract.v1"
CONTRACT_AUTHORITY = "PublicationAuditNormalizationContractOnly"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=no_duplicate_pairs)
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema mismatch")
    require(contract.get("authority") == CONTRACT_AUTHORITY, "contract authority mismatch")
    require(contract.get("relation") == "REL-005A", "relation mismatch")
    require(contract.get("source_pipeline_head") == "460b4c8a8f457f4bbf22820716ed6057d260d4fd", "source pipeline head mismatch")
    require(contract.get("source_run_id") == 35439414230, "source run mismatch")
    files = contract.get("source_files")
    require(isinstance(files, dict), "source_files missing")
    require(set(files) == {"authority-receipt-extraction.json", "sealed-manifest-replay.json"}, "source file contract mismatch")
    require(files["authority-receipt-extraction.json"]["canonical_name"] == "authority-receipt-extraction.json", "extraction canonical name mismatch")
    require(files["sealed-manifest-replay.json"]["canonical_name"] == "sealed-manifest-replay-receipt.json", "replay canonical name mismatch")
    rules = contract["rules"]
    for key in (
        "exact_source_census_required",
        "exact_output_census_required",
        "byte_preserving_copy_required",
        "json_mutation_forbidden",
        "duplicate_json_keys_rejected",
        "source_receipts_must_make_no_qualification_or_scientific_claims",
    ):
        require(rules.get(key) is True, f"{key} must be true")
    require(rules.get("transport_identity_is_scientific_authority") is False, "transport identity promoted")
    boundary = contract["authority_boundary"]
    require(boundary.get("may_rename_verified_metric_free_receipts") is True, "rename authority missing")
    for key in ("may_change_qualification_result", "may_apply_scientific_thresholds", "may_read_raw_scientific_observation", "may_read_execution_logs"):
        require(boundary.get(key) is False, f"authority boundary too broad: {key}")
    require(not any(contract["claims"].values()), "contract claims must all be false")


def validate_receipt(name: str, value: dict[str, Any], spec: dict[str, Any]) -> None:
    require(value.get("schema") == spec["schema"], f"{name}: schema mismatch")
    require(value.get("authority") == spec["authority"], f"{name}: authority mismatch")
    claims = value.get("claims")
    require(isinstance(claims, dict), f"{name}: claims missing")
    for key in ("qualification_completed", "rel_005a_qualified", "scientific_pass", "scientific_fail"):
        require(claims.get(key) is False, f"{name}: claim {key} exceeds authority")
    if name == "sealed-manifest-replay.json":
        require(value.get("exact_file_census_verified") is True, "replay census not verified")
        require(value.get("byte_lengths_verified") is True, "replay lengths not verified")
        require(value.get("sha256_commitments_verified") is True, "replay hashes not verified")
        require(value.get("scientific_observation_fields_parsed") is False, "replay parsed scientific fields")
        require(value.get("execution_logs_parsed") is False, "replay parsed logs")
    else:
        require(value.get("raw_observation_output") is False, "extractor emitted raw observation")
        require(value.get("execution_logs_output") is False, "extractor emitted logs")
        require(value.get("scientific_observation_fields_parsed") is False, "extractor parsed scientific fields")


def normalize(contract: dict[str, Any], source: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    validate_contract(contract)
    require(source.is_dir(), "source directory missing")
    source_files = [p for p in source.rglob("*") if p.is_file()]
    source_rel = {p.relative_to(source).as_posix() for p in source_files}
    expected_source = set(contract["source_files"])
    require(source_rel == expected_source, f"source census mismatch: {sorted(source_rel)}")
    require(not output.exists() or not any(output.iterdir()), "output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    mappings = []
    for source_name in sorted(expected_source):
        spec = contract["source_files"][source_name]
        src = source / source_name
        value = load_json(src)
        validate_receipt(source_name, value, spec)
        dst = output / spec["canonical_name"]
        shutil.copyfile(src, dst)
        require(src.read_bytes() == dst.read_bytes(), f"{source_name}: copy was not byte-preserving")
        mappings.append({
            "source_name": source_name,
            "canonical_name": spec["canonical_name"],
            "byte_length": src.stat().st_size,
            "sha256": sha256(src),
            "bytes_unchanged": True,
        })

    expected_output = {spec["canonical_name"] for spec in contract["source_files"].values()}
    actual_output = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    require(actual_output == expected_output, f"output census mismatch: {sorted(actual_output)}")

    return {
        "schema": "symthaea.rel.publication-audit-normalization-receipt.v1",
        "authority": "PublicationAuditNormalizationOnly",
        "source_pipeline_head": contract["source_pipeline_head"],
        "source_run_id": contract["source_run_id"],
        "source_artifact_name": contract["source_artifact_name"],
        "mappings": mappings,
        "exact_source_census_verified": True,
        "exact_output_census_verified": True,
        "byte_preserving_normalization_verified": True,
        "json_payload_mutated": False,
        "transport_identity_scientific_authority": False,
        "claims": {
            "normalization_completed": True,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def synthetic_source(root: pathlib.Path) -> None:
    extraction = {
        "schema": "symthaea.rel.authority-receipt-extraction-receipt.v2",
        "authority": "AuthorityReceiptExtractionOnly",
        "raw_observation_output": False,
        "execution_logs_output": False,
        "scientific_observation_fields_parsed": False,
        "claims": {
            "comparison_only_adjudicated": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    replay = {
        "schema": "symthaea.rel.sealed-manifest-replay-receipt.v1",
        "authority": "SealedManifestReplayOnly",
        "exact_file_census_verified": True,
        "byte_lengths_verified": True,
        "sha256_commitments_verified": True,
        "scientific_observation_fields_parsed": False,
        "execution_logs_parsed": False,
        "claims": {
            "comparison_only_adjudicated": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "authority-receipt-extraction.json", extraction)
    write_json(root / "sealed-manifest-replay.json", replay)


def self_test(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        src = root / "source"
        out = root / "out"
        synthetic_source(src)
        receipt = normalize(contract, src, out)
        require(receipt["byte_preserving_normalization_verified"] is True, "byte-preserving self-test failed")
        require((src / "sealed-manifest-replay.json").read_bytes() == (out / "sealed-manifest-replay-receipt.json").read_bytes(), "renamed replay bytes changed")

        extra = root / "extra-source"
        shutil.copytree(src, extra)
        (extra / "unexpected.json").write_text("{}\n")
        try:
            normalize(contract, extra, root / "extra-out")
        except ValueError as exc:
            require("source census mismatch" in str(exc), "extra-file rejection failed for wrong reason")
        else:
            raise ValueError("extra source file accepted")

        bad = root / "bad-source"
        shutil.copytree(src, bad)
        value = load_json(bad / "sealed-manifest-replay.json")
        value["authority"] = "QualificationOnly"
        write_json(bad / "sealed-manifest-replay.json", value)
        try:
            normalize(contract, bad, root / "bad-out")
        except ValueError as exc:
            require("authority mismatch" in str(exc), "authority rejection failed for wrong reason")
        else:
            raise ValueError("wrong authority accepted")

    return {
        "schema": "symthaea.rel.publication-audit-normalization-self-test.v1",
        "authority": "PublicationAuditNormalizationContractOnly",
        "valid_source_accepted": True,
        "byte_preserving_rename_verified": True,
        "extra_source_file_rejected": True,
        "wrong_authority_rejected": True,
        "claims": {
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=pathlib.Path, required=True)
    p.add_argument("--source", type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--receipt", type=pathlib.Path)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if args.self_test:
        print(json.dumps(self_test(contract), indent=2, sort_keys=True))
        return
    require(args.source is not None and args.output is not None, "--source and --output are required")
    receipt = normalize(contract, args.source, args.output)
    if args.receipt is not None:
        write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
