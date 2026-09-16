#!/usr/bin/env python3
"""REL-005A ComparisonOnly adjudicator for the corrupted-anchor dispute.

This program is deliberately narrower than the full REL-005A qualification.
It consumes a sealed MeasurementOnly observation and a separately generated
PredictionOnly oracle artifact. It MUST NOT execute Symthaea production code.

Authority: ComparisonOnly (corrupted-anchor dispute only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SEAL_SCHEMA = "symthaea.rel.observation-seal.v2"
EXECUTION_SCHEMA = "symthaea.rel.execution-receipt.v2"
MANIFEST_SCHEMA = "symthaea.rel.execution-capsule-manifest.v1"
CONTEXT_SCHEMA = "symthaea.rel.execution-context.v1"
OBSERVATION_SCHEMA = "symthaea.rel.graft-frame-red-diagnostic.v1"
ORACLE_SCHEMA = "symthaea.rel.graft-frame-independent-oracle.v1"
COMPARISON_SCHEMA = "symthaea.rel.corrupted-anchor-comparison.v1"

FROZEN_SCIENTIFIC_BASE = "7f5826675b44dd0d3f702f62bc9818f7990e4a01"
ORACLE_SOURCE_COMMIT = "8898618544f167e4183ab52d787995779b90eb22"
ORACLE_GIT_BLOB = "5ed47e4fc7548061782d28f03f55a99b577ac856"
NONZERO_ABS_TOL = 5.0e-5
HISTORICAL_CORRUPTED_ANCHOR_THRESHOLD = 0.20

COMPARED_FIELDS = (
    "corrupted_anchor_held_out_max",
    "corrupted_mask_dispersion_max",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected numeric scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field}: expected finite scalar")
    return number


def self_check() -> None:
    assert FROZEN_SCIENTIFIC_BASE == "7f5826675b44dd0d3f702f62bc9818f7990e4a01"
    assert ORACLE_SOURCE_COMMIT == "8898618544f167e4183ab52d787995779b90eb22"
    assert ORACLE_GIT_BLOB == "5ed47e4fc7548061782d28f03f55a99b577ac856"
    assert NONZERO_ABS_TOL == 5.0e-5
    assert HISTORICAL_CORRUPTED_ANCHOR_THRESHOLD == 0.20
    assert COMPARED_FIELDS == (
        "corrupted_anchor_held_out_max",
        "corrupted_mask_dispersion_max",
    )


def require_identity(
    obj: dict[str, object],
    *,
    schema: str,
    authority: str,
    expected_head: str,
    label: str,
) -> None:
    if obj.get("schema") != schema:
        raise ValueError(f"{label} schema mismatch")
    if obj.get("authority") != authority:
        raise ValueError(f"{label} authority mismatch")
    if obj.get("expected_head") != expected_head:
        raise ValueError(f"{label} head mismatch")
    if obj.get("expected_base", "") != FROZEN_SCIENTIFIC_BASE:
        raise ValueError(f"{label} scientific base mismatch")


def validate_evidence_chain(
    *,
    observation: dict[str, object],
    seal: dict[str, object],
    execution: dict[str, object],
    manifest: dict[str, object],
    context: dict[str, object],
    observation_path: Path,
    execution_path: Path,
    manifest_path: Path,
    context_path: Path,
    expected_observation_head: str,
    source_run_id: str,
) -> None:
    require_identity(
        seal,
        schema=SEAL_SCHEMA,
        authority="MeasurementOnly",
        expected_head=expected_observation_head,
        label="observation seal",
    )
    require_identity(
        execution,
        schema=EXECUTION_SCHEMA,
        authority="ExecutionOnly",
        expected_head=expected_observation_head,
        label="execution receipt",
    )
    require_identity(
        manifest,
        schema=MANIFEST_SCHEMA,
        authority="ExecutionOnly",
        expected_head=expected_observation_head,
        label="capsule manifest",
    )
    require_identity(
        context,
        schema=CONTEXT_SCHEMA,
        authority="ExecutionOnly",
        expected_head=expected_observation_head,
        label="workflow context",
    )

    if str(context.get("run_id")) != source_run_id:
        raise ValueError("source run id mismatch")
    if seal.get("observation_status") != "sealed" or seal.get("reason_code") != "OK":
        raise ValueError("observation is not sealed")
    if execution.get("execution_status") != "ok" or execution.get("reason_code") != "OK":
        raise ValueError("execution receipt is not OK")
    if execution.get("measurement_exit_code") != 0:
        raise ValueError("measurement exit code is not zero")
    phase_outcomes = execution.get("phase_outcomes")
    if not isinstance(phase_outcomes, dict) or not phase_outcomes:
        raise ValueError("execution phase outcomes missing")
    if not all(value == "success" for value in phase_outcomes.values()):
        raise ValueError("execution phase outcome is not successful")

    observation_hash = sha256(observation_path)
    execution_hash = sha256(execution_path)
    manifest_hash = sha256(manifest_path)
    if seal.get("observation_sha256") != observation_hash:
        raise ValueError("sealed observation hash mismatch")
    if seal.get("execution_receipt_sha256") != execution_hash:
        raise ValueError("sealed execution receipt hash mismatch")
    if seal.get("capsule_manifest_sha256") != manifest_hash:
        raise ValueError("sealed capsule manifest hash mismatch")
    if execution.get("raw_observation_present") is not True:
        raise ValueError("execution receipt does not declare observation present")
    if execution.get("raw_observation_sha256") != observation_hash:
        raise ValueError("execution receipt observation hash mismatch")
    if execution.get("capsule_manifest_sha256") != manifest_hash:
        raise ValueError("execution receipt manifest hash mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("capsule manifest files missing")
    expected_manifest_files = {
        "workflow-context.json": context_path,
        "observation.json": observation_path,
    }
    for name, path in expected_manifest_files.items():
        meta = files.get(name)
        if not isinstance(meta, dict):
            raise ValueError(f"capsule manifest missing {name}")
        if meta.get("sha256") != sha256(path):
            raise ValueError(f"capsule manifest hash mismatch for {name}")
        if meta.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"capsule manifest byte length mismatch for {name}")

    chain = json.dumps(
        {
            "execution_receipt_sha256": execution_hash,
            "capsule_manifest_sha256": manifest_hash,
            "observation_sha256": observation_hash,
            "expected_head": expected_observation_head,
            "expected_base": FROZEN_SCIENTIFIC_BASE,
            "authority": "MeasurementOnly",
            "observation_status": "sealed",
            "reason_code": "OK",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if seal.get("chain_sha256") != sha256_bytes(chain):
        raise ValueError("observation seal chain commitment mismatch")

    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("observation schema mismatch")
    if observation.get("authority") != "MeasurementOnly":
        raise ValueError("observation authority mismatch")


def adjudicate(
    observation: dict[str, object],
    seal: dict[str, object],
    oracle: dict[str, object],
    execution: dict[str, object],
    manifest: dict[str, object],
    context: dict[str, object],
    *,
    observation_path: Path,
    seal_path: Path,
    oracle_path: Path,
    execution_path: Path,
    manifest_path: Path,
    context_path: Path,
    expected_observation_head: str,
    source_run_id: str,
) -> dict[str, object]:
    validate_evidence_chain(
        observation=observation,
        seal=seal,
        execution=execution,
        manifest=manifest,
        context=context,
        observation_path=observation_path,
        execution_path=execution_path,
        manifest_path=manifest_path,
        context_path=context_path,
        expected_observation_head=expected_observation_head,
        source_run_id=source_run_id,
    )

    if oracle.get("schema") != ORACLE_SCHEMA:
        raise ValueError("oracle schema mismatch")
    if oracle.get("authority") != "PredictionOnly":
        raise ValueError("oracle authority mismatch")

    comparisons: dict[str, object] = {}
    oracle_agreement = True
    for field in COMPARED_FIELDS:
        observed = finite_number(observation.get(field), f"observation.{field}")
        expected = finite_number(oracle.get(field), f"oracle.{field}")
        delta = abs(observed - expected)
        agrees = delta <= NONZERO_ABS_TOL
        oracle_agreement = oracle_agreement and agrees
        comparisons[field] = {
            "observed": observed,
            "oracle": expected,
            "absolute_delta": delta,
            "absolute_tolerance": NONZERO_ABS_TOL,
            "agrees": agrees,
        }

    corrupted = comparisons["corrupted_anchor_held_out_max"]
    assert isinstance(corrupted, dict)
    corrupted_observed = float(corrupted["observed"])
    historical_predicate_pass = corrupted_observed > HISTORICAL_CORRUPTED_ANCHOR_THRESHOLD

    if oracle_agreement and historical_predicate_pass:
        resolution = "RESOLVED_ORACLE_AND_HISTORICAL_PREDICATE_AGREE"
    elif oracle_agreement:
        resolution = "HISTORICAL_PREDICATE_INCONSISTENT_WITH_ORACLE"
    else:
        resolution = "ORACLE_DISAGREEMENT"

    # The production diagnostic carries copied scalar_prediction_* fields for
    # human reference. They are intentionally NOT read as comparison authority.
    copied_prediction_fields_ignored = sorted(
        key for key in observation if key.startswith("scalar_prediction_")
    )

    return {
        "schema": COMPARISON_SCHEMA,
        "authority": "ComparisonOnly",
        "scope": "REL-005A corrupted-anchor dispute only",
        "source_run_id": source_run_id,
        "expected_observation_head": expected_observation_head,
        "frozen_scientific_base": FROZEN_SCIENTIFIC_BASE,
        "execution_receipt_sha256": sha256(execution_path),
        "capsule_manifest_sha256": sha256(manifest_path),
        "workflow_context_sha256": sha256(context_path),
        "observation_seal_sha256": sha256(seal_path),
        "observation_sha256": sha256(observation_path),
        "oracle_artifact_sha256": sha256(oracle_path),
        "oracle_source_commit": ORACLE_SOURCE_COMMIT,
        "oracle_git_blob": ORACLE_GIT_BLOB,
        "frozen_rules": {
            "nonzero_absolute_tolerance": NONZERO_ABS_TOL,
            "historical_corrupted_anchor_threshold": HISTORICAL_CORRUPTED_ANCHOR_THRESHOLD,
            "historical_predicate": "corrupted_anchor_held_out_max > 0.20",
        },
        "comparisons": comparisons,
        "historical_predicate_outcome": "pass" if historical_predicate_pass else "fail",
        "oracle_agreement": oracle_agreement,
        "resolution": resolution,
        "copied_prediction_fields_ignored": copied_prediction_fields_ignored,
        "claims": {
            "dispute_resolved": oracle_agreement and historical_predicate_pass,
            "rel_005a_qualified": False,
            "full_rel_005a_scientific_pass": False,
            "full_rel_005a_scientific_fail": False,
            "production_executed_by_comparison": False,
            "physical_gauge_symmetry_established": False,
            "consciousness_claim_established": False,
        },
    }


def write_failure(
    output: Path,
    *,
    reason: str,
    source_run_id: str,
    expected_observation_head: str,
) -> None:
    receipt = {
        "schema": COMPARISON_SCHEMA,
        "authority": "ComparisonOnly",
        "scope": "REL-005A corrupted-anchor dispute only",
        "source_run_id": source_run_id,
        "expected_observation_head": expected_observation_head,
        "comparison_status": "invalid_input",
        "reason": reason,
        "claims": {
            "dispute_resolved": False,
            "rel_005a_qualified": False,
            "full_rel_005a_scientific_pass": False,
            "full_rel_005a_scientific_fail": False,
            "production_executed_by_comparison": False,
        },
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--expected-observation-head", default="")
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    self_check()
    if args.self_check:
        print("REL005A_COMPARISON_ONLY_SELF_CHECK=PASS")
        return 0

    required = {
        "--observation": args.observation,
        "--seal": args.seal,
        "--oracle": args.oracle,
        "--execution-receipt": args.execution_receipt,
        "--manifest": args.manifest,
        "--context": args.context,
        "--output": args.output,
        "--expected-observation-head": args.expected_observation_head,
        "--source-run-id": args.source_run_id,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required comparison arguments: " + ", ".join(missing))

    assert args.output is not None
    try:
        observation = load_json(args.observation)
        seal = load_json(args.seal)
        oracle = load_json(args.oracle)
        execution = load_json(args.execution_receipt)
        manifest = load_json(args.manifest)
        context = load_json(args.context)
        receipt = adjudicate(
            observation,
            seal,
            oracle,
            execution,
            manifest,
            context,
            observation_path=args.observation,
            seal_path=args.seal,
            oracle_path=args.oracle,
            execution_path=args.execution_receipt,
            manifest_path=args.manifest,
            context_path=args.context,
            expected_observation_head=args.expected_observation_head,
            source_run_id=args.source_run_id,
        )
        receipt["comparison_status"] = "completed"
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        claims = receipt["claims"]
        assert isinstance(claims, dict)
        return 0 if claims["dispute_resolved"] else 2
    except Exception as exc:
        write_failure(
            args.output,
            reason=f"{type(exc).__name__}: {exc}",
            source_run_id=args.source_run_id,
            expected_observation_head=args.expected_observation_head,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
