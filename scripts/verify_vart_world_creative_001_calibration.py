#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_qualified as qualified
import verify_vart_world_creative_001_state as state_verify

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
ERROR_METRIC = "l2_over_declared_effects_v1"


def _dict(value: Any, code: str, detail: str) -> dict[str, Any]:
    core.require(isinstance(value, dict), code, detail)
    return value


def _finite_map(value: Any, name: str) -> dict[str, float]:
    obj = _dict(value, "CALIBRATION_EVIDENCE_INCOMPLETE", name)
    out: dict[str, float] = {}
    for key, raw in obj.items():
        core.require(
            isinstance(key, str) and key and isinstance(raw, (int, float)) and not isinstance(raw, bool),
            "CALIBRATION_EVIDENCE_INCOMPLETE",
            f"{name}.{key}",
        )
        val = float(raw)
        core.require(math.isfinite(val), "CALIBRATION_NONFINITE_VALUE", f"{name}.{key}")
        out[key] = val
    core.require(bool(out), "CALIBRATION_EVIDENCE_INCOMPLETE", name)
    return out


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _compare_map(actual: Any, expected: dict[str, float], name: str, tol: float) -> None:
    got = _finite_map(actual, name)
    core.require(set(got) == set(expected), "CALIBRATION_RECONSTRUCTION_MISMATCH", name)
    for key in expected:
        core.require(
            _close(got[key], expected[key], tol),
            "CALIBRATION_RECONSTRUCTION_MISMATCH",
            f"{name}.{key}: {got[key]} != {expected[key]}",
        )


def _reconstruct(expected: dict[str, float], actual: dict[str, float]) -> dict[str, Any]:
    missing = sorted(set(expected) - set(actual))
    core.require(
        not missing,
        "CALIBRATION_EVIDENCE_INCOMPLETE",
        f"missing actual effects for predicted channels: {missing}",
    )
    signed = {key: actual[key] - expected[key] for key in sorted(expected)}
    absolute = {key: abs(signed[key]) for key in sorted(expected)}
    squared = {key: signed[key] * signed[key] for key in sorted(expected)}
    magnitude = math.sqrt(sum(squared.values()))
    return {
        "expected_effects": {key: expected[key] for key in sorted(expected)},
        "actual_effects": {key: actual[key] for key in sorted(expected)},
        "signed_error": signed,
        "absolute_error": absolute,
        "squared_error": squared,
        "prediction_error_magnitude": magnitude,
    }


def _contract(root: Path, freeze: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
    path = root / "calibration_contract.json"
    expected_sha = core.require_sha256(
        freeze.get("calibration_contract_sha256"), "freeze.calibration_contract_sha256"
    )
    actual_sha = core.sha256_file(path)
    core.require(
        actual_sha == expected_sha,
        "CALIBRATION_CONTRACT_MISMATCH",
        f"{actual_sha} != {expected_sha}",
    )
    contract = _dict(core.read_json(path), "CALIBRATION_CONTRACT_MISMATCH", "contract")
    core.require(
        contract.get("schema") == "symthaea.vart-world-creative-001.calibration-contract.v1"
        and contract.get("experiment_id") == EXPERIMENT_ID
        and contract.get("error_metric") == ERROR_METRIC,
        "CALIBRATION_CONTRACT_MISMATCH",
        "identity/error metric",
    )
    tol_raw = contract.get("numeric_tolerance")
    core.require(
        isinstance(tol_raw, (int, float)) and not isinstance(tol_raw, bool),
        "CALIBRATION_CONTRACT_MISMATCH",
        "numeric_tolerance",
    )
    tol = float(tol_raw)
    core.require(math.isfinite(tol) and 0.0 <= tol <= 1e-6, "CALIBRATION_CONTRACT_MISMATCH", "numeric_tolerance")
    minimum = contract.get("minimum_revisions_per_world_for_trend")
    core.require(
        isinstance(minimum, int) and minimum >= 2,
        "CALIBRATION_CONTRACT_MISMATCH",
        "minimum_revisions_per_world_for_trend",
    )
    return contract, tol, minimum


def _trial_calibration(
    trial_dir: Path,
    manifest: dict[str, Any],
    tol: float,
) -> dict[str, Any] | None:
    if manifest.get("trial_state") != "complete":
        return None
    trial_id = manifest["trial_id"]
    idx = _dict(
        core.read_json(trial_dir / "evidence_index.json"),
        "CALIBRATION_EVIDENCE_INCOMPLETE",
        trial_id,
    )
    hypothesis = _dict(
        core.read_json(core.file_from_index(trial_dir, idx, "revision_hypothesis")),
        "CALIBRATION_EVIDENCE_INCOMPLETE",
        f"{trial_id}: hypothesis",
    )
    outcome = _dict(
        core.read_json(core.file_from_index(trial_dir, idx, "revision_outcome")),
        "CALIBRATION_EVIDENCE_INCOMPLETE",
        f"{trial_id}: outcome",
    )
    expected = _finite_map(hypothesis.get("expected_effects"), f"{trial_id}.expected_effects")
    actual_all = _finite_map(outcome.get("actual_effects"), f"{trial_id}.actual_effects")
    reconstruction = _reconstruct(expected, actual_all)

    receipt_sha = core.require_sha256(
        manifest.get("calibration_receipt_sha256"), "calibration_receipt_sha256"
    )
    receipt_path = core.file_from_index(trial_dir, idx, "calibration_receipt")
    core.require(
        core.sha256_file(receipt_path) == receipt_sha,
        "CALIBRATION_RECONSTRUCTION_MISMATCH",
        f"{trial_id}: calibration receipt digest",
    )
    receipt = _dict(
        core.read_json(receipt_path),
        "CALIBRATION_RECONSTRUCTION_MISMATCH",
        f"{trial_id}: calibration receipt",
    )
    core.require(
        receipt.get("schema") == "symthaea.vart-world-creative-001.calibration-receipt.v1"
        and receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("trial_id") == trial_id
        and receipt.get("revision_hypothesis_sha256") == manifest["revision_hypothesis_sha256"]
        and receipt.get("revision_outcome_sha256") == manifest["revision_outcome_sha256"]
        and receipt.get("error_metric") == ERROR_METRIC,
        "CALIBRATION_RECONSTRUCTION_MISMATCH",
        f"{trial_id}: receipt identity",
    )

    _compare_map(receipt.get("expected_effects"), reconstruction["expected_effects"], f"{trial_id}.receipt.expected", tol)
    _compare_map(receipt.get("actual_effects"), reconstruction["actual_effects"], f"{trial_id}.receipt.actual", tol)
    _compare_map(receipt.get("signed_error"), reconstruction["signed_error"], f"{trial_id}.receipt.signed", tol)
    _compare_map(receipt.get("absolute_error"), reconstruction["absolute_error"], f"{trial_id}.receipt.absolute", tol)
    _compare_map(receipt.get("squared_error"), reconstruction["squared_error"], f"{trial_id}.receipt.squared", tol)

    receipt_magnitude = receipt.get("prediction_error_magnitude")
    core.require(
        isinstance(receipt_magnitude, (int, float))
        and not isinstance(receipt_magnitude, bool)
        and math.isfinite(float(receipt_magnitude))
        and _close(float(receipt_magnitude), reconstruction["prediction_error_magnitude"], tol),
        "CALIBRATION_SCALAR_MISMATCH",
        trial_id,
    )
    if "prediction_error_magnitude" in outcome:
        raw = outcome["prediction_error_magnitude"]
        core.require(
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
            and _close(float(raw), reconstruction["prediction_error_magnitude"], tol),
            "CALIBRATION_SCALAR_MISMATCH",
            f"{trial_id}: outcome scalar",
        )

    return reconstruction


def verify_calibration_qualified(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    result = state_verify.verify_state_qualified(root, expected_freeze_sha256)
    freeze = qualified.preflight_freeze(root, expected_freeze_sha256)
    _, tol, minimum_revisions = _contract(root, freeze)
    inventory = _dict(core.read_json(root / "trial_inventory.json"), "PREREGISTRATION_INVALID", "inventory")
    trial_ids = inventory.get("trial_ids")
    core.require(isinstance(trial_ids, list), "PREREGISTRATION_INVALID", "trial_ids")

    reconstructed: dict[str, Any] = {}
    chains: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for trial_id in trial_ids:
        trial_dir = root / "trials" / trial_id
        manifest = _dict(core.read_json(trial_dir / "manifest.json"), "CALIBRATION_EVIDENCE_INCOMPLETE", trial_id)
        value = _trial_calibration(trial_dir, manifest, tol)
        if value is None:
            continue
        reconstructed[trial_id] = value
        key = (manifest["policy"], manifest["world_fixture_sha256"], manifest["seed"])
        chains.setdefault(key, []).append((manifest["revision_index"], value))

    canonical = json.dumps(reconstructed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    reconstruction_sha = hashlib.sha256(canonical).hexdigest()
    trend_eligible = sum(1 for chain in chains.values() if len(chain) >= minimum_revisions)

    out = dict(result)
    out.update(
        {
            "independent_calibration_reconstruction": "PASS",
            "calibration_reconstruction_sha256": reconstruction_sha,
            "calibration_complete_trial_count": len(reconstructed),
            "calibration_trend_eligible_world_count": trend_eligible,
            "calibration_improvement_claim_authorized": False,
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent calibration reconstructor for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_calibration_qualified(args.root, args.expected_freeze_sha256)
    except core.Reject as exc:
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
            f"ACCEPT: {result['calibration_complete_trial_count']} calibration trials; "
            "independent reconstruction PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
