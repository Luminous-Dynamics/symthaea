#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import test_verify_vart_world_creative_001_identity as i
import test_verify_vart_world_creative_001_n1_n20 as n
import verify_vart_world_creative_001 as core
import verify_vart_world_creative_001_calibration as calibration_verify


def augment_calibration(root: Path, trial_id: str, predicted: float, actual: float) -> None:
    trial_dir = n.tdir(root, trial_id)

    hypothesis_path = n.logical_path(root, trial_id, "revision_hypothesis")
    hypothesis = n.load(hypothesis_path)
    hypothesis["expected_effects"] = {"declared_goal_consequence": predicted}
    hypothesis_sha = n.save(hypothesis_path, hypothesis)

    outcome_path = n.logical_path(root, trial_id, "revision_outcome")
    outcome = n.load(outcome_path)
    outcome["actual_effects"] = {"declared_goal_consequence": actual}
    magnitude = abs(actual - predicted)
    outcome["prediction_error_magnitude"] = magnitude
    outcome_sha = n.save(outcome_path, outcome)

    receipt_path = n.logical_path(root, trial_id, "applied_receipt")
    receipt = n.load(receipt_path)
    receipt["revision_hypothesis_sha256"] = hypothesis_sha
    receipt_sha = n.save(receipt_path, receipt)

    calibration_receipt = {
        "schema": "symthaea.vart-world-creative-001.calibration-receipt.v1",
        "experiment_id": n.EXPERIMENT_ID,
        "trial_id": trial_id,
        "revision_hypothesis_sha256": hypothesis_sha,
        "revision_outcome_sha256": outcome_sha,
        "error_metric": "l2_over_declared_effects_v1",
        "expected_effects": {"declared_goal_consequence": predicted},
        "actual_effects": {"declared_goal_consequence": actual},
        "signed_error": {"declared_goal_consequence": actual - predicted},
        "absolute_error": {"declared_goal_consequence": abs(actual - predicted)},
        "squared_error": {"declared_goal_consequence": (actual - predicted) ** 2},
        "prediction_error_magnitude": magnitude,
    }
    calibration_sha = n.dump(trial_dir / "calibration_receipt.json", calibration_receipt)

    idx = n.load(n.index_path(root, trial_id))
    idx["files"]["calibration_receipt"] = "calibration_receipt.json"
    evidence_sha = n.save(n.index_path(root, trial_id), idx)

    n.update_manifest(
        root,
        trial_id,
        revision_hypothesis_sha256=hypothesis_sha,
        revision_outcome_sha256=outcome_sha,
        applied_receipt_sha256=receipt_sha,
        calibration_receipt_sha256=calibration_sha,
        evidence_bundle_sha256=evidence_sha,
    )


def build_calibration_bundle(root: Path) -> str:
    i.build_identity_bundle(root)
    calibration_contract_sha = n.dump(
        root / "calibration_contract.json",
        {
            "schema": "symthaea.vart-world-creative-001.calibration-contract.v1",
            "experiment_id": n.EXPERIMENT_ID,
            "error_metric": "l2_over_declared_effects_v1",
            "numeric_tolerance": 1e-12,
            "minimum_revisions_per_world_for_trend": 3,
            "pool_channels": False,
            "runtime_ledger_authoritative": False,
            "longitudinal_unit": "world_lineage_sha256",
            "paired_population_resample_unit": "world_cluster_sha256",
        },
    )

    values = {
        n.FULL: (0.10, 0.24),
        n.RANDOM: (0.10, -0.03),
        n.HEURISTIC: (0.10, 0.08),
        n.GENERALIZATION: (0.15, 0.31),
    }
    for trial_id, (predicted, actual) in values.items():
        augment_calibration(root, trial_id, predicted, actual)

    freeze = n.load(root / "confirmatory_freeze.json")
    freeze["calibration_contract_sha256"] = calibration_contract_sha
    return n.save(root / "confirmatory_freeze.json", freeze)


def expect_reject(root: Path, freeze_sha: str, expected: str) -> None:
    try:
        calibration_verify.verify_calibration_qualified(root, freeze_sha)
    except core.Reject as exc:
        assert exc.code == expected, f"expected {expected}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {expected}")


def run_suite(base: Path, freeze_sha: str) -> None:
    result = calibration_verify.verify_calibration_qualified(base, freeze_sha)
    assert result["verdict"] == "ACCEPT", result
    assert result["independent_calibration_reconstruction"] == "PASS"
    assert result["calibration_complete_trial_count"] == 4
    assert result["calibration_complete_world_cluster_count"] == 2
    assert result["calibration_trend_eligible_lineage_count"] == 0
    assert result["calibration_improvement_claim_authorized"] is False

    # K1 — runtime receipt lies about a per-channel reconstructed error.
    b = n.clone(base)
    path = n.logical_path(b, n.FULL, "calibration_receipt")
    receipt = n.load(path)
    receipt["absolute_error"]["declared_goal_consequence"] = 999.0
    sha = n.save(path, receipt)
    n.update_manifest(b, n.FULL, calibration_receipt_sha256=sha)
    expect_reject(b, freeze_sha, "CALIBRATION_RECONSTRUCTION_MISMATCH")

    # K2 — outcome exports a false scalar prediction-error magnitude.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "revision_outcome")
    outcome = n.load(path)
    outcome["prediction_error_magnitude"] = 999.0
    outcome_sha = n.save(path, outcome)
    n.update_manifest(b, n.GENERALIZATION, revision_outcome_sha256=outcome_sha)
    rpath = n.logical_path(b, n.GENERALIZATION, "calibration_receipt")
    receipt = n.load(rpath)
    receipt["revision_outcome_sha256"] = outcome_sha
    rsha = n.save(rpath, receipt)
    n.update_manifest(b, n.GENERALIZATION, calibration_receipt_sha256=rsha)
    expect_reject(b, freeze_sha, "CALIBRATION_SCALAR_MISMATCH")

    # K3 — a prospectively predicted channel disappears from actual measurements.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "revision_outcome")
    outcome = n.load(path)
    outcome["actual_effects"] = {"perceptual_consequence": 0.2}
    outcome_sha = n.save(path, outcome)
    n.update_manifest(b, n.GENERALIZATION, revision_outcome_sha256=outcome_sha)
    rpath = n.logical_path(b, n.GENERALIZATION, "calibration_receipt")
    receipt = n.load(rpath)
    receipt["revision_outcome_sha256"] = outcome_sha
    rsha = n.save(rpath, receipt)
    n.update_manifest(b, n.GENERALIZATION, calibration_receipt_sha256=rsha)
    expect_reject(b, freeze_sha, "CALIBRATION_EVIDENCE_INCOMPLETE")

    # K4 — non-finite outcome values are never admitted as calibration evidence.
    b = n.clone(base)
    path = n.logical_path(b, n.GENERALIZATION, "revision_outcome")
    outcome = n.load(path)
    outcome["actual_effects"]["declared_goal_consequence"] = math.nan
    outcome_sha = n.save(path, outcome)
    n.update_manifest(b, n.GENERALIZATION, revision_outcome_sha256=outcome_sha)
    rpath = n.logical_path(b, n.GENERALIZATION, "calibration_receipt")
    receipt = n.load(rpath)
    receipt["revision_outcome_sha256"] = outcome_sha
    rsha = n.save(rpath, receipt)
    n.update_manifest(b, n.GENERALIZATION, calibration_receipt_sha256=rsha)
    expect_reject(b, freeze_sha, "CALIBRATION_NONFINITE_VALUE")

    # K5 — calibration contract bytes change after the externally anchored freeze.
    b = n.clone(base)
    path = b / "calibration_contract.json"
    contract = n.load(path)
    contract["numeric_tolerance"] = 1e-3
    n.save(path, contract)
    expect_reject(b, freeze_sha, "CALIBRATION_CONTRACT_MISMATCH")

    # K6 — runtime scalar inside the calibration receipt disagrees with reconstruction.
    b = n.clone(base)
    path = n.logical_path(b, n.FULL, "calibration_receipt")
    receipt = n.load(path)
    receipt["prediction_error_magnitude"] = 999.0
    sha = n.save(path, receipt)
    n.update_manifest(b, n.FULL, calibration_receipt_sha256=sha)
    expect_reject(b, freeze_sha, "CALIBRATION_SCALAR_MISMATCH")

    # K7 — receipt retroactively substitutes the predicted effect while preserving
    # hypothesis/outcome artifact identity.
    b = n.clone(base)
    path = n.logical_path(b, n.FULL, "calibration_receipt")
    receipt = n.load(path)
    receipt["expected_effects"]["declared_goal_consequence"] = 0.99
    sha = n.save(path, receipt)
    n.update_manifest(b, n.FULL, calibration_receipt_sha256=sha)
    expect_reject(b, freeze_sha, "CALIBRATION_RECONSTRUCTION_MISMATCH")


with tempfile.TemporaryDirectory(prefix="vart-calibration-") as td:
    base = Path(td) / "base"
    base.mkdir()
    freeze_sha = build_calibration_bundle(base)
    run_suite(base, freeze_sha)

print("PASS: VART independent calibration reconstruction acceptance + K1-K7 deterministic rejection")
