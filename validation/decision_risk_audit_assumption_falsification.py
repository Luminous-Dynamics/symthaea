#!/usr/bin/env python3
"""Falsify decision-risk certificates under audit bias and incomplete support.

Synthetic contract benchmark. A confidence bound is only as good as its sampling
assumptions and audited support. This module does not infer independence from data;
it requires a separately verified audit-assumption receipt.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

SCHEMA = "symthaea-decision-risk-audit-assumption-falsification-v1"
SEED = 20261028
RADIUS_BITS = 0.06664248830771191
REGRET_BUDGET_BITS = 0.20
DELTA = 0.05
LOSS_MIN = 0.0
LOSS_MAX = 4.0
TOL = 1e-12

class AuditEvidenceError(ValueError):
    """Raised when audit evidence is incomplete or its assumptions are unverified."""


def _sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def empirical_bernstein_epsilon_upper(
    planning_matrix: np.ndarray,
    cell_observations: Mapping[str, Sequence[float]],
    policy_ids: Sequence[str],
    world_ids: Sequence[str],
    *,
    delta: float = DELTA,
    loss_min: float = LOSS_MIN,
    loss_max: float = LOSS_MAX,
) -> dict:
    """Compute a simultaneous empirical-Bernstein max-cell error envelope.

    The result is a valid confidence bound only when the caller has independently
    established bounded, independent audit units and predeclared support.
    """
    policies, worlds = tuple(policy_ids), tuple(world_ids)
    model = np.asarray(planning_matrix, dtype=float)
    if len(set(policies)) != len(policies) or len(set(worlds)) != len(worlds):
        raise AuditEvidenceError("duplicate policy/world identifiers")
    if model.shape != (len(policies), len(worlds)) or np.any(~np.isfinite(model)):
        raise AuditEvidenceError("planning matrix shape or values do not match declared IDs")
    if not (0 < delta < 1) or not (loss_max > loss_min):
        raise AuditEvidenceError("invalid confidence or loss-range parameters")
    expected = {f"{p}::{w}" for p in policies for w in worlds}
    actual = set(cell_observations)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise AuditEvidenceError(f"audit support mismatch; missing={missing}; extra={extra}")

    samples = []
    for p in policies:
        row = []
        for w in worlds:
            values = np.asarray(cell_observations[f"{p}::{w}"], dtype=float)
            if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
                raise AuditEvidenceError(f"invalid or insufficient audit samples for {p}::{w}")
            if np.any(values < loss_min) or np.any(values > loss_max):
                raise AuditEvidenceError(f"audit values outside predeclared range for {p}::{w}")
            row.append(values)
        samples.append(row)
    sizes = {values.size for row in samples for values in row}
    if len(sizes) != 1:
        raise AuditEvidenceError("unequal per-cell sample counts are not supported by this benchmark")
    n = sizes.pop()
    matrix = np.asarray([[float(np.mean(v)) for v in row] for row in samples])
    variances = np.asarray([[float(np.var(v, ddof=1)) for v in row] for row in samples])
    k = len(policies) * len(worlds)
    log_term = math.log(4.0 * k / delta)
    span = loss_max - loss_min
    radii = np.sqrt(2.0 * variances * log_term / n) + (7.0 * span * log_term) / (3.0 * (n - 1))
    cell_upper = np.abs(model - matrix) + radii
    epsilon_upper = float(np.max(cell_upper))
    return {
        "epsilon_upper_bits": epsilon_upper,
        "max_cell_plugin_residual_bits": float(np.max(np.abs(model - matrix))),
        "max_cell_confidence_radius_bits": float(np.max(radii)),
        "n_per_cell": n,
        "cells": k,
        "familywise_delta": delta,
        "certificate_regret_bound_bits": 2.0 * (RADIUS_BITS + epsilon_upper),
        "admitted_by_numeric_bound": bool(2.0 * (RADIUS_BITS + epsilon_upper) <= REGRET_BUDGET_BITS),
    }


def fail_closed_preflight(
    *, policy_ids: Sequence[str], world_ids: Sequence[str],
    cell_observations: Mapping[str, Sequence[float]],
    model_manifest_sha256: str, audit_target_manifest_sha256: str,
    audit_protocol_sha256: str, assumption_receipt_verified: bool,
) -> dict:
    """Require complete support and an independently verified assumption receipt.

    assumption_receipt_verified must come from an upstream trusted verifier; a
    boolean supplied by the caller alone is not cryptographic authentication.
    """
    if not _sha256_hex(model_manifest_sha256) or not _sha256_hex(audit_target_manifest_sha256) or not _sha256_hex(audit_protocol_sha256):
        return {"admitted": False, "reason": "INVALID_OR_MISSING_DIGEST"}
    if model_manifest_sha256 != audit_target_manifest_sha256:
        return {"admitted": False, "reason": "AUDIT_TARGET_MANIFEST_MISMATCH"}
    expected = {f"{p}::{w}" for p in policy_ids for w in world_ids}
    actual = set(cell_observations)
    if actual != expected:
        return {"admitted": False, "reason": "AUDIT_SUPPORT_INCOMPLETE", "missing_cells": sorted(expected-actual), "unexpected_cells": sorted(actual-expected)}
    if not assumption_receipt_verified:
        return {"admitted": False, "reason": "AUDIT_ASSUMPTIONS_UNVERIFIED"}
    return {"admitted": True, "reason": "PREFLIGHT_PASSED", "cell_count": len(expected)}


def _observations_from_matrix(matrix: np.ndarray, n: int) -> dict[str, list[float]]:
    return {f"p{i}::w{j}": [float(matrix[i, j])] * n for i in range(matrix.shape[0]) for j in range(matrix.shape[1])}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    policies, worlds = ("p0", "p1"), ("w0", "w1")
    truth = np.array([[1.0, 1.2], [1.5, 2.0]], dtype=float)
    model_clean = truth.copy()
    model_corrupt = truth.copy()
    model_corrupt[0, 0] += 0.04
    epsilon_allowance = REGRET_BUDGET_BITS / 2.0 - RADIUS_BITS
    if np.max(model_corrupt) > LOSS_MAX:
        raise AssertionError("invalid stress matrix")
    n = 4096

    # Positive control: independent, low-noise, unbiased audits.
    iid = {
        f"p{i}::w{j}": (truth[i, j] + rng.uniform(-0.05, 0.05, size=n)).tolist()
        for i in range(2) for j in range(2)
    }
    iid_clean = empirical_bernstein_epsilon_upper(model_clean, iid, policies, worlds)

    # Negative control: a cell-specific shared systematic audit bias masks the
    # same 0.04-bit planning-matrix error. Observations are correlated and biased;
    # the iid formula still produces a numeric admission if assumptions are lied about.
    biased = _observations_from_matrix(model_corrupt, n)
    legacy_biased = empirical_bernstein_epsilon_upper(model_corrupt, biased, policies, worlds)
    true_model_error = float(np.max(np.abs(model_corrupt - truth)))
    blindly_trusted_receipt = "a" * 64
    matched_manifest = "b" * 64
    fail_closed_biased = fail_closed_preflight(
        policy_ids=policies, world_ids=worlds, cell_observations=biased,
        model_manifest_sha256=matched_manifest, audit_target_manifest_sha256=matched_manifest,
        audit_protocol_sha256=blindly_trusted_receipt, assumption_receipt_verified=False,
    )

    # Incomplete support: omit the one cell where the planning matrix differs.
    omitted = {k: v for k, v in biased.items() if k != "p0::w0"}
    # Deliberately compute the unsafe naive bound on only the observed cells, to
    # quantify the failure that the preflight guard must prevent.
    observed_ids = [(0, 1), (1, 0), (1, 1)]
    observed_residuals = [abs(model_corrupt[i, j] - np.mean(omitted[f"p{i}::w{j}"])) for i, j in observed_ids]
    k_obs = len(observed_ids)
    radius_missing = (7.0 * (LOSS_MAX-LOSS_MIN) * math.log(4.0*k_obs/DELTA)) / (3.0*(n-1))
    missing_naive_bound = 2.0 * (RADIUS_BITS + max(observed_residuals) + radius_missing)
    fail_closed_missing = fail_closed_preflight(
        policy_ids=policies, world_ids=worlds, cell_observations=omitted,
        model_manifest_sha256=matched_manifest, audit_target_manifest_sha256=matched_manifest,
        audit_protocol_sha256=blindly_trusted_receipt, assumption_receipt_verified=True,
    )

    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_ASSUMPTION_FALSIFICATION_NOT_DEPLOYMENT_CERTIFICATION",
        "protocol": {
            "seed": SEED, "observations_per_cell": n, "policies": list(policies), "worlds": list(worlds),
            "truth_risk_matrix": truth.tolist(), "declared_loss_range": [LOSS_MIN, LOSS_MAX],
            "decision_radius_bits": RADIUS_BITS, "regret_budget_bits": REGRET_BUDGET_BITS,
            "maximum_admissible_epsilon_bits": epsilon_allowance,
            "systematic_bias_control": "planning model has +0.04 error at p0::w0; audit values share the same +0.04 offset at that cell and repeat identically for all observations",
        },
        "results": {
            "iid_low_noise_positive_control": {**iid_clean, "true_planning_error_bits": 0.0, "preflight": fail_closed_preflight(
                policy_ids=policies, world_ids=worlds, cell_observations=iid,
                model_manifest_sha256=matched_manifest, audit_target_manifest_sha256=matched_manifest,
                audit_protocol_sha256=blindly_trusted_receipt, assumption_receipt_verified=True)},
            "systematic_bias_adversarial_control": {
                **legacy_biased,
                "true_planning_error_bits_scoring_only": true_model_error,
                "epsilon_allowance_bits": epsilon_allowance,
                "unsafe_admission": bool(legacy_biased["admitted_by_numeric_bound"] and true_model_error > epsilon_allowance),
                "fail_closed_preflight": fail_closed_biased,
            },
            "missing_support_control": {
                "observed_cell_count": len(omitted), "expected_cell_count": len(policies)*len(worlds),
                "naive_incomplete_support_epsilon_bound_bits": float(missing_naive_bound),
                "naive_incomplete_support_admitted": bool(missing_naive_bound <= REGRET_BUDGET_BITS),
                "true_planning_error_bits_scoring_only": true_model_error,
                "fail_closed_preflight": fail_closed_missing,
            },
        },
        "limitations": [
            "four-cell synthetic demonstration only; not an empirical estimate of real-world failure prevalence",
            "iid confidence bound is valid only when its bounded independent sampling assumptions and fixed support are justified",
            "sample values alone cannot prove independence or rule out common-mode audit bias; preflight depends on an independently trusted assumption receipt",
            "latent-world cells may not be directly observable; missing support must remain uncertified",
            "no external method baselines or real-world dataset executed; no SOTA claim",
            "contract helper is not integrated into the production Rust optimizer/dispatch path",
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

if __name__ == "__main__":
    payload = run()
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
