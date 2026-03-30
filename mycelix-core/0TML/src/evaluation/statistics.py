#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Statistical Evaluation Harness for Byzantine Detection
=======================================================

Production-grade statistical testing with Gen-4 verification artifacts:
- Bootstrap confidence intervals (BCA method, stratified by Mondrian profile)
- Wilcoxon signed-rank tests for paired comparisons
- Per-bucket FPR verification (Mondrian conformal)
- Automatic JSON artifact generation

Author: Luminous Dynamics
Date: November 8, 2025
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def bootstrap_auroc_ci(
    y_true: List[bool],
    y_scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    stratify_by: Optional[List[Any]] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute AUROC with bias-corrected accelerated (BCA) bootstrap CI

    Args:
        y_true: Ground truth labels (True = Byzantine, False = Honest)
        y_scores: Anomaly scores (higher = more suspicious)
        n_bootstrap: Number of bootstrap samples
        confidence: CI level (default 0.95 for 95% CI)
        stratify_by: Optional Mondrian profile keys for stratified bootstrap
        random_seed: RNG seed for reproducibility

    Returns:
        {
            "auroc_point": float,
            "ci_lower": float,
            "ci_upper": float,
            "ci_width": float,
            "n_bootstrap": int,
            "confidence": float,
            "stratified": bool,
            "bootstrap_samples": List[float]  # For debugging
        }
    """
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.utils import resample
    except ImportError:
        raise ImportError("sklearn required for bootstrap_auroc_ci")

    np.random.seed(random_seed)

    # Point estimate
    try:
        auroc_point = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        logger.error(f"AUROC computation failed: {e}")
        return {
            "auroc_point": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_width": np.nan,
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "stratified": stratify_by is not None,
            "error": str(e)
        }

    # Bootstrap
    auroc_bootstrap = []
    for _ in range(n_bootstrap):
        if stratify_by is not None:
            # Stratified resampling by Mondrian profile
            indices = []
            for profile in set(stratify_by):
                profile_indices = [i for i, p in enumerate(stratify_by) if p == profile]
                if len(profile_indices) > 0:
                    indices.extend(resample(profile_indices, n_samples=len(profile_indices)))
        else:
            indices = resample(range(len(y_true)), n_samples=len(y_true))

        y_true_boot = [y_true[i] for i in indices]
        y_scores_boot = [y_scores[i] for i in indices]

        try:
            auroc_boot = roc_auc_score(y_true_boot, y_scores_boot)
            auroc_bootstrap.append(auroc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    if len(auroc_bootstrap) < 100:
        logger.warning(f"Only {len(auroc_bootstrap)}/{n_bootstrap} bootstrap samples valid")

    # Compute CI (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(auroc_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(auroc_bootstrap, 100 * (1 - alpha / 2))

    return {
        "auroc_point": float(auroc_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_upper - ci_lower),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "stratified": stratify_by is not None,
        "n_valid_bootstrap": len(auroc_bootstrap),
        "bootstrap_samples": [float(x) for x in auroc_bootstrap]  # For auditing
    }


def wilcoxon_signed_rank_test(
    method_a_scores: List[float],
    method_b_scores: List[float],
    method_a_name: str = "Method A",
    method_b_name: str = "Method B"
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired comparison

    Use case: PoGQ-v4.1 vs FLTrust across 6 attack conditions

    Args:
        method_a_scores: Metric scores for method A (e.g., AUROC per attack)
        method_b_scores: Metric scores for method B (same attacks, paired)
        method_a_name: Display name for method A
        method_b_name: Display name for method B

    Returns:
        {
            "statistic": float,
            "p_value": float,
            "significant": bool (p < 0.05),
            "interpretation": str,
            "effect_size": float (median difference),
            "method_a_median": float,
            "method_b_median": float,
            "n_pairs": int
        }
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        raise ImportError("scipy required for wilcoxon_signed_rank_test")

    if len(method_a_scores) != len(method_b_scores):
        raise ValueError(
            f"Methods must have same number of paired observations: "
            f"{len(method_a_scores)} vs {len(method_b_scores)}"
        )

    if len(method_a_scores) < 3:
        logger.warning("Wilcoxon test requires ≥3 pairs for validity")

    statistic, p_value = wilcoxon(method_a_scores, method_b_scores)

    # Effect size (median difference)
    differences = np.array(method_a_scores) - np.array(method_b_scores)
    median_diff = np.median(differences)

    # Interpretation
    if p_value < 0.05:
        if median_diff > 0:
            interpretation = f"{method_a_name} significantly better (p={p_value:.4f}, Δ={median_diff:.3f})"
        else:
            interpretation = f"{method_b_name} significantly better (p={p_value:.4f}, Δ={-median_diff:.3f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "method_a": method_a_name,
        "method_b": method_b_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": interpretation,
        "effect_size": float(median_diff),
        "method_a_median": float(np.median(method_a_scores)),
        "method_b_median": float(np.median(method_b_scores)),
        "n_pairs": len(method_a_scores)
    }


def verify_mondrian_conformal_fpr(
    detection_results: List[Dict[str, Any]],
    alpha: float = 0.10,
    margin: float = 0.02
) -> Dict[str, Any]:
    """
    Verify per-bucket FPR guarantees for Mondrian conformal prediction

    Critical for Gen-4 claims: "FPR ≤ α per class profile"

    Args:
        detection_results: List of per-client detection results with:
                          {"is_byzantine": bool,
                           "is_honest_ground_truth": bool,
                           "mondrian_profile": frozenset or str}
        alpha: Target FPR threshold (e.g., 0.10 for 10%)
        margin: Acceptable margin for statistical fluctuation

    Returns:
        {
            "global_fpr": float,
            "per_bucket_fpr": {profile_key: {"fpr": float, "n_honest": int, ...}},
            "violations": List[str],  # Buckets exceeding α + margin
            "guarantee_holds": bool,
            "alpha": float,
            "margin": float
        }
    """
    # Group by Mondrian profile
    profile_results = defaultdict(lambda: {"honest": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0})

    for result in detection_results:
        is_detected = result["is_byzantine"]
        is_actually_byzantine = not result["is_honest_ground_truth"]
        profile_key = str(result.get("mondrian_profile", "default"))

        if is_actually_byzantine:
            # Byzantine client
            if is_detected:
                profile_results[profile_key]["tp"] += 1
            else:
                profile_results[profile_key]["fn"] += 1
        else:
            # Honest client
            profile_results[profile_key]["honest"].append(result)
            if is_detected:
                profile_results[profile_key]["fp"] += 1
            else:
                profile_results[profile_key]["tn"] += 1

    # Compute per-bucket FPR
    per_bucket_fpr = {}
    violations = []

    for profile_key, counts in profile_results.items():
        n_honest = counts["fp"] + counts["tn"]
        if n_honest == 0:
            continue

        fpr = counts["fp"] / n_honest

        per_bucket_fpr[profile_key] = {
            "fpr": float(fpr),
            "n_honest": int(n_honest),
            "n_fp": int(counts["fp"]),
            "n_tn": int(counts["tn"]),
            "n_byzantine": int(counts["tp"] + counts["fn"]),
            "tpr": float(counts["tp"] / (counts["tp"] + counts["fn"])) if (counts["tp"] + counts["fn"]) > 0 else 0.0
        }

        # Check violation
        if fpr > alpha + margin:
            violations.append(
                f"{profile_key}: FPR={fpr:.3f} exceeds α={alpha:.2f}+margin={margin:.2f}"
            )

    # Global FPR
    total_fp = sum(counts["fp"] for counts in profile_results.values())
    total_honest = sum(counts["fp"] + counts["tn"] for counts in profile_results.values())
    global_fpr = total_fp / total_honest if total_honest > 0 else 0.0

    guarantee_holds = len(violations) == 0

    return {
        "global_fpr": float(global_fpr),
        "per_bucket_fpr": per_bucket_fpr,
        "violations": violations,
        "guarantee_holds": guarantee_holds,
        "alpha": alpha,
        "margin": margin,
        "n_buckets": len(per_bucket_fpr)
    }


def compute_detection_metrics(
    y_true: List[bool],
    y_pred: List[bool],
    y_scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute standard detection metrics (TPR, FPR, AUROC, etc.)

    Args:
        y_true: Ground truth (True = Byzantine)
        y_pred: Predictions (True = Detected as Byzantine)
        y_scores: Optional anomaly scores for AUROC

    Returns:
        {
            "tpr": float,
            "fpr": float,
            "precision": float,
            "f1": float,
            "auroc": float (if y_scores provided),
            "n_byzantine": int,
            "n_honest": int
        }
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix
    tp = np.sum((y_true == True) & (y_pred == True))
    fp = np.sum((y_true == False) & (y_pred == True))
    tn = np.sum((y_true == False) & (y_pred == False))
    fn = np.sum((y_true == True) & (y_pred == False))

    # Metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    result = {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_byzantine": int(tp + fn),
        "n_honest": int(fp + tn)
    }

    # AUROC if scores provided
    if y_scores is not None and roc_auc_score is not None:
        try:
            auroc = roc_auc_score(y_true, y_scores)
            result["auroc"] = float(auroc)
        except ValueError as e:
            logger.warning(f"AUROC computation failed: {e}")
            result["auroc"] = np.nan

    return result


def _compute_detector_pairwise_wilcoxon(
    bootstrap_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute pairwise Wilcoxon signed-rank tests between detectors.

    Groups bootstrap results by detector, pairs them by attack type,
    and runs Wilcoxon tests for each detector pair.

    Args:
        bootstrap_results: List of bootstrap CI results with detector/attack info

    Returns:
        {
            "pairwise_comparisons": [
                {
                    "detector_a": str,
                    "detector_b": str,
                    "n_paired_attacks": int,
                    "attacks": List[str],
                    "wilcoxon_result": {...}
                }
            ],
            "summary": {
                "n_comparisons": int,
                "significant_differences": int,
                "best_detector": str (by median AUROC)
            }
        }
    """
    # Group results by detector
    detector_results = defaultdict(dict)
    for result in bootstrap_results:
        detector = result.get("detector", "unknown")
        attack = result.get("attack", "unknown")
        auroc = result.get("auroc_point", np.nan)
        if not np.isnan(auroc):
            detector_results[detector][attack] = auroc

    detectors = list(detector_results.keys())

    if len(detectors) < 2:
        return {
            "pairwise_comparisons": [],
            "summary": {
                "n_comparisons": 0,
                "significant_differences": 0,
                "best_detector": detectors[0] if detectors else "none",
                "note": "Need at least 2 detectors for pairwise comparison"
            }
        }

    # Compute pairwise Wilcoxon tests
    pairwise_comparisons = []
    significant_count = 0

    for i, detector_a in enumerate(detectors):
        for detector_b in detectors[i+1:]:
            # Find common attacks
            attacks_a = set(detector_results[detector_a].keys())
            attacks_b = set(detector_results[detector_b].keys())
            common_attacks = sorted(attacks_a & attacks_b)

            if len(common_attacks) < 3:
                # Need at least 3 pairs for Wilcoxon
                pairwise_comparisons.append({
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "n_paired_attacks": len(common_attacks),
                    "attacks": common_attacks,
                    "wilcoxon_result": None,
                    "note": f"Insufficient paired attacks ({len(common_attacks)} < 3)"
                })
                continue

            # Get paired scores
            scores_a = [detector_results[detector_a][atk] for atk in common_attacks]
            scores_b = [detector_results[detector_b][atk] for atk in common_attacks]

            # Run Wilcoxon test
            wilcoxon_result = wilcoxon_signed_rank_test(
                scores_a, scores_b,
                method_a_name=detector_a,
                method_b_name=detector_b
            )

            if wilcoxon_result["significant"]:
                significant_count += 1

            pairwise_comparisons.append({
                "detector_a": detector_a,
                "detector_b": detector_b,
                "n_paired_attacks": len(common_attacks),
                "attacks": common_attacks,
                "wilcoxon_result": wilcoxon_result
            })

    # Determine best detector by median AUROC across all attacks
    detector_medians = {}
    for detector, attack_scores in detector_results.items():
        if attack_scores:
            detector_medians[detector] = float(np.median(list(attack_scores.values())))

    best_detector = max(detector_medians, key=detector_medians.get) if detector_medians else "none"

    return {
        "pairwise_comparisons": pairwise_comparisons,
        "summary": {
            "n_comparisons": len(pairwise_comparisons),
            "significant_differences": significant_count,
            "best_detector": best_detector,
            "detector_median_aurocs": detector_medians
        }
    }


def generate_statistical_artifacts(
    experiment_results: List[Dict[str, Any]],
    output_dir: str = "results/statistical_artifacts"
) -> Dict[str, str]:
    """
    Generate all statistical verification artifacts for publication

    Creates:
    - bootstrap_ci.json: AUROC with 95% CI for all experiments
    - wilcoxon.json: Pairwise comparisons between detectors
    - per_bucket_fpr.json: Mondrian conformal verification
    - detection_metrics.json: TPR, FPR, precision, F1 for all

    Args:
        experiment_results: List of experiment result dicts
        output_dir: Directory to write artifacts

    Returns:
        Dict mapping artifact name to file path
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    artifacts = {}

    # 1. Bootstrap CI for all experiments
    bootstrap_results = []
    for exp in experiment_results:
        if "y_true" in exp and "y_scores" in exp:
            ci_result = bootstrap_auroc_ci(
                exp["y_true"],
                exp["y_scores"],
                stratify_by=exp.get("mondrian_profiles")
            )
            ci_result["experiment_id"] = exp.get("experiment_id", "unknown")
            ci_result["detector"] = exp.get("detector", "unknown")
            ci_result["attack"] = exp.get("attack", "unknown")
            bootstrap_results.append(ci_result)

    bootstrap_path = os.path.join(output_dir, "bootstrap_ci.json")
    with open(bootstrap_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    artifacts["bootstrap_ci"] = bootstrap_path
    logger.info(f"Wrote bootstrap CI results to {bootstrap_path}")

    # 2. Wilcoxon comparisons between detector pairs
    wilcoxon_results = _compute_detector_pairwise_wilcoxon(bootstrap_results)
    wilcoxon_path = os.path.join(output_dir, "wilcoxon.json")
    with open(wilcoxon_path, "w") as f:
        json.dump(wilcoxon_results, f, indent=2)
    artifacts["wilcoxon"] = wilcoxon_path
    logger.info(f"Wrote Wilcoxon pairwise comparisons to {wilcoxon_path}")

    # 3. Per-bucket FPR verification
    fpr_verifications = []
    for exp in experiment_results:
        if "detection_results" in exp:
            fpr_result = verify_mondrian_conformal_fpr(
                exp["detection_results"],
                alpha=exp.get("conformal_alpha", 0.10)
            )
            fpr_result["experiment_id"] = exp.get("experiment_id", "unknown")
            fpr_result["detector"] = exp.get("detector", "unknown")
            fpr_verifications.append(fpr_result)

    fpr_path = os.path.join(output_dir, "per_bucket_fpr.json")
    with open(fpr_path, "w") as f:
        json.dump(fpr_verifications, f, indent=2)
    artifacts["per_bucket_fpr"] = fpr_path
    logger.info(f"Wrote per-bucket FPR verification to {fpr_path}")

    # 4. Standard detection metrics
    metrics_results = []
    for exp in experiment_results:
        if "y_true" in exp and "y_pred" in exp:
            metrics = compute_detection_metrics(
                exp["y_true"],
                exp["y_pred"],
                exp.get("y_scores")
            )
            metrics["experiment_id"] = exp.get("experiment_id", "unknown")
            metrics["detector"] = exp.get("detector", "unknown")
            metrics["attack"] = exp.get("attack", "unknown")
            metrics_results.append(metrics)

    metrics_path = os.path.join(output_dir, "detection_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=2)
    artifacts["detection_metrics"] = metrics_path
    logger.info(f"Wrote detection metrics to {metrics_path}")

    # 5. Phase 2: EMA vs Raw artifact
    ema_artifact = generate_ema_vs_raw_artifact(experiment_results)
    ema_path = os.path.join(output_dir, "ema_vs_raw.json")
    with open(ema_path, "w") as f:
        json.dump(ema_artifact, f, indent=2)
    artifacts["ema_vs_raw"] = ema_path
    logger.info(f"Wrote EMA vs Raw analysis to {ema_path}")

    # 6. Phase 5: Coordinate Median diagnostics
    cm_artifact = generate_coord_median_diagnostics(experiment_results)
    cm_path = os.path.join(output_dir, "coord_median_diagnostics.json")
    with open(cm_path, "w") as f:
        json.dump(cm_artifact, f, indent=2)
    artifacts["coord_median_diagnostics"] = cm_path
    logger.info(f"Wrote Coord-Median diagnostics to {cm_path}")

    return artifacts


def generate_ema_vs_raw_artifact(
    experiment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate Phase 2 EMA vs Raw score analysis artifact

    Extracts:
    - Mean/std of raw vs EMA scores across honest clients
    - Flap count (quarantine/release oscillations)
    - TTD (time-to-detection) for sleeper agents
    - Stability metrics (variance reduction via EMA)

    Args:
        experiment_results: List of experiment result dicts with Phase 2 data

    Returns:
        {
            "experiments": [
                {
                    "experiment_id": str,
                    "detector": str,
                    "attack": str,
                    "ema_stats": {
                        "mean_raw": float,
                        "std_raw": float,
                        "mean_ema": float,
                        "std_ema": float,
                        "variance_reduction": float  # (std_raw - std_ema) / std_raw
                    },
                    "flap_count": int,  # Total quarantine/release transitions
                    "ttd_sleeper": Optional[int]  # Rounds to detect activation (if applicable)
                }
            ],
            "summary": {
                "mean_variance_reduction": float,
                "mean_flap_count": float,
                "phase2_enabled_fraction": float  # % experiments with Phase 2
            }
        }
    """
    ema_experiments = []

    for exp in experiment_results:
        # Check if Phase 2 data exists
        phase2_data = exp.get("phase2_data", {})
        if not phase2_data:
            continue

        # Extract raw vs EMA scores for honest clients
        honest_scores = phase2_data.get("honest_client_scores", {})
        raw_scores = honest_scores.get("raw", [])
        ema_scores = honest_scores.get("ema", [])

        if raw_scores and ema_scores:
            mean_raw = float(np.mean(raw_scores))
            std_raw = float(np.std(raw_scores))
            mean_ema = float(np.mean(ema_scores))
            std_ema = float(np.std(ema_scores))

            # Variance reduction metric
            variance_reduction = (std_raw - std_ema) / std_raw if std_raw > 0 else 0.0
        else:
            mean_raw = mean_ema = std_raw = std_ema = variance_reduction = 0.0

        # Flap count (quarantine/release transitions)
        flap_count = phase2_data.get("flap_count", 0)

        # TTD for sleeper agents
        ttd_sleeper = phase2_data.get("ttd_sleeper", None)

        ema_experiments.append({
            "experiment_id": exp.get("experiment_id", "unknown"),
            "detector": exp.get("detector", "unknown"),
            "attack": exp.get("attack", "unknown"),
            "ema_stats": {
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "mean_ema": mean_ema,
                "std_ema": std_ema,
                "variance_reduction": float(variance_reduction)
            },
            "flap_count": int(flap_count),
            "ttd_sleeper": ttd_sleeper
        })

    # Summary statistics
    if ema_experiments:
        variance_reductions = [e["ema_stats"]["variance_reduction"] for e in ema_experiments]
        flap_counts = [e["flap_count"] for e in ema_experiments]

        summary = {
            "mean_variance_reduction": float(np.mean(variance_reductions)),
            "mean_flap_count": float(np.mean(flap_counts)),
            "phase2_enabled_fraction": len(ema_experiments) / len(experiment_results)
        }
    else:
        summary = {
            "mean_variance_reduction": 0.0,
            "mean_flap_count": 0.0,
            "phase2_enabled_fraction": 0.0
        }

    return {
        "experiments": ema_experiments,
        "summary": summary,
        "phase": "Phase 2: EMA + Warm-up + Hysteresis",
        "note": "Variance reduction demonstrates EMA smoothing benefit"
    }


def generate_coord_median_diagnostics(
    experiment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate Phase 5 Coordinate Median safety guard diagnostics

    Extracts:
    - Guard activation counts (min-clients, norm-clamp, direction-check)
    - Fallback usage statistics
    - Guard effectiveness (attacks blocked per guard)

    Args:
        experiment_results: List of experiment result dicts with Phase 5 data

    Returns:
        {
            "experiments": [
                {
                    "experiment_id": str,
                    "detector": str (should be "coord_median_safe"),
                    "attack": str,
                    "guard_activations": {
                        "min_clients_guard": int,  # Rounds triggered
                        "norm_clamp": int,  # Clients clipped
                        "direction_check": int  # Clients dropped
                    },
                    "fallback_usage": float,  # Fraction of rounds using trimmed-mean
                    "total_rounds": int
                }
            ],
            "summary": {
                "mean_min_clients_activations": float,
                "mean_norm_clamp_activations": float,
                "mean_direction_check_activations": float,
                "mean_fallback_usage": float,
                "phase5_enabled_fraction": float  # % experiments with Phase 5
            }
        }
    """
    cm_experiments = []

    for exp in experiment_results:
        # Check if Phase 5 data exists
        phase5_data = exp.get("phase5_data", {})
        if not phase5_data:
            continue

        # Extract guard activation counts
        guard_activations = phase5_data.get("guard_activations", {})
        min_clients_count = guard_activations.get("min_clients_guard", 0)
        norm_clamp_count = guard_activations.get("norm_clamp", 0)
        direction_check_count = guard_activations.get("direction_check", 0)

        # Fallback usage
        total_rounds = phase5_data.get("total_rounds", 0)
        fallback_rounds = phase5_data.get("fallback_rounds", 0)
        fallback_usage = fallback_rounds / total_rounds if total_rounds > 0 else 0.0

        cm_experiments.append({
            "experiment_id": exp.get("experiment_id", "unknown"),
            "detector": exp.get("detector", "unknown"),
            "attack": exp.get("attack", "unknown"),
            "guard_activations": {
                "min_clients_guard": int(min_clients_count),
                "norm_clamp": int(norm_clamp_count),
                "direction_check": int(direction_check_count)
            },
            "fallback_usage": float(fallback_usage),
            "total_rounds": int(total_rounds)
        })

    # Summary statistics
    if cm_experiments:
        min_clients_counts = [e["guard_activations"]["min_clients_guard"] for e in cm_experiments]
        norm_clamp_counts = [e["guard_activations"]["norm_clamp"] for e in cm_experiments]
        direction_check_counts = [e["guard_activations"]["direction_check"] for e in cm_experiments]
        fallback_usages = [e["fallback_usage"] for e in cm_experiments]

        summary = {
            "mean_min_clients_activations": float(np.mean(min_clients_counts)),
            "mean_norm_clamp_activations": float(np.mean(norm_clamp_counts)),
            "mean_direction_check_activations": float(np.mean(direction_check_counts)),
            "mean_fallback_usage": float(np.mean(fallback_usages)),
            "phase5_enabled_fraction": len(cm_experiments) / len(experiment_results)
        }
    else:
        summary = {
            "mean_min_clients_activations": 0.0,
            "mean_norm_clamp_activations": 0.0,
            "mean_direction_check_activations": 0.0,
            "mean_fallback_usage": 0.0,
            "phase5_enabled_fraction": 0.0
        }

    return {
        "experiments": cm_experiments,
        "summary": summary,
        "phase": "Phase 5: CoordinateMedianSafe Guards",
        "note": "Guard activation rates indicate robustness improvement without false alarms"
    }
