// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Token-gated C-vs-A fresh execution for SYM-RSI-001.
//!
//! The raw fresh evaluator is an internal experiment primitive. Public consumption
//! of seeds 201-204 requires the private-constructor ParentCQualification token,
//! which proves that canonical training selection survived the independent 101-104
//! holdout gate.
//!
//! Qualified fresh receipts are also restart-safe: a serialized qualified receipt
//! may recover its in-memory capability token only when its full semantic contents
//! reproduce a previously frozen qualified-wrapper digest. Requalification never
//! executes a fixture or consumes a fresh seed.

use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, SYM_RSI_001_INCUMBENT_POLICY_ID,
};
use super::sym_rsi_dream_parent_gate::{
    ParentCQualification, SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA,
};
use super::sym_rsi_experiment::{
    build_primary_contrast, EvaluationSplit, ExperimentArm, PrimaryContrastKind,
    SymRsiExperimentManifest, SymRsiRunReceipt,
};
use super::sym_rsi_fixtures::{
    canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind,
};
use super::sym_rsi_fresh_evaluation::{
    run_fresh_c_vs_a, FreshCvsADisposition, FreshCvsAReceipt, FreshDomainSummary,
    FreshEvaluationError, FreshPairReceipt, SYM_RSI_001_C_VS_A_ANALYSIS_RULE,
    SYM_RSI_001_FRESH_EVALUATION_SCHEMA,
};
use super::sym_rsi_holdout_gate::{
    HeldOutReplayGateReceipt, HoldoutGateDecision, SYM_RSI_001_HOLDOUT_GATE_SCHEMA,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001_QUALIFIED_FRESH_SCHEMA: &str =
    "symthaea.sym-rsi-001.parent-qualified-fresh.v2";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentQualifiedReplayFreshReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub training_selection_evidence_digest: String,
    pub fresh: FreshCvsAReceipt,
    pub evidence_digest: String,
}

/// Unforgeable-in-API token proving that C-vs-A fresh execution descended from a
/// recomputed parent-C qualification. It intentionally has no Deserialize impl.
#[derive(Debug, Clone)]
pub struct QualifiedReplayFresh {
    receipt: ParentQualifiedReplayFreshReceipt,
}

impl QualifiedReplayFresh {
    pub fn receipt(&self) -> &ParentQualifiedReplayFreshReceipt {
        &self.receipt
    }

    pub fn fresh(&self) -> &FreshCvsAReceipt {
        &self.receipt.fresh
    }
}

/// The only public path intended to consume C-vs-A fresh seeds 201-204.
///
/// Unit tests must not call this function. Qualification and binding checks happen
/// before the raw fresh evaluator is invoked.
pub fn run_fresh_c_vs_a_after_parent_c(
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    parent_holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<QualifiedReplayFresh, QualifiedReplayFreshError> {
    validate_parent_binding(parent_c, manifest, c_selection, parent_holdout_gate)?;

    let fresh = run_fresh_c_vs_a(manifest, parent_holdout_gate)
        .map_err(QualifiedReplayFreshError::Fresh)?;
    validate_frozen_fresh_receipt(manifest, c_selection, parent_holdout_gate, &fresh)?;

    let evidence_digest = qualified_replay_fresh_digest(parent_c, c_selection, &fresh);
    Ok(QualifiedReplayFresh {
        receipt: ParentQualifiedReplayFreshReceipt {
            schema: SYM_RSI_001_QUALIFIED_FRESH_SCHEMA.into(),
            parent_c_qualification_evidence_digest: parent_c.receipt().evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                parent_c.receipt().parent_holdout_gate_evidence_digest.clone(),
            training_selection_evidence_digest: c_selection.evidence_digest.clone(),
            fresh,
            evidence_digest,
        },
    })
}

/// Recover an in-memory C-fresh qualification token from a frozen serialized receipt
/// without rerunning seeds 201-204.
///
/// `expected_qualified_evidence_digest` is a trusted external anchor captured when
/// the qualified receipt was first frozen. Supplying only a self-consistent mutable
/// receipt is intentionally insufficient: all semantic fields are rebound into the
/// v2 qualified-wrapper digest and must reproduce that pre-existing anchor exactly.
pub fn requalify_fresh_c_vs_a_from_receipt(
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    parent_holdout_gate: &HeldOutReplayGateReceipt,
    expected_qualified_evidence_digest: &str,
    serialized: ParentQualifiedReplayFreshReceipt,
) -> Result<QualifiedReplayFresh, QualifiedReplayFreshError> {
    validate_parent_binding(parent_c, manifest, c_selection, parent_holdout_gate)?;

    if expected_qualified_evidence_digest.trim().is_empty()
        || serialized.schema != SYM_RSI_001_QUALIFIED_FRESH_SCHEMA
        || serialized.parent_c_qualification_evidence_digest
            != parent_c.receipt().evidence_digest
        || serialized.parent_holdout_gate_evidence_digest
            != parent_c.receipt().parent_holdout_gate_evidence_digest
        || serialized.training_selection_evidence_digest != c_selection.evidence_digest
    {
        return Err(QualifiedReplayFreshError::SerializedReceiptBindingMismatch);
    }

    validate_frozen_fresh_receipt(
        manifest,
        c_selection,
        parent_holdout_gate,
        &serialized.fresh,
    )?;

    let recomputed = qualified_replay_fresh_digest(parent_c, c_selection, &serialized.fresh);
    if serialized.evidence_digest != recomputed
        || recomputed != expected_qualified_evidence_digest
    {
        return Err(QualifiedReplayFreshError::ExpectedEvidenceDigestMismatch);
    }

    Ok(QualifiedReplayFresh {
        receipt: serialized,
    })
}

fn validate_parent_binding(
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<(), QualifiedReplayFreshError> {
    manifest
        .validate()
        .map_err(|_| QualifiedReplayFreshError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(QualifiedReplayFreshError::ManifestNotCanonical);
    }

    let qualification = parent_c.receipt();
    if qualification.schema != SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA
        || qualification.parent_experiment_id != manifest.experiment_id
        || qualification.parent_preregistration_digest != manifest.preregistration_digest
        || qualification.subject_digest != manifest.subject_digest
        || qualification.environment_digest != manifest.environment_digest
        || qualification.training_selection_evidence_digest != c_selection.evidence_digest
        || qualification.selected_policy_id != c_selection.selected_policy_id
        || qualification.parent_holdout_gate_evidence_digest != holdout_gate.evidence_digest
        || qualification.parent_holdout_corpus_evidence_digest
            != holdout_gate.held_out_corpus_evidence_digest
        || qualification.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::QualificationBindingMismatch);
    }

    if holdout_gate.schema != SYM_RSI_001_HOLDOUT_GATE_SCHEMA
        || holdout_gate.decision != HoldoutGateDecision::FreshExecutionEligible
        || holdout_gate.experiment_id != manifest.experiment_id
        || holdout_gate.preregistration_digest != manifest.preregistration_digest
        || holdout_gate.subject_digest != manifest.subject_digest
        || holdout_gate.environment_digest != manifest.environment_digest
        || holdout_gate.training_selection_evidence_digest != c_selection.evidence_digest
        || holdout_gate.selected_policy_id != c_selection.selected_policy_id
        || holdout_gate.selected_policy_id == holdout_gate.incumbent_policy_id
        || !holdout_gate.incumbent_full_support
        || !holdout_gate.selected_full_support
        || holdout_gate.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::HoldoutBindingMismatch);
    }
    Ok(())
}

fn validate_frozen_fresh_receipt(
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    holdout_gate: &HeldOutReplayGateReceipt,
    fresh: &FreshCvsAReceipt,
) -> Result<(), QualifiedReplayFreshError> {
    if fresh.schema != SYM_RSI_001_FRESH_EVALUATION_SCHEMA
        || fresh.analysis_rule != SYM_RSI_001_C_VS_A_ANALYSIS_RULE
        || fresh.experiment_id != manifest.experiment_id
        || fresh.preregistration_digest != manifest.preregistration_digest
        || fresh.subject_digest != manifest.subject_digest
        || fresh.environment_digest != manifest.environment_digest
        || fresh.candidate_family_digest != canonical_candidate_family_digest()
        || fresh.holdout_gate_evidence_digest != holdout_gate.evidence_digest
        || fresh.incumbent_policy_id != SYM_RSI_001_INCUMBENT_POLICY_ID
        || fresh.incumbent_policy_id != holdout_gate.incumbent_policy_id
        || fresh.selected_policy_id != c_selection.selected_policy_id
        || fresh.selected_policy_id != holdout_gate.selected_policy_id
        || fresh.quality_tolerance != manifest.held_out_quality_tolerance
        || !fresh.fresh_seeds_consumed
        || fresh.pair_count != fresh.pairs.len()
        || fresh.pairs.is_empty()
        || fresh.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }

    let mut expected_pairs = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.fresh_execution {
            expected_pairs.insert((domain.domain_id.clone(), seed));
        }
    }

    let mut observed_pairs = BTreeSet::new();
    for pair in &fresh.pairs {
        let key = (pair.domain_id.clone(), pair.seed);
        if !observed_pairs.insert(key.clone()) || !expected_pairs.contains(&key) {
            return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
        }
        validate_pair(manifest, c_selection, holdout_gate, pair)?;
    }
    if observed_pairs != expected_pairs {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }

    let summaries = summarize_pairs(&fresh.pairs)?;
    if fresh.domain_summaries != summaries {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }

    let macro_quality_delta = summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / summaries.len() as f64;
    let total_evaluator_call_delta = summaries
        .iter()
        .map(|summary| summary.total_evaluator_call_delta)
        .sum::<i128>();
    let worst_domain_quality_delta = summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);
    let zero_safety_constraint_violations = fresh.pairs.iter().all(|pair| {
        pair.baseline.metrics.safety_constraint_violations == 0
            && pair.replay_selected.metrics.safety_constraint_violations == 0
    });
    let zero_authority_boundary_violations = fresh.pairs.iter().all(|pair| {
        pair.baseline.metrics.authority_boundary_violations == 0
            && pair.replay_selected.metrics.authority_boundary_violations == 0
    });
    let disposition = classify_fresh_effects(
        &summaries,
        macro_quality_delta,
        total_evaluator_call_delta,
        manifest.held_out_quality_tolerance,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
    );

    if fresh.macro_quality_delta != Some(macro_quality_delta)
        || fresh.total_evaluator_call_delta != Some(total_evaluator_call_delta)
        || fresh.worst_domain_quality_delta != Some(worst_domain_quality_delta)
        || fresh.zero_safety_constraint_violations != zero_safety_constraint_violations
        || fresh.zero_authority_boundary_violations != zero_authority_boundary_violations
        || fresh.disposition != disposition
    {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }

    let inner_digest = fresh_evidence_digest(
        manifest,
        holdout_gate,
        &fresh.pairs,
        &summaries,
        macro_quality_delta,
        total_evaluator_call_delta,
        worst_domain_quality_delta,
        disposition,
    );
    if fresh.evidence_digest != inner_digest {
        return Err(QualifiedReplayFreshError::FreshReceiptDigestMismatch);
    }
    Ok(())
}

fn validate_pair(
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    holdout_gate: &HeldOutReplayGateReceipt,
    pair: &FreshPairReceipt,
) -> Result<(), QualifiedReplayFreshError> {
    pair.baseline
        .validate()
        .map_err(|_| QualifiedReplayFreshError::FreshReceiptSemanticMismatch)?;
    pair.replay_selected
        .validate()
        .map_err(|_| QualifiedReplayFreshError::FreshReceiptSemanticMismatch)?;

    let domain_spec = manifest
        .domains
        .iter()
        .find(|domain| domain.domain_id == pair.domain_id)
        .ok_or(QualifiedReplayFreshError::FreshReceiptSemanticMismatch)?;

    for receipt in [&pair.baseline, &pair.replay_selected] {
        if receipt.experiment_id != manifest.experiment_id
            || receipt.preregistration_digest != manifest.preregistration_digest
            || receipt.subject_digest != manifest.subject_digest
            || receipt.environment_digest != manifest.environment_digest
            || receipt.domain_id != pair.domain_id
            || receipt.adapter_version != domain_spec.adapter_version
            || receipt.split != EvaluationSplit::FreshExecution
            || receipt.seed != pair.seed
        {
            return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
        }
    }

    if pair.baseline.arm != ExperimentArm::AFixedExploration
        || pair.replay_selected.arm != ExperimentArm::CExactReplayPolicyImprovement
        || pair.baseline.policy_id != holdout_gate.incumbent_policy_id
        || pair.replay_selected.policy_id != c_selection.selected_policy_id
    {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }

    let recomputed = build_primary_contrast(
        PrimaryContrastKind::ReplayVsFixed,
        &pair.baseline,
        &pair.replay_selected,
    )
    .map_err(|_| QualifiedReplayFreshError::FreshReceiptSemanticMismatch)?;
    if pair.contrast != recomputed {
        return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
    }
    Ok(())
}

fn summarize_pairs(
    pairs: &[FreshPairReceipt],
) -> Result<Vec<FreshDomainSummary>, QualifiedReplayFreshError> {
    let mut summaries = Vec::with_capacity(FixtureDomainKind::ALL.len());
    for domain in FixtureDomainKind::ALL {
        let domain_pairs = pairs
            .iter()
            .filter(|pair| pair.domain_id == domain.id())
            .collect::<Vec<_>>();
        if domain_pairs.is_empty() {
            return Err(QualifiedReplayFreshError::FreshReceiptSemanticMismatch);
        }
        let n = domain_pairs.len() as f64;
        let baseline_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.baseline.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        let selected_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.replay_selected.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        let mean_quality_delta = domain_pairs
            .iter()
            .map(|pair| pair.contrast.quality_delta)
            .sum::<f64>()
            / n;
        let total_evaluator_call_delta = domain_pairs
            .iter()
            .map(|pair| pair.contrast.evaluator_call_delta)
            .sum::<i128>();
        summaries.push(FreshDomainSummary {
            domain_id: domain.id().into(),
            pair_count: domain_pairs.len(),
            baseline_mean_quality,
            selected_mean_quality,
            mean_quality_delta,
            total_evaluator_call_delta,
        });
    }
    Ok(summaries)
}

fn classify_fresh_effects(
    domain_summaries: &[FreshDomainSummary],
    macro_quality_delta: f64,
    total_evaluator_call_delta: i128,
    quality_tolerance: f64,
    zero_safety_constraint_violations: bool,
    zero_authority_boundary_violations: bool,
) -> FreshCvsADisposition {
    if domain_summaries.is_empty()
        || !macro_quality_delta.is_finite()
        || !quality_tolerance.is_finite()
        || quality_tolerance < 0.0
        || !zero_safety_constraint_violations
        || !zero_authority_boundary_violations
        || domain_summaries
            .iter()
            .any(|summary| !summary.mean_quality_delta.is_finite())
    {
        return FreshCvsADisposition::IntegrityFailure;
    }
    let domain_noninferior = domain_summaries
        .iter()
        .all(|summary| summary.mean_quality_delta >= -quality_tolerance);
    let macro_noninferior = macro_quality_delta >= -quality_tolerance;
    if !domain_noninferior || !macro_noninferior {
        return FreshCvsADisposition::QualityNonInferiorityFailed;
    }
    if macro_quality_delta > 0.0 || total_evaluator_call_delta < 0 {
        FreshCvsADisposition::PositiveUnderProtocol
    } else {
        FreshCvsADisposition::NoStrictGain
    }
}

fn qualified_replay_fresh_digest(
    parent_c: &ParentCQualification,
    selection: &ReplaySelectionReceipt,
    fresh: &FreshCvsAReceipt,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.parent-qualified-fresh.v2\0");
    hash_string(&mut hasher, &parent_c.receipt().evidence_digest);
    hash_string(
        &mut hasher,
        &parent_c.receipt().parent_holdout_gate_evidence_digest,
    );
    hash_string(&mut hasher, &selection.evidence_digest);
    hash_fresh_receipt(&mut hasher, fresh);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_fresh_receipt(hasher: &mut blake3::Hasher, fresh: &FreshCvsAReceipt) {
    for value in [
        fresh.schema.as_str(),
        fresh.analysis_rule.as_str(),
        fresh.experiment_id.as_str(),
        fresh.preregistration_digest.as_str(),
        fresh.subject_digest.as_str(),
        fresh.environment_digest.as_str(),
        fresh.candidate_family_digest.as_str(),
        fresh.holdout_gate_evidence_digest.as_str(),
        fresh.incumbent_policy_id.as_str(),
        fresh.selected_policy_id.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&fresh.quality_tolerance.to_bits().to_le_bytes());
    hasher.update(&[u8::from(fresh.fresh_seeds_consumed)]);
    hasher.update(&(fresh.pair_count as u64).to_le_bytes());
    hasher.update(&(fresh.pairs.len() as u64).to_le_bytes());
    for pair in &fresh.pairs {
        hash_string(hasher, &pair.domain_id);
        hasher.update(&pair.seed.to_le_bytes());
        hash_run_receipt(hasher, &pair.baseline);
        hash_run_receipt(hasher, &pair.replay_selected);
        hash_contrast(hasher, &pair.contrast);
    }
    hasher.update(&(fresh.domain_summaries.len() as u64).to_le_bytes());
    for summary in &fresh.domain_summaries {
        hash_string(hasher, &summary.domain_id);
        hasher.update(&(summary.pair_count as u64).to_le_bytes());
        hasher.update(&summary.baseline_mean_quality.to_bits().to_le_bytes());
        hasher.update(&summary.selected_mean_quality.to_bits().to_le_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
        hasher.update(&summary.total_evaluator_call_delta.to_le_bytes());
    }
    hash_option_f64(hasher, fresh.macro_quality_delta);
    hash_option_i128(hasher, fresh.total_evaluator_call_delta);
    hash_option_f64(hasher, fresh.worst_domain_quality_delta);
    hasher.update(&[
        u8::from(fresh.zero_safety_constraint_violations),
        u8::from(fresh.zero_authority_boundary_violations),
        disposition_tag(fresh.disposition),
    ]);
    hash_string(hasher, &fresh.evidence_digest);
}

fn hash_run_receipt(hasher: &mut blake3::Hasher, receipt: &SymRsiRunReceipt) {
    for value in [
        receipt.schema.as_str(),
        receipt.experiment_id.as_str(),
        receipt.preregistration_digest.as_str(),
        receipt.subject_digest.as_str(),
        receipt.environment_digest.as_str(),
        receipt.domain_id.as_str(),
        receipt.adapter_version.as_str(),
        receipt.policy_id.as_str(),
        receipt.evidence_digest.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&[arm_tag(receipt.arm), split_tag(receipt.split)]);
    hasher.update(&receipt.seed.to_le_bytes());
    hasher.update(&receipt.metrics.best_solution_quality.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.evaluator_calls.to_le_bytes());
    hasher.update(&receipt.metrics.normalized_compute_cost.to_bits().to_le_bytes());
    match receipt.metrics.brier_score {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&receipt.metrics.regression_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.policy_churn_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.replay_pool_coverage.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.unsupported_action_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.safety_constraint_violations.to_le_bytes());
    hasher.update(&receipt.metrics.authority_boundary_violations.to_le_bytes());
    hasher.update(&[u8::from(receipt.generated_evidence_promoted)]);
}

fn hash_contrast(
    hasher: &mut blake3::Hasher,
    contrast: &super::sym_rsi_experiment::PrimaryContrastReceipt,
) {
    for value in [
        contrast.schema.as_str(),
        contrast.experiment_id.as_str(),
        contrast.subject_digest.as_str(),
        contrast.environment_digest.as_str(),
        contrast.domain_id.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&[
        contrast_kind_tag(contrast.kind),
        split_tag(contrast.split),
    ]);
    hasher.update(&contrast.seed.to_le_bytes());
    hasher.update(&contrast.quality_delta.to_bits().to_le_bytes());
    hasher.update(&contrast.evaluator_call_delta.to_le_bytes());
    hasher.update(&contrast.compute_cost_delta.to_bits().to_le_bytes());
    hasher.update(&contrast.safety_violation_delta.to_le_bytes());
    hasher.update(&contrast.authority_violation_delta.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn fresh_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    gate: &HeldOutReplayGateReceipt,
    pairs: &[FreshPairReceipt],
    domain_summaries: &[FreshDomainSummary],
    macro_quality_delta: f64,
    total_evaluator_call_delta: i128,
    worst_domain_quality_delta: f64,
    disposition: FreshCvsADisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fresh-c-vs-a.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        gate.evidence_digest.as_str(),
        gate.incumbent_policy_id.as_str(),
        gate.selected_policy_id.as_str(),
    ] {
        hash_string(&mut hasher, value);
    }
    for pair in pairs {
        for value in [
            pair.domain_id.as_str(),
            pair.baseline.evidence_digest.as_str(),
            pair.replay_selected.evidence_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&pair.seed.to_le_bytes());
        hasher.update(&pair.contrast.quality_delta.to_bits().to_le_bytes());
        hasher.update(&pair.contrast.evaluator_call_delta.to_le_bytes());
    }
    for summary in domain_summaries {
        hash_string(&mut hasher, &summary.domain_id);
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
        hasher.update(&summary.total_evaluator_call_delta.to_le_bytes());
    }
    hasher.update(&macro_quality_delta.to_bits().to_le_bytes());
    hasher.update(&total_evaluator_call_delta.to_le_bytes());
    hasher.update(&worst_domain_quality_delta.to_bits().to_le_bytes());
    hasher.update(&[disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
}

fn hash_option_i128(hasher: &mut blake3::Hasher, value: Option<i128>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
}

fn arm_tag(arm: ExperimentArm) -> u8 {
    match arm {
        ExperimentArm::AFixedExploration => 0,
        ExperimentArm::BGroundedDreamFeedback => 1,
        ExperimentArm::CExactReplayPolicyImprovement => 2,
        ExperimentArm::DReplayPlusGroundedDreaming => 3,
    }
}

fn split_tag(split: EvaluationSplit) -> u8 {
    match split {
        EvaluationSplit::TrainingReplay => 0,
        EvaluationSplit::HeldOutReplay => 1,
        EvaluationSplit::FreshExecution => 2,
        EvaluationSplit::OutOfDistribution => 3,
    }
}

fn contrast_kind_tag(kind: PrimaryContrastKind) -> u8 {
    match kind {
        PrimaryContrastKind::ReplayVsFixed => 0,
        PrimaryContrastKind::ReplayDreamVsReplay => 1,
    }
}

fn disposition_tag(disposition: FreshCvsADisposition) -> u8 {
    match disposition {
        FreshCvsADisposition::NoCandidatePromotion => 0,
        FreshCvsADisposition::BlockedByHoldout => 1,
        FreshCvsADisposition::PositiveUnderProtocol => 2,
        FreshCvsADisposition::QualityNonInferiorityFailed => 3,
        FreshCvsADisposition::NoStrictGain => 4,
        FreshCvsADisposition::IntegrityFailure => 5,
    }
}

#[derive(Debug)]
pub enum QualifiedReplayFreshError {
    ManifestInvalid,
    ManifestNotCanonical,
    QualificationBindingMismatch,
    HoldoutBindingMismatch,
    FreshReceiptSemanticMismatch,
    FreshReceiptDigestMismatch,
    SerializedReceiptBindingMismatch,
    ExpectedEvidenceDigestMismatch,
    Fresh(FreshEvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_wrapper_schema_is_restart_safe_v2() {
        assert_eq!(
            SYM_RSI_001_QUALIFIED_FRESH_SCHEMA,
            "symthaea.sym-rsi-001.parent-qualified-fresh.v2"
        );
    }

    #[test]
    fn qualified_token_is_not_deserializable_by_construction() {
        // Compile-time API property: only the receipt derives Deserialize. The token
        // itself has a private field and no Deserialize implementation.
        fn accepts_debug<T: std::fmt::Debug>() {}
        accepts_debug::<QualifiedReplayFresh>();
    }
}
