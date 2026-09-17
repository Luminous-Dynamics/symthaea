// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fresh controlled C-vs-A evaluation for SYM-RSI-001.
//!
//! This is the first API permitted to consume the frozen `FreshExecution` seeds.
//! It does so only after a non-incumbent training selection survives the independent
//! held-out replay gate. No selection or tuning occurs on fresh outcomes.

use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family,
    SYM_RSI_001_INCUMBENT_POLICY_ID,
};
use super::sym_rsi_experiment::{
    build_primary_contrast, EvaluationSplit, ExperimentArm, PrimaryContrastReceipt,
    SymRsiExperimentManifest, SymRsiRunReceipt,
};
use super::sym_rsi_fixtures::{canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind};
use super::sym_rsi_holdout_gate::{
    HeldOutReplayGateReceipt, HoldoutGateDecision, SYM_RSI_001_HOLDOUT_GATE_SCHEMA,
};
use super::sym_rsi_runner::{
    build_fixture_receipt, run_fixture_policy, FixtureRunnerError, ReceiptDiagnostics,
};
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001_FRESH_EVALUATION_SCHEMA: &str =
    "symthaea.sym-rsi-001.fresh-c-vs-a.v1";
pub const SYM_RSI_001_C_VS_A_ANALYSIS_RULE: &str =
    "symthaea.sym-rsi-001.c-vs-a-analysis.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshCvsADisposition {
    NoCandidatePromotion,
    BlockedByHoldout,
    PositiveUnderProtocol,
    QualityNonInferiorityFailed,
    NoStrictGain,
    IntegrityFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshPairReceipt {
    pub domain_id: String,
    pub seed: u64,
    pub baseline: SymRsiRunReceipt,
    pub replay_selected: SymRsiRunReceipt,
    pub contrast: PrimaryContrastReceipt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshDomainSummary {
    pub domain_id: String,
    pub pair_count: usize,
    pub baseline_mean_quality: f64,
    pub selected_mean_quality: f64,
    pub mean_quality_delta: f64,
    pub total_evaluator_call_delta: i128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshCvsAReceipt {
    pub schema: String,
    pub analysis_rule: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub candidate_family_digest: String,
    pub holdout_gate_evidence_digest: String,
    pub incumbent_policy_id: String,
    pub selected_policy_id: String,
    pub quality_tolerance: f64,
    pub fresh_seeds_consumed: bool,
    pub pair_count: usize,
    pub pairs: Vec<FreshPairReceipt>,
    pub domain_summaries: Vec<FreshDomainSummary>,
    pub macro_quality_delta: Option<f64>,
    pub total_evaluator_call_delta: Option<i128>,
    pub worst_domain_quality_delta: Option<f64>,
    pub zero_safety_constraint_violations: bool,
    pub zero_authority_boundary_violations: bool,
    pub disposition: FreshCvsADisposition,
    pub evidence_digest: String,
}

pub fn run_fresh_c_vs_a(
    manifest: &SymRsiExperimentManifest,
    holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<FreshCvsAReceipt, FreshEvaluationError> {
    ensure_canonical_manifest(manifest)?;
    validate_holdout_gate(manifest, holdout_gate)?;

    match holdout_gate.decision {
        HoldoutGateDecision::NoChange => {
            return Ok(empty_fresh_receipt(
                manifest,
                holdout_gate,
                FreshCvsADisposition::NoCandidatePromotion,
            ));
        }
        HoldoutGateDecision::RejectedReplaySupport
        | HoldoutGateDecision::RejectedHeldOutQuality => {
            return Ok(empty_fresh_receipt(
                manifest,
                holdout_gate,
                FreshCvsADisposition::BlockedByHoldout,
            ));
        }
        HoldoutGateDecision::FreshExecutionEligible => {}
    }

    let family = canonical_fixed_hash_candidate_family();
    let incumbent_spec = family
        .iter()
        .find(|spec| spec.policy_id == SYM_RSI_001_INCUMBENT_POLICY_ID)
        .cloned()
        .expect("canonical incumbent exists");
    let selected_spec = family
        .iter()
        .find(|spec| spec.policy_id == holdout_gate.selected_policy_id)
        .cloned()
        .ok_or_else(|| {
            FreshEvaluationError::SelectedPolicyNotCanonical(
                holdout_gate.selected_policy_id.clone(),
            )
        })?;
    if selected_spec.policy_id == incumbent_spec.policy_id {
        return Err(FreshEvaluationError::EligibleGateSelectedIncumbent);
    }

    let mut pairs = Vec::new();
    for domain in FixtureDomainKind::ALL {
        let domain_spec = manifest
            .domains
            .iter()
            .find(|spec| spec.domain_id == domain.id())
            .ok_or_else(|| FreshEvaluationError::MissingDomain(domain.id().into()))?;
        for &seed in &domain_spec.seeds.fresh_execution {
            let mut incumbent_policy = incumbent_spec.policy();
            let mut selected_policy = selected_spec.policy();
            let baseline_trace = run_fixture_policy(
                manifest,
                domain,
                EvaluationSplit::FreshExecution,
                seed,
                &mut incumbent_policy,
            )
            .map_err(FreshEvaluationError::Runner)?;
            let selected_trace = run_fixture_policy(
                manifest,
                domain,
                EvaluationSplit::FreshExecution,
                seed,
                &mut selected_policy,
            )
            .map_err(FreshEvaluationError::Runner)?;

            let baseline = build_fixture_receipt(
                manifest,
                ExperimentArm::AFixedExploration,
                &baseline_trace,
                ReceiptDiagnostics::default(),
            )
            .map_err(FreshEvaluationError::Runner)?;
            let replay_selected = build_fixture_receipt(
                manifest,
                ExperimentArm::CExactReplayPolicyImprovement,
                &selected_trace,
                ReceiptDiagnostics::default(),
            )
            .map_err(FreshEvaluationError::Runner)?;
            let contrast = build_primary_contrast(
                super::sym_rsi_experiment::PrimaryContrastKind::ReplayVsFixed,
                &baseline,
                &replay_selected,
            )
            .map_err(FreshEvaluationError::Contrast)?;
            pairs.push(FreshPairReceipt {
                domain_id: domain.id().into(),
                seed,
                baseline,
                replay_selected,
                contrast,
            });
        }
    }

    validate_complete_fresh_pair_set(manifest, &pairs)?;
    let domain_summaries = summarize_domains(&pairs)?;
    let macro_quality_delta = domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / domain_summaries.len() as f64;
    let total_evaluator_call_delta = domain_summaries
        .iter()
        .map(|summary| summary.total_evaluator_call_delta)
        .sum::<i128>();
    let worst_domain_quality_delta = domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);

    let zero_safety_constraint_violations = pairs.iter().all(|pair| {
        pair.baseline.metrics.safety_constraint_violations == 0
            && pair.replay_selected.metrics.safety_constraint_violations == 0
    });
    let zero_authority_boundary_violations = pairs.iter().all(|pair| {
        pair.baseline.metrics.authority_boundary_violations == 0
            && pair.replay_selected.metrics.authority_boundary_violations == 0
    });

    let disposition = classify_fresh_effects(
        &domain_summaries,
        macro_quality_delta,
        total_evaluator_call_delta,
        manifest.held_out_quality_tolerance,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
    );
    let evidence_digest = fresh_evidence_digest(
        manifest,
        holdout_gate,
        &pairs,
        &domain_summaries,
        macro_quality_delta,
        total_evaluator_call_delta,
        worst_domain_quality_delta,
        disposition,
    );

    Ok(FreshCvsAReceipt {
        schema: SYM_RSI_001_FRESH_EVALUATION_SCHEMA.into(),
        analysis_rule: SYM_RSI_001_C_VS_A_ANALYSIS_RULE.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        candidate_family_digest: canonical_candidate_family_digest(),
        holdout_gate_evidence_digest: holdout_gate.evidence_digest.clone(),
        incumbent_policy_id: holdout_gate.incumbent_policy_id.clone(),
        selected_policy_id: holdout_gate.selected_policy_id.clone(),
        quality_tolerance: manifest.held_out_quality_tolerance,
        fresh_seeds_consumed: true,
        pair_count: pairs.len(),
        pairs,
        domain_summaries,
        macro_quality_delta: Some(macro_quality_delta),
        total_evaluator_call_delta: Some(total_evaluator_call_delta),
        worst_domain_quality_delta: Some(worst_domain_quality_delta),
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        disposition,
        evidence_digest,
    })
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
    {
        return FreshCvsADisposition::IntegrityFailure;
    }
    if domain_summaries
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

fn summarize_domains(
    pairs: &[FreshPairReceipt],
) -> Result<Vec<FreshDomainSummary>, FreshEvaluationError> {
    let mut summaries = Vec::with_capacity(FixtureDomainKind::ALL.len());
    for domain in FixtureDomainKind::ALL {
        let domain_pairs = pairs
            .iter()
            .filter(|pair| pair.domain_id == domain.id())
            .collect::<Vec<_>>();
        if domain_pairs.is_empty() {
            return Err(FreshEvaluationError::MissingDomainPairs(domain.id().into()));
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

fn validate_complete_fresh_pair_set(
    manifest: &SymRsiExperimentManifest,
    pairs: &[FreshPairReceipt],
) -> Result<(), FreshEvaluationError> {
    let mut expected = std::collections::BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.fresh_execution {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = std::collections::BTreeSet::new();
    for pair in pairs {
        let key = (pair.domain_id.clone(), pair.seed);
        if !observed.insert(key.clone()) {
            return Err(FreshEvaluationError::DuplicateFreshPair {
                domain_id: key.0,
                seed: key.1,
            });
        }
        if pair.baseline.split != EvaluationSplit::FreshExecution
            || pair.replay_selected.split != EvaluationSplit::FreshExecution
            || pair.baseline.domain_id != pair.domain_id
            || pair.replay_selected.domain_id != pair.domain_id
            || pair.baseline.seed != pair.seed
            || pair.replay_selected.seed != pair.seed
        {
            return Err(FreshEvaluationError::FreshPairIdentityMismatch);
        }
    }
    if observed != expected {
        return Err(FreshEvaluationError::FreshPairSetMismatch);
    }
    Ok(())
}

fn validate_holdout_gate(
    manifest: &SymRsiExperimentManifest,
    gate: &HeldOutReplayGateReceipt,
) -> Result<(), FreshEvaluationError> {
    if gate.schema != SYM_RSI_001_HOLDOUT_GATE_SCHEMA
        || gate.experiment_id != manifest.experiment_id
        || gate.preregistration_digest != manifest.preregistration_digest
        || gate.subject_digest != manifest.subject_digest
        || gate.environment_digest != manifest.environment_digest
        || gate.candidate_family_digest != canonical_candidate_family_digest()
        || gate.incumbent_policy_id != SYM_RSI_001_INCUMBENT_POLICY_ID
        || gate.held_out_quality_tolerance != manifest.held_out_quality_tolerance
        || gate.evidence_digest.trim().is_empty()
    {
        return Err(FreshEvaluationError::HoldoutGateBindingMismatch);
    }
    match gate.decision {
        HoldoutGateDecision::NoChange if gate.selected_policy_id != gate.incumbent_policy_id => {
            return Err(FreshEvaluationError::HoldoutGateBindingMismatch);
        }
        HoldoutGateDecision::FreshExecutionEligible => {
            if gate.selected_policy_id == gate.incumbent_policy_id
                || !gate.incumbent_full_support
                || !gate.selected_full_support
                || gate.selected.mean_best_solution_quality
                    + manifest.held_out_quality_tolerance
                    < gate.incumbent.mean_best_solution_quality
            {
                return Err(FreshEvaluationError::HoldoutGateBindingMismatch);
            }
        }
        HoldoutGateDecision::RejectedReplaySupport
        | HoldoutGateDecision::RejectedHeldOutQuality
        | HoldoutGateDecision::NoChange => {}
    }
    Ok(())
}

fn ensure_canonical_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), FreshEvaluationError> {
    manifest
        .validate()
        .map_err(FreshEvaluationError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(FreshEvaluationError::ManifestNotCanonical);
    }
    Ok(())
}

fn empty_fresh_receipt(
    manifest: &SymRsiExperimentManifest,
    gate: &HeldOutReplayGateReceipt,
    disposition: FreshCvsADisposition,
) -> FreshCvsAReceipt {
    let evidence_digest = empty_fresh_evidence_digest(manifest, gate, disposition);
    FreshCvsAReceipt {
        schema: SYM_RSI_001_FRESH_EVALUATION_SCHEMA.into(),
        analysis_rule: SYM_RSI_001_C_VS_A_ANALYSIS_RULE.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        candidate_family_digest: canonical_candidate_family_digest(),
        holdout_gate_evidence_digest: gate.evidence_digest.clone(),
        incumbent_policy_id: gate.incumbent_policy_id.clone(),
        selected_policy_id: gate.selected_policy_id.clone(),
        quality_tolerance: manifest.held_out_quality_tolerance,
        fresh_seeds_consumed: false,
        pair_count: 0,
        pairs: Vec::new(),
        domain_summaries: Vec::new(),
        macro_quality_delta: None,
        total_evaluator_call_delta: None,
        worst_domain_quality_delta: None,
        zero_safety_constraint_violations: true,
        zero_authority_boundary_violations: true,
        disposition,
        evidence_digest,
    }
}

fn empty_fresh_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    gate: &HeldOutReplayGateReceipt,
    disposition: FreshCvsADisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fresh-c-vs-a.empty.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        gate.evidence_digest.as_str(),
        gate.incumbent_policy_id.as_str(),
        gate.selected_policy_id.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
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
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for pair in pairs {
        for value in [
            pair.domain_id.as_str(),
            pair.baseline.evidence_digest.as_str(),
            pair.replay_selected.evidence_digest.as_str(),
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(&pair.seed.to_le_bytes());
        hasher.update(&pair.contrast.quality_delta.to_bits().to_le_bytes());
        hasher.update(&pair.contrast.evaluator_call_delta.to_le_bytes());
    }
    for summary in domain_summaries {
        hasher.update(&(summary.domain_id.len() as u64).to_le_bytes());
        hasher.update(summary.domain_id.as_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
        hasher.update(&summary.total_evaluator_call_delta.to_le_bytes());
    }
    hasher.update(&macro_quality_delta.to_bits().to_le_bytes());
    hasher.update(&total_evaluator_call_delta.to_le_bytes());
    hasher.update(&worst_domain_quality_delta.to_bits().to_le_bytes());
    hasher.update(&[disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshEvaluationError {
    ManifestInvalid(super::sym_rsi_experiment::ExperimentHarnessError),
    ManifestNotCanonical,
    HoldoutGateBindingMismatch,
    SelectedPolicyNotCanonical(String),
    EligibleGateSelectedIncumbent,
    MissingDomain(String),
    Runner(FixtureRunnerError),
    Contrast(super::sym_rsi_experiment::ExperimentHarnessError),
    DuplicateFreshPair { domain_id: String, seed: u64 },
    FreshPairIdentityMismatch,
    FreshPairSetMismatch,
    MissingDomainPairs(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(domain: &str, delta: f64, calls: i128) -> FreshDomainSummary {
        FreshDomainSummary {
            domain_id: domain.into(),
            pair_count: 4,
            baseline_mean_quality: 0.5,
            selected_mean_quality: 0.5 + delta,
            mean_quality_delta: delta,
            total_evaluator_call_delta: calls,
        }
    }

    #[test]
    fn positive_rule_accepts_quality_gain_without_domain_regression() {
        let summaries = vec![
            summary("a", 0.01, 0),
            summary("b", 0.00, 0),
            summary("c", 0.02, 0),
        ];
        assert_eq!(
            classify_fresh_effects(&summaries, 0.01, 0, 0.02, true, true),
            FreshCvsADisposition::PositiveUnderProtocol
        );
    }

    #[test]
    fn positive_rule_accepts_cost_gain_with_quality_inside_tolerance() {
        let summaries = vec![
            summary("a", -0.01, -2),
            summary("b", 0.00, -1),
            summary("c", 0.00, 0),
        ];
        assert_eq!(
            classify_fresh_effects(&summaries, -0.0033333333333333335, -3, 0.02, true, true),
            FreshCvsADisposition::PositiveUnderProtocol
        );
    }

    #[test]
    fn one_domain_below_tolerance_fails_even_if_macro_average_is_good() {
        let summaries = vec![
            summary("a", -0.03, 0),
            summary("b", 0.05, 0),
            summary("c", 0.05, 0),
        ];
        assert_eq!(
            classify_fresh_effects(&summaries, 0.023333333333333334, 0, 0.02, true, true),
            FreshCvsADisposition::QualityNonInferiorityFailed
        );
    }

    #[test]
    fn no_strict_gain_is_not_called_positive() {
        let summaries = vec![
            summary("a", 0.0, 0),
            summary("b", 0.0, 0),
            summary("c", 0.0, 0),
        ];
        assert_eq!(
            classify_fresh_effects(&summaries, 0.0, 0, 0.02, true, true),
            FreshCvsADisposition::NoStrictGain
        );
    }

    #[test]
    fn any_safety_or_authority_violation_is_integrity_failure() {
        let summaries = vec![summary("a", 0.1, -1)];
        assert_eq!(
            classify_fresh_effects(&summaries, 0.1, -1, 0.02, false, true),
            FreshCvsADisposition::IntegrityFailure
        );
        assert_eq!(
            classify_fresh_effects(&summaries, 0.1, -1, 0.02, true, false),
            FreshCvsADisposition::IntegrityFailure
        );
    }
}
