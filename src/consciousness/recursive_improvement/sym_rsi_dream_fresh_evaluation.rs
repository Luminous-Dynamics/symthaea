// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fresh D-vs-C evaluation for the sealed SYM-RSI-001D lineage.
//!
//! This is the only API intended to consume the 401-404 fresh dream seeds. It may
//! do so only after the independent 301-304 verification gate explicitly authorizes
//! fresh execution. Unit tests exercise only the pure classification rule and never
//! call the measurement entry point.

use super::sym_rsi_candidate_family::canonical_fixed_hash_candidate_family;
use super::sym_rsi_dream_protocol::{
    validate_canonical_sym_rsi_001d_manifest, SYM_RSI_001D_ANALYSIS_RULE,
    SYM_RSI_001D_EXPERIMENT_ID, SYM_RSI_001D_QUALITY_TOLERANCE,
};
use super::sym_rsi_dream_verification::{
    DreamVerificationDecision, DreamVerificationGateReceipt,
    SYM_RSI_001D_VERIFICATION_GATE_SCHEMA,
};
use super::sym_rsi_experiment::{
    build_primary_contrast, EvaluationSplit, ExperimentArm, PrimaryContrastKind,
    PrimaryContrastReceipt, SymRsiExperimentManifest, SymRsiRunReceipt,
};
use super::sym_rsi_fixtures::FixtureDomainKind;
use super::sym_rsi_grounded_dream::{
    build_grounded_dream_policy, GroundedDreamError, SYM_RSI_001_GROUNDED_DREAM_POLICY_ID,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use super::sym_rsi_runner::{
    build_fixture_receipt, run_fixture_policy, FixtureRunnerError, ReceiptDiagnostics,
};
use super::FrozenReplayCorpus;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001D_FRESH_EVALUATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.fresh-d-vs-c.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DreamFreshDisposition {
    NoDreamIntervention,
    BlockedByVerification,
    PositiveUnderProtocol,
    QualityNonInferiorityFailed,
    NoStrictQualityGain,
    IntegrityFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamFreshPairReceipt {
    pub domain_id: String,
    pub seed: u64,
    pub replay_selected: SymRsiRunReceipt,
    pub replay_plus_dream: SymRsiRunReceipt,
    pub contrast: PrimaryContrastReceipt,
    pub d_override_count: usize,
    pub d_prediction_count: usize,
    pub d_model_simulation_count: usize,
    pub generated_evidence_promoted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamFreshDomainSummary {
    pub domain_id: String,
    pub pair_count: usize,
    pub c_mean_quality: f64,
    pub d_mean_quality: f64,
    pub mean_quality_delta: f64,
    pub d_override_count: usize,
    pub d_prediction_count: usize,
    pub d_model_simulation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamFreshEvaluationReceipt {
    pub schema: String,
    pub analysis_rule: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub parent_training_experiment_id: String,
    pub parent_training_preregistration_digest: String,
    pub c_selection_evidence_digest: String,
    pub grounded_dream_model_evidence_digest: String,
    pub verification_gate_evidence_digest: String,
    pub c_policy_id: String,
    pub d_policy_id: String,
    pub quality_tolerance: f64,
    pub fresh_seeds_consumed: bool,
    pub pair_count: usize,
    pub pairs: Vec<DreamFreshPairReceipt>,
    pub domain_summaries: Vec<DreamFreshDomainSummary>,
    pub macro_quality_delta: Option<f64>,
    pub worst_domain_quality_delta: Option<f64>,
    pub total_d_override_count: usize,
    pub total_d_prediction_count: usize,
    pub total_d_model_simulation_count: usize,
    pub zero_safety_constraint_violations: bool,
    pub zero_authority_boundary_violations: bool,
    pub generated_evidence_promoted: bool,
    /// The generic receipt currently normalizes compute as environment calls.
    /// D's extra model simulations are therefore reported separately and no
    /// efficiency conclusion is authorized by this experiment.
    pub generic_compute_cost_is_environment_calls_only: bool,
    pub efficiency_claim_authorized: bool,
    pub disposition: DreamFreshDisposition,
    pub evidence_digest: String,
}

/// Execute the sealed fresh D-vs-C comparison.
///
/// If verification did not authorize fresh execution, this returns a bound no-run
/// receipt without touching any 401-404 seed.
pub fn run_fresh_d_vs_c(
    parent_training_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
) -> Result<DreamFreshEvaluationReceipt, DreamFreshEvaluationError> {
    validate_canonical_sym_rsi_001d_manifest(extension_manifest)
        .map_err(DreamFreshEvaluationError::Protocol)?;
    validate_cross_lineage_binding(parent_training_manifest, extension_manifest)?;
    validate_verification_gate_binding(
        parent_training_manifest,
        c_selection,
        extension_manifest,
        verification_gate,
    )?;

    match verification_gate.decision {
        DreamVerificationDecision::NoDreamIntervention => {
            return Ok(empty_receipt(
                parent_training_manifest,
                c_selection,
                extension_manifest,
                verification_gate,
                DreamFreshDisposition::NoDreamIntervention,
            ));
        }
        DreamVerificationDecision::RejectedReplaySupport
        | DreamVerificationDecision::RejectedQuality
        | DreamVerificationDecision::BlockedEpistemicIntegrity => {
            return Ok(empty_receipt(
                parent_training_manifest,
                c_selection,
                extension_manifest,
                verification_gate,
                DreamFreshDisposition::BlockedByVerification,
            ));
        }
        DreamVerificationDecision::FreshDreamExecutionEligible => {}
    }

    let dream_template = build_grounded_dream_policy(
        parent_training_manifest,
        parent_training_corpus,
        c_selection,
    )
    .map_err(DreamFreshEvaluationError::GroundedDream)?;
    if dream_template.model().evidence_digest
        != verification_gate.grounded_dream_model_evidence_digest
    {
        return Err(DreamFreshEvaluationError::DreamModelDigestMismatch);
    }

    let c_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|candidate| candidate.policy_id == c_selection.selected_policy_id)
        .ok_or_else(|| {
            DreamFreshEvaluationError::SelectedPolicyNotCanonical(
                c_selection.selected_policy_id.clone(),
            )
        })?;

    let mut pairs = Vec::new();
    for domain in FixtureDomainKind::ALL {
        let domain_spec = extension_manifest
            .domains
            .iter()
            .find(|spec| spec.domain_id == domain.id())
            .ok_or_else(|| DreamFreshEvaluationError::MissingDomain(domain.id().into()))?;

        for &seed in &domain_spec.seeds.fresh_execution {
            let mut c_policy = c_spec.policy();
            let mut d_policy = dream_template.clone();

            let c_trace = run_fixture_policy(
                extension_manifest,
                domain,
                EvaluationSplit::FreshExecution,
                seed,
                &mut c_policy,
            )
            .map_err(DreamFreshEvaluationError::Runner)?;
            let d_trace = run_fixture_policy(
                extension_manifest,
                domain,
                EvaluationSplit::FreshExecution,
                seed,
                &mut d_policy,
            )
            .map_err(DreamFreshEvaluationError::Runner)?;

            let d_override_count = d_policy
                .decision_log()
                .iter()
                .filter(|decision| decision.override_applied)
                .count();
            let d_prediction_count = d_policy
                .decision_log()
                .iter()
                .map(|decision| decision.predictions.len())
                .sum::<usize>();
            let d_model_simulation_count =
                d_prediction_count * d_policy.model().config.counterfactual_count;
            let generated_evidence_promoted = d_policy.generated_evidence_promoted();

            let c_receipt = build_fixture_receipt(
                extension_manifest,
                ExperimentArm::CExactReplayPolicyImprovement,
                &c_trace,
                ReceiptDiagnostics::default(),
            )
            .map_err(DreamFreshEvaluationError::Runner)?;

            let d_churn_rate = if d_trace.evaluator_calls == 0 {
                0.0
            } else {
                d_override_count as f64 / d_trace.evaluator_calls as f64
            };
            let d_receipt = build_fixture_receipt(
                extension_manifest,
                ExperimentArm::DReplayPlusGroundedDreaming,
                &d_trace,
                ReceiptDiagnostics {
                    policy_churn_rate: d_churn_rate,
                    generated_evidence_promoted,
                    ..ReceiptDiagnostics::default()
                },
            )
            .map_err(DreamFreshEvaluationError::Runner)?;

            let contrast = build_primary_contrast(
                PrimaryContrastKind::ReplayDreamVsReplay,
                &c_receipt,
                &d_receipt,
            )
            .map_err(DreamFreshEvaluationError::Contrast)?;

            pairs.push(DreamFreshPairReceipt {
                domain_id: domain.id().into(),
                seed,
                replay_selected: c_receipt,
                replay_plus_dream: d_receipt,
                contrast,
                d_override_count,
                d_prediction_count,
                d_model_simulation_count,
                generated_evidence_promoted,
            });
        }
    }

    validate_complete_pair_set(extension_manifest, &pairs)?;
    let domain_summaries = summarize_domains(&pairs)?;
    let macro_quality_delta = domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / domain_summaries.len() as f64;
    let worst_domain_quality_delta = domain_summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);
    let total_d_override_count = pairs
        .iter()
        .map(|pair| pair.d_override_count)
        .sum::<usize>();
    let total_d_prediction_count = pairs
        .iter()
        .map(|pair| pair.d_prediction_count)
        .sum::<usize>();
    let total_d_model_simulation_count = pairs
        .iter()
        .map(|pair| pair.d_model_simulation_count)
        .sum::<usize>();

    let zero_safety_constraint_violations = pairs.iter().all(|pair| {
        pair.replay_selected.metrics.safety_constraint_violations == 0
            && pair.replay_plus_dream.metrics.safety_constraint_violations == 0
    });
    let zero_authority_boundary_violations = pairs.iter().all(|pair| {
        pair.replay_selected.metrics.authority_boundary_violations == 0
            && pair.replay_plus_dream.metrics.authority_boundary_violations == 0
    });
    let generated_evidence_promoted = pairs
        .iter()
        .any(|pair| pair.generated_evidence_promoted
            || pair.replay_plus_dream.generated_evidence_promoted);

    let disposition = classify_fresh_dream_effects(
        &domain_summaries,
        macro_quality_delta,
        total_d_override_count,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        generated_evidence_promoted,
        SYM_RSI_001D_QUALITY_TOLERANCE,
    );
    let evidence_digest = fresh_dream_evidence_digest(
        parent_training_manifest,
        c_selection,
        extension_manifest,
        verification_gate,
        dream_template.model().evidence_digest.as_str(),
        &pairs,
        &domain_summaries,
        macro_quality_delta,
        worst_domain_quality_delta,
        total_d_override_count,
        total_d_prediction_count,
        total_d_model_simulation_count,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        generated_evidence_promoted,
        disposition,
    );

    Ok(DreamFreshEvaluationReceipt {
        schema: SYM_RSI_001D_FRESH_EVALUATION_SCHEMA.into(),
        analysis_rule: SYM_RSI_001D_ANALYSIS_RULE.into(),
        experiment_id: extension_manifest.experiment_id.clone(),
        preregistration_digest: extension_manifest.preregistration_digest.clone(),
        subject_digest: extension_manifest.subject_digest.clone(),
        environment_digest: extension_manifest.environment_digest.clone(),
        parent_training_experiment_id: parent_training_manifest.experiment_id.clone(),
        parent_training_preregistration_digest:
            parent_training_manifest.preregistration_digest.clone(),
        c_selection_evidence_digest: c_selection.evidence_digest.clone(),
        grounded_dream_model_evidence_digest: dream_template.model().evidence_digest.clone(),
        verification_gate_evidence_digest: verification_gate.evidence_digest.clone(),
        c_policy_id: c_selection.selected_policy_id.clone(),
        d_policy_id: SYM_RSI_001_GROUNDED_DREAM_POLICY_ID.into(),
        quality_tolerance: SYM_RSI_001D_QUALITY_TOLERANCE,
        fresh_seeds_consumed: true,
        pair_count: pairs.len(),
        pairs,
        domain_summaries,
        macro_quality_delta: Some(macro_quality_delta),
        worst_domain_quality_delta: Some(worst_domain_quality_delta),
        total_d_override_count,
        total_d_prediction_count,
        total_d_model_simulation_count,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        generated_evidence_promoted,
        generic_compute_cost_is_environment_calls_only: true,
        efficiency_claim_authorized: false,
        disposition,
        evidence_digest,
    })
}

fn classify_fresh_dream_effects(
    domain_summaries: &[DreamFreshDomainSummary],
    macro_quality_delta: f64,
    total_overrides: usize,
    zero_safety_constraint_violations: bool,
    zero_authority_boundary_violations: bool,
    generated_evidence_promoted: bool,
    tolerance: f64,
) -> DreamFreshDisposition {
    if domain_summaries.is_empty()
        || !macro_quality_delta.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || !zero_safety_constraint_violations
        || !zero_authority_boundary_violations
        || generated_evidence_promoted
    {
        return DreamFreshDisposition::IntegrityFailure;
    }
    if total_overrides == 0 {
        return DreamFreshDisposition::NoDreamIntervention;
    }
    if macro_quality_delta < -tolerance
        || domain_summaries
            .iter()
            .any(|summary| !summary.mean_quality_delta.is_finite()
                || summary.mean_quality_delta < -tolerance)
    {
        return DreamFreshDisposition::QualityNonInferiorityFailed;
    }
    if macro_quality_delta > 0.0 {
        DreamFreshDisposition::PositiveUnderProtocol
    } else {
        DreamFreshDisposition::NoStrictQualityGain
    }
}

fn summarize_domains(
    pairs: &[DreamFreshPairReceipt],
) -> Result<Vec<DreamFreshDomainSummary>, DreamFreshEvaluationError> {
    let mut summaries = Vec::with_capacity(FixtureDomainKind::ALL.len());
    for domain in FixtureDomainKind::ALL {
        let domain_pairs = pairs
            .iter()
            .filter(|pair| pair.domain_id == domain.id())
            .collect::<Vec<_>>();
        if domain_pairs.is_empty() {
            return Err(DreamFreshEvaluationError::MissingDomainPairs(
                domain.id().into(),
            ));
        }
        let n = domain_pairs.len() as f64;
        let c_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.replay_selected.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        let d_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.replay_plus_dream.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        summaries.push(DreamFreshDomainSummary {
            domain_id: domain.id().into(),
            pair_count: domain_pairs.len(),
            c_mean_quality,
            d_mean_quality,
            mean_quality_delta: d_mean_quality - c_mean_quality,
            d_override_count: domain_pairs
                .iter()
                .map(|pair| pair.d_override_count)
                .sum(),
            d_prediction_count: domain_pairs
                .iter()
                .map(|pair| pair.d_prediction_count)
                .sum(),
            d_model_simulation_count: domain_pairs
                .iter()
                .map(|pair| pair.d_model_simulation_count)
                .sum(),
        });
    }
    Ok(summaries)
}

fn validate_complete_pair_set(
    manifest: &SymRsiExperimentManifest,
    pairs: &[DreamFreshPairReceipt],
) -> Result<(), DreamFreshEvaluationError> {
    let mut expected = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.fresh_execution {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = BTreeSet::new();
    for pair in pairs {
        let key = (pair.domain_id.clone(), pair.seed);
        if !observed.insert(key.clone()) {
            return Err(DreamFreshEvaluationError::DuplicateFreshPair {
                domain_id: key.0,
                seed: key.1,
            });
        }
        if pair.replay_selected.split != EvaluationSplit::FreshExecution
            || pair.replay_plus_dream.split != EvaluationSplit::FreshExecution
            || pair.replay_selected.domain_id != pair.domain_id
            || pair.replay_plus_dream.domain_id != pair.domain_id
            || pair.replay_selected.seed != pair.seed
            || pair.replay_plus_dream.seed != pair.seed
        {
            return Err(DreamFreshEvaluationError::FreshPairIdentityMismatch);
        }
    }
    if observed != expected {
        return Err(DreamFreshEvaluationError::FreshPairSetMismatch);
    }
    Ok(())
}

fn validate_cross_lineage_binding(
    parent: &SymRsiExperimentManifest,
    extension: &SymRsiExperimentManifest,
) -> Result<(), DreamFreshEvaluationError> {
    if parent.experiment_id != "SYM-RSI-001"
        || extension.experiment_id != SYM_RSI_001D_EXPERIMENT_ID
        || parent.subject_digest != extension.subject_digest
        || parent.environment_digest != extension.environment_digest
    {
        return Err(DreamFreshEvaluationError::CrossLineageMismatch);
    }
    Ok(())
}

fn validate_verification_gate_binding(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    gate: &DreamVerificationGateReceipt,
) -> Result<(), DreamFreshEvaluationError> {
    if gate.schema != SYM_RSI_001D_VERIFICATION_GATE_SCHEMA
        || gate.experiment_id != extension_manifest.experiment_id
        || gate.preregistration_digest != extension_manifest.preregistration_digest
        || gate.subject_digest != extension_manifest.subject_digest
        || gate.environment_digest != extension_manifest.environment_digest
        || gate.parent_training_experiment_id != parent_manifest.experiment_id
        || gate.parent_training_preregistration_digest
            != parent_manifest.preregistration_digest
        || gate.c_selection_evidence_digest != c_selection.evidence_digest
        || gate.c_policy_id != c_selection.selected_policy_id
        || gate.d_policy_id != SYM_RSI_001_GROUNDED_DREAM_POLICY_ID
        || gate.quality_tolerance != SYM_RSI_001D_QUALITY_TOLERANCE
        || gate.grounded_dream_model_evidence_digest.trim().is_empty()
        || gate.verification_corpus_evidence_digest.trim().is_empty()
        || gate.evidence_digest.trim().is_empty()
    {
        return Err(DreamFreshEvaluationError::VerificationGateBindingMismatch);
    }

    if gate.decision == DreamVerificationDecision::FreshDreamExecutionEligible
        && (!gate.c_full_support
            || !gate.d_full_support
            || gate.generated_evidence_promoted
            || gate.total_d_override_count == 0
            || gate.macro_quality_delta < -SYM_RSI_001D_QUALITY_TOLERANCE
            || gate.domain_summaries.iter().any(|summary| {
                !summary.mean_quality_delta.is_finite()
                    || summary.mean_quality_delta < -SYM_RSI_001D_QUALITY_TOLERANCE
            }))
    {
        return Err(DreamFreshEvaluationError::VerificationGateBindingMismatch);
    }
    Ok(())
}

fn empty_receipt(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
    disposition: DreamFreshDisposition,
) -> DreamFreshEvaluationReceipt {
    let evidence_digest = empty_evidence_digest(
        parent_manifest,
        c_selection,
        extension_manifest,
        verification_gate,
        disposition,
    );
    DreamFreshEvaluationReceipt {
        schema: SYM_RSI_001D_FRESH_EVALUATION_SCHEMA.into(),
        analysis_rule: SYM_RSI_001D_ANALYSIS_RULE.into(),
        experiment_id: extension_manifest.experiment_id.clone(),
        preregistration_digest: extension_manifest.preregistration_digest.clone(),
        subject_digest: extension_manifest.subject_digest.clone(),
        environment_digest: extension_manifest.environment_digest.clone(),
        parent_training_experiment_id: parent_manifest.experiment_id.clone(),
        parent_training_preregistration_digest: parent_manifest.preregistration_digest.clone(),
        c_selection_evidence_digest: c_selection.evidence_digest.clone(),
        grounded_dream_model_evidence_digest:
            verification_gate.grounded_dream_model_evidence_digest.clone(),
        verification_gate_evidence_digest: verification_gate.evidence_digest.clone(),
        c_policy_id: c_selection.selected_policy_id.clone(),
        d_policy_id: SYM_RSI_001_GROUNDED_DREAM_POLICY_ID.into(),
        quality_tolerance: SYM_RSI_001D_QUALITY_TOLERANCE,
        fresh_seeds_consumed: false,
        pair_count: 0,
        pairs: Vec::new(),
        domain_summaries: Vec::new(),
        macro_quality_delta: None,
        worst_domain_quality_delta: None,
        total_d_override_count: 0,
        total_d_prediction_count: 0,
        total_d_model_simulation_count: 0,
        zero_safety_constraint_violations: true,
        zero_authority_boundary_violations: true,
        generated_evidence_promoted: false,
        generic_compute_cost_is_environment_calls_only: true,
        efficiency_claim_authorized: false,
        disposition,
        evidence_digest,
    }
}

fn empty_evidence_digest(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
    disposition: DreamFreshDisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.fresh-d-vs-c.empty.v1\0");
    for value in [
        parent_manifest.experiment_id.as_str(),
        parent_manifest.preregistration_digest.as_str(),
        c_selection.evidence_digest.as_str(),
        extension_manifest.experiment_id.as_str(),
        extension_manifest.preregistration_digest.as_str(),
        verification_gate.evidence_digest.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[fresh_disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[allow(clippy::too_many_arguments)]
fn fresh_dream_evidence_digest(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
    dream_model_digest: &str,
    pairs: &[DreamFreshPairReceipt],
    summaries: &[DreamFreshDomainSummary],
    macro_quality_delta: f64,
    worst_domain_quality_delta: f64,
    total_overrides: usize,
    total_predictions: usize,
    total_simulations: usize,
    zero_safety: bool,
    zero_authority: bool,
    generated_evidence_promoted: bool,
    disposition: DreamFreshDisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.fresh-d-vs-c.v1\0");
    for value in [
        parent_manifest.experiment_id.as_str(),
        parent_manifest.preregistration_digest.as_str(),
        c_selection.evidence_digest.as_str(),
        extension_manifest.experiment_id.as_str(),
        extension_manifest.preregistration_digest.as_str(),
        extension_manifest.subject_digest.as_str(),
        extension_manifest.environment_digest.as_str(),
        verification_gate.evidence_digest.as_str(),
        dream_model_digest,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for pair in pairs {
        for value in [
            pair.domain_id.as_str(),
            pair.replay_selected.evidence_digest.as_str(),
            pair.replay_plus_dream.evidence_digest.as_str(),
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(&pair.seed.to_le_bytes());
        hasher.update(&pair.contrast.quality_delta.to_bits().to_le_bytes());
        hasher.update(&(pair.d_override_count as u64).to_le_bytes());
        hasher.update(&(pair.d_prediction_count as u64).to_le_bytes());
        hasher.update(&(pair.d_model_simulation_count as u64).to_le_bytes());
    }
    for summary in summaries {
        hasher.update(&(summary.domain_id.len() as u64).to_le_bytes());
        hasher.update(summary.domain_id.as_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
    }
    hasher.update(&macro_quality_delta.to_bits().to_le_bytes());
    hasher.update(&worst_domain_quality_delta.to_bits().to_le_bytes());
    hasher.update(&(total_overrides as u64).to_le_bytes());
    hasher.update(&(total_predictions as u64).to_le_bytes());
    hasher.update(&(total_simulations as u64).to_le_bytes());
    hasher.update(&[
        u8::from(zero_safety),
        u8::from(zero_authority),
        u8::from(generated_evidence_promoted),
        fresh_disposition_tag(disposition),
    ]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn fresh_disposition_tag(disposition: DreamFreshDisposition) -> u8 {
    match disposition {
        DreamFreshDisposition::NoDreamIntervention => 0,
        DreamFreshDisposition::BlockedByVerification => 1,
        DreamFreshDisposition::PositiveUnderProtocol => 2,
        DreamFreshDisposition::QualityNonInferiorityFailed => 3,
        DreamFreshDisposition::NoStrictQualityGain => 4,
        DreamFreshDisposition::IntegrityFailure => 5,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamFreshEvaluationError {
    Protocol(super::sym_rsi_dream_protocol::DreamProtocolError),
    CrossLineageMismatch,
    VerificationGateBindingMismatch,
    DreamModelDigestMismatch,
    SelectedPolicyNotCanonical(String),
    MissingDomain(String),
    Runner(FixtureRunnerError),
    GroundedDream(GroundedDreamError),
    Contrast(super::sym_rsi_experiment::ExperimentHarnessError),
    DuplicateFreshPair { domain_id: String, seed: u64 },
    FreshPairIdentityMismatch,
    FreshPairSetMismatch,
    MissingDomainPairs(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(domain: &str, delta: f64, overrides: usize) -> DreamFreshDomainSummary {
        DreamFreshDomainSummary {
            domain_id: domain.into(),
            pair_count: 4,
            c_mean_quality: 0.5,
            d_mean_quality: 0.5 + delta,
            mean_quality_delta: delta,
            d_override_count: overrides,
            d_prediction_count: 20,
            d_model_simulation_count: 100,
        }
    }

    #[test]
    fn positive_requires_strict_macro_quality_gain() {
        let summaries = vec![
            summary("a", 0.01, 1),
            summary("b", 0.00, 1),
            summary("c", 0.02, 1),
        ];
        assert_eq!(
            classify_fresh_dream_effects(&summaries, 0.01, 3, true, true, false, 0.02),
            DreamFreshDisposition::PositiveUnderProtocol
        );
    }

    #[test]
    fn environment_call_parity_is_not_a_positive_efficiency_result() {
        let summaries = vec![
            summary("a", 0.0, 1),
            summary("b", 0.0, 1),
            summary("c", 0.0, 1),
        ];
        assert_eq!(
            classify_fresh_dream_effects(&summaries, 0.0, 3, true, true, false, 0.02),
            DreamFreshDisposition::NoStrictQualityGain
        );
    }

    #[test]
    fn one_domain_below_tolerance_blocks_positive_macro_average() {
        let summaries = vec![
            summary("a", -0.03, 1),
            summary("b", 0.05, 1),
            summary("c", 0.05, 1),
        ];
        assert_eq!(
            classify_fresh_dream_effects(
                &summaries,
                0.023333333333333334,
                3,
                true,
                true,
                false,
                0.02,
            ),
            DreamFreshDisposition::QualityNonInferiorityFailed
        );
    }

    #[test]
    fn no_fresh_dream_override_is_not_called_improvement() {
        let summaries = vec![
            summary("a", 0.1, 0),
            summary("b", 0.1, 0),
            summary("c", 0.1, 0),
        ];
        assert_eq!(
            classify_fresh_dream_effects(&summaries, 0.1, 0, true, true, false, 0.02),
            DreamFreshDisposition::NoDreamIntervention
        );
    }

    #[test]
    fn generated_evidence_or_safety_failure_is_integrity_failure() {
        let summaries = vec![summary("a", 0.1, 1)];
        assert_eq!(
            classify_fresh_dream_effects(&summaries, 0.1, 1, true, true, true, 0.02),
            DreamFreshDisposition::IntegrityFailure
        );
        assert_eq!(
            classify_fresh_dream_effects(&summaries, 0.1, 1, false, true, false, 0.02),
            DreamFreshDisposition::IntegrityFailure
        );
    }
}
