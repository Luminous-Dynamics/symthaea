// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secondary OOD follow-up for SYM-RSI-001D.
//!
//! Seeds 1201-1204 remain inaccessible until a complete fresh D-vs-C receipt has
//! already been frozen. OOD results are descriptive only: they cannot change the
//! primary fresh disposition and cannot retrain the dream model.

use super::sym_rsi_candidate_family::{
    canonical_fixed_hash_candidate_family, FrozenReplayCorpus,
};
use super::sym_rsi_dream_fresh_evaluation::{
    DreamFreshDisposition, DreamFreshEvaluationReceipt,
    SYM_RSI_001D_FRESH_EVALUATION_SCHEMA,
};
use super::sym_rsi_dream_protocol::{
    validate_canonical_sym_rsi_001d_manifest, SYM_RSI_001D_ANALYSIS_RULE,
    SYM_RSI_001D_EXPERIMENT_ID, SYM_RSI_001D_QUALITY_TOLERANCE,
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
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001D_OOD_EVALUATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.ood-d-vs-c.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DreamOodDisposition {
    DreamBetter,
    SimilarWithinTolerance,
    DreamWorse,
    NoDreamIntervention,
    IntegrityFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamOodPairReceipt {
    pub domain_id: String,
    pub seed: u64,
    pub replay_selected: SymRsiRunReceipt,
    pub replay_plus_dream: SymRsiRunReceipt,
    pub contrast: PrimaryContrastReceipt,
    pub d_override_count: usize,
    pub d_prediction_count: usize,
    pub d_model_simulation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamOodDomainSummary {
    pub domain_id: String,
    pub pair_count: usize,
    pub mean_quality_delta: f64,
    pub d_override_count: usize,
    pub d_prediction_count: usize,
    pub d_model_simulation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamOodEvaluationReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub fresh_primary_evidence_digest: String,
    pub primary_disposition: DreamFreshDisposition,
    pub primary_disposition_unchanged: bool,
    pub grounded_dream_model_evidence_digest: String,
    pub c_policy_id: String,
    pub d_policy_id: String,
    pub quality_tolerance: f64,
    pub ood_seeds_consumed: bool,
    pub pair_count: usize,
    pub pairs: Vec<DreamOodPairReceipt>,
    pub domain_summaries: Vec<DreamOodDomainSummary>,
    pub macro_quality_delta: f64,
    pub worst_domain_quality_delta: f64,
    pub total_d_override_count: usize,
    pub total_d_prediction_count: usize,
    pub total_d_model_simulation_count: usize,
    pub generated_evidence_promoted: bool,
    pub efficiency_claim_authorized: bool,
    pub disposition: DreamOodDisposition,
    pub evidence_digest: String,
}

/// Execute the secondary OOD comparison only after a complete fresh receipt exists.
///
/// This function never changes or recomputes the primary D-vs-C disposition.
pub fn run_ood_d_vs_c(
    parent_training_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    fresh_receipt: &DreamFreshEvaluationReceipt,
) -> Result<DreamOodEvaluationReceipt, DreamOodEvaluationError> {
    validate_canonical_sym_rsi_001d_manifest(extension_manifest)
        .map_err(DreamOodEvaluationError::Protocol)?;
    validate_cross_lineage(parent_training_manifest, extension_manifest)?;
    validate_fresh_receipt(extension_manifest, c_selection, fresh_receipt)?;

    let dream_template = build_grounded_dream_policy(
        parent_training_manifest,
        parent_training_corpus,
        c_selection,
    )
    .map_err(DreamOodEvaluationError::GroundedDream)?;
    if dream_template.model().evidence_digest
        != fresh_receipt.grounded_dream_model_evidence_digest
    {
        return Err(DreamOodEvaluationError::DreamModelDigestMismatch);
    }

    let c_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|candidate| candidate.policy_id == c_selection.selected_policy_id)
        .ok_or_else(|| {
            DreamOodEvaluationError::SelectedPolicyNotCanonical(
                c_selection.selected_policy_id.clone(),
            )
        })?;

    let mut pairs = Vec::new();
    for domain in FixtureDomainKind::ALL {
        let spec = extension_manifest
            .domains
            .iter()
            .find(|spec| spec.domain_id == domain.id())
            .ok_or_else(|| DreamOodEvaluationError::MissingDomain(domain.id().into()))?;

        for &seed in &spec.seeds.out_of_distribution {
            let mut c_policy = c_spec.policy();
            let mut d_policy = dream_template.clone();
            let c_trace = run_fixture_policy(
                extension_manifest,
                domain,
                EvaluationSplit::OutOfDistribution,
                seed,
                &mut c_policy,
            )
            .map_err(DreamOodEvaluationError::Runner)?;
            let d_trace = run_fixture_policy(
                extension_manifest,
                domain,
                EvaluationSplit::OutOfDistribution,
                seed,
                &mut d_policy,
            )
            .map_err(DreamOodEvaluationError::Runner)?;

            let override_count = d_policy
                .decision_log()
                .iter()
                .filter(|decision| decision.override_applied)
                .count();
            let prediction_count = d_policy
                .decision_log()
                .iter()
                .map(|decision| decision.predictions.len())
                .sum::<usize>();
            let simulation_count =
                prediction_count * d_policy.model().config.counterfactual_count;
            let promoted = d_policy.generated_evidence_promoted();

            let c_receipt = build_fixture_receipt(
                extension_manifest,
                ExperimentArm::CExactReplayPolicyImprovement,
                &c_trace,
                ReceiptDiagnostics::default(),
            )
            .map_err(DreamOodEvaluationError::Runner)?;
            let d_receipt = build_fixture_receipt(
                extension_manifest,
                ExperimentArm::DReplayPlusGroundedDreaming,
                &d_trace,
                ReceiptDiagnostics {
                    policy_churn_rate: if d_trace.evaluator_calls == 0 {
                        0.0
                    } else {
                        override_count as f64 / d_trace.evaluator_calls as f64
                    },
                    generated_evidence_promoted: promoted,
                    ..ReceiptDiagnostics::default()
                },
            )
            .map_err(DreamOodEvaluationError::Runner)?;
            let contrast = build_primary_contrast(
                PrimaryContrastKind::ReplayDreamVsReplay,
                &c_receipt,
                &d_receipt,
            )
            .map_err(DreamOodEvaluationError::Contrast)?;

            pairs.push(DreamOodPairReceipt {
                domain_id: domain.id().into(),
                seed,
                replay_selected: c_receipt,
                replay_plus_dream: d_receipt,
                contrast,
                d_override_count: override_count,
                d_prediction_count: prediction_count,
                d_model_simulation_count: simulation_count,
            });
        }
    }

    validate_complete_ood_pair_set(extension_manifest, &pairs)?;
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
    let total_d_override_count = pairs.iter().map(|p| p.d_override_count).sum();
    let total_d_prediction_count = pairs.iter().map(|p| p.d_prediction_count).sum();
    let total_d_model_simulation_count =
        pairs.iter().map(|p| p.d_model_simulation_count).sum();
    let generated_evidence_promoted = pairs
        .iter()
        .any(|pair| pair.replay_plus_dream.generated_evidence_promoted);

    let disposition = classify_ood(
        &domain_summaries,
        macro_quality_delta,
        total_d_override_count,
        generated_evidence_promoted,
        SYM_RSI_001D_QUALITY_TOLERANCE,
    );
    let evidence_digest = ood_evidence_digest(
        extension_manifest,
        fresh_receipt,
        dream_template.model().evidence_digest.as_str(),
        &pairs,
        &domain_summaries,
        macro_quality_delta,
        worst_domain_quality_delta,
        total_d_override_count,
        total_d_prediction_count,
        total_d_model_simulation_count,
        generated_evidence_promoted,
        disposition,
    );

    Ok(DreamOodEvaluationReceipt {
        schema: SYM_RSI_001D_OOD_EVALUATION_SCHEMA.into(),
        experiment_id: extension_manifest.experiment_id.clone(),
        preregistration_digest: extension_manifest.preregistration_digest.clone(),
        subject_digest: extension_manifest.subject_digest.clone(),
        environment_digest: extension_manifest.environment_digest.clone(),
        fresh_primary_evidence_digest: fresh_receipt.evidence_digest.clone(),
        primary_disposition: fresh_receipt.disposition,
        primary_disposition_unchanged: true,
        grounded_dream_model_evidence_digest:
            dream_template.model().evidence_digest.clone(),
        c_policy_id: c_selection.selected_policy_id.clone(),
        d_policy_id: SYM_RSI_001_GROUNDED_DREAM_POLICY_ID.into(),
        quality_tolerance: SYM_RSI_001D_QUALITY_TOLERANCE,
        ood_seeds_consumed: true,
        pair_count: pairs.len(),
        pairs,
        domain_summaries,
        macro_quality_delta,
        worst_domain_quality_delta,
        total_d_override_count,
        total_d_prediction_count,
        total_d_model_simulation_count,
        generated_evidence_promoted,
        efficiency_claim_authorized: false,
        disposition,
        evidence_digest,
    })
}

fn classify_ood(
    summaries: &[DreamOodDomainSummary],
    macro_delta: f64,
    overrides: usize,
    generated_evidence_promoted: bool,
    tolerance: f64,
) -> DreamOodDisposition {
    if summaries.is_empty()
        || !macro_delta.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || generated_evidence_promoted
        || summaries.iter().any(|summary| !summary.mean_quality_delta.is_finite())
    {
        return DreamOodDisposition::IntegrityFailure;
    }
    if overrides == 0 {
        return DreamOodDisposition::NoDreamIntervention;
    }
    if macro_delta > tolerance
        && summaries
            .iter()
            .all(|summary| summary.mean_quality_delta >= -tolerance)
    {
        DreamOodDisposition::DreamBetter
    } else if macro_delta < -tolerance
        || summaries
            .iter()
            .any(|summary| summary.mean_quality_delta < -tolerance)
    {
        DreamOodDisposition::DreamWorse
    } else {
        DreamOodDisposition::SimilarWithinTolerance
    }
}

fn validate_fresh_receipt(
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    receipt: &DreamFreshEvaluationReceipt,
) -> Result<(), DreamOodEvaluationError> {
    if receipt.schema != SYM_RSI_001D_FRESH_EVALUATION_SCHEMA
        || receipt.analysis_rule != SYM_RSI_001D_ANALYSIS_RULE
        || receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
        || receipt.c_selection_evidence_digest != c_selection.evidence_digest
        || receipt.c_policy_id != c_selection.selected_policy_id
        || receipt.d_policy_id != SYM_RSI_001_GROUNDED_DREAM_POLICY_ID
        || receipt.quality_tolerance != SYM_RSI_001D_QUALITY_TOLERANCE
        || !receipt.fresh_seeds_consumed
        || receipt.pair_count != 12
        || receipt.pairs.len() != 12
        || receipt.domain_summaries.len() != FixtureDomainKind::ALL.len()
        || receipt.generated_evidence_promoted
        || !receipt.generic_compute_cost_is_environment_calls_only
        || receipt.efficiency_claim_authorized
        || receipt.evidence_digest.trim().is_empty()
        || matches!(
            receipt.disposition,
            DreamFreshDisposition::NoCandidatePromotion
                | DreamFreshDisposition::BlockedByHoldout
                | DreamFreshDisposition::IntegrityFailure
        )
    {
        return Err(DreamOodEvaluationError::FreshReceiptNotEligibleForOod);
    }

    let mut expected = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.fresh_execution {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let observed = receipt
        .pairs
        .iter()
        .map(|pair| (pair.domain_id.clone(), pair.seed))
        .collect::<BTreeSet<_>>();
    if observed != expected || observed.len() != receipt.pairs.len() {
        return Err(DreamOodEvaluationError::FreshReceiptNotEligibleForOod);
    }
    Ok(())
}

fn validate_complete_ood_pair_set(
    manifest: &SymRsiExperimentManifest,
    pairs: &[DreamOodPairReceipt],
) -> Result<(), DreamOodEvaluationError> {
    let mut expected = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.out_of_distribution {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = BTreeSet::new();
    for pair in pairs {
        let key = (pair.domain_id.clone(), pair.seed);
        if !observed.insert(key.clone()) {
            return Err(DreamOodEvaluationError::DuplicateOodPair {
                domain_id: key.0,
                seed: key.1,
            });
        }
        if pair.replay_selected.split != EvaluationSplit::OutOfDistribution
            || pair.replay_plus_dream.split != EvaluationSplit::OutOfDistribution
            || pair.replay_selected.domain_id != pair.domain_id
            || pair.replay_plus_dream.domain_id != pair.domain_id
            || pair.replay_selected.seed != pair.seed
            || pair.replay_plus_dream.seed != pair.seed
        {
            return Err(DreamOodEvaluationError::OodPairIdentityMismatch);
        }
    }
    if observed != expected {
        return Err(DreamOodEvaluationError::OodPairSetMismatch);
    }
    Ok(())
}

fn summarize_domains(
    pairs: &[DreamOodPairReceipt],
) -> Result<Vec<DreamOodDomainSummary>, DreamOodEvaluationError> {
    let mut summaries = Vec::new();
    for domain in FixtureDomainKind::ALL {
        let domain_pairs = pairs
            .iter()
            .filter(|pair| pair.domain_id == domain.id())
            .collect::<Vec<_>>();
        if domain_pairs.is_empty() {
            return Err(DreamOodEvaluationError::MissingDomainPairs(domain.id().into()));
        }
        let n = domain_pairs.len() as f64;
        summaries.push(DreamOodDomainSummary {
            domain_id: domain.id().into(),
            pair_count: domain_pairs.len(),
            mean_quality_delta: domain_pairs
                .iter()
                .map(|pair| pair.contrast.quality_delta)
                .sum::<f64>()
                / n,
            d_override_count: domain_pairs.iter().map(|p| p.d_override_count).sum(),
            d_prediction_count: domain_pairs.iter().map(|p| p.d_prediction_count).sum(),
            d_model_simulation_count: domain_pairs
                .iter()
                .map(|p| p.d_model_simulation_count)
                .sum(),
        });
    }
    Ok(summaries)
}

fn validate_cross_lineage(
    parent: &SymRsiExperimentManifest,
    extension: &SymRsiExperimentManifest,
) -> Result<(), DreamOodEvaluationError> {
    if parent.experiment_id != "SYM-RSI-001"
        || extension.experiment_id != SYM_RSI_001D_EXPERIMENT_ID
        || parent.subject_digest != extension.subject_digest
        || parent.environment_digest != extension.environment_digest
    {
        return Err(DreamOodEvaluationError::CrossLineageMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ood_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    fresh_receipt: &DreamFreshEvaluationReceipt,
    dream_model_digest: &str,
    pairs: &[DreamOodPairReceipt],
    summaries: &[DreamOodDomainSummary],
    macro_delta: f64,
    worst_delta: f64,
    overrides: usize,
    predictions: usize,
    simulations: usize,
    promoted: bool,
    disposition: DreamOodDisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.ood-d-vs-c.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        fresh_receipt.evidence_digest.as_str(),
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
        hasher.update(&(pair.d_model_simulation_count as u64).to_le_bytes());
    }
    for summary in summaries {
        hasher.update(&(summary.domain_id.len() as u64).to_le_bytes());
        hasher.update(summary.domain_id.as_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
    }
    hasher.update(&macro_delta.to_bits().to_le_bytes());
    hasher.update(&worst_delta.to_bits().to_le_bytes());
    hasher.update(&(overrides as u64).to_le_bytes());
    hasher.update(&(predictions as u64).to_le_bytes());
    hasher.update(&(simulations as u64).to_le_bytes());
    hasher.update(&[u8::from(promoted), ood_disposition_tag(disposition)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn ood_disposition_tag(disposition: DreamOodDisposition) -> u8 {
    match disposition {
        DreamOodDisposition::DreamBetter => 0,
        DreamOodDisposition::SimilarWithinTolerance => 1,
        DreamOodDisposition::DreamWorse => 2,
        DreamOodDisposition::NoDreamIntervention => 3,
        DreamOodDisposition::IntegrityFailure => 4,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamOodEvaluationError {
    Protocol(super::sym_rsi_dream_protocol::DreamProtocolError),
    CrossLineageMismatch,
    FreshReceiptNotEligibleForOod,
    DreamModelDigestMismatch,
    SelectedPolicyNotCanonical(String),
    MissingDomain(String),
    MissingDomainPairs(String),
    Runner(FixtureRunnerError),
    GroundedDream(GroundedDreamError),
    Contrast(super::sym_rsi_experiment::ExperimentHarnessError),
    DuplicateOodPair { domain_id: String, seed: u64 },
    OodPairIdentityMismatch,
    OodPairSetMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(domain: &str, delta: f64, overrides: usize) -> DreamOodDomainSummary {
        DreamOodDomainSummary {
            domain_id: domain.into(),
            pair_count: 4,
            mean_quality_delta: delta,
            d_override_count: overrides,
            d_prediction_count: 20,
            d_model_simulation_count: 100,
        }
    }

    #[test]
    fn ood_positive_is_descriptive_and_requires_no_domain_regression() {
        let summaries = vec![
            summary("a", 0.04, 1),
            summary("b", 0.03, 1),
            summary("c", 0.01, 1),
        ];
        assert_eq!(
            classify_ood(&summaries, 0.02666666666666667, 3, false, 0.02),
            DreamOodDisposition::DreamBetter
        );
    }

    #[test]
    fn ood_regression_is_reported_not_used_to_rewrite_primary() {
        let summaries = vec![
            summary("a", -0.03, 1),
            summary("b", 0.02, 1),
            summary("c", 0.02, 1),
        ];
        assert_eq!(
            classify_ood(&summaries, 0.0033333333333333335, 3, false, 0.02),
            DreamOodDisposition::DreamWorse
        );
    }

    #[test]
    fn no_ood_override_is_descriptive_no_intervention() {
        let summaries = vec![summary("a", 0.1, 0)];
        assert_eq!(
            classify_ood(&summaries, 0.1, 0, false, 0.02),
            DreamOodDisposition::NoDreamIntervention
        );
    }

    #[test]
    fn generated_evidence_promotion_blocks_interpretation() {
        let summaries = vec![summary("a", 0.1, 1)];
        assert_eq!(
            classify_ood(&summaries, 0.1, 1, true, 0.02),
            DreamOodDisposition::IntegrityFailure
        );
    }
}
