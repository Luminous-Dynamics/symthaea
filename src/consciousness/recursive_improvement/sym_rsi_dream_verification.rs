// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sealed verification-replay gate for SYM-RSI-001D.
//!
//! This module is the only API intended to consume the frozen dream-verification
//! replay seeds (301-304). It is deliberately not invoked by unit tests.
//!
//! Verification remains replay-only: canonical fixed policies first generate an
//! observed branching corpus. The already-frozen C and D policies are then evaluated
//! against that exact corpus. D predictions may affect action choice, but unsupported
//! dream actions fail the gate rather than being simulated into empirical evidence.

use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family, FrozenReplayCorpus,
};
use super::sym_rsi_dream_protocol::{
    validate_canonical_sym_rsi_001d_manifest, DreamProtocolError,
    SYM_RSI_001D_EXPERIMENT_ID, SYM_RSI_001D_QUALITY_TOLERANCE,
};
use super::sym_rsi_experiment::{EvaluationSplit, SymRsiExperimentManifest};
use super::sym_rsi_fixtures::FixtureDomainKind;
use super::sym_rsi_grounded_dream::{build_grounded_dream_policy, GroundedDreamError};
use super::sym_rsi_replay_corpus::{
    merge_observed_traces, ReplayCorpusError, ReplayFixtureWorld, ReplayWorldEvaluation,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use super::sym_rsi_runner::{run_fixture_policy, FixtureRunnerError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001D_VERIFICATION_CORPUS_SCHEMA: &str =
    "symthaea.sym-rsi-001d.verification-corpus.v1";
pub const SYM_RSI_001D_VERIFICATION_GATE_SCHEMA: &str =
    "symthaea.sym-rsi-001d.verification-gate.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamVerificationCorpusReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub candidate_family_digest: String,
    pub split: EvaluationSplit,
    pub domain_count: usize,
    pub seed_count_per_domain: usize,
    pub world_count: usize,
    pub trajectory_count: usize,
    pub collector_policy_ids: Vec<String>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamVerificationCorpus {
    pub receipt: DreamVerificationCorpusReceipt,
    worlds: Vec<ReplayFixtureWorld>,
}

impl DreamVerificationCorpus {
    pub fn worlds(&self) -> &[ReplayFixtureWorld] {
        &self.worlds
    }
}

/// Explicitly acquire the sealed 301-304 verification replay corpus.
///
/// Calling this function consumes the verification partition. Unit tests must not call it.
pub fn acquire_dream_verification_corpus(
    extension_manifest: &SymRsiExperimentManifest,
) -> Result<DreamVerificationCorpus, DreamVerificationError> {
    validate_canonical_sym_rsi_001d_manifest(extension_manifest)
        .map_err(DreamVerificationError::Protocol)?;

    let candidates = canonical_fixed_hash_candidate_family();
    let mut worlds = Vec::new();
    let mut trajectory_bindings = Vec::new();

    for domain in FixtureDomainKind::ALL {
        let domain_spec = extension_manifest
            .domains
            .iter()
            .find(|spec| spec.domain_id == domain.id())
            .ok_or_else(|| DreamVerificationError::MissingDomain(domain.id().into()))?;

        for &seed in &domain_spec.seeds.held_out_replay {
            let mut traces = Vec::with_capacity(candidates.len());
            for candidate in &candidates {
                let mut policy = candidate.policy();
                let trace = run_fixture_policy(
                    extension_manifest,
                    domain,
                    EvaluationSplit::HeldOutReplay,
                    seed,
                    &mut policy,
                )
                .map_err(DreamVerificationError::Runner)?;
                trajectory_bindings.push(VerificationTrajectoryBinding {
                    domain_id: domain.id().into(),
                    seed,
                    policy_id: candidate.policy_id.clone(),
                    final_evidence_digest: trace.evidence_digest.clone(),
                    evaluator_calls: trace.evaluator_calls,
                });
                traces.push(trace);
            }
            worlds.push(
                merge_observed_traces(&traces).map_err(DreamVerificationError::ReplayCorpus)?,
            );
        }
    }

    let collector_policy_ids = candidates
        .iter()
        .map(|candidate| candidate.policy_id.clone())
        .collect::<Vec<_>>();
    let evidence_digest = verification_corpus_evidence_digest(
        extension_manifest,
        &collector_policy_ids,
        &trajectory_bindings,
    );

    Ok(DreamVerificationCorpus {
        receipt: DreamVerificationCorpusReceipt {
            schema: SYM_RSI_001D_VERIFICATION_CORPUS_SCHEMA.into(),
            experiment_id: extension_manifest.experiment_id.clone(),
            preregistration_digest: extension_manifest.preregistration_digest.clone(),
            subject_digest: extension_manifest.subject_digest.clone(),
            environment_digest: extension_manifest.environment_digest.clone(),
            candidate_family_digest: canonical_candidate_family_digest(),
            split: EvaluationSplit::HeldOutReplay,
            domain_count: FixtureDomainKind::ALL.len(),
            seed_count_per_domain: 4,
            world_count: worlds.len(),
            trajectory_count: trajectory_bindings.len(),
            collector_policy_ids,
            evidence_digest,
        },
        worlds,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DreamVerificationDecision {
    NoDreamIntervention,
    RejectedReplaySupport,
    RejectedQuality,
    BlockedEpistemicIntegrity,
    FreshDreamExecutionEligible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamVerificationDomainSummary {
    pub domain_id: String,
    pub world_count: usize,
    pub c_mean_quality: f64,
    pub d_mean_quality: f64,
    pub mean_quality_delta: f64,
    pub c_supported_steps: u64,
    pub d_supported_steps: u64,
    pub c_attempted_steps: u64,
    pub d_attempted_steps: u64,
    pub c_terminal_worlds: usize,
    pub d_terminal_worlds: usize,
    pub d_override_count: usize,
    pub d_prediction_count: usize,
    pub d_model_simulation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamVerificationGateReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub parent_training_experiment_id: String,
    pub parent_training_preregistration_digest: String,
    pub c_selection_evidence_digest: String,
    pub grounded_dream_model_evidence_digest: String,
    pub verification_corpus_evidence_digest: String,
    pub c_policy_id: String,
    pub d_policy_id: String,
    pub quality_tolerance: f64,
    pub domain_summaries: Vec<DreamVerificationDomainSummary>,
    pub macro_quality_delta: f64,
    pub total_d_override_count: usize,
    pub total_d_prediction_count: usize,
    pub total_d_model_simulation_count: usize,
    pub c_full_support: bool,
    pub d_full_support: bool,
    pub generated_evidence_promoted: bool,
    pub decision: DreamVerificationDecision,
    pub evidence_digest: String,
}

impl DreamVerificationGateReceipt {
    pub fn fresh_dream_execution_eligible(&self) -> bool {
        self.decision == DreamVerificationDecision::FreshDreamExecutionEligible
    }
}

/// Evaluate frozen C and grounded D on the already-acquired exact verification corpus.
///
/// This function does not execute the fixture transition function. It only traverses
/// recorded edges in `verification_corpus`.
pub fn validate_grounded_dream_on_verification(
    parent_training_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_corpus: &DreamVerificationCorpus,
) -> Result<DreamVerificationGateReceipt, DreamVerificationError> {
    validate_canonical_sym_rsi_001d_manifest(extension_manifest)
        .map_err(DreamVerificationError::Protocol)?;
    validate_cross_lineage_binding(parent_training_manifest, extension_manifest)?;
    validate_verification_corpus(extension_manifest, verification_corpus)?;

    let dream_policy = build_grounded_dream_policy(
        parent_training_manifest,
        parent_training_corpus,
        c_selection,
    )
    .map_err(DreamVerificationError::GroundedDream)?;
    if dream_policy.model().subject_digest != extension_manifest.subject_digest
        || dream_policy.model().environment_digest != extension_manifest.environment_digest
    {
        return Err(DreamVerificationError::DreamModelLineageMismatch);
    }

    let c_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|candidate| candidate.policy_id == c_selection.selected_policy_id)
        .ok_or_else(|| {
            DreamVerificationError::SelectedPolicyNotCanonical(
                c_selection.selected_policy_id.clone(),
            )
        })?;

    let mut summaries = Vec::with_capacity(FixtureDomainKind::ALL.len());
    let mut total_overrides = 0_usize;
    let mut total_predictions = 0_usize;
    let mut total_simulations = 0_usize;
    let mut any_generated_promotion = false;
    let mut c_full_support = true;
    let mut d_full_support = true;

    for domain in FixtureDomainKind::ALL {
        let worlds = verification_corpus
            .worlds()
            .iter()
            .filter(|world| world.domain == domain)
            .collect::<Vec<_>>();
        if worlds.is_empty() {
            return Err(DreamVerificationError::MissingDomainWorlds(domain.id().into()));
        }

        let mut c_quality_sum = 0.0;
        let mut d_quality_sum = 0.0;
        let mut c_supported_steps = 0_u64;
        let mut d_supported_steps = 0_u64;
        let mut c_attempted_steps = 0_u64;
        let mut d_attempted_steps = 0_u64;
        let mut c_terminal_worlds = 0_usize;
        let mut d_terminal_worlds = 0_usize;
        let mut d_override_count = 0_usize;
        let mut d_prediction_count = 0_usize;
        let mut d_model_simulation_count = 0_usize;

        for world in &worlds {
            let mut c_policy = c_spec.policy();
            let c_eval = world
                .evaluate(&mut c_policy)
                .map_err(DreamVerificationError::ReplayCorpus)?;

            let mut d_policy = dream_policy.clone();
            let d_eval = world
                .evaluate(&mut d_policy)
                .map_err(DreamVerificationError::ReplayCorpus)?;

            c_quality_sum += c_eval.best_solution_quality;
            d_quality_sum += d_eval.best_solution_quality;
            c_supported_steps += c_eval.supported_steps;
            d_supported_steps += d_eval.supported_steps;
            c_attempted_steps += c_eval.attempted_steps;
            d_attempted_steps += d_eval.attempted_steps;
            c_terminal_worlds += if c_eval.reached_terminal { 1 } else { 0 };
            d_terminal_worlds += if d_eval.reached_terminal { 1 } else { 0 };

            let overrides = d_policy
                .decision_log()
                .iter()
                .filter(|decision| decision.override_applied)
                .count();
            let predictions = d_policy
                .decision_log()
                .iter()
                .map(|decision| decision.predictions.len())
                .sum::<usize>();
            let simulations = d_policy
                .decision_log()
                .iter()
                .flat_map(|decision| decision.predictions.iter())
                .map(|prediction| prediction.model_simulation_count)
                .sum::<usize>();
            d_override_count += overrides;
            d_prediction_count += predictions;
            d_model_simulation_count += simulations;
            any_generated_promotion |= d_policy.generated_evidence_promoted();

            c_full_support &= full_support(&c_eval);
            d_full_support &= full_support(&d_eval);
        }

        let n = worlds.len() as f64;
        let c_mean_quality = c_quality_sum / n;
        let d_mean_quality = d_quality_sum / n;
        summaries.push(DreamVerificationDomainSummary {
            domain_id: domain.id().into(),
            world_count: worlds.len(),
            c_mean_quality,
            d_mean_quality,
            mean_quality_delta: d_mean_quality - c_mean_quality,
            c_supported_steps,
            d_supported_steps,
            c_attempted_steps,
            d_attempted_steps,
            c_terminal_worlds,
            d_terminal_worlds,
            d_override_count,
            d_prediction_count,
            d_model_simulation_count,
        });
        total_overrides += d_override_count;
        total_predictions += d_prediction_count;
        total_simulations += d_model_simulation_count;
    }

    let macro_quality_delta = summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / summaries.len() as f64;
    let decision = classify_verification(
        &summaries,
        macro_quality_delta,
        c_full_support,
        d_full_support,
        total_overrides,
        any_generated_promotion,
        SYM_RSI_001D_QUALITY_TOLERANCE,
    );
    let evidence_digest = verification_gate_evidence_digest(
        parent_training_manifest,
        c_selection,
        extension_manifest,
        verification_corpus,
        dream_policy.model().evidence_digest.as_str(),
        &summaries,
        macro_quality_delta,
        total_overrides,
        total_predictions,
        total_simulations,
        c_full_support,
        d_full_support,
        any_generated_promotion,
        decision,
    );

    Ok(DreamVerificationGateReceipt {
        schema: SYM_RSI_001D_VERIFICATION_GATE_SCHEMA.into(),
        experiment_id: extension_manifest.experiment_id.clone(),
        preregistration_digest: extension_manifest.preregistration_digest.clone(),
        subject_digest: extension_manifest.subject_digest.clone(),
        environment_digest: extension_manifest.environment_digest.clone(),
        parent_training_experiment_id: parent_training_manifest.experiment_id.clone(),
        parent_training_preregistration_digest:
            parent_training_manifest.preregistration_digest.clone(),
        c_selection_evidence_digest: c_selection.evidence_digest.clone(),
        grounded_dream_model_evidence_digest: dream_policy.model().evidence_digest.clone(),
        verification_corpus_evidence_digest:
            verification_corpus.receipt.evidence_digest.clone(),
        c_policy_id: c_selection.selected_policy_id.clone(),
        d_policy_id: super::sym_rsi_grounded_dream::SYM_RSI_001_GROUNDED_DREAM_POLICY_ID.into(),
        quality_tolerance: SYM_RSI_001D_QUALITY_TOLERANCE,
        domain_summaries: summaries,
        macro_quality_delta,
        total_d_override_count: total_overrides,
        total_d_prediction_count: total_predictions,
        total_d_model_simulation_count: total_simulations,
        c_full_support,
        d_full_support,
        generated_evidence_promoted: any_generated_promotion,
        decision,
        evidence_digest,
    })
}

fn full_support(evaluation: &ReplayWorldEvaluation) -> bool {
    evaluation.unsupported_actions == 0
        && evaluation.replay_coverage == 1.0
        && evaluation.reached_terminal
}

fn classify_verification(
    summaries: &[DreamVerificationDomainSummary],
    macro_quality_delta: f64,
    c_full_support: bool,
    d_full_support: bool,
    total_overrides: usize,
    generated_evidence_promoted: bool,
    tolerance: f64,
) -> DreamVerificationDecision {
    if summaries.is_empty()
        || !macro_quality_delta.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || generated_evidence_promoted
    {
        return DreamVerificationDecision::BlockedEpistemicIntegrity;
    }
    if !c_full_support || !d_full_support {
        return DreamVerificationDecision::RejectedReplaySupport;
    }
    if total_overrides == 0 {
        return DreamVerificationDecision::NoDreamIntervention;
    }
    if macro_quality_delta < -tolerance
        || summaries
            .iter()
            .any(|summary| !summary.mean_quality_delta.is_finite()
                || summary.mean_quality_delta < -tolerance)
    {
        return DreamVerificationDecision::RejectedQuality;
    }
    DreamVerificationDecision::FreshDreamExecutionEligible
}

fn validate_cross_lineage_binding(
    parent: &SymRsiExperimentManifest,
    extension: &SymRsiExperimentManifest,
) -> Result<(), DreamVerificationError> {
    if parent.experiment_id != "SYM-RSI-001"
        || extension.experiment_id != SYM_RSI_001D_EXPERIMENT_ID
        || parent.subject_digest != extension.subject_digest
        || parent.environment_digest != extension.environment_digest
    {
        return Err(DreamVerificationError::CrossLineageMismatch);
    }
    Ok(())
}

fn validate_verification_corpus(
    manifest: &SymRsiExperimentManifest,
    corpus: &DreamVerificationCorpus,
) -> Result<(), DreamVerificationError> {
    let receipt = &corpus.receipt;
    if receipt.schema != SYM_RSI_001D_VERIFICATION_CORPUS_SCHEMA
        || receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
        || receipt.candidate_family_digest != canonical_candidate_family_digest()
        || receipt.split != EvaluationSplit::HeldOutReplay
        || receipt.evidence_digest.trim().is_empty()
    {
        return Err(DreamVerificationError::VerificationCorpusBindingMismatch);
    }

    let canonical_ids = canonical_fixed_hash_candidate_family()
        .into_iter()
        .map(|candidate| candidate.policy_id)
        .collect::<Vec<_>>();
    if receipt.collector_policy_ids != canonical_ids {
        return Err(DreamVerificationError::VerificationCollectorMismatch);
    }

    let mut expected = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.held_out_replay {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = BTreeSet::new();
    for world in corpus.worlds() {
        if world.split != EvaluationSplit::HeldOutReplay {
            return Err(DreamVerificationError::VerificationCorpusBindingMismatch);
        }
        let key = (world.domain.id().to_owned(), world.seed);
        if !observed.insert(key) {
            return Err(DreamVerificationError::DuplicateVerificationWorld);
        }
    }

    let family_len = canonical_fixed_hash_candidate_family().len();
    if observed != expected
        || receipt.domain_count != manifest.domains.len()
        || receipt.seed_count_per_domain != 4
        || receipt.world_count != expected.len()
        || receipt.world_count != corpus.worlds().len()
        || receipt.trajectory_count != expected.len() * family_len
    {
        return Err(DreamVerificationError::VerificationCorpusIncomplete);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerificationTrajectoryBinding {
    domain_id: String,
    seed: u64,
    policy_id: String,
    final_evidence_digest: String,
    evaluator_calls: u64,
}

fn verification_corpus_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    collector_policy_ids: &[String],
    trajectories: &[VerificationTrajectoryBinding],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.verification-corpus.v1\0");
    let family_digest = canonical_candidate_family_digest();
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        family_digest.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for policy_id in collector_policy_ids {
        hasher.update(&(policy_id.len() as u64).to_le_bytes());
        hasher.update(policy_id.as_bytes());
    }
    for trajectory in trajectories {
        for value in [
            trajectory.domain_id.as_str(),
            trajectory.policy_id.as_str(),
            trajectory.final_evidence_digest.as_str(),
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(&trajectory.seed.to_le_bytes());
        hasher.update(&trajectory.evaluator_calls.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[allow(clippy::too_many_arguments)]
fn verification_gate_evidence_digest(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    corpus: &DreamVerificationCorpus,
    dream_model_digest: &str,
    summaries: &[DreamVerificationDomainSummary],
    macro_quality_delta: f64,
    total_overrides: usize,
    total_predictions: usize,
    total_simulations: usize,
    c_full_support: bool,
    d_full_support: bool,
    generated_evidence_promoted: bool,
    decision: DreamVerificationDecision,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.verification-gate.v1\0");
    for value in [
        parent_manifest.experiment_id.as_str(),
        parent_manifest.preregistration_digest.as_str(),
        c_selection.evidence_digest.as_str(),
        extension_manifest.experiment_id.as_str(),
        extension_manifest.preregistration_digest.as_str(),
        extension_manifest.subject_digest.as_str(),
        extension_manifest.environment_digest.as_str(),
        corpus.receipt.evidence_digest.as_str(),
        dream_model_digest,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for summary in summaries {
        hasher.update(&(summary.domain_id.len() as u64).to_le_bytes());
        hasher.update(summary.domain_id.as_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
        hasher.update(&(summary.d_override_count as u64).to_le_bytes());
        hasher.update(&(summary.d_prediction_count as u64).to_le_bytes());
        hasher.update(&(summary.d_model_simulation_count as u64).to_le_bytes());
    }
    hasher.update(&macro_quality_delta.to_bits().to_le_bytes());
    hasher.update(&(total_overrides as u64).to_le_bytes());
    hasher.update(&(total_predictions as u64).to_le_bytes());
    hasher.update(&(total_simulations as u64).to_le_bytes());
    hasher.update(&[
        u8::from(c_full_support),
        u8::from(d_full_support),
        u8::from(generated_evidence_promoted),
        decision_tag(decision),
    ]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn decision_tag(decision: DreamVerificationDecision) -> u8 {
    match decision {
        DreamVerificationDecision::NoDreamIntervention => 0,
        DreamVerificationDecision::RejectedReplaySupport => 1,
        DreamVerificationDecision::RejectedQuality => 2,
        DreamVerificationDecision::BlockedEpistemicIntegrity => 3,
        DreamVerificationDecision::FreshDreamExecutionEligible => 4,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamVerificationError {
    Protocol(DreamProtocolError),
    CrossLineageMismatch,
    MissingDomain(String),
    MissingDomainWorlds(String),
    Runner(FixtureRunnerError),
    ReplayCorpus(ReplayCorpusError),
    GroundedDream(GroundedDreamError),
    DreamModelLineageMismatch,
    SelectedPolicyNotCanonical(String),
    VerificationCorpusBindingMismatch,
    VerificationCollectorMismatch,
    DuplicateVerificationWorld,
    VerificationCorpusIncomplete,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(domain: &str, delta: f64, overrides: usize) -> DreamVerificationDomainSummary {
        DreamVerificationDomainSummary {
            domain_id: domain.into(),
            world_count: 4,
            c_mean_quality: 0.5,
            d_mean_quality: 0.5 + delta,
            mean_quality_delta: delta,
            c_supported_steps: 10,
            d_supported_steps: 10,
            c_attempted_steps: 10,
            d_attempted_steps: 10,
            c_terminal_worlds: 4,
            d_terminal_worlds: 4,
            d_override_count: overrides,
            d_prediction_count: 20,
            d_model_simulation_count: 100,
        }
    }

    #[test]
    fn verification_requires_an_actual_dream_intervention() {
        let summaries = vec![
            summary("a", 0.0, 0),
            summary("b", 0.0, 0),
            summary("c", 0.0, 0),
        ];
        assert_eq!(
            classify_verification(&summaries, 0.0, true, true, 0, false, 0.02),
            DreamVerificationDecision::NoDreamIntervention
        );
    }

    #[test]
    fn unsupported_dream_path_blocks_fresh_execution() {
        let summaries = vec![summary("a", 0.1, 1)];
        assert_eq!(
            classify_verification(&summaries, 0.1, true, false, 1, false, 0.02),
            DreamVerificationDecision::RejectedReplaySupport
        );
    }

    #[test]
    fn quality_regression_blocks_even_with_dream_overrides() {
        let summaries = vec![
            summary("a", -0.03, 1),
            summary("b", 0.02, 1),
            summary("c", 0.02, 1),
        ];
        assert_eq!(
            classify_verification(&summaries, 0.0033333333333333335, true, true, 3, false, 0.02),
            DreamVerificationDecision::RejectedQuality
        );
    }

    #[test]
    fn generated_evidence_promotion_is_integrity_failure() {
        let summaries = vec![summary("a", 0.1, 1)];
        assert_eq!(
            classify_verification(&summaries, 0.1, true, true, 1, true, 0.02),
            DreamVerificationDecision::BlockedEpistemicIntegrity
        );
    }

    #[test]
    fn noninferior_supported_intervention_may_reach_fresh_gate() {
        let summaries = vec![
            summary("a", -0.01, 1),
            summary("b", 0.00, 0),
            summary("c", 0.02, 2),
        ];
        assert_eq!(
            classify_verification(
                &summaries,
                0.0033333333333333335,
                true,
                true,
                3,
                false,
                0.02,
            ),
            DreamVerificationDecision::FreshDreamExecutionEligible
        );
    }
}
