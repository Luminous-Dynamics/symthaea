// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical training selection and held-out replay validation for SYM-RSI-001.
//!
//! Training replay chooses a policy. Held-out replay may validate or reject that
//! frozen choice, but it can never influence which policy was chosen. Fresh and OOD
//! execution remain untouched by this module.

use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family, FrozenReplayCorpus,
    ReplayCorpusAcquisitionReceipt, SYM_RSI_001_INCUMBENT_POLICY_ID,
};
use super::sym_rsi_experiment::{EvaluationSplit, SymRsiExperimentManifest};
use super::sym_rsi_replay_corpus::{
    score_policy_on_replay_worlds, ReplayCorpusError, ReplayCorpusEvaluation,
};
use super::sym_rsi_replay_selection::{
    select_fixed_hash_policy_from_replay, FixedHashCandidateSpec, ReplaySelectionError,
    ReplaySelectionReceipt, SYM_RSI_001_REPLAY_SELECTION_SCHEMA,
};
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001_HOLDOUT_GATE_SCHEMA: &str =
    "symthaea.sym-rsi-001.held-out-replay-gate.v1";

/// The only SYM-RSI-001 training-selection entry point. It always evaluates the
/// complete frozen candidate family against the complete frozen training corpus.
pub fn select_canonical_training_candidate(
    manifest: &SymRsiExperimentManifest,
    training_corpus: &FrozenReplayCorpus,
) -> Result<ReplaySelectionReceipt, HoldoutGateError> {
    validate_corpus_binding(manifest, training_corpus, EvaluationSplit::TrainingReplay)?;
    let candidates = canonical_fixed_hash_candidate_family();
    let receipt = select_fixed_hash_policy_from_replay(
        manifest,
        SYM_RSI_001_INCUMBENT_POLICY_ID,
        &candidates,
        training_corpus.worlds(),
    )
    .map_err(HoldoutGateError::ReplaySelection)?;
    validate_canonical_selection_receipt(manifest, &receipt)?;
    Ok(receipt)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoldoutGateDecision {
    /// Training replay selected the incumbent, so there is no candidate promotion.
    NoChange,
    /// Candidate survived independent held-out replay and may proceed to fresh execution.
    FreshExecutionEligible,
    /// Selected candidate lacked complete historical support on held-out replay.
    RejectedReplaySupport,
    /// Selected candidate degraded held-out solution quality beyond the frozen tolerance.
    RejectedHeldOutQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeldOutReplayGateReceipt {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub candidate_family_digest: String,
    pub training_selection_evidence_digest: String,
    pub held_out_corpus_evidence_digest: String,
    pub incumbent_policy_id: String,
    pub selected_policy_id: String,
    pub held_out_quality_tolerance: f64,
    pub incumbent: ReplayCorpusEvaluation,
    pub selected: ReplayCorpusEvaluation,
    pub held_out_quality_delta: f64,
    pub evaluator_call_delta: i128,
    pub incumbent_full_support: bool,
    pub selected_full_support: bool,
    pub decision: HoldoutGateDecision,
    pub evidence_digest: String,
}

impl HeldOutReplayGateReceipt {
    pub fn fresh_execution_eligible(&self) -> bool {
        self.decision == HoldoutGateDecision::FreshExecutionEligible
    }
}

pub fn validate_selected_candidate_on_holdout(
    manifest: &SymRsiExperimentManifest,
    training_selection: &ReplaySelectionReceipt,
    held_out_corpus: &FrozenReplayCorpus,
) -> Result<HeldOutReplayGateReceipt, HoldoutGateError> {
    validate_canonical_selection_receipt(manifest, training_selection)?;
    validate_corpus_binding(manifest, held_out_corpus, EvaluationSplit::HeldOutReplay)?;

    let incumbent_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|spec| spec.policy_id == SYM_RSI_001_INCUMBENT_POLICY_ID)
        .expect("canonical family always contains incumbent");
    let selected_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|spec| spec.policy_id == training_selection.selected_policy_id)
        .ok_or_else(|| HoldoutGateError::SelectedPolicyNotCanonical(
            training_selection.selected_policy_id.clone(),
        ))?;

    let incumbent = score_policy_on_replay_worlds(
        &incumbent_spec.policy(),
        held_out_corpus.worlds(),
    )
    .map_err(HoldoutGateError::ReplayCorpus)?;
    let selected = score_policy_on_replay_worlds(
        &selected_spec.policy(),
        held_out_corpus.worlds(),
    )
    .map_err(HoldoutGateError::ReplayCorpus)?;

    let incumbent_full_support = full_support(&incumbent);
    let selected_full_support = full_support(&selected);
    let held_out_quality_delta =
        selected.mean_best_solution_quality - incumbent.mean_best_solution_quality;
    let evaluator_call_delta =
        selected.attempted_steps as i128 - incumbent.attempted_steps as i128;

    let decision = if selected_spec.policy_id == incumbent_spec.policy_id {
        HoldoutGateDecision::NoChange
    } else if !incumbent_full_support || !selected_full_support {
        HoldoutGateDecision::RejectedReplaySupport
    } else if selected.mean_best_solution_quality + manifest.held_out_quality_tolerance
        < incumbent.mean_best_solution_quality
    {
        HoldoutGateDecision::RejectedHeldOutQuality
    } else {
        HoldoutGateDecision::FreshExecutionEligible
    };

    let evidence_digest = holdout_evidence_digest(
        manifest,
        training_selection,
        &held_out_corpus.receipt,
        &incumbent,
        &selected,
        decision,
    );

    Ok(HeldOutReplayGateReceipt {
        schema: SYM_RSI_001_HOLDOUT_GATE_SCHEMA.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        candidate_family_digest: canonical_candidate_family_digest(),
        training_selection_evidence_digest: training_selection.evidence_digest.clone(),
        held_out_corpus_evidence_digest: held_out_corpus.receipt.evidence_digest.clone(),
        incumbent_policy_id: incumbent_spec.policy_id,
        selected_policy_id: selected_spec.policy_id,
        held_out_quality_tolerance: manifest.held_out_quality_tolerance,
        incumbent,
        selected,
        held_out_quality_delta,
        evaluator_call_delta,
        incumbent_full_support,
        selected_full_support,
        decision,
        evidence_digest,
    })
}

fn full_support(evaluation: &ReplayCorpusEvaluation) -> bool {
    evaluation.world_count > 0
        && evaluation.unsupported_worlds == 0
        && evaluation.replay_coverage == 1.0
        && evaluation.terminal_worlds == evaluation.world_count
}

fn validate_corpus_binding(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
    required_split: EvaluationSplit,
) -> Result<(), HoldoutGateError> {
    let receipt = &corpus.receipt;
    if receipt.split != required_split
        || corpus
            .worlds
            .iter()
            .any(|world| world.split != required_split)
    {
        return Err(HoldoutGateError::WrongCorpusSplit {
            required: required_split,
            observed: receipt.split,
        });
    }
    if receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
    {
        return Err(HoldoutGateError::CorpusLineageMismatch);
    }
    if receipt.candidate_family_digest != canonical_candidate_family_digest() {
        return Err(HoldoutGateError::CandidateFamilyMismatch);
    }
    let canonical_ids = canonical_fixed_hash_candidate_family()
        .into_iter()
        .map(|spec| spec.policy_id)
        .collect::<Vec<_>>();
    if receipt.collector_policy_ids != canonical_ids {
        return Err(HoldoutGateError::CollectorSetMismatch);
    }
    if receipt.evidence_digest.trim().is_empty() {
        return Err(HoldoutGateError::MissingCorpusEvidenceDigest);
    }
    Ok(())
}

fn validate_canonical_selection_receipt(
    manifest: &SymRsiExperimentManifest,
    receipt: &ReplaySelectionReceipt,
) -> Result<(), HoldoutGateError> {
    if receipt.schema != SYM_RSI_001_REPLAY_SELECTION_SCHEMA
        || receipt.selection_split != EvaluationSplit::TrainingReplay
        || receipt.incumbent_policy_id != SYM_RSI_001_INCUMBENT_POLICY_ID
    {
        return Err(HoldoutGateError::SelectionReceiptShapeMismatch);
    }
    if receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
    {
        return Err(HoldoutGateError::SelectionLineageMismatch);
    }
    if receipt.beta_cost != manifest.beta_cost
        || receipt.beta_parallelism != manifest.beta_parallelism
        || !receipt.full_historical_support_required
        || receipt.evidence_digest.trim().is_empty()
    {
        return Err(HoldoutGateError::SelectionReceiptShapeMismatch);
    }

    let canonical = canonical_fixed_hash_candidate_family();
    if receipt.assessments.len() != canonical.len() {
        return Err(HoldoutGateError::CandidateSetMismatch);
    }
    for expected in canonical {
        let assessment = receipt
            .assessments
            .iter()
            .find(|assessment| assessment.spec.policy_id == expected.policy_id)
            .ok_or(HoldoutGateError::CandidateSetMismatch)?;
        if assessment.spec.salt != expected.salt {
            return Err(HoldoutGateError::CandidateSetMismatch);
        }
    }
    if !receipt
        .assessments
        .iter()
        .any(|assessment| assessment.spec.policy_id == receipt.selected_policy_id && assessment.eligible)
    {
        return Err(HoldoutGateError::SelectedPolicyNotEligible);
    }
    Ok(())
}

fn holdout_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    selection: &ReplaySelectionReceipt,
    corpus: &ReplayCorpusAcquisitionReceipt,
    incumbent: &ReplayCorpusEvaluation,
    selected: &ReplayCorpusEvaluation,
    decision: HoldoutGateDecision,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.held-out-replay-gate.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        selection.evidence_digest.as_str(),
        corpus.evidence_digest.as_str(),
        incumbent.policy_id.as_str(),
        selected.policy_id.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&manifest.held_out_quality_tolerance.to_bits().to_le_bytes());
    for evaluation in [incumbent, selected] {
        hasher.update(&evaluation.mean_best_solution_quality.to_bits().to_le_bytes());
        hasher.update(&evaluation.attempted_steps.to_le_bytes());
        hasher.update(&evaluation.supported_steps.to_le_bytes());
        hasher.update(&(evaluation.unsupported_worlds as u64).to_le_bytes());
        hasher.update(&(evaluation.terminal_worlds as u64).to_le_bytes());
        hasher.update(&evaluation.replay_coverage.to_bits().to_le_bytes());
    }
    hasher.update(&[decision_tag(decision)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn decision_tag(decision: HoldoutGateDecision) -> u8 {
    match decision {
        HoldoutGateDecision::NoChange => 0,
        HoldoutGateDecision::FreshExecutionEligible => 1,
        HoldoutGateDecision::RejectedReplaySupport => 2,
        HoldoutGateDecision::RejectedHeldOutQuality => 3,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldoutGateError {
    ReplaySelection(ReplaySelectionError),
    ReplayCorpus(ReplayCorpusError),
    WrongCorpusSplit {
        required: EvaluationSplit,
        observed: EvaluationSplit,
    },
    CorpusLineageMismatch,
    CandidateFamilyMismatch,
    CollectorSetMismatch,
    MissingCorpusEvidenceDigest,
    SelectionReceiptShapeMismatch,
    SelectionLineageMismatch,
    CandidateSetMismatch,
    SelectedPolicyNotCanonical(String),
    SelectedPolicyNotEligible,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        acquire_canonical_replay_corpus, canonical_sym_rsi_001_fixture_manifest,
    };

    #[test]
    fn canonical_training_selection_uses_all_eight_frozen_candidates() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay)
            .unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        assert_eq!(selection.assessments.len(), 8);
        assert_eq!(selection.selection_split, EvaluationSplit::TrainingReplay);
        assert_eq!(selection.incumbent_policy_id, SYM_RSI_001_INCUMBENT_POLICY_ID);
    }

    #[test]
    fn held_out_gate_is_lineage_bound_and_does_not_reselect() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay)
            .unwrap();
        let held_out = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::HeldOutReplay)
            .unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let selected_before = selection.selected_policy_id.clone();
        let gate = validate_selected_candidate_on_holdout(&manifest, &selection, &held_out).unwrap();
        assert_eq!(gate.selected_policy_id, selected_before);
        assert!(gate.incumbent_full_support);
        assert!(gate.selected_full_support);
        assert!(gate.evidence_digest.starts_with("blake3:"));
    }

    #[test]
    fn training_corpus_cannot_be_substituted_for_held_out_validation() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training = acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay)
            .unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        assert!(matches!(
            validate_selected_candidate_on_holdout(&manifest, &selection, &training),
            Err(HoldoutGateError::WrongCorpusSplit {
                required: EvaluationSplit::HeldOutReplay,
                observed: EvaluationSplit::TrainingReplay,
            })
        ));
    }
}
