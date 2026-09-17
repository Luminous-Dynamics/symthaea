// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grounded counterfactual dreaming for SYM-RSI-001 arm D.
//!
//! This module deliberately separates:
//! - **observed training transitions**, which may populate the learned transition memory,
//! - **model-generated counterfactual predictions**, which may steer action choice,
//! - **executed outcomes**, which are the only new observations the normal runner may
//!   append to `ExperienceTree`.
//!
//! The dream policy never calls the fixture transition function while choosing an
//! action. It uses Symthaea's existing `symthaea-dream` learned transition memory
//! and counterfactual predictor. Generated predictions remain explicitly non-empirical.

use super::epistemic_world::{EpistemicWorldRecord, WorldEvidenceKind};
use super::sym_rsi_candidate_family::{
    canonical_candidate_family_digest, canonical_fixed_hash_candidate_family, FrozenReplayCorpus,
};
use super::sym_rsi_experiment::{EvaluationSplit, SymRsiExperimentManifest};
use super::sym_rsi_fixtures::{
    canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind, FixtureState,
};
use super::sym_rsi_replay_selection::{ReplaySelectionReceipt, SYM_RSI_001_REPLAY_SELECTION_SCHEMA};
use super::sym_rsi_runner::{
    fixture_action_digest, fixture_state_digest, FixedHashPolicy, FixturePolicy,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_dream::{DreamEngine, DreamEngineConfig, DreamableAction, TransitionMemory};

pub const SYM_RSI_001_GROUNDED_DREAM_SCHEMA: &str =
    "symthaea.sym-rsi-001.grounded-dream-model.v1";
pub const SYM_RSI_001_GROUNDED_DREAM_POLICY_ID: &str =
    "sym-rsi-grounded-dream-policy-v1";
pub const DREAM_STATE_DIM: usize = 16;
pub const DREAM_RISK_PENALTY: f32 = 0.10;
pub const DREAM_OVERRIDE_MARGIN: f32 = 0.01;

const QUALITY_START: usize = 7;
const QUALITY_COPIES: usize = 8;
const TERMINAL_INDEX: usize = 15;

/// Small discrete action adapter for the existing generic dream engine.
///
/// Perturbation is deterministic. Most perturbation seeds retain the proposed
/// action; the remainder explore adjacent action IDs. The policy filters all
/// final choices through the fixture's legal-action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamFixtureAction(pub u8);

impl DreamableAction for DreamFixtureAction {
    fn perturb(&self, seed: u64) -> Self {
        match seed % 5 {
            0 | 1 | 2 => *self,
            3 => Self(self.0.wrapping_add(1) % 4),
            _ => Self(self.0.wrapping_add(3) % 4),
        }
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        let mut predicted = state.to_vec();
        if predicted.len() != DREAM_STATE_DIM {
            return predicted;
        }

        // Generic, deliberately weak motion heuristic. This is not the fixture
        // transition function and contains no seed-specific environment knowledge.
        predicted[3] = (predicted[3] + 0.05).clamp(0.0, 1.0);
        match self.0 % 4 {
            0 => predicted[4] = (predicted[4] - 0.10).clamp(-1.0, 1.0),
            1 => predicted[4] = (predicted[4] + 0.10).clamp(-1.0, 1.0),
            2 => predicted[5] = (predicted[5] - 0.10).clamp(-1.0, 1.0),
            _ => predicted[5] = (predicted[5] + 0.10).clamp(-1.0, 1.0),
        }
        predicted
    }

    fn magnitude(&self) -> f32 {
        (self.0 as f32 + 1.0) / 4.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedDreamModel {
    pub schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub candidate_family_digest: String,
    pub training_corpus_evidence_digest: String,
    pub model_version: String,
    pub observation_count: usize,
    pub transition_memory: TransitionMemory,
    pub config: DreamEngineConfig,
    pub evidence_digest: String,
}

impl GroundedDreamModel {
    fn engine(&self) -> DreamEngine<DreamFixtureAction> {
        let mut engine = DreamEngine::new(self.config.clone());
        engine.world_model = self.transition_memory.clone();
        engine
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamActionPrediction {
    pub action: u8,
    pub score: f32,
    pub expected_phi: f32,
    pub failure_probability: f32,
    pub model_confidence: f32,
    pub epistemic: EpistemicWorldRecord,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamDecisionRecord {
    pub domain: FixtureDomainKind,
    pub split: EvaluationSplit,
    pub seed: u64,
    pub step: u32,
    pub state_digest: String,
    pub base_action: u8,
    pub chosen_action: u8,
    pub override_applied: bool,
    pub predictions: Vec<DreamActionPrediction>,
    /// Hard claim boundary: predictions influenced action selection but were not
    /// themselves inserted into empirical/replay evidence.
    pub generated_evidence_promoted: bool,
}

#[derive(Debug, Clone)]
pub struct GroundedDreamPolicy {
    id: String,
    base: FixedHashPolicy,
    model: GroundedDreamModel,
    override_margin: f32,
    decision_log: Vec<DreamDecisionRecord>,
}

impl GroundedDreamPolicy {
    pub fn new(base: FixedHashPolicy, model: GroundedDreamModel) -> Self {
        Self {
            id: SYM_RSI_001_GROUNDED_DREAM_POLICY_ID.into(),
            base,
            model,
            override_margin: DREAM_OVERRIDE_MARGIN,
            decision_log: Vec::new(),
        }
    }

    pub fn model(&self) -> &GroundedDreamModel {
        &self.model
    }

    pub fn decision_log(&self) -> &[DreamDecisionRecord] {
        &self.decision_log
    }

    pub fn generated_evidence_promoted(&self) -> bool {
        self.decision_log
            .iter()
            .any(|decision| decision.generated_evidence_promoted)
    }
}

impl FixturePolicy for GroundedDreamPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn choose_action(
        &mut self,
        domain: FixtureDomainKind,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        legal_actions: &[u8],
    ) -> Option<u8> {
        if legal_actions.is_empty() {
            return None;
        }

        let base_action = self
            .base
            .choose_action(domain, seed, split, state, legal_actions)?;
        let encoded = encode_fixture_state(domain, split, state);
        let state_digest = fixture_state_digest(domain, seed, split, state);
        let engine = self.model.engine();

        let mut predictions = Vec::with_capacity(legal_actions.len());
        for &action in legal_actions {
            let distribution =
                engine.predict_outcome_distribution(&encoded, &DreamFixtureAction(action));
            let score = distribution.expected_phi
                - DREAM_RISK_PENALTY * distribution.failure_probability;
            let kind = if action == base_action {
                WorldEvidenceKind::ModelPredicted
            } else {
                WorldEvidenceKind::Counterfactual
            };
            let provenance_digest = prediction_provenance_digest(
                &self.model,
                &state_digest,
                action,
                distribution.expected_phi,
                distribution.failure_probability,
                distribution.confidence,
            );
            predictions.push(DreamActionPrediction {
                action,
                score,
                expected_phi: distribution.expected_phi,
                failure_probability: distribution.failure_probability,
                model_confidence: distribution.confidence,
                epistemic: EpistemicWorldRecord {
                    kind,
                    provenance_digest,
                    model_version: Some(self.model.model_version.clone()),
                    confidence: Some(distribution.confidence as f64),
                    support_distance: None,
                    causal_assumptions: vec![
                        "nearest observed state/action transition memory".into(),
                        "heuristic fallback for unsupported action fingerprints".into(),
                    ],
                    empirically_validated: false,
                },
            });
        }

        let base_score = predictions
            .iter()
            .find(|prediction| prediction.action == base_action)
            .map(|prediction| prediction.score)
            .unwrap_or(f32::NEG_INFINITY);
        let best = predictions.iter().max_by(|left, right| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.action.cmp(&left.action))
        });
        let chosen_action = best
            .filter(|prediction| prediction.score > base_score + self.override_margin)
            .map(|prediction| prediction.action)
            .unwrap_or(base_action);

        debug_assert!(predictions
            .iter()
            .all(|prediction| !prediction.epistemic.may_promote_confidence()));

        self.decision_log.push(DreamDecisionRecord {
            domain,
            split,
            seed,
            step: state.step,
            state_digest,
            base_action,
            chosen_action,
            override_applied: chosen_action != base_action,
            predictions,
            generated_evidence_promoted: false,
        });

        Some(chosen_action)
    }
}

/// Train only from the observed branching training corpus.
///
/// No generated dream prediction is fed back into this function. Every transition
/// inserted into the dream engine's `TransitionMemory` is recovered from a recorded
/// child edge in the exact replay graph.
pub fn train_grounded_dream_model(
    manifest: &SymRsiExperimentManifest,
    training_corpus: &FrozenReplayCorpus,
) -> Result<GroundedDreamModel, GroundedDreamError> {
    ensure_canonical_manifest(manifest)?;
    validate_training_corpus_binding(manifest, training_corpus)?;

    let config = DreamEngineConfig {
        surprise_threshold: 0.0,
        counterfactual_count: 5,
        wisdom_threshold: 0.01,
        max_memory_size: 100_000,
        state_dim: DREAM_STATE_DIM,
    };
    let mut engine = DreamEngine::<DreamFixtureAction>::new(config.clone());
    let mut observation_count = 0_usize;

    for world in training_corpus.worlds() {
        let mut stack = vec![world.root_node_id];
        let mut visited = BTreeSet::new();
        while let Some(parent_id) = stack.pop() {
            if !visited.insert(parent_id) {
                continue;
            }
            let parent_state = world
                .state(parent_id)
                .ok_or(GroundedDreamError::MissingRawState(parent_id))?;
            let parent_vector = encode_fixture_state(world.domain, world.split, parent_state);
            let legal = world.domain.legal_actions(parent_state, world.split);

            for &child_id in world.experience_tree.children_of(parent_id) {
                let child_node = world
                    .experience_tree
                    .node(child_id)
                    .ok_or(GroundedDreamError::MissingReplayNode(child_id))?;
                let child_state = world
                    .state(child_id)
                    .ok_or(GroundedDreamError::MissingRawState(child_id))?;
                let action = legal
                    .iter()
                    .copied()
                    .find(|candidate| {
                        fixture_action_digest(world.domain, *candidate) == child_node.action_digest
                    })
                    .ok_or(GroundedDreamError::UnrecoverableObservedAction {
                        node_id: child_id,
                    })?;
                let outcome_vector =
                    encode_fixture_state(world.domain, world.split, child_state);

                // A fixed high surprise is only a storage gate here. The generated
                // timestamp lives in DreamEngine memory but is not included in the
                // extracted TransitionMemory or any SYM-RSI evidence digest.
                engine.record(
                    &parent_vector,
                    DreamFixtureAction(action),
                    &outcome_vector,
                    1.0,
                );
                observation_count += 1;
                stack.push(child_id);
            }
        }
    }

    if observation_count == 0 || engine.world_model.observations.is_empty() {
        return Err(GroundedDreamError::EmptyObservedTrainingSet);
    }

    let transition_memory = engine.world_model.clone();
    let evidence_digest = dream_model_evidence_digest(
        manifest,
        training_corpus,
        &transition_memory,
        observation_count,
    );

    Ok(GroundedDreamModel {
        schema: SYM_RSI_001_GROUNDED_DREAM_SCHEMA.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        candidate_family_digest: canonical_candidate_family_digest(),
        training_corpus_evidence_digest: training_corpus.receipt.evidence_digest.clone(),
        model_version: "symthaea-dream-transition-memory-v1".into(),
        observation_count,
        transition_memory,
        config,
        evidence_digest,
    })
}

/// Build arm D on top of the already-selected arm C policy.
///
/// The selection receipt determines the base C policy. The dream model is trained
/// solely from the frozen training corpus and cannot inspect held-out, fresh, or OOD
/// environment outcomes through this API.
pub fn build_grounded_dream_policy(
    manifest: &SymRsiExperimentManifest,
    training_corpus: &FrozenReplayCorpus,
    replay_selection: &ReplaySelectionReceipt,
) -> Result<GroundedDreamPolicy, GroundedDreamError> {
    ensure_canonical_manifest(manifest)?;
    validate_training_corpus_binding(manifest, training_corpus)?;
    validate_replay_selection_binding(manifest, replay_selection)?;

    let base_spec = canonical_fixed_hash_candidate_family()
        .into_iter()
        .find(|candidate| candidate.policy_id == replay_selection.selected_policy_id)
        .ok_or_else(|| {
            GroundedDreamError::SelectedPolicyNotCanonical(
                replay_selection.selected_policy_id.clone(),
            )
        })?;
    let model = train_grounded_dream_model(manifest, training_corpus)?;
    Ok(GroundedDreamPolicy::new(base_spec.policy(), model))
}

fn validate_training_corpus_binding(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
) -> Result<(), GroundedDreamError> {
    if corpus.receipt.split != EvaluationSplit::TrainingReplay
        || corpus
            .worlds()
            .iter()
            .any(|world| world.split != EvaluationSplit::TrainingReplay)
    {
        return Err(GroundedDreamError::TrainingCorpusRequired);
    }
    if corpus.receipt.experiment_id != manifest.experiment_id
        || corpus.receipt.preregistration_digest != manifest.preregistration_digest
        || corpus.receipt.subject_digest != manifest.subject_digest
        || corpus.receipt.environment_digest != manifest.environment_digest
        || corpus.receipt.candidate_family_digest != canonical_candidate_family_digest()
        || corpus.receipt.evidence_digest.trim().is_empty()
    {
        return Err(GroundedDreamError::TrainingCorpusBindingMismatch);
    }
    Ok(())
}

fn validate_replay_selection_binding(
    manifest: &SymRsiExperimentManifest,
    receipt: &ReplaySelectionReceipt,
) -> Result<(), GroundedDreamError> {
    if receipt.schema != SYM_RSI_001_REPLAY_SELECTION_SCHEMA
        || receipt.selection_split != EvaluationSplit::TrainingReplay
        || receipt.experiment_id != manifest.experiment_id
        || receipt.preregistration_digest != manifest.preregistration_digest
        || receipt.subject_digest != manifest.subject_digest
        || receipt.environment_digest != manifest.environment_digest
        || receipt.evidence_digest.trim().is_empty()
    {
        return Err(GroundedDreamError::ReplaySelectionBindingMismatch);
    }
    let assessment = receipt
        .assessments
        .iter()
        .find(|assessment| assessment.spec.policy_id == receipt.selected_policy_id)
        .ok_or(GroundedDreamError::ReplaySelectionBindingMismatch)?;
    if !assessment.eligible {
        return Err(GroundedDreamError::SelectedPolicyNotReplayEligible);
    }
    Ok(())
}

fn ensure_canonical_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), GroundedDreamError> {
    manifest
        .validate()
        .map_err(GroundedDreamError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(GroundedDreamError::ManifestNotCanonical);
    }
    Ok(())
}

fn encode_fixture_state(
    domain: FixtureDomainKind,
    split: EvaluationSplit,
    state: &FixtureState,
) -> Vec<f32> {
    let mut encoded = vec![0.0; DREAM_STATE_DIM];
    let domain_index = match domain {
        FixtureDomainKind::BranchingSearch => 0,
        FixtureDomainKind::DelayedNavigation => 1,
        FixtureDomainKind::RuggedOptimization => 2,
    };
    encoded[domain_index] = 1.0;
    encoded[3] = state.step as f32 / domain.horizon(split).max(1) as f32;
    encoded[4] = signed_unit(state.a);
    encoded[5] = signed_unit(state.b);
    encoded[6] = state.aux as f32 / (state.aux as f32 + 10.0);
    let quality = state.quality.clamp(0.0, 1.0) as f32;
    for slot in &mut encoded[QUALITY_START..QUALITY_START + QUALITY_COPIES] {
        *slot = quality;
    }
    encoded[TERMINAL_INDEX] = if state.terminal { 1.0 } else { 0.0 };
    encoded
}

fn signed_unit(value: i32) -> f32 {
    let value = value as f32;
    value / (1.0 + value.abs())
}

fn prediction_provenance_digest(
    model: &GroundedDreamModel,
    state_digest: &str,
    action: u8,
    expected_phi: f32,
    failure_probability: f32,
    confidence: f32,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.dream-prediction.v1\0");
    for value in [
        model.evidence_digest.as_str(),
        state_digest,
        model.model_version.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[action]);
    hasher.update(&expected_phi.to_bits().to_le_bytes());
    hasher.update(&failure_probability.to_bits().to_le_bytes());
    hasher.update(&confidence.to_bits().to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn dream_model_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
    memory: &TransitionMemory,
    observation_count: usize,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.grounded-dream-model.v1\0");
    for value in [
        manifest.experiment_id.as_str(),
        manifest.preregistration_digest.as_str(),
        manifest.subject_digest.as_str(),
        manifest.environment_digest.as_str(),
        corpus.receipt.evidence_digest.as_str(),
        SYM_RSI_001_GROUNDED_DREAM_SCHEMA,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&(observation_count as u64).to_le_bytes());
    for observation in &memory.observations {
        hasher.update(&observation.action_fingerprint.to_le_bytes());
        hasher.update(&observation.weight.to_bits().to_le_bytes());
        hasher.update(&(observation.state_context.len() as u64).to_le_bytes());
        for value in &observation.state_context {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&(observation.outcome.len() as u64).to_le_bytes());
        for value in &observation.outcome {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundedDreamError {
    ManifestInvalid(super::sym_rsi_experiment::ExperimentHarnessError),
    ManifestNotCanonical,
    TrainingCorpusRequired,
    TrainingCorpusBindingMismatch,
    ReplaySelectionBindingMismatch,
    SelectedPolicyNotCanonical(String),
    SelectedPolicyNotReplayEligible,
    MissingRawState(u64),
    MissingReplayNode(u64),
    UnrecoverableObservedAction { node_id: u64 },
    EmptyObservedTrainingSet,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        acquire_canonical_replay_corpus, select_canonical_training_candidate,
    };

    #[test]
    fn grounded_model_is_built_only_from_recorded_training_edges() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let model = train_grounded_dream_model(&manifest, &training).unwrap();

        let expected_edges = training
            .worlds()
            .iter()
            .map(|world| world.experience_tree.len().saturating_sub(1))
            .sum::<usize>();
        assert_eq!(model.observation_count, expected_edges);
        assert_eq!(model.transition_memory.observations.len(), expected_edges);
        assert_eq!(
            model.training_corpus_evidence_digest,
            training.receipt.evidence_digest
        );
        assert!(model.evidence_digest.starts_with("blake3:"));
    }

    #[test]
    fn dream_predictions_are_non_empirical_and_cannot_promote_confidence() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let mut policy =
            build_grounded_dream_policy(&manifest, &training, &selection).unwrap();

        let state = FixtureDomainKind::BranchingSearch.reset(
            101,
            EvaluationSplit::HeldOutReplay,
        );
        let legal = FixtureDomainKind::BranchingSearch
            .legal_actions(&state, EvaluationSplit::HeldOutReplay);
        let chosen = policy
            .choose_action(
                FixtureDomainKind::BranchingSearch,
                101,
                EvaluationSplit::HeldOutReplay,
                &state,
                &legal,
            )
            .unwrap();
        assert!(legal.contains(&chosen));

        let decision = policy.decision_log().last().unwrap();
        assert!(!decision.generated_evidence_promoted);
        assert!(decision.predictions.iter().all(|prediction| {
            !prediction.epistemic.kind.is_empirical()
                && !prediction.epistemic.empirically_validated
                && !prediction.epistemic.may_promote_confidence()
                && prediction.epistemic.provenance_digest.starts_with("blake3:")
        }));
    }

    #[test]
    fn grounded_dream_choice_is_deterministic_for_same_model_and_state() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let mut first =
            build_grounded_dream_policy(&manifest, &training, &selection).unwrap();
        let mut second =
            build_grounded_dream_policy(&manifest, &training, &selection).unwrap();

        let domain = FixtureDomainKind::RuggedOptimization;
        let split = EvaluationSplit::HeldOutReplay;
        let state = domain.reset(102, split);
        let legal = domain.legal_actions(&state, split);
        let a = first.choose_action(domain, 102, split, &state, &legal);
        let b = second.choose_action(domain, 102, split, &state, &legal);
        assert_eq!(a, b);
        assert_eq!(first.decision_log(), second.decision_log());
    }

    #[test]
    fn held_out_corpus_cannot_train_dream_model() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let held_out =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::HeldOutReplay).unwrap();
        assert_eq!(
            train_grounded_dream_model(&manifest, &held_out).unwrap_err(),
            GroundedDreamError::TrainingCorpusRequired
        );
    }
}
