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
    SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA, SYM_RSI_001_REPLAY_CORPUS_SCHEMA,
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
use std::collections::{BTreeMap, BTreeSet};
use symthaea_dream::{DreamEngine, DreamEngineConfig, DreamableAction, TransitionMemory};

pub const SYM_RSI_001_GROUNDED_DREAM_SCHEMA: &str =
    "symthaea.sym-rsi-001.grounded-dream-model.v1";
pub const SYM_RSI_001_GROUNDED_DREAM_POLICY_ID: &str =
    "sym-rsi-grounded-dream-policy-v6";
pub const DREAM_STATE_DIM: usize = 16;
pub const DREAM_RISK_PENALTY: f32 = 0.10;
pub const DREAM_OVERRIDE_MARGIN: f32 = 0.01;
pub const SYM_RSI_001_GROUNDED_DREAM_SCORING_RULE: &str =
    "domain-conditioned-action/observed-action-support-gate/local-state-support-shrinkage/5-model-simulations-per-candidate/support-adjusted-task-quality-minus-0.10-task-regression-probability/0.01-override-margin/v6";
pub const DREAM_ACTION_FINGERPRINT_SEMANTICS: &str =
    "symthaea-dream/default-hasher(debug-domain-conditioned-action)/environment-bound-v2";

const QUALITY_START: usize = 7;
const QUALITY_COPIES: usize = 8;
const TERMINAL_INDEX: usize = 15;

/// Small discrete action adapter for the existing generic dream engine.
///
/// Perturbation is deterministic. Most perturbation seeds retain the proposed
/// action; the remainder explore adjacent action IDs. The policy filters all
/// final choices through the fixture's legal-action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamFixtureAction {
    pub domain: FixtureDomainKind,
    pub action: u8,
}

impl DreamFixtureAction {
    pub fn new(domain: FixtureDomainKind, action: u8) -> Self {
        Self { domain, action }
    }
}

impl DreamableAction for DreamFixtureAction {
    fn perturb(&self, seed: u64) -> Self {
        let action = match seed % 5 {
            0 | 1 | 2 => self.action,
            3 => self.action.wrapping_add(1) % 4,
            _ => self.action.wrapping_add(3) % 4,
        };
        Self {
            domain: self.domain,
            action,
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
        match self.action % 4 {
            0 => predicted[4] = (predicted[4] - 0.10).clamp(-1.0, 1.0),
            1 => predicted[4] = (predicted[4] + 0.10).clamp(-1.0, 1.0),
            2 => predicted[5] = (predicted[5] - 0.10).clamp(-1.0, 1.0),
            _ => predicted[5] = (predicted[5] + 0.10).clamp(-1.0, 1.0),
        }
        predicted
    }

    fn magnitude(&self) -> f32 {
        (self.action as f32 + 1.0) / 4.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamActionSupport {
    pub domain_id: String,
    pub action: u8,
    pub observation_count: usize,
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
    /// The upstream dream crate currently fingerprints actions with DefaultHasher
    /// over Debug output. That identity is therefore explicitly environment-bound.
    pub action_fingerprint_semantics: String,
    pub observation_count: usize,
    pub action_support: Vec<DreamActionSupport>,
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

    pub fn support_count(&self, domain: FixtureDomainKind, action: u8) -> usize {
        self.action_support
            .iter()
            .find(|support| support.domain_id == domain.id() && support.action == action)
            .map(|support| support.observation_count)
            .unwrap_or(0)
    }

    fn supported_actions(&self, domain: FixtureDomainKind) -> BTreeSet<u8> {
        self.action_support
            .iter()
            .filter(|support| support.domain_id == domain.id() && support.observation_count > 0)
            .map(|support| support.action)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DreamActionPrediction {
    pub action: u8,
    pub training_support_count: usize,
    pub actionable: bool,
    pub model_simulation_count: usize,
    pub score: f32,
    pub predicted_task_quality: f32,
    pub support_adjusted_task_quality: f32,
    pub mean_support_similarity: f32,
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
    pub base_training_support_count: usize,
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
        let supported_actions = self.model.supported_actions(domain);
        let base_training_support_count = self.model.support_count(domain, base_action);

        let mut predictions = Vec::with_capacity(legal_actions.len());
        for &action in legal_actions {
            let dream_action = DreamFixtureAction::new(domain, action);
            let training_support_count = self.model.support_count(domain, action);
            let actionable = base_training_support_count > 0 && training_support_count > 0;
            let task_prediction = if actionable {
                task_prediction_summary(
                    &engine,
                    &encoded,
                    dream_action,
                    legal_actions,
                    &supported_actions,
                    state.quality as f32,
                )
            } else {
                TaskPredictionSummary {
                    mean_quality: state.quality.clamp(0.0, 1.0) as f32,
                    support_adjusted_mean_quality: state.quality.clamp(0.0, 1.0) as f32,
                    mean_support_similarity: 0.0,
                    failure_probability: 1.0,
                    confidence: 0.0,
                    simulations_run: 0,
                }
            };
            let predicted_task_quality = task_prediction.mean_quality;
            let support_adjusted_task_quality = task_prediction.support_adjusted_mean_quality;
            let score = support_adjusted_task_quality
                - DREAM_RISK_PENALTY * task_prediction.failure_probability;
            let kind = if action == base_action {
                WorldEvidenceKind::ModelPredicted
            } else {
                WorldEvidenceKind::Counterfactual
            };
            let provenance_digest = prediction_provenance_digest(
                &self.model,
                &state_digest,
                action,
                predicted_task_quality,
                support_adjusted_task_quality,
                task_prediction.mean_support_similarity,
                task_prediction.failure_probability,
                task_prediction.confidence,
                training_support_count,
                actionable,
                task_prediction.simulations_run,
            );
            predictions.push(DreamActionPrediction {
                action,
                training_support_count,
                actionable,
                model_simulation_count: task_prediction.simulations_run,
                score,
                predicted_task_quality,
                support_adjusted_task_quality,
                mean_support_similarity: task_prediction.mean_support_similarity,
                failure_probability: task_prediction.failure_probability,
                model_confidence: task_prediction.confidence,
                epistemic: EpistemicWorldRecord {
                    kind,
                    provenance_digest,
                    model_version: Some(self.model.model_version.clone()),
                    confidence: Some(task_prediction.confidence as f64),
                    support_distance: Some(
                        (1.0 - task_prediction.mean_support_similarity.clamp(0.0, 1.0))
                            as f64,
                    ),
                    causal_assumptions: vec![
                        "nearest observed state/action transition memory".into(),
                        "candidate and base action classes require recorded training support".into(),
                        "illegal or unsupported perturbation samples fall back to the original supported legal action".into(),
                        "predicted quality change is shrunk toward current observed quality by nearest training-state cosine support".into(),
                        "task failure means support-adjusted predicted quality regression relative to the current state".into(),
                        SYM_RSI_001_GROUNDED_DREAM_SCORING_RULE.into(),
                    ],
                    empirically_validated: false,
                },
            });
        }

        let base_score = predictions
            .iter()
            .find(|prediction| prediction.action == base_action && prediction.actionable)
            .map(|prediction| prediction.score);
        let best = predictions
            .iter()
            .filter(|prediction| prediction.actionable)
            .max_by(|left, right| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.action.cmp(&left.action))
            });
        let chosen_action = match (base_score, best) {
            (Some(base_score), Some(best))
                if best.score > base_score + self.override_margin =>
            {
                best.action
            }
            _ => base_action,
        };

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
            base_training_support_count,
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
    let mut action_support_counts: BTreeMap<(String, u8), usize> = BTreeMap::new();

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
                    DreamFixtureAction::new(world.domain, action),
                    &outcome_vector,
                    1.0,
                );
                observation_count += 1;
                *action_support_counts
                    .entry((world.domain.id().to_owned(), action))
                    .or_insert(0) += 1;
                stack.push(child_id);
            }
        }
    }

    if observation_count == 0 || engine.world_model.observations.is_empty() {
        return Err(GroundedDreamError::EmptyObservedTrainingSet);
    }

    let transition_memory = engine.world_model.clone();
    let action_support = action_support_counts
        .into_iter()
        .map(|((domain_id, action), observation_count)| DreamActionSupport {
            domain_id,
            action,
            observation_count,
        })
        .collect::<Vec<_>>();
    let evidence_digest = dream_model_evidence_digest(
        manifest,
        training_corpus,
        &transition_memory,
        &action_support,
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
        model_version: "symthaea-dream-transition-memory-v2-support-gated".into(),
        action_fingerprint_semantics: DREAM_ACTION_FINGERPRINT_SEMANTICS.into(),
        observation_count,
        action_support,
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

    let canonical_family = canonical_fixed_hash_candidate_family();
    let canonical_ids = canonical_family
        .iter()
        .map(|candidate| candidate.policy_id.clone())
        .collect::<Vec<_>>();
    if corpus.receipt.schema != SYM_RSI_001_REPLAY_CORPUS_SCHEMA
        || corpus.receipt.experiment_id != manifest.experiment_id
        || corpus.receipt.preregistration_digest != manifest.preregistration_digest
        || corpus.receipt.subject_digest != manifest.subject_digest
        || corpus.receipt.environment_digest != manifest.environment_digest
        || corpus.receipt.candidate_family_schema != SYM_RSI_001_CANDIDATE_FAMILY_SCHEMA
        || corpus.receipt.candidate_family_digest != canonical_candidate_family_digest()
        || corpus.receipt.collector_policy_ids != canonical_ids
        || corpus.receipt.evidence_digest.trim().is_empty()
    {
        return Err(GroundedDreamError::TrainingCorpusBindingMismatch);
    }

    let mut expected = BTreeSet::new();
    for domain in &manifest.domains {
        for &seed in &domain.seeds.training_replay {
            expected.insert((domain.domain_id.clone(), seed));
        }
    }
    let mut observed = BTreeSet::new();
    for world in corpus.worlds() {
        let key = (world.domain.id().to_owned(), world.seed);
        if !observed.insert(key) {
            return Err(GroundedDreamError::DuplicateTrainingWorld);
        }
    }
    let expected_seed_count = manifest
        .domains
        .first()
        .map(|domain| domain.seeds.training_replay.len())
        .unwrap_or(0);
    if observed != expected
        || corpus.receipt.domain_count != manifest.domains.len()
        || corpus.receipt.seed_count_per_domain != expected_seed_count
        || corpus.receipt.world_count != expected.len()
        || corpus.receipt.world_count != corpus.worlds().len()
        || corpus.receipt.trajectory_count != expected.len() * canonical_family.len()
    {
        return Err(GroundedDreamError::TrainingCorpusIncomplete);
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
    let canonical = canonical_fixed_hash_candidate_family();
    if receipt.assessments.len() != canonical.len() {
        return Err(GroundedDreamError::ReplaySelectionCandidateSetMismatch);
    }
    for expected in &canonical {
        let assessment = receipt
            .assessments
            .iter()
            .find(|assessment| assessment.spec.policy_id == expected.policy_id)
            .ok_or(GroundedDreamError::ReplaySelectionCandidateSetMismatch)?;
        if assessment.spec.salt != expected.salt {
            return Err(GroundedDreamError::ReplaySelectionCandidateSetMismatch);
        }
    }

    let assessment = receipt
        .assessments
        .iter()
        .find(|assessment| assessment.spec.policy_id == receipt.selected_policy_id)
        .ok_or(GroundedDreamError::ReplaySelectionBindingMismatch)?;
    if !assessment.eligible {
        return Err(GroundedDreamError::SelectedPolicyNotReplayEligible);
    }
    if assessment.objective != receipt.selected_objective {
        return Err(GroundedDreamError::ReplaySelectionBindingMismatch);
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct TaskPredictionSummary {
    mean_quality: f32,
    failure_probability: f32,
    confidence: f32,
    training_support_count: usize,
    actionable: bool,
    model_simulation_count: usize,
    simulations_run: usize,
}

fn task_prediction_summary(
    engine: &DreamEngine<DreamFixtureAction>,
    state: &[f32],
    action: DreamFixtureAction,
    legal_actions: &[u8],
    supported_actions: &BTreeSet<u8>,
    current_quality: f32,
) -> TaskPredictionSummary {
    let simulations = engine.config().counterfactual_count.max(1);
    let mut total_quality = 0.0_f32;
    let mut total_support_adjusted_quality = 0.0_f32;
    let mut total_support_similarity = 0.0_f32;
    let mut regression_count = 0_usize;

    for index in 0..simulations {
        let perturbed = action.perturb(index as u64);
        let sampled_action = if perturbed.domain == action.domain
            && legal_actions.contains(&perturbed.action)
            && supported_actions.contains(&perturbed.action)
        {
            perturbed
        } else {
            action
        };
        let support_similarity = engine
            .nearest_observed_support_similarity(state, &sampled_action)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let outcome = engine.predict_counterfactual_outcome(state, &sampled_action);
        let predicted_quality = extract_predicted_task_quality(&outcome);
        let support_adjusted_quality = support_adjusted_quality(
            current_quality,
            predicted_quality,
            support_similarity,
        );
        total_quality += predicted_quality;
        total_support_adjusted_quality += support_adjusted_quality;
        total_support_similarity += support_similarity;
        if support_adjusted_quality + f32::EPSILON < current_quality {
            regression_count += 1;
        }
    }

    let failure_probability = regression_count as f32 / simulations as f32;
    let mean_support_similarity = total_support_similarity / simulations as f32;
    TaskPredictionSummary {
        mean_quality: total_quality / simulations as f32,
        support_adjusted_mean_quality: total_support_adjusted_quality / simulations as f32,
        mean_support_similarity,
        failure_probability,
        confidence: (1.0 - failure_probability) * mean_support_similarity,
        simulations_run: simulations,
    }
}

fn support_adjusted_quality(
    current_quality: f32,
    predicted_quality: f32,
    support_similarity: f32,
) -> f32 {
    let support = support_similarity.clamp(0.0, 1.0);
    (current_quality + support * (predicted_quality - current_quality)).clamp(0.0, 1.0)
}

fn extract_predicted_task_quality(outcome: &[f32]) -> f32 {
    if outcome.len() < QUALITY_START + QUALITY_COPIES {
        return 0.0;
    }
    let sum = outcome[QUALITY_START..QUALITY_START + QUALITY_COPIES]
        .iter()
        .copied()
        .sum::<f32>();
    (sum / QUALITY_COPIES as f32).clamp(0.0, 1.0)
}

fn prediction_provenance_digest(
    model: &GroundedDreamModel,
    state_digest: &str,
    action: u8,
    predicted_task_quality: f32,
    support_adjusted_task_quality: f32,
    mean_support_similarity: f32,
    failure_probability: f32,
    confidence: f32,
    training_support_count: usize,
    actionable: bool,
    model_simulation_count: usize,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.dream-prediction.v4\0");
    for value in [
        model.evidence_digest.as_str(),
        state_digest,
        model.model_version.as_str(),
        SYM_RSI_001_GROUNDED_DREAM_SCORING_RULE,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&[action]);
    hasher.update(&predicted_task_quality.to_bits().to_le_bytes());
    hasher.update(&support_adjusted_task_quality.to_bits().to_le_bytes());
    hasher.update(&mean_support_similarity.to_bits().to_le_bytes());
    hasher.update(&failure_probability.to_bits().to_le_bytes());
    hasher.update(&confidence.to_bits().to_le_bytes());
    hasher.update(&(training_support_count as u64).to_le_bytes());
    hasher.update(&[u8::from(actionable)]);
    hasher.update(&(model_simulation_count as u64).to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn dream_model_evidence_digest(
    manifest: &SymRsiExperimentManifest,
    corpus: &FrozenReplayCorpus,
    memory: &TransitionMemory,
    action_support: &[DreamActionSupport],
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
        DREAM_ACTION_FINGERPRINT_SEMANTICS,
        SYM_RSI_001_GROUNDED_DREAM_SCORING_RULE,
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&(observation_count as u64).to_le_bytes());
    for support in action_support {
        hasher.update(&(support.domain_id.len() as u64).to_le_bytes());
        hasher.update(support.domain_id.as_bytes());
        hasher.update(&[support.action]);
        hasher.update(&(support.observation_count as u64).to_le_bytes());
    }
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
    ReplaySelectionCandidateSetMismatch,
    DuplicateTrainingWorld,
    TrainingCorpusIncomplete,
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
        assert_eq!(
            model.action_fingerprint_semantics,
            DREAM_ACTION_FINGERPRINT_SEMANTICS
        );
        assert!(model.evidence_digest.starts_with("blake3:"));
    }

    #[test]
    fn local_support_shrinkage_is_conservative_and_bounded() {
        assert_eq!(support_adjusted_quality(0.4, 0.9, 0.0), 0.4);
        assert!((support_adjusted_quality(0.4, 0.9, 0.5) - 0.65).abs() < 1e-6);
        assert_eq!(support_adjusted_quality(0.4, 0.9, 1.0), 0.9);
        assert_eq!(support_adjusted_quality(0.8, 0.2, 0.0), 0.8);
        assert_eq!(support_adjusted_quality(0.8, 0.2, 1.0), 0.2);
    }

    #[test]
    fn action_support_census_is_complete_for_recorded_training_edges() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let model = train_grounded_dream_model(&manifest, &training).unwrap();

        let support_total = model
            .action_support
            .iter()
            .map(|support| support.observation_count)
            .sum::<usize>();
        assert_eq!(support_total, model.observation_count);
        assert!(model.action_support.iter().all(|support| {
            support.observation_count > 0
                && FixtureDomainKind::ALL
                    .iter()
                    .any(|domain| domain.id() == support.domain_id.as_str())
        }));
    }

    #[test]
    fn no_training_support_means_no_dream_override() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let base_spec = canonical_fixed_hash_candidate_family()
            .into_iter()
            .find(|candidate| candidate.policy_id == selection.selected_policy_id)
            .unwrap();
        let mut model = train_grounded_dream_model(&manifest, &training).unwrap();
        model.action_support.clear();
        let mut policy = GroundedDreamPolicy::new(base_spec.policy(), model);

        let domain = FixtureDomainKind::BranchingSearch;
        let split = EvaluationSplit::HeldOutReplay;
        let state = domain.reset(101, split);
        let legal = domain.legal_actions(&state, split);
        let mut base_policy = base_spec.policy();
        let base_action = base_policy
            .choose_action(domain, 101, split, &state, &legal)
            .unwrap();
        let chosen = policy
            .choose_action(domain, 101, split, &state, &legal)
            .unwrap();

        assert_eq!(chosen, base_action);
        let decision = policy.decision_log().last().unwrap();
        assert_eq!(decision.base_training_support_count, 0);
        assert!(decision.predictions.iter().all(|prediction| {
            prediction.training_support_count == 0
                && !prediction.actionable
                && prediction.model_simulation_count == 0
        }));
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
            prediction.predicted_task_quality.is_finite()
                && prediction.support_adjusted_task_quality.is_finite()
                && (0.0..=1.0).contains(&prediction.predicted_task_quality)
                && (0.0..=1.0).contains(&prediction.support_adjusted_task_quality)
                && (0.0..=1.0).contains(&prediction.mean_support_similarity)
                && !prediction.epistemic.kind.is_empirical()
                && !prediction.epistemic.empirically_validated
                && !prediction.epistemic.may_promote_confidence()
                && prediction.epistemic.provenance_digest.starts_with("blake3:")
        }));
    }

    #[test]
    fn dream_action_fingerprint_is_domain_conditioned() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn debug_fingerprint(action: DreamFixtureAction) -> u64 {
            let mut hasher = DefaultHasher::new();
            format!("{action:?}").hash(&mut hasher);
            hasher.finish()
        }

        let search = DreamFixtureAction::new(FixtureDomainKind::BranchingSearch, 0);
        let navigation = DreamFixtureAction::new(FixtureDomainKind::DelayedNavigation, 0);
        let rugged = DreamFixtureAction::new(FixtureDomainKind::RuggedOptimization, 0);

        assert_ne!(debug_fingerprint(search), debug_fingerprint(navigation));
        assert_ne!(debug_fingerprint(search), debug_fingerprint(rugged));
        assert_ne!(debug_fingerprint(navigation), debug_fingerprint(rugged));
    }

    #[test]
    fn task_prediction_summary_never_executes_illegal_perturbations() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let policy = build_grounded_dream_policy(&manifest, &training, &selection).unwrap();

        let domain = FixtureDomainKind::DelayedNavigation;
        let split = EvaluationSplit::HeldOutReplay;
        let mut state = domain.reset(101, split);
        state.a = 0;
        state.b = 0;
        let legal = domain.legal_actions(&state, split);
        assert!(!legal.contains(&0));
        assert!(!legal.contains(&3));

        let encoded = encode_fixture_state(domain, split, &state);
        let engine = policy.model().engine();
        let supported = policy.model().supported_actions(domain);
        let supported_legal_action = legal
            .iter()
            .copied()
            .find(|action| supported.contains(action))
            .expect("training corpus should support at least one legal navigation action");
        let summary = task_prediction_summary(
            &engine,
            &encoded,
            DreamFixtureAction::new(domain, supported_legal_action),
            &legal,
            &supported,
            state.quality as f32,
        );
        assert!(summary.mean_quality.is_finite());
        assert!(summary.support_adjusted_mean_quality.is_finite());
        assert!((0.0..=1.0).contains(&summary.mean_support_similarity));
        assert!((0.0..=1.0).contains(&summary.failure_probability));
        assert!((0.0..=1.0).contains(&summary.confidence));
        assert_eq!(summary.simulations_run, 5);
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
