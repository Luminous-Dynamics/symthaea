// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical reasoning kernel V2.
//!
//! V2 replaces the caller-supplied single context with evidence-backed competing hypotheses.
//! Ambiguity is preserved and candidate selection is maximin across every active plausible context.
//! The resulting pre-outcome decision is content-bound before later evaluator feedback is accepted.

use super::reasoning_context_competition::{
    assess_and_select, ContextCompetitionError, ContextCompetitionPolicy, ContextHypothesis,
    ContextResolution, RobustContextSelectionReport,
};
use super::reasoning_meta_state::{
    MetaCommitReport, MetaEpistemicState, MetaEpisodeObservation, MetaOutcomeFeedback,
    MetaSignalRevision, MetaStateAuthority, MetaStateCheckpoint, MetaStateError, MetaSupportSignals,
    OutcomeCommitReport,
};
use crate::consciousness::primitive_evolution::CandidatePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub const CANONICAL_REASONING_KERNEL_V2_VERSION: &str = "rq-006-canonical-kernel-v2";
pub const CANONICAL_REASONING_BASELINE_PLASTICITY: f64 = 1.0;
const DECISION_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning-kernel-v2/decision/v1";

#[derive(Debug, Clone)]
pub struct CanonicalReasoningInputV2 {
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub context_hypotheses: Vec<ContextHypothesis>,
    /// Adapter-supplied support for the selected strategy family. This is not correctness
    /// probability and has no control authority.
    pub strategy_support: f64,
    pub abstained: bool,
    pub evidence_items: usize,
    pub weak_assumptions_flagged: usize,
    pub candidates: Vec<CandidatePrimitive>,
}

#[derive(Debug, Clone)]
pub struct CanonicalReasoningDecisionV2 {
    pub kernel_version: String,
    pub authority: MetaStateAuthority,
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub context_selection: RobustContextSelectionReport,
    pub selected_candidate: CandidatePrimitive,
    pub meta_commit: MetaCommitReport,
    pub applied_plasticity_multiplier: f64,
    pub decision_commitment: String,
}

impl CanonicalReasoningDecisionV2 {
    pub fn validate_commitment(&self) -> bool {
        self.decision_commitment == compute_decision_commitment(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSubjectCheckpointV2 {
    pub kernel_version: String,
    pub context_policy: ContextCompetitionPolicy,
    pub state: MetaStateCheckpoint,
}

pub struct CanonicalReasoningKernelV2 {
    history_capacity: usize,
    context_policy: ContextCompetitionPolicy,
    subjects: HashMap<String, MetaEpistemicState>,
}

impl CanonicalReasoningKernelV2 {
    pub fn new(history_capacity: usize) -> Result<Self, ReasoningKernelV2Error> {
        Self::with_context_policy(history_capacity, ContextCompetitionPolicy::development_v1())
    }

    pub fn with_context_policy(
        history_capacity: usize,
        context_policy: ContextCompetitionPolicy,
    ) -> Result<Self, ReasoningKernelV2Error> {
        if history_capacity == 0 {
            return Err(ReasoningKernelV2Error::InvalidHistoryCapacity(
                history_capacity,
            ));
        }
        let context_policy = ContextCompetitionPolicy::try_new(
            context_policy.minimum_support,
            context_policy.ambiguity_band,
        )?;
        Ok(Self {
            history_capacity,
            context_policy,
            subjects: HashMap::new(),
        })
    }

    pub fn context_policy(&self) -> ContextCompetitionPolicy {
        self.context_policy
    }

    pub fn subject_count(&self) -> usize {
        self.subjects.len()
    }

    pub fn subject_state(&self, subject_id: &str) -> Option<&MetaEpistemicState> {
        self.subjects.get(subject_id)
    }

    /// Context competition and candidate admission occur before any longitudinal state mutation.
    /// Therefore insufficient context support or malformed candidate evidence cannot advance an
    /// episode sequence.
    pub fn reason(
        &mut self,
        input: CanonicalReasoningInputV2,
    ) -> Result<CanonicalReasoningDecisionV2, ReasoningKernelV2Error> {
        require_nonempty("subject_id", &input.subject_id)?;
        require_nonempty("episode_id", &input.episode_id)?;
        validate_unit("strategy_support", input.strategy_support)?;

        let context_selection = assess_and_select(
            &input.context_hypotheses,
            self.context_policy,
            &input.candidates,
        )?;
        let selected_candidate = input
            .candidates
            .get(context_selection.selected_source_index)
            .cloned()
            .ok_or(ReasoningKernelV2Error::InvalidSelectedIndex(
                context_selection.selected_source_index,
            ))?;

        let signals = MetaSupportSignals::try_new(
            context_selection.assessment.top_support,
            input.strategy_support,
            context_selection.selected_worst_case_score,
        )?;
        let observation = MetaEpisodeObservation {
            episode_id: input.episode_id.clone(),
            sequence: input.sequence,
            // The full ambiguity surface is retained in `context_selection`. Meta-state keeps the
            // deterministic primary label plus decomposed support for longitudinal calibration.
            context: context_selection.assessment.primary_context,
            signals,
            abstained: input.abstained,
            evidence_items: input.evidence_items,
            weak_assumptions_flagged: input.weak_assumptions_flagged,
        };

        let meta_commit = if let Some(state) = self.subjects.get_mut(&input.subject_id) {
            state.commit_observation(observation)?
        } else {
            let mut state = MetaEpistemicState::new(
                input.subject_id.clone(),
                self.history_capacity,
            )?;
            let report = state.commit_observation(observation)?;
            self.subjects.insert(input.subject_id.clone(), state);
            report
        };

        let mut decision = CanonicalReasoningDecisionV2 {
            kernel_version: CANONICAL_REASONING_KERNEL_V2_VERSION.into(),
            authority: MetaStateAuthority::MeasurementOnly,
            subject_id: input.subject_id,
            episode_id: input.episode_id,
            sequence: input.sequence,
            context_selection,
            selected_candidate,
            meta_commit,
            applied_plasticity_multiplier: CANONICAL_REASONING_BASELINE_PLASTICITY,
            decision_commitment: String::new(),
        };
        decision.decision_commitment = compute_decision_commitment(&decision);
        debug_assert!(decision.validate_commitment());
        debug_assert!(!decision.authority.may_control_plasticity());
        Ok(decision)
    }

    pub fn record_outcome(
        &mut self,
        subject_id: &str,
        feedback: MetaOutcomeFeedback,
    ) -> Result<OutcomeCommitReport, ReasoningKernelV2Error> {
        require_nonempty("subject_id", subject_id)?;
        let state = self
            .subjects
            .get_mut(subject_id)
            .ok_or_else(|| ReasoningKernelV2Error::UnknownSubject(subject_id.to_owned()))?;
        Ok(state.record_outcome(feedback)?)
    }

    pub fn checkpoint_subject(
        &self,
        subject_id: &str,
    ) -> Result<CanonicalSubjectCheckpointV2, ReasoningKernelV2Error> {
        require_nonempty("subject_id", subject_id)?;
        let state = self
            .subjects
            .get(subject_id)
            .ok_or_else(|| ReasoningKernelV2Error::UnknownSubject(subject_id.to_owned()))?;
        Ok(CanonicalSubjectCheckpointV2 {
            kernel_version: CANONICAL_REASONING_KERNEL_V2_VERSION.into(),
            context_policy: self.context_policy,
            state: state.checkpoint(),
        })
    }

    pub fn restore_subject(
        &mut self,
        checkpoint: CanonicalSubjectCheckpointV2,
    ) -> Result<(), ReasoningKernelV2Error> {
        if checkpoint.kernel_version != CANONICAL_REASONING_KERNEL_V2_VERSION {
            return Err(ReasoningKernelV2Error::UnsupportedKernelVersion(
                checkpoint.kernel_version,
            ));
        }
        if checkpoint.context_policy != self.context_policy {
            return Err(ReasoningKernelV2Error::ContextPolicyMismatch);
        }
        let state = MetaEpistemicState::restore(checkpoint.state)?;
        let subject_id = state.subject_id().to_owned();
        if self.subjects.contains_key(&subject_id) {
            return Err(ReasoningKernelV2Error::DuplicateSubject(subject_id));
        }
        self.subjects.insert(subject_id, state);
        Ok(())
    }
}

#[derive(Debug)]
pub enum ReasoningKernelV2Error {
    EmptyField(&'static str),
    InvalidHistoryCapacity(usize),
    InvalidUnitValue { field: &'static str, value: f64 },
    InvalidSelectedIndex(usize),
    UnknownSubject(String),
    DuplicateSubject(String),
    UnsupportedKernelVersion(String),
    ContextPolicyMismatch,
    Context(ContextCompetitionError),
    MetaState(MetaStateError),
}

impl fmt::Display for ReasoningKernelV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::InvalidHistoryCapacity(capacity) => {
                write!(f, "reasoning-kernel history capacity must be positive, got {capacity}")
            }
            Self::InvalidUnitValue { field, value } => {
                write!(f, "`{field}` must be finite and within [0, 1], got {value}")
            }
            Self::InvalidSelectedIndex(index) => write!(
                f,
                "context selector returned source index {index} outside the candidate set"
            ),
            Self::UnknownSubject(subject_id) => write!(f, "reasoning subject `{subject_id}` is not present"),
            Self::DuplicateSubject(subject_id) => write!(f, "reasoning subject `{subject_id}` is already live"),
            Self::UnsupportedKernelVersion(version) => {
                write!(f, "unsupported canonical reasoning kernel V2 version `{version}`")
            }
            Self::ContextPolicyMismatch => write!(f, "checkpoint context policy differs from the live kernel policy"),
            Self::Context(err) => write!(f, "context competition failed: {err}"),
            Self::MetaState(err) => write!(f, "metacognitive state rejected episode: {err}"),
        }
    }
}

impl std::error::Error for ReasoningKernelV2Error {}

impl From<ContextCompetitionError> for ReasoningKernelV2Error {
    fn from(value: ContextCompetitionError) -> Self {
        Self::Context(value)
    }
}

impl From<MetaStateError> for ReasoningKernelV2Error {
    fn from(value: MetaStateError) -> Self {
        Self::MetaState(value)
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ReasoningKernelV2Error> {
    if value.trim().is_empty() {
        Err(ReasoningKernelV2Error::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_unit(field: &'static str, value: f64) -> Result<(), ReasoningKernelV2Error> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ReasoningKernelV2Error::InvalidUnitValue { field, value })
    }
}

fn compute_decision_commitment(decision: &CanonicalReasoningDecisionV2) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, DECISION_COMMITMENT_DOMAIN);
    hash_str(&mut hasher, &decision.kernel_version);
    hash_str(&mut hasher, &decision.subject_id);
    hash_str(&mut hasher, &decision.episode_id);
    hash_u64(&mut hasher, decision.sequence);

    let assessment = &decision.context_selection.assessment;
    hash_u64(&mut hasher, assessment.policy.minimum_support.to_bits());
    hash_u64(&mut hasher, assessment.policy.ambiguity_band.to_bits());
    hash_u64(&mut hasher, context_rank(assessment.primary_context) as u64);
    hash_u64(&mut hasher, assessment.top_support.to_bits());
    hash_optional_f64(&mut hasher, assessment.second_support);
    hash_optional_f64(&mut hasher, assessment.margin);
    hash_u64(
        &mut hasher,
        match assessment.resolution {
            ContextResolution::Resolved => 0,
            ContextResolution::Ambiguous => 1,
        },
    );
    hash_u64(&mut hasher, assessment.active_contexts.len() as u64);
    for context in &assessment.active_contexts {
        hash_u64(&mut hasher, context_rank(*context) as u64);
    }
    hash_u64(&mut hasher, assessment.hypotheses.len() as u64);
    for hypothesis in &assessment.hypotheses {
        hash_u64(&mut hasher, context_rank(hypothesis.context) as u64);
        hash_u64(&mut hasher, hypothesis.support.to_bits());
        hash_str(&mut hasher, &hypothesis.source);
        hash_u64(&mut hasher, hypothesis.evidence_refs.len() as u64);
        for evidence_ref in &hypothesis.evidence_refs {
            hash_str(&mut hasher, evidence_ref);
        }
    }

    hash_u64(
        &mut hasher,
        decision.context_selection.evaluations.len() as u64,
    );
    for evaluation in &decision.context_selection.evaluations {
        hash_u64(&mut hasher, evaluation.source_index as u64);
        hash_str(&mut hasher, &evaluation.candidate_name);
        hash_u64(&mut hasher, evaluation.vector.integration_proxy.to_bits());
        hash_u64(&mut hasher, evaluation.vector.harmonic_alignment.to_bits());
        hash_u64(&mut hasher, evaluation.vector.epistemic_grounding.to_bits());
        hash_u64(&mut hasher, evaluation.worst_case_score.to_bits());
        hash_u64(&mut hasher, evaluation.mean_score.to_bits());
        hash_bool(&mut hasher, evaluation.pareto_optimal);
        hash_u64(&mut hasher, evaluation.context_scores.len() as u64);
        for score in &evaluation.context_scores {
            hash_u64(&mut hasher, context_rank(score.context) as u64);
            hash_u64(&mut hasher, score.weights.integration_proxy.to_bits());
            hash_u64(&mut hasher, score.weights.harmonic_alignment.to_bits());
            hash_u64(&mut hasher, score.weights.epistemic_grounding.to_bits());
            hash_u64(&mut hasher, score.weighted_score.to_bits());
        }
    }

    hash_u64(
        &mut hasher,
        decision.context_selection.selected_source_index as u64,
    );
    hash_str(
        &mut hasher,
        &decision.context_selection.selected_candidate_name,
    );
    hash_u64(
        &mut hasher,
        decision.context_selection.selected_vector.integration_proxy.to_bits(),
    );
    hash_u64(
        &mut hasher,
        decision.context_selection.selected_vector.harmonic_alignment.to_bits(),
    );
    hash_u64(
        &mut hasher,
        decision.context_selection.selected_vector.epistemic_grounding.to_bits(),
    );
    hash_u64(
        &mut hasher,
        decision.context_selection.selected_worst_case_score.to_bits(),
    );
    hash_u64(
        &mut hasher,
        decision.context_selection.selected_mean_score.to_bits(),
    );

    hash_str(&mut hasher, &decision.selected_candidate.name);
    hash_str(&mut hasher, &decision.selected_candidate.definition);
    hash_str(
        &mut hasher,
        &format!("{:?}", decision.selected_candidate.tier),
    );
    hash_bytes(&mut hasher, &decision.selected_candidate.encoding.0);
    hash_u64(&mut hasher, decision.selected_candidate.fitness.to_bits());
    hash_u64(
        &mut hasher,
        decision.selected_candidate.harmonic_alignment.to_bits(),
    );
    hash_str(
        &mut hasher,
        &format!("{:?}", decision.selected_candidate.epistemic_coordinate),
    );

    hash_str(&mut hasher, &decision.meta_commit.subject_id);
    hash_str(&mut hasher, &decision.meta_commit.episode_id);
    hash_u64(&mut hasher, decision.meta_commit.sequence);
    hash_signals(&mut hasher, decision.meta_commit.current_signals);
    match decision.meta_commit.previous_signals {
        Some(signals) => {
            hash_bool(&mut hasher, true);
            hash_signals(&mut hasher, signals);
        }
        None => hash_bool(&mut hasher, false),
    }
    match decision.meta_commit.revision {
        Some(revision) => {
            hash_bool(&mut hasher, true);
            hash_revision(&mut hasher, revision);
        }
        None => hash_bool(&mut hasher, false),
    }
    hash_u64(&mut hasher, decision.meta_commit.retained_records as u64);
    hash_u64(&mut hasher, decision.meta_commit.unresolved_records as u64);
    hash_u64(
        &mut hasher,
        decision.applied_plasticity_multiplier.to_bits(),
    );
    hasher.finalize().to_hex().to_string()
}

fn hash_signals(hasher: &mut blake3::Hasher, signals: MetaSupportSignals) {
    hash_u64(hasher, signals.context_support.to_bits());
    hash_u64(hasher, signals.strategy_support.to_bits());
    hash_u64(hasher, signals.selection_support.to_bits());
}

fn hash_revision(hasher: &mut blake3::Hasher, revision: MetaSignalRevision) {
    hash_u64(hasher, revision.context_support_delta.to_bits());
    hash_u64(hasher, revision.strategy_support_delta.to_bits());
    hash_u64(hasher, revision.selection_support_delta.to_bits());
}

fn hash_optional_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hash_bool(hasher, true);
            hash_u64(hasher, value.to_bits());
        }
        None => hash_bool(hasher, false),
    }
}

fn hash_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

const fn context_rank(context: crate::consciousness::context_aware_evolution::ReasoningContext) -> u8 {
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    match context {
        ReasoningContext::CriticalSafety => 0,
        ReasoningContext::ScientificReasoning => 1,
        ReasoningContext::TechnicalImplementation => 2,
        ReasoningContext::Learning => 3,
        ReasoningContext::SocialInteraction => 4,
        ReasoningContext::PhilosophicalInquiry => 5,
        ReasoningContext::CreativeExploration => 6,
        ReasoningContext::GeneralReasoning => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

    fn hypothesis(context: ReasoningContext, support: f64) -> ContextHypothesis {
        ContextHypothesis {
            context,
            support,
            source: "fixture-adapter-v1".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    fn candidate(
        name: &str,
        fitness: f64,
        harmonic_alignment: f64,
        epistemic_coordinate: EpistemicCoordinate,
    ) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::Physical,
            definition: format!("fixture-{name}"),
            fitness,
            encoding: BinaryHV::random(name.len() as u64 + 900),
            epistemic_coordinate,
            harmonic_alignment,
        }
    }

    fn input(subject: &str, episode: &str, sequence: u64) -> CanonicalReasoningInputV2 {
        CanonicalReasoningInputV2 {
            subject_id: subject.into(),
            episode_id: episode.into(),
            sequence,
            context_hypotheses: vec![
                hypothesis(ReasoningContext::TechnicalImplementation, 0.88),
                hypothesis(ReasoningContext::CriticalSafety, 0.84),
            ],
            strategy_support: 0.7,
            abstained: false,
            evidence_items: 1,
            weak_assumptions_flagged: 0,
            candidates: vec![
                candidate("safety-only", 0.5, 0.9, EpistemicCoordinate::null()),
                candidate("robust", 0.5, 0.55, EpistemicCoordinate::axiom()),
            ],
        }
    }

    #[test]
    fn ambiguous_contexts_select_robust_candidate_and_commit_state() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        let decision = kernel.reason(input("agent-a", "ep-0", 0)).unwrap();
        assert_eq!(decision.selected_candidate.name, "robust");
        assert_eq!(decision.context_selection.assessment.active_contexts.len(), 2);
        assert_eq!(decision.applied_plasticity_multiplier, 1.0);
        assert!(decision.validate_commitment());
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), 1);
    }

    #[test]
    fn insufficient_context_support_does_not_create_subject() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        let mut request = input("agent-a", "ep-0", 0);
        request.context_hypotheses = vec![hypothesis(ReasoningContext::GeneralReasoning, 0.4)];
        assert!(kernel.reason(request).is_err());
        assert_eq!(kernel.subject_count(), 0);
    }

    #[test]
    fn invalid_candidate_does_not_create_subject() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        let mut request = input("agent-a", "ep-0", 0);
        request.candidates[0].fitness = f64::NAN;
        assert!(kernel.reason(request).is_err());
        assert_eq!(kernel.subject_count(), 0);
    }

    #[test]
    fn tampered_winner_fails_commitment_validation() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        let mut decision = kernel.reason(input("agent-a", "ep-0", 0)).unwrap();
        assert!(decision.validate_commitment());
        decision.selected_candidate.name = "tampered".into();
        assert!(!decision.validate_commitment());
    }

    #[test]
    fn tampered_losing_candidate_evaluation_fails_commitment_validation() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        let mut decision = kernel.reason(input("agent-a", "ep-0", 0)).unwrap();
        assert!(decision.validate_commitment());
        decision.context_selection.evaluations[0].candidate_name = "tampered-loser".into();
        assert!(!decision.validate_commitment());
    }

    #[test]
    fn subjects_remain_isolated() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        kernel.reason(input("agent-a", "ep-0", 0)).unwrap();
        kernel.reason(input("agent-b", "ep-0", 0)).unwrap();
        assert_eq!(kernel.subject_count(), 2);
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), 1);
        assert_eq!(kernel.subject_state("agent-b").unwrap().next_sequence(), 1);
    }

    #[test]
    fn checkpoint_binds_context_policy() {
        let mut kernel = CanonicalReasoningKernelV2::new(8).unwrap();
        kernel.reason(input("agent-a", "ep-0", 0)).unwrap();
        let checkpoint = kernel.checkpoint_subject("agent-a").unwrap();

        let different_policy = ContextCompetitionPolicy::try_new(0.6, 0.05).unwrap();
        let mut restored = CanonicalReasoningKernelV2::with_context_policy(8, different_policy).unwrap();
        assert!(matches!(
            restored.restore_subject(checkpoint),
            Err(ReasoningKernelV2Error::ContextPolicyMismatch)
        ));
    }
}
