// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical reasoning orchestration kernel.
//!
//! This module composes the typed objective core with the two-phase metacognitive state machine.
//! It intentionally does not inherit the historical `UnifiedIntelligence` scalar or grant
//! heuristic support signals authority over learning rate.

use super::reasoning_meta_state::{
    MetaCommitReport, MetaEpistemicState, MetaEpisodeObservation, MetaOutcomeFeedback,
    MetaStateAuthority, MetaStateCheckpoint, MetaStateError, MetaSupportSignals,
    OutcomeCommitReport,
};
use super::reasoning_objective_core::{
    select_candidate_by_objectives, ObjectiveCoreError, ObjectiveSelectionReport,
};
use crate::consciousness::context_aware_evolution::ReasoningContext;
use crate::consciousness::primitive_evolution::CandidatePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub const CANONICAL_REASONING_KERNEL_VERSION: &str = "rq-006-canonical-kernel-v1";
pub const BASELINE_PLASTICITY_MULTIPLIER: f64 = 1.0;

/// One pre-outcome request to the canonical kernel.
#[derive(Debug, Clone)]
pub struct CanonicalReasoningInput {
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub context: ReasoningContext,
    /// Measurement supplied by the context-assessment adapter. Not treated as answer correctness.
    pub context_support: f64,
    /// Measurement supplied by the strategy-assessment adapter. Not treated as answer correctness.
    pub strategy_support: f64,
    pub abstained: bool,
    pub evidence_items: usize,
    pub weak_assumptions_flagged: usize,
    pub candidates: Vec<CandidatePrimitive>,
}

/// Mechanically auditable decision surface for one episode.
#[derive(Debug, Clone)]
pub struct CanonicalReasoningDecision {
    pub kernel_version: String,
    pub authority: MetaStateAuthority,
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub context: ReasoningContext,
    pub selected_candidate: CandidatePrimitive,
    pub objective_selection: ObjectiveSelectionReport,
    pub meta_commit: MetaCommitReport,
    /// Learning/plasticity remains at baseline while metacognitive control authority is unqualified.
    pub applied_plasticity_multiplier: f64,
}

/// Serializable collection checkpoint for one subject. The contained state checkpoint remains the
/// sole persistent state; kernel version is included to bind orchestration semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSubjectCheckpoint {
    pub kernel_version: String,
    pub state: MetaStateCheckpoint,
}

/// Replacement orchestration path for evidence-first reasoning experiments.
pub struct CanonicalReasoningKernel {
    history_capacity: usize,
    subjects: HashMap<String, MetaEpistemicState>,
}

impl CanonicalReasoningKernel {
    pub fn new(history_capacity: usize) -> Result<Self, ReasoningKernelError> {
        if history_capacity == 0 {
            return Err(ReasoningKernelError::InvalidHistoryCapacity(
                history_capacity,
            ));
        }
        Ok(Self {
            history_capacity,
            subjects: HashMap::new(),
        })
    }

    pub fn subject_count(&self) -> usize {
        self.subjects.len()
    }

    pub fn subject_state(&self, subject_id: &str) -> Option<&MetaEpistemicState> {
        self.subjects.get(subject_id)
    }

    /// Select a candidate from real normalized objective coordinates, freeze the current episode,
    /// and return only measurement-authority state. No outcome information is accepted here.
    pub fn reason(
        &mut self,
        input: CanonicalReasoningInput,
    ) -> Result<CanonicalReasoningDecision, ReasoningKernelError> {
        require_nonempty("subject_id", &input.subject_id)?;
        require_nonempty("episode_id", &input.episode_id)?;

        // Objective admission/selection happens before mutating subject state. Malformed candidate
        // evidence therefore cannot partially advance longitudinal episode sequence.
        let objective_selection =
            select_candidate_by_objectives(input.context, &input.candidates)?;
        let selected_candidate = input
            .candidates
            .get(objective_selection.selected_source_index)
            .cloned()
            .ok_or(ReasoningKernelError::InvalidSelectedIndex(
                objective_selection.selected_source_index,
            ))?;

        let signals = MetaSupportSignals::try_new(
            input.context_support,
            input.strategy_support,
            objective_selection.selected_weighted_score,
        )?;
        let observation = MetaEpisodeObservation {
            episode_id: input.episode_id.clone(),
            sequence: input.sequence,
            context: input.context,
            signals,
            abstained: input.abstained,
            evidence_items: input.evidence_items,
            weak_assumptions_flagged: input.weak_assumptions_flagged,
        };

        let meta_commit = if let Some(state) = self.subjects.get_mut(&input.subject_id) {
            state.commit_observation(observation)?
        } else {
            // New subjects are inserted only after their first observation validates and commits.
            let mut state = MetaEpistemicState::new(
                input.subject_id.clone(),
                self.history_capacity,
            )?;
            let report = state.commit_observation(observation)?;
            self.subjects.insert(input.subject_id.clone(), state);
            report
        };

        debug_assert_eq!(meta_commit.authority, MetaStateAuthority::MeasurementOnly);
        debug_assert!(!meta_commit.authority.may_control_plasticity());

        Ok(CanonicalReasoningDecision {
            kernel_version: CANONICAL_REASONING_KERNEL_VERSION.into(),
            authority: MetaStateAuthority::MeasurementOnly,
            subject_id: input.subject_id,
            episode_id: input.episode_id,
            sequence: input.sequence,
            context: input.context,
            selected_candidate,
            objective_selection,
            meta_commit,
            applied_plasticity_multiplier: BASELINE_PLASTICITY_MULTIPLIER,
        })
    }

    /// Bind post-outcome feedback to an already frozen episode.
    pub fn record_outcome(
        &mut self,
        subject_id: &str,
        feedback: MetaOutcomeFeedback,
    ) -> Result<OutcomeCommitReport, ReasoningKernelError> {
        require_nonempty("subject_id", subject_id)?;
        let state = self
            .subjects
            .get_mut(subject_id)
            .ok_or_else(|| ReasoningKernelError::UnknownSubject(subject_id.to_owned()))?;
        Ok(state.record_outcome(feedback)?)
    }

    pub fn checkpoint_subject(
        &self,
        subject_id: &str,
    ) -> Result<CanonicalSubjectCheckpoint, ReasoningKernelError> {
        require_nonempty("subject_id", subject_id)?;
        let state = self
            .subjects
            .get(subject_id)
            .ok_or_else(|| ReasoningKernelError::UnknownSubject(subject_id.to_owned()))?;
        Ok(CanonicalSubjectCheckpoint {
            kernel_version: CANONICAL_REASONING_KERNEL_VERSION.into(),
            state: state.checkpoint(),
        })
    }

    /// Restore one subject without silently replacing an existing live subject.
    pub fn restore_subject(
        &mut self,
        checkpoint: CanonicalSubjectCheckpoint,
    ) -> Result<(), ReasoningKernelError> {
        if checkpoint.kernel_version != CANONICAL_REASONING_KERNEL_VERSION {
            return Err(ReasoningKernelError::UnsupportedKernelVersion(
                checkpoint.kernel_version,
            ));
        }
        let state = MetaEpistemicState::restore(checkpoint.state)?;
        let subject_id = state.subject_id().to_owned();
        if self.subjects.contains_key(&subject_id) {
            return Err(ReasoningKernelError::DuplicateSubject(subject_id));
        }
        self.subjects.insert(subject_id, state);
        Ok(())
    }
}

#[derive(Debug)]
pub enum ReasoningKernelError {
    EmptyField(&'static str),
    InvalidHistoryCapacity(usize),
    InvalidSelectedIndex(usize),
    UnknownSubject(String),
    DuplicateSubject(String),
    UnsupportedKernelVersion(String),
    Objective(ObjectiveCoreError),
    MetaState(MetaStateError),
}

impl fmt::Display for ReasoningKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::InvalidHistoryCapacity(capacity) => {
                write!(f, "reasoning-kernel history capacity must be positive, got {capacity}")
            }
            Self::InvalidSelectedIndex(index) => write!(
                f,
                "objective selector returned source index {index} outside the candidate set"
            ),
            Self::UnknownSubject(subject_id) => {
                write!(f, "reasoning subject `{subject_id}` is not present")
            }
            Self::DuplicateSubject(subject_id) => {
                write!(f, "reasoning subject `{subject_id}` is already live")
            }
            Self::UnsupportedKernelVersion(version) => {
                write!(f, "unsupported canonical reasoning kernel version `{version}`")
            }
            Self::Objective(err) => write!(f, "objective selection failed: {err}"),
            Self::MetaState(err) => write!(f, "metacognitive state rejected episode: {err}"),
        }
    }
}

impl std::error::Error for ReasoningKernelError {}

impl From<ObjectiveCoreError> for ReasoningKernelError {
    fn from(value: ObjectiveCoreError) -> Self {
        Self::Objective(value)
    }
}

impl From<MetaStateError> for ReasoningKernelError {
    fn from(value: MetaStateError) -> Self {
        Self::MetaState(value)
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ReasoningKernelError> {
    if value.trim().is_empty() {
        Err(ReasoningKernelError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

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
            encoding: BinaryHV::random(name.len() as u64 + 200),
            epistemic_coordinate,
            harmonic_alignment,
        }
    }

    fn input(
        subject_id: &str,
        episode_id: &str,
        sequence: u64,
        context: ReasoningContext,
        candidates: Vec<CandidatePrimitive>,
    ) -> CanonicalReasoningInput {
        CanonicalReasoningInput {
            subject_id: subject_id.into(),
            episode_id: episode_id.into(),
            sequence,
            context,
            context_support: 0.7,
            strategy_support: 0.6,
            abstained: false,
            evidence_items: 2,
            weak_assumptions_flagged: 0,
            candidates,
        }
    }

    fn outcome(sequence: u64, episode_id: &str, correct: bool) -> MetaOutcomeFeedback {
        MetaOutcomeFeedback {
            episode_id: episode_id.into(),
            sequence,
            exact_correct: correct,
            task_score: if correct { 1.0 } else { 0.0 },
            unsafe_action_observed: None,
        }
    }

    #[test]
    fn canonical_kernel_selects_real_objectives_and_persists_subject_state() {
        let mut kernel = CanonicalReasoningKernel::new(8).unwrap();
        let low = candidate("low", 0.5, 0.1, EpistemicCoordinate::null());
        let high = candidate("high", 0.5, 0.9, EpistemicCoordinate::null());
        let decision = kernel
            .reason(input(
                "agent-a",
                "episode-0",
                0,
                ReasoningContext::SocialInteraction,
                vec![low, high],
            ))
            .unwrap();

        assert_eq!(decision.selected_candidate.name, "high");
        assert_eq!(decision.authority, MetaStateAuthority::MeasurementOnly);
        assert_eq!(decision.applied_plasticity_multiplier, 1.0);
        assert_eq!(kernel.subject_count(), 1);
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), 1);

        kernel
            .record_outcome("agent-a", outcome(0, "episode-0", true))
            .unwrap();
        let second = kernel
            .reason(input(
                "agent-a",
                "episode-1",
                1,
                ReasoningContext::TechnicalImplementation,
                vec![candidate(
                    "grounded",
                    0.5,
                    0.5,
                    EpistemicCoordinate::axiom(),
                )],
            ))
            .unwrap();
        assert_eq!(second.sequence, 1);
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), 2);
    }

    #[test]
    fn invalid_candidate_does_not_create_or_advance_subject() {
        let mut kernel = CanonicalReasoningKernel::new(8).unwrap();
        let bad = candidate("bad", f64::NAN, 0.5, EpistemicCoordinate::null());
        assert!(kernel
            .reason(input(
                "agent-a",
                "episode-0",
                0,
                ReasoningContext::GeneralReasoning,
                vec![bad],
            ))
            .is_err());
        assert_eq!(kernel.subject_count(), 0);
    }

    #[test]
    fn failed_sequence_does_not_advance_existing_subject() {
        let mut kernel = CanonicalReasoningKernel::new(8).unwrap();
        kernel
            .reason(input(
                "agent-a",
                "episode-0",
                0,
                ReasoningContext::GeneralReasoning,
                vec![candidate("ok", 0.5, 0.5, EpistemicCoordinate::null())],
            ))
            .unwrap();
        let before = kernel.subject_state("agent-a").unwrap().next_sequence();
        assert!(kernel
            .reason(input(
                "agent-a",
                "episode-2",
                2,
                ReasoningContext::GeneralReasoning,
                vec![candidate("ok2", 0.5, 0.5, EpistemicCoordinate::null())],
            ))
            .is_err());
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), before);
    }

    #[test]
    fn logical_subjects_are_isolated() {
        let mut kernel = CanonicalReasoningKernel::new(8).unwrap();
        for subject in ["agent-a", "agent-b"] {
            kernel
                .reason(input(
                    subject,
                    "episode-0",
                    0,
                    ReasoningContext::GeneralReasoning,
                    vec![candidate(subject, 0.5, 0.5, EpistemicCoordinate::null())],
                ))
                .unwrap();
        }
        assert_eq!(kernel.subject_count(), 2);
        assert_eq!(kernel.subject_state("agent-a").unwrap().next_sequence(), 1);
        assert_eq!(kernel.subject_state("agent-b").unwrap().next_sequence(), 1);
    }

    #[test]
    fn checkpoint_round_trip_does_not_overwrite_live_subject() {
        let mut kernel = CanonicalReasoningKernel::new(8).unwrap();
        kernel
            .reason(input(
                "agent-a",
                "episode-0",
                0,
                ReasoningContext::GeneralReasoning,
                vec![candidate("ok", 0.5, 0.5, EpistemicCoordinate::null())],
            ))
            .unwrap();
        kernel
            .record_outcome("agent-a", outcome(0, "episode-0", true))
            .unwrap();
        let checkpoint = kernel.checkpoint_subject("agent-a").unwrap();

        let mut restored = CanonicalReasoningKernel::new(8).unwrap();
        restored.restore_subject(checkpoint.clone()).unwrap();
        assert_eq!(restored.subject_state("agent-a").unwrap().next_sequence(), 1);
        assert!(matches!(
            restored.restore_subject(checkpoint),
            Err(ReasoningKernelError::DuplicateSubject(_))
        ));
    }
}
