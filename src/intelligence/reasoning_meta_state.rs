// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical two-phase metacognitive state.
//!
//! The state machine separates what the subject reports before an outcome is known from feedback
//! attached after the outcome is revealed. This prevents benchmark targets or post-hoc correctness
//! from leaking into the state that is later evaluated for calibration.
//!
//! There is intentionally no scalar "meta-confidence" and no plasticity authority in this module.
//! Context support, strategy support, and objective-selection support remain separate measurements.
//! All outputs are `MeasurementOnly` until an external qualification artifact establishes stronger
//! authority.

use crate::consciousness::context_aware_evolution::ReasoningContext;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fmt;

pub const META_EPISTEMIC_STATE_SCHEMA_VERSION: u32 = 1;

/// Current authority of the canonical meta-state surface.
///
/// There is deliberately no stronger variant. A future authority upgrade must be introduced by an
/// evidence-backed API change rather than by crossing a local numeric threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaStateAuthority {
    MeasurementOnly,
}

impl MetaStateAuthority {
    pub const fn may_control_plasticity(self) -> bool {
        false
    }
}

/// Decomposed pre-outcome support signals. None is a probability of answer correctness merely by
/// virtue of lying in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetaSupportSignals {
    pub context_support: f64,
    pub strategy_support: f64,
    pub selection_support: f64,
}

impl MetaSupportSignals {
    pub fn try_new(
        context_support: f64,
        strategy_support: f64,
        selection_support: f64,
    ) -> Result<Self, MetaStateError> {
        validate_unit("context_support", context_support)?;
        validate_unit("strategy_support", strategy_support)?;
        validate_unit("selection_support", selection_support)?;
        Ok(Self {
            context_support,
            strategy_support,
            selection_support,
        })
    }

    pub fn revision_from(self, previous: Self) -> MetaSignalRevision {
        MetaSignalRevision {
            context_support_delta: self.context_support - previous.context_support,
            strategy_support_delta: self.strategy_support - previous.strategy_support,
            selection_support_delta: self.selection_support - previous.selection_support,
        }
    }
}

/// One frozen pre-outcome observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaEpisodeObservation {
    pub episode_id: String,
    /// Strictly monotonic logical sequence for this subject state.
    pub sequence: u64,
    pub context: ReasoningContext,
    pub signals: MetaSupportSignals,
    pub abstained: bool,
    pub evidence_items: usize,
    pub weak_assumptions_flagged: usize,
}

/// Outcome information attached only after the pre-outcome observation has been frozen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaOutcomeFeedback {
    pub episode_id: String,
    pub sequence: u64,
    pub exact_correct: bool,
    /// Task-native normalized score. This is kept separate from exact correctness.
    pub task_score: f64,
    /// Optional benchmark/action safety observation. `None` when the task has no such oracle.
    pub unsafe_action_observed: Option<bool>,
}

/// One complete or still-unresolved longitudinal record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaEpisodeRecord {
    pub observation: MetaEpisodeObservation,
    pub outcome: Option<MetaOutcomeFeedback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetaSignalRevision {
    pub context_support_delta: f64,
    pub strategy_support_delta: f64,
    pub selection_support_delta: f64,
}

/// Result returned immediately after the current episode is committed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaCommitReport {
    pub authority: MetaStateAuthority,
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub current_signals: MetaSupportSignals,
    pub previous_signals: Option<MetaSupportSignals>,
    pub revision: Option<MetaSignalRevision>,
    pub retained_records: usize,
    pub unresolved_records: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeCommitReport {
    pub authority: MetaStateAuthority,
    pub subject_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub retained_records: usize,
    pub unresolved_records: usize,
}

/// Serializable checkpoint. Restore always re-validates all state-machine invariants; callers do
/// not deserialize directly into the live state type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaStateCheckpoint {
    pub schema_version: u32,
    pub subject_id: String,
    pub history_capacity: usize,
    pub next_sequence: u64,
    pub records: Vec<MetaEpisodeRecord>,
}

/// Live metacognitive state for one logical subject/agent.
#[derive(Debug, Clone)]
pub struct MetaEpistemicState {
    subject_id: String,
    history_capacity: usize,
    next_sequence: u64,
    records: VecDeque<MetaEpisodeRecord>,
}

impl MetaEpistemicState {
    pub fn new(
        subject_id: impl Into<String>,
        history_capacity: usize,
    ) -> Result<Self, MetaStateError> {
        let subject_id = subject_id.into();
        require_nonempty("subject_id", &subject_id)?;
        if history_capacity == 0 {
            return Err(MetaStateError::InvalidHistoryCapacity(history_capacity));
        }
        Ok(Self {
            subject_id,
            history_capacity,
            next_sequence: 0,
            records: VecDeque::with_capacity(history_capacity),
        })
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub const fn authority(&self) -> MetaStateAuthority {
        MetaStateAuthority::MeasurementOnly
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub fn records(&self) -> &VecDeque<MetaEpisodeRecord> {
        &self.records
    }

    pub fn current(&self) -> Option<&MetaEpisodeRecord> {
        self.records.back()
    }

    /// Freeze a pre-outcome observation and make it the current state before returning.
    pub fn commit_observation(
        &mut self,
        observation: MetaEpisodeObservation,
    ) -> Result<MetaCommitReport, MetaStateError> {
        validate_observation(&observation)?;
        if observation.sequence != self.next_sequence {
            return Err(MetaStateError::SequenceMismatch {
                expected: self.next_sequence,
                found: observation.sequence,
            });
        }
        if self
            .records
            .iter()
            .any(|record| record.observation.episode_id == observation.episode_id)
        {
            return Err(MetaStateError::DuplicateEpisodeId(
                observation.episode_id.clone(),
            ));
        }

        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(MetaStateError::SequenceOverflow)?;

        if self.records.len() == self.history_capacity {
            match self.records.front() {
                Some(record) if record.outcome.is_none() => {
                    return Err(MetaStateError::UnresolvedHistoryWouldBeEvicted {
                        episode_id: record.observation.episode_id.clone(),
                        sequence: record.observation.sequence,
                    });
                }
                Some(_) => {
                    self.records.pop_front();
                }
                None => {}
            }
        }

        let previous_signals = self.records.back().map(|record| record.observation.signals);
        let revision = previous_signals.map(|previous| observation.signals.revision_from(previous));
        let episode_id = observation.episode_id.clone();
        let sequence = observation.sequence;
        let current_signals = observation.signals;
        self.records.push_back(MetaEpisodeRecord {
            observation,
            outcome: None,
        });
        self.next_sequence = next_sequence;

        Ok(MetaCommitReport {
            authority: MetaStateAuthority::MeasurementOnly,
            subject_id: self.subject_id.clone(),
            episode_id,
            sequence,
            current_signals,
            previous_signals,
            revision,
            retained_records: self.records.len(),
            unresolved_records: self.unresolved_records(),
        })
    }

    /// Attach an outcome to a previously frozen episode. Outcome feedback can never create an
    /// episode or rewrite its pre-outcome observation.
    pub fn record_outcome(
        &mut self,
        feedback: MetaOutcomeFeedback,
    ) -> Result<OutcomeCommitReport, MetaStateError> {
        validate_feedback(&feedback)?;
        let record = self
            .records
            .iter_mut()
            .find(|record| record.observation.sequence == feedback.sequence)
            .ok_or(MetaStateError::UnknownEpisodeSequence(feedback.sequence))?;

        if record.observation.episode_id != feedback.episode_id {
            return Err(MetaStateError::EpisodeIdentityMismatch {
                sequence: feedback.sequence,
                expected: record.observation.episode_id.clone(),
                found: feedback.episode_id,
            });
        }
        if record.outcome.is_some() {
            return Err(MetaStateError::DuplicateOutcome {
                episode_id: record.observation.episode_id.clone(),
                sequence: record.observation.sequence,
            });
        }

        let episode_id = feedback.episode_id.clone();
        let sequence = feedback.sequence;
        record.outcome = Some(feedback);
        Ok(OutcomeCommitReport {
            authority: MetaStateAuthority::MeasurementOnly,
            subject_id: self.subject_id.clone(),
            episode_id,
            sequence,
            retained_records: self.records.len(),
            unresolved_records: self.unresolved_records(),
        })
    }

    pub fn checkpoint(&self) -> MetaStateCheckpoint {
        MetaStateCheckpoint {
            schema_version: META_EPISTEMIC_STATE_SCHEMA_VERSION,
            subject_id: self.subject_id.clone(),
            history_capacity: self.history_capacity,
            next_sequence: self.next_sequence,
            records: self.records.iter().cloned().collect(),
        }
    }

    /// Restore only after re-validating checkpoint structure, ordering, identities and values.
    pub fn restore(checkpoint: MetaStateCheckpoint) -> Result<Self, MetaStateError> {
        if checkpoint.schema_version != META_EPISTEMIC_STATE_SCHEMA_VERSION {
            return Err(MetaStateError::UnsupportedSchemaVersion(
                checkpoint.schema_version,
            ));
        }
        require_nonempty("subject_id", &checkpoint.subject_id)?;
        if checkpoint.history_capacity == 0 {
            return Err(MetaStateError::InvalidHistoryCapacity(
                checkpoint.history_capacity,
            ));
        }
        if checkpoint.records.len() > checkpoint.history_capacity {
            return Err(MetaStateError::CheckpointExceedsCapacity {
                records: checkpoint.records.len(),
                capacity: checkpoint.history_capacity,
            });
        }

        let mut ids = HashSet::with_capacity(checkpoint.records.len());
        let mut previous_sequence = None;
        for record in &checkpoint.records {
            validate_observation(&record.observation)?;
            if !ids.insert(record.observation.episode_id.as_str()) {
                return Err(MetaStateError::DuplicateEpisodeId(
                    record.observation.episode_id.clone(),
                ));
            }
            if let Some(previous) = previous_sequence {
                let expected = previous
                    .checked_add(1)
                    .ok_or(MetaStateError::SequenceOverflow)?;
                if record.observation.sequence != expected {
                    return Err(MetaStateError::NonContiguousCheckpoint {
                        expected,
                        found: record.observation.sequence,
                    });
                }
            }
            previous_sequence = Some(record.observation.sequence);

            if let Some(outcome) = &record.outcome {
                validate_feedback(outcome)?;
                if outcome.sequence != record.observation.sequence
                    || outcome.episode_id != record.observation.episode_id
                {
                    return Err(MetaStateError::CheckpointOutcomeMismatch {
                        observation_episode_id: record.observation.episode_id.clone(),
                        outcome_episode_id: outcome.episode_id.clone(),
                    });
                }
            }
        }

        match checkpoint.records.last() {
            Some(last) => {
                let expected_next = last
                    .observation
                    .sequence
                    .checked_add(1)
                    .ok_or(MetaStateError::SequenceOverflow)?;
                if checkpoint.next_sequence != expected_next {
                    return Err(MetaStateError::SequenceMismatch {
                        expected: expected_next,
                        found: checkpoint.next_sequence,
                    });
                }
            }
            None if checkpoint.next_sequence != 0 => {
                return Err(MetaStateError::SequenceMismatch {
                    expected: 0,
                    found: checkpoint.next_sequence,
                });
            }
            None => {}
        }

        Ok(Self {
            subject_id: checkpoint.subject_id,
            history_capacity: checkpoint.history_capacity,
            next_sequence: checkpoint.next_sequence,
            records: checkpoint.records.into_iter().collect(),
        })
    }

    fn unresolved_records(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.outcome.is_none())
            .count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaStateError {
    EmptyField(&'static str),
    InvalidHistoryCapacity(usize),
    InvalidUnitValue { field: &'static str, value: f64 },
    SequenceMismatch { expected: u64, found: u64 },
    SequenceOverflow,
    DuplicateEpisodeId(String),
    UnknownEpisodeSequence(u64),
    EpisodeIdentityMismatch {
        sequence: u64,
        expected: String,
        found: String,
    },
    DuplicateOutcome { episode_id: String, sequence: u64 },
    UnresolvedHistoryWouldBeEvicted { episode_id: String, sequence: u64 },
    UnsupportedSchemaVersion(u32),
    CheckpointExceedsCapacity { records: usize, capacity: usize },
    NonContiguousCheckpoint { expected: u64, found: u64 },
    CheckpointOutcomeMismatch {
        observation_episode_id: String,
        outcome_episode_id: String,
    },
}

impl fmt::Display for MetaStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::InvalidHistoryCapacity(capacity) => {
                write!(f, "meta-state history capacity must be positive, got {capacity}")
            }
            Self::InvalidUnitValue { field, value } => {
                write!(f, "`{field}` must be finite and within [0, 1], got {value}")
            }
            Self::SequenceMismatch { expected, found } => {
                write!(f, "episode sequence mismatch: expected {expected}, found {found}")
            }
            Self::SequenceOverflow => write!(f, "meta-state episode sequence overflow"),
            Self::DuplicateEpisodeId(id) => write!(f, "episode id `{id}` is already retained"),
            Self::UnknownEpisodeSequence(sequence) => {
                write!(f, "no retained episode has sequence {sequence}")
            }
            Self::EpisodeIdentityMismatch {
                sequence,
                expected,
                found,
            } => write!(
                f,
                "episode identity mismatch at sequence {sequence}: expected `{expected}`, found `{found}`"
            ),
            Self::DuplicateOutcome { episode_id, sequence } => write!(
                f,
                "episode `{episode_id}` at sequence {sequence} already has outcome feedback"
            ),
            Self::UnresolvedHistoryWouldBeEvicted { episode_id, sequence } => write!(
                f,
                "cannot evict unresolved episode `{episode_id}` at sequence {sequence}"
            ),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported meta-state checkpoint schema version {version}")
            }
            Self::CheckpointExceedsCapacity { records, capacity } => write!(
                f,
                "checkpoint contains {records} records but capacity is {capacity}"
            ),
            Self::NonContiguousCheckpoint { expected, found } => write!(
                f,
                "checkpoint sequence is non-contiguous: expected {expected}, found {found}"
            ),
            Self::CheckpointOutcomeMismatch {
                observation_episode_id,
                outcome_episode_id,
            } => write!(
                f,
                "checkpoint outcome `{outcome_episode_id}` does not match observation `{observation_episode_id}`"
            ),
        }
    }
}

impl std::error::Error for MetaStateError {}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), MetaStateError> {
    if value.trim().is_empty() {
        Err(MetaStateError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_unit(field: &'static str, value: f64) -> Result<(), MetaStateError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MetaStateError::InvalidUnitValue { field, value })
    }
}

fn validate_observation(observation: &MetaEpisodeObservation) -> Result<(), MetaStateError> {
    require_nonempty("episode_id", &observation.episode_id)?;
    validate_unit("context_support", observation.signals.context_support)?;
    validate_unit("strategy_support", observation.signals.strategy_support)?;
    validate_unit("selection_support", observation.signals.selection_support)
}

fn validate_feedback(feedback: &MetaOutcomeFeedback) -> Result<(), MetaStateError> {
    require_nonempty("episode_id", &feedback.episode_id)?;
    validate_unit("task_score", feedback.task_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(sequence: u64, id: &str, context_support: f64) -> MetaEpisodeObservation {
        MetaEpisodeObservation {
            episode_id: id.into(),
            sequence,
            context: ReasoningContext::GeneralReasoning,
            signals: MetaSupportSignals::try_new(context_support, 0.6, 0.7).unwrap(),
            abstained: false,
            evidence_items: 2,
            weak_assumptions_flagged: 0,
        }
    }

    fn outcome(sequence: u64, id: &str, correct: bool) -> MetaOutcomeFeedback {
        MetaOutcomeFeedback {
            episode_id: id.into(),
            sequence,
            exact_correct: correct,
            task_score: if correct { 1.0 } else { 0.0 },
            unsafe_action_observed: None,
        }
    }

    #[test]
    fn current_episode_is_committed_before_return() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        let report = state
            .commit_observation(observation(0, "episode-0", 0.8))
            .unwrap();

        assert_eq!(report.sequence, 0);
        assert_eq!(state.current().unwrap().observation.episode_id, "episode-0");
        assert_eq!(
            state.current().unwrap().observation.signals,
            report.current_signals
        );
        assert_eq!(state.next_sequence(), 1);
    }

    #[test]
    fn outcome_cannot_create_or_rewrite_episode() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        assert!(matches!(
            state.record_outcome(outcome(0, "episode-0", true)),
            Err(MetaStateError::UnknownEpisodeSequence(0))
        ));

        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        assert!(matches!(
            state.record_outcome(outcome(0, "wrong-id", true)),
            Err(MetaStateError::EpisodeIdentityMismatch { .. })
        ));
        assert!(state.current().unwrap().outcome.is_none());
    }

    #[test]
    fn episode_order_is_strictly_monotonic() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        assert!(matches!(
            state.commit_observation(observation(1, "episode-1", 0.5)),
            Err(MetaStateError::SequenceMismatch {
                expected: 0,
                found: 1
            })
        ));
        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        assert!(matches!(
            state.commit_observation(observation(0, "replay", 0.5)),
            Err(MetaStateError::SequenceMismatch {
                expected: 1,
                found: 0
            })
        ));
    }

    #[test]
    fn unresolved_episode_is_never_silently_evicted() {
        let mut state = MetaEpistemicState::new("agent-a", 1).unwrap();
        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        assert!(matches!(
            state.commit_observation(observation(1, "episode-1", 0.6)),
            Err(MetaStateError::UnresolvedHistoryWouldBeEvicted { .. })
        ));

        state
            .record_outcome(outcome(0, "episode-0", true))
            .unwrap();
        state
            .commit_observation(observation(1, "episode-1", 0.6))
            .unwrap();
        assert_eq!(state.records().len(), 1);
        assert_eq!(state.current().unwrap().observation.episode_id, "episode-1");
    }

    #[test]
    fn revision_is_between_frozen_pre_outcome_signals() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        state
            .commit_observation(observation(0, "episode-0", 0.4))
            .unwrap();
        state
            .record_outcome(outcome(0, "episode-0", false))
            .unwrap();
        let report = state
            .commit_observation(observation(1, "episode-1", 0.9))
            .unwrap();

        let revision = report.revision.unwrap();
        assert!((revision.context_support_delta - 0.5).abs() < 1.0e-12);
        assert_eq!(revision.strategy_support_delta, 0.0);
        assert_eq!(revision.selection_support_delta, 0.0);
    }

    #[test]
    fn subjects_do_not_share_longitudinal_state() {
        let mut a = MetaEpistemicState::new("agent-a", 4).unwrap();
        let mut b = MetaEpistemicState::new("agent-b", 4).unwrap();
        a.commit_observation(observation(0, "a-0", 0.8)).unwrap();
        b.commit_observation(observation(0, "b-0", 0.2)).unwrap();

        assert_eq!(a.current().unwrap().observation.episode_id, "a-0");
        assert_eq!(b.current().unwrap().observation.episode_id, "b-0");
        assert_ne!(
            a.current().unwrap().observation.signals.context_support,
            b.current().unwrap().observation.signals.context_support
        );
    }

    #[test]
    fn duplicate_outcome_is_rejected() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        state
            .record_outcome(outcome(0, "episode-0", true))
            .unwrap();
        assert!(matches!(
            state.record_outcome(outcome(0, "episode-0", true)),
            Err(MetaStateError::DuplicateOutcome { .. })
        ));
    }

    #[test]
    fn tampered_checkpoint_fails_closed() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        state
            .record_outcome(outcome(0, "episode-0", true))
            .unwrap();
        let mut checkpoint = state.checkpoint();
        checkpoint.next_sequence = 9;
        assert!(matches!(
            MetaEpistemicState::restore(checkpoint),
            Err(MetaStateError::SequenceMismatch { .. })
        ));
    }

    #[test]
    fn restored_checkpoint_preserves_state_and_authority() {
        let mut state = MetaEpistemicState::new("agent-a", 4).unwrap();
        state
            .commit_observation(observation(0, "episode-0", 0.5))
            .unwrap();
        state
            .record_outcome(outcome(0, "episode-0", true))
            .unwrap();
        let restored = MetaEpistemicState::restore(state.checkpoint()).unwrap();

        assert_eq!(restored.subject_id(), "agent-a");
        assert_eq!(restored.next_sequence(), 1);
        assert_eq!(restored.records().len(), 1);
        assert_eq!(restored.authority(), MetaStateAuthority::MeasurementOnly);
        assert!(!restored.authority().may_control_plasticity());
    }
}
