// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! One-shot prediction-only evaluation capability for frozen FEP snapshots.
//!
//! [`FepPredictionSession`] intentionally supports learning because Development
//! and other training workflows need `learn_from_actual`. Confirmatory
//! evaluation needs a narrower capability: start from one immutable
//! [`FepEvaluationSnapshot`], observe one public input, predict one prescribed
//! action, and then lose the capability.
//!
//! This type contains a trainable session internally only as an implementation
//! detail. It exposes no session/agent accessor and no learning method.

use std::fmt;

use crate::prediction_session::{
    FepEvaluationSnapshot, FepPredictionSession, FepPredictionSessionError,
};
use crate::types::ActionOutcome;

/// API revision for the prediction-only frozen-snapshot evaluation boundary.
pub const FEP_EVALUATION_TRIAL_REVISION: &str = "symthaea-fep-evaluation-trial-v1";

/// Fresh one-shot prediction capability derived from a frozen FEP snapshot.
///
/// The value is deliberately not `Clone`. [`predict_once`](Self::predict_once)
/// consumes it, so one trial cannot carry transient inference state into a
/// second evaluated world.
pub struct FepEvaluationTrial {
    session: FepPredictionSession,
    snapshot_replay_digest: u64,
}

/// Deliberately redacted: formatting an evaluation capability must not dump the
/// private trainable session or learned model state into logs.
impl fmt::Debug for FepEvaluationTrial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FepEvaluationTrial")
            .field("revision", &FEP_EVALUATION_TRIAL_REVISION)
            .field("snapshot_replay_digest", &self.snapshot_replay_digest)
            .finish_non_exhaustive()
    }
}

impl FepEvaluationTrial {
    /// Start one fresh evaluation trial from the exact standardized frozen
    /// model state. The snapshot itself remains immutable and reusable for
    /// creating independent trials.
    pub fn from_snapshot(snapshot: &FepEvaluationSnapshot) -> Self {
        Self {
            session: snapshot.session(),
            snapshot_replay_digest: snapshot.replay_digest(),
        }
    }

    /// Deterministic compatibility/replay identity of the source snapshot.
    ///
    /// This remains the existing non-cryptographic snapshot replay digest. It
    /// must not be promoted to cryptographic subject attestation; stronger
    /// evidence custody is a separate concern.
    pub fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot_replay_digest
    }

    /// Observe one exact input and produce one prescribed-action prediction.
    ///
    /// Consuming `self` enforces one-shot trial semantics. There is no outcome
    /// feedback or learning API on this capability.
    pub fn predict_once(
        mut self,
        observation: &[f64],
        precision: f64,
        modality: &str,
        action: usize,
    ) -> Result<ActionOutcome, FepPredictionSessionError> {
        self.session.observe(observation, precision, modality)?;
        self.session.predict(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

    fn snapshot(obs_dim: usize, actions: usize) -> FepEvaluationSnapshot {
        let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim,
            num_actions: actions,
            enable_model_learning: true,
            enable_td_learning: true,
            ..Default::default()
        });
        FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap()
    }

    #[test]
    fn one_shot_trial_matches_existing_frozen_session_prediction() {
        let frozen = snapshot(4, 4);
        let observation = [2.0, 5.0, 9.0, 0.0];

        let mut existing = frozen.session();
        existing.observe(&observation, 1.0, "evaluation").unwrap();
        let expected = existing.predict(2).unwrap();

        let actual = FepEvaluationTrial::from_snapshot(&frozen)
            .predict_once(&observation, 1.0, "evaluation", 2)
            .unwrap();

        assert_eq!(actual.action, expected.action);
        assert_eq!(actual.predicted_next_state.mean, expected.predicted_next_state.mean);
        assert_eq!(
            actual.predicted_next_state.precision,
            expected.predicted_next_state.precision
        );
        assert_eq!(actual.expected_observation, expected.expected_observation);
    }

    #[test]
    fn independent_trials_replay_from_the_same_frozen_state() {
        let frozen = snapshot(4, 4);
        let observation = [1.0, 3.0, 7.0, 2.0];
        let a = FepEvaluationTrial::from_snapshot(&frozen)
            .predict_once(&observation, 1.0, "evaluation", 1)
            .unwrap();
        let b = FepEvaluationTrial::from_snapshot(&frozen)
            .predict_once(&observation, 1.0, "evaluation", 1)
            .unwrap();
        assert_eq!(a.action, b.action);
        assert_eq!(a.predicted_next_state.mean, b.predicted_next_state.mean);
        assert_eq!(a.predicted_next_state.precision, b.predicted_next_state.precision);
        assert_eq!(a.expected_observation, b.expected_observation);
    }

    #[test]
    fn source_snapshot_identity_is_retained_without_becoming_attestation() {
        let frozen = snapshot(4, 4);
        let trial = FepEvaluationTrial::from_snapshot(&frozen);
        assert_eq!(trial.snapshot_replay_digest(), frozen.replay_digest());
    }

    #[test]
    fn debug_output_does_not_expose_private_session_state() {
        let frozen = snapshot(4, 4);
        let rendered = format!("{:?}", FepEvaluationTrial::from_snapshot(&frozen));
        assert!(rendered.contains(FEP_EVALUATION_TRIAL_REVISION));
        assert!(rendered.contains(&frozen.replay_digest().to_string()));
        assert!(!rendered.contains("session"));
        assert!(!rendered.contains("ActiveInferenceAgent"));
        assert!(!rendered.contains("transition_matrices"));
    }

    #[test]
    fn observation_dimension_mismatch_fails_closed() {
        let frozen = snapshot(4, 4);
        assert!(matches!(
            FepEvaluationTrial::from_snapshot(&frozen)
                .predict_once(&[1.0, 2.0, 3.0], 1.0, "evaluation", 0),
            Err(FepPredictionSessionError::ObservationDimensionMismatch {
                expected: 4,
                actual: 3,
            })
        ));
    }

    #[test]
    fn out_of_range_action_fails_instead_of_clamping() {
        let frozen = snapshot(4, 4);
        assert!(matches!(
            FepEvaluationTrial::from_snapshot(&frozen)
                .predict_once(&[1.0, 2.0, 3.0, 4.0], 1.0, "evaluation", 4),
            Err(FepPredictionSessionError::ActionOutOfRange {
                action: 4,
                action_count: 4,
            })
        ));
    }

    #[test]
    fn invalid_precision_and_empty_modality_fail_closed() {
        let frozen = snapshot(4, 4);
        assert!(matches!(
            FepEvaluationTrial::from_snapshot(&frozen)
                .predict_once(&[1.0, 2.0, 3.0, 4.0], 0.0, "evaluation", 0),
            Err(FepPredictionSessionError::InvalidPrecision)
        ));
        assert!(matches!(
            FepEvaluationTrial::from_snapshot(&frozen)
                .predict_once(&[1.0, 2.0, 3.0, 4.0], 1.0, "", 0),
            Err(FepPredictionSessionError::EmptyModality)
        ));
    }
}
