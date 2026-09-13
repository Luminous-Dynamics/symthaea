// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prediction-only evaluation capabilities for frozen FEP snapshots.
//!
//! [`FepPredictionSession`] intentionally supports learning because Development
//! and other training workflows need `learn_from_actual`. Confirmatory
//! evaluation needs a narrower capability: start from one immutable
//! [`FepEvaluationSnapshot`], observe one public input, predict one prescribed
//! action, and then lose the trial capability.
//!
//! [`FepHeldOutSubject`] is the runner-facing ownership boundary. It consumes a
//! frozen snapshot and exposes only fresh one-shot [`FepEvaluationTrial`] values;
//! callers cannot recover the snapshot, agent, or trainable session through the
//! held-out subject API.

use std::fmt;

use crate::prediction_commitment::FrozenPredictionCommitment;
use crate::prediction_session::{
    FepEvaluationSnapshot, FepPredictionSession, FepPredictionSessionError,
};
use crate::types::ActionOutcome;

/// API revision for the prediction-only frozen-snapshot evaluation boundary.
pub const FEP_EVALUATION_TRIAL_REVISION: &str = "symthaea-fep-evaluation-trial-v1";
/// API revision for the sealed held-out subject ownership boundary.
pub const FEP_HELDOUT_SUBJECT_REVISION: &str = "symthaea-fep-heldout-subject-v1";

/// Fresh one-shot prediction capability derived from a frozen FEP snapshot.
///
/// The value is deliberately not `Clone`. [`predict_once`](Self::predict_once)
/// consumes it, so one trial cannot carry transient inference state into a
/// second evaluated world.
pub struct FepEvaluationTrial {
    session: FepPredictionSession,
    snapshot_replay_digest: u64,
    snapshot_commitment: FrozenPredictionCommitment,
}

/// Deliberately redacted: formatting an evaluation capability must not dump the
/// private trainable session or learned model state into logs.
impl fmt::Debug for FepEvaluationTrial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FepEvaluationTrial")
            .field("revision", &FEP_EVALUATION_TRIAL_REVISION)
            .field("snapshot_replay_digest", &self.snapshot_replay_digest)
            .field("snapshot_commitment", &self.snapshot_commitment.to_hex())
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
            snapshot_commitment: snapshot.commitment(),
        }
    }

    /// Deterministic compatibility/replay identity of the source snapshot.
    ///
    /// This remains the existing non-cryptographic snapshot replay digest.
    pub fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot_replay_digest
    }

    /// Collision-resistant identity of the exact standardized learned subject.
    pub fn snapshot_commitment(&self) -> FrozenPredictionCommitment {
        self.snapshot_commitment
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

/// Runner-facing immutable held-out subject.
///
/// Construction consumes the frozen snapshot. This type is deliberately not
/// `Clone` and provides no method that returns the snapshot, agent, or trainable
/// [`FepPredictionSession`]. The only prediction authority it grants is creation
/// of independent one-shot [`FepEvaluationTrial`] capabilities.
///
/// This is defense in depth rather than global API revocation: code that still
/// possesses some other clone of a snapshot could create a trainable session
/// through the legacy snapshot API. EUREKA's held-out runner must therefore own
/// only this type, and its static reachability audit must reject direct imports
/// of `FepEvaluationSnapshot` and `FepPredictionSession`.
pub struct FepHeldOutSubject {
    snapshot: FepEvaluationSnapshot,
}

impl fmt::Debug for FepHeldOutSubject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FepHeldOutSubject")
            .field("revision", &FEP_HELDOUT_SUBJECT_REVISION)
            .field("state_dim", &self.state_dim())
            .field("observation_dim", &self.observation_dim())
            .field("action_count", &self.action_count())
            .field("snapshot_replay_digest", &self.snapshot_replay_digest())
            .field("snapshot_commitment", &self.commitment().to_hex())
            .field("snapshot", &"<redacted>")
            .finish()
    }
}

impl FepHeldOutSubject {
    /// Consume one standardized frozen snapshot into held-out-only authority.
    pub fn seal(snapshot: FepEvaluationSnapshot) -> Self {
        Self { snapshot }
    }

    pub fn state_dim(&self) -> usize {
        self.snapshot.state_dim()
    }

    pub fn observation_dim(&self) -> usize {
        self.snapshot.observation_dim()
    }

    pub fn action_count(&self) -> usize {
        self.snapshot.action_count()
    }

    /// Existing deterministic replay ID of the learned snapshot.
    pub fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot.replay_digest()
    }

    /// Collision-resistant scientific identity of the learned predictor.
    pub fn commitment(&self) -> FrozenPredictionCommitment {
        self.snapshot.commitment()
    }

    /// Mint one fresh prediction-only trial from the sealed learned subject.
    pub fn trial(&self) -> FepEvaluationTrial {
        FepEvaluationTrial::from_snapshot(&self.snapshot)
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
    fn sealed_subject_mints_independent_trials_without_exposing_training_surface() {
        let frozen = snapshot(4, 4);
        let commitment = frozen.commitment();
        let subject = FepHeldOutSubject::seal(frozen);
        assert_eq!(subject.observation_dim(), 4);
        assert_eq!(subject.action_count(), 4);
        assert_eq!(subject.commitment(), commitment);
        let observation = [4.0, 3.0, 2.0, 1.0];
        let first_trial = subject.trial();
        assert_eq!(first_trial.snapshot_commitment(), commitment);
        let a = first_trial
            .predict_once(&observation, 1.0, "heldout", 3)
            .unwrap();
        let b = subject
            .trial()
            .predict_once(&observation, 1.0, "heldout", 3)
            .unwrap();
        assert_eq!(a.action, b.action);
        assert_eq!(a.predicted_next_state.mean, b.predicted_next_state.mean);
        assert_eq!(a.expected_observation, b.expected_observation);
    }

    #[test]
    fn debug_output_redacts_trainable_state() {
        let frozen = snapshot(4, 4);
        let trial = FepEvaluationTrial::from_snapshot(&frozen);
        let trial_debug = format!("{trial:?}");
        assert!(trial_debug.contains(FEP_EVALUATION_TRIAL_REVISION));
        assert!(trial_debug.contains(&frozen.commitment().to_hex()));
        assert!(!trial_debug.contains("ActiveInferenceAgent"));
        assert!(!trial_debug.contains("transition_matrices"));

        let subject = FepHeldOutSubject::seal(frozen);
        let subject_debug = format!("{subject:?}");
        assert!(subject_debug.contains(FEP_HELDOUT_SUBJECT_REVISION));
        assert!(subject_debug.contains(&subject.commitment().to_hex()));
        assert!(subject_debug.contains("<redacted>"));
        assert!(!subject_debug.contains("likelihood_matrix"));
        assert!(!subject_debug.contains("transition_matrices"));
    }

    #[test]
    fn replay_and_cryptographic_subject_identities_are_retained() {
        let frozen = snapshot(4, 4);
        let replay_digest = frozen.replay_digest();
        let commitment = frozen.commitment();
        let subject = FepHeldOutSubject::seal(frozen);
        assert_eq!(subject.snapshot_replay_digest(), replay_digest);
        assert_eq!(subject.commitment(), commitment);
        let trial = subject.trial();
        assert_eq!(trial.snapshot_replay_digest(), replay_digest);
        assert_eq!(trial.snapshot_commitment(), commitment);
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
