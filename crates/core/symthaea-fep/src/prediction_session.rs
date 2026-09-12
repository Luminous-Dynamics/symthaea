// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Detached prescribed-action prediction sessions for the production FEP agent.
//!
//! This module exposes the prediction/learning semantics already implemented by
//! [`ActiveInferenceAgent`] without exposing motor execution or a mutable handle
//! to a live cognitive-loop service.  It is intentionally domain-agnostic: the
//! caller owns the meaning of observation channels and action indices.

use super::agent::ActiveInferenceAgent;
use super::types::{ActionOutcome, Observation, PerceptionResult};

/// Replay-identity revision for frozen prescribed-action prediction models.
pub const FEP_PREDICTION_SNAPSHOT_REVISION: &str = "symthaea-fep-prediction-snapshot-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FepPredictionSessionError {
    ObservationDimensionMismatch { expected: usize, actual: usize },
    ActionOutOfRange { action: usize, action_count: usize },
    NonFiniteObservation { index: usize },
    InvalidPrecision,
    EmptyModality,
    ObservationRequired,
    PredictionAlreadyPending,
    NoPredictionPending,
    OutcomeActionMismatch { expected: usize, actual: usize },
}

/// A detached copy of a production [`ActiveInferenceAgent`] used for
/// prescribed-action prediction and optional learning.
///
/// The session has no motor system and therefore grants no physical or external
/// action authority.  Creating or training a session cannot mutate the source
/// agent from which it was cloned.
#[derive(Debug, Clone)]
pub struct FepPredictionSession {
    agent: ActiveInferenceAgent,
    has_observation: bool,
    pending_action: Option<usize>,
}

impl FepPredictionSession {
    /// Clone the exact current FEP agent state into a detached sandbox.
    pub fn from_agent(agent: &ActiveInferenceAgent) -> Self {
        Self {
            agent: agent.clone(),
            has_observation: false,
            pending_action: None,
        }
    }

    pub fn state_dim(&self) -> usize {
        self.agent.config.state_dim
    }

    pub fn observation_dim(&self) -> usize {
        self.agent.config.obs_dim
    }

    pub fn action_count(&self) -> usize {
        self.agent.config.num_actions
    }

    /// Incorporate one exact observation through the production perception
    /// path.  Dimensions are strict: this API never pads or truncates data.
    pub fn observe(
        &mut self,
        values: &[f64],
        precision: f64,
        modality: &str,
    ) -> Result<PerceptionResult, FepPredictionSessionError> {
        if self.pending_action.is_some() {
            return Err(FepPredictionSessionError::PredictionAlreadyPending);
        }
        self.validate_observation(values, precision, modality)?;
        let observation = Observation::new(values.to_vec(), precision, modality);
        let result = self.agent.perceive(&observation);
        self.has_observation = true;
        Ok(result)
    }

    /// Predict the consequence of one caller-prescribed action using the exact
    /// production `ActiveInferenceAgent::act` path.
    ///
    /// Unlike `GenerativeModel::predict_next_state`, this boundary rejects an
    /// out-of-range action instead of allowing the lower-level model to clamp
    /// it to the final action index.
    pub fn predict(
        &mut self,
        action: usize,
    ) -> Result<ActionOutcome, FepPredictionSessionError> {
        if !self.has_observation {
            return Err(FepPredictionSessionError::ObservationRequired);
        }
        if self.pending_action.is_some() {
            return Err(FepPredictionSessionError::PredictionAlreadyPending);
        }
        self.validate_action(action)?;
        let outcome = self.agent.act(action);
        self.pending_action = Some(action);
        Ok(outcome)
    }

    /// Feed the actual observation following the pending prescribed action back
    /// through the production learning path.
    pub fn learn_from_actual(
        &mut self,
        action: usize,
        actual_values: &[f64],
        precision: f64,
        modality: &str,
    ) -> Result<(), FepPredictionSessionError> {
        let expected = self
            .pending_action
            .ok_or(FepPredictionSessionError::NoPredictionPending)?;
        if action != expected {
            return Err(FepPredictionSessionError::OutcomeActionMismatch {
                expected,
                actual: action,
            });
        }
        self.validate_action(action)?;
        self.validate_observation(actual_values, precision, modality)?;
        let observation = Observation::new(actual_values.to_vec(), precision, modality);
        self.agent.learn_from_outcome(action, &observation);
        self.pending_action = None;
        self.has_observation = true;
        Ok(())
    }

    /// End an episode while preserving the learned generative model.
    ///
    /// `ActiveInferenceAgent::reset()` resets transient belief, precision,
    /// eligibility and action-selection state but deliberately leaves the
    /// learned `GenerativeModel` in place.  This makes episode boundaries
    /// suitable for independent world trials.
    pub fn reset_transient_state_preserving_model(&mut self) {
        self.agent.reset();
        self.has_observation = false;
        self.pending_action = None;
    }

    /// Freeze the learned prescribed-action predictor for held-out evaluation.
    ///
    /// The returned snapshot has standardized transient state.  Each held-out
    /// world should create a fresh session from the snapshot so held-out
    /// outcomes cannot train later held-out trials.
    pub fn freeze_for_evaluation(
        &self,
    ) -> Result<FepEvaluationSnapshot, FepPredictionSessionError> {
        if self.pending_action.is_some() {
            return Err(FepPredictionSessionError::PredictionAlreadyPending);
        }
        let mut agent = self.agent.clone();
        agent.reset();
        let replay_digest = prediction_replay_digest(&agent);
        Ok(FepEvaluationSnapshot {
            agent,
            replay_digest,
        })
    }

    fn validate_action(&self, action: usize) -> Result<(), FepPredictionSessionError> {
        if action >= self.action_count() {
            return Err(FepPredictionSessionError::ActionOutOfRange {
                action,
                action_count: self.action_count(),
            });
        }
        Ok(())
    }

    fn validate_observation(
        &self,
        values: &[f64],
        precision: f64,
        modality: &str,
    ) -> Result<(), FepPredictionSessionError> {
        if values.len() != self.observation_dim() {
            return Err(FepPredictionSessionError::ObservationDimensionMismatch {
                expected: self.observation_dim(),
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(FepPredictionSessionError::NonFiniteObservation { index });
        }
        if !precision.is_finite() || precision <= 0.0 {
            return Err(FepPredictionSessionError::InvalidPrecision);
        }
        if modality.trim().is_empty() {
            return Err(FepPredictionSessionError::EmptyModality);
        }
        Ok(())
    }
}

/// Frozen, standardized learned-model state for independent held-out trials.
#[derive(Debug, Clone)]
pub struct FepEvaluationSnapshot {
    agent: ActiveInferenceAgent,
    replay_digest: u64,
}

impl FepEvaluationSnapshot {
    pub fn state_dim(&self) -> usize {
        self.agent.config.state_dim
    }

    pub fn observation_dim(&self) -> usize {
        self.agent.config.obs_dim
    }

    pub fn action_count(&self) -> usize {
        self.agent.config.num_actions
    }

    /// Deterministic replay identity for the prediction-relevant model/config
    /// state.  This is a fixture/replay digest, not cryptographic attestation.
    pub fn replay_digest(&self) -> u64 {
        self.replay_digest
    }

    /// Start one independent held-out trial from the exact frozen model.
    pub fn session(&self) -> FepPredictionSession {
        FepPredictionSession::from_agent(&self.agent)
    }
}

fn prediction_replay_digest(agent: &ActiveInferenceAgent) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(FEP_PREDICTION_SNAPSHOT_REVISION.as_bytes());
    bytes.push(0);

    push_usize(&mut bytes, agent.config.state_dim);
    push_usize(&mut bytes, agent.config.obs_dim);
    push_usize(&mut bytes, agent.config.num_actions);
    push_usize(&mut bytes, agent.config.inference_iterations);
    push_f64(&mut bytes, agent.config.belief_learning_rate);
    push_usize(&mut bytes, agent.config.planning_horizon);
    push_f64(&mut bytes, agent.config.action_temperature);
    bytes.push(u8::from(agent.config.enable_model_learning));
    bytes.push(u8::from(agent.config.enable_td_learning));

    let model = &agent.model;
    push_usize(&mut bytes, model.state_dim);
    push_usize(&mut bytes, model.obs_dim);
    push_usize(&mut bytes, model.num_actions);
    push_f64(&mut bytes, model.observation_precision);
    push_f64(&mut bytes, model.transition_precision);
    push_f64(&mut bytes, model.learning_rate);

    push_matrix(&mut bytes, &model.likelihood_matrix);
    push_tensor3(&mut bytes, &model.transition_matrices);
    push_matrix(&mut bytes, &model.transition_bias);
    push_f64_slice(&mut bytes, &model.prior_mean);
    push_f64_slice(&mut bytes, &model.prior_precision);

    // The evaluation snapshot has standardized transient state, but include it
    // in the replay identity so a future reset-semantics change cannot silently
    // reuse the same evidence identity.
    push_f64_slice(&mut bytes, &agent.belief.mean);
    push_f64_slice(&mut bytes, &agent.belief.precision);
    push_f64_slice(&mut bytes, &agent.belief.mode_probs);
    push_usize(&mut bytes, agent.belief.current_mode);

    fnv1a64(&bytes)
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_f64_slice(bytes: &mut Vec<u8>, values: &[f64]) {
    push_usize(bytes, values.len());
    for value in values {
        push_f64(bytes, *value);
    }
}

fn push_matrix(bytes: &mut Vec<u8>, matrix: &[Vec<f64>]) {
    push_usize(bytes, matrix.len());
    for row in matrix {
        push_f64_slice(bytes, row);
    }
}

fn push_tensor3(bytes: &mut Vec<u8>, tensor: &[Vec<Vec<f64>>]) {
    push_usize(bytes, tensor.len());
    for matrix in tensor {
        push_matrix(bytes, matrix);
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActiveInferenceAgentConfig, Observation};

    fn agent() -> ActiveInferenceAgent {
        ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 4,
            obs_dim: 3,
            num_actions: 3,
            enable_td_learning: true,
            ..Default::default()
        })
    }

    #[test]
    fn session_prediction_matches_direct_production_agent_path() {
        let source = agent();
        let values = [0.2, 0.4, 0.8];

        let mut direct = source.clone();
        let _ = direct.perceive(&Observation::new(values.to_vec(), 1.0, "test"));
        let direct_outcome = direct.act(1);

        let mut session = FepPredictionSession::from_agent(&source);
        let _ = session.observe(&values, 1.0, "test").unwrap();
        let session_outcome = session.predict(1).unwrap();

        assert_eq!(session_outcome.action, direct_outcome.action);
        assert_eq!(
            session_outcome.predicted_next_state.mean,
            direct_outcome.predicted_next_state.mean
        );
        assert_eq!(
            session_outcome.expected_observation,
            direct_outcome.expected_observation
        );
    }

    #[test]
    fn observation_dimension_mismatch_fails_closed() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        assert_eq!(
            session.observe(&[0.1, 0.2], 1.0, "test"),
            Err(FepPredictionSessionError::ObservationDimensionMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn out_of_range_action_fails_instead_of_clamping() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        session.observe(&[0.1, 0.2, 0.3], 1.0, "test").unwrap();
        assert_eq!(
            session.predict(3),
            Err(FepPredictionSessionError::ActionOutOfRange {
                action: 3,
                action_count: 3,
            })
        );
    }

    #[test]
    fn prediction_requires_observation_and_single_pending_outcome() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        assert_eq!(
            session.predict(0),
            Err(FepPredictionSessionError::ObservationRequired)
        );
        session.observe(&[0.1, 0.2, 0.3], 1.0, "test").unwrap();
        session.predict(0).unwrap();
        assert_eq!(
            session.predict(1),
            Err(FepPredictionSessionError::PredictionAlreadyPending)
        );
    }

    #[test]
    fn learning_is_detached_from_source_agent() {
        let source = agent();
        let source_transitions = source.model.transition_matrices.clone();
        let source_likelihood = source.model.likelihood_matrix.clone();

        let mut session = FepPredictionSession::from_agent(&source);
        session.observe(&[0.1, 0.2, 0.3], 1.0, "development").unwrap();
        session.predict(1).unwrap();
        session
            .learn_from_actual(1, &[0.9, 0.8, 0.7], 1.0, "development")
            .unwrap();

        assert_eq!(source.model.transition_matrices, source_transitions);
        assert_eq!(source.model.likelihood_matrix, source_likelihood);
    }

    #[test]
    fn wrong_action_cannot_be_attached_to_pending_prediction() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        session.observe(&[0.1, 0.2, 0.3], 1.0, "test").unwrap();
        session.predict(1).unwrap();
        assert_eq!(
            session.learn_from_actual(2, &[0.2, 0.3, 0.4], 1.0, "test"),
            Err(FepPredictionSessionError::OutcomeActionMismatch {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn frozen_snapshot_standardizes_transient_state_and_replays() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        session.observe(&[0.7, 0.2, 0.4], 1.0, "development").unwrap();
        let snapshot = session.freeze_for_evaluation().unwrap();

        let mut a = snapshot.session();
        let mut b = snapshot.session();
        a.observe(&[0.3, 0.5, 0.7], 1.0, "heldout").unwrap();
        b.observe(&[0.3, 0.5, 0.7], 1.0, "heldout").unwrap();
        let a_outcome = a.predict(2).unwrap();
        let b_outcome = b.predict(2).unwrap();
        assert_eq!(a_outcome.expected_observation, b_outcome.expected_observation);
        assert_eq!(snapshot.replay_digest(), snapshot.clone().replay_digest());
    }

    #[test]
    fn model_change_changes_frozen_replay_identity() {
        let a = agent();
        let mut b = a.clone();
        b.model.transition_matrices[0][0][0] += 0.01;

        let snapshot_a = FepPredictionSession::from_agent(&a)
            .freeze_for_evaluation()
            .unwrap();
        let snapshot_b = FepPredictionSession::from_agent(&b)
            .freeze_for_evaluation()
            .unwrap();
        assert_ne!(snapshot_a.replay_digest(), snapshot_b.replay_digest());
    }

    #[test]
    fn invalid_numeric_observation_is_rejected() {
        let source = agent();
        let mut session = FepPredictionSession::from_agent(&source);
        assert_eq!(
            session.observe(&[0.1, f64::NAN, 0.3], 1.0, "test"),
            Err(FepPredictionSessionError::NonFiniteObservation { index: 1 })
        );
        assert_eq!(
            session.observe(&[0.1, 0.2, 0.3], 0.0, "test"),
            Err(FepPredictionSessionError::InvalidPrecision)
        );
    }
}