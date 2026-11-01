//! Action-conditioned residual visual-state forecasting.
//!
//! This is a deliberately small, observation-supervised control head for
//! VIS-005. It conditions an autonomous CfC horizon prediction on an explicit
//! action embedding, and updates only when a later real visual observation is
//! supplied to score_and_learn. It does not generate pixels, prove causal
//! coupling, or authorize physical actions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedForecastReceipt {
    pub forecast_id: u64,
    pub history_sha256: String,
    pub source_frame_sequence: u64,
    pub action_manifest_sha256: String,
    pub action_descriptor_sha256: String,
    pub action_id: u64,
    pub horizon_seconds: f32,
    /// Number of supervised updates completed before this forecast was issued.
    pub model_version: u64,
    /// Predicted future visual-manifold state; still a prediction, never an observation.
    pub predicted_future_hv: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingActionConditionedForecastState {
    pub receipt: ActionConditionedForecastReceipt,
    pub autonomous_base_prediction: Vec<f32>,
    pub action_embedding: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedPredictorState {
    pub schema_version: u32,
    pub dim: usize,
    pub action_weights: Vec<f32>,
    pub learning_rate: f32,
    pub max_pending_forecasts: usize,
    pub max_lateness_factor: f32,
    pub next_forecast_id: u64,
    pub training_steps: u64,
    pub expired_forecasts: u64,
    pub pending: Vec<PendingActionConditionedForecastState>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionConditionedForecastScore {
    pub forecast_id: u64,
    pub action_id: u64,
    pub source_frame_sequence: u64,
    pub observed_frame_sequence: u64,
    pub observation_delay_seconds: f32,
    pub lateness_seconds: f32,
    pub prediction_error: f32,
    pub autonomous_baseline_error: f32,
    pub improvement_over_autonomous_baseline: f32,
    pub mean_squared_error: f32,
    pub model_version_before: u64,
    pub model_version_after: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionConditionedPredictorError {
    InvalidConfiguration,
    InvalidDigest,
    InvalidHorizon,
    InvalidVectorDimension,
    NonFiniteValue,
    PendingQueueFull,
    ForecastIdExhausted,
    UnknownForecast,
    ObservationNotLater,
    ObservationBeforeTarget,
    ForecastTooLate,
    InvalidCheckpointState,
    NonFiniteUpdate,
}

/// A diagonal, action-conditioned residual head over an autonomous future HV.
///
/// prediction = autonomous_base + action_embedding ⊙ action_weights.
/// Weights start at zero, so before supervision this head exactly reproduces
/// the autonomous baseline. Its limited diagonal form is intentional: it is a
/// transparent action-sensitivity control before introducing a larger learned
/// action/state interaction model.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionConditionedVisualResidualPredictor {
    dim: usize,
    action_weights: Vec<f32>,
    learning_rate: f32,
    max_pending_forecasts: usize,
    max_lateness_factor: f32,
    next_forecast_id: u64,
    training_steps: u64,
    expired_forecasts: u64,
    pending: BTreeMap<u64, PendingActionConditionedForecastState>,
}

impl ActionConditionedVisualResidualPredictor {
    pub fn new(
        dim: usize,
        learning_rate: f32,
        max_pending_forecasts: usize,
        max_lateness_factor: f32,
    ) -> Result<Self, ActionConditionedPredictorError> {
        if dim == 0
            || !learning_rate.is_finite()
            || !(0.0..=1.0).contains(&learning_rate)
            || max_pending_forecasts == 0
            || !max_lateness_factor.is_finite()
            || max_lateness_factor < 0.0
        {
            return Err(ActionConditionedPredictorError::InvalidConfiguration);
        }
        Ok(Self {
            dim,
            action_weights: vec![0.0; dim],
            learning_rate,
            max_pending_forecasts,
            max_lateness_factor,
            next_forecast_id: 1,
            training_steps: 0,
            expired_forecasts: 0,
            pending: BTreeMap::new(),
        })
    }

    /// Issue a counterfactual visual-state forecast without mutating the
    /// autonomous vision manifold or observation memory.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_forecast(
        &mut self,
        history_sha256: &str,
        source_frame_sequence: u64,
        action_manifest_sha256: &str,
        action_descriptor_sha256: &str,
        action_id: u64,
        horizon_seconds: f32,
        autonomous_base_prediction: &[f32],
        action_embedding: &[f32],
    ) -> Result<ActionConditionedForecastReceipt, ActionConditionedPredictorError> {
        if !is_lower_hex_sha256(history_sha256)
            || !is_lower_hex_sha256(action_manifest_sha256)
            || !is_lower_hex_sha256(action_descriptor_sha256)
        {
            return Err(ActionConditionedPredictorError::InvalidDigest);
        }
        if !horizon_seconds.is_finite() || horizon_seconds <= 0.0 {
            return Err(ActionConditionedPredictorError::InvalidHorizon);
        }
        if autonomous_base_prediction.len() != self.dim || action_embedding.len() != self.dim {
            return Err(ActionConditionedPredictorError::InvalidVectorDimension);
        }
        if autonomous_base_prediction.iter().chain(action_embedding).any(|v| !v.is_finite()) {
            return Err(ActionConditionedPredictorError::NonFiniteValue);
        }
        if self.pending.len() >= self.max_pending_forecasts {
            return Err(ActionConditionedPredictorError::PendingQueueFull);
        }
        let next_id = self.next_forecast_id.checked_add(1)
            .ok_or(ActionConditionedPredictorError::ForecastIdExhausted)?;
        let forecast_id = self.next_forecast_id;
        let predicted_future_hv: Vec<f32> = autonomous_base_prediction
            .iter()
            .zip(action_embedding)
            .zip(&self.action_weights)
            .map(|((&base, &action), &weight)| base + action * weight)
            .collect();
        if predicted_future_hv.iter().any(|v| !v.is_finite()) {
            return Err(ActionConditionedPredictorError::NonFiniteValue);
        }
        let receipt = ActionConditionedForecastReceipt {
            forecast_id,
            history_sha256: history_sha256.to_owned(),
            source_frame_sequence,
            action_manifest_sha256: action_manifest_sha256.to_owned(),
            action_descriptor_sha256: action_descriptor_sha256.to_owned(),
            action_id,
            horizon_seconds,
            model_version: self.training_steps,
            predicted_future_hv,
        };
        let pending = PendingActionConditionedForecastState {
            receipt: receipt.clone(),
            autonomous_base_prediction: autonomous_base_prediction.to_vec(),
            action_embedding: action_embedding.to_vec(),
        };
        self.pending.insert(forecast_id, pending);
        self.next_forecast_id = next_id;
        Ok(receipt)
    }

    /// Score a previously sealed forecast against a later real encoded visual
    /// observation, then apply one online update. Each forecast ID is consumed
    /// at most once. An over-late target expires the forecast without training.
    pub fn score_and_learn(
        &mut self,
        forecast_id: u64,
        actual_future_hv: &[f32],
        observed_frame_sequence: u64,
        observation_delay_seconds: f32,
    ) -> Result<ActionConditionedForecastScore, ActionConditionedPredictorError> {
        let pending = self.pending.get(&forecast_id)
            .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
        if actual_future_hv.len() != self.dim {
            return Err(ActionConditionedPredictorError::InvalidVectorDimension);
        }
        if actual_future_hv.iter().any(|v| !v.is_finite()) {
            return Err(ActionConditionedPredictorError::NonFiniteValue);
        }
        if observed_frame_sequence <= pending.receipt.source_frame_sequence {
            return Err(ActionConditionedPredictorError::ObservationNotLater);
        }
        if !observation_delay_seconds.is_finite() || observation_delay_seconds < 0.0 {
            return Err(ActionConditionedPredictorError::NonFiniteValue);
        }
        if observation_delay_seconds < pending.receipt.horizon_seconds {
            return Err(ActionConditionedPredictorError::ObservationBeforeTarget);
        }
        let lateness = observation_delay_seconds - pending.receipt.horizon_seconds;
        let maximum_lateness = pending.receipt.horizon_seconds * self.max_lateness_factor;
        if lateness > maximum_lateness {
            self.pending.remove(&forecast_id);
            self.expired_forecasts = self.expired_forecasts.saturating_add(1);
            return Err(ActionConditionedPredictorError::ForecastTooLate);
        }

        let prediction = &pending.receipt.predicted_future_hv;
        let base = &pending.autonomous_base_prediction;
        let action = &pending.action_embedding;
        let mut updated_weights = self.action_weights.clone();
        let mut squared_error = 0.0f64;
        for index in 0..self.dim {
            let error = actual_future_hv[index] - prediction[index];
            squared_error += f64::from(error) * f64::from(error);
            let gradient = (error * action[index]).clamp(-1.0, 1.0);
            updated_weights[index] += self.learning_rate * gradient;
        }
        if updated_weights.iter().any(|v| !v.is_finite()) || !squared_error.is_finite() {
            return Err(ActionConditionedPredictorError::NonFiniteUpdate);
        }

        let actual = symthaea_core::hdc::ContinuousHV::from_vec(actual_future_hv.to_vec());
        let predicted = symthaea_core::hdc::ContinuousHV::from_vec(prediction.clone());
        let base_hv = symthaea_core::hdc::ContinuousHV::from_vec(base.clone());
        let prediction_error = 1.0 - actual.similarity(&predicted).clamp(-1.0, 1.0);
        let autonomous_baseline_error = 1.0 - actual.similarity(&base_hv).clamp(-1.0, 1.0);
        let score = ActionConditionedForecastScore {
            forecast_id,
            action_id: pending.receipt.action_id,
            source_frame_sequence: pending.receipt.source_frame_sequence,
            observed_frame_sequence,
            observation_delay_seconds,
            lateness_seconds: lateness,
            prediction_error,
            autonomous_baseline_error,
            improvement_over_autonomous_baseline: autonomous_baseline_error - prediction_error,
            mean_squared_error: (squared_error / self.dim as f64) as f32,
            model_version_before: self.training_steps,
            model_version_after: self.training_steps.saturating_add(1),
        };
        self.pending.remove(&forecast_id);
        self.action_weights = updated_weights;
        self.training_steps = self.training_steps.saturating_add(1);
        Ok(score)
    }

    pub fn pending_forecast_count(&self) -> usize {
        self.pending.len()
    }

    pub fn training_steps(&self) -> u64 {
        self.training_steps
    }

    pub fn expired_forecasts(&self) -> u64 {
        self.expired_forecasts
    }

    pub fn save_state(&self) -> ActionConditionedPredictorState {
        ActionConditionedPredictorState {
            schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
            dim: self.dim,
            action_weights: self.action_weights.clone(),
            learning_rate: self.learning_rate,
            max_pending_forecasts: self.max_pending_forecasts,
            max_lateness_factor: self.max_lateness_factor,
            next_forecast_id: self.next_forecast_id,
            training_steps: self.training_steps,
            expired_forecasts: self.expired_forecasts,
            pending: self.pending.values().cloned().collect(),
        }
    }

    pub fn load_state(
        &mut self,
        state: &ActionConditionedPredictorState,
    ) -> Result<(), ActionConditionedPredictorError> {
        if state.schema_version != ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION
            || state.dim != self.dim
            || state.action_weights.len() != self.dim
            || !state.action_weights.iter().all(|v| v.is_finite())
            || !state.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&state.learning_rate)
            || state.max_pending_forecasts == 0
            || state.pending.len() > state.max_pending_forecasts
            || !state.max_lateness_factor.is_finite()
            || state.max_lateness_factor < 0.0
        {
            return Err(ActionConditionedPredictorError::InvalidCheckpointState);
        }
        let mut pending = BTreeMap::new();
        for forecast in &state.pending {
            let receipt = &forecast.receipt;
            if receipt.forecast_id >= state.next_forecast_id
                || receipt.predicted_future_hv.len() != self.dim
                || forecast.autonomous_base_prediction.len() != self.dim
                || forecast.action_embedding.len() != self.dim
                || !is_lower_hex_sha256(&receipt.history_sha256)
                || !is_lower_hex_sha256(&receipt.action_manifest_sha256)
                || !is_lower_hex_sha256(&receipt.action_descriptor_sha256)
                || !receipt.horizon_seconds.is_finite()
                || receipt.horizon_seconds <= 0.0
                || receipt.predicted_future_hv.iter().chain(&forecast.autonomous_base_prediction)
                    .chain(&forecast.action_embedding).any(|v| !v.is_finite())
                || pending.insert(receipt.forecast_id, forecast.clone()).is_some()
            {
                return Err(ActionConditionedPredictorError::InvalidCheckpointState);
            }
        }

        self.action_weights = state.action_weights.clone();
        self.learning_rate = state.learning_rate;
        self.max_pending_forecasts = state.max_pending_forecasts;
        self.max_lateness_factor = state.max_lateness_factor;
        self.next_forecast_id = state.next_forecast_id;
        self.training_steps = state.training_steps;
        self.expired_forecasts = state.expired_forecasts;
        self.pending = pending;
        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<(), ActionConditionedPredictorError> {
        if !learning_rate.is_finite() || !(0.0..=1.0).contains(&learning_rate) {
            return Err(ActionConditionedPredictorError::InvalidConfiguration);
        }
        self.learning_rate = learning_rate;
        Ok(())
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predictor() -> ActionConditionedVisualResidualPredictor {
        ActionConditionedVisualResidualPredictor::new(2, 0.2, 8, 0.5).unwrap()
    }

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    #[test]
    fn zero_initialized_action_head_exactly_preserves_autonomous_forecast() {
        let mut model = predictor();
        let base = [0.1, -0.3];
        let action = [1.0, -0.5];
        let receipt = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), 7, 0.5, &base, &action
        ).unwrap();
        assert_eq!(receipt.predicted_future_hv, base);
        assert_eq!(receipt.model_version, 0);
        assert_eq!(model.pending_forecast_count(), 1);
    }

    #[test]
    fn action_residual_learns_from_later_observations_not_its_own_prediction() {
        let mut model = predictor();
        let base = [0.0, 0.0];
        let action_positive = [1.0, 1.0];
        let action_negative = [-1.0, -1.0];
        let positive = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), 17, 1.0, &base, &action_positive
        ).unwrap();
        let negative = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('d'), 3, 1.0, &base, &action_negative
        ).unwrap();
        let positive_score = model.score_and_learn(positive.forecast_id, &[0.8, 0.6], 11, 1.0).unwrap();
        let negative_score = model.score_and_learn(negative.forecast_id, &[-0.8, -0.6], 11, 1.0).unwrap();
        assert!(positive_score.mean_squared_error > 0.0);
        assert_eq!(positive_score.model_version_before, 0);
        assert_eq!(negative_score.model_version_before, 1);
        assert_eq!(model.training_steps(), 2);

        let after_positive = model.issue_forecast(
            &digest('a'), 11, &digest('b'), &digest('c'), 17, 1.0, &base, &action_positive
        ).unwrap();
        let after_negative = model.issue_forecast(
            &digest('a'), 11, &digest('b'), &digest('d'), 3, 1.0, &base, &action_negative
        ).unwrap();
        assert!(after_positive.predicted_future_hv[0] > 0.0);
        assert!(after_negative.predicted_future_hv[0] < 0.0);
    }

    #[test]
    fn forecasts_are_one_shot_and_observations_must_be_later_and_on_time() {
        let mut model = predictor();
        let receipt = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]
        ).unwrap();
        assert_eq!(
            model.score_and_learn(receipt.forecast_id, &[0.1, 0.1], 10, 1.0),
            Err(ActionConditionedPredictorError::ObservationNotLater)
        );
        assert_eq!(
            model.score_and_learn(receipt.forecast_id, &[0.1, 0.1], 11, 0.5),
            Err(ActionConditionedPredictorError::ObservationBeforeTarget)
        );
        assert_eq!(
            model.score_and_learn(receipt.forecast_id, &[0.1, 0.1], 11, 2.0),
            Err(ActionConditionedPredictorError::ForecastTooLate)
        );
        assert_eq!(model.pending_forecast_count(), 0);
        assert_eq!(model.expired_forecasts(), 1);
        assert_eq!(
            model.score_and_learn(receipt.forecast_id, &[0.1, 0.1], 11, 1.0),
            Err(ActionConditionedPredictorError::UnknownForecast)
        );
    }

    #[test]
    fn invalid_forecasts_and_checkpoint_mutations_fail_closed() {
        let mut model = predictor();
        assert_eq!(
            model.issue_forecast("bad", 0, &digest('b'), &digest('c'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]),
            Err(ActionConditionedPredictorError::InvalidDigest)
        );
        assert_eq!(
            model.issue_forecast(&digest('a'), 0, &digest('b'), &digest('c'), 1, 0.0, &[0.0, 0.0], &[1.0, 1.0]),
            Err(ActionConditionedPredictorError::InvalidHorizon)
        );
        let receipt = model.issue_forecast(
            &digest('a'), 0, &digest('b'), &digest('c'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]
        ).unwrap();
        let mut state = model.save_state();
        state.pending[0].receipt.predicted_future_hv[0] = f32::NAN;
        assert_eq!(
            model.load_state(&state),
            Err(ActionConditionedPredictorError::InvalidCheckpointState)
        );
        assert_eq!(model.pending_forecast_count(), 1);
        assert!(model.score_and_learn(receipt.forecast_id, &[0.2, 0.2], 1, 1.0).is_ok());
    }

    #[test]
    fn checkpoint_roundtrip_preserves_pending_forecasts_and_model_state() {
        let mut model = predictor();
        let receipt = model.issue_forecast(
            &digest('a'), 20, &digest('b'), &digest('c'), 4, 0.25, &[0.1, 0.2], &[1.0, 0.5]
        ).unwrap();
        let state = model.save_state();
        let mut restored = predictor();
        restored.load_state(&state).unwrap();
        assert_eq!(restored.save_state(), state);
        let score = restored.score_and_learn(receipt.forecast_id, &[0.2, 0.3], 21, 0.25).unwrap();
        assert_eq!(score.forecast_id, receipt.forecast_id);
        assert_eq!(restored.pending_forecast_count(), 0);
    }
}
