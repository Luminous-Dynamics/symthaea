//! Action-conditioned residual visual-state forecasting.
//!
//! This is a deliberately small, observation-supervised control head for
//! VIS-005. It conditions an autonomous CfC horizon prediction on an explicit
//! action embedding, and updates only when a later real visual observation is
//! supplied to score_and_learn. It does not generate pixels, prove causal
//! coupling, or authorize physical actions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedForecastReceipt {
    pub schema_version: u32,
    pub forecast_id: u64,
    pub history_sha256: String,
    pub source_frame_sequence: u64,
    pub action_manifest_sha256: String,
    pub action_descriptor_sha256: String,
    pub model_checkpoint_sha256: String,
    pub action_id: u64,
    pub action_set_id: u64,
    pub action_set_size: usize,
    pub action_set_sha256: String,
    pub model_weights_sha256: String,
    pub horizon_seconds: f32,
    /// Number of supervised updates completed before this forecast was issued.
    pub model_version: u64,
    /// Predicted future visual-manifold state; still a prediction, never an observation.
    pub autonomous_base_prediction_sha256: String,
    pub action_embedding_sha256: String,
    pub predicted_future_hv: Vec<f32>,
    pub predicted_future_hv_sha256: String,
    pub forecast_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedCandidateAction {
    pub action_id: u64,
    pub action_descriptor_sha256: String,
    pub action_embedding: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingActionConditionedForecastState {
    pub receipt: ActionConditionedForecastReceipt,
    pub autonomous_base_prediction: Vec<f32>,
    pub action_embedding: Vec<f32>,
    pub action_weights_used: Vec<f32>,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedForecastScore {
    pub schema_version: u32,
    pub forecast_id: u64,
    pub history_sha256: String,
    pub action_manifest_sha256: String,
    pub action_descriptor_sha256: String,
    pub model_checkpoint_sha256: String,
    pub forecast_sha256: String,
    pub observed_future_hv_sha256: String,
    pub score_sha256: String,
    pub action_id: u64,
    pub forecast_model_version: u64,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConditionedObservedOutcome {
    pub forecast_id: u64,
    pub actual_future_hv: Vec<f32>,
    pub observed_frame_sequence: u64,
    pub observation_delay_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionResponseMetrics {
    pub action_count: usize,
    pub zero_action_prediction_mse: f64,
    pub mean_predicted_action_effect_norm: f64,
    pub mean_observed_action_effect_norm: f64,
    pub mean_action_effect_mse: f64,
    pub mean_action_response_cosine: f64,
    pub cosine_scored_action_count: usize,
    pub missed_action_effect_count: usize,
    pub false_positive_action_effect_count: usize,
    pub both_null_action_effect_count: usize,
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
    ActionSetMismatch,
    DuplicateActionId,
    StaleForecastModelVersion,
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
        model_checkpoint_sha256: &str,
        action_id: u64,
        horizon_seconds: f32,
        autonomous_base_prediction: &[f32],
        action_embedding: &[f32],
    ) -> Result<ActionConditionedForecastReceipt, ActionConditionedPredictorError> {
        if !is_lower_hex_sha256(history_sha256)
            || !is_lower_hex_sha256(action_manifest_sha256)
            || !is_lower_hex_sha256(action_descriptor_sha256)
            || !is_lower_hex_sha256(model_checkpoint_sha256)
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
        let predicted_future_hv = predict_action_residual(
            autonomous_base_prediction, action_embedding, &self.action_weights
        )?;
        let action_set_sha256 = hash_action_descriptors(&[(
            action_id,
            action_descriptor_sha256.to_owned(),
        )]);
        let predicted_future_hv_sha256 = hash_f32_vector(
            b"vis005-action-conditioned-prediction-v1",
            &predicted_future_hv,
        );
        let mut receipt = ActionConditionedForecastReceipt {
            schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
            forecast_id,
            history_sha256: history_sha256.to_owned(),
            source_frame_sequence,
            action_manifest_sha256: action_manifest_sha256.to_owned(),
            action_descriptor_sha256: action_descriptor_sha256.to_owned(),
            model_checkpoint_sha256: model_checkpoint_sha256.to_owned(),
            action_id,
            action_set_id: forecast_id,
            action_set_size: 1,
            action_set_sha256,
            model_weights_sha256: hash_f32_vector(
                b"vis005-action-conditioned-residual-weights-v1",
                &self.action_weights,
            ),
            horizon_seconds,
            model_version: self.training_steps,
            autonomous_base_prediction_sha256: hash_f32_vector(
                b"vis005-autonomous-base-visual-hv-v1",
                autonomous_base_prediction,
            ),
            action_embedding_sha256: hash_f32_vector(
                b"vis005-action-embedding-v1",
                action_embedding,
            ),
            predicted_future_hv,
            predicted_future_hv_sha256,
            forecast_sha256: String::new(),
        };
        receipt.forecast_sha256 = hash_forecast_receipt(&receipt);
        let pending = PendingActionConditionedForecastState {
            receipt: receipt.clone(),
            autonomous_base_prediction: autonomous_base_prediction.to_vec(),
            action_embedding: action_embedding.to_vec(),
            action_weights_used: self.action_weights.clone(),
        };
        self.pending.insert(forecast_id, pending);
        self.next_forecast_id = next_id;
        Ok(receipt)
    }

    /// Issue a frozen candidate action set from one shared observed history,
    /// autonomous baseline and model version. All rows are created atomically
    /// and returned in ascending action-ID order.
    pub fn issue_action_set(
        &mut self,
        history_sha256: &str,
        source_frame_sequence: u64,
        action_manifest_sha256: &str,
        model_checkpoint_sha256: &str,
        horizon_seconds: f32,
        autonomous_base_prediction: &[f32],
        candidates: &[ActionConditionedCandidateAction],
    ) -> Result<Vec<ActionConditionedForecastReceipt>, ActionConditionedPredictorError> {
        if !is_lower_hex_sha256(history_sha256)
            || !is_lower_hex_sha256(action_manifest_sha256)
            || !is_lower_hex_sha256(model_checkpoint_sha256)
        {
            return Err(ActionConditionedPredictorError::InvalidDigest);
        }
        if !horizon_seconds.is_finite() || horizon_seconds <= 0.0 {
            return Err(ActionConditionedPredictorError::InvalidHorizon);
        }
        if autonomous_base_prediction.len() != self.dim
            || autonomous_base_prediction.iter().any(|v| !v.is_finite())
        {
            return Err(if autonomous_base_prediction.len() != self.dim {
                ActionConditionedPredictorError::InvalidVectorDimension
            } else {
                ActionConditionedPredictorError::NonFiniteValue
            });
        }
        if candidates.len() < 2 {
            return Err(ActionConditionedPredictorError::InvalidConfiguration);
        }
        if self.pending.len().checked_add(candidates.len())
            .is_none_or(|count| count > self.max_pending_forecasts)
        {
            return Err(ActionConditionedPredictorError::PendingQueueFull);
        }
        let candidate_count = u64::try_from(candidates.len())
            .map_err(|_| ActionConditionedPredictorError::ForecastIdExhausted)?;
        let next_forecast_id = self.next_forecast_id
            .checked_add(candidate_count)
            .ok_or(ActionConditionedPredictorError::ForecastIdExhausted)?;
        let action_set_id = self.next_forecast_id;

        let mut ordered: Vec<&ActionConditionedCandidateAction> = candidates.iter().collect();
        ordered.sort_unstable_by_key(|candidate| candidate.action_id);
        if ordered.windows(2).any(|pair| pair[0].action_id == pair[1].action_id) {
            return Err(ActionConditionedPredictorError::InvalidConfiguration);
        }
        for candidate in &ordered {
            if !is_lower_hex_sha256(&candidate.action_descriptor_sha256) {
                return Err(ActionConditionedPredictorError::InvalidDigest);
            }
            if candidate.action_embedding.len() != self.dim {
                return Err(ActionConditionedPredictorError::InvalidVectorDimension);
            }
            if candidate.action_embedding.iter().any(|v| !v.is_finite()) {
                return Err(ActionConditionedPredictorError::NonFiniteValue);
            }
        }

        let model_version = self.training_steps;
        let action_descriptors: Vec<(u64, String)> = ordered.iter()
            .map(|candidate| (
                candidate.action_id,
                candidate.action_descriptor_sha256.clone(),
            ))
            .collect();
        let action_set_sha256 = hash_action_descriptors(&action_descriptors);
        let model_weights_sha256 = hash_f32_vector(
            b"vis005-action-conditioned-residual-weights-v1",
            &self.action_weights,
        );
        let mut receipts = Vec::with_capacity(ordered.len());
        let mut pending_records = Vec::with_capacity(ordered.len());
        for (offset, candidate) in ordered.iter().enumerate() {
            let forecast_id = self.next_forecast_id + offset as u64;
            let predicted_future_hv = predict_action_residual(
                autonomous_base_prediction, &candidate.action_embedding, &self.action_weights
            )?;
            let predicted_future_hv_sha256 = hash_f32_vector(
                b"vis005-action-conditioned-prediction-v1",
                &predicted_future_hv,
            );
            let mut receipt = ActionConditionedForecastReceipt {
                schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
                forecast_id,
                history_sha256: history_sha256.to_owned(),
                source_frame_sequence,
                action_manifest_sha256: action_manifest_sha256.to_owned(),
                action_descriptor_sha256: candidate.action_descriptor_sha256.clone(),
                model_checkpoint_sha256: model_checkpoint_sha256.to_owned(),
                action_id: candidate.action_id,
                action_set_id,
                action_set_size: candidates.len(),
                action_set_sha256: action_set_sha256.clone(),
                model_weights_sha256: model_weights_sha256.clone(),
                horizon_seconds,
                model_version,
                autonomous_base_prediction_sha256: hash_f32_vector(
                    b"vis005-autonomous-base-visual-hv-v1",
                    autonomous_base_prediction,
                ),
                action_embedding_sha256: hash_f32_vector(
                    b"vis005-action-embedding-v1",
                    &candidate.action_embedding,
                ),
                predicted_future_hv,
                predicted_future_hv_sha256,
                forecast_sha256: String::new(),
            };
            receipt.forecast_sha256 = hash_forecast_receipt(&receipt);
            pending_records.push((
                forecast_id,
                PendingActionConditionedForecastState {
                    receipt: receipt.clone(),
                    autonomous_base_prediction: autonomous_base_prediction.to_vec(),
                    action_embedding: candidate.action_embedding.clone(),
                    action_weights_used: self.action_weights.clone(),
                },
            ));
            receipts.push(receipt);
        }

        for (forecast_id, pending) in pending_records {
            self.pending.insert(forecast_id, pending);
        }
        self.next_forecast_id = next_forecast_id;
        Ok(receipts)
    }

    /// Score a complete frozen action set against later real encoded visual
    /// observations without changing predictor weights or training version.
    ///
    /// This is the qualification/evaluation path. Held-out evaluator outcomes
    /// must use this read-only method, never score_action_set_and_learn.
    pub fn score_action_set(
        &mut self,
        outcomes: &[ActionConditionedObservedOutcome],
    ) -> Result<Vec<ActionConditionedForecastScore>, ActionConditionedPredictorError> {
        if outcomes.is_empty() {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        let mut forecast_ids = Vec::with_capacity(outcomes.len());
        let mut pending_records = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            if forecast_ids.contains(&outcome.forecast_id) {
                return Err(ActionConditionedPredictorError::ActionSetMismatch);
            }
            forecast_ids.push(outcome.forecast_id);
            let pending = self.pending.get(&outcome.forecast_id)
                .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
            pending_records.push(pending.clone());
        }

        let first = &pending_records[0].receipt;
        if first.action_set_size != outcomes.len() {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        let mut action_ids = Vec::with_capacity(pending_records.len());
        for pending in &pending_records {
            let receipt = &pending.receipt;
            if receipt.action_set_id != first.action_set_id
                || receipt.action_set_size != first.action_set_size
                || receipt.action_set_sha256 != first.action_set_sha256
                || receipt.history_sha256 != first.history_sha256
                || receipt.action_manifest_sha256 != first.action_manifest_sha256
                || receipt.source_frame_sequence != first.source_frame_sequence
                || receipt.horizon_seconds.to_bits() != first.horizon_seconds.to_bits()
                || receipt.model_checkpoint_sha256 != first.model_checkpoint_sha256
                || receipt.model_weights_sha256 != first.model_weights_sha256
                || receipt.model_version != first.model_version
            {
                return Err(ActionConditionedPredictorError::ActionSetMismatch);
            }
            action_ids.push(receipt.action_id);
        }
        action_ids.sort_unstable();
        if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ActionConditionedPredictorError::DuplicateActionId);
        }
        let mut descriptors: Vec<(u64, String)> = pending_records.iter()
            .map(|pending| (
                pending.receipt.action_id,
                pending.receipt.action_descriptor_sha256.clone(),
            ))
            .collect();
        descriptors.sort_unstable_by_key(|(action_id, _)| *action_id);
        if hash_action_descriptors(&descriptors) != first.action_set_sha256 {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }

        let mut all_late = false;
        for (pending, outcome) in pending_records.iter().zip(outcomes) {
            if outcome.actual_future_hv.len() != self.dim {
                return Err(ActionConditionedPredictorError::InvalidVectorDimension);
            }
            if outcome.actual_future_hv.iter().any(|value| !value.is_finite()) {
                return Err(ActionConditionedPredictorError::NonFiniteValue);
            }
            if outcome.observed_frame_sequence <= pending.receipt.source_frame_sequence {
                return Err(ActionConditionedPredictorError::ObservationNotLater);
            }
            if !outcome.observation_delay_seconds.is_finite()
                || outcome.observation_delay_seconds < 0.0
            {
                return Err(ActionConditionedPredictorError::NonFiniteValue);
            }
            if outcome.observation_delay_seconds < pending.receipt.horizon_seconds {
                return Err(ActionConditionedPredictorError::ObservationBeforeTarget);
            }
            let lateness = outcome.observation_delay_seconds - pending.receipt.horizon_seconds;
            all_late |= lateness > pending.receipt.horizon_seconds * self.max_lateness_factor;
        }
        if all_late {
            for forecast_id in &forecast_ids {
                self.pending.remove(forecast_id);
            }
            self.expired_forecasts = self.expired_forecasts
                .saturating_add(forecast_ids.len() as u64);
            return Err(ActionConditionedPredictorError::ForecastTooLate);
        }

        let mut scores = Vec::with_capacity(outcomes.len());
        for (pending, outcome) in pending_records.iter().zip(outcomes) {
            let actual = symthaea_core::hdc::ContinuousHV::from_vec(outcome.actual_future_hv.clone());
            let predicted = symthaea_core::hdc::ContinuousHV::from_vec(
                pending.receipt.predicted_future_hv.clone()
            );
            let base = symthaea_core::hdc::ContinuousHV::from_vec(
                pending.autonomous_base_prediction.clone()
            );
            let prediction_error = 1.0 - actual.similarity(&predicted).clamp(-1.0, 1.0);
            let autonomous_baseline_error = 1.0 - actual.similarity(&base).clamp(-1.0, 1.0);
            let mut squared_error = 0.0f64;
            for (actual_value, predicted_value) in outcome.actual_future_hv.iter()
                .zip(&pending.receipt.predicted_future_hv)
            {
                let error = actual_value - predicted_value;
                squared_error += f64::from(error) * f64::from(error);
            }
            let observed_future_hv_sha256 = hash_f32_vector(
                b"vis005-observed-future-visual-hv-v1",
                &outcome.actual_future_hv,
            );
            let mut score = ActionConditionedForecastScore {
                schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
                forecast_id: pending.receipt.forecast_id,
                history_sha256: pending.receipt.history_sha256.clone(),
                action_manifest_sha256: pending.receipt.action_manifest_sha256.clone(),
                action_descriptor_sha256: pending.receipt.action_descriptor_sha256.clone(),
                model_checkpoint_sha256: pending.receipt.model_checkpoint_sha256.clone(),
                forecast_sha256: pending.receipt.forecast_sha256.clone(),
                observed_future_hv_sha256,
                score_sha256: String::new(),
                action_id: pending.receipt.action_id,
                forecast_model_version: pending.receipt.model_version,
                source_frame_sequence: pending.receipt.source_frame_sequence,
                observed_frame_sequence: outcome.observed_frame_sequence,
                observation_delay_seconds: outcome.observation_delay_seconds,
                lateness_seconds: outcome.observation_delay_seconds - pending.receipt.horizon_seconds,
                prediction_error,
                autonomous_baseline_error,
                improvement_over_autonomous_baseline: autonomous_baseline_error - prediction_error,
                mean_squared_error: (squared_error / self.dim as f64) as f32,
                model_version_before: self.training_steps,
                model_version_after: self.training_steps,
            };
            score.score_sha256 = hash_forecast_score(&score);
            if [
                score.prediction_error,
                score.autonomous_baseline_error,
                score.improvement_over_autonomous_baseline,
                score.mean_squared_error,
            ].iter().any(|value| !value.is_finite())
            {
                return Err(ActionConditionedPredictorError::NonFiniteUpdate);
            }
            scores.push(score);
        }

        for forecast_id in &forecast_ids {
            self.pending.remove(forecast_id);
        }
        Ok(scores)
    }

    /// Score and learn from a complete action set using authorized supervised-
    /// training observations only. Never pass held-out qualification outcomes
    /// here; use score_action_set for those. All forecasts must share
    /// the exact history, action manifest, source frame, horizon, checkpoint
    /// and model version. Scores are computed first; one averaged update is
    /// applied only after the entire set validates.
    pub fn score_action_set_and_learn(
        &mut self,
        outcomes: &[ActionConditionedObservedOutcome],
    ) -> Result<Vec<ActionConditionedForecastScore>, ActionConditionedPredictorError> {
        if outcomes.len() < 2 {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        let mut forecast_ids = Vec::with_capacity(outcomes.len());
        let mut pending_records = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            if forecast_ids.contains(&outcome.forecast_id) {
                return Err(ActionConditionedPredictorError::ActionSetMismatch);
            }
            forecast_ids.push(outcome.forecast_id);
            let pending = self.pending.get(&outcome.forecast_id)
                .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
            pending_records.push(pending.clone());
        }

        let first = &pending_records[0].receipt;
        if first.action_set_size != outcomes.len() {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        let mut action_ids = Vec::with_capacity(pending_records.len());
        for pending in &pending_records {
            let receipt = &pending.receipt;
            if receipt.history_sha256 != first.history_sha256
                || receipt.action_manifest_sha256 != first.action_manifest_sha256
                || receipt.source_frame_sequence != first.source_frame_sequence
                || receipt.horizon_seconds.to_bits() != first.horizon_seconds.to_bits()
                || receipt.model_checkpoint_sha256 != first.model_checkpoint_sha256
                || receipt.action_set_id != first.action_set_id
                || receipt.action_set_size != first.action_set_size
                || receipt.action_set_sha256 != first.action_set_sha256
                || receipt.model_weights_sha256 != first.model_weights_sha256
                || receipt.model_version != first.model_version
                || receipt.model_version != self.training_steps
            {
                return Err(ActionConditionedPredictorError::ActionSetMismatch);
            }
            action_ids.push(receipt.action_id);
        }
        action_ids.sort_unstable();
        if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ActionConditionedPredictorError::DuplicateActionId);
        }
        let descriptors: Vec<(u64, String)> = pending_records.iter()
            .map(|pending| (
                pending.receipt.action_id,
                pending.receipt.action_descriptor_sha256.clone(),
            ))
            .collect();
        let mut descriptors = descriptors;
        descriptors.sort_unstable_by_key(|(action_id, _)| *action_id);
        if hash_action_descriptors(&descriptors) != first.action_set_sha256 {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        if hash_f32_vector(
            b"vis005-action-conditioned-residual-weights-v1",
            &self.action_weights,
        ) != first.model_weights_sha256 {
            return Err(ActionConditionedPredictorError::StaleForecastModelVersion);
        }
        let next_training_step = self.training_steps.checked_add(1)
            .ok_or(ActionConditionedPredictorError::ForecastIdExhausted)?;

        let mut all_late = false;
        for (pending, outcome) in pending_records.iter().zip(outcomes) {
            if outcome.actual_future_hv.len() != self.dim {
                return Err(ActionConditionedPredictorError::InvalidVectorDimension);
            }
            if outcome.actual_future_hv.iter().any(|value| !value.is_finite()) {
                return Err(ActionConditionedPredictorError::NonFiniteValue);
            }
            if outcome.observed_frame_sequence <= pending.receipt.source_frame_sequence {
                return Err(ActionConditionedPredictorError::ObservationNotLater);
            }
            if !outcome.observation_delay_seconds.is_finite()
                || outcome.observation_delay_seconds < 0.0
            {
                return Err(ActionConditionedPredictorError::NonFiniteValue);
            }
            if outcome.observation_delay_seconds < pending.receipt.horizon_seconds {
                return Err(ActionConditionedPredictorError::ObservationBeforeTarget);
            }
            let lateness = outcome.observation_delay_seconds - pending.receipt.horizon_seconds;
            all_late |= lateness > pending.receipt.horizon_seconds * self.max_lateness_factor;
        }
        if all_late {
            for forecast_id in &forecast_ids {
                self.pending.remove(forecast_id);
            }
            self.expired_forecasts = self.expired_forecasts
                .saturating_add(forecast_ids.len() as u64);
            return Err(ActionConditionedPredictorError::ForecastTooLate);
        }

        let mut updated_weights = self.action_weights.clone();
        let mut scores = Vec::with_capacity(outcomes.len());
        let set_count = outcomes.len() as f32;
        for (pending, outcome) in pending_records.iter().zip(outcomes) {
            let prediction = &pending.receipt.predicted_future_hv;
            let base = &pending.autonomous_base_prediction;
            let action = &pending.action_embedding;
            let mut squared_error = 0.0f64;
            for index in 0..self.dim {
                let error = outcome.actual_future_hv[index] - prediction[index];
                squared_error += f64::from(error) * f64::from(error);
                let gradient = (error * action[index]).clamp(-1.0, 1.0);
                updated_weights[index] += self.learning_rate * gradient / set_count;
            }
            if !squared_error.is_finite() {
                return Err(ActionConditionedPredictorError::NonFiniteUpdate);
            }
            let actual = symthaea_core::hdc::ContinuousHV::from_vec(outcome.actual_future_hv.clone());
            let predicted = symthaea_core::hdc::ContinuousHV::from_vec(prediction.clone());
            let base_hv = symthaea_core::hdc::ContinuousHV::from_vec(base.clone());
            let observation_delay_seconds = outcome.observation_delay_seconds;
            let lateness_seconds = observation_delay_seconds - pending.receipt.horizon_seconds;
            let mut score = ActionConditionedForecastScore {
                schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
                forecast_id: pending.receipt.forecast_id,
                history_sha256: pending.receipt.history_sha256.clone(),
                action_manifest_sha256: pending.receipt.action_manifest_sha256.clone(),
                action_descriptor_sha256: pending.receipt.action_descriptor_sha256.clone(),
                model_checkpoint_sha256: pending.receipt.model_checkpoint_sha256.clone(),
                forecast_sha256: pending.receipt.forecast_sha256.clone(),
                observed_future_hv_sha256: hash_f32_vector(
                    b"vis005-observed-future-visual-hv-v1",
                    &outcome.actual_future_hv,
                ),
                score_sha256: String::new(),
                action_id: pending.receipt.action_id,
                forecast_model_version: pending.receipt.model_version,
                source_frame_sequence: pending.receipt.source_frame_sequence,
                observed_frame_sequence: outcome.observed_frame_sequence,
                observation_delay_seconds,
                lateness_seconds,
                prediction_error: 1.0 - actual.similarity(&predicted).clamp(-1.0, 1.0),
                autonomous_baseline_error: 1.0 - actual.similarity(&base_hv).clamp(-1.0, 1.0),
                improvement_over_autonomous_baseline: 0.0,
                mean_squared_error: (squared_error / self.dim as f64) as f32,
                model_version_before: self.training_steps,
                model_version_after: next_training_step,
            };
            score.improvement_over_autonomous_baseline =
                score.autonomous_baseline_error - score.prediction_error;
            score.score_sha256 = hash_forecast_score(&score);
            if [
                score.prediction_error,
                score.autonomous_baseline_error,
                score.improvement_over_autonomous_baseline,
                score.mean_squared_error,
            ].iter().any(|value| !value.is_finite())
            {
                return Err(ActionConditionedPredictorError::NonFiniteUpdate);
            }
            scores.push(score);
        }
        if updated_weights.iter().any(|value| !value.is_finite()) {
            return Err(ActionConditionedPredictorError::NonFiniteUpdate);
        }

        for forecast_id in &forecast_ids {
            self.pending.remove(forecast_id);
        }
        self.action_weights = updated_weights;
        self.training_steps = next_training_step;
        Ok(scores)
    }

    /// Score a previously sealed forecast against a later real encoded visual
    /// observation, then apply one online update. This is a supervised-training
    /// path only; never use held-out qualification outcomes here. Each forecast
    /// ID is consumed at most once. An over-late target expires without training.
    pub fn score_and_learn(
        &mut self,
        forecast_id: u64,
        actual_future_hv: &[f32],
        observed_frame_sequence: u64,
        observation_delay_seconds: f32,
    ) -> Result<ActionConditionedForecastScore, ActionConditionedPredictorError> {
        let pending = self.pending.get(&forecast_id)
            .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
        if pending.receipt.action_set_size != 1 {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        if pending.receipt.model_version != self.training_steps
            || hash_f32_vector(
                b"vis005-action-conditioned-residual-weights-v1",
                &self.action_weights,
            ) != pending.receipt.model_weights_sha256
        {
            return Err(ActionConditionedPredictorError::StaleForecastModelVersion);
        }
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
        let observed_future_hv_sha256 = hash_f32_vector(
            b"vis005-observed-future-visual-hv-v1",
            actual_future_hv,
        );
        let mut score = ActionConditionedForecastScore {
            schema_version: ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION,
            forecast_id,
            history_sha256: pending.receipt.history_sha256.clone(),
            action_manifest_sha256: pending.receipt.action_manifest_sha256.clone(),
            action_descriptor_sha256: pending.receipt.action_descriptor_sha256.clone(),
            model_checkpoint_sha256: pending.receipt.model_checkpoint_sha256.clone(),
            forecast_sha256: pending.receipt.forecast_sha256.clone(),
            observed_future_hv_sha256,
            score_sha256: String::new(),
            action_id: pending.receipt.action_id,
            forecast_model_version: pending.receipt.model_version,
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
        score.score_sha256 = hash_forecast_score(&score);
        if [
            score.prediction_error,
            score.autonomous_baseline_error,
            score.improvement_over_autonomous_baseline,
            score.mean_squared_error,
        ].iter().any(|value| !value.is_finite())
        {
            return Err(ActionConditionedPredictorError::NonFiniteUpdate);
        }
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
            || state.next_forecast_id == 0
            || state.pending.len() > state.max_pending_forecasts
            || !state.max_lateness_factor.is_finite()
            || state.max_lateness_factor < 0.0
        {
            return Err(ActionConditionedPredictorError::InvalidCheckpointState);
        }
        let mut pending = BTreeMap::new();
        for forecast in &state.pending {
            let receipt = &forecast.receipt;
            if receipt.schema_version != ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION
                || receipt.forecast_id == 0
                || receipt.forecast_id >= state.next_forecast_id
                || receipt.action_set_id == 0
                || receipt.action_set_id > receipt.forecast_id
                || receipt.action_set_size == 0
                || receipt.model_version > state.training_steps
                || !is_lower_hex_sha256(&receipt.model_checkpoint_sha256)
                || !is_lower_hex_sha256(&receipt.autonomous_base_prediction_sha256)
                || !is_lower_hex_sha256(&receipt.action_embedding_sha256)
                || !is_lower_hex_sha256(&receipt.action_set_sha256)
                || !is_lower_hex_sha256(&receipt.model_weights_sha256)
                || !is_lower_hex_sha256(&receipt.predicted_future_hv_sha256)
                || !is_lower_hex_sha256(&receipt.forecast_sha256)
                || hash_f32_vector(
                    b"vis005-autonomous-base-visual-hv-v1",
                    &forecast.autonomous_base_prediction,
                ) != receipt.autonomous_base_prediction_sha256
                || hash_f32_vector(
                    b"vis005-action-embedding-v1",
                    &forecast.action_embedding,
                ) != receipt.action_embedding_sha256
                || hash_f32_vector(
                    b"vis005-action-conditioned-residual-weights-v1",
                    &forecast.action_weights_used,
                ) != receipt.model_weights_sha256
                || predict_action_residual(
                    &forecast.autonomous_base_prediction,
                    &forecast.action_embedding,
                    &forecast.action_weights_used,
                ).map_or(true, |predicted| {
                    predicted.iter().zip(&receipt.predicted_future_hv)
                        .any(|(left, right)| left.to_bits() != right.to_bits())
                })
                || receipt.action_set_sha256 != hash_action_descriptors(&[(
                    receipt.action_id,
                    receipt.action_descriptor_sha256.clone(),
                )]) && receipt.action_set_size == 1
                || hash_f32_vector(
                    b"vis005-action-conditioned-prediction-v1",
                    &receipt.predicted_future_hv,
                ) != receipt.predicted_future_hv_sha256
                || hash_forecast_receipt(receipt) != receipt.forecast_sha256
                || receipt.predicted_future_hv.len() != self.dim
                || forecast.autonomous_base_prediction.len() != self.dim
                || forecast.action_embedding.len() != self.dim
                || forecast.action_weights_used.len() != self.dim
                || !is_lower_hex_sha256(&receipt.history_sha256)
                || !is_lower_hex_sha256(&receipt.action_manifest_sha256)
                || !is_lower_hex_sha256(&receipt.action_descriptor_sha256)
                || !receipt.horizon_seconds.is_finite()
                || receipt.horizon_seconds <= 0.0
                || receipt.predicted_future_hv.iter().chain(&forecast.autonomous_base_prediction)
                    .chain(&forecast.action_embedding)
                    .chain(&forecast.action_weights_used).any(|v| !v.is_finite())
                || pending.insert(receipt.forecast_id, forecast.clone()).is_some()
            {
                return Err(ActionConditionedPredictorError::InvalidCheckpointState);
            }
        }

        let mut action_sets: BTreeMap<u64, Vec<&PendingActionConditionedForecastState>> = BTreeMap::new();
        for forecast in pending.values() {
            action_sets.entry(forecast.receipt.action_set_id).or_default().push(forecast);
        }
        for forecasts in action_sets.values() {
            let first = &forecasts[0].receipt;
            if forecasts.len() != first.action_set_size {
                return Err(ActionConditionedPredictorError::InvalidCheckpointState);
            }
            let mut descriptors: Vec<(u64, String)> = forecasts.iter()
                .map(|forecast| (
                    forecast.receipt.action_id,
                    forecast.receipt.action_descriptor_sha256.clone(),
                ))
                .collect();
            descriptors.sort_unstable_by_key(|(action_id, _)| *action_id);
            if forecasts.iter().any(|forecast| {
                forecast.receipt.action_set_id != first.action_set_id
                    || forecast.receipt.action_set_size != first.action_set_size
                    || forecast.receipt.action_set_sha256 != first.action_set_sha256
                    || forecast.receipt.history_sha256 != first.history_sha256
                    || forecast.receipt.action_manifest_sha256 != first.action_manifest_sha256
                    || forecast.receipt.model_checkpoint_sha256 != first.model_checkpoint_sha256
                    || forecast.receipt.model_weights_sha256 != first.model_weights_sha256
                    || forecast.receipt.model_version != first.model_version
                    || forecast.receipt.source_frame_sequence != first.source_frame_sequence
                    || forecast.receipt.horizon_seconds.to_bits() != first.horizon_seconds.to_bits()
            }) || hash_action_descriptors(&descriptors) != first.action_set_sha256 {
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

/// Compare action-induced visual-state changes against one same-history
/// no-op branch. This is an action-response diagnostic, not the exact ARC
/// metric from CoCo and not a calibrated uncertainty or safety certificate.
pub fn evaluate_action_response_set(
    receipts: &[ActionConditionedForecastReceipt],
    outcomes: &[ActionConditionedObservedOutcome],
    zero_action_id: u64,
) -> Result<ActionResponseMetrics, ActionConditionedPredictorError> {
    if receipts.len() < 2 || receipts.len() != outcomes.len() {
        return Err(ActionConditionedPredictorError::ActionSetMismatch);
    }
    let first = &receipts[0];
    if first.action_set_size != receipts.len() {
        return Err(ActionConditionedPredictorError::ActionSetMismatch);
    }
    let dim = first.predicted_future_hv.len();
    if dim == 0 {
        return Err(ActionConditionedPredictorError::InvalidVectorDimension);
    }

    let mut action_ids = Vec::with_capacity(receipts.len());
    let mut descriptors = Vec::with_capacity(receipts.len());
    let mut receipt_by_id = BTreeMap::new();
    for receipt in receipts {
        if receipt.schema_version != ACTION_CONDITIONED_PREDICTOR_STATE_SCHEMA_VERSION
            || receipt.action_set_id != first.action_set_id
            || receipt.action_set_size != first.action_set_size
            || receipt.action_set_sha256 != first.action_set_sha256
            || receipt.history_sha256 != first.history_sha256
            || receipt.action_manifest_sha256 != first.action_manifest_sha256
            || receipt.source_frame_sequence != first.source_frame_sequence
            || receipt.horizon_seconds.to_bits() != first.horizon_seconds.to_bits()
            || receipt.model_checkpoint_sha256 != first.model_checkpoint_sha256
            || receipt.model_weights_sha256 != first.model_weights_sha256
            || receipt.model_version != first.model_version
            || receipt.predicted_future_hv.len() != dim
            || !is_lower_hex_sha256(&receipt.predicted_future_hv_sha256)
            || !is_lower_hex_sha256(&receipt.forecast_sha256)
            || hash_f32_vector(
                b"vis005-action-conditioned-prediction-v1",
                &receipt.predicted_future_hv,
            ) != receipt.predicted_future_hv_sha256
            || hash_forecast_receipt(receipt) != receipt.forecast_sha256
        {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
        action_ids.push(receipt.action_id);
        descriptors.push((receipt.action_id, receipt.action_descriptor_sha256.clone()));
        if receipt_by_id.insert(receipt.forecast_id, receipt).is_some() {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
    }
    action_ids.sort_unstable();
    if action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ActionConditionedPredictorError::DuplicateActionId);
    }
    descriptors.sort_unstable_by_key(|(action_id, _)| *action_id);
    if hash_action_descriptors(&descriptors) != first.action_set_sha256
        || !action_ids.contains(&zero_action_id)
    {
        return Err(ActionConditionedPredictorError::ActionSetMismatch);
    }

    let mut outcome_by_id = BTreeMap::new();
    for outcome in outcomes {
        if outcome_by_id.insert(outcome.forecast_id, outcome).is_some() {
            return Err(ActionConditionedPredictorError::ActionSetMismatch);
        }
    }
    if outcome_by_id.len() != receipts.len() {
        return Err(ActionConditionedPredictorError::ActionSetMismatch);
    }
    for receipt in receipts {
        let outcome = outcome_by_id.get(&receipt.forecast_id)
            .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
        if outcome.actual_future_hv.len() != dim {
            return Err(ActionConditionedPredictorError::InvalidVectorDimension);
        }
        if outcome.actual_future_hv.iter().any(|value| !value.is_finite()) {
            return Err(ActionConditionedPredictorError::NonFiniteValue);
        }
        if outcome.observed_frame_sequence <= receipt.source_frame_sequence {
            return Err(ActionConditionedPredictorError::ObservationNotLater);
        }
        if !outcome.observation_delay_seconds.is_finite()
            || outcome.observation_delay_seconds < receipt.horizon_seconds
        {
            return Err(ActionConditionedPredictorError::ObservationBeforeTarget);
        }
    }

    let zero_receipt = receipts.iter()
        .find(|receipt| receipt.action_id == zero_action_id)
        .ok_or(ActionConditionedPredictorError::ActionSetMismatch)?;
    let zero_outcome = outcome_by_id.get(&zero_receipt.forecast_id)
        .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
    let mut zero_mse = 0.0f64;
    for (predicted, actual) in zero_receipt.predicted_future_hv.iter()
        .zip(&zero_outcome.actual_future_hv)
    {
        let error = f64::from(*predicted) - f64::from(*actual);
        zero_mse += error * error;
    }
    zero_mse /= dim as f64;

    let mut predicted_norm_total = 0.0;
    let mut observed_norm_total = 0.0;
    let mut effect_mse_total = 0.0;
    let mut cosine_total = 0.0;
    let mut cosine_scored = 0usize;
    let mut missed = 0usize;
    let mut false_positive = 0usize;
    let mut both_null = 0usize;
    let mut action_count = 0usize;
    const NULL_EPSILON: f64 = 1e-9;

    for receipt in receipts {
        if receipt.action_id == zero_action_id {
            continue;
        }
        let outcome = outcome_by_id.get(&receipt.forecast_id)
            .ok_or(ActionConditionedPredictorError::UnknownForecast)?;
        let mut predicted_norm_sq = 0.0;
        let mut observed_norm_sq = 0.0;
        let mut effect_error_sq = 0.0;
        let mut dot = 0.0;
        for index in 0..dim {
            let predicted_delta = f64::from(receipt.predicted_future_hv[index])
                - f64::from(zero_receipt.predicted_future_hv[index]);
            let observed_delta = f64::from(outcome.actual_future_hv[index])
                - f64::from(zero_outcome.actual_future_hv[index]);
            let difference = predicted_delta - observed_delta;
            predicted_norm_sq += predicted_delta * predicted_delta;
            observed_norm_sq += observed_delta * observed_delta;
            effect_error_sq += difference * difference;
            dot += predicted_delta * observed_delta;
        }
        let predicted_norm = predicted_norm_sq.sqrt();
        let observed_norm = observed_norm_sq.sqrt();
        predicted_norm_total += predicted_norm;
        observed_norm_total += observed_norm;
        effect_mse_total += effect_error_sq / dim as f64;
        action_count += 1;
        match (predicted_norm > NULL_EPSILON, observed_norm > NULL_EPSILON) {
            (true, true) => {
                cosine_total += (dot / (predicted_norm * observed_norm)).clamp(-1.0, 1.0);
                cosine_scored += 1;
            }
            (false, true) => missed += 1,
            (true, false) => false_positive += 1,
            (false, false) => both_null += 1,
        }
    }
    if action_count == 0 {
        return Err(ActionConditionedPredictorError::ActionSetMismatch);
    }
    let divisor = action_count as f64;
    let metrics = ActionResponseMetrics {
        action_count,
        zero_action_prediction_mse: zero_mse,
        mean_predicted_action_effect_norm: predicted_norm_total / divisor,
        mean_observed_action_effect_norm: observed_norm_total / divisor,
        mean_action_effect_mse: effect_mse_total / divisor,
        mean_action_response_cosine: if cosine_scored == 0 {
            0.0
        } else {
            cosine_total / cosine_scored as f64
        },
        cosine_scored_action_count: cosine_scored,
        missed_action_effect_count: missed,
        false_positive_action_effect_count: false_positive,
        both_null_action_effect_count: both_null,
    };
    if [
        metrics.zero_action_prediction_mse,
        metrics.mean_predicted_action_effect_norm,
        metrics.mean_observed_action_effect_norm,
        metrics.mean_action_effect_mse,
        metrics.mean_action_response_cosine,
    ].iter().any(|value| !value.is_finite()) {
        return Err(ActionConditionedPredictorError::NonFiniteUpdate);
    }
    Ok(metrics)
}

fn hash_action_descriptors(action_descriptors: &[(u64, String)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vis005-action-conditioned-action-set-v1\0");
    hasher.update((action_descriptors.len() as u64).to_le_bytes());
    for (action_id, descriptor_sha256) in action_descriptors {
        hasher.update(action_id.to_le_bytes());
        hash_text(&mut hasher, descriptor_sha256);
    }
    format!("{:x}", hasher.finalize())
}

fn predict_action_residual(
    base: &[f32],
    action: &[f32],
    weights: &[f32],
) -> Result<Vec<f32>, ActionConditionedPredictorError> {
    if base.len() != action.len() || action.len() != weights.len() {
        return Err(ActionConditionedPredictorError::InvalidVectorDimension);
    }
    let predicted: Vec<f32> = base.iter()
        .zip(action)
        .zip(weights)
        .map(|((&base, &action), &weight)| base + action * weight)
        .collect();
    if predicted.iter().any(|value| !value.is_finite()) {
        return Err(ActionConditionedPredictorError::NonFiniteValue);
    }
    Ok(predicted)
}

fn hash_f32_vector(domain: &[u8], values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_forecast_receipt(receipt: &ActionConditionedForecastReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vis005-action-conditioned-forecast-receipt-v1\0");
    hasher.update(receipt.schema_version.to_le_bytes());
    hasher.update(receipt.forecast_id.to_le_bytes());
    hash_text(&mut hasher, &receipt.history_sha256);
    hasher.update(receipt.source_frame_sequence.to_le_bytes());
    hash_text(&mut hasher, &receipt.action_manifest_sha256);
    hash_text(&mut hasher, &receipt.action_descriptor_sha256);
    hash_text(&mut hasher, &receipt.model_checkpoint_sha256);
    hasher.update(receipt.action_id.to_le_bytes());
    hasher.update(receipt.action_set_id.to_le_bytes());
    hasher.update((receipt.action_set_size as u64).to_le_bytes());
    hash_text(&mut hasher, &receipt.action_set_sha256);
    hash_text(&mut hasher, &receipt.model_weights_sha256);
    hasher.update(receipt.horizon_seconds.to_bits().to_le_bytes());
    hasher.update(receipt.model_version.to_le_bytes());
    hash_text(&mut hasher, &receipt.autonomous_base_prediction_sha256);
    hash_text(&mut hasher, &receipt.action_embedding_sha256);
    hash_text(&mut hasher, &receipt.predicted_future_hv_sha256);
    format!("{:x}", hasher.finalize())
}

fn hash_forecast_score(score: &ActionConditionedForecastScore) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vis005-action-conditioned-score-receipt-v1\0");
    hasher.update(score.schema_version.to_le_bytes());
    hasher.update(score.forecast_id.to_le_bytes());
    hash_text(&mut hasher, &score.history_sha256);
    hash_text(&mut hasher, &score.action_manifest_sha256);
    hash_text(&mut hasher, &score.action_descriptor_sha256);
    hash_text(&mut hasher, &score.model_checkpoint_sha256);
    hash_text(&mut hasher, &score.forecast_sha256);
    hash_text(&mut hasher, &score.observed_future_hv_sha256);
    hasher.update(score.action_id.to_le_bytes());
    hasher.update(score.forecast_model_version.to_le_bytes());
    hasher.update(score.source_frame_sequence.to_le_bytes());
    hasher.update(score.observed_frame_sequence.to_le_bytes());
    hasher.update(score.observation_delay_seconds.to_bits().to_le_bytes());
    hasher.update(score.lateness_seconds.to_bits().to_le_bytes());
    hasher.update(score.prediction_error.to_bits().to_le_bytes());
    hasher.update(score.autonomous_baseline_error.to_bits().to_le_bytes());
    hasher.update(score.improvement_over_autonomous_baseline.to_bits().to_le_bytes());
    hasher.update(score.mean_squared_error.to_bits().to_le_bytes());
    hasher.update(score.model_version_before.to_le_bytes());
    hasher.update(score.model_version_after.to_le_bytes());
    format!("{:x}", hasher.finalize())
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
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'), 7, 0.5, &base, &action
        ).unwrap();
        assert_eq!(receipt.predicted_future_hv, base);
        assert_eq!(receipt.model_version, 0);
        assert_eq!(
            receipt.predicted_future_hv_sha256,
            hash_f32_vector(b"vis005-action-conditioned-prediction-v1", &base)
        );
        assert_eq!(receipt.forecast_sha256, hash_forecast_receipt(&receipt));
        assert_eq!(receipt.action_set_size, 1);
        assert_eq!(
            receipt.model_weights_sha256,
            hash_f32_vector(
                b"vis005-action-conditioned-residual-weights-v1",
                &[0.0, 0.0],
            )
        );
        assert_eq!(model.pending_forecast_count(), 1);
    }


    #[test]
    fn action_set_issues_every_candidate_from_one_history_and_model_version() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 50, &digest('b'), &digest('e'), 0.25, &[0.2, -0.1], &candidates
        ).unwrap();
        assert_eq!(receipts.len(), 2);
        assert_eq!(receipts[0].action_id, 3);
        assert_eq!(receipts[1].action_id, 17);
        assert!(receipts.iter().all(|receipt| receipt.history_sha256 == digest('a')));
        assert!(receipts.iter().all(|receipt| receipt.action_manifest_sha256 == digest('b')));
        assert!(receipts.iter().all(|receipt| receipt.source_frame_sequence == 50));
        assert!(receipts.iter().all(|receipt| receipt.model_version == 0));
        assert_ne!(receipts[0].action_descriptor_sha256, receipts[1].action_descriptor_sha256);
        assert_eq!(model.pending_forecast_count(), 2);
    }

    #[test]
    fn action_set_issue_is_atomic_on_duplicate_ids_or_bad_action_embedding() {
        let mut model = predictor();
        let duplicate = vec![
            ActionConditionedCandidateAction {
                action_id: 7,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 7,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![0.0, 1.0],
            },
        ];
        assert_eq!(
            model.issue_action_set(&digest('a'), 1, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &duplicate),
            Err(ActionConditionedPredictorError::InvalidConfiguration)
        );
        let malformed = vec![
            ActionConditionedCandidateAction {
                action_id: 7,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 8,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![0.0],
            },
        ];
        assert_eq!(
            model.issue_action_set(&digest('a'), 1, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &malformed),
            Err(ActionConditionedPredictorError::InvalidVectorDimension)
        );
        assert_eq!(model.pending_forecast_count(), 0);
    }




    #[test]
    fn zero_action_and_inverse_action_metamorphic_controls_hold_after_learning() {
        let mut model = predictor();
        let positive_action = [1.0, -1.0];
        let training_forecast = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'),
            7, 1.0, &[0.0, 0.0], &positive_action
        ).unwrap();
        model.score_and_learn(
            training_forecast.forecast_id,
            &[0.6, -0.4],
            11,
            1.0,
        ).unwrap();

        let base = [0.1, -0.2];
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 10,
                action_descriptor_sha256: digest('f'),
                action_embedding: vec![0.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 11,
                action_descriptor_sha256: digest('c'),
                action_embedding: positive_action.to_vec(),
            },
            ActionConditionedCandidateAction {
                action_id: 12,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-positive_action[0], -positive_action[1]],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 11, &digest('b'), &digest('e'), 1.0, &base, &candidates
        ).unwrap();
        let find = |action_id| receipts.iter()
            .find(|receipt| receipt.action_id == action_id)
            .unwrap();
        let zero = &find(10).predicted_future_hv;
        let reference = &find(11).predicted_future_hv;
        let inverse = &find(12).predicted_future_hv;

        assert_eq!(zero, &base);
        for index in 0..base.len() {
            assert!((reference[index] + inverse[index] - 2.0 * base[index]).abs() < 1e-7);
        }
        assert!(reference.iter().zip(base).any(|(predicted, base)| (predicted - base).abs() > 1e-6));
    }


    #[test]
    fn action_response_evaluator_scores_effects_relative_to_same_history_noop() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 0,
                action_descriptor_sha256: digest('f'),
                action_embedding: vec![0.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 1,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 2,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 20, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let outcome = |action_id: u64, actual: Vec<f32>| ActionConditionedObservedOutcome {
            forecast_id: receipts.iter().find(|r| r.action_id == action_id).unwrap().forecast_id,
            actual_future_hv: actual,
            observed_frame_sequence: 21,
            observation_delay_seconds: 1.0,
        };
        let outcomes = vec![
            outcome(2, vec![-1.0, 0.0]),
            outcome(0, vec![0.0, 0.0]),
            outcome(1, vec![1.0, 0.0]),
        ];
        let metrics = evaluate_action_response_set(&receipts, &outcomes, 0).unwrap();
        assert_eq!(metrics.action_count, 2);
        assert_eq!(metrics.zero_action_prediction_mse, 0.0);
        assert_eq!(metrics.mean_predicted_action_effect_norm, 0.0);
        assert_eq!(metrics.mean_observed_action_effect_norm, 1.0);
        assert_eq!(metrics.mean_action_effect_mse, 1.0);
        assert_eq!(metrics.mean_action_response_cosine, 0.0);
        assert_eq!(metrics.cosine_scored_action_count, 0);
        assert_eq!(metrics.missed_action_effect_count, 2);
    }


    #[test]
    fn action_response_evaluator_rewards_correct_direction_and_magnitude() {
        let mut model = predictor();
        let training = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'),
            1, 1.0, &[0.0, 0.0], &[1.0, 0.0]
        ).unwrap();
        model.score_and_learn(training.forecast_id, &[1.0, 0.0], 11, 1.0).unwrap();

        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 0,
                action_descriptor_sha256: digest('f'),
                action_embedding: vec![0.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 1,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 2,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 11, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let outcomes = receipts.iter().map(|receipt| ActionConditionedObservedOutcome {
            forecast_id: receipt.forecast_id,
            actual_future_hv: match receipt.action_id {
                0 => vec![0.0, 0.0],
                1 => vec![0.2, 0.0],
                2 => vec![-0.2, 0.0],
                _ => unreachable!(),
            },
            observed_frame_sequence: 12,
            observation_delay_seconds: 1.0,
        }).collect::<Vec<_>>();
        let metrics = evaluate_action_response_set(&receipts, &outcomes, 0).unwrap();
        assert_eq!(metrics.action_count, 2);
        assert_eq!(metrics.cosine_scored_action_count, 2);
        assert!((metrics.mean_action_response_cosine - 1.0).abs() < 1e-6);
        assert!((metrics.mean_predicted_action_effect_norm - 0.2).abs() < 1e-6);
        assert!((metrics.mean_observed_action_effect_norm - 0.2).abs() < 1e-6);
        assert!(metrics.mean_action_effect_mse < 1e-12);
        assert_eq!(metrics.missed_action_effect_count, 0);
        assert_eq!(metrics.false_positive_action_effect_count, 0);
    }

    #[test]
    fn action_response_evaluator_rejects_wrong_zero_action_and_mutated_receipt() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 0,
                action_descriptor_sha256: digest('f'),
                action_embedding: vec![0.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 1,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 20, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let outcomes = receipts.iter().map(|receipt| ActionConditionedObservedOutcome {
            forecast_id: receipt.forecast_id,
            actual_future_hv: vec![0.0, 0.0],
            observed_frame_sequence: 21,
            observation_delay_seconds: 1.0,
        }).collect::<Vec<_>>();
        assert_eq!(
            evaluate_action_response_set(&receipts, &outcomes, 99),
            Err(ActionConditionedPredictorError::ActionSetMismatch)
        );
        let mut tampered = receipts.clone();
        tampered[1].predicted_future_hv[0] += 0.1;
        assert_eq!(
            evaluate_action_response_set(&tampered, &outcomes, 0),
            Err(ActionConditionedPredictorError::ActionSetMismatch)
        );
    }

    #[test]
    fn canonical_action_set_digest_is_independent_of_candidate_input_order() {
        let mut first_model = predictor();
        let mut second_model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let reversed = vec![candidates[1].clone(), candidates[0].clone()];
        let first = first_model.issue_action_set(
            &digest('a'), 50, &digest('b'), &digest('e'), 0.25, &[0.2, -0.1], &candidates
        ).unwrap();
        let second = second_model.issue_action_set(
            &digest('a'), 50, &digest('b'), &digest('e'), 0.25, &[0.2, -0.1], &reversed
        ).unwrap();
        assert_eq!(first[0].action_set_sha256, second[0].action_set_sha256);
        assert_eq!(
            first.iter().map(|r| r.action_id).collect::<Vec<_>>(),
            second.iter().map(|r| r.action_id).collect::<Vec<_>>()
        );
        assert_eq!(
            first.iter().map(|r| r.forecast_sha256.clone()).collect::<Vec<_>>(),
            second.iter().map(|r| r.forecast_sha256.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn heldout_action_set_scoring_consumes_receipts_without_updating_the_predictor() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 1.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, -1.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 80, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let before = model.save_state();
        let scores = model.score_action_set(&[
            ActionConditionedObservedOutcome {
                forecast_id: receipts[0].forecast_id,
                actual_future_hv: vec![-0.5, -0.25],
                observed_frame_sequence: 81,
                observation_delay_seconds: 1.0,
            },
            ActionConditionedObservedOutcome {
                forecast_id: receipts[1].forecast_id,
                actual_future_hv: vec![0.5, 0.25],
                observed_frame_sequence: 81,
                observation_delay_seconds: 1.0,
            },
        ]).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score| score.model_version_before == 0));
        assert!(scores.iter().all(|score| score.model_version_after == 0));
        assert!(scores.iter().all(|score| score.forecast_model_version == 0));
        assert!(scores.iter().all(|score| score.score_sha256 == hash_forecast_score(score)));
        let after = model.save_state();
        assert_eq!(after.action_weights, before.action_weights);
        assert_eq!(after.training_steps, before.training_steps);
        assert!(after.pending.is_empty());
    }

    #[test]
    fn action_set_scores_all_branches_at_one_version_then_updates_once() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 1.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, -1.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 100, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let scores = model.score_action_set_and_learn(&[
            ActionConditionedObservedOutcome {
                forecast_id: receipts[0].forecast_id,
                actual_future_hv: vec![-0.5, -0.25],
                observed_frame_sequence: 101,
                observation_delay_seconds: 1.0,
            },
            ActionConditionedObservedOutcome {
                forecast_id: receipts[1].forecast_id,
                actual_future_hv: vec![0.5, 0.25],
                observed_frame_sequence: 101,
                observation_delay_seconds: 1.0,
            },
        ]).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score| score.forecast_model_version == 0));
        assert!(scores.iter().all(|score| score.model_version_before == 0));
        assert!(scores.iter().all(|score| score.model_version_after == 1));
        assert!(scores.iter().all(|score| score.history_sha256 == digest('a')));
        assert!(scores.iter().all(|score| score.action_manifest_sha256 == digest('b')));
        assert!(scores.iter().all(|score| score.score_sha256 == hash_forecast_score(score)));
        assert_eq!(model.training_steps(), 1);
        assert_eq!(model.pending_forecast_count(), 0);
    }

    #[test]
    fn action_set_batch_rejects_missing_duplicate_or_mixed_history_without_partial_update() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 100, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let duplicate = [
            ActionConditionedObservedOutcome {
                forecast_id: receipts[0].forecast_id,
                actual_future_hv: vec![0.1, 0.0],
                observed_frame_sequence: 101,
                observation_delay_seconds: 1.0,
            },
            ActionConditionedObservedOutcome {
                forecast_id: receipts[0].forecast_id,
                actual_future_hv: vec![0.1, 0.0],
                observed_frame_sequence: 101,
                observation_delay_seconds: 1.0,
            },
        ];
        assert_eq!(
            model.score_action_set_and_learn(&duplicate),
            Err(ActionConditionedPredictorError::ActionSetMismatch)
        );
        assert_eq!(model.training_steps(), 0);
        assert_eq!(model.pending_forecast_count(), 2);
        let incomplete = [duplicate[0].clone()];
        assert_eq!(
            model.score_action_set_and_learn(&incomplete),
            Err(ActionConditionedPredictorError::ActionSetMismatch)
        );
        assert_eq!(model.pending_forecast_count(), 2);
        let incomplete_distinct = [duplicate[0].clone(), ActionConditionedObservedOutcome {
            forecast_id: receipts[1].forecast_id,
            actual_future_hv: vec![0.1, 0.0],
            observed_frame_sequence: 101,
            observation_delay_seconds: 1.0,
        }];
        assert_eq!(
            model.score_action_set_and_learn(&incomplete_distinct),
            Err(ActionConditionedPredictorError::ActionSetMismatch)
        );
        assert_eq!(model.training_steps(), 0);
        assert_eq!(model.pending_forecast_count(), 2);
    }

    #[test]
    fn action_residual_learns_from_later_observations_not_its_own_prediction() {
        let mut model = predictor();
        let base = [0.0, 0.0];
        let action_positive = [1.0, 1.0];
        let action_negative = [-1.0, -1.0];
        let positive = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'), 17, 1.0, &base, &action_positive
        ).unwrap();
        let negative = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('d'), &digest('e'), 3, 1.0, &base, &action_negative
        ).unwrap();
        let positive_score = model.score_and_learn(positive.forecast_id, &[0.8, 0.6], 11, 1.0).unwrap();
        let negative_score = model.score_and_learn(negative.forecast_id, &[-0.8, -0.6], 11, 1.0).unwrap();
        assert!(positive_score.mean_squared_error > 0.0);
        assert!(is_lower_hex_sha256(&positive_score.observed_future_hv_sha256));
        assert!(is_lower_hex_sha256(&positive_score.score_sha256));
        assert_eq!(
            positive_score.observed_future_hv_sha256,
            hash_f32_vector(b"vis005-observed-future-visual-hv-v1", &[0.8, 0.6])
        );
        assert_eq!(positive_score.model_version_before, 0);
        assert_eq!(negative_score.model_version_before, 1);
        assert_eq!(positive_score.score_sha256, hash_forecast_score(&positive_score));
        assert_eq!(positive_score.forecast_sha256, positive.forecast_sha256);
        assert_eq!(positive_score.model_checkpoint_sha256, digest('e'));
        assert_eq!(model.training_steps(), 2);

        let after_positive = model.issue_forecast(
            &digest('a'), 11, &digest('b'), &digest('c'), &digest('e'), 17, 1.0, &base, &action_positive
        ).unwrap();
        let after_negative = model.issue_forecast(
            &digest('a'), 11, &digest('b'), &digest('d'), &digest('e'), 3, 1.0, &base, &action_negative
        ).unwrap();
        assert!(after_positive.predicted_future_hv[0] > 0.0);
        assert!(after_negative.predicted_future_hv[0] < 0.0);
    }



    #[test]
    fn stale_single_forecast_can_be_scored_but_cannot_update_newer_weights() {
        let mut model = predictor();
        let first = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'),
            17, 1.0, &[0.0, 0.0], &[1.0, 1.0]
        ).unwrap();
        let second = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('d'), &digest('e'),
            3, 1.0, &[0.0, 0.0], &[-1.0, -1.0]
        ).unwrap();
        model.score_and_learn(first.forecast_id, &[0.5, 0.5], 11, 1.0).unwrap();
        assert_eq!(
            model.score_and_learn(second.forecast_id, &[-0.5, -0.5], 11, 1.0),
            Err(ActionConditionedPredictorError::StaleForecastModelVersion)
        );
        let score = model.score_action_set(&[ActionConditionedObservedOutcome {
            forecast_id: second.forecast_id,
            actual_future_hv: vec![-0.5, -0.5],
            observed_frame_sequence: 11,
            observation_delay_seconds: 1.0,
        }]).unwrap();
        assert_eq!(score.len(), 1);
        assert_eq!(score[0].forecast_model_version, 0);
        assert_eq!(score[0].model_version_before, 1);
        assert_eq!(score[0].model_version_after, 1);
        assert_eq!(model.training_steps(), 1);
    }

    #[test]
    fn incomplete_action_set_checkpoint_restore_is_rejected() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        model.issue_action_set(
            &digest('a'), 100, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let mut state = model.save_state();
        state.pending.pop();
        let mut restored = predictor();
        assert_eq!(
            restored.load_state(&state),
            Err(ActionConditionedPredictorError::InvalidCheckpointState)
        );
        assert_eq!(restored.pending_forecast_count(), 0);
    }

    #[test]
    fn one_late_counterfactual_branch_expires_the_complete_action_set_without_learning() {
        let mut model = predictor();
        let candidates = vec![
            ActionConditionedCandidateAction {
                action_id: 17,
                action_descriptor_sha256: digest('c'),
                action_embedding: vec![1.0, 0.0],
            },
            ActionConditionedCandidateAction {
                action_id: 3,
                action_descriptor_sha256: digest('d'),
                action_embedding: vec![-1.0, 0.0],
            },
        ];
        let receipts = model.issue_action_set(
            &digest('a'), 100, &digest('b'), &digest('e'), 1.0, &[0.0, 0.0], &candidates
        ).unwrap();
        let before = model.save_state();
        assert_eq!(
            model.score_action_set(&[
                ActionConditionedObservedOutcome {
                    forecast_id: receipts[0].forecast_id,
                    actual_future_hv: vec![0.1, 0.0],
                    observed_frame_sequence: 101,
                    observation_delay_seconds: 1.0,
                },
                ActionConditionedObservedOutcome {
                    forecast_id: receipts[1].forecast_id,
                    actual_future_hv: vec![0.1, 0.0],
                    observed_frame_sequence: 101,
                    observation_delay_seconds: 2.0,
                },
            ]),
            Err(ActionConditionedPredictorError::ForecastTooLate)
        );
        assert_eq!(model.pending_forecast_count(), 0);
        assert_eq!(model.expired_forecasts(), 2);
        assert_eq!(model.training_steps(), before.training_steps);
        assert_eq!(model.save_state().action_weights, before.action_weights);
    }

    #[test]
    fn forecasts_are_one_shot_and_observations_must_be_later_and_on_time() {
        let mut model = predictor();
        let receipt = model.issue_forecast(
            &digest('a'), 10, &digest('b'), &digest('c'), &digest('e'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]
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
            model.issue_forecast("bad", 0, &digest('b'), &digest('c'), &digest('e'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]),
            Err(ActionConditionedPredictorError::InvalidDigest)
        );
        assert_eq!(
            model.issue_forecast(&digest('a'), 0, &digest('b'), &digest('c'), &digest('e'), 1, 0.0, &[0.0, 0.0], &[1.0, 1.0]),
            Err(ActionConditionedPredictorError::InvalidHorizon)
        );
        let receipt = model.issue_forecast(
            &digest('a'), 0, &digest('b'), &digest('c'), &digest('e'), 1, 1.0, &[0.0, 0.0], &[1.0, 1.0]
        ).unwrap();
        let mut state = model.save_state();
        state.pending[0].action_embedding[0] += 0.25;
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
            &digest('a'), 20, &digest('b'), &digest('c'), &digest('e'), 4, 0.25, &[0.1, 0.2], &[1.0, 0.5]
        ).unwrap();
        let state = model.save_state();
        let encoded = serde_json::to_vec(&state).unwrap();
        let decoded: ActionConditionedPredictorState = serde_json::from_slice(&encoded).unwrap();
        let mut restored = predictor();
        restored.load_state(&decoded).unwrap();
        assert_eq!(restored.save_state(), state);
        let score = restored.score_and_learn(receipt.forecast_id, &[0.2, 0.3], 21, 0.25).unwrap();
        assert_eq!(score.forecast_id, receipt.forecast_id);
        assert_eq!(restored.pending_forecast_count(), 0);
    }
}
