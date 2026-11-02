//! Action-conditioned visual forecast contracts and exact-target scoring.
//!
//! This module deliberately does not implement a visual dynamics model. It
//! defines the producer/evaluator seam required to compare hypothetical
//! actions from one immutable observed belief and to score each branch only
//! against the exact later observation that belongs to that branch.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ACTION_CONDITIONED_FORECAST_SCHEMA_VERSION: u32 = 1;

/// The declared source of pairing across action-conditioned forecast draws.
///
/// This declaration is metadata, not proof that a producer actually used the
/// same latent/noise draw. A future adapter receipt must retain and verify the
/// source draw tape or model-side sample provenance independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionDrawCoupling {
    SharedPosteriorAndWorldNoise,
    PairedExogenousDisturbance,
    DeterministicSharedLatent,
}

/// One candidate action in the frozen, canonical action manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionDescriptor {
    pub action_id: u64,
    /// Canonical serialized action payload, including units and action-space version.
    pub canonical_descriptor: String,
}

/// Read-only evidence the predictor is allowed to consume.
///
/// There is intentionally no simulator truth, unchosen outcome, future frame,
/// evaluator label, or branch result in this type.
#[derive(Clone, Copy, Debug)]
pub struct ObservedVisualBelief<'a> {
    pub history_sha256: &'a str,
    pub frame_sequence: u64,
    pub captured_at_seconds: f64,
    pub belief: &'a [f32],
}

/// The complete request for one same-history action-set forecast.
#[derive(Clone, Debug)]
pub struct ActionForecastRequest<'a> {
    pub observed: ObservedVisualBelief<'a>,
    pub action_manifest_sha256: &'a str,
    pub joint_draw_manifest_sha256: &'a str,
    pub model_sha256: &'a str,
    pub horizon_seconds: f64,
    pub target_frame_sequence: u64,
    pub target_time_seconds: f64,
    pub actions: &'a [ActionDescriptor],
    pub joint_draw_ids: &'a [u64],
    pub coupling: ActionDrawCoupling,
}

impl ActionForecastRequest<'_> {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("history_sha256", self.observed.history_sha256)?;
        validate_sha256("action_manifest_sha256", self.action_manifest_sha256)?;
        validate_sha256("joint_draw_manifest_sha256", self.joint_draw_manifest_sha256)?;
        validate_sha256("model_sha256", self.model_sha256)?;
        validate_vector("observed belief", self.observed.belief)?;
        if !self.observed.captured_at_seconds.is_finite()
            || self.observed.captured_at_seconds < 0.0
        {
            return Err("observed capture time must be finite and non-negative".into());
        }
        if !self.horizon_seconds.is_finite() || self.horizon_seconds <= 0.0 {
            return Err("forecast horizon must be finite and > 0".into());
        }
        if self.target_frame_sequence <= self.observed.frame_sequence {
            return Err("target frame sequence must follow the observed history".into());
        }
        if !self.target_time_seconds.is_finite()
            || self.target_time_seconds <= self.observed.captured_at_seconds
        {
            return Err("target time must be finite and follow the observed history".into());
        }
        let elapsed = self.target_time_seconds - self.observed.captured_at_seconds;
        let tolerance = 1e-6_f64.max(self.horizon_seconds.abs() * 1e-6);
        if (elapsed - self.horizon_seconds).abs() > tolerance {
            return Err("target time and forecast horizon disagree".into());
        }
        if self.actions.len() < 2 {
            return Err("an action set requires at least two candidate actions".into());
        }
        let mut action_ids = HashSet::with_capacity(self.actions.len());
        for action in self.actions {
            if action.canonical_descriptor.trim().is_empty() {
                return Err(format!("action {} has an empty canonical descriptor", action.action_id));
            }
            if !action_ids.insert(action.action_id) {
                return Err(format!("duplicate action ID {}", action.action_id));
            }
        }
        let action_digest = compute_action_manifest_sha256(self.actions)?;
        if action_digest != self.action_manifest_sha256 {
            return Err("action manifest digest does not match canonical action descriptors".into());
        }
        if self.joint_draw_ids.len() < 2 {
            return Err("a joint action bank requires at least two joint draws".into());
        }
        let mut draw_ids = HashSet::with_capacity(self.joint_draw_ids.len());
        for draw_id in self.joint_draw_ids {
            if !draw_ids.insert(*draw_id) {
                return Err(format!("duplicate joint draw ID {draw_id}"));
            }
        }
        let draw_digest = compute_joint_draw_manifest_sha256(self.joint_draw_ids, self.coupling)?;
        if draw_digest != self.joint_draw_manifest_sha256 {
            return Err("joint draw manifest digest does not match draw IDs and coupling".into());
        }
        Ok(())
    }
}

/// One predicted visual belief for one action and one joint draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionForecastSample {
    pub action_id: u64,
    pub joint_draw_id: u64,
    /// Unique simulator/model branch identity for this action/draw pair.
    pub branch_id: String,
    /// Predicted visual belief; never an acquired observation.
    pub predicted_belief: Vec<f32>,
}

/// Output of the predictor for a single immutable history and action set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionForecastBank {
    pub schema_version: u32,
    pub history_sha256: String,
    pub action_manifest_sha256: String,
    pub joint_draw_manifest_sha256: String,
    pub model_sha256: String,
    pub bank_sha256: String,
    pub source_frame_sequence: u64,
    pub issued_at_seconds: f64,
    pub horizon_seconds: f64,
    pub target_frame_sequence: u64,
    pub target_time_seconds: f64,
    pub action_ids: Vec<u64>,
    pub joint_draw_ids: Vec<u64>,
    pub coupling: ActionDrawCoupling,
    /// Persistence baseline copied from the observed starting belief.
    pub persistence_belief: Vec<f32>,
    /// Complete Cartesian product: one forecast per action and joint draw.
    pub samples: Vec<ActionForecastSample>,
}

impl ActionForecastBank {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ACTION_CONDITIONED_FORECAST_SCHEMA_VERSION {
            return Err(format!(
                "unsupported action forecast schema version {}",
                self.schema_version
            ));
        }
        validate_sha256("history_sha256", &self.history_sha256)?;
        validate_sha256("action_manifest_sha256", &self.action_manifest_sha256)?;
        validate_sha256("joint_draw_manifest_sha256", &self.joint_draw_manifest_sha256)?;
        validate_sha256("model_sha256", &self.model_sha256)?;
        validate_sha256("bank_sha256", &self.bank_sha256)?;
        validate_vector("persistence belief", &self.persistence_belief)?;
        if !self.issued_at_seconds.is_finite() || self.issued_at_seconds < 0.0 {
            return Err("issued_at_seconds must be finite and non-negative".into());
        }
        if !self.horizon_seconds.is_finite() || self.horizon_seconds <= 0.0 {
            return Err("horizon_seconds must be finite and > 0".into());
        }
        if !self.target_time_seconds.is_finite() || self.target_time_seconds <= self.issued_at_seconds {
            return Err("target_time_seconds must follow issued_at_seconds".into());
        }
        let elapsed = self.target_time_seconds - self.issued_at_seconds;
        let tolerance = 1e-6_f64.max(self.horizon_seconds.abs() * 1e-6);
        if (elapsed - self.horizon_seconds).abs() > tolerance {
            return Err("target time and forecast horizon disagree".into());
        }
        if self.target_frame_sequence <= self.source_frame_sequence {
            return Err("target frame sequence must follow source frame sequence".into());
        }
        if self.action_ids.len() < 2 || self.joint_draw_ids.len() < 2 {
            return Err("action forecast bank requires at least two actions and two draws".into());
        }
        let mut actions = HashSet::with_capacity(self.action_ids.len());
        for action_id in &self.action_ids {
            if !actions.insert(*action_id) {
                return Err(format!("duplicate action ID {action_id}"));
            }
        }
        let mut draws = HashSet::with_capacity(self.joint_draw_ids.len());
        for draw_id in &self.joint_draw_ids {
            if !draws.insert(*draw_id) {
                return Err(format!("duplicate joint draw ID {draw_id}"));
            }
        }
        let expected = self.action_ids.len()
            .checked_mul(self.joint_draw_ids.len())
            .ok_or_else(|| "action/draw cardinality overflow".to_string())?;
        if self.samples.len() != expected {
            return Err(format!(
                "incomplete action/draw matrix: got {}, expected {expected}",
                self.samples.len()
            ));
        }
        let mut pairs = HashSet::with_capacity(expected);
        let mut branches = HashSet::with_capacity(expected);
        for sample in &self.samples {
            if !actions.contains(&sample.action_id) {
                return Err(format!("sample has unknown action ID {}", sample.action_id));
            }
            if !draws.contains(&sample.joint_draw_id) {
                return Err(format!("sample has unknown joint draw ID {}", sample.joint_draw_id));
            }
            if !pairs.insert((sample.action_id, sample.joint_draw_id)) {
                return Err(format!(
                    "duplicate action/draw pair ({}, {})",
                    sample.action_id, sample.joint_draw_id
                ));
            }
            if sample.branch_id.trim().is_empty() || !branches.insert(sample.branch_id.as_str()) {
                return Err("branch IDs must be non-empty and unique within a bank".into());
            }
            if sample.predicted_belief.len() != self.persistence_belief.len() {
                return Err("predicted/persistence belief dimension mismatch".into());
            }
            validate_vector("predicted belief", &sample.predicted_belief)?;
        }
        let expected_draw_digest =
            compute_joint_draw_manifest_sha256(&self.joint_draw_ids, self.coupling)?;
        if expected_draw_digest != self.joint_draw_manifest_sha256 {
            return Err("joint draw manifest digest mismatch".into());
        }
        if compute_action_forecast_bank_sha256(self) != self.bank_sha256 {
            return Err("action forecast bank content digest mismatch".into());
        }
        Ok(())
    }

    /// Check that a producer returned a bank for exactly the request it received.
    ///
    /// The action manifest is recomputed from the request descriptors. The
    /// joint-draw manifest is recomputed from sorted draw IDs and coupling.
    /// The predictor bank digest is recomputed over the exact sample bytes.
    pub fn validate_for_request(&self, request: &ActionForecastRequest<'_>) -> Result<(), String> {
        request.validate()?;
        self.validate()?;
        if self.history_sha256 != request.observed.history_sha256
            || self.source_frame_sequence != request.observed.frame_sequence
            || self.issued_at_seconds.to_bits() != request.observed.captured_at_seconds.to_bits()
        {
            return Err("producer changed the observed-history identity".into());
        }
        if self.action_manifest_sha256 != request.action_manifest_sha256 {
            return Err("producer changed the action-manifest identity".into());
        }
        if self.joint_draw_manifest_sha256 != request.joint_draw_manifest_sha256 {
            return Err("producer changed the joint-draw manifest identity".into());
        }
        if self.model_sha256 != request.model_sha256 {
            return Err("producer changed the model/checkpoint identity".into());
        }
        if self.horizon_seconds.to_bits() != request.horizon_seconds.to_bits()
            || self.target_frame_sequence != request.target_frame_sequence
            || self.target_time_seconds.to_bits() != request.target_time_seconds.to_bits()
        {
            return Err("producer changed the requested forecast target".into());
        }
        if self.coupling != request.coupling {
            return Err("producer changed the declared draw coupling".into());
        }
        let requested_actions: HashSet<u64> = request.actions.iter().map(|item| item.action_id).collect();
        let returned_actions: HashSet<u64> = self.action_ids.iter().copied().collect();
        if returned_actions != requested_actions {
            return Err("producer changed candidate action identities".into());
        }
        let requested_draws: HashSet<u64> = request.joint_draw_ids.iter().copied().collect();
        let returned_draws: HashSet<u64> = self.joint_draw_ids.iter().copied().collect();
        if returned_draws != requested_draws {
            return Err("producer changed joint draw identities".into());
        }
        if self.persistence_belief.len() != request.observed.belief.len()
            || self
                .persistence_belief
                .iter()
                .zip(request.observed.belief)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            return Err("producer changed the observed persistence baseline".into());
        }
        Ok(())
    }
}

/// Producer seam for future CfC/vision or learned action-conditioned predictors.
///
/// Implementations receive the observed belief, canonical candidate actions,
/// horizon and joint draw IDs only. They must not query evaluator truth or
/// mutate observation memory while constructing hypothetical futures.
pub trait ActionConditionedVisualPredictor {
    fn predict_joint_action_set(
        &self,
        request: &ActionForecastRequest<'_>,
    ) -> Result<ActionForecastBank, String>;

    fn predict_validated(
        &self,
        request: &ActionForecastRequest<'_>,
    ) -> Result<ActionForecastBank, String> {
        let bank = self.predict_joint_action_set(request)?;
        bank.validate_for_request(request)?;
        Ok(bank)
    }
}

/// The evaluator-side observation for one exact simulator branch.
#[derive(Clone, Copy, Debug)]
pub struct BranchVisualObservation<'a> {
    pub history_sha256: &'a str,
    pub action_manifest_sha256: &'a str,
    pub action_id: u64,
    pub joint_draw_id: u64,
    pub branch_id: &'a str,
    pub frame_sequence: u64,
    pub captured_at_seconds: f64,
    pub observed_belief: &'a [f32],
}

/// Why a pending forecast could not be scored at its exact target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastExpiryReason {
    TargetFrameSkipped,
    ObservationArrivedTooLate,
}

/// Score of one forecast against its exact later branch observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionVisualForecastScore {
    pub history_sha256: String,
    pub action_manifest_sha256: String,
    pub model_sha256: String,
    pub action_id: u64,
    pub joint_draw_id: u64,
    pub branch_id: String,
    pub source_frame_sequence: u64,
    pub target_frame_sequence: u64,
    pub issued_at_seconds: f64,
    pub horizon_seconds: f64,
    pub timing_error_seconds: f64,
    pub prediction_error: f64,
    pub persistence_error: f64,
    /// (persistence error - model error) / persistence error.
    /// None means persistence had zero error, so relative skill is undefined.
    pub relative_skill: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "score", rename_all = "snake_case")]
pub enum ForecastObservationResult {
    Unmatched,
    NotDue,
    Scored(ActionVisualForecastScore),
    Expired(ForecastExpiryReason),
}

#[derive(Clone, Debug)]
struct PendingForecast {
    history_sha256: String,
    action_manifest_sha256: String,
    model_sha256: String,
    source_frame_sequence: u64,
    issued_at_seconds: f64,
    horizon_seconds: f64,
    target_frame_sequence: u64,
    target_time_seconds: f64,
    action_id: u64,
    joint_draw_id: u64,
    branch_id: String,
    predicted_belief: Vec<f32>,
    persistence_belief: Vec<f32>,
}

/// Bounded exact-target scorer for action-conditioned future visual beliefs.
///
/// This is deliberately separate from the existing autonomous
/// DelayedHorizonEvaluator. A forecast is scored only against the named
/// branch at the requested frame sequence and within the configured timestamp
/// tolerance; a later frame is never silently substituted.
#[derive(Clone, Debug)]
pub struct ActionConditionedForecastEvaluator {
    max_pending: usize,
    max_timestamp_error_seconds: f64,
    pending: Vec<PendingForecast>,
    matured_count: u64,
    expired_count: u64,
}

impl ActionConditionedForecastEvaluator {
    pub fn new(max_pending: usize, max_timestamp_error_seconds: f64) -> Result<Self, String> {
        if max_pending == 0 {
            return Err("max_pending must be > 0".into());
        }
        if !max_timestamp_error_seconds.is_finite() || max_timestamp_error_seconds < 0.0 {
            return Err("max_timestamp_error_seconds must be finite and non-negative".into());
        }
        Ok(Self {
            max_pending,
            max_timestamp_error_seconds,
            pending: Vec::new(),
            matured_count: 0,
            expired_count: 0,
        })
    }

    /// Atomically admit a complete action × joint-draw bank.
    pub fn issue(&mut self, bank: &ActionForecastBank) -> Result<(), String> {
        bank.validate()?;
        if self.pending.len().saturating_add(bank.samples.len()) > self.max_pending {
            return Err("pending action forecast capacity would be exceeded".into());
        }
        let existing: HashSet<&str> = self.pending.iter().map(|item| item.branch_id.as_str()).collect();
        if bank.samples.iter().any(|sample| existing.contains(sample.branch_id.as_str())) {
            return Err("branch ID is already pending in the evaluator".into());
        }
        let mut staged = Vec::with_capacity(bank.samples.len());
        for sample in &bank.samples {
            staged.push(PendingForecast {
                history_sha256: bank.history_sha256.clone(),
                action_manifest_sha256: bank.action_manifest_sha256.clone(),
                model_sha256: bank.model_sha256.clone(),
                source_frame_sequence: bank.source_frame_sequence,
                issued_at_seconds: bank.issued_at_seconds,
                horizon_seconds: bank.horizon_seconds,
                target_frame_sequence: bank.target_frame_sequence,
                target_time_seconds: bank.target_time_seconds,
                action_id: sample.action_id,
                joint_draw_id: sample.joint_draw_id,
                branch_id: sample.branch_id.clone(),
                predicted_belief: sample.predicted_belief.clone(),
                persistence_belief: bank.persistence_belief.clone(),
            });
        }
        self.pending.extend(staged);
        Ok(())
    }

    /// Supply one real observation from one named branch.
    ///
    /// A frame before the requested target leaves the forecast pending. A
    /// skipped target frame or too-late target observation expires the
    /// forecast rather than scoring against an unrelated later frame.
    pub fn observe_branch(
        &mut self,
        observation: &BranchVisualObservation<'_>,
    ) -> Result<ForecastObservationResult, String> {
        validate_sha256("history_sha256", observation.history_sha256)?;
        validate_sha256("action_manifest_sha256", observation.action_manifest_sha256)?;
        validate_vector("observed branch belief", observation.observed_belief)?;
        if observation.branch_id.trim().is_empty() {
            return Err("branch observation requires a non-empty branch ID".into());
        }
        if !observation.captured_at_seconds.is_finite() || observation.captured_at_seconds < 0.0 {
            return Err("branch observation timestamp must be finite and non-negative".into());
        }

        let Some(index) = self
            .pending
            .iter()
            .position(|item| item.branch_id == observation.branch_id)
        else {
            return Ok(ForecastObservationResult::Unmatched);
        };
        let pending = &self.pending[index];
        if pending.history_sha256 != observation.history_sha256
            || pending.action_manifest_sha256 != observation.action_manifest_sha256
            || pending.action_id != observation.action_id
            || pending.joint_draw_id != observation.joint_draw_id
        {
            return Err("branch observation identity does not match its pending forecast".into());
        }
        if observation.observed_belief.len() != pending.predicted_belief.len() {
            return Err("observed branch belief dimension mismatch".into());
        }
        if observation.frame_sequence < pending.target_frame_sequence {
            return Ok(ForecastObservationResult::NotDue);
        }
        if observation.frame_sequence > pending.target_frame_sequence {
            self.pending.swap_remove(index);
            self.expired_count = self.expired_count.saturating_add(1);
            return Ok(ForecastObservationResult::Expired(
                ForecastExpiryReason::TargetFrameSkipped,
            ));
        }
        let timing_error = observation.captured_at_seconds - pending.target_time_seconds;
        if timing_error.abs() > self.max_timestamp_error_seconds {
            if timing_error > 0.0 {
                self.pending.swap_remove(index);
                self.expired_count = self.expired_count.saturating_add(1);
                return Ok(ForecastObservationResult::Expired(
                    ForecastExpiryReason::ObservationArrivedTooLate,
                ));
            }
            return Err("target frame timestamp precedes the requested target time".into());
        }

        let prediction_error = cosine_error(&pending.predicted_belief, observation.observed_belief)?;
        let persistence_error = cosine_error(&pending.persistence_belief, observation.observed_belief)?;
        let relative_skill = (persistence_error > f64::EPSILON)
            .then_some((persistence_error - prediction_error) / persistence_error);
        let score = ActionVisualForecastScore {
            history_sha256: pending.history_sha256.clone(),
            action_manifest_sha256: pending.action_manifest_sha256.clone(),
            model_sha256: pending.model_sha256.clone(),
            action_id: pending.action_id,
            joint_draw_id: pending.joint_draw_id,
            branch_id: pending.branch_id.clone(),
            source_frame_sequence: pending.source_frame_sequence,
            target_frame_sequence: pending.target_frame_sequence,
            issued_at_seconds: pending.issued_at_seconds,
            horizon_seconds: pending.horizon_seconds,
            timing_error_seconds: timing_error,
            prediction_error,
            persistence_error,
            relative_skill,
        };
        self.pending.swap_remove(index);
        self.matured_count = self.matured_count.saturating_add(1);
        Ok(ForecastObservationResult::Scored(score))
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn matured_count(&self) -> u64 {
        self.matured_count
    }

    pub fn expired_count(&self) -> u64 {
        self.expired_count
    }
}

fn append_len_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn append_u64s(bytes: &mut Vec<u8>, values: &[u64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn coupling_tag(coupling: ActionDrawCoupling) -> u8 {
    match coupling {
        ActionDrawCoupling::SharedPosteriorAndWorldNoise => 0,
        ActionDrawCoupling::PairedExogenousDisturbance => 1,
        ActionDrawCoupling::DeterministicSharedLatent => 2,
    }
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Content hash of action IDs and canonical descriptors, independent of row order.
pub fn compute_action_manifest_sha256(actions: &[ActionDescriptor]) -> Result<String, String> {
    if actions.len() < 2 {
        return Err("an action manifest requires at least two actions".into());
    }
    let mut ordered: Vec<&ActionDescriptor> = actions.iter().collect();
    ordered.sort_by_key(|action| action.action_id);
    let mut bytes = b"cmaq-018-vision-action-manifest-v0.1\0".to_vec();
    bytes.extend_from_slice(&(ordered.len() as u64).to_le_bytes());
    for action in ordered {
        if action.canonical_descriptor.trim().is_empty() {
            return Err(format!("action {} has an empty canonical descriptor", action.action_id));
        }
        bytes.extend_from_slice(&action.action_id.to_le_bytes());
        append_len_prefixed(&mut bytes, action.canonical_descriptor.as_bytes());
    }
    Ok(digest_hex(&bytes))
}

/// Content hash of the common draw identities and the declared coupling policy.
pub fn compute_joint_draw_manifest_sha256(
    draw_ids: &[u64],
    coupling: ActionDrawCoupling,
) -> Result<String, String> {
    if draw_ids.len() < 2 {
        return Err("a joint draw manifest requires at least two draw IDs".into());
    }
    let mut ordered = draw_ids.to_vec();
    ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("joint draw manifest contains duplicate IDs".into());
    }
    let mut bytes = b"cmaq-018-vision-joint-draw-manifest-v0.1\0".to_vec();
    bytes.push(coupling_tag(coupling));
    append_u64s(&mut bytes, &ordered);
    Ok(digest_hex(&bytes))
}

/// Canonical content hash of the exact action-conditioned predicted belief bank.
///
/// Action/draw identities and samples are sorted by their semantic keys, so a
/// harmless row permutation does not change the digest. Exact f32 bit patterns,
/// branch IDs, baseline belief and all provenance identities are bound.
pub fn compute_action_forecast_bank_sha256(bank: &ActionForecastBank) -> String {
    let mut bytes = b"cmaq-018-vision-action-forecast-bank-v0.1\0".to_vec();
    append_len_prefixed(&mut bytes, bank.history_sha256.as_bytes());
    append_len_prefixed(&mut bytes, bank.action_manifest_sha256.as_bytes());
    append_len_prefixed(&mut bytes, bank.joint_draw_manifest_sha256.as_bytes());
    append_len_prefixed(&mut bytes, bank.model_sha256.as_bytes());
    bytes.extend_from_slice(&bank.source_frame_sequence.to_le_bytes());
    bytes.extend_from_slice(&bank.issued_at_seconds.to_bits().to_le_bytes());
    bytes.extend_from_slice(&bank.horizon_seconds.to_bits().to_le_bytes());
    bytes.extend_from_slice(&bank.target_frame_sequence.to_le_bytes());
    bytes.extend_from_slice(&bank.target_time_seconds.to_bits().to_le_bytes());
    bytes.push(coupling_tag(bank.coupling));
    let mut action_ids = bank.action_ids.clone();
    action_ids.sort_unstable();
    append_u64s(&mut bytes, &action_ids);
    let mut draw_ids = bank.joint_draw_ids.clone();
    draw_ids.sort_unstable();
    append_u64s(&mut bytes, &draw_ids);
    bytes.extend_from_slice(&(bank.persistence_belief.len() as u64).to_le_bytes());
    for value in &bank.persistence_belief {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let mut samples: Vec<&ActionForecastSample> = bank.samples.iter().collect();
    samples.sort_by_key(|sample| (sample.action_id, sample.joint_draw_id));
    bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.action_id.to_le_bytes());
        bytes.extend_from_slice(&sample.joint_draw_id.to_le_bytes());
        append_len_prefixed(&mut bytes, sample.branch_id.as_bytes());
        bytes.extend_from_slice(&(sample.predicted_belief.len() as u64).to_le_bytes());
        for value in &sample.predicted_belief {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    digest_hex(&bytes)
}

fn validate_sha256(field: &str, digest: &str) -> Result<(), String> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn validate_vector(field: &str, values: &[f32]) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{field} contains non-finite values"));
    }
    if values.iter().all(|value| *value == 0.0) {
        return Err(format!("{field} must have non-zero magnitude"));
    }
    Ok(())
}

fn cosine_error(left: &[f32], right: &[f32]) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err("cosine comparison dimension mismatch".into());
    }
    validate_vector("left cosine vector", left)?;
    validate_vector("right cosine vector", right)?;
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let left = *left as f64;
        let right = *right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let similarity = (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0);
    Ok(1.0 - similarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTORY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ACTION_MANIFEST_SHA: &str = "8cd89f9287f4cfe2f5b69b6e35fa8b33f74150f6f4850271ab283a95418b48aa";
    const JOINT_DRAW_MANIFEST_SHA: &str = "f0fd63f7e8fbc17b1c19e93e7437488528f5db2d254507e93fac1595b7637616";

    fn request() -> ActionForecastRequest<'static> {
        static BELIEF: [f32; 3] = [1.0, 0.0, 0.0];
        static DRAWS: [u64; 2] = [100, 101];
        let actions = Box::leak(Box::new([
            ActionDescriptor { action_id: 17, canonical_descriptor: "move:left".into() },
            ActionDescriptor { action_id: 9, canonical_descriptor: "move:right".into() },
        ]));
        ActionForecastRequest {
            observed: ObservedVisualBelief {
                history_sha256: HISTORY,
                frame_sequence: 40,
                captured_at_seconds: 1.0,
                belief: &BELIEF,
            },
            action_manifest_sha256: ACTION_MANIFEST_SHA,
            joint_draw_manifest_sha256: JOINT_DRAW_MANIFEST_SHA,
            model_sha256: MODEL,
            horizon_seconds: 0.5,
            target_frame_sequence: 45,
            target_time_seconds: 1.5,
            actions,
            joint_draw_ids: &DRAWS,
            coupling: ActionDrawCoupling::PairedExogenousDisturbance,
        }
    }

    fn bank() -> ActionForecastBank {
        let mut bank = ActionForecastBank {
            schema_version: ACTION_CONDITIONED_FORECAST_SCHEMA_VERSION,
            history_sha256: HISTORY.into(),
            action_manifest_sha256: compute_action_manifest_sha256(&[
                ActionDescriptor { action_id: 17, canonical_descriptor: "move:left".into() },
                ActionDescriptor { action_id: 9, canonical_descriptor: "move:right".into() },
            ]).unwrap(),
            joint_draw_manifest_sha256: compute_joint_draw_manifest_sha256(
                &[100, 101], ActionDrawCoupling::PairedExogenousDisturbance
            ).unwrap(),
            model_sha256: MODEL.into(),
            bank_sha256: String::new(),
            source_frame_sequence: 40,
            issued_at_seconds: 1.0,
            horizon_seconds: 0.5,
            target_frame_sequence: 45,
            target_time_seconds: 1.5,
            action_ids: vec![17, 9],
            joint_draw_ids: vec![100, 101],
            coupling: ActionDrawCoupling::PairedExogenousDisturbance,
            persistence_belief: vec![1.0, 0.0, 0.0],
            samples: vec![
                ActionForecastSample { action_id: 17, joint_draw_id: 100, branch_id: "b17-100".into(), predicted_belief: vec![0.0, 1.0, 0.0] },
                ActionForecastSample { action_id: 17, joint_draw_id: 101, branch_id: "b17-101".into(), predicted_belief: vec![0.0, 1.0, 0.0] },
                ActionForecastSample { action_id: 9, joint_draw_id: 100, branch_id: "b9-100".into(), predicted_belief: vec![0.0, 1.0, 0.0] },
                ActionForecastSample { action_id: 9, joint_draw_id: 101, branch_id: "b9-101".into(), predicted_belief: vec![0.0, 1.0, 0.0] },
            ],
        };
        bank.bank_sha256 = compute_action_forecast_bank_sha256(&bank);
        bank
    }

    fn observation<'a>(
        branch_id: &'a str,
        action_id: u64,
        draw_id: u64,
        frame_sequence: u64,
        captured_at_seconds: f64,
        belief: &'a [f32],
    ) -> BranchVisualObservation<'a> {
        BranchVisualObservation {
            history_sha256: HISTORY,
            action_manifest_sha256: ACTION_MANIFEST_SHA,
            action_id,
            joint_draw_id: draw_id,
            branch_id,
            frame_sequence,
            captured_at_seconds,
            observed_belief: belief,
        }
    }

    #[test]
    fn action_forecast_request_and_complete_joint_bank_validate() {
        let request = request();
        request.validate().unwrap();
        let forecast_bank = bank();
        forecast_bank.validate_for_request(&request).unwrap();
    }

    #[test]
    fn request_rejects_bad_identity_timing_actions_and_draws() {
        let mut bad_horizon = request();
        assert!(bad_horizon.validate().is_ok());
        bad_horizon.horizon_seconds = f64::NAN;
        assert!(bad_horizon.validate().is_err());

        let mut bad_target = request();
        bad_target.target_time_seconds = 1.75;
        assert!(bad_target.validate().is_err());

        let mut bad_manifest = request();
        bad_manifest.action_manifest_sha256 = "not-a-digest";
        assert!(bad_manifest.validate().is_err());

        let valid = request();
        let bad_actions = Box::leak(Box::new([
            valid.actions[0].clone(),
            ActionDescriptor { action_id: 9, canonical_descriptor: String::new() },
        ]));
        let bad_action = ActionForecastRequest {
            actions: bad_actions,
            ..valid
        };
        assert!(bad_action.validate().is_err());
    }

    struct FixedPredictor;

    impl ActionConditionedVisualPredictor for FixedPredictor {
        fn predict_joint_action_set(
            &self,
            _request: &ActionForecastRequest<'_>,
        ) -> Result<ActionForecastBank, String> {
            Ok(bank())
        }
    }

    #[test]
    fn predictor_adapter_validates_returned_bank_against_immutable_request() {
        let request = request();
        let prediction = FixedPredictor.predict_validated(&request).unwrap();
        prediction.validate_for_request(&request).unwrap();

        let mut tampered = prediction;
        tampered.persistence_belief[0] = 0.5;
        assert!(tampered.validate_for_request(&request).is_err());

        let mut tampered = bank();
        tampered.action_manifest_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into();
        tampered.bank_sha256 = compute_action_forecast_bank_sha256(&tampered);
        assert!(tampered.validate_for_request(&request).is_err());
    }

    #[test]
    fn manifest_hashes_match_pinned_canonical_known_answers() {
        let actions = [
            ActionDescriptor { action_id: 17, canonical_descriptor: "move:left".into() },
            ActionDescriptor { action_id: 9, canonical_descriptor: "move:right".into() },
        ];
        assert_eq!(compute_action_manifest_sha256(&actions).unwrap(), ACTION_MANIFEST_SHA);
        assert_eq!(
            compute_joint_draw_manifest_sha256(
                &[100, 101], ActionDrawCoupling::PairedExogenousDisturbance
            ).unwrap(),
            JOINT_DRAW_MANIFEST_SHA
        );
    }

    #[test]
    fn action_and_draw_manifest_hashes_ignore_input_row_order() {
        let original = [
            ActionDescriptor { action_id: 17, canonical_descriptor: "move:left".into() },
            ActionDescriptor { action_id: 9, canonical_descriptor: "move:right".into() },
        ];
        let reversed = [
            original[1].clone(),
            original[0].clone(),
        ];
        assert_eq!(
            compute_action_manifest_sha256(&original).unwrap(),
            compute_action_manifest_sha256(&reversed).unwrap()
        );
        assert_eq!(
            compute_joint_draw_manifest_sha256(
                &[100, 101], ActionDrawCoupling::PairedExogenousDisturbance
            ).unwrap(),
            compute_joint_draw_manifest_sha256(
                &[101, 100], ActionDrawCoupling::PairedExogenousDisturbance
            ).unwrap()
        );
    }

    #[test]
    fn forecast_bank_digest_binds_predictions_but_not_row_order() {
        let original = bank();
        let mut permuted = bank();
        permuted.action_ids.swap(0, 1);
        permuted.samples.swap(0, 3);
        permuted.samples.swap(1, 2);
        assert_eq!(
            compute_action_forecast_bank_sha256(&original),
            compute_action_forecast_bank_sha256(&permuted)
        );
        let mut tampered = original;
        tampered.samples[0].predicted_belief[0] = 0.25;
        assert_ne!(
            compute_action_forecast_bank_sha256(&tampered),
            tampered.bank_sha256
        );
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn bank_rejects_incomplete_duplicate_and_nonfinite_action_samples() {
        let mut forecast_bank = bank();
        forecast_bank.samples.pop();
        assert!(forecast_bank.validate().is_err());
        forecast_bank = bank();
        forecast_bank.samples[1].joint_draw_id = 100;
        assert!(forecast_bank.validate().is_err());
        forecast_bank = bank();
        forecast_bank.samples[0].predicted_belief[0] = f32::NAN;
        assert!(forecast_bank.validate().is_err());
    }

    #[test]
    fn exact_target_branch_observation_scores_against_persistence() {
        let forecast_bank = bank();
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&forecast_bank).unwrap();
        assert_eq!(evaluator.pending_count(), 4);
        let target = [0.0, 1.0, 0.0];
        let result = evaluator
            .observe_branch(&observation("b17-100", 17, 100, 45, 1.505, &target))
            .unwrap();
        let ForecastObservationResult::Scored(score) = result else {
            panic!("expected a scored exact-target forecast");
        };
        assert_eq!(score.history_sha256, HISTORY);
        assert_eq!(score.action_id, 17);
        assert_eq!(score.joint_draw_id, 100);
        assert!((score.timing_error_seconds - 0.005).abs() < 1e-9);
        assert!(score.prediction_error.abs() < 1e-12);
        assert!((score.persistence_error - 1.0).abs() < 1e-12);
        assert_eq!(score.relative_skill, Some(1.0));
        assert_eq!(evaluator.pending_count(), 3);
        assert_eq!(evaluator.matured_count(), 1);
    }

    #[test]
    fn observations_before_target_wait_and_skipped_target_expires() {
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&bank()).unwrap();
        let early = [1.0, 0.0, 0.0];
        assert_eq!(
            evaluator.observe_branch(&observation("b17-100", 17, 100, 44, 1.4, &early)).unwrap(),
            ForecastObservationResult::NotDue
        );
        assert_eq!(evaluator.pending_count(), 4);
        assert_eq!(
            evaluator.observe_branch(&observation("b17-100", 17, 100, 46, 1.6, &early)).unwrap(),
            ForecastObservationResult::Expired(ForecastExpiryReason::TargetFrameSkipped)
        );
        assert_eq!(evaluator.expired_count(), 1);
    }

    #[test]
    fn late_observation_expires_instead_of_scoring_the_wrong_frame() {
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&bank()).unwrap();
        let actual = [0.0, 1.0, 0.0];
        assert_eq!(
            evaluator.observe_branch(&observation("b17-100", 17, 100, 45, 1.6, &actual)).unwrap(),
            ForecastObservationResult::Expired(ForecastExpiryReason::ObservationArrivedTooLate)
        );
        assert_eq!(evaluator.expired_count(), 1);
        assert_eq!(evaluator.matured_count(), 0);
    }

    #[test]
    fn branch_identity_mismatch_never_consumes_pending_forecast() {
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&bank()).unwrap();
        let actual = [0.0, 1.0, 0.0];
        let wrong = BranchVisualObservation {
            history_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            ..observation("b17-100", 17, 100, 45, 1.5, &actual)
        };
        assert!(evaluator.observe_branch(&wrong).is_err());
        assert_eq!(evaluator.pending_count(), 4);
    }

    #[test]
    fn queue_capacity_is_atomic_and_duplicate_branch_ids_are_rejected() {
        let forecast_bank = bank();
        let mut full = ActionConditionedForecastEvaluator::new(3, 0.01).unwrap();
        assert!(full.issue(&forecast_bank).is_err());
        assert_eq!(full.pending_count(), 0);

        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&forecast_bank).unwrap();
        assert!(evaluator.issue(&forecast_bank).is_err());
        assert_eq!(evaluator.pending_count(), 4);
    }

    #[test]
    fn timestamp_before_target_and_invalid_vectors_are_rejected_without_consumption() {
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.01).unwrap();
        evaluator.issue(&bank()).unwrap();
        let actual = [0.0, 1.0, 0.0];
        assert!(evaluator
            .observe_branch(&observation("b17-100", 17, 100, 45, 1.4, &actual))
            .is_err());
        let invalid = [0.0, f32::INFINITY, 0.0];
        assert!(evaluator
            .observe_branch(&observation("b17-100", 17, 100, 45, 1.5, &invalid))
            .is_err());
        assert_eq!(evaluator.pending_count(), 4);
    }

    #[test]
    fn persistence_perfect_prediction_has_undefined_relative_skill() {
        let mut forecast_bank = bank();
        forecast_bank.samples[0].predicted_belief = forecast_bank.persistence_belief.clone();
        let mut evaluator = ActionConditionedForecastEvaluator::new(8, 0.0).unwrap();
        evaluator.issue(&forecast_bank).unwrap();
        let actual = [1.0, 0.0, 0.0];
        let result = evaluator
            .observe_branch(&observation("b17-100", 17, 100, 45, 1.5, &actual))
            .unwrap();
        let ForecastObservationResult::Scored(score) = result else {
            panic!("expected scored forecast");
        };
        assert_eq!(score.prediction_error, score.persistence_error);
        assert_eq!(score.relative_skill, None);
    }
}
