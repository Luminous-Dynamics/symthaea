// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-jepa
//!
//! **Joint Embedding Predictive Architecture** for thermodynamically efficient
//! prediction in Symthaea's cognitive loop.
//!
//! ## Core Idea
//!
//! Instead of predicting raw observations (16,384D HDC vectors), JEPA predicts
//! in a learned 128D latent space. This provides:
//!
//! 1. **Thermodynamic efficiency**: 128x fewer dimensions → exponentially fewer
//!    bits erased per prediction → lower Landauer floor → more cycles before
//!    energy budget exhaustion.
//!
//! 2. **Compositional generalization**: The latent space learns to capture
//!    abstract state transitions rather than raw observation correlations.
//!
//! 3. **Collapse prevention**: EMA target encoder (momentum 0.996) prevents
//!    the trivial solution of mapping everything to a constant. Variance
//!    regularization provides a secondary defense.
//!
//! ## Architecture
//!
//! ```text
//! Current HV ──→ [Context Encoder] ──→ z_t ──→ [Predictor] ──→ ẑ_{t+1}
//!                                              ↑ action                 │
//! Next HV    ──→ [Target Encoder]  ──→ z*_{t+1} ←── cosine_loss ──────┘
//!                  (EMA, no grad)
//! ```
//!
//! ## Integration
//!
//! JEPA runs **parallel to** the CfC temporal network. Its latent prediction
//! error feeds into the FEP bridge's learning signal alongside TD error,
//! motor PE, and free energy.
//!
//! ## References
//!
//! - LeCun (2022): "A Path Towards Autonomous Machine Intelligence"
//! - Assran et al. (2023): I-JEPA (CVPR 2023)
//! - Grill et al. (2020): BYOL (NeurIPS 2020) — EMA target encoder origin
//! - Friston (2010): "The free-energy principle: a unified brain theory?"

#![deny(unsafe_code)]

pub mod config;
pub mod encoder;
pub mod loss;
pub mod predictor;
pub mod state;
pub mod target;

pub use config::JepaConfig;
pub use encoder::ContextEncoder;
pub use loss::{cosine_loss, cosine_loss_gradient, cosine_similarity, prediction_error};
pub use predictor::LatentPredictor;
pub use state::JepaState;
pub use target::TargetEncoder;

use sha2::{Digest, Sha256};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Telemetry snapshot from JEPA engine, for inclusion in `CycleMetadata`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct JepaTelemetry {
    /// Latent prediction error (cosine loss, 0.0 = perfect).
    pub latent_pe: f32,
    /// Total energy spent by JEPA (joules).
    pub total_energy: f64,
    /// Number of forward passes completed.
    pub steps: u64,
    /// Whether representation collapse has been detected.
    pub collapse_detected: bool,
    /// Current variance regularization penalty.
    pub variance_penalty: f32,
}

pub const OBSERVED_TRANSITION_EVIDENCE_SCHEMA_VERSION: &str =
    "symthaea-jepa-observed-transition-evidence-v0.1";
pub const ONE_STEP_ACTION_SET_SCHEMA_VERSION: &str = "symthaea-jepa-one-step-action-set-v0.1";

/// Evidence supplied by an observation adapter for one action-conditioned
/// transition. Embedding digests are recomputed here; raw-observation, history
/// and manifest digests identify upstream artifacts and require adapter-level
/// byte verification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedTransitionEvidence {
    pub schema_version: String,
    pub source_history_sha256: String,
    pub source_observation_sha256: String,
    pub source_embedding_sha256: String,
    pub target_observation_sha256: String,
    pub target_embedding_sha256: String,
    pub action_manifest_sha256: String,
    pub action_binding_sha256: String,
    pub episode_id: String,
    pub branch_id: String,
    pub forecast_id: String,
    pub source_observation_id: String,
    pub target_observation_id: String,
    pub source_sequence: u64,
    pub target_sequence: u64,
    pub action_id: u64,
    pub action_index: u8,
    pub source_observed_at_seconds: f64,
    pub forecast_issued_at_seconds: f64,
    pub action_started_at_seconds: f64,
    pub action_ended_at_seconds: f64,
    pub requested_target_time_seconds: f64,
    pub target_observed_at_seconds: f64,
    pub max_target_lateness_seconds: f64,
}

/// Stable candidate identity mapped to the JEPA discrete action embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCandidate {
    pub action_id: u64,
    pub action_index: u8,
}

/// One deterministic JEPA latent prediction for a candidate action.
/// This is a point prediction, not a calibrated distribution or joint draw.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionLatentPrediction {
    pub action_id: u64,
    pub action_index: u8,
    pub predicted_latent: Vec<f32>,
}

/// Predictions for a frozen action set, all computed from one encoded context.
/// The manifest/history digests are identifiers supplied by the caller.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneStepActionSetPrediction {
    pub schema_version: String,
    pub source_history_sha256: String,
    pub source_embedding_sha256: String,
    pub action_manifest_sha256: String,
    pub candidate_set_sha256: String,
    pub predictions: Vec<ActionLatentPrediction>,
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Canonical digest of the finite HDC embedding values consumed by JEPA.
/// This binds the exact f32 bit patterns and dimensionality, not the original
/// camera bytes or the upstream observation capture.
pub fn embedding_sha256(hv: &ContinuousHV) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-jepa-observation-embedding-v0.1\0");
    hasher.update((hv.dim() as u64).to_le_bytes());
    for value in hv.as_slice() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    digest_to_hex(&hasher.finalize())
}

/// Canonical binding of one stable action ID and its discrete JEPA index to
/// the caller-supplied digest of the full action manifest.
pub fn action_binding_sha256(
    action_manifest_sha256: &str,
    action_id: u64,
    action_index: u8,
) -> Result<String, String> {
    if !is_lower_hex_sha256(action_manifest_sha256) {
        return Err("action_manifest_sha256 must be a lowercase SHA-256 digest".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-jepa-action-binding-v0.1\0");
    hasher.update(action_manifest_sha256.as_bytes());
    hasher.update(action_id.to_le_bytes());
    hasher.update([action_index]);
    Ok(digest_to_hex(&hasher.finalize()))
}

/// Canonical digest of the frozen action IDs and their discrete JEPA indices.
/// The action-manifest digest remains the upstream binding to full descriptors.
pub fn candidate_set_sha256(
    action_manifest_sha256: &str,
    candidates: &[ActionCandidate],
) -> Result<String, String> {
    if !is_lower_hex_sha256(action_manifest_sha256) {
        return Err("action_manifest_sha256 must be a lowercase SHA-256 digest".into());
    }
    if candidates.len() < 2 {
        return Err("at least two action candidates are required".into());
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index].iter().any(|previous| {
            previous.action_id == candidate.action_id
                || previous.action_index == candidate.action_index
        }) {
            return Err("candidate action IDs and action indices must be unique".into());
        }
    }
    let mut ordered = candidates.to_vec();
    ordered.sort_by_key(|candidate| candidate.action_id);
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-jepa-action-candidate-set-v0.1\0");
    hasher.update(action_manifest_sha256.as_bytes());
    hasher.update((ordered.len() as u64).to_le_bytes());
    for candidate in ordered {
        hasher.update(candidate.action_id.to_le_bytes());
        hasher.update([candidate.action_index]);
    }
    Ok(digest_to_hex(&hasher.finalize()))
}

fn digest_to_hex(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let byte = *byte;
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn validate_observed_transition(
    config: &JepaConfig,
    current_hv: &ContinuousHV,
    next_hv: &ContinuousHV,
    evidence: &ObservedTransitionEvidence,
) -> Result<(), String> {
    if evidence.schema_version != OBSERVED_TRANSITION_EVIDENCE_SCHEMA_VERSION {
        return Err("unsupported observed-transition evidence schema version".into());
    }
    for (name, digest) in [
        ("source_history_sha256", &evidence.source_history_sha256),
        (
            "source_observation_sha256",
            &evidence.source_observation_sha256,
        ),
        ("source_embedding_sha256", &evidence.source_embedding_sha256),
        (
            "target_observation_sha256",
            &evidence.target_observation_sha256,
        ),
        ("target_embedding_sha256", &evidence.target_embedding_sha256),
        ("action_manifest_sha256", &evidence.action_manifest_sha256),
        ("action_binding_sha256", &evidence.action_binding_sha256),
    ] {
        if !is_lower_hex_sha256(digest) {
            return Err(format!("{name} must be a lowercase SHA-256 digest"));
        }
    }
    if [
        &evidence.episode_id,
        &evidence.branch_id,
        &evidence.forecast_id,
        &evidence.source_observation_id,
        &evidence.target_observation_id,
    ]
    .iter()
    .any(|id| id.trim().is_empty())
        || evidence.source_observation_id == evidence.target_observation_id
    {
        return Err(
            "episode, branch, forecast and distinct observation IDs must be non-empty".into(),
        );
    }
    if evidence.source_sequence >= evidence.target_sequence {
        return Err("target observation sequence must be later than source sequence".into());
    }
    let times = [
        evidence.source_observed_at_seconds,
        evidence.forecast_issued_at_seconds,
        evidence.action_started_at_seconds,
        evidence.action_ended_at_seconds,
        evidence.requested_target_time_seconds,
        evidence.target_observed_at_seconds,
        evidence.max_target_lateness_seconds,
    ];
    if !times.iter().all(|value| value.is_finite())
        || evidence.max_target_lateness_seconds < 0.0
        || !(times[0] <= times[1]
            && times[1] <= times[2]
            && times[3] <= times[4]
            && times[4] <= times[5]
            && times[2] <= times[3])
    {
        return Err(
            "transition times must be finite and satisfy source <= forecast <= action start <= requested target <= actual target, with action end >= action start".into(),
        );
    }
    if evidence.target_observed_at_seconds - evidence.requested_target_time_seconds
        > evidence.max_target_lateness_seconds
    {
        return Err("target observation exceeds the declared maximum forecast lateness".into());
    }
    if action_binding_sha256(
        &evidence.action_manifest_sha256,
        evidence.action_id,
        evidence.action_index,
    )? != evidence.action_binding_sha256
    {
        return Err("action ID/index binding digest mismatch".into());
    }
    if evidence.action_index as usize >= config.num_actions {
        return Err(format!(
            "action index {} is outside configured action space {}",
            evidence.action_index, config.num_actions
        ));
    }
    if current_hv.dim() != config.input_dim || next_hv.dim() != config.input_dim {
        return Err(format!(
            "observed transition dimensions must both equal configured input dimension {}",
            config.input_dim
        ));
    }
    if !current_hv.as_slice().iter().all(|value| value.is_finite())
        || !next_hv.as_slice().iter().all(|value| value.is_finite())
    {
        return Err("observed transition contains non-finite vector values".into());
    }
    if embedding_sha256(current_hv) != evidence.source_embedding_sha256
        || embedding_sha256(next_hv) != evidence.target_embedding_sha256
    {
        return Err("observed transition embedding digest mismatch".into());
    }
    Ok(())
}

/// JEPA engine: the top-level API composing all components.
///
/// Call `forward()` each cognitive cycle with the current and next HDC states
/// to get a latent prediction error. Call `train_step()` to update the
/// context encoder and predictor (target encoder updates via EMA).
pub struct JepaEngine {
    context: ContextEncoder,
    target: TargetEncoder,
    predictor: LatentPredictor,
    config: JepaConfig,
    state: JepaState,
}

impl JepaEngine {
    /// Create a new JEPA engine from configuration and deterministic seed.
    pub fn new(config: JepaConfig, seed: u64) -> Self {
        let context = ContextEncoder::new(config.input_dim, config.latent_dim, seed);
        let target = TargetEncoder::from_context(&context);
        let predictor =
            LatentPredictor::new(config.latent_dim, config.num_actions, seed.wrapping_add(42));
        let state = JepaState::new(config.latent_dim);
        Self {
            context,
            target,
            predictor,
            config,
            state,
        }
    }

    /// Forward pass: compute latent prediction error without updating weights.
    ///
    /// 1. Context encoder encodes `current_hv` → z_t
    /// 2. Predictor maps (z_t, action) → ẑ_{t+1}
    /// 3. Target encoder encodes `next_hv` → z*_{t+1}
    /// 4. Cosine loss(ẑ_{t+1}, z*_{t+1}) → latent PE
    ///
    /// Returns (latent_pe, energy_cost).
    /// `tau_factor` modulates nothing in forward (reserved for multi-horizon).
    pub fn forward(
        &mut self,
        current_hv: &ContinuousHV,
        next_hv: &ContinuousHV,
        action: u8,
        _tau_factor: f32,
    ) -> (f32, f64) {
        let z_t = self.context.encode(current_hv);
        let z_hat = self.predictor.predict(&z_t, action);
        let z_star = self.target.encode(next_hv);

        let pe = prediction_error(&z_hat, &z_star);

        // Update state
        self.state.last_pe = pe;
        self.state.cumulative_loss += pe as f64;
        self.state.total_energy_spent += self.config.effective_energy_per_forward();

        // Update variance tracking from target encoder output
        self.state.update_variance(&z_star);

        (pe, self.config.effective_energy_per_forward())
    }

    /// Train step: forward + backward + weight update.
    ///
    /// Updates context encoder and predictor via backpropagation.
    /// Updates target encoder via EMA (no gradient).
    /// Returns the loss (cosine loss + variance penalty).
    pub fn train_step(
        &mut self,
        current_hv: &ContinuousHV,
        next_hv: &ContinuousHV,
        action: u8,
        learning_rate: f32,
    ) -> f32 {
        // Forward
        let z_t = self.context.encode(current_hv);
        let z_hat = self.predictor.predict(&z_t, action);
        let z_star = self.target.encode(next_hv);

        // Compute loss + variance penalty
        let cos_loss = cosine_loss(&z_hat, &z_star);
        let var_penalty = self.state.variance_penalty(self.config.variance_floor);
        let total_loss = cos_loss + var_penalty;

        // Backward through predictor
        let output_grad = cosine_loss_gradient(&z_hat, &z_star);
        let latent_grad = self
            .predictor
            .backward(&z_t, action, &output_grad, learning_rate);

        // Backward through context encoder
        let (weight_grad, bias_grad) = self.context.backward(current_hv, &latent_grad);
        self.context
            .apply_gradients(&weight_grad, &bias_grad, learning_rate, 1e-5);

        // EMA update target encoder (no gradient)
        self.target
            .update_from_context(&self.context, self.config.ema_momentum);

        // Update state
        self.state.last_pe = prediction_error(&z_hat, &z_star);
        self.state.cumulative_loss += total_loss as f64;
        self.state.total_energy_spent += self.config.effective_energy_per_forward();
        self.state.update_variance(&z_star);

        total_loss
    }

    /// Train JEPA against a separately acquired later observation.
    ///
    /// Unlike the low-level train_step API, this path requires explicit
    /// observation/action provenance and strict temporal ordering. The evidence
    /// digests are identifiers only: upstream adapters remain responsible for
    /// authenticating that the vectors correspond to the retained observation
    /// bytes and that the target was not generated by this model.
    pub fn train_observed_transition(
        &mut self,
        current_hv: &ContinuousHV,
        next_observed_hv: &ContinuousHV,
        evidence: &ObservedTransitionEvidence,
        learning_rate: f32,
    ) -> Result<f32, String> {
        validate_observed_transition(&self.config, current_hv, next_observed_hv, evidence)?;
        if !learning_rate.is_finite()
            || !(0.0..=1.0).contains(&learning_rate)
            || learning_rate == 0.0
        {
            return Err("learning rate must be finite and in (0, 1]".into());
        }
        Ok(self.train_step(
            current_hv,
            next_observed_hv,
            evidence.action_index,
            learning_rate,
        ))
    }

    /// Score an observation-grounded transition without updating model weights.
    pub fn evaluate_observed_transition(
        &mut self,
        current_hv: &ContinuousHV,
        next_observed_hv: &ContinuousHV,
        evidence: &ObservedTransitionEvidence,
    ) -> Result<(f32, f64), String> {
        validate_observed_transition(&self.config, current_hv, next_observed_hv, evidence)?;
        Ok(self.forward(current_hv, next_observed_hv, evidence.action_index, 1.0))
    }

    /// Produce one-step latent predictions for a frozen set of candidate
    /// actions. The current observation is encoded exactly once; each candidate
    /// is then evaluated from that same context. Outputs are deterministic
    /// point predictions and must not be relabelled as uncertainty samples.
    pub fn predict_action_set_one_step(
        &self,
        current_hv: &ContinuousHV,
        source_history_sha256: &str,
        action_manifest_sha256: &str,
        candidates: &[ActionCandidate],
    ) -> Result<OneStepActionSetPrediction, String> {
        for (name, digest) in [
            ("source_history_sha256", source_history_sha256),
            ("action_manifest_sha256", action_manifest_sha256),
        ] {
            if !is_lower_hex_sha256(digest) {
                return Err(format!("{name} must be a lowercase SHA-256 digest"));
            }
        }
        if current_hv.dim() != self.config.input_dim
            || !current_hv.as_slice().iter().all(|value| value.is_finite())
        {
            return Err("current observation has invalid dimension or non-finite values".into());
        }
        if candidates.len() < 2 {
            return Err("at least two candidate actions are required".into());
        }
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.action_index as usize >= self.config.num_actions {
                return Err(format!(
                    "candidate {index} has an out-of-range action index"
                ));
            }
            if candidates[..index].iter().any(|previous| {
                previous.action_id == candidate.action_id
                    || previous.action_index == candidate.action_index
            }) {
                return Err("candidate action IDs and action indices must be unique".into());
            }
        }

        let source_embedding_sha256 = embedding_sha256(current_hv);
        let candidate_set_digest = candidate_set_sha256(action_manifest_sha256, candidates)?;
        let context = self.context.encode(current_hv);
        let predictions = candidates
            .iter()
            .map(|candidate| ActionLatentPrediction {
                action_id: candidate.action_id,
                action_index: candidate.action_index,
                predicted_latent: self.predictor.predict(&context, candidate.action_index),
            })
            .collect::<Vec<_>>();
        if predictions.iter().any(|prediction| {
            !prediction
                .predicted_latent
                .iter()
                .all(|value| value.is_finite())
        }) {
            return Err("JEPA action-set prediction produced non-finite values".into());
        }
        Ok(OneStepActionSetPrediction {
            schema_version: ONE_STEP_ACTION_SET_SCHEMA_VERSION.into(),
            source_history_sha256: source_history_sha256.to_string(),
            source_embedding_sha256,
            action_manifest_sha256: action_manifest_sha256.to_string(),
            candidate_set_sha256: candidate_set_digest,
            predictions,
        })
    }

    /// Get the most recent latent prediction error.
    pub fn latent_pe(&self) -> f32 {
        self.state.last_pe
    }

    /// Get total energy spent by JEPA (joules).
    pub fn energy_spent(&self) -> f64 {
        self.state.total_energy_spent
    }

    /// Get telemetry snapshot for `CycleMetadata`.
    pub fn telemetry(&self) -> JepaTelemetry {
        JepaTelemetry {
            latent_pe: self.state.last_pe,
            total_energy: self.state.total_energy_spent,
            steps: self.state.steps,
            collapse_detected: self.state.has_collapse(self.config.variance_floor),
            variance_penalty: self.state.variance_penalty(self.config.variance_floor),
        }
    }

    /// Get configuration (for substrate integration).
    pub fn config(&self) -> &JepaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;

    fn make_engine() -> JepaEngine {
        JepaEngine::new(JepaConfig::default(), 12345)
    }

    fn observed_evidence(
        current_hv: &ContinuousHV,
        next_hv: &ContinuousHV,
        action_index: u8,
    ) -> ObservedTransitionEvidence {
        ObservedTransitionEvidence {
            schema_version: OBSERVED_TRANSITION_EVIDENCE_SCHEMA_VERSION.into(),
            source_history_sha256: "a".repeat(64),
            source_observation_sha256: "b".repeat(64),
            source_embedding_sha256: embedding_sha256(current_hv),
            target_observation_sha256: "c".repeat(64),
            target_embedding_sha256: embedding_sha256(next_hv),
            action_manifest_sha256: "d".repeat(64),
            action_binding_sha256: action_binding_sha256(&"d".repeat(64), 42, action_index)
                .unwrap(),
            episode_id: "episode-1".into(),
            branch_id: "branch-action-42".into(),
            forecast_id: "forecast-10-a42".into(),
            source_observation_id: "frame-10".into(),
            target_observation_id: "frame-11".into(),
            source_sequence: 10,
            target_sequence: 11,
            action_id: 42,
            action_index,
            source_observed_at_seconds: 1.0,
            forecast_issued_at_seconds: 1.005,
            action_started_at_seconds: 1.01,
            action_ended_at_seconds: 1.05,
            requested_target_time_seconds: 1.07,
            target_observed_at_seconds: 1.08,
            max_target_lateness_seconds: 0.02,
        }
    }

    #[test]
    fn observed_transition_training_accepts_later_observation_with_bound_action() {
        let mut engine = make_engine();
        let current = random_hv(501);
        let observed_next = random_hv(502);
        let loss = engine
            .train_observed_transition(
                &current,
                &observed_next,
                &observed_evidence(&current, &observed_next, 2),
                0.001,
            )
            .unwrap();
        assert!(loss.is_finite());
        assert!(engine.latent_pe().is_finite());
    }

    #[test]
    fn observed_transition_training_rejects_unverifiable_or_misaligned_evidence() {
        let current = random_hv(511);
        let next = random_hv(512);

        let mut malformed = observed_evidence(&current, &next, 2);
        malformed.target_observation_sha256 = "not-a-digest".into();
        let mut candidate = make_engine();
        assert!(
            candidate
                .train_observed_transition(&current, &next, &malformed, 0.001)
                .is_err()
        );

        let mut malformed = observed_evidence(&current, &next, 2);
        malformed.target_sequence = malformed.source_sequence;
        assert!(
            candidate
                .train_observed_transition(&current, &next, &malformed, 0.001)
                .is_err()
        );

        let mut malformed = observed_evidence(&current, &next, 2);
        malformed.target_observed_at_seconds =
            malformed.requested_target_time_seconds + malformed.max_target_lateness_seconds + 0.001;
        assert!(
            candidate
                .train_observed_transition(&current, &next, &malformed, 0.001)
                .is_err()
        );

        let malformed = observed_evidence(&current, &next, 250);
        assert!(
            candidate
                .train_observed_transition(&current, &next, &malformed, 0.001)
                .is_err()
        );

        let malformed = observed_evidence(&current, &next, 2);
        assert!(
            candidate
                .train_observed_transition(&current, &next, &malformed, f32::NAN)
                .is_err()
        );
    }

    #[test]
    fn observed_transition_rejects_cross_episode_missing_branch_and_forecast_order_errors() {
        let current = random_hv(517);
        let next = random_hv(518);
        let mut engine = make_engine();

        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.branch_id.clear();
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );

        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.forecast_issued_at_seconds = evidence.action_started_at_seconds + 0.001;
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );

        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.action_ended_at_seconds = evidence.requested_target_time_seconds + 0.001;
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );

        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.requested_target_time_seconds = evidence.target_observed_at_seconds + 0.001;
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );
    }

    #[test]
    fn observed_transition_rejects_action_id_index_binding_mismatch() {
        let current = random_hv(519);
        let next = random_hv(520);
        let mut engine = make_engine();
        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.action_id = 43;
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );
    }

    #[test]
    fn observed_transition_training_rejects_embedding_byte_mismatch() {
        let mut engine = make_engine();
        let current = random_hv(515);
        let next = random_hv(516);
        let mut evidence = observed_evidence(&current, &next, 2);
        evidence.target_embedding_sha256 = "c".repeat(64);
        assert!(
            engine
                .train_observed_transition(&current, &next, &evidence, 0.001)
                .is_err()
        );
    }

    #[test]
    fn jepa_digest_helpers_match_sha256_known_answer() {
        assert_eq!(
            digest_to_hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let left = random_hv(551);
        let right = random_hv(551);
        assert_eq!(embedding_sha256(&left), embedding_sha256(&right));
        assert_eq!(embedding_sha256(&left).len(), 64);
    }

    #[test]
    fn candidate_set_digest_is_row_order_invariant_and_binds_action_mapping() {
        let manifest = "b".repeat(64);
        let left = [
            ActionCandidate {
                action_id: 700,
                action_index: 1,
            },
            ActionCandidate {
                action_id: 300,
                action_index: 3,
            },
        ];
        let right = [left[1], left[0]];
        assert_eq!(
            candidate_set_sha256(&manifest, &left).unwrap(),
            candidate_set_sha256(&manifest, &right).unwrap()
        );
        let changed = [
            ActionCandidate {
                action_id: 700,
                action_index: 2,
            },
            ActionCandidate {
                action_id: 300,
                action_index: 3,
            },
        ];
        assert_ne!(
            candidate_set_sha256(&manifest, &left).unwrap(),
            candidate_set_sha256(&manifest, &changed).unwrap()
        );
    }

    #[test]
    fn action_set_predictions_share_one_context_and_preserve_candidate_identity() {
        let engine = make_engine();
        let current = random_hv(521);
        let result = engine
            .predict_action_set_one_step(
                &current,
                &"a".repeat(64),
                &"b".repeat(64),
                &[
                    ActionCandidate {
                        action_id: 700,
                        action_index: 1,
                    },
                    ActionCandidate {
                        action_id: 300,
                        action_index: 3,
                    },
                    ActionCandidate {
                        action_id: 900,
                        action_index: 5,
                    },
                ],
            )
            .unwrap();
        assert_eq!(
            result
                .predictions
                .iter()
                .map(|item| item.action_id)
                .collect::<Vec<_>>(),
            vec![700, 300, 900]
        );
        assert_eq!(
            result
                .predictions
                .iter()
                .map(|item| item.predicted_latent.len())
                .collect::<Vec<_>>(),
            vec![engine.config().latent_dim; 3]
        );
        assert!(
            result
                .predictions
                .iter()
                .all(|item| { item.predicted_latent.iter().all(|value| value.is_finite()) })
        );
    }

    #[test]
    fn action_set_predictions_are_invariant_to_candidate_row_order() {
        let engine = make_engine();
        let current = random_hv(541);
        let history = "a".repeat(64);
        let manifest = "b".repeat(64);
        let forward = engine
            .predict_action_set_one_step(
                &current,
                &history,
                &manifest,
                &[
                    ActionCandidate {
                        action_id: 70,
                        action_index: 2,
                    },
                    ActionCandidate {
                        action_id: 20,
                        action_index: 4,
                    },
                ],
            )
            .unwrap();
        let reversed = engine
            .predict_action_set_one_step(
                &current,
                &history,
                &manifest,
                &[
                    ActionCandidate {
                        action_id: 20,
                        action_index: 4,
                    },
                    ActionCandidate {
                        action_id: 70,
                        action_index: 2,
                    },
                ],
            )
            .unwrap();
        for expected in &forward.predictions {
            let actual = reversed
                .predictions
                .iter()
                .find(|candidate| candidate.action_id == expected.action_id)
                .unwrap();
            assert_eq!(actual.action_index, expected.action_index);
            assert_eq!(actual.predicted_latent, expected.predicted_latent);
        }
    }

    #[test]
    fn action_set_prediction_rejects_duplicate_ids_indices_and_bad_digests() {
        let engine = make_engine();
        let current = random_hv(531);
        let candidates = [
            ActionCandidate {
                action_id: 1,
                action_index: 1,
            },
            ActionCandidate {
                action_id: 1,
                action_index: 2,
            },
        ];
        assert!(
            engine
                .predict_action_set_one_step(
                    &current,
                    &"a".repeat(64),
                    &"b".repeat(64),
                    &candidates
                )
                .is_err()
        );

        let candidates = [
            ActionCandidate {
                action_id: 1,
                action_index: 1,
            },
            ActionCandidate {
                action_id: 2,
                action_index: 1,
            },
        ];
        assert!(
            engine
                .predict_action_set_one_step(
                    &current,
                    &"a".repeat(64),
                    &"b".repeat(64),
                    &candidates
                )
                .is_err()
        );

        assert!(
            engine
                .predict_action_set_one_step(&current, "bad", &"b".repeat(64), &candidates)
                .is_err()
        );
        assert!(
            engine
                .predict_action_set_one_step(
                    &current,
                    &"a".repeat(64),
                    &"b".repeat(64),
                    &candidates[..1]
                )
                .is_err()
        );
    }

    fn random_hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    #[test]
    fn test_encode_dimensions() {
        let engine = make_engine();
        let hv = random_hv(1);
        let latent = engine.context.encode(&hv);
        assert_eq!(latent.len(), 128, "latent should be 128D");
        assert!(
            latent.iter().all(|v| v.is_finite()),
            "all values should be finite"
        );
    }

    #[test]
    fn test_target_ema_convergence() {
        let mut engine = make_engine();
        // After many EMA updates with constant context, target should converge
        let hv = random_hv(2);
        let initial_target = engine.target.encode(&hv);

        for _ in 0..500 {
            engine.target.update_from_context(&engine.context, 0.996);
        }
        let converged_target = engine.target.encode(&hv);
        let context_out = engine.context.encode(&hv);

        // After 500 EMA updates with no context change, target ≈ context
        let sim = cosine_similarity(&converged_target, &context_out);
        assert!(
            sim > 0.99,
            "target should converge to context after many EMA updates, got sim={sim}"
        );

        // But initial target should differ from converged (due to EMA lag on initial random weights being the same — they start identical, so this tests that EMA is a no-op when already equal)
        let initial_sim = cosine_similarity(&initial_target, &converged_target);
        assert!(
            initial_sim > 0.99,
            "target starts as copy of context, so should remain similar: {initial_sim}"
        );
    }

    #[test]
    fn test_cosine_loss_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(cosine_loss(&a, &a) < 1e-6);
    }

    #[test]
    fn test_cosine_loss_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!((cosine_loss(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_energy_tracking() {
        let mut engine = make_engine();
        let hv1 = random_hv(10);
        let hv2 = random_hv(11);

        assert_eq!(engine.energy_spent(), 0.0);

        let (_, cost1) = engine.forward(&hv1, &hv2, 0, 1.0);
        assert!(cost1 > 0.0);
        assert_eq!(engine.energy_spent(), cost1);

        let (_, cost2) = engine.forward(&hv1, &hv2, 1, 1.0);
        assert_eq!(engine.energy_spent(), cost1 + cost2);
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let mut engine = make_engine();
        let hv1 = random_hv(100);
        let hv2 = random_hv(101);
        let action = 3u8;

        // Collect initial loss
        let initial_loss = engine.train_step(&hv1, &hv2, action, 0.001);

        // Train for several steps on the same pair
        let mut last_loss = initial_loss;
        for _ in 0..50 {
            last_loss = engine.train_step(&hv1, &hv2, action, 0.001);
        }

        assert!(
            last_loss < initial_loss,
            "loss should decrease after training: initial={initial_loss}, final={last_loss}"
        );
    }

    #[test]
    fn test_forward_returns_bounded_pe() {
        let mut engine = make_engine();
        let hv1 = random_hv(200);
        let hv2 = random_hv(201);

        let (pe, _) = engine.forward(&hv1, &hv2, 0, 1.0);
        assert!(pe >= 0.0 && pe <= 1.0, "PE should be in [0,1], got {pe}");
    }

    #[test]
    fn test_telemetry_populated() {
        let mut engine = make_engine();
        let hv1 = random_hv(300);
        let hv2 = random_hv(301);

        engine.forward(&hv1, &hv2, 0, 1.0);
        let telem = engine.telemetry();

        assert!(telem.latent_pe >= 0.0);
        assert!(telem.total_energy > 0.0);
        assert_eq!(telem.steps, 1);
    }
}
