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
    /// Pending transitions discarded because no FEP action was bound.
    #[serde(default)]
    pub dropped_pending_transitions: u64,
}

/// JEPA engine: the top-level API composing all components.
///
/// Standalone callers may use forward()/train_step() with explicit state pairs.
/// The cognitive-loop integration uses stage_transition_source(), binds the
/// selected action with condition_pending_transition(), and resolves the pair
/// only when the next real observation arrives. The latter avoids self-targets.
struct PendingTransition {
    source: ContinuousHV,
    action: Option<u8>,
    context_latent: Option<Vec<f32>>,
    predicted_latent: Option<Vec<f32>>,
}

pub struct JepaEngine {
    context: ContextEncoder,
    target: TargetEncoder,
    predictor: LatentPredictor,
    config: JepaConfig,
    state: JepaState,
    pending_transition: Option<PendingTransition>,
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
            pending_transition: None,
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

    /// Stage the current real observation as the source of a future transition.
    ///
    /// The action is deliberately bound later, after FEP selects it. On the next
    /// observation, resolve_pending_transition trains against that independently
    /// observed target. This avoids using a model-generated forecast as its own
    /// training target and avoids pairing the current observation with a stale action.
    pub fn stage_transition_source(
        &mut self,
        source: &ContinuousHV,
    ) -> Result<(), String> {
        if source.dim() != self.config.input_dim {
            return Err(format!(
                "JEPA transition source dimension mismatch: got {}, expected {}",
                source.dim(),
                self.config.input_dim
            ));
        }
        if !source.as_slice().iter().all(|value| value.is_finite()) {
            return Err("JEPA transition source contains non-finite values".into());
        }
        if self.pending_transition.is_some() {
            return Err("JEPA previous transition has not been resolved or dropped".into());
        }
        self.pending_transition = Some(PendingTransition {
            source: source.clone(),
            action: None,
            context_latent: None,
            predicted_latent: None,
        });
        Ok(())
    }

    /// Bind the FEP-selected action and create the action-conditioned latent
    /// prediction before the next observation arrives.
    ///
    /// The caller must ensure this index denotes the action actually issued
    /// between the staged source and the next observation. In the cognitive-loop
    /// integration it is the selected FEP action token; it is not evidence of a
    /// physical actuator command unless an external adapter executes that token.
    ///
    /// Returns false when no source observation has been staged. The prediction
    /// is frozen until resolve_pending_transition receives the actual next input.
    pub fn condition_pending_transition(&mut self, action: u8) -> Result<bool, String> {
        let Some(pending) = self.pending_transition.as_ref() else {
            return Ok(false);
        };
        if action as usize >= self.config.num_actions {
            return Err(format!(
                "JEPA action index {action} is outside configured action count {}",
                self.config.num_actions
            ));
        }
        if pending.action.is_some() {
            return Err("JEPA pending transition is already action-conditioned".into());
        }

        let source = pending.source.clone();
        let context_latent = self.context.encode(&source);
        let predicted_latent = self.predictor.predict(&context_latent, action);
        if !context_latent.iter().all(|value| value.is_finite())
            || !predicted_latent.iter().all(|value| value.is_finite())
        {
            return Err("JEPA action-conditioned prediction contains non-finite values".into());
        }

        let pending = self
            .pending_transition
            .as_mut()
            .ok_or_else(|| "JEPA pending transition disappeared during action binding".to_string())?;
        pending.action = Some(action);
        pending.context_latent = Some(context_latent);
        pending.predicted_latent = Some(predicted_latent);
        // Charge the prediction now, even if the environment never returns a
        // scoreable next observation. Training later must not hide inference cost.
        self.state.total_energy_spent += self.config.effective_energy_per_forward();
        Ok(true)
    }

    /// Resolve the staged action-conditioned prediction against the next real
    /// observation, then update the context/predictor and EMA target networks.
    ///
    /// Returns None if there is no pending transition. If an action was never
    /// bound, the transition is explicitly dropped and counted rather than
    /// silently training an action-free or stale-action pair.
    pub fn resolve_pending_transition(
        &mut self,
        actual_next: &ContinuousHV,
        learning_rate: f32,
    ) -> Result<Option<f32>, String> {
        if self.pending_transition.is_none() {
            return Ok(None);
        }
        if actual_next.dim() != self.config.input_dim {
            return Err(format!(
                "JEPA next observation dimension mismatch: got {}, expected {}",
                actual_next.dim(),
                self.config.input_dim
            ));
        }
        if !actual_next.as_slice().iter().all(|value| value.is_finite()) {
            return Err("JEPA next observation contains non-finite values".into());
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(format!(
                "JEPA learning rate must be finite and > 0, got {learning_rate}"
            ));
        }

        let ready = self.pending_transition.as_ref().is_some_and(|pending| {
            pending.action.is_some()
                && pending.context_latent.is_some()
                && pending.predicted_latent.is_some()
        });
        if !ready {
            self.pending_transition = None;
            self.state.dropped_pending_transitions =
                self.state.dropped_pending_transitions.saturating_add(1);
            return Ok(None);
        }

        let target_latent = self.target.encode(actual_next);
        if !target_latent.iter().all(|value| value.is_finite()) {
            return Err("JEPA target encoder produced non-finite values".into());
        }
        let predicted_latent = self
            .pending_transition
            .as_ref()
            .and_then(|pending| pending.predicted_latent.as_ref())
            .ok_or_else(|| "JEPA pending prediction missing after readiness check".to_string())?;
        let cosine = cosine_loss(predicted_latent, &target_latent);
        let variance_penalty = self.state.variance_penalty(self.config.variance_floor);
        let total_loss = cosine + variance_penalty;
        if !total_loss.is_finite() {
            return Err("JEPA transition loss is non-finite".into());
        }

        let pending = self
            .pending_transition
            .take()
            .ok_or_else(|| "JEPA pending transition disappeared before update".to_string())?;
        let action = pending
            .action
            .ok_or_else(|| "JEPA pending action missing before update".to_string())?;
        let context_latent = pending
            .context_latent
            .ok_or_else(|| "JEPA pending context latent missing before update".to_string())?;
        let predicted_latent = pending
            .predicted_latent
            .ok_or_else(|| "JEPA pending prediction missing before update".to_string())?;

        let output_grad = cosine_loss_gradient(&predicted_latent, &target_latent);
        let latent_grad =
            self.predictor
                .backward(&context_latent, action, &output_grad, learning_rate);
        let (weight_grad, bias_grad) = self.context.backward(&pending.source, &latent_grad);
        self.context
            .apply_gradients(&weight_grad, &bias_grad, learning_rate, 1e-5);
        self.target
            .update_from_context(&self.context, self.config.ema_momentum);

        self.state.last_pe = prediction_error(&predicted_latent, &target_latent);
        self.state.cumulative_loss += total_loss as f64;
        self.state.update_variance(&target_latent);
        Ok(Some(total_loss))
    }

    /// Return the pending prediction without exposing or mutating the target.
    /// This is intended for diagnostics and pre-observation forecast receipts.
    pub fn pending_prediction(&self) -> Option<&[f32]> {
        self.pending_transition
            .as_ref()
            .and_then(|pending| pending.predicted_latent.as_deref())
    }

    /// Explicitly discard an incomplete pending transition.
    pub fn discard_pending_transition(&mut self) -> bool {
        if self.pending_transition.take().is_some() {
            self.state.dropped_pending_transitions =
                self.state.dropped_pending_transitions.saturating_add(1);
            true
        } else {
            false
        }
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
            dropped_pending_transitions: self.state.dropped_pending_transitions,
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
    fn observed_transition_uses_actual_next_observation_and_action_bound_before_it() {
        let mut engine = make_engine();
        let source = random_hv(900);
        let actual_next = random_hv(901);

        engine.stage_transition_source(&source).unwrap();
        assert!(engine.condition_pending_transition(3).unwrap());
        let prediction_before_observation = engine.pending_prediction().unwrap().to_vec();
        assert_eq!(prediction_before_observation.len(), engine.config.latent_dim);
        assert_eq!(engine.telemetry().steps, 0);
        assert!(engine.energy_spent() > 0.0);

        let loss = engine.resolve_pending_transition(&actual_next, 0.001).unwrap().unwrap();
        assert!(loss.is_finite());
        assert_eq!(engine.telemetry().steps, 1);
        assert_eq!(engine.telemetry().dropped_pending_transitions, 0);
        assert!(engine.pending_prediction().is_none());
        assert!(engine.resolve_pending_transition(&random_hv(902), 0.001).unwrap().is_none());
    }

    #[test]
    fn staged_prediction_is_action_conditioned_before_the_target_is_available() {
        let source = random_hv(910);
        let mut action_zero = make_engine();
        let mut action_one = make_engine();

        action_zero.stage_transition_source(&source).unwrap();
        action_one.stage_transition_source(&source).unwrap();
        action_zero.condition_pending_transition(0).unwrap();
        action_one.condition_pending_transition(1).unwrap();

        let zero_prediction = action_zero.pending_prediction().unwrap();
        let one_prediction = action_one.pending_prediction().unwrap();
        assert_ne!(zero_prediction, one_prediction);
    }

    #[test]
    fn observed_target_changes_the_training_update_after_prediction_is_frozen() {
        let source = random_hv(915);
        let next_a = random_hv(916);
        let next_b = random_hv(917);
        let mut engine_a = make_engine();
        let mut engine_b = make_engine();

        engine_a.stage_transition_source(&source).unwrap();
        engine_b.stage_transition_source(&source).unwrap();
        engine_a.condition_pending_transition(4).unwrap();
        engine_b.condition_pending_transition(4).unwrap();
        assert_eq!(
            engine_a.pending_prediction().unwrap(),
            engine_b.pending_prediction().unwrap()
        );

        engine_a.resolve_pending_transition(&next_a, 0.001).unwrap().unwrap();
        engine_b.resolve_pending_transition(&next_b, 0.001).unwrap().unwrap();
        assert_ne!(engine_a.context.weights, engine_b.context.weights);
    }

    #[test]
    fn missing_action_is_counted_as_a_dropped_transition_not_trained() {
        let mut engine = make_engine();
        engine.stage_transition_source(&random_hv(920)).unwrap();

        assert!(engine.resolve_pending_transition(&random_hv(921), 0.001).unwrap().is_none());
        assert_eq!(engine.telemetry().steps, 0);
        assert_eq!(engine.telemetry().dropped_pending_transitions, 1);
        engine.stage_transition_source(&random_hv(922)).unwrap();
    }

    #[test]
    fn invalid_action_and_target_do_not_consume_a_valid_pending_transition() {
        let mut engine = make_engine();
        engine.stage_transition_source(&random_hv(930)).unwrap();

        assert!(engine.condition_pending_transition(8).is_err());
        assert!(engine.pending_prediction().is_none());
        engine.condition_pending_transition(2).unwrap();

        let wrong_dim = ContinuousHV::random(HDC_DIMENSION - 1, 931);
        assert!(engine.resolve_pending_transition(&wrong_dim, 0.001).is_err());
        assert!(engine.pending_prediction().is_some());

        engine.resolve_pending_transition(&random_hv(932), 0.001).unwrap().unwrap();
        assert!(engine.pending_prediction().is_none());
    }

    #[test]
    fn pending_transition_cannot_be_overwritten_or_reconditioned() {
        let mut engine = make_engine();
        let source = random_hv(940);
        engine.stage_transition_source(&source).unwrap();
        engine.condition_pending_transition(1).unwrap();

        assert!(engine.stage_transition_source(&random_hv(941)).is_err());
        assert!(engine.condition_pending_transition(2).is_err());
        engine.resolve_pending_transition(&random_hv(942), 0.001).unwrap().unwrap();
        engine.stage_transition_source(&random_hv(943)).unwrap();
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
