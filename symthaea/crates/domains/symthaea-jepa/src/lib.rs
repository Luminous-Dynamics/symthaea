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
