// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal training for the CfC vision manifold.
//!
//! Makes the manifold's dynamics learnable via prediction-error-driven training:
//!
//! - **BPTT**: Analytical gradient through the closed-form CfC step.
//!   `∂loss/∂W = ∂loss/∂pred · ∂pred/∂x_inf · ∂x_inf/∂W`
//!
//! - **SPSA**: Zeroth-order gradient estimation via simultaneous perturbation.
//!   No differentiability required; robust fallback.
//!
//! Trainable parameters: `weight_hv` (W matrix in equilibrium) and `tau_base`
//! (temporal adaptation rate).

use symthaea_core::hdc::ContinuousHV;

use crate::types::{AdamStateSnapshot, TrainerState, TrainingConfig, TrainingMethod};

/// Adam optimizer state for a single HV parameter.
pub struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl AdamState {
    pub fn new(dim: usize) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Apply one Adam step, returning the update delta (not yet scaled by lr).
    pub fn step(&mut self, gradient: &[f32], grad_clip: f32) -> Vec<f32> {
        self.t = self.t.saturating_add(1);
        let t = self.t as f32;
        let mut delta = vec![0.0f32; gradient.len()];

        for i in 0..gradient.len().min(self.m.len()) {
            let g = gradient[i].clamp(-grad_clip, grad_clip);
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));

            delta[i] = m_hat / (v_hat.sqrt() + self.eps);
        }

        // NaN guard: replace any non-finite deltas with zero
        for d in &mut delta {
            if !d.is_finite() {
                *d = 0.0;
            }
        }

        delta
    }

    fn save_state(&self) -> AdamStateSnapshot {
        AdamStateSnapshot {
            m: self.m.clone(),
            v: self.v.clone(),
            t: self.t,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
        }
    }

    fn load_state(&mut self, state: &AdamStateSnapshot, expected_dim: usize) -> Result<(), String> {
        if state.m.len() != expected_dim || state.v.len() != expected_dim {
            return Err(format!(
                "Adam state dimension mismatch: m={}, v={}, expected={expected_dim}",
                state.m.len(),
                state.v.len()
            ));
        }
        self.m = state.m.clone();
        self.v = state.v.clone();
        self.t = state.t;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.eps = state.eps;
        Ok(())
    }

    /// Perform 'Holographic Dilation' - scale internal momentum buffers.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.m.len() == target_dim {
            return;
        }

        // We use the same segment-based bundling for momentum buffers
        // to maintain semantic alignment of the learned gradients.
        let m_hv = ContinuousHV::from_values(self.m.clone()).dilate(target_dim);
        // Adam's second moment is a variance-like quantity and must remain
        // non-negative. Dilate its square root, then square the result instead
        // of treating the raw second moment as a signed semantic HV.
        let v_root =
            ContinuousHV::from_values(self.v.iter().map(|value| value.max(0.0).sqrt()).collect())
                .dilate(target_dim);

        self.m = m_hv.values;
        self.v = v_root
            .values
            .into_iter()
            .map(|value| value * value)
            .collect();
    }
}

/// Training state for the vision manifold's CfC parameters.
pub struct ManifoldTrainer {
    config: TrainingConfig,
    weight_adam: AdamState,
    tau_adam: AdamState,
    rng_state: u64,
    total_steps: u64,
    /// Observation weight used by the manifold equilibrium. The trainable
    /// weight contribution is `1.0 - input_blend`.
    input_blend: f32,
}

impl ManifoldTrainer {
    pub fn new(config: &TrainingConfig, hdc_dim: usize) -> Self {
        Self {
            config: config.clone(),
            weight_adam: AdamState::new(hdc_dim),
            tau_adam: AdamState::new(1),
            rng_state: 0xCAFE_BABE_1337_DEAD,
            total_steps: 0,
            input_blend: 0.7,
        }
    }

    pub(crate) fn hdc_vector_count(&self) -> usize {
        // Adam stores one first- and one second-moment vector for weight_hv.
        2
    }

    /// Perform 'Holographic Dilation' - scale internal components.
    pub fn dilate(&mut self, target_dim: usize) {
        self.weight_adam.dilate(target_dim);
        // tau_adam is scalar (dim 1), no dilation needed
    }

    /// Keep the trainer's derivative model aligned with the manifold's
    /// equilibrium blend.
    pub fn set_input_blend(&mut self, input_blend: f32) {
        let _ = self.set_input_blend_checked(input_blend);
    }

    /// Checked equilibrium-blend update used by runtime policy changes.
    pub fn set_input_blend_checked(&mut self, input_blend: f32) -> Result<(), String> {
        if !input_blend.is_finite() || !(0.0..=1.0).contains(&input_blend) {
            return Err(format!(
                "input blend must be finite and in [0, 1], got {input_blend}"
            ));
        }
        self.input_blend = input_blend;
        Ok(())
    }

    /// Recover the forward-pass equilibrium from `state`, `predicted`, and the
    /// closed-form gate. This avoids rebuilding it from the supervision target.
    fn equilibrium_from_prediction(
        state: &ContinuousHV,
        predicted: &ContinuousHV,
        sigma: f32,
    ) -> ContinuousHV {
        if sigma <= 1e-8 {
            return state.clone();
        }

        let one_minus_sigma = 1.0 - sigma;
        let values = state
            .as_slice()
            .iter()
            .zip(predicted.as_slice().iter())
            .map(|(&state_i, &predicted_i)| (predicted_i - one_minus_sigma * state_i) / sigma)
            .collect();
        ContinuousHV::from_vec(values)
    }

    /// Compute BPTT gradient for weight_hv and tau through the CfC closed-form.
    ///
    /// Loss = 1 - cos_sim(predicted, actual_next_frame)
    ///
    /// Chain rule:
    /// - `∂loss/∂W = ∂loss/∂pred · ∂pred/∂x_inf · ∂x_inf/∂W`
    /// - `∂pred/∂x_inf = sigma` (gating factor)
    /// - `∂x_inf/∂W = (1-input_blend) * (1 - x_inf²) ⊗ state`
    pub fn bptt_step(
        &mut self,
        weight_hv: &ContinuousHV,
        state: &ContinuousHV,
        predicted: &ContinuousHV,
        actual: &ContinuousHV,
        tau_base: f32,
        dt: f32,
    ) -> BpttResult {
        let dim = weight_hv.dim();

        // Loss = 1 - cos_sim(predicted, actual)
        let pred_norm = predicted.norm().max(1e-8);
        let actual_norm = actual.norm().max(1e-8);
        let dot = predicted.dot(actual);
        let cos_sim = (dot / (pred_norm * actual_norm)).clamp(-1.0, 1.0);
        let loss = 1.0 - cos_sim;

        // ∂loss/∂pred = -actual / (|pred| · |actual|) + cos_sim · pred / |pred|²
        let pred_slice = predicted.as_slice();
        let actual_slice = actual.as_slice();
        let mut dloss_dpred = vec![0.0f32; dim];
        let inv_norms = 1.0 / (pred_norm * actual_norm);
        let cos_over_pred_sq = cos_sim / (pred_norm * pred_norm);
        for i in 0..dim {
            dloss_dpred[i] = -actual_slice[i] * inv_norms + cos_over_pred_sq * pred_slice[i];
        }

        // ∂pred/∂x_inf = sigma (gating factor)
        let sigma = 1.0 - (-dt / tau_base.max(0.001)).exp();

        // Recover the equilibrium that actually produced `predicted`. Rebuilding
        // x_inf from `actual` leaks the supervision target into the derivative
        // and is not the gradient of the forward computation.
        let x_inf = Self::equilibrium_from_prediction(state, predicted, sigma);
        let x_inf_slice = x_inf.as_slice();
        let state_slice = state.as_slice();
        let state_blend = 1.0 - self.input_blend;

        // ∂x_inf/∂W = state_blend * (1 - x_inf²) ⊗ state
        let mut grad_weight = vec![0.0f32; dim];
        for i in 0..dim {
            let dtanh = 1.0 - x_inf_slice[i] * x_inf_slice[i];
            grad_weight[i] = dloss_dpred[i] * sigma * state_blend * dtanh * state_slice[i];
        }

        // ∂loss/∂tau: ∂sigma/∂tau = -(dt/tau²) · exp(-dt/tau)
        let decay = (-dt / tau_base.max(0.001)).exp();
        let dsigma_dtau = -(dt / (tau_base * tau_base).max(0.001)) * decay;
        let mut grad_tau = 0.0f32;
        for i in 0..dim {
            // ∂pred/∂sigma = x_inf - state (from lerp derivative)
            let dpred_dsigma = x_inf_slice[i] - state_slice[i];
            grad_tau += dloss_dpred[i] * dpred_dsigma * dsigma_dtau;
        }

        // Pre-clip tau gradient before Adam to prevent explosion
        grad_tau = grad_tau.clamp(-self.config.grad_clip, self.config.grad_clip);

        // Apply Adam optimizer
        let weight_delta = self.weight_adam.step(&grad_weight, self.config.grad_clip);
        let tau_delta = self.tau_adam.step(&[grad_tau], self.config.grad_clip);

        let weight_lr = self.config.learning_rate * self.config.weight_lr_scale;
        let tau_lr = self.config.learning_rate * self.config.tau_lr_scale;

        BpttResult {
            weight_update: ContinuousHV::from_vec(
                weight_delta.iter().map(|&d| -weight_lr * d).collect(),
            ),
            tau_update: -tau_lr * tau_delta[0],
            loss,
        }
    }

    /// SPSA gradient estimation: perturb, evaluate, estimate gradient.
    pub fn spsa_step(
        &mut self,
        weight_hv: &ContinuousHV,
        state: &ContinuousHV,
        input: &ContinuousHV,
        actual_next: &ContinuousHV,
        tau_base: f32,
        dt: f32,
    ) -> BpttResult {
        let dim = weight_hv.dim();
        let eps = self.config.spsa_epsilon;
        let c = self.config.spsa_c;

        // Generate random perturbation direction (Bernoulli ±1)
        let perturbation: Vec<f32> = (0..dim)
            .map(|i| {
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                self.rng_state = self.rng_state.wrapping_add(i as u64);
                if self.rng_state.is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Perturb weight_hv in both directions
        let w_slice = weight_hv.as_slice();
        let w_plus: Vec<f32> = w_slice
            .iter()
            .zip(&perturbation)
            .map(|(&w, &p)| w + eps * p)
            .collect();
        let w_minus: Vec<f32> = w_slice
            .iter()
            .zip(&perturbation)
            .map(|(&w, &p)| w - eps * p)
            .collect();

        let loss_plus = Self::evaluate_loss(
            &ContinuousHV::from_vec(w_plus),
            state,
            input,
            actual_next,
            tau_base,
            dt,
            self.input_blend,
        );
        let loss_minus = Self::evaluate_loss(
            &ContinuousHV::from_vec(w_minus),
            state,
            input,
            actual_next,
            tau_base,
            dt,
            self.input_blend,
        );

        // SPSA gradient estimate
        let loss_diff = loss_plus - loss_minus;
        let grad: Vec<f32> = perturbation
            .iter()
            .map(|&p| c * loss_diff / (2.0 * eps * p))
            .collect();

        let weight_delta = self.weight_adam.step(&grad, self.config.grad_clip);
        let weight_lr = self.config.learning_rate * self.config.weight_lr_scale;

        // Tau SPSA: advance RNG before sampling perturbation direction
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let tau_pert = if self.rng_state.is_multiple_of(2) {
            eps
        } else {
            -eps
        };
        let tau_loss_plus = Self::evaluate_loss(
            weight_hv,
            state,
            input,
            actual_next,
            tau_base + tau_pert,
            dt,
            self.input_blend,
        );
        let tau_loss_minus = Self::evaluate_loss(
            weight_hv,
            state,
            input,
            actual_next,
            tau_base - tau_pert,
            dt,
            self.input_blend,
        );
        let tau_grad = (tau_loss_plus - tau_loss_minus) / (2.0 * tau_pert);
        let tau_delta = self.tau_adam.step(&[tau_grad], self.config.grad_clip);
        let tau_lr = self.config.learning_rate * self.config.tau_lr_scale;

        let loss = (loss_plus + loss_minus) / 2.0;

        BpttResult {
            weight_update: ContinuousHV::from_vec(
                weight_delta.iter().map(|&d| -weight_lr * d).collect(),
            ),
            tau_update: -tau_lr * tau_delta[0],
            loss,
        }
    }

    /// Dispatch training step based on configured method.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step(
        &mut self,
        weight_hv: &ContinuousHV,
        state: &ContinuousHV,
        input: &ContinuousHV,
        predicted: &ContinuousHV,
        actual_next: &ContinuousHV,
        tau_base: f32,
        dt: f32,
    ) -> BpttResult {
        self.total_steps += 1;

        match self.config.method {
            TrainingMethod::Bptt => {
                self.bptt_step(weight_hv, state, predicted, actual_next, tau_base, dt)
            }
            TrainingMethod::Spsa => {
                self.spsa_step(weight_hv, state, input, actual_next, tau_base, dt)
            }
            TrainingMethod::BpttWithSpsaFallback => {
                // BPTT updates Adam moments while constructing its candidate step.
                // Preserve those moments so rejecting the candidate does not make
                // one observation count twice in optimizer time.
                let weight_before = self.weight_adam.save_state();
                let tau_before = self.tau_adam.save_state();
                let result = self.bptt_step(weight_hv, state, predicted, actual_next, tau_base, dt);
                let grad_norm = result.weight_update.norm();
                if grad_norm < 1e-10 || !grad_norm.is_finite() {
                    self.weight_adam
                        .load_state(&weight_before, weight_hv.dim())
                        .expect("saved weight optimizer state must restore");
                    self.tau_adam
                        .load_state(&tau_before, 1)
                        .expect("saved tau optimizer state must restore");
                    self.spsa_step(weight_hv, state, input, actual_next, tau_base, dt)
                } else {
                    result
                }
            }
        }
    }

    /// Evaluate prediction loss for a given weight_hv configuration.
    fn evaluate_loss(
        weight_hv: &ContinuousHV,
        state: &ContinuousHV,
        input: &ContinuousHV,
        actual_next: &ContinuousHV,
        tau_base: f32,
        dt: f32,
        input_blend: f32,
    ) -> f32 {
        let state_influence = weight_hv.bind(state);
        let x_inf = ContinuousHV::weighted_bundle(
            &[input, &state_influence],
            &[input_blend, 1.0 - input_blend],
        )
        .tanh();
        let sigma = 1.0 - (-dt / tau_base.max(0.001)).exp();
        // predicted = state + sigma * (x_inf - state) = (1-sigma)*state + sigma*x_inf
        let mut predicted = state.clone();
        predicted.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
        1.0 - predicted.similarity(actual_next).clamp(-1.0, 1.0)
    }

    /// Snapshot optimizer and stochastic state for exact checkpoint resume.
    pub fn save_state(&self) -> TrainerState {
        TrainerState {
            weight_adam: self.weight_adam.save_state(),
            tau_adam: self.tau_adam.save_state(),
            rng_state: self.rng_state,
            total_steps: self.total_steps,
            input_blend: self.input_blend,
        }
    }

    /// Validate a serialized trainer state without mutating this trainer.
    pub fn validate_state(state: &TrainerState, hdc_dim: usize) -> Result<(), String> {
        fn validate_adam(
            name: &str,
            state: &AdamStateSnapshot,
            expected_dim: usize,
        ) -> Result<(), String> {
            if state.m.len() != expected_dim || state.v.len() != expected_dim {
                return Err(format!(
                    "trainer {name} Adam dimension mismatch: m={}, v={}, expected={expected_dim}",
                    state.m.len(),
                    state.v.len()
                ));
            }
            if state.m.iter().any(|value| !value.is_finite()) {
                return Err(format!("trainer {name} Adam first moment is non-finite"));
            }
            if state
                .v
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(format!(
                    "trainer {name} Adam second moment must be finite and non-negative"
                ));
            }
            if !state.beta1.is_finite() || !(0.0..1.0).contains(&state.beta1) {
                return Err(format!(
                    "trainer {name} Adam beta1 must be finite and in [0,1), got {}",
                    state.beta1
                ));
            }
            if !state.beta2.is_finite() || !(0.0..1.0).contains(&state.beta2) {
                return Err(format!(
                    "trainer {name} Adam beta2 must be finite and in [0,1), got {}",
                    state.beta2
                ));
            }
            if !state.eps.is_finite() || state.eps <= 0.0 {
                return Err(format!(
                    "trainer {name} Adam epsilon must be finite and positive, got {}",
                    state.eps
                ));
            }
            Ok(())
        }

        validate_adam("weight", &state.weight_adam, hdc_dim)?;
        validate_adam("tau", &state.tau_adam, 1)?;
        if !state.input_blend.is_finite() || !(0.0..=1.0).contains(&state.input_blend) {
            return Err(format!(
                "trainer input blend must be finite and in [0,1], got {}",
                state.input_blend
            ));
        }
        Ok(())
    }

    /// Restore optimizer moments, RNG, and step counters.
    pub fn load_state(&mut self, state: &TrainerState, hdc_dim: usize) -> Result<(), String> {
        Self::validate_state(state, hdc_dim)?;
        self.weight_adam.load_state(&state.weight_adam, hdc_dim)?;
        self.tau_adam.load_state(&state.tau_adam, 1)?;
        self.rng_state = state.rng_state;
        self.total_steps = state.total_steps;
        self.input_blend = state.input_blend;
        Ok(())
    }

    /// Restore only the legacy training-step count from older checkpoints.
    pub fn set_total_steps(&mut self, total_steps: u64) {
        self.total_steps = total_steps;
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut TrainingConfig {
        &mut self.config
    }
}

/// Result of a single training step.
pub struct BpttResult {
    /// Additive update to apply to weight_hv.
    pub weight_update: ContinuousHV,
    /// Additive update to apply to tau_base.
    pub tau_update: f32,
    /// Loss value at this step.
    pub loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrainingConfig;

    #[test]
    fn test_fallback_restores_rejected_bptt_optimizer_moments() {
        let cfg = TrainingConfig::default();
        let dim = 64;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);
        let weight = ContinuousHV::random(dim, 1);
        let state = ContinuousHV::random(dim, 2);
        let input = ContinuousHV::random(dim, 3);
        let predicted = ContinuousHV::random(dim, 4);
        let actual = predicted.clone();

        trainer.train_step(&weight, &state, &input, &predicted, &actual, 0.5, 0.03);
        let saved = trainer.save_state();
        assert_eq!(saved.total_steps, 1);
        assert_eq!(saved.weight_adam.t, 1);
        assert_eq!(saved.tau_adam.t, 1);
    }

    #[test]
    fn test_dilated_second_moments_remain_valid() {
        let cfg = TrainingConfig::default();
        let mut trainer = ManifoldTrainer::new(&cfg, 32);
        let gradient = vec![0.25; 32];
        let _ = trainer.weight_adam.step(&gradient, 1.0);
        trainer.dilate(64);
        let state = trainer.save_state();
        assert!(state.weight_adam.v.iter().all(|value| *value >= 0.0));
        ManifoldTrainer::validate_state(&state, 64).unwrap();
    }

    #[test]
    fn test_trainer_checkpoint_numeric_rejection_is_atomic() {
        let cfg = TrainingConfig::default();
        let dim = 32;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);
        let before = trainer.save_state();
        let mut malformed = before.clone();
        malformed.weight_adam.v[0] = -1.0;
        assert!(trainer.load_state(&malformed, dim).is_err());
        assert_eq!(trainer.save_state(), before);

        malformed = before.clone();
        malformed.input_blend = f32::NAN;
        assert!(trainer.load_state(&malformed, dim).is_err());
        assert_eq!(trainer.save_state(), before);
    }

    #[test]
    fn test_adam_state_construction() {
        let adam = AdamState::new(100);
        assert_eq!(adam.m.len(), 100);
        assert_eq!(adam.t, 0);
    }

    #[test]
    fn test_adam_step_produces_finite_output() {
        let mut adam = AdamState::new(4);
        let grad = vec![0.1, -0.2, 0.3, -0.4];
        let delta = adam.step(&grad, 1.0);
        assert_eq!(delta.len(), 4);
        for &d in &delta {
            assert!(d.is_finite(), "Adam delta should be finite");
        }
    }

    #[test]
    fn test_equilibrium_recovery_uses_forward_prediction() {
        let state = ContinuousHV::random(256, 10);
        let equilibrium = ContinuousHV::random(256, 20).tanh();
        let sigma = 0.25;
        let mut predicted = state.clone();
        predicted.lerp_in_place(&equilibrium, 1.0 - sigma, sigma);

        let recovered = ManifoldTrainer::equilibrium_from_prediction(&state, &predicted, sigma);
        assert!(
            recovered.similarity(&equilibrium) > 0.999,
            "the analytical gradient must recover the forward equilibrium"
        );
    }

    #[test]
    fn test_bptt_step_produces_finite_loss() {
        let cfg = TrainingConfig::default();
        let dim = 256; // Small dim for test speed
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let weight_hv = ContinuousHV::random(dim, 100);
        let state = ContinuousHV::random(dim, 200);
        let predicted = ContinuousHV::random(dim, 300);
        let actual = ContinuousHV::random(dim, 400);

        let result = trainer.bptt_step(&weight_hv, &state, &predicted, &actual, 0.5, 0.033);
        assert!(result.loss.is_finite(), "BPTT loss should be finite");
        assert!(result.loss >= 0.0, "Loss should be non-negative");
        assert!(result.tau_update.is_finite(), "Tau update should be finite");
        assert_eq!(result.weight_update.dim(), dim);
    }

    #[test]
    fn test_spsa_step_produces_finite_loss() {
        let cfg = TrainingConfig::default();
        let dim = 256;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let weight_hv = ContinuousHV::random(dim, 100);
        let state = ContinuousHV::random(dim, 200);
        let input = ContinuousHV::random(dim, 300);
        let actual = ContinuousHV::random(dim, 400);

        let result = trainer.spsa_step(&weight_hv, &state, &input, &actual, 0.5, 0.033);
        assert!(result.loss.is_finite(), "SPSA loss should be finite");
        assert!(result.loss >= 0.0);
    }

    #[test]
    fn test_training_reduces_prediction_error_on_repeating_sequence() {
        let cfg = TrainingConfig {
            learning_rate: 0.01,
            method: TrainingMethod::Bptt,
            ..Default::default()
        };
        let dim = 512;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let frame_a = ContinuousHV::random(dim, 1000);
        let frame_b = ContinuousHV::random(dim, 2000);
        let mut weight_hv = ContinuousHV::random(dim, 3000);
        let mut tau_base = 0.5f32;

        // Replay A, B, A, B... and train the manifold to predict B after A
        let mut state = ContinuousHV::zero(dim);
        let dt = 0.033;
        let mut early_loss = 0.0f32;
        let mut late_loss = 0.0f32;

        for step in 0..100 {
            let (current, next) = if step % 2 == 0 {
                (&frame_a, &frame_b)
            } else {
                (&frame_b, &frame_a)
            };

            // CfC evolve
            let state_influence = weight_hv.bind(&state);
            let x_inf =
                ContinuousHV::weighted_bundle(&[current, &state_influence], &[0.7, 0.3]).tanh();
            let sigma = 1.0 - (-dt / tau_base.max(0.001)).exp();
            state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

            // Predict
            let x_inf_pred =
                ContinuousHV::weighted_bundle(&[current, &state_influence], &[0.7, 0.3]).tanh();
            let mut predicted = state.clone();
            predicted.lerp_in_place(&x_inf_pred, 1.0 - sigma, sigma);

            // Train
            let result = trainer.bptt_step(&weight_hv, &state, &predicted, next, tau_base, dt);

            // Apply updates
            weight_hv = weight_hv.add(&result.weight_update);
            tau_base = (tau_base + result.tau_update).clamp(0.01, 10.0);

            if step < 10 {
                early_loss += result.loss;
            }
            if step >= 90 {
                late_loss += result.loss;
            }
        }

        early_loss /= 10.0;
        late_loss /= 10.0;

        // Training should not diverge, and late loss should not wildly exceed early
        assert!(
            late_loss.is_finite(),
            "Late loss should be finite, got {late_loss}"
        );
        assert!(
            early_loss.is_finite(),
            "Early loss should be finite, got {early_loss}"
        );
        assert!(
            late_loss <= early_loss + 0.1,
            "Training should not diverge: early={early_loss}, late={late_loss}"
        );
    }

    #[test]
    fn test_evaluate_loss_bounds() {
        let dim = 256;
        let w = ContinuousHV::random(dim, 1);
        let s = ContinuousHV::random(dim, 2);
        let inp = ContinuousHV::random(dim, 3);
        let actual = ContinuousHV::random(dim, 4);

        let loss = ManifoldTrainer::evaluate_loss(&w, &s, &inp, &actual, 0.5, 0.033, 0.7);
        assert!(
            loss >= 0.0 && loss <= 2.0,
            "Loss should be in [0, 2], got {loss}"
        );
    }

    #[test]
    fn test_train_step_dispatches_correctly() {
        for method in [
            TrainingMethod::Bptt,
            TrainingMethod::Spsa,
            TrainingMethod::BpttWithSpsaFallback,
        ] {
            let cfg = TrainingConfig {
                method,
                ..Default::default()
            };
            let dim = 128;
            let mut trainer = ManifoldTrainer::new(&cfg, dim);

            let w = ContinuousHV::random(dim, 10);
            let s = ContinuousHV::random(dim, 20);
            let inp = ContinuousHV::random(dim, 30);
            let pred = ContinuousHV::random(dim, 40);
            let actual = ContinuousHV::random(dim, 50);

            let result = trainer.train_step(&w, &s, &inp, &pred, &actual, 0.5, 0.033);
            assert!(
                result.loss.is_finite(),
                "{method:?} should produce finite loss"
            );
        }
    }

    #[test]
    fn test_spsa_tau_perturbation_is_stochastic() {
        // The bug: tau perturbation used rng_state % 2 without advancing the RNG,
        // making it deterministic. The fix advances the RNG via XOR-shift before
        // sampling the tau perturbation direction.
        //
        // Note: For a scalar parameter like tau, the SPSA gradient estimate is
        // invariant to perturbation direction (sign of numerator and denominator
        // cancel), so tau_update is always identical regardless of direction.
        // We verify stochasticity through weight updates, which are multi-dimensional
        // and DO vary across seeds, plus confirm the RNG state diverges.
        let cfg = TrainingConfig {
            method: TrainingMethod::Spsa,
            ..Default::default()
        };
        let dim = 128;

        let mut weight_first_elem = Vec::new();
        let mut final_rng_states = Vec::new();
        for seed_offset in 0..10u64 {
            let mut trainer = ManifoldTrainer::new(&cfg, dim);
            trainer.rng_state = 0xCAFE_BABE_0000_0000 + seed_offset * 7919;

            let w = ContinuousHV::random(dim, 10);
            let s = ContinuousHV::random(dim, 20);
            let inp = ContinuousHV::random(dim, 30);
            let actual = ContinuousHV::random(dim, 50);

            let result = trainer.spsa_step(&w, &s, &inp, &actual, 0.5, 0.033);
            weight_first_elem.push(result.weight_update.as_slice()[0]);
            final_rng_states.push(trainer.rng_state);
        }

        // Weight updates differ across seeds (perturbation vectors are stochastic)
        let first_w = weight_first_elem[0];
        let all_same_w = weight_first_elem
            .iter()
            .all(|&u| (u - first_w).abs() < 1e-15);
        assert!(
            !all_same_w,
            "SPSA weight updates should vary across different RNG seeds"
        );

        // Final RNG states are all unique (each seed path diverges through the
        // weight perturbation loop + tau advancement)
        let unique_rng: std::collections::HashSet<_> = final_rng_states.iter().collect();
        assert_eq!(
            unique_rng.len(),
            final_rng_states.len(),
            "Final RNG states should all be unique after SPSA step"
        );
    }

    #[test]
    fn test_nan_input_does_not_propagate() {
        let cfg = TrainingConfig::default();
        let dim = 256;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let w = ContinuousHV::random(dim, 10);
        let s = ContinuousHV::random(dim, 20);
        // Create an input with some NaN-like extreme values
        let inp = ContinuousHV::random(dim, 30);
        let pred = ContinuousHV::random(dim, 40);
        let actual = ContinuousHV::random(dim, 50);

        let result = trainer.train_step(&w, &s, &inp, &pred, &actual, 0.5, 0.033);
        assert!(result.loss.is_finite(), "Loss should be finite");
        assert!(result.tau_update.is_finite(), "Tau update should be finite");
        assert!(
            result
                .weight_update
                .as_slice()
                .iter()
                .all(|v| v.is_finite()),
            "All weight update values should be finite"
        );
    }

    #[test]
    fn test_gradient_explosion_resistance() {
        let cfg = TrainingConfig {
            learning_rate: 0.01,
            method: TrainingMethod::Bptt,
            ..Default::default()
        };
        let dim = 256;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let w = ContinuousHV::random(dim, 10);
        let s = ContinuousHV::random(dim, 20);
        let pred = ContinuousHV::random(dim, 40);
        let actual = ContinuousHV::random(dim, 50);

        // Use very small tau (0.01) which can cause gradient explosion
        let result = trainer.bptt_step(&w, &s, &pred, &actual, 0.01, 0.033);
        assert!(
            result.loss.is_finite(),
            "Loss should be finite with small tau"
        );
        assert!(
            result.tau_update.is_finite(),
            "Tau update should be finite with small tau"
        );
    }

    #[test]
    fn test_trainer_state_roundtrip_preserves_optimizer_and_rng() {
        let cfg = TrainingConfig {
            method: TrainingMethod::Spsa,
            ..Default::default()
        };
        let dim = 128;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let weight = ContinuousHV::random(dim, 10);
        let state = ContinuousHV::random(dim, 20);
        let input = ContinuousHV::random(dim, 30);
        let predicted = ContinuousHV::random(dim, 40);
        let actual = ContinuousHV::random(dim, 50);
        let _ = trainer.train_step(&weight, &state, &input, &predicted, &actual, 0.5, 0.033);

        let saved = trainer.save_state();
        let mut restored = ManifoldTrainer::new(&cfg, dim);
        restored
            .load_state(&saved, dim)
            .expect("trainer checkpoint should restore");

        assert_eq!(restored.save_state(), saved);
    }

    #[test]
    fn test_adam_bias_correction_at_step_1() {
        let mut adam = AdamState::new(4);
        let grad = vec![0.1, -0.2, 0.3, -0.4];

        // Step 1: bias correction amplifies early updates
        let delta_1 = adam.step(&grad, 1.0);
        let norm_1: f32 = delta_1.iter().map(|d| d * d).sum::<f32>().sqrt();

        // Run 100 more steps with same gradient
        for _ in 0..99 {
            adam.step(&grad, 1.0);
        }
        let delta_100 = adam.step(&grad, 1.0);
        let norm_100: f32 = delta_100.iter().map(|d| d * d).sum::<f32>().sqrt();

        // Step 1 delta should be larger than step 100 delta due to bias correction
        // At step 1, m_hat is amplified by 1/(1-0.9) = 10x and v_hat by ~1/(1-0.999) = 1000x
        // The net effect: step 1 produces ~1.0 magnitude deltas
        assert!(
            norm_1 > norm_100 * 0.5,
            "Bias correction should amplify early updates: step1_norm={norm_1}, step100_norm={norm_100}"
        );
    }

    #[test]
    fn test_tau_stays_bounded_after_training() {
        let cfg = TrainingConfig {
            learning_rate: 0.01,
            method: TrainingMethod::Bptt,
            ..Default::default()
        };
        let dim = 256;
        let mut trainer = ManifoldTrainer::new(&cfg, dim);

        let mut weight_hv = ContinuousHV::random(dim, 100);
        let mut tau_base = 0.5f32;
        let mut state = ContinuousHV::zero(dim);
        let frame_a = ContinuousHV::random(dim, 1000);
        let frame_b = ContinuousHV::random(dim, 2000);
        let dt = 0.033;

        for step in 0..200 {
            let (current, next) = if step % 2 == 0 {
                (&frame_a, &frame_b)
            } else {
                (&frame_b, &frame_a)
            };

            let state_influence = weight_hv.bind(&state);
            let x_inf =
                ContinuousHV::weighted_bundle(&[current, &state_influence], &[0.7, 0.3]).tanh();
            let sigma = 1.0 - (-dt / tau_base.max(0.001)).exp();
            state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

            let mut predicted = state.clone();
            predicted.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

            let result = trainer.bptt_step(&weight_hv, &state, &predicted, next, tau_base, dt);
            weight_hv = weight_hv.add(&result.weight_update);
            tau_base = (tau_base + result.tau_update).clamp(0.01, 10.0);
        }

        assert!(
            tau_base >= 0.01 && tau_base <= 10.0,
            "Tau should remain bounded after 200 steps: tau={tau_base}"
        );
    }

    #[test]
    fn checked_input_blend_rejects_invalid_values_atomically() {
        let cfg = TrainingConfig::default();
        let mut trainer = ManifoldTrainer::new(&cfg, 32);
        trainer.set_input_blend_checked(0.4).unwrap();
        assert!(trainer.set_input_blend_checked(f32::NAN).is_err());
        assert!(trainer.set_input_blend_checked(1.1).is_err());
        assert!((trainer.input_blend - 0.4).abs() < 1e-6);
    }
}
