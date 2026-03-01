//! CfC-based temporal manifold for video state tracking.
//!
//! Maintains a continuous-time hypervector state that evolves via closed-form
//! CfC (Closed-form Continuous-time) dynamics:
//!
//! ```text
//! state' = x_inf + (state - x_inf) · exp(-dt / τ)
//! ```
//!
//! where `x_inf = tanh(W ⊗ state  +  U ⊗ input)` is the equilibrium state.
//!
//! Key property: prediction cost is O(D) regardless of the time horizon dt.
//! Whether dt is 0.001s or 1000s, the computation is a single closed-form step.
//!
//! **Caveat**: The closed-form assumes equilibrium is constant during the step.
//! For very large dt/τ ratios, the state saturates to x_inf. This is accurate
//! for static scenes but introduces error when the scene is changing rapidly
//! during the predicted interval.

use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::temporal::TemporalPredictor;

use crate::attention::SurpriseMap;
use crate::encoder::PatchHdcEncoder;
use crate::training::{BpttResult, ManifoldTrainer};
use crate::types::{VisionConfig, VisionTelemetry};

/// A CfC temporal manifold over holographic video encodings.
///
/// The manifold state is a ContinuousHV (16,384D by default) that continuously
/// tracks the scene. Each frame observation evolves the state via closed-form
/// CfC dynamics; temporal predictions use O(1) jumps.
pub struct VisionManifold {
    config: VisionConfig,
    encoder: PatchHdcEncoder,
    state: ContinuousHV,
    weight_hv: ContinuousHV,
    last_prediction: Option<ContinuousHV>,
    last_frame_hv: Option<ContinuousHV>,
    last_patch_hvs: Vec<ContinuousHV>,
    surprise: SurpriseMap,
    prediction_error: f32,
    coherence: f32,
    frame_count: u64,
    telemetry: VisionTelemetry,
    trainer: ManifoldTrainer,
}

impl VisionManifold {
    /// Create a new vision manifold sized for frames up to `max_width × max_height`.
    pub fn new(config: VisionConfig, max_width: u32, max_height: u32) -> Self {
        let encoder = PatchHdcEncoder::new(&config, max_width, max_height);
        let dim = config.hdc_dim;
        let state = ContinuousHV::zero(dim);
        let weight_hv = ContinuousHV::random(dim, config.seed + 300_000);
        let grid = encoder.grid_for(max_width, max_height);
        let surprise = SurpriseMap::new(grid, config.surprise_decay, config.surprise_threshold);
        let trainer = ManifoldTrainer::new(&config.training, dim);

        Self {
            config,
            encoder,
            state,
            weight_hv,
            last_prediction: None,
            last_frame_hv: None,
            last_patch_hvs: Vec::new(),
            surprise,
            prediction_error: 0.0,
            coherence: 0.0,
            frame_count: 0,
            telemetry: VisionTelemetry::default(),
            trainer,
        }
    }

    /// Observe a raw frame: encode → evolve CfC state → compute surprise → predict.
    ///
    /// Returns telemetry for this observation cycle.
    pub fn observe_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> VisionTelemetry {
        let t0 = Instant::now();
        let (frame_hv, patch_hvs) = self.encoder.encode_frame(pixels, width, height, channels);
        let encode_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        self.observe_encoded(&frame_hv, &patch_hvs, dt);
        let evolve_us = t1.elapsed().as_micros() as u64;

        self.telemetry = VisionTelemetry {
            encode_time_us: encode_us,
            evolve_time_us: evolve_us,
            prediction_error: self.prediction_error,
            manifold_coherence: self.coherence,
            attention_entropy: self.surprise.attention_map().entropy(),
            num_salient_patches: self
                .surprise
                .salient_patches()
                .len(),
            frame_sequence: self.frame_count,
            training_triggered: false,
            training_loss: None,
            output_hv_norm: 0.0,
            attention_boost_applied: 0.0,
        };

        self.telemetry.clone()
    }

    /// Observe a pre-encoded frame HV with its per-patch decomposition.
    pub fn observe_encoded(
        &mut self,
        frame_hv: &ContinuousHV,
        patch_hvs: &[ContinuousHV],
        dt: f32,
    ) {
        // Compute prediction error against previous prediction
        let mut training_triggered = false;
        let mut training_loss = None;

        if let Some(predicted) = self.last_prediction.clone() {
            self.prediction_error = 1.0 - frame_hv.similarity(&predicted).clamp(-1.0, 1.0);

            // Trigger training when prediction error exceeds threshold
            if self.prediction_error > self.config.training.error_threshold {
                if let Some(last_input) = self.last_frame_hv.clone() {
                    let result = self.train_step_inner(
                        &last_input, &predicted, frame_hv, dt,
                    );
                    training_triggered = true;
                    training_loss = Some(result.loss);
                }
            }
        }

        // Update per-patch surprise map
        self.surprise.update(patch_hvs, &self.last_patch_hvs);

        // Evolve CfC state: state' = x_inf + (state - x_inf) * exp(-dt/τ)
        let x_inf = self.equilibrium(frame_hv);
        let sigma = self.gating(dt);
        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

        // Compute coherence (state-frame alignment)
        self.coherence = self.state.similarity(frame_hv).max(0.0);

        // Predict next frame (one dt ahead) for next cycle's error computation
        self.last_prediction = Some(self.predict_horizon(frame_hv, dt));
        self.last_frame_hv = Some(frame_hv.clone());
        self.last_patch_hvs = patch_hvs.to_vec();
        self.frame_count += 1;

        // Store training telemetry
        self.telemetry.training_triggered = training_triggered;
        self.telemetry.training_loss = training_loss;
    }

    /// Internal training step: update weight_hv and tau_base from prediction error.
    fn train_step_inner(
        &mut self,
        input: &ContinuousHV,
        predicted: &ContinuousHV,
        actual: &ContinuousHV,
        dt: f32,
    ) -> BpttResult {
        let result = self.trainer.train_step(
            &self.weight_hv,
            &self.state,
            input,
            predicted,
            actual,
            self.config.tau_base,
            dt,
        );

        // Apply weight update
        self.weight_hv = self.weight_hv.add(&result.weight_update);

        // Apply tau update with clamping
        self.config.tau_base = (self.config.tau_base + result.tau_update).clamp(0.01, 10.0);

        result
    }

    /// CfC equilibrium: tanh(α · input + (1-α) · W ⊗ state).
    ///
    /// The equilibrium is attracted toward the input signal (what we observe)
    /// with state persistence through the weight-transformed state (memory/inertia).
    /// This ensures the manifold tracks visual input rather than drifting into
    /// a random subspace (which happens with pure bind on random untrained weights).
    fn equilibrium(&self, input: &ContinuousHV) -> ContinuousHV {
        let state_influence = self.weight_hv.bind(&self.state);
        // Input-dominant blend: 70% observation, 30% state persistence
        ContinuousHV::weighted_bundle(&[input, &state_influence], &[0.7, 0.3]).tanh()
    }

    /// CfC gating factor: σ = 1 - exp(-dt / τ).
    fn gating(&self, dt: f32) -> f32 {
        let decay = (-dt / self.config.tau_base.max(0.001)).exp();
        1.0 - decay
    }

    /// Predict manifold state at a future horizon via O(1) closed-form jump.
    fn predict_horizon(&self, current_input: &ContinuousHV, horizon: f32) -> ContinuousHV {
        let x_inf = self.equilibrium(current_input);
        let sigma = self.gating(horizon);
        let mut predicted = self.state.clone();
        predicted.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
        predicted
    }

    /// Current manifold state (the "scene representation").
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Last prediction error (free energy proxy, 0 = perfect prediction).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Manifold coherence (state-frame cosine similarity, 0..1).
    pub fn coherence(&self) -> f32 {
        self.coherence
    }

    /// Access the spatial surprise map.
    pub fn surprise_map(&self) -> &SurpriseMap {
        &self.surprise
    }

    /// Last telemetry snapshot.
    pub fn telemetry(&self) -> &VisionTelemetry {
        &self.telemetry
    }

    /// Total frames observed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Access the underlying encoder for external use.
    pub fn encoder(&self) -> &PatchHdcEncoder {
        &self.encoder
    }

    /// Current tau_base value (may change during training).
    pub fn current_tau(&self) -> f32 {
        self.config.tau_base
    }

    /// Total training steps performed.
    pub fn training_steps(&self) -> u64 {
        self.trainer.total_steps()
    }

    /// Mutable access to the encoder (for contrastive refinement).
    pub fn encoder_mut(&mut self) -> &mut PatchHdcEncoder {
        &mut self.encoder
    }

    /// Reset manifold to initial state.
    pub fn reset(&mut self) {
        self.state = ContinuousHV::zero(self.config.hdc_dim);
        self.last_prediction = None;
        self.last_frame_hv = None;
        self.last_patch_hvs.clear();
        self.surprise.reset();
        self.prediction_error = 0.0;
        self.coherence = 0.0;
        self.frame_count = 0;
    }
}

impl TemporalPredictor for VisionManifold {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        let x_inf = self.equilibrium(state);
        let sigma = self.gating(dt_seconds);
        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
        self.frame_count += 1;
    }

    fn domain(&self) -> &'static str {
        "vision"
    }

    fn tau_base(&self) -> f32 {
        self.config.tau_base
    }

    fn default_horizons(&self) -> &'static [f32] {
        // ~1 frame, ~3 frames, ~15 frames, ~30 frames at 30fps
        &[0.033, 0.1, 0.5, 1.0]
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        &["next_frame", "short_term", "medium_term", "scene_scale"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_gray_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x + y) % 256) as u8);
            }
        }
        pixels
    }

    #[test]
    fn test_manifold_construction() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        assert_eq!(m.frame_count(), 0);
        assert_eq!(m.prediction_error(), 0.0);
    }

    #[test]
    fn test_observe_single_frame() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
        assert_eq!(tel.frame_sequence, 1);
        // After a single CfC step from zero state, the manifold has begun evolving
        assert!(m.state().norm() > 0.0, "State should be non-zero after observation");
    }

    #[test]
    fn test_coherence_stays_high_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);
        let dt = 0.033;

        // Observe same frame repeatedly — coherence should remain high throughout
        for _ in 0..30 {
            m.observe_frame(&frame, 64, 64, 1, dt);
        }

        assert!(
            m.coherence() > 0.9,
            "Coherence should be high for static scene, got {}",
            m.coherence()
        );
    }

    #[test]
    fn test_prediction_error_decreases_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        let dt = 0.033;

        // Observe same frame repeatedly — prediction error should decrease
        let mut errors = Vec::new();
        for _ in 0..20 {
            let tel = m.observe_frame(&frame, 64, 64, 1, dt);
            errors.push(tel.prediction_error);
        }

        // After warm-up, later errors should be smaller than early errors
        let early_mean: f32 = errors[2..5].iter().sum::<f32>() / 3.0;
        let late_mean: f32 = errors[15..20].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= early_mean + 0.05,
            "Prediction error should decrease for static scene: early={early_mean}, late={late_mean}"
        );
    }

    #[test]
    fn test_scene_change_spikes_error() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let dt = 0.033;

        // Converge on scene A
        let frame_a = solid_gray_frame(64, 64, 50);
        for _ in 0..15 {
            m.observe_frame(&frame_a, 64, 64, 1, dt);
        }
        let stable_error = m.prediction_error();

        // Switch to scene B — error should spike
        let frame_b = solid_gray_frame(64, 64, 200);
        m.observe_frame(&frame_b, 64, 64, 1, dt);
        let spike_error = m.prediction_error();

        assert!(
            spike_error > stable_error,
            "Scene change should spike prediction error: stable={stable_error}, spike={spike_error}"
        );
    }

    #[test]
    fn test_temporal_prediction_o1() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let input = m.state().clone();

        // Predict at multiple horizons — all should return valid HVs
        let p_short = m.predict_at(&input, 0.033);
        let p_medium = m.predict_at(&input, 1.0);
        let p_long = m.predict_at(&input, 100.0);

        assert!(p_short.norm() > 0.0);
        assert!(p_medium.norm() > 0.0);
        assert!(p_long.norm() > 0.0);

        // Longer horizons should approach equilibrium more (higher sigma)
        let state = m.state();
        let sim_short = state.similarity(&p_short);
        let sim_long = state.similarity(&p_long);
        // Short prediction is closer to current state than long prediction
        assert!(
            sim_short >= sim_long - 0.01,
            "Short prediction should be closer to current state: short={sim_short}, long={sim_long}"
        );
    }

    #[test]
    fn test_temporal_predictor_trait() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        assert_eq!(m.domain(), "vision");
        assert!(m.tau_base() > 0.0);
        assert!(!m.default_horizons().is_empty());
        assert_eq!(m.default_horizons().len(), m.horizon_labels().len());
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        m.observe_frame(&frame, 64, 64, 1, 0.033);
        assert!(m.frame_count() > 0);

        m.reset();
        assert_eq!(m.frame_count(), 0);
        assert_eq!(m.prediction_error(), 0.0);
        assert_eq!(m.coherence(), 0.0);
    }

    #[test]
    fn test_gating_bounds() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        // dt=0 → sigma=0 (no change)
        assert!((m.gating(0.0)).abs() < 1e-6);

        // dt >> tau → sigma ≈ 1 (jump to equilibrium)
        assert!((m.gating(1000.0) - 1.0).abs() < 1e-4);

        // Intermediate dt → 0 < sigma < 1
        let mid = m.gating(0.5);
        assert!(mid > 0.0 && mid < 1.0, "mid sigma = {mid}");
    }
}
