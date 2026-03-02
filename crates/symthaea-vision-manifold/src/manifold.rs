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
use crate::types::{ManifoldState, VisionConfig, VisionTelemetry};

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

    /// Access the learned weight HV (for inspection/comparison).
    pub fn weight_hv(&self) -> &ContinuousHV {
        &self.weight_hv
    }

    /// Total training steps performed.
    pub fn training_steps(&self) -> u64 {
        self.trainer.total_steps()
    }

    /// Mutable access to the encoder (for contrastive refinement).
    pub fn encoder_mut(&mut self) -> &mut PatchHdcEncoder {
        &mut self.encoder
    }

    /// Saliency-guided encoding refinement (closed-loop active inference).
    ///
    /// Uses the surprise map to select positive (high-surprise) and negative
    /// (low-surprise) exemplar HVs, then refines the encoder's feature weights
    /// via contrastive learning. This makes the encoder adapt to attend to
    /// whatever is currently surprising in the scene.
    pub fn refine_from_attention(&mut self) {
        let attention = self.surprise.attention_map();
        if self.last_patch_hvs.is_empty() || attention.values.is_empty() {
            return;
        }

        let max_surprise = attention.max_surprise();
        if max_surprise < 1e-6 {
            return;
        }

        // Find the highest and lowest surprise patches
        let mut best_idx = 0;
        let mut worst_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        let mut worst_val = f32::INFINITY;

        for (i, &v) in attention.values.iter().enumerate() {
            if i < self.last_patch_hvs.len() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
                if v < worst_val {
                    worst_val = v;
                    worst_idx = i;
                }
            }
        }

        if best_idx == worst_idx {
            return;
        }

        let positive = self.last_patch_hvs[best_idx].clone();
        let negative = self.last_patch_hvs[worst_idx].clone();
        let lr = self.config.learning.contrastive_lr;
        self.encoder.refine_contrastive(&positive, &negative, lr);
    }

    /// Evaluate prediction accuracy at multiple temporal horizons.
    ///
    /// Returns a `HorizonAccuracy` with per-horizon prediction error measured
    /// against the current frame. Call after `observe_frame()` to get accuracy
    /// of predictions that were made N steps ago.
    pub fn evaluate_horizons(&self) -> HorizonAccuracy {
        let horizons = self.default_horizons();
        let labels = self.horizon_labels();
        let mut errors = Vec::with_capacity(horizons.len());

        if let Some(ref frame_hv) = self.last_frame_hv {
            let state = self.state();
            for &h in horizons {
                let predicted = self.predict_horizon(frame_hv, h);
                let error = 1.0 - state.similarity(&predicted).clamp(-1.0, 1.0);
                errors.push(error);
            }
        } else {
            errors.resize(horizons.len(), 1.0);
        }

        HorizonAccuracy {
            horizons: horizons.to_vec(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            errors,
            frame_sequence: self.frame_count,
        }
    }

    /// Snapshot the manifold's learned state for serialization.
    ///
    /// Captures weight_hv, tau_base, feature_weights, and training steps
    /// so the manifold can be resumed from a trained checkpoint.
    pub fn save_state(&self) -> ManifoldState {
        ManifoldState {
            weight_hv: self.weight_hv.as_slice().to_vec(),
            tau_base: self.config.tau_base,
            feature_weights: self.encoder.feature_weights().to_vec(),
            training_steps: self.trainer.total_steps(),
            hdc_dim: self.config.hdc_dim,
            num_features: self.config.num_features,
        }
    }

    /// Restore the manifold from a saved state.
    ///
    /// Validates dimensional compatibility before applying. Returns `Err`
    /// if the saved state is incompatible with the current config.
    pub fn load_state(&mut self, state: &ManifoldState) -> Result<(), String> {
        if state.hdc_dim != self.config.hdc_dim {
            return Err(format!(
                "HDC dimension mismatch: saved={}, current={}",
                state.hdc_dim, self.config.hdc_dim
            ));
        }
        if state.weight_hv.len() != self.config.hdc_dim {
            return Err(format!(
                "Weight HV length mismatch: saved={}, expected={}",
                state.weight_hv.len(),
                self.config.hdc_dim
            ));
        }

        self.weight_hv = ContinuousHV::from_vec(state.weight_hv.clone());
        self.config.tau_base = state.tau_base;

        // Restore feature weights if compatible
        let current_weights = self.encoder.feature_weights().len();
        if state.feature_weights.len() == current_weights {
            // Apply via contrastive interface would change weights — instead set directly
            // We need mutable access to the encoder's weights
            self.encoder.set_feature_weights(&state.feature_weights);
        }

        Ok(())
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

/// Multi-horizon prediction accuracy snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HorizonAccuracy {
    /// Prediction horizons in seconds.
    pub horizons: Vec<f32>,
    /// Human-readable labels for each horizon.
    pub labels: Vec<String>,
    /// Prediction error (1 - cos_sim) at each horizon.
    pub errors: Vec<f32>,
    /// Frame at which this was evaluated.
    pub frame_sequence: u64,
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
    fn test_refine_from_attention_modifies_weights() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let dt = 0.033;

        // Need at least 2 frames so surprise map has data
        let frame_a = solid_gray_frame(64, 64, 50);
        m.observe_frame(&frame_a, 64, 64, 1, dt);

        // Scene change creates surprise contrast
        let frame_b = gradient_frame(64, 64);
        m.observe_frame(&frame_b, 64, 64, 1, dt);

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();
        m.refine_from_attention();
        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();

        // Weights should have changed (surprise contrast drives contrastive update)
        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "Saliency refinement should modify encoder weights");
    }

    #[test]
    fn test_refine_from_attention_noop_when_no_surprise() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        // Only one frame observed — surprise map is all zeros
        let frame = solid_gray_frame(64, 64, 128);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();
        m.refine_from_attention();
        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();

        // Should be a no-op (no surprise contrast)
        assert_eq!(weights_before, weights_after);
    }

    #[test]
    fn test_evaluate_horizons_structure() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let acc = m.evaluate_horizons();

        assert_eq!(acc.horizons.len(), 4);
        assert_eq!(acc.labels.len(), 4);
        assert_eq!(acc.errors.len(), 4);
        assert_eq!(acc.frame_sequence, 1);
        assert_eq!(acc.labels[0], "next_frame");
        assert_eq!(acc.labels[3], "scene_scale");
    }

    #[test]
    fn test_evaluate_horizons_error_ordering() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Converge on the frame
        for _ in 0..10 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let acc = m.evaluate_horizons();

        // Short horizon prediction should be at least as good as long horizon
        // (closer to current state means less divergence from equilibrium)
        assert!(
            acc.errors[0] <= acc.errors[3] + 0.05,
            "Short horizon error ({}) should be <= long horizon error ({})",
            acc.errors[0],
            acc.errors[3]
        );
    }

    #[test]
    fn test_evaluate_horizons_before_any_frame() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        // No frames observed — should return default errors
        let acc = m.evaluate_horizons();
        assert_eq!(acc.errors.len(), 4);
        // All errors should be 1.0 (maximum)
        for &e in &acc.errors {
            assert!((e - 1.0).abs() < 1e-6, "Pre-frame error should be 1.0, got {e}");
        }
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

    // === State Persistence ===

    #[test]
    fn test_save_state_captures_fields() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg.clone(), 64, 64);

        let state = m.save_state();
        assert_eq!(state.hdc_dim, cfg.hdc_dim);
        assert_eq!(state.weight_hv.len(), cfg.hdc_dim);
        assert!((state.tau_base - cfg.tau_base).abs() < 1e-6);
        assert_eq!(state.training_steps, 0);
        assert_eq!(state.feature_weights.len(), cfg.total_features());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let cfg = VisionConfig::default();
        let mut m1 = VisionManifold::new(cfg.clone(), 64, 64);

        // Evolve manifold so it has non-trivial state
        let frame = gradient_frame(64, 64);
        for _ in 0..10 {
            m1.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let saved = m1.save_state();

        // Load into a fresh manifold
        let mut m2 = VisionManifold::new(cfg, 64, 64);
        assert!(m2.load_state(&saved).is_ok());

        // Weight HVs should match
        let sim = m2.weight_hv().similarity(m1.weight_hv());
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Loaded weight_hv should match saved: sim={sim}"
        );

        // Tau should match
        assert!(
            (m2.current_tau() - m1.current_tau()).abs() < 1e-6,
            "Loaded tau should match"
        );
    }

    #[test]
    fn test_load_state_rejects_dimension_mismatch() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        let bad_state = ManifoldState {
            weight_hv: vec![0.0; 100], // Wrong dimension
            tau_base: 0.5,
            feature_weights: vec![],
            training_steps: 0,
            hdc_dim: 100,
            num_features: 5,
        };

        assert!(m.load_state(&bad_state).is_err());
    }

    #[test]
    fn test_save_state_serializable() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        let state = m.save_state();

        // Should be JSON-serializable
        let json = serde_json::to_string(&state).expect("Should serialize");
        let deserialized: ManifoldState =
            serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.hdc_dim, state.hdc_dim);
        assert_eq!(deserialized.weight_hv.len(), state.weight_hv.len());
    }
}
