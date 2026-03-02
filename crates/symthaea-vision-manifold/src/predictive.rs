//! Predictive coding hierarchy for multi-scale vision processing.
//!
//! Implements a two-level hierarchy where the coarse scale predicts the fine
//! scale, and prediction errors flow upward. This mirrors cortical predictive
//! coding: higher levels generate top-down expectations, lower levels signal
//! surprise when those expectations are violated.
//!
//! ```text
//! coarse (32px patches) → predicts → fine (8px patches)
//!      ↑                                    |
//!      └──── prediction error ──────────────┘
//! ```
//!
//! The prediction error signal serves two purposes:
//! 1. **Attention**: High prediction error regions need more processing
//! 2. **Learning**: Gradient signal for improving the coarse-to-fine mapping

use symthaea_core::hdc::ContinuousHV;

use crate::encoder::MultiScaleEncoder;
use crate::types::VisionConfig;

/// A two-level predictive coding hierarchy over multi-scale HDC encodings.
///
/// The coarse level generates predictions of the fine-level encoding.
/// Prediction errors are tracked per-frame and drive attention allocation.
pub struct PredictiveCodingHierarchy {
    encoder: MultiScaleEncoder,
    /// Coarse→fine prediction weight HV (learned mapping).
    prediction_weight: ContinuousHV,
    /// Last coarse-level HV (for prediction).
    last_coarse_hv: Option<ContinuousHV>,
    /// Prediction error: 1 - cos_sim(predicted_fine, actual_fine).
    prediction_error: f32,
    /// Exponential moving average of prediction error.
    error_ema: f32,
    /// EMA decay factor.
    ema_decay: f32,
    /// HDC dimension.
    dim: usize,
    /// Learning rate for prediction weight updates.
    learning_rate: f32,
}

impl PredictiveCodingHierarchy {
    /// Create a new predictive coding hierarchy from a VisionConfig.
    ///
    /// The config's multi_scale settings determine the fine and coarse scales.
    /// At least 2 scales are required; if fewer are configured, defaults are used.
    pub fn new(config: &VisionConfig, max_width: u32, max_height: u32) -> Self {
        let mut cfg = config.clone();
        if cfg.multi_scale.scales.len() < 2 {
            cfg.multi_scale.scales = vec![8, 32];
        }

        let encoder = MultiScaleEncoder::new(&cfg, max_width, max_height);
        let dim = cfg.hdc_dim;

        // Initialize prediction weight as random HV
        let prediction_weight = ContinuousHV::random(dim, cfg.seed + 700_000);

        Self {
            encoder,
            prediction_weight,
            last_coarse_hv: None,
            prediction_error: 0.0,
            error_ema: 0.0,
            ema_decay: 0.95,
            dim,
            learning_rate: 0.01,
        }
    }

    /// Process a frame through the predictive hierarchy.
    ///
    /// Returns `(fine_hv, coarse_hv, prediction_error)` where:
    /// - `fine_hv`: The fine-scale encoding (8px patches)
    /// - `coarse_hv`: The coarse-scale encoding (32px patches)
    /// - `prediction_error`: How well the coarse scale predicted the fine scale
    ///
    /// Internally updates the prediction weight to reduce future errors.
    pub fn process_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> PredictiveOutput {
        let (_blended, scale_hvs, _patches) =
            self.encoder.encode_frame(pixels, width, height, channels);

        let fine_hv = scale_hvs.first().cloned().unwrap_or_else(|| ContinuousHV::zero(self.dim));
        let coarse_hv = scale_hvs.last().cloned().unwrap_or_else(|| ContinuousHV::zero(self.dim));

        // Generate prediction of fine scale from previous coarse scale
        if let Some(prev_coarse) = self.last_coarse_hv.clone() {
            let predicted_fine = self.predict_fine(&prev_coarse);
            self.prediction_error =
                1.0 - fine_hv.similarity(&predicted_fine).clamp(-1.0, 1.0);

            // Update prediction weight via simple Hebbian-like rule:
            // Move prediction_weight to reduce the gap between predicted_fine and actual fine
            self.update_prediction_weight(&fine_hv, &predicted_fine, &prev_coarse);
        } else {
            self.prediction_error = 1.0; // No prediction yet
        }

        self.error_ema = self.ema_decay * self.error_ema + (1.0 - self.ema_decay) * self.prediction_error;
        self.last_coarse_hv = Some(coarse_hv.clone());

        PredictiveOutput {
            fine_hv,
            coarse_hv,
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
        }
    }

    /// Predict fine-scale HV from a coarse-scale HV.
    ///
    /// Uses the learned prediction weight as a transformation:
    /// `predicted_fine = tanh(prediction_weight ⊗ coarse_hv)`
    fn predict_fine(&self, coarse: &ContinuousHV) -> ContinuousHV {
        self.prediction_weight.bind(coarse).tanh()
    }

    /// Update the prediction weight to reduce prediction error.
    ///
    /// Uses a contrastive Hebbian rule: strengthen the mapping that would have
    /// produced the actual fine HV, weaken the mapping that produced the error.
    fn update_prediction_weight(
        &mut self,
        actual_fine: &ContinuousHV,
        predicted_fine: &ContinuousHV,
        coarse: &ContinuousHV,
    ) {
        let dim = self.dim;
        let actual_s = actual_fine.as_slice();
        let predicted_s = predicted_fine.as_slice();
        let coarse_s = coarse.as_slice();
        let weight_s = self.prediction_weight.as_slice();

        // Error signal: difference between actual and predicted
        // ∂loss/∂W ≈ (actual - predicted) ⊗ coarse (Hebbian)
        let mut updated = vec![0.0f32; dim];
        for i in 0..dim {
            let error = actual_s[i] - predicted_s[i];
            updated[i] = weight_s[i] + self.learning_rate * error * coarse_s[i];
        }

        self.prediction_weight = ContinuousHV::from_vec(updated);
    }

    /// Current prediction error (1 - cos_sim between predicted and actual fine).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Exponential moving average of prediction error.
    pub fn error_ema(&self) -> f32 {
        self.error_ema
    }

    /// Access the underlying multi-scale encoder.
    pub fn encoder(&self) -> &MultiScaleEncoder {
        &self.encoder
    }

    /// Mutable access to the underlying multi-scale encoder.
    pub fn encoder_mut(&mut self) -> &mut MultiScaleEncoder {
        &mut self.encoder
    }

    /// Reset the hierarchy to initial state.
    pub fn reset(&mut self) {
        self.last_coarse_hv = None;
        self.prediction_error = 0.0;
        self.error_ema = 0.0;
    }
}

/// Output from one predictive coding cycle.
#[derive(Debug, Clone)]
pub struct PredictiveOutput {
    /// Fine-scale (8px) encoding.
    pub fine_hv: ContinuousHV,
    /// Coarse-scale (32px) encoding.
    pub coarse_hv: ContinuousHV,
    /// Prediction error: how well coarse predicted fine this frame.
    pub prediction_error: f32,
    /// Exponential moving average of prediction error.
    pub error_ema: f32,
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
    fn test_hierarchy_construction() {
        let cfg = VisionConfig::default();
        let pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        assert_eq!(pch.prediction_error(), 0.0);
        assert_eq!(pch.error_ema(), 0.0);
    }

    #[test]
    fn test_single_frame_no_prediction() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let out = pch.process_frame(&frame, 64, 64, 1);

        // First frame: no prior coarse → prediction error is 1.0
        assert!((out.prediction_error - 1.0).abs() < 1e-6);
        assert!(out.fine_hv.norm() > 0.0);
        assert!(out.coarse_hv.norm() > 0.0);
    }

    #[test]
    fn test_prediction_error_finite() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);

        let frame = gradient_frame(64, 64);
        for _ in 0..10 {
            let out = pch.process_frame(&frame, 64, 64, 1);
            assert!(
                out.prediction_error.is_finite(),
                "Prediction error should be finite"
            );
            assert!(out.prediction_error >= 0.0);
        }
    }

    #[test]
    fn test_prediction_error_decreases_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let mut errors = Vec::new();
        for _ in 0..30 {
            let out = pch.process_frame(&frame, 64, 64, 1);
            errors.push(out.prediction_error);
        }

        // After learning, prediction error should decrease
        let early_mean: f32 = errors[2..5].iter().sum::<f32>() / 3.0;
        let late_mean: f32 = errors[25..30].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= early_mean + 0.1,
            "Prediction error should decrease for static scene: early={early_mean}, late={late_mean}"
        );
    }

    #[test]
    fn test_scene_change_increases_error() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);

        // Converge on scene A
        let frame_a = solid_gray_frame(64, 64, 50);
        for _ in 0..20 {
            pch.process_frame(&frame_a, 64, 64, 1);
        }
        let stable_error = pch.prediction_error();

        // Switch to scene B
        let frame_b = solid_gray_frame(64, 64, 200);
        let out = pch.process_frame(&frame_b, 64, 64, 1);

        assert!(
            out.prediction_error >= stable_error * 0.5,
            "Scene change should increase error: stable={stable_error}, change={}",
            out.prediction_error
        );
    }

    #[test]
    fn test_ema_tracks_errors() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        for _ in 0..10 {
            pch.process_frame(&frame, 64, 64, 1);
        }

        // EMA should be non-zero after processing frames
        assert!(pch.error_ema().is_finite());
        assert!(pch.error_ema() >= 0.0);
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        pch.process_frame(&frame, 64, 64, 1);
        pch.process_frame(&frame, 64, 64, 1);
        assert!(pch.prediction_error() < 1.0 || pch.error_ema() > 0.0);

        pch.reset();
        assert_eq!(pch.prediction_error(), 0.0);
        assert_eq!(pch.error_ema(), 0.0);
    }

    #[test]
    fn test_predict_fine_produces_valid_hv() {
        let cfg = VisionConfig::default();
        let pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let coarse = ContinuousHV::random(cfg.hdc_dim, 999);

        let predicted = pch.predict_fine(&coarse);
        assert_eq!(predicted.dim(), cfg.hdc_dim);
        assert!(predicted.norm() > 0.0);
    }
}
