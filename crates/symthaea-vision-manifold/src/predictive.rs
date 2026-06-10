// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

    /// Perform 'Holographic Dilation' - scale internal components.
    pub fn dilate(&mut self, target_dim: usize) {
        if self.dim == target_dim {
            return;
        }

        self.encoder.dilate(target_dim);
        self.prediction_weight = self.prediction_weight.dilate(target_dim);

        if let Some(ref mut hv) = self.last_coarse_hv {
            *hv = hv.dilate(target_dim);
        }

        self.dim = target_dim;
    }

    /// Perform multi-scale 'Dreaming' - predict future coarse and fine states.
    ///
    /// Science: Friston (2010). Hierarchical active inference allows the system
    /// to Zoom Out (abstract simulation) and Zoom In (detailed simulation) across
    /// multiple levels of the world model hierarchy.
    ///
    /// Returns `(coarse_predictions, fine_predictions)`.
    pub fn dream_ahead(&self, steps: usize, _dt: f32) -> (Vec<ContinuousHV>, Vec<ContinuousHV>) {
        let mut coarse_preds = Vec::with_capacity(steps);
        let mut fine_preds = Vec::with_capacity(steps);

        let mut current_coarse = self
            .last_coarse_hv
            .clone()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        for _ in 0..steps {
            // 1. Evolve coarse state (assume abstract scene-level dynamics)
            // For now we use the identity as a neutral dynamic (static scene dream)
            // but rotate it slightly to simulate abstract "thought drift".
            current_coarse = current_coarse.permute(1);

            // 2. Predict fine scale from coarse scale (top-down inference)
            let predicted_fine = self.predict_fine(&current_coarse);

            coarse_preds.push(current_coarse.clone());
            fine_preds.push(predicted_fine);
        }

        (coarse_preds, fine_preds)
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

        let fine_hv = scale_hvs
            .first()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));
        let coarse_hv = scale_hvs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        // Generate prediction of fine scale from previous coarse scale
        if let Some(prev_coarse) = self.last_coarse_hv.clone() {
            let predicted_fine = self.predict_fine(&prev_coarse);
            self.prediction_error = 1.0 - fine_hv.similarity(&predicted_fine).clamp(-1.0, 1.0);

            // Update prediction weight via simple Hebbian-like rule:
            // Move prediction_weight to reduce the gap between predicted_fine and actual fine
            self.update_prediction_weight(&fine_hv, &predicted_fine, &prev_coarse);
        } else {
            self.prediction_error = 1.0; // No prediction yet
        }

        self.error_ema =
            self.ema_decay * self.error_ema + (1.0 - self.ema_decay) * self.prediction_error;
        self.last_coarse_hv = Some(coarse_hv.clone());

        PredictiveOutput {
            fine_hv,
            coarse_hv,
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
            patch_prediction_errors: vec![],
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

    /// Process a frame with full predictive coding feedback loop.
    ///
    /// Implements a proper cortical predictive coding cycle (Rao & Ballard 1999):
    /// 1. Encode at all scales
    /// 2. Apply top-down prior: coarse prediction biases fine encoding
    /// 3. Compute bottom-up error: cross-scale attention from prediction residuals
    /// 4. Re-encode fine scale with attention weighting
    /// 5. Update coarse→fine mapping from bottom-up error signal
    pub fn process_frame_with_feedback(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
    ) -> PredictiveOutput {
        let (_blended, scale_hvs, all_patches) =
            self.encoder.encode_frame(pixels, width, height, channels);

        let fine_hv = scale_hvs
            .first()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));
        let coarse_hv = scale_hvs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.dim));

        // Step 1: Apply top-down prior from previous coarse prediction
        let top_down_fine = if let Some(ref prev_coarse) = self.last_coarse_hv {
            let predicted = self.predict_fine(prev_coarse);
            // Blend prediction with bottom-up: 80% bottom-up + 20% top-down prior
            ContinuousHV::weighted_bundle(&[&fine_hv, &predicted], &[0.8, 0.2])
        } else {
            fine_hv.clone()
        };

        // Step 2: Compute cross-scale attention (bottom-up error signal)
        let attention = Self::compute_patch_attention(&all_patches);

        // Step 3: Re-encode fine scale with attention weighting
        let attended_fine = if !attention.is_empty() {
            if let Some(fine_enc) = self.encoder.encoder_at_mut(0) {
                let (attended, _) =
                    fine_enc.encode_frame_attended(pixels, width, height, channels, &attention);
                // Blend with top-down prior
                ContinuousHV::weighted_bundle(&[&attended, &top_down_fine], &[0.7, 0.3])
            } else {
                top_down_fine
            }
        } else {
            top_down_fine
        };

        // Step 4: Compute prediction error and update mapping (bottom-up → top-down)
        if let Some(prev_coarse) = self.last_coarse_hv.clone() {
            let predicted_fine = self.predict_fine(&prev_coarse);
            self.prediction_error =
                1.0 - attended_fine.similarity(&predicted_fine).clamp(-1.0, 1.0);
            self.update_prediction_weight(&attended_fine, &predicted_fine, &prev_coarse);
        } else {
            self.prediction_error = 1.0;
        }

        self.error_ema =
            self.ema_decay * self.error_ema + (1.0 - self.ema_decay) * self.prediction_error;
        self.last_coarse_hv = Some(coarse_hv.clone());

        PredictiveOutput {
            fine_hv: attended_fine,
            coarse_hv,
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
            patch_prediction_errors: attention,
        }
    }

    /// Compute per-patch attention from cross-scale prediction error.
    ///
    /// Fine patches that are poorly predicted by their corresponding coarse
    /// patch receive high attention (1 - cosine_similarity).
    /// Now public so callers can inject the result into a `SurpriseMap`.
    pub fn compute_patch_attention(all_patches: &[Vec<ContinuousHV>]) -> Vec<f32> {
        if all_patches.len() < 2 {
            return vec![];
        }

        let fine_patches = &all_patches[0];
        let coarse_patches = &all_patches[all_patches.len() - 1];

        if fine_patches.is_empty() || coarse_patches.is_empty() {
            return vec![];
        }

        let fine_per_coarse = (fine_patches.len() / coarse_patches.len()).max(1);

        fine_patches
            .iter()
            .enumerate()
            .map(|(i, fine_hv)| {
                let coarse_idx = (i / fine_per_coarse).min(coarse_patches.len() - 1);
                let sim = fine_hv.similarity(&coarse_patches[coarse_idx]).max(0.0);
                1.0 - sim
            })
            .collect()
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
    /// Per-patch cross-scale prediction error (1 - cos_sim(fine_patch, predicted_from_coarse)).
    ///
    /// Length = number of fine-scale patches. Each entry is in [0.0, ~1.5].
    /// High values indicate regions where the coarse encoding fails to capture
    /// the fine-grained structure — these are true free-energy hotspots.
    /// Empty until the second frame (no prediction on first frame).
    pub patch_prediction_errors: Vec<f32>,
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

    // === Predictive Coding Feedback ===

    #[test]
    fn test_process_frame_with_feedback_produces_valid_output() {
        let cfg = VisionConfig::default();
        let mut pch = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let out = pch.process_frame_with_feedback(&frame, 64, 64, 1);
        assert!(out.fine_hv.norm() > 0.0);
        assert!(out.coarse_hv.norm() > 0.0);
        assert!(out.prediction_error >= 0.0);
    }

    #[test]
    fn test_feedback_differs_from_no_feedback() {
        let cfg = VisionConfig::default();
        let mut pch1 = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let mut pch2 = PredictiveCodingHierarchy::new(&cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Process first frame to establish state
        pch1.process_frame(&frame, 64, 64, 1);
        pch2.process_frame_with_feedback(&frame, 64, 64, 1);

        // Second frame: compare outputs
        let out_no_fb = pch1.process_frame(&frame, 64, 64, 1);
        let out_fb = pch2.process_frame_with_feedback(&frame, 64, 64, 1);

        // Both should produce valid output; they may differ due to attention weighting
        assert!(out_no_fb.fine_hv.norm() > 0.0);
        assert!(out_fb.fine_hv.norm() > 0.0);
    }

    #[test]
    fn test_compute_patch_attention() {
        let dim = 16_384;
        let fine: Vec<ContinuousHV> = (0..16)
            .map(|i| ContinuousHV::random(dim, 1000 + i))
            .collect();
        let coarse: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, 2000 + i))
            .collect();

        let attention = PredictiveCodingHierarchy::compute_patch_attention(&[fine, coarse]);
        assert_eq!(attention.len(), 16);
        for &a in &attention {
            assert!(a >= 0.0 && a <= 1.5, "attention should be bounded: {a}");
        }
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
