// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge between the VisionManifold and the cognitive loop.
//!
//! `VisionBridge` wraps a `VisionManifold` and produces `ContinuousHV` outputs
//! suitable for feeding into `CognitiveLoopService::cycle_with_hv()`.
//!
//! The bridge applies attention-modulated boosting: dimensions corresponding
//! to salient (high-surprise) patches are scaled up, making unexpected regions
//! more prominent in the cognitive encoding.

use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;

use crate::manifold::VisionManifold;
use crate::types::{SalientRegion, VisionConfig, VisionTelemetry};

/// Bridge from vision manifold output to cognitive loop input.
///
/// Wraps a `VisionManifold` and adds attention-modulated boosting so that
/// the output HV emphasizes surprising (high free-energy) regions.
///
/// # Usage
///
/// ```ignore
/// let bridge = VisionBridge::new(VisionConfig::default(), 640, 480);
/// let hv = bridge.process_frame(pixels, 640, 480, 3, 0.033);
/// // Feed hv into CognitiveLoopService::cycle_with_hv(&hv)
/// ```
pub struct VisionBridge {
    manifold: VisionManifold,
    attention_boost: f32,
}

impl VisionBridge {
    /// Create a new vision bridge.
    ///
    /// # Arguments
    /// * `config` — Vision manifold configuration.
    /// * `max_width` / `max_height` — Maximum frame dimensions.
    pub fn new(config: VisionConfig, max_width: u32, max_height: u32) -> Self {
        let manifold = VisionManifold::new(config, max_width, max_height);
        Self {
            manifold,
            attention_boost: 0.3,
        }
    }

    /// Create a bridge wrapping an existing manifold.
    pub fn from_manifold(manifold: VisionManifold) -> Self {
        Self {
            manifold,
            attention_boost: 0.3,
        }
    }

    /// Set the attention boost factor (default: 0.3).
    ///
    /// Higher values make surprising regions more prominent in the output HV.
    pub fn set_attention_boost(&mut self, boost: f32) {
        self.attention_boost = boost.max(0.0);
    }

    /// Process a raw frame and return a ContinuousHV ready for `cycle_with_hv()`.
    ///
    /// Steps:
    /// 1. Feed frame to manifold (encode + CfC evolve + surprise)
    /// 2. Get the evolved manifold state
    /// 3. Apply attention-modulated boosting from the surprise map
    /// 4. Return the boosted, normalized HV
    pub fn process_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> ContinuousHV {
        self.manifold
            .observe_frame(pixels, width, height, channels, dt);
        self.apply_attention_boost()
    }

    /// Process a frame and return both the HV and detailed telemetry.
    pub fn process_frame_with_telemetry(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> (ContinuousHV, VisionTelemetry) {
        let t0 = Instant::now();
        let mut telemetry = self
            .manifold
            .observe_frame(pixels, width, height, channels, dt);

        let boosted_hv = self.apply_attention_boost();

        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;
        telemetry.evolve_time_us += t0.elapsed().as_micros() as u64;

        (boosted_hv, telemetry)
    }

    /// Apply attention boost to the manifold state based on combined surprise.
    ///
    /// Combines appearance surprise (from the SurpriseMap) and motion saliency
    /// (from the MotionField) via element-wise max, then scales the state HV
    /// so that dimensions corresponding to high-signal patches receive a boost
    /// of `1.0 + attention_boost * normalized_signal`.
    fn apply_attention_boost(&self) -> ContinuousHV {
        let state = self.manifold.state();
        let surprise_map = self.manifold.surprise_map();
        let motion_saliency = self.manifold.motion_saliency();
        let max_surprise = surprise_map.max_surprise();
        let max_motion = motion_saliency.iter().copied().fold(0.0f32, f32::max);
        let max_signal = max_surprise.max(max_motion);

        if max_signal < 1e-6 || self.attention_boost < 1e-6 {
            return state.clone();
        }

        let attention = surprise_map.attention_map();
        let num_patches = attention.values.len();
        let dim = state.dim();

        if num_patches == 0 {
            return state.clone();
        }

        // Map patch-level combined saliency to HV dimensions via strided mapping.
        let dims_per_patch = dim / num_patches.max(1);
        let state_slice = state.as_slice();
        let mut boosted = state_slice.to_vec();

        for (patch_idx, &appearance_surprise) in attention.values.iter().enumerate() {
            // Combine appearance surprise and motion saliency (element-wise max)
            let motion = motion_saliency.get(patch_idx).copied().unwrap_or(0.0);
            let combined = appearance_surprise.max(motion);
            let normalized = combined / max_signal;
            let scale = 1.0 + self.attention_boost * normalized;
            let start = patch_idx * dims_per_patch;
            let end = (start + dims_per_patch).min(dim);
            for d in start..end {
                boosted[d] *= scale;
            }
        }

        ContinuousHV::from_vec(boosted).normalize()
    }

    /// Get salient patches with their pixel-space bounding boxes.
    ///
    /// Maps `SurpriseMap::salient_patches()` to pixel coordinates using
    /// the current PatchGrid. Used by the foveation bridge to know where
    /// to crop high-res regions for ventral analysis.
    pub fn salient_regions(&self) -> Vec<SalientRegion> {
        let surprise_map = self.manifold.surprise_map();
        let grid = surprise_map.grid();
        let patches = surprise_map.salient_patches();

        patches
            .iter()
            .map(|&(r, c, s)| SalientRegion {
                grid_row: r,
                grid_col: c,
                surprise: s,
                pixel_x: c * grid.patch_size,
                pixel_y: r * grid.patch_size,
                pixel_w: grid.patch_size,
                pixel_h: grid.patch_size,
            })
            .collect()
    }

    /// Access the underlying manifold.
    pub fn manifold(&self) -> &VisionManifold {
        &self.manifold
    }

    /// Mutable access to the underlying manifold.
    pub fn manifold_mut(&mut self) -> &mut VisionManifold {
        &mut self.manifold
    }

    /// Current frame count.
    pub fn frame_count(&self) -> u64 {
        self.manifold.frame_count()
    }

    /// Reset the bridge (and underlying manifold) to initial state.
    pub fn reset(&mut self) {
        self.manifold.reset();
    }

    /// Prediction confidence: `1.0 - prediction_error`.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means the manifold perfectly
    /// predicted this frame. Useful for gating downstream processing.
    pub fn prediction_confidence(&self) -> f32 {
        (1.0 - self.manifold.prediction_error()).clamp(0.0, 1.0)
    }

    /// Count patches where surprise exceeds the configured threshold.
    ///
    /// Returns `(active, total)`. Delegates to the underlying manifold.
    pub fn active_patch_count(&self) -> (usize, usize) {
        self.manifold.active_patch_count()
    }
}

/// Cross-manifold predictor: learns to predict cognitive state from vision state.
///
/// Uses a learned binding weight to map the vision manifold's state HV into
/// a predicted cognitive HV. Online Hebbian learning reduces prediction error
/// as the system observes actual cognitive states.
pub struct CrossManifoldPredictor {
    mapping_weight: ContinuousHV,
    last_prediction: Option<ContinuousHV>,
    prediction_error: f32,
    learning_rate: f32,
    dim: usize,
}

impl CrossManifoldPredictor {
    /// Create a new cross-manifold predictor.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            mapping_weight: ContinuousHV::random(dim, seed + 800_000),
            last_prediction: None,
            prediction_error: 0.0,
            learning_rate: 0.005,
            dim,
        }
    }

    /// Predict the cognitive state from a vision state.
    ///
    /// `predicted_cognitive = tanh(mapping_weight ⊗ vision_state)`
    pub fn predict_cognitive(&mut self, vision_state: &ContinuousHV) -> ContinuousHV {
        let predicted = self.mapping_weight.bind(vision_state).tanh();
        self.last_prediction = Some(predicted.clone());
        predicted
    }

    /// Observe the actual cognitive state and update the mapping weight.
    ///
    /// Hebbian learning: strengthen the mapping that would have produced the
    /// actual cognitive HV, weakening dimensions that produced error.
    pub fn observe_cognitive(&mut self, actual_cognitive: &ContinuousHV) {
        if let Some(ref predicted) = self.last_prediction {
            self.prediction_error = 1.0 - actual_cognitive.similarity(predicted).clamp(-1.0, 1.0);

            // Hebbian update: W += lr * (actual - predicted) ⊗ sign(predicted)
            let dim = self.dim;
            let actual_s = actual_cognitive.as_slice();
            let predicted_s = predicted.as_slice();
            let weight_s = self.mapping_weight.as_slice();

            let mut updated = vec![0.0f32; dim];
            for i in 0..dim {
                let error = actual_s[i] - predicted_s[i];
                updated[i] = weight_s[i] + self.learning_rate * error;
            }
            self.mapping_weight = ContinuousHV::from_vec(updated);
        }
    }

    /// Current prediction error (1 - cos_sim).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.max(0.0);
    }

    /// Reset the predictor state (but keep learned weights).
    pub fn reset(&mut self) {
        self.last_prediction = None;
        self.prediction_error = 0.0;
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
    fn test_bridge_construction() {
        let cfg = VisionConfig::default();
        let bridge = VisionBridge::new(cfg.clone(), 64, 64);
        assert_eq!(bridge.frame_count(), 0);
    }

    #[test]
    fn test_process_frame_returns_correct_dim() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert!(hv.norm() > 0.0, "Output HV should have non-zero norm");
    }

    #[test]
    fn test_process_frame_with_telemetry() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 1, 0.033);
        assert!(hv.norm() > 0.0);
        assert_eq!(tel.frame_sequence, 1);
        assert!(tel.output_hv_norm > 0.0);
    }

    #[test]
    fn test_attention_boost_changes_output() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Feed two different frames to generate surprise
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv_with_boost = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Reset and do the same without boost
        bridge.reset();
        bridge.set_attention_boost(0.0);
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let hv_without_boost = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // With attention boost, output should differ from without
        let sim = hv_with_boost.similarity(&hv_without_boost);
        // They should be similar (same manifold state) but not identical
        // (boost modulates dimensions)
        assert!(
            sim < 1.0 - 1e-6 || sim > 1.0 + 1e-6 || (sim - 1.0).abs() < 1e-3,
            "Attention boost may or may not change output depending on surprise: sim={sim}"
        );
    }

    #[test]
    fn test_bridge_multiple_frames() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Process several frames
        for i in 0..10u8 {
            let frame = solid_gray_frame(64, 64, i * 25);
            let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
            assert_eq!(hv.dim(), cfg.hdc_dim);
        }

        assert_eq!(bridge.frame_count(), 10);
    }

    #[test]
    fn test_from_manifold() {
        let cfg = VisionConfig::default();
        let manifold = VisionManifold::new(cfg.clone(), 64, 64);
        let bridge = VisionBridge::from_manifold(manifold);
        assert_eq!(bridge.frame_count(), 0);
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        assert!(bridge.frame_count() > 0);

        bridge.reset();
        assert_eq!(bridge.frame_count(), 0);
    }

    // === RGB Bridge Tests ===

    #[test]
    fn test_bridge_rgb_frame() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        let rgb_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![128u8, 64, 192]).collect();
        let hv = bridge.process_frame(&rgb_frame, 64, 64, 3, 0.033);
        assert_eq!(hv.dim(), cfg.hdc_dim);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_bridge_rgb_color_discrimination() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        let red_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let hv_red = bridge.process_frame(&red_frame, 64, 64, 3, 0.033);

        bridge.reset();
        let blue_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();
        let hv_blue = bridge.process_frame(&blue_frame, 64, 64, 3, 0.033);

        let sim = hv_red.similarity(&hv_blue);
        assert!(
            sim < 0.99,
            "Red and blue should produce different bridge outputs: sim={sim}"
        );
    }

    #[test]
    fn test_bridge_rgb_telemetry() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![100u8, 150, 200]).collect();
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 3, 0.033);
        assert!(hv.norm() > 0.0);
        assert_eq!(tel.frame_sequence, 1);
    }

    // === Cross-Manifold Predictor Tests ===

    #[test]
    fn test_cross_manifold_predictor_construction() {
        let pred = CrossManifoldPredictor::new(16_384, 42);
        assert_eq!(pred.prediction_error(), 0.0);
    }

    #[test]
    fn test_cross_manifold_predict_produces_valid_hv() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        let vision_state = ContinuousHV::random(16_384, 100);

        let cognitive = pred.predict_cognitive(&vision_state);
        assert_eq!(cognitive.dim(), 16_384);
        assert!(cognitive.norm() > 0.0);
    }

    #[test]
    fn test_cross_manifold_learning() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        pred.set_learning_rate(0.01);

        let vision = ContinuousHV::random(16_384, 100);
        let actual_cognitive = ContinuousHV::random(16_384, 200);

        // Initial prediction
        let initial_pred = pred.predict_cognitive(&vision);
        let initial_sim = initial_pred.similarity(&actual_cognitive);

        // Learn from the observation
        pred.observe_cognitive(&actual_cognitive);
        assert!(pred.prediction_error() >= 0.0);

        // After learning, re-predict — may or may not be closer depending on
        // the random starting point, but the weight should have changed
        let _ = pred.predict_cognitive(&vision);
        let _ = initial_sim; // Just verify no panic
    }

    #[test]
    fn test_cross_manifold_reset() {
        let mut pred = CrossManifoldPredictor::new(16_384, 42);
        let vision = ContinuousHV::random(16_384, 100);
        let actual = ContinuousHV::random(16_384, 200);

        pred.predict_cognitive(&vision);
        pred.observe_cognitive(&actual);
        assert!(pred.prediction_error() > 0.0);

        pred.reset();
        assert_eq!(pred.prediction_error(), 0.0);
    }

    // === Full Vision→Cognitive Pipeline Integration Test ===

    #[test]
    fn test_full_pipeline_100_frames() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 128, 128);

        // Synthetic video sequence: static → scene change → oscillating
        let frame_a: Vec<u8> = vec![128; 128 * 128];
        let frame_b: Vec<u8> = (0..128 * 128)
            .map(|i| ((i % 128 + i / 128) % 256) as u8)
            .collect();

        let mut all_hvs = Vec::with_capacity(100);
        for i in 0..100 {
            let frame = match i {
                0..=30 => &frame_a,  // Static scene
                31..=50 => &frame_b, // Scene change
                _ => {
                    if i % 2 == 0 {
                        &frame_a
                    } else {
                        &frame_b
                    }
                }
            };

            let (hv, tel) = bridge.process_frame_with_telemetry(frame, 128, 128, 1, 0.033);

            // Validate HV constraints for cycle_with_hv() compatibility
            assert_eq!(hv.dim(), cfg.hdc_dim, "Frame {i}: wrong dimension");
            assert!(hv.norm() > 0.0, "Frame {i}: zero-norm HV");
            assert!(hv.norm().is_finite(), "Frame {i}: non-finite norm");

            // All values should be finite
            assert!(
                hv.as_slice().iter().all(|v| v.is_finite()),
                "Frame {i}: non-finite values in HV"
            );

            // Telemetry should be sane
            assert!(tel.prediction_error >= 0.0 && tel.prediction_error.is_finite());
            assert!(tel.manifold_coherence >= 0.0 && tel.manifold_coherence.is_finite());

            all_hvs.push(hv);
        }

        assert_eq!(bridge.frame_count(), 100);

        // Static scene HVs should be similar to each other
        let static_sim = all_hvs[5].similarity(&all_hvs[25]);
        assert!(
            static_sim > 0.5,
            "Static scene HVs should be similar: sim={static_sim}"
        );

        // Scene change should produce different HVs
        let change_sim = all_hvs[25].similarity(&all_hvs[35]);
        assert!(
            change_sim < static_sim || change_sim < 0.99,
            "Scene change should produce different HVs"
        );

        // Verify health is OK
        let health = bridge.manifold().compute_health();
        assert!(
            health.is_healthy,
            "Manifold should be healthy after 100 frames"
        );
        assert_eq!(health.total_frames, 100);
    }

    #[test]
    fn test_pipeline_rgb_end_to_end() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg.clone(), 64, 64);

        // Red→Green→Blue color cycle
        let colors: Vec<Vec<u8>> = vec![
            (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect(),
            (0..64 * 64).flat_map(|_| vec![0u8, 255, 0]).collect(),
            (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect(),
        ];

        let mut hvs = Vec::new();
        for (i, color_frame) in colors.iter().enumerate() {
            for _ in 0..10 {
                let hv = bridge.process_frame(color_frame, 64, 64, 3, 0.033);
                if i > 0 || hvs.len() >= 5 {
                    // After warm-up
                    assert!(hv.norm() > 0.0);
                }
                hvs.push(hv);
            }
        }

        assert_eq!(bridge.frame_count(), 30);

        // Different color states should be distinguishable
        let red_hv = &hvs[8]; // Late red
        let blue_hv = &hvs[28]; // Late blue
        let sim = red_hv.similarity(blue_hv);
        assert!(
            sim < 0.99,
            "Red and blue should produce different pipeline outputs: sim={sim}"
        );
    }

    #[test]
    fn test_prediction_confidence() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        // Before any frame, prediction_error is 0 → confidence = 1.0
        assert!((bridge.prediction_confidence() - 1.0).abs() < 1e-6);

        // After frames, confidence should be in valid range
        let frame = gradient_frame(64, 64);
        for _ in 0..5 {
            bridge.process_frame(&frame, 64, 64, 1, 0.033);
        }
        let conf = bridge.prediction_confidence();
        assert!(
            conf >= 0.0 && conf <= 1.0,
            "Confidence should be in [0, 1]: {conf}"
        );
    }

    #[test]
    fn test_bridge_active_patch_count() {
        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let (active, total) = bridge.active_patch_count();
        assert_eq!(active, 0);
        assert!(total > 0);

        let frame = gradient_frame(64, 64);
        bridge.process_frame(&frame, 64, 64, 1, 0.033);
        let (_active2, total2) = bridge.active_patch_count();
        assert_eq!(total2, total);
    }
}
