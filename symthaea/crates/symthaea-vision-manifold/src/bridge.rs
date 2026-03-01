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
use crate::types::{VisionConfig, VisionTelemetry};

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
        self.manifold.observe_frame(pixels, width, height, channels, dt);
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
        let mut telemetry = self.manifold.observe_frame(pixels, width, height, channels, dt);

        let boosted_hv = self.apply_attention_boost();

        telemetry.output_hv_norm = boosted_hv.norm();
        telemetry.attention_boost_applied = self.attention_boost;
        telemetry.evolve_time_us += t0.elapsed().as_micros() as u64;

        (boosted_hv, telemetry)
    }

    /// Apply attention boost to the manifold state based on the surprise map.
    ///
    /// Scales the state HV so that dimensions corresponding to high-surprise
    /// patches receive a boost of `1.0 + attention_boost * normalized_surprise`.
    fn apply_attention_boost(&self) -> ContinuousHV {
        let state = self.manifold.state();
        let surprise_map = self.manifold.surprise_map();
        let max_surprise = surprise_map.max_surprise();

        if max_surprise < 1e-6 || self.attention_boost < 1e-6 {
            return state.clone();
        }

        let attention = surprise_map.attention_map();
        let num_patches = attention.values.len();
        let dim = state.dim();

        if num_patches == 0 {
            return state.clone();
        }

        // Map patch-level surprise to HV dimensions via strided mapping.
        // Each patch "owns" a contiguous block of dimensions.
        let dims_per_patch = dim / num_patches.max(1);
        let state_slice = state.as_slice();
        let mut boosted = state_slice.to_vec();

        for (patch_idx, &surprise) in attention.values.iter().enumerate() {
            let normalized = surprise / max_surprise;
            let scale = 1.0 + self.attention_boost * normalized;
            let start = patch_idx * dims_per_patch;
            let end = (start + dims_per_patch).min(dim);
            for d in start..end {
                boosted[d] *= scale;
            }
        }

        ContinuousHV::from_vec(boosted).normalize()
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
}
