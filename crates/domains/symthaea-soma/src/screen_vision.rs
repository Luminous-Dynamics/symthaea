// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Screen embodiment: framebuffer capture → holographic visual perception.
//!
//! This module lets Soma "see" the phone screen by feeding screen framebuffer
//! captures through the existing vision pipeline: `symthaea-vision-manifold`
//! (dorsal stream: patch-based HDC encoding + CfC surprise) and
//! `symthaea-foveation` (ventral stream: cropped region → semantic recognition).
//!
//! # Data flow
//!
//! ```text
//! screen pixels → ScreenVisionBridge::process_frame()
//!                       ↓
//!                 VisionManifold::observe_frame()      (dorsal: 16,384D scene HV + surprise)
//!                       ↓
//!                 extract salient regions               (top-K by surprise)
//!                       ↓ (if foveation enabled)
//!                 FoveationManager::on_saliency()      (ventral: OCR / embed / caption)
//!                       ↓
//!                 ScreenPerception                      (scene HV + surprise + regions)
//! ```
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use symthaea_soma::screen_vision::{ScreenVisionBridge, ScreenVisionConfig};
//!
//! let mut bridge = ScreenVisionBridge::new(ScreenVisionConfig::default());
//! let perception = bridge.process_frame(&frame_rgb, 360, 640);
//! if perception.surprise_level > 0.3 {
//!     // screen changed significantly — attend to salient regions
//! }
//! let ventral_results = bridge.drain_foveation_results();
//! ```

use serde::{Deserialize, Serialize};
use symthaea_foveation::{
    FoveationConfig, FoveationManager, FoveationResult, FrameBuffer, RoutingStrategy,
};
use symthaea_vision_manifold::{VisionConfig, VisionManifold, VisionTelemetry};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the screen vision bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenVisionConfig {
    /// Downscaled capture width in pixels (default: 360).
    pub capture_width: u32,
    /// Downscaled capture height in pixels (default: 640).
    pub capture_height: u32,
    /// Target frame rate in Hz (default: 5.0).
    ///
    /// Screen content changes far less frequently than camera video, so 5 Hz
    /// balances visual awareness with CPU budget on mobile.
    pub target_fps: f32,
    /// Minimum surprise level to include a region in the output (default: 0.3).
    pub surprise_threshold: f32,
    /// Enable ventral stream dispatch via `FoveationManager` (default: true).
    pub enable_foveation: bool,
    /// Maximum number of salient regions reported per frame (default: 4).
    pub max_salient_regions: usize,
}

impl Default for ScreenVisionConfig {
    fn default() -> Self {
        Self {
            capture_width: 360,
            capture_height: 640,
            target_fps: 5.0,
            surprise_threshold: 0.3,
            enable_foveation: true,
            max_salient_regions: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A region of the screen that the dorsal stream flagged as surprising.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalientRegion {
    /// Pixel X coordinate (top-left corner) in the capture coordinate space.
    pub x: u32,
    /// Pixel Y coordinate (top-left corner) in the capture coordinate space.
    pub y: u32,
    /// Region width in pixels.
    pub width: u32,
    /// Region height in pixels.
    pub height: u32,
    /// Surprise value for this region (higher = more unexpected change).
    pub surprise: f32,
    /// Motion velocity `[dx, dy]` in pixels/frame at this patch.
    pub velocity: [f32; 2],
}

/// The output of a single `process_frame()` call.
#[derive(Debug, Clone)]
pub struct ScreenPerception {
    /// 16,384-dimensional holographic scene encoding (ContinuousHV values).
    pub scene_hv: Vec<f32>,
    /// Overall scene surprise: the maximum per-patch surprise across the frame.
    /// Range 0.0 (identical to prediction) to 1.0 (completely novel).
    pub surprise_level: f32,
    /// Top-K attention targets sorted by descending surprise.
    pub salient_regions: Vec<SalientRegion>,
    /// Monotonically increasing frame sequence number.
    pub frame_id: u64,
}

/// Telemetry snapshot from the screen vision subsystem.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScreenVisionTelemetry {
    /// Total frames processed since creation.
    pub frames_processed: u64,
    /// Surprise level of the most recent frame.
    pub last_surprise: f32,
    /// Number of salient regions in the most recent frame.
    pub last_salient_count: usize,
    /// Last VisionManifold telemetry snapshot.
    pub vision_telemetry: Option<VisionTelemetry>,
    /// Foveation subsystem stats (pending / in-flight / completed).
    pub foveation_pending: usize,
    /// Total foveation dispatches since creation.
    pub foveation_total_dispatched: u64,
    /// Total foveation completions since creation.
    pub foveation_total_completed: u64,
}

// ---------------------------------------------------------------------------
// Main bridge
// ---------------------------------------------------------------------------

/// Bridge between screen framebuffer captures and the holographic vision pipeline.
///
/// Wraps a `VisionManifold` (dorsal stream) and a `FoveationManager` (ventral
/// stream) configured for phone-screen content analysis.
pub struct ScreenVisionBridge {
    /// Dorsal stream: HDC encoding + CfC temporal prediction + surprise.
    vision: VisionManifold,
    /// Ventral stream: crop dispatch + OCR/embed/caption.
    foveation: FoveationManager,
    /// User configuration.
    config: ScreenVisionConfig,
    /// Monotonically increasing frame counter.
    frame_count: u64,
    /// Per-patch surprise values from the last frame (cached for telemetry).
    last_surprise_map: Vec<f32>,
}

impl ScreenVisionBridge {
    /// Create a new screen vision bridge.
    ///
    /// Internally creates a `VisionManifold` sized for the configured capture
    /// resolution, and a `FoveationManager` tuned for screen content:
    ///
    /// - Patch size 8 — good for UI elements and text
    /// - Motion features enabled — detects scrolling and animations
    /// - Lower surprise decay (0.85) — screen content is more static than video
    /// - Auto routing strategy — lets foveation decide OCR vs embed vs caption
    pub fn new(config: ScreenVisionConfig) -> Self {
        // Build a VisionConfig tuned for screen content
        let vision_config = VisionConfig {
            patch_size: 8,
            surprise_threshold: config.surprise_threshold,
            surprise_decay: 0.85,
            enable_motion: true,
            enable_color: true,
            input_blend: 0.7,
            enable_predictive_hierarchy: false,
            ..VisionConfig::default()
        };

        let vision =
            VisionManifold::new(vision_config, config.capture_width, config.capture_height);

        // Build a FoveationConfig tuned for screen analysis
        let foveation_config = FoveationConfig {
            max_concurrent: 2,
            channel_depth: config.max_salient_regions,
            min_surprise_threshold: config.surprise_threshold,
            cooldown_ms: 100, // slower than camera — screen changes less often
            max_crop_pixels: 256 * 256,
            routing: RoutingStrategy::Auto,
        };

        let patch_size = 8;
        let foveation = FoveationManager::new(foveation_config, patch_size);

        Self {
            vision,
            foveation,
            config,
            frame_count: 0,
            last_surprise_map: Vec::new(),
        }
    }

    /// Process a single screen frame.
    ///
    /// # Arguments
    ///
    /// * `frame_rgb` — Raw pixel data in RGB format (3 bytes per pixel,
    ///   row-major). Length must be `width * height * 3`.
    /// * `width` — Frame width in pixels.
    /// * `height` — Frame height in pixels.
    ///
    /// # Returns
    ///
    /// A `ScreenPerception` containing the holographic scene encoding,
    /// overall surprise level, and up to `max_salient_regions` attention
    /// targets sorted by descending surprise.
    ///
    /// # Panics
    ///
    /// Panics if `frame_rgb.len() != width * height * 3`.
    pub fn process_frame(&mut self, frame_rgb: &[u8], width: u32, height: u32) -> ScreenPerception {
        let expected_len = (width as usize) * (height as usize) * 3;
        assert_eq!(
            frame_rgb.len(),
            expected_len,
            "frame_rgb length {} does not match width({}) * height({}) * 3 = {}",
            frame_rgb.len(),
            width,
            height,
            expected_len,
        );

        let dt = if self.config.target_fps > 0.0 {
            1.0 / self.config.target_fps
        } else {
            0.5
        };

        // Step 1: Feed frame to the dorsal stream (HDC encode + CfC evolve)
        let _telemetry = self.vision.observe_frame(frame_rgb, width, height, 3, dt);

        // Step 2: Extract the surprise map
        let surprise_map = self.vision.surprise_map();
        let attention = surprise_map.attention_map();
        self.last_surprise_map = attention.values.clone();

        // Overall surprise = max patch surprise (clamped to 1.0)
        let surprise_level = attention.max_surprise().min(1.0);

        // Step 3: Extract top-K salient regions
        let salient_patches = surprise_map.salient_patches();
        let motion_vectors = self.vision.motion_vectors();
        let patch_size = 8u32;

        let mut regions: Vec<SalientRegion> = salient_patches
            .iter()
            .filter(|&&(_, _, s)| s >= self.config.surprise_threshold)
            .map(|&(row, col, surprise)| {
                let velocity = motion_vectors
                    .get(row * (width as usize / patch_size as usize) + col)
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                SalientRegion {
                    x: (col as u32) * patch_size,
                    y: (row as u32) * patch_size,
                    width: patch_size,
                    height: patch_size,
                    surprise,
                    velocity,
                }
            })
            .collect();

        // Sort by descending surprise and cap at max_salient_regions
        regions.sort_by(|a, b| {
            b.surprise
                .partial_cmp(&a.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        regions.truncate(self.config.max_salient_regions);

        // Step 4: If foveation is enabled, dispatch salient regions to ventral stream
        if self.config.enable_foveation && !regions.is_empty() {
            // Provide the full-res frame for cropping
            let frame_buffer = FrameBuffer {
                pixels: frame_rgb.to_vec(),
                width,
                height,
                channels: 3,
                frame_id: self.frame_count,
                timestamp_us: (self.frame_count as f64 * dt as f64 * 1_000_000.0) as u64,
            };
            self.foveation.on_frame(frame_buffer);

            // Convert our SalientRegions into the format on_saliency expects:
            // (row, col, surprise, velocity)
            let saliency_tuples: Vec<(usize, usize, f32, [f32; 2])> = regions
                .iter()
                .map(|r| {
                    let row = (r.y / patch_size) as usize;
                    let col = (r.x / patch_size) as usize;
                    (row, col, r.surprise, r.velocity)
                })
                .collect();
            self.foveation.on_saliency(&saliency_tuples);

            // Tick the foveation manager to dispatch/collect
            let now_us = (self.frame_count as f64 * dt as f64 * 1_000_000.0) as u64;
            self.foveation.tick(now_us);
        }

        // Step 5: Build the scene HV from the manifold state
        let scene_hv = self.vision.state().values.to_vec();

        let frame_id = self.frame_count;
        self.frame_count += 1;

        ScreenPerception {
            scene_hv,
            surprise_level,
            salient_regions: regions,
            frame_id,
        }
    }

    /// Drain completed ventral recognition results.
    ///
    /// Call this after `process_frame()` to collect any semantic recognitions
    /// (OCR text, object embeddings, captions) that the background ventral
    /// pipeline has finished processing.
    pub fn drain_foveation_results(&mut self) -> Vec<FoveationResult> {
        self.foveation.drain_results()
    }

    /// Forward neuromodulator levels to the foveation manager.
    ///
    /// - **NE** (norepinephrine, 0.0-2.0): scales the surprise threshold.
    ///   Higher NE = higher threshold = only very surprising regions are dispatched.
    /// - **DA** (dopamine, 0.0-2.0): scales concurrent dispatch budget.
    ///   Higher DA = more concurrent ventral requests = broader attention.
    pub fn modulate(&mut self, ne: f32, da: f32) {
        self.foveation.modulate(ne, da);
    }

    /// Current telemetry snapshot.
    pub fn telemetry(&self) -> ScreenVisionTelemetry {
        let fov_tel = self.foveation.telemetry();
        ScreenVisionTelemetry {
            frames_processed: self.frame_count,
            last_surprise: self
                .last_surprise_map
                .iter()
                .copied()
                .fold(0.0f32, f32::max),
            last_salient_count: self.foveation.pending_count()
                + self.foveation.in_flight_count()
                + self.foveation.ready_count(),
            vision_telemetry: Some(self.vision.telemetry().clone()),
            foveation_pending: fov_tel.pending_count,
            foveation_total_dispatched: fov_tel.total_dispatched,
            foveation_total_completed: fov_tel.total_completed,
        }
    }

    /// Access the underlying `VisionManifold` (for advanced integration).
    pub fn vision(&self) -> &VisionManifold {
        &self.vision
    }

    /// Mutable access to the underlying `VisionManifold`.
    pub fn vision_mut(&mut self) -> &mut VisionManifold {
        &mut self.vision
    }

    /// Access the underlying `FoveationManager`.
    pub fn foveation(&self) -> &FoveationManager {
        &self.foveation
    }

    /// Mutable access to the underlying `FoveationManager`.
    pub fn foveation_mut(&mut self) -> &mut FoveationManager {
        &mut self.foveation
    }

    /// Current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid-color RGB frame.
    fn solid_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let num_pixels = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(num_pixels * 3);
        for _ in 0..num_pixels {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    /// Create a frame with a bright rectangle on a dark background.
    fn frame_with_rectangle(
        width: u32,
        height: u32,
        rx: u32,
        ry: u32,
        rw: u32,
        rh: u32,
    ) -> Vec<u8> {
        let mut data = vec![16u8; (width as usize) * (height as usize) * 3];
        for y in ry..(ry + rh).min(height) {
            for x in rx..(rx + rw).min(width) {
                let idx = ((y * width + x) as usize) * 3;
                data[idx] = 255; // R
                data[idx + 1] = 255; // G
                data[idx + 2] = 255; // B
            }
        }
        data
    }

    #[test]
    fn test_process_frame_produces_valid_perception() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation: false,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        let frame = solid_frame(64, 64, 128, 128, 128);
        let perception = bridge.process_frame(&frame, 64, 64);

        // Scene HV must be 16,384D
        assert_eq!(perception.scene_hv.len(), 16_384);
        // Surprise is in valid range
        assert!(perception.surprise_level >= 0.0);
        assert!(perception.surprise_level <= 1.0);
        // Frame ID starts at 0
        assert_eq!(perception.frame_id, 0);
        // Second frame increments
        let p2 = bridge.process_frame(&frame, 64, 64);
        assert_eq!(p2.frame_id, 1);
    }

    #[test]
    fn test_surprise_detection_on_frame_change() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.1, // low threshold to catch changes
            enable_foveation: false,
            max_salient_regions: 8,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        // Feed several identical frames to establish a baseline
        let dark_frame = solid_frame(64, 64, 16, 16, 16);
        for _ in 0..5 {
            bridge.process_frame(&dark_frame, 64, 64);
        }
        let baseline = bridge.process_frame(&dark_frame, 64, 64);

        // Now feed a very different frame
        let bright_frame = solid_frame(64, 64, 240, 240, 240);
        let changed = bridge.process_frame(&bright_frame, 64, 64);

        // The changed frame should have higher surprise than baseline
        assert!(
            changed.surprise_level > baseline.surprise_level,
            "surprise after scene change ({}) should exceed baseline ({})",
            changed.surprise_level,
            baseline.surprise_level,
        );
    }

    #[test]
    fn test_foveation_results_drain_correctly() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation: true,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        // Process a frame — foveation manager gets initialized
        let frame = solid_frame(64, 64, 128, 128, 128);
        bridge.process_frame(&frame, 64, 64);

        // Drain should succeed (may be empty if nothing dispatched)
        let results = bridge.drain_foveation_results();
        // Results is a valid vec (we just check it doesn't panic)
        assert!(results.len() <= 4);

        // Draining again should return empty (no new results)
        let results2 = bridge.drain_foveation_results();
        assert!(results2.is_empty());
    }

    #[test]
    fn test_neuromod_modulation_affects_thresholds() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation: true,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        // Default NE=1.0 equivalent → threshold should be near 0.3
        let baseline_threshold = bridge.foveation().effective_surprise_threshold();

        // High NE → higher threshold (more selective)
        bridge.modulate(2.0, 1.0);
        let high_ne_threshold = bridge.foveation().effective_surprise_threshold();
        assert!(
            high_ne_threshold > baseline_threshold,
            "high NE ({}) should raise threshold above baseline ({})",
            high_ne_threshold,
            baseline_threshold,
        );

        // Low NE → lower threshold (more sensitive)
        bridge.modulate(0.5, 1.0);
        let low_ne_threshold = bridge.foveation().effective_surprise_threshold();
        assert!(
            low_ne_threshold < baseline_threshold,
            "low NE ({}) should lower threshold below baseline ({})",
            low_ne_threshold,
            baseline_threshold,
        );

        // DA affects concurrent dispatch budget
        bridge.modulate(1.0, 2.0);
        let high_da_concurrent = bridge.foveation().effective_max_concurrent();
        bridge.modulate(1.0, 0.5);
        let low_da_concurrent = bridge.foveation().effective_max_concurrent();
        assert!(
            high_da_concurrent >= low_da_concurrent,
            "high DA ({}) should allow >= concurrent requests than low DA ({})",
            high_da_concurrent,
            low_da_concurrent,
        );
    }

    #[test]
    fn test_config_defaults_are_reasonable() {
        let config = ScreenVisionConfig::default();

        // Capture dimensions match a phone in portrait
        assert_eq!(config.capture_width, 360);
        assert_eq!(config.capture_height, 640);
        assert!(config.capture_height > config.capture_width);

        // Frame rate is low (screen, not camera)
        assert!(config.target_fps > 0.0);
        assert!(config.target_fps <= 10.0);

        // Surprise threshold in valid range
        assert!(config.surprise_threshold > 0.0);
        assert!(config.surprise_threshold <= 1.0);

        // Foveation enabled by default
        assert!(config.enable_foveation);

        // Reasonable salient region cap
        assert!(config.max_salient_regions > 0);
        assert!(config.max_salient_regions <= 16);
    }

    #[test]
    fn test_telemetry_updates_after_processing() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation: false,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        let tel_before = bridge.telemetry();
        assert_eq!(tel_before.frames_processed, 0);

        let frame = solid_frame(64, 64, 128, 128, 128);
        bridge.process_frame(&frame, 64, 64);

        let tel_after = bridge.telemetry();
        assert_eq!(tel_after.frames_processed, 1);
        assert!(tel_after.vision_telemetry.is_some());
    }

    #[test]
    fn test_salient_regions_respect_max_cap() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.01, // very low to catch many regions
            enable_foveation: false,
            max_salient_regions: 2,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        // Establish a baseline with dark frames
        let dark = solid_frame(64, 64, 0, 0, 0);
        for _ in 0..3 {
            bridge.process_frame(&dark, 64, 64);
        }

        // Feed a frame with a bright rectangle to create spatial surprise
        let rect_frame = frame_with_rectangle(64, 64, 8, 8, 32, 32);
        let perception = bridge.process_frame(&rect_frame, 64, 64);

        // Regardless of how many patches are surprising, cap at 2
        assert!(
            perception.salient_regions.len() <= 2,
            "salient_regions count {} exceeds max_salient_regions 2",
            perception.salient_regions.len(),
        );
    }

    #[test]
    fn test_scene_hv_dimension_matches_hdc_constant() {
        let config = ScreenVisionConfig {
            capture_width: 32,
            capture_height: 32,
            target_fps: 1.0,
            surprise_threshold: 0.3,
            enable_foveation: false,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        let frame = solid_frame(32, 32, 100, 100, 100);
        let perception = bridge.process_frame(&frame, 32, 32);

        // Must match symthaea_core::hdc::HDC_DIMENSION (16,384)
        assert_eq!(perception.scene_hv.len(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    #[should_panic(expected = "frame_rgb length")]
    fn test_process_frame_panics_on_size_mismatch() {
        let config = ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation: false,
            max_salient_regions: 4,
        };
        let mut bridge = ScreenVisionBridge::new(config);

        // Wrong size: 32x32 data for a 64x64 frame
        let bad_frame = vec![128u8; 32 * 32 * 3];
        bridge.process_frame(&bad_frame, 64, 64);
    }
}
