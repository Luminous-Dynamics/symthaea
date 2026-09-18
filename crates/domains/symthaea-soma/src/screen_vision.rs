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
//! The compatibility `process_frame()` path remains intentionally unproven: raw pixels plus an
//! internally synthesized frame counter are not enough to establish capture provenance.
//! Capture owners that possess a stable stream identity and acquisition timestamp may instead use
//! `new_with_observation_source()` + `process_observed_frame()`.
//!
//! # Data flow
//!
//! ```text
//! capture owner: VisualStreamRef + clock + acquisition timestamp
//!                       ↓
//! screen pixels → ScreenVisionBridge::process_observed_frame()
//!                       ↓
//!                 VisualObservationRef
//!                       ↓
//!                 VisionManifold::observe_frame()      (dorsal)
//!                       ↓
//!                 FoveationManager::on_observed_frame()
//!                       ↓
//!                 FoveationManager::on_saliency()      (ventral)
//!                       ↓
//!                 ScreenPerception + provenance
//! ```

use serde::{Deserialize, Serialize};
use symthaea_foveation::{
    FoveationConfig, FoveationManager, FoveationResult, FrameBuffer, FrameObservationError,
    RoutingStrategy,
};
use symthaea_vision_manifold::{
    VisualCaptureClock, VisualObservationRef, VisualStreamRef, VisionConfig, VisionManifold,
    VisionTelemetry,
};

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

/// Capture-owner identity used by the provenance-bearing screen path.
///
/// The stream reference is validated by VIS-000R. The clock label describes timestamp semantics
/// but does not prove synchronization quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenObservationSource {
    pub stream: VisualStreamRef,
    pub clock: VisualCaptureClock,
}

impl ScreenObservationSource {
    pub const fn new(stream: VisualStreamRef, clock: VisualCaptureClock) -> Self {
        Self { stream, clock }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenVisionObservationError {
    ObservationSourceNotConfigured,
    FoveationBinding(FrameObservationError),
}

impl std::fmt::Display for ScreenVisionObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObservationSourceNotConfigured => f.write_str(
                "screen observation provenance requires an explicitly configured capture source",
            ),
            Self::FoveationBinding(error) => write!(f, "screen foveation provenance error: {error}"),
        }
    }
}

impl std::error::Error for ScreenVisionObservationError {}

impl From<FrameObservationError> for ScreenVisionObservationError {
    fn from(value: FrameObservationError) -> Self {
        Self::FoveationBinding(value)
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

/// The output of a single screen perception call.
#[derive(Debug, Clone)]
pub struct ScreenPerception {
    /// 16,384-dimensional holographic scene encoding (ContinuousHV values).
    pub scene_hv: Vec<f32>,
    /// Overall scene surprise: maximum per-patch surprise across the frame.
    pub surprise_level: f32,
    /// Top-K attention targets sorted by descending surprise.
    pub salient_regions: Vec<SalientRegion>,
    /// Monotonically increasing frame sequence number inside this bridge instance.
    pub frame_id: u64,
    /// Exact observation identity when the provenance-bearing API was used.
    pub observation: Option<VisualObservationRef>,
}

/// Telemetry snapshot from the screen vision subsystem.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScreenVisionTelemetry {
    pub frames_processed: u64,
    pub last_surprise: f32,
    pub last_salient_count: usize,
    pub vision_telemetry: Option<VisionTelemetry>,
    pub foveation_pending: usize,
    pub foveation_total_dispatched: u64,
    pub foveation_total_completed: u64,
}

// ---------------------------------------------------------------------------
// Main bridge
// ---------------------------------------------------------------------------

/// Bridge between screen framebuffer captures and the holographic vision pipeline.
pub struct ScreenVisionBridge {
    vision: VisionManifold,
    foveation: FoveationManager,
    config: ScreenVisionConfig,
    frame_count: u64,
    last_surprise_map: Vec<f32>,
    observation_source: Option<ScreenObservationSource>,
}

impl ScreenVisionBridge {
    /// Create a screen bridge whose frames remain unproven unless the caller later supplies an
    /// explicit observation source through `new_with_observation_source` instead.
    pub fn new(config: ScreenVisionConfig) -> Self {
        Self::build(config, None)
    }

    /// Create a screen bridge whose capture owner has an explicit stream identity and clock.
    ///
    /// The caller must still use `process_observed_frame()` and provide each acquisition timestamp;
    /// merely configuring a source does not upgrade calls to `process_frame()` into observations.
    pub fn new_with_observation_source(
        config: ScreenVisionConfig,
        stream: VisualStreamRef,
        clock: VisualCaptureClock,
    ) -> Self {
        Self::build(config, Some(ScreenObservationSource::new(stream, clock)))
    }

    fn build(config: ScreenVisionConfig, observation_source: Option<ScreenObservationSource>) -> Self {
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

        let foveation_config = FoveationConfig {
            max_concurrent: 2,
            channel_depth: config.max_salient_regions,
            min_surprise_threshold: config.surprise_threshold,
            cooldown_ms: 100,
            max_crop_pixels: 256 * 256,
            routing: RoutingStrategy::Auto,
        };

        let foveation = FoveationManager::new(foveation_config, 8);

        Self {
            vision,
            foveation,
            config,
            frame_count: 0,
            last_surprise_map: Vec::new(),
            observation_source,
        }
    }

    /// Process raw screen pixels without asserting capture provenance.
    ///
    /// The internal nominal frame clock is retained for scheduling/legacy temporal binding only.
    /// `ScreenPerception::observation` is always `None` on this path, even if the bridge was
    /// constructed with an observation source.
    pub fn process_frame(&mut self, frame_rgb: &[u8], width: u32, height: u32) -> ScreenPerception {
        self.process_frame_inner(frame_rgb, width, height, None)
            .expect("unproven screen frame path cannot fail provenance admission")
    }

    /// Process a screen frame as a direct observation from the configured capture stream.
    ///
    /// `captured_at_us` is interpreted according to the clock supplied to
    /// `new_with_observation_source`. The source owner, not Soma, is responsible for that clock
    /// assertion. This method does not infer Unix time or synchronization quality.
    pub fn process_observed_frame(
        &mut self,
        frame_rgb: &[u8],
        width: u32,
        height: u32,
        captured_at_us: u64,
    ) -> Result<ScreenPerception, ScreenVisionObservationError> {
        let source = self
            .observation_source
            .ok_or(ScreenVisionObservationError::ObservationSourceNotConfigured)?;
        let observation = VisualObservationRef::new(
            source.stream,
            self.frame_count,
            captured_at_us,
            source.clock,
        );
        self.process_frame_inner(frame_rgb, width, height, Some(observation))
    }

    fn process_frame_inner(
        &mut self,
        frame_rgb: &[u8],
        width: u32,
        height: u32,
        observation: Option<VisualObservationRef>,
    ) -> Result<ScreenPerception, ScreenVisionObservationError> {
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
        // Scheduling uses this process-local nominal clock, never an arbitrary device/acquisition
        // clock supplied for provenance.
        let nominal_now_us = (self.frame_count as f64 * dt as f64 * 1_000_000.0) as u64;

        // Step 1: Feed frame to the dorsal stream (HDC encode + CfC evolve)
        let _telemetry = self.vision.observe_frame(frame_rgb, width, height, 3, dt);

        // Step 2: Extract the surprise map
        let surprise_map = self.vision.surprise_map();
        let attention = surprise_map.attention_map();
        self.last_surprise_map = attention.values.clone();
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

        regions.sort_by(|a, b| {
            b.surprise
                .partial_cmp(&a.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        regions.truncate(self.config.max_salient_regions);

        // Step 4: If foveation is enabled, bind the exact frame before saliency dispatch.
        if self.config.enable_foveation && !regions.is_empty() {
            let capture_timestamp_us = observation
                .map(VisualObservationRef::captured_at_us)
                .unwrap_or(nominal_now_us);
            let frame_buffer = FrameBuffer {
                pixels: frame_rgb.to_vec(),
                width,
                height,
                channels: 3,
                frame_id: self.frame_count,
                timestamp_us: capture_timestamp_us,
            };

            match observation {
                Some(observation) => self
                    .foveation
                    .on_observed_frame(frame_buffer, observation)?,
                None => self.foveation.on_frame(frame_buffer),
            }

            let saliency_tuples: Vec<(usize, usize, f32, [f32; 2])> = regions
                .iter()
                .map(|r| {
                    let row = (r.y / patch_size) as usize;
                    let col = (r.x / patch_size) as usize;
                    (row, col, r.surprise, r.velocity)
                })
                .collect();
            self.foveation.on_saliency(&saliency_tuples);

            // Cooldown/backpressure scheduling stays process-local. Acquisition clocks may be
            // device-local, Unix, or otherwise unsuitable as a scheduler clock.
            self.foveation.tick(nominal_now_us);
        }

        // Step 5: Build the scene HV from the manifold state
        let scene_hv = self.vision.state().values.to_vec();
        let frame_id = self.frame_count;
        self.frame_count += 1;

        Ok(ScreenPerception {
            scene_hv,
            surprise_level,
            salient_regions: regions,
            frame_id,
            observation,
        })
    }

    /// Drain completed ventral recognition results.
    pub fn drain_foveation_results(&mut self) -> Vec<FoveationResult> {
        self.foveation.drain_results()
    }

    /// Forward neuromodulator levels to the foveation manager.
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

    pub fn vision(&self) -> &VisionManifold {
        &self.vision
    }

    pub fn vision_mut(&mut self) -> &mut VisionManifold {
        &mut self.vision
    }

    pub fn foveation(&self) -> &FoveationManager {
        &self.foveation
    }

    pub fn foveation_mut(&mut self) -> &mut FoveationManager {
        &mut self.foveation
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub const fn observation_source(&self) -> Option<ScreenObservationSource> {
        self.observation_source
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
        data
    }

    fn small_config(enable_foveation: bool) -> ScreenVisionConfig {
        ScreenVisionConfig {
            capture_width: 64,
            capture_height: 64,
            target_fps: 2.0,
            surprise_threshold: 0.3,
            enable_foveation,
            max_salient_regions: 4,
        }
    }

    #[test]
    fn test_process_frame_produces_valid_unproven_perception() {
        let mut bridge = ScreenVisionBridge::new(small_config(false));
        let frame = solid_frame(64, 64, 128, 128, 128);
        let perception = bridge.process_frame(&frame, 64, 64);
        assert_eq!(perception.scene_hv.len(), 16_384);
        assert!(perception.surprise_level >= 0.0 && perception.surprise_level <= 1.0);
        assert_eq!(perception.frame_id, 0);
        assert_eq!(perception.observation, None);
        let p2 = bridge.process_frame(&frame, 64, 64);
        assert_eq!(p2.frame_id, 1);
        assert_eq!(p2.observation, None);
    }

    #[test]
    fn test_observed_path_requires_explicit_source() {
        let mut bridge = ScreenVisionBridge::new(small_config(false));
        let frame = solid_frame(64, 64, 128, 128, 128);
        assert_eq!(
            bridge.process_observed_frame(&frame, 64, 64, 99_000).unwrap_err(),
            ScreenVisionObservationError::ObservationSourceNotConfigured
        );
        assert_eq!(bridge.frame_count(), 0, "failed provenance admission must not advance frame id");
    }

    #[test]
    fn test_observed_path_binds_exact_stream_clock_and_capture_time() {
        let stream = VisualStreamRef::new(55, 77).unwrap();
        let mut bridge = ScreenVisionBridge::new_with_observation_source(
            small_config(false),
            stream,
            VisualCaptureClock::DeviceLocal,
        );
        let frame = solid_frame(64, 64, 128, 128, 128);
        let perception = bridge
            .process_observed_frame(&frame, 64, 64, 123_456)
            .unwrap();
        let observation = perception.observation.unwrap();
        assert_eq!(observation.stream(), stream);
        assert_eq!(observation.frame_id(), 0);
        assert_eq!(observation.captured_at_us(), 123_456);
        assert_eq!(observation.clock_domain(), VisualCaptureClock::DeviceLocal);
    }

    #[test]
    fn test_configured_source_does_not_upgrade_legacy_process_frame() {
        let stream = VisualStreamRef::new(55, 78).unwrap();
        let mut bridge = ScreenVisionBridge::new_with_observation_source(
            small_config(false),
            stream,
            VisualCaptureClock::StreamMonotonic,
        );
        let frame = solid_frame(64, 64, 128, 128, 128);
        let perception = bridge.process_frame(&frame, 64, 64);
        assert_eq!(perception.observation, None);
    }

    #[test]
    fn test_surprise_detection_on_frame_change() {
        let config = ScreenVisionConfig {
            surprise_threshold: 0.1,
            max_salient_regions: 8,
            ..small_config(false)
        };
        let mut bridge = ScreenVisionBridge::new(config);
        let dark_frame = solid_frame(64, 64, 16, 16, 16);
        for _ in 0..5 {
            bridge.process_frame(&dark_frame, 64, 64);
        }
        let baseline = bridge.process_frame(&dark_frame, 64, 64);
        let bright_frame = solid_frame(64, 64, 240, 240, 240);
        let changed = bridge.process_frame(&bright_frame, 64, 64);
        assert!(changed.surprise_level > baseline.surprise_level);
    }

    #[test]
    fn test_foveation_results_drain_correctly() {
        let mut bridge = ScreenVisionBridge::new(small_config(true));
        let frame = solid_frame(64, 64, 128, 128, 128);
        bridge.process_frame(&frame, 64, 64);
        let results = bridge.drain_foveation_results();
        assert!(results.len() <= 4);
        assert!(bridge.drain_foveation_results().is_empty());
    }

    #[test]
    fn test_neuromod_modulation_affects_thresholds() {
        let mut bridge = ScreenVisionBridge::new(small_config(true));
        let baseline_threshold = bridge.foveation().effective_surprise_threshold();
        bridge.modulate(2.0, 1.0);
        let high_ne_threshold = bridge.foveation().effective_surprise_threshold();
        assert!(high_ne_threshold > baseline_threshold);
        bridge.modulate(0.5, 1.0);
        let low_ne_threshold = bridge.foveation().effective_surprise_threshold();
        assert!(low_ne_threshold < baseline_threshold);
        bridge.modulate(1.0, 2.0);
        let high_da_concurrent = bridge.foveation().effective_max_concurrent();
        bridge.modulate(1.0, 0.5);
        let low_da_concurrent = bridge.foveation().effective_max_concurrent();
        assert!(high_da_concurrent >= low_da_concurrent);
    }

    #[test]
    fn test_config_defaults_are_reasonable() {
        let config = ScreenVisionConfig::default();
        assert_eq!(config.capture_width, 360);
        assert_eq!(config.capture_height, 640);
        assert!(config.capture_height > config.capture_width);
        assert!(config.target_fps > 0.0 && config.target_fps <= 10.0);
        assert!(config.surprise_threshold > 0.0 && config.surprise_threshold <= 1.0);
        assert!(config.enable_foveation);
        assert!(config.max_salient_regions > 0 && config.max_salient_regions <= 16);
    }

    #[test]
    fn test_telemetry_updates_after_processing() {
        let mut bridge = ScreenVisionBridge::new(small_config(false));
        assert_eq!(bridge.telemetry().frames_processed, 0);
        let frame = solid_frame(64, 64, 128, 128, 128);
        bridge.process_frame(&frame, 64, 64);
        let tel_after = bridge.telemetry();
        assert_eq!(tel_after.frames_processed, 1);
        assert!(tel_after.vision_telemetry.is_some());
    }

    #[test]
    fn test_salient_regions_respect_max_cap() {
        let config = ScreenVisionConfig {
            surprise_threshold: 0.01,
            max_salient_regions: 2,
            ..small_config(false)
        };
        let mut bridge = ScreenVisionBridge::new(config);
        let dark = solid_frame(64, 64, 0, 0, 0);
        for _ in 0..3 {
            bridge.process_frame(&dark, 64, 64);
        }
        let rect_frame = frame_with_rectangle(64, 64, 8, 8, 32, 32);
        let perception = bridge.process_frame(&rect_frame, 64, 64);
        assert!(perception.salient_regions.len() <= 2);
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
        assert_eq!(perception.scene_hv.len(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    #[should_panic(expected = "frame_rgb length")]
    fn test_process_frame_panics_on_size_mismatch() {
        let mut bridge = ScreenVisionBridge::new(small_config(false));
        let bad_frame = vec![128u8; 32 * 32 * 3];
        bridge.process_frame(&bad_frame, 64, 64);
    }
}
