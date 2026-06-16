// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UCL Cross-Domain Frame integration for the cognitive loop.
//!
//! Wraps `symthaea_core::hdc::UCLFrameSystem` into a cycle-compatible
//! interface with interval-gated frame detection and compressed_state enrichment.
//! Detects semantic frames (TRADE, CONFLICT, FEEDBACK_LOOP, NORM_ENFORCEMENT,
//! COOPERATION, ADAPTATION) from the perception HDC encoding.

#[cfg(feature = "ucl-frames")]
use symthaea_core::hdc::UCLFrameSystem;

/// Telemetry snapshot from UCL frame detection.
#[derive(Debug, Clone, Default)]
pub struct UCLFrameTelemetry {
    pub frames_detected: usize,
    pub top_frame: String,
    pub top_score: f32,
    pub detected_names: Vec<String>,
    pub queried_this_cycle: bool,
    pub effective_interval: usize,
}

/// A single detected frame with its similarity score.
#[derive(Debug, Clone)]
pub struct DetectedFrame {
    pub name: String,
    pub similarity: f32,
}

#[cfg(feature = "ucl-frames")]
pub(crate) struct UCLFrameIntegration {
    system: UCLFrameSystem,
    last_detected: Vec<DetectedFrame>,
    queried_this_cycle: bool,
    last_effective_interval: usize,
}

#[cfg(feature = "ucl-frames")]
impl std::fmt::Debug for UCLFrameIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UCLFrameIntegration")
            .field("queried_this_cycle", &self.queried_this_cycle)
            .field("frames_detected", &self.last_detected.len())
            .field("last_effective_interval", &self.last_effective_interval)
            .finish()
    }
}

#[cfg(feature = "ucl-frames")]
impl UCLFrameIntegration {
    pub fn new() -> Self {
        Self {
            system: UCLFrameSystem::new(),
            last_detected: Vec::new(),
            queried_this_cycle: false,
            last_effective_interval: 0,
        }
    }

    /// Detect semantic frames from the perception encoding.
    ///
    /// Runs frame detection against all 6 UCL frames, storing matches
    /// above `threshold` and enriching `compressed_state` with detected
    /// frame encodings weighted by similarity.
    ///
    /// Returns true if detection was performed this cycle.
    pub fn detect_cycle(
        &mut self,
        cycle_number: usize,
        base_interval: usize,
        detection_threshold: f32,
        blend_weight: f32,
        hv16: &symthaea_core::hdc::BinaryHV,
        compressed_state: &mut [f32],
    ) -> bool {
        let effective_interval = base_interval.max(1);
        self.last_effective_interval = effective_interval;

        if cycle_number % effective_interval != 0 {
            return false;
        }

        self.queried_this_cycle = true;

        // Detect frames from the binary perception encoding
        let detections = self.system.detect_frame(hv16, detection_threshold);
        self.last_detected = detections
            .iter()
            .map(|(name, sim)| DetectedFrame {
                name: name.to_string(),
                similarity: *sim,
            })
            .collect();

        // Sort by similarity (highest first)
        self.last_detected.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Blend detected frame encodings into compressed_state
        for detected in &self.last_detected {
            if let Some(frame) = self.system.get_frame(&detected.name) {
                let w = blend_weight * detected.similarity.clamp(0.0, 1.0);
                let frame_bits = &frame.frame_encoding;
                // Convert BinaryHV to continuous for blending
                let frame_continuous = frame_bits.to_continuous();
                let frame_vals = &frame_continuous.values;
                let stride = if frame_vals.len() > compressed_state.len() {
                    frame_vals.len() / compressed_state.len()
                } else {
                    1
                };
                for (i, state_val) in compressed_state.iter_mut().enumerate() {
                    let frame_component = frame_vals.get(i * stride).copied().unwrap_or(0.0);
                    *state_val = (1.0 - w) * *state_val + w * frame_component;
                }
            }
        }

        true
    }

    /// Get the last detected frames.
    pub fn last_detected(&self) -> &[DetectedFrame] {
        &self.last_detected
    }

    /// Collect telemetry and reset per-cycle flag.
    pub fn telemetry(&mut self) -> UCLFrameTelemetry {
        let (top_frame, top_score) = self
            .last_detected
            .first()
            .map(|d| (d.name.clone(), d.similarity))
            .unwrap_or_default();
        let telem = UCLFrameTelemetry {
            frames_detected: self.last_detected.len(),
            top_frame,
            top_score,
            detected_names: self.last_detected.iter().map(|d| d.name.clone()).collect(),
            queried_this_cycle: self.queried_this_cycle,
            effective_interval: self.last_effective_interval,
        };
        self.queried_this_cycle = false;
        telem
    }
}

#[cfg(all(test, feature = "ucl-frames"))]
mod tests {
    use super::*;

    #[test]
    fn test_ucl_frame_integration_creates() {
        let ufi = UCLFrameIntegration::new();
        assert!(!ufi.queried_this_cycle);
        assert!(ufi.last_detected.is_empty());
    }

    #[test]
    fn test_ucl_frame_interval_skips() {
        let mut ufi = UCLFrameIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random();
        let mut state = vec![0.0f32; 128];
        // Cycle 1 with interval 7 should not query
        let detected = ufi.detect_cycle(1, 7, 0.3, 0.1, &hv, &mut state);
        assert!(!detected);
    }

    #[test]
    fn test_ucl_frame_fires_on_interval() {
        let mut ufi = UCLFrameIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random();
        let mut state = vec![0.0f32; 128];
        // Cycle 7 with interval 7 should query
        let detected = ufi.detect_cycle(7, 7, 0.3, 0.1, &hv, &mut state);
        assert!(detected);
        assert!(ufi.queried_this_cycle);
    }

    #[test]
    fn test_ucl_frame_telemetry_resets() {
        let mut ufi = UCLFrameIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random();
        let mut state = vec![0.0f32; 128];
        ufi.detect_cycle(7, 7, 0.3, 0.1, &hv, &mut state);
        assert!(ufi.queried_this_cycle);
        let telem = ufi.telemetry();
        assert!(telem.queried_this_cycle);
        assert!(!ufi.queried_this_cycle); // reset after drain
    }
}
