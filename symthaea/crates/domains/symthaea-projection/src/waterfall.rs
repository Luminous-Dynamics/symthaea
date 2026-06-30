// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Time-Waterfall buffer — the 64-frame history ring for the Time-Waterfall projection.
//!
//! ## Visual Grammar
//!
//! - Age 0 (front plane): present state — full opacity, full size
//! - Age N (rear planes): past states — exponentially decaying opacity
//! - Narrowing shape across ages: collapse
//! - Widening chaotic shape: instability
//! - Stable repeated shape: attractor
//! - Broken / discontinuous: missing evidence
//! - Over-smooth: suspicious false-green / Null masking

use serde::{Deserialize, Serialize};

use crate::frame::ProjectionFrame;

/// Configuration for the Time-Waterfall renderer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallConfig {
    /// Number of historical frames to render (depth slots).
    pub history_depth: usize,
    /// Opacity decay rate per frame. 0.95 = gentle, 0.70 = aggressive.
    pub opacity_decay_rate: f64,
    /// Y-offset per frame for 2.5D depth simulation (pixels or normalized units).
    pub depth_y_offset: f32,
    /// Scale reduction per frame (1.0 = no reduction, 0.98 = slight).
    pub depth_scale_reduction: f32,
    /// Whether to show event markers on the waterfall ribbon.
    pub show_event_markers: bool,
    /// Whether to show anomaly highlighting.
    pub show_anomaly_highlighting: bool,
    /// Pause/scrub state.
    pub paused: bool,
    /// Scrub position when paused (0 = live, N = N frames back).
    pub scrub_offset: usize,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            history_depth: 64,
            opacity_decay_rate: 0.95,
            depth_y_offset: 8.0,
            depth_scale_reduction: 0.985,
            show_event_markers: true,
            show_anomaly_highlighting: true,
            paused: false,
            scrub_offset: 0,
        }
    }
}

/// A 64-frame ring buffer of [`ProjectionFrame`]s for the Time-Waterfall.
///
/// Age 0 = most recent (front plane). Age N = oldest (rear plane).
pub struct WaterfallBuffer {
    frames: Vec<Option<ProjectionFrame>>,
    capacity: usize,
    head: usize,
    len: usize,
    pub config: WaterfallConfig,
}

impl WaterfallBuffer {
    pub fn new(config: WaterfallConfig) -> Self {
        let cap = config.history_depth;
        Self {
            frames: vec![None; cap],
            capacity: cap,
            head: 0,
            len: 0,
            config,
        }
    }

    /// Push a new frame into the waterfall.
    pub fn push(&mut self, frame: ProjectionFrame) {
        self.frames[self.head] = Some(frame);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get a frame by age (0 = most recent, N = oldest stored).
    pub fn get_by_age(&self, age: usize) -> Option<&ProjectionFrame> {
        if age >= self.len {
            return None;
        }
        let idx = (self.head + self.capacity - 1 - age) % self.capacity;
        self.frames[idx].as_ref()
    }

    /// Number of frames currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Compute opacity for a frame at `age`.
    pub fn opacity_at(&self, age: usize) -> f32 {
        let decay = self.config.opacity_decay_rate;
        (decay.powi(age as i32) as f32).max(0.05)
    }

    /// Compute the depth Y offset for a frame at `age` (for 2.5D rendering).
    pub fn depth_offset_at(&self, age: usize) -> f32 {
        self.config.depth_y_offset * age as f32
    }

    /// Compute scale for a frame at `age`.
    pub fn scale_at(&self, age: usize) -> f32 {
        self.config.depth_scale_reduction.powi(age as i32)
    }

    /// Return an iterator over (age, frame, opacity, depth_offset, scale).
    ///
    /// Suitable for direct use by a renderer.
    pub fn render_frames(&self) -> impl Iterator<Item = RenderSlot<'_>> {
        let effective_age = if self.config.paused {
            self.config.scrub_offset
        } else {
            0
        };

        (0..self.len).filter_map(move |i| {
            let age = i + effective_age;
            self.get_by_age(age).map(|frame| RenderSlot {
                age,
                frame,
                opacity: self.opacity_at(age),
                depth_offset: self.depth_offset_at(i), // visual depth from front
                scale: self.scale_at(i),
            })
        })
    }

    /// Detect waterfall shape anomalies in the current history.
    ///
    /// Returns a list of human-readable anomaly descriptions.
    pub fn detect_shape_anomalies(&self) -> Vec<WaterfallAnomaly> {
        let mut anomalies = vec![];

        if self.len < 4 {
            return anomalies;
        }

        // Check for false-green: all recent frames have high confidence, no anomaly tags,
        // but scalar variance is suspiciously low.
        let recent_confidences: Vec<f64> = (0..self.len.min(16))
            .filter_map(|age| self.get_by_age(age).map(|f| f.confidence))
            .collect();

        let mean_conf = recent_confidences.iter().sum::<f64>() / recent_confidences.len() as f64;
        let conf_variance = recent_confidences
            .iter()
            .map(|c| (c - mean_conf).powi(2))
            .sum::<f64>()
            / recent_confidences.len() as f64;

        if mean_conf > 0.9 && conf_variance < 0.001 {
            anomalies.push(WaterfallAnomaly::SuspiciousFalseGreen {
                mean_confidence: mean_conf,
                confidence_variance: conf_variance,
            });
        }

        // Check for data loss (hard disappearances — None frames in history)
        let missing = (0..self.len.min(16))
            .filter(|&age| self.get_by_age(age).is_none())
            .count();
        if missing > 0 {
            anomalies.push(WaterfallAnomaly::MissingData {
                frame_count: missing,
            });
        }

        anomalies
    }
}

/// One slot in the waterfall ready for rendering.
pub struct RenderSlot<'a> {
    pub age: usize,
    pub frame: &'a ProjectionFrame,
    pub opacity: f32,
    /// Y offset from front plane (for 2.5D simulation).
    pub depth_offset: f32,
    /// Scale multiplier (smaller = deeper).
    pub scale: f32,
}

/// Anomalies detected in the waterfall shape.
#[derive(Debug, Clone)]
pub enum WaterfallAnomaly {
    /// Confidence is very high and variance is suspiciously low — possible Null masking.
    SuspiciousFalseGreen {
        mean_confidence: f64,
        confidence_variance: f64,
    },
    /// Some frames in the history are missing (data loss or archive damage).
    MissingData { frame_count: usize },
    /// Metric variance has collapsed to near-zero (over-smooth ribbon).
    MetricVarianceCollapse { metric_name: String },
}

/// Placeholder for Phase 0 visual grammar reference doc.
pub mod visual_grammar_ref {
    /// Returns a text summary of the visual grammar rules.
    /// This is used to populate the Phase 0 static reference sheet.
    pub fn grammar_summary() -> &'static str {
        r#"
# Holographic 2.5D Projection System — Visual Grammar Reference

## Depth Meanings (IMMUTABLE — depth is never decorative)
- Time-Waterfall:     depth = time (front=present, rear=past)
- Stratified Stack:   depth = abstraction layer (low=physical, high=civic)
- Cross-Section:      depth = evidence/source-chain depth

## Color Roles
- blue/cyan      → physical signal, pressure, flow (PhysicalSignal)
- amber/gold     → Chronicle, durable civic truth (Chronicle)
- green/organic  → ecology, mycelium, living signal (Ecology)
- violet/purple  → memory, replay, hidden structure (Memory)
- red/orange     → danger, heat, instability (Danger)
- white/clean    → machine diagnostic truth (MachineTruth)
- sterile white  → SUSPICIOUS false-green / Null masking (FalseGreen)
- grey/static    → archive damage, missing evidence (ArchiveDamage)

## Line Styles
- crisp          → verified signal
- dashed         → inferred signal
- broken         → missing data
- trembling      → high variance / unstable
- too-smooth     → SUSPICIOUS artificial consistency (Null masking)
- braided        → multi-source agreement
- diverging      → contradiction

## Opacity
- high           → current or high-confidence
- low            → past, uncertain, weak evidence
- flickering     → unstable sensor or archive damage
- fading         → decaying relevance
- hard vanish    → data loss

## Motion (state-driven only — no decorative pulsing)
- ripple         → perturbation entering system
- contraction    → collapse
- expansion      → growing uncertainty
- spiral         → recursive/recurrent process
- drift          → slow bias accumulation
- snap           → discrete authority change
- bloom          → ecological response
- fracture       → MIP cut or trust break

## Waterfall Shape Grammar
- stable ribbon      → attractor
- narrowing tunnel   → collapse
- widening chaos     → instability
- warped ribbon      → perturbation
- broken ribbon      → missing evidence
- over-smooth ribbon → SUSPICIOUS false-green / Null masking
- amber marker ring  → Chronicle durable event

## Anti-Patterns (forbidden)
- decorative depth
- glowing everything
- too many simultaneous labels
- particle effects hiding evidence
- dashboard looking more certain than the data
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{ProjectionFrame, ProjectionMode, SourceSystem};
    use crate::layer::LayerId;

    fn make_frame(id: u64, confidence: f64) -> ProjectionFrame {
        let mut f = ProjectionFrame::new(
            id,
            id as f64,
            SourceSystem::Fep,
            ProjectionMode::TimeWaterfall,
            LayerId::Fep,
        );
        f.confidence = confidence;
        f
    }

    #[test]
    fn push_and_retrieve_by_age() {
        let mut buf = WaterfallBuffer::new(WaterfallConfig::default());
        buf.push(make_frame(1, 0.9));
        buf.push(make_frame(2, 0.8));
        buf.push(make_frame(3, 0.7));
        assert_eq!(buf.get_by_age(0).unwrap().frame_id, 3); // most recent
        assert_eq!(buf.get_by_age(2).unwrap().frame_id, 1); // oldest
    }

    #[test]
    fn opacity_decays_with_age() {
        let buf = WaterfallBuffer::new(WaterfallConfig::default());
        assert_eq!(buf.opacity_at(0), 1.0);
        assert!(buf.opacity_at(10) < 0.8);
        assert!(buf.opacity_at(63) >= 0.05);
    }

    #[test]
    fn false_green_detected_when_all_high_confidence() {
        let mut buf = WaterfallBuffer::new(WaterfallConfig::default());
        for i in 0..20 {
            buf.push(make_frame(i, 0.99)); // suspiciously uniform high confidence
        }
        let anomalies = buf.detect_shape_anomalies();
        let has_false_green = anomalies
            .iter()
            .any(|a| matches!(a, WaterfallAnomaly::SuspiciousFalseGreen { .. }));
        assert!(
            has_false_green,
            "should detect suspicious false-green pattern"
        );
    }

    #[test]
    fn render_frames_count_matches_len() {
        let mut buf = WaterfallBuffer::new(WaterfallConfig::default());
        for i in 0..10u64 {
            buf.push(make_frame(i, 0.9));
        }
        let count = buf.render_frames().count();
        assert_eq!(count, 10);
    }
}
