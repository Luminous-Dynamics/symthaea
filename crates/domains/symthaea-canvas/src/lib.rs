// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-canvas
//!
//! Living topology UI: real-time SVG projection of Symthaea's cognitive geometry.
//!
//! Converts cognitive telemetry (consciousness level, neuromodulators, Betti numbers,
//! Cantor depth, harmony activations) into a declarative scene graph, then renders
//! self-contained animated SVG.
//!
//! Pipeline: `CognitiveSnapshot → AestheticEngine → AestheticState → build_scene() → render_svg()`
//!
//! Total pipeline target: <500µs per frame.

#![deny(unsafe_code)]

pub mod aesthetic_engine;
pub mod aesthetic_score;
pub mod animation;
pub mod color;
pub mod geometry;
pub mod scene_features;
pub mod scene_graph;
pub mod svg_renderer;
pub mod validation;

use serde::{Deserialize, Serialize};

// Re-exports
pub use aesthetic_engine::{AestheticEngine, AestheticState};
pub use aesthetic_score::aesthetic_score;
pub use animation::{FrameContext, MotionPreference};
pub use color::{Color, Palette};
pub use geometry::build_scene;
pub use scene_features::{SceneFeatures, extract_scene_features};
pub use scene_graph::SceneNode;
pub use svg_renderer::{SvgRenderOptions, render_svg, render_svg_with_options};
pub use validation::{SnapshotLimits, SnapshotSanitization};

/// Lightweight snapshot of cognitive state, decoupled from CycleMetadata.
///
/// Extracted once per generation interval from the cognitive loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSnapshot {
    /// Consciousness level Ψ (0.0–1.0).
    pub consciousness_level: f64,
    /// Current prediction error magnitude.
    pub prediction_error: f32,
    /// Living mind vitality measure.
    pub living_mind_vitality: f64,
    /// Living mind coherence measure.
    pub living_mind_coherence: f64,

    // Neuromodulators (effective levels, 0.0–1.0)
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub oxytocin: f32,
    pub gaba: f32,
    pub allostatic_load: f32,

    // Topology
    /// Betti-0: connected components.
    pub betti_0: usize,
    /// Betti-1: loops/cycles.
    pub betti_1: usize,
    /// Betti-2: voids/cavities.
    pub betti_2: usize,
    /// Persistence diagram: birth-death pairs for components.
    pub persistence_components: Vec<[f64; 2]>,
    /// Persistence diagram: birth-death pairs for cycles.
    pub persistence_cycles: Vec<[f64; 2]>,

    // Cantor fractal
    pub cantor_metacognitive_depth: f32,
    pub cantor_last_depth: u8,

    // Affect
    /// Valence in [-1, 1].
    pub valence: f32,
    /// Arousal in [0, 1].
    pub arousal: f32,

    // Harmonies
    /// Eight Harmony activations (0.0–1.0 each).
    pub harmony_activations: [f32; 8],

    // Spatial
    /// Low-dimensional thought vector (typically 32D, first 2 dims used for layout).
    pub thought_vector: Vec<f32>,
    /// Monotonic cycle counter.
    pub cycle_count: u64,
}

impl CognitiveSnapshot {
    /// Create a default/dormant snapshot (minimal consciousness).
    pub fn dormant() -> Self {
        Self {
            consciousness_level: 0.05,
            prediction_error: 0.0,
            living_mind_vitality: 0.0,
            living_mind_coherence: 0.0,
            dopamine: 0.1,
            noradrenaline: 0.1,
            serotonin: 0.3,
            acetylcholine: 0.1,
            oxytocin: 0.2,
            gaba: 0.5,
            allostatic_load: 0.0,
            betti_0: 1,
            betti_1: 0,
            betti_2: 0,
            persistence_components: Vec::new(),
            persistence_cycles: Vec::new(),
            cantor_metacognitive_depth: 0.0,
            cantor_last_depth: 0,
            valence: 0.0,
            arousal: 0.1,
            harmony_activations: [0.0; 8],
            thought_vector: vec![0.0, 0.0],
            cycle_count: 0,
        }
    }
}

/// Full pipeline: snapshot → SVG string.
///
/// Convenience function that runs the entire canvas pipeline.
/// For stateful usage (EMA smoothing), use `AestheticEngine` directly.
pub fn render_snapshot(snap: &CognitiveSnapshot) -> (String, AestheticState) {
    let frame = FrameContext::from_cycle_count(snap.cycle_count);
    render_snapshot_frame(snap, frame)
}

/// Full pipeline on an explicit monotonic frame timeline.
pub fn render_snapshot_frame(
    snap: &CognitiveSnapshot,
    frame: FrameContext,
) -> (String, AestheticState) {
    let mut engine = AestheticEngine::new();
    let mut state = engine.process_frame(snap, frame);
    let scene = build_scene(&state);
    state.aesthetic_score = aesthetic_score(&state, &scene);
    let svg = render_svg_with_options(
        &scene,
        snap.consciousness_level,
        SvgRenderOptions::from_frame(frame),
    );
    (svg, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dormant_snapshot_renders() {
        let snap = CognitiveSnapshot::dormant();
        let (svg, state) = render_snapshot(&snap);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(state.luminosity < 0.2);
    }

    #[test]
    fn full_pipeline_deterministic() {
        let snap = CognitiveSnapshot {
            consciousness_level: 0.7,
            prediction_error: 0.15,
            living_mind_vitality: 0.6,
            living_mind_coherence: 0.5,
            dopamine: 0.6,
            noradrenaline: 0.4,
            serotonin: 0.5,
            acetylcholine: 0.3,
            oxytocin: 0.4,
            gaba: 0.5,
            allostatic_load: 0.2,
            betti_0: 4,
            betti_1: 2,
            betti_2: 1,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            cantor_metacognitive_depth: 0.7,
            cantor_last_depth: 3,
            valence: 0.3,
            arousal: 0.6,
            harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
            thought_vector: vec![0.3, -0.2],
            cycle_count: 500,
        };
        let (svg1, state1) = render_snapshot(&snap);
        let (svg2, state2) = render_snapshot(&snap);
        assert_eq!(svg1, svg2, "SVG should be deterministic");
        assert_eq!(state1.aesthetic_score, state2.aesthetic_score);
    }

    #[test]
    fn low_vs_high_consciousness_differ() {
        let low = CognitiveSnapshot::dormant();
        let mut high = CognitiveSnapshot::dormant();
        high.consciousness_level = 0.9;
        high.betti_0 = 5;
        high.betti_1 = 3;
        high.betti_2 = 1;
        high.arousal = 0.7;
        high.harmony_activations = [0.8; 8];
        high.cantor_last_depth = 4;

        let (svg_low, state_low) = render_snapshot(&low);
        let (svg_high, state_high) = render_snapshot(&high);

        assert!(state_high.luminosity > state_low.luminosity);
        assert!(
            svg_high.len() > svg_low.len(),
            "high Ψ should produce more SVG"
        );
    }

    #[test]
    fn aesthetic_score_bounded() {
        let snap = CognitiveSnapshot {
            consciousness_level: 0.6,
            prediction_error: 0.1,
            living_mind_vitality: 0.5,
            living_mind_coherence: 0.5,
            dopamine: 0.5,
            noradrenaline: 0.3,
            serotonin: 0.5,
            acetylcholine: 0.4,
            oxytocin: 0.3,
            gaba: 0.5,
            allostatic_load: 0.2,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            persistence_components: vec![],
            persistence_cycles: vec![],
            cantor_metacognitive_depth: 0.5,
            cantor_last_depth: 2,
            valence: 0.0,
            arousal: 0.4,
            harmony_activations: [0.5; 8],
            thought_vector: vec![0.0, 0.0],
            cycle_count: 200,
        };
        let (_, state) = render_snapshot(&snap);
        assert!(state.aesthetic_score >= 0.0 && state.aesthetic_score <= 1.0);
    }

    #[test]
    fn snapshot_serialization() {
        let snap = CognitiveSnapshot::dormant();
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: CognitiveSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snap.consciousness_level, back.consciousness_level);
        assert_eq!(snap.betti_0, back.betti_0);
    }
}
