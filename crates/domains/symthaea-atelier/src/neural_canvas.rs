// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-driven visual generation: consciousness → neural dynamics → art.
//!
//! Replaces hand-designed parametric generators with a `HdcLtcUnifiedNetwork`
//! that evolves the cognitive state through temporal dynamics and projects
//! the output HV to visual parameters. This follows the exact pattern of
//! Broca (text) and Vocal Tract (speech), making visual art a native output
//! modality of Symthaea's neural substrate.
//!
//! # Architecture
//!
//! ```text
//! CognitiveSnapshot → encode 12 channels to 16,384D HV
//!                   → CfC network (2 layers × 4 neurons) evolves state
//!                   → output HV projected to VisualFrame (12D)
//!                   → VisualFrame → SVG scene elements
//! ```
//!
//! Each autoregressive step generates one visual element. The CfC temporal
//! dynamics mean that element N is influenced by all previous elements
//! through the network's hidden state — creating compositional coherence.

use std::fmt::Write;
use symthaea_canvas::scene_graph::Style;
use symthaea_canvas::{CognitiveSnapshot, Color, SceneNode};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::AtelierConfig;

/// A single visual element's parameters, decoded from a 16,384D output HV.
struct VisualFrame {
    /// Center X position (normalized 0-1).
    cx: f32,
    /// Center Y position (normalized 0-1).
    cy: f32,
    /// Size/radius (normalized 0-1).
    size: f32,
    /// Rotation angle in degrees.
    rotation: f32,
    /// Hue (0-360).
    hue: f32,
    /// Saturation (0-1).
    saturation: f32,
    /// Lightness (0-1).
    lightness: f32,
    /// Opacity (0-1).
    opacity: f32,
    /// Shape selector (0-1, binned into shape types).
    shape_selector: f32,
    /// Curve amplitude (for path elements).
    curve_amplitude: f32,
    /// Curve frequency.
    curve_frequency: f32,
    /// Stroke width.
    stroke_width: f32,
}

/// CfC-driven visual generation controller.
pub struct NeuralCanvas {
    /// The CfC network that evolves visual state.
    network: HdcLtcUnifiedNetwork,
    /// Pre-computed projection vectors for decoding output HV → VisualFrame.
    /// Each of the 12 VisualFrame dimensions gets a random projection vector.
    projections: Vec<ContinuousHV>,
    /// Time step per visual element (controls temporal coherence).
    dt: f32,
}

impl NeuralCanvas {
    /// Create a new neural canvas from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![4, 4],
            neuron_config: UnifiedConfig {
                tau_base: 0.05,
                backbone_tau: 0.3,
                activation: symthaea_core::hdc::hdc_ltc_unified::UnifiedActivation::Tanh,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        // Create 12 projection vectors for decoding output to VisualFrame
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let projections: Vec<ContinuousHV> = (0..12)
            .map(|i| genesis.hv(&format!("visual_projection_{i}"), dim))
            .collect();

        Self {
            network,
            projections,
            dt: 0.02, // 20ms per element (same as Broca)
        }
    }

    /// Number of projection vectors (VisualFrame dimensions).
    pub fn projection_count(&self) -> usize {
        self.projections.len()
    }

    /// Get a reference to a projection vector by index.
    pub fn get_projection(&self, idx: usize) -> &ContinuousHV {
        &self.projections[idx]
    }

    /// Set a projection vector by index (used by training loop).
    pub fn set_projection(&mut self, idx: usize, hv: ContinuousHV) {
        self.projections[idx] = hv;
    }

    /// Encode a cognitive snapshot into a 16,384D input HV.
    fn encode_snapshot(snapshot: &CognitiveSnapshot, genesis: &GenesisSeed) -> ContinuousHV {
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

        // Encode key dimensions as level-coded HVs, then bundle
        let channels: Vec<(f32, &str)> = vec![
            (snapshot.consciousness_level as f32, "consciousness"),
            (snapshot.valence, "valence"),
            (snapshot.arousal, "arousal"),
            (snapshot.dopamine, "dopamine"),
            (snapshot.serotonin, "serotonin"),
            (snapshot.noradrenaline, "noradrenaline"),
            (snapshot.harmony_activations[0], "harmony_0"),
            (snapshot.harmony_activations[3], "harmony_3"), // InfinitePlay
            (snapshot.harmony_activations[7], "harmony_7"), // SacredStillness
            (snapshot.prediction_error, "prediction_error"),
        ];

        let mut result = ContinuousHV::zero(dim);
        for (value, label) in &channels {
            let basis = genesis.hv(label, dim);
            // Level-code: scale basis by channel value
            let scaled = basis.scale(*value);
            result = result.add(&scaled);
        }
        result.normalize()
    }

    /// Decode a 16,384D output HV into a VisualFrame.
    fn decode_frame(&self, output: &ContinuousHV) -> VisualFrame {
        let vals: Vec<f32> = self
            .projections
            .iter()
            .map(|proj| {
                // Cosine similarity → [-1, 1] → [0, 1]
                (output.similarity(proj) + 1.0) / 2.0
            })
            .collect();

        VisualFrame {
            cx: vals[0],
            cy: vals[1],
            size: vals[2] * 0.3 + 0.01, // min 1%, max 31% of viewport
            rotation: vals[3] * 360.0,
            hue: vals[4] * 360.0,
            saturation: vals[5] * 0.6 + 0.2, // 0.2-0.8
            lightness: vals[6] * 0.4 + 0.3,  // 0.3-0.7
            opacity: vals[7] * 0.5 + 0.2,    // 0.2-0.7
            shape_selector: vals[8],
            curve_amplitude: vals[9],
            curve_frequency: vals[10] * 4.0 + 1.0,
            stroke_width: vals[11] * 4.0 + 0.5,
        }
    }

    /// Generate a scene graph by autoregressively producing visual elements.
    ///
    /// Each CfC step evolves the hidden state and produces one visual element.
    /// The temporal dynamics create compositional coherence — later elements
    /// are influenced by the network's memory of earlier ones.
    pub fn generate(&mut self, config: &AtelierConfig, snapshot: &CognitiveSnapshot) -> SceneNode {
        let genesis = GenesisSeed::from_phrase("atelier-neural-canvas");
        let input_hv = Self::encode_snapshot(snapshot, &genesis);
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

        let num_elements = 5 + (snapshot.consciousness_level * 15.0) as usize;
        let num_elements = num_elements.min(config.max_elements);

        let mut root = SceneNode::group(Some("neural-canvas"));

        for step in 0..num_elements {
            // Perturb input each step to prevent CfC convergence
            let step_basis = genesis.hv(&format!("nc_step_{}", step % 12), dim);
            let phase = (step as f32 * 0.5).sin() * 0.25;
            let perturbed = input_hv.add(&step_basis.scale(phase)).normalize();

            self.network.evolve_closed_form(self.dt, &perturbed);
            let output = self.network.output();
            let frame = self.decode_frame(&output);

            // Convert VisualFrame to SVG element
            let element = self.frame_to_element(&frame, config, step);
            root.children.push(element);
        }

        root
    }

    /// Convert a VisualFrame to an SVG scene node.
    fn frame_to_element(
        &self,
        frame: &VisualFrame,
        config: &AtelierConfig,
        step: usize,
    ) -> SceneNode {
        let x = frame.cx * config.width;
        let y = frame.cy * config.height;
        let size = frame.size * config.width;
        let color = Color::from_hsl(frame.hue, frame.saturation, frame.lightness);

        let style = Style {
            fill: Some(color),
            opacity: Some(frame.opacity),
            stroke: Some(Color::from_hsl(
                frame.hue + 30.0,
                frame.saturation * 0.8,
                frame.lightness - 0.1,
            )),
            stroke_width: Some(frame.stroke_width),
            ..Style::default()
        };

        // Shape selection based on frame.shape_selector
        if frame.shape_selector < 0.25 {
            // Circle
            SceneNode::circle(x, y, size).with_style(style)
        } else if frame.shape_selector < 0.5 {
            // Ellipse (size modulated by step for variety)
            let rx = size * (0.5 + frame.curve_amplitude * 0.5);
            let ry = size * (1.0 - frame.curve_amplitude * 0.3);
            SceneNode::ellipse(x, y, rx, ry).with_style(Style {
                fill: None,
                ..style
            })
        } else if frame.shape_selector < 0.75 {
            // Bezier curve
            let path = self.frame_to_curve(frame, config);
            SceneNode::path(path).with_style(Style {
                fill: None,
                ..style
            })
        } else {
            // Polygon (triangle to hexagon based on step)
            let sides = 3 + (step % 4);
            let points: Vec<(f32, f32)> = (0..sides)
                .map(|i| {
                    let angle = (i as f32 / sides as f32) * std::f32::consts::TAU
                        + frame.rotation.to_radians();
                    (x + size * angle.cos(), y + size * angle.sin())
                })
                .collect();
            SceneNode::polygon(points, true).with_style(style)
        }
    }

    /// Generate a Bezier curve path from a VisualFrame.
    fn frame_to_curve(&self, frame: &VisualFrame, config: &AtelierConfig) -> String {
        let x = frame.cx * config.width;
        let y = frame.cy * config.height;
        let amp = frame.curve_amplitude * config.width * 0.2;
        let freq = frame.curve_frequency;
        let steps = 30;

        let mut path = String::with_capacity(steps * 30);
        let points: Vec<(f32, f32)> = (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let px = x - config.width * 0.2 + t * config.width * 0.4;
                let py = y + amp * (freq * t * std::f32::consts::TAU).sin();
                (px, py)
            })
            .collect();

        let _ = write!(path, "M {:.1} {:.1}", points[0].0, points[0].1);
        for i in 0..points.len() - 1 {
            let p0 = if i > 0 { points[i - 1] } else { points[i] };
            let p1 = points[i];
            let p2 = points[i + 1];
            let p3 = if i + 2 < points.len() {
                points[i + 2]
            } else {
                points[i + 1]
            };

            let tension = 6.0;
            let cp1x = p1.0 + (p2.0 - p0.0) / tension;
            let cp1y = p1.1 + (p2.1 - p0.1) / tension;
            let cp2x = p2.0 - (p3.0 - p1.0) / tension;
            let cp2y = p2.1 - (p3.1 - p1.1) / tension;

            let _ = write!(
                path,
                " C {cp1x:.1} {cp1y:.1}, {cp2x:.1} {cp2y:.1}, {:.1} {:.1}",
                p2.0, p2.1
            );
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-neural-canvas")
    }

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            valence: 0.3,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            prediction_error: 0.1,
            thought_vector: vec![0.3, -0.2],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn neural_canvas_generates_elements() {
        let genesis = test_genesis();
        let mut canvas = NeuralCanvas::new(&genesis);
        let config = AtelierConfig::default();
        let snapshot = test_snapshot();

        let scene = canvas.generate(&config, &snapshot);
        // Should have multiple elements
        assert!(
            scene.children.len() >= 5,
            "got {} elements",
            scene.children.len()
        );
    }

    #[test]
    fn consciousness_affects_element_count() {
        let genesis = test_genesis();
        let config = AtelierConfig::default();

        let mut low_canvas = NeuralCanvas::new(&genesis);
        let low_snap = CognitiveSnapshot {
            consciousness_level: 0.1,
            ..CognitiveSnapshot::dormant()
        };
        let low_scene = low_canvas.generate(&config, &low_snap);

        let mut high_canvas = NeuralCanvas::new(&genesis);
        let high_snap = CognitiveSnapshot {
            consciousness_level: 0.9,
            ..test_snapshot()
        };
        let high_scene = high_canvas.generate(&config, &high_snap);

        assert!(
            high_scene.children.len() > low_scene.children.len(),
            "high {} should > low {}",
            high_scene.children.len(),
            low_scene.children.len()
        );
    }

    #[test]
    fn different_states_different_art() {
        let genesis = test_genesis();
        let config = AtelierConfig::default();

        let mut canvas1 = NeuralCanvas::new(&genesis);
        let snap1 = CognitiveSnapshot {
            valence: 0.8,
            arousal: 0.2,
            harmony_activations: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            ..test_snapshot()
        };
        let scene1 = canvas1.generate(&config, &snap1);

        let mut canvas2 = NeuralCanvas::new(&genesis);
        let snap2 = CognitiveSnapshot {
            valence: -0.8,
            arousal: 0.9,
            harmony_activations: [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1],
            ..test_snapshot()
        };
        let scene2 = canvas2.generate(&config, &snap2);

        // Different states should produce different element counts at minimum
        // (CfC dynamics are state-dependent)
        let svg1 = symthaea_canvas::render_svg(&scene1, snap1.consciousness_level);
        let svg2 = symthaea_canvas::render_svg(&scene2, snap2.consciousness_level);
        assert_ne!(svg1, svg2);
    }

    #[test]
    fn encode_produces_normalized_hv() {
        let genesis = test_genesis();
        let snapshot = test_snapshot();
        let hv = NeuralCanvas::encode_snapshot(&snapshot, &genesis);
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Should be approximately unit norm (normalized)
        assert!((norm - 1.0).abs() < 0.1, "norm = {norm}");
    }

    #[test]
    fn visual_frame_bounded() {
        let genesis = test_genesis();
        let canvas = NeuralCanvas::new(&genesis);
        // Create a random output HV and decode it
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let output = ContinuousHV::random(dim, 42);
        let frame = canvas.decode_frame(&output);

        assert!(frame.cx >= 0.0 && frame.cx <= 1.0);
        assert!(frame.cy >= 0.0 && frame.cy <= 1.0);
        assert!(frame.size > 0.0);
        assert!(frame.opacity >= 0.0 && frame.opacity <= 1.0);
        assert!(frame.hue >= 0.0 && frame.hue <= 360.0);
    }
}
