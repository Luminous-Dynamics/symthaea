// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hybrid neural-procedural generation: CfC dynamics modulate hand-designed generators.
//!
//! Instead of generating raw coordinates from CfC output (which produces noise),
//! the neural network selects WHICH procedural generators to use, parameterizes
//! them, and controls transitions between styles. The procedural generators
//! (L-systems, harmony shapes, curves, color fields) provide the visual quality;
//! the CfC dynamics provide the compositional intelligence.
//!
//! ```text
//! CfC output HV → decode to HybridParams:
//!   - which generators to activate (selection weights)
//!   - L-system depth/angle modulation
//!   - curve frequency/amplitude modulation
//!   - color field warmth/energy modulation
//!   - harmony shape scale/position
//!   - overall composition layout bias
//! → Feed into existing procedural generators with these params
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use symthaea_canvas::scene_graph::{Style, Transform};
use symthaea_canvas::{CognitiveSnapshot, SceneNode};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::AtelierConfig;

/// Parameters decoded from CfC output for hybrid generation.
struct HybridParams {
    /// Activation weights for each generator (0-1, softmax-normalized).
    generator_weights: [f32; 5], // lsystem, curves, persistence, colorfield, harmony_shapes
    /// L-system modulation: depth scale, angle scale.
    lsystem_depth_mod: f32,
    lsystem_angle_mod: f32,
    /// Curve modulation: frequency and amplitude scales.
    curve_freq_mod: f32,
    curve_amp_mod: f32,
    /// Color warmth shift (-1 to 1).
    color_warmth: f32,
    /// Harmony shape scale.
    shape_scale: f32,
    /// Layout center offset (normalized -0.3 to 0.3).
    layout_x: f32,
    layout_y: f32,
    /// Overall opacity modulation.
    opacity_mod: f32,
}

/// CfC controller for hybrid generation.
pub struct HybridController {
    network: HdcLtcUnifiedNetwork,
    projections: Vec<ContinuousHV>,
    dt: f32,
}

impl HybridController {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![4, 4],
            neuron_config: UnifiedConfig {
                tau_base: 0.04,
                backbone_tau: 0.35,
                activation: UnifiedActivation::Tanh,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: true,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        // 14 projection vectors for HybridParams
        let projections: Vec<ContinuousHV> = (0..14)
            .map(|i| genesis.hv(&format!("hybrid_proj_{i}"), dim))
            .collect();

        Self {
            network,
            projections,
            dt: 0.03,
        }
    }

    fn encode_snapshot(snapshot: &CognitiveSnapshot, genesis: &GenesisSeed) -> ContinuousHV {
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let channels: Vec<(f32, &str)> = vec![
            (snapshot.consciousness_level as f32, "psi"),
            (snapshot.valence, "valence"),
            (snapshot.arousal, "arousal"),
            (snapshot.dopamine, "dopamine"),
            (snapshot.serotonin, "serotonin"),
            (snapshot.noradrenaline, "noradrenaline"),
            (snapshot.harmony_activations[0], "h0"),
            (snapshot.harmony_activations[3], "h3"),
            (snapshot.harmony_activations[7], "h7"),
            (snapshot.prediction_error, "pe"),
        ];
        let mut result = ContinuousHV::zero(dim);
        for (value, label) in &channels {
            result = result.add(&genesis.hv(label, dim).scale(*value));
        }
        result.normalize()
    }

    fn decode_params(&self, output: &ContinuousHV) -> HybridParams {
        let v: Vec<f32> = self
            .projections
            .iter()
            .map(|p| (output.similarity(p) + 1.0) / 2.0)
            .collect();

        // Softmax the first 5 for generator selection
        let raw_weights = [v[0], v[1], v[2], v[3], v[4]];
        let max_w = raw_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_weights: Vec<f32> = raw_weights.iter().map(|w| (w - max_w).exp()).collect();
        let sum_exp: f32 = exp_weights.iter().sum();
        let mut generator_weights = [0.0f32; 5];
        for i in 0..5 {
            generator_weights[i] = exp_weights[i] / sum_exp;
        }

        HybridParams {
            generator_weights,
            lsystem_depth_mod: 0.5 + v[5] * 1.0, // 0.5-1.5x
            lsystem_angle_mod: 0.7 + v[6] * 0.6, // 0.7-1.3x
            curve_freq_mod: 0.5 + v[7] * 1.5,    // 0.5-2.0x
            curve_amp_mod: 0.5 + v[8] * 1.0,     // 0.5-1.5x
            color_warmth: v[9] * 2.0 - 1.0,      // -1 to 1
            shape_scale: 0.5 + v[10] * 1.0,      // 0.5-1.5x
            layout_x: v[11] * 0.6 - 0.3,         // -0.3 to 0.3
            layout_y: v[12] * 0.6 - 0.3,         // -0.3 to 0.3
            opacity_mod: 0.6 + v[13] * 0.4,      // 0.6-1.0
        }
    }

    /// Generate a hybrid composition: CfC selects and parameterizes procedural generators.
    pub fn generate(&mut self, config: &AtelierConfig, snapshot: &CognitiveSnapshot) -> SceneNode {
        let genesis = GenesisSeed::from_phrase("hybrid-controller");
        let input = Self::encode_snapshot(snapshot, &genesis);

        // Evolve CfC to get compositional parameters
        self.network.evolve_closed_form(self.dt, &input);
        let output = self.network.output();
        let params = self.decode_params(&output);

        let mut root = SceneNode::group(Some("hybrid"));
        let mut rng = StdRng::seed_from_u64(snapshot.cycle_count.wrapping_add(42));

        let threshold = 0.15; // minimum weight to activate a generator

        // Layer generators based on CfC-determined weights
        if params.generator_weights[0] > threshold {
            // L-system: modulated depth and angle
            let mut mod_snapshot = snapshot.clone();
            // Increase/decrease EvolutionaryProgression (controls L-system depth)
            mod_snapshot.harmony_activations[6] *= params.lsystem_depth_mod;
            let ls = crate::lsystem::generate(config, &mod_snapshot, &mut rng);
            let layer = SceneNode::group(None)
                .with_child(ls)
                .with_style(Style {
                    opacity: Some(params.generator_weights[0] * params.opacity_mod),
                    ..Style::default()
                })
                .with_transform(Transform {
                    translate_x: config.width * params.layout_x,
                    translate_y: config.height * params.layout_y,
                    ..Transform::identity()
                });
            root.children.push(layer);
        }

        if params.generator_weights[1] > threshold {
            // Curves: modulated frequency and amplitude
            let mut mod_snapshot = snapshot.clone();
            // Scale thought vector to affect curve params
            for v in mod_snapshot.thought_vector.iter_mut() {
                *v *= params.curve_freq_mod;
            }
            mod_snapshot.arousal *= params.curve_amp_mod;
            let curves = crate::curves::generate(config, &mod_snapshot, &mut rng);
            let layer = SceneNode::group(None).with_child(curves).with_style(Style {
                opacity: Some(params.generator_weights[1] * params.opacity_mod),
                ..Style::default()
            });
            root.children.push(layer);
        }

        if params.generator_weights[2] > threshold {
            let tex = crate::persistence::generate(config, snapshot, &mut rng);
            let layer = SceneNode::group(None).with_child(tex).with_style(Style {
                opacity: Some(params.generator_weights[2] * params.opacity_mod),
                ..Style::default()
            });
            root.children.push(layer);
        }

        if params.generator_weights[3] > threshold {
            // Color field: warmth modulation
            let mut mod_snapshot = snapshot.clone();
            mod_snapshot.valence =
                (mod_snapshot.valence + params.color_warmth * 0.5).clamp(-1.0, 1.0);
            let cf = crate::color_field::generate(config, &mod_snapshot, &mut rng);
            let layer = SceneNode::group(None).with_child(cf).with_style(Style {
                opacity: Some(params.generator_weights[3] * params.opacity_mod * 0.7),
                ..Style::default()
            });
            root.children.push(layer);
        }

        if params.generator_weights[4] > threshold {
            // Harmony shapes: scaled
            let mut harmony_layer = SceneNode::group(Some("hybrid-shapes"));
            let mut pairs: Vec<(usize, f32)> = snapshot
                .harmony_activations
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, &(idx, activation)) in pairs.iter().take(3).enumerate() {
                if activation < 0.3 {
                    break;
                }
                let angle = rank as f32 * std::f32::consts::TAU * 0.618;
                let dist = config.width * 0.15 * (rank as f32 + 1.0);
                let hx =
                    config.width / 2.0 + dist * angle.cos() * 0.3 + config.width * params.layout_x;
                let hy = config.height / 2.0
                    + dist * angle.sin() * 0.3
                    + config.height * params.layout_y;
                let radius =
                    config.width * 0.06 * params.shape_scale + activation * config.width * 0.04;
                let hue = idx as f32 * 45.0;
                let shape =
                    crate::harmony_shapes::harmony_shape(idx, activation, hx, hy, radius, hue);
                harmony_layer.children.push(shape);
            }

            let layer = SceneNode::group(None)
                .with_child(harmony_layer)
                .with_style(Style {
                    opacity: Some(params.generator_weights[4] * params.opacity_mod),
                    ..Style::default()
                });
            root.children.push(layer);
        }

        root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn hybrid_generates_multi_layer() {
        let genesis = GenesisSeed::from_phrase("test-hybrid");
        let mut ctrl = HybridController::new(&genesis);
        let config = AtelierConfig::default();
        let scene = ctrl.generate(&config, &test_snapshot());
        // Should activate at least 2 generators
        assert!(
            scene.children.len() >= 2,
            "got {} layers",
            scene.children.len()
        );
    }

    #[test]
    fn hybrid_different_states_different_composition() {
        let genesis = GenesisSeed::from_phrase("test-hybrid-diff");
        let config = AtelierConfig::default();

        let mut c1 = HybridController::new(&genesis);
        let serene = CognitiveSnapshot {
            consciousness_level: 0.9,
            valence: 0.7,
            arousal: 0.2,
            harmony_activations: [0.8, 0.7, 0.6, 0.2, 0.7, 0.6, 0.4, 0.9],
            thought_vector: vec![0.1, 0.0],
            ..CognitiveSnapshot::dormant()
        };
        let s1 = c1.generate(&config, &serene);

        let mut c2 = HybridController::new(&genesis);
        let turb = CognitiveSnapshot {
            consciousness_level: 0.5,
            valence: -0.7,
            arousal: 0.9,
            harmony_activations: [0.2, 0.1, 0.3, 0.9, 0.5, 0.1, 0.8, 0.1],
            noradrenaline: 0.8,
            thought_vector: vec![0.8, -0.7],
            ..CognitiveSnapshot::dormant()
        };
        let s2 = c2.generate(&config, &turb);

        // Different states should produce different layer counts or content
        let svg1 = symthaea_canvas::render_svg(&s1, serene.consciousness_level);
        let svg2 = symthaea_canvas::render_svg(&s2, turb.consciousness_level);
        assert_ne!(svg1, svg2);
    }

    #[test]
    fn hybrid_generator_weights_sum_to_one() {
        let genesis = GenesisSeed::from_phrase("test-hybrid-weights");
        let ctrl = HybridController::new(&genesis);
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let output = ContinuousHV::random(dim, 42);
        let params = ctrl.decode_params(&output);
        let sum: f32 = params.generator_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "weights sum = {sum}");
    }
}
