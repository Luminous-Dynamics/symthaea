// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Golden-ratio multi-layer composition.
//!
//! Combines multiple generative subsystems into a single artwork,
//! using consciousness level to control how many layers are active
//! and golden-ratio subdivision for layout.

use rand::rngs::StdRng;
use symthaea_canvas::scene_graph::{Style, Transform};
use symthaea_canvas::{CognitiveSnapshot, SceneNode};

use crate::AtelierConfig;

/// Inverse golden ratio for layout subdivision.
const INV_PHI: f32 = 0.618;

/// Generate a composite artwork layering multiple subsystems.
///
/// Consciousness level controls layer count:
/// - Low Psi (< 0.3): single subsystem (color field only)
/// - Medium Psi (0.3-0.6): two subsystems (color field + curves)
/// - High Psi (> 0.6): all subsystems layered
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    let psi = snapshot.consciousness_level as f32;
    let mut root = SceneNode::group(Some("composition"));

    // Layer 1: Color field (always present) — background wash
    let bg_config = AtelierConfig {
        max_elements: config.max_elements / 3,
        ..*config
    };
    let bg = crate::color_field::generate(&bg_config, snapshot, rng);
    root.children.push(bg);

    // Layer 2: Persistence texture (if Psi > 0.2)
    if psi > 0.2 && !snapshot.persistence_components.is_empty() {
        let tex_config = AtelierConfig {
            max_elements: config.max_elements / 4,
            ..*config
        };
        let tex = crate::persistence::generate(&tex_config, snapshot, rng);
        // Position in golden-ratio subdivision
        let tex_layer = SceneNode::group(None)
            .with_child(tex)
            .with_transform(Transform {
                translate_x: config.width * (1.0 - INV_PHI) * 0.5,
                translate_y: config.height * (1.0 - INV_PHI) * 0.5,
                scale: INV_PHI,
                ..Transform::identity()
            })
            .with_style(Style {
                opacity: Some(0.6),
                ..Style::default()
            });
        root.children.push(tex_layer);
    }

    // Layer 3: Parametric curves (if Psi > 0.4)
    if psi > 0.4 {
        let curve_config = AtelierConfig {
            max_elements: config.max_elements / 3,
            ..*config
        };
        let curves = crate::curves::generate(&curve_config, snapshot, rng);
        let curve_layer = SceneNode::group(None).with_child(curves).with_style(Style {
            opacity: Some(0.5 + psi * 0.3),
            ..Style::default()
        });
        root.children.push(curve_layer);
    }

    // Layer 4: Harmony shapes — one shape per dominant harmony (if Psi > 0.5)
    if psi > 0.5 {
        let mut harmony_layer = SceneNode::group(Some("harmony-shapes"));
        // Find top 3 active harmonies
        let mut harmony_pairs: Vec<(usize, f32)> = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        harmony_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, &(idx, activation)) in harmony_pairs.iter().take(3).enumerate() {
            if activation < 0.3 {
                break;
            }
            // Position: golden spiral layout
            let angle = rank as f32 * std::f32::consts::TAU * INV_PHI;
            let dist = config.width * 0.2 * (rank as f32 + 1.0);
            let hx = config.width / 2.0 + dist * angle.cos() * 0.3;
            let hy = config.height / 2.0 + dist * angle.sin() * 0.3;
            let radius = config.width * 0.08 + activation * config.width * 0.06;
            let hue = idx as f32 * 45.0;

            let shape = crate::harmony_shapes::harmony_shape(idx, activation, hx, hy, radius, hue);
            harmony_layer.children.push(shape);
        }
        root.children.push(harmony_layer);
    }

    // Layer 5: L-system (if Psi > 0.6) — foreground detail
    if psi > 0.6 {
        let ls_config = AtelierConfig {
            max_elements: config.max_elements / 4,
            ..*config
        };
        let lsys = crate::lsystem::generate(&ls_config, snapshot, rng);
        let ls_layer = SceneNode::group(None)
            .with_child(lsys)
            .with_transform(Transform {
                translate_x: config.width * INV_PHI * 0.3,
                translate_y: config.height * INV_PHI * 0.2,
                scale: 0.8,
                ..Transform::identity()
            })
            .with_style(Style {
                opacity: Some(0.7),
                ..Style::default()
            });
        root.children.push(ls_layer);
    }

    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn low_psi_single_layer() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.1,
            dopamine: 0.5,
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Only color field layer
        assert_eq!(scene.children.len(), 1);
    }

    #[test]
    fn high_psi_all_layers() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.9,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5; 8],
            persistence_components: vec![[0.0, 0.5]],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // All 5 layers (color field + persistence + curves + harmony shapes + L-system)
        assert_eq!(scene.children.len(), 5);
    }

    #[test]
    fn medium_psi_partial() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.5,
            dopamine: 0.5,
            persistence_components: vec![[0.0, 0.5]],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let scene = generate(&config, &snapshot, &mut rng);
        // Color field + persistence + curves = 3 (harmony shapes need Psi > 0.5, not >=)
        assert!(
            scene.children.len() >= 3,
            "got {} layers",
            scene.children.len()
        );
    }
}
