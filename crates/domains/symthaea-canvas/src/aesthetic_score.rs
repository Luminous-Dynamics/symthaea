// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Birkhoff beauty metric: M = O / C.
//!
//! Order measures symmetry, harmony balance, and consciousness coupling.
//! Complexity measures scene richness (node count, color diversity, path length).

use crate::aesthetic_engine::AestheticState;
use crate::scene_graph::SceneNode;

/// Compute the Birkhoff aesthetic score for a scene.
///
/// Returns a value in [0, 1] where higher = more beautiful.
pub fn aesthetic_score(state: &AestheticState, scene: &SceneNode) -> f32 {
    let order = compute_order(state);
    let complexity = compute_complexity(state, scene);
    if complexity < 0.01 {
        return 0.0;
    }
    (order / complexity).clamp(0.0, 1.0)
}

/// Order = 0.4*symmetry + 0.4*harmony_balance + 0.2*consciousness_coupling
fn compute_order(state: &AestheticState) -> f32 {
    let symmetry = compute_symmetry(state);
    let harmony_balance = compute_harmony_balance(state);
    let consciousness_coupling = state.luminosity; // higher Ψ → more ordered

    0.4 * symmetry + 0.4 * harmony_balance + 0.2 * consciousness_coupling
}

/// Symmetry: how evenly distributed the harmony radii are.
/// Perfect octagon = 1.0, single spike = low.
fn compute_symmetry(state: &AestheticState) -> f32 {
    let radii = &state.harmony_radii;
    let mean: f32 = radii.iter().sum::<f32>() / 8.0;
    if mean < 0.01 {
        return 0.0;
    }
    let variance: f32 = radii.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / 8.0;
    let cv = variance.sqrt() / mean; // coefficient of variation
    // Low CV → high symmetry. CV of 0 → 1.0, CV of 1 → ~0
    (1.0 - cv).clamp(0.0, 1.0)
}

/// Harmony balance: how many harmonies are active (above 0.3 threshold).
fn compute_harmony_balance(state: &AestheticState) -> f32 {
    let active = state.harmony_radii.iter().filter(|&&r| r > 0.3).count();
    (active as f32) / 8.0
}

/// Complexity from node count + visual diversity.
fn compute_complexity(state: &AestheticState, scene: &SceneNode) -> f32 {
    let node_count = scene.node_count() as f32;
    // Logarithmic: 1 node → ~0, 10 nodes → ~0.33, 100 → ~0.66, 1000 → ~1.0
    let structural = (node_count.ln().max(0.0)) / 7.0; // ln(1000) ≈ 6.9

    let topo = (state.component_count + state.ring_count + state.void_count) as f32;
    let topological = (topo / 10.0).min(1.0);

    // Measured from the actual scene colors (was a hardcoded 0.5 until
    // 2026-07-10, which made complexity blind to the palette in use).
    let color_diversity = crate::scene_features::extract_scene_features(scene).color_diversity();

    (0.5 * structural + 0.3 * topological + 0.2 * color_diversity).clamp(0.01, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Palette;

    fn make_state(harmony_radii: [f32; 8], luminosity: f32) -> AestheticState {
        AestheticState {
            luminosity,
            complexity: 0.5,
            turbulence: 0.1,
            energy: 0.5,
            warmth: 0.5,
            fractal_depth: 2,
            palette: Palette::default(),
            component_count: 3,
            ring_count: 1,
            void_count: 0,
            harmony_radii,
            layout_center: (256.0, 256.0),
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            cycle_phase: 0.5,
            aesthetic_score: 0.0,
        }
    }

    #[test]
    fn symmetric_scores_higher() {
        let symmetric = make_state([0.8; 8], 0.7);
        let asymmetric = make_state([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 0.7);
        // Need enough nodes AND real color variety for meaningful complexity
        // so scores don't both clamp to 1.0 — color_diversity is measured
        // from actual fills since 2026-07-10, and unstyled circles score 0.
        let mut scene = SceneNode::group(None);
        for i in 0..20 {
            scene
                .children
                .push(SceneNode::circle(i as f32 * 10.0, 0.0, 5.0).with_style(
                    crate::scene_graph::Style {
                        fill: Some(crate::color::Color::from_hsl(i as f32 * 18.0, 0.8, 0.5)),
                        ..Default::default()
                    },
                ));
        }
        let s1 = aesthetic_score(&symmetric, &scene);
        let s2 = aesthetic_score(&asymmetric, &scene);
        assert!(s1 > s2, "symmetric {s1} should > asymmetric {s2}");
    }

    #[test]
    fn score_in_range() {
        let state = make_state([0.5; 8], 0.5);
        let scene = SceneNode::group(None).with_child(SceneNode::circle(0.0, 0.0, 10.0));
        let s = aesthetic_score(&state, &scene);
        assert!(s >= 0.0 && s <= 1.0, "score {s} out of range");
    }

    #[test]
    fn empty_scene_zero() {
        let state = make_state([0.0; 8], 0.0);
        let scene = SceneNode::group(None);
        let s = aesthetic_score(&state, &scene);
        // With only 1 node and zero harmony, score should be low
        assert!(s < 0.5);
    }

    #[test]
    fn more_harmonies_more_beauty() {
        let few = make_state([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5);
        let many = make_state([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], 0.5);
        let scene = SceneNode::group(None)
            .with_child(SceneNode::circle(0.0, 0.0, 10.0))
            .with_child(SceneNode::circle(10.0, 10.0, 5.0))
            .with_child(SceneNode::circle(20.0, 20.0, 5.0));
        let s1 = aesthetic_score(&few, &scene);
        let s2 = aesthetic_score(&many, &scene);
        assert!(s2 > s1, "more harmonies {s2} should > fewer {s1}");
    }
}
