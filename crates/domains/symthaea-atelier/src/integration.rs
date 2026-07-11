// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compositional integration: a Φ-style measure of how irreducible an
//! artwork's composition is.
//!
//! Builds a similarity graph over the artwork's elements (spatial proximity,
//! color affinity, size kinship), then runs the existing Hodge pipeline
//! (`symthaea-hodge::consciousness_topology`) over it: the resulting
//! `TopologicalPhi` measures how much of the composition's relational
//! structure is topologically irreducible — a scattered collage partitions
//! into independent sub-compositions (low Φ), a tightly interlocked
//! composition with echoed palette/proportion/placement resists partition
//! (high Φ). Art theory calls the same property *unity*; Birkhoff's "order"
//! term gestured at it without the machinery to formalize it.
//!
//! **Honesty note — what this is and is not.** This is integrated-information
//! *mathematics applied to a static relational structure*, not consciousness
//! of the artwork: IIT's Φ proper is defined over a causal dynamical system,
//! and an SVG has no dynamics. The name "compositional integration" is chosen
//! deliberately. (The observer-side measure — how perceiving an artwork moves
//! *Symthaea's own* live Φ — is the separate `art-observer` path.)
//!
//! References: Hodge (1941), Petri et al. (2014) — topological strata of
//! weighted networks; Tononi (2004) for the integration framing; Arnheim
//! (1954) *Art and Visual Perception* for compositional unity.

use symthaea_canvas::SceneNode;
use symthaea_canvas::scene_features::{ElementFeature, extract_element_features};
use symthaea_hodge::consciousness_topology::{
    ConsciousnessComplex, HodgeConsciousnessDecomposition, TopologicalPhi,
};

/// Maximum elements analyzed — keeps the O(n³)-ish Hodge pipeline bounded.
/// When a scene has more, the largest elements are kept (they dominate the
/// composition perceptually).
pub const MAX_INTEGRATION_ELEMENTS: usize = 32;

/// Similarity threshold for including an edge in the element complex.
/// Below ~0.5 nearly everything connects (Φ saturates); above ~0.7 nearly
/// nothing does (Φ collapses).
const SIMILARITY_THRESHOLD: f64 = 0.6;

/// Weights of the three relation channels in element similarity. Spatial
/// proximity carries half the weight ON PURPOSE: color+size max out at 0.5,
/// below the edge threshold, so an identical-palette identical-size scene
/// does NOT degenerate into a complete graph — two elements only relate if
/// they are also near each other (within ~1.35σ ≈ 0.4× the content
/// diagonal). Layout always matters.
const W_SPATIAL: f32 = 0.5;
const W_COLOR: f32 = 0.25;
const W_SIZE: f32 = 0.25;

/// Result of the compositional-integration analysis.
#[derive(Debug, Clone, Default)]
pub struct CompositionalIntegration {
    /// Combined topological integration in [0, 1] — the headline number.
    pub phi_topological: f32,
    /// Fraction of relational energy in topologically-protected modes.
    pub phi_harmonic: f32,
    /// Integration from Betti structure (independent loops + voids).
    pub phi_betti: f32,
    /// Independent relational loops (β₁ of the element complex).
    pub integration_loops: usize,
    /// Elements actually analyzed (post-subsampling).
    pub element_count: usize,
}

/// Measure the compositional integration of a scene.
///
/// Deterministic pure function. Scenes with fewer than 3 drawable elements
/// return the zero default — integration of a composition needs parts to
/// integrate.
pub fn compositional_integration(scene: &SceneNode) -> CompositionalIntegration {
    let mut elements = extract_element_features(scene);
    if elements.len() < 3 {
        return CompositionalIntegration::default();
    }

    // Keep the perceptually dominant elements when over budget.
    if elements.len() > MAX_INTEGRATION_ELEMENTS {
        elements.sort_by(|a, b| {
            b.size
                .partial_cmp(&a.size)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        elements.truncate(MAX_INTEGRATION_ELEMENTS);
    }
    let n = elements.len();

    // Scale for spatial similarity: the content bbox diagonal.
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for e in &elements {
        min_x = min_x.min(e.center.0);
        min_y = min_y.min(e.center.1);
        max_x = max_x.max(e.center.0);
        max_y = max_y.max(e.center.1);
    }
    let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2))
        .sqrt()
        .max(f32::EPSILON);

    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        1.0
                    } else {
                        element_similarity(&elements[i], &elements[j], diag) as f64
                    }
                })
                .collect()
        })
        .collect();

    let complex = ConsciousnessComplex::from_correlation_matrix(&matrix, SIMILARITY_THRESHOLD);
    if complex.edges.is_empty() {
        // Fully scattered: no relations above threshold → zero integration.
        return CompositionalIntegration {
            element_count: n,
            ..CompositionalIntegration::default()
        };
    }

    // Edge signal: salience difference across each relation (the gradient
    // proxy the Hodge decomposition splits into feedforward / recurrent /
    // harmonic parts) — mirrors `topological_measure`'s use of state
    // differences. Salience = luma blended with normalized size; unstyled
    // elements fall back to size alone.
    let max_size = elements.iter().map(|e| e.size).fold(f32::EPSILON, f32::max);
    let salience: Vec<f64> = elements
        .iter()
        .map(|e| {
            let size_term = e.size / max_size;
            match e.color {
                Some((_h, _s, l)) => (0.5 * l + 0.5 * size_term) as f64,
                None => size_term as f64,
            }
        })
        .collect();
    let edge_signal: Vec<f64> = complex
        .edges
        .iter()
        .map(|&(i, j, _)| salience[i] - salience[j])
        .collect();

    let decomposition = HodgeConsciousnessDecomposition::decompose(&complex, &edge_signal);
    let phi = TopologicalPhi::from_decomposition(&decomposition);

    // Signal-degenerate compositions (perfectly uniform elements — identical
    // salience everywhere, e.g. a mandala of same-hue same-size circles)
    // have a ZERO edge signal: there is no relational flow to decompose, the
    // harmonic ratio is trivially zero, and upstream's geometric-mean
    // phi_topological collapses even when the complex has real loops. Found
    // by this module's own ring test (Φ=0 on a closed 8-ring with β₁ ≥ 1).
    // Integration is then purely structural: fall back to phi_betti, which
    // is signal-independent.
    const SIGNAL_ENERGY_EPSILON: f64 = 1e-9;
    let signal_energy: f64 = edge_signal.iter().map(|s| s * s).sum();
    let phi_topological = if signal_energy > SIGNAL_ENERGY_EPSILON {
        phi.phi_topological
    } else {
        phi.phi_betti
    };

    CompositionalIntegration {
        phi_topological: phi_topological.clamp(0.0, 1.0) as f32,
        phi_harmonic: phi.phi_harmonic.clamp(0.0, 1.0) as f32,
        phi_betti: phi.phi_betti.clamp(0.0, 1.0) as f32,
        integration_loops: phi.integration_loops,
        element_count: n,
    }
}

/// Pairwise element similarity in [0, 1]: spatial proximity (Gaussian on
/// center distance, σ = 0.3 × content diagonal), color affinity (hue
/// closeness for chromatic pairs, lightness closeness for achromatic pairs,
/// weak neutral for mixed/unstyled), and size kinship (min/max ratio).
fn element_similarity(a: &ElementFeature, b: &ElementFeature, diag: f32) -> f32 {
    let sigma = 0.3 * diag;
    let dist2 = (a.center.0 - b.center.0).powi(2) + (a.center.1 - b.center.1).powi(2);
    let spatial = (-dist2 / (2.0 * sigma * sigma)).exp();

    let color = match (a.color, b.color) {
        (Some((ha, sa, la)), Some((hb, sb, lb))) => {
            let a_chromatic = sa >= 0.1;
            let b_chromatic = sb >= 0.1;
            if a_chromatic && b_chromatic {
                let dh = (ha - hb).rem_euclid(360.0);
                let dh = dh.min(360.0 - dh); // shortest arc, [0, 180]
                1.0 - dh / 180.0
            } else if !a_chromatic && !b_chromatic {
                1.0 - (la - lb).abs()
            } else {
                0.3
            }
        }
        (None, None) => 0.5,
        _ => 0.3,
    };

    let size = if a.size > 0.0 && b.size > 0.0 {
        (a.size.min(b.size) / a.size.max(b.size)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (W_SPATIAL * spatial + W_COLOR * color + W_SIZE * size).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_canvas::color::Color;
    use symthaea_canvas::scene_graph::Style;

    fn circle(x: f32, y: f32, r: f32, hue: f32) -> SceneNode {
        SceneNode::circle(x, y, r).with_style(Style {
            fill: Some(Color::from_hsl(hue, 0.8, 0.5)),
            ..Style::default()
        })
    }

    /// A closed ring of same-hue, same-size, evenly spaced circles — echoed
    /// palette + proportion + proximity chain closing on itself.
    fn interlocked_ring(n: usize, radius: f32) -> SceneNode {
        let mut scene = SceneNode::group(None);
        for i in 0..n {
            let theta = (i as f32 / n as f32) * std::f32::consts::TAU;
            scene.children.push(circle(
                radius * theta.cos(),
                radius * theta.sin(),
                20.0,
                200.0,
            ));
        }
        scene
    }

    /// Distant clusters with clashing hues and disparate sizes — a
    /// composition that partitions into independent pieces.
    fn scattered_clusters() -> SceneNode {
        let mut scene = SceneNode::group(None);
        for (cx, cy, hue, r) in [
            (0.0f32, 0.0f32, 0.0f32, 40.0f32),
            (30.0, 20.0, 20.0, 35.0),
            (5000.0, 5000.0, 130.0, 6.0),
            (5030.0, 5010.0, 150.0, 5.0),
            (-5000.0, 5000.0, 270.0, 90.0),
            (-5040.0, 5020.0, 290.0, 3.0),
        ] {
            scene.children.push(circle(cx, cy, r, hue));
        }
        scene
    }

    #[test]
    fn too_few_elements_zero() {
        let scene = SceneNode::group(None)
            .with_child(circle(0.0, 0.0, 10.0, 100.0))
            .with_child(circle(50.0, 0.0, 10.0, 100.0));
        let result = compositional_integration(&scene);
        assert_eq!(result.phi_topological, 0.0);
        assert_eq!(result.element_count, 0);
    }

    #[test]
    fn interlocked_beats_scattered() {
        let ring = compositional_integration(&interlocked_ring(8, 100.0));
        let scattered = compositional_integration(&scattered_clusters());
        assert!(
            ring.phi_topological > scattered.phi_topological,
            "interlocked ring Φ {} should beat scattered clusters Φ {}",
            ring.phi_topological,
            scattered.phi_topological
        );
    }

    #[test]
    fn ring_has_integration_loops() {
        // A closed relational ring is exactly what β₁ counts.
        let ring = compositional_integration(&interlocked_ring(8, 100.0));
        assert!(
            ring.integration_loops >= 1,
            "closed ring should yield at least one loop, got {}",
            ring.integration_loops
        );
    }

    #[test]
    fn bounded_and_deterministic() {
        let scene = interlocked_ring(6, 80.0);
        let a = compositional_integration(&scene);
        let b = compositional_integration(&scene);
        for v in [a.phi_topological, a.phi_harmonic, a.phi_betti] {
            assert!((0.0..=1.0).contains(&v), "value {v} out of bounds");
        }
        assert_eq!(a.phi_topological.to_bits(), b.phi_topological.to_bits());
    }

    #[test]
    fn subsampling_keeps_it_bounded() {
        // 200 elements — must subsample to MAX_INTEGRATION_ELEMENTS, not
        // blow up.
        let mut scene = SceneNode::group(None);
        for i in 0..200 {
            let theta = (i as f32 / 200.0) * std::f32::consts::TAU;
            scene
                .children
                .push(circle(300.0 * theta.cos(), 300.0 * theta.sin(), 8.0, 210.0));
        }
        let result = compositional_integration(&scene);
        assert_eq!(result.element_count, MAX_INTEGRATION_ELEMENTS);
        assert!((0.0..=1.0).contains(&result.phi_topological));
    }

    #[test]
    fn real_artwork_measures() {
        // The pipeline must handle every real generator's output shape.
        use symthaea_canvas::CognitiveSnapshot;
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };
        let config = crate::AtelierConfig {
            style: crate::AtelierStyle::Composite,
            iteration_budget: 1,
            ..Default::default()
        };
        let artwork = crate::create_artwork(&config, &snapshot, 42);
        let result = compositional_integration(&artwork.scene);
        assert!((0.0..=1.0).contains(&result.phi_topological));
    }
}
