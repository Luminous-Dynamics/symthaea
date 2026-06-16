// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-atelier
//!
//! Generative visual art engine for Symthaea. Produces SVG compositions
//! driven by consciousness state, Eight Harmony activations, neuromodulators,
//! and topological structure.
//!
//! Unlike `symthaea-canvas` (diagnostic mandala visualization), the atelier
//! creates genuinely novel artwork through iterative generate-evaluate-mutate cycles.
//!
//! # Architecture
//!
//! ```text
//! CognitiveSnapshot → AtelierStyle selection → Generate → Evaluate → Mutate → Best Artwork
//! ```
//!
//! Five generative subsystems:
//! - **L-Systems**: Harmony-driven fractal branching
//! - **Parametric Curves**: Thought-vector-driven Lissajous/rose curves
//! - **Persistence Textures**: Topology-driven Voronoi cellular patterns
//! - **Color Fields**: Neuromodulator gradient paintings
//! - **Composition**: Golden-ratio multi-layer compositing

#![deny(unsafe_code)]

pub mod color_field;
pub mod composition;
pub mod critic;
pub mod curves;
pub mod harmony_shapes;
pub mod iterate;
pub mod lsystem;
pub mod persistence;
pub mod reaction_diffusion;
pub mod strange_attractors;
pub mod timeline;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use symthaea_aesthetic::AestheticScore;
use symthaea_canvas::{CognitiveSnapshot, SceneNode};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the atelier art generation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtelierConfig {
    /// Viewport width in SVG units.
    pub width: f32,
    /// Viewport height in SVG units.
    pub height: f32,
    /// Maximum scene graph elements (performance cap).
    pub max_elements: usize,
    /// Which generative subsystem(s) to use.
    pub style: AtelierStyle,
    /// Maximum generate-evaluate iterations per artwork.
    pub iteration_budget: usize,
}

impl Default for AtelierConfig {
    fn default() -> Self {
        Self {
            width: 1024.0,
            height: 1024.0,
            max_elements: 500,
            style: AtelierStyle::Composite,
            iteration_budget: 8,
        }
    }
}

/// Which generative subsystem to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtelierStyle {
    /// Harmony-driven L-system fractals.
    LSystem,
    /// Thought-vector parametric curves.
    ParametricCurve,
    /// Topology-driven Voronoi textures.
    PersistenceTexture,
    /// Neuromodulator color field paintings.
    ColorField,
    /// All subsystems, layered via golden-ratio composition.
    Composite,
    /// Gray-Scott reaction-diffusion: living, emergent chemical patterns.
    /// Harmony activations map to (F, k) parameter space; consciousness controls steps.
    ReactionDiffusion,
    /// Strange attractor trajectory: Clifford, Lorenz, Rössler, or Duffing.
    /// Dominant harmony selects the attractor family; Phi gates orbit depth.
    StrangeAttractor,
}

// ─── Artwork ─────────────────────────────────────────────────────────────────

/// A generated artwork with its scene graph and metadata.
#[derive(Debug, Clone)]
pub struct Artwork {
    /// The scene graph (renderable via `symthaea_canvas::render_svg`).
    pub scene: SceneNode,
    /// SVG string (pre-rendered for convenience).
    pub svg: String,
    /// Aesthetic quality score.
    pub aesthetic_score: AestheticScore,
    /// Style used to generate this artwork.
    pub style: AtelierStyle,
    /// Number of generate-evaluate cycles used.
    pub generation_cycles: usize,
}

// ─── Top-Level API ───────────────────────────────────────────────────────────

/// Generate a single artwork from a cognitive snapshot.
///
/// This is the main entry point. It selects the generative subsystem,
/// runs the iterative creative loop, and returns the best artwork.
pub fn create_artwork(config: &AtelierConfig, snapshot: &CognitiveSnapshot, seed: u64) -> Artwork {
    let mut rng = StdRng::seed_from_u64(seed);
    let scene = generate(config, snapshot, &mut rng);
    let score = score_scene(&scene, snapshot);
    let svg = symthaea_canvas::render_svg(&scene, snapshot.consciousness_level);

    Artwork {
        scene,
        svg,
        aesthetic_score: score,
        style: config.style,
        generation_cycles: 1,
    }
}

/// Generate artwork with iterative refinement.
pub fn create_artwork_iterative(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    seed: u64,
) -> Artwork {
    iterate::create_iterative(config, snapshot, seed)
}

/// Generate a scene graph from the selected style.
pub fn generate(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    rng: &mut StdRng,
) -> SceneNode {
    match config.style {
        AtelierStyle::LSystem => lsystem::generate(config, snapshot, rng),
        AtelierStyle::ParametricCurve => curves::generate(config, snapshot, rng),
        AtelierStyle::PersistenceTexture => persistence::generate(config, snapshot, rng),
        AtelierStyle::ColorField => color_field::generate(config, snapshot, rng),
        AtelierStyle::Composite => composition::generate(config, snapshot, rng),
        AtelierStyle::ReactionDiffusion => reaction_diffusion::generate(config, snapshot, rng),
        AtelierStyle::StrangeAttractor => strange_attractors::generate(config, snapshot, rng),
    }
}

/// Score a scene graph aesthetically using the snapshot's harmony activations.
fn score_scene(scene: &SceneNode, snapshot: &CognitiveSnapshot) -> AestheticScore {
    let node_count = scene.node_count() as f32;
    let structural = (node_count.ln().max(0.0)) / 7.0;

    let topo = (snapshot.betti_0 + snapshot.betti_1 + snapshot.betti_2) as f32;
    let topological = (topo / 10.0).min(1.0);

    let features = symthaea_aesthetic::birkhoff::extract_common_features(
        &snapshot.harmony_activations,
        snapshot.consciousness_level as f32,
        structural.clamp(0.0, 1.0),
        topological.clamp(0.0, 1.0),
        0.5, // diversity (approximate)
    );

    features.to_score()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_artwork_produces_svg() {
        let config = AtelierConfig {
            style: AtelierStyle::ParametricCurve,
            iteration_budget: 1,
            ..Default::default()
        };
        let snapshot = CognitiveSnapshot::dormant();
        let artwork = create_artwork(&config, &snapshot, 42);
        assert!(artwork.svg.contains("<svg"));
        assert!(artwork.svg.contains("</svg>"));
        assert!(artwork.aesthetic_score.composite >= 0.0);
    }

    #[test]
    fn all_styles_generate() {
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
            dopamine: 0.6,
            noradrenaline: 0.4,
            serotonin: 0.5,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, -0.1],
            ..CognitiveSnapshot::dormant()
        };

        for style in [
            AtelierStyle::LSystem,
            AtelierStyle::ParametricCurve,
            AtelierStyle::PersistenceTexture,
            AtelierStyle::ColorField,
            AtelierStyle::Composite,
        ] {
            let config = AtelierConfig {
                style,
                iteration_budget: 1,
                ..Default::default()
            };
            let artwork = create_artwork(&config, &snapshot, 42);
            assert!(
                artwork.svg.contains("<svg"),
                "{style:?} failed to produce SVG"
            );
            assert!(
                artwork.scene.node_count() > 1,
                "{style:?} produced empty scene"
            );
        }
    }

    #[test]
    fn iterative_produces_artwork() {
        let config = AtelierConfig {
            style: AtelierStyle::ParametricCurve,
            iteration_budget: 3,
            ..Default::default()
        };
        let snapshot = CognitiveSnapshot::dormant();
        let artwork = create_artwork_iterative(&config, &snapshot, 42);
        assert!(artwork.svg.contains("<svg"));
        assert!(artwork.generation_cycles >= 1);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = AtelierConfig::default();
        let snapshot = CognitiveSnapshot::dormant();
        let a1 = create_artwork(&config, &snapshot, 42);
        let a2 = create_artwork(&config, &snapshot, 42);
        assert_eq!(a1.svg, a2.svg);
    }

    #[test]
    fn different_seeds_different_art() {
        let config = AtelierConfig {
            style: AtelierStyle::ParametricCurve,
            ..Default::default()
        };
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };
        let a1 = create_artwork(&config, &snapshot, 42);
        let a2 = create_artwork(&config, &snapshot, 123);
        assert_ne!(a1.svg, a2.svg);
    }
}
