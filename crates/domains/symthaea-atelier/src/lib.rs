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
//! creates novel artwork through explore-then-mutate hill climbing:
//! independent generations first, then genuine mutation of the best scene
//! (see `iterate` for the exact semantics).
//!
//! # Architecture
//!
//! ```text
//! CognitiveSnapshot → AtelierStyle selection → Generate ×N → Evaluate → Mutate Best → Artwork
//! ```
//!
//! Nine generative subsystems:
//! - **L-Systems**: Harmony-driven fractal branching
//! - **Parametric Curves**: Thought-vector-driven Lissajous/rose curves
//! - **Persistence Textures**: Topology-driven Voronoi cellular patterns
//! - **Color Fields**: Neuromodulator gradient paintings
//! - **Composition**: Golden-ratio multi-layer compositing
//! - **Reaction-Diffusion**: Gray-Scott emergent chemical patterns
//! - **Strange Attractors**: Clifford/Lorenz/Rössler/Duffing orbits
//! - **Hofstadter Butterfly**: real Harper-model spectra (fractal-time-lab)
//! - **Persistence Diagrams**: real birth/death topology as diagram art

#![deny(unsafe_code)]

pub mod color_field;
pub mod composition;
pub mod critic;
pub mod curves;
pub mod harmony_shapes;
pub mod hofstadter_art;
pub mod integration;
pub mod iterate;
pub mod lsystem;
pub mod persistence;
pub mod persistence_diagram;
pub mod reaction_diffusion;
pub mod strange_attractors;
pub mod timeline;

// Showcase-tier modules: experimental art subsystems used by the
// `showcase`-gated examples. Before 2026-07-06 these files existed but were
// never declared, so the examples referencing them could not compile at all.
#[cfg(feature = "showcase")]
pub mod art_protocol;
#[cfg(feature = "showcase")]
pub mod creative_loop;
#[cfg(feature = "showcase")]
pub mod dream_art;
#[cfg(feature = "showcase")]
pub mod hybrid;
#[cfg(feature = "showcase")]
pub mod living_art;
#[cfg(feature = "showcase")]
pub mod neural_canvas;
#[cfg(feature = "showcase")]
pub mod pixel_canvas;
#[cfg(feature = "showcase")]
pub mod self_perception;
#[cfg(feature = "showcase")]
pub mod training;

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
    /// Hofstadter butterfly: real Harper-model spectral slices (flux on x,
    /// energy on y). Consciousness scales spectral resolution (bounded).
    HofstadterButterfly,
    /// Persistence diagram: the snapshot's real birth/death pairs above the
    /// diagonal, with a barcode band; degrades honestly when diagrams are
    /// empty (see `persistence_diagram` module docs).
    PersistenceDiagram,
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

// Crate-root re-exports of the externally-scored iterate API, matching how
// `create_artwork_iterative` is exposed (callers shouldn't need to know
// which submodule owns the loop).
pub use iterate::{EXTERNAL_SCORE_WEIGHT, ExternalScorer, create_iterative_scored};

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
        AtelierStyle::HofstadterButterfly => hofstadter_art::generate(config, snapshot, rng),
        AtelierStyle::PersistenceDiagram => persistence_diagram::generate(config, snapshot, rng),
    }
}

/// Score a scene graph aesthetically, combining artifact-measured features
/// with the snapshot's harmony activations.
///
/// Public so callers outside this crate (e.g. Symthaea's cultural-memory
/// imitation path, which builds a scene manually via [`generate`] +
/// [`iterate::mutate_scene`] instead of going through [`create_artwork`] /
/// [`create_artwork_iterative`]) can score a hand-assembled scene the same
/// way the top-level API does.
///
/// Until 2026-07-10 this function's only artwork-dependent signal was node
/// count (diversity was a hardcoded 0.5 and every order feature came from the
/// snapshot, which is constant across an iteration budget) — so hill-climbing
/// in [`iterate`] selected color/spatial mutations essentially at random. It
/// now measures the artwork itself: golden proportions over the real element
/// size hierarchy, spatial balance of the actual layout, and entropy of the
/// actual color/kind distributions. Snapshot-derived features (harmony
/// balance, consciousness coupling) remain deliberately — art reflecting the
/// generating cognitive state is a design goal — but they no longer crowd out
/// the artifact.
pub fn score_scene(scene: &SceneNode, snapshot: &CognitiveSnapshot) -> AestheticScore {
    let node_count = scene.node_count() as f32;
    let structural = (node_count.ln().max(0.0)) / 7.0;

    let topo = (snapshot.betti_0 + snapshot.betti_1 + snapshot.betti_2) as f32;
    let topological = (topo / 10.0).min(1.0);

    // Artifact-measured features (see symthaea_canvas::scene_features).
    let measured = symthaea_canvas::extract_scene_features(scene);
    let diversity = 0.5 * measured.color_diversity() + 0.5 * measured.kind_diversity();
    let golden = symthaea_aesthetic::golden::golden_proportions_score(&measured.element_sizes);
    let spatial_balance = measured.spatial_balance();

    let mut features = symthaea_aesthetic::birkhoff::extract_common_features(
        &snapshot.harmony_activations,
        snapshot.consciousness_level as f32,
        structural.clamp(0.0, 1.0),
        topological.clamp(0.0, 1.0),
        diversity,
    );

    // Compositional integration: Φ-style irreducibility of the artwork's
    // own element-relation graph (see `integration` module docs for what
    // this is and is not). This is the principled version of what
    // Birkhoff's "order" gestures at — unity that resists partition.
    let phi = integration::compositional_integration(scene).phi_topological;

    // Blend visual order measured from the artwork into the symmetry channel.
    // extract_common_features derives symmetry purely from harmony-activation
    // CV (cognitive intent); weight the layout, proportion, and integration
    // evidence of the artifact itself ahead of it.
    features.symmetry =
        (0.3 * spatial_balance + 0.3 * golden + 0.2 * phi + 0.2 * features.symmetry)
            .clamp(0.0, 1.0);

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
            AtelierStyle::HofstadterButterfly,
            AtelierStyle::PersistenceDiagram,
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

    /// Two scenes with identical geometry and node count but different color
    /// arrangements must score differently — this is the acceptance test for
    /// the 2026-07-10 artifact-inspecting scorer (before it, only node count
    /// mattered and these two scenes were indistinguishable).
    #[test]
    fn color_arrangement_changes_score() {
        use symthaea_canvas::color::Color;
        use symthaea_canvas::scene_graph::Style;

        let snapshot = CognitiveSnapshot::dormant();
        let build = |hues: &[f32]| {
            let mut scene = SceneNode::group(None);
            for (i, &hue) in hues.iter().enumerate() {
                scene.children.push(
                    SceneNode::circle(i as f32 * 40.0, (i % 2) as f32 * 40.0, 10.0).with_style(
                        Style {
                            fill: Some(Color::from_hsl(hue, 0.8, 0.5)),
                            ..Style::default()
                        },
                    ),
                );
            }
            scene
        };

        let mono = build(&[200.0; 6]);
        let varied = build(&[0.0, 60.0, 120.0, 180.0, 240.0, 300.0]);
        assert_eq!(mono.node_count(), varied.node_count());

        let s_mono = score_scene(&mono, &snapshot);
        let s_varied = score_scene(&varied, &snapshot);
        assert_ne!(
            s_mono.composite.to_bits(),
            s_varied.composite.to_bits(),
            "same geometry, different palette must not score identically"
        );
    }

    /// Compositional integration reaches the live score: a closed relational
    /// ring (one irreducible composition) must out-order the same eight
    /// circles — same hue, same size, same balanced quadrant spread —
    /// scattered as four independent corner pairs. Every other order channel
    /// (golden: identical sizes; harmony CV: same dormant snapshot; spatial
    /// balance: both centered and quadrant-balanced) is held ~equal, so the
    /// difference is the Φ channel.
    #[test]
    fn integrated_ring_out_orders_scattered_pairs() {
        use symthaea_canvas::color::Color;
        use symthaea_canvas::scene_graph::Style;
        let snapshot = CognitiveSnapshot::dormant();
        let styled_circle = |x: f32, y: f32| {
            SceneNode::circle(x, y, 20.0).with_style(Style {
                fill: Some(Color::from_hsl(200.0, 0.8, 0.5)),
                ..Style::default()
            })
        };

        let mut ring = SceneNode::group(None);
        for i in 0..8 {
            let theta = (i as f32 / 8.0) * std::f32::consts::TAU;
            ring.children
                .push(styled_circle(100.0 * theta.cos(), 100.0 * theta.sin()));
        }

        let mut pairs = SceneNode::group(None);
        for (cx, cy) in [
            (-100.0f32, -100.0f32),
            (100.0, -100.0),
            (-100.0, 100.0),
            (100.0, 100.0),
        ] {
            pairs.children.push(styled_circle(cx - 10.0, cy));
            pairs.children.push(styled_circle(cx + 10.0, cy));
        }

        assert_eq!(ring.node_count(), pairs.node_count());
        let s_ring = score_scene(&ring, &snapshot);
        let s_pairs = score_scene(&pairs, &snapshot);
        assert!(
            s_ring.order > s_pairs.order,
            "integrated ring order {} should beat scattered pairs {}",
            s_ring.order,
            s_pairs.order
        );
    }

    /// Golden-ratio size hierarchies score higher than uniform sizes, all
    /// else equal — proof the hill-climb now has a proportion gradient.
    #[test]
    fn golden_hierarchy_beats_uniform_sizes() {
        let snapshot = CognitiveSnapshot::dormant();
        let build = |radii: &[f32]| {
            let mut scene = SceneNode::group(None);
            for (i, &r) in radii.iter().enumerate() {
                scene
                    .children
                    .push(SceneNode::circle(i as f32 * 150.0, 0.0, r));
            }
            scene
        };

        // Consecutive ratios of φ vs. all-equal sizes.
        let golden = build(&[100.0, 61.8, 38.2, 23.6]);
        let uniform = build(&[55.9, 55.9, 55.9, 55.9]);
        assert_eq!(golden.node_count(), uniform.node_count());

        let s_golden = score_scene(&golden, &snapshot);
        let s_uniform = score_scene(&uniform, &snapshot);
        assert!(
            s_golden.order > s_uniform.order,
            "golden hierarchy order {} should beat uniform {}",
            s_golden.order,
            s_uniform.order
        );
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
