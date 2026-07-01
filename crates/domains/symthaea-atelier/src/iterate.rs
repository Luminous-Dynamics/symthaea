// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Iterative generate-evaluate-mutate creative loop.
//!
//! Generates multiple artwork variants, evaluates each aesthetically,
//! and keeps the best. This is a microcosm of the Wallas cycle:
//! divergent mutation → convergent selection.

use rand::SeedableRng;
use rand::rngs::StdRng;
use symthaea_canvas::CognitiveSnapshot;

use crate::{Artwork, AtelierConfig, generate, score_scene};

/// Create artwork with iterative refinement.
///
/// Generates `config.iteration_budget` variants with different random seeds,
/// keeps the one with the highest aesthetic composite score.
pub fn create_iterative(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    base_seed: u64,
) -> Artwork {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_scene = None;
    let mut best_aesthetic = symthaea_aesthetic::AestheticScore::zero();
    let mut cycles = 0;

    for i in 0..config.iteration_budget.max(1) {
        let seed = base_seed.wrapping_add(i as u64 * 7919); // prime step
        let mut rng = StdRng::seed_from_u64(seed);
        let scene = generate(config, snapshot, &mut rng);
        let score = score_scene(&scene, snapshot);

        cycles += 1;

        if score.composite > best_score {
            best_score = score.composite;
            best_scene = Some(scene);
            best_aesthetic = score;
        }
    }

    let scene = best_scene.unwrap_or_else(|| {
        let mut rng = StdRng::seed_from_u64(base_seed);
        generate(config, snapshot, &mut rng)
    });

    let svg = symthaea_canvas::render_svg(&scene, snapshot.consciousness_level);

    Artwork {
        scene,
        svg,
        aesthetic_score: best_aesthetic,
        style: config.style,
        generation_cycles: cycles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterative_improves_or_maintains() {
        let config = AtelierConfig {
            iteration_budget: 5,
            ..AtelierConfig::default()
        };
        let snapshot = CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            dopamine: 0.6,
            serotonin: 0.5,
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        };

        let artwork = create_iterative(&config, &snapshot, 42);
        assert!(artwork.generation_cycles == 5);
        assert!(artwork.aesthetic_score.composite >= 0.0);
    }

    #[test]
    fn zero_budget_still_produces() {
        let config = AtelierConfig {
            iteration_budget: 0,
            ..AtelierConfig::default()
        };
        let snapshot = CognitiveSnapshot::dormant();
        let artwork = create_iterative(&config, &snapshot, 42);
        assert!(artwork.svg.contains("<svg"));
        assert!(artwork.generation_cycles >= 1);
    }
}
