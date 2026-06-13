// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use symthaea_cosmogenesis::{CognitiveCosmologyParams, CosmogenesisSimulator, SemanticParticle};

fn main() {
    let dim = 16384;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create base classes with high-dimensional jitter (noise)
    let mut particles = vec![
        SemanticParticle {
            id: "class0_a".into(),
            class_id: 0,
            position: vec![0.1; dim],
            velocity: vec![0.0; dim],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "class0_b".into(),
            class_id: 0,
            position: vec![0.2; dim],
            velocity: vec![0.0; dim],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "class1_a".into(),
            class_id: 1,
            position: vec![0.8; dim],
            velocity: vec![0.0; dim],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "class1_b".into(),
            class_id: 1,
            position: vec![0.9; dim],
            velocity: vec![0.0; dim],
            mass: 1.0,
            latent_mass: 1.0,
        },
    ];

    // Apply "Semantic Noise" (jittering positions by 0.2 across all dimensions)
    for p in &mut particles {
        for pos in p.position.iter_mut() {
            *pos += rng.gen_range(-0.2..0.2);
        }
    }

    let params = CognitiveCosmologyParams {
        attraction_strength: 0.5,
        lambda: 0.5,
        steps: 20,
        ..Default::default()
    };

    println!("NoiseLevel,InitialSeparation,FinalSeparation,DeltaSep,Finite");

    let mut sim = CosmogenesisSimulator::new(params, particles);
    let initial_metrics = sim.calculate_metrics();
    let final_metrics = sim.run_simulation();

    let delta = final_metrics.separation_proxy - initial_metrics.separation_proxy;
    let finite = final_metrics.separation_proxy.is_finite();

    println!(
        "0.2,{:.4},{:.4},{:.4},{}",
        initial_metrics.separation_proxy, final_metrics.separation_proxy, delta, finite
    );
}
