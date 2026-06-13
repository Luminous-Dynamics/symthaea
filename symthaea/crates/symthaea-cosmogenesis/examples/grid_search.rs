// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_cosmogenesis::{CognitiveCosmologyParams, CosmogenesisSimulator, SemanticParticle};

fn main() {
    let particles = vec![
        SemanticParticle {
            id: "1".into(),
            class_id: 0,
            position: vec![0.0, 0.0],
            velocity: vec![0.0, 0.0],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "2".into(),
            class_id: 0,
            position: vec![0.1, 0.1],
            velocity: vec![0.0, 0.0],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "3".into(),
            class_id: 1,
            position: vec![1.0, 1.0],
            velocity: vec![0.0, 0.0],
            mass: 1.0,
            latent_mass: 1.0,
        },
        SemanticParticle {
            id: "4".into(),
            class_id: 1,
            position: vec![1.1, 1.1],
            velocity: vec![0.0, 0.0],
            mass: 1.0,
            latent_mass: 1.0,
        },
    ];

    println!("Attraction,Lambda,InitSeparation,FinalSeparation,DeltaSep,Finite");

    let attraction_values = [0.5, 1.0, 2.0, 4.0];
    let lambda_values = [0.0, 0.01, 0.1, 0.5];

    for &attraction in &attraction_values {
        for &lambda in &lambda_values {
            let params = CognitiveCosmologyParams {
                attraction_strength: attraction,
                lambda,
                ..Default::default()
            };

            let mut sim = CosmogenesisSimulator::new(params, particles.clone());
            let initial_metrics = sim.calculate_metrics();
            let final_metrics = sim.run_simulation();

            let delta = final_metrics.separation_proxy - initial_metrics.separation_proxy;
            let finite = final_metrics.separation_proxy.is_finite();

            println!(
                "{},{},{:.4},{:.4},{:.4},{}",
                attraction,
                lambda,
                initial_metrics.separation_proxy,
                final_metrics.separation_proxy,
                delta,
                finite
            );
        }
    }
}
