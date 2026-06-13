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

    println!("Attraction,Lambda,SeparationProxy,IntraDist,InterDist");

    let attraction_values = [0.1, 0.5, 1.0, 2.0, 5.0];
    let lambda_values = [0.0, 0.001, 0.01, 0.1, 0.5];

    let mut best_sep = -1.0;
    let mut best_params = None;

    for &attraction in &attraction_values {
        for &lambda in &lambda_values {
            let params = CognitiveCosmologyParams {
                attraction_strength: attraction,
                lambda,
                steps: 50,
                ..Default::default()
            };

            let mut sim = CosmogenesisSimulator::new(params.clone(), particles.clone());
            let metrics = sim.run_simulation();

            println!(
                "{},{},{:.4},{:.4},{:.4}",
                attraction,
                lambda,
                metrics.separation_proxy,
                metrics.mean_intra_class_distance,
                metrics.mean_inter_class_distance
            );

            if metrics.separation_proxy > best_sep {
                best_sep = metrics.separation_proxy;
                best_params = Some(params);
            }
        }
    }

    if let Some(p) = best_params {
        eprintln!("Best Improvement: {}", best_sep);
        eprintln!("Best Params: {:?}", p);
    }
}
