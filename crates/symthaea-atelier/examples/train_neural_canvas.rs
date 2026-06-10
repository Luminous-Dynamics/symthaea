// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train the NeuralCanvas projections via evolution strategy.
//!
//! Run: cargo run -p symthaea-atelier --example train_neural_canvas

use symthaea_atelier::AtelierConfig;
use symthaea_atelier::neural_canvas::NeuralCanvas;
use symthaea_atelier::training::{TrainingConfig, train_projections};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_core::genesis::GenesisSeed;

fn main() {
    println!("╔═══════════════════════════════════════════════╗");
    println!("║  Training NeuralCanvas — Learning to See Beauty ║");
    println!("╚═══════════════════════════════════════════════╝\n");

    let genesis = GenesisSeed::from_phrase("training-session");
    let mut canvas = NeuralCanvas::new(&genesis);

    let atelier_config = AtelierConfig {
        max_elements: 100,
        ..AtelierConfig::default()
    };

    let training_snapshots = vec![
        CognitiveSnapshot {
            consciousness_level: 0.85,
            valence: 0.6,
            arousal: 0.3,
            dopamine: 0.5,
            serotonin: 0.8,
            noradrenaline: 0.2,
            harmony_activations: [0.7, 0.8, 0.6, 0.3, 0.7, 0.6, 0.5, 0.8],
            prediction_error: 0.05,
            thought_vector: vec![0.3, -0.1],
            ..CognitiveSnapshot::dormant()
        },
        CognitiveSnapshot {
            consciousness_level: 0.6,
            valence: -0.6,
            arousal: 0.9,
            dopamine: 0.3,
            serotonin: 0.2,
            noradrenaline: 0.9,
            harmony_activations: [0.3, 0.2, 0.4, 0.9, 0.5, 0.2, 0.8, 0.1],
            prediction_error: 0.7,
            thought_vector: vec![0.8, -0.7],
            ..CognitiveSnapshot::dormant()
        },
        CognitiveSnapshot {
            consciousness_level: 0.75,
            valence: 0.4,
            arousal: 0.7,
            dopamine: 0.8,
            serotonin: 0.5,
            noradrenaline: 0.5,
            harmony_activations: [0.4, 0.5, 0.3, 0.95, 0.6, 0.4, 0.7, 0.1],
            prediction_error: 0.3,
            thought_vector: vec![0.5, 0.3],
            ..CognitiveSnapshot::dormant()
        },
        CognitiveSnapshot {
            consciousness_level: 0.9,
            valence: 0.2,
            arousal: 0.15,
            dopamine: 0.4,
            serotonin: 0.7,
            noradrenaline: 0.1,
            harmony_activations: [0.8, 0.6, 0.9, 0.2, 0.5, 0.7, 0.4, 0.95],
            prediction_error: 0.02,
            thought_vector: vec![0.1, 0.0],
            ..CognitiveSnapshot::dormant()
        },
    ];

    let config = TrainingConfig {
        population_size: 6,
        noise_sigma: 0.03,
        learning_rate: 0.008,
        epochs: 15,
    };

    println!(
        "  Training on {} cognitive states, {} epochs, population {}\n",
        training_snapshots.len(),
        config.epochs,
        config.population_size
    );

    let result = train_projections(&mut canvas, &config, &atelier_config, &training_snapshots);

    println!("  ━━━ Training Results ━━━\n");
    println!("  Initial score:     {:.4}", result.initial_score);
    println!("  Final score:       {:.4}", result.final_score);
    let improvement =
        (result.final_score - result.initial_score) / result.initial_score.max(0.001) * 100.0;
    println!("  Improvement:       {:+.1}%", improvement);
    println!("  Total evaluations: {}", result.total_evaluations);
    println!();

    println!("  Score trajectory:");
    for (i, &score) in result.score_history.iter().enumerate() {
        let bar_len = (score * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    epoch {:>2}: {:.4} {}", i, score, bar);
    }
}
