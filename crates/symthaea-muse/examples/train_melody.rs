// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train melody projections on MAESTRO MIDI data.
//!
//! ```bash
//! cargo run -p symthaea-muse --example train_melody --release
//! ```

use std::path::Path;
use symthaea_muse::MuseConfig;
use symthaea_muse::training::{TrainingConfig, save_projections, train_projections};

fn main() {
    let maestro_dir = Path::new("data/midi-training/maestro/maestro-v3.0.0");

    if !maestro_dir.exists() {
        eprintln!("MAESTRO dataset not found at {:?}", maestro_dir);
        eprintln!(
            "Download: wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        );
        return;
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  Melody Projection Training (MAESTRO)");
    println!("═══════════════════════════════════════════════════════\n");

    let config = TrainingConfig {
        population_size: 12,
        noise_sigma: 0.03,
        learning_rate: 0.008,
        epochs: 300,
        max_melodies: 500, // More diverse phrases
        seed: 42,
    };

    let muse_config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        ..Default::default()
    };

    println!(
        "  Config: {} epochs, pop={}, σ={}, lr={}",
        config.epochs, config.population_size, config.noise_sigma, config.learning_rate
    );
    println!("  Dataset: MAESTRO (max {} melodies)", config.max_melodies);
    println!("  Loading melodies...\n");

    let result = train_projections(maestro_dir, &config, &muse_config);

    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Training Results");
    println!("═══════════════════════════════════════════════════════\n");
    println!("  Melodies used: {}", result.melodies_used);
    println!("  Epochs: {}", result.epochs_completed);
    println!("  Best fitness: {:.4}", result.best_fitness);

    if !result.fitness_trajectory.is_empty() {
        let first = result.fitness_trajectory[0];
        let last = *result.fitness_trajectory.last().unwrap();
        println!("  First epoch: {:.4}", first);
        println!("  Last epoch:  {:.4}", last);
        println!("  Improvement: {:+.4}", last - first);

        // Mini trajectory
        println!("\n  Fitness trajectory:");
        let step = (result.fitness_trajectory.len() / 10).max(1);
        for (i, &f) in result.fitness_trajectory.iter().enumerate() {
            if i % step == 0 || i == result.fitness_trajectory.len() - 1 {
                let bar_len = (f * 40.0) as usize;
                let bar: String = "█".repeat(bar_len);
                println!("    Epoch {:3}: {:.4} |{bar}", i + 1, f);
            }
        }
    }

    // Save projections
    let proj_path = Path::new("data/midi-training/melody_projections.json");
    if let Err(e) = save_projections(&result.projections, proj_path) {
        eprintln!("  Failed to save projections: {e}");
    } else {
        println!("\n  Projections saved to: {}", proj_path.display());
    }

    // Generate before/after comparison
    println!("\n\n═══════════════════════════════════════════════════════");
    println!("  Before/After Comparison");
    println!("═══════════════════════════════════════════════════════\n");

    let state = symthaea_muse::MusicalState::default();

    // Before: default config (random projections)
    let before = symthaea_muse::compose(
        &MuseConfig {
            max_notes: 16,
            duration_secs: 4.0,
            ..Default::default()
        },
        &state,
        42,
    );
    let tempo = symthaea_muse::rhythm::compute_tempo(&MuseConfig::default(), &state);
    let before_path = Path::new("target/training-comparison");
    std::fs::create_dir_all(before_path).ok();
    before
        .to_midi_file(&before_path.join("before.mid"), tempo)
        .ok();

    // After: trained projections
    // (For now, compose uses whatever projections are in NeuralMelody::new)
    // The real "after" requires loading projections back — implemented but needs
    // NeuralMelody to accept loaded projections, which is a future integration step.
    let after = symthaea_muse::compose(
        &MuseConfig {
            max_notes: 16,
            duration_secs: 4.0,
            melody_mode: symthaea_muse::MelodyMode::Neural,
            ..Default::default()
        },
        &state,
        42,
    );
    after
        .to_midi_file(&before_path.join("after_neural.mid"), tempo)
        .ok();

    println!("  Before (Classic): {} notes", before.notes.len());
    println!("  After (Neural):   {} notes", after.notes.len());
    println!("  MIDI files: target/training-comparison/{{before,after_neural}}.mid");
    println!("\n  Load in a DAW and listen. The difference is the CfC temporal coherence.");
}
