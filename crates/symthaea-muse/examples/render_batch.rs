// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch render compositions across multiple seeds for statistical CLAP validation.
//!
//! ```bash
//! cargo run -p symthaea-muse --example render_batch --release -- 10
//! # Generates: audio_output/batch/joyful_001.wav ... joyful_010.wav (×4 emotions)
//! ```

use symthaea_muse::export::write_wav;
use symthaea_muse::{compose, MelodyMode, MuseConfig, MusicalState};

fn main() {
    let n_seeds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let dir = "audio_output/batch";
    std::fs::create_dir_all(dir).ok();

    let base_config = MuseConfig {
        duration_secs: 8.0,
        max_notes: 32,
        melody_mode: MelodyMode::Neural,
        enable_antialiasing: true,
        num_partials: 8,
        ..Default::default()
    };

    let joyful_config = MuseConfig {
        max_notes: 48,
        melody_mode: MelodyMode::Classic,
        ..base_config.clone()
    };

    // Tense needs dense rhythmic activity (dramatic = intense, not sparse)
    let tense_config = MuseConfig {
        max_notes: 48,
        melody_mode: MelodyMode::Classic,
        enable_sub_bass: true, // sub-bass for weight
        ..base_config.clone()
    };

    let states: Vec<(&str, MusicalState, &MuseConfig)> = vec![
        (
            "joyful",
            MusicalState {
                valence: 0.8,
                arousal: 0.85,
                consciousness_level: 0.7,
                dopamine: 0.9,
                serotonin: 0.5,
                noradrenaline: 0.5,
                prediction_error: 0.1,
                harmony_activations: [0.9, 0.7, 0.8, 0.2, 0.6, 0.5, 0.7, 0.1],
            },
            &joyful_config,
        ),
        (
            "contemplative",
            MusicalState {
                valence: 0.2,
                arousal: 0.2,
                consciousness_level: 0.8,
                dopamine: 0.3,
                serotonin: 0.7,
                noradrenaline: 0.2,
                prediction_error: 0.05,
                harmony_activations: [0.5, 0.3, 0.8, 0.2, 0.3, 0.6, 0.4, 0.9],
            },
            &base_config,
        ),
        (
            "tense",
            MusicalState {
                valence: -0.6,
                arousal: 0.9, // higher arousal = faster, more intense
                consciousness_level: 0.5,
                dopamine: 0.3,
                serotonin: 0.1,
                noradrenaline: 0.95, // maximum aggression
                prediction_error: 0.7,
                harmony_activations: [0.2, 0.1, 0.3, 0.95, 0.8, 0.2, 0.9, 0.0],
            },
            &tense_config,
        ),
        (
            "sorrowful",
            MusicalState {
                valence: -0.6,
                arousal: 0.2,
                consciousness_level: 0.4,
                dopamine: 0.2,
                serotonin: 0.4, // slight warmth boost
                noradrenaline: 0.2,
                prediction_error: 0.3,
                harmony_activations: [0.4, 0.2, 0.3, 0.5, 0.3, 0.2, 0.3, 0.7],
            },
            &base_config,
        ),
    ];

    println!(
        "Rendering {} seeds × {} emotions = {} compositions...\n",
        n_seeds,
        states.len(),
        n_seeds * states.len()
    );

    for (name, state, cfg) in &states {
        for seed in 1..=n_seeds as u64 {
            let comp = compose(cfg, state, seed);
            let path = format!("{dir}/{name}_{seed:03}.wav");
            match write_wav(&path, &comp) {
                Ok(()) => {
                    print!(".");
                }
                Err(e) => {
                    eprintln!("\n  {name}_{seed:03}: FAILED: {e}");
                }
            }
        }
        println!(" {name} ({n_seeds} files)");
    }

    println!("\nDone. Run CLAP scoring:");
    println!("  python3 scripts/audio-bench/batch_clap.py audio_output/batch/");
}
