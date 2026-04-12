// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Render a listenable demo composition to disk.
//!
//! Generates 4 WAV files across different emotional states so you can hear
//! what Symthaea's music system actually sounds like.
//!
//! ```bash
//! cargo run -p symthaea-muse --example render_demo --release
//! # Then play: audio_output/demo_joyful.wav
//! ```

use symthaea_muse::export::write_wav;
use symthaea_muse::{compose, MelodyMode, MuseConfig, MusicalState};

fn main() {
    let dir = "audio_output";
    std::fs::create_dir_all(dir).ok();

    let base_config = MuseConfig {
        duration_secs: 8.0,
        max_notes: 32,
        melody_mode: MelodyMode::Neural,
        enable_antialiasing: true,
        num_partials: 8,
        ..Default::default()
    };
    // Joyful needs dense rhythmic notes (upbeat = arpeggiator-like density)
    // Use Classic mode (reliable grid timing) with high note count
    let joyful_config = MuseConfig {
        max_notes: 48,
        melody_mode: MelodyMode::Classic,
        ..base_config.clone()
    };
    let config = base_config;

    let states: Vec<(&str, MusicalState)> = vec![
        (
            "joyful",
            MusicalState {
                valence: 0.8,
                arousal: 0.85, // high arousal = fast tempo, dense notes
                consciousness_level: 0.7,
                dopamine: 0.9,
                serotonin: 0.5,
                noradrenaline: 0.5,
                prediction_error: 0.1,
                harmony_activations: [0.9, 0.7, 0.8, 0.2, 0.6, 0.5, 0.7, 0.1],
            },
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
        ),
        (
            "tense",
            MusicalState {
                valence: -0.5,
                arousal: 0.8,
                consciousness_level: 0.5,
                dopamine: 0.4,
                serotonin: 0.2,
                noradrenaline: 0.9,
                prediction_error: 0.6,
                harmony_activations: [0.3, 0.2, 0.4, 0.9, 0.7, 0.3, 0.8, 0.1],
            },
        ),
        (
            "sorrowful",
            MusicalState {
                valence: -0.6,
                arousal: 0.2,
                consciousness_level: 0.4,
                dopamine: 0.2,
                serotonin: 0.3,
                noradrenaline: 0.3,
                prediction_error: 0.3,
                harmony_activations: [0.4, 0.2, 0.3, 0.5, 0.3, 0.2, 0.3, 0.7],
            },
        ),
    ];

    println!("Rendering Symthaea demo compositions...\n");

    for (name, state) in &states {
        let cfg = if *name == "joyful" { &joyful_config } else { &config };
        let comp = compose(cfg, state, 42);
        let path = format!("{dir}/demo_{name}.wav");
        match write_wav(&path, &comp) {
            Ok(()) => {
                println!(
                    "  {} — {} notes, {:.1}s → {}",
                    name,
                    comp.notes.len(),
                    comp.duration_secs,
                    path,
                );
            }
            Err(e) => {
                eprintln!("  {} — FAILED: {}", name, e);
            }
        }
    }

    println!("\nDone. Play with: aplay audio_output/demo_joyful.wav");
}
