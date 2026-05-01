// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export WAV files using the compose() pipeline — the same path benchmarked.
//!
//! Unlike export_emotional_wavs (which uses StreamingSynth), this uses the
//! full compose() → arrange() → render_arrangement() path with chord
//! accompaniment, emotional gestures, and phrase contour.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example export_composed_wavs
//! ```

use symthaea_muse::{compose, export, MuseConfig, MusicalState};

fn main() {
    println!("═══ Symthaea Muse: Composed WAV Export ═══\n");

    let out_dir = "data/composed_wav";
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let config = MuseConfig {
        sample_rate: 44100,
        duration_secs: 15.0,
        max_notes: 64,
        num_partials: 12,
        enable_antialiasing: true,
        enable_sub_bass: true,
        ..Default::default()
    };

    let scenarios: Vec<(&str, &str, MusicalState)> = vec![
        (
            "Contentment",
            "01_contentment.wav",
            MusicalState {
                valence: 0.7,
                arousal: 0.2,
                dopamine: 0.3,
                serotonin: 0.8,
                noradrenaline: 0.1,
                consciousness_level: 0.6,
                prediction_error: 0.1,
                harmony_activations: [0.7, 0.5, 0.6, 0.1, 0.4, 0.3, 0.2, 0.5],
            },
        ),
        (
            "Excitement",
            "02_excitement.wav",
            MusicalState {
                valence: 0.8,
                arousal: 0.85,
                dopamine: 0.9,
                serotonin: 0.3,
                noradrenaline: 0.4,
                consciousness_level: 0.8,
                prediction_error: 0.2,
                harmony_activations: [0.5, 0.7, 0.8, 0.3, 0.6, 0.5, 0.7, 0.1],
            },
        ),
        (
            "Grief",
            "03_grief.wav",
            MusicalState {
                valence: -0.8,
                arousal: 0.15,
                dopamine: 0.1,
                serotonin: 0.2,
                noradrenaline: 0.2,
                consciousness_level: 0.4,
                prediction_error: 0.1,
                harmony_activations: [0.2, 0.1, 0.2, 0.4, 0.1, 0.1, 0.1, 0.6],
            },
        ),
        (
            "Panic",
            "04_panic.wav",
            MusicalState {
                valence: -0.7,
                arousal: 0.95,
                dopamine: 0.3,
                serotonin: 0.1,
                noradrenaline: 0.9,
                consciousness_level: 0.3,
                prediction_error: 0.7,
                harmony_activations: [0.1, 0.2, 0.1, 0.9, 0.3, 0.1, 0.5, 0.0],
            },
        ),
        (
            "Wonder",
            "05_wonder.wav",
            MusicalState {
                valence: 0.6,
                arousal: 0.5,
                dopamine: 0.6,
                serotonin: 0.5,
                noradrenaline: 0.2,
                consciousness_level: 0.9,
                prediction_error: 0.3,
                harmony_activations: [0.6, 0.5, 0.7, 0.2, 0.5, 0.4, 0.3, 0.3],
            },
        ),
        (
            "Sacred Stillness",
            "06_sacred_stillness.wav",
            MusicalState {
                valence: 0.2,
                arousal: 0.05,
                dopamine: 0.1,
                serotonin: 0.6,
                noradrenaline: 0.0,
                consciousness_level: 0.5,
                prediction_error: 0.0,
                harmony_activations: [0.3, 0.2, 0.3, 0.0, 0.2, 0.2, 0.1, 0.9],
            },
        ),
        (
            "Flow",
            "07_flow.wav",
            MusicalState {
                valence: 0.5,
                arousal: 0.6,
                dopamine: 0.7,
                serotonin: 0.5,
                noradrenaline: 0.3,
                consciousness_level: 0.85,
                prediction_error: 0.2,
                harmony_activations: [0.6, 0.6, 0.7, 0.2, 0.5, 0.5, 0.5, 0.2],
            },
        ),
        (
            "Tension",
            "08_tension.wav",
            MusicalState {
                valence: -0.3,
                arousal: 0.7,
                dopamine: 0.4,
                serotonin: 0.2,
                noradrenaline: 0.6,
                consciousness_level: 0.5,
                prediction_error: 0.5,
                harmony_activations: [0.2, 0.3, 0.2, 0.8, 0.3, 0.2, 0.6, 0.0],
            },
        ),
    ];

    for (name, filename, state) in &scenarios {
        print!("  {} ... ", name);

        let comp = compose(&config, state, 42);
        let path = format!("{}/{}", out_dir, filename);

        match export::write_wav(&path, &comp) {
            Ok(()) => {
                let size_kb = std::fs::metadata(&path)
                    .map(|m| m.len() / 1024)
                    .unwrap_or(0);
                println!("{} ({} notes, {} KB)", path, comp.notes.len(), size_kb);
            }
            Err(e) => println!("ERROR: {}", e),
        }
    }

    println!("\n  Done. These use the compose() pipeline with chords,");
    println!("  emotional gestures, and phrase contour.");
}
