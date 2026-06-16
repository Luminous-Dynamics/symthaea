// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare compositions with random vs trained melody projections.
//!
//! Generates WAV audio + MIDI files across 4 emotional states for A/B listening.
//! Output goes to `audio_output/` (WAV) and `target/trained-vs-random/` (MIDI).
//!
//! ```bash
//! cargo run -p symthaea-muse --example trained_vs_random --release
//! # Then play: audio_output/trained_joyful_neural.wav
//! ```

use std::path::Path;
use symthaea_muse::critic::evaluate_composition;
use symthaea_muse::export::write_wav;
use symthaea_muse::rhythm::compute_tempo;
use symthaea_muse::{MelodyMode, MuseConfig, MusicalState, compose};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Trained vs Random: A/B Comparison");
    println!("═══════════════════════════════════════════════════════\n");

    let midi_dir = Path::new("target/trained-vs-random");
    let audio_dir = Path::new("audio_output");
    std::fs::create_dir_all(midi_dir).ok();
    std::fs::create_dir_all(audio_dir).ok();

    let states: Vec<(&str, MusicalState)> = vec![
        (
            "serene",
            MusicalState {
                harmony_activations: [0.3, 0.7, 0.6, 0.2, 0.5, 0.6, 0.3, 0.8],
                serotonin: 0.8,
                dopamine: 0.3,
                noradrenaline: 0.1,
                arousal: 0.2,
                valence: 0.6,
                consciousness_level: 0.8,
                prediction_error: 0.05,
            },
        ),
        (
            "turbulent",
            MusicalState {
                harmony_activations: [0.2, 0.2, 0.3, 0.8, 0.4, 0.2, 0.7, 0.1],
                serotonin: 0.2,
                dopamine: 0.8,
                noradrenaline: 0.9,
                arousal: 0.9,
                valence: -0.4,
                consciousness_level: 0.6,
                prediction_error: 0.7,
            },
        ),
        (
            "contemplative",
            MusicalState {
                harmony_activations: [0.6, 0.4, 0.8, 0.1, 0.3, 0.4, 0.2, 0.9],
                serotonin: 0.7,
                dopamine: 0.4,
                noradrenaline: 0.2,
                arousal: 0.15,
                valence: 0.1,
                consciousness_level: 0.9,
                prediction_error: 0.1,
            },
        ),
        (
            "joyful",
            MusicalState {
                harmony_activations: [0.7, 0.9, 0.5, 0.7, 0.6, 0.8, 0.6, 0.2],
                serotonin: 0.6,
                dopamine: 0.9,
                noradrenaline: 0.3,
                arousal: 0.7,
                valence: 0.8,
                consciousness_level: 0.85,
                prediction_error: 0.15,
            },
        ),
    ];

    println!(
        "  {:12} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>7}",
        "State", "C:note", "C:mel", "C:comp", "N:note", "N:mel", "N:comp", "Δcomp"
    );
    println!(
        "  {:─>12}─┼─{:─>6}─{:─>6}─{:─>6}─┼─{:─>6}─{:─>6}─{:─>6}─┼─{:─>7}",
        "", "", "", "", "", "", "", ""
    );

    for (name, state) in &states {
        let config_classic = MuseConfig {
            duration_secs: 8.0,
            max_notes: 24,
            melody_mode: MelodyMode::Classic,
            ..Default::default()
        };
        let config_neural = MuseConfig {
            duration_secs: 8.0,
            max_notes: 24,
            melody_mode: MelodyMode::Neural,
            ..Default::default()
        };

        // Classic (random) composition
        let classic = compose(&config_classic, state, 42);
        let classic_verdict = evaluate_composition(&classic, state);
        let tempo_c = compute_tempo(&config_classic, state);
        classic
            .to_midi_file(&midi_dir.join(format!("{name}_classic.mid")), tempo_c)
            .ok();
        write_wav(
            &audio_dir
                .join(format!("trained_{name}_classic.wav"))
                .to_string_lossy(),
            &classic,
        )
        .ok();

        // Neural (trained projections loaded automatically in compose)
        let neural = compose(&config_neural, state, 42);
        let neural_verdict = evaluate_composition(&neural, state);
        let tempo_n = compute_tempo(&config_neural, state);
        neural
            .to_midi_file(&midi_dir.join(format!("{name}_neural.mid")), tempo_n)
            .ok();
        write_wav(
            &audio_dir
                .join(format!("trained_{name}_neural.wav"))
                .to_string_lossy(),
            &neural,
        )
        .ok();

        let delta = neural_verdict.composite - classic_verdict.composite;
        println!(
            "  {:12} | {:5} {:6.3} {:6.3} | {:5} {:6.3} {:6.3} | {:+7.3}",
            name,
            classic.notes.len(),
            classic_verdict.melodic_interest,
            classic_verdict.composite,
            neural.notes.len(),
            neural_verdict.melodic_interest,
            neural_verdict.composite,
            delta,
        );
    }

    println!("\n  C = Classic (random projections)");
    println!("  N = Neural (trained on MAESTRO)");
    println!("  note = note count, mel = melodic interest, comp = composite score");
    println!("  Δcomp = Neural minus Classic (positive = Neural better)\n");
    println!("  WAV audio: audio_output/trained_{{state}}_{{classic,neural}}.wav");
    println!("  MIDI data: target/trained-vs-random/{{state}}_{{classic,neural}}.mid\n");
    println!("  Play the WAV files directly — no DAW needed.");
    println!("  Listen for: phrases, contour, rhythmic coherence, emotional tone.\n");
}
