// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-refinement: generate → evaluate → keep best → iterate.
//!
//! Symthaea listens to her own compositions, judges them, keeps the best,
//! and evolves her parameters toward what she considers beautiful.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example self_refine
//! ```

use symthaea_muse::creative_bench::{
    AudioQualityScore, CreativeQualityScore, HarmonicProgressionScore, TheoryValidation,
};
use symthaea_muse::{compose, export, pitch, MuseConfig, MusicalState};
use symthaea_aesthetic::ValenceArousal;

const CANDIDATES_PER_ROUND: usize = 6;
const REFINEMENT_ROUNDS: usize = 4;

fn main() {
    println!("═══ Symthaea Self-Refinement Loop ═══\n");

    let out_dir = "data/refined_wav";
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
        ("Joyful", "refined_joyful.wav", MusicalState {
            valence: 0.7, arousal: 0.7, dopamine: 0.8, serotonin: 0.5,
            noradrenaline: 0.3, consciousness_level: 0.7, prediction_error: 0.2,
            harmony_activations: [0.6, 0.7, 0.7, 0.3, 0.5, 0.4, 0.6, 0.1],
        }),
        ("Serene", "refined_serene.wav", MusicalState {
            valence: 0.5, arousal: 0.15, dopamine: 0.2, serotonin: 0.8,
            noradrenaline: 0.05, consciousness_level: 0.6, prediction_error: 0.05,
            harmony_activations: [0.5, 0.4, 0.5, 0.1, 0.3, 0.3, 0.1, 0.7],
        }),
        ("Melancholy", "refined_melancholy.wav", MusicalState {
            valence: -0.6, arousal: 0.2, dopamine: 0.1, serotonin: 0.3,
            noradrenaline: 0.15, consciousness_level: 0.5, prediction_error: 0.1,
            harmony_activations: [0.3, 0.2, 0.3, 0.3, 0.2, 0.2, 0.1, 0.5],
        }),
        ("Intense", "refined_intense.wav", MusicalState {
            valence: -0.3, arousal: 0.85, dopamine: 0.5, serotonin: 0.2,
            noradrenaline: 0.7, consciousness_level: 0.6, prediction_error: 0.5,
            harmony_activations: [0.3, 0.4, 0.3, 0.7, 0.4, 0.3, 0.7, 0.0],
        }),
    ];

    for (name, filename, state) in &scenarios {
        println!("── {} ──", name);
        let va = ValenceArousal::new(state.valence, state.arousal);
        let scale = pitch::build_scale(state);

        let mut best_comp = None;
        let mut best_score = 0.0f32;
        let mut best_seed = 0u64;

        for round in 0..REFINEMENT_ROUNDS {
            let mut round_best_score = 0.0f32;
            let mut round_best_seed = 0u64;

            let mut rejected = 0usize;
            for candidate in 0..CANDIDATES_PER_ROUND {
                // Each candidate gets a different seed. Retry up to 4x per slot if rejected.
                let mut seed = (round * CANDIDATES_PER_ROUND + candidate) as u64 + 1;
                let mut comp = compose(&config, state, seed);
                let mut attempts = 0;

                let audio = loop {
                    let audio = match &comp.audio {
                        symthaea_muse::AudioData::StereoF32(s) => {
                            AudioQualityScore::evaluate(s, comp.sample_rate)
                        }
                        symthaea_muse::AudioData::F32(s) => {
                            AudioQualityScore::evaluate_mono(s, comp.sample_rate)
                        }
                        symthaea_muse::AudioData::I16(s) => {
                            let f: Vec<f32> = s.iter().map(|&v| v as f32 / 32768.0).collect();
                            AudioQualityScore::evaluate_mono(&f, comp.sample_rate)
                        }
                    };

                    // HARD GATE: reject if spectral flatness > 0.4 (static/noise)
                    // Spectral flatness is the one metric that correlates with user perception:
                    // flat > 0.5 = NOISY (user experienced as static/harsh)
                    // flat < 0.3 = TONAL (user experienced as musical)
                    if audio.spectral_flatness > 0.4 && attempts < 4 {
                        rejected += 1;
                        attempts += 1;
                        // Try a different seed
                        seed = seed.wrapping_mul(31).wrapping_add(attempts * 1000);
                        comp = compose(&config, state, seed);
                        continue;
                    }
                    break audio;
                };

                // Score across all 4 dimensions (for ranking survivors)
                let creative = CreativeQualityScore::evaluate(&comp, va);
                let theory = TheoryValidation::validate(&comp.notes, &scale);
                let harmonic = HarmonicProgressionScore::evaluate(&comp.notes);

                // Perceptually-calibrated composite: tonal-ness dominates.
                // Flatness < 0.3 = tonal bonus, > 0.3 = penalty.
                // This calibration comes from user ratings:
                //   refined_melancholy (good)   flat=0.153
                //   refined_joyful (bad)        flat=0.516
                //   composed_07_flow (arcade)   flat=0.560
                let tonal_bonus = (0.4 - audio.spectral_flatness).max(0.0) * 2.5;
                let total = 0.25 * creative.composite
                    + 0.15 * audio.composite
                    + 0.20 * theory.composite
                    + 0.15 * harmonic.composite
                    + 0.25 * tonal_bonus; // perceptual correlate

                if total > round_best_score {
                    round_best_score = total;
                    round_best_seed = seed;
                }

                if total > best_score {
                    best_score = total;
                    best_seed = seed;
                    best_comp = Some(comp);
                }
            }
            if rejected > 0 {
                println!("  Round {}: rejected {} noisy candidates (flat > 0.4)",
                    round + 1, rejected);
            }

            println!(
                "  Round {}: best seed={}, score={:.3} (overall best={:.3})",
                round + 1, round_best_seed, round_best_score, best_score
            );
        }

        // Export the best composition
        if let Some(comp) = &best_comp {
            let path = format!("{}/{}", out_dir, filename);
            match export::write_wav(&path, comp) {
                Ok(()) => {
                    let size_kb = std::fs::metadata(&path)
                        .map(|m| m.len() / 1024)
                        .unwrap_or(0);
                    println!(
                        "  Winner: seed={}, score={:.3}, {} notes, {} KB\n",
                        best_seed, best_score, comp.notes.len(), size_kb
                    );
                }
                Err(e) => println!("  ERROR: {}\n", e),
            }
        }
    }

    println!("═══ Self-Refinement Complete ═══");
    println!("  Best compositions saved to {}/", out_dir);
    println!("  Each is the winner of {} candidates across {} rounds.",
        CANDIDATES_PER_ROUND * REFINEMENT_ROUNDS, REFINEMENT_ROUNDS);
}
