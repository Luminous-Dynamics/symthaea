// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure music quality test — no voice, starts at an audible consciousness level.
//! Tests VCSL sample integration with humanization.
//!
//! Run: cargo run --release -p symthaea-muse --example music_quality_test

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("=== Music Quality Test (no voice, VCSL samples, humanized) ===\n");

    let sample_rate = 44100u32;
    let config = MuseConfig {
        duration_secs: 300.0,
        max_notes: 16,
        num_partials: 12,
        enable_antialiasing: true,
        enable_sub_bass: true,
        ..Default::default()
    };

    // Three test renders at different consciousness states
    let tests = vec![
        // Peaceful: very low arousal, very low dopamine, high serotonin (warmth),
        // high stillness harmony. Should be sparse, gentle, warm.
        (
            "peaceful_piano",
            45.0,
            MusicalState {
                consciousness_level: 0.6,
                arousal: 0.12, // very calm — slow tempo, sparse notes
                valence: 0.4,
                dopamine: 0.15,      // low reward drive — no urgency
                serotonin: 0.85,     // high warmth — gentle rolloff
                noradrenaline: 0.05, // no stress — no harshness
                harmony_activations: [0.5, 0.4, 0.5, 0.0, 0.3, 0.3, 0.0, 0.6],
                prediction_error: 0.05, // no surprise — settled, predictable
            },
        ),
        // Energetic: high arousal, high dopamine, moderate phi, ascending harmonies
        (
            "energetic_flow",
            30.0,
            MusicalState {
                consciousness_level: 0.85,
                arousal: 0.75,
                valence: 0.6,
                dopamine: 0.8,
                serotonin: 0.4,
                noradrenaline: 0.35,
                harmony_activations: [0.5, 0.7, 0.8, 0.2, 0.5, 0.5, 0.7, 0.1],
                prediction_error: 0.25,
            },
        ),
        // Dark tension: high arousal, negative valence, high noradrenaline,
        // InfinitePlay (minor 7th) dominant, high prediction error
        (
            "dark_tension",
            30.0,
            MusicalState {
                consciousness_level: 0.4,
                arousal: 0.8,
                valence: -0.6,
                dopamine: 0.3,
                serotonin: 0.1,     // no warmth — cold, harsh
                noradrenaline: 0.8, // high stress — bright, edgy
                harmony_activations: [0.1, 0.1, 0.1, 0.9, 0.2, 0.0, 0.7, 0.0],
                prediction_error: 0.7, // high surprise — jittery timing
            },
        ),
        // Sacred stillness: near-silence, drone, sub-octave, contemplation
        (
            "sacred_stillness",
            45.0,
            MusicalState {
                consciousness_level: 0.5,
                arousal: 0.05,
                valence: 0.15,
                dopamine: 0.1,
                serotonin: 0.6,
                noradrenaline: 0.0,
                harmony_activations: [0.3, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0, 0.95],
                prediction_error: 0.0,
            },
        ),
    ];

    for (name, duration, state) in &tests {
        print!("  Rendering {} ({:.0}s)... ", name, duration);

        let mut synth = StreamingSynth::new(config.clone(), sample_rate);
        synth.enable_sidechain = true;
        synth.enable_binaural = false; // Keep it simple for testing
        synth.enable_fep = false;
        synth.feedback_strength = 0.2;

        synth.update_state(state);

        let chunks_per_sec = sample_rate as usize / synth.chunk_samples();
        let total_chunks = (*duration * chunks_per_sec as f32) as usize;
        let reassert_every = chunks_per_sec * 3; // re-assert state every 3s

        let mut all: Vec<[f32; 2]> = Vec::new();
        for i in 0..total_chunks {
            if i % reassert_every == 0 && i > 0 {
                synth.update_state(state);
            }
            all.extend_from_slice(&synth.render_chunk());
        }

        // Skip auto-mastering: let consciousness dynamics drive the level naturally
        // Auto-mastering normalizes everything to the same peak, destroying the
        // dynamic contrast between peaceful (quiet) and energetic (loud) states.

        // Write WAV
        let path = dir.join(format!("{name}.wav"));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav");
        for pair in &all {
            for &s in pair {
                w.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .expect("write");
            }
        }
        w.finalize().expect("finalize");

        // Analyze
        let peak = all
            .iter()
            .flat_map(|p| p.iter())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let rms = (all
            .iter()
            .flat_map(|p| p.iter())
            .map(|s| s * s)
            .sum::<f32>()
            / (all.len() * 2) as f32)
            .sqrt();
        let size_kb = std::fs::metadata(&path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0);

        println!(
            "{} ({}KB) peak={:.3} rms={:.4}",
            path.display(),
            size_kb,
            peak,
            rms
        );
    }

    println!("\n  Play: aplay audio_output/peaceful_piano.wav");
}
