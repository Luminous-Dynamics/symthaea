// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export WAV files for each emotional state so you can listen.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example export_emotional_wavs
//! # Output: data/muse_wav/01_contentment.wav, 02_excitement.wav, etc.
//! ```

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

struct Phase {
    name: &'static str,
    filename: &'static str,
    duration_secs: f32,
    state: MusicalState,
}

fn phases() -> Vec<Phase> {
    vec![
        Phase {
            name: "Contentment (warm, calm, major)",
            filename: "01_contentment.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: 0.7, arousal: 0.2,
                dopamine: 0.3, serotonin: 0.8, noradrenaline: 0.1,
                consciousness_level: 0.6, prediction_error: 0.1,
                harmony_activations: [0.7, 0.5, 0.6, 0.1, 0.4, 0.3, 0.2, 0.5],
            },
        },
        Phase {
            name: "Excitement (bright, fast, triumphant)",
            filename: "02_excitement.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: 0.8, arousal: 0.85,
                dopamine: 0.9, serotonin: 0.3, noradrenaline: 0.4,
                consciousness_level: 0.8, prediction_error: 0.2,
                harmony_activations: [0.5, 0.7, 0.8, 0.3, 0.6, 0.5, 0.7, 0.1],
            },
        },
        Phase {
            name: "Grief (dark, slow, minor, detuned)",
            filename: "03_grief.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: -0.8, arousal: 0.15,
                dopamine: 0.1, serotonin: 0.2, noradrenaline: 0.2,
                consciousness_level: 0.4, prediction_error: 0.1,
                harmony_activations: [0.2, 0.1, 0.2, 0.4, 0.1, 0.1, 0.1, 0.6],
            },
        },
        Phase {
            name: "Panic (harsh, fast, dissonant, jittery)",
            filename: "04_panic.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: -0.7, arousal: 0.95,
                dopamine: 0.3, serotonin: 0.1, noradrenaline: 0.9,
                consciousness_level: 0.3, prediction_error: 0.7,
                harmony_activations: [0.1, 0.2, 0.1, 0.9, 0.3, 0.1, 0.5, 0.0],
            },
        },
        Phase {
            name: "Wonder (ascending, spacious, high Phi)",
            filename: "05_wonder.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: 0.6, arousal: 0.5,
                dopamine: 0.6, serotonin: 0.5, noradrenaline: 0.2,
                consciousness_level: 0.9, prediction_error: 0.3,
                harmony_activations: [0.6, 0.5, 0.7, 0.2, 0.5, 0.4, 0.3, 0.3],
            },
        },
        Phase {
            name: "Sacred Stillness (drone, sparse, release)",
            filename: "06_sacred_stillness.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: 0.2, arousal: 0.05,
                dopamine: 0.1, serotonin: 0.6, noradrenaline: 0.0,
                consciousness_level: 0.5, prediction_error: 0.0,
                harmony_activations: [0.3, 0.2, 0.3, 0.0, 0.2, 0.2, 0.1, 0.9],
            },
        },
        Phase {
            name: "Flow (high Phi, balanced, generative)",
            filename: "07_flow.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: 0.5, arousal: 0.6,
                dopamine: 0.7, serotonin: 0.5, noradrenaline: 0.3,
                consciousness_level: 0.85, prediction_error: 0.2,
                harmony_activations: [0.6, 0.6, 0.7, 0.2, 0.5, 0.5, 0.5, 0.2],
            },
        },
        Phase {
            name: "Tension (building, unresolved, tritones)",
            filename: "08_tension.wav",
            duration_secs: 15.0,
            state: MusicalState {
                valence: -0.3, arousal: 0.7,
                dopamine: 0.4, serotonin: 0.2, noradrenaline: 0.6,
                consciousness_level: 0.5, prediction_error: 0.5,
                harmony_activations: [0.2, 0.3, 0.2, 0.8, 0.3, 0.2, 0.6, 0.0],
            },
        },
    ]
}

fn main() {
    println!("═══ Symthaea Muse: Exporting Emotional WAV Files ═══\n");

    let out_dir = "data/muse_wav";
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let config = MuseConfig {
        sample_rate: 44100,
        num_partials: 12,
        enable_antialiasing: true,
        ..Default::default()
    };

    for phase in phases() {
        print!("  {} ... ", phase.name);

        let mut synth = StreamingSynth::new(config.clone(), 44100);
        synth.update_state(&phase.state);

        let total_chunks = (phase.duration_secs * 44100.0 / synth.chunk_samples() as f32) as usize;
        let reassert_interval = (5.0 * 44100.0 / synth.chunk_samples() as f32) as usize;

        let mut all_samples: Vec<[f32; 2]> = Vec::with_capacity(44100 * phase.duration_secs as usize);

        for chunk_idx in 0..total_chunks {
            if chunk_idx % reassert_interval == 0 && chunk_idx > 0 {
                synth.update_state(&phase.state);
            }
            let chunk = synth.render_chunk();
            all_samples.extend_from_slice(&chunk);
        }

        // Write WAV
        let path = format!("{}/{}", out_dir, phase.filename);
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
        for pair in &all_samples {
            for &s in pair {
                let sample = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer.write_sample(sample).expect("write sample");
            }
        }
        writer.finalize().expect("finalize wav");

        let size_kb = std::fs::metadata(&path).map(|m| m.len() / 1024).unwrap_or(0);
        println!("{} ({} KB)", path, size_kb);
    }

    println!("\n  Done. Play with: aplay data/muse_wav/01_contentment.wav");
    println!("  Or open in any audio player / DAW.");
}
