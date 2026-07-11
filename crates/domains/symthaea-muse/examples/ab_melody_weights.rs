// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blind A/B listening pairs: trained melody weights vs pre-training fallback.
//!
//! Held-out metrics say the MAESTRO-trained MelodyPredictor beats the old
//! hand-tuned constants (69.1% vs 38.2% direction accuracy) — but metrics
//! are not ears. This harness renders the streaming engine twice per
//! scenario, identical except for the predictor weights, and writes blind
//! WAV pairs plus an answer key. Listen first, then open the key.
//!
//! The predictor shapes pitch via a consciousness-scaled blend
//! (streaming.rs), so scenarios use high Ψ to make the difference audible.
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --example ab_melody_weights
//! # listen to audio_output/ab_melody/*.wav  (each scenario: _A and _B)
//! # then: cat audio_output/ab_melody/ANSWER_KEY.json
//! ```

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

const SAMPLE_RATE: u32 = 44100;
const RENDER_SECS: usize = 20;

struct Scenario {
    name: &'static str,
    state: MusicalState,
}

fn main() {
    let out_dir = std::path::Path::new("audio_output/ab_melody");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let scenarios = vec![
        Scenario {
            name: "contemplative",
            state: MusicalState {
                consciousness_level: 0.9,
                valence: 0.4,
                arousal: 0.25,
                serotonin: 0.7,
                ..Default::default()
            },
        },
        Scenario {
            name: "joyful",
            state: MusicalState {
                consciousness_level: 0.85,
                valence: 0.6,
                arousal: 0.6,
                dopamine: 0.8,
                ..Default::default()
            },
        },
        Scenario {
            name: "melancholy",
            state: MusicalState {
                consciousness_level: 0.9,
                valence: -0.5,
                arousal: 0.3,
                serotonin: 0.3,
                ..Default::default()
            },
        },
        Scenario {
            name: "tense",
            state: MusicalState {
                consciousness_level: 0.8,
                valence: -0.4,
                arousal: 0.7,
                noradrenaline: 0.8,
                ..Default::default()
            },
        },
    ];

    println!("=== Blind A/B: trained vs fallback melody weights ===\n");
    let mut key_lines = Vec::new();

    for (idx, sc) in scenarios.iter().enumerate() {
        let trained = render(&sc.state, false);
        let fallback = render(&sc.state, true);

        // Deterministic blind assignment: FNV of the scenario name decides
        // whether "trained" is A or B.
        let trained_is_a = fnv(sc.name) % 2 == 0;
        let (a, b, a_label, b_label) = if trained_is_a {
            (&trained, &fallback, "trained", "fallback")
        } else {
            (&fallback, &trained, "fallback", "trained")
        };

        let path_a = out_dir.join(format!("{:02}_{}_A.wav", idx + 1, sc.name));
        let path_b = out_dir.join(format!("{:02}_{}_B.wav", idx + 1, sc.name));
        write_wav_stereo(&path_a, a);
        write_wav_stereo(&path_b, b);
        println!(
            "  {}: wrote {} + {} ({RENDER_SECS}s each)",
            sc.name,
            path_a.display(),
            path_b.display()
        );
        key_lines.push(format!(
            "  {{\"scenario\": \"{}\", \"A\": \"{a_label}\", \"B\": \"{b_label}\"}}",
            sc.name
        ));
    }

    let key = format!(
        "{{\n  \"note\": \"listen BEFORE reading; record which of A/B you prefer per scenario\",\n\
         \"pairs\": [\n{}\n  ]\n}}\n",
        key_lines.join(",\n")
    );
    let key_path = out_dir.join("ANSWER_KEY.json");
    std::fs::write(&key_path, key).expect("write answer key");
    println!(
        "\nAnswer key: {} — listen first, then look.\n\
         Record preferences; they are the human-evaluation evidence the\n\
         improvement plan calls for (metrics ≠ ears).",
        key_path.display()
    );
}

/// Render RENDER_SECS of streaming audio for a state, with either the
/// trained (default) or fallback melody-predictor weights.
fn render(state: &MusicalState, use_fallback: bool) -> Vec<[f32; 2]> {
    let config = MuseConfig {
        sample_rate: SAMPLE_RATE,
        ..Default::default()
    };
    let mut synth = StreamingSynth::new(config, SAMPLE_RATE);
    if use_fallback {
        synth.use_fallback_melody_weights();
    }
    synth.update_state(state);

    let target = RENDER_SECS * SAMPLE_RATE as usize;
    let mut frames = Vec::with_capacity(target);
    while frames.len() < target {
        frames.extend(synth.render_chunk());
    }
    frames.truncate(target);
    frames
}

fn write_wav_stereo(path: &std::path::Path, frames: &[[f32; 2]]) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for [l, r] in frames {
        writer
            .write_sample((l.clamp(-1.0, 1.0) * 32767.0) as i16)
            .expect("write");
        writer
            .write_sample((r.clamp(-1.0, 1.0) * 32767.0) as i16)
            .expect("write");
    }
    writer.finalize().expect("finalize wav");
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
