// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blind A/B: the new music-theory composer vs the current random-walk engine.
//!
//! For each emotional scenario this renders the SAME intent two ways —
//! (T) `symthaea-music-theory`'s structural composer (motif development,
//!     antecedent/consequent phrasing, functional harmony, cadences,
//!     structure-driven rubato), realized by muse; and
//! (E) muse's current `compose()` engine (the constrained random walk) —
//! and writes blind, labeled WAV pairs plus an answer key.
//!
//! Listen BEFORE reading the key. The question this answers with your ears:
//! does explicit musical structure sound less like "a child / a robot" than
//! the random walk?
//!
//! Run: cargo run --release -p symthaea-muse --features theory \
//!          --example theory_vs_engine

use symthaea_muse::theory_realize::realize;
use symthaea_muse::{AudioData, Composition, MuseConfig, MusicalState};
use symthaea_music_theory::{MusicalIntent, PitchClass};

const SAMPLE_RATE: u32 = 44100;

struct Scenario {
    name: &'static str,
    intent: MusicalIntent,
    state: MusicalState,
}

fn main() {
    let out_dir = std::path::Path::new("audio_output/theory_ab");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let scenarios = vec![
        Scenario {
            name: "serene",
            intent: MusicalIntent {
                valence: 0.5,
                arousal: 0.2,
                energy: 0.5,
                bars: 4,
                seed: 1,
                tonic: PitchClass::C,
            },
            state: MusicalState {
                valence: 0.5,
                arousal: 0.2,
                serotonin: 0.7,
                consciousness_level: 0.7,
                ..Default::default()
            },
        },
        Scenario {
            name: "joyful",
            intent: MusicalIntent {
                valence: 0.7,
                arousal: 0.7,
                energy: 0.8,
                bars: 4,
                seed: 2,
                tonic: PitchClass::G,
            },
            state: MusicalState {
                valence: 0.7,
                arousal: 0.7,
                dopamine: 0.8,
                consciousness_level: 0.8,
                ..Default::default()
            },
        },
        Scenario {
            name: "melancholy",
            intent: MusicalIntent {
                valence: -0.6,
                arousal: 0.3,
                energy: 0.5,
                bars: 4,
                seed: 3,
                tonic: PitchClass::A,
            },
            state: MusicalState {
                valence: -0.6,
                arousal: 0.3,
                serotonin: 0.3,
                consciousness_level: 0.7,
                ..Default::default()
            },
        },
        Scenario {
            name: "yearning",
            intent: MusicalIntent {
                valence: -0.2,
                arousal: 0.5,
                energy: 0.7,
                bars: 4,
                seed: 4,
                tonic: PitchClass::D,
            },
            state: MusicalState {
                valence: -0.2,
                arousal: 0.5,
                noradrenaline: 0.5,
                consciousness_level: 0.8,
                ..Default::default()
            },
        },
    ];

    println!("=== Blind A/B: theory composer vs random-walk engine ===\n");
    let mut key_lines = Vec::new();

    for (idx, sc) in scenarios.iter().enumerate() {
        // (T) theory composer → realized by muse
        let theory = realize(
            &symthaea_music_theory::compose(&sc.intent),
            &sc.state,
            SAMPLE_RATE,
        );

        // (E) current engine: compose() with a matching duration
        let engine = {
            let config = MuseConfig {
                sample_rate: SAMPLE_RATE,
                duration_secs: theory.duration_secs.max(8.0),
                max_notes: 32,
                ..Default::default()
            };
            symthaea_muse::compose(&config, &sc.state, sc.intent.seed)
        };

        // Deterministic blind assignment: seed parity decides which is A.
        let theory_is_a = fnv(sc.name) % 2 == 0;
        let (a, b, a_label, b_label) = if theory_is_a {
            (&theory, &engine, "theory", "engine")
        } else {
            (&engine, &theory, "engine", "theory")
        };

        let pa = out_dir.join(format!("{:02}_{}_A.wav", idx + 1, sc.name));
        let pb = out_dir.join(format!("{:02}_{}_B.wav", idx + 1, sc.name));
        write_wav(&pa, a);
        write_wav(&pb, b);
        println!("  {}: wrote A + B", sc.name);
        key_lines.push(format!(
            "  {{\"scenario\": \"{}\", \"A\": \"{a_label}\", \"B\": \"{b_label}\"}}",
            sc.name
        ));
    }

    let key = format!(
        "{{\n  \"note\": \"listen BEFORE reading; note which of A/B sounds more \
         intentional / less aimless per scenario\",\n  \"pairs\": [\n{}\n  ]\n}}\n",
        key_lines.join(",\n")
    );
    let key_path = out_dir.join("ANSWER_KEY.json");
    std::fs::write(&key_path, key).expect("write key");
    println!(
        "\nWrote {} pairs to {} + ANSWER_KEY.json.\nListen first; the T files are \
         the structural composer, E the current engine.",
        scenarios.len(),
        out_dir.display()
    );
}

fn write_wav(path: &std::path::Path, comp: &Composition) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: comp.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    match &comp.audio {
        AudioData::StereoF32(frames) => {
            for [l, r] in frames {
                w.write_sample((l.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .unwrap();
                w.write_sample((r.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .unwrap();
            }
        }
        AudioData::F32(s) => {
            for &x in s {
                let v = (x.clamp(-1.0, 1.0) * 32767.0) as i16;
                w.write_sample(v).unwrap();
                w.write_sample(v).unwrap();
            }
        }
        AudioData::I16(s) => {
            for &x in s {
                w.write_sample(x).unwrap();
                w.write_sample(x).unwrap();
            }
        }
    }
    w.finalize().expect("finalize");
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
