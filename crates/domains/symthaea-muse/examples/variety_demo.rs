// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hear the variety pass: the SAME style, several different seeds, through
//! the real synthesis + mastering pipeline. Each seed can land on a
//! different instrument pool member, a different B/C-section
//! transformation, a different motif orientation, ternary vs rondo form,
//! and occasionally the new V7/V chromatic color (whenever a ii-before-V
//! measure occurs in the generated progression).
//!
//! Run: cargo run --release -p symthaea-muse --features theory \
//!          --example variety_demo
//! Output: audio_output/variety_demo/*.wav

use symthaea_muse::AudioData;
use symthaea_muse::MusicalState;
use symthaea_muse::theory_realize::compose_and_realize_styled;
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

const SAMPLE_RATE: u32 = 44100;

fn main() {
    let out = std::path::Path::new("audio_output/variety_demo");
    std::fs::create_dir_all(out).expect("mkdir");
    let state = MusicalState::default();

    println!("=== Same style (Classical), five different seeds ===\n");
    for seed in 0..5u64 {
        let intent = MusicalIntent {
            valence: 0.4,
            arousal: 0.5,
            energy: 0.6,
            bars: 4,
            seed,
            tonic: PitchClass::C,
        };
        let comp = compose_and_realize_styled(&intent, Style::Classical, &state, SAMPLE_RATE);
        let form_kind = if seed % 2 == 0 { "ternary" } else { "rondo" };
        let path = out.join(format!("classical_seed{seed}.wav"));
        write_wav(&path, &comp.audio);
        println!(
            "  seed={seed}  form={form_kind:8}  {:.1}s  {} notes -> {}",
            comp.duration_secs,
            comp.notes.len(),
            path.display()
        );
    }

    println!("\n=== Same seed, five different styles (for contrast) ===\n");
    let styles = [
        ("classical", Style::Classical),
        ("waltz", Style::Waltz),
        ("folk", Style::Folk),
        ("cinematic", Style::Cinematic),
        ("playful", Style::Playful),
    ];
    let intent = MusicalIntent {
        valence: 0.4,
        arousal: 0.5,
        energy: 0.6,
        bars: 4,
        seed: 2,
        tonic: PitchClass::C,
    };
    for (name, style) in styles {
        let comp = compose_and_realize_styled(&intent, style, &state, SAMPLE_RATE);
        let path = out.join(format!("style_{name}.wav"));
        write_wav(&path, &comp.audio);
        println!(
            "  {name:10} {:.1}s  {} notes -> {}",
            comp.duration_secs,
            comp.notes.len(),
            path.display()
        );
    }

    println!("\nListen to audio_output/variety_demo/*.wav:");
    println!("  classical_seed*.wav -- same style, should sound like DIFFERENT");
    println!("  pieces now (instrument, development, form all vary by seed).");
    println!("  style_*.wav -- same seed, five genuinely different ensembles/forms.");
}

fn write_wav(path: &std::path::Path, audio: &AudioData) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create");
    match audio {
        AudioData::StereoF32(frames) => {
            for [l, r] in frames {
                w.write_sample((l.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .unwrap();
                w.write_sample((r.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .unwrap();
            }
        }
        AudioData::F32(samples) => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                w.write_sample(v).unwrap();
                w.write_sample(v).unwrap();
            }
        }
        AudioData::I16(samples) => {
            for &s in samples {
                w.write_sample(s).unwrap();
                w.write_sample(s).unwrap();
            }
        }
    }
    w.finalize().expect("finalize");
}
