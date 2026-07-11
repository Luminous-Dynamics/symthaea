// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hear the real instrument wiring: the SAME `MusicalIntent`, rendered once
//! per [`Style`] through muse's ACTUAL synthesis engine (additive acoustic
//! partials, Karplus-Strong plucked strings, and FM synthesis) rather than
//! `symthaea-music-theory`'s standalone diagnostic sine-synth demos.
//!
//! This is the fix for the "every voice is the same chiptune patch in a
//! different register" character: melody/harmony/bass now each get a real,
//! distinct instrument, and that instrument is chosen per genre.
//!
//! Run: cargo run --release -p symthaea-muse --features theory \
//!          --example instrument_ensembles
//! Output: audio_output/instrument_ensembles/*.wav

use symthaea_muse::theory_realize::compose_and_realize_styled;
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

const SAMPLE_RATE: u32 = 44100;

fn main() {
    let out = std::path::Path::new("audio_output/instrument_ensembles");
    std::fs::create_dir_all(out).expect("mkdir");

    let intent = MusicalIntent {
        valence: 0.4,
        arousal: 0.5,
        energy: 0.6,
        bars: 4,
        seed: 5,
        tonic: PitchClass::C,
    };
    let state = MusicalState::default();

    let styles = [
        ("classical", Style::Classical),
        ("waltz", Style::Waltz),
        ("folk", Style::Folk),
        ("cinematic", Style::Cinematic),
        ("playful", Style::Playful),
    ];

    println!("=== Rendering one intent through every Style's real instrument ensemble ===\n");
    for (name, style) in styles {
        let comp = compose_and_realize_styled(&intent, style, &state, SAMPLE_RATE);
        let path = out.join(format!("{name}.wav"));
        write_wav(&path, &comp.audio);
        println!(
            "  {name:10} {:.1}s  {} notes -> {}",
            comp.duration_secs,
            comp.notes.len(),
            path.display()
        );
    }
    println!("\nListen to audio_output/instrument_ensembles/*.wav -- each style is a");
    println!("different real ensemble now: classical/waltz = string trio, folk =");
    println!("flute + plucked guitar + plucked upright bass, cinematic = violin +");
    println!("organ + cello, playful = clarinet + FM electric piano + plucked guitar.");
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
