// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A piece composed to hold the feeling of "The Luminous Library" — a
//! philosophical text (Evolving Resonant Co-creationism) that explicitly
//! frames itself as a "hymn" and a "Music of Becoming," moving through the
//! same seven-fold refrain at three ascending scales: the individual heart,
//! society, then the whole Kosmos.
//!
//! That shape — a recurring core theme, revisited between widening
//! explorations — is structurally a rondo (ABACA), so an odd seed is chosen
//! deliberately (see `symthaea_music_theory::composer::compose_styled`,
//! which picks `Form::rondo` for odd seeds). `Style::Cinematic` (wide
//! melodic leaps, sustained notes, a warm-but-not-saccharine "sensitive
//! female" progression) carries the text's blend of reverence and cosmic
//! sweep; a bright valence keeps it in a major key (the text is explicitly
//! about love and luminous coherence, not darkness), while moderate arousal
//! leaves room for the piece's own long-range tension arc (quiet at the
//! start, peaking at the piece's structural climax) to do the emotional
//! work of "ascending scale" on its own.
//!
//! Run: cargo run --release -p symthaea-muse --features theory \
//!          --example luminous_library_hymn
//! Output: audio_output/luminous_library_hymn.wav

use symthaea_muse::AudioData;
use symthaea_muse::MusicalState;
use symthaea_muse::theory_realize::compose_and_realize_styled;
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

const SAMPLE_RATE: u32 = 44100;

fn main() {
    let out = std::path::Path::new("audio_output");
    std::fs::create_dir_all(out).expect("mkdir");
    let state = MusicalState::default();

    let intent = MusicalIntent {
        valence: 0.75,
        arousal: 0.55,
        energy: 0.65,
        bars: 8,
        seed: 7, // odd -> rondo: a recurring theme, revisited between wider explorations
        tonic: PitchClass::D,
    };

    println!("Composing 'The Luminous Library' hymn...");
    println!(
        "  valence={} arousal={} energy={} bars={} seed={} tonic={:?} style=Cinematic",
        intent.valence, intent.arousal, intent.energy, intent.bars, intent.seed, intent.tonic
    );

    let comp = compose_and_realize_styled(&intent, Style::Cinematic, &state, SAMPLE_RATE);

    let path = out.join("luminous_library_hymn.wav");
    write_wav(&path, &comp.audio);

    println!(
        "\n{:.1}s, {} notes -> {}",
        comp.duration_secs,
        comp.notes.len(),
        path.display()
    );
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
