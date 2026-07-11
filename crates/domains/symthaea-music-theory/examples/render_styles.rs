// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hear the genre/style system: the SAME intent (valence/arousal/energy/
//! tonic/seed), rendered once per [`Style`], through the same plain
//! additive synth `render_wav` uses. The point is to isolate the variable
//! Tristan is asking about: does changing ONLY the style genuinely change
//! the character (meter, tempo, melodic shape, harmonic loop), or is it
//! cosmetic?
//!
//! Run: cargo run --release -p symthaea-music-theory --example render_styles
//! Output: audio_output/theory_styles/*.wav

use symthaea_music_theory::score::{Emphasis, Score, VoiceRole};
use symthaea_music_theory::{MusicalIntent, PitchClass, Style, compose_styled};

const SR: u32 = 44100;

fn main() {
    let out = std::path::Path::new("audio_output/theory_styles");
    std::fs::create_dir_all(out).expect("mkdir");

    let intent = MusicalIntent {
        valence: 0.4,
        arousal: 0.5,
        energy: 0.6,
        bars: 4,
        seed: 5,
        tonic: PitchClass::C,
    };

    let styles = [
        ("classical", Style::Classical),
        ("waltz", Style::Waltz),
        ("folk", Style::Folk),
        ("cinematic", Style::Cinematic),
        ("playful", Style::Playful),
    ];

    println!("=== Rendering one intent through every Style ===\n");
    for (name, style) in styles {
        let score = compose_styled(&intent, style);
        let frames = synth(&score);
        let path = out.join(format!("{name}.wav"));
        write_wav(&path, &frames);
        println!(
            "  {name:10} meter={}/4 tempo={:.0}  {} notes  {:.1}s -> {}",
            score.meter,
            score.tempo_bpm,
            score.notes.len(),
            frames.len() as f32 / SR as f32,
            path.display()
        );
    }
    println!("\nListen to audio_output/theory_styles/*.wav — same intent, five");
    println!("genuinely different characters: waltz's 3/4 lilt, folk's open");
    println!("pentatonic-flavored melody, cinematic's wide sustained leaps,");
    println!("playful's syncopated bounce.");
}

/// Rubato: a shared beat->seconds map from the score's structural emphases
/// (breath before phrases, dwell on climax, ritardando into cadences).
fn beat_map(score: &Score) -> impl Fn(f64) -> f32 {
    let spb = 60.0 / score.tempo_bpm as f64;
    let mut events: Vec<(f64, f64)> = Vec::new();
    let mel = score.voice(VoiceRole::Melody);
    let last = mel.len().saturating_sub(1);
    for (i, n) in mel.iter().enumerate() {
        let onset = n.onset.beats();
        let end = (n.onset + n.duration).beats();
        match n.emphasis {
            Emphasis::PhraseStart => events.push((onset, 0.10 * spb)),
            Emphasis::Climax => events.push((end, 0.18 * spb)),
            Emphasis::Cadential => events.push((end, if i == last { 0.9 } else { 0.35 } * spb)),
            Emphasis::Normal => {}
        }
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    move |beat: f64| {
        let extra: f64 = events
            .iter()
            .take_while(|(b, _)| *b <= beat + 1e-9)
            .map(|(_, d)| d)
            .sum();
        (beat * spb + extra) as f32
    }
}

fn midi_to_hz(m: u8) -> f32 {
    440.0 * 2.0_f32.powf((m as f32 - 69.0) / 12.0)
}

/// Simple additive polyphonic synth: a few partials + ADSR per note.
fn synth(score: &Score) -> Vec<[f32; 2]> {
    let map = beat_map(score);
    let total = map(score.total_beats.beats()) + 1.0;
    let n = (total * SR as f32) as usize + 1;
    let mut buf = vec![[0.0f32; 2]; n];

    for note in &score.notes {
        let start = map(note.onset.beats());
        let end = map((note.onset + note.duration).beats());
        let dur = (end - start).max(0.05);
        let freq = midi_to_hz(note.pitch.midi());
        let s0 = (start * SR as f32) as usize;

        // Per-role colour: melody bright + centred, harmony soft, bass low+mono.
        let (partials, gain, pan): (&[f32], f32, f32) = match note.role {
            VoiceRole::Melody => (&[1.0, 0.5, 0.33, 0.22, 0.15], 0.5, 0.0),
            VoiceRole::Harmony => (&[1.0, 0.4, 0.2], 0.28, 0.25),
            VoiceRole::Bass => (&[1.0, 0.5, 0.25], 0.5, 0.0),
            VoiceRole::CounterMelody => (&[1.0, 0.45, 0.28], 0.38, -0.15),
        };
        let amp = gain * note.velocity;
        let gl = (0.5 * (1.0 - pan)).sqrt();
        let gr = (0.5 * (1.0 + pan)).sqrt();

        let len = (dur * SR as f32) as usize;
        for i in 0..len {
            if s0 + i >= n {
                break;
            }
            let t = i as f32 / SR as f32;
            let env = adsr(t, dur);
            let mut s = 0.0f32;
            for (h, &pa) in partials.iter().enumerate() {
                let cf = freq * (h + 1) as f32;
                if cf >= SR as f32 * 0.5 {
                    break;
                }
                s += pa * (std::f32::consts::TAU * cf * t).sin();
            }
            let v = s * env * amp;
            buf[s0 + i][0] += v * gl;
            buf[s0 + i][1] += v * gr;
        }
    }

    // Normalize to about -3 dBFS, then a gentle tanh safety limiter.
    let peak = buf
        .iter()
        .flat_map(|f| [f[0].abs(), f[1].abs()])
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let g = 0.707 / peak;
    for f in &mut buf {
        f[0] = (f[0] * g).tanh();
        f[1] = (f[1] * g).tanh();
    }
    buf
}

/// A gentle ADSR envelope (attack 8ms, decay to 0.7 sustain, 0.12s release).
fn adsr(t: f32, dur: f32) -> f32 {
    let a = 0.008;
    let d = 0.10;
    let sus = 0.7;
    let rel = 0.12;
    if t < a {
        t / a
    } else if t < a + d {
        1.0 - (t - a) / d * (1.0 - sus)
    } else if t < dur {
        sus
    } else {
        (sus * (1.0 - (t - dur) / rel)).max(0.0)
    }
}

fn write_wav(path: &std::path::Path, frames: &[[f32; 2]]) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create");
    for [l, r] in frames {
        w.write_sample((l.clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
        w.write_sample((r.clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().expect("finalize");
}
