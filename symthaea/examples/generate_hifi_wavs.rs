// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate A/B comparison WAV files for Hi-Fi audio validation.
//!
//! Produces pairs of WAV files: old (4 partial, mono) vs new (8 partial, stereo)
//! plus substrate-specific renders and consciousness variation demos.
//!
//! Run: cargo run --features muse,humanoid --example generate_hifi_wavs
//! Output: audio_output/*.wav

use symthaea_muse::{AudioData, MelodyMode, MuseConfig, MusicalState, OutputFormat, compose};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("Generating Hi-Fi comparison WAV files...\n");

    let rich_state = MusicalState {
        consciousness_level: 0.8,
        arousal: 0.5,
        dopamine: 0.6,
        serotonin: 0.4,
        noradrenaline: 0.3,
        harmony_activations: [0.7, 0.5, 0.6, 0.4, 0.3, 0.5, 0.6, 0.3],
        valence: 0.3,
        prediction_error: 0.2,
    };

    // ── A/B: Old vs New ──
    println!("1. A/B Comparison");
    let old_cfg = MuseConfig {
        duration_secs: 8.0,
        max_notes: 24,
        output_format: OutputFormat::Mono16,
        num_partials: 4,
        enable_antialiasing: false,
        melody_mode: MelodyMode::Classic,
        cfc_layer_sizes: vec![4, 4],
        ..Default::default()
    };
    let new_cfg = MuseConfig {
        duration_secs: 8.0,
        max_notes: 24,
        melody_mode: MelodyMode::Neural,
        ..Default::default()
    };
    let old = compose(&old_cfg, &rich_state, 42);
    let new = compose(&new_cfg, &rich_state, 42);
    write_wav(
        &dir.join("A_old_mono_4partial.wav"),
        &old.audio,
        old.sample_rate,
    );
    write_wav(
        &dir.join("B_new_stereo_8partial.wav"),
        &new.audio,
        new.sample_rate,
    );
    println!("   A_old_mono_4partial.wav  ({} samples)", old.audio.len());
    println!(
        "   B_new_stereo_8partial.wav ({} samples)\n",
        new.audio.len()
    );

    // ── Consciousness Variation ──
    println!("2. Consciousness Variation");
    let states = [
        (
            "calm_contemplative",
            MusicalState {
                consciousness_level: 0.9,
                arousal: 0.1,
                serotonin: 0.8,
                harmony_activations: [0.3, 0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.9],
                ..Default::default()
            },
        ),
        (
            "focused_creative",
            MusicalState {
                consciousness_level: 0.7,
                arousal: 0.4,
                dopamine: 0.7,
                harmony_activations: [0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.6, 0.2],
                valence: 0.5,
                ..Default::default()
            },
        ),
        (
            "intense_urgent",
            MusicalState {
                consciousness_level: 0.3,
                arousal: 0.9,
                noradrenaline: 0.9,
                dopamine: 0.8,
                harmony_activations: [0.2, 0.1, 0.3, 0.9, 0.1, 0.1, 0.8, 0.0],
                valence: -0.3,
                prediction_error: 0.7,
                ..Default::default()
            },
        ),
        (
            "joyful_expansive",
            MusicalState {
                consciousness_level: 0.85,
                arousal: 0.6,
                dopamine: 0.9,
                serotonin: 0.3,
                harmony_activations: [0.8, 0.9, 0.7, 0.6, 0.8, 0.7, 0.5, 0.3],
                valence: 0.8,
                ..Default::default()
            },
        ),
    ];
    let var_cfg = MuseConfig {
        duration_secs: 6.0,
        max_notes: 20,
        ..Default::default()
    };
    for (name, state) in &states {
        let comp = compose(&var_cfg, state, 42);
        let path = dir.join(format!("consciousness_{name}.wav"));
        write_wav(&path, &comp.audio, comp.sample_rate);
        println!("   consciousness_{name}.wav ({} notes)", comp.notes.len());
    }
    println!();

    // ── Streaming Pipeline Demo ──
    println!("3. Streaming Pipeline (100 chunks, full pipeline)");
    {
        use symthaea_muse::streaming::StreamingSynth;
        let mut synth = StreamingSynth::new(
            MuseConfig {
                duration_secs: 10.0,
                max_notes: 32,
                ..Default::default()
            },
            44100,
        );
        synth.update_state(&rich_state);
        synth.feedback_strength = 0.5; // enable strange loop

        let mut all_samples: Vec<[f32; 2]> = Vec::new();
        for _ in 0..200 {
            let chunk = synth.render_chunk();
            all_samples.extend_from_slice(&chunk);
        }

        let audio = AudioData::StereoF32(all_samples);
        let path = dir.join("streaming_full_pipeline.wav");
        write_wav(&path, &audio, 44100);
        println!(
            "   streaming_full_pipeline.wav ({} samples, {:.1}s)",
            audio.len(),
            audio.len() as f32 / 44100.0
        );
        let features = synth.feedback_features();
        println!(
            "   Feedback: centroid={:.3}, flux={:.3}, entropy={:.3}",
            features.spectral_centroid, features.spectral_flux, features.rhythm_entropy
        );
    }
    println!();

    println!("Done! Files in audio_output/");
    println!("Compare with: aplay audio_output/A_old_mono_4partial.wav");
    println!("         vs:  aplay audio_output/B_new_stereo_8partial.wav");
}

/// Write AudioData to WAV (manual RIFF/WAVE, no hound dependency).
fn write_wav(path: &std::path::Path, audio: &AudioData, sample_rate: u32) {
    use std::io::Write;
    let (channels, bits, data): (u16, u16, Vec<u8>) = match audio {
        AudioData::I16(samples) => {
            let mut d = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                d.extend_from_slice(&s.to_le_bytes());
            }
            (1, 16, d)
        }
        AudioData::F32(samples) => {
            let mut d = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                d.extend_from_slice(&i.to_le_bytes());
            }
            (1, 16, d)
        }
        AudioData::StereoF32(samples) => {
            let mut d = Vec::with_capacity(samples.len() * 4);
            for s in samples {
                let l = (s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16;
                let r = (s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16;
                d.extend_from_slice(&l.to_le_bytes());
                d.extend_from_slice(&r.to_le_bytes());
            }
            (2, 16, d)
        }
    };
    let byte_rate = sample_rate * channels as u32 * bits as u32 / 8;
    let block_align = channels * bits / 8;
    let data_len = data.len() as u32;
    let file_len = 36 + data_len;

    let mut f = std::fs::File::create(path).expect("create file");
    f.write_all(b"RIFF").ok();
    f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok(); // fmt chunk size
    f.write_all(&1u16.to_le_bytes()).ok(); // PCM
    f.write_all(&channels.to_le_bytes()).ok();
    f.write_all(&sample_rate.to_le_bytes()).ok();
    f.write_all(&byte_rate.to_le_bytes()).ok();
    f.write_all(&block_align.to_le_bytes()).ok();
    f.write_all(&bits.to_le_bytes()).ok();
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    f.write_all(&data).ok();
}