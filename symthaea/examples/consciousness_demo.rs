// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Demo: A 3-minute composition through 4 consciousness states.
//!
//! Demonstrates the full pipeline: FEP active inference, temporal hierarchy,
//! percussion, mixing chain, binaural rendering, consciousness reverb,
//! sidechain ducking, wavetable morphing, and audio feedback loop.
//!
//! The piece evolves through 4 phases mirroring the section arc:
//! 1. Awakening (30s): Low consciousness, sparse, emerging awareness
//! 2. Integration (45s): Rising Phi, richer harmonies, building complexity
//! 3. Flourishing (60s): Peak consciousness, full polyphony, spatial depth
//! 4. Stillness (45s): Sacred stillness, contemplative, fading to silence
//!
//! Run: cargo run --features muse,humanoid --example consciousness_demo

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::substrate_timbre::SubstrateTimbreType;
use symthaea_muse::{AudioData, MuseConfig, MusicalState};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Consciousness Demo: A Journey Through Awareness        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 44100u32;
    let chunks_per_second = (sample_rate as f32 / (32.0 / 1000.0 * sample_rate as f32)) as usize;

    let mut synth = StreamingSynth::new(
        MuseConfig {
            duration_secs: 300.0, // long session
            max_notes: 32,
            ..Default::default()
        },
        sample_rate,
    );

    // Enable full pipeline
    synth.enable_binaural = true;
    synth.enable_sidechain = true;
    synth.enable_fep = true;
    synth.enable_phi_optimizer = false; // let the composition control Phi
    synth.feedback_strength = 0.4;
    synth.set_substrate(SubstrateTimbreType::Biological); // warm, organic

    let mut all_samples: Vec<[f32; 2]> = Vec::new();
    let total_duration_secs = 180.0; // 3 minutes
    let total_chunks = (total_duration_secs * chunks_per_second as f32) as usize;

    println!("Rendering {} chunks ({:.0}s)...\n", total_chunks, total_duration_secs);

    for chunk_idx in 0..total_chunks {
        let progress = chunk_idx as f32 / total_chunks as f32;

        // ── Phase-based consciousness state ──
        let state = if progress < 0.167 {
            // Phase 1: Awakening (0-30s)
            let p = progress / 0.167;
            println_phase(chunk_idx, "Awakening", p);
            MusicalState {
                consciousness_level: 0.15 + p * 0.25,
                arousal: 0.1 + p * 0.15,
                dopamine: 0.2 + p * 0.1,
                serotonin: 0.6,
                noradrenaline: 0.1,
                harmony_activations: [
                    0.2 + p * 0.2, 0.1, 0.1, 0.05,
                    0.1, 0.1, 0.05, 0.6 - p * 0.3, // stillness fading
                ],
                valence: -0.1 + p * 0.2,
                prediction_error: 0.1 + p * 0.1,
            }
        } else if progress < 0.417 {
            // Phase 2: Integration (30-75s)
            let p = (progress - 0.167) / 0.25;
            println_phase(chunk_idx, "Integration", p);
            MusicalState {
                consciousness_level: 0.4 + p * 0.3,
                arousal: 0.25 + p * 0.25,
                dopamine: 0.3 + p * 0.3,
                serotonin: 0.5 - p * 0.1,
                noradrenaline: 0.1 + p * 0.2,
                harmony_activations: [
                    0.4 + p * 0.2, 0.3 + p * 0.3, 0.3 + p * 0.3, 0.1 + p * 0.2,
                    0.2 + p * 0.2, 0.3 + p * 0.2, 0.3 + p * 0.3, 0.3 - p * 0.1,
                ],
                valence: 0.1 + p * 0.3,
                prediction_error: 0.2 + p * 0.15,
            }
        } else if progress < 0.75 {
            // Phase 3: Flourishing (75-135s)
            let p = (progress - 0.417) / 0.333;
            println_phase(chunk_idx, "Flourishing", p);
            MusicalState {
                consciousness_level: 0.7 + p * 0.2,
                arousal: 0.5 + p * 0.2,
                dopamine: 0.6 + p * 0.25,
                serotonin: 0.4,
                noradrenaline: 0.3 + p * 0.15,
                harmony_activations: [
                    0.6 + p * 0.2, 0.6 + p * 0.2, 0.6 + p * 0.15, 0.3 + p * 0.3,
                    0.4 + p * 0.2, 0.5 + p * 0.2, 0.6 + p * 0.2, 0.2,
                ],
                valence: 0.4 + p * 0.35,
                prediction_error: 0.35 - p * 0.1,
            }
        } else {
            // Phase 4: Stillness (135-180s)
            let p = (progress - 0.75) / 0.25;
            println_phase(chunk_idx, "Stillness", p);
            MusicalState {
                consciousness_level: 0.9 - p * 0.4,
                arousal: 0.7 - p * 0.6,
                dopamine: 0.85 - p * 0.6,
                serotonin: 0.4 + p * 0.4,
                noradrenaline: 0.45 - p * 0.35,
                harmony_activations: [
                    0.8 - p * 0.4, 0.8 - p * 0.5, 0.75 - p * 0.4, 0.6 - p * 0.5,
                    0.6 - p * 0.4, 0.7 - p * 0.4, 0.8 - p * 0.6, 0.2 + p * 0.7, // stillness rising
                ],
                valence: 0.75 - p * 0.5,
                prediction_error: 0.25 - p * 0.2,
            }
        };

        synth.update_state(&state);
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);
    }

    // Write WAV
    let path = dir.join("consciousness_demo_3min.wav");
    let audio = AudioData::StereoF32(all_samples.clone());
    write_wav(&path, &audio, sample_rate);

    let duration = all_samples.len() as f32 / sample_rate as f32;
    let fe_history = synth.free_energy_history();
    let early_fe = if fe_history.len() > 20 { fe_history[..20].iter().sum::<f64>() / 20.0 } else { 0.0 };
    let late_fe = if fe_history.len() > 20 { fe_history[fe_history.len()-20..].iter().sum::<f64>() / 20.0 } else { 0.0 };

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Composition complete                                      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Duration:  {:.1}s ({:.1} minutes)                          ║", duration, duration / 60.0);
    println!("║  Samples:   {}                                     ║", all_samples.len());
    println!("║  FEP cycles: {}                                         ║", synth.fep_cycles());
    println!("║  FE learning: {:.4} → {:.4} (Δ={:.4})                  ║", early_fe, late_fe, early_fe - late_fe);
    println!("║  Output:    {}          ║", path.display());
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nPlay: aplay {}", path.display());
}

fn println_phase(chunk: usize, name: &str, progress: f32) {
    if chunk % 500 == 0 {
        println!("  [{name}] {:.0}% — chunk {chunk}", progress * 100.0);
    }
}

fn write_wav(path: &std::path::Path, audio: &AudioData, sample_rate: u32) {
    use std::io::Write;
    let stereo = match audio {
        AudioData::StereoF32(s) => s,
        _ => return,
    };
    let data_len = (stereo.len() * 4) as u32; // 2ch × 16bit = 4 bytes/frame
    let file_len = 36 + data_len;

    let mut f = std::fs::File::create(path).expect("create file");
    f.write_all(b"RIFF").ok();
    f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok(); // PCM
    f.write_all(&2u16.to_le_bytes()).ok(); // stereo
    f.write_all(&sample_rate.to_le_bytes()).ok();
    f.write_all(&(sample_rate * 4).to_le_bytes()).ok(); // byte rate
    f.write_all(&4u16.to_le_bytes()).ok(); // block align
    f.write_all(&16u16.to_le_bytes()).ok(); // bits
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    for s in stereo {
        let l = (s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        let r = (s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&l.to_le_bytes()).ok();
        f.write_all(&r.to_le_bytes()).ok();
    }
}
