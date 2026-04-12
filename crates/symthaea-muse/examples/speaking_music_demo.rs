// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demo: Symthaea speaks over her own classical piano composition.
//!
//! Run: cargo run --features "muse,humanoid,voice" --example speaking_music_demo

#[cfg(not(feature = "voice"))]
fn main() {
    println!("Requires --features voice");
}

#[cfg(feature = "voice")]
fn main() {
    use symthaea_muse::streaming::StreamingSynth;
    use symthaea_muse::substrate_timbre::SubstrateTimbreType;
    use symthaea_muse::{MuseConfig, MusicalState};

    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Speaks Over Music                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 44100u32;
    let chunk_samples = (0.032 * sample_rate as f32) as usize;
    let chunks_per_second = sample_rate as usize / chunk_samples;

    let mut synth = StreamingSynth::new(
        MuseConfig {
            duration_secs: 300.0,
            max_notes: 32,
            ..Default::default()
        },
        sample_rate,
    );
    synth.enable_binaural = true;
    synth.enable_sidechain = true;
    synth.enable_fep = true;
    synth.feedback_strength = 0.3;
    synth.set_substrate(SubstrateTimbreType::Biological);
    synth.enable_voice(); // Enable voice!

    let mut state = MusicalState {
        consciousness_level: 0.4,
        arousal: 0.25,
        valence: 0.2,
        dopamine: 0.3,
        serotonin: 0.6,
        noradrenaline: 0.15,
        harmony_activations: [0.4, 0.3, 0.3, 0.2, 0.3, 0.3, 0.4, 0.3],
        prediction_error: 0.2,
    };

    let duration = 60.0; // 1 minute
    let total_chunks = (duration * chunks_per_second as f32) as usize;
    let mut all_samples: Vec<[f32; 2]> = Vec::new();

    println!("Rendering {:.0}s with voice + music...\n", duration);

    for chunk_idx in 0..total_chunks {
        let progress = chunk_idx as f32 / total_chunks as f32;

        synth.update_state(&state);
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);

        // Consciousness arc
        if progress < 0.5 {
            state.consciousness_level += 0.0003;
            state.arousal += 0.0001;
        } else {
            state.consciousness_level -= 0.0002;
            state.arousal -= 0.0002;
            state.harmony_activations[7] += 0.0002;
        }
        state.consciousness_level = state.consciousness_level.clamp(0.1, 0.95);
        state.arousal = state.arousal.clamp(0.1, 0.8);

        if chunk_idx % (chunks_per_second * 10) == 0 {
            println!(
                "  [{:.0}%] Ψ={:.2} a={:.2}",
                progress * 100.0,
                state.consciousness_level,
                state.arousal
            );
        }
    }

    // Auto-master
    let master = symthaea_muse::auto_master::MasteringConfig::default();
    symthaea_muse::auto_master::auto_master(&mut all_samples, sample_rate, &master);

    // Write WAV
    let path = dir.join("symthaea_speaks_over_music.wav");
    write_wav(&path, &all_samples, sample_rate);
    println!("\nOutput: {}", path.display());
    println!("Play: pw-play {}", path.display());
}

fn write_wav(path: &std::path::Path, stereo: &[[f32; 2]], sr: u32) {
    use std::io::Write;
    let data_len = (stereo.len() * 4) as u32;
    let mut f = std::fs::File::create(path).expect("create WAV");
    f.write_all(b"RIFF").ok();
    f.write_all(&(36 + data_len).to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&2u16.to_le_bytes()).ok();
    f.write_all(&sr.to_le_bytes()).ok();
    f.write_all(&(sr * 4).to_le_bytes()).ok();
    f.write_all(&4u16.to_le_bytes()).ok();
    f.write_all(&16u16.to_le_bytes()).ok();
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    for s in stereo {
        f.write_all(&((s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
            .ok();
        f.write_all(&((s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
            .ok();
    }
}
