// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compose a novel song using the full StreamingSynth pipeline.
//!
//! Uses genre presets to seed consciousness state, then renders through
//! all 52 modules (learned melody, rhythm engine, creative agency,
//! composer mind, aesthetic listener, etc.). Exports both WAV and MIDI.
//!
//! Usage:
//!   cargo run --release -p symthaea-muse --example compose_song -- jazz
//!   cargo run --release -p symthaea-muse --example compose_song -- prog-metal
//!   cargo run --release -p symthaea-muse --example compose_song -- all

use symthaea_muse::genre_presets::Genre;
use symthaea_muse::midi_export;
use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::substrate_timbre::SubstrateTimbreType;
use symthaea_muse::{AudioData, MuseConfig, MusicalState};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    let genre = parse_genre(dir);

    compose_genre(genre, dir);
}

fn compose_genre(genre: Genre, dir: &std::path::Path) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Composing: {:44}║", genre.name());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 44100u32;
    let chunk_ms = 32.0;
    let chunk_samples = (chunk_ms / 1000.0 * sample_rate as f32) as usize;
    let chunks_per_second = (sample_rate as usize + chunk_samples - 1) / chunk_samples;

    let mut synth = StreamingSynth::new(
        MuseConfig {
            duration_secs: 600.0,
            max_notes: 32,
            ..Default::default()
        },
        sample_rate,
    );

    // Enable full pipeline
    synth.enable_binaural = true;
    synth.enable_sidechain = true;
    synth.enable_fep = true;
    synth.enable_phi_optimizer = false;
    synth.feedback_strength = 0.5;
    synth.set_substrate(SubstrateTimbreType::Biological);

    let mut state = genre.seed_state();
    let duration = genre.duration_secs().min(180.0); // cap for speed
    let total_chunks = (duration * chunks_per_second as f32) as usize;

    let (tempo_lo, tempo_hi) = genre.tempo_range();
    let tempo = (tempo_lo + tempo_hi) / 2.0;

    println!(
        "  Seed: Ψ={:.2} arousal={:.2} valence={:.2}",
        state.consciousness_level, state.arousal, state.valence
    );
    println!(
        "  Tempo: {:.0} BPM | Duration: {:.0}s ({:.1} min)",
        tempo,
        duration,
        duration / 60.0
    );
    println!("  Chunks: {} | Pipeline: all 52 modules\n", total_chunks);

    let mut all_samples: Vec<[f32; 2]> = Vec::new();

    for chunk_idx in 0..total_chunks {
        let progress = chunk_idx as f32 / total_chunks as f32;

        synth.update_state(&state);
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);

        // Natural state evolution — clear arc: build → peak at 60% → fade
        if progress < 0.6 {
            // Build phase: rising energy
            state.consciousness_level += 0.0002;
            state.arousal += 0.0001;
            state.dopamine += 0.00005;
        } else if progress < 0.8 {
            // Sustain/transition: hold then start fading
            state.arousal -= 0.0002;
            state.dopamine -= 0.0001;
        } else {
            // Fade: clear wind-down
            state.consciousness_level -= 0.0004;
            state.arousal -= 0.0003;
            state.dopamine -= 0.0002;
            state.serotonin += 0.0001; // warmer as it fades
            state.harmony_activations[7] += 0.0004; // stillness rising
        }
        state.valence += 0.00002;

        // Clamp
        state.consciousness_level = state.consciousness_level.clamp(0.05, 0.95);
        state.arousal = state.arousal.clamp(0.05, 0.95);
        state.valence = state.valence.clamp(-0.8, 0.8);
        for h in &mut state.harmony_activations {
            *h = h.clamp(0.0, 1.0);
        }

        if chunk_idx % 500 == 0 {
            println!(
                "  [{:.0}%] Ψ={:.2} a={:.2} v={:.2} notes={}",
                progress * 100.0,
                state.consciousness_level,
                state.arousal,
                state.valence,
                synth.generated_notes.len()
            );
        }
    }

    // Auto-master
    println!("\n  Auto-mastering...");
    let master_config = symthaea_muse::auto_master::MasteringConfig::default();
    let master_result =
        symthaea_muse::auto_master::auto_master(&mut all_samples, sample_rate, &master_config);
    println!(
        "  LUFS: {:.1} → {:.1} (gain {:.1} dB)",
        master_result.input_lufs, master_result.output_lufs, master_result.gain_applied_db
    );

    // Export WAV
    let wav_path = dir.join(format!("song_{}.wav", slug(genre.name())));
    let audio = AudioData::StereoF32(all_samples);
    write_wav(&wav_path, &audio, sample_rate);
    println!("  WAV: {}", wav_path.display());

    // Export MIDI from actual generated notes (not estimated from audio)
    let midi_path = dir.join(format!("song_{}.mid", slug(genre.name())));
    let time_sig_num = if state.consciousness_level > 0.7 {
        7
    } else {
        4
    };
    let time_sig_den = if time_sig_num == 7 { 8 } else { 4 };
    let midi_notes = &synth.generated_notes;
    match midi_export::export_midi(midi_notes, tempo, time_sig_num, time_sig_den, &midi_path) {
        Ok(()) => println!(
            "  MIDI: {} ({} notes)",
            midi_path.display(),
            midi_notes.len()
        ),
        Err(e) => println!("  MIDI failed: {e}"),
    }

    println!("\n  Play: aplay {}", wav_path.display());
    println!("  DAW:  {}", midi_path.display());
}

fn parse_genre(dir: &std::path::Path) -> Genre {
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).map(|s| s.as_str()).unwrap_or("jazz");
    match name.to_lowercase().as_str() {
        "prog-metal" | "prog" | "metal" => Genre::ProgMetal,
        "jazz" => Genre::Jazz,
        "atmospheric" | "progressive" | "prog-atmo" => Genre::ProgressiveAtmospheric,
        "funk" | "groove" | "funk-groove" => Genre::FunkGroove,
        "ambient" | "contemplative" => Genre::AmbientContemplative,
        "rock" | "classic-rock" => Genre::ClassicRock,
        "electronic" | "psych" | "psychedelic" => Genre::ElectronicPsych,
        "island" | "reggae" | "island-groove" => Genre::IslandGroove,
        "classical" | "bach" | "beethoven" | "chopin" => Genre::Classical,
        "all" => {
            for g in Genre::all() {
                compose_genre(*g, dir);
                println!();
            }
            std::process::exit(0);
        }
        _ => {
            println!("Unknown genre '{}'. Available:", name);
            for g in Genre::all() {
                println!("  {}", slug(g.name()));
            }
            std::process::exit(1);
        }
    }
}

fn slug(s: &str) -> String {
    s.to_lowercase().replace(' ', "_")
}

fn write_wav(path: &std::path::Path, audio: &AudioData, sample_rate: u32) {
    use std::io::Write;
    let stereo = match audio {
        AudioData::StereoF32(s) => s,
        _ => return,
    };
    let data_len = (stereo.len() * 4) as u32;
    let file_len = 36 + data_len;
    let mut f = std::fs::File::create(path).expect("create WAV");
    f.write_all(b"RIFF").ok();
    f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&2u16.to_le_bytes()).ok();
    f.write_all(&sample_rate.to_le_bytes()).ok();
    f.write_all(&(sample_rate * 4).to_le_bytes()).ok();
    f.write_all(&4u16.to_le_bytes()).ok();
    f.write_all(&16u16.to_le_bytes()).ok();
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    for s in stereo {
        let l = (s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        let r = (s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&l.to_le_bytes()).ok();
        f.write_all(&r.to_le_bytes()).ok();
    }
}
