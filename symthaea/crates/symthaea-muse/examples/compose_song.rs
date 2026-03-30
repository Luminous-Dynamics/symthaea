// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compose a novel song in a chosen genre and export as MIDI + WAV.
//!
//! Usage:
//!   cargo run -p symthaea-muse --example compose_song
//!   cargo run -p symthaea-muse --example compose_song -- jazz
//!   cargo run -p symthaea-muse --example compose_song -- prog-metal
//!
//! Outputs: audio_output/song_<genre>.mid + audio_output/song_<genre>.wav

use symthaea_muse::genre_presets::Genre;
use symthaea_muse::midi_export;
use symthaea_muse::{compose, AudioData, MuseConfig, MusicalState};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    let genre = parse_genre();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Composing: {:44}║", genre.name());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let state = genre.seed_state();
    let (tempo_lo, tempo_hi) = genre.tempo_range();
    let tempo = (tempo_lo + tempo_hi) / 2.0;
    let duration = genre.duration_secs();

    println!("  Seed: Ψ={:.2} arousal={:.2} valence={:.2}",
        state.consciousness_level, state.arousal, state.valence);
    println!("  Tempo: {:.0} BPM | Duration: {:.0}s ({:.1} min)\n",
        tempo, duration, duration / 60.0);

    let config = MuseConfig {
        duration_secs: duration,
        max_notes: 64,
        ..Default::default()
    };

    println!("Composing...");
    let comp = compose(&config, &state, 42);
    println!("  Generated {} notes\n", comp.notes.len());

    // Export MIDI
    let midi_path = dir.join(format!("song_{}.mid", slug(genre.name())));
    match midi_export::export_midi(&comp.notes, tempo, 4, 4, &midi_path) {
        Ok(()) => println!("  MIDI: {}", midi_path.display()),
        Err(e) => println!("  MIDI export failed: {e}"),
    }

    // Export WAV
    let wav_path = dir.join(format!("song_{}.wav", slug(genre.name())));
    write_wav(&wav_path, &comp.audio, 44100);
    println!("  WAV:  {}", wav_path.display());

    println!("\nPlay:");
    println!("  aplay {}", wav_path.display());
    println!("\nImport MIDI into DAW for professional sound:");
    println!("  {}", midi_path.display());
}

fn parse_genre() -> Genre {
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
        "all" => {
            println!("Composing all genres...\n");
            for g in Genre::all() {
                println!("── {} ──", g.name());
                let state = g.seed_state();
                let config = MuseConfig {
                    duration_secs: g.duration_secs().min(180.0), // cap at 3 min for speed
                    max_notes: 64,
                    ..Default::default()
                };
                let comp = compose(&config, &state, 42);
                let (lo, hi) = g.tempo_range();
                let tempo = (lo + hi) / 2.0;

                let midi_path = dir.join(format!("song_{}.mid", slug(g.name())));
                midi_export::export_midi(&comp.notes, tempo, 4, 4, &midi_path).ok();

                let wav_path = dir.join(format!("song_{}.wav", slug(g.name())));
                write_wav(&wav_path, &comp.audio, 44100);
                println!("  {} notes → {} + {}", comp.notes.len(), midi_path.display(), wav_path.display());
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

fn dir() -> &'static std::path::Path {
    std::path::Path::new("audio_output")
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
    f.write_all(b"RIFF").ok(); f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok(); f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok(); f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&2u16.to_le_bytes()).ok(); f.write_all(&sample_rate.to_le_bytes()).ok();
    f.write_all(&(sample_rate * 4).to_le_bytes()).ok();
    f.write_all(&4u16.to_le_bytes()).ok(); f.write_all(&16u16.to_le_bytes()).ok();
    f.write_all(b"data").ok(); f.write_all(&data_len.to_le_bytes()).ok();
    for s in stereo {
        let l = (s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        let r = (s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&l.to_le_bytes()).ok(); f.write_all(&r.to_le_bytes()).ok();
    }
}
