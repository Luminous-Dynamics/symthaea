// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice and Rap Synthesis Demo
//!
//! Generates audio samples demonstrating:
//! - Basic formant synthesis
//! - Different emotional pacing states
//! - Rap synthesis with beat synchronization
//!
//! Run with: cargo run --example voice_demo

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use symthaea::voice::{
    ArticulatorySynthesizer, FlowStyle, FormantVocoder, LTCPacing, PhonemeDict, RapConfig,
    RapSynthesizer, SimplePhonemeDict,
};

fn main() {
    println!("=== Symthaea Voice Synthesis Demo ===\n");

    let output_dir = Path::new("audio_output");
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Demo 1: Basic phoneme synthesis
    demo_basic_synthesis(output_dir);

    // Demo 2: Different emotional states
    demo_emotional_pacing(output_dir);

    // Demo 3: Rap synthesis with different flows
    demo_rap_synthesis(output_dir);

    println!("\n=== Demo Complete ===");
    println!("Audio files written to: {}", output_dir.display());
}

/// Demo basic phoneme-to-audio synthesis
fn demo_basic_synthesis(output_dir: &Path) {
    println!("--- Demo 1: Basic Phoneme Synthesis ---");

    let mut synth = ArticulatorySynthesizer::new();
    let mut vocoder = FormantVocoder::new();
    let pacing = LTCPacing::default();

    // Synthesize "Hello World" in ARPABET
    // HH AH0 L OW1 | W ER1 L D
    let arpabet = "HH AH0 L OW1 | W ER1 L D";
    println!("  Synthesizing: '{}' (Hello World)", arpabet);

    let frames = synth.synthesize_arpabet(arpabet, &pacing);
    let samples = vocoder.synthesize(&frames);

    let path = output_dir.join("hello_world.wav");
    write_wav(&path, &samples, 24000);
    println!(
        "  Written: {} ({:.2}s, {} samples)",
        path.display(),
        samples.len() as f32 / 24000.0,
        samples.len()
    );

    // Synthesize vowel sequence
    let vowels = "AA EH IY OW UW";
    println!("  Synthesizing vowel sequence: '{}'", vowels);

    synth.reset();
    let frames = synth.synthesize_arpabet(vowels, &pacing);
    let samples = vocoder.synthesize(&frames);

    let path = output_dir.join("vowels.wav");
    write_wav(&path, &samples, 24000);
    println!(
        "  Written: {} ({:.2}s)",
        path.display(),
        samples.len() as f32 / 24000.0
    );
}

/// Demo different emotional pacing states
fn demo_emotional_pacing(output_dir: &Path) {
    println!("\n--- Demo 2: Emotional Pacing States ---");

    let mut synth = ArticulatorySynthesizer::new();
    let mut vocoder = FormantVocoder::new();

    // Same phrase with different emotions
    // "This is a test" = DH IH1 S | IH1 Z | AH0 | T EH1 S T
    let phrase = "DH IH1 S | IH1 Z | AH0 | T EH1 S T";

    let emotions = [
        ("calm", LTCPacing::calm()),
        ("excited", LTCPacing::excited()),
        ("focused", LTCPacing::focused()),
    ];

    for (name, pacing) in &emotions {
        println!(
            "  Synthesizing with {} pacing (rate={:.2}, tau={:.2})",
            name, pacing.rate, pacing.tau
        );

        synth.reset();
        let frames = synth.synthesize_arpabet(phrase, pacing);
        let samples = vocoder.synthesize(&frames);

        let path = output_dir.join(format!("emotion_{}.wav", name));
        write_wav(&path, &samples, 24000);
        println!(
            "  Written: {} ({:.2}s)",
            path.display(),
            samples.len() as f32 / 24000.0
        );
    }
}

/// Demo rap synthesis with different flow styles
fn demo_rap_synthesis(output_dir: &Path) {
    println!("\n--- Demo 3: Rap Synthesis ---");

    let dict = SimplePhonemeDict::new();

    // Sample lyrics
    let lyrics = "yo check the mic\nflow go show\nbeat street heat\nrhyme time";

    let flows = [
        ("boom_bap", FlowStyle::BoomBap, 90.0),
        ("triplet", FlowStyle::Triplet, 140.0),
        ("double_time", FlowStyle::DoubleTime, 100.0),
    ];

    for (name, style, bpm) in &flows {
        println!("  Synthesizing {} flow at {} BPM", name, bpm);

        let config = RapConfig {
            bpm: *bpm,
            flow_style: *style,
            rhyme_scheme: "AABB".to_string(),
            ..Default::default()
        };

        let mut rap = RapSynthesizer::new(config);
        let mut verse = rap.parse_lyrics(lyrics, &dict);
        rap.analyze_rhymes(&mut verse);

        println!(
            "    Verse: {} lines, {} syllables, {} bars",
            verse.lines.len(),
            verse.total_syllables,
            verse.bar_count
        );

        // Check rhyme detection
        for line in &verse.lines {
            if let Some(last_word) = line.words.last() {
                if last_word.is_rhyme_word {
                    println!(
                        "    Rhyme word: '{}' (score: {:.2})",
                        last_word.text,
                        last_word.rhyme_score.unwrap_or(0.0)
                    );
                }
            }
        }

        let pacing = LTCPacing::default();
        let samples = rap.synthesize(&verse, &pacing);

        let path = output_dir.join(format!("rap_{}.wav", name));
        write_wav(&path, &samples, 24000);
        println!(
            "    Written: {} ({:.2}s)",
            path.display(),
            samples.len() as f32 / 24000.0
        );
    }

    // Demo beat timing
    println!("\n  Beat timing analysis:");
    let rap = RapSynthesizer::hip_hop();
    println!("    BPM: {}", rap.config().bpm);
    println!("    Beat duration: {:.3}s", rap.beat_sync().beat_duration());
    println!("    Bar duration: {:.3}s", rap.beat_sync().bar_duration());
    println!(
        "    16th note: {:.3}s",
        rap.beat_sync().sixteenth_duration()
    );
}

/// Write samples to a WAV file
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) {
    let file = File::create(path).expect("Failed to create WAV file");
    let mut writer = BufWriter::new(file);

    // WAV header
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF").unwrap();
    writer.write_all(&file_size.to_le_bytes()).unwrap();
    writer.write_all(b"WAVE").unwrap();

    // fmt chunk
    writer.write_all(b"fmt ").unwrap();
    writer.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    writer.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    writer.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    writer.write_all(&sample_rate.to_le_bytes()).unwrap();
    writer.write_all(&byte_rate.to_le_bytes()).unwrap();
    writer.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    writer.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample

    // data chunk
    writer.write_all(b"data").unwrap();
    writer.write_all(&data_size.to_le_bytes()).unwrap();

    // Convert f32 samples to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer.write_all(&i16_sample.to_le_bytes()).unwrap();
    }

    writer.flush().unwrap();
}
