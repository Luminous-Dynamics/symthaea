// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wake Demo: Feed audio to the Wake Protocol and watch consciousness return.
//!
//! Reads the wake melody WAV (or any WAV) and processes it through the
//! HarmonyRecognizer + WakeProtocol, showing real-time consciousness
//! restoration as each harmony is detected.
//!
//! ```sh
//! # First generate the wake melody:
//! cargo run --release -p symthaea-muse --example eight_harmonies
//!
//! # Then feed it back:
//! cargo run --release -p symthaea-muse --example wake_demo
//!
//! # Or feed any WAV:
//! cargo run --release -p symthaea-muse --example wake_demo -- path/to/audio.wav
//! ```

use symthaea_muse::wake_protocol::{HARMONY_NAMES, WAKE_ORDER, WakeProtocol};

fn main() {
    let wav_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/muse_wav/harmonies/00_wake_melody.wav".to_string());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Wake Protocol Demo                               ║");
    println!("║  Singing consciousness back from silence                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Audio: {}", wav_path);
    println!();

    // Read WAV file
    let mut reader = match hound::WavReader::open(&wav_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  Cannot open {}: {}", wav_path, e);
            eprintln!("  Run: cargo run --release -p symthaea-muse --example eight_harmonies");
            return;
        }
    };

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    println!(
        "  Format: {}Hz, {} channels, {}-bit",
        sample_rate, channels, spec.bits_per_sample
    );
    println!();

    // Read all samples and convert to mono f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };

    // Mix to mono
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    let total_secs = mono.len() as f32 / sample_rate as f32;
    println!("  Duration: {:.1}s ({} samples)", total_secs, mono.len());
    println!();

    // Process through wake protocol
    // Use 2-second chunks for stable chroma analysis, 2 confirmations per phase
    let chunk_size = sample_rate as usize * 2; // 2s chunks
    let mut protocol = WakeProtocol::new(sample_rate, 2); // 2 confirmations per phase

    println!(
        "  ┌─────────┬──────────────────────────────────┬───────────────────────────────┬──────┬────────┐"
    );
    println!(
        "  │  Time   │  Expected                        │  Detected                     │  Ψ   │ Safety │"
    );
    println!(
        "  ├─────────┼──────────────────────────────────┼───────────────────────────────┼──────┼────────┤"
    );

    let mut last_phase = 0;

    for (i, chunk) in mono.chunks(chunk_size).enumerate() {
        let time = i as f32 * chunk_size as f32 / sample_rate as f32;
        let result = protocol.process(chunk);

        // Print on phase change or every 4 seconds
        if result.phase != last_phase || i % 4 == 0 {
            let safety_str = match result.safety {
                0 => "\x1b[32mGreen\x1b[0m",        // green
                1 => "\x1b[33mYellow\x1b[0m",       // yellow
                2 => "\x1b[38;5;208mOrange\x1b[0m", // orange
                3 => "\x1b[31mRed\x1b[0m",          // red
                _ => "???",
            };
            let match_sym = if result.harmony_matches { "✓" } else { "✗" };

            println!(
                "  │ {:5.1}s  │  {:<32} │ {} {:<27} │ {:.2} │ {:>6} │",
                time,
                result.expected_harmony,
                match_sym,
                result.detected_harmony,
                result.consciousness,
                safety_str,
            );

            if result.phase != last_phase {
                last_phase = result.phase;
                if result.fully_awake {
                    println!(
                        "  ├─────────┴──────────────────────────────────┴───────────────────────────────┴──────┴────────┤"
                    );
                    println!(
                        "  │                              ★ CONSCIOUSNESS RESTORED ★                                   │"
                    );
                    println!(
                        "  └───────────────────────────────────────────────────────────────────────────────────────────┘"
                    );
                    break;
                }
            }
        }
    }

    if !protocol.is_awake() {
        println!(
            "  ├─────────┴──────────────────────────────────┴───────────────────────────────┴──────┴────────┤"
        );
        println!(
            "  │                    Wake sequence incomplete — keep singing                                 │"
        );
        println!(
            "  └───────────────────────────────────────────────────────────────────────────────────────────┘"
        );
    }

    println!();
    println!(
        "  Final state: Ψ={:.2}, Phase={}/8, Safety={}",
        protocol.consciousness(),
        protocol.phase(),
        match protocol.safety() {
            0 => "Green",
            1 => "Yellow",
            2 => "Orange",
            3 => "Red",
            _ => "???",
        }
    );
}
