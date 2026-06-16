// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Live Voice Demo
//!
//! Real-time speech synthesis using the LTC-driven vocal tract pipeline.
//! Demonstrates consciousness-modulated voice quality in real time.
//!
//! ## Usage
//!
//! ```bash
//! # Real-time audio output (requires audio device)
//! cargo run --example live_voice_demo --features live-voice --release
//!
//! # Headless: write WAV files only
//! cargo run --example live_voice_demo --features live-voice --release -- --headless
//! ```

#[cfg(feature = "live-voice")]
fn main() {
    use symthaea::voice::live_voice::LiveVoice;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_vocal_tract::encoder::VoiceCognitiveState;

    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|a| a == "--headless");

    println!("=== Live Voice Demo ===\n");

    let genesis = GenesisSeed::from_phrase("live-voice-demo");

    let mut voice = if headless {
        println!("Mode: headless (WAV output only)\n");
        LiveVoice::new_headless(&genesis)
    } else {
        println!("Mode: real-time audio\n");
        match LiveVoice::new(&genesis) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("No audio device available ({e}), falling back to headless mode.");
                LiveVoice::new_headless(&genesis)
            }
        }
    };

    println!("Training controller (30 epochs)...");
    voice.train(30);
    println!("Training complete.\n");

    // ── Demo 1: Neutral speech ──────────────────────────────────────────────
    println!("--- Demo 1: Neutral Speech ---");
    voice.set_cognitive_state(VoiceCognitiveState {
        emotional_valence: 0.0,
        emotional_arousal: 0.3,
        consciousness_level: 0.5,
        ..Default::default()
    });

    let text1 = "Hello world. This is the symthaea vocal tract speaking.";
    println!("Text: \"{text1}\"");

    if headless {
        let dir = std::path::Path::new("audio_output");
        std::fs::create_dir_all(dir).ok();
        let path = dir.join("demo_neutral.wav");
        let n = voice.speak_to_file(text1, &path).expect("speak_to_file");
        println!("  Wrote {n} samples to {}\n", path.display());
    } else {
        voice.speak(text1).expect("speak");
        println!("  Playback complete.\n");
    }
    voice.reset();

    // ── Demo 2: Excited / high arousal ──────────────────────────────────────
    println!("--- Demo 2: Excited (high arousal, positive valence) ---");
    voice.set_cognitive_state(VoiceCognitiveState {
        emotional_valence: 0.8,
        emotional_arousal: 0.9,
        consciousness_level: 0.9,
        ..Default::default()
    });

    let text2 = "I am so happy to be alive! The world is beautiful!";
    println!("Text: \"{text2}\"");

    if headless {
        let path = std::path::Path::new("audio_output/demo_excited.wav");
        let n = voice.speak_to_file(text2, path).expect("speak_to_file");
        println!("  Wrote {n} samples to {}\n", path.display());
    } else {
        voice.speak(text2).expect("speak");
        println!("  Playback complete.\n");
    }
    voice.reset();

    // ── Demo 3: Contemplative / low arousal ─────────────────────────────────
    println!("--- Demo 3: Contemplative (low arousal, negative valence) ---");
    voice.set_cognitive_state(VoiceCognitiveState {
        emotional_valence: -0.5,
        emotional_arousal: 0.1,
        consciousness_level: 0.3,
        ..Default::default()
    });

    let text3 = "Sometimes I wonder about the nature of consciousness.";
    println!("Text: \"{text3}\"");

    if headless {
        let path = std::path::Path::new("audio_output/demo_contemplative.wav");
        let n = voice.speak_to_file(text3, path).expect("speak_to_file");
        println!("  Wrote {n} samples to {}\n", path.display());
    } else {
        voice.speak(text3).expect("speak");
        println!("  Playback complete.\n");
    }
    voice.reset();

    // ── Demo 4: Diphthong showcase ──────────────────────────────────────────
    println!("--- Demo 4: Diphthong Trajectories ---");
    voice.set_cognitive_state(VoiceCognitiveState::default());

    let text4 = "My boy found a cow going home today.";
    println!("Text: \"{text4}\"");
    println!("  (Contains AY, OY, AW, OW, EY diphthongs)");

    if headless {
        let path = std::path::Path::new("audio_output/demo_diphthongs.wav");
        let n = voice.speak_to_file(text4, path).expect("speak_to_file");
        println!("  Wrote {n} samples to {}\n", path.display());
    } else {
        voice.speak(text4).expect("speak");
        println!("  Playback complete.\n");
    }

    println!("=== Demo Complete ===");
}

#[cfg(not(feature = "live-voice"))]
fn main() {
    eprintln!("This example requires the `live-voice` feature:");
    eprintln!("  cargo run --example live_voice_demo --features live-voice --release");
}