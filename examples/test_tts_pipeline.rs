// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TTS Pipeline Integration Test
//!
//! Tests the complete text-to-speech pipeline:
//! Text → G2P (espeak-ng + Misaki vocab) → Kokoro ONNX → Audio samples
//!
//! Run with:
//!   cargo run --example test_tts_pipeline --features voice-tts
//!
//! Requires: espeak-ng installed on system

use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Symthaea TTS Pipeline Integration Test ===\n");

    // Test 1: G2P Converter
    println!("1. Testing G2P Converter...");
    test_g2p();

    // Test 2: Kokoro Engine Loading (if feature enabled)
    println!("\n2. Testing Kokoro Engine...");
    test_kokoro_engine();

    // Test 3: Full VoiceOutput pipeline
    println!("\n3. Testing VoiceOutput Pipeline...");
    test_voice_output();

    println!("\n=== All TTS pipeline tests complete ===");
    Ok(())
}

fn test_g2p() {
    use symthaea::voice::G2PConverter;

    let g2p = G2PConverter::new();

    // Test basic words
    let test_phrases = [
        "hello",
        "world",
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Consciousness emerges from complex dynamics.",
        "How are you today?",
    ];

    for phrase in &test_phrases {
        let phonemes = g2p.text_to_phonemes(phrase);
        let phoneme_count = phonemes.len();

        // Check if we got phonemes
        if phonemes.is_empty() {
            println!("  [FAIL] '{}' -> no phonemes generated", phrase);
        } else {
            // Check for variety (not all the same ID)
            let unique_ids: std::collections::HashSet<_> = phonemes.iter().copied().collect();
            if unique_ids.len() > 1 {
                println!(
                    "  [OK] '{}' -> {} phonemes, {} unique IDs",
                    phrase,
                    phoneme_count,
                    unique_ids.len()
                );
            } else {
                println!(
                    "  [WARN] '{}' -> {} phonemes but only {} unique (fallback mode?)",
                    phrase,
                    phoneme_count,
                    unique_ids.len()
                );
            }
        }
    }

    // Check espeak-ng availability
    #[cfg(feature = "voice-tts")]
    {
        if g2p.has_espeak() {
            println!("  [INFO] espeak-ng is available for high-quality phoneme conversion");
        } else {
            println!("  [INFO] espeak-ng not available, using fallback character mapping");
        }
    }
}

fn test_kokoro_engine() {
    use symthaea::voice::{KokoroConfig, KokoroEngine};

    println!("  Attempting to load Kokoro TTS model...");

    let config = KokoroConfig::default();
    println!("  Model repo: {}", config.repo_id);
    println!("  Model file: {}", config.model_filename);

    match KokoroEngine::load(config) {
        Some(engine) => {
            println!("  [OK] Kokoro engine loaded successfully");
            println!("  Sample rate: {} Hz", engine.sample_rate());
            println!("  Available voices: {}", engine.num_voices());

            // Try synthesizing a short phrase
            println!("\n  Attempting synthesis...");
            match engine.synthesize("Hello, this is a test.", None) {
                Some(samples) => {
                    let duration_s = samples.len() as f32 / engine.sample_rate() as f32;
                    println!(
                        "  [OK] Synthesis successful: {} samples ({:.2}s)",
                        samples.len(),
                        duration_s
                    );

                    // Check audio quality indicators
                    let max_amplitude = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let mean_amplitude =
                        samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32;
                    println!("  Max amplitude: {:.3}", max_amplitude);
                    println!("  Mean amplitude: {:.3}", mean_amplitude);

                    // Save to file for manual listening
                    let path = "/tmp/symthaea_tts_test.wav";
                    match symthaea::voice::save_wav(&samples, engine.sample_rate(), path) {
                        Ok(()) => println!("  [OK] Saved audio to {}", path),
                        Err(e) => println!("  [WARN] Failed to save WAV: {}", e),
                    }
                }
                None => {
                    println!("  [FAIL] Synthesis returned no audio");
                }
            }
        }
        None => {
            println!("  [INFO] Kokoro engine not available (model download may have failed)");
            println!("  This is normal if running without network access or in CI");
        }
    }
}

fn test_voice_output() {
    use symthaea::voice::{LTCPacing, VoiceOutput, VoiceOutputConfig};

    let config = VoiceOutputConfig {
        enable_tts: true,
        ..VoiceOutputConfig::default()
    };

    let mut voice = VoiceOutput::new(config);

    // Initialize (loads Kokoro if available)
    match voice.initialize() {
        Ok(()) => println!("  [OK] VoiceOutput initialized"),
        Err(e) => println!("  [WARN] VoiceOutput initialization error: {}", e),
    }

    // Test synthesize with default pacing
    let test_text = "The consciousness field resonates with harmonic frequencies.";
    match voice.synthesize(test_text) {
        Ok(samples) => {
            let duration_s = samples.len() as f32 / voice.config().sample_rate as f32;
            println!(
                "  [OK] Default synthesis: {} samples ({:.2}s)",
                samples.len(),
                duration_s
            );
        }
        Err(e) => println!("  [FAIL] Synthesis failed: {}", e),
    }

    // Test with calm pacing
    let calm_pacing = LTCPacing::calm();
    match voice.synthesize_with_pacing(test_text, &calm_pacing) {
        Ok(samples) => {
            let duration_s = samples.len() as f32 / voice.config().sample_rate as f32;
            println!(
                "  [OK] Calm pacing synthesis: {} samples ({:.2}s)",
                samples.len(),
                duration_s
            );
        }
        Err(e) => println!("  [FAIL] Calm synthesis failed: {}", e),
    }

    // Test with excited pacing
    let excited_pacing = LTCPacing::excited();
    match voice.synthesize_with_pacing(test_text, &excited_pacing) {
        Ok(samples) => {
            let duration_s = samples.len() as f32 / voice.config().sample_rate as f32;
            println!(
                "  [OK] Excited pacing synthesis: {} samples ({:.2}s)",
                samples.len(),
                duration_s
            );
        }
        Err(e) => println!("  [FAIL] Excited synthesis failed: {}", e),
    }

    // Print stats
    let stats = voice.stats();
    println!("\n  Voice Statistics:");
    println!("    Total utterances: {}", stats.total_utterances);
    println!("    Total characters: {}", stats.total_chars);
    println!(
        "    Avg synthesis time: {:.2} ms",
        stats.avg_synthesis_time_ms
    );
    println!(
        "    Total audio generated: {:.2} s",
        stats.total_audio_seconds
    );
}
