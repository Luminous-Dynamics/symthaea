//! Minimal TTS test - tests only voice synthesis
//!
//! Run with: cargo run --release --example tts_minimal_test --features "voice-tts"
//! Or with full audio playback: cargo run --release --example tts_minimal_test --features "audio"

use std::time::Instant;
use std::path::Path;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🎤 TTS Minimal Performance Test");
    println!("================================\n");

    // Test tokenization directly
    let test_phrases = [
        ("Short", "Hello world."),
        ("Medium", "Welcome to Symthaea, the consciousness-first AI system."),
        ("Long", "The quick brown fox jumps over the lazy dog. This is a longer test."),
    ];

    // Test tokenization speed
    println!("📝 Tokenization Tests (Misaki G2P):");
    for (name, text) in &test_phrases {
        let start = Instant::now();
        match symthaea::voice::tokenizer::tokenize(text) {
            Ok(tokens) => {
                let elapsed = start.elapsed();
                let ms = elapsed.as_secs_f64() * 1000.0;
                let status = if ms < 50.0 { "✅ FAST (server)" } else if ms < 500.0 { "⚠️ MODERATE" } else { "❌ SLOW (subprocess?)" };
                println!("   {}: {} tokens in {:.1}ms {}", name, tokens.len(), ms, status);
            }
            Err(e) => {
                println!("   {}: ERROR - {}", name, e);
            }
        }
    }

    println!();

    #[cfg(feature = "voice-tts")]
    {
        use symthaea::voice::{VoiceOutput, VoiceOutputConfig, LTCPacing};

        println!("🔊 Full TTS Synthesis Tests (12-thread CPU):");

        // Initialize voice output
        let config = VoiceOutputConfig::default();
        println!("   Loading Kokoro model...");
        let load_start = Instant::now();
        let mut voice = VoiceOutput::new(config)?;
        println!("   Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());

        // Test synthesis - warmup is the first synthesis
        println!("   Running synthesis tests...\n");

        // Test synthesis
        for (name, text) in &test_phrases {
            let start = Instant::now();
            match voice.synthesize_with_pacing(text, LTCPacing::default()) {
                Ok(audio) => {
                    let gen_time = start.elapsed().as_secs_f64();
                    // Calculate audio duration from samples
                    let audio_duration = audio.samples.len() as f64 / audio.sample_rate as f64;
                    let rtf = gen_time / audio_duration;
                    let status = if rtf < 1.0 { "✅ REAL-TIME!" } else if rtf < 5.0 { "⚠️ ACCEPTABLE" } else { "❌ TOO SLOW" };
                    println!("   {}: Gen {:.2}s | Audio {:.2}s | RTF {:.1}x {}", name, gen_time, audio_duration, rtf, status);

                    // Save to file
                    let filename = format!("/tmp/tts_test_{}.wav", name.to_lowercase());
                    audio.save_wav(Path::new(&filename))?;
                    println!("      Saved to: {}", filename);
                }
                Err(e) => {
                    println!("   {}: ERROR - {}", name, e);
                }
            }
        }
    }

    #[cfg(not(feature = "voice-tts"))]
    {
        println!("⚠️ voice-tts feature not enabled. Enable with --features \"voice-tts\"");
    }

    println!("\n✅ Done!");
    Ok(())
}
