//! Test TTS audio quality by saving to WAV file
//!
//! Run with: cargo run --example test_tts_audio --features voice

use std::path::Path;
use std::time::Instant;

fn main() {
    #[cfg(feature = "voice-tts")]
    {
        use symthaea::voice::{VoiceOutput, VoiceOutputConfig};

        println!("=== TTS Audio Quality Test ===\n");

        // Initialize TTS
        println!("Initializing Kokoro TTS...");
        let start = Instant::now();
        let config = VoiceOutputConfig::default();
        let mut tts = match VoiceOutput::new(config) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to initialize TTS: {}", e);
                return;
            }
        };
        println!("TTS initialized in {:?}\n", start.elapsed());

        // Test phrases
        let phrases = [
            ("Hello world", "/tmp/tts_hello.wav"),
            ("Welcome to Symthaea", "/tmp/tts_welcome.wav"),
            ("The quick brown fox jumps over the lazy dog", "/tmp/tts_pangram.wav"),
        ];

        for (text, output_path) in phrases {
            println!("Synthesizing: \"{}\"", text);

            let start = Instant::now();
            match tts.synthesize(text) {
                Ok(result) => {
                    let gen_time = start.elapsed();
                    println!("  Generation time: {:?}", gen_time);
                    println!("  Audio duration: {}ms ({:.1}s)", result.duration_ms, result.duration_ms as f32 / 1000.0);
                    println!("  Sample rate: {}Hz", result.sample_rate);
                    println!("  Samples: {}", result.samples.len());

                    // Calculate real-time factor
                    let rtf = gen_time.as_secs_f32() / (result.duration_ms as f32 / 1000.0);
                    println!("  Real-time factor: {:.2}x ({})", rtf,
                        if rtf <= 1.0 { "real-time or faster" } else { "slower than real-time" });

                    // Save to WAV
                    match result.save_wav(Path::new(output_path)) {
                        Ok(_) => println!("  Saved to: {}", output_path),
                        Err(e) => eprintln!("  Failed to save: {}", e),
                    }
                }
                Err(e) => {
                    eprintln!("  Synthesis failed: {}", e);
                }
            }
            println!();
        }

        println!("=== Test Complete ===");
        println!("Listen to the WAV files to verify audio quality:");
        for (_, path) in phrases {
            println!("  {}", path);
        }
    }

    #[cfg(not(feature = "voice-tts"))]
    {
        eprintln!("This example requires the voice-tts feature.");
        eprintln!("Run with: cargo run --example test_tts_audio --features voice");
    }
}
