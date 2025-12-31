//! Test Piper TTS - Fast native Rust text-to-speech
//!
//! Run with: cargo run --example test_piper_tts --features voice-tts-piper --release

use std::time::Instant;

fn main() {
    #[cfg(feature = "voice-tts-piper")]
    {
        use std::path::Path;
        use piper_rs;

        println!("🎤 Piper TTS Test");
        println!("==================");

        // Check for model
        let config_path = Path::new("models/piper/en_US-amy-medium.onnx.json");
        if !config_path.exists() {
            eprintln!("❌ Model not found: {:?}", config_path);
            eprintln!("\nPlease download the model first:");
            eprintln!("  mkdir -p models/piper");
            eprintln!("  curl -L -o models/piper/en_US-amy-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx");
            eprintln!("  curl -L -o models/piper/en_US-amy-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json");
            return;
        }

        println!("📂 Loading model from: {:?}", config_path);
        let start = Instant::now();

        match piper_rs::from_config_path(config_path) {
            Ok(model) => {
                println!("✅ Model loaded in {:.2}s", start.elapsed().as_secs_f32());

                let audio_info = model.audio_output_info();
                println!("📊 Audio: {} Hz, {} channels", audio_info.sample_rate, audio_info.channels);

                // Test phrases
                let phrases = [
                    ("Hello world", "/tmp/piper_hello.wav"),
                    ("Welcome to Symthaea, the consciousness-first AI.", "/tmp/piper_welcome.wav"),
                    ("The quick brown fox jumps over the lazy dog.", "/tmp/piper_pangram.wav"),
                ];

                for (text, output) in &phrases {
                    println!("\n📝 Synthesizing: \"{}\"", text);
                    let start = Instant::now();

                    // Phonemize
                    match model.phonemize_text(text, None) {
                        Ok(phonemes) => {
                            println!("   Phonemes: {:?}", phonemes.len());

                            // Synthesize
                            let mut all_samples: Vec<i16> = Vec::new();
                            for phoneme in phonemes {
                                match model.speak_one_sentence(&phoneme, None) {
                                    Ok(audio) => {
                                        let samples = audio.into_samples();
                                        all_samples.extend(samples);
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Synthesis failed: {:?}", e);
                                        continue;
                                    }
                                }
                            }

                            if all_samples.is_empty() {
                                eprintln!("❌ No samples generated");
                                continue;
                            }

                            let elapsed = start.elapsed();
                            let duration_s = all_samples.len() as f32 / audio_info.sample_rate as f32;
                            let rtf = elapsed.as_secs_f32() / duration_s;

                            // Write WAV file
                            let spec = hound::WavSpec {
                                channels: 1,
                                sample_rate: audio_info.sample_rate as u32,
                                bits_per_sample: 16,
                                sample_format: hound::SampleFormat::Int,
                            };

                            match hound::WavWriter::create(output, spec) {
                                Ok(mut writer) => {
                                    for sample in &all_samples {
                                        let _ = writer.write_sample(*sample);
                                    }
                                    if let Err(e) = writer.finalize() {
                                        eprintln!("❌ WAV finalize failed: {}", e);
                                        continue;
                                    }

                                    println!("✅ Saved to: {}", output);
                                    println!("   Duration: {:.2}s", duration_s);
                                    println!("   Generation time: {:.2}s", elapsed.as_secs_f32());
                                    println!("   Real-time factor: {:.2}x", rtf);

                                    if rtf < 1.0 {
                                        println!("   🚀 FASTER than real-time!");
                                    }
                                }
                                Err(e) => {
                                    eprintln!("❌ WAV write failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Phonemization failed: {:?}", e);
                        }
                    }
                }

                println!("\n🎉 Test complete!");
                println!("\nTo play the audio:");
                println!("  aplay /tmp/piper_hello.wav");
                println!("  # or");
                println!("  mpv /tmp/piper_hello.wav");
            }
            Err(e) => {
                eprintln!("❌ Failed to load model: {:?}", e);
            }
        }
    }

    #[cfg(not(feature = "voice-tts-piper"))]
    {
        eprintln!("❌ This example requires the voice-tts-piper feature");
        eprintln!("Run with: cargo run --example test_piper_tts --features voice-tts-piper --release");
    }
}
