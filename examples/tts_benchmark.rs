//! TTS Performance Benchmark
//!
//! Benchmarks Kokoro TTS with various configurations:
//! - CPU vs GPU execution providers
//! - Regular vs streaming synthesis
//! - Short vs long text
//!
//! Run with: cargo run --example tts_benchmark --release --features voice-tts

use std::time::{Duration, Instant};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for detailed logs
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║         Symthaea TTS Performance Benchmark             ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Check for model
    let model_path = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("symthaea/models/kokoro/kokoro-v0_19.onnx");

    let voice_path = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("symthaea/models/kokoro/voices/af_bella.bin");

    if !model_path.exists() {
        println!("❌ Model not found at: {:?}", model_path);
        println!("   Please download the Kokoro model first.");
        return Ok(());
    }

    println!("📂 Model path: {:?}", model_path);
    println!("🎤 Voice file: {:?}\n", voice_path);

    // Test texts of varying lengths
    let test_texts = vec![
        ("Short", "Hello, how are you today?"),
        ("Medium", "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing fonts and keyboards."),
        ("Long", "In the depths of consciousness, patterns emerge and dissolve like waves upon the shore. Each thought carries the weight of meaning, transforming the landscape of awareness. The mind dances between certainty and mystery, finding beauty in both the known and unknown."),
    ];

    // Benchmark configuration
    let warmup_runs = 1;
    let benchmark_runs = 3;

    #[cfg(feature = "voice-tts")]
    {
        use symthaea::voice::{VoiceOutput, VoiceOutputConfig, LTCPacing, ExecutionProvider};

        let config = VoiceOutputConfig::default();
        let pacing = LTCPacing::default();

        // Try to create output handler
        println!("🔧 Initializing TTS engine...");
        let start = Instant::now();

        let mut output = VoiceOutput::from_paths(&model_path, &voice_path, config.clone())?;

        let init_time = start.elapsed();
        println!("✅ Model loaded in {:.2}s\n", init_time.as_secs_f64());

        // Check system info
        println!("═══════════════════════════════════════════════════════════");
        println!("SYSTEM INFO");
        println!("═══════════════════════════════════════════════════════════");

        #[cfg(feature = "voice-tts-cuda")]
        {
            println!("🎮 CUDA support: ENABLED (feature flag)");
        }
        #[cfg(not(feature = "voice-tts-cuda"))]
        {
            println!("🖥️  CUDA support: DISABLED (use --features voice-tts-cuda)");
        }

        println!("🔢 Sample rate: {} Hz", config.sample_rate);
        println!("🎙️  Voice: {}", config.voice_name);
        println!();

        // Benchmark each text length
        println!("═══════════════════════════════════════════════════════════");
        println!("BENCHMARK RESULTS");
        println!("═══════════════════════════════════════════════════════════\n");

        for (name, text) in &test_texts {
            println!("📝 {} text ({} chars):", name, text.len());
            println!("   \"{}...\"", &text[..std::cmp::min(50, text.len())]);
            println!();

            // Warmup
            for _ in 0..warmup_runs {
                let _ = output.synthesize_with_pacing(text, pacing.clone());
            }

            // Benchmark runs
            let mut synth_times = Vec::new();
            let mut audio_durations = Vec::new();

            for _ in 0..benchmark_runs {
                let start = Instant::now();
                let result = output.synthesize_with_pacing(text, pacing.clone())?;
                let synth_time = start.elapsed();

                synth_times.push(synth_time);
                audio_durations.push(result.duration_ms);
            }

            // Calculate statistics
            let avg_synth_ms: f64 = synth_times.iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .sum::<f64>() / synth_times.len() as f64;

            let avg_audio_ms = audio_durations.iter().sum::<u64>() as f64 / audio_durations.len() as f64;

            let rtf = avg_synth_ms / avg_audio_ms;

            // Report
            println!("   ⏱️  Synthesis time: {:.1}ms (avg of {} runs)", avg_synth_ms, benchmark_runs);
            println!("   🎵 Audio duration: {:.0}ms", avg_audio_ms);
            println!("   📊 RTF (Real-time Factor): {:.2}x", rtf);

            if rtf < 1.0 {
                println!("   ✅ REAL-TIME capable! ({:.1}% margin)", (1.0 - rtf) * 100.0);
            } else {
                println!("   ⚠️  Not real-time ({:.1}% over)", (rtf - 1.0) * 100.0);
            }
            println!();
        }

        // Overall performance summary
        println!("═══════════════════════════════════════════════════════════");
        println!("PERFORMANCE SUMMARY");
        println!("═══════════════════════════════════════════════════════════\n");

        // Quick single measurement for summary
        let test_sentence = "Hello, this is a test sentence for benchmarking.";
        let start = Instant::now();
        let result = output.synthesize_with_pacing(test_sentence, pacing.clone())?;
        let synth_time = start.elapsed();

        let chars_per_sec = test_sentence.len() as f64 / synth_time.as_secs_f64();
        let samples_per_sec = result.samples.len() as f64 / synth_time.as_secs_f64();

        println!("   📊 Throughput: {:.0} chars/sec", chars_per_sec);
        println!("   🎵 Sample rate: {:.0} samples/sec generated", samples_per_sec);
        println!("   💾 Memory per second of audio: ~{:.0}KB",
            (config.sample_rate as f64 * 4.0) / 1024.0);

        // Latency estimate
        let first_chunk_latency = output.estimate_latency_ms(10);
        println!("   ⚡ First chunk latency estimate: {}ms", first_chunk_latency);

        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("RECOMMENDATIONS");
        println!("═══════════════════════════════════════════════════════════\n");

        let rtf = synth_time.as_secs_f64() * 1000.0 / result.duration_ms as f64;

        if rtf < 0.5 {
            println!("   ✅ Excellent performance! RTF < 0.5x");
            println!("   🚀 System is well-suited for real-time voice applications");
        } else if rtf < 1.0 {
            println!("   ✅ Good performance! RTF < 1.0x (real-time capable)");
            println!("   💡 Consider GPU acceleration for even better results");
        } else {
            println!("   ⚠️  Below real-time (RTF > 1.0x)");
            println!("   💡 Recommendations:");
            println!("      - Enable GPU: --features voice-tts-cuda");
            println!("      - Use streaming mode for lower latency");
            println!("      - Consider shorter sentences");
        }

        #[cfg(not(feature = "voice-tts-cuda"))]
        {
            println!("\n   💡 To enable GPU acceleration:");
            println!("      cargo run --example tts_benchmark --release --features voice-tts-cuda");
        }
    }

    #[cfg(not(feature = "voice-tts"))]
    {
        println!("❌ voice-tts feature not enabled");
        println!("   Run with: cargo run --example tts_benchmark --release --features voice-tts");
    }

    Ok(())
}
