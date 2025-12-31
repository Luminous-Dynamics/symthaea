//! TTS Performance Benchmark
//!
//! Profiles each component of the TTS pipeline:
//! - Model loading
//! - Warmup
//! - Tokenization (Misaki)
//! - Inference (ONNX)

use std::time::Instant;
use symthaea::voice::{VoiceOutput, VoiceOutputConfig, tokenizer};

fn main() -> anyhow::Result<()> {
    println!("🎤 TTS Performance Benchmark");
    println!("============================\n");

    // Test phrases
    let test_phrases = [
        ("Short", "Hello world."),
        ("Medium", "Welcome to Symthaea, the consciousness-first AI system."),
        ("Long", "The quick brown fox jumps over the lazy dog. This is a longer sentence to test performance scaling."),
    ];

    // 1. Measure model loading
    println!("📦 Phase 1: Model Loading");
    let load_start = Instant::now();
    let config = VoiceOutputConfig::default();
    let mut voice = VoiceOutput::new(config)?;
    let load_time = load_start.elapsed();
    println!("   Model load time: {:.2}s\n", load_time.as_secs_f64());

    // 2. Measure warmup
    println!("🔥 Phase 2: Model Warmup");
    let warmup_start = Instant::now();
    voice.warmup()?;
    let warmup_time = warmup_start.elapsed();
    println!("   Warmup time: {:.2}s\n", warmup_time.as_secs_f64());

    // 3. Benchmark tokenization separately
    println!("📝 Phase 3: Tokenization Benchmark (Misaki G2P)");
    for (name, text) in &test_phrases {
        let tok_start = Instant::now();
        let tokens = tokenizer::tokenize(text).map_err(|e| anyhow::anyhow!(e))?;
        let tok_time = tok_start.elapsed();
        println!("   {} ({} tokens): {:.3}s", name, tokens.len(), tok_time.as_secs_f64());
    }
    println!();

    // 4. Full synthesis benchmark (after warmup)
    println!("🔊 Phase 4: Full Synthesis Benchmark (post-warmup)");
    println!("   (Each phrase run 3 times, showing average)\n");

    let mut total_generation_time = 0.0;
    let mut total_audio_duration = 0.0;

    for (name, text) in &test_phrases {
        let mut times = Vec::new();
        let mut audio_duration = 0.0;

        for _ in 0..3 {
            let synth_start = Instant::now();
            let result = voice.synthesize(text)?;
            let synth_time = synth_start.elapsed();
            times.push(synth_time.as_secs_f64());
            audio_duration = result.duration_ms as f64 / 1000.0;
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let rtf = avg_time / audio_duration;  // Real-time factor

        println!("   {} phrase:", name);
        println!("      Text: \"{}\"", if text.len() > 50 { &text[..50] } else { text });
        println!("      Generation: {:.2}s (avg of 3 runs)", avg_time);
        println!("      Audio: {:.2}s", audio_duration);
        println!("      RTF: {:.1}x (should be <1.0 for real-time)\n", rtf);

        total_generation_time += avg_time;
        total_audio_duration += audio_duration;
    }

    // Summary
    println!("📊 Summary");
    println!("==========");
    println!("   Model load:     {:.2}s", load_time.as_secs_f64());
    println!("   Warmup:         {:.2}s", warmup_time.as_secs_f64());
    println!("   Avg RTF:        {:.1}x", total_generation_time / total_audio_duration);

    let target_rtf = 1.0;
    if total_generation_time / total_audio_duration <= target_rtf {
        println!("\n   ✅ Target met! RTF ≤ {:.1}x (real-time capable)", target_rtf);
    } else {
        let speedup_needed = (total_generation_time / total_audio_duration) / target_rtf;
        println!("\n   ❌ Need {:.1}x speedup to reach real-time", speedup_needed);
        println!("   💡 Recommendations:");
        println!("      - Enable CUDA: Add 'cuda' feature to ort in Cargo.toml");
        println!("      - Use INT8 quantized model (if available)");
        println!("      - Pre-cache common phrases");
    }

    Ok(())
}
