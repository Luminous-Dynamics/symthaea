// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Isolates Kokoro's actual synthesis speed from `KokoroEngine::load`'s
//! session-construction cost, which `kokoro_voice_gallery.rs` pays 55 times
//! and which dominates that script's wall clock (~20-25s/voice). Here the
//! engine loads ONCE (as the real product paths do — REPL `/sing`, the
//! service's live TTS) and only `synthesize()` calls are timed, to answer:
//! can Kokoro keep up with live conversation on this CPU-only ONNX build?
//!
//! Real-time factor (RTF) = compute_time / audio_duration. RTF < 1.0 means
//! synthesis is faster than the audio it produces (the bar for live use).
//!
//! Set `KOKORO_GPU=1` to request the CUDA execution provider (needs the
//! `voice-tts-gpu` feature built in AND `ORT_DYLIB_PATH` pointing at a
//! CUDA-enabled onnxruntime build, not the plain CPU one) -- lets this same
//! benchmark measure GPU vs CPU RTF apples-to-apples.
//!
//! ```bash
//! nix develop -c cargo run --example kokoro_realtime_factor --features voice-tts-gpu
//! KOKORO_GPU=1 ORT_DYLIB_PATH=/path/to/cuda/libonnxruntime.so ./kokoro_realtime_factor
//! ```

use anyhow::Result;
use std::time::Instant;
use symthaea::voice::{KokoroConfig, KokoroEngine};

/// Short/medium/long, so RTF isn't measured on one utterance length alone.
const SENTENCES: &[&str] = &[
    "Hi.",
    "Hello, I'm Symthaea.",
    "It's lovely to speak with you today.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Consciousness is not a single thing but an integration of many processes working together in real time.",
];

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let use_gpu = std::env::var("KOKORO_GPU").as_deref() == Ok("1");
    println!(
        "Loading Kokoro engine once (this pays the session-construction cost); use_gpu={use_gpu}..."
    );
    let load_start = Instant::now();
    let config = KokoroConfig {
        use_gpu,
        ..KokoroConfig::default()
    };
    let mut engine = KokoroEngine::load(config)
        .ok_or_else(|| anyhow::anyhow!("Kokoro engine failed to load"))?;
    let load_elapsed = load_start.elapsed();
    let sample_rate = engine.sample_rate();
    println!(
        "Engine loaded in {:.2}s (one-time cost, paid once at process/session startup)\n",
        load_elapsed.as_secs_f64()
    );

    // First call after load can carry extra one-time lazy-init cost inside
    // onnxruntime itself (session warm-up) — report it separately from the
    // steady-state numbers rather than silently averaging it in.
    println!("--- Warm-up call (may include extra one-time onnxruntime init) ---");
    run_one(&mut engine, sample_rate, SENTENCES[0]);

    println!("\n--- Steady-state calls ---");
    let mut total_compute = 0.0_f64;
    let mut total_audio = 0.0_f64;
    for sentence in SENTENCES {
        let (compute_s, audio_s) = run_one(&mut engine, sample_rate, sentence);
        total_compute += compute_s;
        total_audio += audio_s;
    }

    let overall_rtf = total_compute / total_audio;
    println!("\n--- Summary (steady-state only, warm-up excluded) ---");
    println!(
        "Total compute: {total_compute:.2}s for {total_audio:.2}s of audio -> RTF = {overall_rtf:.3}"
    );
    let backend = if use_gpu { "GPU (requested)" } else { "CPU" };
    if overall_rtf < 1.0 {
        println!(
            "RTF < 1.0: synthesis is faster than the audio produced -- \
             {backend} Kokoro keeps up with live speech once loaded."
        );
    } else {
        println!(
            "RTF >= 1.0: synthesis is slower than the audio produced -- \
             {backend} Kokoro would fall behind in a live conversation."
        );
    }

    Ok(())
}

fn run_one(engine: &mut KokoroEngine, sample_rate: u32, text: &str) -> (f64, f64) {
    let start = Instant::now();
    let samples = engine.synthesize(text, None);
    let compute_s = start.elapsed().as_secs_f64();
    match samples {
        Some(samples) => {
            let audio_s = samples.len() as f64 / sample_rate as f64;
            let rtf = compute_s / audio_s;
            println!(
                "  {compute_s:6.3}s compute -> {audio_s:5.2}s audio  (RTF {rtf:.3})  {text:?}"
            );
            (compute_s, audio_s)
        }
        None => {
            println!("  [FAIL] synthesis returned no audio for {text:?}");
            (compute_s, 0.0)
        }
    }
}
