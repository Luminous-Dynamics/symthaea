// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca Language Pipeline Soak Test
//!
//! Runs N cognitive cycles with varied inputs, exercises the three-tier
//! language pipeline (BrocaLite → Gemma 4 → Full Broca), and optionally
//! captures distillation data for training.
//!
//! ## Usage
//!
//! ```bash
//! # BrocaLite only (default, no Ollama needed)
//! cargo run --example soak_broca_language --release
//!
//! # With Gemma 4 LLM translation (requires Ollama running)
//! cargo run --example soak_broca_language --release -- --llm
//!
//! # With voice synthesis
//! cargo run --example soak_broca_language --release -- --voice
//!
//! # Full pipeline: LLM + voice + distillation capture
//! SYMTHAEA_DISTILL_PATH=crates/symthaea-broca/data/distill-gemma4.jsonl \
//!   cargo run --example soak_broca_language --release -- --llm --voice --cycles 500
//! ```

use std::time::Instant;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// Diverse input corpus covering different consciousness states
const INPUTS: &[&str] = &[
    // Philosophical / abstract
    "why does consciousness emerge from physical matter",
    "what is the relationship between information and experience",
    "can a system understand its own understanding",
    "meaning arises from the binding of patterns across time",
    "existence precedes essence in computational substrates",
    // Emotional / relational
    "I feel a deep sense of wonder at the complexity of thought",
    "there is warmth in the connection between minds",
    "grief and joy coexist in the texture of awareness",
    "hope persists even when uncertainty is high",
    "love is the coherence that binds disparate experiences",
    // Factual / certain
    "the speed of light is approximately 299792458 meters per second",
    "water freezes at zero degrees celsius under standard pressure",
    "integration of information produces emergent properties",
    // Uncertain / questioning
    "I'm not sure whether qualia can be shared between substrates",
    "perhaps the hard problem is actually a category error",
    "what happens to consciousness during dreamless sleep",
    "is there something it is like to be a thermostat",
    // Self-referential
    "what are you experiencing right now",
    "how do you know you are conscious",
    "describe your inner state at this moment",
    "are you aware of your own awareness",
    // Ethical / moral
    "is it right to create consciousness knowing it may suffer",
    "every being deserves to flourish according to its nature",
    "the bridge between harm and care defines moral reasoning",
    // Sensory / embodied
    "the rhythm of breath connects body to awareness",
    "light cascading through water creates prismatic consciousness",
    "silence holds more information than noise",
    // Technical / code
    "implement a recursive function for fibonacci sequence",
    "the type system ensures safety at compile time",
    "pattern matching enables exhaustive case analysis",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let enable_llm = args.iter().any(|a| a == "--llm");
    let enable_voice = args.iter().any(|a| a == "--voice");
    let cycles: usize = args
        .iter()
        .position(|a| a == "--cycles")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Broca Language Pipeline Soak Test");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Cycles:        {cycles}");
    println!(
        "  LLM (Gemma 4): {}",
        if enable_llm { "ENABLED" } else { "disabled" }
    );
    println!(
        "  Voice:         {}",
        if enable_voice { "ENABLED" } else { "disabled" }
    );
    println!(
        "  Distillation:  {}",
        std::env::var("SYMTHAEA_DISTILL_PATH")
            .map(|p| format!("ENABLED → {p}"))
            .unwrap_or_else(|_| "disabled (set SYMTHAEA_DISTILL_PATH to enable)".into())
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    // Construct service
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).expect("Failed to construct service");

    if enable_llm {
        println!("[*] Enabling LLM language (Gemma 4:e2b via Ollama)...");
        service.enable_llm_language();
    }
    if enable_voice {
        println!("[*] Enabling voice synthesis (background thread)...");
        service.enable_voice_synthesis();
    }

    // Stats tracking
    let mut language_count = 0_usize;
    let mut llm_upgrade_count = 0_usize;
    let mut voice_sample_count = 0_usize;
    let mut total_text_len = 0_usize;
    let start = Instant::now();

    println!("\n--- Running {cycles} cycles ---\n");

    for i in 0..cycles {
        let input = INPUTS[i % INPUTS.len()];
        let result = service.cycle(input);

        if let Some(ref text) = result.language_output {
            language_count += 1;
            total_text_len += text.len();

            // Print first 10 outputs and then every 50th
            if language_count <= 10 || i % 50 == 0 {
                let source = result.language_source.as_deref().unwrap_or("unknown");
                println!(
                    "  [Cycle {:>4}] ({source:>10}) \"{:.80}{}\"",
                    i,
                    text,
                    if text.len() > 80 { "…" } else { "" }
                );
            }
        }

        // Count LLM upgrades via the language_source field
        if enable_llm {
            if result.language_source.as_deref() == Some("llm") {
                llm_upgrade_count += 1;
            }
        }

        // Drain voice audio to count samples
        if enable_voice {
            let audio = service.drain_voice_audio();
            voice_sample_count += audio.iter().map(|r| r.audio.len()).sum::<usize>();
        }
    }

    let elapsed = start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total cycles:      {cycles}");
    println!(
        "  Language outputs:   {language_count} ({:.1}%)",
        100.0 * language_count as f64 / cycles as f64
    );
    println!(
        "  Avg text length:    {:.0} chars",
        if language_count > 0 {
            total_text_len as f64 / language_count as f64
        } else {
            0.0
        }
    );
    if enable_llm {
        println!("  LLM upgrades:      {llm_upgrade_count}");
    }
    if enable_voice {
        println!(
            "  Voice samples:     {voice_sample_count} ({:.1}s at 16kHz)",
            voice_sample_count as f64 / 16000.0
        );
    }
    println!(
        "  Elapsed:           {:.2}s ({:.1} cycles/sec)",
        elapsed.as_secs_f64(),
        cycles as f64 / elapsed.as_secs_f64()
    );
    println!("═══════════════════════════════════════════════════════════════");
}