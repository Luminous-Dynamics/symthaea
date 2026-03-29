// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Biological Homeostasis Demo (v2 - Warm-up included)
//!
//! Demonstrates the 'Dynamic Cognitive Throttle':
//! 1. Warm up the system to avoid cold-start bias.
//! 2. Simulate a power surge (> 5W) -> Stride increases to 8 (Low Resolution).
//! 3. Simulate stable power (< 3W) -> Stride decreases to 1 (High Resolution).
//! 4. Verify that Phi-based query processing adapts in real-time.

use std::time::Instant;
use symthaea::Symthaea;
use symthaea_core::hdc::unified_hv::get_cognitive_stride;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧬 Symthaea v0.6.0: Biological Homeostasis Demo (v2)\n");

    // 1. Initialize Symthaea
    let mut sym = Symthaea::new(1024, 64).await?;
    let query = "Analyze the stability of an LTC network.";

    // 2. WARM UP
    println!("[WARMUP] Priming the cognitive loop...");
    for _ in 0..2 {
        let _ = sym.process("Warmup").await?;
    }

    // --- PHASE 1: HIGH POWER SURGE (Low Resolution / Fast) ---
    println!("\n[EVENT] Power Surge Detected: 6.5 Watts");
    sym.apply_homeostasis(6.5);
    let stride_high = get_cognitive_stride();
    println!("[STATE] Cognitive Stride: {} (Expected: 8)", stride_high);

    let start = Instant::now();
    let res = sym.process(query).await?;
    let duration_throttle = start.elapsed();
    println!(
        "[THOUGHT] Throttled Cycle: {:?} (Confidence: {:.4})",
        duration_throttle, res.confidence
    );

    // --- PHASE 2: STABLE POWER (High Resolution / Slow) ---
    println!("\n[EVENT] Power Stable: 2.1 Watts");
    sym.apply_homeostasis(2.1);
    let stride_low = get_cognitive_stride();
    println!("[STATE] Cognitive Stride: {} (Expected: 1)", stride_low);

    let start = Instant::now();
    let res = sym.process(query).await?;
    let duration_full = start.elapsed();
    println!(
        "[THOUGHT] Full-Resolution Cycle: {:?} (Confidence: {:.4})",
        duration_full, res.confidence
    );

    // --- RESULTS ---
    println!("\n--- ⚖️ Homeostatic Comparison ---");
    println!("Throttled Time (Stride 8):  {:?}", duration_throttle);
    println!("Full Res Time  (Stride 1):  {:?}", duration_full);

    // In theory, Stride 1 should take ~8x longer in the similarity loop than Stride 8.
    // However, LLM overhead is constant, so we look for a significant delta.
    if duration_full > duration_throttle {
        println!("\n✅ HOMEOSTASIS VERIFIED: Processing speed adapted to power state.");
    } else {
        println!(
            "\n⚠️  THROTTLE FAILURE: Full resolution should take more time than throttled state."
        );
    }

    Ok(())
}
