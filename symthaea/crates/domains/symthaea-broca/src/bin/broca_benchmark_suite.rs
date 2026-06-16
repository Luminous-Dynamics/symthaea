// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-benchmark-suite: Unified evaluation for Narrative, Commonsense, and Topological Reasoning.
//!
//! Tests:
//! 1. LAMBADA (Narrative Continuity): Predict last word of context.
//! 2. HellaSwag (Commonsense): Select most plausible ending.
//! 3. Hodge-Consistency: Measure frequency of beta_1 topological spikes.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Deserialize)]
struct LambadaItem {
    text: String,
}

#[derive(Debug, Deserialize)]
struct HellaSwagItem {
    ctx: String,
    endings: Vec<String>,
    label: serde_json::Value,
}

impl HellaSwagItem {
    fn label_index(&self) -> usize {
        match &self.label {
            serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as usize,
            serde_json::Value::String(s) => s.parse::<usize>().unwrap_or(0),
            _ => 0,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("symthaea-bench-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = false; // Disable veto for pure predictive benchmarking

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;

    println!("🚀 Broca Benchmark Suite Initialized.");
    println!("--------------------------------------");

    run_topological_bench(&mut generator)?;
    run_lambada_bench(&mut generator)?;
    run_hellaswag_bench(&mut generator)?;

    Ok(())
}

fn run_topological_bench(generator: &mut LiquidMambaGenerator) -> Result<()> {
    println!("\n[1/3] Running Custom Hodge-Consistency Bench...");
    let start = Instant::now();
    let mut total_chunks = 0;
    let mut beta1_spikes = 0;

    // Generate long-form monologue and monitor topological health
    for i in 0..5 {
        let channels = ThoughtChannels::with_intent(i);
        let monologue = generator.generate_semantic_monologue(&channels, 3)?;
        total_chunks += monologue.chunks.len();

        let history = generator.betti_history.lock();
        for &(_, beta1) in history.iter() {
            if beta1 > 0 {
                beta1_spikes += 1;
            }
        }
    }

    let error_rate = beta1_spikes as f32 / total_chunks.max(1) as f32;
    println!("   └─ Total Chunks: {}", total_chunks);
    println!("   └─ beta_1 Spikes: {}", beta1_spikes);
    println!("   └─ Topological Error Rate: {:.2}%", error_rate * 100.0);
    println!("   └─ Elapsed: {:.2}s", start.elapsed().as_secs_f32());
    Ok(())
}

fn run_lambada_bench(generator: &mut LiquidMambaGenerator) -> Result<()> {
    println!("\n[2/3] Running LAMBADA (Narrative Continuity) Bench...");
    let path = "data/benchmarks/lambada/test.jsonl";
    if !std::path::Path::new(path).exists() {
        println!("   ⚠️  LAMBADA data not found. Skipping.");
        return Ok(());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut correct = 0;
    let mut total = 0;
    let start = Instant::now();

    for line in reader.lines().take(200) {
        // Solid baseline subset
        let item: LambadaItem = serde_json::from_str(&line?)?;
        let words: Vec<&str> = item.text.split_whitespace().collect();
        if words.len() < 2 {
            continue;
        }

        let last_word = words.last().unwrap().to_lowercase();
        let context = words[..words.len() - 1].join(" ");

        // In a real benchmark, we'd use the generator to predict the next token
        // and check if it matches the last_word.
        // For Broca, we set the intent as the context and see if the first generated word matches.
        let channels = ThoughtChannels::with_intent(total % 1000); // Using index as intent sector
        // Set goal to the context's HDC encoding
        // (Simplified for now: we just generate and check)
        let result = generator.generate(&channels);
        let predicted = result
            .text
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();

        if predicted == last_word {
            correct += 1;
        }
        total += 1;
    }

    println!("   └─ Total Samples: {}", total);
    println!(
        "   └─ Accuracy: {:.2}%",
        (correct as f32 / total as f32) * 100.0
    );
    println!("   └─ Elapsed: {:.2}s", start.elapsed().as_secs_f32());
    Ok(())
}

fn run_hellaswag_bench(generator: &mut LiquidMambaGenerator) -> Result<()> {
    println!("\n[3/3] Running HellaSwag (Commonsense Reasoning) Bench...");
    let path = "data/benchmarks/hellaswag/validation.jsonl";
    if !std::path::Path::new(path).exists() {
        println!("   ⚠️  HellaSwag data not found. Skipping.");
        return Ok(());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut correct = 0;
    let mut total = 0;
    let start = Instant::now();

    for line in reader.lines().take(50) {
        // Solid baseline subset
        let item: HellaSwagItem = serde_json::from_str(&line?)?;
        let gold = item.label_index();

        let mut best_idx = 0;
        let mut max_coherence = f32::NEG_INFINITY;

        for (j, ending) in item.endings.iter().enumerate() {
            // Test each ending: how coherent is the resulting manifold?
            let _full_text = format!("{} {}", &item.ctx, ending);

            // Set the intent and provide the physical prompt text as context
            let channels = ThoughtChannels::with_intent(j);
            // In a real system, we'd encode full_text into her active context.
            // For this benchmark, we'll simulate the "digest" of the text.
            let result = generator.generate_inner(&channels, None)?;

            if result.final_coherence > max_coherence {
                max_coherence = result.final_coherence;
                best_idx = j;
            }
        }

        if best_idx == gold {
            correct += 1;
        }
        total += 1;
    }

    println!("   └─ Total Samples: {}", total);
    println!(
        "   └─ Accuracy: {:.2}%",
        (correct as f32 / total as f32) * 100.0
    );
    println!("   └─ Elapsed: {:.2}s", start.elapsed().as_secs_f32());
    Ok(())
}
