// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-benchmark-suite: exploratory diagnostics for Liquid Broca.
//!
//! The custom Betti-history probe below is diagnostic only and is not a
//! validated language-quality metric. The historical LAMBADA and HellaSwag
//! paths are deliberately fail-closed: they did not condition predictions on
//! the benchmark text and therefore must not emit task-labelled accuracy.
//! Re-enable those tasks only after implementing their actual scoring protocol.

use anyhow::Result;
use std::time::Instant;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("symthaea-bench-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = false; // Disable veto for this exploratory diagnostic.

    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;

    println!("Broca exploratory diagnostics initialized.");
    println!("These outputs are not comparable to standard language benchmarks.");
    println!("---------------------------------------------------------------");

    run_topological_bench(&mut generator)?;
    run_lambada_bench()?;
    run_hellaswag_bench()?;

    Ok(())
}

fn run_topological_bench(generator: &mut LiquidMambaGenerator) -> Result<()> {
    println!("\n[1/3] Running experimental Betti-history diagnostic...");
    let start = Instant::now();
    let mut total_chunks = 0usize;
    let mut beta1_samples = 0usize;
    let mut beta1_spikes = 0usize;

    // Count only history entries emitted by each run. Previously, each
    // iteration re-counted the entire accumulated history, inflating spikes.
    for i in 0..5 {
        let previous_history_len = generator.betti_history.lock().len();
        let channels = ThoughtChannels::with_intent(i);
        let monologue = generator.generate_semantic_monologue(&channels, 3)?;
        total_chunks += monologue.chunks.len();

        let history = generator.betti_history.lock();
        for &(_, beta1) in history.iter().skip(previous_history_len) {
            beta1_samples += 1;
            if beta1 > 0 {
                beta1_spikes += 1;
            }
        }
    }

    println!("   Chunks generated: {total_chunks}");
    println!("   New Betti-history samples: {beta1_samples}");
    println!("   beta_1-positive samples: {beta1_spikes}");
    if beta1_samples > 0 {
        println!(
            "   Exploratory beta_1-positive fraction: {:.2}%",
            beta1_spikes as f64 / beta1_samples as f64 * 100.0
        );
    } else {
        println!("   Exploratory beta_1-positive fraction: unavailable (no new samples)");
    }
    println!("   Elapsed: {:.2}s", start.elapsed().as_secs_f32());
    println!("   NOTE: This is not a validated model-quality or safety metric.");
    Ok(())
}

fn run_lambada_bench() -> Result<()> {
    println!("\n[2/3] LAMBADA word prediction: NOT RUN");
    println!("   Reason: the legacy path did not condition on the context or score the");
    println!("   held-out target token sequence. No LAMBADA accuracy is reported.");
    Ok(())
}

fn run_hellaswag_bench() -> Result<()> {
    println!("\n[3/3] HellaSwag multiple-choice: NOT RUN");
    println!("   Reason: the legacy path did not feed the context/endings to the model");
    println!("   and selected answers using generation coherence, not conditional");
    println!("   likelihood. No HellaSwag accuracy is reported.");
    Ok(())
}
