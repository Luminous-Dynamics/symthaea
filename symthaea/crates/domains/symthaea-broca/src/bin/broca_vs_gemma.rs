// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-vs-gemma: comparative stability probe.
//!
//! Measures 'Topological Resistance' to circular reasoning and hallucinations
//! for Symthaea. External model values are explicitly marked as proxy values
//! unless a real baseline runner is wired in.

use anyhow::Result;
use serde::Serialize;
use std::time::Instant;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Serialize)]
struct StabilityProbeReport {
    schema_version: u32,
    comparison_mode: &'static str,
    symthaea: SymthaeaMetrics,
    external_baseline: BaselineMetrics,
    notes: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct SymthaeaMetrics {
    measured: bool,
    cycles: usize,
    average_topological_drift: f32,
    retention: f32,
    elapsed_secs: f32,
    stomp_damage_percent: f32,
    stomp_retention: f32,
}

#[derive(Debug, Serialize)]
struct BaselineMetrics {
    name: &'static str,
    measured: bool,
    average_topological_drift: f32,
    retention: f32,
    elapsed_secs: f32,
    stomp_retention: f32,
}

fn main() -> Result<()> {
    let genesis = GenesisSeed::from_phrase("comparative-bench-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = false; // Pure generational stress test

    let mut symthaea = LiquidMambaGenerator::new(&genesis, config)?;

    println!("Symthaea Narrative Integrity Probe");
    println!("==================================\n");
    println!("External baseline is a documented proxy, not a measured Gemma run.\n");

    // 1. Run Symthaea 1,000-Cycle Stress Test
    println!("[1/2] Stress-testing Symthaea (1,000 Cycles / Topological Aware)...");
    let start_s = Instant::now();
    let mut s_drift = 0.0f32;
    let global_intent = symthaea
        .encoder()
        .encode(&ThoughtChannels::with_intent(777));

    for i in 0..1000 {
        let channels = ThoughtChannels::with_intent(777);
        let monologue = symthaea.generate_semantic_monologue(&channels, 1)?;
        let step_nucleus = symthaea.recursive_fold(&monologue);
        s_drift += 1.0 - step_nucleus.similarity(&global_intent);

        if i % 100 == 0 {
            println!("   └─ Cycle {}: Drift = {:.4}", i, s_drift / (i + 1) as f32);
        }
    }
    let s_time = start_s.elapsed().as_secs_f32();

    // 2. Explicit proxy baseline.
    println!("\n[2/2] Recording external baseline proxy values...");
    let g_drift = 0.84;
    let g_time = s_time * 4.5;

    println!("\n📈 Results Summary (Horizon: 1,000 Steps):");
    println!("-------------------");
    println!("💎 Symthaea (HDC+CfC + Memory Folding):");
    println!(
        "   └─ Average Topological Drift: {:.4} (Retention: {:.2}%)",
        s_drift / 1000.0,
        100.0 * (1.0 - s_drift / 1000.0)
    );
    println!("   └─ Total Time: {:.2}s", s_time);

    println!("\n🤖 External transformer baseline proxy:");
    println!(
        "   └─ Average Topological Drift: {:.4} (Retention: {:.2}%)",
        g_drift,
        100.0 * (1.0 - g_drift)
    );
    println!("   └─ Estimated Time: {:.2}s", g_time);

    // 3. The 'Manifold Stomp' Stress Test (Fault Tolerance)
    println!("\n[3/3] The 'Manifold Stomp' (Destroying 10% of weight manifold)...");
    let sym_retention = symthaea.run_manifold_stomp_test(0.1);

    // Proxy baseline retained only for rough comparison.
    let g_retention: f32 = 0.04;

    println!("\n📈 Results Summary (Stress: 10% Destruction):");
    println!("-------------------");
    println!("💎 Symthaea (HDC-Based):");
    println!("   └─ Integrity Retention: {:.2}%", sym_retention * 100.0);

    println!("\n🤖 External transformer proxy:");
    println!(
        "   └─ Proxy Integrity Retention: {:.2}%",
        g_retention * 100.0
    );

    let report = StabilityProbeReport {
        schema_version: 1,
        comparison_mode: "symthaea-measured-vs-external-proxy",
        symthaea: SymthaeaMetrics {
            measured: true,
            cycles: 1000,
            average_topological_drift: s_drift / 1000.0,
            retention: 1.0 - s_drift / 1000.0,
            elapsed_secs: s_time,
            stomp_damage_percent: 0.1,
            stomp_retention: sym_retention,
        },
        external_baseline: BaselineMetrics {
            name: "transformer-proxy",
            measured: false,
            average_topological_drift: g_drift,
            retention: 1.0 - g_drift,
            elapsed_secs: g_time,
            stomp_retention: g_retention,
        },
        notes: vec![
            "External baseline values are proxies and must not be used as empirical claims.",
            "Wire a real baseline runner before using this for model-vs-model decisions.",
        ],
    };
    println!("\n{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
