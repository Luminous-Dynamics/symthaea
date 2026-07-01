// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! JEPA A/B Test on Real Cognitive Loop
//!
//! Runs CognitiveLoopService with and without the `jepa` feature,
//! comparing Phi, PE, and JEPA telemetry across 200 cycles of natural
//! language input. Proves (or disproves) that JEPA helps consciousness.
//!
//! Usage:
//!   Without JEPA:  cargo run --release --example benchmark_jepa_cognitive
//!   With JEPA:     cargo run --release --features jepa --example benchmark_jepa_cognitive

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const NUM_CYCLES: usize = 200;

const INPUTS: &[&str] = &[
    "the sun rises over the mountain, warming the valley below",
    "a sudden crash echoes through the darkness",
    "gentle waves lap at the shore as the moon rises",
    "the machine grinds to a halt, alarms blaring",
    "children laughing in the garden after the rain",
    "silence falls across the empty room like snow",
    "new patterns emerge from chaos, self-organizing",
    "the old bridge creaks under the weight of memory",
    "music drifts through the open window at dusk",
    "a warning light blinks red on the console",
    "the forest canopy filters sunlight into emerald shards",
    "thunder rolls across the plains, shaking the earth",
    "a single bird calls across the frozen lake",
    "the reactor core temperature stabilizes at nominal",
    "two strangers share a knowing glance on the train",
    "the algorithm converges after ten thousand iterations",
    "moonlight paints silver paths across the still water",
    "the city hums with ten million overlapping conversations",
    "a seed cracks open in the dark soil, reaching upward",
    "the satellite passes silently above, recording everything",
];

fn main() {
    let jepa_enabled = cfg!(feature = "jepa");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  JEPA A/B Test — Real Cognitive Loop");
    println!(
        "  JEPA: {}",
        if jepa_enabled {
            "ENABLED"
        } else {
            "DISABLED (baseline)"
        }
    );
    println!("  {} cycles, {} unique inputs", NUM_CYCLES, INPUTS.len());
    println!("═══════════════════════════════════════════════════════════════\n");

    let config = CognitiveLoopConfig::default();
    let mut service = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create CognitiveLoopService: {e}");
            return;
        }
    };

    let mut phi_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut pe_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut coherence_trajectory = Vec::with_capacity(NUM_CYCLES);
    let mut total_learning = 0usize;

    #[cfg(feature = "jepa")]
    let mut jepa_pe_trajectory = Vec::with_capacity(NUM_CYCLES);

    for i in 0..NUM_CYCLES {
        let input = INPUTS[i % INPUTS.len()];
        let result = service.cycle(input);

        let phi = result.metadata.consciousness.consciousness_level;
        phi_trajectory.push(phi);
        pe_trajectory.push(result.prediction_error);
        coherence_trajectory.push(result.metadata.prediction_coherence);
        if result.learning_occurred {
            total_learning += 1;
        }

        #[cfg(feature = "jepa")]
        {
            jepa_pe_trajectory.push(result.metadata.jepa_latent_pe);
        }
    }

    // Compute statistics
    let mean_phi: f64 = phi_trajectory.iter().sum::<f64>() / NUM_CYCLES as f64;
    let mean_pe: f32 = pe_trajectory.iter().sum::<f32>() / NUM_CYCLES as f32;
    let mean_coh: f32 = coherence_trajectory.iter().sum::<f32>() / NUM_CYCLES as f32;
    let final_phi = *phi_trajectory.last().unwrap_or(&0.0);
    let first_20_phi: f64 = phi_trajectory[..20].iter().sum::<f64>() / 20.0;
    let last_20_phi: f64 = phi_trajectory[phi_trajectory.len() - 20..]
        .iter()
        .sum::<f64>()
        / 20.0;

    println!("  Results:");
    println!("    Mean Φ:          {:.4}", mean_phi);
    println!("    Final Φ:         {:.4}", final_phi);
    println!(
        "    Φ improvement:   {:.4} → {:.4} ({:+.1}%)",
        first_20_phi,
        last_20_phi,
        if first_20_phi > 1e-6 {
            (last_20_phi - first_20_phi) / first_20_phi * 100.0
        } else {
            0.0
        }
    );
    println!("    Mean PE:         {:.4}", mean_pe);
    println!("    Mean Coherence:  {:.4}", mean_coh);
    println!("    Learning cycles: {}/{}", total_learning, NUM_CYCLES);

    #[cfg(feature = "jepa")]
    {
        let mean_jepa_pe: f32 = jepa_pe_trajectory.iter().sum::<f32>() / NUM_CYCLES as f32;
        let first_20_jpe: f32 = jepa_pe_trajectory[..20].iter().sum::<f32>() / 20.0;
        let last_20_jpe: f32 = jepa_pe_trajectory[jepa_pe_trajectory.len() - 20..]
            .iter()
            .sum::<f32>()
            / 20.0;
        println!("\n  JEPA Telemetry:");
        println!("    Mean latent PE:  {:.4}", mean_jepa_pe);
        println!(
            "    JEPA PE trend:   {:.4} → {:.4}",
            first_20_jpe, last_20_jpe
        );
    }

    // Trajectory samples
    println!("\n  Trajectory (every 20 cycles):");
    #[cfg(feature = "jepa")]
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Cycle", "Phi", "PE", "Coh", "JEPA_PE"
    );
    #[cfg(not(feature = "jepa"))]
    println!("  {:>6}  {:>8}  {:>8}  {:>8}", "Cycle", "Phi", "PE", "Coh");

    for i in (0..NUM_CYCLES).step_by(20) {
        #[cfg(feature = "jepa")]
        println!(
            "  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.4}",
            i + 1,
            phi_trajectory[i],
            pe_trajectory[i],
            coherence_trajectory[i],
            jepa_pe_trajectory[i]
        );
        #[cfg(not(feature = "jepa"))]
        println!(
            "  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}",
            i + 1,
            phi_trajectory[i],
            pe_trajectory[i],
            coherence_trajectory[i]
        );
    }

    // JSON for automated comparison
    println!("\n--- JSON ---");
    println!(
        r#"{{"jepa_enabled": {}, "mean_phi": {:.6}, "final_phi": {:.6}, "mean_pe": {:.6}, "mean_coherence": {:.6}, "learning_cycles": {}, "cycles": {}}}"#,
        jepa_enabled, mean_phi, final_phi, mean_pe, mean_coh, total_learning, NUM_CYCLES
    );
}