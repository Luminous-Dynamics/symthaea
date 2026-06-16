// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Three-Way Threshold Comparison on Real Cognitive Loop
//!
//! Compares 1000 cognitive cycles across three threshold configurations:
//! 1. Defaults — compile-time constants
//! 2. Consistency-evolved — from heuristic fitness (internal coherence only)
//! 3. Spectral-Phi-evolved — from spectral gap Phi + multi-objective Pareto
//!
//! This is the definitive test: does Phi-directed evolution outperform
//! heuristic evolution on the real consciousness pipeline?

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const CYCLES: usize = 1000;

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

struct RunResult {
    name: String,
    mean_phi: f64,
    final_phi: f64,
    first_100_phi: f64,
    last_100_phi: f64,
    mean_pe: f64,
    mean_coherence: f64,
    learning_cycles: usize,
}

fn run_with_label(name: &str, overrides: Option<Overrides>) -> RunResult {
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).expect("failed to create CLS");

    if let Some(ov) = overrides {
        let t = service.threshold_overrides_mut();
        t.fep_surprise_scale = Some(ov.fep_surprise_scale);
        t.fep_lr_decay = Some(ov.fep_lr_decay);
        t.neuromod_d2_baseline = Some(ov.neuromod_d2_baseline);
        t.neuromod_ne_phasic_threshold = Some(ov.neuromod_ne_phasic_threshold);
        t.neuromod_arousal_ema_decay = Some(ov.neuromod_arousal_ema_decay);
        t.homeostasis_recalibrate_high = Some(ov.homeostasis_recalibrate_high);
        t.homeostasis_recalibrate_low = Some(ov.homeostasis_recalibrate_low);
        t.frustration_dampen_threshold = Some(ov.frustration_dampen_threshold);
        t.self_model_weight_high = Some(ov.self_model_weight_high);
        t.homeostasis_pull_cruise = Some(ov.homeostasis_pull_cruise);
        println!("    Applied {} overrides", t.active_count());
    }

    let mut phi_values = Vec::with_capacity(CYCLES);
    let mut total_pe = 0.0f64;
    let mut total_coh = 0.0f64;
    let mut learn_count = 0usize;

    for i in 0..CYCLES {
        let input = INPUTS[i % INPUTS.len()];
        let result = service.cycle(input);
        let phi = result.metadata.consciousness.consciousness_level;
        phi_values.push(phi);
        total_pe += result.prediction_error as f64;
        total_coh += result.metadata.prediction_coherence as f64;
        if result.learning_occurred {
            learn_count += 1;
        }

        if (i + 1) % 200 == 0 {
            let recent_phi: f64 = phi_values[phi_values.len() - 50..].iter().sum::<f64>() / 50.0;
            println!("    cycle {}: Φ={:.4} (recent 50 avg)", i + 1, recent_phi);
        }
    }

    let n = CYCLES as f64;
    let mean_phi = phi_values.iter().sum::<f64>() / n;
    let final_phi = *phi_values.last().unwrap_or(&0.0);
    let first_100 = phi_values[..100].iter().sum::<f64>() / 100.0;
    let last_100 = phi_values[CYCLES - 100..].iter().sum::<f64>() / 100.0;

    RunResult {
        name: name.to_string(),
        mean_phi,
        final_phi,
        first_100_phi: first_100,
        last_100_phi: last_100,
        mean_pe: total_pe / n,
        mean_coherence: total_coh / n,
        learning_cycles: learn_count,
    }
}

struct Overrides {
    fep_surprise_scale: f32,
    fep_lr_decay: f32,
    neuromod_d2_baseline: f64,
    neuromod_ne_phasic_threshold: f32,
    neuromod_arousal_ema_decay: f32,
    homeostasis_recalibrate_high: f32,
    homeostasis_recalibrate_low: f32,
    frustration_dampen_threshold: f64,
    self_model_weight_high: f32,
    homeostasis_pull_cruise: f32,
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Three-Way Threshold Comparison");
    println!(
        "  {} cycles per configuration, {} unique inputs",
        CYCLES,
        INPUTS.len()
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    // Run 1: Defaults
    println!("  [1/3] Running DEFAULTS ({} cycles)...", CYCLES);
    let defaults = run_with_label("Defaults", None);

    // Run 2: Consistency-evolved (from evolve_thresholds output)
    println!(
        "\n  [2/3] Running CONSISTENCY-EVOLVED ({} cycles)...",
        CYCLES
    );
    let consistency = run_with_label(
        "Consistency",
        Some(Overrides {
            fep_surprise_scale: 6.75,
            fep_lr_decay: 0.872,
            neuromod_d2_baseline: 0.599,
            neuromod_ne_phasic_threshold: 0.162,
            neuromod_arousal_ema_decay: 0.883,
            homeostasis_recalibrate_high: 1.160,
            homeostasis_recalibrate_low: 0.923,
            frustration_dampen_threshold: 0.701,
            self_model_weight_high: 0.672,
            homeostasis_pull_cruise: 0.689,
        }),
    );

    // Run 3: Spectral-Phi-evolved (from evolve_for_phi with spectral proxy)
    println!(
        "\n  [3/3] Running SPECTRAL-PHI-EVOLVED ({} cycles)...",
        CYCLES
    );
    let spectral = run_with_label(
        "Spectral-Phi",
        Some(Overrides {
            fep_surprise_scale: 1.055,
            fep_lr_decay: 0.926,
            neuromod_d2_baseline: 0.861,
            neuromod_ne_phasic_threshold: 0.681,
            neuromod_arousal_ema_decay: 0.898,
            homeostasis_recalibrate_high: 1.232,
            homeostasis_recalibrate_low: 0.703,
            frustration_dampen_threshold: 0.490,
            self_model_weight_high: 0.763,
            homeostasis_pull_cruise: 1.498,
        }),
    );

    // Run 4: CLS-evolved "vigilant learner" (from evolve_cls)
    println!("\n  [4/4] Running CLS-EVOLVED ({} cycles)...", CYCLES);
    let cls_evolved = run_with_label(
        "CLS-Evolved",
        Some(Overrides {
            fep_surprise_scale: 9.360,
            fep_lr_decay: 0.922,
            neuromod_d2_baseline: 0.244,
            neuromod_ne_phasic_threshold: 0.197,
            neuromod_arousal_ema_decay: 0.936,
            homeostasis_recalibrate_high: 1.273,
            homeostasis_recalibrate_low: 0.725,
            frustration_dampen_threshold: 0.490,
            self_model_weight_high: 0.423,
            homeostasis_pull_cruise: 1.440,
        }),
    );

    // Results table
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results (4-way comparison)");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!(
        "  {:>16}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Metric", "Default", "Consist", "Spectr", "CLS-Evo"
    );
    println!("  {}", "-".repeat(56));

    fn row(name: &str, a: f64, b: f64, c: f64, d: f64) {
        let db = if a.abs() > 1e-6 {
            (b - a) / a * 100.0
        } else {
            0.0
        };
        let dc = if a.abs() > 1e-6 {
            (c - a) / a * 100.0
        } else {
            0.0
        };
        let dd = if a.abs() > 1e-6 {
            (d - a) / a * 100.0
        } else {
            0.0
        };
        println!(
            "  {:>16}  {:>8.4}  {:>+7.1}%  {:>+7.1}%  {:>+7.1}%",
            name, a, db, dc, dd
        );
    }

    row(
        "Mean Φ",
        defaults.mean_phi,
        consistency.mean_phi,
        spectral.mean_phi,
        cls_evolved.mean_phi,
    );
    row(
        "Final Φ",
        defaults.final_phi,
        consistency.final_phi,
        spectral.final_phi,
        cls_evolved.final_phi,
    );
    row(
        "First 100 Φ",
        defaults.first_100_phi,
        consistency.first_100_phi,
        spectral.first_100_phi,
        cls_evolved.first_100_phi,
    );
    row(
        "Last 100 Φ",
        defaults.last_100_phi,
        consistency.last_100_phi,
        spectral.last_100_phi,
        cls_evolved.last_100_phi,
    );
    row(
        "Mean PE",
        defaults.mean_pe,
        consistency.mean_pe,
        spectral.mean_pe,
        cls_evolved.mean_pe,
    );
    row(
        "Mean Coherence",
        defaults.mean_coherence,
        consistency.mean_coherence,
        spectral.mean_coherence,
        cls_evolved.mean_coherence,
    );

    // Winner determination
    println!("\n  Winner (by last 100 Φ):");
    let results = [&defaults, &consistency, &spectral, &cls_evolved];
    let best = results
        .iter()
        .max_by(|a, b| a.last_100_phi.partial_cmp(&b.last_100_phi).unwrap())
        .unwrap();
    println!("    {} — Φ={:.4}", best.name, best.last_100_phi);
}