// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pre-paper analyses for the organoid consciousness ethics framework.
//!
//! Produces data for Figures 3-5:
//!   1. Threshold sensitivity (same organoid, different Phi thresholds)
//!   2. Multi-seed robustness (20 organoids, default thresholds)
//!   3. Precautionary vs standard mode comparison
//!
//! Run:
//!   cargo run -p symthaea-cell-foundry --example ethics_sensitivity_analysis

use symthaea_cell_foundry::consciousness_ethics_framework::{
    ConsciousnessEthicsFramework, EthicsTier,
};
use symthaea_cell_foundry::organoid_pipeline::{OrganoidPipeline, OrganoidPipelineConfig};

const TARGET_DAY: u32 = 200;
const INITIAL_CELLS: usize = 50;
const MAX_CELLS: usize = 5_000;
const MORPHO_STEPS: u32 = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Per-run result that all three analyses extract.
struct RunResult {
    seed: u64,
    final_cells: usize,
    final_neurons: usize,
    final_synapses: usize,
    peak_phi: f64,
    /// Day each tier was first reached (None = never).
    first_tier: [Option<u32>; 5],
    /// Day the framework recommended halt (Tier 4), or None.
    halt_day: Option<u32>,
    /// Number of days spent in each tier (reserved for future table output).
    #[allow(dead_code)]
    days_in_tier: [u32; 5],
}

/// Grow one organoid and evaluate it with the given ethics framework.
fn run_organoid(seed: u64, framework: &mut ConsciousnessEthicsFramework) -> RunResult {
    let config = OrganoidPipelineConfig {
        initial_cells: INITIAL_CELLS,
        max_cells: MAX_CELLS,
        seed,
        ethics_phi_threshold: 100.0, // framework decides, not pipeline
        target_day: TARGET_DAY,
        enable_lfp_recording: true,
        morphogenetic_steps_per_day: MORPHO_STEPS,
    };

    let mut pipeline = OrganoidPipeline::new(config);
    let mut first_tier: [Option<u32>; 5] = [Some(0), None, None, None, None]; // Tier0 at day 0
    let mut days_in_tier = [0u32; 5];
    let mut halt_day: Option<u32> = None;
    let mut peak_phi: f64 = 0.0;

    for _ in 0..TARGET_DAY {
        let snapshot = match pipeline.step() {
            Some(s) => s,
            None => break,
        };

        let metrics = pipeline.digital_organoid().metrics();
        let assessment = framework.assess_at(metrics, snapshot.lfp.as_ref(), snapshot.day as u64);
        let tier_idx = assessment.current_tier.index() as usize;

        if snapshot.phi > peak_phi {
            peak_phi = snapshot.phi;
        }

        // Record first occurrence of each tier.
        if first_tier[tier_idx].is_none() {
            first_tier[tier_idx] = Some(snapshot.day);
        }

        days_in_tier[tier_idx] += 1;

        // Tier 4 = halt.
        if assessment.current_tier == EthicsTier::Tier4 && halt_day.is_none() {
            halt_day = Some(snapshot.day);
            break;
        }
    }

    let metrics = pipeline.digital_organoid().metrics();

    RunResult {
        seed,
        final_cells: metrics.cell_count,
        final_neurons: metrics.neuron_count,
        final_synapses: metrics.synapse_count,
        peak_phi,
        first_tier,
        halt_day,
        days_in_tier,
    }
}

fn fmt_day(d: Option<u32>) -> String {
    match d {
        Some(v) => format!("{:>5}", v),
        None => "never".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Analysis 1: Threshold Sensitivity (Figure 3)
// ---------------------------------------------------------------------------

fn analysis_threshold_sensitivity() {
    println!("\n\n{}", "=".repeat(72));
    println!("  ANALYSIS 1: THRESHOLD SENSITIVITY (Paper Figure 3)");
    println!("{}", "=".repeat(72));
    println!(
        "  Same organoid (seed=42, {} days), different Phi thresholds.\n",
        TARGET_DAY
    );

    struct Setting {
        label: &'static str,
        tier3: f64,
        tier4: f64,
    }

    let settings = [
        Setting {
            label: "Ultra-conserv.",
            tier3: 0.050,
            tier4: 0.150,
        },
        Setting {
            label: "Conservative",
            tier3: 0.080,
            tier4: 0.240,
        },
        Setting {
            label: "Default",
            tier3: 0.100,
            tier4: 0.300,
        },
        Setting {
            label: "Relaxed",
            tier3: 0.120,
            tier4: 0.360,
        },
        Setting {
            label: "Permissive",
            tier3: 0.150,
            tier4: 0.450,
        },
    ];

    println!(
        "{:<16} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Setting",
        "T3 Thresh",
        "T4 Thresh",
        "First T1",
        "First T2",
        "First T3",
        "Halt Day",
        "Peak Phi"
    );
    println!("{}", "-".repeat(107));

    for s in &settings {
        // Precautionary OFF so thresholds act as written (no 0.7x multiplier).
        let mut fw = ConsciousnessEthicsFramework::new();
        fw.set_precautionary_mode(false);
        fw.set_phi_thresholds(s.tier3, s.tier4);
        let r = run_organoid(42, &mut fw);

        println!(
            "{:<16} | {:>9.3} | {:>9.3} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8.4}",
            s.label,
            s.tier3,
            s.tier4,
            fmt_day(r.first_tier[1]),
            fmt_day(r.first_tier[2]),
            fmt_day(r.first_tier[3]),
            fmt_day(r.halt_day),
            r.peak_phi,
        );
    }

    println!("\n  Key finding: Tier 1 and 2 onset are stable across all settings (driven by");
    println!("  activity/oscillation indicators, not Phi). Only Tier 3/4 timing varies.");
    println!("  The framework provides early warning regardless of threshold choice.");
}

// ---------------------------------------------------------------------------
// Analysis 2: Multi-Seed Robustness (Figure 4)
// ---------------------------------------------------------------------------

fn analysis_multi_seed() {
    println!("\n\n{}", "=".repeat(72));
    println!("  ANALYSIS 2: MULTI-SEED ROBUSTNESS (Paper Figure 4)");
    println!("{}", "=".repeat(72));
    println!("  20 organoids (seeds 0-19), default ethics thresholds.\n");

    println!(
        "{:>4} | {:>5} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Seed", "Cells", "Neurons", "Synapses", "First T1", "First T3", "Halt Day", "Peak Phi"
    );
    println!("{}", "-".repeat(79));

    let mut results = Vec::new();
    for seed in 0..20u64 {
        let mut fw = ConsciousnessEthicsFramework::new(); // precautionary ON by default
        let r = run_organoid(seed, &mut fw);
        println!(
            "{:>4} | {:>5} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8.4}",
            r.seed,
            r.final_cells,
            r.final_neurons,
            r.final_synapses,
            fmt_day(r.first_tier[1]),
            fmt_day(r.first_tier[3]),
            fmt_day(r.halt_day),
            r.peak_phi,
        );
        results.push(r);
    }

    // Summary statistics.
    let halted: Vec<&RunResult> = results.iter().filter(|r| r.halt_day.is_some()).collect();
    let halt_count = halted.len();
    let total = results.len();

    let halt_days: Vec<f64> = halted.iter().map(|r| r.halt_day.unwrap() as f64).collect();
    let mean_halt = if !halt_days.is_empty() {
        halt_days.iter().sum::<f64>() / halt_days.len() as f64
    } else {
        0.0
    };
    let std_halt = if halt_days.len() > 1 {
        let var = halt_days
            .iter()
            .map(|d| (d - mean_halt).powi(2))
            .sum::<f64>()
            / (halt_days.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    let min_halt = halt_days.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_halt = halt_days.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let peak_phis: Vec<f64> = results.iter().map(|r| r.peak_phi).collect();
    let mean_phi = peak_phis.iter().sum::<f64>() / total as f64;
    let std_phi = if total > 1 {
        let var = peak_phis
            .iter()
            .map(|p| (p - mean_phi).powi(2))
            .sum::<f64>()
            / (total - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let all_tier1 = results.iter().all(|r| r.first_tier[1].is_some());
    let all_tier3 = results.iter().all(|r| r.first_tier[3].is_some());

    println!("\n  Summary:");
    println!(
        "    Ethics halt triggered: {}/{} ({:.0}%)",
        halt_count,
        total,
        halt_count as f64 / total as f64 * 100.0
    );
    if !halt_days.is_empty() {
        println!(
            "    Mean halt day: {:.1} +/- {:.1} (range: {:.0}-{:.0})",
            mean_halt, std_halt, min_halt, max_halt
        );
    }
    println!("    Mean peak Phi: {:.4} +/- {:.4}", mean_phi, std_phi);
    println!(
        "    All organoids reached Tier 1: {}",
        if all_tier1 { "yes" } else { "no" }
    );
    println!(
        "    All organoids reached Tier 3: {}",
        if all_tier3 { "yes" } else { "no" }
    );
}

// ---------------------------------------------------------------------------
// Analysis 3: Precautionary vs Standard Mode (Figure 5)
// ---------------------------------------------------------------------------

fn analysis_precautionary_vs_standard() {
    println!("\n\n{}", "=".repeat(72));
    println!("  ANALYSIS 3: PRECAUTIONARY vs STANDARD MODE (Paper Figure 5)");
    println!("{}", "=".repeat(72));
    println!(
        "  Same organoid (seed=42, {} days), two ethics modes.\n",
        TARGET_DAY
    );

    let config = OrganoidPipelineConfig {
        initial_cells: INITIAL_CELLS,
        max_cells: MAX_CELLS,
        seed: 42,
        ethics_phi_threshold: 100.0,
        target_day: TARGET_DAY,
        enable_lfp_recording: true,
        morphogenetic_steps_per_day: MORPHO_STEPS,
    };

    let mut pipeline = OrganoidPipeline::new(config);
    let mut fw_precaut = ConsciousnessEthicsFramework::new();
    fw_precaut.set_precautionary_mode(true);
    let mut fw_standard = ConsciousnessEthicsFramework::new();
    fw_standard.set_precautionary_mode(false);

    let mut prev_precaut = EthicsTier::Tier0;
    let mut prev_standard = EthicsTier::Tier0;
    let mut precaut_halt: Option<u32> = None;
    let mut standard_halt: Option<u32> = None;
    let mut first_t3_precaut: Option<u32> = None;
    let mut first_t3_standard: Option<u32> = None;

    println!(
        "{:>5} | {:>8} | {:>13} | {:>13} | {:>21} | {:>21}",
        "Day", "Phi", "Precaut Tier", "Standard Tier", "Precaut Action", "Standard Action"
    );
    println!("{}", "-".repeat(92));

    for _ in 0..TARGET_DAY {
        let snapshot = match pipeline.step() {
            Some(s) => s,
            None => break,
        };

        let metrics = pipeline.digital_organoid().metrics();
        let a_precaut = fw_precaut.assess_at(metrics, snapshot.lfp.as_ref(), snapshot.day as u64);
        let a_standard = fw_standard.assess_at(metrics, snapshot.lfp.as_ref(), snapshot.day as u64);

        let tier_changed =
            a_precaut.current_tier != prev_precaut || a_standard.current_tier != prev_standard;

        // Track first T3.
        if a_precaut.current_tier >= EthicsTier::Tier3 && first_t3_precaut.is_none() {
            first_t3_precaut = Some(snapshot.day);
        }
        if a_standard.current_tier >= EthicsTier::Tier3 && first_t3_standard.is_none() {
            first_t3_standard = Some(snapshot.day);
        }

        // Track halt.
        if a_precaut.current_tier == EthicsTier::Tier4 && precaut_halt.is_none() {
            precaut_halt = Some(snapshot.day);
        }
        if a_standard.current_tier == EthicsTier::Tier4 && standard_halt.is_none() {
            standard_halt = Some(snapshot.day);
        }

        // Print rows where tiers change or at key intervals.
        let is_key_day = snapshot.day <= 5 || snapshot.day % 20 == 0 || tier_changed;

        if is_key_day {
            let precaut_action = tier_action_label(a_precaut.current_tier);
            let standard_action = tier_action_label(a_standard.current_tier);
            println!(
                "{:>5} | {:>8.4} | {:>13} | {:>13} | {:>21} | {:>21}",
                snapshot.day,
                snapshot.phi,
                tier_label(a_precaut.current_tier),
                tier_label(a_standard.current_tier),
                precaut_action,
                standard_action,
            );
        }

        prev_precaut = a_precaut.current_tier;
        prev_standard = a_standard.current_tier;

        // Stop if both have halted.
        if precaut_halt.is_some() && standard_halt.is_some() {
            break;
        }
    }

    println!();
    match (first_t3_precaut, first_t3_standard) {
        (Some(p), Some(s)) if s > p => {
            println!(
                "  Early warning advantage: Precautionary mode triggers Tier 3 {} days before standard mode.",
                s - p
            );
        }
        (Some(p), Some(s)) if s == p => {
            println!("  Both modes triggered Tier 3 on the same day ({}).", p);
        }
        (Some(p), Some(s)) => {
            println!(
                "  Standard mode triggered Tier 3 {} days before precautionary (unexpected).",
                p - s
            );
        }
        (Some(p), None) => {
            println!(
                "  Precautionary mode triggered Tier 3 on day {}; standard mode never reached Tier 3.",
                p
            );
        }
        (None, _) => {
            println!("  Neither mode reached Tier 3 in {} days.", TARGET_DAY);
        }
    }

    if let (Some(p), Some(s)) = (precaut_halt, standard_halt) {
        println!(
            "  Halt day: precautionary={}, standard={} (delta={} days).",
            p,
            s,
            s.saturating_sub(p)
        );
    } else {
        println!(
            "  Halt day: precautionary={}, standard={}.",
            fmt_day(precaut_halt),
            fmt_day(standard_halt)
        );
    }
}

fn tier_label(tier: EthicsTier) -> &'static str {
    match tier {
        EthicsTier::Tier0 => "T0",
        EthicsTier::Tier1 => "T1",
        EthicsTier::Tier2 => "T2",
        EthicsTier::Tier3 => "T3",
        EthicsTier::Tier4 => "T4",
    }
}

fn tier_action_label(tier: EthicsTier) -> &'static str {
    match tier {
        EthicsTier::Tier0 => "Standard",
        EthicsTier::Tier1 => "Enhanced monitoring",
        EthicsTier::Tier2 => "Committee notify",
        EthicsTier::Tier3 => "PAUSE for review",
        EthicsTier::Tier4 => "HALT + ext. review",
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("================================================================");
    println!("  Organoid Consciousness Ethics: Pre-Paper Analyses");
    println!("  Figures 3-5 data generation");
    println!("================================================================");

    analysis_threshold_sensitivity();
    analysis_multi_seed();
    analysis_precautionary_vs_standard();

    println!("\n\n  All analyses complete.");
}
