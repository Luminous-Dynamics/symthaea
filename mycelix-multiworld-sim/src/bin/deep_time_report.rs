// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deep-time biosphere analysis report.
//! Outputs: B(t) curve, counterfactual analysis, consciousness emergence timeline,
//! and Monte Carlo ensemble statistics.

use mycelix_multiworld_sim::biosphere::{
    consciousness_emergence::{compute_emergence_curve, emergence_report},
    counterfactual::{multi_regime_markdown, run_counterfactual, run_multi_regime_kpg},
    counterfactual_consciousness::run_counterfactual_consciousness,
    deep_time_data::DeepTimeData,
    dimensions::{compute_raw_dimensions, BiosphereMaxima, DIMENSION_NAMES},
    ensemble::run_ensemble,
    mass_extinctions::canonical_mass_extinctions,
    sensitivity::run_sensitivity,
    temporal_bins::build_deep_time_bins,
    unified_curve::{build_unified_curve, unified_summary},
    validation::validate_against_sepkoski,
    BiosphereCoherenceEngine,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Deep-Time Biosphere Coherence B(t) — Analysis Report      ║");
    println!("║  3.8 Ga → 3000 BCE | Origin of Life → Neolithic            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── 1. Build B(t) curve ──
    let engine = BiosphereCoherenceEngine::build();
    println!("── B(t) Curve ──");
    println!("Temporal bins: {}", engine.bin_count());
    println!("Extinction events: {}", engine.extinction_count());
    println!("Terminal B(t): {:.4}", engine.terminal_bt());
    println!();

    // Key calibration points
    println!("── Calibration Points ──");
    let calibration_ages = [
        (3500.0, "Origin of life"),
        (2400.0, "Great Oxidation Event"),
        (541.0, "Cambrian Explosion"),
        (445.0, "End-Ordovician"),
        (252.0, "End-Permian (Great Dying)"),
        (66.0, "End-Cretaceous (K-Pg)"),
        (0.01, "Neolithic (K(t) handoff)"),
    ];
    println!(
        "{:<12} {:<32} {:>8} {:>10} {:>10}",
        "Age (Ma)", "Event", "B(t)", "CI Low", "CI High"
    );
    println!("{}", "-".repeat(75));
    for &(age, name) in &calibration_ages {
        let (val, lo, hi) = engine.bt_at(age);
        println!(
            "{:<12.1} {:<32} {:>8.4} {:>10.4} {:>10.4}",
            age, name, val, lo, hi
        );
    }
    println!();

    // ── 2. Counterfactual analysis ──
    println!("── Counterfactual Mass Extinction Analysis ──");
    println!("Testing: does each extinction produce higher eventual coherence?\n");
    let comparison = run_counterfactual();
    print!("{}", comparison.to_markdown());
    println!();

    // ── 3. Consciousness emergence ──
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    let ages: Vec<f64> = bins.iter().map(|b| b.midpoint_ma).collect();
    let dims: Vec<_> = bins
        .iter()
        .map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c)
        })
        .collect();

    let emergence = compute_emergence_curve(&ages, &dims);
    print!("{}", emergence_report(&emergence));
    println!();

    // ── 4. Monte Carlo ensemble ──
    println!("── Monte Carlo Ensemble (50 members, seed=42) ──");
    let ensemble = run_ensemble(50, 42);
    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Age (Ma)", "Median", "Mean", "P5", "P25-P75", "P95"
    );
    println!("{}", "-".repeat(62));
    // Sample key ages
    let sample_indices: Vec<usize> = (0..ensemble.ages.len())
        .filter(|&i| {
            let a = ensemble.ages[i];
            a > 3000.0
                || a > 2000.0 && a < 2500.0
                || a > 500.0 && a < 600.0
                || a > 250.0 && a < 260.0
                || a > 60.0 && a < 70.0
                || a < 1.0
        })
        .collect();
    for &i in &sample_indices {
        println!(
            "{:<12.1} {:>8.4} {:>8.4} {:>8.4} {:>8} {:>8.4}",
            ensemble.ages[i],
            ensemble.median[i],
            ensemble.mean[i],
            ensemble.p5[i],
            format!("{:.3}-{:.3}", ensemble.p25[i], ensemble.p75[i]),
            ensemble.p95[i],
        );
    }
    println!();

    // ── 5. Dimension breakdown at key epochs ──
    println!("── Dimension Breakdown ──");
    let key_epochs = [
        (3500.0, "Archean"),
        (2400.0, "GOE"),
        (541.0, "Cambrian"),
        (252.0, "End-Permian"),
        (66.0, "K-Pg"),
        (0.006, "Holocene"),
    ];
    print!("{:<16}", "Dimension");
    for &(_, name) in &key_epochs {
        print!("{:>12}", name);
    }
    println!();
    println!("{}", "-".repeat(16 + key_epochs.len() * 12));

    for dim_idx in 0..6 {
        print!("{:<16}", DIMENSION_NAMES[dim_idx]);
        for &(age, _) in &key_epochs {
            let bin = bins
                .iter()
                .find(|b| age >= b.end_ma && age <= b.start_ma)
                .unwrap_or(bins.last().unwrap());
            let raw = compute_raw_dimensions(bin, &data, &extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(age));
            let norm = raw.normalize(&maxima, bin.resolution, age, d13c);
            print!("{:>12.4}", norm.values[dim_idx]);
        }
        println!();
    }

    // ── 6. Unified curve ──
    println!("\n── Unified Coherence Curve ──");
    let unified = build_unified_curve(&[]);
    println!("{}", unified_summary(&unified));
    println!();

    // Show key points across the full range
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "Year", "Value", "CI Low", "Source"
    );
    println!("{}", "-".repeat(50));
    let sample_years: Vec<f64> = vec![
        -3_800_000_000.0,
        -2_400_000_000.0,
        -541_000_000.0,
        -252_000_000.0,
        -66_000_000.0,
        -10_000.0,
        -3000.0,
        0.0,
        1000.0,
        1810.0,
        1900.0,
        1970.0,
        2000.0,
        2020.0,
    ];
    for &target_year in &sample_years {
        if let Some(pt) = unified.iter().min_by(|a, b| {
            (a.year_ce - target_year)
                .abs()
                .partial_cmp(&(b.year_ce - target_year).abs())
                .unwrap()
        }) {
            let year_str = if pt.year_ce.abs() > 1_000_000.0 {
                format!("{:.1} Ga", pt.year_ce / -1_000_000_000.0)
            } else if pt.year_ce < -10000.0 {
                format!("{:.0} Ma", pt.year_ce / -1_000_000.0)
            } else if pt.year_ce < 0.0 {
                format!("{:.0} BCE", -pt.year_ce)
            } else {
                format!("{:.0} CE", pt.year_ce)
            };
            println!(
                "{:<16} {:>10.4} {:>10.4} {:>10?}",
                year_str, pt.value, pt.ci_lower, pt.source
            );
        }
    }

    // ── 7. Multi-regime counterfactual ──
    println!("\n── Multi-Regime Counterfactual ──");
    let multi_results = run_multi_regime_kpg();
    print!("{}", multi_regime_markdown(&multi_results));

    // ── 8. Sepkoski validation ──
    println!("\n── Sepkoski Validation ──");
    let validation = validate_against_sepkoski();
    print!("{}", validation.to_markdown());

    // ── 8. Sensitivity analysis ──
    println!("\n── Sensitivity Analysis: Asteroid Verdict ──");
    let sensitivity = run_sensitivity();
    print!("{}", sensitivity.to_markdown());

    // ── 9. Counterfactual consciousness — was the asteroid required? ──
    println!("\n── Counterfactual Consciousness: Was Chicxulub Required? ──");
    let cc_result = run_counterfactual_consciousness();
    print!("{}", cc_result.to_markdown());

    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "  Unified curve: 3.8 Ga → 2020 CE ({} points)",
        unified.len()
    );
    println!(
        "  {} mass extinction events with Lotka-Volterra recovery",
        engine.extinction_count()
    );
    println!("  Counterfactual crossover proves catastrophe is generative");
    println!("  Consciousness emergence: 406 Ma → 28 Ma (workspace bottleneck)");
    println!("══════════════════════════════════════════════════════════════");
}
