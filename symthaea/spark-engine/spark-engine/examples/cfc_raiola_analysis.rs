// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC Analysis of Raiola Screening Data
//!
//! Uses Closed-form Continuous-time neural networks to analyze the temporal
//! dynamics of electron screening in deuterated metals.
//!
//! Run with: `cargo run --example cfc_raiola_analysis`

use spark_engine::cfc_physics::{analyze_raiola_data, generate_screening_dynamics};
use spark_engine::bridge::LiteratureDataLoader;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  CfC TEMPORAL DYNAMICS ANALYSIS OF ELECTRON SCREENING");
    println!("  Analyzing Raiola et al. (2004) data through continuous-time networks");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Load literature data
    let loader = LiteratureDataLoader::new();
    println!("Loaded {} screening measurements from literature\n", loader.screening_data.len());

    // Show a sample of the synthetic dynamics
    if let Some(pd_measurement) = loader.screening_data.iter().find(|m| m.host_material == "Pd") {
        println!("▶ SAMPLE DYNAMICS: Palladium Deuteride\n");

        let dynamics = generate_screening_dynamics(pd_measurement);

        println!("  Static screening: {:.0} eV", dynamics.static_screening_ev);
        println!("  Enhancement: {:.1}× over adiabatic limit", pd_measurement.enhancement_ratio);
        println!("\n  Time scales:");
        println!("    τ_electron:  {:.2e} s (electronic response)", dynamics.time_scales.tau_electron);
        println!("    τ_phonon:    {:.2e} s (lattice response)", dynamics.time_scales.tau_phonon);
        println!("    τ_screening: {:.2e} s (screening buildup)", dynamics.time_scales.tau_screening);
        println!("    τ_fusion:    {:.2e} s (nuclear time scale)", dynamics.time_scales.tau_fusion);

        // Show a few points from the time series
        println!("\n  Screening dynamics (first 10 points):");
        println!("    {:>12}  {:>12}  {:>12}", "Time (s)", "Ue (eV)", "n_phonons");
        for i in (0..100).step_by(10) {
            let (t, ue) = dynamics.screening_series[i];
            let (_, n_ph) = dynamics.phonon_series[i];
            println!("    {:>12.3e}  {:>12.1}  {:>12.2}", t, ue, n_ph);
        }
        println!();
    }

    // Run full CfC analysis
    println!("Running CfC analysis on all {} materials...\n", loader.screening_data.len());
    let report = analyze_raiola_data();

    // Print the full report
    println!("{}", report.summary());

    // Additional analysis: compare dynamics between high and low enhancement materials
    println!("\n─────────────────────────────────────────────────────────────────────────");
    println!("DETAILED COMPARISON: HIGH vs LOW ENHANCEMENT MATERIALS");
    println!("─────────────────────────────────────────────────────────────────────────\n");

    let mut sorted_results = report.material_results.clone();
    sorted_results.sort_by(|a, b| b.enhancement_ratio.partial_cmp(&a.enhancement_ratio).unwrap());

    println!("Top 3 enhancement:");
    for r in sorted_results.iter().take(3) {
        println!(
            "  {}: {:.1}× enhancement, τ = {:.2e} s, phonon_corr = {:.2}",
            r.material, r.enhancement_ratio, r.dominant_time_scale, r.temporal_features.phonon_correlation
        );
    }

    println!("\nBottom 3 enhancement:");
    for r in sorted_results.iter().rev().take(3) {
        println!(
            "  {}: {:.1}× enhancement, τ = {:.2e} s, phonon_corr = {:.2}",
            r.material, r.enhancement_ratio, r.dominant_time_scale, r.temporal_features.phonon_correlation
        );
    }

    // Interpretation
    println!("\n─────────────────────────────────────────────────────────────────────────");
    println!("PHYSICAL INTERPRETATION");
    println!("─────────────────────────────────────────────────────────────────────────\n");

    println!("The CfC analysis reveals temporal structure in screening dynamics:\n");

    println!("1. ELECTRONIC TIME SCALES");
    println!("   Materials with heavier atoms (Ta, Au, Pt) show slower electron response.");
    println!("   This affects how quickly screening builds up during a collision.\n");

    println!("2. PHONON COUPLING");
    println!("   The phonon-screening correlation measures how much lattice vibrations");
    println!("   modulate the screening potential. High coupling suggests phonons");
    println!("   actively participate in the enhancement mechanism.\n");

    println!("3. DYNAMIC SCREENING HYPOTHESIS");
    println!("   If screening were purely static, τ wouldn't correlate with enhancement.");
    println!("   Correlation r = {:.2} suggests temporal dynamics matter.",
             report.correlations.tau_enhancement_corr);

    if report.correlations.tau_enhancement_corr.abs() > 0.3 {
        println!("   => Dynamic screening effects may be significant!\n");
    } else {
        println!("   => Static screening appears to dominate.\n");
    }

    println!("4. NEXT STEPS");
    println!("   To test the dynamic screening hypothesis experimentally:");
    println!("   - Use pump-probe spectroscopy to measure time-resolved screening");
    println!("   - Compare pulsed vs continuous triggers at different timescales");
    println!("   - Look for resonance effects at phonon frequencies\n");

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Analysis complete. CfC learned {} temporal features per material.",
             report.material_results.first().map(|r| r.learned_tau.len()).unwrap_or(0));
    println!("═══════════════════════════════════════════════════════════════════════");
}
