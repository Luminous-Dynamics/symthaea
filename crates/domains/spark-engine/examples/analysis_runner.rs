// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Full Anomaly Investigation Pipeline
//!
//! This example runs the complete spark-engine analysis pipeline:
//! 1. Load literature database
//! 2. Analyze NASA results against standard physics
//! 3. Evaluate all hypotheses
//! 4. Calculate required conditions for each hypothesis
//! 5. Rank hypotheses by plausibility
//! 6. Generate experimental program
//! 7. Output comprehensive report

use spark_engine::anomaly::AnomalyAnalysis;
use spark_engine::constants::*;
use spark_engine::experimental_design::ExperimentDesigner;
use spark_engine::hypothesis_models::{
    HotSpotModel, HypothesisComparison, PhononCascadeModel, SuperScreeningModel,
};
use spark_engine::literature::{LiteratureDatabase, PatternAnalysis};
use spark_engine::physics::GamowIntegration;
use spark_engine::prelude::*;
use spark_engine::rate_gap::{
    ExperimentalConditions, HostMaterial, RateGapCalculator, TriggerType,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         SPARK ENGINE: LCF Anomaly Investigation Report           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Physics-Honest Assessment of Lattice Confinement Fusion         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // Section 1: Standard Physics Baseline
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 1: STANDARD GAMOW PHYSICS BASELINE                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let gamow_300k = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 0);
    let gamow_1000k = GamowIntegration::dd_rate(1000.0, SCREENING_PD_EV, 0);
    let gamow_10000k = GamowIntegration::dd_rate(10000.0, SCREENING_PD_EV, 0);

    println!("D-D Fusion Rate <σv> at Different Temperatures:");
    println!(
        "  • 300 K (room temp):   {:.2e} cm³/s",
        gamow_300k.sigma_v_cm3_s
    );
    println!(
        "  • 1,000 K:             {:.2e} cm³/s",
        gamow_1000k.sigma_v_cm3_s
    );
    println!(
        "  • 10,000 K:            {:.2e} cm³/s",
        gamow_10000k.sigma_v_cm3_s
    );
    println!();

    println!("Gamow Peak Analysis (300 K, Ue = {} eV):", SCREENING_PD_EV);
    println!(
        "  • Gamow peak energy:   {:.3} keV",
        gamow_300k.gamow_peak_kev
    );
    println!(
        "  • Gamow width:         {:.3} keV",
        gamow_300k.gamow_width_kev
    );
    println!(
        "  • Screening factor:    {:.2e}",
        gamow_300k.screening_enhancement
    );
    println!();

    // Q Factor
    let q_params = QFactorParams::lcf_typical();
    let (q_factor, achievable) = QFactor::compute(&gamow_300k, &q_params);
    println!("Q Factor Assessment (room temperature):");
    println!("  • Q = {:.2e}", q_factor);
    println!(
        "  • Energy gain achievable: {}",
        if achievable { "YES" } else { "NO" }
    );
    println!(
        "  • Verdict: {}",
        if q_factor < 1.0 {
            "NET ENERGY CONSUMER (Q < 1)"
        } else {
            "NET ENERGY PRODUCER (Q > 1)"
        }
    );
    println!();

    // ========================================================================
    // Section 2: Literature Survey
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 2: LITERATURE DATABASE ANALYSIS                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let db = LiteratureDatabase::canonical();
    let patterns = db.compute_patterns();

    println!("Database Statistics:");
    println!("  • Total observations:    {}", patterns.total_observations);
    println!(
        "  • Average credibility:   {:.1}/10",
        patterns.average_credibility
    );
    println!();

    println!("Observations with Neutron Detection:");
    for obs in db.with_neutrons() {
        println!(
            "  • {} ({}) - {} [credibility: {:.1}]",
            obs.author, obs.year, obs.venue, obs.credibility.score
        );
        if let Some(effect) = obs
            .effects
            .iter()
            .find(|e| matches!(e, spark_engine::literature::ObservedEffect::Neutrons { .. }))
        {
            if let spark_engine::literature::ObservedEffect::Neutrons {
                rate_per_s,
                energy_mev,
            } = effect
            {
                println!(
                    "    Rate: {:.0} n/s, Energy: {} MeV",
                    rate_per_s,
                    energy_mev
                        .map(|e| format!("{:.2}", e))
                        .unwrap_or("unknown".to_string())
                );
            }
        }
    }
    println!();

    println!("High-Credibility Observations (≥7.0):");
    for obs in db.by_credibility(7.0) {
        println!(
            "  • {} ({}) - credibility {:.1}: {}",
            obs.author,
            obs.year,
            obs.credibility.score,
            obs.notes.chars().take(60).collect::<String>()
        );
    }
    println!();

    // ========================================================================
    // Section 3: The Central Anomaly
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 3: THE CENTRAL ANOMALY - NASA vs GAMOW                  │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let nasa_analysis = RateGapCalculator::analyze_nasa_steinetz();

    println!("NASA Steinetz 2020 Observation:");
    println!(
        "  • Observed neutron rate:  {:.0} n/s",
        nasa_analysis.observed_rate
    );
    println!(
        "  • Gamow predicted rate:   {:.2e} n/s",
        nasa_analysis.gamow_predicted_rate
    );
    println!();

    println!("THE GAP:");
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!(
        "  ║  Gap factor: {:.2e} ({:.0} orders of magnitude)        ║",
        nasa_analysis.gap_factor, nasa_analysis.gap_orders
    );
    println!(
        "  ║  Classification: {:?}                          ║",
        nasa_analysis.assessment.classification
    );
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();

    println!("Gap Breakdown:");
    println!(
        "  • Screening enhancement:  {:.2e}×",
        nasa_analysis.gap_breakdown.screening_enhancement
    );
    println!(
        "  • Phonon enhancement:     {:.2e}×",
        nasa_analysis.gap_breakdown.phonon_enhancement
    );
    println!(
        "  • Hot spot enhancement:   {:.2e}×",
        nasa_analysis.gap_breakdown.hot_spot_enhancement
    );
    println!(
        "  • UNEXPLAINED:            {:.2e}× ({:.0} orders)",
        nasa_analysis.gap_breakdown.unexplained_factor,
        nasa_analysis.gap_breakdown.unexplained_orders
    );
    println!();

    // ========================================================================
    // Section 4: Hypothesis Evaluation
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 4: HYPOTHESIS EVALUATION                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let comparison = HypothesisComparison::compare_all();

    println!("Hypothesis 1: TRANSIENT HOT SPOTS");
    println!("  X-ray absorption creates brief high-T regions");
    let hot_spot = HotSpotModel::compute(12.0, 1e12);
    println!(
        "  • Predicted peak T:      {:.0} K",
        hot_spot.peak_temperature_k
    );
    println!(
        "  • Cooling time:          {:.2} ps",
        hot_spot.cooling_time_ps
    );
    println!(
        "  • Predicted rate:        {:.2e} n/s",
        comparison.hot_spot.predicted_rate
    );
    println!(
        "  • Gap from NASA:         {:.2e}×",
        comparison.hot_spot.gap_factor
    );
    println!(
        "  • Required T for NASA:   {:.2e} K",
        comparison.hot_spot.required_temperature_k
    );
    println!(
        "  • Plausible:             {}",
        if comparison.hot_spot.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("Hypothesis 2: COHERENT PHONON CASCADE");
    println!("  Multiple phonons coherently boost CM energy");
    let phonon = PhononCascadeModel::compute(3, 1.0);
    println!(
        "  • Energy boost (3 phonons): {:.1} meV",
        phonon.energy_boost_kev * 1000.0
    );
    println!(
        "  • Enhancement factor:    {:.2e}×",
        phonon.enhancement_factor
    );
    println!(
        "  • Predicted rate (3):    {:.2e} n/s",
        comparison.phonon_cascade.predicted_rate_n3
    );
    println!(
        "  • Gap from NASA:         {:.2e}×",
        comparison.phonon_cascade.gap_factor
    );
    println!(
        "  • Phonons needed:        {}",
        comparison.phonon_cascade.phonons_needed
    );
    println!(
        "  • Plausible:             {}",
        if comparison.phonon_cascade.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("Hypothesis 3: SUPER-SCREENING");
    println!("  Enhanced Ue at high D/Pd loading");
    let screening = SuperScreeningModel::compute(0.7, 0.0);
    println!(
        "  • Base screening:        {:.0} eV",
        screening.base_screening_ev
    );
    println!(
        "  • Predicted rate:        {:.2e} n/s",
        comparison.super_screening.predicted_rate
    );
    println!(
        "  • Gap from NASA:         {:.2e}×",
        comparison.super_screening.gap_factor
    );
    println!(
        "  • Screening needed:      {:.0} eV",
        comparison.super_screening.screening_needed_ev
    );
    println!(
        "  • Plausible:             {}",
        if comparison.super_screening.plausible {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    println!("BEST FIT HYPOTHESIS: {}", comparison.best_fit);
    println!();

    // ========================================================================
    // Section 5: Required Conditions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 5: WHAT WOULD EXPLAIN THE OBSERVATIONS?                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let conditions = ExperimentalConditions::default();
    let required_screening =
        RateGapCalculator::required_screening_for_rate(&conditions, NASA_OBSERVED_RATE);
    let required_temp =
        RateGapCalculator::required_temperature_for_rate(&conditions, NASA_OBSERVED_RATE);

    println!(
        "To explain {:.0} n/s from standard Gamow physics:",
        NASA_OBSERVED_RATE
    );
    println!();
    println!("  Option A: Increase screening energy");
    println!(
        "    • Required Ue:  {:.0} eV (vs measured {:.0} eV)",
        required_screening, SCREENING_PD_EV
    );
    println!(
        "    • Factor:       {:.0}× higher than measured",
        required_screening / SCREENING_PD_EV
    );
    println!(
        "    • Assessment:   {}",
        if required_screening < 10000.0 {
            "Potentially achievable"
        } else {
            "Implausible"
        }
    );
    println!();
    println!("  Option B: Increase temperature");
    println!(
        "    • Required T:   {:.2e} K (vs room temp 300 K)",
        required_temp
    );
    println!("    • Factor:       {:.0}× higher", required_temp / 300.0);
    println!(
        "    • Assessment:   {}",
        if required_temp < 1e6 {
            "Hot spots could achieve"
        } else {
            "Requires extreme conditions"
        }
    );
    println!();

    // ========================================================================
    // Section 6: Experimental Program
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 6: RECOMMENDED EXPERIMENTAL PROGRAM                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let program = ExperimentDesigner::design_program();

    println!("Program: {}", program.name);
    println!("Total Cost: ${:.0}K", program.total_cost_usd / 1000.0);
    println!(
        "Total Duration: {:.0} months",
        program.total_duration_months
    );
    println!();

    for (i, phase) in program.phases.iter().enumerate() {
        println!(
            "Phase {}: {} (${:.0}K, {} months)",
            i + 1,
            phase.name,
            phase.cost_usd / 1000.0,
            phase.duration_months
        );
        println!("  Goal: {}", phase.goal);
        println!("  Experiments:");
        for exp in &phase.experiments {
            println!("    • {} (priority: {:.2})", exp.name, exp.priority);
        }
        println!("  Success Criteria:");
        for criterion in &phase.success_criteria {
            println!("    ✓ {}", criterion);
        }
        println!("  Go/No-Go: {}", phase.go_nogo);
        println!();
    }

    // ========================================================================
    // Section 7: Conclusions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 7: CONCLUSIONS                                          │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("1. PHYSICS HONESTY");
    println!("   Standard Gamow physics predicts <σv> ~ 10⁻⁵⁰ cm³/s at room");
    println!("   temperature. This means Q << 1: LCF is a net energy CONSUMER,");
    println!("   not a producer, under standard physics.");
    println!();

    println!("2. THE ANOMALY IS REAL (if observations are valid)");
    println!(
        "   NASA observed ~10³ n/s, creating a {:.0}-order-of-magnitude gap.",
        nasa_analysis.gap_orders
    );
    println!("   This is either:");
    println!("   a) New physics beyond standard Gamow theory");
    println!("   b) Systematic measurement error");
    println!("   c) A combination of known enhancements (screening + hot spots + phonons)");
    println!();

    println!("3. MOST PLAUSIBLE EXPLANATION");
    println!("   {}", comparison.best_fit);
    println!();

    println!("4. PATH FORWARD");
    println!("   Even with Q < 1, LCF may be valuable as a COMPACT NEUTRON SOURCE");
    println!("   for applications like NAA, BNCT, and security screening where the");
    println!("   metric is neutrons-per-dollar, not energy gain.");
    println!();

    println!("5. CRITICAL NEXT STEP");
    println!("   Independent replication of NASA results with:");
    println!("   • Hydrogen control (should show no neutrons)");
    println!("   • Neutron spectroscopy (should show 2.45 MeV peak)");
    println!("   • Time-resolved diagnostics during X-ray trigger");
    println!();

    println!("══════════════════════════════════════════════════════════════════");
    println!("                    END OF ANALYSIS REPORT");
    println!("══════════════════════════════════════════════════════════════════");
}
