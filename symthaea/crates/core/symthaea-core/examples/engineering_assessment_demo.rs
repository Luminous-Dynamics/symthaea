// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Integrated Engineering Assessment Demo
//!
//! Combines uncertainty quantification, manufacturing assessment, and economic
//! analysis to provide a comprehensive engineering evaluation of Spark Engine designs.
//!
//! Run with: cargo run --example engineering_assessment_demo

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CapitalCosts, CoupledPhysicsEngine, EconomicEngine, FuelCosts, ManufacturingEngine, OmCosts,
    OperatingConditions, UncertaintyEngine,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              INTEGRATED ENGINEERING ASSESSMENT                       ║");
    println!("║              Spark Engine Comprehensive Evaluation                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Engineering Assessment 2024");

    // =========================================================================
    // Consumer Unit (5 kW D-D)
    // =========================================================================
    println!("\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CONSUMER UNIT: 5 kW D-D FUSION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let conditions = OperatingConditions::consumer();

    // 1. Base Physics Simulation
    println!("\n[1/4] Running coupled physics simulation...");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);
    let physics_result = physics.simulate(&conditions);

    println!("      Shell: {}", physics_result.shell_material);
    println!("      Core: {}", physics_result.core_material);
    println!(
        "      Max temp: {:.0} K",
        physics_result.thermal_profile.t_max
    );
    println!(
        "      Dose rate: {:.6} mSv/hr",
        physics_result.geometry_shielding.shielding.final_dose
    );
    println!(
        "      Total mass: {:.0} kg",
        physics_result.geometry_shielding.total_mass_kg
    );
    println!(
        "      Feasible: {}",
        if physics_result.feasible { "YES" } else { "NO" }
    );

    // 2. Uncertainty Quantification
    println!("\n[2/4] Running Monte Carlo uncertainty analysis (N=100)...");
    let uncertainty = UncertaintyEngine::new(&genesis);
    let mc_results = uncertainty.monte_carlo(&conditions, 100, 42);

    println!(
        "      Dose rate: {:.6} / {:.6} / {:.6} mSv/hr (5th/50th/95th)",
        mc_results.dose_stats.p5, mc_results.dose_stats.p50, mc_results.dose_stats.p95
    );
    println!(
        "      Temperature: {:.0} / {:.0} / {:.0} K",
        mc_results.temp_stats.p5, mc_results.temp_stats.p50, mc_results.temp_stats.p95
    );
    println!(
        "      Lifetime: {:.1} / {:.1} / {:.1} years",
        mc_results.lifetime_stats.p5, mc_results.lifetime_stats.p50, mc_results.lifetime_stats.p95
    );
    println!(
        "      Feasibility rate: {:.1}%",
        mc_results.feasibility_rate * 100.0
    );

    // Top sensitivities
    println!("\n      Most sensitive parameters:");
    for sens in mc_results.sensitivities.iter().take(3) {
        println!(
            "        {}. {} (dose: {:.2}, temp: {:.2})",
            sens.rank, sens.parameter, sens.dose_sensitivity, sens.temp_sensitivity
        );
    }

    // 3. Manufacturing Assessment
    println!("\n[3/4] Assessing manufacturing feasibility...");
    let manufacturing = ManufacturingEngine::new();
    let mfg_result = manufacturing.assess_consumer_spark();

    println!(
        "      Relative cost: {:.1}x baseline",
        mfg_result.total_cost
    );
    println!("      Lead time: {:.0} weeks", mfg_result.lead_time_weeks);
    println!("      Risk score: {:.0}%", mfg_result.risk_score * 100.0);
    println!(
        "      Assembly steps: {}",
        mfg_result.assembly_sequence.len()
    );

    if !mfg_result.compatibility_issues.is_empty() {
        println!("\n      Compatibility warnings:");
        for issue in &mfg_result.compatibility_issues {
            println!("        ⚠ {}", issue);
        }
    }

    // 4. Economic Analysis
    println!("\n[4/4] Calculating economic viability...");
    let econ = EconomicEngine::consumer(conditions.power_kw);
    let capital = CapitalCosts::consumer_5kw();
    let om = OmCosts::consumer();
    let fuel = FuelCosts::dd_fusion();

    let lcoe = econ.calculate_lcoe(&capital, &om, &fuel);
    let sensitivities = econ.sensitivity_analysis(&capital, &om, &fuel);

    println!(
        "      LCOE: ${:.1}/MWh (${:.3}/kWh)",
        lcoe.lcoe_usd_mwh, lcoe.lcoe_usd_kwh
    );
    println!("      Capital cost: ${:.0}", capital.total());
    println!(
        "      Cost per kW: ${:.0}/kW",
        capital.cost_per_kw(conditions.power_kw)
    );
    println!("      Payback vs grid: {:.1} years", lcoe.payback_years);
    println!("      NPV: ${:.0}", lcoe.npv_usd);

    // =========================================================================
    // Summary Dashboard
    // =========================================================================
    println!("\n\n");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    CONSUMER UNIT SUMMARY DASHBOARD");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Traffic light indicators
    let physics_status = if physics_result.feasible {
        "🟢 PASS"
    } else {
        "🔴 FAIL"
    };
    let uncertainty_status = if mc_results.feasibility_rate > 0.9 {
        "🟢 HIGH CONF"
    } else if mc_results.feasibility_rate > 0.7 {
        "🟡 MODERATE"
    } else {
        "🔴 LOW CONF"
    };
    let mfg_status = if mfg_result.risk_score < 0.3 {
        "🟢 LOW RISK"
    } else if mfg_result.risk_score < 0.5 {
        "🟡 MED RISK"
    } else {
        "🔴 HIGH RISK"
    };
    let econ_status = if lcoe.payback_years < 15.0 {
        "🟢 VIABLE"
    } else if lcoe.payback_years < 25.0 {
        "🟡 MARGINAL"
    } else {
        "🔴 UNVIABLE"
    };

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  CATEGORY              STATUS          KEY METRIC                   │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Physics Feasibility   {:12}    Max T = {:.0} K                │",
        physics_status, mc_results.temp_stats.p50
    );
    println!(
        "│  Design Confidence     {:12}    {:.0}% feasible (MC)           │",
        uncertainty_status,
        mc_results.feasibility_rate * 100.0
    );
    println!(
        "│  Manufacturing         {:12}    {:.0} weeks lead time          │",
        mfg_status, mfg_result.lead_time_weeks
    );
    println!(
        "│  Economic Viability    {:12}    {:.1} year payback             │",
        econ_status, lcoe.payback_years
    );
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  KEY PERFORMANCE INDICATORS (with 90% confidence intervals)        │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Power output:         5.0 kW (continuous)                         │");
    println!(
        "│  System mass:          {:.0} kg (±{:.0}%)                         │",
        mc_results.mass_stats.p50,
        mc_results.mass_stats.cv * 100.0
    );
    println!(
        "│  Dose rate:            {:.4} mSv/hr (95th: {:.4})             │",
        mc_results.dose_stats.p50, mc_results.dose_stats.p95
    );
    println!(
        "│  Design lifetime:      {:.0} years (95th: {:.0})                   │",
        mc_results.lifetime_stats.p50.min(100.0),
        mc_results.lifetime_stats.p5.min(100.0)
    );
    println!(
        "│  LCOE:                 ${:.1}/MWh ± ${}                         │",
        lcoe.lcoe_usd_mwh,
        (sensitivities[0].2 - lcoe.lcoe_usd_mwh).abs() as i64
    );
    println!(
        "│  Total investment:     ${:.0}                                 │",
        capital.total()
    );
    println!("└─────────────────────────────────────────────────────────────────────┘");

    // =========================================================================
    // Recommendations
    // =========================================================================
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  RECOMMENDATIONS                                                   │");
    println!("├─────────────────────────────────────────────────────────────────────┤");

    if physics_result.feasible && mc_results.feasibility_rate > 0.8 {
        println!("│  ✓ Design is technically feasible with high confidence            │");
    } else {
        println!("│  ⚠ Review limiting factors and uncertainty drivers                │");
    }

    if mfg_result.risk_score < 0.4 {
        println!("│  ✓ Manufacturing approach is well-characterized                   │");
    } else {
        println!("│  ⚠ Develop HEA supplier qualification program                     │");
    }

    if lcoe.payback_years < 20.0 {
        println!("│  ✓ Economics favorable compared to grid electricity               │");
    } else {
        println!("│  ⚠ Target capital cost reduction of 30-40% for viability         │");
    }

    println!("│                                                                     │");
    println!("│  Next steps:                                                        │");
    println!("│  1. Prototype validation of key subsystems                          │");
    println!("│  2. HEA supplier development and qualification                      │");
    println!("│  3. Regulatory pre-engagement for novel fusion device               │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ASSESSMENT COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}
