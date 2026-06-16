// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Industrial Scale Assessment (100 MW D-T)
//!
//! Run with: cargo run --example industrial_assessment

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CapitalCosts, CoupledPhysicsEngine, EconomicEngine, FuelCosts, OmCosts, OperatingConditions,
    UncertaintyEngine,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              INDUSTRIAL SCALE ASSESSMENT                             ║");
    println!("║              100 MW D-T Fusion Power Plant                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Industrial Assessment 2024");

    println!("\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INDUSTRIAL UNIT: 100 MW D-T FUSION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let conditions = OperatingConditions::industrial();

    // 1. Base Physics Simulation
    println!("\n[1/3] Running coupled physics simulation...");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);
    let result = physics.simulate(&conditions);

    println!("      Reaction: D-T (14.1 MeV neutrons)");
    println!("      Power: {:.0} MW", conditions.power_kw / 1000.0);
    println!("      Shell: {}", result.shell_material);
    println!("      Core: {}", result.core_material);
    println!("      Max temp: {:.0} K", result.thermal_profile.t_max);
    println!(
        "      Dose rate: {:.6} mSv/hr",
        result.geometry_shielding.shielding.final_dose
    );
    println!(
        "      Total mass: {:.0} kg ({:.1} tonnes)",
        result.geometry_shielding.total_mass_kg,
        result.geometry_shielding.total_mass_kg / 1000.0
    );
    println!(
        "      Shielding: {} ({:.2} m)",
        result.geometry_shielding.shielding.primary_material,
        result.geometry_shielding.shielding.thickness_m
    );
    println!(
        "      Feasible: {}",
        if result.feasible { "YES" } else { "NO" }
    );

    if !result.limiting_factors.is_empty() {
        println!("\n      Limiting factors:");
        for factor in &result.limiting_factors {
            println!("        ⚠ {}", factor);
        }
    }

    // 2. Uncertainty Quantification (fewer samples for industrial - expensive)
    println!("\n[2/3] Running Monte Carlo uncertainty analysis (N=50)...");
    let uncertainty = UncertaintyEngine::new(&genesis);
    let mc_results = uncertainty.monte_carlo(&conditions, 50, 42);

    println!(
        "      Dose rate: {:.6} / {:.6} / {:.6} mSv/hr (5th/50th/95th)",
        mc_results.dose_stats.p5, mc_results.dose_stats.p50, mc_results.dose_stats.p95
    );
    println!(
        "      Temperature: {:.0} / {:.0} / {:.0} K",
        mc_results.temp_stats.p5, mc_results.temp_stats.p50, mc_results.temp_stats.p95
    );
    println!(
        "      Feasibility rate: {:.1}%",
        mc_results.feasibility_rate * 100.0
    );

    // 3. Economic Analysis
    println!("\n[3/3] Calculating economic viability...");
    let econ = EconomicEngine::industrial(conditions.power_kw);
    let capital = CapitalCosts::industrial_100mw();
    let om = OmCosts::industrial();
    let fuel = FuelCosts::dt_fusion();

    let lcoe = econ.calculate_lcoe(&capital, &om, &fuel);

    println!(
        "      LCOE: ${:.1}/MWh (${:.4}/kWh)",
        lcoe.lcoe_usd_mwh, lcoe.lcoe_usd_kwh
    );
    println!("      Capital cost: ${:.0}M", capital.total() / 1e6);
    println!(
        "      Cost per kW: ${:.0}/kW",
        capital.cost_per_kw(conditions.power_kw)
    );
    println!(
        "      Payback vs wholesale: {:.1} years",
        lcoe.payback_years
    );
    println!("      NPV: ${:.0}M", lcoe.npv_usd / 1e6);

    // Summary
    println!("\n\n");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    INDUSTRIAL UNIT COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  METRIC                 CONSUMER (5kW)     INDUSTRIAL (100MW)       │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Power                  5 kW               100,000 kW               │");
    println!("│  Reaction               D-D                D-T                      │");
    println!(
        "│  Mass                   ~2,800 kg          {:.0} tonnes             │",
        result.geometry_shielding.total_mass_kg / 1000.0
    );
    println!(
        "│  Specific mass          560 kg/kW          {:.2} kg/kW              │",
        result.geometry_shielding.total_mass_kg / conditions.power_kw
    );
    println!(
        "│  Dose rate              0.001 mSv/hr       {:.4} mSv/hr           │",
        result.geometry_shielding.shielding.final_dose
    );
    println!(
        "│  LCOE                   $631/MWh           ${:.1}/MWh              │",
        lcoe.lcoe_usd_mwh
    );
    println!(
        "│  Feasible               YES                {}                     │",
        if result.feasible { "YES" } else { "NO " }
    );
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let physics_status = if result.feasible {
        "🟢 PASS"
    } else {
        "🔴 FAIL"
    };
    let econ_status = if lcoe.lcoe_usd_mwh < 100.0 {
        "🟢 COMPETITIVE"
    } else if lcoe.lcoe_usd_mwh < 200.0 {
        "🟡 MARGINAL"
    } else {
        "🔴 UNVIABLE"
    };

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  INDUSTRIAL ASSESSMENT SUMMARY                                      │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Physics Feasibility:   {:12}                                │",
        physics_status
    );
    println!(
        "│  Economic Status:       {:12}                                │",
        econ_status
    );
    println!("│  Key Advantage:         Economy of scale - lower $/kW              │");
    println!("│  Key Challenge:         14 MeV neutron shielding                   │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ASSESSMENT COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}
