// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! D-He3 Aneutronic Fusion Comparison
//!
//! Compares D-He3 (aneutronic) with D-D and D-T fusion for shielding requirements.
//!
//! Run with: cargo run --example aneutronic_comparison

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CapitalCosts, CoupledPhysicsEngine, EconomicEngine, FuelCosts, OmCosts, OperatingConditions,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              D-He3 ANEUTRONIC FUSION COMPARISON                      ║");
    println!("║              Neutron Shielding vs Ignition Temperature               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Aneutronic Comparison 2024");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CONSUMER SCALE COMPARISON (5 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // D-D Consumer
    let dd_conditions = OperatingConditions::consumer();
    let dd_result = physics.simulate(&dd_conditions);

    // D-He3 Consumer
    let dhe3_conditions = OperatingConditions::aneutronic_consumer();
    let dhe3_result = physics.simulate(&dhe3_conditions);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  METRIC              D-D (2.45 MeV n)    D-He3 (aneutronic)       │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  Neutron Energy      2.45 MeV            None (protons only)      │");
    println!("│  Ignition Temp       15 keV              58 keV (3.9× harder)     │");
    println!(
        "│  Dose Rate           {:<8.4} mSv/hr     {:<8.6} mSv/hr         │",
        dd_result.geometry_shielding.shielding.final_dose,
        dhe3_result.geometry_shielding.shielding.final_dose
    );
    println!(
        "│  Total Mass          {:<8.0} kg         {:<8.0} kg               │",
        dd_result.geometry_shielding.total_mass_kg, dhe3_result.geometry_shielding.total_mass_kg
    );
    println!(
        "│  Shielding           {:<8.2} m          {:<8.2} m                │",
        dd_result.geometry_shielding.shielding.thickness_m,
        dhe3_result.geometry_shielding.shielding.thickness_m
    );
    println!(
        "│  Shell Material      {}           {}               │",
        &dd_result.shell_material[..13.min(dd_result.shell_material.len())],
        &dhe3_result.shell_material[..13.min(dhe3_result.shell_material.len())]
    );
    println!(
        "│  Max Temp            {:<8.0} K           {:<8.0} K                │",
        dd_result.thermal_profile.t_max, dhe3_result.thermal_profile.t_max
    );
    println!(
        "│  Feasible            {}                {}                    │",
        if dd_result.feasible { "YES" } else { "NO " },
        if dhe3_result.feasible { "YES" } else { "NO " }
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INDUSTRIAL SCALE COMPARISON (100 MW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // D-T Industrial
    let dt_conditions = OperatingConditions::industrial();
    let dt_result = physics.simulate(&dt_conditions);

    // D-He3 Industrial
    let dhe3_ind_conditions = OperatingConditions::aneutronic_industrial();
    let dhe3_ind_result = physics.simulate(&dhe3_ind_conditions);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  METRIC              D-T (14.1 MeV n)   D-He3 (aneutronic)        │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  Neutron Energy      14.1 MeV           None (protons only)       │");
    println!("│  Ignition Temp       10 keV             58 keV (5.8× harder)      │");
    println!(
        "│  Dose Rate           {:<8.4} mSv/hr    {:<8.6} mSv/hr          │",
        dt_result.geometry_shielding.shielding.final_dose,
        dhe3_ind_result.geometry_shielding.shielding.final_dose
    );
    println!(
        "│  Total Mass          {:<8.0} t         {:<8.0} t                │",
        dt_result.geometry_shielding.total_mass_kg / 1000.0,
        dhe3_ind_result.geometry_shielding.total_mass_kg / 1000.0
    );
    println!(
        "│  Shielding           {:<8.2} m         {:<8.2} m                 │",
        dt_result.geometry_shielding.shielding.thickness_m,
        dhe3_ind_result.geometry_shielding.shielding.thickness_m
    );
    println!(
        "│  Max Temp            {:<8.0} K          {:<8.0} K                 │",
        dt_result.thermal_profile.t_max, dhe3_ind_result.thermal_profile.t_max
    );
    println!(
        "│  Feasible            {}                {}                     │",
        if dt_result.feasible { "YES" } else { "NO " },
        if dhe3_ind_result.feasible {
            "YES"
        } else {
            "NO "
        }
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    // Economic comparison
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  ECONOMIC COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Consumer economics
    let econ_dd = EconomicEngine::consumer(dd_conditions.power_kw);
    let capital_dd = CapitalCosts::consumer_5kw();
    let om_dd = OmCosts::consumer();
    let fuel_dd = FuelCosts::dd_fusion();
    let lcoe_dd = econ_dd.calculate_lcoe(&capital_dd, &om_dd, &fuel_dd);

    // D-He3 needs He-3 fuel which is expensive (~$15k/g)
    let fuel_dhe3 = FuelCosts::dhe3_fusion();
    let lcoe_dhe3 = econ_dd.calculate_lcoe(&capital_dd, &om_dd, &fuel_dhe3);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  LCOE COMPARISON (5 kW Consumer)                                  │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  D-D Fusion:   ${:.1}/MWh                                       │",
        lcoe_dd.lcoe_usd_mwh
    );
    println!(
        "│  D-He3 Fusion: ${:.1}/MWh (He-3 fuel cost dominant)           │",
        lcoe_dhe3.lcoe_usd_mwh
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    // Summary
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ANEUTRONIC FUSION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  ADVANTAGES OF D-He3 (Aneutronic)                                  │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  ✓ No neutron shielding required (minimal mass)                    │");
    println!("│  ✓ No neutron activation (easier decommissioning)                  │");
    println!("│  ✓ Direct energy conversion possible (higher efficiency)           │");
    println!("│  ✓ No tritium handling (reduced safety requirements)               │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  CHALLENGES OF D-He3                                               │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  ✗ 5.8× higher ignition temperature than D-T                       │");
    println!("│  ✗ He-3 costs ~$15M/kg (vs $30k/kg for tritium)                    │");
    println!("│  ✗ He-3 availability limited (lunar mining required?)              │");
    println!("│  ✗ Lower cross-section (harder to sustain reaction)                │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let recommendation = if dhe3_result.feasible && dd_result.feasible {
        if dhe3_result.geometry_shielding.total_mass_kg
            < dd_result.geometry_shielding.total_mass_kg * 0.5
        {
            "D-He3 offers significant mass advantage if ignition can be achieved"
        } else {
            "D-D preferred for near-term due to lower ignition requirements"
        }
    } else {
        "Further analysis needed"
    };

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  RECOMMENDATION                                                     │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  {}  │", format!("{:<65}", recommendation));
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    COMPARISON COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}
