// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine Engineering Demo
//!
//! Complete integrated demonstration of the Spark Engine design system:
//! 1. Coupled multi-physics simulation
//! 2. Prototype specification generation
//! 3. Consumer (5 kW) and Industrial (100 MW) unit designs
//!
//! Run with: cargo run --example spark_engineering_demo

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{CoupledPhysicsEngine, OperatingConditions, PrototypeSpecification};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           SPARK ENGINE INTEGRATED DESIGN SYSTEM                      ║");
    println!("║           Coupled Physics → Prototype Specification                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let genesis = GenesisSeed::from_phrase("Spark Engine v1.0");
    let engine = CoupledPhysicsEngine::from_genesis(&genesis);

    // =========================================================================
    // CONSUMER UNIT: 5 kW Home Fusion Generator
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DESIGN 1: Consumer Home Unit (5 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let consumer_conditions = OperatingConditions::consumer();
    println!("Target Specifications:");
    println!("  Power: {:.0} kW thermal", consumer_conditions.power_kw);
    println!("  Fuel: {:?}", consumer_conditions.reaction);
    println!(
        "  Lifetime: {:.0} years",
        consumer_conditions.target_lifetime_years
    );
    println!(
        "  Max dose rate: {:.4} mSv/hr",
        consumer_conditions.max_dose_rate
    );
    println!();

    // Run coupled simulation
    println!("Running coupled multi-physics simulation...");
    let consumer_result = engine.simulate(&consumer_conditions);

    // Display simulation results
    println!();
    println!("Simulation Results:");
    println!("  Materials:");
    println!("    Shell: {}", consumer_result.shell_material);
    println!("    Interface: {}", consumer_result.interface_material);
    println!("    Core: {}", consumer_result.core_material);
    println!();
    println!("  Thermal Profile:");
    println!(
        "    Max temperature: {:.0} K ({:.0}°C)",
        consumer_result.thermal_profile.t_max,
        consumer_result.thermal_profile.t_max - 273.15
    );
    println!(
        "    Surface temperature: {:.0} K ({:.0}°C)",
        consumer_result.thermal_profile.t_shell_outer,
        consumer_result.thermal_profile.t_shell_outer - 273.15
    );
    println!(
        "    Effective healing rate: {:.2e} DPA/s",
        consumer_result.effective_healing_rate
    );
    println!();
    println!("  Geometry & Shielding:");
    println!(
        "    Core radius: {:.1} cm",
        consumer_result.geometry_shielding.geometry.core_radius * 100.0
    );
    println!(
        "    Total radius: {:.1} cm",
        consumer_result.geometry_shielding.geometry.outer_radius * 100.0
    );
    println!(
        "    Shielding: {} @ {:.1} cm",
        consumer_result
            .geometry_shielding
            .shielding
            .primary_material,
        consumer_result.geometry_shielding.shielding.thickness_m * 100.0
    );
    println!(
        "    System mass: {:.1} kg",
        consumer_result.geometry_shielding.total_mass_kg
    );
    println!();
    println!("  Lifetime Analysis:");
    println!(
        "    Operating mode: Pulsed @ {:.0}% duty",
        consumer_result.pulse_thermal.pulse.duty_cycle * 100.0
    );
    println!(
        "    Equilibrium DPA: {:.2}",
        consumer_result.pulse_thermal.equilibrium_dpa
    );
    println!(
        "    Radiation lifetime: {} years",
        format_lifetime(consumer_result.pulse_thermal.lifetime_years)
    );
    println!(
        "    Fatigue lifetime: {} years",
        format_lifetime(consumer_result.pulse_thermal.fatigue_lifetime_years)
    );
    println!(
        "    Limiting factor: {:?}",
        consumer_result.pulse_thermal.limiting_factor
    );
    println!();

    // Assessment
    println!(
        "  Assessment: {}",
        if consumer_result.feasible {
            "✓ FEASIBLE"
        } else {
            "✗ NEEDS REVISION"
        }
    );
    if !consumer_result.limiting_factors.is_empty() {
        println!("  Issues:");
        for factor in &consumer_result.limiting_factors {
            println!("    • {}", factor);
        }
    }
    println!("  Recommendations:");
    for rec in &consumer_result.recommendations {
        println!("    • {}", rec);
    }
    println!();

    // Generate prototype specification
    println!("Generating prototype specification...");
    let consumer_spec = PrototypeSpecification::from_simulation(&consumer_result);
    println!();
    println!("{}", consumer_spec.to_string());

    // =========================================================================
    // INDUSTRIAL UNIT: 100 MW Power Plant Module
    // =========================================================================
    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DESIGN 2: Industrial Power Module (100 MW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let industrial_conditions = OperatingConditions::industrial();
    println!("Target Specifications:");
    println!(
        "  Power: {:.0} MW thermal",
        industrial_conditions.power_kw / 1000.0
    );
    println!("  Fuel: {:?}", industrial_conditions.reaction);
    println!(
        "  Lifetime: {:.0} years",
        industrial_conditions.target_lifetime_years
    );
    println!(
        "  Max dose rate: {:.4} mSv/hr (occupational)",
        industrial_conditions.max_dose_rate
    );
    println!();

    println!("Running coupled multi-physics simulation...");
    let industrial_result = engine.simulate(&industrial_conditions);

    println!();
    println!("Simulation Results:");
    println!("  Materials:");
    println!("    Shell: {}", industrial_result.shell_material);
    println!("    Interface: {}", industrial_result.interface_material);
    println!("    Core: {}", industrial_result.core_material);
    println!();
    println!("  Thermal Profile:");
    println!(
        "    Max temperature: {:.0} K ({:.0}°C)",
        industrial_result.thermal_profile.t_max,
        industrial_result.thermal_profile.t_max - 273.15
    );
    println!(
        "    Surface temperature: {:.0} K ({:.0}°C)",
        industrial_result.thermal_profile.t_shell_outer,
        industrial_result.thermal_profile.t_shell_outer - 273.15
    );
    println!();
    println!("  Geometry & Shielding:");
    println!(
        "    Core radius: {:.1} m",
        industrial_result.geometry_shielding.geometry.core_radius
    );
    println!(
        "    Total radius: {:.1} m",
        industrial_result.geometry_shielding.geometry.outer_radius
    );
    println!(
        "    Shielding: {} @ {:.1} cm",
        industrial_result
            .geometry_shielding
            .shielding
            .primary_material,
        industrial_result.geometry_shielding.shielding.thickness_m * 100.0
    );
    println!(
        "    System mass: {:.0} kg ({:.1} tonnes)",
        industrial_result.geometry_shielding.total_mass_kg,
        industrial_result.geometry_shielding.total_mass_kg / 1000.0
    );
    println!();
    println!("  Lifetime Analysis:");
    println!(
        "    Operating mode: Pulsed @ {:.0}% duty",
        industrial_result.pulse_thermal.pulse.duty_cycle * 100.0
    );
    println!(
        "    Radiation lifetime: {} years",
        format_lifetime(industrial_result.pulse_thermal.lifetime_years)
    );
    println!(
        "    Fatigue lifetime: {} years",
        format_lifetime(industrial_result.pulse_thermal.fatigue_lifetime_years)
    );
    println!();

    println!(
        "  Assessment: {}",
        if industrial_result.feasible {
            "✓ FEASIBLE"
        } else {
            "✗ NEEDS REVISION"
        }
    );
    if !industrial_result.limiting_factors.is_empty() {
        println!("  Issues:");
        for factor in &industrial_result.limiting_factors {
            println!("    • {}", factor);
        }
    }
    println!();

    // Generate specification
    println!("Generating prototype specification...");
    let industrial_spec = PrototypeSpecification::from_simulation(&industrial_result);
    println!();
    println!("{}", industrial_spec.to_string());

    // =========================================================================
    // COMPARISON SUMMARY
    // =========================================================================
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     DESIGN COMPARISON SUMMARY                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "{:30} {:>20} {:>20}",
        "Parameter", "Consumer (5 kW)", "Industrial (100 MW)"
    );
    println!("{}", "─".repeat(70));
    println!(
        "{:30} {:>20} {:>20}",
        "Model",
        consumer_spec.model.full_name(),
        industrial_spec.model.full_name()
    );
    println!(
        "{:30} {:>20.1} {:>20.1}",
        "Thermal Power (kW)", consumer_spec.thermal_power_kw, industrial_spec.thermal_power_kw
    );
    println!(
        "{:30} {:>20.1} {:>20.1}",
        "Electrical Output (kW)",
        consumer_spec.electrical_output_kw,
        industrial_spec.electrical_output_kw
    );
    println!(
        "{:30} {:>20.1} {:>20.1}",
        "Total Diameter (m)",
        consumer_result.geometry_shielding.geometry.outer_radius * 2.0,
        industrial_result.geometry_shielding.geometry.outer_radius * 2.0
    );
    println!(
        "{:30} {:>20.1} {:>20.1}",
        "System Mass (kg)",
        consumer_result.geometry_shielding.total_mass_kg,
        industrial_result.geometry_shielding.total_mass_kg
    );
    println!(
        "{:30} {:>20} {:>20}",
        "Predicted Lifetime",
        format_lifetime(consumer_result.pulse_thermal.lifetime_years),
        format_lifetime(industrial_result.pulse_thermal.lifetime_years)
    );
    println!(
        "{:30} {:>17.0} USD {:>17.0} USD",
        "Est. Material Cost",
        consumer_spec.total_material_cost_usd,
        industrial_spec.total_material_cost_usd
    );
    println!(
        "{:30} {:>17.0} USD {:>17.0} USD",
        "Est. Total Cost",
        consumer_spec.estimated_total_cost_usd,
        industrial_spec.estimated_total_cost_usd
    );
    println!(
        "{:30} {:>15.0} USD/kW {:>15.0} USD/kW",
        "Cost per kW",
        consumer_spec.estimated_total_cost_usd / consumer_spec.thermal_power_kw,
        industrial_spec.estimated_total_cost_usd / industrial_spec.thermal_power_kw
    );
    println!(
        "{:30} {:>20} {:>20}",
        "Feasibility",
        if consumer_result.feasible {
            "✓ YES"
        } else {
            "✗ NO"
        },
        if industrial_result.feasible {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!();

    // Final notes
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  NOTES:");
    println!("  • Consumer unit uses D-D fusion (lower neutron flux, simpler fuel)");
    println!("  • Industrial unit uses D-T fusion (higher power density, tritium handling)");
    println!("  • Cost estimates are preliminary and exclude R&D, certification, etc.");
    println!("  • All designs include self-healing HEA shell and liquid metal core");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}

fn format_lifetime(years: f64) -> String {
    if years.is_infinite() || years > 1000.0 {
        "∞".to_string()
    } else if years > 100.0 {
        format!(">{:.0}", 100.0)
    } else {
        format!("{:.1}", years)
    }
}
