// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Scaling Study Demo
//!
//! Runs a parametric scaling analysis of LCF reactor performance across
//! 6 orders of magnitude (1W to 1MW), identifying optimal architectures,
//! trigger systems, and scaling laws.
//!
//! ## Output
//!
//! - Per-power-level design summary (architecture, trigger, mass, cost)
//! - Fitted scaling laws (Y = A × P^n) for volume, mass, cost, etc.
//! - Architecture transition points
//! - Parametric sensitivity analysis
//!
//! Run with: cargo run --example scaling_study_demo

use symthaea_core::physics::{
    POWER_LEVELS_W, RagoneComparison, ReadinessComparison, ScaleAwareComparison, ScalingStudy,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           LCF REACTOR SCALING STUDY: 1W → 1MW                      ║");
    println!("║           Parametric Analysis Across 6 Orders of Magnitude          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create the study
    let study = ScalingStudy::new();

    println!(
        "Running full scaling study across {} power levels...",
        POWER_LEVELS_W.len()
    );
    println!(
        "  Range: {:.0}W to {:.0}MW",
        POWER_LEVELS_W[0],
        POWER_LEVELS_W[6] / 1e6
    );
    println!();

    // Run the study
    let results = study.run_full_study();

    // Generate and print the built-in report
    let report = study.generate_report(&results);
    println!("{}", report);

    // Run parametric sweep for sensitivity analysis
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("  PARAMETRIC SENSITIVITY ANALYSIS");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // Sweep lifetime at 1kW: 5, 10, 20, 30, 50 years
    let lifetime_values = [5.0, 10.0, 20.0, 30.0, 50.0];
    let sweep = study.parametric_sweep(1_000.0, "lifetime", &lifetime_values);

    println!("  Lifetime sensitivity at 1kW:");
    println!(
        "  {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Lifetime(yr)", "Mass(kg)", "Cost($)", "$/W", "Architecture"
    );
    println!("  {}", "-".repeat(70));
    for (val, point) in &sweep {
        println!(
            "  {:>12.0} {:>12.1} {:>12.0} {:>12.2} {:>15?}",
            val, point.mass_kg, point.cost_usd, point.specific_cost, point.architecture
        );
    }

    // Sweep cost budget at 1kW
    let budget_values = [10_000.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0];
    let sweep = study.parametric_sweep(1_000.0, "cost_budget", &budget_values);

    println!();
    println!("  Cost budget sensitivity at 1kW:");
    println!(
        "  {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Budget($)", "Mass(kg)", "Actual($)", "$/W", "Architecture"
    );
    println!("  {}", "-".repeat(70));
    for (val, point) in &sweep {
        println!(
            "  {:>12.0} {:>12.1} {:>12.0} {:>12.2} {:>15?}",
            val, point.mass_kg, point.cost_usd, point.specific_cost, point.architecture
        );
    }

    // Summary statistics
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("  SCALING LAW SUMMARY");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Fitted Laws (Y = A × P^n):");
    println!(
        "  {:20} {:>8} {:>8}   {}",
        "Parameter", "n", "R²", "Interpretation"
    );
    println!("  {}", "-".repeat(75));
    for law in &results.scaling_laws {
        println!(
            "  {:20} {:>8.3} {:>8.3}   {}",
            law.parameter, law.exponent, law.r_squared, law.interpretation
        );
    }

    println!();
    println!("  Architecture Transitions:");
    for (power, arch, reason) in &results.transitions {
        println!("    {:.0}W → {:?}: {}", power, arch, reason);
    }

    println!();
    println!("  Trigger Transitions:");
    for (power, trigger, reason) in &results.trigger_transitions {
        println!("    {:.0}W → {:?}: {}", power, trigger, reason);
    }

    println!();
    if let (Some(first), Some(last)) = (results.data.first(), results.data.last()) {
        println!(
            "  Cost reduction: ${:.0}/W (1W) → ${:.2}/W (1MW) = {:.0}× improvement",
            first.specific_cost,
            last.specific_cost,
            first.specific_cost / last.specific_cost
        );
        println!(
            "  Power density:  {:.0} W/m³ (1W) → {:.0} W/m³ (1MW) = {:.1}× improvement",
            first.power_density,
            last.power_density,
            last.power_density / first.power_density
        );
    }

    println!();
    println!("  Key Findings:");
    for (i, finding) in results.findings.iter().enumerate() {
        println!("    {}. {}", i + 1, finding);
    }

    println!();
    println!("  Recommendations:");
    for (application, rec) in &results.recommendations {
        println!("    [{}] {}", application, rec);
    }

    // === Direction D: Comparative Benchmarking ===
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("  COMPARATIVE BENCHMARKING (Direction D)");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // Build Ragone comparison data from scaling study results
    let lcf_data: Vec<(String, f64, f64, f64)> = results
        .data
        .iter()
        .map(|dp| {
            let lifetime_hours = dp.lifetime_years * 8760.0;
            (
                format!("LCF {:.0}W", dp.power_w),
                dp.power_w,
                dp.mass_kg,
                lifetime_hours,
            )
        })
        .collect();

    let ragone = RagoneComparison::build(&lcf_data);

    println!("  RAGONE PLOT DATA (Specific Power vs Energy Density)");
    println!(
        "  {:25} {:>12} {:>15} {:>18}",
        "Technology", "W/kg", "Wh/kg", "Category"
    );
    println!("  {}", "-".repeat(75));
    for point in &ragone.data_points {
        println!(
            "  {:25} {:>12.1} {:>15.0} {:>18}",
            point.name, point.specific_power_w_kg, point.energy_density_wh_kg, point.category
        );
    }

    // Scale-aware cost comparison
    let lcf_cost_data: Vec<(f64, f64, f64)> = results
        .data
        .iter()
        .map(|dp| {
            // Estimate LCOE from specific cost (rough: $/W × CRF / CF / 8.76)
            let lcoe_approx = dp.specific_cost * 0.07 / 0.85 / 8.76 * 1000.0;
            (dp.power_w, dp.cost_usd, lcoe_approx)
        })
        .collect();

    let scale_comparisons = ScaleAwareComparison::build_all(&POWER_LEVELS_W, &lcf_cost_data);

    println!();
    println!("  SCALE-AWARE COST COMPARISON");
    for comp in scale_comparisons.iter().take(3) {
        // Show first 3 power levels
        println!();
        println!("  At {:.0}W:", comp.power_w);
        println!("    {:25} {:>10} {:>12}", "Technology", "$/W", "LCOE $/MWh");
        println!("    {}", "-".repeat(50));
        for tech in &comp.technologies {
            println!(
                "    {:25} {:>10.2} {:>12.0}",
                tech.technology, tech.cost_per_w, tech.lcoe_usd_mwh
            );
        }
    }

    // Technology readiness comparison
    println!();
    println!("  TECHNOLOGY READINESS COMPARISON");
    println!(
        "  {:30} {:>5} {:>10} {:>12}",
        "Technology", "TRL", "Years", "R&D ($M)"
    );
    println!("  {}", "-".repeat(65));

    let lcf = ReadinessComparison::lcf_assessment();
    println!(
        "  {:30} {:>5} {:>10.0} {:>12.0}",
        lcf.technology,
        lcf.trl,
        lcf.years_to_deployment,
        lcf.rd_funding_needed_usd / 1e6
    );

    for comp in ReadinessComparison::competing_technologies() {
        println!(
            "  {:30} {:>5} {:>10.0} {:>12.0}",
            comp.technology,
            comp.trl,
            comp.years_to_deployment,
            comp.rd_funding_needed_usd / 1e6
        );
    }

    println!();
    println!("  LCF Key Risks:");
    for (i, risk) in lcf.key_risks.iter().enumerate() {
        println!("    {}. {}", i + 1, risk);
    }

    println!();
    println!(
        "Study complete. {} data points, {} scaling laws, {} transitions.",
        results.data.len(),
        results.scaling_laws.len(),
        results.transitions.len()
    );
}
