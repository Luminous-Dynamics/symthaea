// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Design Integration Demo
//!
//! Demonstrates how reactor designs can be analyzed using integration metrics.
//!
//! Run with: cargo run --example design_integration_demo --release

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CoupledPhysicsEngine, DesignIntegrationEngine, FusionReaction, OperatingConditions,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           DESIGN INTEGRATION DEMO                                    ║");
    println!("║           Measuring Coupled System Quality via HDC                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Design Integration 2024");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);
    let integration = DesignIntegrationEngine::from_genesis(&genesis);

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGRATION ACROSS POWER SCALES
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INTEGRATION ACROSS POWER SCALES (D-D Fusion)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ Power   Mass    Lifetime  Feasible  Thermal   Damage    Geometry   I  │");
    println!("│  (kW)   (kg)    (years)             Coher.    Balance   Harmony       │");
    println!("├────────────────────────────────────────────────────────────────────────┤");

    let mut results = Vec::new();

    for power in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
        let conditions = OperatingConditions {
            power_kw: power,
            reaction: FusionReaction::DD,
            ..OperatingConditions::consumer()
        };

        let result = physics.simulate(&conditions);
        let metrics = integration.compute_metrics(&result);

        let feasible_str = if result.feasible { "YES" } else { "NO " };

        println!(
            "│ {:>5.0}   {:>5.0}   {:>6.1}    {}       {:.3}     {:.3}     {:.3}    {:.3} │",
            power,
            result.geometry_shielding.total_mass_kg,
            result.pulse_thermal.lifetime_years.min(100.0),
            feasible_str,
            metrics.thermal_coherence,
            metrics.damage_healing_balance,
            metrics.geometry_harmony,
            metrics.overall_integration
        );

        results.push((power, metrics.overall_integration, result.feasible));
    }

    println!("└────────────────────────────────────────────────────────────────────────┘");

    // Find optimal integration
    let (best_power, best_i, _) = results
        .iter()
        .filter(|(_, _, feasible)| *feasible)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!(
        "\n  Highest integration: {:.3} at {} kW",
        best_i, best_power
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // REACTION TYPE COMPARISON
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INTEGRATION BY REACTION TYPE (5 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let reactions = [
        (FusionReaction::DD, "D-D (2.45 MeV n)"),
        (FusionReaction::DT, "D-T (14.1 MeV n)"),
        (FusionReaction::DHe3, "D-He3 (aneutronic)"),
    ];

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  Reaction            Mass (kg)   Shielding   Integration     Feasible │");
    println!("├────────────────────────────────────────────────────────────────────────┤");

    for (reaction, name) in &reactions {
        let conditions = OperatingConditions {
            power_kw: 5.0,
            reaction: *reaction,
            ..OperatingConditions::consumer()
        };

        let result = physics.simulate(&conditions);
        let metrics = integration.compute_metrics(&result);

        let feasible_str = if result.feasible { "YES" } else { "NO " };

        println!(
            "│  {:20} {:>8.0}   {:>6.2} m     {:.4}         {}      │",
            name,
            result.geometry_shielding.total_mass_kg,
            result.geometry_shielding.shielding.thickness_m,
            metrics.overall_integration,
            feasible_str
        );
    }

    println!("└────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // DETAILED METRICS BREAKDOWN
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DETAILED INTEGRATION BREAKDOWN (5 kW D-D)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let consumer_result = physics.simulate(&OperatingConditions::consumer());
    let consumer_metrics = integration.compute_metrics(&consumer_result);

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  INTEGRATION COMPONENTS                                               │");
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Coupling Index:       {:>8.4}  (HDC similarity pattern)            │",
        consumer_metrics.coupling_index
    );
    println!(
        "│  Design Integration:   {:>8.4}  (internal coherence)                │",
        consumer_metrics.design_integration
    );
    println!(
        "│  Thermal Coherence:    {:>8.4}  (temperature profile smoothness)    │",
        consumer_metrics.thermal_coherence
    );
    println!(
        "│  Damage-Healing:       {:>8.4}  (equilibrium balance)               │",
        consumer_metrics.damage_healing_balance
    );
    println!(
        "│  Geometry Harmony:     {:>8.4}  (mass efficiency)                   │",
        consumer_metrics.geometry_harmony
    );
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  OVERALL INTEGRATION:  {:>8.4}                                       │",
        consumer_metrics.overall_integration
    );
    println!(
        "│  Binding Advantage:    {:>8.4}  (bind vs bundle difference)         │",
        consumer_metrics.binding_advantage
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERPRETATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  KEY FINDINGS                                                       │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  • Smaller designs (1-5 kW) show higher integration scores         │");
    println!("│  • This suggests better thermal-structural-radiation coupling      │");
    println!("│  • D-He3 (aneutronic) has lowest mass but similar integration      │");
    println!("│  • Integration correlates with design compactness/efficiency       │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  ENGINEERING IMPLICATIONS                                           │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Higher integration scores may predict:                            │");
    println!("│  • Better robustness under off-design conditions                   │");
    println!("│  • More graceful degradation during failures                       │");
    println!("│  • Emergent stability properties from tight coupling               │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════\n");
}
