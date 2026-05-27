// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine Demo: Complete LCF Power System Specification
//!
//! This example demonstrates the full integration of:
//! - Resonance Discovery (fuel/reaction selection)
//! - Wolverine Protocol (self-healing materials)
//! - Advanced Materials (MAX phases, nano-laminates)
//! - Spark Engine (unified specification)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example spark_engine_demo --release
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    AdvancedMaterials, FusionReaction, HEATarget, RadiationDamageSystem, SparkEngineSpec,
    SparkTarget, multi_seed_hea_search,
};

fn main() {
    println!("\n{}", "═".repeat(72));
    println!("   SPARK ENGINE DESIGN SYSTEM");
    println!("   Unified Lattice Confinement Fusion Specification");
    println!("{}", "═".repeat(72));

    let genesis = GenesisSeed::from_phrase("Spark Engine v1.0 - Sovereign Power");

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: Material Discovery via Multi-Seed Consensus
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n{}", "━".repeat(72));
    println!("PHASE 1: Multi-Seed Material Consensus");
    println!("{}", "━".repeat(72));

    println!("\n  Running {} parallel universe searches...", 5);

    let target = HEATarget::lcf_wolverine();
    let consensus = multi_seed_hea_search(
        "Spark Engine",
        &target,
        FusionReaction::DT,
        5, // 5 seeds
        5, // top 5 per seed
    );

    println!("\n  {:─^68}", " CONSENSUS CANDIDATES ");
    println!();

    for (i, result) in consensus.iter().take(5).enumerate() {
        let rank = i + 1;
        let stars = "★".repeat((result.consensus_score * 5.0) as usize);

        println!(
            "  #{} {} (Consensus: {:.0}%)",
            rank,
            result.alloy.name,
            result.consensus_score * 100.0
        );
        println!(
            "     {} Seeds: {}/5, Avg Match: {:.1}%",
            stars,
            result.occurrence_count,
            result.avg_match_score * 100.0
        );
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: Advanced Materials Analysis
    // ═══════════════════════════════════════════════════════════════════════

    println!("{}", "━".repeat(72));
    println!("PHASE 2: Advanced Materials Analysis");
    println!("{}", "━".repeat(72));

    let radiation = RadiationDamageSystem::from_genesis(&genesis);
    let advanced = AdvancedMaterials::from_genesis(&genesis);

    let damage = radiation.calculate_damage(14.1, 1e14, 22);

    println!("\n  MAX PHASES (Kink Band Healing):\n");

    let rankings = advanced.rank_all_materials(&damage);
    for (i, mat) in rankings
        .iter()
        .filter(|m| m.material_type == symthaea_core::physics::MaterialType::MAXPhase)
        .take(3)
        .enumerate()
    {
        println!(
            "    {}. {} (LCF: {:.0}%)",
            i + 1,
            mat.name,
            mat.lcf_score * 100.0
        );
        println!("       Role: {}", mat.recommended_role);
    }

    println!("\n  NANO-LAMINATES (Interface Sinks):\n");

    for (i, mat) in rankings
        .iter()
        .filter(|m| m.material_type == symthaea_core::physics::MaterialType::NanoLaminate)
        .take(3)
        .enumerate()
    {
        println!(
            "    {}. {} (LCF: {:.0}%)",
            i + 1,
            mat.name,
            mat.lcf_score * 100.0
        );
        println!("       Role: {}", mat.recommended_role);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: Engine Specifications
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n{}", "━".repeat(72));
    println!("PHASE 3: Engine Specifications");
    println!("{}", "━".repeat(72));

    // Consumer-grade engine
    println!("\n  {:─^68}", " CONSUMER MODEL ");
    let consumer_spec = SparkEngineSpec::design(&genesis, SparkTarget::consumer());
    println!("{}", consumer_spec.summary());

    // Industrial-grade engine
    println!("  {:─^68}", " INDUSTRIAL MODEL ");
    let industrial_spec = SparkEngineSpec::design(&genesis, SparkTarget::industrial());
    println!("{}", industrial_spec.summary());

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 4: Comparison Summary
    // ═══════════════════════════════════════════════════════════════════════

    println!("{}", "━".repeat(72));
    println!("PHASE 4: Model Comparison");
    println!("{}", "━".repeat(72));

    println!(
        "\n  {:<20} {:>15} {:>15}",
        "Metric", "Consumer", "Industrial"
    );
    println!("  {:─<20} {:─>15} {:─>15}", "", "", "");
    println!(
        "  {:<20} {:>12.1} kW {:>12.0} kW",
        "Power Output", consumer_spec.power_kw, industrial_spec.power_kw
    );
    println!(
        "  {:<20} {:>12.1} yr {:>12.1} yr",
        "Lifetime", consumer_spec.lifetime_years, industrial_spec.lifetime_years
    );
    println!(
        "  {:<20} {:>12.4} g {:>12.2} g",
        "Fuel/Year", consumer_spec.fuel_rate_g_year, industrial_spec.fuel_rate_g_year
    );
    println!(
        "  {:<20} {:>15?} {:>15?}",
        "Reaction", consumer_spec.reaction, industrial_spec.reaction
    );
    println!(
        "  {:<20} {:>14.0}% {:>14.0}%",
        "Confidence",
        consumer_spec.confidence * 100.0,
        industrial_spec.confidence * 100.0
    );

    // ═══════════════════════════════════════════════════════════════════════
    // FINAL OUTPUT: Complete Specification
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n{}", "━".repeat(72));
    println!("SPARK ENGINE COMPLETE SPECIFICATION");
    println!("{}", "━".repeat(72));

    println!(
        r#"

  ┌─────────────────────────────────────────────────────────────────────┐
  │                     SPARK ENGINE v1.0                               │
  │                 Sovereign Power Technology                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │   FUEL SYSTEM                                                       │
  │   ────────────                                                      │
  │   Source:     Deuterium (D₂) from seawater                          │
  │   Cost:       ~$0.01 per gram                                       │
  │   Density:    ~{:.0} GJ/g (fusion scale)                             │
  │                                                                     │
  │   REACTION                                                          │
  │   ────────                                                          │
  │   Consumer:   D + D → He-3 + n (2.45 MeV, safer)                    │
  │   Industrial: D + T → He-4 + n (14.1 MeV, higher output)            │
  │                                                                     │
  │   ARCHITECTURE                                                      │
  │   ────────────                                                      │
  │   ╔═══════════════════════════════════════════════════════════╗     │
  │   ║  SHELL: {:<30} (HEA)                         ║     │
  │   ║    • H-storage: {:.0}%                                      ║     │
  │   ║    • Self-healing via sluggish diffusion                  ║     │
  │   ╠═══════════════════════════════════════════════════════════╣     │
  │   ║  INTERFACE: {:<18} (Nano-laminate)                     ║     │
  │   ║    • Defect sink efficiency: {:.0}%                         ║     │
  │   ╠═══════════════════════════════════════════════════════════╣     │
  │   ║  CORE: Galinstan (Liquid Metal)                           ║     │
  │   ║    • Self-healing: 100%                                   ║     │
  │   ║    • Heat transfer medium                                 ║     │
  │   ╚═══════════════════════════════════════════════════════════╝     │
  │                                                                     │
  │   TRIGGER PROTOCOL                                                  │
  │   ────────────────                                                  │
  │   Method:      X-ray @ 0.1 nm                                       │
  │   Pulse:       Femtosecond                                          │
  │   Temperature: 300 K                                                │
  │   Mechanism:   Phonon-assisted bremsstrahlung                       │
  │                                                                     │
  │   KEY INNOVATIONS                                                   │
  │   ───────────────                                                   │
  │   1. Inverse vector search identifies optimal fuel/materials        │
  │   2. HEA sluggish diffusion extends lifetime 10-100x                │
  │   3. Liquid metal core provides infinite self-healing               │
  │   4. Nano-laminate interface maximizes defect absorption            │
  │   5. Multi-seed consensus ensures robust material selection         │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

"#,
        consumer_spec.energy_density_gj_g,
        consumer_spec.shell.name,
        consumer_spec.shell.h_solubility * 100.0,
        consumer_spec.interface.name,
        consumer_spec.interface.sink_efficiency * 100.0,
    );

    println!("{}", "═".repeat(72));
    println!("   SPARK ENGINE DESIGN COMPLETE");
    println!("{}\n", "═".repeat(72));
}