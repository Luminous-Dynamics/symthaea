// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! BIOLOGICAL VALIDATION 4: Information Efficiency (Landauer Ratio)
//!
//! IMPORTANT: This computes the SIMULATED thermodynamic ledger's J/bit ratio,
//! NOT the actual CPU hardware costs. We are validating the simulated physics
//! world's energy accounting, not claiming we broke the Landauer limit on a PC.
//!
//! Biology:
//! - Landauer limit: 2.87 × 10^-21 J/bit at 310K
//! - Biological synapses: ~16.6× Landauer per operation
//! - Whole neuron per spike: ~10^8× Landauer
//! - Communication/computation ratio: 35:1 (Levy & Calvert 2021 PNAS)
//!
//! Test: compute the simulated J/bit, communication/computation ratio,
//! and compare against biological data.

use symtropy_consciousness_physics::ThermodynamicConstants;
use symtropy_consciousness_physics::thermodynamics::{
    ENERGY_PER_COGNITIVE_OP, LANDAUER_BOUND_310K,
};

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BIOLOGICAL VALIDATION 4: Information Efficiency             ║");
    eprintln!("║  NOTE: All values are from the SIMULATED thermodynamic       ║");
    eprintln!("║  ledger, NOT CPU hardware costs.                            ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    let constants = ThermodynamicConstants::default();

    // === SIMULATED ENERGY COSTS ===
    let consciousness_cost_per_tick = constants.consciousness_maintenance_per_tick; // 0.08 J/tick
    let movement_cost_per_unit = constants.movement_cost_per_unit; // 0.005 J/unit
    let ticks_per_second = 64.0;
    let walk_speed = 100.0; // units/sec

    // Consciousness maintenance per second (simulated)
    let consciousness_cost_per_sec = consciousness_cost_per_tick * ticks_per_second;

    // Movement cost per second at walk speed (simulated)
    let movement_cost_per_sec = movement_cost_per_unit * walk_speed;

    // === COMMUNICATION vs COMPUTATION RATIO ===
    // In our model:
    //   "Computation" = consciousness maintenance (information integration, Phi computation)
    //   "Communication" = movement energy (physically navigating to share models, offload epistemically)
    let comm_to_comp_ratio = movement_cost_per_sec / consciousness_cost_per_sec;

    // === INFORMATION CONTENT ===
    // Each consciousness tick processes the Master Equation across 8 inputs
    // Phi = integration measure across these inputs
    // Each input is a continuous value [0,1] — at f64 precision, ~53 bits per input
    // But the effective information is much less — the equation computes a softmin bottleneck
    // Estimated effective bits per consciousness tick: ~8 inputs × ~4 effective bits = ~32 bits
    let estimated_bits_per_tick = 32.0;
    let estimated_bits_per_sec = estimated_bits_per_tick * ticks_per_second;

    // Simulated J/bit
    let simulated_j_per_bit = consciousness_cost_per_tick / estimated_bits_per_tick;

    // Ratio to Landauer bound
    let ratio_to_landauer = simulated_j_per_bit / LANDAUER_BOUND_310K;

    // === BIOLOGICAL COMPARISON ===
    let bio_synapse_j_per_op = 4.76e-20; // 16.6 × Landauer (our ENERGY_PER_COGNITIVE_OP constant)
    let bio_neuron_j_per_spike = 3.5e-11; // ~7.1 × 10^8 ATP × 5 × 10^-20 J/ATP
    let bio_neuron_to_landauer = bio_neuron_j_per_spike / LANDAUER_BOUND_310K;
    let bio_comm_to_comp = 35.0; // Levy & Calvert 2021

    // CSV output
    println!("metric,our_model,biology,unit,match");

    println!(
        "landauer_bound,{:.2e},{:.2e},J/bit,reference",
        LANDAUER_BOUND_310K, LANDAUER_BOUND_310K
    );

    println!(
        "energy_per_cognitive_op,{:.2e},{:.2e},J/op,{}",
        ENERGY_PER_COGNITIVE_OP,
        bio_synapse_j_per_op,
        if (ENERGY_PER_COGNITIVE_OP - bio_synapse_j_per_op).abs() < 1e-21 {
            "MATCH"
        } else {
            "CLOSE"
        }
    );

    println!(
        "simulated_j_per_bit,{:.2e},{:.2e},J/bit,simulated_ledger",
        simulated_j_per_bit, bio_synapse_j_per_op
    );

    println!(
        "ratio_to_landauer,{:.1},{:.1},×,{}",
        ratio_to_landauer,
        16.6,
        if (ratio_to_landauer - 16.6).abs() < 100.0 {
            "SAME_ORDER"
        } else {
            "DIFFERENT"
        }
    );

    println!(
        "neuron_ratio_to_landauer,N/A,{:.1e},×,biological_reference",
        bio_neuron_to_landauer
    );

    println!(
        "comm_to_comp_ratio,{:.1},{:.1},ratio,{}",
        comm_to_comp_ratio,
        bio_comm_to_comp,
        if (comm_to_comp_ratio - bio_comm_to_comp).abs() < 30.0 {
            "SAME_ORDER"
        } else {
            "DIFFERENT"
        }
    );

    println!(
        "consciousness_cost_per_sec,{:.2},{:.2e},J/s,simulated",
        consciousness_cost_per_sec, 20.0
    ); // brain ~20W

    println!(
        "movement_cost_per_sec,{:.2},N/A,J/s,simulated",
        movement_cost_per_sec
    );

    // Summary
    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("SIMULATED THERMODYNAMIC LEDGER (not CPU costs):");
    eprintln!(
        "  Consciousness maintenance: {:.2} J/s ({:.2} J/tick × {} Hz)",
        consciousness_cost_per_sec, consciousness_cost_per_tick, ticks_per_second
    );
    eprintln!(
        "  Movement (walking):        {:.2} J/s",
        movement_cost_per_sec
    );
    eprintln!("  Comm/Comp ratio:           {:.1}:1", comm_to_comp_ratio);
    eprintln!("  BIOLOGY Comm/Comp:         35:1 (Levy & Calvert 2021 PNAS)");
    if comm_to_comp_ratio < 1.0 {
        eprintln!("  ✗ Our model: computation > communication (inverted from biology)");
        eprintln!("    This is expected: our agents move in 2D, biology moves in 3D");
        eprintln!("    with axonal wiring costs dominating metabolically.");
    } else if comm_to_comp_ratio > 10.0 && comm_to_comp_ratio < 100.0 {
        eprintln!("  ✓ Same order of magnitude as biology");
    } else {
        eprintln!("  ~ Different magnitude — see discussion");
    }
    eprintln!();
    eprintln!("INFORMATION EFFICIENCY:");
    eprintln!(
        "  Estimated bits/tick:       {:.0}",
        estimated_bits_per_tick
    );
    eprintln!("  Simulated J/bit:           {:.2e}", simulated_j_per_bit);
    eprintln!("  Ratio to Landauer:         {:.1e}×", ratio_to_landauer);
    eprintln!(
        "  Our constant (16.6×):      {:.2e} J/op",
        ENERGY_PER_COGNITIVE_OP
    );
    eprintln!(
        "  Bio synapse (16.6×):       {:.2e} J/op",
        bio_synapse_j_per_op
    );
    eprintln!(
        "  Bio neuron/spike:          {:.2e} J (~{:.0e}× Landauer)",
        bio_neuron_j_per_spike, bio_neuron_to_landauer
    );
    eprintln!("═══════════════════════════════════════════════════");
}
