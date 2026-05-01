// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 4: 2×2 Factorial — Consciousness × Coordination Understanding
//!
//! Tests for INTERACTION EFFECT: do consciousness + understanding together
//! produce more than the sum of their individual contributions?
//!
//! If D >> (C - A) + (B - A), the third regime is real.
//!
//! |                    | Low CU (0.0)  | High CU (0.5) |
//! |--------------------|---------------|---------------|
//! | Low Consciousness  | Cell A        | Cell B        |
//! | High Consciousness | Cell C        | Cell D        |

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

struct CellConfig {
    name: &'static str,
    consciousness_boost: f64,
    coordination_understanding: f64,
}

fn run_cell(cell: &CellConfig, seed: u64) -> (f64, f64, usize, usize) {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.policy.coordination_understanding_initial = cell.coordination_understanding;
    config.policy.consciousness_boost = cell.consciousness_boost;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    (
        report.final_cvs,
        report.final_collective_phi,
        report.final_population,
        report.civil_wars,
    )
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    let cells = [
        CellConfig {
            name: "A: Low_C + Low_CU",
            consciousness_boost: 0.0,
            coordination_understanding: 0.0,
        },
        CellConfig {
            name: "B: Low_C + High_CU",
            consciousness_boost: 0.0,
            coordination_understanding: 0.5,
        },
        CellConfig {
            name: "C: High_C + Low_CU",
            consciousness_boost: 0.4,
            coordination_understanding: 0.0,
        },
        CellConfig {
            name: "D: High_C + High_CU",
            consciousness_boost: 0.4,
            coordination_understanding: 0.5,
        },
    ];

    let seeds: Vec<u64> = (42..52).collect();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 4: 2×2 Factorial (Consciousness × Coordination) ║");
    println!(
        "║  4 cells × {} seeds = {} simulations                         ║",
        seeds.len(),
        cells.len() * seeds.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut cell_cvs: Vec<Vec<f64>> = vec![Vec::new(); 4];
    let mut cell_phi: Vec<Vec<f64>> = vec![Vec::new(); 4];
    let mut cell_pop: Vec<Vec<f64>> = vec![Vec::new(); 4];
    let mut cell_wars: Vec<Vec<f64>> = vec![Vec::new(); 4];

    for (ci, cell) in cells.iter().enumerate() {
        println!("── {} ──", cell.name);
        for &seed in &seeds {
            let (cvs, phi, pop, wars) = run_cell(cell, seed);
            print!(
                "  seed={}: CVS={:.4} Phi={:.4} pop={} wars={}",
                seed, cvs, phi, pop, wars
            );
            println!();
            cell_cvs[ci].push(cvs);
            cell_phi[ci].push(phi);
            cell_pop[ci].push(pop as f64);
            cell_wars[ci].push(wars as f64);
        }
        println!();
    }

    // ── Summary ──
    println!("══════════════════════════════════════════════════════════════");
    println!("  2×2 FACTORIAL RESULTS");
    println!("══════════════════════════════════════════════════════════════\n");

    println!(
        "{:<25} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Cell", "CVS", "±", "Phi", "Pop", "Wars"
    );
    println!("{}", "-".repeat(68));
    for (ci, cell) in cells.iter().enumerate() {
        println!(
            "{:<25} {:>8.4} {:>8.4} {:>8.4} {:>8.0} {:>8.1}",
            cell.name,
            mean(&cell_cvs[ci]),
            std_dev(&cell_cvs[ci]),
            mean(&cell_phi[ci]),
            mean(&cell_pop[ci]),
            mean(&cell_wars[ci]),
        );
    }

    // ── Interaction analysis ──
    let a = mean(&cell_cvs[0]); // Low C, Low CU
    let b = mean(&cell_cvs[1]); // Low C, High CU
    let c = mean(&cell_cvs[2]); // High C, Low CU
    let d = mean(&cell_cvs[3]); // High C, High CU

    let main_effect_cu = ((b - a) + (d - c)) / 2.0;
    let main_effect_consciousness = ((c - a) + (d - b)) / 2.0;
    let interaction = (d - c) - (b - a); // The key number

    println!("\n── Main Effects (CVS) ──");
    println!("  Coordination Understanding: {:+.4}", main_effect_cu);
    println!(
        "  Consciousness:              {:+.4}",
        main_effect_consciousness
    );
    println!("  INTERACTION (C×CU):         {:+.4}", interaction);

    // Phi interaction
    let a_phi = mean(&cell_phi[0]);
    let b_phi = mean(&cell_phi[1]);
    let c_phi = mean(&cell_phi[2]);
    let d_phi = mean(&cell_phi[3]);
    let phi_interaction = (d_phi - c_phi) - (b_phi - a_phi);

    println!("\n── Main Effects (Phi) ──");
    println!(
        "  Coordination Understanding: {:+.4}",
        ((b_phi - a_phi) + (d_phi - c_phi)) / 2.0
    );
    println!(
        "  Consciousness:              {:+.4}",
        ((c_phi - a_phi) + (d_phi - b_phi)) / 2.0
    );
    println!("  INTERACTION (C×CU):         {:+.4}", phi_interaction);

    // Additive prediction vs actual
    let predicted_d = a + (b - a) + (c - a); // What D would be if effects are purely additive
    let synergy = d - predicted_d;

    println!("\n── Synergy Analysis ──");
    println!("  Predicted D (additive):  {:.4}", predicted_d);
    println!("  Actual D:                {:.4}", d);
    println!("  Synergy (D - predicted): {:+.4}", synergy);

    println!("\n── Verdict ──");
    if interaction > 0.005 {
        println!("  INTERACTION EFFECT DETECTED: Consciousness × Coordination");
        println!("  Understanding produces MORE than the sum of individual effects.");
        println!("  The THIRD REGIME is supported by the data.");
        println!(
            "  Synergy = {:+.4} CVS beyond additive prediction.",
            synergy
        );
    } else if interaction > -0.005 {
        println!("  NO INTERACTION: Effects are additive.");
        println!("  Consciousness and coordination understanding contribute independently.");
        println!("  The third regime is NOT supported — the two factors are separable.");
    } else {
        println!("  NEGATIVE INTERACTION: Combined effect is LESS than sum of parts.");
        println!("  Consciousness and coordination understanding may interfere.");
    }
}
