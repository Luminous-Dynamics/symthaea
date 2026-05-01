// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A/B Sweep: 5-seed comparison of Symthaea vs Legacy vs Utopia.
//!
//! Run: cargo run --release --bin ab_sweep

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::MultiWorldSimulator;

struct RunResult {
    seed: u64,
    cvs: f64,
    pop: usize,
    phi: f64,
    wars: usize,
    turchin: usize,
    survived: bool,
}

fn run(policy: PolicyConfig, seed: u64, years: u32) -> RunResult {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = policy;
    let mut sim = MultiWorldSimulator::new(config);
    let r = sim.run();
    RunResult {
        seed,
        cvs: r.final_cvs,
        pop: r.final_population,
        phi: r.final_collective_phi,
        wars: r.civil_wars,
        turchin: r.turchin_transitions,
        survived: r.survived,
    }
}

fn main() {
    let seeds = [42, 123, 789, 1337, 2718];
    let years = 150;

    println!("=== 5-SEED A/B SWEEP ({} years) ===\n", years);

    for label in &["Symthaea", "Legacy", "Utopia"] {
        let policy = match *label {
            "Symthaea" => PolicyConfig::default(),
            "Legacy" => PolicyConfig::legacy_mode(),
            "Utopia" => PolicyConfig::utopia_mode(),
            _ => unreachable!(),
        };

        let mut cvs_sum = 0.0;
        let mut pop_sum = 0usize;
        let mut phi_sum = 0.0;
        let mut wars_sum = 0usize;
        let mut survived_count = 0;

        println!("  --- {} ---", label);
        for &seed in &seeds {
            let r = run(policy.clone(), seed, years);
            println!(
                "    Seed {:>4}: CVS {:.3} | Pop {:>5} | Phi {:.3} | Wars {} | Turchin {} | {}",
                r.seed,
                r.cvs,
                r.pop,
                r.phi,
                r.wars,
                r.turchin,
                if r.survived { "OK" } else { "FAIL" }
            );
            cvs_sum += r.cvs;
            pop_sum += r.pop;
            phi_sum += r.phi;
            wars_sum += r.wars;
            if r.survived {
                survived_count += 1;
            }
        }

        let n = seeds.len() as f64;
        println!(
            "    MEAN:      CVS {:.3} | Pop {:>5} | Phi {:.3} | Wars {} | Survival {}/{}",
            cvs_sum / n,
            pop_sum / seeds.len(),
            phi_sum / n,
            wars_sum,
            survived_count,
            seeds.len()
        );
        println!();
    }

    println!("=== INTERPRETATION ===");
    println!("Compare mean CVS, population, and Phi across conditions.");
    println!("If Symthaea consistently outperforms Legacy, the thesis holds.");
}
