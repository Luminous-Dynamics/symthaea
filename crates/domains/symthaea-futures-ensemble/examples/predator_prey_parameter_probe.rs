// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cheap probe, not a backtest: finds parameters where predator extinction is a real, variable,
//! non-negligible event across seeds -- the previous backtest's parameters
//! (`tests/phase1_predator_prey.rs`'s fixture) were tuned specifically to prevent that. Just
//! prints whether/when predators go extinct across several seeds and a couple of parameter
//! variants, no forecasting involved.
//!
//! Run: `cargo run --example predator_prey_parameter_probe -p symthaea-futures-ensemble`

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};

const TICKS: u64 = 8000;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];

fn config(predation_scale: f64, predation_efficiency: f64) -> PredatorPreyConfig {
    let organism_cfg = OrganismConfig {
        forage_efficiency: 0.6,
        ..OrganismConfig::default()
    };
    let pop_cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg,
        ..Default::default()
    };
    PredatorPreyConfig {
        prey_cfg: pop_cfg,
        predator_cfg: pop_cfg,
        plant_resource_total: 3.0,
        predation_scale,
        predation_efficiency,
    }
}

fn probe(label: &str, initial_prey: usize, initial_predators: usize, cfg: PredatorPreyConfig) {
    println!("== {label} (prey={initial_prey}, predators={initial_predators}) ==");
    for &seed in &SEEDS {
        let mut sim = PredatorPreySim::new(cfg, initial_prey, initial_predators, seed);
        let mut extinct_at: Option<u64> = None;
        let mut min_predator_count = initial_predators;
        for _ in 0..TICKS {
            sim.step();
            let count = sim.predator.len();
            min_predator_count = min_predator_count.min(count);
            if count == 0 && extinct_at.is_none() {
                extinct_at = Some(sim.t);
            }
        }
        println!(
            "  seed {seed:3}: min_predator_count={min_predator_count:3}  extinct_at={:?}  final_predator_count={}",
            extinct_at,
            sim_final_count(&sim)
        );
    }
    println!();
}

fn sim_final_count(sim: &PredatorPreySim) -> usize {
    sim.predator.len()
}

fn main() {
    // Variant 1: the original sustaining config, for comparison.
    probe("original (sustaining)", 10, 3, config(0.05, 0.05));

    // Variant 2: fewer predators, same pressure -- smaller population more vulnerable to
    // stochastic bad luck.
    probe("fewer initial predators", 10, 2, config(0.05, 0.05));

    // Variant 3: less prey to start, same predator count.
    probe("scarcer prey", 5, 3, config(0.05, 0.05));

    // Variant 4: higher predation_scale (predators need more prey density per capita to thrive).
    probe("higher predation_scale", 10, 3, config(0.15, 0.05));

    // Variant 5: lower predation_efficiency (predators catch prey less effectively).
    probe("lower predation_efficiency", 10, 3, config(0.05, 0.015));

    // Variant 6: combine scarcer prey + fewer predators + weaker efficiency.
    probe("combined stress", 5, 2, config(0.05, 0.015));
}
