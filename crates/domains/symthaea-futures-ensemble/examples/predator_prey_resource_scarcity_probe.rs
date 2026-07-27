// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A genuinely different lever than `predator_prey_parameter_probe.rs` tried: that probe varied
//! predation-specific knobs (`predation_scale`, `predation_efficiency`, initial counts) across 6
//! configurations and found the system too resilient -- predator count never dropped below its
//! starting value in any of 30 seed-runs. `plant_resource_total` (the prey's entire shared food
//! supply) was left fixed at `3.0` in every one of those configurations, in both that probe and
//! the original `tests/phase1_predator_prey.rs` fixture -- never varied. This is the
//! predator/prey family's closest analog to the ecological family's "dimmed sun" lever (which
//! *did* reliably force real collapse): starve the base of the food chain directly, rather than
//! tune how effectively predators exploit whatever prey exists.
//!
//! Not a backtest -- like the prior probe, just tracks whether/when predators go extinct across
//! scarcity levels and seeds, to find out whether ANY lever can make predator extinction a real,
//! variable event here (a prerequisite for ever testing `HistogramCalibrator`'s generalization on
//! this scenario family, still open per the plan doc).
//!
//! Run: `cargo run --example predator_prey_resource_scarcity_probe -p symthaea-futures-ensemble`

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};

const TICKS: u64 = 8000;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];

fn config(
    plant_resource_total: f64,
    predation_scale: f64,
    predation_efficiency: f64,
) -> PredatorPreyConfig {
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
        plant_resource_total,
        predation_scale,
        predation_efficiency,
    }
}

fn probe(label: &str, initial_prey: usize, initial_predators: usize, cfg: PredatorPreyConfig) {
    println!(
        "== {label} (prey={initial_prey}, predators={initial_predators}, resource={:.2}) ==",
        cfg.plant_resource_total
    );
    for &seed in &SEEDS {
        let mut sim = PredatorPreySim::new(cfg, initial_prey, initial_predators, seed);
        let mut predator_extinct_at: Option<u64> = None;
        let mut prey_extinct_at: Option<u64> = None;
        let mut min_predator_count = initial_predators;
        let mut min_prey_count = initial_prey;
        for _ in 0..TICKS {
            sim.step();
            let predator_count = sim.predator.len();
            let prey_count = sim.prey.len();
            min_predator_count = min_predator_count.min(predator_count);
            min_prey_count = min_prey_count.min(prey_count);
            if predator_count == 0 && predator_extinct_at.is_none() {
                predator_extinct_at = Some(sim.t);
            }
            if prey_count == 0 && prey_extinct_at.is_none() {
                prey_extinct_at = Some(sim.t);
            }
        }
        println!(
            "  seed {seed:3}: min_predator={min_predator_count:3}  min_prey={min_prey_count:3}  \
             predator_extinct_at={predator_extinct_at:?}  prey_extinct_at={prey_extinct_at:?}  \
             final_predator={}  final_prey={}",
            sim.predator.len(),
            sim.prey.len()
        );
    }
    println!();
}

fn main() {
    // Baseline for comparison -- same as the original fixture and the prior probe's variant 1.
    probe("baseline resource=3.0", 10, 3, config(3.0, 0.05, 0.05));

    // Progressive resource scarcity, predation knobs held at the original sustaining values.
    probe("resource=1.5", 10, 3, config(1.5, 0.05, 0.05));
    probe("resource=1.0", 10, 3, config(1.0, 0.05, 0.05));
    probe("resource=0.5", 10, 3, config(0.5, 0.05, 0.05));
    probe("resource=0.25", 10, 3, config(0.25, 0.05, 0.05));
    probe("resource=0.10", 10, 3, config(0.10, 0.05, 0.05));

    // Combined: scarce resource + fewer initial predators + higher predation_scale (compounding
    // every stressor the prior probe tried individually, plus this session's new lever).
    probe(
        "combined: resource=0.5 + fewer predators + higher predation_scale",
        10,
        2,
        config(0.5, 0.15, 0.05),
    );
}
