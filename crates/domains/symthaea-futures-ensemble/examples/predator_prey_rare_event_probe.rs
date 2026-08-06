// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dose-response probe for the Phase 2 addendum's item 3 (rare-event calibration), predator_prey
//! side. `predator_prey_resource_scarcity_probe.rs` found `resource=3.0` gives 0/5 seeds ever
//! extinct and `resource<=1.0` gives 5/5 fast and reliable; `resource=1.5` gave exactly 1/5
//! seeds extinct (late, tick 7503 in an 8000-tick run) -- a real but very sparse event. This
//! probe measures the same checkpoint-level true-rate metric every train/test calibration test
//! in this plan has used (fraction of scored checkpoints where `actual=true` at `HORIZON=100`,
//! not just "did extinction ever happen"), across a `resource` sweep between the known rare and
//! never-happens endpoints, over the longer 8000-tick window the rare regime needs.
//!
//! Run: `cargo run --example predator_prey_rare_event_probe -p symthaea-futures-ensemble`

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};

const HORIZON: u64 = 100;
const TICKS: u64 = 8000;
const CHECKPOINT_STRIDE: u64 = 200;
const INITIAL_PREY: usize = 10;
const INITIAL_PREDATORS: usize = 3;
// A wider seed set than the usual 5 -- the finer sweep below found this family's transition
// zone genuinely chaotic/seed-sensitive (individual seeds tip in or out unpredictably at nearby
// resource values), unlike ecological's clean, near-uniform-per-seed transition. Averaging over
// more seeds is the honest way to find a real population-level rare rate here, rather than
// cherry-picking whichever exact value happens to look rare in a 5-seed sample.
const SEEDS: [u64; 20] = [
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211,
];

fn config(plant_resource_total: f64) -> PredatorPreyConfig {
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
        predation_scale: 0.05,
        predation_efficiency: 0.05,
    }
}

/// (checkpoints scored, checkpoints where actual=true at HORIZON) for one seed at one resource
/// level.
fn true_rate_for_seed(resource: f64, seed: u64) -> (usize, usize) {
    let mut sim = PredatorPreySim::new(config(resource), INITIAL_PREY, INITIAL_PREDATORS, seed);
    let mut trajectory: Vec<bool> = vec![sim.predator.len() == 0];

    for _ in 0..TICKS {
        sim.step();
        trajectory.push(sim.predator.len() == 0);
    }

    let mut scored = 0usize;
    let mut true_count = 0usize;
    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        scored += 1;
        if trajectory[(checkpoint + HORIZON) as usize] {
            true_count += 1;
        }
        checkpoint += CHECKPOINT_STRIDE;
    }
    (scored, true_count)
}

fn probe(resource: f64) {
    let mut total_scored = 0usize;
    let mut total_true = 0usize;
    let mut per_seed = Vec::new();
    for &seed in &SEEDS {
        let (scored, true_count) = true_rate_for_seed(resource, seed);
        per_seed.push((seed, scored, true_count));
        total_scored += scored;
        total_true += true_count;
    }
    let rate = 100.0 * total_true as f64 / total_scored as f64;
    println!(
        "resource={resource:5.2}: overall true rate = {rate:5.1}%  ({total_true}/{total_scored})  per-seed: {per_seed:?}"
    );
}

fn main() {
    println!(
        "Predator/prey rare-event dose-response probe ({} seeds, horizon=100, 8000 ticks)\n",
        SEEDS.len()
    );
    println!(
        "5-seed sweep found a chaotic, non-monotonic transition between 1.5 and 1.25 (e.g. 1.45:"
    );
    println!(
        "42%, 1.42: 15%, 1.40: 0%, 1.38: 18.5%) -- individual seeds tip in/out unpredictably at"
    );
    println!("nearby values. Wider seed set to find the real population-level rare rate:\n");
    for &r in &[1.45, 1.40, 1.35, 1.30] {
        probe(r);
    }
}
