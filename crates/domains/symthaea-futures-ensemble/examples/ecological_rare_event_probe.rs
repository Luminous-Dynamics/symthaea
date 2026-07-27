// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dose-response probe for the Phase 2 addendum's item 3 (rare-event calibration): every
//! `ecological` regime tested so far is either 0% true (habitable, `solar_constant=1361.0`, the
//! real Earth default `IceAlbedoModel::earth()` uses) or 90%+ true (dimmed collapse,
//! `solar_constant=600.0`). Sweeps `solar_constant` between those two known endpoints to find a
//! mid-range value giving a genuinely rare-but-real checkpoint-level true rate (single digits,
//! not 0% or 90%+) -- the same checkpoint-level metric every train/test calibration test in this
//! plan has used (fraction of scored checkpoints where `actual=true` at `HORIZON=100`), not just
//! "did extinction ever happen in the run."
//!
//! Run: `cargo run --example ecological_rare_event_probe -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, PopulationCensusObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;
const SAMPLE_SIZE: usize = 3;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];

fn population_config() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            forage_efficiency: 0.6,
            ..OrganismConfig::default()
        },
        ..Default::default()
    }
}

/// (checkpoints scored, checkpoints where actual=true at HORIZON) for one seed at one
/// `solar_constant`.
fn true_rate_for_seed(solar_constant: f64, seed: u64) -> (usize, usize) {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = solar_constant;
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);
    let mut policy = PopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let _ = policy.observe(&truth, 0);
    let mut trajectory: Vec<bool> = vec![truth.is_extinct()];

    for _ in 0..TICKS {
        truth.step();
        trajectory.push(truth.is_extinct());
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

fn probe(solar_constant: f64) {
    let mut total_scored = 0usize;
    let mut total_true = 0usize;
    let mut per_seed = Vec::new();
    for &seed in &SEEDS {
        let (scored, true_count) = true_rate_for_seed(solar_constant, seed);
        per_seed.push((seed, scored, true_count));
        total_scored += scored;
        total_true += true_count;
    }
    let rate = 100.0 * total_true as f64 / total_scored as f64;
    println!(
        "solar_constant={solar_constant:7.1}: overall true rate = {rate:5.1}%  ({total_true}/{total_scored})  per-seed: {per_seed:?}"
    );
}

fn main() {
    println!("Ecological rare-event dose-response probe (5 seeds, horizon=100)\n");
    println!("Second refinement: 1280.0 gave 0%, 1260.0 gave 25.0% (uniformly across all 5 seeds,");
    println!("unlike predator_prey's chaotic transition) -- narrowing further:\n");
    for &sc in &[1280.0, 1276.0, 1272.0, 1268.0, 1264.0, 1260.0] {
        probe(sc);
    }
}
