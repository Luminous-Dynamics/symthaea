// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runs the confirmatory gate predeclared in `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`
//! (committed *before* this example was ever run): does rung 5 (FEP-driven) beat rung 2
//! (historical-frequency, uninformative) on aggregate Brier score across 5 seeds, habitable
//! regime only? `examples/ecological_backtest.rs`'s single-seed exploratory run showed rung 5
//! losing to rung 2 — this checks whether that was one unlucky seed or a repeatable property.
//!
//! Run: `cargo run --example confirmatory_gate -p symthaea-futures-ensemble`
//!
//! The threshold (rung 5 mean Brier strictly less than rung 2's, aggregated across all 5 seeds)
//! was fixed in the plan file before this code ran even once — this file must not be edited to
//! change the seeds, horizon, or comparison after seeing a result.

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::{FepDrivenGenerator, HistoricalFrequencyGenerator};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, ExtinctionObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;
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

/// Returns (rung5_sum, rung5_n, rung2_sum, rung2_n) for one seed, habitable regime only.
fn run_seed(seed: u64) -> (f64, usize, f64, usize) {
    let env = EarthForcedEnvironment::earth_like(200.0); // habitable -- no dimming
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);

    let mut policy = ExtinctionObservationPolicy::new(6, 0.5, 1, 0.02, false, 99);

    let mut observations: Vec<EcologicalObservation> = vec![policy.observe(&truth, 0)];
    let mut trajectory: Vec<bool> = vec![truth.is_extinct()];

    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        observations.push(policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
    }

    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let fep_driven = FepDrivenGenerator::default();

    let (mut r5_sum, mut r5_n, mut r2_sum, mut r2_n) = (0.0, 0usize, 0.0, 0usize);

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let obs = observations[checkpoint as usize];
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        if let ForecastOutput::Distribution(dist) =
            fep_driven.generate(&history_slice, Horizon(HORIZON))
        {
            r5_sum += BrierScore
                .score(&dist, &actual)
                .expect("scoring a validated forecast cannot fail")
                .get();
            r5_n += 1;
        }
        if let ForecastOutput::Distribution(dist) = historical.generate(&obs, Horizon(HORIZON)) {
            r2_sum += BrierScore
                .score(&dist, &actual)
                .expect("scoring a validated forecast cannot fail")
                .get();
            r2_n += 1;
        }

        checkpoint += CHECKPOINT_STRIDE;
    }

    (r5_sum, r5_n, r2_sum, r2_n)
}

fn main() {
    println!("Confirmatory gate: rung 5 (fep_driven) vs rung 2 (historical_frequency)");
    println!("Habitable regime, {} seeds: {SEEDS:?}\n", SEEDS.len());

    let mut per_seed_r5 = Vec::new();
    let mut per_seed_r2 = Vec::new();
    let (mut total_r5_sum, mut total_r5_n, mut total_r2_sum, mut total_r2_n) =
        (0.0, 0usize, 0.0, 0usize);

    for &seed in &SEEDS {
        let (r5_sum, r5_n, r2_sum, r2_n) = run_seed(seed);
        let r5_mean = r5_sum / r5_n as f64;
        let r2_mean = r2_sum / r2_n as f64;
        println!("  seed {seed:3}: fep_driven={r5_mean:.4}  historical_frequency={r2_mean:.4}");
        per_seed_r5.push(r5_mean);
        per_seed_r2.push(r2_mean);
        total_r5_sum += r5_sum;
        total_r5_n += r5_n;
        total_r2_sum += r2_sum;
        total_r2_n += r2_n;
    }

    let aggregate_r5 = total_r5_sum / total_r5_n as f64;
    let aggregate_r2 = total_r2_sum / total_r2_n as f64;

    println!(
        "\nAggregate mean Brier across all {} checkpoints, all 5 seeds:",
        total_r5_n
    );
    println!("  fep_driven           = {aggregate_r5:.4}");
    println!("  historical_frequency = {aggregate_r2:.4}");

    let gate_passes = aggregate_r5 < aggregate_r2;
    println!(
        "\nGATE {}: fep_driven {} historical_frequency",
        if gate_passes { "PASSED" } else { "FAILED" },
        if gate_passes {
            "beats"
        } else {
            "does NOT beat"
        }
    );
}
