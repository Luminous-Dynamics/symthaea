// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! First real backtest of the second forecasting target — time-to-extinction, conditional on
//! extinction occurring — scored via `symthaea-futures-calibration::Crps` instead of Brier.
//! Real dimmed-sun collapse (`symthaea-alife`'s ground-truth-tested guaranteed-collapse
//! fixture), `PopulationCensusObservationPolicy` (the proven fix, not the original flawed
//! fixed-cohort policy), 5 seeds. Only evaluates checkpoints strictly before the ground truth's
//! actual recorded extinction tick — "conditional on extinction occurring" means the question
//! is only meaningful once we already know it happens.
//!
//! Also backtests `TimeToExtinctionUncensoredLinearGenerator` alongside the naive
//! `TimeToExtinctionLinearGenerator` — the fix for the whole-history-OLS-dominated-by-the-flat-
//! cap-segment failure this same backtest originally found (see the plan doc's "Time-to-
//! extinction / CRPS target" section for the diagnosed root cause).
//!
//! And `TimeToExtinctionEnsembleGenerator` — the uncertainty-propagation follow-up: every other
//! generator here reports a single point as if certain, which throws away exactly what `Crps`
//! is designed to reward (calibrated spread). This one reports a real bootstrap ensemble
//! instead, at the cost of occasionally abstaining where the point generators would still
//! (over-confidently) guess.
//!
//! Run: `cargo run --example time_to_extinction_backtest -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{Crps, ScoringRule};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::{
    TimeToExtinctionEnsembleGenerator, TimeToExtinctionLinearGenerator,
    TimeToExtinctionOracleGenerator, TimeToExtinctionUncensoredLinearGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, PopulationCensusObservationPolicy,
};

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

#[derive(Default, Clone, Copy)]
struct RungTally {
    sum: f64,
    n: usize,
}

impl RungTally {
    fn record(&mut self, output: &ForecastOutput, actual: &OutcomeRegion) {
        if let ForecastOutput::Distribution(dist) = output {
            self.sum += Crps.score(dist, actual);
            self.n += 1;
        }
    }

    fn mean(&self) -> f64 {
        if self.n > 0 {
            self.sum / self.n as f64
        } else {
            f64::NAN
        }
    }
}

const BOOTSTRAP_REPLICATES: usize = 100;

#[derive(Default, Clone, Copy)]
struct SeedTallies {
    oracle: RungTally,
    linear: RungTally,
    uncensored: RungTally,
    ensemble: RungTally,
}

fn run_seed(seed: u64) -> SeedTallies {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = 600.0; // dimmed past the snowball threshold -- guaranteed collapse
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);

    let mut policy = PopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut observations: Vec<EcologicalObservation> = vec![policy.observe(&truth, 0)];
    let mut trajectory: Vec<bool> = vec![truth.is_extinct()];

    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        observations.push(policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
    }

    let extinction_tick = trajectory
        .iter()
        .position(|&extinct| extinct)
        .expect("dimmed-sun regime should always collapse within 4000 ticks")
        as u64;

    let oracle = TimeToExtinctionOracleGenerator::from_trajectory(trajectory);
    let linear = TimeToExtinctionLinearGenerator;
    let uncensored = TimeToExtinctionUncensoredLinearGenerator;
    // Seed reuses the scenario seed itself -- deterministic and varied across seeds without
    // needing a separate seed-management scheme for a single-use backtest.
    let ensemble = TimeToExtinctionEnsembleGenerator::new(BOOTSTRAP_REPLICATES, seed);

    let mut tallies = SeedTallies::default();

    let mut checkpoint = 0u64;
    while checkpoint < extinction_tick {
        let actual = OutcomeRegion::Interval {
            low: (extinction_tick - checkpoint) as f64,
            high: (extinction_tick - checkpoint) as f64,
        };
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        tallies
            .oracle
            .record(&oracle.generate(&checkpoint, Horizon(0)), &actual);
        tallies
            .linear
            .record(&linear.generate(&history_slice, Horizon(0)), &actual);
        tallies
            .uncensored
            .record(&uncensored.generate(&history_slice, Horizon(0)), &actual);
        tallies
            .ensemble
            .record(&ensemble.generate(&history_slice, Horizon(0)), &actual);

        checkpoint += CHECKPOINT_STRIDE;
    }

    tallies
}

fn main() {
    println!(
        "Time-to-extinction backtest (dimmed-sun collapse, {} seeds, {} bootstrap replicates)\n",
        SEEDS.len(),
        BOOTSTRAP_REPLICATES
    );

    let mut total = SeedTallies::default();

    for &seed in &SEEDS {
        let t = run_seed(seed);
        println!(
            "  seed {seed:3}: oracle CRPS={:.4} (n={})   linear CRPS={:.4} (n={})   uncensored CRPS={:.4} (n={})   ensemble CRPS={:.4} (n={})",
            t.oracle.mean(),
            t.oracle.n,
            t.linear.mean(),
            t.linear.n,
            t.uncensored.mean(),
            t.uncensored.n,
            t.ensemble.mean(),
            t.ensemble.n
        );
        total.oracle.sum += t.oracle.sum;
        total.oracle.n += t.oracle.n;
        total.linear.sum += t.linear.sum;
        total.linear.n += t.linear.n;
        total.uncensored.sum += t.uncensored.sum;
        total.uncensored.n += t.uncensored.n;
        total.ensemble.sum += t.ensemble.sum;
        total.ensemble.n += t.ensemble.n;
    }

    println!("\nAggregate mean CRPS (ticks -- lower is better, 0 is perfect):");
    println!(
        "  oracle (upper bound)          = {:.4}  (n={})",
        total.oracle.mean(),
        total.oracle.n
    );
    println!(
        "  linear (whole-history OLS)    = {:.4}  (n={})",
        total.linear.mean(),
        total.linear.n
    );
    println!(
        "  uncensored (drops cap reads)  = {:.4}  (n={})",
        total.uncensored.mean(),
        total.uncensored.n
    );
    println!(
        "  ensemble (bootstrap, uncensored) = {:.4}  (n={})",
        total.ensemble.mean(),
        total.ensemble.n
    );
}
