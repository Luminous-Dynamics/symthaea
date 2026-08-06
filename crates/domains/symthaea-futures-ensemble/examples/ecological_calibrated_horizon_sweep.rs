// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2 addendum item 4 (ecological half): `horizon_decay_sweep.rs` only tested raw
//! (uncalibrated) rungs, at `HORIZON=100` only for every calibration test in this plan. This asks
//! whether `HistogramCalibrator`'s correction -- fit *once*, at `HORIZON=100`, on the same 5
//! seeds `calibration_correction_verification.rs` uses -- still helps when the underlying
//! prediction is made at a horizon the calibrator was never fit on. Held-out seeds only; the
//! calibrator never sees test-seed data at any horizon.
//!
//! Uses `PopulationCensusObservationPolicy` (not `horizon_decay_sweep.rs`'s
//! `ExtinctionObservationPolicy`) because that's what `HistogramCalibrator` was actually fit
//! against in every prior calibration test in this plan -- applying a calibrator to predictions
//! from a different observation policy than it was validated on would be invalid. Also reuses
//! the `MAX_CHECKPOINT` fix `horizon_decay_sweep.rs` needed: capping evaluated checkpoints to the
//! genuinely uncertain boundary region, not the full 4000-tick trajectory.
//!
//! Predeclared before running: whether the calibrator's benefit holds, degrades, or vanishes
//! away from its fitting horizon is genuinely unknown in advance.
//!
//! Run: `cargo run --example ecological_calibrated_horizon_sweep -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{
    HistogramCalibrator, boolean_prediction_pair, reliability_diagram,
};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::FepDrivenGenerator;
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, PopulationCensusObservationPolicy,
};

const FIT_HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 50;
const MAX_CHECKPOINT: u64 = 350;
const HORIZONS: [u64; 6] = [10, 25, 50, 100, 200, 400];
const SAMPLE_SIZE: usize = 3;
const NUM_BUCKETS: usize = 5;
const TRAIN_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const TEST_SEEDS: [u64; 5] = [66, 77, 88, 99, 111];

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

fn run_seed_observations(seed: u64) -> (Vec<EcologicalObservation>, Vec<bool>) {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = 600.0; // dimmed past the snowball threshold -- guaranteed collapse
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);
    let mut policy = PopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut observations = vec![policy.observe(&truth, 0)];
    let mut trajectory = vec![truth.is_extinct()];
    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        observations.push(policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
    }
    (observations, trajectory)
}

/// Collects (predicted P(true), actual) pairs at a given horizon, checkpoints capped to
/// `0..=MAX_CHECKPOINT` (the same fix `horizon_decay_sweep.rs` needed).
fn collect_pairs_at_horizon(
    seeds: &[u64],
    horizon: u64,
    fep_driven: &FepDrivenGenerator,
) -> Vec<(f64, bool)> {
    let mut pairs = Vec::new();
    for &seed in seeds {
        let (observations, trajectory) = run_seed_observations(seed);
        let mut checkpoint = 0u64;
        while checkpoint <= MAX_CHECKPOINT && checkpoint + horizon < trajectory.len() as u64 {
            let actual = trajectory[(checkpoint + horizon) as usize];
            let history_slice: Vec<EcologicalObservation> =
                observations[..=checkpoint as usize].to_vec();
            if let ForecastOutput::Distribution(dist) =
                fep_driven.generate(&history_slice, Horizon(horizon))
            {
                if let Some(pair) = boolean_prediction_pair(&dist, &OutcomeRegion::Boolean(actual))
                {
                    pairs.push(pair);
                }
            }
            checkpoint += CHECKPOINT_STRIDE;
        }
    }
    pairs
}

fn main() {
    println!("Calibrated horizon-decay sweep -- does a calibrator fit at one horizon generalize?");
    println!(
        "Fit at HORIZON={FIT_HORIZON} on train seeds {TRAIN_SEEDS:?}, evaluated on held-out {TEST_SEEDS:?}\n"
    );

    let fep_driven = FepDrivenGenerator::default();

    let train_pairs = collect_pairs_at_horizon(&TRAIN_SEEDS, FIT_HORIZON, &fep_driven);
    println!(
        "Fit calibrator on {} training predictions at horizon={FIT_HORIZON}.\n",
        train_pairs.len()
    );
    let calibrator = HistogramCalibrator::fit(&train_pairs, 10);

    println!(
        "{:>8}  {:>14}  {:>14}",
        "horizon", "raw ECE", "calibrated ECE"
    );
    for &horizon in &HORIZONS {
        let test_pairs = collect_pairs_at_horizon(&TEST_SEEDS, horizon, &fep_driven);
        let raw_ece = reliability_diagram(&test_pairs, NUM_BUCKETS).expected_calibration_error();

        let calibrated_pairs: Vec<(f64, bool)> = test_pairs
            .iter()
            .map(|&(p, actual)| (calibrator.calibrate(p), actual))
            .collect();
        let calibrated_ece =
            reliability_diagram(&calibrated_pairs, NUM_BUCKETS).expected_calibration_error();

        println!(
            "{horizon:>8}  {raw_ece:>14.4}  {calibrated_ece:>14.4}  (n={})",
            test_pairs.len()
        );
    }
}
