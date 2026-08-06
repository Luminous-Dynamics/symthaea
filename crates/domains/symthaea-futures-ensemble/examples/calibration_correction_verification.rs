// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Does `HistogramCalibrator` actually help on genuinely held-out data? This is the honest
//! version of the question `rung5_convergence_probe.rs` raised: rung 5, under
//! `PopulationCensusObservationPolicy`, tracks real decline but plateaus underconfident
//! (~0.74 instead of ~1.0) once truly extinct.
//!
//! **Train/test seed split, the whole point of this file**: [`TRAIN_SEEDS`] are the same 5
//! seeds every prior analysis in this plan used (11/22/33/44/55) — legitimate to reuse for
//! *fitting*, since they were already "seen" during diagnosis. [`TEST_SEEDS`] are five seeds
//! that have never appeared anywhere in this plan before (66/77/88/99/111) — genuinely held out.
//! The calibrator is fit ONLY on `TRAIN_SEEDS`' predictions and evaluated ONLY on `TEST_SEEDS`'.
//! Reusing `TRAIN_SEEDS` for evaluation, or picking `TEST_SEEDS` after peeking at their result,
//! would be exactly the look-ahead bias this whole exercise exists to avoid.
//!
//! Run: `cargo run --example calibration_correction_verification -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{
    BrierScore, HistogramCalibrator, ScoringRule, boolean_prediction_pair, reliability_diagram,
};
use symthaea_futures_core::{
    ForecastBranch, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, OutcomeSpaceId,
    TrajectoryGenerator,
};
use symthaea_futures_ensemble::ecological::FepDrivenGenerator;
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, PopulationCensusObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;
const SAMPLE_SIZE: usize = 3;
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

/// Collects (raw predicted P(true), actual) pairs from rung 5, dimmed-collapse regime, one seed.
fn collect_predictions(seed: u64) -> Vec<(f64, bool)> {
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

    let fep_driven = FepDrivenGenerator::default();
    let mut pairs = Vec::new();

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        if let ForecastOutput::Distribution(dist) =
            fep_driven.generate(&history_slice, Horizon(HORIZON))
        {
            if let Some(pair) = boolean_prediction_pair(&dist, &actual) {
                pairs.push(pair);
            }
        }

        checkpoint += CHECKPOINT_STRIDE;
    }

    pairs
}

fn boolean_forecast(p_true: f64) -> ForecastDistribution {
    ForecastDistribution::try_from_raw(
        0,
        Horizon(HORIZON),
        OutcomeSpaceId("calibration_correction_verification".to_string()),
        vec![
            (p_true, OutcomeRegion::Boolean(true), Vec::new()),
            (1.0 - p_true, OutcomeRegion::Boolean(false), Vec::new()),
        ],
        0.0,
    )
    .expect("complementary boolean masses are valid by construction")
}

fn mean_brier(pairs: &[(f64, bool)]) -> f64 {
    let sum: f64 = pairs
        .iter()
        .map(|&(p, actual)| {
            BrierScore
                .score(&boolean_forecast(p), &OutcomeRegion::Boolean(actual))
                .expect("scoring a validated forecast cannot fail")
                .get()
        })
        .sum();
    sum / pairs.len() as f64
}

fn main() {
    println!("Calibration correction verification -- proper train/test seed separation");
    println!("Train seeds (fit only): {TRAIN_SEEDS:?}");
    println!("Test seeds (evaluate only, never used for fitting): {TEST_SEEDS:?}\n");

    let mut train_pairs = Vec::new();
    for &seed in &TRAIN_SEEDS {
        train_pairs.extend(collect_predictions(seed));
    }
    println!("Collected {} training predictions.", train_pairs.len());

    let calibrator = HistogramCalibrator::fit(&train_pairs, 10);

    let mut test_pairs = Vec::new();
    for &seed in &TEST_SEEDS {
        test_pairs.extend(collect_predictions(seed));
    }
    println!(
        "Collected {} held-out test predictions.\n",
        test_pairs.len()
    );

    let raw_mean_brier = mean_brier(&test_pairs);
    let raw_diagram = reliability_diagram(&test_pairs, 10);
    let raw_ece = raw_diagram.expected_calibration_error();

    let calibrated_pairs: Vec<(f64, bool)> = test_pairs
        .iter()
        .map(|&(p, actual)| (calibrator.calibrate(p), actual))
        .collect();
    let calibrated_mean_brier = mean_brier(&calibrated_pairs);
    let calibrated_diagram = reliability_diagram(&calibrated_pairs, 10);
    let calibrated_ece = calibrated_diagram.expected_calibration_error();

    println!("On held-out test seeds (never used for fitting):");
    println!("  Raw        mean Brier = {raw_mean_brier:.4}   ECE = {raw_ece:.4}");
    println!("  Calibrated mean Brier = {calibrated_mean_brier:.4}   ECE = {calibrated_ece:.4}");

    let helped = calibrated_ece < raw_ece;
    println!(
        "\n{}: calibration correction {} ECE on genuinely held-out data",
        if helped { "HELPED" } else { "DID NOT HELP" },
        if helped { "reduced" } else { "did not reduce" }
    );
}
