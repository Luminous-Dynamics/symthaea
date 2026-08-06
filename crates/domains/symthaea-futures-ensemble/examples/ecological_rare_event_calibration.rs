// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2 addendum item 3: does `HistogramCalibrator`'s post-hoc correction still help when the
//! positive class is genuinely rare (single digits), not the 50/50-ish or 90/10 splits every
//! prior calibration test in this plan has used? `ecological_rare_event_probe.rs` found
//! `solar_constant=1272.0` gives a clean, uniform 5.0% checkpoint-level true rate (1/20 per seed,
//! identical across all 5 probed seeds) -- unlike `predator_prey`'s chaotic transition zone, this
//! family's dose-response here is remarkably deterministic.
//!
//! Same train/test seed discipline as `calibration_correction_verification.rs`, but with 15
//! seeds per split rather than 5 -- at a ~5% base rate, 5 seeds would only give ~5 positive
//! examples per split, too thin to draw an honest conclusion from. Train and test seed sets are
//! both freshly chosen (1000-1014 and 2000-2014), disjoint from each other and from every seed
//! used elsewhere in this plan.
//!
//! Predeclared before running: whether calibration still helps under this rare-event regime is
//! genuinely unknown in advance.
//!
//! Run: `cargo run --example ecological_rare_event_calibration -p symthaea-futures-ensemble`

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
const SOLAR_CONSTANT: f64 = 1272.0; // the rare-event probe's clean, uniform 5% base-rate value
const TRAIN_SEEDS: [u64; 15] = [
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
];
const TEST_SEEDS: [u64; 15] = [
    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
];

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

/// Collects (raw predicted P(true), actual) pairs from rung 5, rare-event regime, one seed.
fn collect_predictions(seed: u64) -> Vec<(f64, bool)> {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = SOLAR_CONSTANT;
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
        OutcomeSpaceId("ecological_rare_event_calibration".to_string()),
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
    println!("Ecological rare-event calibration generalization -- solar_constant={SOLAR_CONSTANT}");
    println!(
        "Train seeds (fit only, {}): {TRAIN_SEEDS:?}",
        TRAIN_SEEDS.len()
    );
    println!(
        "Test seeds (evaluate only, {}): {TEST_SEEDS:?}\n",
        TEST_SEEDS.len()
    );

    let mut train_pairs = Vec::new();
    for &seed in &TRAIN_SEEDS {
        train_pairs.extend(collect_predictions(seed));
    }
    let train_true_count = train_pairs.iter().filter(|&&(_, a)| a).count();
    println!(
        "Collected {} training predictions ({} true, {} false).",
        train_pairs.len(),
        train_true_count,
        train_pairs.len() - train_true_count
    );

    let calibrator = HistogramCalibrator::fit(&train_pairs, 10);

    let mut test_pairs = Vec::new();
    for &seed in &TEST_SEEDS {
        test_pairs.extend(collect_predictions(seed));
    }
    let test_true_count = test_pairs.iter().filter(|&&(_, a)| a).count();
    println!(
        "Collected {} held-out test predictions ({} true, {} false).\n",
        test_pairs.len(),
        test_true_count,
        test_pairs.len() - test_true_count
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
        "\n{}: calibration correction {} ECE under this rare-event (~5% base rate) regime",
        if helped { "HELPED" } else { "DID NOT HELP" },
        if helped { "reduced" } else { "did not reduce" }
    );
}
