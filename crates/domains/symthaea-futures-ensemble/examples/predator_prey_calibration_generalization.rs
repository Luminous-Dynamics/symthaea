// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Answers the plan's long-open question: does `HistogramCalibrator`'s post-hoc correction
//! generalize beyond the `ecological` family's bimodal regime? Two prior attempts on
//! `predator_prey` failed to even construct a genuinely varied target -- the original sustaining
//! parameters and a 6-configuration parameter probe both found predator extinction essentially
//! never happens. `examples/predator_prey_resource_scarcity_probe.rs` found the missing lever:
//! `plant_resource_total` (fixed at `3.0` in every prior attempt, never varied) is this family's
//! analog to `ecological`'s "dimmed sun." At `plant_resource_total=0.5`, predator extinction
//! occurs in all 5 probe seeds, with real timing variance (39-270 ticks) and prey surviving in
//! every seed -- a clean, predator-specific, genuinely varied target at last.
//!
//! Same train/test seed discipline as `calibration_correction_verification.rs`: `TRAIN_SEEDS`
//! are the 5 seeds every predator_prey probe so far has used (legitimate to reuse for fitting,
//! already "seen"); `TEST_SEEDS` are five seeds that have never appeared in any predator_prey
//! run in this plan -- genuinely held out. Fit only on `TRAIN_SEEDS`, evaluate only on
//! `TEST_SEEDS`.
//!
//! Run: `cargo run --example predator_prey_calibration_generalization -p symthaea-futures-ensemble`

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};
use symthaea_futures_calibration::{
    BrierScore, HistogramCalibrator, ScoringRule, boolean_prediction_pair, reliability_diagram,
};
use symthaea_futures_core::{
    ForecastBranch, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, OutcomeSpaceId,
    TrajectoryGenerator,
};
use symthaea_futures_ensemble::predator_prey::FepDrivenGenerator;
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::predator_prey::{
    PredatorPopulationCensusObservationPolicy, PredatorPreyObservation,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 500;
const CHECKPOINT_STRIDE: u64 = 20;
const SAMPLE_SIZE: usize = 3;
const INITIAL_PREY: usize = 10;
const INITIAL_PREDATORS: usize = 3;
const TRAIN_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const TEST_SEEDS: [u64; 5] = [66, 77, 88, 99, 111];

fn config() -> PredatorPreyConfig {
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
        plant_resource_total: 0.5, // the winning scarcity level from the resource-scarcity probe
        predation_scale: 0.05,
        predation_efficiency: 0.05,
    }
}

/// Collects (raw predicted P(true), actual) pairs from rung 5, resource-scarce predator/prey
/// regime, one seed.
fn collect_predictions(seed: u64) -> Vec<(f64, bool)> {
    let mut sim = PredatorPreySim::new(config(), INITIAL_PREY, INITIAL_PREDATORS, seed);
    let mut policy = PredatorPopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut observations: Vec<PredatorPreyObservation> = vec![policy.observe(&sim, 0)];
    let mut trajectory: Vec<bool> = vec![sim.predator.len() == 0];

    for _ in 0..TICKS {
        sim.step();
        let tick = sim.t;
        observations.push(policy.observe(&sim, tick));
        trajectory.push(sim.predator.len() == 0);
    }

    let fep_driven = FepDrivenGenerator::default();
    let mut pairs = Vec::new();

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<PredatorPreyObservation> =
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
        OutcomeSpaceId("predator_prey_calibration_generalization".to_string()),
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
    println!("Predator/prey calibration generalization -- proper train/test seed separation");
    println!(
        "plant_resource_total=0.5 (the resource-scarcity probe's clean predator-specific collapse)"
    );
    println!("Train seeds (fit only): {TRAIN_SEEDS:?}");
    println!("Test seeds (evaluate only, never used for fitting): {TEST_SEEDS:?}\n");

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
        "\n{}: calibration correction {} ECE on genuinely held-out predator/prey data",
        if helped { "HELPED" } else { "DID NOT HELP" },
        if helped { "reduced" } else { "did not reduce" }
    );
}
