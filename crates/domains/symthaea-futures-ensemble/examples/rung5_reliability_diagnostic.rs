// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic, not a gate: `examples/confirmatory_gate.rs` established that rung 5
//! (`FepDrivenGenerator`) loses to rung 2 (`HistoricalFrequencyGenerator`) on aggregate Brier
//! score in the habitable regime, consistently across 5 seeds. This example asks *how* it
//! fails — is it uniformly overconfident, systematically wrong in one direction, or just noisy —
//! by running `symthaea-futures-calibration::reliability_diagram` against the same real
//! backtest data for the first time (that function has only ever been exercised against
//! synthetic unit-test data before this).
//!
//! Run: `cargo run --example rung5_reliability_diagnostic -p symthaea-futures-ensemble`
//!
//! Unlike `confirmatory_gate.rs`, this file is diagnostic, not a predeclared test — it's fine to
//! read its output and let that inform which fix to try next.

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{boolean_prediction_pair, reliability_diagram};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::FepDrivenGenerator;
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, ExtinctionObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55]; // same 5 seeds confirmatory_gate.rs used

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

fn collect_predictions(seed: u64) -> Vec<(f64, bool)> {
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

fn main() {
    let mut all_predictions = Vec::new();
    for &seed in &SEEDS {
        all_predictions.extend(collect_predictions(seed));
    }

    println!(
        "rung 5 (fep_driven) reliability diagnostic -- habitable regime, {} seeds, {} predictions\n",
        SEEDS.len(),
        all_predictions.len()
    );

    let true_count = all_predictions
        .iter()
        .filter(|&&(_, actual)| actual)
        .count();
    println!(
        "Base rate (actual extinction-within-horizon = true): {:.4} ({}/{})\n",
        true_count as f64 / all_predictions.len() as f64,
        true_count,
        all_predictions.len()
    );

    let diagram = reliability_diagram(&all_predictions, 10);
    println!("Bucket                Predicted  Empirical  Count");
    for bucket in &diagram.buckets {
        println!(
            "[{:.1}, {:.1})            {:.4}     {:.4}     {}",
            bucket.bucket_low,
            bucket.bucket_high,
            bucket.mean_predicted_probability,
            bucket.empirical_frequency,
            bucket.count
        );
    }
    println!(
        "\nExpected Calibration Error: {:.4}",
        diagram.expected_calibration_error()
    );
}
