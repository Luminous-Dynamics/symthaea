// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Closes the exact original predeclared confirmatory gate
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`: "aggregate rung 5's and rung 2's Brier
//! score across N=5 seeds, habitable regime only") non-degenerately, using
//! `PopulationCensusObservationPolicy` at a `sample_size` chosen to actually vary in this regime.
//!
//! `population_census_verification.rs` already re-ran this exact gate with the fix and got a
//! clear pass (`fep_driven=0.1276` vs `historical_frequency=0.5000`), but disclosed a caveat:
//! at `sample_size=3`, the census signal was constant the entire run (population never dropped
//! near that threshold), so it demonstrated the *absence* of the old policy's bias, not
//! necessarily *presence* of new predictive skill. A direct probe confirmed why: true population
//! count in the habitable regime never drops below its 6-organism starting floor in any of the
//! same 5 seeds (min_count=6 in all 5), so any `sample_size <= 6` is structurally doomed to a
//! constant signal here -- but the population does range up to 28-32, so `sample_size=15`
//! (roughly mid-range) should see genuine crossings.
//!
//! Predeclared before running: whether `fep_driven` beats `historical_frequency` under a
//! genuinely varying signal is not predicted in advance -- observed honestly either way.
//!
//! Run: `cargo run --example population_census_verification_habitable_nondegenerate -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{
    BrierScore, ScoringRule, boolean_prediction_pair, reliability_diagram,
};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::{FepDrivenGenerator, HistoricalFrequencyGenerator};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, PopulationCensusObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55]; // same 5 seeds the original confirmatory gate used
const SAMPLE_SIZE: usize = 15; // mid-range of the observed 6-32 habitable-regime population span

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

struct SeedResult {
    r5_sum: f64,
    r5_n: usize,
    r2_sum: f64,
    r2_n: usize,
    pairs: Vec<(f64, bool)>,
    min_reported: usize,
    max_reported: usize,
}

fn run_seed(seed: u64) -> SeedResult {
    let env = EarthForcedEnvironment::earth_like(200.0); // habitable -- no dimming
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

    let reported: Vec<usize> = observations
        .iter()
        .filter_map(|o| o.sample.map(|s| s.sampled_alive_count))
        .collect();
    let min_reported = reported.iter().copied().min().unwrap_or(0);
    let max_reported = reported.iter().copied().max().unwrap_or(0);

    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let fep_driven = FepDrivenGenerator::default();

    let mut result = SeedResult {
        r5_sum: 0.0,
        r5_n: 0,
        r2_sum: 0.0,
        r2_n: 0,
        pairs: Vec::new(),
        min_reported,
        max_reported,
    };

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let obs = observations[checkpoint as usize];
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        if let ForecastOutput::Distribution(dist) =
            fep_driven.generate(&history_slice, Horizon(HORIZON))
        {
            result.r5_sum += BrierScore
                .score(&dist, &actual)
                .expect("scoring a validated forecast cannot fail")
                .get();
            result.r5_n += 1;
            if let Some(pair) = boolean_prediction_pair(&dist, &actual) {
                result.pairs.push(pair);
            }
        }
        if let ForecastOutput::Distribution(dist) = historical.generate(&obs, Horizon(HORIZON)) {
            result.r2_sum += BrierScore
                .score(&dist, &actual)
                .expect("scoring a validated forecast cannot fail")
                .get();
            result.r2_n += 1;
        }

        checkpoint += CHECKPOINT_STRIDE;
    }

    result
}

fn main() {
    println!(
        "Non-degenerate re-run of the original predeclared gate: habitable regime, sample_size={SAMPLE_SIZE}"
    );
    println!("{} seeds: {SEEDS:?}\n", SEEDS.len());

    let mut all_pairs = Vec::new();
    let (mut total_r5_sum, mut total_r5_n, mut total_r2_sum, mut total_r2_n) =
        (0.0, 0usize, 0.0, 0usize);

    for &seed in &SEEDS {
        let result = run_seed(seed);
        let r5_mean = result.r5_sum / result.r5_n as f64;
        let r2_mean = result.r2_sum / result.r2_n as f64;
        println!(
            "  seed {seed:3}: fep_driven={r5_mean:.4}  historical_frequency={r2_mean:.4}  reported_range=[{},{}]",
            result.min_reported, result.max_reported
        );
        total_r5_sum += result.r5_sum;
        total_r5_n += result.r5_n;
        total_r2_sum += result.r2_sum;
        total_r2_n += result.r2_n;
        all_pairs.extend(result.pairs);
    }

    let aggregate_r5 = total_r5_sum / total_r5_n as f64;
    let aggregate_r2 = total_r2_sum / total_r2_n as f64;

    println!("\nAggregate mean Brier across all {total_r5_n} checkpoints, all 5 seeds:");
    println!("  fep_driven           = {aggregate_r5:.4}");
    println!("  historical_frequency = {aggregate_r2:.4}");

    let fixed = aggregate_r5 < aggregate_r2;
    println!(
        "\n{}: fep_driven {} historical_frequency",
        if fixed { "GATE PASSES" } else { "GATE FAILS" },
        if fixed {
            "now beats"
        } else {
            "still does NOT beat"
        }
    );

    let true_count = all_pairs.iter().filter(|&&(_, actual)| actual).count();
    println!(
        "\nReliability check -- base rate (actual extinction-within-horizon = true): {:.4} ({}/{})",
        true_count as f64 / all_pairs.len() as f64,
        true_count,
        all_pairs.len()
    );
    let diagram = reliability_diagram(&all_pairs, 10);
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
        "Expected Calibration Error: {:.4}",
        diagram.expected_calibration_error()
    );
}
