// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Follow-up to `population_census_verification.rs`: that run showed
//! `PopulationCensusObservationPolicy` fixing rung 5's habitable-regime failure, but every
//! prediction landed in one bucket with an identical value, because the population never
//! actually approached `sample_size` during any of those runs — the fix removed a bias but
//! didn't yet demonstrate the new signal responds to real variation.
//!
//! This runs the SAME methodology in the dimmed-sun collapse regime, where the population
//! genuinely declines to zero (a real, ground-truth-verified event — see
//! `symthaea-alife`'s own `phase5_earth_forcing.rs` and this codebase's `oracle_matches_a_real_
//! simulated_collapse` test), so the census signal actually crosses `sample_size` during the
//! run. Does `p_true` correctly rise as the true population approaches and crosses the
//! threshold, or does it stay flat like the habitable-regime run?
//!
//! Run: `cargo run --example population_census_collapse_verification -p symthaea-futures-ensemble`

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
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const SAMPLE_SIZE: usize = 3;

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
    /// (checkpoint tick, true population count, predicted p_true) -- for tracing whether the
    /// prediction tracks the real decline, not just the aggregate score.
    trace: Vec<(u64, usize, f64)>,
}

fn run_seed(seed: u64) -> SeedResult {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = 600.0; // dimmed past the snowball threshold -- guaranteed collapse
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);

    let mut policy = PopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut observations: Vec<EcologicalObservation> = vec![policy.observe(&truth, 0)];
    let mut trajectory: Vec<bool> = vec![truth.is_extinct()];
    let mut true_counts: Vec<usize> = vec![truth.true_population_count()];

    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        observations.push(policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
        true_counts.push(truth.true_population_count());
    }

    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let fep_driven = FepDrivenGenerator::default();

    let mut result = SeedResult {
        r5_sum: 0.0,
        r5_n: 0,
        r2_sum: 0.0,
        r2_n: 0,
        pairs: Vec::new(),
        trace: Vec::new(),
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
            if let Some((p_true, actual_bool)) = boolean_prediction_pair(&dist, &actual) {
                result.pairs.push((p_true, actual_bool));
                result
                    .trace
                    .push((checkpoint, true_counts[checkpoint as usize], p_true));
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
        "Does PopulationCensusObservationPolicy track a REAL decline? (dimmed-sun collapse regime)"
    );
    println!(
        "{} seeds: {SEEDS:?}, sample_size={SAMPLE_SIZE}\n",
        SEEDS.len()
    );

    let mut all_pairs = Vec::new();
    let (mut total_r5_sum, mut total_r5_n, mut total_r2_sum, mut total_r2_n) =
        (0.0, 0usize, 0.0, 0usize);

    for &seed in &SEEDS {
        let result = run_seed(seed);
        let r5_mean = result.r5_sum / result.r5_n as f64;
        let r2_mean = result.r2_sum / result.r2_n as f64;
        println!("  seed {seed:3}: fep_driven={r5_mean:.4}  historical_frequency={r2_mean:.4}");
        println!("    trace (checkpoint_tick, true_population, predicted_p_true):");
        for (tick, count, p) in &result.trace {
            println!("      tick={tick:5}  true_population={count:3}  p_true={p:.4}");
        }
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
