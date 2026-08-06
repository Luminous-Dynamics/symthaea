// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! First real use of `symthaea-futures-analysis`, per the Phase 2 addendum
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`, item 2): does inter-model disagreement at
//! a checkpoint predict where the best individual model (`FepDrivenGenerator`, rung 5) is more
//! likely to be wrong? Uses data this apparatus already knows how to produce -- no new scenario
//! simulation -- reusing the exact dimmed-sun collapse fixture
//! `population_census_collapse_verification.rs` already proved is non-degenerate
//! (`PopulationCensusObservationPolicy`, `sample_size=3`, 5 seeds).
//!
//! At every checkpoint where all 5 real rungs (persistence, historical_frequency,
//! simple_statistical, scenario_mechanistic, fep_driven -- oracle excluded, it isn't a real
//! predictor) produce a forecast, computes `boolean_disagreement_variance` across their five
//! `P(true)` values, and pairs it with `fep_driven`'s squared error at that checkpoint.
//!
//! Predeclared before running: the direction of this relationship is genuinely unknown in
//! advance -- reported honestly either way, not fit to a story after the fact.
//!
//! Run: `cargo run --example model_disagreement_probe -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_analysis::{boolean_disagreement_variance, boolean_p_true};
use symthaea_futures_core::{ForecastOutput, Horizon, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::{
    FepDrivenGenerator, HistoricalFrequencyGenerator, PersistenceGenerator,
    ScenarioMechanisticGenerator, SimpleStatisticalGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, PopulationCensusObservationPolicy,
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

/// One evaluated checkpoint: how much the 5 real rungs disagreed, and how wrong the best of
/// them (`fep_driven`) turned out to be.
struct Point {
    disagreement: f64,
    fep_squared_error: f64,
}

fn collect_points(seed: u64) -> Vec<Point> {
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

    let persistence = PersistenceGenerator;
    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let statistical = SimpleStatisticalGenerator;
    let mechanistic = ScenarioMechanisticGenerator {
        per_member_death_probability: 0.01,
    };
    let fep_driven = FepDrivenGenerator::default();

    let mut points = Vec::new();

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let obs = observations[checkpoint as usize];
        let actual = trajectory[(checkpoint + HORIZON) as usize];
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        let outputs = [
            persistence.generate(&obs, Horizon(HORIZON)),
            historical.generate(&obs, Horizon(HORIZON)),
            statistical.generate(&history_slice, Horizon(HORIZON)),
            mechanistic.generate(&obs, Horizon(HORIZON)),
            fep_driven.generate(&history_slice, Horizon(HORIZON)),
        ];

        // Require all 5 to have an opinion -- a clean, unambiguous 5-way disagreement metric,
        // rather than diluting it with partial sets when a rung abstains.
        let dists: Option<Vec<_>> = outputs
            .iter()
            .map(|o| match o {
                ForecastOutput::Distribution(d) => Some(d.clone()),
                ForecastOutput::Abstain(_) => None,
            })
            .collect();
        let Some(dists) = dists else {
            checkpoint += CHECKPOINT_STRIDE;
            continue;
        };

        let Some(disagreement) = boolean_disagreement_variance(&dists) else {
            checkpoint += CHECKPOINT_STRIDE;
            continue;
        };

        let fep_p_true = boolean_p_true(&dists[4]);
        let actual_f = if actual { 1.0 } else { 0.0 };
        let fep_squared_error = (fep_p_true - actual_f).powi(2);

        points.push(Point {
            disagreement,
            fep_squared_error,
        });

        checkpoint += CHECKPOINT_STRIDE;
    }

    points
}

fn pearson_correlation(points: &[Point]) -> f64 {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.disagreement).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.fep_squared_error).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for p in points {
        let dx = p.disagreement - mean_x;
        let dy = p.fep_squared_error - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 0.0 || var_y <= 0.0 {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn main() {
    println!("Model-disagreement probe -- does inter-rung disagreement predict rung-5 error?");
    println!(
        "Dimmed-sun collapse, {} seeds, sample_size={SAMPLE_SIZE}, horizon={HORIZON}\n",
        SEEDS.len()
    );

    let mut all_points: Vec<Point> = Vec::new();
    for &seed in &SEEDS {
        all_points.extend(collect_points(seed));
    }

    println!(
        "Collected {} checkpoints with all 5 rungs opining.\n",
        all_points.len()
    );

    let mut sorted_by_disagreement: Vec<&Point> = all_points.iter().collect();
    sorted_by_disagreement.sort_by(|a, b| a.disagreement.partial_cmp(&b.disagreement).unwrap());
    let median_idx = sorted_by_disagreement.len() / 2;

    let (low_half, high_half) = sorted_by_disagreement.split_at(median_idx);
    let mean_error = |half: &[&Point]| -> f64 {
        half.iter().map(|p| p.fep_squared_error).sum::<f64>() / half.len() as f64
    };

    let low_mean_error = mean_error(low_half);
    let high_mean_error = mean_error(high_half);
    let correlation = pearson_correlation(&all_points);

    println!(
        "Low-disagreement half (n={}): mean fep_driven squared error = {:.4}",
        low_half.len(),
        low_mean_error
    );
    println!(
        "High-disagreement half (n={}): mean fep_driven squared error = {:.4}",
        high_half.len(),
        high_mean_error
    );
    println!("\nPearson correlation (disagreement, fep_driven squared error) = {correlation:.4}");

    let predicts_difficulty = high_mean_error > low_mean_error && correlation > 0.1;
    println!(
        "\n{}: high inter-rung disagreement {} associated with higher rung-5 error here",
        if predicts_difficulty {
            "SIGNAL FOUND"
        } else {
            "NO CLEAR SIGNAL"
        },
        if predicts_difficulty {
            "is"
        } else {
            "is not clearly"
        }
    );
}
