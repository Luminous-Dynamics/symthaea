// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only diagnostic for the `time_to_extinction_backtest` result: dumps the raw
//! `sampled_alive_count` trajectory for seed 11 to confirm (or refute) the hypothesis that
//! `PopulationCensusObservationPolicy`'s capped signal (`min(true_count, sample_size)`) is flat
//! at the cap for most of the run, then crashes abruptly right at the end -- which would make a
//! single whole-history OLS fit a poor model (dominated by the flat segment, giving a slope near
//! zero and therefore a wildly displaced crossing point).

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, PopulationCensusObservationPolicy,
};

const TICKS: u64 = 4000;
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

fn main() {
    let seed = 11u64;
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = 600.0;
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);
    let mut policy = PopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut reported: Vec<(u64, usize, usize)> = Vec::new(); // (tick, reported, true_count)
    let obs0 = policy.observe(&truth, 0);
    reported.push((
        0,
        obs0.sample.map(|s| s.sampled_alive_count).unwrap_or(0),
        truth.true_population_count(),
    ));

    let mut extinction_tick = None;
    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        let obs = policy.observe(&truth, tick);
        reported.push((
            tick,
            obs.sample.map(|s| s.sampled_alive_count).unwrap_or(0),
            truth.true_population_count(),
        ));
        if truth.is_extinct() && extinction_tick.is_none() {
            extinction_tick = Some(tick);
        }
    }
    let extinction_tick = extinction_tick.expect("should collapse");
    println!("seed {seed}: extinction_tick={extinction_tick}");

    // Show the transition: first tick where reported count drops below SAMPLE_SIZE, and the
    // last 15 points before extinction.
    let first_below_cap = reported.iter().find(|&&(_, r, _)| r < SAMPLE_SIZE);
    println!(
        "first tick with reported < sample_size cap: {:?}",
        first_below_cap
    );

    println!("\nlast 20 points before extinction (tick, reported, true_count):");
    for &(t, r, tc) in reported
        .iter()
        .filter(|&&(t, _, _)| t + 20 >= extinction_tick && t <= extinction_tick)
    {
        println!("  tick={t:5}  reported={r}  true={tc}");
    }

    // Now replicate the OLS fit exactly as TimeToExtinctionLinearGenerator does, using the
    // FULL history up to a checkpoint just before extinction (matching the backtest's stride
    // logic: last checkpoint < extinction_tick).
    let checkpoint = (extinction_tick / 200) * 200;
    let points: Vec<(f64, f64)> = reported
        .iter()
        .filter(|&&(t, _, _)| t <= checkpoint)
        .map(|&(t, r, _)| (t as f64, r as f64))
        .collect();
    let n = points.len() as f64;
    let x_mean = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let y_mean = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let denom: f64 = points.iter().map(|&(x, _)| (x - x_mean).powi(2)).sum();
    let slope = points
        .iter()
        .map(|&(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / denom;
    let intercept = y_mean - slope * x_mean;
    let cross_tick = -intercept / slope;

    println!(
        "\nOLS fit at checkpoint={checkpoint} (n={} points, whole history from tick 0):",
        points.len()
    );
    println!(
        "  x_mean={x_mean:.4}  y_mean={y_mean:.4}  slope={slope:.8}  intercept={intercept:.4}"
    );
    println!("  predicted cross_tick={cross_tick:.2}  (true extinction_tick={extinction_tick})");
    println!(
        "  predicted time_to_extinction from checkpoint = {:.2}  (true = {})",
        cross_tick - checkpoint as f64,
        extinction_tick - checkpoint
    );
}
