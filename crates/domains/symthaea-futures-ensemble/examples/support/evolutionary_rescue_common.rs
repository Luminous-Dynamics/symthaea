// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared world-generation logic for the evolutionary-rescue Phase 2.2 examples. Included via
//! `#[path] mod common;` rather than promoted into the crate's own `src/` -- this is
//! example/fixture-generation code (`symthaea-alife` is a dev-dependency of this crate, not a
//! real one), not part of the crate's public API.
//!
//! **Phase 2.2C engineering prerequisite** (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`):
//! serialize the expensive world trajectories once so every subsequent model-comparison
//! experiment replays against them instead of re-running ~110,000 simulation ticks per pass.
//! [`record_world`] reproduces `evolutionary_rescue_backtest_and_gate.rs`'s original
//! `record_seed` logic exactly (same constants, same policy construction order, same RNG seeding
//! scheme) so a freshly-generated fixture is bit-identical to what that already-verified,
//! already-committed harness would compute inline -- this file does not change Phase 2.2B-ii's
//! result, only makes it (and follow-up experiments) replayable.

// This module is shared via `#[path]` across two separate example binaries, each of which uses
// only a subset of its items -- the rest are legitimately unused from that binary's own
// perspective, not dead code in the usual sense.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use symthaea_alife::{
    EarthForcedEnvironment, InheritanceMode, OrganismConfig, Population, PopulationConfig,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::evolutionary_rescue::{
    EvolutionaryRescueGroundTruth, EvolutionaryRescueObservation, EvolutionaryRescueSample,
    NoisyTraitObservationPolicy, PopulationCensusObservationPolicy,
    PrivilegedTraitObservationPolicy, shuffle_trait_readings,
};

// Same calibration `tests/phase7_evolutionary_rescue.rs` traced and validated, and the same
// values `evolutionary_rescue_backtest_and_gate.rs` already used -- not re-derived here.
pub const SEASONAL_PERIOD_TICKS: f64 = 200.0;
pub const SECULAR_DRIFT_PER_TICK: f64 = -0.01;
pub const PLANT_RESOURCE_TOTAL: f64 = 3.0;
pub const INITIAL_COUNT: usize = 12;
pub const MUTATION_RATE: f64 = 0.1;
pub const TICKS: u64 = 11_000;

pub const CENSUS_SAMPLE_SIZE: usize = 500;
pub const TRAIT_SAMPLE_SIZE: usize = 15;
pub const TRAIT_NOISE_AMPLITUDE: f64 = 0.03;

pub const TRAIN_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
pub const TEST_SEEDS: [u64; 5] = [66, 77, 88, 99, 111];

/// Bump this whenever `record_world`'s simulation/policy construction changes, so a fixture file
/// on disk can be recognized as stale rather than silently misread by a newer ablation script.
pub const WORLD_FORMAT_VERSION: &str = "evolutionary_rescue_world_v1";

pub fn population_config() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(), // 0.15 forage_efficiency -- knife-edge, real room to evolve
        mutation_rate: MUTATION_RATE,
        mutation_std: 0.05,
        inheritance: InheritanceMode::FromParent,
    }
}

/// One seed's fully-recorded world: every observation stream a Phase 2.2 model-comparison
/// experiment might need, plus enough provenance to tell a stale fixture apart from a fresh one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedWorld {
    pub format_version: String,
    pub seed: u64,
    pub census_obs: Vec<EvolutionaryRescueObservation>,
    pub noisy_obs: Vec<EvolutionaryRescueObservation>,
    pub shuffled_obs: Vec<EvolutionaryRescueObservation>,
    pub privileged_obs: Vec<EvolutionaryRescueObservation>,
    pub trajectory: Vec<bool>, // true == collapsed (extinct) at that tick
    pub max_population: usize,
    /// `None` if the population never went extinct within `TICKS`.
    pub first_extinction_tick: Option<u64>,
    /// The last trait level actually measured *before* extinction, if extinction ever happened
    /// (`true_mean_forage_efficiency` falls back to `0.0` once extinct, which would otherwise be
    /// misread as "the trait declined to zero").
    pub trait_level_just_before_extinction: Option<f64>,
}

/// Runs the full ~11,000-tick simulation for one seed and records every observation stream a
/// Phase 2.2 experiment needs. Expensive -- call once per seed via
/// `evolutionary_rescue_generate_worlds`, not per experiment.
pub fn record_world(seed: u64) -> SerializedWorld {
    let environment = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS)
        .with_secular_drift(SECULAR_DRIFT_PER_TICK);
    let population = Population::new(population_config(), INITIAL_COUNT, seed);
    let mut truth =
        EvolutionaryRescueGroundTruth::new(environment, population, PLANT_RESOURCE_TOTAL);

    // Independent noise-seed streams per policy -- must never derive from the simulation's own
    // seed (see symthaea-futures-symtropy::evolutionary_rescue's module docs).
    let mut census_policy = PopulationCensusObservationPolicy::new(CENSUS_SAMPLE_SIZE, 1);
    let mut noisy_policy = NoisyTraitObservationPolicy::new(
        CENSUS_SAMPLE_SIZE,
        TRAIT_SAMPLE_SIZE,
        TRAIT_NOISE_AMPLITUDE,
        1,
        seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
    );
    let mut privileged_policy = PrivilegedTraitObservationPolicy::new(CENSUS_SAMPLE_SIZE, 1);

    let mut census_obs = vec![census_policy.observe(&truth, 0)];
    let mut noisy_obs = vec![noisy_policy.observe(&truth, 0)];
    let mut privileged_obs = vec![privileged_policy.observe(&truth, 0)];
    let mut trajectory = vec![truth.is_extinct()];
    let mut max_population = truth.true_population_count();
    let mut first_extinction_tick: Option<u64> = None;
    let mut trait_level_just_before_extinction: Option<f64> = None;

    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        if !truth.is_extinct() {
            trait_level_just_before_extinction = Some(truth.true_mean_forage_efficiency());
        } else if first_extinction_tick.is_none() {
            first_extinction_tick = Some(tick);
        }
        census_obs.push(census_policy.observe(&truth, tick));
        noisy_obs.push(noisy_policy.observe(&truth, tick));
        privileged_obs.push(privileged_policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
        max_population = max_population.max(truth.true_population_count());
    }

    let trait_readings: Vec<Option<f64>> = noisy_obs
        .iter()
        .map(|o| o.sample.and_then(|s| s.observed_mean_forage_efficiency))
        .collect();
    let shuffled_readings = shuffle_trait_readings(&trait_readings, seed.wrapping_add(777));
    let shuffled_obs: Vec<EvolutionaryRescueObservation> = noisy_obs
        .iter()
        .zip(shuffled_readings)
        .map(|(o, shuffled_trait)| EvolutionaryRescueObservation {
            tick: o.tick,
            sample: o.sample.map(|s| EvolutionaryRescueSample {
                sampled_alive_count: s.sampled_alive_count,
                observed_mean_forage_efficiency: shuffled_trait,
            }),
        })
        .collect();

    SerializedWorld {
        format_version: WORLD_FORMAT_VERSION.to_string(),
        seed,
        census_obs,
        noisy_obs,
        shuffled_obs,
        privileged_obs,
        trajectory,
        max_population,
        first_extinction_tick,
        trait_level_just_before_extinction,
    }
}

pub fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/evolutionary_rescue")
}

pub fn fixture_path(seed: u64) -> std::path::PathBuf {
    fixtures_dir().join(format!("world_{seed}.json"))
}

pub fn load_world(seed: u64) -> SerializedWorld {
    let path = fixture_path(seed);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {}: {e} -- run `cargo run --release --example \
             evolutionary_rescue_generate_worlds -p symthaea-futures-ensemble` first",
            path.display()
        )
    });
    let world: SerializedWorld = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()));
    assert_eq!(
        world.format_version,
        WORLD_FORMAT_VERSION,
        "fixture {} was generated by a different format version ({} vs. expected {}) -- \
         regenerate it",
        path.display(),
        world.format_version,
        WORLD_FORMAT_VERSION
    );
    world
}
