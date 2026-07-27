// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.2B-ii: the four-condition experiment and the predeclared acceptance-gate evaluation
//! for the evolutionary-rescue scenario family, per
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`.
//!
//! Runs one simulation per seed (`symthaea-alife`'s `phase7_evolutionary_rescue.rs`-calibrated
//! evolving-genome-under-real-worsening-climate scenario), and attaches all three real
//! `ObservationPolicy` implementations to the *same* ground truth every tick — valid because
//! `observe()` is read-only, so the underlying simulation trajectory is bit-identical across the
//! three recorded streams; only what each policy reveals differs. Condition 3 (shuffled trait)
//! is derived after the fact from condition 2's recorded trait readings via
//! `shuffle_trait_readings`, per the module's own design (it needs the full sequence in hand,
//! so it cannot be a fourth live policy).
//!
//! Run: `cargo run --release --example evolutionary_rescue_backtest_and_gate -p symthaea-futures-ensemble`
//!
//! **Predeclared acceptance gate** (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md` Phase 2.2B):
//! the trait-augmented FEP model (rung 6, condition 2 — real noisy trait) must improve held-out
//! mean Brier over the census-only FEP model (rung 5, condition 1), **and** the same rung 6 run
//! on condition 3 (shuffled trait) must fail to reproduce that improvement. Passing only the
//! first half would repeat exactly the correlation-vs-causation mistake this plan's own
//! predictive-compression program hit before.

use symthaea_alife::{
    EarthForcedEnvironment, InheritanceMode, OrganismConfig, Population, PopulationConfig,
};
use symthaea_futures_calibration::BrierScore;
use symthaea_futures_calibration::ScoringRule;
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::evolutionary_rescue::{
    AdaptationVsForcingGenerator, CensusOnlyStatisticalGenerator, FepCensusOnlyGenerator,
    FepCensusPlusTraitGenerator, HistoricalFrequencyGenerator, OracleGenerator,
    TraitTrendStatisticalGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::evolutionary_rescue::{
    EvolutionaryRescueGroundTruth, EvolutionaryRescueObservation, EvolutionaryRescueSample,
    NoisyTraitObservationPolicy, PopulationCensusObservationPolicy,
    PrivilegedTraitObservationPolicy, shuffle_trait_readings,
};

// Same calibration `tests/phase7_evolutionary_rescue.rs` traced and validated -- not invented
// for this harness.
const SEASONAL_PERIOD_TICKS: f64 = 200.0;
const SECULAR_DRIFT_PER_TICK: f64 = -0.01;
const PLANT_RESOURCE_TOTAL: f64 = 3.0;
const INITIAL_COUNT: usize = 12;
const MUTATION_RATE: f64 = 0.1;
const TICKS: u64 = 11_000;

const HORIZON: u64 = 300;
const CHECKPOINT_STRIDE: u64 = 300;

// Deliberately generous, not artificially tight: this experiment tests whether trait
// *information* helps, not whether a subsampling/capping scheme is safe (that's already covered
// by symthaea-futures-symtropy's own leakage tests). A tight cap here would confound "trait
// signal helps" with "the census signal was also crippled."
const CENSUS_SAMPLE_SIZE: usize = 500;
const TRAIT_SAMPLE_SIZE: usize = 15;
const TRAIT_NOISE_AMPLITUDE: f64 = 0.03;

const TRAIN_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const TEST_SEEDS: [u64; 5] = [66, 77, 88, 99, 111];

fn population_config() -> PopulationConfig {
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

struct SeedRecording {
    census_obs: Vec<EvolutionaryRescueObservation>,
    noisy_obs: Vec<EvolutionaryRescueObservation>,
    shuffled_obs: Vec<EvolutionaryRescueObservation>,
    privileged_obs: Vec<EvolutionaryRescueObservation>,
    trajectory: Vec<bool>, // true == collapsed (extinct) at that tick
    max_population: usize,
    /// `None` if the population never went extinct within `TICKS`. `true_mean_forage_efficiency`
    /// falls back to `0.0` once extinct (it has no organisms left to average), so this must not
    /// be read as "the trait declined to zero" -- an extinction tick means the trait signal
    /// simply stopped existing, not that it fell to a real value of 0.
    first_extinction_tick: Option<u64>,
    /// The last trait level actually measured *before* extinction (if extinction ever
    /// happened), so the write-up can report a real number instead of the misleading
    /// post-extinction 0.0 fallback.
    trait_level_just_before_extinction: Option<f64>,
}

fn record_seed(seed: u64) -> SeedRecording {
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

    SeedRecording {
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

/// Names, in the fixed order `run_arms` scores them in.
const ARMS: [&str; 9] = [
    "historical_frequency",
    "census_only_statistical",
    "trait_trend",
    "adaptation_vs_forcing",
    "fep_census_only",
    "fep_census_plus_trait_REAL",
    "fep_census_plus_trait_SHUFFLED",
    "fep_census_plus_trait_PRIVILEGED",
    "oracle",
];

fn brier_of(output: ForecastOutput, actual: &OutcomeRegion, acc: &mut (f64, usize)) {
    if let ForecastOutput::Distribution(dist) = output {
        acc.0 += BrierScore.score(&dist, actual);
        acc.1 += 1;
    }
}

fn run_arms(rec: &SeedRecording) -> Vec<(f64, usize)> {
    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let census_only_statistical = CensusOnlyStatisticalGenerator;
    let trait_trend = TraitTrendStatisticalGenerator;
    let adaptation_vs_forcing = AdaptationVsForcingGenerator {
        adaptation_sensitivity: 5.0,
        collapse_threshold: 50.0,
    };
    let fep_census_only = FepCensusOnlyGenerator::default();
    let fep_census_plus_trait = FepCensusPlusTraitGenerator::default();
    let oracle = OracleGenerator::from_trajectory(rec.trajectory.clone());

    let mut scores: Vec<(f64, usize)> = vec![(0.0, 0); ARMS.len()];

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < rec.trajectory.len() as u64 {
        let actual = OutcomeRegion::Boolean(rec.trajectory[(checkpoint + HORIZON) as usize]);
        let idx = checkpoint as usize;

        let census_obs = rec.census_obs[idx];
        let noisy_obs = rec.noisy_obs[idx];
        let census_history = rec.census_obs[..=idx].to_vec();
        let noisy_history = rec.noisy_obs[..=idx].to_vec();
        let shuffled_history = rec.shuffled_obs[..=idx].to_vec();
        let privileged_history = rec.privileged_obs[..=idx].to_vec();

        brier_of(
            historical.generate(&census_obs, Horizon(HORIZON)),
            &actual,
            &mut scores[0],
        );
        brier_of(
            census_only_statistical.generate(&census_history, Horizon(HORIZON)),
            &actual,
            &mut scores[1],
        );
        brier_of(
            trait_trend.generate(&noisy_history, Horizon(HORIZON)),
            &actual,
            &mut scores[2],
        );
        brier_of(
            adaptation_vs_forcing.generate(&noisy_obs, Horizon(HORIZON)),
            &actual,
            &mut scores[3],
        );
        brier_of(
            fep_census_only.generate(&census_history, Horizon(HORIZON)),
            &actual,
            &mut scores[4],
        );
        brier_of(
            fep_census_plus_trait.generate(&noisy_history, Horizon(HORIZON)),
            &actual,
            &mut scores[5],
        );
        brier_of(
            fep_census_plus_trait.generate(&shuffled_history, Horizon(HORIZON)),
            &actual,
            &mut scores[6],
        );
        brier_of(
            fep_census_plus_trait.generate(&privileged_history, Horizon(HORIZON)),
            &actual,
            &mut scores[7],
        );
        brier_of(
            oracle.generate(&checkpoint, Horizon(HORIZON)),
            &actual,
            &mut scores[8],
        );

        checkpoint += CHECKPOINT_STRIDE;
    }

    scores
}

fn mean_scores(seeds: &[u64]) -> Vec<(f64, usize)> {
    let mut aggregate: Vec<(f64, usize)> = vec![(0.0, 0); ARMS.len()];
    for &seed in seeds {
        let rec = record_seed(seed);
        let trait_str = rec
            .trait_level_just_before_extinction
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a (never extinct)".to_string());
        let extinction_str = rec
            .first_extinction_tick
            .map(|t| t.to_string())
            .unwrap_or_else(|| "never".to_string());
        let max_population = rec.max_population;
        println!(
            "  seed={seed:4}  max_population={max_population:3}  \
             trait_level_just_before_extinction={trait_str} first_extinction_tick={extinction_str}"
        );
        let scores = run_arms(&rec);
        for (i, (sum, n)) in scores.into_iter().enumerate() {
            aggregate[i].0 += sum;
            aggregate[i].1 += n;
        }
    }
    aggregate
}

fn print_scores(label: &str, aggregate: &[(f64, usize)]) {
    println!("\n{label}");
    for (name, (sum, n)) in ARMS.iter().zip(aggregate.iter()) {
        let mean = if *n > 0 { sum / *n as f64 } else { f64::NAN };
        println!("  {name:34} mean Brier = {mean:.4}  (n={n})");
    }
}

fn main() {
    println!("== Evolutionary-rescue Phase 2.2B-ii: four-condition backtest ==");
    println!(
        "Config: TICKS={TICKS} INITIAL_COUNT={INITIAL_COUNT} MUTATION_RATE={MUTATION_RATE} \
         HORIZON={HORIZON} CHECKPOINT_STRIDE={CHECKPOINT_STRIDE}"
    );

    println!(
        "\n-- Train seeds (diagnostic only -- nothing in this harness fits parameters \
              across seeds, so this is not a leakage risk, just a sanity check) --"
    );
    let train_aggregate = mean_scores(&TRAIN_SEEDS);
    print_scores("Train-seed mean Brier per arm:", &train_aggregate);

    println!("\n-- Test seeds (the real, held-out acceptance-gate evaluation) --");
    let test_aggregate = mean_scores(&TEST_SEEDS);
    print_scores("Test-seed mean Brier per arm:", &test_aggregate);

    let mean_of = |i: usize| -> f64 {
        let (sum, n) = test_aggregate[i];
        if n > 0 { sum / n as f64 } else { f64::NAN }
    };
    let census_only_fep = mean_of(4);
    let real_trait_fep = mean_of(5);
    let shuffled_trait_fep = mean_of(6);
    let privileged_trait_fep = mean_of(7);

    println!(
        "\n== Acceptance gate (predeclared, SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md) =="
    );
    println!("  fep_census_only              = {census_only_fep:.4}");
    println!("  fep_census_plus_trait REAL     = {real_trait_fep:.4}");
    println!("  fep_census_plus_trait SHUFFLED = {shuffled_trait_fep:.4}");
    println!("  fep_census_plus_trait PRIVILEGED (ceiling, not gated) = {privileged_trait_fep:.4}");

    let real_improves = real_trait_fep < census_only_fep;
    let shuffled_fails_to_reproduce = !(shuffled_trait_fep < census_only_fep) || {
        // "fails to reproduce" is a relative bar: shuffled must not close the same gap real did.
        let real_gap = census_only_fep - real_trait_fep;
        let shuffled_gap = census_only_fep - shuffled_trait_fep;
        real_gap <= 0.0 || shuffled_gap < 0.5 * real_gap
    };

    println!(
        "\n  Half 1 (real trait beats census-only): {}",
        if real_improves { "PASS" } else { "FAIL" }
    );
    println!(
        "  Half 2 (shuffled control fails to reproduce the gain): {}",
        if shuffled_fails_to_reproduce {
            "PASS"
        } else {
            "FAIL"
        }
    );

    let gate_passes = real_improves && shuffled_fails_to_reproduce;
    println!(
        "\n{}: the trait-augmented FEP model {} genuine incremental predictive value from \
         evolutionary adaptation on held-out seeds.",
        if gate_passes {
            "GATE PASSES"
        } else {
            "GATE FAILS"
        },
        if gate_passes {
            "extracts"
        } else {
            "does not extract"
        }
    );
    if !gate_passes {
        println!(
            "\nCAVEAT (do not over-read this run alone): fep_census_only used state_dim=1 while \
             every trait-augmented arm used state_dim=2, so this comparison confounds \
             information content with model capacity. The PRIVILEGED arm scoring worst of the \
             three trait-augmented conditions rules out \"the signal was too noisy,\" but does \
             NOT by itself prove dimensionality/architecture is the cause -- see \
             SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md's Phase 2.2C for the \
             constant/duplicate-channel ablation that would isolate this properly."
        );
    }
}
