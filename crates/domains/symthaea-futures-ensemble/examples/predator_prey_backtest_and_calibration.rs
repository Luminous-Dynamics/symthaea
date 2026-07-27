// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! First real backtest of the predator/prey scenario family, plus the calibration-generalization
//! test this plan's own honest caveat called for: `calibration_correction_verification.rs`
//! found `HistogramCalibrator` reduces ECE to exactly `0.0000` on held-out ecological-regime
//! seeds, but flagged that result as too clean to trust in general — a consequence of that
//! scenario's strongly bimodal, largely deterministic-given-the-regime dynamics. This uses
//! `symthaea-alife`'s real, ground-truth-tested predator/prey oscillation instead
//! (`tests/phase1_predator_prey.rs`: predator CV=0.198, real variance, not a monotone collapse)
//! to check whether the technique holds up on a genuinely noisier target.
//!
//! Run: `cargo run --example predator_prey_backtest_and_calibration -p symthaea-futures-ensemble`
//!
//! Train/test seeds reuse the same values `calibration_correction_verification.rs` used
//! (11/22/33/44/55 train, 66/77/88/99/111 test) — reusing seed *values* across unrelated scenario
//! families doesn't violate the train/test split (each run is independently seeded and
//! self-contained); what matters is that no single fit/evaluate pair shares seeds, which holds
//! here exactly as it did there.

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};
use symthaea_futures_calibration::{
    BrierScore, HistogramCalibrator, ScoringRule, boolean_prediction_pair, reliability_diagram,
};
use symthaea_futures_core::{
    ForecastBranch, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, OutcomeSpaceId,
    TrajectoryGenerator,
};
use symthaea_futures_ensemble::predator_prey::{
    FepDrivenGenerator, HistoricalFrequencyGenerator, OracleGenerator, PersistenceGenerator,
    ScenarioMechanisticGenerator, SimpleStatisticalGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::predator_prey::{
    PredatorExtinctionObservationPolicy, PredatorPreyObservation,
};

const HORIZON: u64 = 200;
const TICKS: u64 = 8000;
const CHECKPOINT_STRIDE: u64 = 400;
const INITIAL_PREY: usize = 10;
const INITIAL_PREDATORS: usize = 3;
const TRAIN_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const TEST_SEEDS: [u64; 5] = [66, 77, 88, 99, 111];

/// Same fixture `symthaea-alife`'s own `tests/phase1_predator_prey.rs` uses -- real,
/// ground-truth-verified oscillation with genuine variance, not invented for this example.
fn scenario_config() -> PredatorPreyConfig {
    let sustainable_organism_cfg = OrganismConfig {
        forage_efficiency: 0.6,
        ..OrganismConfig::default()
    };
    PredatorPreyConfig {
        prey_cfg: PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: sustainable_organism_cfg,
            ..Default::default()
        },
        predator_cfg: PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: sustainable_organism_cfg,
            ..Default::default()
        },
        plant_resource_total: 3.0,
        predation_scale: 0.05,
        predation_efficiency: 0.05,
    }
}

struct SeedRun {
    scores: Vec<(&'static str, f64, usize)>,
    fep_pairs: Vec<(f64, bool)>,
}

fn run_seed(seed: u64) -> SeedRun {
    let mut sim = PredatorPreySim::new(scenario_config(), INITIAL_PREY, INITIAL_PREDATORS, seed);
    let mut policy = PredatorExtinctionObservationPolicy::new(INITIAL_PREDATORS, 1.0, 1);

    let mut observations: Vec<PredatorPreyObservation> = vec![policy.observe(&sim, 0)];
    let mut trajectory: Vec<bool> = vec![sim.predator.is_empty()];

    for _ in 0..TICKS {
        sim.step();
        let tick = sim.t;
        observations.push(policy.observe(&sim, tick));
        trajectory.push(sim.predator.is_empty());
    }

    let oracle = OracleGenerator::from_trajectory(trajectory.clone());
    let persistence = PersistenceGenerator;
    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 };
    let mechanistic = ScenarioMechanisticGenerator {
        per_member_death_probability: 0.01,
    };
    let statistical = SimpleStatisticalGenerator;
    let fep_driven = FepDrivenGenerator::default();

    let mut scores: Vec<(&str, f64, usize)> = vec![
        ("persistence", 0.0, 0),
        ("historical_frequency", 0.0, 0),
        ("simple_statistical", 0.0, 0),
        ("scenario_mechanistic", 0.0, 0),
        ("fep_driven", 0.0, 0),
        ("oracle", 0.0, 0),
    ];
    let mut fep_pairs = Vec::new();

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let obs = observations[checkpoint as usize];
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<PredatorPreyObservation> =
            observations[..=checkpoint as usize].to_vec();

        score(
            &mut scores[0],
            persistence.generate(&obs, Horizon(HORIZON)),
            &actual,
        );
        score(
            &mut scores[1],
            historical.generate(&obs, Horizon(HORIZON)),
            &actual,
        );
        score(
            &mut scores[2],
            statistical.generate(&history_slice, Horizon(HORIZON)),
            &actual,
        );
        score(
            &mut scores[3],
            mechanistic.generate(&obs, Horizon(HORIZON)),
            &actual,
        );
        let fep_output = fep_driven.generate(&history_slice, Horizon(HORIZON));
        if let ForecastOutput::Distribution(dist) = &fep_output
            && let Some(pair) = boolean_prediction_pair(dist, &actual)
        {
            fep_pairs.push(pair);
        }
        score(&mut scores[4], fep_output, &actual);
        score(
            &mut scores[5],
            oracle.generate(&checkpoint, Horizon(HORIZON)),
            &actual,
        );

        checkpoint += CHECKPOINT_STRIDE;
    }

    SeedRun { scores, fep_pairs }
}

fn score(acc: &mut (&str, f64, usize), output: ForecastOutput, actual: &OutcomeRegion) {
    if let ForecastOutput::Distribution(dist) = output {
        acc.1 += BrierScore.score(&dist, actual);
        acc.2 += 1;
    }
}

fn boolean_forecast(p_true: f64) -> ForecastDistribution {
    ForecastDistribution {
        issued_at_tick: 0,
        horizon: Horizon(HORIZON),
        outcome_space: OutcomeSpaceId("predator_prey_calibration_check".to_string()),
        branches: vec![
            ForecastBranch {
                probability: p_true,
                outcome: OutcomeRegion::Boolean(true),
                assumptions: Vec::new(),
            },
            ForecastBranch {
                probability: 1.0 - p_true,
                outcome: OutcomeRegion::Boolean(false),
                assumptions: Vec::new(),
            },
        ],
        unsupported_mass: 0.0,
    }
}

fn mean_brier(pairs: &[(f64, bool)]) -> f64 {
    let sum: f64 = pairs
        .iter()
        .map(|&(p, actual)| BrierScore.score(&boolean_forecast(p), &OutcomeRegion::Boolean(actual)))
        .sum();
    sum / pairs.len() as f64
}

fn main() {
    println!(
        "== Predator/prey backtest: all six rungs, {} train seeds ==\n",
        TRAIN_SEEDS.len()
    );

    let mut aggregate: Vec<(&str, f64, usize)> = vec![
        ("persistence", 0.0, 0),
        ("historical_frequency", 0.0, 0),
        ("simple_statistical", 0.0, 0),
        ("scenario_mechanistic", 0.0, 0),
        ("fep_driven", 0.0, 0),
        ("oracle", 0.0, 0),
    ];
    let mut train_fep_pairs = Vec::new();

    for &seed in &TRAIN_SEEDS {
        let run = run_seed(seed);
        for (i, (_, sum, n)) in run.scores.iter().enumerate() {
            aggregate[i].1 += sum;
            aggregate[i].2 += n;
        }
        train_fep_pairs.extend(run.fep_pairs);
    }

    for (rung, sum, n) in &aggregate {
        let mean = if *n > 0 { sum / *n as f64 } else { f64::NAN };
        println!("  {rung:22} mean Brier = {mean:.4}  (n={n})");
    }

    let train_diagram = reliability_diagram(&train_fep_pairs, 10);
    println!(
        "\nfep_driven reliability on train seeds (base rate {:.4}, {} predictions):",
        train_fep_pairs.iter().filter(|&&(_, a)| a).count() as f64 / train_fep_pairs.len() as f64,
        train_fep_pairs.len()
    );
    for bucket in &train_diagram.buckets {
        println!(
            "  [{:.1}, {:.1})  predicted={:.4}  empirical={:.4}  n={}",
            bucket.bucket_low,
            bucket.bucket_high,
            bucket.mean_predicted_probability,
            bucket.empirical_frequency,
            bucket.count
        );
    }
    println!(
        "ECE (train) = {:.4}",
        train_diagram.expected_calibration_error()
    );

    // Now the generalization check: fit on TRAIN_SEEDS' fep_driven predictions, evaluate on
    // genuinely held-out TEST_SEEDS.
    let calibrator = HistogramCalibrator::fit(&train_fep_pairs, 10);

    let mut test_fep_pairs = Vec::new();
    for &seed in &TEST_SEEDS {
        test_fep_pairs.extend(run_seed(seed).fep_pairs);
    }

    let raw_mean = mean_brier(&test_fep_pairs);
    let raw_ece = reliability_diagram(&test_fep_pairs, 10).expected_calibration_error();
    let calibrated_pairs: Vec<(f64, bool)> = test_fep_pairs
        .iter()
        .map(|&(p, a)| (calibrator.calibrate(p), a))
        .collect();
    let calibrated_mean = mean_brier(&calibrated_pairs);
    let calibrated_ece = reliability_diagram(&calibrated_pairs, 10).expected_calibration_error();

    println!("\n== Calibration generalization check (held-out test seeds) ==");
    println!("Train seeds: {TRAIN_SEEDS:?}   Test seeds: {TEST_SEEDS:?}");
    println!("  Raw        mean Brier = {raw_mean:.4}   ECE = {raw_ece:.4}");
    println!("  Calibrated mean Brier = {calibrated_mean:.4}   ECE = {calibrated_ece:.4}");
    let helped = calibrated_ece < raw_ece;
    println!(
        "\n{}: calibration correction {} ECE on held-out predator/prey seeds",
        if helped { "HELPED" } else { "DID NOT HELP" },
        if helped { "reduced" } else { "did not reduce" }
    );
}
