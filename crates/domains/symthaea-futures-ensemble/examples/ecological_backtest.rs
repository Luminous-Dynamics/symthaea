// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The first real end-to-end run of the Symthaea Futures Laboratory's Phase 1 apparatus
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`) — not a unit test of one piece in
//! isolation. Runs the ecological-collapse scenario forward under two regimes (habitable,
//! dimmed-sun collapse — the same fixture `symthaea-alife`'s own `phase5_earth_forcing.rs` test
//! uses), generates forecasts from every implemented baseline rung at a series of checkpoints,
//! scores each against the actual future outcome via `symthaea-futures-calibration::BrierScore`,
//! and reports mean Brier score per rung per scenario.
//!
//! Run: `cargo run --example ecological_backtest -p symthaea-futures-ensemble`
//!
//! ## What this run is and isn't
//!
//! This is a real, honest empirical result — not a hand-picked demo. It is **not** the plan's
//! formal predeclared acceptance gate (no threshold was set in advance of this run). What it
//! *does* establish: whether all six implemented rungs actually rank the way the baseline
//! hierarchy is supposed to (oracle best, naive extrapolation-based rungs worst where the
//! cohort/population mismatch bites) on a scenario this apparatus has never been run against
//! end to end before — including rung 5 (`FepDrivenGenerator`), now included.
//!
//! `HistoricalFrequencyGenerator`'s `base_rate` and `ScenarioMechanisticGenerator`'s
//! `per_member_death_probability` are **illustrative placeholder constants** here, not values
//! calibrated across training seeds the way the plan specifies those parameters should be —
//! disclosed, not hidden. Abstentions are silently excluded from each rung's mean Brier score
//! rather than tracked separately; a real evaluation harness should report abstention rate
//! alongside score, which this quick example doesn't do.
//!
//! ## Evidence ledger — first real use
//!
//! Every scored forecast also builds a real `symthaea_futures_ledger::EvidenceRecord` — the
//! first time that type has been used anywhere (previously a schema with no consumer). Several
//! fields are honestly filled with disclosed placeholders rather than fabricated precision:
//! `belief_state_snapshot_hash` (no generator here exposes its internal belief through a public
//! API), `calibration_bucket` (this example doesn't call `reliability_diagram` — that's a
//! separate, already-tested mechanism, not duplicated here), and `wall_clock_cost_ms` (not
//! measured). Real, meaningful fields: `scenario_family`, `world_seed`,
//! `observation_cutoff_tick`, `trajectory_generator_ids`, `predicted_distribution`,
//! `actual_continuation`, and `score`.

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::ecological::{
    FepDrivenGenerator, HistoricalFrequencyGenerator, OracleGenerator, PersistenceGenerator,
    ScenarioMechanisticGenerator, SimpleStatisticalGenerator,
};
use symthaea_futures_ledger::EvidenceRecord;
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, ExtinctionObservationPolicy,
};

const HORIZON: u64 = 100;
const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 200;

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

struct ScenarioResult {
    name: &'static str,
    scores: Vec<(&'static str, f64, usize)>,
    records: Vec<EvidenceRecord>,
}

#[allow(clippy::too_many_arguments)]
fn score_and_record(
    acc: &mut (&str, f64, usize),
    rung_name: &str,
    output: ForecastOutput,
    actual: &OutcomeRegion,
    scenario_name: &str,
    seed: u64,
    checkpoint: u64,
    records: &mut Vec<EvidenceRecord>,
) {
    let ForecastOutput::Distribution(dist) = output else {
        return; // abstentions are excluded, per this example's disclosed limitation
    };
    let score = BrierScore.score(&dist, actual);
    acc.1 += score;
    acc.2 += 1;

    records.push(EvidenceRecord {
        scenario_family: scenario_name.to_string(),
        world_seed: seed,
        observation_policy_version:
            "ExtinctionObservationPolicy(sample_fraction=0.5, frequency=1, noise=0.02)".to_string(),
        observation_cutoff_tick: checkpoint,
        belief_state_snapshot_hash:
            "n/a -- no generator in this example exposes its internal belief through a public API"
                .to_string(),
        model_versions: vec![rung_name.to_string()],
        trajectory_generator_ids: vec![rung_name.to_string()],
        branch_clustering_method: "none (2-branch boolean outcome space)".to_string(),
        predicted_distribution: dist,
        scoring_rule: "Brier".to_string(),
        actual_continuation: actual.clone(),
        score,
        calibration_bucket: "not computed in this example -- see symthaea-futures-calibration::reliability_diagram for the real mechanism".to_string(),
        wall_clock_cost_ms: 0, // not measured in this example
        notes: "produced by examples/ecological_backtest.rs".to_string(),
    });
}

fn run_scenario(name: &'static str, dim_sun: bool, seed: u64) -> ScenarioResult {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    if dim_sun {
        env.model.solar_constant = 600.0; // matches ice_albedo.rs's own snowball fixture
    }
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

    let oracle = OracleGenerator::from_trajectory(trajectory.clone());
    let persistence = PersistenceGenerator;
    let historical = HistoricalFrequencyGenerator { base_rate: 0.5 }; // illustrative, not calibrated
    let mechanistic = ScenarioMechanisticGenerator {
        per_member_death_probability: 0.01, // illustrative, not calibrated
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
    let mut records: Vec<EvidenceRecord> = Vec::new();

    let mut checkpoint = 0u64;
    while checkpoint + HORIZON < trajectory.len() as u64 {
        let obs = observations[checkpoint as usize];
        let actual = OutcomeRegion::Boolean(trajectory[(checkpoint + HORIZON) as usize]);
        let history_slice: Vec<EcologicalObservation> =
            observations[..=checkpoint as usize].to_vec();

        score_and_record(
            &mut scores[0],
            "persistence",
            persistence.generate(&obs, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );
        score_and_record(
            &mut scores[1],
            "historical_frequency",
            historical.generate(&obs, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );
        score_and_record(
            &mut scores[2],
            "simple_statistical",
            statistical.generate(&history_slice, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );
        score_and_record(
            &mut scores[3],
            "scenario_mechanistic",
            mechanistic.generate(&obs, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );
        score_and_record(
            &mut scores[4],
            "fep_driven",
            fep_driven.generate(&history_slice, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );
        score_and_record(
            &mut scores[5],
            "oracle",
            oracle.generate(&checkpoint, Horizon(HORIZON)),
            &actual,
            name,
            seed,
            checkpoint,
            &mut records,
        );

        checkpoint += CHECKPOINT_STRIDE;
    }

    ScenarioResult {
        name,
        scores,
        records,
    }
}

fn main() {
    let scenarios = [
        run_scenario("habitable", false, 11),
        run_scenario("dimmed_collapse", true, 11),
    ];

    for scenario in &scenarios {
        println!(
            "== {} (horizon={HORIZON}, {} checkpoints) ==",
            scenario.name,
            scenario.scores.first().map(|(_, _, n)| *n).unwrap_or(0)
        );
        for (rung, sum, n) in &scenario.scores {
            let mean = if *n > 0 { sum / *n as f64 } else { f64::NAN };
            println!("  {rung:22} mean Brier = {mean:.4}  (n={n})");
        }
        println!(
            "  {} EvidenceRecords built this scenario. Sample (first record):",
            scenario.records.len()
        );
        if let Some(first) = scenario.records.first() {
            println!(
                "{}",
                serde_json::to_string_pretty(first).expect("EvidenceRecord must serialize")
            );
        }
        println!();
    }
}
