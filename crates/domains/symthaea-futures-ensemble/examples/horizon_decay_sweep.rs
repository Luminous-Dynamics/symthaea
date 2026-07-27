// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests the one bullet of the plan's "What 'done' looks like for Phase 1" checklist
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`) that no backtest example has ever actually
//! exercised: "attaches calibrated uncertainty that visibly decays with forecast horizon." Every
//! prior example uses one fixed `HORIZON` constant per run; this one sweeps `Horizon` across
//! several values and reports `expected_calibration_error()` per rung per horizon, so an honest
//! degradation (or lack of one) is directly visible.
//!
//! Predeclared in the plan doc before this was run: `PersistenceGenerator` (reports the same
//! prediction regardless of horizon) should show ECE visibly worsening as horizon grows on the
//! dimmed-sun collapse regime (true extinction probability rises monotonically with horizon, its
//! frozen forecast doesn't). `ScenarioMechanisticGenerator` (a real closed-form hazard-decay
//! term keyed to `horizon.0`) should track that growth and hold a flatter/lower ECE.
//! `FepDrivenGenerator`'s behavior was not predicted in advance.
//!
//! Run: `cargo run --example horizon_decay_sweep -p symthaea-futures-ensemble`

use symthaea_alife::{EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig};
use symthaea_futures_calibration::reliability_diagram;
use symthaea_futures_core::{
    ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator,
};
use symthaea_futures_ensemble::ecological::{
    FepDrivenGenerator, PersistenceGenerator, ScenarioMechanisticGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::ecological::{
    EcologicalGroundTruth, EcologicalObservation, ExtinctionObservationPolicy,
};

/// Extracts the forecast's assigned `P(true)` directly -- not via
/// `symthaea_futures_calibration::boolean_prediction_pair`, which bundles this together with a
/// ground-truth "actual" argument this helper has no use for (the actual outcome is already
/// tracked separately by `collect_pairs`).
fn p_true(dist: &ForecastDistribution) -> f64 {
    dist.branches
        .iter()
        .find(|b| b.outcome == OutcomeRegion::Boolean(true))
        .map(|b| b.probability)
        .unwrap_or(0.0)
}

const TICKS: u64 = 4000;
const CHECKPOINT_STRIDE: u64 = 50;
// Restricts evaluated checkpoints to 0..=MAX_CHECKPOINT (8 per seed at this stride) rather than
// sweeping the full 4000-tick trajectory. Real extinction ticks on this regime cluster around
// ~400-600 (per earlier traces this session, e.g. seed 11 = 402, seed 44 = 411) -- with the full
// range, checkpoint+horizon lands well past extinction for almost every checkpoint regardless of
// which horizon in [10, 400] is chosen, and every horizon "trivially" confirms an already-decided
// outcome. Capping at 350 means even the smallest horizon (10) mostly evaluates while the
// population is still alive, and the largest (400) spans past the typical extinction tick --
// keeping every horizon in the genuinely uncertain, decision-relevant boundary region.
const MAX_CHECKPOINT: u64 = 350;
const HORIZONS: [u64; 6] = [10, 25, 50, 100, 200, 400];
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const NUM_BUCKETS: usize = 5;

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

/// One seed's full observation/trajectory record -- generated once per seed, reused across every
/// horizon value tested (the ground truth doesn't change; only how far ahead we ask it to look
/// does), so the sweep isn't paying for N re-simulations of the same world.
struct SeedRecord {
    observations: Vec<EcologicalObservation>,
    trajectory: Vec<bool>,
}

fn run_seed(seed: u64) -> SeedRecord {
    let mut env = EarthForcedEnvironment::earth_like(200.0);
    env.model.solar_constant = 600.0; // dimmed past the snowball threshold -- guaranteed collapse
    let population = Population::new(population_config(), 6, seed);
    let mut truth = EcologicalGroundTruth::new(env, population, 3.0);
    let mut policy = ExtinctionObservationPolicy::new(6, 0.5, 1, 0.02, false, 99);

    let mut observations = vec![policy.observe(&truth, 0)];
    let mut trajectory = vec![truth.is_extinct()];

    for _ in 0..TICKS {
        truth.step();
        let tick = truth.tick();
        observations.push(policy.observe(&truth, tick));
        trajectory.push(truth.is_extinct());
    }

    SeedRecord {
        observations,
        trajectory,
    }
}

/// Collects `(predicted P(true), actual)` pairs for one rung at one horizon, across every seed
/// and checkpoint -- the raw material `reliability_diagram` needs.
fn collect_pairs(
    records: &[SeedRecord],
    horizon: u64,
    mut predict: impl FnMut(&SeedRecord, u64) -> Option<f64>,
) -> Vec<(f64, bool)> {
    let mut pairs = Vec::new();
    for record in records {
        let mut checkpoint = 0u64;
        while checkpoint <= MAX_CHECKPOINT && checkpoint + horizon < record.trajectory.len() as u64
        {
            let actual = record.trajectory[(checkpoint + horizon) as usize];
            if let Some(p) = predict(record, checkpoint) {
                pairs.push((p, actual));
            }
            checkpoint += CHECKPOINT_STRIDE;
        }
    }
    pairs
}

fn ece_from_pairs(pairs: &[(f64, bool)]) -> (f64, usize) {
    let diagram = reliability_diagram(pairs, NUM_BUCKETS);
    (diagram.expected_calibration_error(), pairs.len())
}

fn main() {
    println!(
        "Horizon-decay calibration sweep (dimmed-sun collapse, {} seeds, horizons {:?})\n",
        SEEDS.len(),
        HORIZONS
    );

    let records: Vec<SeedRecord> = SEEDS.iter().map(|&s| run_seed(s)).collect();

    let persistence = PersistenceGenerator;
    let mechanistic = ScenarioMechanisticGenerator {
        per_member_death_probability: 0.01, // illustrative, same value ecological_backtest.rs uses
    };
    let fep_driven = FepDrivenGenerator::default();

    println!(
        "{:>8}  {:>18}  {:>18}  {:>18}",
        "horizon", "persistence ECE", "mechanistic ECE", "fep_driven ECE"
    );

    for &horizon in &HORIZONS {
        let persistence_pairs = collect_pairs(&records, horizon, |record, checkpoint| {
            let obs = record.observations[checkpoint as usize];
            match persistence.generate(&obs, Horizon(horizon)) {
                ForecastOutput::Distribution(dist) => Some(p_true(&dist)),
                ForecastOutput::Abstain(_) => None,
            }
        });
        let mechanistic_pairs = collect_pairs(&records, horizon, |record, checkpoint| {
            let obs = record.observations[checkpoint as usize];
            match mechanistic.generate(&obs, Horizon(horizon)) {
                ForecastOutput::Distribution(dist) => Some(p_true(&dist)),
                ForecastOutput::Abstain(_) => None,
            }
        });
        let fep_pairs = collect_pairs(&records, horizon, |record, checkpoint| {
            let history_slice: Vec<EcologicalObservation> =
                record.observations[..=checkpoint as usize].to_vec();
            match fep_driven.generate(&history_slice, Horizon(horizon)) {
                ForecastOutput::Distribution(dist) => Some(p_true(&dist)),
                ForecastOutput::Abstain(_) => None,
            }
        });

        let (p_ece, p_n) = ece_from_pairs(&persistence_pairs);
        let (m_ece, m_n) = ece_from_pairs(&mechanistic_pairs);
        let (f_ece, f_n) = ece_from_pairs(&fep_pairs);

        println!(
            "{:>8}  {:>10.4} (n={:>3})  {:>10.4} (n={:>3})  {:>10.4} (n={:>3})",
            horizon, p_ece, p_n, m_ece, m_n, f_ece, f_n
        );
    }
}
