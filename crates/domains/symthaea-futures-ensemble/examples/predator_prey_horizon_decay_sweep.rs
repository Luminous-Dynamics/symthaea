// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2 addendum item 4 (predator_prey half): `horizon_decay_sweep.rs` only ever tested
//! `ecological`. This builds the first horizon-decay sweep for `predator_prey`, reusing the
//! clean, decisive, non-degenerate `resource=0.5` collapse regime
//! `predator_prey_calibration_generalization.rs` established (5/5 seeds extinct, real timing
//! variance 39-270 ticks, prey surviving in every seed).
//!
//! Horizons are scaled to this regime's actual timescale (collapse happens within ~40-270
//! ticks, not the ecological sweep's 0-4000), and checkpoints are capped to a range that keeps
//! every horizon in the genuinely uncertain boundary region -- the same `MAX_CHECKPOINT`
//! discipline `horizon_decay_sweep.rs` needed for `ecological`.
//!
//! Predeclared before running: direction genuinely unknown in advance.
//!
//! Run: `cargo run --example predator_prey_horizon_decay_sweep -p symthaea-futures-ensemble`

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};
use symthaea_futures_calibration::reliability_diagram;
use symthaea_futures_core::{ForecastOutput, Horizon, OutcomeRegion, TrajectoryGenerator};
use symthaea_futures_ensemble::predator_prey::{
    FepDrivenGenerator, PersistenceGenerator, ScenarioMechanisticGenerator,
};
use symthaea_futures_symtropy::ObservationPolicy;
use symthaea_futures_symtropy::predator_prey::{
    PredatorPopulationCensusObservationPolicy, PredatorPreyObservation,
};

const TICKS: u64 = 500;
const CHECKPOINT_STRIDE: u64 = 10;
const MAX_CHECKPOINT: u64 = 200;
const HORIZONS: [u64; 6] = [5, 10, 20, 40, 60, 100];
const SAMPLE_SIZE: usize = 3;
const NUM_BUCKETS: usize = 5;
const INITIAL_PREY: usize = 10;
const INITIAL_PREDATORS: usize = 3;
const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];

fn config() -> PredatorPreyConfig {
    let organism_cfg = OrganismConfig {
        forage_efficiency: 0.6,
        ..OrganismConfig::default()
    };
    let pop_cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg,
        ..Default::default()
    };
    PredatorPreyConfig {
        prey_cfg: pop_cfg,
        predator_cfg: pop_cfg,
        plant_resource_total: 0.5,
        predation_scale: 0.05,
        predation_efficiency: 0.05,
    }
}

fn run_seed_observations(seed: u64) -> (Vec<PredatorPreyObservation>, Vec<bool>) {
    let mut sim = PredatorPreySim::new(config(), INITIAL_PREY, INITIAL_PREDATORS, seed);
    let mut policy = PredatorPopulationCensusObservationPolicy::new(SAMPLE_SIZE, 1);

    let mut observations = vec![policy.observe(&sim, 0)];
    let mut trajectory = vec![sim.predator.len() == 0];
    for _ in 0..TICKS {
        sim.step();
        let tick = sim.t;
        observations.push(policy.observe(&sim, tick));
        trajectory.push(sim.predator.len() == 0);
    }
    (observations, trajectory)
}

fn p_true(dist: &symthaea_futures_core::ForecastDistribution) -> f64 {
    dist.branches()
        .iter()
        .find(|b| b.outcome == OutcomeRegion::Boolean(true))
        .map(|b| b.probability.get())
        .unwrap_or(0.0)
}

fn main() {
    println!(
        "Predator/prey horizon-decay sweep (resource=0.5, {} seeds)\n",
        SEEDS.len()
    );

    let persistence = PersistenceGenerator;
    let mechanistic = ScenarioMechanisticGenerator {
        per_member_death_probability: 0.01,
    };
    let fep_driven = FepDrivenGenerator::default();

    let all_observations: Vec<_> = SEEDS.iter().map(|&s| run_seed_observations(s)).collect();

    println!(
        "{:>8}  {:>18}  {:>18}  {:>18}",
        "horizon", "persistence ECE", "mechanistic ECE", "fep_driven ECE"
    );

    for &horizon in &HORIZONS {
        let mut persistence_pairs = Vec::new();
        let mut mechanistic_pairs = Vec::new();
        let mut fep_pairs = Vec::new();

        for (observations, trajectory) in &all_observations {
            let mut checkpoint = 0u64;
            while checkpoint <= MAX_CHECKPOINT && checkpoint + horizon < trajectory.len() as u64 {
                let actual = trajectory[(checkpoint + horizon) as usize];
                let obs = observations[checkpoint as usize];
                let history_slice: Vec<PredatorPreyObservation> =
                    observations[..=checkpoint as usize].to_vec();

                if let ForecastOutput::Distribution(dist) =
                    persistence.generate(&obs, Horizon(horizon))
                {
                    persistence_pairs.push((p_true(&dist), actual));
                }
                if let ForecastOutput::Distribution(dist) =
                    mechanistic.generate(&obs, Horizon(horizon))
                {
                    mechanistic_pairs.push((p_true(&dist), actual));
                }
                if let ForecastOutput::Distribution(dist) =
                    fep_driven.generate(&history_slice, Horizon(horizon))
                {
                    fep_pairs.push((p_true(&dist), actual));
                }

                checkpoint += CHECKPOINT_STRIDE;
            }
        }

        let p_ece =
            reliability_diagram(&persistence_pairs, NUM_BUCKETS).expected_calibration_error();
        let m_ece =
            reliability_diagram(&mechanistic_pairs, NUM_BUCKETS).expected_calibration_error();
        let f_ece = reliability_diagram(&fep_pairs, NUM_BUCKETS).expected_calibration_error();

        println!(
            "{:>8}  {:>10.4} (n={:>3})  {:>10.4} (n={:>3})  {:>10.4} (n={:>3})",
            horizon,
            p_ece,
            persistence_pairs.len(),
            m_ece,
            mechanistic_pairs.len(),
            f_ece,
            fep_pairs.len()
        );
    }
}
