// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measurement-only repeated-shock evolvability harness.
//!
//! This example compares an evolving population (`mutation_rate = 0.1`) with an otherwise
//! identical frozen control (`mutation_rate = 0.0`) under two identical resource shocks.
//! Each shock receives the same fixed post-shock evaluation horizon, preventing the earlier shock
//! from receiving more recovery time merely because it occurred earlier in the run.
//! It deliberately makes no superiority assertion: the output is evidence for later analysis,
//! not a preregistered claim that evolution must improve every recovery dimension.

use symthaea_alife::{
    EncounterScheduler, InheritanceMode, OrganismConfig, PairingMode, PerturbationSchedule,
    Population, PopulationConfig, ResourcePerturbation, analyze_genesis_events,
    analyze_recovery_through, compare_recovery,
};

const TICKS: u64 = 1_400;
const INITIAL_COUNT: usize = 24;
const PLANT_RESOURCE_TOTAL: f64 = 12.0;
const BASELINE_LOOKBACK_TICKS: u64 = 100;
const RECOVERY_FRACTION: f64 = 0.90;
const POST_SHOCK_EVALUATION_TICKS: u64 = 300;

const SHOCK_1_START: u64 = 400;
const SHOCK_2_START: u64 = 900;
const SHOCK_DURATION: u64 = 60;
const SHOCK_MULTIPLIER: f64 = 0.65;

fn population_config(mutation_rate: f64) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(),
        mutation_rate,
        mutation_std: 0.05,
        inheritance: InheritanceMode::FromParent,
    }
}

fn evaluation_end_tick(shock: ResourcePerturbation) -> u64 {
    shock
        .end_tick_exclusive()
        .checked_add(POST_SHOCK_EVALUATION_TICKS - 1)
        .expect("fixed evaluation horizon must fit in u64")
}

fn main() {
    let shock_1 = ResourcePerturbation::new(
        SHOCK_1_START,
        SHOCK_DURATION,
        SHOCK_MULTIPLIER,
        0.0,
    )
    .expect("valid first shock");
    let shock_2 = ResourcePerturbation::new(
        SHOCK_2_START,
        SHOCK_DURATION,
        SHOCK_MULTIPLIER,
        0.0,
    )
    .expect("valid second shock");
    let schedule = PerturbationSchedule::new(vec![shock_1, shock_2]);

    // Matching seeds deliberately hold initialization and encounter-scheduler randomness fixed
    // between conditions. Mutation rate is the intended treatment difference.
    let population_seed = 17;
    let scheduler_seed = 91;
    let mut frozen = Population::new(population_config(0.0), INITIAL_COUNT, population_seed);
    let mut evolving = Population::new(population_config(0.1), INITIAL_COUNT, population_seed);
    let mut frozen_scheduler = EncounterScheduler::new(PairingMode::Random, scheduler_seed);
    let mut evolving_scheduler = EncounterScheduler::new(PairingMode::Random, scheduler_seed);

    let mut frozen_events = Vec::new();
    let mut evolving_events = Vec::new();

    for tick in 0..TICKS {
        let frozen_baseline = PLANT_RESOURCE_TOTAL / frozen.len().max(1) as f64;
        let frozen_resource = schedule
            .apply_resource(tick, frozen_baseline)
            .expect("finite frozen resource");
        frozen.step_social(|_| frozen_resource, &mut frozen_scheduler);
        frozen_events.extend(frozen.drain_event_log());

        let evolving_baseline = PLANT_RESOURCE_TOTAL / evolving.len().max(1) as f64;
        let evolving_resource = schedule
            .apply_resource(tick, evolving_baseline)
            .expect("finite evolving resource");
        evolving.step_social(|_| evolving_resource, &mut evolving_scheduler);
        evolving_events.extend(evolving.drain_event_log());
    }

    let frozen_report = analyze_genesis_events(&frozen_events).expect("valid frozen event stream");
    let evolving_report =
        analyze_genesis_events(&evolving_events).expect("valid evolving event stream");

    for (label, shock) in [("shock-1", shock_1), ("shock-2", shock_2)] {
        let evaluation_end_tick = evaluation_end_tick(shock);
        let frozen_recovery = analyze_recovery_through(
            &frozen_report,
            shock,
            BASELINE_LOOKBACK_TICKS,
            RECOVERY_FRACTION,
            evaluation_end_tick,
        );
        let evolving_recovery = analyze_recovery_through(
            &evolving_report,
            shock,
            BASELINE_LOOKBACK_TICKS,
            RECOVERY_FRACTION,
            evaluation_end_tick,
        );

        println!("{label} (fixed post-shock horizon: {POST_SHOCK_EVALUATION_TICKS} ticks)");
        match (evolving_recovery, frozen_recovery) {
            (Ok(evolving_recovery), Ok(frozen_recovery)) => {
                let comparison = compare_recovery(&evolving_recovery, &frozen_recovery)
                    .expect("candidate/control experiment boundaries must match");
                println!("  frozen recovery:   {frozen_recovery:#?}");
                println!("  evolving recovery: {evolving_recovery:#?}");
                println!("  comparison:        {comparison:#?}");
            }
            (evolving, frozen) => {
                println!("  recovery comparison unavailable under current evidence boundary");
                println!("  frozen analysis:   {frozen:#?}");
                println!("  evolving analysis: {evolving:#?}");
            }
        }
    }
}
