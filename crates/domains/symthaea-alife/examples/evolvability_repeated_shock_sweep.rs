// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired multi-seed repeated-shock evolvability sweep.
//!
//! This example applies the already-defined repeated-shock transfer criterion unchanged across a
//! fixed eight-seed panel. Frozen and evolving conditions share the same population and encounter
//! seeds within each pair. The output is descriptive evidence: verdict counts and paired outcomes,
//! not a statistical-significance or mechanism-causality claim.

use symthaea_alife::{
    EncounterScheduler, EvolvabilityError, GenesisEvent, InheritanceMode, ObservatoryReport,
    OrganismConfig, PairingMode, PerturbationSchedule, Population, PopulationConfig,
    RecoveryMetrics, RepeatShockErrorV1, RepeatShockTransferV1, RepeatShockTransferVerdictV1,
    ResourcePerturbation, analyze_genesis_events, analyze_recovery_through,
    compare_repeated_shocks,
};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
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

#[derive(Debug, Default)]
struct VerdictCounts {
    pareto_improved: usize,
    mixed: usize,
    no_directional_change: usize,
    pareto_worse: usize,
    unavailable: usize,
}

impl VerdictCounts {
    fn record(&mut self, verdict: RepeatShockTransferVerdictV1) {
        match verdict {
            RepeatShockTransferVerdictV1::ParetoImproved => self.pareto_improved += 1,
            RepeatShockTransferVerdictV1::Mixed => self.mixed += 1,
            RepeatShockTransferVerdictV1::NoDirectionalChange => {
                self.no_directional_change += 1
            }
            RepeatShockTransferVerdictV1::ParetoWorse => self.pareto_worse += 1,
        }
    }
}

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

fn run_condition(
    mutation_rate: f64,
    population_seed: u64,
    scheduler_seed: u64,
    schedule: &PerturbationSchedule,
) -> Vec<GenesisEvent> {
    let mut population = Population::new(
        population_config(mutation_rate),
        INITIAL_COUNT,
        population_seed,
    );
    let mut scheduler = EncounterScheduler::new(PairingMode::Random, scheduler_seed);
    let mut events = Vec::new();

    for tick in 0..TICKS {
        let baseline = PLANT_RESOURCE_TOTAL / population.len().max(1) as f64;
        let resource = schedule
            .apply_resource(tick, baseline)
            .expect("finite resource");
        population.step_social(|_| resource, &mut scheduler);
        events.extend(population.drain_event_log());
    }
    events
}

fn recovery(
    report: &ObservatoryReport,
    shock: ResourcePerturbation,
) -> Result<RecoveryMetrics, EvolvabilityError> {
    analyze_recovery_through(
        report,
        shock,
        BASELINE_LOOKBACK_TICKS,
        RECOVERY_FRACTION,
        evaluation_end_tick(shock),
    )
}

fn transfer(
    report: &ObservatoryReport,
    shock_1: ResourcePerturbation,
    shock_2: ResourcePerturbation,
) -> Result<RepeatShockTransferV1, String> {
    let first = recovery(report, shock_1)
        .map_err(|error| format!("shock-1 recovery unavailable: {error:?}"))?;
    let second = recovery(report, shock_2)
        .map_err(|error| format!("shock-2 recovery unavailable: {error:?}"))?;
    compare_repeated_shocks(shock_1, &first, shock_2, &second)
        .map_err(|error: RepeatShockErrorV1| format!("transfer comparison unavailable: {error:?}"))
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

    let mut frozen_counts = VerdictCounts::default();
    let mut evolving_counts = VerdictCounts::default();
    let mut paired_valid = 0usize;
    let mut evolving_only_pareto_improved = 0usize;
    let mut frozen_only_pareto_improved = 0usize;
    let mut both_pareto_improved = 0usize;
    let mut neither_pareto_improved = 0usize;

    for &seed in SEEDS {
        let scheduler_seed = seed.wrapping_add(100_003);
        let frozen_events = run_condition(0.0, seed, scheduler_seed, &schedule);
        let evolving_events = run_condition(0.1, seed, scheduler_seed, &schedule);
        let frozen_report = analyze_genesis_events(&frozen_events)
            .expect("frozen Genesis event stream must satisfy observatory invariants");
        let evolving_report = analyze_genesis_events(&evolving_events)
            .expect("evolving Genesis event stream must satisfy observatory invariants");

        let frozen_transfer = transfer(&frozen_report, shock_1, shock_2);
        let evolving_transfer = transfer(&evolving_report, shock_1, shock_2);

        println!("seed={seed}");
        match &frozen_transfer {
            Ok(report) => {
                frozen_counts.record(report.verdict);
                println!("  frozen transfer:   {report:#?}");
            }
            Err(error) => {
                frozen_counts.unavailable += 1;
                println!("  frozen transfer unavailable: {error}");
            }
        }
        match &evolving_transfer {
            Ok(report) => {
                evolving_counts.record(report.verdict);
                println!("  evolving transfer: {report:#?}");
            }
            Err(error) => {
                evolving_counts.unavailable += 1;
                println!("  evolving transfer unavailable: {error}");
            }
        }

        if let (Ok(frozen), Ok(evolving)) = (&frozen_transfer, &evolving_transfer) {
            paired_valid += 1;
            let frozen_improved =
                frozen.verdict == RepeatShockTransferVerdictV1::ParetoImproved;
            let evolving_improved =
                evolving.verdict == RepeatShockTransferVerdictV1::ParetoImproved;
            match (frozen_improved, evolving_improved) {
                (false, true) => evolving_only_pareto_improved += 1,
                (true, false) => frozen_only_pareto_improved += 1,
                (true, true) => both_pareto_improved += 1,
                (false, false) => neither_pareto_improved += 1,
            }
        }
    }

    println!("fixed seed panel: {SEEDS:?}");
    println!("frozen verdict counts:   {frozen_counts:#?}");
    println!("evolving verdict counts: {evolving_counts:#?}");
    println!("paired-valid seeds: {paired_valid}/{}", SEEDS.len());
    println!("paired Pareto-improvement table:");
    println!("  evolving-only: {evolving_only_pareto_improved}");
    println!("  frozen-only:   {frozen_only_pareto_improved}");
    println!("  both:          {both_pareto_improved}");
    println!("  neither:       {neither_pareto_improved}");
    println!(
        "These counts are descriptive evidence only; no p-value, causal mechanism, or general \
         population claim is established by this example."
    );
}
