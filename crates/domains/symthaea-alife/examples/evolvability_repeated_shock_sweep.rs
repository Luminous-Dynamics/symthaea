// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired multi-seed repeated-shock evolvability sweep.
//!
//! This example applies the already-defined repeated-shock transfer criterion unchanged across a
//! fixed eight-seed panel. Frozen and evolving conditions share the same population and encounter
//! seeds within each pair. The output is descriptive evidence: verdict counts, paired outcomes,
//! and a conservative difference-in-differences contrast — not a statistical-significance or
//! mechanism-causality claim.

use sha2::{Digest, Sha256};
use symthaea_alife::{
    EncounterScheduler, EvolvabilityError, GenesisEvent, InheritanceMode, ObservatoryReport,
    OrganismConfig, PairingMode, PerturbationSchedule, Population, PopulationConfig,
    RecoveryMetrics, RepeatShockErrorV1, RepeatShockTransferV1, RepeatShockTransferVerdictV1,
    RepeatedShockDidVerdictV1, ResourcePerturbation, analyze_genesis_events,
    analyze_recovery_through, compare_repeated_shock_did, compare_repeated_shocks,
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
const SHOCK_DELTA: f64 = 0.0;
const SCHEDULER_SEED_OFFSET: u64 = 100_003;
const FROZEN_MUTATION_RATE: f64 = 0.0;
const EVOLVING_MUTATION_RATE: f64 = 0.1;
const MUTATION_STD: f64 = 0.05;

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

#[derive(Debug, Default)]
struct DidCounts {
    pareto_evolving: usize,
    mixed: usize,
    no_directional_difference: usize,
    pareto_frozen: usize,
    unavailable: usize,
}

impl DidCounts {
    fn record(&mut self, verdict: RepeatedShockDidVerdictV1) {
        match verdict {
            RepeatedShockDidVerdictV1::ParetoEvolving => self.pareto_evolving += 1,
            RepeatedShockDidVerdictV1::Mixed => self.mixed += 1,
            RepeatedShockDidVerdictV1::NoDirectionalDifference => {
                self.no_directional_difference += 1
            }
            RepeatedShockDidVerdictV1::ParetoFrozen => self.pareto_frozen += 1,
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
        mutation_std: MUTATION_STD,
        inheritance: InheritanceMode::FromParent,
    }
}

fn protocol_evidence_json() -> serde_json::Value {
    let frozen = population_config(FROZEN_MUTATION_RATE);
    let evolving = population_config(EVOLVING_MUTATION_RATE);
    let organism = frozen.organism_cfg;

    serde_json::json!({
        "schema": "symthaea.alife.repeated-shock.protocol.v1",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "seed_panel": SEEDS,
        "simulation": {
            "ticks": TICKS,
            "initial_count": INITIAL_COUNT,
        },
        "resource": {
            "rule": "shared_pool_total_div_population_max_1",
            "plant_resource_total": PLANT_RESOURCE_TOTAL,
        },
        "scheduler": {
            "pairing_mode": "Random",
            "seed_rule": "population_seed.wrapping_add(offset)",
            "seed_offset": SCHEDULER_SEED_OFFSET,
        },
        "population": {
            "death_energy_threshold": frozen.death_energy_threshold,
            "reproduction_energy_threshold": frozen.reproduction_energy_threshold,
            "reproduction_energy_cost": frozen.reproduction_energy_cost,
            "inheritance_mode": "FromParent",
            "mutation_std": MUTATION_STD,
            "frozen_mutation_rate": frozen.mutation_rate,
            "evolving_mutation_rate": evolving.mutation_rate,
        },
        "organism_config": {
            "set_point": organism.set_point,
            "metabolic_cost": organism.metabolic_cost,
            "forage_activity_cost": organism.forage_activity_cost,
            "forage_efficiency": organism.forage_efficiency,
            "goal_precision": organism.goal_precision,
            "effective_temperature": organism.effective_temperature,
            "dissipation_rate": organism.dissipation_rate,
            "death_energy_threshold": organism.death_energy_threshold,
            "action_temperature": organism.action_temperature,
            "perceptual_grain": organism.perceptual_grain,
            "spoilage_sigma": organism.spoilage_sigma,
            "resource_preference": organism.resource_preference,
            "resource_prior": organism.resource_prior,
            "social_enabled": organism.social_enabled,
            "transfer_quantum": organism.transfer_quantum,
        },
        "shocks": [
            {
                "start_tick": SHOCK_1_START,
                "duration_ticks": SHOCK_DURATION,
                "multiplier": SHOCK_MULTIPLIER,
                "delta": SHOCK_DELTA,
            },
            {
                "start_tick": SHOCK_2_START,
                "duration_ticks": SHOCK_DURATION,
                "multiplier": SHOCK_MULTIPLIER,
                "delta": SHOCK_DELTA,
            }
        ],
        "analysis": {
            "baseline_lookback_ticks": BASELINE_LOOKBACK_TICKS,
            "recovery_fraction": RECOVERY_FRACTION,
            "post_shock_evaluation_ticks": POST_SHOCK_EVALUATION_TICKS,
            "repeat_shock_verdict": "pareto_predeclared_dimensions",
            "difference_in_differences": "evolving_transfer_minus_frozen_transfer",
            "did_latency_in_pareto_verdict": false,
        }
    })
}

fn emit_protocol_evidence() {
    let protocol = protocol_evidence_json();
    let canonical = serde_json::to_vec(&protocol).expect("protocol evidence must serialize");
    let digest = Sha256::digest(&canonical);
    let digest_hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let canonical = String::from_utf8(canonical).expect("serde_json output is UTF-8");
    println!("protocol_sha256={digest_hex}");
    println!("protocol_json={canonical}");
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
    emit_protocol_evidence();

    let shock_1 = ResourcePerturbation::new(
        SHOCK_1_START,
        SHOCK_DURATION,
        SHOCK_MULTIPLIER,
        SHOCK_DELTA,
    )
    .expect("valid first shock");
    let shock_2 = ResourcePerturbation::new(
        SHOCK_2_START,
        SHOCK_DURATION,
        SHOCK_MULTIPLIER,
        SHOCK_DELTA,
    )
    .expect("valid second shock");
    let schedule = PerturbationSchedule::new(vec![shock_1, shock_2]);

    let mut frozen_counts = VerdictCounts::default();
    let mut evolving_counts = VerdictCounts::default();
    let mut did_counts = DidCounts::default();
    let mut paired_valid = 0usize;
    let mut evolving_only_pareto_improved = 0usize;
    let mut frozen_only_pareto_improved = 0usize;
    let mut both_pareto_improved = 0usize;
    let mut neither_pareto_improved = 0usize;

    for &seed in SEEDS {
        let scheduler_seed = seed.wrapping_add(SCHEDULER_SEED_OFFSET);
        let frozen_events = run_condition(FROZEN_MUTATION_RATE, seed, scheduler_seed, &schedule);
        let evolving_events = run_condition(EVOLVING_MUTATION_RATE, seed, scheduler_seed, &schedule);
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

            match compare_repeated_shock_did(frozen, evolving) {
                Ok(did) => {
                    did_counts.record(did.verdict);
                    println!("  paired difference-in-differences: {did:#?}");
                }
                Err(error) => {
                    did_counts.unavailable += 1;
                    println!("  paired difference-in-differences unavailable: {error:?}");
                }
            }
        } else {
            did_counts.unavailable += 1;
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
    println!("paired repeated-shock difference-in-differences counts: {did_counts:#?}");
    println!(
        "Difference-in-differences is a directional paired contrast over predeclared continuous \
         recovery dimensions. These counts remain descriptive evidence only; no p-value, causal \
         mutation mechanism, or population-level generalization is established by this example."
    );
}
