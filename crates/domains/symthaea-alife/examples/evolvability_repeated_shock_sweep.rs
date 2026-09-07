// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired multi-seed repeated-shock evolvability sweep.
//!
//! Three conditions share the same founder and encounter-scheduler seeds within every seed panel
//! entry:
//!
//! - frozen: no mutation, fitness-linked `FromParent` inheritance;
//! - selected evolving: mutation enabled, fitness-linked `FromParent` inheritance;
//! - random-peer evolving: the same mutation rate, but `RandomPeer` deliberately breaks the
//!   successful-reproducer -> inherited-genome link.
//!
//! The RandomPeer arm is a selection-link ablation, not an exact common-random-number replay of
//! mutation draws: source selection itself consumes the population RNG and therefore changes the
//! later RNG trajectory. Results remain descriptive evidence rather than a statistical or causal
//! mechanism claim.

use sha2::{Digest, Sha256};
use symthaea_alife::{
    EncounterScheduler, EvolvabilityError, GenesisEvent, InheritanceMode, ObservatoryReport,
    OrganismConfig, PairingMode, PerturbationSchedule, Population, PopulationConfig,
    RecoveryMetrics, RepeatShockErrorV1, RepeatShockTransferV1, RepeatShockTransferVerdictV1,
    RepeatedShockDidVerdictV1, RepeatedShockRelativeVerdictV1, ResourcePerturbation,
    analyze_genesis_events, analyze_recovery_through, compare_repeated_shock_did,
    compare_repeated_shock_relative, compare_repeated_shocks,
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

#[derive(Debug, Default)]
struct RelativeCounts {
    pareto_candidate: usize,
    mixed: usize,
    no_directional_difference: usize,
    pareto_reference: usize,
    unavailable: usize,
}

impl RelativeCounts {
    fn record(&mut self, verdict: RepeatedShockRelativeVerdictV1) {
        match verdict {
            RepeatedShockRelativeVerdictV1::ParetoCandidate => self.pareto_candidate += 1,
            RepeatedShockRelativeVerdictV1::Mixed => self.mixed += 1,
            RepeatedShockRelativeVerdictV1::NoDirectionalDifference => {
                self.no_directional_difference += 1
            }
            RepeatedShockRelativeVerdictV1::ParetoReference => self.pareto_reference += 1,
        }
    }
}

fn inheritance_name(inheritance: InheritanceMode) -> &'static str {
    match inheritance {
        InheritanceMode::FromParent => "FromParent",
        InheritanceMode::RandomPeer => "RandomPeer",
    }
}

fn population_config(mutation_rate: f64, inheritance: InheritanceMode) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(),
        mutation_rate,
        mutation_std: MUTATION_STD,
        inheritance,
    }
}

fn protocol_evidence_json() -> serde_json::Value {
    let frozen = population_config(FROZEN_MUTATION_RATE, InheritanceMode::FromParent);
    let selected = population_config(EVOLVING_MUTATION_RATE, InheritanceMode::FromParent);
    let random_peer = population_config(EVOLVING_MUTATION_RATE, InheritanceMode::RandomPeer);
    let organism = frozen.organism_cfg;

    serde_json::json!({
        "schema": "symthaea.alife.repeated-shock.protocol.v2",
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
            "same_initial_seed_across_conditions": true,
        },
        "conditions": {
            "frozen": {
                "mutation_rate": frozen.mutation_rate,
                "inheritance_mode": inheritance_name(frozen.inheritance),
            },
            "selected_evolving": {
                "mutation_rate": selected.mutation_rate,
                "inheritance_mode": inheritance_name(selected.inheritance),
            },
            "random_peer_evolving": {
                "mutation_rate": random_peer.mutation_rate,
                "inheritance_mode": inheritance_name(random_peer.inheritance),
                "ablation": "break_fitness_link_to_inherited_genome_source",
            },
        },
        "population": {
            "death_energy_threshold": frozen.death_energy_threshold,
            "reproduction_energy_threshold": frozen.reproduction_energy_threshold,
            "reproduction_energy_cost": frozen.reproduction_energy_cost,
            "mutation_std": MUTATION_STD,
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
            "contrasts": [
                "selected_evolving_minus_frozen_transfer",
                "random_peer_evolving_minus_frozen_transfer",
                "selected_evolving_minus_random_peer_evolving_transfer"
            ],
            "selected_vs_frozen_did": "selected_transfer_minus_frozen_transfer",
            "relative_latency_in_pareto_verdict": false,
        },
        "known_control_limitation": {
            "random_peer_source_selection_consumes_population_rng": true,
            "exact_common_random_number_mutation_replay": false,
            "interpretation": "selection_link_ablation_not_exact_mutation_draw_replay",
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
    inheritance: InheritanceMode,
    population_seed: u64,
    scheduler_seed: u64,
    schedule: &PerturbationSchedule,
) -> Vec<GenesisEvent> {
    let mut population = Population::new(
        population_config(mutation_rate, inheritance),
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

fn record_transfer(
    label: &str,
    result: &Result<RepeatShockTransferV1, String>,
    counts: &mut VerdictCounts,
) {
    match result {
        Ok(report) => {
            counts.record(report.verdict);
            println!("  {label} transfer: {report:#?}");
        }
        Err(error) => {
            counts.unavailable += 1;
            println!("  {label} transfer unavailable: {error}");
        }
    }
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
    let mut selected_counts = VerdictCounts::default();
    let mut random_peer_counts = VerdictCounts::default();
    let mut selected_vs_frozen_did_counts = DidCounts::default();
    let mut random_peer_vs_frozen_counts = RelativeCounts::default();
    let mut selected_vs_random_peer_counts = RelativeCounts::default();
    let mut all_three_valid = 0usize;

    for &seed in SEEDS {
        let scheduler_seed = seed.wrapping_add(SCHEDULER_SEED_OFFSET);
        let frozen_events = run_condition(
            FROZEN_MUTATION_RATE,
            InheritanceMode::FromParent,
            seed,
            scheduler_seed,
            &schedule,
        );
        let selected_events = run_condition(
            EVOLVING_MUTATION_RATE,
            InheritanceMode::FromParent,
            seed,
            scheduler_seed,
            &schedule,
        );
        let random_peer_events = run_condition(
            EVOLVING_MUTATION_RATE,
            InheritanceMode::RandomPeer,
            seed,
            scheduler_seed,
            &schedule,
        );

        let frozen_report = analyze_genesis_events(&frozen_events)
            .expect("frozen Genesis event stream must satisfy observatory invariants");
        let selected_report = analyze_genesis_events(&selected_events)
            .expect("selected Genesis event stream must satisfy observatory invariants");
        let random_peer_report = analyze_genesis_events(&random_peer_events)
            .expect("RandomPeer Genesis event stream must satisfy observatory invariants");

        let frozen_transfer = transfer(&frozen_report, shock_1, shock_2);
        let selected_transfer = transfer(&selected_report, shock_1, shock_2);
        let random_peer_transfer = transfer(&random_peer_report, shock_1, shock_2);

        println!("seed={seed}");
        record_transfer("frozen", &frozen_transfer, &mut frozen_counts);
        record_transfer("selected", &selected_transfer, &mut selected_counts);
        record_transfer(
            "random-peer",
            &random_peer_transfer,
            &mut random_peer_counts,
        );

        match (&frozen_transfer, &selected_transfer) {
            (Ok(frozen), Ok(selected)) => match compare_repeated_shock_did(frozen, selected) {
                Ok(did) => {
                    selected_vs_frozen_did_counts.record(did.verdict);
                    println!("  selected-vs-frozen DID: {did:#?}");
                }
                Err(error) => {
                    selected_vs_frozen_did_counts.unavailable += 1;
                    println!("  selected-vs-frozen DID unavailable: {error:?}");
                }
            },
            _ => selected_vs_frozen_did_counts.unavailable += 1,
        }

        match (&frozen_transfer, &random_peer_transfer) {
            (Ok(frozen), Ok(random_peer)) => {
                match compare_repeated_shock_relative(frozen, random_peer) {
                    Ok(relative) => {
                        random_peer_vs_frozen_counts.record(relative.verdict);
                        println!("  random-peer-vs-frozen relative transfer: {relative:#?}");
                    }
                    Err(error) => {
                        random_peer_vs_frozen_counts.unavailable += 1;
                        println!("  random-peer-vs-frozen relative unavailable: {error:?}");
                    }
                }
            }
            _ => random_peer_vs_frozen_counts.unavailable += 1,
        }

        match (&random_peer_transfer, &selected_transfer) {
            (Ok(random_peer), Ok(selected)) => {
                match compare_repeated_shock_relative(random_peer, selected) {
                    Ok(relative) => {
                        selected_vs_random_peer_counts.record(relative.verdict);
                        println!("  selected-vs-random-peer relative transfer: {relative:#?}");
                    }
                    Err(error) => {
                        selected_vs_random_peer_counts.unavailable += 1;
                        println!("  selected-vs-random-peer relative unavailable: {error:?}");
                    }
                }
            }
            _ => selected_vs_random_peer_counts.unavailable += 1,
        }

        if frozen_transfer.is_ok() && selected_transfer.is_ok() && random_peer_transfer.is_ok() {
            all_three_valid += 1;
        }
    }

    println!("fixed seed panel: {SEEDS:?}");
    println!("frozen transfer verdict counts:       {frozen_counts:#?}");
    println!("selected transfer verdict counts:     {selected_counts:#?}");
    println!("random-peer transfer verdict counts:  {random_peer_counts:#?}");
    println!("all-three-valid seeds: {all_three_valid}/{}", SEEDS.len());
    println!(
        "selected-vs-frozen repeated-shock DID counts: \
         {selected_vs_frozen_did_counts:#?}"
    );
    println!(
        "random-peer-vs-frozen relative-transfer counts: \
         {random_peer_vs_frozen_counts:#?}"
    );
    println!(
        "selected-vs-random-peer relative-transfer counts: \
         {selected_vs_random_peer_counts:#?}"
    );
    println!(
        "The selected-vs-random-peer contrast is a selection-link ablation: both mutation-enabled \
         arms use the same nominal mutation rate, while RandomPeer breaks successful-parent -> \
         genome-source inheritance. RandomPeer also consumes RNG for source selection, so this is \
         not an exact common-random-number mutation replay. All counts remain descriptive evidence \
         only; no p-value, lineage mechanism, or population-level causal generalization is \
         established by this example."
    );
}
