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
    RecoveryMetrics, RepeatShockErrorV1, RepeatShockLatencyV1, RepeatShockTransferV1,
    RepeatShockTransferVerdictV1, RepeatedShockDidV1, RepeatedShockDidVerdictV1,
    RepeatedShockRelativeV1, RepeatedShockRelativeVerdictV1, ResourcePerturbation,
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

    fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pareto_improved": self.pareto_improved,
            "mixed": self.mixed,
            "no_directional_change": self.no_directional_change,
            "pareto_worse": self.pareto_worse,
            "unavailable": self.unavailable,
        })
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

    fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pareto_evolving": self.pareto_evolving,
            "mixed": self.mixed,
            "no_directional_difference": self.no_directional_difference,
            "pareto_frozen": self.pareto_frozen,
            "unavailable": self.unavailable,
        })
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

    fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pareto_candidate": self.pareto_candidate,
            "mixed": self.mixed,
            "no_directional_difference": self.no_directional_difference,
            "pareto_reference": self.pareto_reference,
            "unavailable": self.unavailable,
        })
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

fn digest_json(value: &serde_json::Value) -> (String, String) {
    let bytes = serde_json::to_vec(value).expect("evidence JSON must serialize");
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let json = String::from_utf8(bytes).expect("serde_json output is UTF-8");
    (digest, json)
}

fn emit_protocol_evidence() -> String {
    let (digest, json) = digest_json(&protocol_evidence_json());
    println!("protocol_sha256={digest}");
    println!("protocol_json={json}");
    digest
}

fn repeat_shock_verdict_name(verdict: RepeatShockTransferVerdictV1) -> &'static str {
    match verdict {
        RepeatShockTransferVerdictV1::ParetoImproved => "ParetoImproved",
        RepeatShockTransferVerdictV1::Mixed => "Mixed",
        RepeatShockTransferVerdictV1::NoDirectionalChange => "NoDirectionalChange",
        RepeatShockTransferVerdictV1::ParetoWorse => "ParetoWorse",
    }
}

fn did_verdict_name(verdict: RepeatedShockDidVerdictV1) -> &'static str {
    match verdict {
        RepeatedShockDidVerdictV1::ParetoEvolving => "ParetoEvolving",
        RepeatedShockDidVerdictV1::Mixed => "Mixed",
        RepeatedShockDidVerdictV1::NoDirectionalDifference => "NoDirectionalDifference",
        RepeatedShockDidVerdictV1::ParetoFrozen => "ParetoFrozen",
    }
}

fn relative_verdict_name(verdict: RepeatedShockRelativeVerdictV1) -> &'static str {
    match verdict {
        RepeatedShockRelativeVerdictV1::ParetoCandidate => "ParetoCandidate",
        RepeatedShockRelativeVerdictV1::Mixed => "Mixed",
        RepeatedShockRelativeVerdictV1::NoDirectionalDifference => "NoDirectionalDifference",
        RepeatedShockRelativeVerdictV1::ParetoReference => "ParetoReference",
    }
}

fn latency_json(latency: RepeatShockLatencyV1) -> serde_json::Value {
    match latency {
        RepeatShockLatencyV1::BothRecovered {
            first_latency_ticks,
            second_latency_ticks,
            latency_advantage_ticks,
        } => serde_json::json!({
            "status": "BothRecovered",
            "first_latency_ticks": first_latency_ticks,
            "second_latency_ticks": second_latency_ticks,
            "latency_advantage_ticks": latency_advantage_ticks.to_string(),
        }),
        RepeatShockLatencyV1::SecondOnlyRecovered => {
            serde_json::json!({"status": "SecondOnlyRecovered"})
        }
        RepeatShockLatencyV1::FirstOnlyRecovered => {
            serde_json::json!({"status": "FirstOnlyRecovered"})
        }
        RepeatShockLatencyV1::NeitherRecovered => {
            serde_json::json!({"status": "NeitherRecovered"})
        }
    }
}

fn transfer_json(result: &Result<RepeatShockTransferV1, String>) -> serde_json::Value {
    match result {
        Ok(report) => serde_json::json!({
            "status": "ok",
            "verdict": repeat_shock_verdict_name(report.verdict),
            "first_baseline_mean_observed_population": report.first_baseline_mean_observed_population,
            "second_baseline_mean_observed_population": report.second_baseline_mean_observed_population,
            "minimum_during_fraction_delta": report.minimum_during_fraction_delta,
            "minimum_after_fraction_delta": report.minimum_after_fraction_delta,
            "final_fraction_delta": report.final_fraction_delta,
            "deficit_area_advantage": report.deficit_area_advantage,
            "latency": latency_json(report.latency),
        }),
        Err(error) => serde_json::json!({
            "status": "unavailable",
            "error": error,
        }),
    }
}

fn did_json(result: &Result<RepeatedShockDidV1, String>) -> serde_json::Value {
    match result {
        Ok(report) => serde_json::json!({
            "status": "ok",
            "verdict": did_verdict_name(report.verdict),
            "minimum_during_fraction_did": report.minimum_during_fraction_did,
            "minimum_after_fraction_did": report.minimum_after_fraction_did,
            "final_fraction_did": report.final_fraction_did,
            "deficit_area_advantage_did": report.deficit_area_advantage_did,
            "frozen_latency_transfer": latency_json(report.frozen_latency_transfer),
            "evolving_latency_transfer": latency_json(report.evolving_latency_transfer),
        }),
        Err(error) => serde_json::json!({
            "status": "unavailable",
            "error": error,
        }),
    }
}

fn relative_json(result: &Result<RepeatedShockRelativeV1, String>) -> serde_json::Value {
    match result {
        Ok(report) => serde_json::json!({
            "status": "ok",
            "verdict": relative_verdict_name(report.verdict),
            "minimum_during_candidate_minus_reference": report.minimum_during_candidate_minus_reference,
            "minimum_after_candidate_minus_reference": report.minimum_after_candidate_minus_reference,
            "final_fraction_candidate_minus_reference": report.final_fraction_candidate_minus_reference,
            "deficit_area_advantage_candidate_minus_reference": report.deficit_area_advantage_candidate_minus_reference,
            "reference_latency_transfer": latency_json(report.reference_latency_transfer),
            "candidate_latency_transfer": latency_json(report.candidate_latency_transfer),
        }),
        Err(error) => serde_json::json!({
            "status": "unavailable",
            "error": error,
        }),
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

fn did(
    frozen: &Result<RepeatShockTransferV1, String>,
    selected: &Result<RepeatShockTransferV1, String>,
) -> Result<RepeatedShockDidV1, String> {
    let frozen = frozen
        .as_ref()
        .map_err(|error| format!("frozen transfer unavailable: {error}"))?;
    let selected = selected
        .as_ref()
        .map_err(|error| format!("selected transfer unavailable: {error}"))?;
    compare_repeated_shock_did(frozen, selected)
        .map_err(|error| format!("DID unavailable: {error:?}"))
}

fn relative(
    reference_name: &str,
    reference: &Result<RepeatShockTransferV1, String>,
    candidate_name: &str,
    candidate: &Result<RepeatShockTransferV1, String>,
) -> Result<RepeatedShockRelativeV1, String> {
    let reference = reference
        .as_ref()
        .map_err(|error| format!("{reference_name} transfer unavailable: {error}"))?;
    let candidate = candidate
        .as_ref()
        .map_err(|error| format!("{candidate_name} transfer unavailable: {error}"))?;
    compare_repeated_shock_relative(reference, candidate)
        .map_err(|error| format!("relative transfer unavailable: {error:?}"))
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

fn record_did(label: &str, result: &Result<RepeatedShockDidV1, String>, counts: &mut DidCounts) {
    match result {
        Ok(report) => {
            counts.record(report.verdict);
            println!("  {label}: {report:#?}");
        }
        Err(error) => {
            counts.unavailable += 1;
            println!("  {label} unavailable: {error}");
        }
    }
}

fn record_relative(
    label: &str,
    result: &Result<RepeatedShockRelativeV1, String>,
    counts: &mut RelativeCounts,
) {
    match result {
        Ok(report) => {
            counts.record(report.verdict);
            println!("  {label}: {report:#?}");
        }
        Err(error) => {
            counts.unavailable += 1;
            println!("  {label} unavailable: {error}");
        }
    }
}

fn main() {
    let protocol_sha256 = emit_protocol_evidence();

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
    let mut seed_results = Vec::with_capacity(SEEDS.len());

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
        let selected_vs_frozen_did = did(&frozen_transfer, &selected_transfer);
        let random_peer_vs_frozen = relative(
            "frozen",
            &frozen_transfer,
            "random-peer",
            &random_peer_transfer,
        );
        let selected_vs_random_peer = relative(
            "random-peer",
            &random_peer_transfer,
            "selected",
            &selected_transfer,
        );

        println!("seed={seed}");
        record_transfer("frozen", &frozen_transfer, &mut frozen_counts);
        record_transfer("selected", &selected_transfer, &mut selected_counts);
        record_transfer(
            "random-peer",
            &random_peer_transfer,
            &mut random_peer_counts,
        );
        record_did(
            "selected-vs-frozen DID",
            &selected_vs_frozen_did,
            &mut selected_vs_frozen_did_counts,
        );
        record_relative(
            "random-peer-vs-frozen relative transfer",
            &random_peer_vs_frozen,
            &mut random_peer_vs_frozen_counts,
        );
        record_relative(
            "selected-vs-random-peer relative transfer",
            &selected_vs_random_peer,
            &mut selected_vs_random_peer_counts,
        );

        if frozen_transfer.is_ok() && selected_transfer.is_ok() && random_peer_transfer.is_ok() {
            all_three_valid += 1;
        }

        seed_results.push(serde_json::json!({
            "seed": seed,
            "scheduler_seed": scheduler_seed,
            "frozen_transfer": transfer_json(&frozen_transfer),
            "selected_transfer": transfer_json(&selected_transfer),
            "random_peer_transfer": transfer_json(&random_peer_transfer),
            "selected_vs_frozen_did": did_json(&selected_vs_frozen_did),
            "random_peer_vs_frozen_relative": relative_json(&random_peer_vs_frozen),
            "selected_vs_random_peer_relative": relative_json(&selected_vs_random_peer),
        }));
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

    let results = serde_json::json!({
        "schema": "symthaea.alife.repeated-shock.results.v1",
        "protocol_sha256": protocol_sha256,
        "seed_results": seed_results,
        "aggregate": {
            "seed_count": SEEDS.len(),
            "all_three_valid": all_three_valid,
            "frozen_transfer": frozen_counts.as_json(),
            "selected_transfer": selected_counts.as_json(),
            "random_peer_transfer": random_peer_counts.as_json(),
            "selected_vs_frozen_did": selected_vs_frozen_did_counts.as_json(),
            "random_peer_vs_frozen_relative": random_peer_vs_frozen_counts.as_json(),
            "selected_vs_random_peer_relative": selected_vs_random_peer_counts.as_json(),
        },
        "evidence_boundary": {
            "descriptive_only": true,
            "statistical_population_generalization": false,
            "lineage_mechanism_established": false,
            "exact_common_random_number_mutation_replay": false,
        }
    });
    let (results_sha256, results_json) = digest_json(&results);
    println!("results_sha256={results_sha256}");
    println!("results_json={results_json}");
    println!(
        "The selected-vs-random-peer contrast is a selection-link ablation: both mutation-enabled \
         arms use the same nominal mutation rate, while RandomPeer breaks successful-parent -> \
         genome-source inheritance. RandomPeer also consumes RNG for source selection, so this is \
         not an exact common-random-number mutation replay. All counts remain descriptive evidence \
         only; no p-value, lineage mechanism, or population-level causal generalization is \
         established by this example."
    );
}
