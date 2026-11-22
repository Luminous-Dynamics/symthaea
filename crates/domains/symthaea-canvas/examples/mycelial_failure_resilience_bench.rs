// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic controlled-failure benchmark for the junction–hypha graph.
//!
//! Run:
//! cargo run -p symthaea-canvas --example mycelial_failure_resilience_bench -- 100
//!
//! The graph and demand weights are synthetic. This is structural reachability
//! and geometric shortest-path analysis only; it is not biological transport,
//! flow, repair, adaptation or viability evidence.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    JunctionKind, JunctionLengthUnit, MAX_FAILURE_TOTAL_DEMAND_WEIGHT,
    MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION, MYCELIAL_MODEL_VERSION, MycelialDemandPair,
    MycelialFailureAttackConfig, MycelialFailureAttackStrategy,
    MycelialFailureDeliveryConfig, MycelialHyphaSegment, MycelialJunction,
    MycelialJunctionHyphaGraph, PARAMETER_SCALE, SpatialPoint3,
    build_mycelial_failure_attack_plan, build_mycelial_failure_delivery_profile,
    build_single_failure_resilience_profile, evaluate_mycelial_failure_attack_plan,
};

const DEFAULT_SAMPLES: usize = 100;
const WARMUP_RUNS: usize = 5;
const MAX_SAMPLES: usize = 10_000;

fn segment(
    id: &str,
    filament_id: u32,
    start: &str,
    end: &str,
    length: f64,
) -> MycelialHyphaSegment {
    MycelialHyphaSegment {
        id: id.to_string(),
        filament_id,
        t_start_micros: 0,
        t_end_micros: PARAMETER_SCALE,
        start_junction_id: start.to_string(),
        end_junction_id: end.to_string(),
        arc_length_model_units: length,
        start_junction_offset_model_units: 0.0,
        end_junction_offset_model_units: 0.0,
        radius_start_model_units: 1.0,
        radius_end_model_units: 1.0,
    }
}

fn fixture_graph() -> MycelialJunctionHyphaGraph {
    MycelialJunctionHyphaGraph {
        schema_version: MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION,
        source_model_version: MYCELIAL_MODEL_VERSION,
        length_unit: JunctionLengthUnit::OrganismLocalModelUnit,
        source_model_digest: "a".repeat(64),
        source_filament_ids: vec![1, 2, 3],
        maximum_junction_gap_model_units: 0.0,
        junctions: vec![
            MycelialJunction {
                id: "A".to_string(),
                kind: JunctionKind::Branch,
                position: SpatialPoint3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                incident_filament_ids: vec![1, 3],
                terminal_filament_ids: Vec::new(),
                anastomosis_ids: Vec::new(),
            },
            MycelialJunction {
                id: "B".to_string(),
                kind: JunctionKind::Tip,
                position: SpatialPoint3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                incident_filament_ids: vec![1, 2],
                terminal_filament_ids: vec![1],
                anastomosis_ids: Vec::new(),
            },
            MycelialJunction {
                id: "C".to_string(),
                kind: JunctionKind::Anastomosis,
                position: SpatialPoint3 {
                    x: 0.5,
                    y: 1.0,
                    z: 0.0,
                },
                incident_filament_ids: vec![2, 3],
                terminal_filament_ids: vec![2],
                anastomosis_ids: vec![7],
            },
        ],
        segments: vec![
            segment("edge-ab", 1, "A", "B", 1.0),
            segment("edge-bc", 2, "B", "C", 1.2),
            segment("edge-ca", 3, "C", "A", 1.2),
        ],
    }
}

fn demand_pairs() -> Vec<MycelialDemandPair> {
    vec![
        MycelialDemandPair {
            id: "demand-ab".to_string(),
            source_junction_id: "A".to_string(),
            target_junction_id: "B".to_string(),
            weight: 1,
        },
        MycelialDemandPair {
            id: "demand-ac".to_string(),
            source_junction_id: "A".to_string(),
            target_junction_id: "C".to_string(),
            weight: 2,
        },
        MycelialDemandPair {
            id: "demand-bc".to_string(),
            source_junction_id: "B".to_string(),
            target_junction_id: "C".to_string(),
            weight: 1,
        },
    ]
}

fn run_attack_family(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
    strategy: MycelialFailureAttackStrategy,
    seed: u64,
    spatial_seed_junction_id: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let plan = build_mycelial_failure_attack_plan(
        graph,
        demands,
        MycelialFailureAttackConfig {
            strategy,
            seed,
            maximum_failure_budget: 2,
            spatial_seed_junction_id: spatial_seed_junction_id.map(str::to_string),
        },
    )?;
    let evaluation = evaluate_mycelial_failure_attack_plan(graph, demands, &plan)?;
    println!(
        "attack_strategy:{} seed={} ranked_ids={:?} plan_digest={}",
        strategy.stable_id(),
        seed,
        plan.ranked_entity_ids,
        evaluation.attack_plan_digest
    );
    for (step, ranking) in plan.ranked_entity_ids_per_step.iter().enumerate() {
        println!("attack_ranking_step:{} candidates={:?}", step + 1, ranking);
    }
    for scenario in &evaluation.resilience_profile.scenarios {
        println!(
            "attack_scenario:{} failed_edges={:?} failed_junctions={:?} weighted_reachability={:?} weighted_directness={:?}",
            scenario.scenario_id,
            scenario.failed_segment_ids,
            scenario.failed_junction_ids,
            scenario.weighted_reachability,
            scenario.weighted_mean_directness
        );
    }
    println!(
        "attack_evaluation_digest:{}",
        evaluation.canonical_digest_hex()?
    );
    Ok(())
}

fn run_attack_delivery_uncertainty(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
) -> Result<(), Box<dyn Error>> {
    let plan = build_mycelial_failure_attack_plan(
        graph,
        demands,
        MycelialFailureAttackConfig {
            strategy: MycelialFailureAttackStrategy::DemandPathLoadSegments,
            seed: 0xacc0_0015,
            maximum_failure_budget: 2,
            spatial_seed_junction_id: None,
        },
    )?;
    println!("attack_delivery_uncertainty: simulated-not-empirical");
    for probability in [0, 250_000, 500_000, 750_000, 1_000_000] {
        let profile = build_mycelial_failure_delivery_profile(
            graph,
            demands,
            &plan,
            MycelialFailureDeliveryConfig {
                trials_per_scenario: 128,
                removal_success_probability_micros: probability,
                seed: 0xfeed_2026,
            },
        )?;
        println!(
            "delivery_probability_micros:{} profile_digest:{}",
            probability,
            profile.canonical_digest_hex()?
        );
        for summary in &profile.scenarios {
            println!(
                "delivery_scenario:{} attempts={} trials={} mean_realized_failures={:.3} probability_any_failure={:.3} reachability_mean={:?} reachability_p05={:?} reachability_p95={:?}",
                summary.scenario_id,
                summary.attempted_entity_count,
                summary.trial_count,
                summary.mean_realized_failure_count,
                summary.probability_at_least_one_failure,
                summary.mean_weighted_reachability,
                summary.weighted_reachability_p05,
                summary.weighted_reachability_p95
            );
        }
    }
    Ok(())
}

fn percentile(sorted: &[u128], percentage: usize) -> u128 {
    sorted[((sorted.len() - 1) * percentage) / 100]
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = match std::env::args().nth(1) {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_SAMPLES,
    };
    if !(1..=MAX_SAMPLES).contains(&samples) {
        return Err(format!("sample count must be in 1..={MAX_SAMPLES}").into());
    }

    let graph = fixture_graph();
    let demands = demand_pairs();
    if demands
        .iter()
        .map(|demand| u128::from(demand.weight))
        .sum::<u128>()
        > MAX_FAILURE_TOTAL_DEMAND_WEIGHT
    {
        return Err("fixture demand weight exceeds exact accumulation bound".into());
    }
    graph.validate()?;

    for _ in 0..WARMUP_RUNS {
        black_box(build_single_failure_resilience_profile(&graph, &demands)?);
    }
    let mut durations_ns = Vec::with_capacity(samples);
    let mut final_profile = None;
    for _ in 0..samples {
        let started = Instant::now();
        let profile = build_single_failure_resilience_profile(&graph, &demands)?;
        black_box(profile.canonical_digest_hex()?);
        durations_ns.push(started.elapsed().as_nanos());
        final_profile = Some(profile);
    }
    durations_ns.sort_unstable();

    let profile = final_profile.expect("samples are non-zero");
    println!("benchmark: mycelial-failure-resilience-v1");
    println!("scope: synthetic-structural-conformance-only");
    println!("junction_count: {}", graph.junctions.len());
    println!("hypha_segment_count: {}", graph.segments.len());
    println!("demand_pair_count: {}", demands.len());
    println!(
        "total_demand_weight: {}",
        profile.baseline.total_demand_weight
    );
    println!(
        "baseline_components: {}",
        profile.baseline.connected_components
    );
    println!("baseline_cycle_rank: {}", profile.baseline.cycle_rank);
    println!(
        "baseline_weighted_reachability: {:?}",
        profile.baseline.weighted_reachability
    );
    println!(
        "baseline_weighted_directness: {:?}",
        profile.baseline.weighted_mean_directness
    );
    println!(
        "baseline_weighted_path_length: {:?}",
        profile.baseline.weighted_mean_network_path_length
    );
    println!("single_failure_scenarios: {}", profile.scenarios.len());
    for scenario in &profile.scenarios {
        println!(
            "scenario:{} failed_edges={} failed_junctions={} components={} cycle_rank={} weighted_reachability={:?} weighted_directness={:?} weighted_path_length={:?}",
            scenario.scenario_id,
            scenario.failed_segment_count,
            scenario.failed_junction_count,
            scenario.connected_components,
            scenario.cycle_rank,
            scenario.weighted_reachability,
            scenario.weighted_mean_directness,
            scenario.weighted_mean_network_path_length
        );
    }
    println!(
        "worst_case_weighted_reachability: {:?}",
        profile.worst_case_weighted_reachability
    );
    println!(
        "worst_case_weighted_directness_among_reachable: {:?}",
        profile.worst_case_weighted_directness
    );
    println!("profile_digest: {}", profile.canonical_digest_hex()?);
    println!("attack_family_comparison: demand-weighted deterministic fixture policies");
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::RandomSegments,
        0x5eed,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::RandomJunctions,
        0x5eed,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::HighestDegreeJunctions,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::AdaptiveHighestDegreeJunctions,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::LowestDegreeJunctions,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::DemandPathLoadSegments,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::DemandPathLoadJunctions,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::AdaptiveDemandPathLoadSegments,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::AdaptiveDemandPathLoadJunctions,
        0,
        None,
    )?;
    run_attack_family(
        &graph,
        &demands,
        MycelialFailureAttackStrategy::SpatialClusterJunctions,
        0,
        Some("A"),
    )?;
    run_attack_delivery_uncertainty(&graph, &demands)?;
    println!("warmup_runs: {WARMUP_RUNS}");
    println!("samples: {samples}");
    println!(
        "single_failure_profile_time_ns_p50: {}",
        percentile(&durations_ns, 50)
    );
    println!(
        "single_failure_profile_time_ns_p95: {}",
        percentile(&durations_ns, 95)
    );
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!("length_unit: organism_local_model_unit");
    println!(
        "claim_boundary: structural reachability and geometric paths only; no biology, flow, transport or repair claims"
    );
    Ok(())
}
