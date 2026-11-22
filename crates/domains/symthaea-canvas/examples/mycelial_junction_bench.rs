// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic baseline for the junction–hypha graph adapter.
//!
//! Run from the repository root:
//! cargo run -p symthaea-canvas --example mycelial_junction_bench -- 100
//!
//! This measures deterministic renderer-local geometry only. Generated
//! origins are not biological roots; no fusion proposal is accepted here;
//! no physical dimensions, biological fidelity or flow transport is claimed.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    EmbodimentSnapshot, MycelialContactFiltrationConfig, MycelialJunctionGraphConfig,
    MycelialNetworkConfig, PresenceMode, build_mycelial_contact_filtration,
    build_mycelial_junction_hypha_graph, build_mycelial_model,
    build_observed_network_minimum_spanning_forest, detect_anastomosis_candidates,
};

const DEFAULT_SAMPLES: usize = 100;
const WARMUP_RUNS: usize = 5;
const MAX_SAMPLES: usize = 10_000;

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

    let model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Cognitive);
    let config = MycelialJunctionGraphConfig::default();
    let fusion_config = MycelialNetworkConfig::default();

    let candidates = detect_anastomosis_candidates(&model, fusion_config)?;
    let filtration_started = Instant::now();
    let contact_filtration = build_mycelial_contact_filtration(
        &model,
        &candidates,
        MycelialContactFiltrationConfig {
            maximum_contact_tolerance_model_units: fusion_config.max_fusion_distance,
        },
    )?;
    let contact_filtration_elapsed_ns = filtration_started.elapsed().as_nanos();
    let contact_filtration_digest = contact_filtration.canonical_digest_hex()?;
    let root_ids: Vec<_> = model
        .filaments
        .iter()
        .filter(|filament| filament.parent_id.is_none())
        .map(|filament| format!("origin:{}", filament.id))
        .collect();

    for _ in 0..WARMUP_RUNS {
        let graph = build_mycelial_junction_hypha_graph(black_box(&model), &[], config)?;
        black_box(graph.canonical_digest_hex()?);
        black_box(graph.metrics()?);
        black_box(graph.global_geometric_efficiency()?);
        black_box(graph.root_directed_geometric_efficiency(&root_ids)?);
    }

    let mut durations_ns = Vec::with_capacity(samples);
    let mut final_graph = None;
    for _ in 0..samples {
        let started = Instant::now();
        let graph = build_mycelial_junction_hypha_graph(black_box(&model), &[], config)?;
        black_box(graph.canonical_digest_hex()?);
        let metrics = graph.metrics()?;
        let global = graph.global_geometric_efficiency()?;
        let root = graph.root_directed_geometric_efficiency(&root_ids)?;
        black_box((metrics, global, root));
        durations_ns.push(started.elapsed().as_nanos());
        final_graph = Some(graph);
    }
    durations_ns.sort_unstable();

    let graph = final_graph.expect("sample count is non-zero");
    let metrics = graph.metrics()?;
    let global = graph.global_geometric_efficiency()?;
    let root = graph.root_directed_geometric_efficiency(&root_ids)?;
    let mst_report = build_observed_network_minimum_spanning_forest(&graph, &root_ids)?;
    let digest = graph.canonical_digest_hex()?;

    println!("benchmark: mycelial-junction-hypha-v1");
    println!("model_schema_version: {}", model.schema_version);
    println!("model_filaments: {}", model.filament_count());
    println!(
        "model_terminal_filaments: {}",
        model.terminal_filament_count()
    );
    println!("accepted_anastomoses: 0");
    println!("anastomosis_candidates_proposed: {}", candidates.len());
    println!(
        "candidate_contacts_admitted_by_tolerance: {}",
        contact_filtration.metrics.candidate_count_in_filtration
    );
    println!(
        "candidate_contacts_above_tolerance: {}",
        contact_filtration.metrics.candidate_count_above_maximum
    );
    println!(
        "candidate_component_merges: {}",
        contact_filtration.metrics.component_merge_count
    );
    println!(
        "candidate_cycle_births_right_censored: {}",
        contact_filtration.metrics.cycle_birth_count
    );
    println!(
        "candidate_contact_conflict_groups: {}",
        contact_filtration.metrics.contact_conflict_group_count
    );
    println!(
        "candidate_conflicting_candidate_count: {}",
        contact_filtration.metrics.conflicting_candidate_count
    );
    println!(
        "contact_filtration_interpretation: all_candidate_union_upper_bound_not_jointly_accepted"
    );
    println!(
        "contact_filtration_threshold_snapshots: {}",
        contact_filtration.snapshots.len()
    );
    println!(
        "contact_filtration_time_ns: {contact_filtration_elapsed_ns}"
    );
    println!("contact_filtration_digest: {contact_filtration_digest}");
    println!("junctions: {}", metrics.junction_count);
    println!("hypha_segments: {}", metrics.hypha_segment_count);
    println!("connected_components: {}", metrics.connected_components);
    println!("cycle_rank: {}", metrics.cycle_rank);
    println!(
        "hypha_arc_length_model_units_microunits: {}",
        metrics.total_hypha_length_model_units_microunits
    );
    println!(
        "junction_weld_offsets_model_units_microunits: {}",
        metrics.total_junction_offset_model_units_microunits
    );
    println!(
        "maximum_junction_gap_model_units_microunits: {}",
        metrics.maximum_junction_gap_microunits
    );
    println!("length_unit: organism_local_model_unit");
    println!("global_geometric_directness: {:?}", global.mean_directness);
    println!("global_eligible_pairs: {}", global.eligible_pairs);
    println!("global_reachable_pairs: {}", global.reachable_pairs);
    println!("global_unreachable_pairs: {}", global.unreachable_pairs);
    println!("root_directness_scope: declared_renderer_origin_anchors_not_biological_roots");
    println!("declared_origin_anchors: {}", root_ids.len());
    println!("root_geometric_directness: {:?}", root.mean_directness);
    println!("root_eligible_terminal_junctions: {}", root.eligible_pairs);
    println!(
        "root_reachable_terminal_junctions: {}",
        root.reachable_pairs
    );
    println!(
        "root_unreachable_terminal_junctions: {}",
        root.unreachable_pairs
    );
    println!("source_geometry_digest: {}", graph.source_geometry_digest);
    println!("junction_graph_digest: {digest}");
    println!("observed_network_mst_selected_segments: {}", mst_report.selected_segment_count);
    println!("observed_network_mst_components: {}", mst_report.connected_components);
    println!(
        "observed_network_mst_arc_cost_model_units_microunits: {}",
        mst_report.hypha_arc_cost_model_units_microunits
    );
    println!(
        "observed_network_mst_traversal_cost_model_units_microunits: {}",
        mst_report.traversal_cost_model_units_microunits
    );
    println!(
        "observed_network_mst_global_directness: {:?}",
        mst_report.global_geometric_efficiency.mean_directness
    );
    println!(
        "observed_network_mst_root_directness: {:?}",
        mst_report.root_directed_geometric_efficiency.mean_directness
    );
    println!("observed_network_mst_source_graph_digest: {}", mst_report.source_graph_digest);
    println!("warmup_runs: {WARMUP_RUNS}");
    println!("samples: {samples}");
    println!(
        "graph_build_plus_metrics_time_ns_p50: {}",
        percentile(&durations_ns, 50)
    );
    println!(
        "graph_build_plus_metrics_time_ns_p95: {}",
        percentile(&durations_ns, 95)
    );
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!(
        "benchmark_scope: deterministic synthetic renderer-local graph; proposals only; no biology/physical units/transport/GPU claims"
    );
    Ok(())
}
