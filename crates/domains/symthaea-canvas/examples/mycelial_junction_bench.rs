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
    EmbodimentSnapshot, MycelialJunctionGraphConfig, MycelialNetworkConfig, PresenceMode,
    build_mycelial_junction_hypha_graph, build_mycelial_model, detect_anastomosis_candidates,
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
