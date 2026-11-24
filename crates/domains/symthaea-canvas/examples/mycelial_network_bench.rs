// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CPU-only baseline for the Mycelial branching network and anastomosis search.
//!
//! Run from the repository root:
//! cargo run -p symthaea-canvas --example mycelial_network_bench -- 250
//!
//! The benchmark does not render a frame or model fungal nutrient transport.
//! All presence signals come from a dormant synthetic snapshot.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    AnastomosisKind, EmbodimentSnapshot, MycelialNetworkConfig, PresenceMode, build_mycelial_model,
    build_mycelial_network_graph, detect_anastomosis_candidates,
};

const WARMUP_RUNS: usize = 5;
const DEFAULT_SAMPLES: usize = 250;
const MAX_SAMPLES: usize = 100_000;

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

    let snapshot = EmbodimentSnapshot::default();
    let config = MycelialNetworkConfig::default();

    for mode in [
        PresenceMode::Ambient,
        PresenceMode::Cognitive,
        PresenceMode::Conversational,
    ] {
        let model = build_mycelial_model(&snapshot, mode);
        let baseline = build_mycelial_network_graph(&model, &[], config)?;
        let baseline_metrics = baseline.metrics();

        for _ in 0..WARMUP_RUNS {
            let candidates = detect_anastomosis_candidates(black_box(&model), config)?;
            black_box(candidates.len());
        }

        let mut durations_ns = Vec::with_capacity(samples);
        let mut final_candidates = Vec::new();
        for _ in 0..samples {
            let started = Instant::now();
            let candidates = detect_anastomosis_candidates(black_box(&model), config)?;
            durations_ns.push(started.elapsed().as_nanos());
            final_candidates = candidates;
        }
        durations_ns.sort_unstable();

        let tip_to_tip = final_candidates
            .iter()
            .filter(|candidate| candidate.fusion.kind == AnastomosisKind::TipToTip)
            .count();
        let tip_to_side = final_candidates
            .iter()
            .filter(|candidate| candidate.fusion.kind == AnastomosisKind::TipToSide)
            .count();

        println!("mode: {mode:?}");
        println!("model_schema_version: {}", model.schema_version);
        println!("filaments: {}", baseline_metrics.filament_count);
        println!("tree_edges: {}", baseline_metrics.parent_branch_count);
        println!("components: {}", baseline_metrics.connected_components);
        println!("cycle_rank: {}", baseline_metrics.cycle_rank);
        println!(
            "articulation_filaments: {}",
            baseline_metrics.articulation_filament_count
        );
        println!(
            "max_components_after_single_filament_failure: {}",
            baseline_metrics.max_components_after_single_filament_failure
        );
        println!("candidate_count: {}", final_candidates.len());
        println!("tip_to_tip_candidates: {tip_to_tip}");
        println!("tip_to_side_candidates: {tip_to_side}");
        println!(
            "candidate_search_time_ns_p50: {}",
            percentile(&durations_ns, 50)
        );
        println!(
            "candidate_search_time_ns_p95: {}",
            percentile(&durations_ns, 95)
        );
        println!("candidate_search_warmup_runs: {WARMUP_RUNS}");
        println!("candidate_search_samples: {samples}");
        println!("fusion_distance_limit: {}", config.max_fusion_distance);
        println!("provenance: dormant synthetic snapshot; no live cognitive activity");
        println!();
    }

    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!(
        "benchmark_scope: CPU network topology and candidate search only; no GPU/render/biology claims"
    );
    Ok(())
}
