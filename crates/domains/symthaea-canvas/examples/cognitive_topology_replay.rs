// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate, project and microbenchmark a Mycelial cognitive-topology fixture.
//!
//! Run from the repository root:
//! cargo run -p symthaea-canvas --example cognitive_topology_replay --
//!   crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_v1.json 500

use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{CognitiveLayoutConfig, CognitiveTopology, layout_cognitive_topology};

const DEFAULT_FIXTURE: &str =
    "crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_v1.json";
const WARMUP_RUNS: usize = 20;
const DEFAULT_SAMPLES: usize = 500;
const MAX_SAMPLES: usize = 100_000;

fn percentile(sorted: &[u128], percentage: usize) -> u128 {
    let index = ((sorted.len() - 1) * percentage) / 100;
    sorted[index]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_FIXTURE.to_string());
    let samples = match args.get(1) {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_SAMPLES,
    };
    let svg_output = args.get(2).cloned();
    if !(1..=MAX_SAMPLES).contains(&samples) {
        return Err(format!("sample count must be in 1..={MAX_SAMPLES}").into());
    }

    let source = fs::read_to_string(&path)?;
    let topology: CognitiveTopology = serde_json::from_str(&source)?;
    topology.validate()?;
    let semantic_digest = topology.canonical_digest_hex()?;
    let config = CognitiveLayoutConfig::default();

    for _ in 0..WARMUP_RUNS {
        let projection = layout_cognitive_topology(black_box(&topology), None, config)?;
        black_box(projection.geometry_digest_hex()?);
    }

    let mut durations_ns = Vec::with_capacity(samples);
    let mut geometry_digest = None;
    let mut final_projection = None;
    for _ in 0..samples {
        let started = Instant::now();
        let projection = layout_cognitive_topology(black_box(&topology), None, config)?;
        let digest = projection.geometry_digest_hex()?;
        durations_ns.push(started.elapsed().as_nanos());
        geometry_digest = Some(digest);
        final_projection = Some(projection);
    }
    durations_ns.sort_unstable();

    let projection = final_projection.expect("sample count is non-zero");
    let geometry_digest = geometry_digest.expect("sample count is non-zero");

    println!("fixture: {path}");
    println!("schema_version: {}", topology.schema_version);
    println!("session_id: {}", topology.session_id);
    println!("revision: {}", topology.revision);
    println!("last_sequence: {}", topology.last_sequence);
    println!("nodes: {}", topology.nodes.len());
    println!("edges: {}", topology.edges.len());
    println!("episodes: {}", topology.episodes.len());
    println!("semantic_digest: {semantic_digest}");
    println!("projection_schema_version: {}", projection.schema_version);
    println!("spatial_anchors: {}", projection.anchors.len());
    println!("spatial_links: {}", projection.links.len());
    println!("layout_algorithm: deterministic-force-v1");
    println!("layout_iterations: {}", config.iterations);
    println!("geometry_digest_1e-3: {geometry_digest}");
    println!(
        "svg_output: {}",
        svg_output.as_deref().unwrap_or("not requested")
    );
    println!("layout_warmup_runs: {WARMUP_RUNS}");
    println!("layout_samples: {samples}");
    println!("layout_time_ns_p50: {}", percentile(&durations_ns, 50));
    println!("layout_time_ns_p95: {}", percentile(&durations_ns, 95));
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!("provenance: synthetic fixture; no live runtime events asserted");
    Ok(())
}
