// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic temporal-cycle recurrence / TCN-TCR conformance benchmark.
//!
//! Run:
//! cargo run -p symthaea-canvas --example mycelial_temporal_cycle_bench -- 100
//!
//! The fixture is synthetic and explicitly opts into synthetic evidence.
//! The reported TCN/TCR values are contract-test outputs, not biological
//! measurements, transport rates or resilience results.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    MAX_TEMPORAL_CYCLE_SEARCH_STEPS, MycelialTemporalCycleConfig, TemporalEdgeEvidence,
    TemporalMycelialEdge, TemporalMycelialSequence, TemporalMycelialSnapshot,
    TemporalWindowCoverage, build_temporal_mycelial_cycle_profile,
};

const DEFAULT_SAMPLES: usize = 100;
const WARMUP_RUNS: usize = 5;
const MAX_SAMPLES: usize = 10_000;

fn edge(id: &str, source: &str, target: &str) -> TemporalMycelialEdge {
    TemporalMycelialEdge {
        id: id.to_string(),
        source_node_id: source.to_string(),
        target_node_id: target.to_string(),
        evidence: TemporalEdgeEvidence::SyntheticFixture,
    }
}

fn snapshot(
    id: &str,
    start_ns: u64,
    end_ns: u64,
    coverage: TemporalWindowCoverage,
    mut edges: Vec<TemporalMycelialEdge>,
) -> TemporalMycelialSnapshot {
    edges.sort_by(|left, right| left.id.cmp(&right.id));
    TemporalMycelialSnapshot {
        snapshot_id: id.to_string(),
        window_start_ns: start_ns,
        window_end_ns: end_ns,
        observation_scope_id: "synthetic-whole-domain-v1".to_string(),
        source_observation_digest: "a".repeat(64),
        coverage,
        node_ids: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        edges,
    }
}

fn fixture() -> TemporalMycelialSequence {
    let cycle = vec![
        edge("edge:AB", "A", "B"),
        edge("edge:AC", "A", "C"),
        edge("edge:BC", "B", "C"),
    ];
    TemporalMycelialSequence {
        schema_version: 1,
        session_id: "synthetic-mycelial-temporal-fixture-v1".to_string(),
        coordinate_frame_id: "synthetic-organism-frame".to_string(),
        calibration_digest: "c".repeat(64),
        snapshots: vec![
            snapshot(
                "window-001",
                0,
                1_000_000,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle.clone(),
            ),
            snapshot(
                "window-002",
                1_000_000,
                2_000_000,
                TemporalWindowCoverage::PartialOrOccluded,
                vec![edge("edge:AB", "A", "B")],
            ),
            snapshot(
                "window-003",
                2_000_000,
                3_000_000,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle,
            ),
        ],
    }
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

    let sequence = fixture();
    let config = MycelialTemporalCycleConfig {
        minimum_present_windows: 2,
        minimum_present_duration_ns: 2_000_000,
        minimum_persistence_fraction: 0.5,
        minimum_known_duration_fraction: 0.5,
        tcr_epsilon: 1.0e-12,
        allow_synthetic_evidence: true,
    };
    for _ in 0..WARMUP_RUNS {
        black_box(build_temporal_mycelial_cycle_profile(&sequence, config)?);
    }

    let mut durations_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        let profile = build_temporal_mycelial_cycle_profile(&sequence, config)?;
        black_box(profile.canonical_digest_hex()?);
        durations_ns.push(started.elapsed().as_nanos());
    }
    durations_ns.sort_unstable();

    let profile = build_temporal_mycelial_cycle_profile(&sequence, config)?;
    println!("benchmark: mycelial-temporal-cycle-recurrence-v1");
    println!("scope: synthetic-conformance-only");
    println!("snapshot_windows: {}", profile.snapshot_metrics.len());
    println!("observed_duration_ns: {}", profile.observed_duration_ns);
    println!("complete_duration_ns: {}", profile.complete_duration_ns);
    println!("partial_duration_ns: {}", profile.partial_duration_ns);
    println!(
        "edge_observations: {}",
        profile.edge_evidence_counts.total_edge_observations
    );
    println!(
        "synthetic_edge_observations: {}",
        profile.edge_evidence_counts.synthetic_fixture_observations
    );
    println!("distinct_cycles: {}", profile.cycles.len());
    println!("maximum_cycle_search_steps_per_window: {MAX_TEMPORAL_CYCLE_SEARCH_STEPS}");
    println!(
        "persistent_cycle_hyperedges: {}",
        profile.persistent_cycle_hyperedges.len()
    );
    for node in &profile.node_metrics {
        println!(
            "node:{} tcn_duration_ns={} tcr={:?} degree_exposure_ns={} exposure_complete={}",
            node.node_id,
            node.tcn_duration_ns,
            node.tcr,
            node.degree_exposure_ns,
            node.degree_exposure_complete
        );
    }
    println!("profile_digest: {}", profile.canonical_digest_hex()?);
    println!("warmup_runs: {WARMUP_RUNS}");
    println!("samples: {samples}");
    println!(
        "profile_build_time_ns_p50: {}",
        percentile(&durations_ns, 50)
    );
    println!(
        "profile_build_time_ns_p95: {}",
        percentile(&durations_ns, 95)
    );
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!(
        "claim_boundary: synthetic recurrence contract only; not empirical biology, transport or resilience"
    );
    Ok(())
}
