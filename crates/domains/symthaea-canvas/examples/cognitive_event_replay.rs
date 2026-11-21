// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay a versioned cognitive event log into a topology, layout and SVG.
//!
//! Run from the repository root:
//! cargo run -p symthaea-canvas --example cognitive_event_replay --
//!   crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_events_v1.json
//!
//! Optional arguments: event fixture path, SVG output path, replay sample count.

use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    COGNITIVE_TOPOLOGY_SCHEMA_VERSION, CognitiveLayoutConfig, CognitiveMorphologyConfig,
    CognitiveSvgOptions, CognitiveTopology, CognitiveTopologyEvent, CognitiveTopologyReducer,
    build_cognitive_morphology, layout_cognitive_topology, render_cognitive_morphology_svg,
};

const DEFAULT_EVENTS: &str =
    "crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_events_v1.json";
const DEFAULT_SAMPLES: usize = 100;
const WARMUP_RUNS: usize = 5;
const MAX_SAMPLES: usize = 10_000;

fn percentile(sorted: &[u128], percentage: usize) -> u128 {
    sorted[((sorted.len() - 1) * percentage) / 100]
}

fn empty_base(session_id: String) -> CognitiveTopology {
    CognitiveTopology {
        schema_version: COGNITIVE_TOPOLOGY_SCHEMA_VERSION,
        session_id,
        revision: 0,
        last_sequence: 0,
        nodes: Vec::new(),
        edges: Vec::new(),
        episodes: Vec::new(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let event_path = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_EVENTS.to_string());
    let svg_path = args.get(1).cloned();
    let samples = match args.get(2) {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_SAMPLES,
    };
    if !(1..=MAX_SAMPLES).contains(&samples) {
        return Err(format!("sample count must be in 1..={MAX_SAMPLES}").into());
    }

    let source = fs::read_to_string(&event_path)?;
    let events: Vec<CognitiveTopologyEvent> = serde_json::from_str(&source)?;
    let session_id = events
        .first()
        .ok_or("event log is empty")?
        .session_id
        .clone();
    if events.iter().any(|event| event.session_id != session_id) {
        return Err("event log mixes sessions".into());
    }
    let base = empty_base(session_id.clone());

    for _ in 0..WARMUP_RUNS {
        let reducer =
            CognitiveTopologyReducer::replay(black_box(base.clone()), black_box(&events))?;
        black_box(reducer.snapshot_digest_hex()?);
        black_box(reducer.event_log_digest_hex()?);
    }

    let mut replay_durations = Vec::with_capacity(samples);
    let mut final_reducer = None;
    for _ in 0..samples {
        let started = Instant::now();
        let reducer =
            CognitiveTopologyReducer::replay(black_box(base.clone()), black_box(&events))?;
        replay_durations.push(started.elapsed().as_nanos());
        final_reducer = Some(reducer);
    }
    replay_durations.sort_unstable();

    let reducer = final_reducer.expect("sample count is non-zero");
    let snapshot = reducer.snapshot();
    let spatial = layout_cognitive_topology(snapshot, None, CognitiveLayoutConfig::default())?;
    let morphology =
        build_cognitive_morphology(snapshot, &spatial, CognitiveMorphologyConfig::default())?;
    let semantic_digest = reducer.snapshot_digest_hex()?;
    let event_log_digest = reducer.event_log_digest_hex()?;
    let replay_verifier =
        CognitiveTopologyReducer::replay(black_box(base.clone()), black_box(&events))?;
    let replay_trace_agreement =
        replay_verifier.event_log_digest_hex()? == event_log_digest;
    let replay_projection_agreement =
        replay_verifier.snapshot_digest_hex()? == semantic_digest;
    if !replay_trace_agreement || !replay_projection_agreement {
        return Err("independent projection replay diverged".into());
    }

    let replay_assurance_receipt = serde_json::json!({
        "schema_version": 1,
        "claim_scope": "same_binary_same_process_projection_replay",
        "input": {
            "session_id": session_id,
            "accepted_event_count": reducer.events().len(),
            "event_log_digest": event_log_digest,
            "semantic_snapshot_digest": semantic_digest,
        },
        "assertions": {
            "byte_identity": {
                "status": "not_assessed",
                "reason": "source artifact bytes are not compared by this CLI"
            },
            "trace_identity": {
                "status": if replay_trace_agreement { "established" } else { "failed" },
                "scope": "accepted event log digest agreement",
                "verifier": "independent reducer replay in same process"
            },
            "projection_equivalence": {
                "status": if replay_projection_agreement { "established" } else { "failed" },
                "scope": "canonical semantic snapshot digest agreement",
                "verifier": "independent reducer replay in same process"
            },
            "path_identity": {
                "status": "not_assessed",
                "reason": "model and tool path receipts are not part of the graph event fixture"
            },
            "decision_equivalence": {
                "status": "not_assessed",
                "reason": "no task-specific decision equivalence oracle is supplied"
            },
            "execution_replay": {
                "status": "not_assessed",
                "reason": "no recorded model/tool execution receipts or live-call guard are supplied"
            },
            "environment_equivalence": {
                "status": "not_assessed",
                "reason": "graph projection forks do not model or reconcile external effects"
            }
        }
    });
    let geometry_digest = morphology.geometry_digest_hex()?;

    if let Some(path) = &svg_path {
        let svg = render_cognitive_morphology_svg(&morphology, CognitiveSvgOptions::default())?;
        fs::write(path, svg)?;
    }

    let synthetic = events.iter().all(|event| {
        matches!(
            event.evidence.provenance,
            symthaea_canvas::CognitiveProvenance::SyntheticFixture
        )
    });
    println!("event_fixture: {event_path}");
    println!("session_id: {session_id}");
    println!("event_schema_version: {}", events[0].schema_version);
    println!("events_replayed: {}", reducer.events().len());
    println!("nodes: {}", snapshot.nodes.len());
    println!("edges: {}", snapshot.edges.len());
    println!("episodes: {}", snapshot.episodes.len());
    println!("last_sequence: {}", snapshot.last_sequence);
    println!("semantic_snapshot_digest: {semantic_digest}");
    println!("event_log_digest: {event_log_digest}");
    println!("replay_assurance_json: {}", serde_json::to_string(&replay_assurance_receipt)?);
    println!("replay_assurance_schema_version: 1");
    println!("replay_assurance_scope: same-binary same-process projection replay");
    println!("replay_assertion.byte_identity: not_assessed");
    println!(
        "replay_assertion.trace_identity: {}",
        if replay_trace_agreement { "established_for_accepted_log" } else { "failed" }
    );
    println!(
        "replay_assertion.projection: {}",
        if replay_projection_agreement { "established_in_process" } else { "failed" }
    );
    println!("replay_assertion.path_identity: not_assessed");
    println!("replay_assertion.decision_equivalence: not_assessed");
    println!("replay_assertion.execution_replay: not_assessed_no_effect_receipts");
    println!("replay_assertion.external_environment: not_assessed");
    println!("layout_algorithm: deterministic-force-v1");
    println!("morphology_algorithm: cognitive-bezier-hypha-v1");
    println!("morphology_nodes: {}", morphology.nodes.len());
    println!("morphology_hyphae: {}", morphology.hyphae.len());
    println!("geometry_digest_1e-3: {geometry_digest}");
    println!(
        "svg_output: {}",
        svg_path.as_deref().unwrap_or("not requested")
    );
    println!("replay_warmup_runs: {WARMUP_RUNS}");
    println!("replay_samples: {samples}");
    println!("replay_time_ns_p50: {}", percentile(&replay_durations, 50));
    println!("replay_time_ns_p95: {}", percentile(&replay_durations, 95));
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!(
        "provenance: {}",
        if synthetic {
            "synthetic fixture; not live cognitive telemetry"
        } else {
            "runtime event log"
        }
    );
    Ok(())
}
