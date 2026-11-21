// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate and fingerprint a Mycelial cognitive-topology fixture.
//!
//! Run from the repository root:
//! cargo run -p symthaea-canvas --example cognitive_topology_replay --
//!   crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_v1.json

use std::error::Error;
use std::fs;

use symthaea_canvas::{CognitiveTopology, project_cognitive_topology};

const DEFAULT_FIXTURE: &str =
    "crates/domains/symthaea-canvas/tests/fixtures/cognitive_topology_v1.json";

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_FIXTURE.to_string());
    let source = fs::read_to_string(&path)?;
    let topology: CognitiveTopology = serde_json::from_str(&source)?;
    topology.validate()?;
    let digest = topology.canonical_digest_hex()?;
    let projection = project_cognitive_topology(&topology)?;

    println!("fixture: {path}");
    println!("schema_version: {}", topology.schema_version);
    println!("session_id: {}", topology.session_id);
    println!("revision: {}", topology.revision);
    println!("last_sequence: {}", topology.last_sequence);
    println!("nodes: {}", topology.nodes.len());
    println!("edges: {}", topology.edges.len());
    println!("episodes: {}", topology.episodes.len());
    println!("semantic_digest: {digest}");
    println!("projection_schema_version: {}", projection.schema_version);
    println!("spatial_anchors: {}", projection.anchors.len());
    println!("spatial_links: {}", projection.links.len());
    println!("projection_topology_digest: {}", projection.topology_digest);
    println!("provenance: synthetic fixture; no live runtime events asserted");
    Ok(())
}
