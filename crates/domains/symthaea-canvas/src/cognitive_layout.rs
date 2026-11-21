// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic spatial projection of an evidence-bearing cognitive topology.
//!
//! The projection is a renderer-neutral baseline. Node anchors are derived from
//! stable semantic IDs, so adding another node does not move existing anchors.
//! It does not infer new edges, confidence, memories or runtime activity.

use serde::{Deserialize, Serialize};

use crate::cognitive_topology::{
    CognitiveProvenance, CognitiveRelation, CognitiveTopology, CognitiveTopologyError,
    EpistemicStatus,
};
use crate::mycelial_model::SpatialPoint3;

pub const COGNITIVE_PROJECTION_SCHEMA_VERSION: u16 = 1;
const ANCHOR_SCALE: f32 = 6.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveAnchor {
    pub node_id: String,
    pub position: SpatialPoint3,
    pub source_event_id: String,
    pub source_sequence: u64,
    pub provenance: CognitiveProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveSpatialLink {
    pub edge_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub relation: CognitiveRelation,
    pub status: EpistemicStatus,
    pub source_event_id: String,
    pub source_sequence: u64,
    pub provenance: CognitiveProvenance,
}

/// A presentation-neutral projection. Camera, colour, lighting, animation and
/// LOD are intentionally absent and therefore cannot change semantic identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveSpatialProjection {
    pub schema_version: u16,
    pub topology_digest: String,
    pub anchors: Vec<CognitiveAnchor>,
    pub links: Vec<CognitiveSpatialLink>,
}

/// Project a validated topology to stable renderer-neutral anchors and links.
/// Coordinates are a hash-based identity baseline, not a relation-aware layout.
pub fn project_cognitive_topology(
    topology: &CognitiveTopology,
) -> Result<CognitiveSpatialProjection, CognitiveTopologyError> {
    let topology_digest = topology.canonical_digest_hex()?;

    let mut nodes: Vec<_> = topology.nodes.iter().collect();
    nodes.sort_by(|left, right| left.id.cmp(&right.id));
    let anchors = nodes
        .into_iter()
        .map(|node| CognitiveAnchor {
            node_id: node.id.clone(),
            position: stable_anchor(&node.id),
            source_event_id: node.evidence.event_id.clone(),
            source_sequence: node.evidence.sequence,
            provenance: node.evidence.provenance,
        })
        .collect();

    let mut edges: Vec<_> = topology.edges.iter().collect();
    edges.sort_by(|left, right| left.id.cmp(&right.id));
    let links = edges
        .into_iter()
        .map(|edge| CognitiveSpatialLink {
            edge_id: edge.id.clone(),
            source_node_id: edge.source.clone(),
            target_node_id: edge.target.clone(),
            relation: edge.relation,
            status: edge.status,
            source_event_id: edge.evidence.event_id.clone(),
            source_sequence: edge.evidence.sequence,
            provenance: edge.evidence.provenance,
        })
        .collect();

    Ok(CognitiveSpatialProjection {
        schema_version: COGNITIVE_PROJECTION_SCHEMA_VERSION,
        topology_digest,
        anchors,
        links,
    })
}

/// Stable ID-to-anchor mapping. It is independent of input ordering and graph
/// size, so appending a node cannot reposition an existing semantic entity.
fn stable_anchor(node_id: &str) -> SpatialPoint3 {
    let digest = blake3::hash(node_id.as_bytes());
    let bytes = digest.as_bytes();
    let axis = |offset: usize| {
        let value = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        (value as f32 / 32_768.0) * ANCHOR_SCALE
    };
    SpatialPoint3 {
        x: axis(0),
        y: axis(2),
        z: axis(4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> CognitiveTopology {
        serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
            .expect("checked-in cognitive topology fixture")
    }

    #[test]
    fn projection_is_deterministic_and_preserves_evidence_references() {
        let topology = fixture();
        let first = project_cognitive_topology(&topology).expect("valid topology");
        let second = project_cognitive_topology(&topology).expect("valid topology");
        assert_eq!(first, second);
        assert_eq!(first.anchors.len(), topology.nodes.len());
        assert_eq!(first.links.len(), topology.edges.len());
        for anchor in &first.anchors {
            let node = topology
                .nodes
                .iter()
                .find(|node| node.id == anchor.node_id)
                .unwrap();
            assert_eq!(anchor.source_event_id, node.evidence.event_id);
            assert_eq!(anchor.source_sequence, node.evidence.sequence);
            assert_eq!(anchor.provenance, node.evidence.provenance);
            assert!(anchor.position.is_finite());
        }
        for link in &first.links {
            let edge = topology
                .edges
                .iter()
                .find(|edge| edge.id == link.edge_id)
                .unwrap();
            assert_eq!(link.source_event_id, edge.evidence.event_id);
            assert_eq!(link.provenance, edge.evidence.provenance);
        }
    }

    #[test]
    fn adding_a_node_does_not_move_existing_anchors() {
        let topology = fixture();
        let before = project_cognitive_topology(&topology).expect("valid topology");
        let mut extended = topology;
        extended.last_sequence += 1;
        extended.revision += 1;
        extended
            .nodes
            .push(crate::cognitive_topology::CognitiveNode {
                id: "concept:additional".to_string(),
                kind: crate::cognitive_topology::CognitiveNodeKind::Concept,
                label: "Additional concept".to_string(),
                status: EpistemicStatus::Observed,
                evidence: crate::cognitive_topology::CognitiveEvidence {
                    event_id: "event:additional".to_string(),
                    producer: "synthetic-fixture".to_string(),
                    sequence: 4,
                    monotonic_ns: 4_000,
                    provenance: CognitiveProvenance::SyntheticFixture,
                    trace_digest: None,
                },
                activation: None,
            });
        let after = project_cognitive_topology(&extended).expect("valid extended topology");
        for anchor in &before.anchors {
            let matching = after
                .anchors
                .iter()
                .find(|value| value.node_id == anchor.node_id);
            assert_eq!(matching.map(|value| value.position), Some(anchor.position));
        }
    }

    #[test]
    fn candidate_prediction_links_remain_candidates() {
        let topology = fixture();
        let projection = project_cognitive_topology(&topology).expect("valid topology");
        let prediction = projection
            .links
            .iter()
            .find(|link| link.relation == CognitiveRelation::Predicts)
            .expect("prediction link");
        assert_eq!(prediction.status, EpistemicStatus::Candidate);
        assert_eq!(prediction.provenance, CognitiveProvenance::SyntheticFixture);
    }

    #[test]
    fn invalid_topology_is_rejected_before_projection() {
        let mut topology = fixture();
        topology.edges[0].target = "missing-node".to_string();
        assert_eq!(
            project_cognitive_topology(&topology),
            Err(CognitiveTopologyError::DanglingEdgeEndpoint(
                "missing-node".to_string()
            ))
        );
    }
}
