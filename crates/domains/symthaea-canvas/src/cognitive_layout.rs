// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic spatial projection of an evidence-bearing cognitive topology.
//!
//! The projection is a renderer-neutral baseline. Node anchors are derived from
//! stable semantic IDs, so adding another node does not move existing anchors.
//! It does not infer new edges, confidence, memories or runtime activity.

use std::collections::HashMap;
use std::fmt;

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
    pub session_id: String,
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
        session_id: topology.session_id.clone(),
        topology_digest,
        anchors,
        links,
    })
}

/// Bounded parameters for deterministic, relation-aware spatial refinement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CognitiveLayoutConfig {
    pub iterations: u16,
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub ideal_edge_length: f32,
    pub anchor_strength: f32,
    pub max_step: f32,
    pub coordinate_limit: f32,
}

impl Default for CognitiveLayoutConfig {
    fn default() -> Self {
        Self {
            iterations: 32,
            repulsion_strength: 0.10,
            attraction_strength: 0.08,
            ideal_edge_length: 2.8,
            anchor_strength: 0.16,
            max_step: 0.05,
            coordinate_limit: 8.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveLayoutError {
    InvalidConfig,
    InvalidPreviousProjection,
    Topology(CognitiveTopologyError),
}

impl fmt::Display for CognitiveLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid or unbounded cognitive layout configuration"),
            Self::InvalidPreviousProjection => {
                write!(f, "previous projection has an incompatible session, schema, or anchor set")
            }
            Self::Topology(error) => write!(f, "cognitive topology validation failed: {error}"),
        }
    }
}

impl std::error::Error for CognitiveLayoutError {}

impl From<CognitiveTopologyError> for CognitiveLayoutError {
    fn from(value: CognitiveTopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Refine identity-stable anchors using a deterministic 3D force-directed pass.
///
/// Existing anchors from a previous projection are used as temporal reference
/// positions. This regularization limits mental-map disruption when a graph
/// grows or changes. Layout is derived presentation, never semantic evidence.
pub fn layout_cognitive_topology(
    topology: &CognitiveTopology,
    previous: Option<&CognitiveSpatialProjection>,
    config: CognitiveLayoutConfig,
) -> Result<CognitiveSpatialProjection, CognitiveLayoutError> {
    if config.iterations == 0
        || config.iterations > 128
        || !config.repulsion_strength.is_finite()
        || !(0.0..=4.0).contains(&config.repulsion_strength)
        || !config.attraction_strength.is_finite()
        || !(0.0..=4.0).contains(&config.attraction_strength)
        || !config.ideal_edge_length.is_finite()
        || !(0.1..=24.0).contains(&config.ideal_edge_length)
        || !config.anchor_strength.is_finite()
        || !(0.0..=4.0).contains(&config.anchor_strength)
        || !config.max_step.is_finite()
        || !(0.001..=1.0).contains(&config.max_step)
        || !config.coordinate_limit.is_finite()
        || !(1.0..=64.0).contains(&config.coordinate_limit)
    {
        return Err(CognitiveLayoutError::InvalidConfig);
    }

    let mut projection = project_cognitive_topology(topology)?;
    let mut previous_positions = HashMap::new();
    if let Some(previous) = previous {
        if previous.schema_version != COGNITIVE_PROJECTION_SCHEMA_VERSION
            || previous.session_id != topology.session_id
        {
            return Err(CognitiveLayoutError::InvalidPreviousProjection);
        }
        for anchor in &previous.anchors {
            if !anchor.position.is_finite()
                || previous_positions
                    .insert(anchor.node_id.as_str(), anchor.position)
                    .is_some()
            {
                return Err(CognitiveLayoutError::InvalidPreviousProjection);
            }
        }
    }

    let mut initial_positions = Vec::with_capacity(projection.anchors.len());
    for anchor in &mut projection.anchors {
        if let Some(position) = previous_positions.get(anchor.node_id.as_str()) {
            anchor.position = *position;
        }
        initial_positions.push(anchor.position);
    }

    let mut node_indices = HashMap::with_capacity(projection.anchors.len());
    for (index, anchor) in projection.anchors.iter().enumerate() {
        node_indices.insert(anchor.node_id.as_str(), index);
    }
    let mut edges: Vec<_> = topology.edges.iter().collect();
    edges.sort_by(|left, right| left.id.cmp(&right.id));

    let mut positions = initial_positions.clone();
    for _ in 0..config.iterations {
        let mut forces = vec![SpatialPoint3::ZERO; positions.len()];

        // Pairwise repulsion prevents unrelated anchors from collapsing.
        // Nodes are already sorted by stable ID, so accumulation order is fixed.
        for left in 0..positions.len() {
            for right in (left + 1)..positions.len() {
                let delta = positions[left].sub(positions[right]);
                let distance = delta.length();
                let direction = if distance > 1.0e-5 {
                    delta.scale(1.0 / distance)
                } else {
                    pair_direction(
                        &projection.anchors[left].node_id,
                        &projection.anchors[right].node_id,
                    )
                };
                let magnitude = (config.repulsion_strength / distance.max(0.5).powi(2))
                    .min(2.0);
                let force = direction.scale(magnitude);
                forces[left] = forces[left].add(force);
                forces[right] = forces[right].sub(force);
            }
        }

        // Every accepted relation attracts its endpoints toward a common,
        // readable edge length. Epistemic status is not converted into force.
        for edge in &edges {
            let source = *node_indices
                .get(edge.source.as_str())
                .ok_or(CognitiveLayoutError::InvalidPreviousProjection)?;
            let target = *node_indices
                .get(edge.target.as_str())
                .ok_or(CognitiveLayoutError::InvalidPreviousProjection)?;
            let delta = positions[target].sub(positions[source]);
            let distance = delta.length();
            let direction = if distance > 1.0e-5 {
                delta.scale(1.0 / distance)
            } else {
                pair_direction(&edge.source, &edge.target)
            };
            let magnitude = ((distance - config.ideal_edge_length)
                * config.attraction_strength)
                .clamp(-2.0, 2.0);
            let force = direction.scale(magnitude);
            forces[source] = forces[source].add(force);
            forces[target] = forces[target].sub(force);
        }

        // Anchor springs preserve temporal identity when previous geometry is
        // supplied. This is a layout regularizer, not a confidence signal.
        for index in 0..positions.len() {
            let restorative = initial_positions[index]
                .sub(positions[index])
                .scale(config.anchor_strength);
            let mut step = forces[index].add(restorative);
            let magnitude = step.length();
            if magnitude > config.max_step {
                step = step.scale(config.max_step / magnitude);
            }
            let next = positions[index].add(step);
            positions[index] = SpatialPoint3 {
                x: next.x.clamp(-config.coordinate_limit, config.coordinate_limit),
                y: next.y.clamp(-config.coordinate_limit, config.coordinate_limit),
                z: next.z.clamp(-config.coordinate_limit, config.coordinate_limit),
            };
        }
    }

    for (anchor, position) in projection.anchors.iter_mut().zip(positions) {
        anchor.position = position;
    }
    Ok(projection)
}

fn pair_direction(first_id: &str, second_id: &str) -> SpatialPoint3 {
    let (left, right, sign) = if first_id <= second_id {
        (first_id, second_id, 1.0)
    } else {
        (second_id, first_id, -1.0)
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(left.as_bytes());
    hasher.update(&[0]);
    hasher.update(right.as_bytes());
    let bytes = hasher.finalize();
    let bytes = bytes.as_bytes();
    let axis = |offset: usize| {
        let value = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        value as f32 / 32_768.0
    };
    SpatialPoint3 {
        x: axis(0) * sign,
        y: axis(2) * sign,
        z: axis(4) * sign,
    }
    .normalized()
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
    fn force_layout_is_deterministic_and_independent_of_input_order() {
        let topology = fixture();
        let first = layout_cognitive_topology(
            &topology,
            None,
            CognitiveLayoutConfig::default(),
        )
        .expect("valid layout");
        let mut reordered = topology;
        reordered.nodes.reverse();
        reordered.edges.reverse();
        reordered.episodes.reverse();
        let second = layout_cognitive_topology(
            &reordered,
            None,
            CognitiveLayoutConfig::default(),
        )
        .expect("valid reordered layout");
        assert_eq!(first, second);
        assert_eq!(first.topology_digest, second.topology_digest);
    }

    #[test]
    fn incremental_layout_uses_previous_positions_and_keeps_semantics_separate() {
        let topology = fixture();
        let previous = layout_cognitive_topology(
            &topology,
            None,
            CognitiveLayoutConfig::default(),
        )
        .expect("initial layout");
        let semantic_digest = topology.canonical_digest_hex().expect("valid topology");

        let mut extended = topology.clone();
        extended.revision += 1;
        extended.last_sequence += 1;
        extended.nodes.push(crate::cognitive_topology::CognitiveNode {
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
        let next = layout_cognitive_topology(
            &extended,
            Some(&previous),
            CognitiveLayoutConfig {
                iterations: 8,
                anchor_strength: 1.5,
                max_step: 0.01,
                ..CognitiveLayoutConfig::default()
            },
        )
        .expect("incremental layout");
        for old_anchor in &previous.anchors {
            let current = next
                .anchors
                .iter()
                .find(|anchor| anchor.node_id == old_anchor.node_id)
                .expect("existing identity remains present");
            let drift = current.position.sub(old_anchor.position).length();
            assert!(drift < 0.08, "excessive anchor drift: {drift}");
        }
        assert_eq!(next.session_id, previous.session_id);
        assert_ne!(next.topology_digest, previous.topology_digest);
        assert_eq!(semantic_digest, topology.canonical_digest_hex().unwrap());
    }

    #[test]
    fn invalid_layout_configuration_and_foreign_session_projection_fail_closed() {
        let topology = fixture();
        assert_eq!(
            layout_cognitive_topology(
                &topology,
                None,
                CognitiveLayoutConfig {
                    iterations: u16::MAX,
                    ..CognitiveLayoutConfig::default()
                }
            ),
            Err(CognitiveLayoutError::InvalidConfig)
        );
        let mut previous = project_cognitive_topology(&topology).expect("valid projection");
        previous.session_id = "another-session".to_string();
        assert_eq!(
            layout_cognitive_topology(
                &topology,
                Some(&previous),
                CognitiveLayoutConfig::default()
            ),
            Err(CognitiveLayoutError::InvalidPreviousProjection)
        );
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
