// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direct semantic-graph-to-mycelial morphology projection.
//!
//! Every cognitive node maps to one stable spatial anchor and every semantic
//! edge maps to one evidence-bearing Bézier hypha. This module never invents
//! graph edges, infers relations from geometry, or changes CognitiveTopology.
//! Decorative MycelialModel branches and anastomosis experiments are separate.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::cognitive_layout::{CognitiveLayoutError, CognitiveSpatialProjection};
use crate::cognitive_topology::{
    CognitiveEdge, CognitiveEvidence, CognitiveNodeKind, CognitiveProvenance, CognitiveRelation,
    CognitiveTopology, CognitiveTopologyError, EpistemicStatus,
};
use crate::mycelial_model::SpatialPoint3;

pub const COGNITIVE_MORPHOLOGY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CognitiveMorphologyConfig {
    /// Uniform hypha radius; not derived from confidence or epistemic status.
    pub edge_radius: f32,
    /// Maximum world-space offset used to separate parallel directed edges.
    pub max_control_offset: f32,
    /// Minimum bend offset for a single edge.
    pub min_control_offset: f32,
    /// Hard bound for all generated coordinates.
    pub coordinate_limit: f32,
}

impl Default for CognitiveMorphologyConfig {
    fn default() -> Self {
        Self {
            edge_radius: 0.035,
            max_control_offset: 2.0,
            min_control_offset: 0.25,
            coordinate_limit: 64.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveMorphologyNode {
    pub node_id: String,
    pub kind: CognitiveNodeKind,
    pub label: String,
    pub status: EpistemicStatus,
    pub position: SpatialPoint3,
    pub evidence: CognitiveEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveHypha {
    pub edge_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub relation: CognitiveRelation,
    pub status: EpistemicStatus,
    pub start: SpatialPoint3,
    pub control_a: SpatialPoint3,
    pub control_b: SpatialPoint3,
    pub end: SpatialPoint3,
    pub radius: f32,
    pub evidence: CognitiveEvidence,
}

/// Renderer-neutral organism projected from exactly one CognitiveTopology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveMorphologyProjection {
    pub schema_version: u16,
    pub session_id: String,
    pub topology_digest: String,
    pub nodes: Vec<CognitiveMorphologyNode>,
    pub hyphae: Vec<CognitiveHypha>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveMorphologyError {
    InvalidConfig,
    ProjectionMismatch,
    DuplicateAnchor(String),
    MissingAnchor(String),
    InvalidGeometry(String),
    Topology(CognitiveTopologyError),
    Layout(CognitiveLayoutError),
}

impl fmt::Display for CognitiveMorphologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid cognitive morphology configuration"),
            Self::ProjectionMismatch => {
                write!(
                    f,
                    "spatial projection does not exactly match the semantic topology"
                )
            }
            Self::DuplicateAnchor(id) => write!(f, "duplicate spatial anchor for node {id}"),
            Self::MissingAnchor(id) => write!(f, "missing spatial anchor for node {id}"),
            Self::InvalidGeometry(id) => {
                write!(
                    f,
                    "generated cognitive morphology geometry is invalid for {id}"
                )
            }
            Self::Topology(error) => write!(f, "cognitive topology validation failed: {error}"),
            Self::Layout(error) => write!(f, "cognitive spatial projection failed: {error}"),
        }
    }
}

impl Error for CognitiveMorphologyError {}

impl From<CognitiveTopologyError> for CognitiveMorphologyError {
    fn from(value: CognitiveTopologyError) -> Self {
        Self::Topology(value)
    }
}

impl From<CognitiveLayoutError> for CognitiveMorphologyError {
    fn from(value: CognitiveLayoutError) -> Self {
        Self::Layout(value)
    }
}

/// Build an organism-like projection from the validated semantic graph and its
/// exact spatial projection. All semantic IDs and source evidence are retained.
pub fn build_cognitive_morphology(
    topology: &CognitiveTopology,
    spatial: &CognitiveSpatialProjection,
    config: CognitiveMorphologyConfig,
) -> Result<CognitiveMorphologyProjection, CognitiveMorphologyError> {
    topology.validate()?;
    validate_config(config)?;

    let topology_digest = topology.canonical_digest_hex()?;
    if spatial.schema_version != crate::cognitive_layout::COGNITIVE_PROJECTION_SCHEMA_VERSION
        || spatial.session_id != topology.session_id
        || spatial.topology_digest != topology_digest
        || spatial.anchors.len() != topology.nodes.len()
        || spatial.links.len() != topology.edges.len()
    {
        return Err(CognitiveMorphologyError::ProjectionMismatch);
    }
    spatial.geometry_digest_hex()?;

    let mut link_by_id = HashMap::with_capacity(spatial.links.len());
    for link in &spatial.links {
        if link_by_id.insert(link.edge_id.as_str(), link).is_some() {
            return Err(CognitiveMorphologyError::ProjectionMismatch);
        }
    }
    for edge in &topology.edges {
        let link = link_by_id
            .get(edge.id.as_str())
            .ok_or(CognitiveMorphologyError::ProjectionMismatch)?;
        if link.source_node_id != edge.source
            || link.target_node_id != edge.target
            || link.relation != edge.relation
            || link.status != edge.status
            || link.source_event_id != edge.evidence.event_id
            || link.source_sequence != edge.evidence.sequence
            || link.provenance != edge.evidence.provenance
        {
            return Err(CognitiveMorphologyError::ProjectionMismatch);
        }
    }

    let mut anchor_by_id = HashMap::with_capacity(spatial.anchors.len());
    for anchor in &spatial.anchors {
        if anchor_by_id
            .insert(anchor.node_id.as_str(), anchor)
            .is_some()
        {
            return Err(CognitiveMorphologyError::DuplicateAnchor(
                anchor.node_id.clone(),
            ));
        }
    }

    let mut nodes = Vec::with_capacity(topology.nodes.len());
    let mut sorted_nodes: Vec<_> = topology.nodes.iter().collect();
    sorted_nodes.sort_by(|left, right| left.id.cmp(&right.id));
    for node in sorted_nodes {
        let anchor = anchor_by_id
            .get(node.id.as_str())
            .ok_or_else(|| CognitiveMorphologyError::MissingAnchor(node.id.clone()))?;
        if anchor.node_kind != node.kind
            || anchor.label != node.label
            || anchor.status != node.status
            || anchor.source_event_id != node.evidence.event_id
            || anchor.source_producer != node.evidence.producer
            || anchor.source_sequence != node.evidence.sequence
            || anchor.provenance != node.evidence.provenance
            || !within_bounds(anchor.position, config.coordinate_limit)
        {
            return Err(CognitiveMorphologyError::ProjectionMismatch);
        }
        nodes.push(CognitiveMorphologyNode {
            node_id: node.id.clone(),
            kind: node.kind,
            label: node.label.clone(),
            status: node.status,
            position: anchor.position,
            evidence: node.evidence.clone(),
        });
    }

    let mut edges_by_pair: BTreeMap<(String, String), Vec<&CognitiveEdge>> = BTreeMap::new();
    for edge in &topology.edges {
        edges_by_pair
            .entry((edge.source.clone(), edge.target.clone()))
            .or_default()
            .push(edge);
    }

    let mut hyphae = Vec::with_capacity(topology.edges.len());
    for ((_source_id, _target_id), mut edges) in edges_by_pair {
        edges.sort_by(|left, right| left.id.cmp(&right.id));
        let edge_count = edges.len();
        for (lane_index, edge) in edges.into_iter().enumerate() {
            let source = anchor_by_id
                .get(edge.source.as_str())
                .ok_or_else(|| CognitiveMorphologyError::MissingAnchor(edge.source.clone()))?;
            let target = anchor_by_id
                .get(edge.target.as_str())
                .ok_or_else(|| CognitiveMorphologyError::MissingAnchor(edge.target.clone()))?;
            let (control_a, control_b) = edge_controls(
                edge,
                source.position,
                target.position,
                lane_index,
                edge_count,
                config,
            );
            if !within_bounds(control_a, config.coordinate_limit)
                || !within_bounds(control_b, config.coordinate_limit)
            {
                return Err(CognitiveMorphologyError::InvalidGeometry(edge.id.clone()));
            }
            hyphae.push(CognitiveHypha {
                edge_id: edge.id.clone(),
                source_node_id: edge.source.clone(),
                target_node_id: edge.target.clone(),
                relation: edge.relation,
                status: edge.status,
                start: source.position,
                control_a,
                control_b,
                end: target.position,
                radius: config.edge_radius,
                evidence: edge.evidence.clone(),
            });
        }
    }
    hyphae.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));

    Ok(CognitiveMorphologyProjection {
        schema_version: COGNITIVE_MORPHOLOGY_SCHEMA_VERSION,
        session_id: topology.session_id.clone(),
        topology_digest,
        nodes,
        hyphae,
    })
}

fn validate_config(config: CognitiveMorphologyConfig) -> Result<(), CognitiveMorphologyError> {
    if !config.edge_radius.is_finite()
        || !(0.005..=0.5).contains(&config.edge_radius)
        || !config.max_control_offset.is_finite()
        || !(0.05..=16.0).contains(&config.max_control_offset)
        || !config.min_control_offset.is_finite()
        || !(0.0..=config.max_control_offset).contains(&config.min_control_offset)
        || !config.coordinate_limit.is_finite()
        || !(1.0..=128.0).contains(&config.coordinate_limit)
    {
        return Err(CognitiveMorphologyError::InvalidConfig);
    }
    Ok(())
}

fn edge_controls(
    edge: &CognitiveEdge,
    start: SpatialPoint3,
    end: SpatialPoint3,
    lane_index: usize,
    edge_count: usize,
    config: CognitiveMorphologyConfig,
) -> (SpatialPoint3, SpatialPoint3) {
    let chord = end.sub(start);
    let length = chord.length();
    let direction = if length > 1.0e-5 {
        chord.scale(1.0 / length)
    } else {
        stable_direction(&edge.id)
    };
    let reference = if direction.z.abs() < 0.8 {
        SpatialPoint3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    } else {
        SpatialPoint3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    };
    let perpendicular = direction.cross(reference).normalized();

    let digest = blake3::hash(edge.id.as_bytes());
    let bytes = digest.as_bytes();
    let sign = if bytes[0] & 1 == 0 { 1.0 } else { -1.0 };
    let random_fraction = f32::from(bytes[1]) / 255.0;
    let base_offset = config.min_control_offset
        + (config.max_control_offset - config.min_control_offset) * random_fraction;
    let lane = lane_index as f32 - (edge_count.saturating_sub(1) as f32 * 0.5);
    let lane_offset = lane * (config.min_control_offset.max(0.05) * 1.5);
    let offset = (sign * base_offset + lane_offset)
        .clamp(-config.max_control_offset, config.max_control_offset);

    let effective_offset = if length > 0.0 {
        offset.min(length * 0.45).max(-length * 0.45)
    } else {
        offset
    };
    let bend = perpendicular.scale(effective_offset);
    let control_a = start.add(chord.scale(1.0 / 3.0)).add(bend);
    let control_b = start.add(chord.scale(2.0 / 3.0)).add(bend);
    (control_a, control_b)
}

fn stable_direction(id: &str) -> SpatialPoint3 {
    let digest = blake3::hash(id.as_bytes());
    let bytes = digest.as_bytes();
    let axis = |offset: usize| f32::from(bytes[offset]) / 127.5 - 1.0;
    SpatialPoint3 {
        x: axis(0),
        y: axis(1),
        z: axis(2),
    }
    .normalized()
}

fn within_bounds(point: SpatialPoint3, limit: f32) -> bool {
    point.is_finite() && point.x.abs() <= limit && point.y.abs() <= limit && point.z.abs() <= limit
}

impl CognitiveMorphologyProjection {
    /// Digest exact semantic identity plus quantized generated curves. This
    /// excludes camera, renderer, lighting, colour and animation state.
    pub fn geometry_digest_hex(&self) -> Result<String, CognitiveMorphologyError> {
        if self.schema_version != COGNITIVE_MORPHOLOGY_SCHEMA_VERSION
            || self.session_id.is_empty()
            || self.topology_digest.len() != 64
            || !self
                .topology_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CognitiveMorphologyError::ProjectionMismatch);
        }

        let mut node_positions = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if node.node_id.is_empty()
                || !within_bounds(node.position, 128.0)
                || node_positions
                    .insert(node.node_id.as_str(), node.position)
                    .is_some()
            {
                return Err(CognitiveMorphologyError::ProjectionMismatch);
            }
        }

        let mut edge_ids = HashSet::with_capacity(self.hyphae.len());
        for hypha in &self.hyphae {
            let Some(source) = node_positions.get(hypha.source_node_id.as_str()) else {
                return Err(CognitiveMorphologyError::ProjectionMismatch);
            };
            let Some(target) = node_positions.get(hypha.target_node_id.as_str()) else {
                return Err(CognitiveMorphologyError::ProjectionMismatch);
            };
            if hypha.edge_id.is_empty()
                || !edge_ids.insert(hypha.edge_id.as_str())
                || *source != hypha.start
                || *target != hypha.end
                || !within_bounds(hypha.control_a, 128.0)
                || !within_bounds(hypha.control_b, 128.0)
                || !hypha.radius.is_finite()
                || !(0.005..=0.5).contains(&hypha.radius)
            {
                return Err(CognitiveMorphologyError::ProjectionMismatch);
            }
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-cognitive-morphology-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.session_id);
        hash_string(&mut hasher, &self.topology_digest);

        let mut nodes: Vec<_> = self.nodes.iter().collect();
        nodes.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        hasher.update(&(nodes.len() as u64).to_le_bytes());
        for node in nodes {
            hash_string(&mut hasher, &node.node_id);
            hash_point(&mut hasher, node.position)?;
            hash_string(&mut hasher, status_name(node.status));
            hash_evidence(&mut hasher, &node.evidence);
        }

        let mut hyphae: Vec<_> = self.hyphae.iter().collect();
        hyphae.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
        hasher.update(&(hyphae.len() as u64).to_le_bytes());
        for hypha in hyphae {
            hash_string(&mut hasher, &hypha.edge_id);
            hash_string(&mut hasher, &hypha.source_node_id);
            hash_string(&mut hasher, &hypha.target_node_id);
            hash_string(&mut hasher, relation_name(hypha.relation));
            hash_string(&mut hasher, status_name(hypha.status));
            hash_point(&mut hasher, hypha.start)?;
            hash_point(&mut hasher, hypha.control_a)?;
            hash_point(&mut hasher, hypha.control_b)?;
            hash_point(&mut hasher, hypha.end)?;
            hash_quantized(&mut hasher, hypha.radius)?;
            hash_evidence(&mut hasher, &hypha.evidence);
        }

        let mut output = String::with_capacity(64);
        for byte in hasher.finalize().as_bytes() {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        Ok(output)
    }
}

fn hash_point(
    hasher: &mut blake3::Hasher,
    point: SpatialPoint3,
) -> Result<(), CognitiveMorphologyError> {
    for coordinate in [point.x, point.y, point.z] {
        hash_quantized(hasher, coordinate)?;
    }
    Ok(())
}

fn hash_quantized(hasher: &mut blake3::Hasher, value: f32) -> Result<(), CognitiveMorphologyError> {
    if !value.is_finite() || value.abs() > 128.0 {
        return Err(CognitiveMorphologyError::ProjectionMismatch);
    }
    let quantized = (f64::from(value) * 1_000.0).round() as i64;
    hasher.update(&quantized.to_le_bytes());
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_evidence(hasher: &mut blake3::Hasher, evidence: &CognitiveEvidence) {
    hash_string(hasher, &evidence.event_id);
    hash_string(hasher, &evidence.producer);
    hasher.update(&evidence.sequence.to_le_bytes());
    hasher.update(&evidence.monotonic_ns.to_le_bytes());
    hasher.update(&[match evidence.provenance {
        CognitiveProvenance::RuntimeEvent => 0,
        CognitiveProvenance::RuntimeDerived => 1,
        CognitiveProvenance::SyntheticFixture => 2,
    }]);
    if let Some(digest) = &evidence.trace_digest {
        hasher.update(&[1]);
        hash_string(hasher, digest);
    } else {
        hasher.update(&[0]);
    }
}

fn status_name(value: EpistemicStatus) -> &'static str {
    match value {
        EpistemicStatus::Observed => "observed",
        EpistemicStatus::Derived => "derived",
        EpistemicStatus::Candidate => "candidate",
        EpistemicStatus::Resolved => "resolved",
        EpistemicStatus::Rejected => "rejected",
        EpistemicStatus::Stale => "stale",
        EpistemicStatus::Unavailable => "unavailable",
    }
}

fn relation_name(value: CognitiveRelation) -> &'static str {
    match value {
        CognitiveRelation::RetrievedFrom => "retrieved_from",
        CognitiveRelation::AssociatedWith => "associated_with",
        CognitiveRelation::Supports => "supports",
        CognitiveRelation::Contradicts => "contradicts",
        CognitiveRelation::Predicts => "predicts",
        CognitiveRelation::DependsOn => "depends_on",
        CognitiveRelation::ExecutedBy => "executed_by",
        CognitiveRelation::CandidateRelation => "candidate_relation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_layout::{CognitiveLayoutConfig, layout_cognitive_topology};

    fn fixture_topology() -> CognitiveTopology {
        serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
            .expect("checked-in topology fixture")
    }

    fn fixture_projection() -> (CognitiveTopology, CognitiveSpatialProjection) {
        let topology = fixture_topology();
        let spatial = layout_cognitive_topology(&topology, None, CognitiveLayoutConfig::default())
            .expect("fixture layout");
        (topology, spatial)
    }

    #[test]
    fn each_semantic_node_and_edge_maps_to_exactly_one_morphology_element() {
        let (topology, spatial) = fixture_projection();
        let morphology =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .expect("morphology projection");
        assert_eq!(morphology.nodes.len(), topology.nodes.len());
        assert_eq!(morphology.hyphae.len(), topology.edges.len());
        assert_eq!(
            morphology
                .nodes
                .iter()
                .map(|node| node.node_id.as_str())
                .collect::<Vec<_>>(),
            spatial
                .anchors
                .iter()
                .map(|anchor| anchor.node_id.as_str())
                .collect::<Vec<_>>()
        );
        let mut semantic_edge_ids: Vec<_> =
            topology.edges.iter().map(|edge| edge.id.as_str()).collect();
        semantic_edge_ids.sort_unstable();
        assert_eq!(
            morphology
                .hyphae
                .iter()
                .map(|hypha| hypha.edge_id.as_str())
                .collect::<Vec<_>>(),
            semantic_edge_ids
        );
        assert!(
            morphology.hyphae.iter().all(|hypha| {
                hypha.evidence.provenance == CognitiveProvenance::SyntheticFixture
            })
        );
    }

    #[test]
    fn candidate_prediction_and_relation_status_are_preserved() {
        let (topology, spatial) = fixture_projection();
        let morphology =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .expect("morphology projection");
        let prediction = morphology
            .hyphae
            .iter()
            .find(|hypha| hypha.relation == CognitiveRelation::Predicts)
            .expect("prediction hypha");
        assert_eq!(prediction.status, EpistemicStatus::Candidate);
        assert_eq!(prediction.source_node_id, "concept:mycelium".to_string());
        assert_eq!(
            prediction.target_node_id,
            "prediction:resilience".to_string()
        );
    }

    #[test]
    fn parallel_semantic_edges_get_distinct_deterministic_curves() {
        let mut topology = fixture_topology();
        topology.revision += 1;
        topology.last_sequence += 1;
        topology.edges.push(CognitiveEdge {
            id: "edge:association-parallel".to_string(),
            source: "concept:soil".to_string(),
            target: "concept:mycelium".to_string(),
            relation: CognitiveRelation::Supports,
            status: EpistemicStatus::Derived,
            evidence: CognitiveEvidence {
                event_id: "event:parallel".to_string(),
                producer: "synthetic-fixture".to_string(),
                sequence: 4,
                monotonic_ns: 4_000,
                provenance: CognitiveProvenance::SyntheticFixture,
                trace_digest: None,
            },
            activation: None,
        });
        let spatial = layout_cognitive_topology(&topology, None, CognitiveLayoutConfig::default())
            .expect("layout with parallel edge");
        let morphology =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .expect("parallel morphology");
        let first = morphology
            .hyphae
            .iter()
            .find(|hypha| hypha.edge_id == "edge:association-a")
            .unwrap();
        let second = morphology
            .hyphae
            .iter()
            .find(|hypha| hypha.edge_id == "edge:association-parallel")
            .unwrap();
        assert_ne!(first.control_a, second.control_a);
        assert_ne!(first.control_b, second.control_b);
    }

    #[test]
    fn semantic_digest_is_unchanged_by_morphology_and_geometry_digest_is_stable() {
        let (topology, spatial) = fixture_projection();
        let semantic_before = topology.canonical_digest_hex().unwrap();
        let first =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .unwrap();
        let second =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .unwrap();
        assert_eq!(
            first.geometry_digest_hex().unwrap(),
            second.geometry_digest_hex().unwrap()
        );
        assert_eq!(topology.canonical_digest_hex().unwrap(), semantic_before);
    }

    #[test]
    fn stale_or_foreign_spatial_projection_is_rejected() {
        let (topology, mut spatial) = fixture_projection();
        spatial.topology_digest = "0".repeat(64);
        assert_eq!(
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default()),
            Err(CognitiveMorphologyError::ProjectionMismatch)
        );
    }

    #[test]
    fn semantically_mismatched_spatial_link_is_rejected() {
        let (topology, mut spatial) = fixture_projection();
        let existing_node = topology.nodes[0].id.clone();
        spatial.links[0].target_node_id = existing_node;
        assert_eq!(
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default()),
            Err(CognitiveMorphologyError::ProjectionMismatch)
        );
    }

    #[test]
    fn malformed_deserialized_morphology_with_dangling_hypha_is_rejected() {
        let (topology, spatial) = fixture_projection();
        let mut morphology =
            build_cognitive_morphology(&topology, &spatial, CognitiveMorphologyConfig::default())
                .unwrap();
        morphology.hyphae[0].target_node_id = "missing-node".to_string();
        assert_eq!(
            morphology.geometry_digest_hex(),
            Err(CognitiveMorphologyError::ProjectionMismatch)
        );
    }

    #[test]
    fn invalid_config_and_out_of_bounds_layout_are_rejected() {
        let (topology, spatial) = fixture_projection();
        assert_eq!(
            build_cognitive_morphology(
                &topology,
                &spatial,
                CognitiveMorphologyConfig {
                    edge_radius: f32::NAN,
                    ..CognitiveMorphologyConfig::default()
                }
            ),
            Err(CognitiveMorphologyError::InvalidConfig)
        );
        let limited = CognitiveMorphologyConfig {
            coordinate_limit: 1.0,
            ..CognitiveMorphologyConfig::default()
        };
        assert!(matches!(
            build_cognitive_morphology(&topology, &spatial, limited),
            Err(CognitiveMorphologyError::ProjectionMismatch)
                | Err(CognitiveMorphologyError::InvalidGeometry(_))
        ));
    }
}
