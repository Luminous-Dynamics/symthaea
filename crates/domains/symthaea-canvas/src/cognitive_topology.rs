// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing cognitive topology, independent of any renderer or layout.
//!
//! This module stores semantic entities, relationships and episodes. It does not
//! generate geometry, infer cognition, or authorize runtime actions. A renderer
//! may project this topology, but presentation-only changes must not mutate it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

pub const COGNITIVE_TOPOLOGY_SCHEMA_VERSION: u16 = 1;
pub const MAX_COGNITIVE_NODES: usize = 256;
pub const MAX_COGNITIVE_EDGES: usize = 1_024;
pub const MAX_COGNITIVE_EPISODES: usize = 128;
pub const MAX_TOPOLOGY_ID_BYTES: usize = 128;
pub const MAX_TOPOLOGY_LABEL_BYTES: usize = 256;
pub const MAX_TOPOLOGY_PRODUCER_BYTES: usize = 96;

/// The provenance class of an item in the graph. Synthetic fixtures are
/// intentionally distinct from runtime evidence and must never be upgraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveProvenance {
    RuntimeEvent,
    RuntimeDerived,
    SyntheticFixture,
}

/// Evidence reference shared by nodes, edges and episodes. Multiple graph
/// elements may be caused by one event; the same event identity must always
/// carry exactly the same metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveEvidence {
    pub event_id: String,
    pub producer: String,
    pub sequence: u64,
    pub monotonic_ns: u64,
    pub provenance: CognitiveProvenance,
    pub trace_digest: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveNodeKind {
    Memory,
    Concept,
    Observation,
    Task,
    Tool,
    Plan,
    Prediction,
    Entity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStatus {
    Observed,
    Derived,
    Candidate,
    Resolved,
    Rejected,
    Stale,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveRelation {
    RetrievedFrom,
    AssociatedWith,
    Supports,
    Contradicts,
    Predicts,
    DependsOn,
    ExecutedBy,
    CandidateRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveEpisodeKind {
    Retrieval,
    Association,
    Prediction,
    Planning,
    ToolExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveEpisodeState {
    Started,
    Completed,
    Cancelled,
    Failed,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveNode {
    pub id: String,
    pub kind: CognitiveNodeKind,
    pub label: String,
    pub status: EpistemicStatus,
    pub evidence: CognitiveEvidence,
    /// Optional activity/activation signal. This is not confidence or truth.
    pub activation: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relation: CognitiveRelation,
    pub status: EpistemicStatus,
    pub evidence: CognitiveEvidence,
    /// Optional activity/activation signal. This is not confidence or truth.
    pub activation: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEpisode {
    pub id: String,
    pub kind: CognitiveEpisodeKind,
    pub state: CognitiveEpisodeState,
    pub evidence: CognitiveEvidence,
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
}

/// Canonical semantic topology. No camera, material, colour, geometry, LOD,
/// animation or renderer-specific fields are permitted in this structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveTopology {
    pub schema_version: u16,
    pub session_id: String,
    pub revision: u64,
    pub last_sequence: u64,
    pub nodes: Vec<CognitiveNode>,
    pub edges: Vec<CognitiveEdge>,
    pub episodes: Vec<CognitiveEpisode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveTopologyError {
    UnsupportedVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidLabel,
    InvalidProducer,
    InvalidTraceDigest,
    InvalidSequence,
    ConflictingEventMetadata,
    DuplicateNode(String),
    DuplicateEdge(String),
    DuplicateEpisode(String),
    NodeLimitExceeded(usize),
    EdgeLimitExceeded(usize),
    EpisodeLimitExceeded(usize),
    DanglingEdgeEndpoint(String),
    DanglingEpisodeNode(String),
    DanglingEpisodeEdge(String),
    InvalidActivation(String),
}

impl fmt::Display for CognitiveTopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported cognitive topology schema version {version}")
            }
            Self::InvalidIdentifier(kind) => write!(f, "invalid {kind} identifier"),
            Self::InvalidLabel => write!(f, "invalid node label"),
            Self::InvalidProducer => write!(f, "invalid evidence producer"),
            Self::InvalidTraceDigest => {
                write!(
                    f,
                    "trace digest must be 64 lowercase hexadecimal characters"
                )
            }
            Self::InvalidSequence => write!(
                f,
                "evidence sequence is zero, exceeds last_sequence, or has non-monotonic time"
            ),
            Self::ConflictingEventMetadata => {
                write!(
                    f,
                    "one event identity or sequence has conflicting evidence metadata"
                )
            }
            Self::DuplicateNode(id) => write!(f, "duplicate cognitive node {id}"),
            Self::DuplicateEdge(id) => write!(f, "duplicate cognitive edge {id}"),
            Self::DuplicateEpisode(id) => write!(f, "duplicate cognitive episode {id}"),
            Self::NodeLimitExceeded(count) => {
                write!(f, "{count} nodes exceeds limit {MAX_COGNITIVE_NODES}")
            }
            Self::EdgeLimitExceeded(count) => {
                write!(f, "{count} edges exceeds limit {MAX_COGNITIVE_EDGES}")
            }
            Self::EpisodeLimitExceeded(count) => {
                write!(f, "{count} episodes exceeds limit {MAX_COGNITIVE_EPISODES}")
            }
            Self::DanglingEdgeEndpoint(id) => write!(f, "edge references missing node {id}"),
            Self::DanglingEpisodeNode(id) => write!(f, "episode references missing node {id}"),
            Self::DanglingEpisodeEdge(id) => write!(f, "episode references missing edge {id}"),
            Self::InvalidActivation(id) => {
                write!(f, "node/edge {id} has activation outside finite [0, 1]")
            }
        }
    }
}

impl std::error::Error for CognitiveTopologyError {}

impl CognitiveTopology {
    /// Validate all IDs, provenance references, event ordering, graph links and
    /// resource bounds before this object is consumed by a layout or renderer.
    pub fn validate(&self) -> Result<(), CognitiveTopologyError> {
        if self.schema_version != COGNITIVE_TOPOLOGY_SCHEMA_VERSION {
            return Err(CognitiveTopologyError::UnsupportedVersion(
                self.schema_version,
            ));
        }
        validate_identifier(&self.session_id, "session")?;
        if self.nodes.len() > MAX_COGNITIVE_NODES {
            return Err(CognitiveTopologyError::NodeLimitExceeded(self.nodes.len()));
        }
        if self.edges.len() > MAX_COGNITIVE_EDGES {
            return Err(CognitiveTopologyError::EdgeLimitExceeded(self.edges.len()));
        }
        if self.episodes.len() > MAX_COGNITIVE_EPISODES {
            return Err(CognitiveTopologyError::EpisodeLimitExceeded(
                self.episodes.len(),
            ));
        }

        let mut node_ids = HashSet::with_capacity(self.nodes.len());
        let mut edge_ids = HashSet::with_capacity(self.edges.len());
        let mut episode_ids = HashSet::with_capacity(self.episodes.len());
        let mut event_metadata =
            HashMap::<String, (String, u64, u64, CognitiveProvenance, Option<String>)>::new();
        let mut sequence_events = BTreeMap::<u64, String>::new();
        let mut sequence_times = BTreeMap::<u64, u64>::new();

        for node in &self.nodes {
            validate_identifier(&node.id, "node")?;
            if !node_ids.insert(node.id.as_str()) {
                return Err(CognitiveTopologyError::DuplicateNode(node.id.clone()));
            }
            if node.label.is_empty()
                || node.label.len() > MAX_TOPOLOGY_LABEL_BYTES
                || node.label.chars().any(char::is_control)
            {
                return Err(CognitiveTopologyError::InvalidLabel);
            }
            validate_activation(&node.id, node.activation)?;
            validate_evidence(
                &node.evidence,
                self.last_sequence,
                &mut event_metadata,
                &mut sequence_events,
                &mut sequence_times,
            )?;
        }

        for edge in &self.edges {
            validate_identifier(&edge.id, "edge")?;
            if !edge_ids.insert(edge.id.as_str()) {
                return Err(CognitiveTopologyError::DuplicateEdge(edge.id.clone()));
            }
            validate_identifier(&edge.source, "edge source")?;
            validate_identifier(&edge.target, "edge target")?;
            if !node_ids.contains(edge.source.as_str()) {
                return Err(CognitiveTopologyError::DanglingEdgeEndpoint(
                    edge.source.clone(),
                ));
            }
            if !node_ids.contains(edge.target.as_str()) {
                return Err(CognitiveTopologyError::DanglingEdgeEndpoint(
                    edge.target.clone(),
                ));
            }
            validate_activation(&edge.id, edge.activation)?;
            validate_evidence(
                &edge.evidence,
                self.last_sequence,
                &mut event_metadata,
                &mut sequence_events,
                &mut sequence_times,
            )?;
        }

        for episode in &self.episodes {
            validate_identifier(&episode.id, "episode")?;
            if !episode_ids.insert(episode.id.as_str()) {
                return Err(CognitiveTopologyError::DuplicateEpisode(episode.id.clone()));
            }
            validate_evidence(
                &episode.evidence,
                self.last_sequence,
                &mut event_metadata,
                &mut sequence_events,
                &mut sequence_times,
            )?;
            let mut seen_nodes = HashSet::new();
            for node_id in &episode.node_ids {
                if !node_ids.contains(node_id.as_str()) {
                    return Err(CognitiveTopologyError::DanglingEpisodeNode(node_id.clone()));
                }
                if !seen_nodes.insert(node_id.as_str()) {
                    return Err(CognitiveTopologyError::DanglingEpisodeNode(node_id.clone()));
                }
            }
            let mut seen_edges = HashSet::new();
            for edge_id in &episode.edge_ids {
                if !edge_ids.contains(edge_id.as_str()) {
                    return Err(CognitiveTopologyError::DanglingEpisodeEdge(edge_id.clone()));
                }
                if !seen_edges.insert(edge_id.as_str()) {
                    return Err(CognitiveTopologyError::DanglingEpisodeEdge(edge_id.clone()));
                }
            }
        }

        let mut previous_time = None;
        for (_, time) in sequence_times {
            if previous_time.is_some_and(|previous| time < previous) {
                return Err(CognitiveTopologyError::InvalidSequence);
            }
            previous_time = Some(time);
        }
        Ok(())
    }

    /// Digest semantic identity/content only. Input array order is ignored;
    /// renderer, camera, materials, animation and LOD are absent by design.
    pub fn canonical_digest(&self) -> Result<[u8; 32], CognitiveTopologyError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_u64(&mut hasher, u64::from(self.schema_version));
        hash_string(&mut hasher, &self.session_id);
        hash_u64(&mut hasher, self.revision);
        hash_u64(&mut hasher, self.last_sequence);

        let mut nodes: Vec<_> = self.nodes.iter().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        hash_u64(&mut hasher, nodes.len() as u64);
        for node in nodes {
            hash_string(&mut hasher, &node.id);
            hash_string(&mut hasher, node_kind_name(node.kind));
            hash_string(&mut hasher, node.label.as_str());
            hash_string(&mut hasher, epistemic_status_name(node.status));
            hash_evidence(&mut hasher, &node.evidence);
            hash_optional_f32(&mut hasher, node.activation);
        }

        let mut edges: Vec<_> = self.edges.iter().collect();
        edges.sort_by(|a, b| a.id.cmp(&b.id));
        hash_u64(&mut hasher, edges.len() as u64);
        for edge in edges {
            hash_string(&mut hasher, &edge.id);
            hash_string(&mut hasher, &edge.source);
            hash_string(&mut hasher, &edge.target);
            hash_string(&mut hasher, relation_name(edge.relation));
            hash_string(&mut hasher, epistemic_status_name(edge.status));
            hash_evidence(&mut hasher, &edge.evidence);
            hash_optional_f32(&mut hasher, edge.activation);
        }

        let mut episodes: Vec<_> = self.episodes.iter().collect();
        episodes.sort_by(|a, b| a.id.cmp(&b.id));
        hash_u64(&mut hasher, episodes.len() as u64);
        for episode in episodes {
            hash_string(&mut hasher, &episode.id);
            hash_string(&mut hasher, episode_kind_name(episode.kind));
            hash_string(&mut hasher, episode_state_name(episode.state));
            hash_evidence(&mut hasher, &episode.evidence);
            let mut node_ids = episode.node_ids.clone();
            node_ids.sort();
            hash_string_list(&mut hasher, &node_ids);
            let mut edge_ids = episode.edge_ids.clone();
            edge_ids.sort();
            hash_string_list(&mut hasher, &edge_ids);
        }
        Ok(*hasher.finalize().as_bytes())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, CognitiveTopologyError> {
        let digest = self.canonical_digest()?;
        let mut output = String::with_capacity(64);
        for byte in digest {
            use fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        Ok(output)
    }
}

fn validate_identifier(value: &str, kind: &'static str) -> Result<(), CognitiveTopologyError> {
    if value.is_empty()
        || value.len() > MAX_TOPOLOGY_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(CognitiveTopologyError::InvalidIdentifier(kind));
    }
    Ok(())
}

fn validate_activation(id: &str, value: Option<f32>) -> Result<(), CognitiveTopologyError> {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(CognitiveTopologyError::InvalidActivation(id.to_string()));
    }
    Ok(())
}

fn validate_evidence(
    evidence: &CognitiveEvidence,
    last_sequence: u64,
    event_metadata: &mut HashMap<String, (String, u64, u64, CognitiveProvenance, Option<String>)>,
    sequence_events: &mut BTreeMap<u64, String>,
    sequence_times: &mut BTreeMap<u64, u64>,
) -> Result<(), CognitiveTopologyError> {
    validate_identifier(&evidence.event_id, "event")?;
    if evidence.producer.is_empty()
        || evidence.producer.len() > MAX_TOPOLOGY_PRODUCER_BYTES
        || evidence.producer.chars().any(char::is_control)
    {
        return Err(CognitiveTopologyError::InvalidProducer);
    }
    if evidence.sequence == 0 || evidence.sequence > last_sequence {
        return Err(CognitiveTopologyError::InvalidSequence);
    }
    if let Some(digest) = &evidence.trace_digest
        && (digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(CognitiveTopologyError::InvalidTraceDigest);
    }

    let metadata = (
        evidence.producer.clone(),
        evidence.sequence,
        evidence.monotonic_ns,
        evidence.provenance,
        evidence.trace_digest.clone(),
    );
    if let Some(previous) = event_metadata.insert(evidence.event_id.clone(), metadata.clone())
        && previous != metadata
    {
        return Err(CognitiveTopologyError::ConflictingEventMetadata);
    }
    if let Some(previous_event) =
        sequence_events.insert(evidence.sequence, evidence.event_id.clone())
        && previous_event != evidence.event_id
    {
        return Err(CognitiveTopologyError::ConflictingEventMetadata);
    }
    if let Some(previous_time) = sequence_times.insert(evidence.sequence, evidence.monotonic_ns)
        && previous_time != evidence.monotonic_ns
    {
        return Err(CognitiveTopologyError::ConflictingEventMetadata);
    }
    Ok(())
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn hash_optional_f32(hasher: &mut blake3::Hasher, value: Option<f32>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_string_list(hasher: &mut blake3::Hasher, values: &[String]) {
    hash_u64(hasher, values.len() as u64);
    for value in values {
        hash_string(hasher, value);
    }
}

fn hash_evidence(hasher: &mut blake3::Hasher, evidence: &CognitiveEvidence) {
    hash_string(hasher, &evidence.event_id);
    hash_string(hasher, &evidence.producer);
    hash_u64(hasher, evidence.sequence);
    hash_u64(hasher, evidence.monotonic_ns);
    hash_string(hasher, provenance_name(evidence.provenance));
    match &evidence.trace_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hash_string(hasher, digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn provenance_name(value: CognitiveProvenance) -> &'static str {
    match value {
        CognitiveProvenance::RuntimeEvent => "runtime_event",
        CognitiveProvenance::RuntimeDerived => "runtime_derived",
        CognitiveProvenance::SyntheticFixture => "synthetic_fixture",
    }
}

fn node_kind_name(value: CognitiveNodeKind) -> &'static str {
    match value {
        CognitiveNodeKind::Memory => "memory",
        CognitiveNodeKind::Concept => "concept",
        CognitiveNodeKind::Observation => "observation",
        CognitiveNodeKind::Task => "task",
        CognitiveNodeKind::Tool => "tool",
        CognitiveNodeKind::Plan => "plan",
        CognitiveNodeKind::Prediction => "prediction",
        CognitiveNodeKind::Entity => "entity",
    }
}

fn epistemic_status_name(value: EpistemicStatus) -> &'static str {
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

fn episode_kind_name(value: CognitiveEpisodeKind) -> &'static str {
    match value {
        CognitiveEpisodeKind::Retrieval => "retrieval",
        CognitiveEpisodeKind::Association => "association",
        CognitiveEpisodeKind::Prediction => "prediction",
        CognitiveEpisodeKind::Planning => "planning",
        CognitiveEpisodeKind::ToolExecution => "tool_execution",
    }
}

fn episode_state_name(value: CognitiveEpisodeState) -> &'static str {
    match value {
        CognitiveEpisodeState::Started => "started",
        CognitiveEpisodeState::Completed => "completed",
        CognitiveEpisodeState::Cancelled => "cancelled",
        CognitiveEpisodeState::Failed => "failed",
        CognitiveEpisodeState::Unresolved => "unresolved",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(event_id: &str, sequence: u64) -> CognitiveEvidence {
        CognitiveEvidence {
            event_id: event_id.to_string(),
            producer: "synthetic-fixture".to_string(),
            sequence,
            monotonic_ns: sequence * 1_000,
            provenance: CognitiveProvenance::SyntheticFixture,
            trace_digest: None,
        }
    }

    fn fixture() -> CognitiveTopology {
        CognitiveTopology {
            schema_version: COGNITIVE_TOPOLOGY_SCHEMA_VERSION,
            session_id: "fixture-session".to_string(),
            revision: 3,
            last_sequence: 3,
            nodes: vec![
                CognitiveNode {
                    id: "memory:forest".to_string(),
                    kind: CognitiveNodeKind::Memory,
                    label: "Forest restoration notes".to_string(),
                    status: EpistemicStatus::Observed,
                    evidence: evidence("event:retrieval", 1),
                    activation: Some(0.6),
                },
                CognitiveNode {
                    id: "concept:mycelium".to_string(),
                    kind: CognitiveNodeKind::Concept,
                    label: "Mycelial networks".to_string(),
                    status: EpistemicStatus::Observed,
                    evidence: evidence("event:retrieval", 1),
                    activation: None,
                },
                CognitiveNode {
                    id: "concept:soil".to_string(),
                    kind: CognitiveNodeKind::Concept,
                    label: "Soil restoration".to_string(),
                    status: EpistemicStatus::Observed,
                    evidence: evidence("event:retrieval", 1),
                    activation: None,
                },
                CognitiveNode {
                    id: "concept:water".to_string(),
                    kind: CognitiveNodeKind::Concept,
                    label: "Water retention".to_string(),
                    status: EpistemicStatus::Candidate,
                    evidence: evidence("event:association", 2),
                    activation: None,
                },
                CognitiveNode {
                    id: "prediction:resilience".to_string(),
                    kind: CognitiveNodeKind::Prediction,
                    label: "Possible ecosystem resilience".to_string(),
                    status: EpistemicStatus::Candidate,
                    evidence: evidence("event:prediction", 3),
                    activation: None,
                },
            ],
            edges: vec![
                CognitiveEdge {
                    id: "edge:retrieval".to_string(),
                    source: "concept:soil".to_string(),
                    target: "memory:forest".to_string(),
                    relation: CognitiveRelation::RetrievedFrom,
                    status: EpistemicStatus::Observed,
                    evidence: evidence("event:retrieval", 1),
                    activation: None,
                },
                CognitiveEdge {
                    id: "edge:association-a".to_string(),
                    source: "concept:soil".to_string(),
                    target: "concept:mycelium".to_string(),
                    relation: CognitiveRelation::AssociatedWith,
                    status: EpistemicStatus::Candidate,
                    evidence: evidence("event:association", 2),
                    activation: None,
                },
                CognitiveEdge {
                    id: "edge:association-b".to_string(),
                    source: "concept:soil".to_string(),
                    target: "concept:water".to_string(),
                    relation: CognitiveRelation::AssociatedWith,
                    status: EpistemicStatus::Candidate,
                    evidence: evidence("event:association", 2),
                    activation: None,
                },
                CognitiveEdge {
                    id: "edge:prediction".to_string(),
                    source: "concept:mycelium".to_string(),
                    target: "prediction:resilience".to_string(),
                    relation: CognitiveRelation::Predicts,
                    status: EpistemicStatus::Candidate,
                    evidence: evidence("event:prediction", 3),
                    activation: None,
                },
            ],
            episodes: vec![
                CognitiveEpisode {
                    id: "episode:retrieval".to_string(),
                    kind: CognitiveEpisodeKind::Retrieval,
                    state: CognitiveEpisodeState::Completed,
                    evidence: evidence("event:retrieval", 1),
                    node_ids: vec!["memory:forest".to_string(), "concept:soil".to_string()],
                    edge_ids: vec!["edge:retrieval".to_string()],
                },
                CognitiveEpisode {
                    id: "episode:association".to_string(),
                    kind: CognitiveEpisodeKind::Association,
                    state: CognitiveEpisodeState::Completed,
                    evidence: evidence("event:association", 2),
                    node_ids: vec![
                        "concept:soil".to_string(),
                        "concept:mycelium".to_string(),
                        "concept:water".to_string(),
                    ],
                    edge_ids: vec![
                        "edge:association-a".to_string(),
                        "edge:association-b".to_string(),
                    ],
                },
                CognitiveEpisode {
                    id: "episode:prediction".to_string(),
                    kind: CognitiveEpisodeKind::Prediction,
                    state: CognitiveEpisodeState::Unresolved,
                    evidence: evidence("event:prediction", 3),
                    node_ids: vec![
                        "concept:mycelium".to_string(),
                        "prediction:resilience".to_string(),
                    ],
                    edge_ids: vec!["edge:prediction".to_string()],
                },
            ],
        }
    }

    #[test]
    fn checked_in_fixture_matches_the_reference_topology() {
        let checked_in: CognitiveTopology =
            serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
                .expect("checked-in cognitive topology fixture must deserialize");
        assert_eq!(checked_in, fixture());
        checked_in
            .validate()
            .expect("checked-in fixture must validate");
    }

    #[test]
    fn fixture_has_retrieval_competing_associations_and_unresolved_prediction() {
        let topology = fixture();
        topology.validate().expect("fixture must be valid");
        assert_eq!(topology.nodes.len(), 5);
        assert_eq!(
            topology
                .edges
                .iter()
                .filter(|edge| edge.status == EpistemicStatus::Candidate)
                .count(),
            3
        );
        assert!(topology.episodes.iter().any(|episode| {
            episode.kind == CognitiveEpisodeKind::Prediction
                && episode.state == CognitiveEpisodeState::Unresolved
        }));
        assert!(
            topology
                .nodes
                .iter()
                .filter(|node| node.status == EpistemicStatus::Candidate)
                .any(|node| { node.kind == CognitiveNodeKind::Prediction })
        );
    }

    #[test]
    fn canonical_digest_ignores_input_array_order() {
        let topology = fixture();
        let expected = topology.canonical_digest().expect("valid fixture");
        let mut reordered = topology.clone();
        reordered.nodes.reverse();
        reordered.edges.reverse();
        reordered.episodes.reverse();
        for episode in &mut reordered.episodes {
            episode.node_ids.reverse();
            episode.edge_ids.reverse();
        }
        assert_eq!(
            expected,
            reordered
                .canonical_digest()
                .expect("valid reordered fixture")
        );
    }

    #[test]
    fn semantic_content_changes_digest() {
        let topology = fixture();
        let original = topology.canonical_digest().expect("valid fixture");
        let mut changed = topology;
        changed.nodes[0].label.push_str(" updated");
        assert_ne!(
            original,
            changed.canonical_digest().expect("valid changed fixture")
        );
    }

    #[test]
    fn dangling_endpoints_are_not_admitted() {
        let mut topology = fixture();
        topology.edges[0].target = "missing-node".to_string();
        assert_eq!(
            topology.validate(),
            Err(CognitiveTopologyError::DanglingEdgeEndpoint(
                "missing-node".to_string()
            ))
        );
    }

    #[test]
    fn renderer_local_provenance_cannot_be_admitted_as_cognitive_evidence() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
                .expect("fixture JSON");
        value["nodes"][0]["evidence"]["provenance"] =
            serde_json::Value::String("renderer_local".to_string());
        assert!(serde_json::from_value::<CognitiveTopology>(value).is_err());
    }

    #[test]
    fn non_monotonic_event_time_is_rejected() {
        let mut topology = fixture();
        for node in &mut topology.nodes {
            if node.evidence.event_id == "event:retrieval" {
                node.evidence.monotonic_ns = 4_000;
            }
        }
        for edge in &mut topology.edges {
            if edge.evidence.event_id == "event:retrieval" {
                edge.evidence.monotonic_ns = 4_000;
            }
        }
        for episode in &mut topology.episodes {
            if episode.evidence.event_id == "event:retrieval" {
                episode.evidence.monotonic_ns = 4_000;
            }
        }
        assert_eq!(
            topology.validate(),
            Err(CognitiveTopologyError::InvalidSequence)
        );
    }

    #[test]
    fn conflicting_metadata_for_one_event_is_rejected() {
        let mut topology = fixture();
        topology.edges[0].evidence.monotonic_ns += 1;
        assert_eq!(
            topology.validate(),
            Err(CognitiveTopologyError::ConflictingEventMetadata)
        );
    }

    #[test]
    fn invalid_activation_and_out_of_range_sequence_are_rejected() {
        let mut topology = fixture();
        topology.nodes[0].activation = Some(f32::NAN);
        assert!(matches!(
            topology.validate(),
            Err(CognitiveTopologyError::InvalidActivation(_))
        ));

        let mut topology = fixture();
        topology.nodes[0].evidence.sequence = topology.last_sequence + 1;
        assert_eq!(
            topology.validate(),
            Err(CognitiveTopologyError::InvalidSequence)
        );
    }

    #[test]
    fn oversized_graph_is_rejected_before_layout_or_rendering() {
        let mut topology = fixture();
        while topology.nodes.len() <= MAX_COGNITIVE_NODES {
            let next = topology.nodes.len();
            topology.nodes.push(CognitiveNode {
                id: format!("extra:{next}"),
                kind: CognitiveNodeKind::Entity,
                label: "Synthetic fixture node".to_string(),
                status: EpistemicStatus::Observed,
                evidence: evidence("event:prediction", 3),
                activation: None,
            });
        }
        assert_eq!(
            topology.validate(),
            Err(CognitiveTopologyError::NodeLimitExceeded(
                MAX_COGNITIVE_NODES + 1
            ))
        );
    }
}
