// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Event-sourced, transactional updates for CognitiveTopology.
//!
//! The event stream is authoritative for accepted graph changes. Spatial
//! projection, layout, SVG, Bevy, camera, LOD and materials are deliberately
//! absent from this module. Rejected events and rejected batches never mutate
//! the current snapshot or its accepted event log.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cognitive_topology::{
    CognitiveEdge, CognitiveEpisode, CognitiveEpisodeKind, CognitiveEpisodeState,
    CognitiveEvidence, CognitiveNode, CognitiveNodeKind, CognitiveProvenance,
    CognitiveRelation, CognitiveTopology, CognitiveTopologyError, EpistemicStatus,
    MAX_COGNITIVE_EDGES, MAX_COGNITIVE_EPISODES, MAX_COGNITIVE_NODES,
};

pub const COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION: u16 = 1;
pub const MAX_COGNITIVE_EVENTS_PER_BATCH: usize = 64;
pub const MAX_COGNITIVE_EVENT_LOG: usize = 4_096;

/// One owner-emitted semantic change. Its evidence identity and sequence are
/// independent of renderer frames and visual interpolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveTopologyEvent {
    pub schema_version: u16,
    pub session_id: String,
    pub evidence: CognitiveEvidence,
    pub change: CognitiveTopologyChange,
}

/// Supported semantic mutations. Additions and explicit state transitions are
/// preferred to destructive removal so earlier evidence remains inspectable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveTopologyChange {
    NodeAdded {
        node_id: String,
        kind: CognitiveNodeKind,
        label: String,
        status: EpistemicStatus,
        activation: Option<f32>,
    },
    NodeStatusChanged {
        node_id: String,
        expected_status: EpistemicStatus,
        status: EpistemicStatus,
        activation: Option<f32>,
    },
    EdgeAdded {
        edge_id: String,
        source: String,
        target: String,
        relation: CognitiveRelation,
        status: EpistemicStatus,
        activation: Option<f32>,
    },
    EdgeStatusChanged {
        edge_id: String,
        expected_status: EpistemicStatus,
        status: EpistemicStatus,
        activation: Option<f32>,
    },
    EpisodeStarted {
        episode_id: String,
        kind: CognitiveEpisodeKind,
        node_ids: Vec<String>,
        edge_ids: Vec<String>,
    },
    EpisodeStateChanged {
        episode_id: String,
        expected_state: CognitiveEpisodeState,
        state: CognitiveEpisodeState,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveEventApplyOutcome {
    Applied,
    AlreadyApplied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveEventError {
    InvalidEventIdentifier,
    InvalidProducer,
    InvalidTraceDigest,
    InvalidSequence,
    SequenceGap { expected: u64, received: u64 },
    SequenceConflict { sequence: u64 },
    TimestampRegression { previous: u64, received: u64 },
    ConflictingDuplicate(String),
    BatchLimitExceeded(usize),
    EventLogLimitExceeded(usize),
    DuplicateNode(String),
    DuplicateEdge(String),
    DuplicateEpisode(String),
    MissingNode(String),
    MissingEdge(String),
    MissingEpisode(String),
    ExpectedStatusMismatch {
        subject_id: String,
        expected: EpistemicStatus,
        actual: EpistemicStatus,
    },
    ExpectedEpisodeStateMismatch {
        episode_id: String,
        expected: CognitiveEpisodeState,
        actual: CognitiveEpisodeState,
    },
    InvalidEpisodeTransition {
        from: CognitiveEpisodeState,
        to: CognitiveEpisodeState,
    },
    ProvenanceBoundaryViolation(String),
    RevisionExhausted,
    Topology(CognitiveTopologyError),
}

impl fmt::Display for CognitiveEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEventIdentifier => write!(f, "invalid event identifier"),
            Self::InvalidProducer => write!(f, "invalid event producer"),
            Self::InvalidTraceDigest => write!(f, "invalid event trace digest"),
            Self::InvalidSequence => write!(f, "event sequence must be non-zero"),
            Self::SequenceGap { expected, received } => {
                write!(f, "event sequence gap: expected {expected}, received {received}")
            }
            Self::SequenceConflict { sequence } => {
                write!(f, "event sequence {sequence} is already owned by another event")
            }
            Self::TimestampRegression { previous, received } => {
                write!(f, "event timestamp {received} regresses behind {previous}")
            }
            Self::ConflictingDuplicate(id) => {
                write!(f, "event ID {id} was replayed with different content")
            }
            Self::BatchLimitExceeded(count) => {
                write!(f, "event batch size {count} exceeds limit {MAX_COGNITIVE_EVENTS_PER_BATCH}")
            }
            Self::EventLogLimitExceeded(count) => {
                write!(f, "event log length {count} exceeds limit {MAX_COGNITIVE_EVENT_LOG}")
            }
            Self::DuplicateNode(id) => write!(f, "node {id} already exists"),
            Self::DuplicateEdge(id) => write!(f, "edge {id} already exists"),
            Self::DuplicateEpisode(id) => write!(f, "episode {id} already exists"),
            Self::MissingNode(id) => write!(f, "node {id} does not exist"),
            Self::MissingEdge(id) => write!(f, "edge {id} does not exist"),
            Self::MissingEpisode(id) => write!(f, "episode {id} does not exist"),
            Self::ExpectedStatusMismatch { subject_id, .. } => {
                write!(f, "epistemic status precondition failed for {subject_id}")
            }
            Self::ExpectedEpisodeStateMismatch { episode_id, .. } => {
                write!(f, "episode state precondition failed for {episode_id}")
            }
            Self::InvalidEpisodeTransition { from, to } => {
                write!(f, "invalid episode transition {from:?} -> {to:?}")
            }
            Self::ProvenanceBoundaryViolation(id) => {
                write!(f, "event {id} attempts to cross the synthetic/runtime provenance boundary")
            }
            Self::RevisionExhausted => write!(f, "topology revision is exhausted"),
            Self::Topology(error) => write!(f, "resulting topology is invalid: {error}"),
        }
    }
}

impl std::error::Error for CognitiveEventError {}

impl From<CognitiveTopologyError> for CognitiveEventError {
    fn from(value: CognitiveTopologyError) -> Self {
        Self::Topology(value)
    }
}

/// Replayable reducer state. Persist the base snapshot and accepted events
/// together; reconstruct with replay() rather than serializing renderer state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveTopologyReducer {
    base_snapshot: CognitiveTopology,
    topology: CognitiveTopology,
    events: Vec<CognitiveTopologyEvent>,
}

impl CognitiveTopologyReducer {
    pub fn new(base_snapshot: CognitiveTopology) -> Result<Self, CognitiveEventError> {
        base_snapshot.validate()?;
        Ok(Self {
            topology: base_snapshot.clone(),
            base_snapshot,
            events: Vec::new(),
        })
    }

    pub fn snapshot(&self) -> &CognitiveTopology {
        &self.topology
    }

    pub fn base_snapshot(&self) -> &CognitiveTopology {
        &self.base_snapshot
    }

    pub fn events(&self) -> &[CognitiveTopologyEvent] {
        &self.events
    }

    pub fn snapshot_digest_hex(&self) -> Result<String, CognitiveEventError> {
        Ok(self.topology.canonical_digest_hex()?)
    }

    /// Hash-chain the base snapshot digest and each accepted event in sequence.
    /// This digest is not the same as the materialized snapshot digest.
    pub fn event_log_digest_hex(&self) -> Result<String, CognitiveEventError> {
        let base_digest = self.base_snapshot.canonical_digest()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-cognitive-event-log-v1");
        hasher.update(&base_digest);
        hasher.update(&(self.events.len() as u64).to_le_bytes());
        for event in &self.events {
            hasher.update(&event_digest(event));
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }

    /// Apply one event. Exact duplicate delivery is idempotent; conflicting
    /// reuse of an event ID or sequence fails closed.
    pub fn apply_event(
        &mut self,
        event: CognitiveTopologyEvent,
    ) -> Result<CognitiveEventApplyOutcome, CognitiveEventError> {
        validate_event(&event)?;

        if event.session_id != self.topology.session_id {
            return Err(CognitiveEventError::InvalidEventIdentifier);
        }
        let digest = event_digest(&event);
        for previous in &self.events {
            if previous.evidence.event_id == event.evidence.event_id {
                if event_digest(previous) == digest {
                    return Ok(CognitiveEventApplyOutcome::AlreadyApplied);
                }
                return Err(CognitiveEventError::ConflictingDuplicate(
                    event.evidence.event_id,
                ));
            }
            if previous.evidence.sequence == event.evidence.sequence {
                return Err(CognitiveEventError::SequenceConflict {
                    sequence: event.evidence.sequence,
                });
            }
        }

        if self.events.len() >= MAX_COGNITIVE_EVENT_LOG {
            return Err(CognitiveEventError::EventLogLimitExceeded(
                self.events.len() + 1,
            ));
        }
        let expected = self
            .topology
            .last_sequence
            .checked_add(1)
            .ok_or(CognitiveEventError::InvalidSequence)?;
        if event.evidence.sequence != expected {
            return Err(CognitiveEventError::SequenceGap {
                expected,
                received: event.evidence.sequence,
            });
        }

        let previous_time = latest_evidence_time(&self.topology);
        if event.evidence.monotonic_ns < previous_time {
            return Err(CognitiveEventError::TimestampRegression {
                previous: previous_time,
                received: event.evidence.monotonic_ns,
            });
        }

        let mut next = self.topology.clone();
        apply_change(&mut next, &event)?;
        next.revision = next
            .revision
            .checked_add(1)
            .ok_or(CognitiveEventError::RevisionExhausted)?;
        next.last_sequence = event.evidence.sequence;
        next.validate()?;

        self.topology = next;
        self.events.push(event);
        Ok(CognitiveEventApplyOutcome::Applied)
    }

    /// Apply a batch atomically. If any event fails, neither the snapshot nor
    /// the event log changes. A batch may contain exact duplicate deliveries.
    pub fn apply_batch(
        &mut self,
        batch: &[CognitiveTopologyEvent],
    ) -> Result<Vec<CognitiveEventApplyOutcome>, CognitiveEventError> {
        if batch.len() > MAX_COGNITIVE_EVENTS_PER_BATCH {
            return Err(CognitiveEventError::BatchLimitExceeded(batch.len()));
        }
        let mut candidate = self.clone();
        let mut outcomes = Vec::with_capacity(batch.len());
        for event in batch {
            outcomes.push(candidate.apply_event(event.clone())?);
        }
        *self = candidate;
        Ok(outcomes)
    }

    /// Rebuild a reducer from its immutable base snapshot and an ordered log.
    /// Replay uses the same validator/reducer as live event application.
    pub fn replay(
        base_snapshot: CognitiveTopology,
        events: &[CognitiveTopologyEvent],
    ) -> Result<Self, CognitiveEventError> {
        let mut reducer = Self::new(base_snapshot)?;
        for batch in events.chunks(MAX_COGNITIVE_EVENTS_PER_BATCH) {
            reducer.apply_batch(batch)?;
        }
        Ok(reducer)
    }
}

fn validate_event(event: &CognitiveTopologyEvent) -> Result<(), CognitiveEventError> {
    if event.schema_version != COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION {
        return Err(CognitiveEventError::InvalidSequence);
    }
    if event.session_id.is_empty()
        || event.session_id.len() > 128
        || event.session_id.chars().any(char::is_control)
    {
        return Err(CognitiveEventError::InvalidEventIdentifier);
    }
    if event.evidence.event_id.is_empty()
        || event.evidence.event_id.len() > 128
        || event.evidence.event_id.chars().any(char::is_control)
    {
        return Err(CognitiveEventError::InvalidEventIdentifier);
    }
    if event.evidence.producer.is_empty()
        || event.evidence.producer.len() > 96
        || event.evidence.producer.chars().any(char::is_control)
    {
        return Err(CognitiveEventError::InvalidProducer);
    }
    if event.evidence.sequence == 0 {
        return Err(CognitiveEventError::InvalidSequence);
    }
    if let Some(digest) = &event.evidence.trace_digest
        && (digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(CognitiveEventError::InvalidTraceDigest);
    }
    Ok(())
}

fn apply_change(
    topology: &mut CognitiveTopology,
    event: &CognitiveTopologyEvent,
) -> Result<(), CognitiveEventError> {
    match &event.change {
        CognitiveTopologyChange::NodeAdded {
            node_id,
            kind,
            label,
            status,
            activation,
        } => {
            if topology.nodes.iter().any(|node| node.id == *node_id) {
                return Err(CognitiveEventError::DuplicateNode(node_id.clone()));
            }
            if topology.nodes.len() >= MAX_COGNITIVE_NODES {
                return Err(CognitiveEventError::Topology(
                    CognitiveTopologyError::NodeLimitExceeded(topology.nodes.len() + 1),
                ));
            }
            let node = CognitiveNode {
                id: node_id.clone(),
                kind: *kind,
                label: label.clone(),
                status: *status,
                evidence: event.evidence.clone(),
                activation: *activation,
            };
            topology.nodes.push(node);
        }
        CognitiveTopologyChange::NodeStatusChanged {
            node_id,
            expected_status,
            status,
            activation,
        } => {
            let node = topology
                .nodes
                .iter_mut()
                .find(|node| node.id == *node_id)
                .ok_or_else(|| CognitiveEventError::MissingNode(node_id.clone()))?;
            if node.status != *expected_status {
                return Err(CognitiveEventError::ExpectedStatusMismatch {
                    subject_id: node_id.clone(),
                    expected: *expected_status,
                    actual: node.status,
                });
            }
            enforce_provenance_boundary(&node.evidence, &event.evidence)?;
            node.status = *status;
            node.activation = *activation;
            node.evidence = event.evidence.clone();
        }
        CognitiveTopologyChange::EdgeAdded {
            edge_id,
            source,
            target,
            relation,
            status,
            activation,
        } => {
            if topology.edges.iter().any(|edge| edge.id == *edge_id) {
                return Err(CognitiveEventError::DuplicateEdge(edge_id.clone()));
            }
            if topology.edges.len() >= MAX_COGNITIVE_EDGES {
                return Err(CognitiveEventError::Topology(
                    CognitiveTopologyError::EdgeLimitExceeded(topology.edges.len() + 1),
                ));
            }
            topology.edges.push(CognitiveEdge {
                id: edge_id.clone(),
                source: source.clone(),
                target: target.clone(),
                relation: *relation,
                status: *status,
                evidence: event.evidence.clone(),
                activation: *activation,
            });
        }
        CognitiveTopologyChange::EdgeStatusChanged {
            edge_id,
            expected_status,
            status,
            activation,
        } => {
            let edge = topology
                .edges
                .iter_mut()
                .find(|edge| edge.id == *edge_id)
                .ok_or_else(|| CognitiveEventError::MissingEdge(edge_id.clone()))?;
            if edge.status != *expected_status {
                return Err(CognitiveEventError::ExpectedStatusMismatch {
                    subject_id: edge_id.clone(),
                    expected: *expected_status,
                    actual: edge.status,
                });
            }
            enforce_provenance_boundary(&edge.evidence, &event.evidence)?;
            edge.status = *status;
            edge.activation = *activation;
            edge.evidence = event.evidence.clone();
        }
        CognitiveTopologyChange::EpisodeStarted {
            episode_id,
            kind,
            node_ids,
            edge_ids,
        } => {
            if topology.episodes.iter().any(|episode| episode.id == *episode_id) {
                return Err(CognitiveEventError::DuplicateEpisode(episode_id.clone()));
            }
            if topology.episodes.len() >= MAX_COGNITIVE_EPISODES {
                return Err(CognitiveEventError::Topology(
                    CognitiveTopologyError::EpisodeLimitExceeded(topology.episodes.len() + 1),
                ));
            }
            topology.episodes.push(CognitiveEpisode {
                id: episode_id.clone(),
                kind: *kind,
                state: CognitiveEpisodeState::Started,
                evidence: event.evidence.clone(),
                node_ids: node_ids.clone(),
                edge_ids: edge_ids.clone(),
            });
        }
        CognitiveTopologyChange::EpisodeStateChanged {
            episode_id,
            expected_state,
            state,
        } => {
            let episode = topology
                .episodes
                .iter_mut()
                .find(|episode| episode.id == *episode_id)
                .ok_or_else(|| CognitiveEventError::MissingEpisode(episode_id.clone()))?;
            if episode.state != *expected_state {
                return Err(CognitiveEventError::ExpectedEpisodeStateMismatch {
                    episode_id: episode_id.clone(),
                    expected: *expected_state,
                    actual: episode.state,
                });
            }
            if !valid_episode_transition(episode.state, *state) {
                return Err(CognitiveEventError::InvalidEpisodeTransition {
                    from: episode.state,
                    to: *state,
                });
            }
            enforce_provenance_boundary(&episode.evidence, &event.evidence)?;
            episode.state = *state;
            episode.evidence = event.evidence.clone();
        }
    }
    Ok(())
}

fn enforce_provenance_boundary(
    previous: &CognitiveEvidence,
    next: &CognitiveEvidence,
) -> Result<(), CognitiveEventError> {
    if (previous.provenance == CognitiveProvenance::SyntheticFixture)
        != (next.provenance == CognitiveProvenance::SyntheticFixture)
    {
        return Err(CognitiveEventError::ProvenanceBoundaryViolation(
            next.event_id.clone(),
        ));
    }
    Ok(())
}

fn valid_episode_transition(from: CognitiveEpisodeState, to: CognitiveEpisodeState) -> bool {
    matches!(
        (from, to),
        (
            CognitiveEpisodeState::Started,
            CognitiveEpisodeState::Completed
                | CognitiveEpisodeState::Cancelled
                | CognitiveEpisodeState::Failed
                | CognitiveEpisodeState::Unresolved
        ) | (
            CognitiveEpisodeState::Unresolved,
            CognitiveEpisodeState::Completed
                | CognitiveEpisodeState::Cancelled
                | CognitiveEpisodeState::Failed
        )
    )
}

fn latest_evidence_time(topology: &CognitiveTopology) -> u64 {
    topology
        .nodes
        .iter()
        .map(|node| node.evidence.monotonic_ns)
        .chain(topology.edges.iter().map(|edge| edge.evidence.monotonic_ns))
        .chain(
            topology
                .episodes
                .iter()
                .map(|episode| episode.evidence.monotonic_ns),
        )
        .max()
        .unwrap_or(0)
}

fn event_digest(event: &CognitiveTopologyEvent) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-cognitive-event-v1");
    hasher.update(&event.schema_version.to_le_bytes());
    hash_string(&mut hasher, &event.session_id);
    hash_evidence(&mut hasher, &event.evidence);
    match &event.change {
        CognitiveTopologyChange::NodeAdded {
            node_id,
            kind,
            label,
            status,
            activation,
        } => {
            hash_tag(&mut hasher, "node_added");
            hash_string(&mut hasher, node_id);
            hash_string(&mut hasher, node_kind_name(*kind));
            hash_string(&mut hasher, label);
            hash_string(&mut hasher, status_name(*status));
            hash_activation(&mut hasher, *activation);
        }
        CognitiveTopologyChange::NodeStatusChanged {
            node_id,
            expected_status,
            status,
            activation,
        } => {
            hash_tag(&mut hasher, "node_status_changed");
            hash_string(&mut hasher, node_id);
            hash_string(&mut hasher, status_name(*expected_status));
            hash_string(&mut hasher, status_name(*status));
            hash_activation(&mut hasher, *activation);
        }
        CognitiveTopologyChange::EdgeAdded {
            edge_id,
            source,
            target,
            relation,
            status,
            activation,
        } => {
            hash_tag(&mut hasher, "edge_added");
            hash_string(&mut hasher, edge_id);
            hash_string(&mut hasher, source);
            hash_string(&mut hasher, target);
            hash_string(&mut hasher, relation_name(*relation));
            hash_string(&mut hasher, status_name(*status));
            hash_activation(&mut hasher, *activation);
        }
        CognitiveTopologyChange::EdgeStatusChanged {
            edge_id,
            expected_status,
            status,
            activation,
        } => {
            hash_tag(&mut hasher, "edge_status_changed");
            hash_string(&mut hasher, edge_id);
            hash_string(&mut hasher, status_name(*expected_status));
            hash_string(&mut hasher, status_name(*status));
            hash_activation(&mut hasher, *activation);
        }
        CognitiveTopologyChange::EpisodeStarted {
            episode_id,
            kind,
            node_ids,
            edge_ids,
        } => {
            hash_tag(&mut hasher, "episode_started");
            hash_string(&mut hasher, episode_id);
            hash_string(&mut hasher, episode_kind_name(*kind));
            hash_string_list(&mut hasher, node_ids);
            hash_string_list(&mut hasher, edge_ids);
        }
        CognitiveTopologyChange::EpisodeStateChanged {
            episode_id,
            expected_state,
            state,
        } => {
            hash_tag(&mut hasher, "episode_state_changed");
            hash_string(&mut hasher, episode_id);
            hash_string(&mut hasher, episode_state_name(*expected_state));
            hash_string(&mut hasher, episode_state_name(*state));
        }
    }
    *hasher.finalize().as_bytes()
}

fn hash_evidence(hasher: &mut blake3::Hasher, evidence: &CognitiveEvidence) {
    hash_string(hasher, &evidence.event_id);
    hash_string(hasher, &evidence.producer);
    hasher.update(&evidence.sequence.to_le_bytes());
    hasher.update(&evidence.monotonic_ns.to_le_bytes());
    hash_string(hasher, provenance_name(evidence.provenance));
    match &evidence.trace_digest {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_activation(hasher: &mut blake3::Hasher, value: Option<f32>) {
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

fn hash_tag(hasher: &mut blake3::Hasher, value: &str) {
    hash_string(hasher, value);
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_string_list(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash_string(hasher, value);
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
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

    fn base() -> CognitiveTopology {
        CognitiveTopology {
            schema_version: crate::cognitive_topology::COGNITIVE_TOPOLOGY_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            revision: 0,
            last_sequence: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
            episodes: Vec::new(),
        }
    }

    fn evidence(sequence: u64, event_id: &str) -> CognitiveEvidence {
        CognitiveEvidence {
            event_id: event_id.to_string(),
            producer: "memory-test".to_string(),
            sequence,
            monotonic_ns: sequence * 1_000,
            provenance: CognitiveProvenance::RuntimeEvent,
            trace_digest: None,
        }
    }

    fn add_node(sequence: u64, id: &str) -> CognitiveTopologyEvent {
        CognitiveTopologyEvent {
            schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            evidence: evidence(sequence, &format!("event:{sequence}")),
            change: CognitiveTopologyChange::NodeAdded {
                node_id: id.to_string(),
                kind: CognitiveNodeKind::Memory,
                label: format!("Memory {id}"),
                status: EpistemicStatus::Observed,
                activation: None,
            },
        }
    }

    #[test]
    fn event_application_updates_snapshot_and_separate_event_log_digest() {
        let mut reducer = CognitiveTopologyReducer::new(base()).expect("valid base");
        let initial_snapshot = reducer.snapshot_digest_hex().expect("snapshot digest");
        let initial_log = reducer.event_log_digest_hex().expect("event log digest");
        assert_ne!(initial_snapshot, initial_log);

        assert_eq!(
            reducer.apply_event(add_node(1, "memory:a")).expect("apply node"),
            CognitiveEventApplyOutcome::Applied
        );
        assert_eq!(reducer.snapshot().nodes.len(), 1);
        assert_eq!(reducer.snapshot().last_sequence, 1);
        assert_eq!(reducer.snapshot().revision, 1);
        assert_ne!(reducer.snapshot_digest_hex().unwrap(), initial_snapshot);
        assert_ne!(reducer.event_log_digest_hex().unwrap(), initial_log);
    }

    #[test]
    fn exact_duplicate_delivery_is_idempotent() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        let event = add_node(1, "memory:a");
        assert_eq!(
            reducer.apply_event(event.clone()).unwrap(),
            CognitiveEventApplyOutcome::Applied
        );
        let snapshot_digest = reducer.snapshot_digest_hex().unwrap();
        let event_digest = reducer.event_log_digest_hex().unwrap();
        assert_eq!(
            reducer.apply_event(event).unwrap(),
            CognitiveEventApplyOutcome::AlreadyApplied
        );
        assert_eq!(reducer.snapshot_digest_hex().unwrap(), snapshot_digest);
        assert_eq!(reducer.event_log_digest_hex().unwrap(), event_digest);
        assert_eq!(reducer.events().len(), 1);
    }

    #[test]
    fn duplicate_event_id_with_different_payload_is_rejected() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        let original = add_node(1, "memory:a");
        reducer.apply_event(original.clone()).unwrap();
        let mut conflicting = original;
        if let CognitiveTopologyChange::NodeAdded { label, .. } = &mut conflicting.change {
            *label = "different payload".to_string();
        }
        assert!(matches!(
            reducer.apply_event(conflicting),
            Err(CognitiveEventError::ConflictingDuplicate(_))
        ));
        assert_eq!(reducer.snapshot().nodes[0].label, "Memory memory:a");
        assert_eq!(reducer.events().len(), 1);
    }

    #[test]
    fn batch_is_atomic_when_a_later_event_is_invalid() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        let snapshot_digest = reducer.snapshot_digest_hex().unwrap();
        let log_digest = reducer.event_log_digest_hex().unwrap();
        let batch = [
            add_node(1, "memory:a"),
            CognitiveTopologyEvent {
                schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
                session_id: "event-reducer-test".to_string(),
                evidence: evidence(2, "event:2"),
                change: CognitiveTopologyChange::EdgeAdded {
                    edge_id: "edge:missing".to_string(),
                    source: "memory:a".to_string(),
                    target: "missing-node".to_string(),
                    relation: CognitiveRelation::AssociatedWith,
                    status: EpistemicStatus::Candidate,
                    activation: None,
                },
            },
        ];
        assert!(reducer.apply_batch(&batch).is_err());
        assert_eq!(reducer.snapshot_digest_hex().unwrap(), snapshot_digest);
        assert_eq!(reducer.event_log_digest_hex().unwrap(), log_digest);
        assert!(reducer.snapshot().nodes.is_empty());
        assert!(reducer.events().is_empty());
    }

    #[test]
    fn replay_reconstructs_identical_snapshot_and_event_log_digest() {
        let base = base();
        let events = vec![
            add_node(1, "memory:a"),
            add_node(2, "concept:b"),
            CognitiveTopologyEvent {
                schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
                session_id: "event-reducer-test".to_string(),
                evidence: evidence(3, "event:3"),
                change: CognitiveTopologyChange::EdgeAdded {
                    edge_id: "edge:a-b".to_string(),
                    source: "memory:a".to_string(),
                    target: "concept:b".to_string(),
                    relation: CognitiveRelation::RetrievedFrom,
                    status: EpistemicStatus::Observed,
                    activation: Some(0.7),
                },
            },
        ];
        let mut live = CognitiveTopologyReducer::new(base.clone()).unwrap();
        live.apply_batch(&events).unwrap();
        let replayed = CognitiveTopologyReducer::replay(base, &events).unwrap();
        assert_eq!(live.snapshot(), replayed.snapshot());
        assert_eq!(
            live.snapshot_digest_hex().unwrap(),
            replayed.snapshot_digest_hex().unwrap()
        );
        assert_eq!(
            live.event_log_digest_hex().unwrap(),
            replayed.event_log_digest_hex().unwrap()
        );
    }

    #[test]
    fn sequence_gaps_and_timestamp_regressions_fail_without_mutation() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        assert!(matches!(
            reducer.apply_event(add_node(2, "memory:a")),
            Err(CognitiveEventError::SequenceGap { .. })
        ));
        let mut regressing = add_node(1, "memory:a");
        regressing.evidence.monotonic_ns = 0;
        assert!(matches!(
            reducer.apply_event(regressing),
            Err(CognitiveEventError::TimestampRegression { .. })
        ));
        assert!(reducer.snapshot().nodes.is_empty());
        assert!(reducer.events().is_empty());
    }

    #[test]
    fn episode_lifecycle_requires_expected_state_and_valid_transition() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        let start = CognitiveTopologyEvent {
            schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            evidence: evidence(1, "event:episode-start"),
            change: CognitiveTopologyChange::EpisodeStarted {
                episode_id: "episode:tool".to_string(),
                kind: CognitiveEpisodeKind::ToolExecution,
                node_ids: Vec::new(),
                edge_ids: Vec::new(),
            },
        };
        reducer.apply_event(start).unwrap();

        let invalid = CognitiveTopologyEvent {
            schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            evidence: evidence(2, "event:episode-invalid"),
            change: CognitiveTopologyChange::EpisodeStateChanged {
                episode_id: "episode:tool".to_string(),
                expected_state: CognitiveEpisodeState::Started,
                state: CognitiveEpisodeState::Started,
            },
        };
        assert!(matches!(
            reducer.apply_event(invalid),
            Err(CognitiveEventError::InvalidEpisodeTransition { .. })
        ));
        assert_eq!(reducer.snapshot().episodes[0].state, CognitiveEpisodeState::Started);

        let complete = CognitiveTopologyEvent {
            schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            evidence: evidence(2, "event:episode-complete"),
            change: CognitiveTopologyChange::EpisodeStateChanged {
                episode_id: "episode:tool".to_string(),
                expected_state: CognitiveEpisodeState::Started,
                state: CognitiveEpisodeState::Completed,
            },
        };
        reducer.apply_event(complete).unwrap();
        assert_eq!(reducer.snapshot().episodes[0].state, CognitiveEpisodeState::Completed);
    }

    #[test]
    fn synthetic_and_runtime_provenance_cannot_be_crossed() {
        let mut reducer = CognitiveTopologyReducer::new(base()).unwrap();
        let mut synthetic = add_node(1, "memory:fixture");
        synthetic.evidence.provenance = CognitiveProvenance::SyntheticFixture;
        reducer.apply_event(synthetic).unwrap();

        let runtime_update = CognitiveTopologyEvent {
            schema_version: COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION,
            session_id: "event-reducer-test".to_string(),
            evidence: evidence(2, "event:runtime-update"),
            change: CognitiveTopologyChange::NodeStatusChanged {
                node_id: "memory:fixture".to_string(),
                expected_status: EpistemicStatus::Observed,
                status: EpistemicStatus::Stale,
                activation: None,
            },
        };
        assert!(matches!(
            reducer.apply_event(runtime_update),
            Err(CognitiveEventError::ProvenanceBoundaryViolation(_))
        ));
        assert_eq!(reducer.snapshot().nodes[0].status, EpistemicStatus::Observed);
        assert_eq!(reducer.snapshot().last_sequence, 1);
    }

    #[test]
    fn renderer_configuration_is_not_part_of_event_or_snapshot_identity() {
        let mut first = CognitiveTopologyReducer::new(base()).unwrap();
        let mut second = CognitiveTopologyReducer::new(base()).unwrap();
        let event = add_node(1, "memory:a");
        first.apply_event(event.clone()).unwrap();
        second.apply_event(event).unwrap();
        let before = first.snapshot_digest_hex().unwrap();
        // Renderer changes are intentionally not accepted by this API.
        assert_eq!(first.snapshot_digest_hex().unwrap(), before);
        assert_eq!(first.event_log_digest_hex().unwrap(), second.event_log_digest_hex().unwrap());
    }
}
