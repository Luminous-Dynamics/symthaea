// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal recurrence of evidence-backed mycelial cycles.
//!
//! This module implements a bounded, exact-cycle-identity recurrence track over
//! non-overlapping observed time windows. A cycle is present in a window only
//! when all of its stable edge IDs co-exist in that window. Complete windows
//! make missing cycles known-absent; partial/occluded windows make them unknown.
//!
//! This is not zigzag persistence, a continuous-time path algorithm, or a
//! transport simulator. It computes recurrent simultaneous cycles and
//! persistence-weighted node participation (TCN/TCR-inspired metrics). It
//! accepts only observed/accepted edge evidence; candidate contacts are not a
//! supported edge kind.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::AnnotatedMycelialSkeleton;

pub const MYCELIAL_TEMPORAL_CYCLE_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_TEMPORAL_CYCLE_PROFILE_SCHEMA_VERSION: u16 = 1;
pub const MAX_TEMPORAL_MYCELIAL_WINDOWS: usize = 4_096;
pub const MAX_TEMPORAL_MYCELIAL_NODES: usize = 128;
pub const MAX_TEMPORAL_MYCELIAL_EDGES: usize = 512;
pub const MAX_TEMPORAL_CYCLES_PER_WINDOW: usize = 2_048;
const MAX_TEMPORAL_NODE_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalWindowCoverage {
    /// The declared graph region and edge-extraction process were fully observed.
    /// Missing edges are therefore known absent within this declared scope.
    CompleteWithinDeclaredScope,
    /// Occlusion, partial imaging or incomplete annotation means missing edges
    /// are unknown, not absent. Positively observed edges remain usable.
    PartialOrOccluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalEdgeEvidence {
    DirectlyObservedHypha,
    AcceptedParentBranch,
    AcceptedAnastomosis,
    SyntheticFixture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalMycelialEdge {
    pub id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub evidence: TemporalEdgeEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalMycelialSnapshot {
    pub snapshot_id: String,
    /// Half-open observation interval [start_ns, end_ns).
    pub window_start_ns: u64,
    pub window_end_ns: u64,
    /// Stable ROI/mask/scope identity; must not silently change across windows.
    pub observation_scope_id: String,
    pub source_observation_digest: String,
    pub coverage: TemporalWindowCoverage,
    /// Sorted, unique persistent node identities visible in this window.
    pub node_ids: Vec<String>,
    /// Sorted by stable edge ID. Only positively observed/accepted edges appear.
    pub edges: Vec<TemporalMycelialEdge>,
}

impl TemporalMycelialSnapshot {
    pub fn duration_ns(&self) -> u64 {
        self.window_end_ns.saturating_sub(self.window_start_ns)
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialTemporalCycleError> {
        validate_snapshot(self)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-temporal-mycelial-snapshot-v1");
        hash_string(&mut hasher, &self.snapshot_id);
        hasher.update(&self.window_start_ns.to_le_bytes());
        hasher.update(&self.window_end_ns.to_le_bytes());
        hash_string(&mut hasher, &self.observation_scope_id);
        hasher.update(self.source_observation_digest.as_bytes());
        hasher.update(&[match self.coverage {
            TemporalWindowCoverage::CompleteWithinDeclaredScope => 0,
            TemporalWindowCoverage::PartialOrOccluded => 1,
        }]);
        hasher.update(&(self.node_ids.len() as u64).to_le_bytes());
        for node_id in &self.node_ids {
            hash_string(&mut hasher, node_id);
        }
        hasher.update(&(self.edges.len() as u64).to_le_bytes());
        for edge in &self.edges {
            hash_string(&mut hasher, &edge.id);
            hash_string(&mut hasher, &edge.source_node_id);
            hash_string(&mut hasher, &edge.target_node_id);
            hasher.update(&[match edge.evidence {
                TemporalEdgeEvidence::DirectlyObservedHypha => 0,
                TemporalEdgeEvidence::AcceptedParentBranch => 1,
                TemporalEdgeEvidence::AcceptedAnastomosis => 2,
                TemporalEdgeEvidence::SyntheticFixture => 3,
            }]);
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

/// Convert one annotated junction–hypha reconstruction into a temporal
/// snapshot. Every annotated segment requires an explicit evidence record;
/// this function never assumes that a predicted segment is directly observed.
/// Candidate fusion links have no evidence enum variant and cannot be imported.
///
/// Use `DirectlyObservedHypha` only for segments supported by the source image
/// annotation protocol. Use `AcceptedAnastomosis` only after a separate
/// acceptance/evidence process has established that fusion. Synthetic fixtures
/// require the profile config's explicit synthetic-evidence opt-in.
pub fn temporal_snapshot_from_annotated_skeleton(
    skeleton: &AnnotatedMycelialSkeleton,
    window_start_ns: u64,
    window_end_ns: u64,
    observation_scope_id: &str,
    coverage: TemporalWindowCoverage,
    edge_evidence: &[TemporalMycelialEdgeEvidenceRecord],
) -> Result<TemporalMycelialSnapshot, MycelialTemporalCycleError> {
    skeleton.validate().map_err(|error| {
        MycelialTemporalCycleError::InvalidAnnotatedSkeleton(error.to_string())
    })?;
    if window_end_ns <= window_start_ns || !valid_id(observation_scope_id) {
        return Err(MycelialTemporalCycleError::InvalidSnapshot {
            snapshot_id: skeleton.frame_id.clone(),
            reason: "invalid temporal interval or observation scope",
        });
    }
    let mut evidence_by_id = BTreeMap::new();
    for record in edge_evidence {
        if !valid_id(&record.edge_id)
            || evidence_by_id.insert(record.edge_id.as_str(), record.evidence).is_some()
        {
            return Err(MycelialTemporalCycleError::EdgeEvidenceInventoryMismatch(
                record.edge_id.clone(),
            ));
        }
    }
    if evidence_by_id.len() != skeleton.segments.len() {
        return Err(MycelialTemporalCycleError::EdgeEvidenceInventoryMismatch(
            "segment/evidence inventory cardinality differs".to_string(),
        ));
    }

    let mut node_ids: Vec<_> = skeleton
        .junctions
        .iter()
        .map(|junction| junction.id.clone())
        .collect();
    node_ids.sort_unstable();
    let mut edges = Vec::with_capacity(skeleton.segments.len());
    for segment in &skeleton.segments {
        let evidence = evidence_by_id
            .get(segment.id.as_str())
            .copied()
            .ok_or_else(|| MycelialTemporalCycleError::EdgeEvidenceInventoryMismatch(
                segment.id.clone(),
            ))?;
        edges.push(TemporalMycelialEdge {
            id: segment.id.clone(),
            source_node_id: segment.start_junction_id.clone(),
            target_node_id: segment.end_junction_id.clone(),
            evidence,
        });
    }
    edges.sort_by(|left, right| left.id.cmp(&right.id));
    let snapshot = TemporalMycelialSnapshot {
        snapshot_id: skeleton.frame_id.clone(),
        window_start_ns,
        window_end_ns,
        observation_scope_id: observation_scope_id.to_string(),
        source_observation_digest: skeleton.source_observation_digest.clone(),
        coverage,
        node_ids,
        edges,
    };
    validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialSequence {
    pub schema_version: u16,
    pub session_id: String,
    pub coordinate_frame_id: String,
    /// Digest of the fixed calibration manifest shared by all windows.
    pub calibration_digest: String,
    /// Sorted, non-overlapping observations. Gaps are unobserved time, not absent edges.
    pub snapshots: Vec<TemporalMycelialSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialCycleConfig {
    pub minimum_present_windows: u16,
    pub minimum_present_duration_ns: u64,
    /// Fraction of known duration (present + known-absent) in which the cycle exists.
    pub minimum_persistence_fraction: f64,
    /// Known duration / total observed duration. Unknown partial windows lower this.
    pub minimum_known_duration_fraction: f64,
    /// Small positive constant for the TCR degree-exposure denominator.
    pub tcr_epsilon: f64,
}

impl Default for TemporalMycelialCycleConfig {
    fn default() -> Self {
        Self {
            minimum_present_windows: 2,
            minimum_present_duration_ns: 2,
            minimum_persistence_fraction: 0.5,
            minimum_known_duration_fraction: 0.8,
            tcr_epsilon: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalCycleWindowState {
    Present,
    Absent,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalCycleWindowObservation {
    pub snapshot_id: String,
    pub duration_ns: u64,
    pub state: TemporalCycleWindowState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialCycleRecord {
    /// Stable digest of the sorted edge-identity set that defines the cycle.
    pub cycle_id: String,
    /// Stable edge identities in this cycle, sorted for canonical identity.
    pub edge_ids: Vec<String>,
    /// Stable node identities in this cycle, sorted.
    pub node_ids: Vec<String>,
    pub observations: Vec<TemporalCycleWindowObservation>,
    pub present_window_count: usize,
    pub known_absent_window_count: usize,
    pub unknown_window_count: usize,
    pub present_duration_ns: u128,
    pub known_absent_duration_ns: u128,
    pub unknown_duration_ns: u128,
    pub persistence_fraction: Option<f64>,
    pub known_duration_fraction: f64,
    pub persistent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialSnapshotMetrics {
    pub snapshot_id: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
    pub simple_cycle_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialNodeMetrics {
    pub node_id: String,
    /// Sum of present duration for every persistent cycle containing this node.
    pub tcn_duration_ns: u128,
    /// TCN divided by degree × duration exposure. Undefined when any window is partial.
    pub tcr: Option<f64>,
    /// Sum over windows of degree(node, window) × window duration.
    pub degree_exposure_ns: u128,
    pub degree_exposure_complete: bool,
    pub persistent_cycle_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialCycleHyperedge {
    pub cycle_id: String,
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
    pub present_duration_ns: u128,
    pub persistence_fraction: f64,
    pub known_duration_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMycelialCycleProfile {
    pub schema_version: u16,
    pub session_id: String,
    pub coordinate_frame_id: String,
    pub calibration_digest: String,
    pub observation_scope_id: String,
    pub sequence_digest: String,
    pub config: TemporalMycelialCycleConfig,
    /// Complete sorted node identity inventory over every snapshot.
    pub source_node_ids: Vec<String>,
    pub snapshot_metrics: Vec<TemporalMycelialSnapshotMetrics>,
    pub cycles: Vec<TemporalMycelialCycleRecord>,
    /// Persistent cycles represented as higher-order hyperedges over their nodes.
    pub persistent_cycle_hyperedges: Vec<TemporalMycelialCycleHyperedge>,
    pub node_metrics: Vec<TemporalMycelialNodeMetrics>,
    pub observed_duration_ns: u128,
    pub complete_duration_ns: u128,
    pub partial_duration_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialTemporalCycleError {
    InvalidSchemaVersion(u16),
    InvalidSequenceMetadata,
    InvalidConfig,
    InvalidAnnotatedSkeleton(String),
    EdgeEvidenceInventoryMismatch(String),
    TooManyWindows(usize),
    TooManyNodes { snapshot_id: String, count: usize },
    TooManyEdges { snapshot_id: String, count: usize },
    TooManyCycles { snapshot_id: String, count: usize },
    InvalidSnapshot { snapshot_id: String, reason: &'static str },
    DuplicateSnapshotId(String),
    OverlappingWindows { previous: String, current: String },
    ObservationScopeMismatch { previous: String, current: String },
    UnstableNodeIdentity(String),
    UnstableEdgeIdentity(String),
    InvalidProfile(&'static str),
}

impl fmt::Display for MycelialTemporalCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported temporal mycelial schema version {version}")
            }
            Self::InvalidSequenceMetadata => {
                write!(f, "invalid temporal sequence identity, frame or calibration digest")
            }
            Self::InvalidConfig => write!(f, "invalid temporal cycle recurrence configuration"),
            Self::InvalidAnnotatedSkeleton(reason) => {
                write!(f, "annotated skeleton cannot be adapted to a temporal snapshot: {reason}")
            }
            Self::EdgeEvidenceInventoryMismatch(id) => {
                write!(f, "temporal edge evidence does not map exactly to annotated segment {id}")
            }
            Self::TooManyWindows(count) => {
                write!(f, "snapshot count {count} exceeds {MAX_TEMPORAL_MYCELIAL_WINDOWS}")
            }
            Self::TooManyNodes { snapshot_id, count } => {
                write!(f, "snapshot {snapshot_id} node count {count} exceeds its bound")
            }
            Self::TooManyEdges { snapshot_id, count } => {
                write!(f, "snapshot {snapshot_id} edge count {count} exceeds its bound")
            }
            Self::TooManyCycles { snapshot_id, count } => {
                write!(f, "snapshot {snapshot_id} simple-cycle count {count} exceeds its bound")
            }
            Self::InvalidSnapshot { snapshot_id, reason } => {
                write!(f, "invalid temporal snapshot {snapshot_id}: {reason}")
            }
            Self::DuplicateSnapshotId(id) => write!(f, "duplicate snapshot ID {id}"),
            Self::OverlappingWindows { previous, current } => {
                write!(f, "snapshot windows {previous} and {current} overlap or are unordered")
            }
            Self::ObservationScopeMismatch { previous, current } => {
                write!(f, "observation scope changed across temporal sequence: {previous} vs {current}")
            }
            Self::UnstableNodeIdentity(id) => {
                write!(f, "node ID {id} is reused with conflicting identity evidence")
            }
            Self::UnstableEdgeIdentity(id) => {
                write!(f, "edge ID {id} changes endpoints or evidence kind across snapshots")
            }
            Self::InvalidProfile(reason) => write!(f, "invalid temporal cycle profile: {reason}"),
        }
    }
}

impl Error for MycelialTemporalCycleError {}

fn validate_snapshot(
    snapshot: &TemporalMycelialSnapshot,
) -> Result<(), MycelialTemporalCycleError> {
    let fail = |reason| MycelialTemporalCycleError::InvalidSnapshot {
        snapshot_id: snapshot.snapshot_id.clone(),
        reason,
    };
    if !valid_id(&snapshot.snapshot_id)
        || !valid_id(&snapshot.observation_scope_id)
        || !valid_digest(&snapshot.source_observation_digest)
        || snapshot.window_end_ns <= snapshot.window_start_ns
    {
        return Err(fail("invalid ID, source digest or observation interval"));
    }
    if snapshot.node_ids.len() > MAX_TEMPORAL_MYCELIAL_NODES {
        return Err(MycelialTemporalCycleError::TooManyNodes {
            snapshot_id: snapshot.snapshot_id.clone(),
            count: snapshot.node_ids.len(),
        });
    }
    if snapshot.edges.len() > MAX_TEMPORAL_MYCELIAL_EDGES {
        return Err(MycelialTemporalCycleError::TooManyEdges {
            snapshot_id: snapshot.snapshot_id.clone(),
            count: snapshot.edges.len(),
        });
    }
    if snapshot.node_ids.iter().any(|id| !valid_id(id))
        || snapshot.node_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(fail("node IDs must be valid, sorted and unique"));
    }
    let nodes: BTreeSet<_> = snapshot.node_ids.iter().map(String::as_str).collect();
    if snapshot.edges.iter().any(|edge| {
        !valid_id(&edge.id)
            || !valid_id(&edge.source_node_id)
            || !valid_id(&edge.target_node_id)
            || edge.source_node_id == edge.target_node_id
            || !nodes.contains(edge.source_node_id.as_str())
            || !nodes.contains(edge.target_node_id.as_str())
    }) || snapshot.edges.windows(2).any(|pair| pair[0].id >= pair[1].id)
    {
        return Err(fail("edges must be sorted/unique with valid IDs and visible endpoints"));
    }
    Ok(())
}

impl TemporalMycelialSequence {
    pub fn validate(&self) -> Result<(), MycelialTemporalCycleError> {
        if self.schema_version != MYCELIAL_TEMPORAL_CYCLE_SCHEMA_VERSION {
            return Err(MycelialTemporalCycleError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_id(&self.session_id)
            || !valid_id(&self.coordinate_frame_id)
            || !valid_digest(&self.calibration_digest)
            || !valid_id(&self.observation_scope_id)
        {
            return Err(MycelialTemporalCycleError::InvalidSequenceMetadata);
        }
        if self.snapshots.is_empty() || self.snapshots.len() > MAX_TEMPORAL_MYCELIAL_WINDOWS {
            return Err(MycelialTemporalCycleError::TooManyWindows(
                self.snapshots.len(),
            ));
        }

        let mut snapshot_ids = BTreeSet::new();
        let mut edge_identity =
            BTreeMap::<&str, (String, String, TemporalEdgeEvidence)>::new();
        let mut previous_end = None;
        let mut previous_snapshot = String::new();
        let mut observation_scope_id: Option<&str> = None;
        for snapshot in &self.snapshots {
            validate_snapshot(snapshot)?;
            if observation_scope_id
                .is_some_and(|previous| previous != snapshot.observation_scope_id.as_str())
            {
                return Err(MycelialTemporalCycleError::ObservationScopeMismatch {
                    previous: observation_scope_id.unwrap_or_default().to_string(),
                    current: snapshot.observation_scope_id.clone(),
                });
            }
            observation_scope_id = Some(&snapshot.observation_scope_id);
            if !snapshot_ids.insert(snapshot.snapshot_id.as_str()) {
                return Err(MycelialTemporalCycleError::DuplicateSnapshotId(
                    snapshot.snapshot_id.clone(),
                ));
            }
            if let Some(end) = previous_end {
                if snapshot.window_start_ns < end {
                    return Err(MycelialTemporalCycleError::OverlappingWindows {
                        previous: previous_snapshot.clone(),
                        current: snapshot.snapshot_id.clone(),
                    });
                }
            }
            previous_end = Some(snapshot.window_end_ns);
            previous_snapshot.clone_from(&snapshot.snapshot_id);
            for edge in &snapshot.edges {
                let endpoints = if edge.source_node_id <= edge.target_node_id {
                    (
                        edge.source_node_id.clone(),
                        edge.target_node_id.clone(),
                        edge.evidence,
                    )
                } else {
                    (
                        edge.target_node_id.clone(),
                        edge.source_node_id.clone(),
                        edge.evidence,
                    )
                };
                if edge_identity
                    .get(edge.id.as_str())
                    .is_some_and(|previous| previous != &endpoints)
                {
                    return Err(MycelialTemporalCycleError::UnstableEdgeIdentity(
                        edge.id.clone(),
                    ));
                }
                edge_identity.entry(edge.id.as_str()).or_insert(endpoints);
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialTemporalCycleError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-temporal-mycelial-sequence-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.session_id);
        hash_string(&mut hasher, &self.coordinate_frame_id);
        hasher.update(self.calibration_digest.as_bytes());
        hasher.update(&(self.snapshots.len() as u64).to_le_bytes());
        for snapshot in &self.snapshots {
            hasher.update(snapshot.canonical_digest_hex()?.as_bytes());
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DetectedCycle {
    cycle_id: String,
    edge_ids: Vec<String>,
    node_ids: Vec<String>,
}

fn enumerate_snapshot_cycles(
    snapshot: &TemporalMycelialSnapshot,
) -> Result<Vec<DetectedCycle>, MycelialTemporalCycleError> {
    let mut adjacency = BTreeMap::<&str, Vec<(&str, &str)>>::new();
    for node_id in &snapshot.node_ids {
        adjacency.insert(node_id.as_str(), Vec::new());
    }
    for edge in &snapshot.edges {
        adjacency
            .get_mut(edge.source_node_id.as_str())
            .expect("validated source node")
            .push((edge.target_node_id.as_str(), edge.id.as_str()));
        adjacency
            .get_mut(edge.target_node_id.as_str())
            .expect("validated target node")
            .push((edge.source_node_id.as_str(), edge.id.as_str()));
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
    }

    let mut cycles = BTreeMap::<String, DetectedCycle>::new();
    for start in snapshot.node_ids.iter().map(String::as_str) {
        let mut visited = BTreeSet::from([start]);
        let mut path_nodes = vec![start];
        let mut path_edges = Vec::<&str>::new();
        enumerate_from(
            snapshot,
            &adjacency,
            start,
            start,
            &mut visited,
            &mut path_nodes,
            &mut path_edges,
            &mut cycles,
        )?;
    }
    Ok(cycles.into_values().collect())
}

#[allow(clippy::too_many_arguments)]
fn enumerate_from<'a>(
    snapshot: &'a TemporalMycelialSnapshot,
    adjacency: &BTreeMap<&'a str, Vec<(&'a str, &'a str)>>,
    start: &'a str,
    current: &'a str,
    visited: &mut BTreeSet<&'a str>,
    path_nodes: &mut Vec<&'a str>,
    path_edges: &mut Vec<&'a str>,
    cycles: &mut BTreeMap<String, DetectedCycle>,
) -> Result<(), MycelialTemporalCycleError> {
    if let Some(neighbours) = adjacency.get(current) {
        for &(next, edge_id) in neighbours {
            if path_edges.contains(&edge_id) {
                continue;
            }
            if next == start {
                if path_edges.is_empty() {
                    continue;
                }
                let mut edge_ids: Vec<_> = path_edges
                    .iter()
                    .map(|id| (*id).to_string())
                    .collect();
                edge_ids.push(edge_id.to_string());
                edge_ids.sort_unstable();
                edge_ids.dedup();
                if edge_ids.len() < 2 {
                    continue;
                }
                let mut node_ids: Vec<_> =
                    path_nodes.iter().map(|id| (*id).to_string()).collect();
                node_ids.sort_unstable();
                node_ids.dedup();
                let cycle_id = cycle_digest(&edge_ids);
                cycles.entry(cycle_id.clone()).or_insert(DetectedCycle {
                    cycle_id,
                    edge_ids,
                    node_ids,
                });
                if cycles.len() > MAX_TEMPORAL_CYCLES_PER_WINDOW {
                    return Err(MycelialTemporalCycleError::TooManyCycles {
                        snapshot_id: snapshot.snapshot_id.clone(),
                        count: cycles.len(),
                    });
                }
            } else if next > start && !visited.contains(next) {
                visited.insert(next);
                path_nodes.push(next);
                path_edges.push(edge_id);
                enumerate_from(
                    snapshot,
                    adjacency,
                    start,
                    next,
                    visited,
                    path_nodes,
                    path_edges,
                    cycles,
                )?;
                path_edges.pop();
                path_nodes.pop();
                visited.remove(next);
            }
        }
    }
    Ok(())
}

fn cycle_digest(edge_ids: &[String]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-temporal-cycle-edge-set-v1");
    hasher.update(&(edge_ids.len() as u64).to_le_bytes());
    for edge_id in edge_ids {
        hash_string(&mut hasher, edge_id);
    }
    hex_digest(hasher.finalize().as_bytes())
}

pub fn build_temporal_mycelial_cycle_profile(
    sequence: &TemporalMycelialSequence,
    config: TemporalMycelialCycleConfig,
) -> Result<TemporalMycelialCycleProfile, MycelialTemporalCycleError> {
    sequence.validate()?;
    if config.minimum_present_windows == 0
        || config.minimum_present_duration_ns == 0
        || !config.minimum_persistence_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_persistence_fraction)
        || config.minimum_persistence_fraction == 0.0
        || !config.minimum_known_duration_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_known_duration_fraction)
        || config.minimum_known_duration_fraction == 0.0
        || !config.tcr_epsilon.is_finite()
        || !(1.0e-15..=1.0e-3).contains(&config.tcr_epsilon)
    {
        return Err(MycelialTemporalCycleError::InvalidConfig);
    }

    let sequence_digest = sequence.canonical_digest_hex()?;
    let mut cycles = BTreeMap::<String, DetectedCycle>::new();
    let mut snapshot_cycle_ids = Vec::with_capacity(sequence.snapshots.len());
    let mut snapshot_metrics = Vec::with_capacity(sequence.snapshots.len());
    let mut observed_duration_ns = 0u128;
    let mut complete_duration_ns = 0u128;
    let mut partial_duration_ns = 0u128;

    for snapshot in &sequence.snapshots {
        let detected = enumerate_snapshot_cycles(snapshot)?;
        let cycle_ids: BTreeSet<_> =
            detected.iter().map(|cycle| cycle.cycle_id.clone()).collect();
        for cycle in &detected {
            cycles
                .entry(cycle.cycle_id.clone())
                .or_insert_with(|| cycle.clone());
        }
        let (connected_components, cycle_rank) = snapshot_topology(snapshot);
        snapshot_metrics.push(TemporalMycelialSnapshotMetrics {
            snapshot_id: snapshot.snapshot_id.clone(),
            node_count: snapshot.node_ids.len(),
            edge_count: snapshot.edges.len(),
            connected_components,
            cycle_rank,
            simple_cycle_count: detected.len(),
        });
        snapshot_cycle_ids.push(cycle_ids);
        let duration = u128::from(snapshot.duration_ns());
        observed_duration_ns += duration;
        match snapshot.coverage {
            TemporalWindowCoverage::CompleteWithinDeclaredScope => {
                complete_duration_ns += duration;
            }
            TemporalWindowCoverage::PartialOrOccluded => {
                partial_duration_ns += duration;
            }
        }
    }

    let mut cycle_records = Vec::with_capacity(cycles.len());
    for cycle in cycles.values() {
        let mut observations = Vec::with_capacity(sequence.snapshots.len());
        let mut present_window_count = 0usize;
        let mut known_absent_window_count = 0usize;
        let mut unknown_window_count = 0usize;
        let mut present_duration_ns = 0u128;
        let mut known_absent_duration_ns = 0u128;
        let mut unknown_duration_ns = 0u128;
        for (snapshot, snapshot_cycles) in
            sequence.snapshots.iter().zip(&snapshot_cycle_ids)
        {
            let duration = snapshot.duration_ns();
            let state = if snapshot_cycles.contains(&cycle.cycle_id) {
                present_window_count += 1;
                present_duration_ns += u128::from(duration);
                TemporalCycleWindowState::Present
            } else if snapshot.coverage == TemporalWindowCoverage::CompleteWithinDeclaredScope {
                known_absent_window_count += 1;
                known_absent_duration_ns += u128::from(duration);
                TemporalCycleWindowState::Absent
            } else {
                unknown_window_count += 1;
                unknown_duration_ns += u128::from(duration);
                TemporalCycleWindowState::Unknown
            };
            observations.push(TemporalCycleWindowObservation {
                snapshot_id: snapshot.snapshot_id.clone(),
                duration_ns: duration,
                state,
            });
        }
        let known_duration = present_duration_ns + known_absent_duration_ns;
        let persistence_fraction = (known_duration > 0)
            .then_some(present_duration_ns as f64 / known_duration as f64);
        let known_duration_fraction = if observed_duration_ns > 0 {
            known_duration as f64 / observed_duration_ns as f64
        } else {
            0.0
        };
        let persistent = present_window_count >= usize::from(config.minimum_present_windows)
            && present_duration_ns >= u128::from(config.minimum_present_duration_ns)
            && persistence_fraction.is_some_and(|fraction| {
                fraction >= config.minimum_persistence_fraction
            })
            && known_duration_fraction >= config.minimum_known_duration_fraction;
        cycle_records.push(TemporalMycelialCycleRecord {
            cycle_id: cycle.cycle_id.clone(),
            edge_ids: cycle.edge_ids.clone(),
            node_ids: cycle.node_ids.clone(),
            observations,
            present_window_count,
            known_absent_window_count,
            unknown_window_count,
            present_duration_ns,
            known_absent_duration_ns,
            unknown_duration_ns,
            persistence_fraction,
            known_duration_fraction,
            persistent,
        });
    }

    let mut degree_exposure = BTreeMap::<String, u128>::new();
    // Any partial window makes degree exposure incomplete for this profile.
    // Missing nodes/edges in a partial window are unknown, not zero degree.
    let degree_exposure_is_complete = sequence.snapshots.iter().all(|snapshot| {
        snapshot.coverage == TemporalWindowCoverage::CompleteWithinDeclaredScope
    });
    for snapshot in &sequence.snapshots {
        let mut degree = BTreeMap::<&str, u128>::new();
        for node_id in &snapshot.node_ids {
            degree.insert(node_id.as_str(), 0);
        }
        for edge in &snapshot.edges {
            if let Some(value) = degree.get_mut(edge.source_node_id.as_str()) {
                *value += 1;
            }
            if let Some(value) = degree.get_mut(edge.target_node_id.as_str()) {
                *value += 1;
            }
        }
        for (node_id, value) in degree {
            *degree_exposure.entry(node_id.to_string()).or_default() +=
                value * u128::from(snapshot.duration_ns());
        }
    }

    let mut tcn = BTreeMap::<String, u128>::new();
    let mut node_cycle_count = BTreeMap::<String, usize>::new();
    let mut hyperedges = Vec::new();
    for cycle in cycle_records.iter().filter(|cycle| cycle.persistent) {
        for node_id in &cycle.node_ids {
            *tcn.entry(node_id.clone()).or_default() += cycle.present_duration_ns;
            *node_cycle_count.entry(node_id.clone()).or_default() += 1;
        }
        hyperedges.push(TemporalMycelialCycleHyperedge {
            cycle_id: cycle.cycle_id.clone(),
            node_ids: cycle.node_ids.clone(),
            edge_ids: cycle.edge_ids.clone(),
            present_duration_ns: cycle.present_duration_ns,
            persistence_fraction: cycle.persistence_fraction.unwrap_or(0.0),
            known_duration_fraction: cycle.known_duration_fraction,
        });
    }

    let mut all_node_ids = BTreeSet::new();
    for snapshot in &sequence.snapshots {
        all_node_ids.extend(snapshot.node_ids.iter().cloned());
    }
    let source_node_ids: Vec<_> = all_node_ids.iter().cloned().collect();
    let mut node_metrics = Vec::with_capacity(all_node_ids.len());
    for node_id in all_node_ids {
        let tcn_duration_ns = tcn.get(&node_id).copied().unwrap_or(0);
        let degree_exposure_ns = degree_exposure.get(&node_id).copied().unwrap_or(0);
        let exposure_complete = degree_exposure_is_complete;
        let tcr = if exposure_complete {
            Some(
                tcn_duration_ns as f64
                    / (degree_exposure_ns as f64 + config.tcr_epsilon),
            )
        } else {
            None
        };
        node_metrics.push(TemporalMycelialNodeMetrics {
            persistent_cycle_count: node_cycle_count.get(&node_id).copied().unwrap_or(0),
            node_id,
            tcn_duration_ns,
            tcr,
            degree_exposure_ns,
            degree_exposure_complete: exposure_complete,
        });
    }

    let profile = TemporalMycelialCycleProfile {
        schema_version: MYCELIAL_TEMPORAL_CYCLE_PROFILE_SCHEMA_VERSION,
        session_id: sequence.session_id.clone(),
        coordinate_frame_id: sequence.coordinate_frame_id.clone(),
        calibration_digest: sequence.calibration_digest.clone(),
        observation_scope_id: sequence.snapshots[0].observation_scope_id.clone(),
        sequence_digest,
        config,
        source_node_ids,
        snapshot_metrics,
        cycles: cycle_records,
        persistent_cycle_hyperedges: hyperedges,
        node_metrics,
        observed_duration_ns,
        complete_duration_ns,
        partial_duration_ns,
    };
    profile.validate()?;
    Ok(profile)
}

fn snapshot_topology(snapshot: &TemporalMycelialSnapshot) -> (usize, usize) {
    let mut parent: BTreeMap<_, _> = snapshot
        .node_ids
        .iter()
        .map(|node_id| (node_id.as_str(), node_id.as_str()))
        .collect();
    fn find<'a>(
        parent: &mut BTreeMap<&'a str, &'a str>,
        node: &'a str,
    ) -> &'a str {
        let value = parent[node];
        if value == node {
            node
        } else {
            let root = find(parent, value);
            parent.insert(node, root);
            root
        }
    }
    for edge in &snapshot.edges {
        let left = find(&mut parent, &edge.source_node_id);
        let right = find(&mut parent, &edge.target_node_id);
        if left != right {
            let (lower, higher) = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            parent.insert(higher, lower);
        }
    }
    let roots: BTreeSet<_> = snapshot
        .node_ids
        .iter()
        .map(|node| find(&mut parent, node))
        .collect();
    let components = roots.len();
    let cycle_rank = snapshot
        .edges
        .len()
        .saturating_add(components)
        .saturating_sub(snapshot.node_ids.len());
    (components, cycle_rank)
}

impl TemporalMycelialCycleProfile {
    pub fn validate(&self) -> Result<(), MycelialTemporalCycleError> {
        if self.schema_version != MYCELIAL_TEMPORAL_CYCLE_PROFILE_SCHEMA_VERSION
            || !valid_id(&self.session_id)
            || !valid_id(&self.coordinate_frame_id)
            || !valid_digest(&self.calibration_digest)
            || !valid_id(&self.observation_scope_id)
            || !valid_digest(&self.sequence_digest)
            || self.config.minimum_present_windows == 0
            || self.config.minimum_present_duration_ns == 0
            || !self.config.minimum_persistence_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.config.minimum_persistence_fraction)
            || self.config.minimum_persistence_fraction == 0.0
            || !self.config.minimum_known_duration_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.config.minimum_known_duration_fraction)
            || self.config.minimum_known_duration_fraction == 0.0
            || !self.config.tcr_epsilon.is_finite()
            || !(1.0e-15..=1.0e-3).contains(&self.config.tcr_epsilon)
            || self.snapshot_metrics.is_empty()
            || self.snapshot_metrics.len() > MAX_TEMPORAL_MYCELIAL_WINDOWS
            || self.source_node_ids.len()
                > MAX_TEMPORAL_MYCELIAL_NODES.saturating_mul(self.snapshot_metrics.len())
            || self.source_node_ids.iter().any(|id| !valid_id(id))
            || self.source_node_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || self.snapshot_metrics.iter().any(|snapshot| {
                !valid_id(&snapshot.snapshot_id)
                    || snapshot.duration_ns == 0
                    || snapshot.node_count > MAX_TEMPORAL_MYCELIAL_NODES
                    || snapshot.node_ids.len() != snapshot.node_count
                    || snapshot.node_ids.iter().any(|id| !valid_id(id))
                    || snapshot.node_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || snapshot.edge_count > MAX_TEMPORAL_MYCELIAL_EDGES
                    || snapshot.edge_ids.len() != snapshot.edge_count
                    || snapshot.edge_ids.iter().any(|id| !valid_id(id))
                    || snapshot.edge_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || snapshot.simple_cycle_count > MAX_TEMPORAL_CYCLES_PER_WINDOW
                    || snapshot.cycle_rank > snapshot.simple_cycle_count
            })
            || self.snapshot_metrics.windows(2).any(|pair| {
                pair[0].snapshot_id == pair[1].snapshot_id
            })
            || self.edge_evidence_inventory.len()
                > MAX_TEMPORAL_MYCELIAL_WINDOWS.saturating_mul(MAX_TEMPORAL_MYCELIAL_EDGES)
            || self.edge_evidence_inventory.iter().any(|edge| {
                !valid_id(&edge.edge_id)
                    || !valid_id(&edge.source_node_id)
                    || !valid_id(&edge.target_node_id)
                    || edge.source_node_id >= edge.target_node_id
                    || self.source_node_ids.binary_search(&edge.source_node_id).is_err()
                    || self.source_node_ids.binary_search(&edge.target_node_id).is_err()
            })
            || self.edge_evidence_inventory.windows(2).any(|pair| {
                pair[0].edge_id >= pair[1].edge_id
            })
            || self.cycles.len() > MAX_TEMPORAL_CYCLE_RECORDS
            || self.cycles.len().saturating_mul(self.snapshot_metrics.len())
                > MAX_TEMPORAL_CYCLE_WINDOW_OBSERVATIONS
            || self.cycles.windows(2).any(|pair| pair[0].cycle_id >= pair[1].cycle_id)
            || self.node_metrics.windows(2).any(|pair| pair[0].node_id >= pair[1].node_id)
            || self.node_metrics.len() != self.source_node_ids.len()
            || self.node_metrics.iter().zip(&self.source_node_ids).any(|(metric, id)| {
                metric.node_id.as_str() != id.as_str()
            })
            || self.persistent_cycle_hyperedges.windows(2).any(|pair| {
                pair[0].cycle_id >= pair[1].cycle_id
            })
        {
            return Err(MycelialTemporalCycleError::InvalidProfile(
                "invalid metadata, limits, inventories or canonical ordering",
            ));
        }

        let edge_evidence_by_id: BTreeMap<_, _> = self
            .edge_evidence_inventory
            .iter()
            .map(|edge| (edge.edge_id.as_str(), edge))
            .collect();
        let inventory_edge_ids: BTreeSet<_> = edge_evidence_by_id.keys().copied().collect();
        let mut source_node_union = BTreeSet::new();
        let mut snapshot_edge_union = BTreeSet::new();
        let mut expected_observed_duration = 0u128;
        let mut expected_complete_duration = 0u128;
        let mut expected_partial_duration = 0u128;
        let mut expected_edge_counts = TemporalMycelialEdgeEvidenceCounts {
            total_edge_observations: 0,
            directly_observed_hypha_observations: 0,
            accepted_parent_branch_observations: 0,
            accepted_anastomosis_observations: 0,
            synthetic_fixture_observations: 0,
        };
        let mut expected_degree_exposure = BTreeMap::<String, u128>::new();

        for snapshot in &self.snapshot_metrics {
            expected_observed_duration += u128::from(snapshot.duration_ns);
            match snapshot.coverage {
                TemporalWindowCoverage::CompleteWithinDeclaredScope => {
                    expected_complete_duration += u128::from(snapshot.duration_ns);
                }
                TemporalWindowCoverage::PartialOrOccluded => {
                    expected_partial_duration += u128::from(snapshot.duration_ns);
                }
            }
            source_node_union.extend(snapshot.node_ids.iter().map(String::as_str));
            snapshot_edge_union.extend(snapshot.edge_ids.iter().map(String::as_str));
            expected_edge_counts.total_edge_observations += snapshot.edge_ids.len();

            let mut node_degree = BTreeMap::<&str, usize>::new();
            for node_id in &snapshot.node_ids {
                node_degree.insert(node_id.as_str(), 0);
            }
            for edge_id in &snapshot.edge_ids {
                let edge = edge_evidence_by_id.get(edge_id.as_str()).ok_or(
                    MycelialTemporalCycleError::InvalidProfile(
                        "snapshot references an edge absent from the evidence inventory",
                    ),
                )?;
                if !node_degree.contains_key(edge.source_node_id.as_str())
                    || !node_degree.contains_key(edge.target_node_id.as_str())
                {
                    return Err(MycelialTemporalCycleError::InvalidProfile(
                        "snapshot edge endpoint is not visible in that window",
                    ));
                }
                *node_degree.get_mut(edge.source_node_id.as_str()).expect("checked source") += 1;
                *node_degree.get_mut(edge.target_node_id.as_str()).expect("checked target") += 1;
                match edge.evidence {
                    TemporalEdgeEvidence::DirectlyObservedHypha => {
                        expected_edge_counts.directly_observed_hypha_observations += 1;
                    }
                    TemporalEdgeEvidence::AcceptedParentBranch => {
                        expected_edge_counts.accepted_parent_branch_observations += 1;
                    }
                    TemporalEdgeEvidence::AcceptedAnastomosis => {
                        expected_edge_counts.accepted_anastomosis_observations += 1;
                    }
                    TemporalEdgeEvidence::SyntheticFixture => {
                        expected_edge_counts.synthetic_fixture_observations += 1;
                    }
                }
            }
            for (node_id, degree) in node_degree {
                *expected_degree_exposure.entry(node_id.to_string()).or_default() +=
                    (degree as u128) * u128::from(snapshot.duration_ns);
            }
        }

        if source_node_union.into_iter().ne(self.source_node_ids.iter().map(String::as_str))
            || snapshot_edge_union != inventory_edge_ids
            || expected_observed_duration
                != expected_complete_duration + expected_partial_duration
            || expected_observed_duration != self.observed_duration_ns
            || expected_complete_duration != self.complete_duration_ns
            || expected_partial_duration != self.partial_duration_ns
            || expected_edge_counts != self.edge_evidence_counts
            || (expected_edge_counts.synthetic_fixture_observations > 0
                && !self.config.allow_synthetic_evidence)
        {
            return Err(MycelialTemporalCycleError::InvalidProfile(
                "source identity, edge evidence or observed-duration inventory is inconsistent",
            ));
        }

        let mut expected_tcn = BTreeMap::<String, u128>::new();
        let mut expected_node_cycle_counts = BTreeMap::<String, usize>::new();
        let mut expected_cycle_counts_per_snapshot =
            vec![0usize; self.snapshot_metrics.len()];
        let mut persistent_cycle_count = 0usize;

        for cycle in &self.cycles {
            if !valid_digest(&cycle.cycle_id)
                || cycle.cycle_id != cycle_digest(&cycle.edge_ids)
                || cycle.edge_ids.len() < 2
                || cycle.edge_ids.iter().any(|id| {
                    !valid_id(id) || !inventory_edge_ids.contains(id.as_str())
                })
                || cycle.edge_ids.windows(2).any(|pair| pair[0] >= pair[1])
                || cycle.node_ids.len() < 2
                || cycle.node_ids.iter().any(|id| {
                    !valid_id(id) || self.source_node_ids.binary_search(id).is_err()
                })
                || cycle.node_ids.windows(2).any(|pair| pair[0] >= pair[1])
                || cycle.observations.len() != self.snapshot_metrics.len()
                || cycle.present_duration_ns
                    + cycle.known_absent_duration_ns
                    + cycle.unknown_duration_ns
                    != self.observed_duration_ns
                || cycle.present_window_count
                    + cycle.known_absent_window_count
                    + cycle.unknown_window_count
                    != cycle.observations.len()
            {
                return Err(MycelialTemporalCycleError::InvalidProfile(
                    "cycle record has invalid identities, evidence references or window totals",
                ));
            }

            let cycle_node_set: BTreeSet<_> =
                cycle.node_ids.iter().map(String::as_str).collect();
            let mut cycle_adjacency = BTreeMap::<&str, Vec<&str>>::new();
            for node_id in &cycle.node_ids {
                cycle_adjacency.insert(node_id.as_str(), Vec::new());
            }
            for edge_id in &cycle.edge_ids {
                let edge = edge_evidence_by_id[edge_id.as_str()];
                if !cycle_node_set.contains(edge.source_node_id.as_str())
                    || !cycle_node_set.contains(edge.target_node_id.as_str())
                {
                    return Err(MycelialTemporalCycleError::InvalidProfile(
                        "cycle edge endpoints do not match its node identity set",
                    ));
                }
                cycle_adjacency.get_mut(edge.source_node_id.as_str())
                    .expect("cycle source node exists")
                    .push(edge.target_node_id.as_str());
                cycle_adjacency.get_mut(edge.target_node_id.as_str())
                    .expect("cycle target node exists")
                    .push(edge.source_node_id.as_str());
            }
            if cycle_adjacency.values().any(|neighbours| neighbours.len() != 2) {
                return Err(MycelialTemporalCycleError::InvalidProfile(
                    "cycle edge inventory does not form a degree-two multigraph",
                ));
            }
            let start_node = cycle.node_ids[0].as_str();
            let mut visited_nodes = BTreeSet::from([start_node]);
            let mut pending_nodes = vec![start_node];
            while let Some(current) = pending_nodes.pop() {
                for neighbour in &cycle_adjacency[current] {
                    if visited_nodes.insert(*neighbour) {
                        pending_nodes.push(neighbour);
                    }
                }
            }
            if visited_nodes.len() != cycle.node_ids.len()
                || cycle.edge_ids.len() != cycle.node_ids.len()
            {
                return Err(MycelialTemporalCycleError::InvalidProfile(
                    "cycle edge inventory is disconnected or not a simple cycle",
                ));
            }

            let mut present_windows = 0usize;
            let mut absent_windows = 0usize;
            let mut unknown_windows = 0usize;
            let mut present_duration = 0u128;
            let mut absent_duration = 0u128;
            let mut unknown_duration = 0u128;
            for (index, observation) in cycle.observations.iter().enumerate() {
                let snapshot = &self.snapshot_metrics[index];
                if observation.snapshot_id != snapshot.snapshot_id
                    || observation.duration_ns != snapshot.duration_ns
                {
                    return Err(MycelialTemporalCycleError::InvalidProfile(
                        "cycle observations are not aligned with snapshot identity and duration",
                    ));
                }
                let all_cycle_edges_present = cycle.edge_ids.iter().all(|edge_id| {
                    snapshot.edge_ids.binary_search(edge_id).is_ok()
                });
                let all_cycle_nodes_present = cycle.node_ids.iter().all(|node_id| {
                    snapshot.node_ids.binary_search(node_id).is_ok()
                });
                let expected_state = if all_cycle_edges_present && all_cycle_nodes_present {
                    TemporalCycleWindowState::Present
                } else if snapshot.coverage == TemporalWindowCoverage::CompleteWithinDeclaredScope {
                    TemporalCycleWindowState::Absent
                } else {
                    TemporalCycleWindowState::Unknown
                };
                if observation.state != expected_state {
                    return Err(MycelialTemporalCycleError::InvalidProfile(
                        "cycle present/absent/unknown state disagrees with the observed edge inventory and window coverage",
                    ));
                }
                match observation.state {
                    TemporalCycleWindowState::Present => {
                        present_windows += 1;
                        present_duration += u128::from(observation.duration_ns);
                        expected_cycle_counts_per_snapshot[index] += 1;
                    }
                    TemporalCycleWindowState::Absent => {
                        absent_windows += 1;
                        absent_duration += u128::from(observation.duration_ns);
                    }
                    TemporalCycleWindowState::Unknown => {
                        unknown_windows += 1;
                        unknown_duration += u128::from(observation.duration_ns);
                    }
                }
            }
            let known_duration = present_duration + absent_duration;
            let expected_persistence = (known_duration > 0)
                .then_some(present_duration as f64 / known_duration as f64);
            let expected_known_fraction = if self.observed_duration_ns > 0 {
                known_duration as f64 / self.observed_duration_ns as f64
            } else {
                0.0
            };
            let expected_persistent =
                present_windows >= usize::from(self.config.minimum_present_windows)
                    && present_duration
                        >= u128::from(self.config.minimum_present_duration_ns)
                    && expected_persistence.is_some_and(|fraction| {
                        fraction >= self.config.minimum_persistence_fraction
                    })
                    && expected_known_fraction
                        >= self.config.minimum_known_duration_fraction;
            if cycle.present_window_count != present_windows
                || cycle.known_absent_window_count != absent_windows
                || cycle.unknown_window_count != unknown_windows
                || cycle.present_duration_ns != present_duration
                || cycle.known_absent_duration_ns != absent_duration
                || cycle.unknown_duration_ns != unknown_duration
                || cycle.persistence_fraction != expected_persistence
                || (cycle.known_duration_fraction - expected_known_fraction).abs() > 1.0e-12
                || cycle.persistent != expected_persistent
            {
                return Err(MycelialTemporalCycleError::InvalidProfile(
                    "cycle recurrence/persistence totals disagree with window observations",
                ));
            }
            if cycle.persistent {
                persistent_cycle_count += 1;
                for node_id in &cycle.node_ids {
                    *expected_tcn.entry(node_id.clone()).or_default() += present_duration;
                    *expected_node_cycle_counts.entry(node_id.clone()).or_default() += 1;
                }
            }
        }

        if self.snapshot_metrics.iter().enumerate().any(|(index, snapshot)| {
            snapshot.simple_cycle_count != expected_cycle_counts_per_snapshot[index]
        }) {
            return Err(MycelialTemporalCycleError::InvalidProfile(
                "snapshot cycle counts disagree with cycle observation records",
            ));
        }
        if self.persistent_cycle_hyperedges.len() != persistent_cycle_count
            || self.persistent_cycle_hyperedges.iter().any(|hyperedge| {
                !self.cycles.iter().any(|cycle| {
                    cycle.persistent
                        && cycle.cycle_id == hyperedge.cycle_id
                        && cycle.edge_ids == hyperedge.edge_ids
                        && cycle.node_ids == hyperedge.node_ids
                        && cycle.present_duration_ns == hyperedge.present_duration_ns
                        && cycle.persistence_fraction == Some(hyperedge.persistence_fraction)
                        && cycle.known_duration_fraction == hyperedge.known_duration_fraction
                })
            })
        {
            return Err(MycelialTemporalCycleError::InvalidProfile(
                "persistent cycle hyperedges do not exactly represent persistent cycle records",
            ));
        }

        let degree_exposure_complete = self.partial_duration_ns == 0;
        if self.node_metrics.iter().any(|node| {
            node.tcn_duration_ns != expected_tcn.get(&node.node_id).copied().unwrap_or(0)
                || node.persistent_cycle_count
                    != expected_node_cycle_counts.get(&node.node_id).copied().unwrap_or(0)
                || node.degree_exposure_ns
                    != expected_degree_exposure.get(&node.node_id).copied().unwrap_or(0)
                || node.degree_exposure_complete != degree_exposure_complete
                || node.tcr.is_some() != degree_exposure_complete
                || node.tcr.is_some_and(|value| {
                    !value.is_finite()
                        || value < 0.0
                        || (value
                            - node.tcn_duration_ns as f64
                                / (node.degree_exposure_ns as f64 + self.config.tcr_epsilon))
                            .abs()
                            > 1.0e-12
                })
        }) {
            return Err(MycelialTemporalCycleError::InvalidProfile(
                "node TCN/TCR metrics disagree with observed edge exposure and persistent cycles",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialTemporalCycleError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-temporal-mycelial-cycle-profile-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.session_id);
        hash_string(&mut hasher, &self.coordinate_frame_id);
        hasher.update(self.calibration_digest.as_bytes());
        hash_string(&mut hasher, &self.observation_scope_id);
        hasher.update(self.sequence_digest.as_bytes());
        hasher.update(&self.config.minimum_present_windows.to_le_bytes());
        hasher.update(&self.config.minimum_present_duration_ns.to_le_bytes());
        hasher.update(&self.config.minimum_persistence_fraction.to_bits().to_le_bytes());
        hasher.update(&self.config.minimum_known_duration_fraction.to_bits().to_le_bytes());
        hasher.update(&self.config.tcr_epsilon.to_bits().to_le_bytes());
        hasher.update(&(self.source_node_ids.len() as u64).to_le_bytes());
        for node_id in &self.source_node_ids {
            hash_string(&mut hasher, node_id);
        }
        hasher.update(&(self.edge_evidence_inventory.len() as u64).to_le_bytes());
        for edge in &self.edge_evidence_inventory {
            hash_string(&mut hasher, &edge.edge_id);
            hasher.update(&[match edge.evidence {
                TemporalEdgeEvidence::DirectlyObservedHypha => 0,
                TemporalEdgeEvidence::AcceptedParentBranch => 1,
                TemporalEdgeEvidence::AcceptedAnastomosis => 2,
                TemporalEdgeEvidence::SyntheticFixture => 3,
            }]);
        }
        for value in [
            self.edge_evidence_counts.total_edge_observations,
            self.edge_evidence_counts.directly_observed_hypha_observations,
            self.edge_evidence_counts.accepted_parent_branch_observations,
            self.edge_evidence_counts.accepted_anastomosis_observations,
            self.edge_evidence_counts.synthetic_fixture_observations,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for snapshot in &self.snapshot_metrics {
            hash_string(&mut hasher, &snapshot.snapshot_id);
            hasher.update(&(snapshot.node_count as u64).to_le_bytes());
            hasher.update(&(snapshot.edge_count as u64).to_le_bytes());
            hasher.update(&(snapshot.connected_components as u64).to_le_bytes());
            hasher.update(&(snapshot.cycle_rank as u64).to_le_bytes());
            hasher.update(&(snapshot.simple_cycle_count as u64).to_le_bytes());
        }
        for cycle in &self.cycles {
            hasher.update(cycle.cycle_id.as_bytes());
            for edge_id in &cycle.edge_ids {
                hash_string(&mut hasher, edge_id);
            }
            for node_id in &cycle.node_ids {
                hash_string(&mut hasher, node_id);
            }
            for observation in &cycle.observations {
                hash_string(&mut hasher, &observation.snapshot_id);
                hasher.update(&observation.duration_ns.to_le_bytes());
                hasher.update(&[match observation.state {
                    TemporalCycleWindowState::Present => 0,
                    TemporalCycleWindowState::Absent => 1,
                    TemporalCycleWindowState::Unknown => 2,
                }]);
            }
            hasher.update(&(cycle.present_window_count as u64).to_le_bytes());
            hasher.update(&(cycle.known_absent_window_count as u64).to_le_bytes());
            hasher.update(&(cycle.unknown_window_count as u64).to_le_bytes());
            hasher.update(&cycle.present_duration_ns.to_le_bytes());
            hasher.update(&cycle.known_absent_duration_ns.to_le_bytes());
            hasher.update(&cycle.unknown_duration_ns.to_le_bytes());
            match cycle.persistence_fraction {
                Some(value) => {
                    hasher.update(&[1]);
                    hasher.update(&value.to_bits().to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            hasher.update(&cycle.known_duration_fraction.to_bits().to_le_bytes());
            hasher.update(&[u8::from(cycle.persistent)]);
        }
        for node in &self.node_metrics {
            hash_string(&mut hasher, &node.node_id);
            hasher.update(&node.tcn_duration_ns.to_le_bytes());
            match node.tcr {
                Some(value) => {
                    hasher.update(&[1]);
                    hasher.update(&value.to_bits().to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            hasher.update(&node.degree_exposure_ns.to_le_bytes());
            hasher.update(&[u8::from(node.degree_exposure_complete)]);
            hasher.update(&(node.persistent_cycle_count as u64).to_le_bytes());
        }
        hasher.update(&self.observed_duration_ns.to_le_bytes());
        hasher.update(&self.complete_duration_ns.to_le_bytes());
        hasher.update(&self.partial_duration_ns.to_le_bytes());
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEMPORAL_NODE_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        AnnotatedMycelialSkeleton, ReconstructionDimensionality, ReconstructionLengthUnit,
        ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn edge(id: &str, source: &str, target: &str) -> TemporalMycelialEdge {
        TemporalMycelialEdge {
            id: id.to_string(),
            source_node_id: source.to_string(),
            target_node_id: target.to_string(),
            evidence: TemporalEdgeEvidence::DirectlyObservedHypha,
        }
    }

    fn snapshot(
        id: &str,
        start: u64,
        end: u64,
        coverage: TemporalWindowCoverage,
        edges: Vec<TemporalMycelialEdge>,
    ) -> TemporalMycelialSnapshot {
        let mut edges = edges;
        edges.sort_by(|left, right| left.id.cmp(&right.id));
        TemporalMycelialSnapshot {
            snapshot_id: id.to_string(),
            window_start_ns: start,
            window_end_ns: end,
            observation_scope_id: "scope:whole-domain-v1".to_string(),
            source_observation_digest: digest('a'),
            coverage,
            node_ids: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            edges,
        }
    }

    fn sequence(snapshots: Vec<TemporalMycelialSnapshot>) -> TemporalMycelialSequence {
        TemporalMycelialSequence {
            schema_version: MYCELIAL_TEMPORAL_CYCLE_SCHEMA_VERSION,
            session_id: "session:1".to_string(),
            coordinate_frame_id: "organism-frame".to_string(),
            calibration_digest: digest('b'),
            snapshots,
        }
    }

    fn cycle_config() -> TemporalMycelialCycleConfig {
        TemporalMycelialCycleConfig {
            minimum_present_windows: 2,
            minimum_present_duration_ns: 20,
            minimum_persistence_fraction: 0.5,
            minimum_known_duration_fraction: 0.5,
            tcr_epsilon: 1.0e-12,
        }
    }

    fn annotated_fixture() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: crate::mycelial_reconstruction_metrics::MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-001".to_string(),
            coordinate_frame_id: "microscope-stage-v1".to_string(),
            calibration_digest: digest('c'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "annotator-v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Branch,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::ProjectedCrossing,
                    position: ReconstructionPoint3 { x: 0.0, y: 10.0, z: 0.0 },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "segment-ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "segment-ac".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "C".to_string(),
                    length: 10.0,
                },
            ],
        }
    }

    #[test]
    fn annotated_skeleton_adapter_preserves_source_ids_and_does_not_invent_crossing_edges() {
        let annotation = annotated_fixture();
        let evidence = vec![
            TemporalMycelialEdgeEvidenceRecord {
                edge_id: "segment-ab".to_string(),
                evidence: TemporalEdgeEvidence::DirectlyObservedHypha,
            },
            TemporalMycelialEdgeEvidenceRecord {
                edge_id: "segment-ac".to_string(),
                evidence: TemporalEdgeEvidence::DirectlyObservedHypha,
            },
        ];
        let snapshot = temporal_snapshot_from_annotated_skeleton(
            &annotation,
            0,
            10,
            "roi:whole-domain-v1",
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            &evidence,
        )
        .unwrap();
        assert_eq!(snapshot.snapshot_id, "frame-001");
        assert_eq!(snapshot.source_observation_digest, annotation.source_observation_digest);
        assert_eq!(snapshot.node_ids, vec!["A", "B", "C"]);
        assert_eq!(snapshot.edges.len(), 2);
        assert!(!snapshot.edges.iter().any(|edge| {
            (edge.source_node_id == "B" && edge.target_node_id == "C")
                || (edge.source_node_id == "C" && edge.target_node_id == "B")
        }));
    }

    #[test]
    fn annotated_skeleton_adapter_rejects_missing_or_extra_evidence_records() {
        let annotation = annotated_fixture();
        let evidence = vec![TemporalMycelialEdgeEvidenceRecord {
            edge_id: "wrong-segment-id".to_string(),
            evidence: TemporalEdgeEvidence::DirectlyObservedHypha,
        }];
        assert!(matches!(
            temporal_snapshot_from_annotated_skeleton(
                &annotation,
                0,
                10,
                "roi:whole-domain-v1",
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                &evidence,
            ),
            Err(MycelialTemporalCycleError::EdgeEvidenceInventoryMismatch(_))
        ));
    }

    #[test]
    fn synthetic_edge_evidence_is_rejected_without_explicit_fixture_opt_in() {
        let cycle_edges = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc", "B", "C"),
        ];
        let input = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges,
            ),
        ]);
        assert!(matches!(
            build_temporal_mycelial_cycle_profile(
                &input,
                TemporalMycelialCycleConfig::default()
            ),
            Err(MycelialTemporalCycleError::SyntheticEvidenceNotAllowed(3))
        ));
    }

    #[test]
    fn recurring_cycle_becomes_hyperedge_and_contributes_tcn_tcr() {
        let cycle_edges = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc", "B", "C"),
        ];
        let sequence = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges.clone(),
            ),
            snapshot(
                "w2",
                10,
                20,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                vec![edge("e-ab", "A", "B"), edge("e-bc", "B", "C")],
            ),
            snapshot(
                "w3",
                20,
                30,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges,
            ),
        ]);
        let profile =
            build_temporal_mycelial_cycle_profile(&sequence, cycle_config()).unwrap();
        assert_eq!(profile.cycles.len(), 1);
        let cycle = &profile.cycles[0];
        assert_eq!(cycle.present_window_count, 2);
        assert_eq!(cycle.known_absent_window_count, 1);
        assert_eq!(cycle.unknown_window_count, 0);
        assert_eq!(cycle.present_duration_ns, 20);
        assert_eq!(cycle.persistence_fraction, Some(2.0 / 3.0));
        assert!(cycle.persistent);
        assert_eq!(profile.persistent_cycle_hyperedges.len(), 1);
        let node_a = profile.node_metrics.iter().find(|node| node.node_id == "A").unwrap();
        assert_eq!(node_a.tcn_duration_ns, 20);
        assert!(node_a.tcr.is_some());
        profile.validate().unwrap();
        let digest = profile.canonical_digest_hex().unwrap();
        assert_eq!(digest, profile.canonical_digest_hex().unwrap());
        let bytes = serde_json::to_vec(&profile).unwrap();
        let restored: TemporalMycelialCycleProfile = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.canonical_digest_hex().unwrap(), digest);
    }

    #[test]
    fn partial_windows_make_missing_cycles_unknown_not_absent() {
        let cycle_edges = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc", "B", "C"),
        ];
        let sequence = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges.clone(),
            ),
            snapshot(
                "w2",
                10,
                20,
                TemporalWindowCoverage::PartialOrOccluded,
                vec![edge("e-ab", "A", "B")],
            ),
            snapshot(
                "w3",
                20,
                30,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges,
            ),
        ]);
        let profile =
            build_temporal_mycelial_cycle_profile(&sequence, cycle_config()).unwrap();
        let cycle = &profile.cycles[0];
        assert_eq!(cycle.present_window_count, 2);
        assert_eq!(cycle.known_absent_window_count, 0);
        assert_eq!(cycle.unknown_window_count, 1);
        assert_eq!(cycle.present_duration_ns, 20);
        assert_eq!(cycle.unknown_duration_ns, 10);
        let node_a = profile.node_metrics.iter().find(|node| node.node_id == "A").unwrap();
        assert!(node_a.tcr.is_none());
        assert!(!node_a.degree_exposure_complete);
    }

    #[test]
    fn temporal_profile_validator_rejects_tampered_cycle_state() {
        let cycle_edges = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc", "B", "C"),
        ];
        let input = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges.clone(),
            ),
            snapshot(
                "w2",
                10,
                20,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                Vec::new(),
            ),
            snapshot(
                "w3",
                20,
                30,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                cycle_edges,
            ),
        ]);
        let mut profile =
            build_temporal_mycelial_cycle_profile(&input, cycle_config()).unwrap();
        profile.cycles[0].observations[1].state = TemporalCycleWindowState::Unknown;
        assert!(matches!(
            profile.validate(),
            Err(MycelialTemporalCycleError::InvalidProfile(_))
        ));
    }

    #[test]
    fn aggregating_edges_across_time_cannot_create_a_simultaneous_cycle() {
        let input = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                vec![edge("e-ab", "A", "B"), edge("e-ac", "A", "C")],
            ),
            snapshot(
                "w2",
                10,
                20,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                vec![edge("e-bc", "B", "C")],
            ),
        ]);
        let profile = build_temporal_mycelial_cycle_profile(
            &input,
            TemporalMycelialCycleConfig {
                minimum_present_windows: 1,
                minimum_present_duration_ns: 1,
                minimum_persistence_fraction: 0.5,
                minimum_known_duration_fraction: 0.5,
                tcr_epsilon: 1.0e-12,
                allow_synthetic_evidence: true,
            },
        )
        .unwrap();
        assert_eq!(profile.snapshot_metrics[0].cycle_rank, 0);
        assert_eq!(profile.snapshot_metrics[1].cycle_rank, 0);
        assert!(profile.cycles.is_empty());
        assert!(profile.persistent_cycle_hyperedges.is_empty());
    }

    #[test]
    fn cycle_identity_requires_exact_stable_edge_ids() {
        let first = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc", "B", "C"),
        ];
        let changed = vec![
            edge("e-ab", "A", "B"),
            edge("e-ac", "A", "C"),
            edge("e-bc-v2", "B", "C"),
        ];
        let sequence = sequence(vec![
            snapshot("w1", 0, 10, TemporalWindowCoverage::CompleteWithinDeclaredScope, first),
            snapshot("w2", 10, 20, TemporalWindowCoverage::CompleteWithinDeclaredScope, changed),
        ]);
        let profile =
            build_temporal_mycelial_cycle_profile(&sequence, cycle_config()).unwrap();
        assert_eq!(profile.cycles.len(), 2);
        assert!(!profile.cycles.iter().any(|cycle| cycle.persistent));
    }

    fn brute_force_simple_cycle_count(snapshot: &TemporalMycelialSnapshot) -> usize {
        assert!(snapshot.edges.len() < usize::BITS as usize);
        let mut count = 0usize;
        for mask in 1usize..(1usize << snapshot.edges.len()) {
            if mask.count_ones() < 2 {
                continue;
            }
            let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
            let mut degrees = BTreeMap::<&str, usize>::new();
            for (index, edge) in snapshot.edges.iter().enumerate() {
                if mask & (1usize << index) == 0 {
                    continue;
                }
                adjacency
                    .entry(edge.source_node_id.as_str())
                    .or_default()
                    .push(edge.target_node_id.as_str());
                adjacency
                    .entry(edge.target_node_id.as_str())
                    .or_default()
                    .push(edge.source_node_id.as_str());
                *degrees.entry(edge.source_node_id.as_str()).or_default() += 1;
                *degrees.entry(edge.target_node_id.as_str()).or_default() += 1;
            }
            if degrees.values().any(|degree| *degree != 2) {
                continue;
            }
            let Some(start) = adjacency.keys().next().copied() else {
                continue;
            };
            let mut visited = BTreeSet::from([start]);
            let mut stack = vec![start];
            while let Some(current) = stack.pop() {
                if let Some(neighbours) = adjacency.get(current) {
                    for neighbour in neighbours {
                        if visited.insert(*neighbour) {
                            stack.push(neighbour);
                        }
                    }
                }
            }
            if visited.len() == adjacency.len() {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn simple_cycle_enumerator_matches_exhaustive_edge_subset_oracle() {
        let mut input = snapshot(
            "square-with-diagonal",
            0,
            10,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            vec![
                edge("e-ab", "A", "B"),
                edge("e-bc", "B", "C"),
                edge("e-cd", "C", "D"),
                edge("e-da", "D", "A"),
                edge("e-ac", "A", "C"),
            ],
        );
        input.node_ids.push("D".to_string());
        input.node_ids.sort_unstable();
        let cycles = enumerate_snapshot_cycles(&input).unwrap();
        assert_eq!(cycles.len(), 3);
        assert_eq!(cycles.len(), brute_force_simple_cycle_count(&input));
        assert_eq!(snapshot_topology(&input), (1, 2));
    }

    #[test]
    fn parallel_hypha_segments_form_a_two_edge_multigraph_cycle() {
        let mut input = snapshot(
            "parallel-segments",
            0,
            10,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            vec![edge("parallel-1", "A", "B"), edge("parallel-2", "A", "B")],
        );
        input.node_ids = vec!["A".to_string(), "B".to_string()];
        let cycles = enumerate_snapshot_cycles(&input).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].edge_ids, vec!["parallel-1", "parallel-2"]);
        assert_eq!(cycles[0].node_ids, vec!["A", "B"]);
    }

    #[test]
    fn unstable_edge_identity_and_overlapping_windows_fail_closed() {
        let sequence_with_edge_change = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                vec![edge("edge-1", "A", "B")],
            ),
            snapshot(
                "w2",
                10,
                20,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                vec![edge("edge-1", "A", "C")],
            ),
        ]);
        assert!(matches!(
            sequence_with_edge_change.validate(),
            Err(MycelialTemporalCycleError::UnstableEdgeIdentity(_))
        ));
        let overlapping = sequence(vec![
            snapshot(
                "w1",
                0,
                10,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                Vec::new(),
            ),
            snapshot(
                "w2",
                9,
                20,
                TemporalWindowCoverage::CompleteWithinDeclaredScope,
                Vec::new(),
            ),
        ]);
        assert!(matches!(
            overlapping.validate(),
            Err(MycelialTemporalCycleError::OverlappingWindows { .. })
        ));
    }

    #[test]
    fn observation_scope_changes_fail_closed() {
        let first = snapshot(
            "w1",
            0,
            10,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            Vec::new(),
        );
        let mut second = snapshot(
            "w2",
            10,
            20,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            Vec::new(),
        );
        second.observation_scope_id = "scope:shifted-roi".to_string();
        let input = sequence(vec![first, second]);
        assert!(matches!(
            input.validate(),
            Err(MycelialTemporalCycleError::ObservationScopeMismatch { .. })
        ));
    }

    #[test]
    fn sequence_order_permutations_are_not_silently_reordered() {
        let first = snapshot(
            "w1",
            0,
            10,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            Vec::new(),
        );
        let second = snapshot(
            "w2",
            10,
            20,
            TemporalWindowCoverage::CompleteWithinDeclaredScope,
            Vec::new(),
        );
        let mut reversed = sequence(vec![second, first]);
        assert!(reversed.validate().is_err());
        reversed.snapshots.reverse();
        assert!(reversed.validate().is_ok());
    }
}
