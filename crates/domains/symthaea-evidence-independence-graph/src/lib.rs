// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded common-cause lineage assurance for verifier independence.
//!
//! Different verifier or fault-domain identifiers do not establish independence.
//! This crate interprets the direct domains in `VerifierFaultDomainProfile` as
//! entry points into a reviewed DAG and checks whether two verifier lineages share
//! a policy-relevant common ancestor.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;

pub const FAULT_DOMAIN_GRAPH_SCHEMA_V1: &str = "symthaea.assurance.fault-domain-graph.v1";
pub const INDEPENDENCE_POLICY_SCHEMA_V1: &str = "symthaea.assurance.independence-policy.v1";
pub const MAX_GRAPH_NODES: usize = 4_096;
pub const MAX_GRAPH_EDGES: usize = 16_384;
pub const MAX_ANCESTRY_DEPTH: usize = 64;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const GRAPH_DIGEST_DOMAIN: &[u8] = b"symthaea.assurance.fault-domain-graph.digest.v1\0";
const POLICY_DIGEST_DOMAIN: &[u8] = b"symthaea.assurance.independence-policy.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FaultDomainNodeKind {
    Organization,
    ReviewProcess,
    Toolchain,
    EvidenceSource,
    ControlPlane,
    DataRoot,
    GovernanceRoot,
    Infrastructure,
    Other,
}

impl FaultDomainNodeKind {
    fn code(self) -> &'static str {
        match self {
            Self::Organization => "organization",
            Self::ReviewProcess => "review-process",
            Self::Toolchain => "toolchain",
            Self::EvidenceSource => "evidence-source",
            Self::ControlPlane => "control-plane",
            Self::DataRoot => "data-root",
            Self::GovernanceRoot => "governance-root",
            Self::Infrastructure => "infrastructure",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FaultDomainRelation {
    ControlledBy,
    OperatedBy,
    DependsOn,
    DerivedFrom,
    GovernedBy,
}

impl FaultDomainRelation {
    fn code(self) -> &'static str {
        match self {
            Self::ControlledBy => "controlled-by",
            Self::OperatedBy => "operated-by",
            Self::DependsOn => "depends-on",
            Self::DerivedFrom => "derived-from",
            Self::GovernedBy => "governed-by",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IndependenceAxis {
    Organization,
    ReviewProcess,
    Toolchain,
    EvidenceSource,
}

impl IndependenceAxis {
    fn code(self) -> &'static str {
        match self {
            Self::Organization => "organization",
            Self::ReviewProcess => "review-process",
            Self::Toolchain => "toolchain",
            Self::EvidenceSource => "evidence-source",
        }
    }

    fn expected_kind(self) -> FaultDomainNodeKind {
        match self {
            Self::Organization => FaultDomainNodeKind::Organization,
            Self::ReviewProcess => FaultDomainNodeKind::ReviewProcess,
            Self::Toolchain => FaultDomainNodeKind::Toolchain,
            Self::EvidenceSource => FaultDomainNodeKind::EvidenceSource,
        }
    }

    fn entry_node<'a>(self, profile: &'a VerifierFaultDomainProfile) -> &'a str {
        match self {
            Self::Organization => &profile.organization_domain,
            Self::ReviewProcess => &profile.review_process_domain,
            Self::Toolchain => &profile.toolchain_domain,
            Self::EvidenceSource => &profile.evidence_source_domain,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDomainNode {
    pub node_id: String,
    pub kind: FaultDomainNodeKind,
    /// True means the reviewed graph asserts that all policy-relevant immediate
    /// ancestry for this node is represented. False means absence of an edge
    /// cannot be interpreted as absence of a common cause.
    pub lineage_complete: bool,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDomainEdge {
    /// More-specific child domain.
    pub child_node_id: String,
    /// More-general or upstream ancestor/common-cause domain.
    pub ancestor_node_id: String,
    pub relation: FaultDomainRelation,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultDomainGraph {
    pub schema_version: String,
    pub graph_id: String,
    pub nodes: Vec<FaultDomainNode>,
    pub edges: Vec<FaultDomainEdge>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceGraphPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub required_axes: Vec<IndependenceAxis>,
    /// If true, any shared ancestor across the union of required axes blocks
    /// separation, even when the shared node is reached from different axes.
    pub require_global_separation: bool,
    pub max_ancestry_depth: usize,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependenceStatus {
    Invalid,
    Indeterminate,
    Correlated,
    Separated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndependenceIssue {
    InvalidGraphSchema,
    InvalidGraphId,
    EmptyGraph,
    TooManyNodes { observed: usize, maximum: usize },
    TooManyEdges { observed: usize, maximum: usize },
    InvalidGraphEvidence,
    InvalidNode(String),
    DuplicateNode(String),
    InvalidEdge { child: String, ancestor: String },
    DuplicateEdge { child: String, ancestor: String, relation: FaultDomainRelation },
    UnknownEdgeNode(String),
    GraphCycle,
    InvalidPolicy,
    InvalidLeftProfile,
    InvalidRightProfile,
    SameVerifierIdentity,
    MissingProfileNode { verifier_ref: String, axis: IndependenceAxis, node_id: String },
    ProfileNodeKindMismatch {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
        observed: FaultDomainNodeKind,
    },
    IncompleteLineage { verifier_ref: String, axis: IndependenceAxis, node_id: String },
    AncestryDepthLimited { verifier_ref: String, axis: IndependenceAxis, node_id: String },
    SharedAncestor { axis: IndependenceAxis, node_id: String },
    SharedGlobalAncestor(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisIndependenceReport {
    pub axis: IndependenceAxis,
    pub left_entry_node: String,
    pub right_entry_node: String,
    pub shared_ancestors: Vec<String>,
    pub incomplete_nodes: Vec<String>,
    pub depth_limited: bool,
    pub status: IndependenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceGraphReport {
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub graph_id: String,
    pub graph_digest: Option<String>,
    pub left_verifier_ref: String,
    pub right_verifier_ref: String,
    pub status: IndependenceStatus,
    pub issues: Vec<IndependenceIssue>,
    pub axis_reports: Vec<AxisIndependenceReport>,
    pub global_shared_ancestors: Vec<String>,
}

impl IndependenceGraphReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

impl FaultDomainGraph {
    pub fn validate(&self) -> Result<(), Vec<IndependenceIssue>> {
        let issues = graph_issues(self);
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if self.validate().is_err() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(GRAPH_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.graph_id);

        let mut nodes = self.nodes.iter().collect::<Vec<_>>();
        nodes.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        for node in nodes {
            push_field(&mut hasher, &node.node_id);
            push_field(&mut hasher, node.kind.code());
            push_field(&mut hasher, if node.lineage_complete { "1" } else { "0" });
            push_sorted_refs(&mut hasher, &node.evidence_refs);
        }

        let mut edges = self.edges.iter().collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            (
                left.child_node_id.as_str(),
                left.ancestor_node_id.as_str(),
                left.relation,
            )
                .cmp(&(
                    right.child_node_id.as_str(),
                    right.ancestor_node_id.as_str(),
                    right.relation,
                ))
        });
        for edge in edges {
            push_field(&mut hasher, &edge.child_node_id);
            push_field(&mut hasher, &edge.ancestor_node_id);
            push_field(&mut hasher, edge.relation.code());
            push_sorted_refs(&mut hasher, &edge.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl IndependenceGraphPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != INDEPENDENCE_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.required_axes.is_empty()
            || self.max_ancestry_depth == 0
            || self.max_ancestry_depth > MAX_ANCESTRY_DEPTH
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }
        let unique = self.required_axes.iter().copied().collect::<BTreeSet<_>>();
        unique.len() == self.required_axes.len()
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        let mut axes = self.required_axes.clone();
        axes.sort();
        for axis in axes {
            push_field(&mut hasher, axis.code());
        }
        push_field(
            &mut hasher,
            if self.require_global_separation { "1" } else { "0" },
        );
        push_field(&mut hasher, &self.max_ancestry_depth.to_string());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

pub fn assess_verifier_independence(
    left: &VerifierFaultDomainProfile,
    right: &VerifierFaultDomainProfile,
    graph: &FaultDomainGraph,
    policy: &IndependenceGraphPolicy,
) -> IndependenceGraphReport {
    let mut issues = graph_issues(graph);
    if !policy.validate() {
        issues.push(IndependenceIssue::InvalidPolicy);
    }
    if !left.validate() {
        issues.push(IndependenceIssue::InvalidLeftProfile);
    }
    if !right.validate() {
        issues.push(IndependenceIssue::InvalidRightProfile);
    }
    if left.verifier_ref == right.verifier_ref {
        issues.push(IndependenceIssue::SameVerifierIdentity);
    }

    let policy_digest = policy.canonical_digest();
    let graph_digest = graph.canonical_digest();

    if issues.iter().any(is_structurally_invalid) {
        return IndependenceGraphReport {
            policy_id: policy.policy_id.clone(),
            policy_digest,
            graph_id: graph.graph_id.clone(),
            graph_digest,
            left_verifier_ref: left.verifier_ref.clone(),
            right_verifier_ref: right.verifier_ref.clone(),
            status: IndependenceStatus::Invalid,
            issues,
            axis_reports: Vec::new(),
            global_shared_ancestors: Vec::new(),
        };
    }

    let nodes = graph
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
    for edge in &graph.edges {
        adjacency
            .entry(edge.child_node_id.as_str())
            .or_default()
            .push(edge.ancestor_node_id.as_str());
    }
    for ancestors in adjacency.values_mut() {
        ancestors.sort_unstable();
        ancestors.dedup();
    }

    let mut axis_reports = Vec::new();
    let mut left_global = BTreeSet::<String>::new();
    let mut right_global = BTreeSet::<String>::new();

    for axis in &policy.required_axes {
        let left_entry = axis.entry_node(left).to_string();
        let right_entry = axis.entry_node(right).to_string();

        let Some(left_node) = nodes.get(left_entry.as_str()) else {
            issues.push(IndependenceIssue::MissingProfileNode {
                verifier_ref: left.verifier_ref.clone(),
                axis: *axis,
                node_id: left_entry,
            });
            continue;
        };
        let Some(right_node) = nodes.get(right_entry.as_str()) else {
            issues.push(IndependenceIssue::MissingProfileNode {
                verifier_ref: right.verifier_ref.clone(),
                axis: *axis,
                node_id: right_entry,
            });
            continue;
        };
        if left_node.kind != axis.expected_kind() {
            issues.push(IndependenceIssue::ProfileNodeKindMismatch {
                verifier_ref: left.verifier_ref.clone(),
                axis: *axis,
                node_id: left_entry,
                observed: left_node.kind,
            });
            continue;
        }
        if right_node.kind != axis.expected_kind() {
            issues.push(IndependenceIssue::ProfileNodeKindMismatch {
                verifier_ref: right.verifier_ref.clone(),
                axis: *axis,
                node_id: right_entry,
                observed: right_node.kind,
            });
            continue;
        }

        let left_walk = ancestry(
            &left.verifier_ref,
            *axis,
            axis.entry_node(left),
            &nodes,
            &adjacency,
            policy.max_ancestry_depth,
        );
        let right_walk = ancestry(
            &right.verifier_ref,
            *axis,
            axis.entry_node(right),
            &nodes,
            &adjacency,
            policy.max_ancestry_depth,
        );
        issues.extend(left_walk.issues.clone());
        issues.extend(right_walk.issues.clone());
        left_global.extend(left_walk.visited.iter().cloned());
        right_global.extend(right_walk.visited.iter().cloned());

        let shared = left_walk
            .visited
            .intersection(&right_walk.visited)
            .cloned()
            .collect::<Vec<_>>();
        for node_id in &shared {
            issues.push(IndependenceIssue::SharedAncestor {
                axis: *axis,
                node_id: node_id.clone(),
            });
        }

        let mut incomplete_nodes = left_walk.incomplete_nodes;
        incomplete_nodes.extend(right_walk.incomplete_nodes);
        incomplete_nodes.sort();
        incomplete_nodes.dedup();
        let depth_limited = left_walk.depth_limited || right_walk.depth_limited;
        let status = if !shared.is_empty() {
            IndependenceStatus::Correlated
        } else if !incomplete_nodes.is_empty() || depth_limited {
            IndependenceStatus::Indeterminate
        } else {
            IndependenceStatus::Separated
        };

        axis_reports.push(AxisIndependenceReport {
            axis: *axis,
            left_entry_node: axis.entry_node(left).to_string(),
            right_entry_node: axis.entry_node(right).to_string(),
            shared_ancestors: shared,
            incomplete_nodes,
            depth_limited,
            status,
        });
    }

    if issues.iter().any(is_structurally_invalid) {
        return IndependenceGraphReport {
            policy_id: policy.policy_id.clone(),
            policy_digest,
            graph_id: graph.graph_id.clone(),
            graph_digest,
            left_verifier_ref: left.verifier_ref.clone(),
            right_verifier_ref: right.verifier_ref.clone(),
            status: IndependenceStatus::Invalid,
            issues,
            axis_reports,
            global_shared_ancestors: Vec::new(),
        };
    }

    let global_shared_ancestors = if policy.require_global_separation {
        let shared = left_global
            .intersection(&right_global)
            .cloned()
            .collect::<Vec<_>>();
        for node_id in &shared {
            issues.push(IndependenceIssue::SharedGlobalAncestor(node_id.clone()));
        }
        shared
    } else {
        Vec::new()
    };

    let status = if axis_reports
        .iter()
        .any(|report| report.status == IndependenceStatus::Correlated)
        || !global_shared_ancestors.is_empty()
    {
        IndependenceStatus::Correlated
    } else if axis_reports.len() != policy.required_axes.len()
        || axis_reports
            .iter()
            .any(|report| report.status == IndependenceStatus::Indeterminate)
    {
        IndependenceStatus::Indeterminate
    } else {
        IndependenceStatus::Separated
    };

    IndependenceGraphReport {
        policy_id: policy.policy_id.clone(),
        policy_digest,
        graph_id: graph.graph_id.clone(),
        graph_digest,
        left_verifier_ref: left.verifier_ref.clone(),
        right_verifier_ref: right.verifier_ref.clone(),
        status,
        issues,
        axis_reports,
        global_shared_ancestors,
    }
}

#[derive(Debug)]
struct AncestryWalk {
    visited: BTreeSet<String>,
    incomplete_nodes: Vec<String>,
    depth_limited: bool,
    issues: Vec<IndependenceIssue>,
}

fn ancestry(
    verifier_ref: &str,
    axis: IndependenceAxis,
    start: &str,
    nodes: &BTreeMap<&str, &FaultDomainNode>,
    adjacency: &BTreeMap<&str, Vec<&str>>,
    max_depth: usize,
) -> AncestryWalk {
    let mut visited = BTreeSet::new();
    let mut incomplete_nodes = Vec::new();
    let mut issues = Vec::new();
    let mut depth_limited = false;
    let mut queue = VecDeque::from([(start, 0usize)]);

    while let Some((node_id, depth)) = queue.pop_front() {
        if !visited.insert(node_id.to_string()) {
            continue;
        }
        let Some(node) = nodes.get(node_id) else {
            continue;
        };
        if !node.lineage_complete {
            incomplete_nodes.push(node_id.to_string());
            issues.push(IndependenceIssue::IncompleteLineage {
                verifier_ref: verifier_ref.to_string(),
                axis,
                node_id: node_id.to_string(),
            });
        }
        let ancestors = adjacency.get(node_id).cloned().unwrap_or_default();
        if depth >= max_depth && !ancestors.is_empty() {
            depth_limited = true;
            issues.push(IndependenceIssue::AncestryDepthLimited {
                verifier_ref: verifier_ref.to_string(),
                axis,
                node_id: node_id.to_string(),
            });
            continue;
        }
        for ancestor in ancestors {
            queue.push_back((ancestor, depth + 1));
        }
    }

    AncestryWalk {
        visited,
        incomplete_nodes,
        depth_limited,
        issues,
    }
}

fn graph_issues(graph: &FaultDomainGraph) -> Vec<IndependenceIssue> {
    let mut issues = Vec::new();
    if graph.schema_version != FAULT_DOMAIN_GRAPH_SCHEMA_V1 {
        issues.push(IndependenceIssue::InvalidGraphSchema);
    }
    if !canonical_text(&graph.graph_id) {
        issues.push(IndependenceIssue::InvalidGraphId);
    }
    if graph.nodes.is_empty() {
        issues.push(IndependenceIssue::EmptyGraph);
    }
    if graph.nodes.len() > MAX_GRAPH_NODES {
        issues.push(IndependenceIssue::TooManyNodes {
            observed: graph.nodes.len(),
            maximum: MAX_GRAPH_NODES,
        });
    }
    if graph.edges.len() > MAX_GRAPH_EDGES {
        issues.push(IndependenceIssue::TooManyEdges {
            observed: graph.edges.len(),
            maximum: MAX_GRAPH_EDGES,
        });
    }
    if !valid_refs(&graph.evidence_refs) {
        issues.push(IndependenceIssue::InvalidGraphEvidence);
    }

    let mut node_ids = BTreeSet::new();
    for node in &graph.nodes {
        if !canonical_text(&node.node_id) || !valid_refs(&node.evidence_refs) {
            issues.push(IndependenceIssue::InvalidNode(node.node_id.clone()));
        }
        if !node_ids.insert(node.node_id.clone()) {
            issues.push(IndependenceIssue::DuplicateNode(node.node_id.clone()));
        }
    }

    let mut edge_ids = BTreeSet::new();
    for edge in &graph.edges {
        if !canonical_text(&edge.child_node_id)
            || !canonical_text(&edge.ancestor_node_id)
            || edge.child_node_id == edge.ancestor_node_id
            || !valid_refs(&edge.evidence_refs)
        {
            issues.push(IndependenceIssue::InvalidEdge {
                child: edge.child_node_id.clone(),
                ancestor: edge.ancestor_node_id.clone(),
            });
            continue;
        }
        if !node_ids.contains(&edge.child_node_id) {
            issues.push(IndependenceIssue::UnknownEdgeNode(edge.child_node_id.clone()));
        }
        if !node_ids.contains(&edge.ancestor_node_id) {
            issues.push(IndependenceIssue::UnknownEdgeNode(edge.ancestor_node_id.clone()));
        }
        if !edge_ids.insert((
            edge.child_node_id.clone(),
            edge.ancestor_node_id.clone(),
            edge.relation,
        )) {
            issues.push(IndependenceIssue::DuplicateEdge {
                child: edge.child_node_id.clone(),
                ancestor: edge.ancestor_node_id.clone(),
                relation: edge.relation,
            });
        }
    }

    if !issues.iter().any(|issue| {
        matches!(
            issue,
            IndependenceIssue::DuplicateNode(_)
                | IndependenceIssue::UnknownEdgeNode(_)
                | IndependenceIssue::InvalidEdge { .. }
        )
    }) && graph_has_cycle(graph)
    {
        issues.push(IndependenceIssue::GraphCycle);
    }

    issues
}

fn graph_has_cycle(graph: &FaultDomainGraph) -> bool {
    let mut indegree = graph
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
    for edge in &graph.edges {
        if let Some(value) = indegree.get_mut(edge.ancestor_node_id.as_str()) {
            *value += 1;
        }
        adjacency
            .entry(edge.child_node_id.as_str())
            .or_default()
            .push(edge.ancestor_node_id.as_str());
    }
    let mut queue = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(*node))
        .collect::<VecDeque<_>>();
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        for ancestor in adjacency.get(node).into_iter().flatten() {
            if let Some(degree) = indegree.get_mut(ancestor) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(ancestor);
                }
            }
        }
    }
    visited != graph.nodes.len()
}

fn is_structurally_invalid(issue: &IndependenceIssue) -> bool {
    matches!(
        issue,
        IndependenceIssue::InvalidGraphSchema
            | IndependenceIssue::InvalidGraphId
            | IndependenceIssue::EmptyGraph
            | IndependenceIssue::TooManyNodes { .. }
            | IndependenceIssue::TooManyEdges { .. }
            | IndependenceIssue::InvalidGraphEvidence
            | IndependenceIssue::InvalidNode(_)
            | IndependenceIssue::DuplicateNode(_)
            | IndependenceIssue::InvalidEdge { .. }
            | IndependenceIssue::DuplicateEdge { .. }
            | IndependenceIssue::UnknownEdgeNode(_)
            | IndependenceIssue::GraphCycle
            | IndependenceIssue::InvalidPolicy
            | IndependenceIssue::InvalidLeftProfile
            | IndependenceIssue::InvalidRightProfile
            | IndependenceIssue::SameVerifierIdentity
            | IndependenceIssue::MissingProfileNode { .. }
            | IndependenceIssue::ProfileNodeKindMismatch { .. }
    )
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    if values.is_empty() || values.len() > MAX_EVIDENCE_REFS {
        return false;
    }
    let mut unique = BTreeSet::new();
    values
        .iter()
        .all(|value| canonical_text(value) && unique.insert(value.as_str()))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.iter().map(String::as_str).collect::<Vec<_>>();
    refs.sort_unstable();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for value in refs {
        push_field(hasher, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(
        verifier: &str,
        org: &str,
        process: &str,
        tool: &str,
        source: &str,
    ) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: org.into(),
            review_process_domain: process.into(),
            toolchain_domain: tool.into(),
            evidence_source_domain: source.into(),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn node(id: &str, kind: FaultDomainNodeKind, complete: bool) -> FaultDomainNode {
        FaultDomainNode {
            node_id: id.into(),
            kind,
            lineage_complete: complete,
            evidence_refs: vec![format!("evidence:{id}")],
        }
    }

    fn edge(child: &str, ancestor: &str, relation: FaultDomainRelation) -> FaultDomainEdge {
        FaultDomainEdge {
            child_node_id: child.into(),
            ancestor_node_id: ancestor.into(),
            relation,
            evidence_refs: vec![format!("edge:{child}->{ancestor}")],
        }
    }

    fn graph(nodes: Vec<FaultDomainNode>, edges: Vec<FaultDomainEdge>) -> FaultDomainGraph {
        FaultDomainGraph {
            schema_version: FAULT_DOMAIN_GRAPH_SCHEMA_V1.into(),
            graph_id: "graph:v1".into(),
            nodes,
            edges,
            evidence_refs: vec!["review:graph:v1".into()],
        }
    }

    fn policy(axis: IndependenceAxis) -> IndependenceGraphPolicy {
        IndependenceGraphPolicy {
            schema_version: INDEPENDENCE_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:v1".into(),
            required_axes: vec![axis],
            require_global_separation: false,
            max_ancestry_depth: 8,
            evidence_refs: vec!["review:policy:v1".into()],
        }
    }

    #[test]
    fn different_org_leaves_with_shared_control_root_are_correlated() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("org:b", FaultDomainNodeKind::Organization, true),
                node("control:shared", FaultDomainNodeKind::ControlPlane, true),
            ],
            vec![
                edge("org:a", "control:shared", FaultDomainRelation::ControlledBy),
                edge("org:b", "control:shared", FaultDomainRelation::ControlledBy),
            ],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::Organization),
        );
        assert_eq!(report.status, IndependenceStatus::Correlated);
        assert_eq!(
            report.axis_reports[0].shared_ancestors,
            vec!["control:shared".to_string()]
        );
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn complete_disjoint_lineages_are_separated() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("org:b", FaultDomainNodeKind::Organization, true),
                node("control:a", FaultDomainNodeKind::ControlPlane, true),
                node("control:b", FaultDomainNodeKind::ControlPlane, true),
            ],
            vec![
                edge("org:a", "control:a", FaultDomainRelation::ControlledBy),
                edge("org:b", "control:b", FaultDomainRelation::ControlledBy),
            ],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::Organization),
        );
        assert_eq!(report.status, IndependenceStatus::Separated);
    }

    #[test]
    fn incomplete_lineage_is_indeterminate_not_separated() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, false),
                node("org:b", FaultDomainNodeKind::Organization, true),
            ],
            vec![],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::Organization),
        );
        assert_eq!(report.status, IndependenceStatus::Indeterminate);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            IndependenceIssue::IncompleteLineage { verifier_ref, .. }
                if verifier_ref == "verifier:a"
        )));
    }

    #[test]
    fn graph_cycle_is_invalid() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("org:b", FaultDomainNodeKind::Organization, true),
                node("control:x", FaultDomainNodeKind::ControlPlane, true),
            ],
            vec![
                edge("org:a", "control:x", FaultDomainRelation::ControlledBy),
                edge("control:x", "org:a", FaultDomainRelation::DependsOn),
            ],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::Organization),
        );
        assert_eq!(report.status, IndependenceStatus::Invalid);
        assert!(report.issues.contains(&IndependenceIssue::GraphCycle));
    }

    #[test]
    fn missing_profile_entry_node_is_invalid() {
        let left = profile("verifier:a", "org:missing", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![node("org:b", FaultDomainNodeKind::Organization, true)],
            vec![],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::Organization),
        );
        assert_eq!(report.status, IndependenceStatus::Invalid);
    }

    #[test]
    fn distinct_mirrors_of_one_data_root_are_correlated() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:mirror-a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:mirror-b");
        let graph = graph(
            vec![
                node("source:mirror-a", FaultDomainNodeKind::EvidenceSource, true),
                node("source:mirror-b", FaultDomainNodeKind::EvidenceSource, true),
                node("data:root", FaultDomainNodeKind::DataRoot, true),
            ],
            vec![
                edge("source:mirror-a", "data:root", FaultDomainRelation::DerivedFrom),
                edge("source:mirror-b", "data:root", FaultDomainRelation::DerivedFrom),
            ],
        );
        let report = assess_verifier_independence(
            &left,
            &right,
            &graph,
            &policy(IndependenceAxis::EvidenceSource),
        );
        assert_eq!(report.status, IndependenceStatus::Correlated);
    }

    #[test]
    fn depth_truncation_is_indeterminate() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("org:b", FaultDomainNodeKind::Organization, true),
                node("control:a", FaultDomainNodeKind::ControlPlane, true),
                node("root:a", FaultDomainNodeKind::GovernanceRoot, true),
            ],
            vec![
                edge("org:a", "control:a", FaultDomainRelation::ControlledBy),
                edge("control:a", "root:a", FaultDomainRelation::GovernedBy),
            ],
        );
        let mut policy = policy(IndependenceAxis::Organization);
        policy.max_ancestry_depth = 1;
        let report = assess_verifier_independence(&left, &right, &graph, &policy);
        assert_eq!(report.status, IndependenceStatus::Indeterminate);
    }

    #[test]
    fn graph_digest_is_order_independent_for_set_like_content() {
        let first = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("root:a", FaultDomainNodeKind::GovernanceRoot, true),
            ],
            vec![edge("org:a", "root:a", FaultDomainRelation::GovernedBy)],
        );
        let second = graph(
            vec![
                node("root:a", FaultDomainNodeKind::GovernanceRoot, true),
                node("org:a", FaultDomainNodeKind::Organization, true),
            ],
            vec![edge("org:a", "root:a", FaultDomainRelation::GovernedBy)],
        );
        assert_eq!(first.canonical_digest(), second.canonical_digest());
    }

    #[test]
    fn global_separation_catches_cross_axis_common_control() {
        let left = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let right = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization, true),
                node("org:b", FaultDomainNodeKind::Organization, true),
                node("tool:a", FaultDomainNodeKind::Toolchain, true),
                node("tool:b", FaultDomainNodeKind::Toolchain, true),
                node("control:shared", FaultDomainNodeKind::ControlPlane, true),
            ],
            vec![
                edge("org:a", "control:shared", FaultDomainRelation::ControlledBy),
                edge("tool:b", "control:shared", FaultDomainRelation::OperatedBy),
            ],
        );
        let policy = IndependenceGraphPolicy {
            schema_version: INDEPENDENCE_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:global".into(),
            required_axes: vec![IndependenceAxis::Organization, IndependenceAxis::Toolchain],
            require_global_separation: true,
            max_ancestry_depth: 8,
            evidence_refs: vec!["review:policy:global".into()],
        };
        let report = assess_verifier_independence(&left, &right, &graph, &policy);
        assert_eq!(report.status, IndependenceStatus::Correlated);
        assert_eq!(report.global_shared_ancestors, vec!["control:shared".to_string()]);
    }
}
