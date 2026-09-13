// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Relation-scoped completeness for verifier independence graphs.
//!
//! This crate strengthens ASSURE-015A without mutating its exact subject.
//! The parent graph answers which common-cause ancestry is represented. This
//! overlay answers which relation classes are reviewed as complete enough for a
//! particular independence theorem.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use symthaea_evidence_independence_graph::{
    FaultDomainGraph, FaultDomainNodeKind, FaultDomainRelation, IndependenceAxis,
    IndependenceStatus, MAX_ANCESTRY_DEPTH,
};
use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;

pub const RELATION_COMPLETENESS_SCHEMA_V1: &str =
    "symthaea.assurance.relation-completeness.v1";
pub const RELATION_AWARE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.relation-aware-independence-policy.v1";
pub const MAX_COMPLETENESS_CLAIMS: usize = 32_768;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const COMPLETENESS_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.relation-completeness.digest.v1\0";
const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.relation-aware-independence-policy.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RelationCompletenessState {
    Complete,
    Incomplete,
}

impl RelationCompletenessState {
    fn code(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Incomplete => "incomplete",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationCompletenessClaim {
    pub node_id: String,
    pub relation: FaultDomainRelation,
    pub state: RelationCompletenessState,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationCompletenessMap {
    pub schema_version: String,
    pub map_id: String,
    /// Exact digest of the ASSURE-015A graph this map qualifies.
    pub graph_digest: String,
    pub claims: Vec<RelationCompletenessClaim>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisRelationRequirement {
    pub axis: IndependenceAxis,
    pub required_relations: Vec<FaultDomainRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationAwareIndependencePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub axis_requirements: Vec<AxisRelationRequirement>,
    pub require_global_separation: bool,
    pub max_ancestry_depth: usize,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationAwareIssue {
    InvalidGraph,
    InvalidCompletenessMap,
    CompletenessGraphDigestMismatch,
    InvalidPolicy,
    InvalidLeftProfile,
    InvalidRightProfile,
    SameVerifierIdentity,
    DuplicateCompletenessClaim {
        node_id: String,
        relation: FaultDomainRelation,
    },
    UnknownCompletenessNode(String),
    MissingProfileNode {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
    },
    ProfileNodeKindMismatch {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
        observed: FaultDomainNodeKind,
    },
    MissingRelationCompleteness {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
        relation: FaultDomainRelation,
    },
    IncompleteRelation {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
        relation: FaultDomainRelation,
    },
    AncestryDepthLimited {
        verifier_ref: String,
        axis: IndependenceAxis,
        node_id: String,
    },
    SharedAncestor {
        axis: IndependenceAxis,
        node_id: String,
    },
    SharedGlobalAncestor(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationAwareAxisReport {
    pub axis: IndependenceAxis,
    pub required_relations: Vec<FaultDomainRelation>,
    pub left_entry_node: String,
    pub right_entry_node: String,
    pub shared_ancestors: Vec<String>,
    pub unresolved_nodes: Vec<String>,
    pub depth_limited: bool,
    pub status: IndependenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationAwareIndependenceReport {
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub graph_id: String,
    pub graph_digest: Option<String>,
    pub completeness_map_id: String,
    pub completeness_digest: Option<String>,
    pub left_verifier_ref: String,
    pub right_verifier_ref: String,
    pub status: IndependenceStatus,
    pub issues: Vec<RelationAwareIssue>,
    pub axis_reports: Vec<RelationAwareAxisReport>,
    pub global_shared_ancestors: Vec<String>,
}

impl RelationAwareIndependenceReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

impl RelationCompletenessMap {
    pub fn validate(&self) -> bool {
        if self.schema_version != RELATION_COMPLETENESS_SCHEMA_V1
            || !canonical_text(&self.map_id)
            || !digest_text(&self.graph_digest)
            || self.claims.is_empty()
            || self.claims.len() > MAX_COMPLETENESS_CLAIMS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }

        let mut seen = BTreeSet::new();
        for claim in &self.claims {
            if !canonical_text(&claim.node_id) || !valid_refs(&claim.evidence_refs) {
                return false;
            }
            if !seen.insert((claim.node_id.as_str(), claim.relation)) {
                return false;
            }
        }
        true
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMPLETENESS_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.map_id);
        push_field(&mut hasher, &self.graph_digest);

        let mut claims = self.claims.iter().collect::<Vec<_>>();
        claims.sort_by(|left, right| {
            (left.node_id.as_str(), left.relation)
                .cmp(&(right.node_id.as_str(), right.relation))
        });
        for claim in claims {
            push_field(&mut hasher, &claim.node_id);
            push_field(&mut hasher, relation_code(claim.relation));
            push_field(&mut hasher, claim.state.code());
            push_sorted_refs(&mut hasher, &claim.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl RelationAwareIndependencePolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != RELATION_AWARE_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.axis_requirements.is_empty()
            || self.max_ancestry_depth == 0
            || self.max_ancestry_depth > MAX_ANCESTRY_DEPTH
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }

        let mut axes = BTreeSet::new();
        for requirement in &self.axis_requirements {
            if requirement.required_relations.is_empty()
                || !axes.insert(requirement.axis)
            {
                return false;
            }
            let unique = requirement
                .required_relations
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            if unique.len() != requirement.required_relations.len() {
                return false;
            }
        }
        true
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);

        let mut requirements = self.axis_requirements.clone();
        requirements.sort_by_key(|requirement| requirement.axis);
        for mut requirement in requirements {
            push_field(&mut hasher, axis_code(requirement.axis));
            requirement.required_relations.sort();
            for relation in requirement.required_relations {
                push_field(&mut hasher, relation_code(relation));
            }
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

pub fn assess_relation_aware_independence(
    left: &VerifierFaultDomainProfile,
    right: &VerifierFaultDomainProfile,
    graph: &FaultDomainGraph,
    completeness: &RelationCompletenessMap,
    policy: &RelationAwareIndependencePolicy,
) -> RelationAwareIndependenceReport {
    let graph_digest = graph.canonical_digest();
    let completeness_digest = completeness.canonical_digest();
    let policy_digest = policy.canonical_digest();
    let mut issues = Vec::new();

    if graph.validate().is_err() {
        issues.push(RelationAwareIssue::InvalidGraph);
    }
    if !completeness.validate() {
        issues.push(RelationAwareIssue::InvalidCompletenessMap);
    }
    if !policy.validate() {
        issues.push(RelationAwareIssue::InvalidPolicy);
    }
    if !left.validate() {
        issues.push(RelationAwareIssue::InvalidLeftProfile);
    }
    if !right.validate() {
        issues.push(RelationAwareIssue::InvalidRightProfile);
    }
    if left.verifier_ref == right.verifier_ref {
        issues.push(RelationAwareIssue::SameVerifierIdentity);
    }
    if graph_digest.as_deref() != Some(completeness.graph_digest.as_str()) {
        issues.push(RelationAwareIssue::CompletenessGraphDigestMismatch);
    }

    let node_ids = graph
        .nodes
        .iter()
        .map(|node| node.node_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut completeness_index = BTreeMap::new();
    for claim in &completeness.claims {
        if !node_ids.contains(claim.node_id.as_str()) {
            issues.push(RelationAwareIssue::UnknownCompletenessNode(
                claim.node_id.clone(),
            ));
        }
        if completeness_index
            .insert((claim.node_id.as_str(), claim.relation), claim.state)
            .is_some()
        {
            issues.push(RelationAwareIssue::DuplicateCompletenessClaim {
                node_id: claim.node_id.clone(),
                relation: claim.relation,
            });
        }
    }

    if !issues.is_empty() {
        return invalid_report(
            left,
            right,
            graph,
            completeness,
            policy,
            graph_digest,
            completeness_digest,
            policy_digest,
            issues,
        );
    }

    let nodes = graph
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node.kind))
        .collect::<BTreeMap<_, _>>();

    let mut adjacency = BTreeMap::<(&str, FaultDomainRelation), Vec<&str>>::new();
    for edge in &graph.edges {
        adjacency
            .entry((edge.child_node_id.as_str(), edge.relation))
            .or_default()
            .push(edge.ancestor_node_id.as_str());
    }
    for values in adjacency.values_mut() {
        values.sort_unstable();
        values.dedup();
    }

    let mut axis_reports = Vec::new();
    let mut left_global = BTreeSet::new();
    let mut right_global = BTreeSet::new();

    let mut requirements = policy.axis_requirements.clone();
    requirements.sort_by_key(|requirement| requirement.axis);

    for requirement in requirements {
        let axis = requirement.axis;
        let left_entry = entry_node(axis, left).to_string();
        let right_entry = entry_node(axis, right).to_string();

        let Some(left_kind) = nodes.get(left_entry.as_str()) else {
            issues.push(RelationAwareIssue::MissingProfileNode {
                verifier_ref: left.verifier_ref.clone(),
                axis,
                node_id: left_entry,
            });
            continue;
        };
        let Some(right_kind) = nodes.get(right_entry.as_str()) else {
            issues.push(RelationAwareIssue::MissingProfileNode {
                verifier_ref: right.verifier_ref.clone(),
                axis,
                node_id: right_entry,
            });
            continue;
        };
        if *left_kind != expected_kind(axis) {
            issues.push(RelationAwareIssue::ProfileNodeKindMismatch {
                verifier_ref: left.verifier_ref.clone(),
                axis,
                node_id: left_entry,
                observed: *left_kind,
            });
            continue;
        }
        if *right_kind != expected_kind(axis) {
            issues.push(RelationAwareIssue::ProfileNodeKindMismatch {
                verifier_ref: right.verifier_ref.clone(),
                axis,
                node_id: right_entry,
                observed: *right_kind,
            });
            continue;
        }

        let required_relations = requirement
            .required_relations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let left_walk = walk(
            &left.verifier_ref,
            axis,
            &left_entry,
            &required_relations,
            &adjacency,
            &completeness_index,
            policy.max_ancestry_depth,
        );
        let right_walk = walk(
            &right.verifier_ref,
            axis,
            &right_entry,
            &required_relations,
            &adjacency,
            &completeness_index,
            policy.max_ancestry_depth,
        );

        issues.extend(left_walk.issues.iter().cloned());
        issues.extend(right_walk.issues.iter().cloned());
        left_global.extend(left_walk.visited.iter().cloned());
        right_global.extend(right_walk.visited.iter().cloned());

        let shared_ancestors = left_walk
            .visited
            .intersection(&right_walk.visited)
            .cloned()
            .collect::<Vec<_>>();
        for node_id in &shared_ancestors {
            issues.push(RelationAwareIssue::SharedAncestor {
                axis,
                node_id: node_id.clone(),
            });
        }

        let mut unresolved_nodes = left_walk.unresolved_nodes;
        unresolved_nodes.extend(right_walk.unresolved_nodes);
        unresolved_nodes.sort();
        unresolved_nodes.dedup();
        let depth_limited = left_walk.depth_limited || right_walk.depth_limited;
        let status = if !shared_ancestors.is_empty() {
            IndependenceStatus::Correlated
        } else if !unresolved_nodes.is_empty() || depth_limited {
            IndependenceStatus::Indeterminate
        } else {
            IndependenceStatus::Separated
        };

        let mut required_relations = requirement.required_relations;
        required_relations.sort();
        axis_reports.push(RelationAwareAxisReport {
            axis,
            required_relations,
            left_entry_node: left_entry,
            right_entry_node: right_entry,
            shared_ancestors,
            unresolved_nodes,
            depth_limited,
            status,
        });
    }

    if issues.iter().any(is_structural_issue) {
        return invalid_report(
            left,
            right,
            graph,
            completeness,
            policy,
            graph_digest,
            completeness_digest,
            policy_digest,
            issues,
        );
    }

    let global_shared_ancestors = if policy.require_global_separation {
        left_global
            .intersection(&right_global)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    for node_id in &global_shared_ancestors {
        issues.push(RelationAwareIssue::SharedGlobalAncestor(node_id.clone()));
    }

    let status = if !global_shared_ancestors.is_empty()
        || axis_reports
            .iter()
            .any(|report| report.status == IndependenceStatus::Correlated)
    {
        IndependenceStatus::Correlated
    } else if axis_reports
        .iter()
        .any(|report| report.status == IndependenceStatus::Indeterminate)
    {
        IndependenceStatus::Indeterminate
    } else {
        IndependenceStatus::Separated
    };

    RelationAwareIndependenceReport {
        policy_id: policy.policy_id.clone(),
        policy_digest,
        graph_id: graph.graph_id.clone(),
        graph_digest,
        completeness_map_id: completeness.map_id.clone(),
        completeness_digest,
        left_verifier_ref: left.verifier_ref.clone(),
        right_verifier_ref: right.verifier_ref.clone(),
        status,
        issues,
        axis_reports,
        global_shared_ancestors,
    }
}

#[derive(Debug)]
struct WalkResult {
    visited: BTreeSet<String>,
    unresolved_nodes: Vec<String>,
    depth_limited: bool,
    issues: Vec<RelationAwareIssue>,
}

fn walk(
    verifier_ref: &str,
    axis: IndependenceAxis,
    entry_node: &str,
    required_relations: &BTreeSet<FaultDomainRelation>,
    adjacency: &BTreeMap<(&str, FaultDomainRelation), Vec<&str>>,
    completeness: &BTreeMap<(&str, FaultDomainRelation), RelationCompletenessState>,
    max_depth: usize,
) -> WalkResult {
    let mut visited = BTreeSet::new();
    let mut unresolved_nodes = Vec::new();
    let mut issues = Vec::new();
    let mut depth_limited = false;
    let mut queue = VecDeque::from([(entry_node.to_string(), 0usize)]);

    while let Some((node_id, depth)) = queue.pop_front() {
        if !visited.insert(node_id.clone()) {
            continue;
        }

        for relation in required_relations {
            match completeness.get(&(node_id.as_str(), *relation)) {
                Some(RelationCompletenessState::Complete) => {}
                Some(RelationCompletenessState::Incomplete) => {
                    unresolved_nodes.push(node_id.clone());
                    issues.push(RelationAwareIssue::IncompleteRelation {
                        verifier_ref: verifier_ref.to_string(),
                        axis,
                        node_id: node_id.clone(),
                        relation: *relation,
                    });
                }
                None => {
                    unresolved_nodes.push(node_id.clone());
                    issues.push(RelationAwareIssue::MissingRelationCompleteness {
                        verifier_ref: verifier_ref.to_string(),
                        axis,
                        node_id: node_id.clone(),
                        relation: *relation,
                    });
                }
            }

            let ancestors = adjacency
                .get(&(node_id.as_str(), *relation))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if depth >= max_depth {
                if !ancestors.is_empty() {
                    depth_limited = true;
                    unresolved_nodes.push(node_id.clone());
                    issues.push(RelationAwareIssue::AncestryDepthLimited {
                        verifier_ref: verifier_ref.to_string(),
                        axis,
                        node_id: node_id.clone(),
                    });
                }
                continue;
            }
            for ancestor in ancestors {
                queue.push_back(((*ancestor).to_string(), depth + 1));
            }
        }
    }

    unresolved_nodes.sort();
    unresolved_nodes.dedup();
    WalkResult {
        visited,
        unresolved_nodes,
        depth_limited,
        issues,
    }
}

#[allow(clippy::too_many_arguments)]
fn invalid_report(
    left: &VerifierFaultDomainProfile,
    right: &VerifierFaultDomainProfile,
    graph: &FaultDomainGraph,
    completeness: &RelationCompletenessMap,
    policy: &RelationAwareIndependencePolicy,
    graph_digest: Option<String>,
    completeness_digest: Option<String>,
    policy_digest: Option<String>,
    issues: Vec<RelationAwareIssue>,
) -> RelationAwareIndependenceReport {
    RelationAwareIndependenceReport {
        policy_id: policy.policy_id.clone(),
        policy_digest,
        graph_id: graph.graph_id.clone(),
        graph_digest,
        completeness_map_id: completeness.map_id.clone(),
        completeness_digest,
        left_verifier_ref: left.verifier_ref.clone(),
        right_verifier_ref: right.verifier_ref.clone(),
        status: IndependenceStatus::Invalid,
        issues,
        axis_reports: Vec::new(),
        global_shared_ancestors: Vec::new(),
    }
}

fn is_structural_issue(issue: &RelationAwareIssue) -> bool {
    matches!(
        issue,
        RelationAwareIssue::InvalidGraph
            | RelationAwareIssue::InvalidCompletenessMap
            | RelationAwareIssue::CompletenessGraphDigestMismatch
            | RelationAwareIssue::InvalidPolicy
            | RelationAwareIssue::InvalidLeftProfile
            | RelationAwareIssue::InvalidRightProfile
            | RelationAwareIssue::SameVerifierIdentity
            | RelationAwareIssue::DuplicateCompletenessClaim { .. }
            | RelationAwareIssue::UnknownCompletenessNode(_)
            | RelationAwareIssue::MissingProfileNode { .. }
            | RelationAwareIssue::ProfileNodeKindMismatch { .. }
    )
}

fn entry_node<'a>(axis: IndependenceAxis, profile: &'a VerifierFaultDomainProfile) -> &'a str {
    match axis {
        IndependenceAxis::Organization => &profile.organization_domain,
        IndependenceAxis::ReviewProcess => &profile.review_process_domain,
        IndependenceAxis::Toolchain => &profile.toolchain_domain,
        IndependenceAxis::EvidenceSource => &profile.evidence_source_domain,
    }
}

fn expected_kind(axis: IndependenceAxis) -> FaultDomainNodeKind {
    match axis {
        IndependenceAxis::Organization => FaultDomainNodeKind::Organization,
        IndependenceAxis::ReviewProcess => FaultDomainNodeKind::ReviewProcess,
        IndependenceAxis::Toolchain => FaultDomainNodeKind::Toolchain,
        IndependenceAxis::EvidenceSource => FaultDomainNodeKind::EvidenceSource,
    }
}

fn axis_code(axis: IndependenceAxis) -> &'static str {
    match axis {
        IndependenceAxis::Organization => "organization",
        IndependenceAxis::ReviewProcess => "review-process",
        IndependenceAxis::Toolchain => "toolchain",
        IndependenceAxis::EvidenceSource => "evidence-source",
    }
}

fn relation_code(relation: FaultDomainRelation) -> &'static str {
    match relation {
        FaultDomainRelation::ControlledBy => "controlled-by",
        FaultDomainRelation::OperatedBy => "operated-by",
        FaultDomainRelation::DependsOn => "depends-on",
        FaultDomainRelation::DerivedFrom => "derived-from",
        FaultDomainRelation::GovernedBy => "governed-by",
    }
}

fn canonical_text(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed == value && value.len() <= MAX_TEXT_BYTES
}

fn digest_text(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_refs(refs: &[String]) -> bool {
    !refs.is_empty()
        && refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && refs.iter().collect::<BTreeSet<_>>().len() == refs.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.iter().collect::<Vec<_>>();
    refs.sort();
    for value in refs {
        push_field(hasher, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_independence_graph::{
        FaultDomainEdge, FaultDomainNode, FAULT_DOMAIN_GRAPH_SCHEMA_V1,
    };

    fn profile(verifier: &str, org: &str, tool: &str, source: &str) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: org.into(),
            review_process_domain: format!("process:{verifier}"),
            toolchain_domain: tool.into(),
            evidence_source_domain: source.into(),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn node(id: &str, kind: FaultDomainNodeKind) -> FaultDomainNode {
        FaultDomainNode {
            node_id: id.into(),
            kind,
            // Deliberately false: this child does not rely on the coarse V1 bit.
            lineage_complete: false,
            evidence_refs: vec![format!("node:{id}")],
        }
    }

    fn edge(child: &str, ancestor: &str, relation: FaultDomainRelation) -> FaultDomainEdge {
        FaultDomainEdge {
            child_node_id: child.into(),
            ancestor_node_id: ancestor.into(),
            relation,
            evidence_refs: vec![format!("edge:{child}:{ancestor}")],
        }
    }

    fn graph(nodes: Vec<FaultDomainNode>, edges: Vec<FaultDomainEdge>) -> FaultDomainGraph {
        FaultDomainGraph {
            schema_version: FAULT_DOMAIN_GRAPH_SCHEMA_V1.into(),
            graph_id: "graph:v1".into(),
            nodes,
            edges,
            evidence_refs: vec!["review:graph".into()],
        }
    }

    fn completeness(
        graph: &FaultDomainGraph,
        claims: Vec<RelationCompletenessClaim>,
    ) -> RelationCompletenessMap {
        RelationCompletenessMap {
            schema_version: RELATION_COMPLETENESS_SCHEMA_V1.into(),
            map_id: "completeness:v1".into(),
            graph_digest: graph.canonical_digest().unwrap(),
            claims,
            evidence_refs: vec!["review:completeness".into()],
        }
    }

    fn claim(
        node_id: &str,
        relation: FaultDomainRelation,
        state: RelationCompletenessState,
    ) -> RelationCompletenessClaim {
        RelationCompletenessClaim {
            node_id: node_id.into(),
            relation,
            state,
            evidence_refs: vec![format!("claim:{node_id}:{}", relation_code(relation))],
        }
    }

    fn policy(
        requirements: Vec<AxisRelationRequirement>,
        global: bool,
        depth: usize,
    ) -> RelationAwareIndependencePolicy {
        RelationAwareIndependencePolicy {
            schema_version: RELATION_AWARE_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:v1".into(),
            axis_requirements: requirements,
            require_global_separation: global,
            max_ancestry_depth: depth,
            evidence_refs: vec!["review:policy".into()],
        }
    }

    #[test]
    fn shared_control_root_is_correlated() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
                node("control:x", FaultDomainNodeKind::ControlPlane),
            ],
            vec![
                edge("org:a", "control:x", FaultDomainRelation::ControlledBy),
                edge("org:b", "control:x", FaultDomainRelation::ControlledBy),
            ],
        );
        let completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("control:x", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
            ],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Correlated);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn irrelevant_incomplete_relation_does_not_poison_narrow_policy() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
            ],
            vec![],
        );
        let completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:a", FaultDomainRelation::DependsOn, RelationCompletenessState::Incomplete),
                claim("org:b", FaultDomainRelation::DependsOn, RelationCompletenessState::Incomplete),
            ],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Separated);
    }

    #[test]
    fn missing_required_completeness_is_indeterminate() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
            ],
            vec![],
        );
        let completeness = completeness(
            &graph,
            vec![claim(
                "org:a",
                FaultDomainRelation::ControlledBy,
                RelationCompletenessState::Complete,
            )],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Indeterminate);
    }

    #[test]
    fn required_incomplete_relation_is_indeterminate() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
            ],
            vec![],
        );
        let completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Incomplete),
            ],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Indeterminate);
    }

    #[test]
    fn known_correlation_dominates_incomplete_elsewhere() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
                node("control:x", FaultDomainNodeKind::ControlPlane),
            ],
            vec![
                edge("org:a", "control:x", FaultDomainRelation::ControlledBy),
                edge("org:b", "control:x", FaultDomainRelation::ControlledBy),
            ],
        );
        let completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Incomplete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Incomplete),
                claim("control:x", FaultDomainRelation::ControlledBy, RelationCompletenessState::Incomplete),
            ],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Correlated);
    }

    #[test]
    fn graph_digest_mismatch_is_invalid() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
            ],
            vec![],
        );
        let mut completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
            ],
        );
        completeness.graph_digest = format!("blake3:{}", "0".repeat(64));
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            8,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Invalid);
    }

    #[test]
    fn depth_limit_is_indeterminate() {
        let left = profile("a", "org:a", "tool:a", "source:a");
        let right = profile("b", "org:b", "tool:b", "source:b");
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
                node("control:a", FaultDomainNodeKind::ControlPlane),
                node("root:a", FaultDomainNodeKind::GovernanceRoot),
            ],
            vec![
                edge("org:a", "control:a", FaultDomainRelation::ControlledBy),
                edge("control:a", "root:a", FaultDomainRelation::ControlledBy),
            ],
        );
        let completeness = completeness(
            &graph,
            vec![
                claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("control:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
                claim("root:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete),
            ],
        );
        let policy = policy(
            vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            false,
            1,
        );
        let report = assess_relation_aware_independence(
            &left,
            &right,
            &graph,
            &completeness,
            &policy,
        );
        assert_eq!(report.status, IndependenceStatus::Indeterminate);
    }

    #[test]
    fn completeness_digest_is_order_independent() {
        let graph = graph(
            vec![
                node("org:a", FaultDomainNodeKind::Organization),
                node("org:b", FaultDomainNodeKind::Organization),
            ],
            vec![],
        );
        let a = claim("org:a", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete);
        let b = claim("org:b", FaultDomainRelation::ControlledBy, RelationCompletenessState::Complete);
        let left = completeness(&graph, vec![a.clone(), b.clone()]);
        let right = completeness(&graph, vec![b, a]);
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn policy_digest_is_order_independent() {
        let a = AxisRelationRequirement {
            axis: IndependenceAxis::Organization,
            required_relations: vec![FaultDomainRelation::GovernedBy, FaultDomainRelation::ControlledBy],
        };
        let b = AxisRelationRequirement {
            axis: IndependenceAxis::Toolchain,
            required_relations: vec![FaultDomainRelation::DependsOn, FaultDomainRelation::OperatedBy],
        };
        let left = policy(vec![a.clone(), b.clone()], true, 8);
        let right = policy(vec![b, a], true, 8);
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }
}
