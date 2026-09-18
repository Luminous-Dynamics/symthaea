// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed acyclic provenance for scientific research artifacts.
//!
//! `ProvenanceGraph` is validated at construction and intentionally is not
//! directly deserializable. Recovery/import must rebuild it from explicit node
//! and edge declarations. Scientific semantic relations such as `supports`,
//! `contradicts`, and `alternative-to` belong in a separate relation graph.

use crate::{
    AuthorityFacet, AuthorityLevel, AuthorityProfile, EvidenceRecord, FramedDigest, ResearchId,
    Sha256Digest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const GRAPH_DOMAIN: &str = "symthaea.science-provenance-graph.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ProvenanceNodeKind {
    SourceArtifact,
    RawData,
    Calibration,
    Preprocessing,
    Analysis,
    Execution,
    Observation,
    Measurement,
    Evidence,
    ClaimEvaluation,
    Qualification,
    Other,
}

impl ProvenanceNodeKind {
    const fn can_anchor_lineage(self) -> bool {
        matches!(self, Self::SourceArtifact | Self::RawData | Self::Observation)
    }
}

/// Immutable provenance node. `qualification_sha256` is a lineage reference,
/// not a qualification capability and grants no authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceNode {
    node_id: ResearchId,
    kind: ProvenanceNodeKind,
    subject_sha256: Sha256Digest,
    artifact_sha256: Sha256Digest,
    qualification_sha256: Option<Sha256Digest>,
    authority: AuthorityProfile,
}

impl ProvenanceNode {
    pub fn new(
        node_id: ResearchId,
        kind: ProvenanceNodeKind,
        subject_sha256: Sha256Digest,
        artifact_sha256: Sha256Digest,
        qualification_sha256: Option<Sha256Digest>,
        authority: AuthorityProfile,
    ) -> Self {
        Self {
            node_id,
            kind,
            subject_sha256,
            artifact_sha256,
            qualification_sha256,
            authority,
        }
    }

    pub fn from_evidence(record: &EvidenceRecord) -> Self {
        Self::new(
            record.evidence_id().clone(),
            ProvenanceNodeKind::Evidence,
            record.subject_sha256().clone(),
            record.artifact_sha256().clone(),
            record.qualification_sha256().cloned(),
            record.authority().clone(),
        )
    }

    pub fn node_id(&self) -> &ResearchId {
        &self.node_id
    }

    pub fn kind(&self) -> ProvenanceNodeKind {
        self.kind
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn artifact_sha256(&self) -> &Sha256Digest {
        &self.artifact_sha256
    }

    pub fn qualification_sha256(&self) -> Option<&Sha256Digest> {
        self.qualification_sha256.as_ref()
    }

    pub fn authority(&self) -> &AuthorityProfile {
        &self.authority
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ProvenanceEdgeKind {
    DerivedFrom,
    Consumes,
    CalibratedBy,
    Evaluates,
    Qualifies,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    pub source: ResearchId,
    pub target: ResearchId,
    pub kind: ProvenanceEdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceIssue {
    EmptyGraph,
    DuplicateNode { node_id: ResearchId },
    DuplicateEdge { edge: ProvenanceEdge },
    UnknownSource { node_id: ResearchId },
    UnknownTarget { node_id: ResearchId },
    SelfEdge { node_id: ResearchId },
    CycleDetected,
    EvaluateTargetMustBeClaimEvaluation { node_id: ResearchId },
    QualifyTargetMustBeQualification { node_id: ResearchId },
    AuthorityEdgeSubjectMismatch {
        source: ResearchId,
        target: ResearchId,
    },
    OrphanAuthorityNode { node_id: ResearchId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceGraph {
    nodes: BTreeMap<ResearchId, ProvenanceNode>,
    edges: BTreeSet<ProvenanceEdge>,
}

impl ProvenanceGraph {
    pub fn new(
        nodes: impl IntoIterator<Item = ProvenanceNode>,
        edges: impl IntoIterator<Item = ProvenanceEdge>,
    ) -> Result<Self, Vec<ProvenanceIssue>> {
        let mut issues = Vec::new();
        let mut node_map = BTreeMap::new();
        for node in nodes {
            let id = node.node_id.clone();
            if node_map.insert(id.clone(), node).is_some() {
                issues.push(ProvenanceIssue::DuplicateNode { node_id: id });
            }
        }

        let mut edge_set = BTreeSet::new();
        for edge in edges {
            if !edge_set.insert(edge.clone()) {
                issues.push(ProvenanceIssue::DuplicateEdge { edge });
            }
        }

        let graph = Self {
            nodes: node_map,
            edges: edge_set,
        };
        issues.extend(graph.validate());
        if issues.is_empty() {
            Ok(graph)
        } else {
            Err(issues)
        }
    }

    pub fn validate(&self) -> Vec<ProvenanceIssue> {
        let mut issues = Vec::new();
        if self.nodes.is_empty() {
            issues.push(ProvenanceIssue::EmptyGraph);
            return issues;
        }

        for edge in &self.edges {
            let Some(source) = self.nodes.get(&edge.source) else {
                issues.push(ProvenanceIssue::UnknownSource {
                    node_id: edge.source.clone(),
                });
                continue;
            };
            let Some(target) = self.nodes.get(&edge.target) else {
                issues.push(ProvenanceIssue::UnknownTarget {
                    node_id: edge.target.clone(),
                });
                continue;
            };
            if edge.source == edge.target {
                issues.push(ProvenanceIssue::SelfEdge {
                    node_id: edge.source.clone(),
                });
            }

            match edge.kind {
                ProvenanceEdgeKind::Evaluates => {
                    if target.kind != ProvenanceNodeKind::ClaimEvaluation {
                        issues.push(ProvenanceIssue::EvaluateTargetMustBeClaimEvaluation {
                            node_id: target.node_id.clone(),
                        });
                    }
                    check_authority_subject(source, target, &mut issues);
                }
                ProvenanceEdgeKind::Qualifies => {
                    if target.kind != ProvenanceNodeKind::Qualification {
                        issues.push(ProvenanceIssue::QualifyTargetMustBeQualification {
                            node_id: target.node_id.clone(),
                        });
                    }
                    check_authority_subject(source, target, &mut issues);
                }
                ProvenanceEdgeKind::DerivedFrom
                | ProvenanceEdgeKind::Consumes
                | ProvenanceEdgeKind::CalibratedBy => {
                    // Cross-subject dependency is explicit and permitted here.
                }
            }
        }

        if self.has_cycle() {
            issues.push(ProvenanceIssue::CycleDetected);
            return issues;
        }

        for node in self.nodes.values() {
            let requires_root = !node.authority.is_empty()
                || matches!(
                    node.kind,
                    ProvenanceNodeKind::Evidence
                        | ProvenanceNodeKind::ClaimEvaluation
                        | ProvenanceNodeKind::Qualification
                );
            if requires_root && !node.kind.can_anchor_lineage() {
                let anchored = self.ancestors_internal(&node.node_id).iter().any(|id| {
                    self.nodes
                        .get(id)
                        .is_some_and(|ancestor| ancestor.kind.can_anchor_lineage())
                });
                if !anchored {
                    issues.push(ProvenanceIssue::OrphanAuthorityNode {
                        node_id: node.node_id.clone(),
                    });
                }
            }
        }
        issues
    }

    pub fn node(&self, node_id: &ResearchId) -> Option<&ProvenanceNode> {
        self.nodes.get(node_id)
    }

    pub fn ancestors(&self, node_id: &ResearchId) -> Option<BTreeSet<ResearchId>> {
        self.nodes
            .contains_key(node_id)
            .then(|| self.ancestors_internal(node_id))
    }

    /// Exposes common ancestor artifacts so different downstream outputs cannot
    /// hide correlated data/code/calibration roots behind different identities.
    pub fn shared_ancestor_artifacts(
        &self,
        left: &ResearchId,
        right: &ResearchId,
    ) -> Option<BTreeSet<Sha256Digest>> {
        let left = self.ancestors(left)?;
        let right = self.ancestors(right)?;
        Some(
            left.intersection(&right)
                .filter_map(|id| self.nodes.get(id))
                .map(|node| node.artifact_sha256.clone())
                .collect(),
        )
    }

    pub fn graph_sha256(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(GRAPH_DOMAIN);
        for node in self.nodes.values() {
            digest.text(node.node_id.as_str());
            digest.text(node_kind_tag(node.kind));
            digest.text(node.subject_sha256.as_str());
            digest.text(node.artifact_sha256.as_str());
            digest.text(
                node.qualification_sha256
                    .as_ref()
                    .map_or("", Sha256Digest::as_str),
            );
            for facet in AuthorityFacet::ALL {
                digest.text(authority_facet_tag(facet));
                digest.text(authority_level_tag(node.authority.get(facet)));
            }
        }
        for edge in &self.edges {
            digest.text(edge.source.as_str());
            digest.text(edge.target.as_str());
            digest.text(edge_kind_tag(edge.kind));
        }
        digest.digest()
    }

    fn ancestors_internal(&self, node_id: &ResearchId) -> BTreeSet<ResearchId> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![node_id.clone()];
        while let Some(current) = stack.pop() {
            for edge in self.edges.iter().filter(|edge| edge.target == current) {
                if seen.insert(edge.source.clone()) {
                    stack.push(edge.source.clone());
                }
            }
        }
        seen
    }

    fn has_cycle(&self) -> bool {
        let mut indegree: BTreeMap<ResearchId, usize> =
            self.nodes.keys().cloned().map(|id| (id, 0)).collect();
        let mut outgoing: BTreeMap<ResearchId, Vec<ResearchId>> = BTreeMap::new();
        for edge in &self.edges {
            if self.nodes.contains_key(&edge.source) && self.nodes.contains_key(&edge.target) {
                *indegree.entry(edge.target.clone()).or_default() += 1;
                outgoing
                    .entry(edge.source.clone())
                    .or_default()
                    .push(edge.target.clone());
            }
        }

        let mut queue: VecDeque<ResearchId> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
            .collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(children) = outgoing.get(&node) {
                for child in children {
                    let degree = indegree
                        .get_mut(child)
                        .expect("outgoing nodes were validated against the node map");
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(child.clone());
                    }
                }
            }
        }
        visited != self.nodes.len()
    }
}

fn check_authority_subject(
    source: &ProvenanceNode,
    target: &ProvenanceNode,
    issues: &mut Vec<ProvenanceIssue>,
) {
    if source.subject_sha256 != target.subject_sha256 {
        issues.push(ProvenanceIssue::AuthorityEdgeSubjectMismatch {
            source: source.node_id.clone(),
            target: target.node_id.clone(),
        });
    }
}

fn node_kind_tag(kind: ProvenanceNodeKind) -> &'static str {
    match kind {
        ProvenanceNodeKind::SourceArtifact => "source-artifact",
        ProvenanceNodeKind::RawData => "raw-data",
        ProvenanceNodeKind::Calibration => "calibration",
        ProvenanceNodeKind::Preprocessing => "preprocessing",
        ProvenanceNodeKind::Analysis => "analysis",
        ProvenanceNodeKind::Execution => "execution",
        ProvenanceNodeKind::Observation => "observation",
        ProvenanceNodeKind::Measurement => "measurement",
        ProvenanceNodeKind::Evidence => "evidence",
        ProvenanceNodeKind::ClaimEvaluation => "claim-evaluation",
        ProvenanceNodeKind::Qualification => "qualification",
        ProvenanceNodeKind::Other => "other",
    }
}

fn edge_kind_tag(kind: ProvenanceEdgeKind) -> &'static str {
    match kind {
        ProvenanceEdgeKind::DerivedFrom => "derived-from",
        ProvenanceEdgeKind::Consumes => "consumes",
        ProvenanceEdgeKind::CalibratedBy => "calibrated-by",
        ProvenanceEdgeKind::Evaluates => "evaluates",
        ProvenanceEdgeKind::Qualifies => "qualifies",
    }
}

fn authority_facet_tag(facet: AuthorityFacet) -> &'static str {
    match facet {
        AuthorityFacet::Provenance => "provenance",
        AuthorityFacet::Execution => "execution",
        AuthorityFacet::Empirical => "empirical",
        AuthorityFacet::Causal => "causal",
        AuthorityFacet::Formal => "formal",
        AuthorityFacet::Replication => "replication",
        AuthorityFacet::Independence => "independence",
    }
}

fn authority_level_tag(level: AuthorityLevel) -> &'static str {
    match level {
        AuthorityLevel::None => "none",
        AuthorityLevel::Declared => "declared",
        AuthorityLevel::Bound => "bound",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn node(name: &str, kind: ProvenanceNodeKind, subject: &Sha256Digest) -> ProvenanceNode {
        ProvenanceNode::new(
            id(name),
            kind,
            subject.clone(),
            digest(name),
            None,
            AuthorityProfile::empty(),
        )
    }

    fn edge(source: &str, target: &str, kind: ProvenanceEdgeKind) -> ProvenanceEdge {
        ProvenanceEdge {
            source: id(source),
            target: id(target),
            kind,
        }
    }

    #[test]
    fn cycle_fails_closed() {
        let subject = digest("subject");
        let result = ProvenanceGraph::new(
            [
                node("A", ProvenanceNodeKind::SourceArtifact, &subject),
                node("B", ProvenanceNodeKind::Analysis, &subject),
            ],
            [
                edge("A", "B", ProvenanceEdgeKind::DerivedFrom),
                edge("B", "A", ProvenanceEdgeKind::DerivedFrom),
            ],
        );
        assert!(result.unwrap_err().contains(&ProvenanceIssue::CycleDetected));
    }

    #[test]
    fn authority_edge_cannot_substitute_subject() {
        let a = digest("subject-a");
        let b = digest("subject-b");
        let result = ProvenanceGraph::new(
            [
                node("ROOT", ProvenanceNodeKind::SourceArtifact, &a),
                node("E", ProvenanceNodeKind::Evidence, &a),
                node("C", ProvenanceNodeKind::ClaimEvaluation, &b),
            ],
            [
                edge("ROOT", "E", ProvenanceEdgeKind::DerivedFrom),
                edge("E", "C", ProvenanceEdgeKind::Evaluates),
            ],
        );
        assert!(result.unwrap_err().iter().any(|issue| matches!(
            issue,
            ProvenanceIssue::AuthorityEdgeSubjectMismatch { .. }
        )));
    }

    #[test]
    fn cross_subject_data_consumption_is_allowed() {
        let data_subject = digest("dataset-subject");
        let claim_subject = digest("claim-subject");
        let graph = ProvenanceGraph::new(
            [
                node("DATA", ProvenanceNodeKind::RawData, &data_subject),
                node("ANALYSIS", ProvenanceNodeKind::Analysis, &claim_subject),
            ],
            [edge("DATA", "ANALYSIS", ProvenanceEdgeKind::Consumes)],
        );
        assert!(graph.is_ok());
    }

    #[test]
    fn shared_ancestor_exposes_correlated_root() {
        let subject = digest("subject");
        let graph = ProvenanceGraph::new(
            [
                node("DATA", ProvenanceNodeKind::RawData, &subject),
                node("IMPL-A", ProvenanceNodeKind::Execution, &subject),
                node("IMPL-B", ProvenanceNodeKind::Execution, &subject),
            ],
            [
                edge("DATA", "IMPL-A", ProvenanceEdgeKind::Consumes),
                edge("DATA", "IMPL-B", ProvenanceEdgeKind::Consumes),
            ],
        )
        .unwrap();
        let shared = graph
            .shared_ancestor_artifacts(&id("IMPL-A"), &id("IMPL-B"))
            .unwrap();
        assert_eq!(shared, BTreeSet::from([digest("DATA")]));
    }

    #[test]
    fn authority_node_requires_real_root() {
        let subject = digest("subject");
        let isolated = ProvenanceNode::new(
            id("EXEC"),
            ProvenanceNodeKind::Execution,
            subject,
            digest("exec"),
            None,
            AuthorityProfile::empty().with(AuthorityFacet::Execution, AuthorityLevel::Bound),
        );
        let result = ProvenanceGraph::new([isolated], []);
        assert!(result.unwrap_err().iter().any(|issue| matches!(
            issue,
            ProvenanceIssue::OrphanAuthorityNode { .. }
        )));
    }

    #[test]
    fn qualification_digest_is_only_a_graph_reference() {
        let subject = digest("subject");
        let qualification = digest("qualification-reference");
        let node = ProvenanceNode::new(
            id("Q"),
            ProvenanceNodeKind::Qualification,
            subject,
            digest("artifact"),
            Some(qualification.clone()),
            AuthorityProfile::empty().with(AuthorityFacet::Provenance, AuthorityLevel::Bound),
        );
        assert_eq!(node.qualification_sha256(), Some(&qualification));
        assert_eq!(node.authority().get(AuthorityFacet::Provenance), AuthorityLevel::Bound);
    }

    #[test]
    fn graph_digest_is_order_independent_for_same_graph() {
        let subject = digest("subject");
        let a = node("A", ProvenanceNodeKind::SourceArtifact, &subject);
        let b = node("B", ProvenanceNodeKind::Analysis, &subject);
        let e = edge("A", "B", ProvenanceEdgeKind::DerivedFrom);
        let left = ProvenanceGraph::new([a.clone(), b.clone()], [e.clone()]).unwrap();
        let right = ProvenanceGraph::new([b, a], [e]).unwrap();
        assert_eq!(left.graph_sha256(), right.graph_sha256());
    }
}
