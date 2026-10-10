// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-neutral scientific discovery graph.
//!
//! Records epistemic lineage without declaring scientific truth. Computational
//! outputs, observations, replications, failures, and official-criterion
//! evidence remain distinct node kinds.
//!
//! Model agreement is not independent corroboration: every causal model may
//! carry a lineage root, and shared ancestry remains visible.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::prospective::ProspectivePredictionCommitment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Quest,
    Observation,
    MechanismHypothesis,
    CausalModel,
    Candidate,
    ProspectiveCommitment,
    IndependentReplication,
    OfficialCriterionEvidence,
    Failure,
    NullResult,
    Contradiction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    DerivedFrom,
    Predicts,
    TestedBy,
    Supports,
    Contradicts,
    Replicates,
    SatisfiesCriterion,
    Supersedes,
    SharesModelAncestry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    pub actor_id: String,
    pub artifact_digest: String,
    pub input_digest: String,
    pub model_lineage: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
}

impl ProvenanceRef {
    pub fn new(
        actor_id: impl Into<String>,
        artifact_digest: impl Into<String>,
        input_digest: impl Into<String>,
    ) -> Result<Self, GraphError> {
        let value = Self {
            actor_id: actor_id.into(),
            artifact_digest: artifact_digest.into(),
            input_digest: input_digest.into(),
            model_lineage: None,
            knowledge_cutoff: None,
            exposure_cutoff: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_model_lineage(mut self, lineage: impl Into<String>) -> Self {
        self.model_lineage = Some(lineage.into());
        self
    }

    pub fn with_cutoffs(
        mut self,
        knowledge: impl Into<String>,
        exposure: impl Into<String>,
    ) -> Result<Self, GraphError> {
        self.knowledge_cutoff = Some(knowledge.into());
        self.exposure_cutoff = Some(exposure.into());
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), GraphError> {
        for (name, value) in [
            ("actor_id", &self.actor_id),
            ("artifact_digest", &self.artifact_digest),
            ("input_digest", &self.input_digest),
        ] {
            if value.trim().is_empty() {
                return Err(GraphError::MissingField(name));
            }
        }
        match (&self.knowledge_cutoff, &self.exposure_cutoff) {
            (Some(k), Some(e)) if e < k => Err(GraphError::ExposureBeforeKnowledgeCutoff),
            (Some(_), None) | (None, Some(_)) => Err(GraphError::IncompleteCutoffPair),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryNode {
    pub id: String,
    pub kind: NodeKind,
    pub provenance: ProvenanceRef,
    /// Stable digest of the domain payload. The graph does not interpret it.
    pub payload_digest: String,
}

impl DiscoveryNode {
    pub fn new(
        id: impl Into<String>,
        kind: NodeKind,
        provenance: ProvenanceRef,
        payload_digest: impl Into<String>,
    ) -> Result<Self, GraphError> {
        let value = Self {
            id: id.into(),
            kind,
            provenance,
            payload_digest: payload_digest.into(),
        };
        if value.id.trim().is_empty() {
            return Err(GraphError::MissingField("node_id"));
        }
        if value.payload_digest.trim().is_empty() {
            return Err(GraphError::MissingField("payload_digest"));
        }
        value.provenance.validate()?;
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DiscoveryEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScientificDiscoveryGraph {
    nodes: BTreeMap<String, DiscoveryNode>,
    edges: BTreeSet<DiscoveryEdge>,
}

impl ScientificDiscoveryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: DiscoveryNode) -> Result<(), GraphError> {
        if self.nodes.contains_key(&node.id) {
            return Err(GraphError::DuplicateNode(node.id));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub fn add_prospective_commitment(
        &mut self,
        commitment: &ProspectivePredictionCommitment,
        provenance: ProvenanceRef,
        payload_digest: impl Into<String>,
    ) -> Result<(), GraphError> {
        self.add_node(DiscoveryNode::new(
            commitment.event_id(),
            NodeKind::ProspectiveCommitment,
            provenance,
            payload_digest,
        )?)
    }

    pub fn add_edge(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        kind: EdgeKind,
    ) -> Result<(), GraphError> {
        let from = from.into();
        let to = to.into();
        if from == to {
            return Err(GraphError::SelfLoop(from));
        }
        let source = self.nodes.get(&from)
            .ok_or_else(|| GraphError::UnknownNode(from.clone()))?;
        let target = self.nodes.get(&to)
            .ok_or_else(|| GraphError::UnknownNode(to.clone()))?;
        if !edge_allowed(source.kind, target.kind, kind) {
            return Err(GraphError::InvalidEdge {
                from: source.kind,
                to: target.kind,
                edge: kind,
            });
        }
        self.edges.insert(DiscoveryEdge { from, to, kind });
        Ok(())
    }

    pub fn node(&self, id: &str) -> Option<&DiscoveryNode> {
        self.nodes.get(id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &DiscoveryNode> {
        self.nodes.values()
    }

    pub fn edges(&self) -> impl Iterator<Item = &DiscoveryEdge> {
        self.edges.iter()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn incoming(&self, id: &str) -> Vec<&DiscoveryEdge> {
        self.edges.iter().filter(|edge| edge.to == id).collect()
    }

    pub fn outgoing(&self, id: &str) -> Vec<&DiscoveryEdge> {
        self.edges.iter().filter(|edge| edge.from == id).collect()
    }

    /// Return distinct causal-model lineage roots reachable upstream.
    /// Shared roots are deliberately counted once.
    pub fn model_lineage_roots(&self, node_id: &str) -> Result<BTreeSet<String>, GraphError> {
        if !self.nodes.contains_key(node_id) {
            return Err(GraphError::UnknownNode(node_id.to_string()));
        }
        let mut roots = BTreeSet::new();
        let mut stack = vec![node_id.to_string()];
        let mut visited = BTreeSet::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id.clone()) {
                continue;
            }
            let node = self.nodes.get(&id)
                .ok_or_else(|| GraphError::UnknownNode(id.clone()))?;
            if node.kind == NodeKind::CausalModel {
                if let Some(lineage) = &node.provenance.model_lineage {
                    roots.insert(lineage.clone());
                }
            }
            for edge in self.incoming(&id) {
                if matches!(
                    edge.kind,
                    EdgeKind::DerivedFrom
                        | EdgeKind::Predicts
                        | EdgeKind::Supports
                        | EdgeKind::Contradicts
                        | EdgeKind::TestedBy
                        | EdgeKind::SharesModelAncestry
                ) {
                    stack.push(edge.from.clone());
                }
            }
        }
        Ok(roots)
    }

    /// Structural inventory only; this is not a scientific confidence score.
    pub fn evidence_kinds(&self) -> BTreeSet<NodeKind> {
        self.nodes.values().filter_map(|node| match node.kind {
            NodeKind::Observation
            | NodeKind::IndependentReplication
            | NodeKind::OfficialCriterionEvidence
            | NodeKind::Failure
            | NodeKind::NullResult
            | NodeKind::Contradiction => Some(node.kind),
            _ => None,
        }).collect()
    }
}

fn edge_allowed(from: NodeKind, to: NodeKind, edge: EdgeKind) -> bool {
    match edge {
        EdgeKind::DerivedFrom => matches!(
            (from, to),
            (NodeKind::MechanismHypothesis, NodeKind::Quest)
                | (NodeKind::CausalModel, NodeKind::MechanismHypothesis)
                | (NodeKind::Candidate, NodeKind::MechanismHypothesis)
                | (NodeKind::ProspectiveCommitment, NodeKind::Candidate)
                | (NodeKind::ProspectiveCommitment, NodeKind::MechanismHypothesis)
                | (NodeKind::Observation, NodeKind::ProspectiveCommitment)
                | (NodeKind::Failure, NodeKind::ProspectiveCommitment)
                | (NodeKind::NullResult, NodeKind::ProspectiveCommitment)
                | (NodeKind::Contradiction, NodeKind::Observation)
                | (NodeKind::Contradiction, NodeKind::CausalModel)
                | (NodeKind::IndependentReplication, NodeKind::Observation)
                | (NodeKind::OfficialCriterionEvidence, NodeKind::IndependentReplication)
                | (NodeKind::OfficialCriterionEvidence, NodeKind::Observation)
        ),
        EdgeKind::Predicts => matches!(
            (from, to),
            (NodeKind::CausalModel, NodeKind::Candidate)
                | (NodeKind::CausalModel, NodeKind::Observation)
                | (NodeKind::MechanismHypothesis, NodeKind::Candidate)
        ),
        EdgeKind::TestedBy => matches!(
            (from, to),
            (NodeKind::Candidate, NodeKind::Observation)
                | (NodeKind::Candidate, NodeKind::Failure)
                | (NodeKind::Candidate, NodeKind::NullResult)
                | (NodeKind::CausalModel, NodeKind::Observation)
                | (NodeKind::MechanismHypothesis, NodeKind::Observation)
        ),
        EdgeKind::Supports => matches!(
            (from, to),
            (NodeKind::Observation, NodeKind::MechanismHypothesis)
                | (NodeKind::Observation, NodeKind::CausalModel)
                | (NodeKind::IndependentReplication, NodeKind::MechanismHypothesis)
                | (NodeKind::IndependentReplication, NodeKind::CausalModel)
                | (NodeKind::OfficialCriterionEvidence, NodeKind::Candidate)
        ),
        EdgeKind::Contradicts => matches!(
            (from, to),
            (NodeKind::Observation, NodeKind::MechanismHypothesis)
                | (NodeKind::Observation, NodeKind::CausalModel)
                | (NodeKind::Failure, NodeKind::Candidate)
                | (NodeKind::Failure, NodeKind::MechanismHypothesis)
                | (NodeKind::Failure, NodeKind::CausalModel)
                | (NodeKind::NullResult, NodeKind::Candidate)
                | (NodeKind::NullResult, NodeKind::MechanismHypothesis)
                | (NodeKind::NullResult, NodeKind::CausalModel)
                | (NodeKind::Contradiction, NodeKind::MechanismHypothesis)
                | (NodeKind::Contradiction, NodeKind::CausalModel)
        ),
        EdgeKind::Replicates => matches!(
            (from, to),
            (NodeKind::IndependentReplication, NodeKind::Observation)
                | (NodeKind::IndependentReplication, NodeKind::OfficialCriterionEvidence)
        ),
        EdgeKind::SatisfiesCriterion => matches!(
            (from, to),
            (NodeKind::OfficialCriterionEvidence, NodeKind::Candidate)
        ),
        EdgeKind::Supersedes => matches!(
            (from, to),
            (NodeKind::ProspectiveCommitment, NodeKind::ProspectiveCommitment)
        ),
        EdgeKind::SharesModelAncestry => matches!(
            (from, to),
            (NodeKind::CausalModel, NodeKind::CausalModel)
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    MissingField(&'static str),
    DuplicateNode(String),
    UnknownNode(String),
    SelfLoop(String),
    InvalidEdge { from: NodeKind, to: NodeKind, edge: EdgeKind },
    ExposureBeforeKnowledgeCutoff,
    IncompleteCutoffPair,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::DuplicateNode(id) => write!(f, "duplicate node: {id}"),
            Self::UnknownNode(id) => write!(f, "unknown node: {id}"),
            Self::SelfLoop(id) => write!(f, "self-loop is not permitted: {id}"),
            Self::InvalidEdge { from, to, edge } => {
                write!(f, "invalid {edge:?} edge from {from:?} to {to:?}")
            }
            Self::ExposureBeforeKnowledgeCutoff => {
                write!(f, "exposure cutoff must not precede knowledge cutoff")
            }
            Self::IncompleteCutoffPair => {
                write!(f, "knowledge and exposure cutoffs must be supplied together")
            }
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(actor: &str) -> ProvenanceRef {
        ProvenanceRef::new(actor, "sha256:artifact", "sha256:input").unwrap()
    }

    fn node(id: &str, kind: NodeKind) -> DiscoveryNode {
        DiscoveryNode::new(id, kind, provenance("actor:test"), "sha256:payload").unwrap()
    }

    #[test]
    fn graph_preserves_competing_model_ancestry() {
        let mut graph = ScientificDiscoveryGraph::new();
        let p1 = provenance("model:a").with_model_lineage("lineage:root-a");
        let p2 = provenance("model:b").with_model_lineage("lineage:root-b");

        graph.add_node(DiscoveryNode::new("m1", NodeKind::CausalModel, p1, "sha256:m1").unwrap()).unwrap();
        graph.add_node(DiscoveryNode::new("m2", NodeKind::CausalModel, p2, "sha256:m2").unwrap()).unwrap();
        graph.add_node(node("h", NodeKind::MechanismHypothesis)).unwrap();
        graph.add_edge("m1", "h", EdgeKind::DerivedFrom).unwrap();
        graph.add_edge("m2", "h", EdgeKind::DerivedFrom).unwrap();

        let roots = graph.model_lineage_roots("h").unwrap();
        assert_eq!(roots, BTreeSet::from(["lineage:root-a".into(), "lineage:root-b".into()]));
    }

    #[test]
    fn shared_ancestry_is_not_double_counted() {
        let mut graph = ScientificDiscoveryGraph::new();
        for (id, lineage) in [("m1", "root"), ("m2", "root")] {
            let p = provenance(id).with_model_lineage(lineage);
            graph.add_node(DiscoveryNode::new(id, NodeKind::CausalModel, p, "sha256:m").unwrap()).unwrap();
        }
        graph.add_node(node("h", NodeKind::MechanismHypothesis)).unwrap();
        graph.add_edge("m1", "h", EdgeKind::DerivedFrom).unwrap();
        graph.add_edge("m2", "h", EdgeKind::DerivedFrom).unwrap();

        assert_eq!(graph.model_lineage_roots("h").unwrap().len(), 1);
    }

    #[test]
    fn computational_nodes_cannot_become_observations_by_edge_label() {
        let mut graph = ScientificDiscoveryGraph::new();
        graph.add_node(node("model", NodeKind::CausalModel)).unwrap();
        graph.add_node(node("obs", NodeKind::Observation)).unwrap();
        assert!(matches!(
            graph.add_edge("model", "obs", EdgeKind::Supports),
            Err(GraphError::InvalidEdge { .. })
        ));
        graph.add_edge("model", "obs", EdgeKind::Predicts).unwrap();
    }

    #[test]
    fn criterion_evidence_has_explicit_evidence_parent() {
        let mut graph = ScientificDiscoveryGraph::new();
        graph.add_node(node("obs", NodeKind::Observation)).unwrap();
        graph.add_node(node("criterion", NodeKind::OfficialCriterionEvidence)).unwrap();
        graph.add_edge("criterion", "obs", EdgeKind::DerivedFrom).unwrap();
    }

    #[test]
    fn duplicate_nodes_and_self_loops_are_rejected() {
        let mut graph = ScientificDiscoveryGraph::new();
        graph.add_node(node("x", NodeKind::Quest)).unwrap();
        assert!(matches!(graph.add_node(node("x", NodeKind::Quest)), Err(GraphError::DuplicateNode(_))));
        assert!(matches!(graph.add_edge("x", "x", EdgeKind::DerivedFrom), Err(GraphError::SelfLoop(_))));
    }

    #[test]
    fn cutoff_order_is_enforced() {
        let result = ProvenanceRef::new("a", "artifact", "input")
            .unwrap()
            .with_cutoffs("2026-09-28T10:00:00Z", "2026-09-28T09:00:00Z");
        assert_eq!(result, Err(GraphError::ExposureBeforeKnowledgeCutoff));
    }
}
