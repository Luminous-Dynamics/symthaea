// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Export spectroscopy evidence into the Mycelix epistemic-DKG wire schema.
//!
//! Symthaea remains the physics/reasoning producer; Mycelix remains the durable
//! epistemic authority. This module has no Mycelix crate dependency and emits a
//! versioned transport bundle that the Mycelix-DeSci bridge validates.

use crate::spectroscopy_evidence::{
    EvidenceNodeKind, EvidenceRelation, GraphNodeRef, SpectroscopyEvidenceGraph,
};
use serde::{Deserialize, Serialize};

pub const MYCELIX_SPECTROSCOPY_DKG_PROTOCOL: &str = "mycelix-symthaea-spectroscopy";
pub const MYCELIX_SPECTROSCOPY_DKG_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelixNodeKind {
    Claim,
    Observation,
    UpperLimit,
    LatticePrediction,
    PhenomenologicalModel,
    Replication,
    Contradiction,
    SupersedingEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelixRelation {
    Supports,
    Challenges,
    Constrains,
    Replicates,
    Supersedes,
    ContextFor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelixProvenance {
    pub source: String,
    pub persistent_id: String,
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelixDkgNode {
    pub id: String,
    pub kind: MycelixNodeKind,
    pub subject: String,
    pub statement: String,
    pub observable: Option<String>,
    pub provenance: Option<MycelixProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelixDkgEdge {
    pub from: String,
    pub relation: MycelixRelation,
    pub to: String,
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelixDkgBundle {
    pub protocol: String,
    pub schema_version: u16,
    pub producer: String,
    pub nodes: Vec<MycelixDkgNode>,
    pub edges: Vec<MycelixDkgEdge>,
}

fn node_kind(kind: EvidenceNodeKind) -> MycelixNodeKind {
    match kind {
        EvidenceNodeKind::Observation => MycelixNodeKind::Observation,
        EvidenceNodeKind::UpperLimit => MycelixNodeKind::UpperLimit,
        EvidenceNodeKind::LatticePrediction => MycelixNodeKind::LatticePrediction,
        EvidenceNodeKind::PhenomenologicalModel => MycelixNodeKind::PhenomenologicalModel,
        EvidenceNodeKind::Replication => MycelixNodeKind::Replication,
        EvidenceNodeKind::Contradiction => MycelixNodeKind::Contradiction,
        EvidenceNodeKind::SupersedingEvidence => MycelixNodeKind::SupersedingEvidence,
    }
}

fn relation(relation: EvidenceRelation) -> MycelixRelation {
    match relation {
        EvidenceRelation::Supports => MycelixRelation::Supports,
        EvidenceRelation::Challenges => MycelixRelation::Challenges,
        EvidenceRelation::Constrains => MycelixRelation::Constrains,
        EvidenceRelation::Replicates => MycelixRelation::Replicates,
        EvidenceRelation::Supersedes => MycelixRelation::Supersedes,
        EvidenceRelation::ContextFor => MycelixRelation::ContextFor,
    }
}

fn node_id(node: &GraphNodeRef) -> &str {
    match node {
        GraphNodeRef::Claim(id) | GraphNodeRef::Evidence(id) => id,
    }
}

/// Export a validated spectroscopy graph into Mycelix's versioned transport.
///
/// Graph validation should be performed before persistence; this function is a
/// pure projection and never mutates or assigns consensus to source evidence.
pub fn export_to_mycelix(
    graph: &SpectroscopyEvidenceGraph,
    producer: impl Into<String>,
) -> MycelixDkgBundle {
    let mut nodes = Vec::with_capacity(graph.claims().len() + graph.evidence().len());

    nodes.extend(graph.claims().iter().map(|claim| MycelixDkgNode {
        id: claim.id.clone(),
        kind: MycelixNodeKind::Claim,
        subject: claim.subject.clone(),
        statement: claim.statement.clone(),
        observable: None,
        provenance: None,
    }));

    nodes.extend(graph.evidence().iter().map(|evidence| MycelixDkgNode {
        id: evidence.id.clone(),
        kind: node_kind(evidence.kind),
        subject: evidence.subject.clone(),
        statement: evidence.summary.clone(),
        observable: Some(evidence.observable.clone()),
        provenance: Some(MycelixProvenance {
            source: evidence.source.clone(),
            persistent_id: evidence.persistent_id.clone(),
            content_hash: None,
        }),
    }));

    let edges = graph
        .edges()
        .iter()
        .map(|edge| MycelixDkgEdge {
            from: node_id(&edge.from).to_string(),
            relation: relation(edge.relation),
            to: node_id(&edge.to).to_string(),
            rationale: None,
        })
        .collect();

    MycelixDkgBundle {
        protocol: MYCELIX_SPECTROSCOPY_DKG_PROTOCOL.into(),
        schema_version: MYCELIX_SPECTROSCOPY_DKG_SCHEMA_VERSION,
        producer: producer.into(),
        nodes,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectroscopy_evidence::{
        EvidenceEdge, ScientificClaim, SpectroscopyEvidence,
    };

    #[test]
    fn exports_claim_evidence_and_support_edge_without_collapsing_them() {
        let mut graph = SpectroscopyEvidenceGraph::new();
        graph
            .add_claim(ScientificClaim {
                id: "claim::x2370::glueball".into(),
                subject: "X(2370)".into(),
                statement: "dominant pseudoscalar glueball component".into(),
            })
            .unwrap();
        graph
            .add_evidence(SpectroscopyEvidence {
                id: "evidence::x2370::jpc".into(),
                kind: EvidenceNodeKind::Observation,
                subject: "X(2370)".into(),
                observable: "jpc".into(),
                source: "BESIII".into(),
                persistent_id: "arXiv:2312.05324".into(),
                summary: "J^PC = 0-+".into(),
            })
            .unwrap();
        graph
            .add_edge(EvidenceEdge {
                from: GraphNodeRef::Evidence("evidence::x2370::jpc".into()),
                relation: EvidenceRelation::Supports,
                to: GraphNodeRef::Claim("claim::x2370::glueball".into()),
            })
            .unwrap();

        let bundle = export_to_mycelix(&graph, "symthaea:test-lineage");
        assert_eq!(bundle.protocol, MYCELIX_SPECTROSCOPY_DKG_PROTOCOL);
        assert_eq!(bundle.nodes.len(), 2);
        assert_eq!(bundle.edges.len(), 1);
        assert_eq!(bundle.edges[0].relation, MycelixRelation::Supports);
    }
}
