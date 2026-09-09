// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic, append-only evidence graph for spectroscopy.
//!
//! The graph deliberately separates claims from evidence and preserves old
//! records when newer work supersedes them. Consensus is therefore derived
//! from graph state rather than written back into historical observations.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceNodeKind {
    Observation,
    UpperLimit,
    LatticePrediction,
    PhenomenologicalModel,
    Replication,
    Contradiction,
    SupersedingEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceRelation {
    Supports,
    Challenges,
    Constrains,
    Replicates,
    Supersedes,
    ContextFor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphNodeRef {
    Claim(String),
    Evidence(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificClaim {
    pub id: String,
    pub subject: String,
    pub statement: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectroscopyEvidence {
    pub id: String,
    pub kind: EvidenceNodeKind,
    pub subject: String,
    /// Stable observable key such as `mass`, `jpc`, or `decay::kstar_k`.
    pub observable: String,
    pub source: String,
    /// DOI, arXiv identifier, collaboration data release, or other stable ID.
    pub persistent_id: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEdge {
    pub from: GraphNodeRef,
    pub relation: EvidenceRelation,
    pub to: GraphNodeRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceGraphError {
    DuplicateId(String),
    MissingNode(GraphNodeRef),
    SelfEdge(GraphNodeRef),
}

/// Append-only in-memory graph. No mutation/deletion API is exposed for nodes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectroscopyEvidenceGraph {
    claims: Vec<ScientificClaim>,
    evidence: Vec<SpectroscopyEvidence>,
    edges: Vec<EvidenceEdge>,
}

impl SpectroscopyEvidenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn claims(&self) -> &[ScientificClaim] {
        &self.claims
    }

    pub fn evidence(&self) -> &[SpectroscopyEvidence] {
        &self.evidence
    }

    pub fn edges(&self) -> &[EvidenceEdge] {
        &self.edges
    }

    pub fn add_claim(&mut self, claim: ScientificClaim) -> Result<(), EvidenceGraphError> {
        if self.contains_id(&claim.id) {
            return Err(EvidenceGraphError::DuplicateId(claim.id));
        }
        self.claims.push(claim);
        Ok(())
    }

    pub fn add_evidence(
        &mut self,
        record: SpectroscopyEvidence,
    ) -> Result<(), EvidenceGraphError> {
        if self.contains_id(&record.id) {
            return Err(EvidenceGraphError::DuplicateId(record.id));
        }
        self.evidence.push(record);
        Ok(())
    }

    pub fn add_edge(&mut self, edge: EvidenceEdge) -> Result<(), EvidenceGraphError> {
        if edge.from == edge.to {
            return Err(EvidenceGraphError::SelfEdge(edge.from));
        }
        if !self.contains_ref(&edge.from) {
            return Err(EvidenceGraphError::MissingNode(edge.from));
        }
        if !self.contains_ref(&edge.to) {
            return Err(EvidenceGraphError::MissingNode(edge.to));
        }
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
        Ok(())
    }

    /// Append newer evidence and link it to the retained historical record.
    pub fn supersede_evidence(
        &mut self,
        newer: SpectroscopyEvidence,
        older_id: &str,
    ) -> Result<(), EvidenceGraphError> {
        let older = GraphNodeRef::Evidence(older_id.to_string());
        if !self.contains_ref(&older) {
            return Err(EvidenceGraphError::MissingNode(older));
        }
        let newer_ref = GraphNodeRef::Evidence(newer.id.clone());
        self.add_evidence(newer)?;
        self.add_edge(EvidenceEdge {
            from: newer_ref,
            relation: EvidenceRelation::Supersedes,
            to: older,
        })
    }

    pub fn evidence_for_claim(&self, claim_id: &str) -> Vec<(&SpectroscopyEvidence, EvidenceRelation)> {
        let claim_ref = GraphNodeRef::Claim(claim_id.to_string());
        self.edges
            .iter()
            .filter_map(|edge| {
                if edge.to != claim_ref {
                    return None;
                }
                let GraphNodeRef::Evidence(ref evidence_id) = edge.from else {
                    return None;
                };
                self.evidence
                    .iter()
                    .find(|record| record.id == *evidence_id)
                    .map(|record| (record, edge.relation))
            })
            .collect()
    }

    pub fn validate(&self) -> Result<(), EvidenceGraphError> {
        let mut ids = HashSet::new();
        for id in self.claims.iter().map(|c| &c.id).chain(self.evidence.iter().map(|e| &e.id)) {
            if !ids.insert(id) {
                return Err(EvidenceGraphError::DuplicateId(id.clone()));
            }
        }
        for edge in &self.edges {
            if !self.contains_ref(&edge.from) {
                return Err(EvidenceGraphError::MissingNode(edge.from.clone()));
            }
            if !self.contains_ref(&edge.to) {
                return Err(EvidenceGraphError::MissingNode(edge.to.clone()));
            }
        }
        Ok(())
    }

    fn contains_id(&self, id: &str) -> bool {
        self.claims.iter().any(|claim| claim.id == id)
            || self.evidence.iter().any(|record| record.id == id)
    }

    fn contains_ref(&self, node: &GraphNodeRef) -> bool {
        match node {
            GraphNodeRef::Claim(id) => self.claims.iter().any(|claim| &claim.id == id),
            GraphNodeRef::Evidence(id) => self.evidence.iter().any(|record| &record.id == id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim() -> ScientificClaim {
        ScientificClaim {
            id: "claim::x2370::glueball".into(),
            subject: "X(2370)".into(),
            statement: "X(2370) has a dominant pseudoscalar-glueball component".into(),
        }
    }

    fn evidence(id: &str) -> SpectroscopyEvidence {
        SpectroscopyEvidence {
            id: id.into(),
            kind: EvidenceNodeKind::Observation,
            subject: "X(2370)".into(),
            observable: "jpc".into(),
            source: "BESIII".into(),
            persistent_id: "arXiv:2312.05324".into(),
            summary: "J^PC = 0-+".into(),
        }
    }

    #[test]
    fn graph_links_evidence_to_claim_without_collapsing_them() {
        let mut graph = SpectroscopyEvidenceGraph::new();
        graph.add_claim(claim()).unwrap();
        graph.add_evidence(evidence("evidence::jpc")).unwrap();
        graph.add_edge(EvidenceEdge {
            from: GraphNodeRef::Evidence("evidence::jpc".into()),
            relation: EvidenceRelation::Supports,
            to: GraphNodeRef::Claim("claim::x2370::glueball".into()),
        }).unwrap();
        assert_eq!(graph.evidence_for_claim("claim::x2370::glueball").len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn supersession_preserves_prior_evidence() {
        let mut graph = SpectroscopyEvidenceGraph::new();
        graph.add_evidence(evidence("evidence::old")).unwrap();
        let mut newer = evidence("evidence::new");
        newer.kind = EvidenceNodeKind::SupersedingEvidence;
        graph.supersede_evidence(newer, "evidence::old").unwrap();
        assert_eq!(graph.evidence().len(), 2);
        assert!(graph.edges().iter().any(|edge| edge.relation == EvidenceRelation::Supersedes));
    }

    #[test]
    fn dangling_edges_are_rejected() {
        let mut graph = SpectroscopyEvidenceGraph::new();
        graph.add_claim(claim()).unwrap();
        let result = graph.add_edge(EvidenceEdge {
            from: GraphNodeRef::Evidence("missing".into()),
            relation: EvidenceRelation::Challenges,
            to: GraphNodeRef::Claim("claim::x2370::glueball".into()),
        });
        assert!(matches!(result, Err(EvidenceGraphError::MissingNode(_))));
    }
}
