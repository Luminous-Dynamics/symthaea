// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic evidence-lineage projection for DKG indexers.
//!
//! This module deliberately contains no scientific inference. It turns already
//! verified immutable envelopes into a canonical provenance graph. Consumers may
//! index/query the graph, but graph membership or connectivity is never evidence
//! of scientific truth or criterion completion.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::candidate_commitment::CandidatePredictionCommitment;
use crate::criterion_evidence::CriterionEvidenceEligibility;
use crate::external_observation::ExternalExperimentalObservation;
use crate::independent_assessment::IndependentAssessment;
use crate::replication::ReplicationRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DkgNodeType {
    CandidateCommitment,
    ExternalObservation,
    IndependentAssessment,
    Replication,
    CriterionEvidence,
    EvidenceChainAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DkgEdgeType {
    CommitsTo,
    ObservedFrom,
    Assesses,
    Replicates,
    EligibleFor,
    AuditedBy,
    Supersedes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DkgNode {
    pub node_id: String,
    pub node_type: DkgNodeType,
    pub record_digest: String,
    pub supersedes_node_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DkgEdge {
    pub source_node_id: String,
    pub edge_type: DkgEdgeType,
    pub target_node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageDkgProjection {
    pub projection_version: String,
    pub nodes: Vec<DkgNode>,
    pub edges: Vec<DkgEdge>,
    pub projection_digest: String,
}

impl EvidenceLineageDkgProjection {

    /// Derive canonical graph topology from one verified evidence chain.
    pub fn from_chain(
        commitment: &CandidatePredictionCommitment,
        observation: &ExternalExperimentalObservation,
        assessment: &IndependentAssessment,
        replication: &ReplicationRecord,
        evidence: &CriterionEvidenceEligibility,
    ) -> Result<Self, DkgProjectionError> {
        if !commitment.commitment.verify_integrity() { return Err(DkgProjectionError::InvalidCommitment); }
        if !observation.verify_integrity() { return Err(DkgProjectionError::InvalidObservation); }
        if !assessment.verify_integrity() { return Err(DkgProjectionError::InvalidAssessment); }
        if !replication.verify_integrity() { return Err(DkgProjectionError::InvalidReplication); }
        if !evidence.verify_integrity() { return Err(DkgProjectionError::InvalidCriterionEvidence); }

        let commitment_id = commitment.commitment.event_id().to_string();
        if observation.commitment_event_id != commitment_id
            || observation.candidate_id != commitment.candidate_id
            || observation.binding_digest != commitment.binding_digest {
            return Err(DkgProjectionError::CommitmentObservationMismatch);
        }
        if assessment.observation_id != observation.observation_id
            || assessment.observation_record_digest != observation.record_digest
            || assessment.commitment_event_id != observation.commitment_event_id
            || assessment.candidate_id != observation.candidate_id
            || assessment.binding_digest != observation.binding_digest {
            return Err(DkgProjectionError::AssessmentMismatch);
        }
        if replication.observation_id != observation.observation_id
            || replication.observation_record_digest != observation.record_digest
            || replication.assessment_id != assessment.assessment_id
            || replication.assessment_record_digest != assessment.record_digest
            || replication.commitment_event_id != observation.commitment_event_id
            || replication.candidate_id != observation.candidate_id
            || replication.binding_digest != observation.binding_digest {
            return Err(DkgProjectionError::ReplicationMismatch);
        }
        if evidence.challenge_id != observation.challenge_id
            || evidence.criterion_id != observation.criterion_id
            || evidence.criterion_generation != observation.criterion_generation
            || evidence.observation_id != observation.observation_id
            || evidence.observation_record_digest != observation.record_digest
            || evidence.assessment_id != assessment.assessment_id
            || evidence.assessment_record_digest != assessment.record_digest
            || evidence.replication_id != replication.replication_id
            || evidence.replication_record_digest != replication.record_digest
            || evidence.commitment_event_id != observation.commitment_event_id
            || evidence.candidate_id != observation.candidate_id
            || evidence.binding_digest != observation.binding_digest {
            return Err(DkgProjectionError::CriterionEvidenceMismatch);
        }

        if evidence.supersedes_evidence_id.is_some() {
            return Err(DkgProjectionError::SupersessionParentNotIncluded);
        }

        let nodes = vec![
            DkgNode { node_id: commitment_id.clone(), node_type: DkgNodeType::CandidateCommitment,
                record_digest: commitment.commitment.payload_digest(), supersedes_node_id: None },
            DkgNode { node_id: observation.observation_id.clone(), node_type: DkgNodeType::ExternalObservation,
                record_digest: observation.record_digest.clone(), supersedes_node_id: None },
            DkgNode { node_id: assessment.assessment_id.clone(), node_type: DkgNodeType::IndependentAssessment,
                record_digest: assessment.record_digest.clone(), supersedes_node_id: None },
            DkgNode { node_id: replication.replication_id.clone(), node_type: DkgNodeType::Replication,
                record_digest: replication.record_digest.clone(), supersedes_node_id: None },
            DkgNode { node_id: evidence.evidence_id.clone(), node_type: DkgNodeType::CriterionEvidence,
                record_digest: evidence.record_digest.clone(), supersedes_node_id: evidence.supersedes_evidence_id.clone() },
        ];
        let mut edges = vec![
            DkgEdge { source_node_id: observation.observation_id.clone(), edge_type: DkgEdgeType::ObservedFrom, target_node_id: commitment_id },
            DkgEdge { source_node_id: assessment.assessment_id.clone(), edge_type: DkgEdgeType::Assesses, target_node_id: observation.observation_id.clone() },
            DkgEdge { source_node_id: replication.replication_id.clone(), edge_type: DkgEdgeType::Replicates, target_node_id: assessment.assessment_id.clone() },
            DkgEdge { source_node_id: evidence.evidence_id.clone(), edge_type: DkgEdgeType::EligibleFor, target_node_id: replication.replication_id.clone() },
        ];
        Ok(Self::new(nodes, edges))
    }

    pub fn new(mut nodes: Vec<DkgNode>, mut edges: Vec<DkgEdge>) -> Self {
        nodes.sort_by_key(|n| (node_type(n.node_type), n.node_id.clone(), n.record_digest.clone()));
        edges.sort_by_key(|e| (e.source_node_id.clone(), edge_type(e.edge_type), e.target_node_id.clone()));
        let mut p = Self {
            projection_version: "1.0.0".into(),
            nodes,
            edges,
            projection_digest: String::new(),
        };
        p.projection_digest = p.digest();
        p
    }

    pub fn verify_integrity(&self) -> bool {
        self.projection_version == "1.0.0"
            && self.projection_digest == self.digest()
            && self.nodes.iter().all(|n| !n.node_id.trim().is_empty() && !n.record_digest.trim().is_empty())
            && self.edges.iter().all(|e| !e.source_node_id.trim().is_empty() && !e.target_node_id.trim().is_empty())
            && self.edges.iter().all(|e| {
                self.nodes.iter().any(|n| n.node_id == e.source_node_id)
                    && self.nodes.iter().any(|n| n.node_id == e.target_node_id)
            })
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:evidence-lineage-dkg-projection:v1 ");
        put(&mut h, &self.projection_version);
        h.update((self.nodes.len() as u64).to_be_bytes());
        for n in &self.nodes {
            put(&mut h, &n.node_id);
            put(&mut h, node_type(n.node_type));
            put(&mut h, &n.record_digest);
            put(&mut h, n.supersedes_node_id.as_deref().unwrap_or(""));
        }
        h.update((self.edges.len() as u64).to_be_bytes());
        for e in &self.edges {
            put(&mut h, &e.source_node_id);
            put(&mut h, edge_type(e.edge_type));
            put(&mut h, &e.target_node_id);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DkgProjectionError {
    InvalidCommitment, InvalidObservation, InvalidAssessment, InvalidReplication, InvalidCriterionEvidence,
    CommitmentObservationMismatch, AssessmentMismatch, ReplicationMismatch, CriterionEvidenceMismatch,
    SupersessionParentNotIncluded,
}

fn put(h: &mut Sha256, s: &str) {
    h.update((s.len() as u64).to_be_bytes());
    h.update(s.as_bytes());
}
fn node_type(t: DkgNodeType) -> &'static str {
    match t {
        DkgNodeType::CandidateCommitment => "candidate_commitment",
        DkgNodeType::ExternalObservation => "external_observation",
        DkgNodeType::IndependentAssessment => "independent_assessment",
        DkgNodeType::Replication => "replication",
        DkgNodeType::CriterionEvidence => "criterion_evidence",
        DkgNodeType::EvidenceChainAudit => "evidence_chain_audit",
    }
}
fn edge_type(t: DkgEdgeType) -> &'static str {
    match t {
        DkgEdgeType::CommitsTo => "commits_to",
        DkgEdgeType::ObservedFrom => "observed_from",
        DkgEdgeType::Assesses => "assesses",
        DkgEdgeType::Replicates => "replicates",
        DkgEdgeType::EligibleFor => "eligible_for",
        DkgEdgeType::AuditedBy => "audited_by",
        DkgEdgeType::Supersedes => "supersedes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, t: DkgNodeType) -> DkgNode {
        DkgNode { node_id: id.into(), node_type: t, record_digest: format!("sha256:{id}"), supersedes_node_id: None }
    }

    #[test]
    fn projection_is_deterministic_under_input_order() {
        let n1 = node("observation:1", DkgNodeType::ExternalObservation);
        let n2 = node("assessment:1", DkgNodeType::IndependentAssessment);
        let a = EvidenceLineageDkgProjection::new(vec![n1.clone(), n2.clone()], vec![
            DkgEdge { source_node_id: n2.node_id.clone(), edge_type: DkgEdgeType::Assesses, target_node_id: n1.node_id.clone() },
        ]);
        let b = EvidenceLineageDkgProjection::new(vec![n2.clone(), n1.clone()], vec![
            DkgEdge { source_node_id: n2.node_id, edge_type: DkgEdgeType::Assesses, target_node_id: n1.node_id },
        ]);
        assert_eq!(a, b);
        assert!(a.verify_integrity());
    }

    #[test]
    fn tampering_changes_integrity() {
        let n1 = node("observation:1", DkgNodeType::ExternalObservation);
        let mut p = EvidenceLineageDkgProjection::new(vec![n1], vec![]);
        p.nodes[0].record_digest = "sha256:tampered".into();
        assert!(!p.verify_integrity());
    }

    #[test]
    fn unknown_edge_endpoint_is_rejected() {
        let n = node("observation:1", DkgNodeType::ExternalObservation);
        let p = EvidenceLineageDkgProjection::new(vec![n], vec![DkgEdge {
            source_node_id: "observation:1".into(), edge_type: DkgEdgeType::Assesses, target_node_id: "missing".into(),
        }]);
        assert!(!p.verify_integrity());
    }

    #[test]
    fn serde_roundtrip_preserves_digest() {
        let n = node("observation:1", DkgNodeType::ExternalObservation);
        let p = EvidenceLineageDkgProjection::new(vec![n], vec![]);
        let encoded = serde_json::to_vec(&p).unwrap();
        let decoded: EvidenceLineageDkgProjection = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(p, decoded);
        assert!(decoded.verify_integrity());
    }
}
