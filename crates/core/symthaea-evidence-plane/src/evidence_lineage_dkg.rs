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
        Self::from_chain_history(
            commitment,
            observation,
            assessment,
            replication,
            std::slice::from_ref(evidence),
        )
    }

    /// Derive a canonical graph from a closed criterion-evidence supersession
    /// history. Every historical record remains a first-class node; the graph
    /// records replacement ancestry without selecting a scientifically "true"
    /// disposition.
    pub fn from_chain_history(
        commitment: &CandidatePredictionCommitment,
        observation: &ExternalExperimentalObservation,
        assessment: &IndependentAssessment,
        replication: &ReplicationRecord,
        evidence_history: &[CriterionEvidenceEligibility],
    ) -> Result<Self, DkgProjectionError> {
        if !commitment.commitment.verify_integrity() {
            return Err(DkgProjectionError::InvalidCommitment);
        }
        if !observation.verify_integrity() {
            return Err(DkgProjectionError::InvalidObservation);
        }
        if !assessment.verify_integrity() {
            return Err(DkgProjectionError::InvalidAssessment);
        }
        if !replication.verify_integrity() {
            return Err(DkgProjectionError::InvalidReplication);
        }
        if evidence_history.is_empty() {
            return Err(DkgProjectionError::EmptySupersessionHistory);
        }

        let commitment_id = commitment.commitment.event_id().to_string();
        if observation.commitment_event_id != commitment_id
            || observation.candidate_id != commitment.candidate_id
            || observation.binding_digest != commitment.binding_digest
        {
            return Err(DkgProjectionError::CommitmentObservationMismatch);
        }
        if assessment.observation_id != observation.observation_id
            || assessment.observation_record_digest != observation.record_digest
            || assessment.commitment_event_id != observation.commitment_event_id
            || assessment.candidate_id != observation.candidate_id
            || assessment.binding_digest != observation.binding_digest
        {
            return Err(DkgProjectionError::AssessmentMismatch);
        }
        if replication.observation_id != observation.observation_id
            || replication.observation_record_digest != observation.record_digest
            || replication.assessment_id != assessment.assessment_id
            || replication.assessment_record_digest != assessment.record_digest
            || replication.commitment_event_id != observation.commitment_event_id
            || replication.candidate_id != observation.candidate_id
            || replication.binding_digest != observation.binding_digest
        {
            return Err(DkgProjectionError::ReplicationMismatch);
        }

        for evidence in evidence_history {
            if !evidence.verify_integrity() {
                return Err(DkgProjectionError::InvalidCriterionEvidence);
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
                || evidence.binding_digest != observation.binding_digest
            {
                return Err(DkgProjectionError::CriterionEvidenceMismatch);
            }
        }

        let mut ids = std::collections::BTreeSet::new();
        for evidence in evidence_history {
            if !ids.insert(evidence.evidence_id.clone()) {
                return Err(DkgProjectionError::DuplicateEvidenceId);
            }
            if evidence.supersedes_evidence_id.as_deref() == Some(evidence.evidence_id.as_str()) {
                return Err(DkgProjectionError::SelfSupersession);
            }
        }

        let mut parent_to_child = std::collections::BTreeMap::<String, String>::new();
        let mut child_to_parent = std::collections::BTreeMap::<String, String>::new();
        for evidence in evidence_history {
            if let Some(parent) = &evidence.supersedes_evidence_id {
                if !ids.contains(parent) {
                    return Err(DkgProjectionError::MissingSupersessionParent);
                }
                if parent_to_child.insert(parent.clone(), evidence.evidence_id.clone()).is_some() {
                    return Err(DkgProjectionError::BranchingSupersession);
                }
                child_to_parent.insert(evidence.evidence_id.clone(), parent.clone());
            }
        }

        // A closed linear history has exactly one root and one terminal record.
        // Following parents from every node catches cycles deterministically.
        let roots: Vec<String> = ids.iter().filter(|id| !child_to_parent.contains_key(id.as_str())).cloned().collect();
        if roots.len() != 1 {
            return Err(DkgProjectionError::SupersessionCycleOrDisconnected);
        }
        let mut visited = std::collections::BTreeSet::new();
        let mut cursor = roots[0].clone();
        loop {
            if !visited.insert(cursor.clone()) {
                return Err(DkgProjectionError::SupersessionCycleOrDisconnected);
            }
            match parent_to_child.get(&cursor) {
                Some(next) => cursor = next.clone(),
                None => break,
            }
        }
        if visited.len() != ids.len() {
            return Err(DkgProjectionError::SupersessionCycleOrDisconnected);
        }

        let mut nodes = vec![
            DkgNode {
                node_id: commitment_id.clone(),
                node_type: DkgNodeType::CandidateCommitment,
                record_digest: commitment.commitment.payload_digest(),
                supersedes_node_id: None,
            },
            DkgNode {
                node_id: observation.observation_id.clone(),
                node_type: DkgNodeType::ExternalObservation,
                record_digest: observation.record_digest.clone(),
                supersedes_node_id: None,
            },
            DkgNode {
                node_id: assessment.assessment_id.clone(),
                node_type: DkgNodeType::IndependentAssessment,
                record_digest: assessment.record_digest.clone(),
                supersedes_node_id: None,
            },
            DkgNode {
                node_id: replication.replication_id.clone(),
                node_type: DkgNodeType::Replication,
                record_digest: replication.record_digest.clone(),
                supersedes_node_id: None,
            },
        ];
        let mut edges = vec![
            DkgEdge {
                source_node_id: observation.observation_id.clone(),
                edge_type: DkgEdgeType::ObservedFrom,
                target_node_id: commitment_id,
            },
            DkgEdge {
                source_node_id: assessment.assessment_id.clone(),
                edge_type: DkgEdgeType::Assesses,
                target_node_id: observation.observation_id.clone(),
            },
            DkgEdge {
                source_node_id: replication.replication_id.clone(),
                edge_type: DkgEdgeType::Replicates,
                target_node_id: assessment.assessment_id.clone(),
            },
        ];

        for evidence in evidence_history {
            nodes.push(DkgNode {
                node_id: evidence.evidence_id.clone(),
                node_type: DkgNodeType::CriterionEvidence,
                record_digest: evidence.record_digest.clone(),
                supersedes_node_id: evidence.supersedes_evidence_id.clone(),
            });
            edges.push(DkgEdge {
                source_node_id: evidence.evidence_id.clone(),
                edge_type: DkgEdgeType::EligibleFor,
                target_node_id: replication.replication_id.clone(),
            });
            if let Some(parent) = &evidence.supersedes_evidence_id {
                edges.push(DkgEdge {
                    source_node_id: evidence.evidence_id.clone(),
                    edge_type: DkgEdgeType::Supersedes,
                    target_node_id: parent.clone(),
                });
            }
        }

        Ok(Self::new(nodes, edges))
    }

    pub fn new(mut nodes: Vec<DkgNode>, mut edges: Vec<DkgEdge>) -> Self {
        nodes.sort_by_key(|n| (node_type(n.node_type), n.node_id.clone(), n.record_digest.clone()));
        edges.sort_by_key(|e| (e.source_node_id.clone(), edge_type(e.edge_type), e.target_node_id.clone()));
        let mut p = Self {
            projection_version: "1.1.0".into(),
            nodes,
            edges,
            projection_digest: String::new(),
        };
        p.projection_digest = p.digest();
        p
    }

    pub fn verify_integrity(&self) -> bool {
        let unique_nodes = self.nodes.iter().map(|n| n.node_id.as_str()).collect::<std::collections::BTreeSet<_>>().len() == self.nodes.len();
        let unique_edges = self.edges.iter().map(|e| (&e.source_node_id, e.edge_type, &e.target_node_id)).collect::<std::collections::BTreeSet<_>>().len() == self.edges.len();
        self.projection_version == "1.1.0"
            && self.projection_digest == self.digest()
            && unique_nodes
            && unique_edges
            && self.nodes.iter().all(|n| !n.node_id.trim().is_empty() && !n.record_digest.trim().is_empty())
            && self.edges.iter().all(|e| !e.source_node_id.trim().is_empty() && !e.target_node_id.trim().is_empty())
            && self.edges.iter().all(|e| {
                self.nodes.iter().any(|n| n.node_id == e.source_node_id)
                    && self.nodes.iter().any(|n| n.node_id == e.target_node_id)
            })
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:evidence-lineage-dkg-projection:v2 ");
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
    EmptySupersessionHistory, DuplicateEvidenceId, SelfSupersession, MissingSupersessionParent,
    BranchingSupersession, SupersessionCycleOrDisconnected,
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
    use crate::candidate_commitment::{
        commit_candidate_envelope, CandidatePredictionBinding, CandidatePredictionSource,
    };
    use crate::criterion_evidence::{
        CriterionEvidenceDisposition, CriterionEvidenceEligibility, CriterionEvidenceInput,
    };
    use crate::external_observation::{ExternalExperimentalObservation, ExternalObservationInput, ObservationDisposition};
    use crate::independent_assessment::{AssessmentInput, AssessmentOutcome, IndependentAssessment};
    use crate::prospective::ProspectiveProvenance;
    use crate::replication::{ReplicationInput, ReplicationOutcome, ReplicationRecord};

    fn evidence_chain() -> (
        crate::candidate_commitment::CandidatePredictionCommitment,
        ExternalExperimentalObservation,
        IndependentAssessment,
        ReplicationRecord,
        CriterionEvidenceEligibility,
        CriterionEvidenceEligibility,
    ) {
        let source = CandidatePredictionSource {
            candidate_id: "candidate:1".into(),
            source_candidate_id: "source:1".into(),
            test_specification_id: "test:1".into(),
            measurement_specification_id: "measure:1".into(),
            left_lineage: "left".into(),
            right_lineage: "right".into(),
        };
        let binding = CandidatePredictionBinding::from_source(&source, b"prediction").unwrap();
        let provenance = ProspectiveProvenance::new(
            "input", "artifact", binding.lineage_digest().unwrap(),
            "2026-09-28T08:00:00Z", "2026-09-28T09:00:00Z",
        ).unwrap();
        let commitment = commit_candidate_envelope(
            &source, "challenge-1", "criterion-1", "generation-1",
            "predictor", "2026-09-28T09:00:00Z", provenance, b"prediction",
        ).unwrap();
        let observation = ExternalExperimentalObservation::ingest(
            &commitment, &binding,
            ExternalObservationInput {
                observation_id: "obs:1".into(),
                execution_id: "exec:original".into(),
                observer_id: "observer:original".into(),
                institution_id: "institution:original".into(),
                observed_at: "2026-09-29T10:00:00Z".into(),
                disposition: ObservationDisposition::Reported,
                observation_payload: b"observation".to_vec(),
            },
        ).unwrap();
        let assessment = IndependentAssessment::assess(
            &observation,
            AssessmentInput {
                assessment_id: "assessment:1".into(),
                assessor_id: "assessor:1".into(),
                assessor_institution_id: "review".into(),
                independent_from_observer: true,
                independence_basis: "separate".into(),
                assessed_at: "2026-09-29T11:00:00Z".into(),
                outcome: AssessmentOutcome::Supports,
                assessment_payload: b"assessment".to_vec(),
            },
        ).unwrap();
        let replication = ReplicationRecord::record(
            &observation, &assessment,
            ReplicationInput {
                replication_id: "replication:1".into(),
                execution_id: "exec:replica".into(),
                observer_id: "observer:replica".into(),
                institution_id: "institution:replica".into(),
                independent_from_original_observer: true,
                independence_basis: "distinct execution".into(),
                replicated_at: "2026-09-29T12:00:00Z".into(),
                outcome: ReplicationOutcome::ReplicatedSupportive,
                replication_payload: b"replication".to_vec(),
            },
        ).unwrap();
        let input = |evidence_id: &str, disposition| CriterionEvidenceInput {
            evidence_id: evidence_id.into(),
            challenge_id: "challenge-1".into(),
            criterion_id: "criterion-1".into(),
            criterion_generation: "generation-1".into(),
            authority_id: "authority:1".into(),
            authority_institution_id: "institution:authority".into(),
            authority_basis: "official scorer designation".into(),
            adjudicated_at: "2026-09-29T13:00:00Z".into(),
            disposition,
            authority_payload: evidence_id.as_bytes().to_vec(),
        };
        let original = CriterionEvidenceEligibility::adjudicate(
            &observation, &assessment, &replication,
            input("evidence:1", CriterionEvidenceDisposition::Deferred),
        ).unwrap();
        let replacement = original.supersede(
            input("evidence:2", CriterionEvidenceDisposition::Eligible),
            &observation, &assessment, &replication,
        ).unwrap();
        (commitment, observation, assessment, replication, original, replacement)
    }

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

    #[test]
    fn closed_history_projects_all_dispositions_and_supersession() {
        let (c, o, a, r, original, replacement) = evidence_chain();
        let p = EvidenceLineageDkgProjection::from_chain_history(
            &c, &o, &a, &r, &[original.clone(), replacement.clone()],
        ).unwrap();
        assert!(p.verify_integrity());
        assert_eq!(p.nodes.len(), 6);
        assert_eq!(p.edges.len(), 7);
        assert!(p.edges.iter().any(|e|
            e.source_node_id == replacement.evidence_id
                && e.edge_type == DkgEdgeType::Supersedes
                && e.target_node_id == original.evidence_id
        ));
        let replacement_node = p.nodes.iter().find(|n| n.node_id == replacement.evidence_id).unwrap();
        assert_eq!(replacement_node.supersedes_node_id.as_deref(), Some(original.evidence_id.as_str()));
    }

    #[test]
    fn single_superseded_record_requires_closed_history() {
        let (c, o, a, r, _original, replacement) = evidence_chain();
        assert_eq!(
            EvidenceLineageDkgProjection::from_chain(&c, &o, &a, &r, &replacement),
            Err(DkgProjectionError::MissingSupersessionParent)
        );
    }

    #[test]
    fn history_order_does_not_change_projection_digest() {
        let (c, o, assessment, r, original, replacement) = evidence_chain();
        let first = EvidenceLineageDkgProjection::from_chain_history(&c, &o, &assessment, &r, &[original.clone(), replacement.clone()]).unwrap();
        let second = EvidenceLineageDkgProjection::from_chain_history(&c, &o, &assessment, &r, &[replacement, original]).unwrap();
        assert_eq!(first, second);
        assert!(second.verify_integrity());
    }

    #[test]
    fn duplicate_node_ids_are_not_integrity_valid() {
        let n1 = node("same", DkgNodeType::ExternalObservation);
        let n2 = node("same", DkgNodeType::CriterionEvidence);
        let p = EvidenceLineageDkgProjection::new(vec![n1, n2], vec![]);
        assert!(!p.verify_integrity());
    }

    #[test]
    fn duplicate_edges_are_not_integrity_valid() {
        let n1 = node("a", DkgNodeType::ExternalObservation);
        let n2 = node("b", DkgNodeType::IndependentAssessment);
        let edge = DkgEdge { source_node_id: "b".into(), edge_type: DkgEdgeType::Assesses, target_node_id: "a".into() };
        let p = EvidenceLineageDkgProjection::new(vec![n1, n2], vec![edge.clone(), edge]);
        assert!(!p.verify_integrity());
    }
}
