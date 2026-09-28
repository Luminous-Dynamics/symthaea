// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only audit projection for prospective prediction commitments.
//!
//! Audit history records what commitment was made and how it relates to prior
//! commitments. It is not experimental evidence and cannot promote a record.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::candidate_commitment::CandidatePredictionCommitment;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProspectiveCommitmentAudit {
    pub event_id: String,
    pub parent_event_ids: Vec<String>,
    pub challenge_id: String,
    pub criteria_generation: String,
    pub mapping_generation: String,
    pub actor_id: String,
    pub created_at: String,
    pub knowledge_cutoff: String,
    pub exposure_cutoff: String,
    pub binding_digest: String,
    pub lineage_digest: String,
    pub audit_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditError {
    InvalidCommitment,
}

impl ProspectiveCommitmentAudit {
    pub fn from_commitment(commitment: &CandidatePredictionCommitment) -> Result<Self, AuditError> {
        if !commitment.commitment.verify_integrity() {
            return Err(AuditError::InvalidCommitment);
        }
        let p = commitment.commitment.provenance();
        let audit_digest = compute_audit_digest(
            commitment.commitment.event_id(),
            commitment.commitment.parent_event_ids(),
            commitment.commitment.challenge_id(),
            commitment.commitment.criteria_generation(),
            commitment.commitment.mapping_generation(),
            commitment.commitment.actor_id(),
            commitment.commitment.created_at(),
            &p.knowledge_cutoff,
            &p.exposure_cutoff,
            &commitment.binding_digest,
            &commitment.lineage_digest,
        );
        Ok(Self {
            event_id: commitment.commitment.event_id().to_owned(),
            parent_event_ids: commitment.commitment.parent_event_ids().to_vec(),
            challenge_id: commitment.commitment.challenge_id().to_owned(),
            criteria_generation: commitment.commitment.criteria_generation().to_owned(),
            mapping_generation: commitment.commitment.mapping_generation().to_owned(),
            actor_id: commitment.commitment.actor_id().to_owned(),
            created_at: commitment.commitment.created_at().to_owned(),
            knowledge_cutoff: p.knowledge_cutoff.clone(),
            exposure_cutoff: p.exposure_cutoff.clone(),
            binding_digest: commitment.binding_digest.clone(),
            lineage_digest: commitment.lineage_digest.clone(),
            audit_digest,
        })
    }

    pub fn verify_integrity(&self) -> bool {
        !self.event_id.trim().is_empty()
            && !self.challenge_id.trim().is_empty()
            && !self.criteria_generation.trim().is_empty()
            && !self.mapping_generation.trim().is_empty()
            && !self.actor_id.trim().is_empty()
            && !self.created_at.trim().is_empty()
            && !self.knowledge_cutoff.trim().is_empty()
            && !self.exposure_cutoff.trim().is_empty()
            && !self.binding_digest.trim().is_empty()
            && !self.lineage_digest.trim().is_empty()
            && self.audit_digest == compute_audit_digest(
                &self.event_id,
                &self.parent_event_ids,
                &self.challenge_id,
                &self.criteria_generation,
                &self.mapping_generation,
                &self.actor_id,
                &self.created_at,
                &self.knowledge_cutoff,
                &self.exposure_cutoff,
                &self.binding_digest,
                &self.lineage_digest,
            )
    }

    /// Supersession is lineage history, not corroboration.
    pub fn is_parent_of(&self, child: &Self) -> bool {
        self.verify_integrity()
            && child.verify_integrity()
            && child.parent_event_ids.len() == 1
            && child.parent_event_ids[0] == self.event_id
    }
}

fn compute_audit_digest(
    event_id: &str,
    parents: &[String],
    challenge_id: &str,
    criteria_generation: &str,
    mapping_generation: &str,
    actor_id: &str,
    created_at: &str,
    knowledge_cutoff: &str,
    exposure_cutoff: &str,
    binding_digest: &str,
    lineage_digest: &str,
) -> String {
    let mut h = Sha256::new();
    h.update(b"symthaea:prospective-commitment-audit:v1\0");
    for value in [
        event_id, challenge_id, criteria_generation, mapping_generation, actor_id,
        created_at, knowledge_cutoff, exposure_cutoff, binding_digest, lineage_digest,
    ] {
        h.update((value.len() as u64).to_be_bytes());
        h.update(value.as_bytes());
    }
    h.update((parents.len() as u64).to_be_bytes());
    for parent in parents {
        h.update((parent.len() as u64).to_be_bytes());
        h.update(parent.as_bytes());
    }
    format!("sha256:{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_commitment::{
        CandidatePredictionBinding, CandidatePredictionSource, commit_candidate_envelope,
    };
    use crate::prospective::ProspectiveProvenance;

    fn candidate() -> CandidatePredictionSource {
        CandidatePredictionSource {
            candidate_id: "candidate:1".into(),
            source_candidate_id: "source:1".into(),
            test_specification_id: "test:1".into(),
            measurement_specification_id: "measurement:1".into(),
            left_lineage: "left:lineage".into(),
            right_lineage: "right:lineage".into(),
        }
    }

    fn provenance() -> ProspectiveProvenance {
        let lineage = CandidatePredictionBinding::from_source(&candidate(), b"p")
            .unwrap().lineage_digest().unwrap();
        ProspectiveProvenance::new(
            "sha256:input", "sha256:artifact", lineage,
            "2026-09-28T08:00:00Z", "2026-09-28T09:00:00Z",
        ).unwrap()
    }

    fn commitment() -> CandidatePredictionCommitment {
        commit_candidate_envelope(
            &candidate(), "challenge:1", "criteria:1", "mapping:1",
            "actor:1", "2026-09-28T09:00:00Z", provenance(), b"prediction",
        ).unwrap()
    }

    #[test]
    fn projection_is_deterministic_and_round_trips() {
        let audit = ProspectiveCommitmentAudit::from_commitment(&commitment()).unwrap();
        let encoded = serde_json::to_vec(&audit).unwrap();
        let decoded: ProspectiveCommitmentAudit = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(audit, decoded);
        assert!(decoded.verify_integrity());
    }

    #[test]
    fn tampering_identity_or_digest_is_detected() {
        let mut audit = ProspectiveCommitmentAudit::from_commitment(&commitment()).unwrap();
        audit.actor_id = "actor:tampered".into();
        assert!(!audit.verify_integrity());
        let mut audit = ProspectiveCommitmentAudit::from_commitment(&commitment()).unwrap();
        audit.audit_digest = "sha256:tampered".into();
        assert!(!audit.verify_integrity());
    }

    #[test]
    fn supersession_is_lineage_not_corroboration() {
        let parent = commitment();
        let child = parent.supersede(
            &candidate(), "actor:2", "2026-09-28T10:00:00Z",
            provenance(), b"prediction-v2",
        ).unwrap();
        let parent_audit = ProspectiveCommitmentAudit::from_commitment(&parent).unwrap();
        let child_audit = ProspectiveCommitmentAudit::from_commitment(&child).unwrap();
        assert!(parent_audit.is_parent_of(&child_audit));
    }

    #[test]
    fn invalid_commitment_is_rejected() {
        let mut value = commitment();
        value.commitment = serde_json::from_value({
            let mut raw = serde_json::to_value(&value.commitment).unwrap();
            raw["event_id"] = serde_json::json!("sha256:tampered");
            raw
        }).unwrap();
        assert_eq!(
            ProspectiveCommitmentAudit::from_commitment(&value),
            Err(AuditError::InvalidCommitment)
        );
    }
}
