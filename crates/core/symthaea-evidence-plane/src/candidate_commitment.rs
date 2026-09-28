// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifiable binding between a materialized scientific test candidate and a
//! prospective prediction commitment.
//!
//! This layer records planning identity only. It cannot emit observations,
//! replications, or official-criterion evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::prospective::{
    CommitmentError, ProspectivePredictionCommitment, ProspectiveProvenance,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePredictionSource {
    pub candidate_id: String,
    pub source_candidate_id: String,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePredictionBinding {
    pub candidate_id: String,
    pub source_candidate_id: String,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub prediction_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    EmptyPredictionPayload,
    MissingBindingField(&'static str),
    Serialization,
    LineageMismatch,
    Commitment(CommitmentError),
}

impl From<CommitmentError> for BindingError {
    fn from(value: CommitmentError) -> Self {
        Self::Commitment(value)
    }
}

impl CandidatePredictionBinding {
    pub fn validate(&self) -> Result<(), BindingError> {
        for (name, value) in [
            ("candidate_id", &self.candidate_id),
            ("source_candidate_id", &self.source_candidate_id),
            ("test_specification_id", &self.test_specification_id),
            ("measurement_specification_id", &self.measurement_specification_id),
            ("left_lineage", &self.left_lineage),
            ("right_lineage", &self.right_lineage),
        ] {
            if value.trim().is_empty() {
                return Err(BindingError::MissingBindingField(name));
            }
        }
        if self.prediction_payload.is_empty() {
            return Err(BindingError::EmptyPredictionPayload);
        }
        Ok(())
    }

    /// Canonical serialized binding bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BindingError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| BindingError::Serialization)
    }

    /// Domain-separated digest of the canonical binding.
    pub fn digest(&self) -> Result<String, BindingError> {
        let bytes = self.canonical_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:candidate-prediction-binding:v1\0");
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }

    /// Plain SHA-256 of the canonical bytes, matching the generic commitment
    /// payload digest.
    pub fn payload_digest(&self) -> Result<String, BindingError> {
        let bytes = self.canonical_bytes()?;
        Ok(sha256(&bytes))
    }

    pub fn lineage_digest(&self) -> Result<String, BindingError> {
        self.validate()?;
        Ok(lineage_digest(&self.left_lineage, &self.right_lineage))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePredictionCommitment {
    pub commitment: ProspectivePredictionCommitment,
    pub candidate_id: String,
    pub source_candidate_id: String,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
    pub lineage_digest: String,
    pub binding_digest: String,
}

impl CandidatePredictionCommitment {
    /// Verify every independently recomputable identity relationship.
    pub fn verify_binding(
        &self,
        binding: &CandidatePredictionBinding,
    ) -> Result<bool, BindingError> {
        let binding_digest = binding.digest()?;
        let payload_digest = binding.payload_digest()?;
        let lineage_digest = binding.lineage_digest()?;

        Ok(
            self.commitment.verify_integrity()
                && self.commitment.payload_digest() == payload_digest
                && self.binding_digest == binding_digest
                && self.candidate_id == binding.candidate_id
                && self.source_candidate_id == binding.source_candidate_id
                && self.test_specification_id == binding.test_specification_id
                && self.measurement_specification_id == binding.measurement_specification_id
                && self.lineage_digest == lineage_digest
                && self.commitment.provenance().model_lineage == lineage_digest,
        )
    }
}

pub fn commit_candidate(
    candidate: &CandidatePredictionSource,
    challenge_id: &str,
    criteria_generation: &str,
    mapping_generation: &str,
    actor_id: &str,
    created_at: &str,
    provenance: ProspectiveProvenance,
    prediction_payload: &[u8],
) -> Result<ProspectivePredictionCommitment, BindingError> {
    let binding = binding_for(candidate, prediction_payload);
    validate_provenance_lineage(candidate, &provenance)?;
    let bytes = binding.canonical_bytes()?;
    Ok(ProspectivePredictionCommitment::commit(
        challenge_id,
        criteria_generation,
        mapping_generation,
        actor_id,
        created_at,
        provenance,
        &bytes,
    )?)
}

pub fn commit_candidate_envelope(
    candidate: &CandidatePredictionSource,
    challenge_id: &str,
    criteria_generation: &str,
    mapping_generation: &str,
    actor_id: &str,
    created_at: &str,
    provenance: ProspectiveProvenance,
    prediction_payload: &[u8],
) -> Result<CandidatePredictionCommitment, BindingError> {
    let binding = binding_for(candidate, prediction_payload);
    validate_provenance_lineage(candidate, &provenance)?;
    let bytes = binding.canonical_bytes()?;
    let commitment = ProspectivePredictionCommitment::commit(
        challenge_id,
        criteria_generation,
        mapping_generation,
        actor_id,
        created_at,
        provenance,
        &bytes,
    )?;
    Ok(CandidatePredictionCommitment {
        commitment,
        candidate_id: binding.candidate_id,
        source_candidate_id: binding.source_candidate_id,
        test_specification_id: binding.test_specification_id,
        measurement_specification_id: binding.measurement_specification_id,
        lineage_digest: binding.lineage_digest()?,
        binding_digest: binding.digest()?,
    })
}

fn binding_for(
    candidate: &CandidatePredictionSource,
    prediction_payload: &[u8],
) -> CandidatePredictionBinding {
    CandidatePredictionBinding {
        candidate_id: candidate.candidate_id.clone(),
        source_candidate_id: candidate.source_candidate_id.clone(),
        test_specification_id: candidate.test_specification_id.clone(),
        measurement_specification_id: candidate.measurement_specification_id.clone(),
        left_lineage: candidate.left_lineage.clone(),
        right_lineage: candidate.right_lineage.clone(),
        prediction_payload: prediction_payload.to_vec(),
    }
}

fn lineage_digest(left: &str, right: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:candidate-model-lineage:v1\0");
    for value in [left, right] {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn validate_provenance_lineage(
    candidate: &CandidatePredictionSource,
    provenance: &ProspectiveProvenance,
) -> Result<(), BindingError> {
    let expected = lineage_digest(&candidate.left_lineage, &candidate.right_lineage);
    if provenance.model_lineage != expected {
        return Err(BindingError::LineageMismatch);
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> CandidatePredictionSource {
        CandidatePredictionSource {
            candidate_id: "test-candidate-1".into(),
            source_candidate_id: "temporal-source-1".into(),
            test_specification_id: "test-spec-1".into(),
            measurement_specification_id: "measurement-1".into(),
            left_lineage: "lineage-a".into(),
            right_lineage: "lineage-b".into(),
        }
    }

    fn provenance(candidate: &TemporalTestCandidateSpec) -> ProspectiveProvenance {
        ProspectiveProvenance::new(
            "sha256:input",
            "sha256:artifact",
            lineage_digest(&candidate.left_lineage, &candidate.right_lineage),
            "2026-09-28T08:00:00Z",
            "2026-09-28T09:00:00Z",
        )
        .unwrap()
    }

    #[test]
    fn envelope_verifies_all_identity_layers() {
        let c = candidate();
        let envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        let binding = binding_for(&c, b"forecast");

        assert!(envelope.verify_binding(&binding).unwrap());
        assert_eq!(envelope.binding_digest, binding.digest().unwrap());
        assert_eq!(envelope.commitment.payload_digest(), binding.payload_digest().unwrap());
        assert_eq!(envelope.lineage_digest, binding.lineage_digest().unwrap());
    }

    #[test]
    fn tampered_commitment_payload_digest_fails_verification() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.commitment = ProspectivePredictionCommitment::commit(
            "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"tampered",
        ).unwrap();
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn tampered_event_id_fails_verification() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.commitment = {
            let mut commitment = envelope.commitment.clone();
            // Private event_id cannot be mutated directly outside this module;
            // serde round-trip below models an untrusted deserialized record.
            let mut value = serde_json::to_value(&commitment).unwrap();
            value["event_id"] = serde_json::Value::String("sha256:tampered".into());
            commitment = serde_json::from_value(value).unwrap();
            commitment
        };
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn mismatched_provenance_lineage_is_rejected() {
        let c = candidate();
        let bad = ProspectiveProvenance::new(
            "sha256:input",
            "sha256:artifact",
            lineage_digest("wrong-left", &c.right_lineage),
            "2026-09-28T08:00:00Z",
            "2026-09-28T09:00:00Z",
        ).unwrap();
        assert_eq!(
            commit_candidate(
                &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
                "2026-09-28T09:00:00Z", bad, b"forecast",
            ),
            Err(BindingError::LineageMismatch)
        );
    }

    #[test]
    fn lineage_digest_is_order_sensitive_and_delimiter_safe() {
        assert_ne!(lineage_digest("a:b", "c"), lineage_digest("a", "b:c"));
        assert_ne!(lineage_digest("left", "right"), lineage_digest("right", "left"));
    }
}
