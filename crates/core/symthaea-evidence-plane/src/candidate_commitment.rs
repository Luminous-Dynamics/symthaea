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
pub enum VerificationFailure {
    CommitmentIntegrity,
    PayloadDigestMismatch,
    BindingDigestMismatch,
    CandidateIdMismatch,
    SourceCandidateIdMismatch,
    TestSpecificationMismatch,
    MeasurementSpecificationMismatch,
    LineageDigestMismatch,
    ProvenanceLineageMismatch,
    InvalidBinding(BindingError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    EmptyPredictionPayload,
    MissingBindingField(&'static str),
    Serialization,
    LineageMismatch,
    SourceMismatch,
    Commitment(CommitmentError),
}

impl VerificationFailure {
    /// Stable machine-readable diagnostic code for downstream logs and metrics.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::CommitmentIntegrity => "commitment_integrity",
            Self::PayloadDigestMismatch => "payload_digest_mismatch",
            Self::BindingDigestMismatch => "binding_digest_mismatch",
            Self::CandidateIdMismatch => "candidate_id_mismatch",
            Self::SourceCandidateIdMismatch => "source_candidate_id_mismatch",
            Self::TestSpecificationMismatch => "test_specification_mismatch",
            Self::MeasurementSpecificationMismatch => "measurement_specification_mismatch",
            Self::LineageDigestMismatch => "lineage_digest_mismatch",
            Self::ProvenanceLineageMismatch => "provenance_lineage_mismatch",
            Self::InvalidBinding(_) => "invalid_binding",
        }
    }
}

impl std::fmt::Display for VerificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBinding(error) => write!(f, "{}: {error:?}", self.code()),
            _ => write!(f, "{}", self.code()),
        }
    }
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
    ///
    /// Malformed bindings still return Err; a well-formed but mismatched
    /// envelope returns Ok(false), preserving the original API.
    pub fn verify_binding(
        &self,
        binding: &CandidatePredictionBinding,
    ) -> Result<bool, BindingError> {
        match self.verify_binding_detailed(binding) {
            Ok(()) => Ok(true),
            Err(VerificationFailure::InvalidBinding(error)) => Err(error),
            Err(_) => Ok(false),
        }
    }

    /// Verify the envelope and identify the first integrity relationship that
    /// does not hold. This is a pure diagnostic operation.
    pub fn verify_binding_detailed(
        &self,
        binding: &CandidatePredictionBinding,
    ) -> Result<(), VerificationFailure> {
        binding.validate().map_err(VerificationFailure::InvalidBinding)?;
        let binding_digest = binding.digest().map_err(VerificationFailure::InvalidBinding)?;
        let payload_digest = binding.payload_digest().map_err(VerificationFailure::InvalidBinding)?;
        let lineage_digest = binding.lineage_digest().map_err(VerificationFailure::InvalidBinding)?;

        if !self.commitment.verify_integrity() {
            return Err(VerificationFailure::CommitmentIntegrity);
        }
        if self.commitment.payload_digest() != payload_digest {
            return Err(VerificationFailure::PayloadDigestMismatch);
        }
        if self.binding_digest != binding_digest {
            return Err(VerificationFailure::BindingDigestMismatch);
        }
        if self.candidate_id != binding.candidate_id {
            return Err(VerificationFailure::CandidateIdMismatch);
        }
        if self.source_candidate_id != binding.source_candidate_id {
            return Err(VerificationFailure::SourceCandidateIdMismatch);
        }
        if self.test_specification_id != binding.test_specification_id {
            return Err(VerificationFailure::TestSpecificationMismatch);
        }
        if self.measurement_specification_id != binding.measurement_specification_id {
            return Err(VerificationFailure::MeasurementSpecificationMismatch);
        }
        if self.lineage_digest != lineage_digest {
            return Err(VerificationFailure::LineageDigestMismatch);
        }
        if self.commitment.provenance().model_lineage != lineage_digest {
            return Err(VerificationFailure::ProvenanceLineageMismatch);
        }
        Ok(())
    }

    /// Supersede this envelope without allowing the raw commitment primitive to
    /// bypass candidate identity and lineage invariants. A changed candidate
    /// source must be committed as a new candidate, not a supersession.
    pub fn supersede(
        &self,
        candidate: &CandidatePredictionSource,
        actor_id: &str,
        created_at: &str,
        provenance: ProspectiveProvenance,
        prediction_payload: &[u8],
    ) -> Result<Self, BindingError> {
        if candidate.candidate_id != self.candidate_id
            || candidate.source_candidate_id != self.source_candidate_id
            || candidate.test_specification_id != self.test_specification_id
            || candidate.measurement_specification_id != self.measurement_specification_id
            || lineage_digest(&candidate.left_lineage, &candidate.right_lineage) != self.lineage_digest
        {
            return Err(BindingError::SourceMismatch);
        }
        validate_provenance_lineage(candidate, &provenance)?;
        let binding = binding_for(candidate, prediction_payload);
        let bytes = binding.canonical_bytes()?;
        let commitment = self.commitment.supersede(
            actor_id,
            created_at,
            &bytes,
            provenance,
        )?;
        Ok(Self {
            commitment,
            candidate_id: binding.candidate_id,
            source_candidate_id: binding.source_candidate_id,
            test_specification_id: binding.test_specification_id,
            measurement_specification_id: binding.measurement_specification_id,
            lineage_digest: binding.lineage_digest()?,
            binding_digest: binding.digest()?,
        })
    }

    /// Verify that this envelope is an immutable supersession of the supplied parent.
    pub fn is_supersession_of(&self, parent: &Self) -> bool {
        self.commitment.verify_integrity()
            && parent.commitment.verify_integrity()
            && self.commitment.parent_event_ids().len() == 1
            && self.commitment.parent_event_ids()[0] == parent.commitment.event_id()
            && self.candidate_id == parent.candidate_id
            && self.source_candidate_id == parent.source_candidate_id
            && self.test_specification_id == parent.test_specification_id
            && self.measurement_specification_id == parent.measurement_specification_id
            && self.lineage_digest == parent.lineage_digest
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

    fn provenance(candidate: &CandidatePredictionSource) -> ProspectiveProvenance {
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
    fn envelope_supersession_preserves_parent_and_rebinds_payload() {
        let c = candidate();
        let original = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast-v1",
        ).unwrap();
        let replacement = original.supersede(
            &c, "actor-v2", "2026-09-28T10:00:00Z", provenance(&c), b"forecast-v2",
        ).unwrap();
        assert!(replacement.is_supersession_of(&original));
        assert_ne!(replacement.commitment.event_id(), original.commitment.event_id());
        assert!(replacement.verify_binding(&binding_for(&c, b"forecast-v2")).unwrap());
    }

    #[test]
    fn envelope_supersession_rejects_changed_source_identity() {
        let c = candidate();
        let original = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        let mut changed = c.clone();
        changed.candidate_id = "different-candidate".into();
        assert_eq!(
            original.supersede(
                &changed, "actor-v2", "2026-09-28T10:00:00Z", provenance(&c), b"forecast-v2",
            ),
            Err(BindingError::SourceMismatch)
        );
    }

    #[test]
    fn verification_failure_codes_are_stable_and_distinct() {
        let failures = [
            VerificationFailure::CommitmentIntegrity,
            VerificationFailure::PayloadDigestMismatch,
            VerificationFailure::BindingDigestMismatch,
            VerificationFailure::CandidateIdMismatch,
            VerificationFailure::SourceCandidateIdMismatch,
            VerificationFailure::TestSpecificationMismatch,
            VerificationFailure::MeasurementSpecificationMismatch,
            VerificationFailure::LineageDigestMismatch,
            VerificationFailure::ProvenanceLineageMismatch,
            VerificationFailure::InvalidBinding(BindingError::EmptyPredictionPayload),
        ];
        let codes: Vec<&str> = failures.iter().map(VerificationFailure::code).collect();
        let mut unique = codes.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(codes.len(), unique.len());
        assert_eq!(VerificationFailure::LineageDigestMismatch.to_string(), "lineage_digest_mismatch");
        assert_eq!(
            VerificationFailure::InvalidBinding(BindingError::EmptyPredictionPayload).to_string(),
            "invalid_binding: EmptyPredictionPayload"
        );
    }

    #[test]
    fn detailed_verification_reports_each_integrity_class() {
        let c = candidate();
        let binding = binding_for(&c, b"forecast");
        let make = || {
            commit_candidate_envelope(
                &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
                "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
            ).unwrap()
        };

        let mut commitment = make();
        let mut value = serde_json::to_value(&commitment.commitment).unwrap();
        value["event_id"] = serde_json::Value::String("sha256:tampered".into());
        commitment.commitment = serde_json::from_value(value).unwrap();
        assert_eq!(commitment.verify_binding_detailed(&binding), Err(VerificationFailure::CommitmentIntegrity));

        let mut payload = make();
        payload.commitment = ProspectivePredictionCommitment::commit(
            "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"tampered",
        ).unwrap();
        assert_eq!(payload.verify_binding_detailed(&binding), Err(VerificationFailure::PayloadDigestMismatch));

        let mut digest = make();
        digest.binding_digest = "sha256:tampered".into();
        assert_eq!(digest.verify_binding_detailed(&binding), Err(VerificationFailure::BindingDigestMismatch));

        let mut candidate_id = make();
        candidate_id.candidate_id = "tampered".into();
        assert_eq!(candidate_id.verify_binding_detailed(&binding), Err(VerificationFailure::CandidateIdMismatch));

        let mut source_id = make();
        source_id.source_candidate_id = "tampered".into();
        assert_eq!(source_id.verify_binding_detailed(&binding), Err(VerificationFailure::SourceCandidateIdMismatch));

        let mut test_spec = make();
        test_spec.test_specification_id = "tampered".into();
        assert_eq!(test_spec.verify_binding_detailed(&binding), Err(VerificationFailure::TestSpecificationMismatch));

        let mut measurement = make();
        measurement.measurement_specification_id = "tampered".into();
        assert_eq!(measurement.verify_binding_detailed(&binding), Err(VerificationFailure::MeasurementSpecificationMismatch));

        let mut lineage = make();
        lineage.lineage_digest = "sha256:tampered".into();
        assert_eq!(lineage.verify_binding_detailed(&binding), Err(VerificationFailure::LineageDigestMismatch));

        let mut provenance_lineage = make();
        provenance_lineage.commitment = {
            let mut value = serde_json::to_value(&provenance_lineage.commitment).unwrap();
            let wrong = lineage_digest("wrong-left", &c.right_lineage);
            value["provenance"]["model_lineage"] = serde_json::Value::String(wrong);
            serde_json::from_value(value).unwrap()
        };
        assert_eq!(
            provenance_lineage.verify_binding_detailed(&binding),
            Err(VerificationFailure::ProvenanceLineageMismatch)
        );
    }

    #[test]
    fn detailed_verification_preserves_malformed_binding_error() {
        let c = candidate();
        let envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        let mut binding = binding_for(&c, b"forecast");
        binding.prediction_payload.clear();

        assert_eq!(
            envelope.verify_binding_detailed(&binding),
            Err(VerificationFailure::InvalidBinding(BindingError::EmptyPredictionPayload))
        );
        assert_eq!(
            envelope.verify_binding(&binding),
            Err(BindingError::EmptyPredictionPayload)
        );
    }

    #[test]
    fn detailed_verification_reports_typed_mismatch() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.lineage_digest = "sha256:tampered".into();

        assert_eq!(
            envelope.verify_binding_detailed(&binding_for(&c, b"forecast")),
            Err(VerificationFailure::LineageDigestMismatch)
        );
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn tampered_outer_binding_digest_fails_verification() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.binding_digest = "sha256:tampered".into();
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn tampered_outer_lineage_digest_fails_verification() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.lineage_digest = "sha256:tampered".into();
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn tampered_outer_identity_fails_verification() {
        let c = candidate();
        let mut envelope = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        envelope.candidate_id = "tampered-candidate".into();
        assert!(!envelope.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn envelope_serde_round_trip_retains_integrity() {
        let c = candidate();
        let original = commit_candidate_envelope(
            &c, "challenge-v1", "criteria-v1", "mapping-v1", "actor-v1",
            "2026-09-28T09:00:00Z", provenance(&c), b"forecast",
        ).unwrap();
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: CandidatePredictionCommitment = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, original);
        assert!(decoded.verify_binding(&binding_for(&c, b"forecast")).unwrap());
    }

    #[test]
    fn lineage_digest_is_order_sensitive_and_delimiter_safe() {
        assert_ne!(lineage_digest("a:b", "c"), lineage_digest("a", "b:c"));
        assert_ne!(lineage_digest("left", "right"), lineage_digest("right", "left"));
    }
}
