//! Explicit boundary receipts between federated projections and external observations.
//!
//! A projection is never itself an external observation. This module records
//! the narrower fact that one exact projection output has an independently
//! ingested external observation counterpart. The receipt is provenance only:
//! it does not qualify the observation, establish truth, authenticate actors,
//! or satisfy an official challenge criterion.

use super::cross_dkg_projection::FederatedProjectionReceipt;
use super::external_observation::ExternalExperimentalObservation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:federated-evidence-boundary:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FederatedEvidenceBoundaryDisposition {
    ExternalObservationBacked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedEvidenceBoundaryReceipt {
    pub receipt_version: String,
    pub projection_digest: String,
    pub output_record_id: String,
    pub output_record_digest: String,
    pub observation_id: String,
    pub observation_record_digest: String,
    pub execution_id: String,
    pub observer_id: String,
    pub institution_id: String,
    pub disposition: FederatedEvidenceBoundaryDisposition,
    pub boundary_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceBoundaryError {
    InvalidProjection,
    OutputNotFound,
    OutputDigestMismatch,
    InvalidObservation,
    EmptyField(&'static str),
    UnsupportedDisposition,
    BoundaryDigestMismatch,
}

impl FederatedEvidenceBoundaryReceipt {
    /// Bind one exact projection output to an independently ingested external
    /// observation. The projection remains provenance; the observation remains
    /// the evidence-bearing external record.
    pub fn from_external_observation(
        projection: &FederatedProjectionReceipt,
        collection: &super::cross_dkg_link_collection::CrossDkgLinkCollection,
        output_record_id: impl Into<String>,
        observation: &ExternalExperimentalObservation,
    ) -> Result<Self, EvidenceBoundaryError> {
        projection
            .verify_integrity(collection)
            .map_err(|_| EvidenceBoundaryError::InvalidProjection)?;

        if !observation.verify_integrity() {
            return Err(EvidenceBoundaryError::InvalidObservation);
        }

        let output_record_id = output_record_id.into();
        if output_record_id.trim().is_empty() {
            return Err(EvidenceBoundaryError::EmptyField("output_record_id"));
        }

        let output = projection
            .outputs
            .iter()
            .find(|o| o.record_id == output_record_id)
            .ok_or(EvidenceBoundaryError::OutputNotFound)?;

        if output.record_digest.trim().is_empty() {
            return Err(EvidenceBoundaryError::OutputDigestMismatch);
        }

        for (name, value) in [
            ("observation_id", observation.observation_id.as_str()),
            ("observation_record_digest", observation.record_digest.as_str()),
            ("execution_id", observation.execution_id.as_str()),
            ("observer_id", observation.observer_id.as_str()),
            ("institution_id", observation.institution_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(EvidenceBoundaryError::EmptyField(name));
            }
        }

        let mut receipt = Self {
            receipt_version: VERSION.into(),
            projection_digest: projection.projection_digest.clone(),
            output_record_id,
            output_record_digest: output.record_digest.clone(),
            observation_id: observation.observation_id.clone(),
            observation_record_digest: observation.record_digest.clone(),
            execution_id: observation.execution_id.clone(),
            observer_id: observation.observer_id.clone(),
            institution_id: observation.institution_id.clone(),
            disposition: FederatedEvidenceBoundaryDisposition::ExternalObservationBacked,
            boundary_digest: String::new(),
        };
        receipt.boundary_digest = receipt.compute_digest();
        Ok(receipt)
    }

    /// Verify the boundary receipt and the exact external observation it names.
    /// This verifies provenance/linkage only, not scientific validity.
    pub fn verify(
        &self,
        projection: &FederatedProjectionReceipt,
        collection: &super::cross_dkg_link_collection::CrossDkgLinkCollection,
        observation: &ExternalExperimentalObservation,
    ) -> Result<(), EvidenceBoundaryError> {
        projection
            .verify_integrity(collection)
            .map_err(|_| EvidenceBoundaryError::InvalidProjection)?;

        if !observation.verify_integrity() {
            return Err(EvidenceBoundaryError::InvalidObservation);
        }

        if self.receipt_version != VERSION
            || self.projection_digest != projection.projection_digest
            || self.observation_id != observation.observation_id
            || self.observation_record_digest != observation.record_digest
            || self.execution_id != observation.execution_id
            || self.observer_id != observation.observer_id
            || self.institution_id != observation.institution_id
            || self.disposition != FederatedEvidenceBoundaryDisposition::ExternalObservationBacked
        {
            return Err(EvidenceBoundaryError::BoundaryDigestMismatch);
        }

        let output = projection
            .outputs
            .iter()
            .find(|o| o.record_id == self.output_record_id)
            .ok_or(EvidenceBoundaryError::OutputNotFound)?;

        if output.record_digest != self.output_record_digest {
            return Err(EvidenceBoundaryError::OutputDigestMismatch);
        }

        if self.boundary_digest != self.compute_digest() {
            return Err(EvidenceBoundaryError::BoundaryDigestMismatch);
        }

        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            self.receipt_version.as_str(),
            self.projection_digest.as_str(),
            self.output_record_id.as_str(),
            self.output_record_digest.as_str(),
            self.observation_id.as_str(),
            self.observation_record_digest.as_str(),
            self.execution_id.as_str(),
            self.observer_id.as_str(),
            self.institution_id.as_str(),
            "external_observation_backed",
        ] {
            h.update((value.len() as u64).to_be_bytes());
            h.update(value.as_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_commitment::{
        commit_candidate_envelope, CandidatePredictionBinding, CandidatePredictionSource,
    };
    use crate::external_observation::{ExternalObservationInput, ObservationDisposition};
    use crate::prospective::ProspectiveProvenance;
    use crate::cross_dkg_adapter::{
        AdapterCapability, AdapterDisposition, CrossDkgAdapterDeclaration, CrossDkgAdapterReceipt,
    };
    use crate::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgRelation};
    use crate::cross_dkg_link_collection::CrossDkgLinkCollection;

    fn projection() -> (FederatedProjectionReceipt, CrossDkgLinkCollection) {
        let declaration = CrossDkgAdapterDeclaration {
            adapter_id: "mycelix-to-symthaea".into(),
            adapter_version: "1.0".into(),
            source_dkg_id: "mycelix".into(),
            target_dkg_id: "symthaea".into(),
            allowed_relations: vec![CrossDkgRelation::References],
            capability: AdapterCapability::ReferenceOnly,
            policy_ref: "policy:research-reference".into(),
        };
        let link = CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:1", "sha256:r1",
            CrossDkgRelation::References, "symthaea", "sha256:g2",
            "observation:1", "sha256:t1",
        ).unwrap();
        let receipt = CrossDkgAdapterReceipt::derive_reference(
            &declaration, &link, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1",
        ).unwrap();
        let collection = CrossDkgLinkCollection::new(vec![link]).unwrap();
        (
            FederatedProjectionReceipt::new(
                "projection:1", "1", &collection, &[receipt], &[],
            ).unwrap(),
            collection,
        )
    }

    fn observation() -> ExternalExperimentalObservation {
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
            &source, "challenge", "criteria", "mapping", "predictor",
            "2026-09-28T09:00:00Z", provenance, b"prediction",
        ).unwrap();
        ExternalExperimentalObservation::ingest(
            &commitment, &binding,
            ExternalObservationInput {
                observation_id: "obs:1".into(),
                execution_id: "exec:1".into(),
                observer_id: "observer:1".into(),
                institution_id: "institution:1".into(),
                observed_at: "2026-09-29T10:00:00Z".into(),
                disposition: ObservationDisposition::Reported,
                observation_payload: b"external observation".to_vec(),
            },
        ).unwrap()
    }

    #[test]
    fn binds_only_when_external_observation_exists() {
        let (p, c) = projection();
        let o = observation();
        let r = FederatedEvidenceBoundaryReceipt::from_external_observation(
            &p, &c, "local:1", &o,
        ).unwrap();
        assert!(r.verify(&p, &c, &o).is_ok());
    }

    #[test]
    fn rejects_observation_tampering() {
        let (p, c) = projection();
        let o = observation();
        let r = FederatedEvidenceBoundaryReceipt::from_external_observation(
            &p, &c, "local:1", &o,
        ).unwrap();
        let mut tampered = o;
        tampered.observer_id = "observer:changed".into();
        assert_eq!(
            r.verify(&p, &c, &tampered),
            Err(EvidenceBoundaryError::InvalidObservation)
        );
    }

    #[test]
    fn rejects_unknown_projection_output() {
        let (p, c) = projection();
        let o = observation();
        assert_eq!(
            FederatedEvidenceBoundaryReceipt::from_external_observation(
                &p, &c, "not-in-projection", &o,
            ),
            Err(EvidenceBoundaryError::OutputNotFound)
        );
    }

    #[test]
    fn boundary_receipt_tampering_is_rejected() {
        let (p, c) = projection();
        let o = observation();
        let mut r = FederatedEvidenceBoundaryReceipt::from_external_observation(
            &p, &c, "local:1", &o,
        ).unwrap();
        r.institution_id = "changed".into();
        assert_eq!(
            r.verify(&p, &c, &o),
            Err(EvidenceBoundaryError::BoundaryDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let (p, c) = projection();
        let o = observation();
        let r = FederatedEvidenceBoundaryReceipt::from_external_observation(
            &p, &c, "local:1", &o,
        ).unwrap();
        let bytes = serde_json::to_vec(&r).unwrap();
        let decoded: FederatedEvidenceBoundaryReceipt = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(r, decoded);
        assert!(decoded.verify(&p, &c, &o).is_ok());
    }
}
