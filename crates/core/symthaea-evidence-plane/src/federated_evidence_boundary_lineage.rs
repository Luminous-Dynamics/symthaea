//! Transitive audit receipts for federated evidence-boundary provenance.
//!
//! This layer closes the gap between a single projection/output-to-observation
//! boundary receipt and a multi-hop federated provenance graph. It commits to
//! the exact projection ancestry used to explain that boundary. It remains a
//! provenance artifact: ancestry does not qualify evidence, establish truth,
//! authenticate actors, or transfer authority.

use super::cross_dkg_projection::FederatedProjectionReceipt;
use super::federated_evidence_boundary::FederatedEvidenceBoundaryReceipt;
use super::external_observation::ExternalExperimentalObservation;
use super::cross_dkg_link_collection::CrossDkgLinkCollection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:federated-evidence-boundary-lineage:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedEvidenceBoundaryLineageReceipt {
    pub receipt_version: String,
    pub boundary_digest: String,
    pub projection_digest: String,
    pub ancestor_projection_digests: Vec<String>,
    pub observation_record_digest: String,
    pub lineage_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceBoundaryLineageError {
    InvalidBoundary,
    BoundaryProjectionMismatch,
    ObservationMismatch,
    InvalidProjectionAncestry,
    ProjectionNotFound,
    MissingAncestor,
    DuplicateAncestor,
    NonCanonicalAncestorOrder,
    InvalidDigest,
    LineageDigestMismatch,
}

impl FederatedEvidenceBoundaryLineageReceipt {
    /// Commit to the exact closed projection ancestry supporting a boundary.
    /// The caller supplies the terminal projection's exact collection because
    /// projection receipts intentionally commit to, rather than embed, their
    /// link collections.
    pub fn new(
        boundary: &FederatedEvidenceBoundaryReceipt,
        observation: &ExternalExperimentalObservation,
        projection: &FederatedProjectionReceipt,
        collection: &CrossDkgLinkCollection,
        ancestry: &[FederatedProjectionReceipt],
    ) -> Result<Self, EvidenceBoundaryLineageError> {
        boundary
            .verify(projection, collection, observation)
            .map_err(|_| EvidenceBoundaryLineageError::InvalidBoundary)?;
        if projection.projection_digest != boundary.projection_digest {
            return Err(EvidenceBoundaryLineageError::BoundaryProjectionMismatch);
        }
        if !observation.verify_integrity() {
            return Err(EvidenceBoundaryLineageError::ObservationMismatch);
        }
        if !ancestry.iter().any(|p| p.projection_digest == projection.projection_digest) {
            return Err(EvidenceBoundaryLineageError::ProjectionNotFound);
        }

        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| EvidenceBoundaryLineageError::InvalidProjectionAncestry)?;

        let mut ancestors = ancestry
            .iter()
            .map(|p| p.projection_digest.clone())
            .collect::<Vec<_>>();
        ancestors.sort();
        if ancestors.windows(2).any(|w| w[0] == w[1]) {
            return Err(EvidenceBoundaryLineageError::DuplicateAncestor);
        }
        if ancestors.iter().any(|d| !valid_digest(d)) {
            return Err(EvidenceBoundaryLineageError::InvalidDigest);
        }

        let mut receipt = Self {
            receipt_version: VERSION.into(),
            boundary_digest: boundary.boundary_digest.clone(),
            projection_digest: projection.projection_digest.clone(),
            ancestor_projection_digests: ancestors,
            observation_record_digest: observation.record_digest.clone(),
            lineage_digest: String::new(),
        };
        receipt.lineage_digest = receipt.compute_digest();
        Ok(receipt)
    }

    /// Verify the boundary, exact observation, closed ancestry, and the
    /// content-addressed lineage commitment. This answers only whether the
    /// provenance path is intact.
    pub fn verify(
        &self,
        boundary: &FederatedEvidenceBoundaryReceipt,
        observation: &ExternalExperimentalObservation,
        projection: &FederatedProjectionReceipt,
        collection: &CrossDkgLinkCollection,
        ancestry: &[FederatedProjectionReceipt],
    ) -> Result<(), EvidenceBoundaryLineageError> {
        boundary
            .verify(projection, collection, observation)
            .map_err(|_| EvidenceBoundaryLineageError::InvalidBoundary)?;
        if self.receipt_version != VERSION
            || self.boundary_digest != boundary.boundary_digest
            || self.projection_digest != projection.projection_digest
            || self.observation_record_digest != observation.record_digest
        {
            return Err(EvidenceBoundaryLineageError::InvalidBoundary);
        }
        if !observation.verify_integrity() {
            return Err(EvidenceBoundaryLineageError::ObservationMismatch);
        }

        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| EvidenceBoundaryLineageError::InvalidProjectionAncestry)?;

        if !ancestry.iter().any(|p| p.projection_digest == projection.projection_digest) {
            return Err(EvidenceBoundaryLineageError::ProjectionNotFound);
        }

        let mut expected = ancestry
            .iter()
            .map(|p| p.projection_digest.clone())
            .collect::<Vec<_>>();
        expected.sort();

        if expected != self.ancestor_projection_digests {
            return Err(EvidenceBoundaryLineageError::MissingAncestor);
        }
        if self.ancestor_projection_digests.windows(2).any(|w| w[0] >= w[1]) {
            return Err(EvidenceBoundaryLineageError::NonCanonicalAncestorOrder);
        }
        if self.ancestor_projection_digests.iter().any(|d| !valid_digest(d)) {
            return Err(EvidenceBoundaryLineageError::InvalidDigest);
        }
        if self.lineage_digest != self.compute_digest() {
            return Err(EvidenceBoundaryLineageError::LineageDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        for value in [
            self.receipt_version.as_str(),
            self.boundary_digest.as_str(),
            self.projection_digest.as_str(),
            self.observation_record_digest.as_str(),
        ] {
            put(&mut h, value);
        }
        for digest in &self.ancestor_projection_digests {
            put(&mut h, digest);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn valid_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..].iter().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_commitment::{
        commit_candidate_envelope, CandidatePredictionBinding, CandidatePredictionSource,
    };
    use crate::cross_dkg_adapter::{
        AdapterCapability, AdapterDisposition, CrossDkgAdapterDeclaration, CrossDkgAdapterReceipt,
    };
    use crate::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgRelation};
    use crate::external_observation::{ExternalObservationInput, ObservationDisposition};
    use crate::prospective::ProspectiveProvenance;

    fn setup() -> (FederatedProjectionReceipt, CrossDkgLinkCollection, ExternalExperimentalObservation) {
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
        let adapter = CrossDkgAdapterReceipt::derive_reference(
            &declaration, &link, AdapterDisposition::ReferenceOnly,
            "local:1", "sha256:o1",
        ).unwrap();
        let collection = CrossDkgLinkCollection::new(vec![link]).unwrap();
        let projection = FederatedProjectionReceipt::new(
            "projection:1", "1", &collection, &[adapter], &[],
        ).unwrap();

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
        let observation = ExternalExperimentalObservation::ingest(
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
        ).unwrap();

        (projection, collection, observation)
    }

    fn boundary(
        projection: &FederatedProjectionReceipt,
        collection: &CrossDkgLinkCollection,
        observation: &ExternalExperimentalObservation,
    ) -> FederatedEvidenceBoundaryReceipt {
        FederatedEvidenceBoundaryReceipt::from_external_observation(
            projection, collection, "local:1", observation,
        ).unwrap()
    }

    #[test]
    fn commits_exact_closed_ancestry() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let receipt = FederatedEvidenceBoundaryLineageReceipt::new(
            &boundary, &observation, &projection, &collection, &[projection.clone()],
        ).unwrap();
        assert_eq!(receipt.ancestor_projection_digests, vec![projection.projection_digest]);
        assert!(receipt.verify(
            &boundary, &observation, &projection, &collection, &[projection],
        ).is_ok());
    }

    #[test]
    fn rejects_truncated_ancestry() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let receipt = FederatedEvidenceBoundaryLineageReceipt::new(
            &boundary, &observation, &projection, &collection, &[projection.clone()],
        ).unwrap();
        assert_eq!(
            receipt.verify(
                &boundary, &observation, &projection, &collection, &[],
            ),
            Err(EvidenceBoundaryLineageError::InvalidProjectionAncestry)
        );
    }

    #[test]
    fn rejects_boundary_projection_substitution() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let mut substituted = boundary.clone();
        substituted.projection_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert_eq!(
            FederatedEvidenceBoundaryLineageReceipt::new(
                &substituted, &observation, &projection, &collection, &[projection],
            ),
            Err(EvidenceBoundaryLineageError::InvalidBoundary)
        );
    }

    #[test]
    fn rejects_observation_substitution() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let mut substituted = observation.clone();
        substituted.observer_id = "observer:changed".into();
        assert_eq!(
            FederatedEvidenceBoundaryLineageReceipt::new(
                &boundary, &substituted, &projection, &collection, &[projection],
            ),
            Err(EvidenceBoundaryLineageError::InvalidObservation)
        );
    }

    #[test]
    fn rejects_lineage_digest_tampering() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let mut receipt = FederatedEvidenceBoundaryLineageReceipt::new(
            &boundary, &observation, &projection, &collection, &[projection.clone()],
        ).unwrap();
        receipt.lineage_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert_eq!(
            receipt.verify(
                &boundary, &observation, &projection, &collection, &[projection],
            ),
            Err(EvidenceBoundaryLineageError::LineageDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let (projection, collection, observation) = setup();
        let boundary = boundary(&projection, &collection, &observation);
        let receipt = FederatedEvidenceBoundaryLineageReceipt::new(
            &boundary, &observation, &projection, &collection, &[projection],
        ).unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        assert_eq!(
            serde_json::from_slice::<FederatedEvidenceBoundaryLineageReceipt>(&bytes).unwrap(),
            receipt
        );
    }
}
