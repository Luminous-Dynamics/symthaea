//! Deterministic bounded provenance proofs for federated projection outputs.
//!
//! This is the query/receipt layer above exact output lineage (003X) and
//! qualified derivation context (003Y). It can optionally carry the exact
//! external-observation boundary lineage. It is provenance-only: it does not
//! establish scientific truth, qualify evidence, authenticate actors, transfer
//! authority, or prove global graph completeness.
//!
//! W3C PROV-AQ explicitly supports provenance query services and scoped
//! retrieval. This module adopts that idea without importing an RDF query
//! language into the evidence-plane wire model.

use super::cross_dkg_link_collection::CrossDkgLinkCollection;
use super::cross_dkg_projection::FederatedProjectionReceipt;
use super::federated_evidence_boundary::{
    FederatedEvidenceBoundaryReceipt,
};
use super::federated_evidence_boundary_lineage::{
    EvidenceBoundaryLineageError, FederatedEvidenceBoundaryLineageReceipt,
};
use super::federated_projection_output_lineage::{
    FederatedProjectionOutputLineageReceipt, ProjectionOutputLineageError,
};
use super::federated_qualified_derivation::{
    FederatedQualifiedDerivationReceipt, QualifiedDerivationError,
};
use super::external_observation::ExternalExperimentalObservation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:federated-provenance-proof:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceScope {
    ClosedProjectionSet,
    DepthLimited { max_depth: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceCompleteness {
    CompleteWithinScope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProvenanceQuery {
    pub query_version: String,
    pub target_projection_digest: String,
    pub target_output_record_id: String,
    pub target_output_record_digest: String,
    pub scope: ProvenanceScope,
    pub include_derivation_context: bool,
    pub include_external_observation_boundary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProvenanceProof {
    pub proof_version: String,
    pub query: FederatedProvenanceQuery,
    pub target_output_lineage_digest: String,
    pub target_projection_ancestry: Vec<String>,
    pub completeness: ProvenanceCompleteness,
    pub derivation_context_digest: Option<String>,
    pub boundary_lineage_digest: Option<String>,
    pub observation_record_digest: Option<String>,
    pub proof_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceProofError {
    InvalidQuery,
    InvalidDigest,
    QueryTargetMismatch,
    InvalidProjectionAncestry,
    OutputLineage(ProjectionOutputLineageError),
    QualifiedDerivation(QualifiedDerivationError),
    BoundaryLineage(EvidenceBoundaryLineageError),
    MissingDerivationContext,
    UnexpectedDerivationContext,
    MissingBoundaryLineage,
    UnexpectedBoundaryLineage,
    NonCanonicalAncestry,
    ProofDigestMismatch,
}

impl FederatedProvenanceQuery {
    pub fn new(
        target_projection_digest: impl Into<String>,
        target_output_record_id: impl Into<String>,
        target_output_record_digest: impl Into<String>,
        scope: ProvenanceScope,
        include_derivation_context: bool,
        include_external_observation_boundary: bool,
    ) -> Result<Self, ProvenanceProofError> {
        let query = Self {
            query_version: VERSION.into(),
            target_projection_digest: target_projection_digest.into(),
            target_output_record_id: target_output_record_id.into(),
            target_output_record_digest: target_output_record_digest.into(),
            scope,
            include_derivation_context,
            include_external_observation_boundary,
        };
        query.validate()?;
        Ok(query)
    }

    fn validate(&self) -> Result<(), ProvenanceProofError> {
        if self.query_version != VERSION
            || !valid_digest(&self.target_projection_digest)
            || self.target_output_record_id.trim().is_empty()
            || !valid_digest(&self.target_output_record_digest)
        {
            return Err(ProvenanceProofError::InvalidQuery);
        }
        if matches!(self.scope, ProvenanceScope::DepthLimited { max_depth: 0 }) {
            return Err(ProvenanceProofError::InvalidQuery);
        }
        Ok(())
    }
}

impl FederatedProvenanceProof {
    /// Construct a content-addressed proof from exact, closed provenance
    /// receipts. A depth-limited query bounds the requested closure; it does
    /// not turn omitted graph records into claims of absence.
    pub fn new(
        query: FederatedProvenanceQuery,
        output_lineage: &FederatedProjectionOutputLineageReceipt,
        ancestry: &[FederatedProjectionReceipt],
        derivation: Option<&FederatedQualifiedDerivationReceipt>,
        boundary: Option<(
            &FederatedEvidenceBoundaryLineageReceipt,
            &FederatedEvidenceBoundaryReceipt,
            &FederatedProjectionReceipt,
            &CrossDkgLinkCollection,
            &ExternalExperimentalObservation,
        )>,
    ) -> Result<Self, ProvenanceProofError> {
        query.validate()?;

        if output_lineage.projection_digest != query.target_projection_digest
            || output_lineage.output_record_id != query.target_output_record_id
            || output_lineage.output_record_digest != query.target_output_record_digest
        {
            return Err(ProvenanceProofError::QueryTargetMismatch);
        }

        let target = ancestry
            .iter()
            .find(|p| p.projection_digest == query.target_projection_digest)
            .ok_or(ProvenanceProofError::InvalidProjectionAncestry)?;

        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| ProvenanceProofError::InvalidProjectionAncestry)?;

        output_lineage
            .verify(target, ancestry)
            .map_err(ProvenanceProofError::OutputLineage)?;

        if query.include_derivation_context {
            let derivation = derivation.ok_or(ProvenanceProofError::MissingDerivationContext)?;
            derivation
                .verify(output_lineage)
                .map_err(ProvenanceProofError::QualifiedDerivation)?;
        } else if derivation.is_some() {
            return Err(ProvenanceProofError::UnexpectedDerivationContext);
        }

        let (boundary_lineage_digest, observation_record_digest) =
            if query.include_external_observation_boundary {
                let (boundary_lineage, boundary_receipt, projection, collection, observation) =
                    boundary.ok_or(ProvenanceProofError::MissingBoundaryLineage)?;
                boundary_lineage
                    .verify(
                        boundary_receipt,
                        observation,
                        projection,
                        collection,
                        ancestry,
                    )
                    .map_err(ProvenanceProofError::BoundaryLineage)?;
                (
                    Some(boundary_lineage.lineage_digest.clone()),
                    Some(observation.record_digest.clone()),
                )
            } else {
                if boundary.is_some() {
                    return Err(ProvenanceProofError::UnexpectedBoundaryLineage);
                }
                (None, None)
            };

        let mut proof = Self {
            proof_version: VERSION.into(),
            query,
            target_output_lineage_digest: output_lineage.lineage_digest.clone(),
            target_projection_ancestry: canonical_ancestry(ancestry)?,
            completeness: ProvenanceCompleteness::CompleteWithinScope,
            derivation_context_digest: derivation.map(|d| d.context_digest.clone()),
            boundary_lineage_digest,
            observation_record_digest,
            proof_digest: String::new(),
        };
        proof.proof_digest = proof.compute_digest();
        Ok(proof)
    }

    pub fn verify(
        &self,
        output_lineage: &FederatedProjectionOutputLineageReceipt,
        ancestry: &[FederatedProjectionReceipt],
        derivation: Option<&FederatedQualifiedDerivationReceipt>,
        boundary: Option<(
            &FederatedEvidenceBoundaryLineageReceipt,
            &FederatedEvidenceBoundaryReceipt,
            &FederatedProjectionReceipt,
            &CrossDkgLinkCollection,
            &ExternalExperimentalObservation,
        )>,
    ) -> Result<(), ProvenanceProofError> {
        if self.proof_version != VERSION
            || self.completeness != ProvenanceCompleteness::CompleteWithinScope
        {
            return Err(ProvenanceProofError::InvalidQuery);
        }
        self.query.validate()?;

        if output_lineage.projection_digest != self.query.target_projection_digest
            || output_lineage.output_record_id != self.query.target_output_record_id
            || output_lineage.output_record_digest != self.query.target_output_record_digest
        {
            return Err(ProvenanceProofError::QueryTargetMismatch);
        }

        let target = ancestry
            .iter()
            .find(|p| p.projection_digest == self.query.target_projection_digest)
            .ok_or(ProvenanceProofError::InvalidProjectionAncestry)?;

        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| ProvenanceProofError::InvalidProjectionAncestry)?;

        output_lineage
            .verify(target, ancestry)
            .map_err(ProvenanceProofError::OutputLineage)?;

        let expected_ancestry = canonical_ancestry(ancestry)?;
        if expected_ancestry != self.target_projection_ancestry {
            return Err(ProvenanceProofError::NonCanonicalAncestry);
        }

        if self.query.include_derivation_context {
            let derivation = derivation.ok_or(ProvenanceProofError::MissingDerivationContext)?;
            derivation
                .verify(output_lineage)
                .map_err(ProvenanceProofError::QualifiedDerivation)?;
            if self.derivation_context_digest.as_deref()
                != Some(derivation.context_digest.as_str())
            {
                return Err(ProvenanceProofError::ProofDigestMismatch);
            }
        } else if derivation.is_some() || self.derivation_context_digest.is_some() {
            return Err(ProvenanceProofError::UnexpectedDerivationContext);
        }

        if self.query.include_external_observation_boundary {
            let (boundary_lineage, boundary_receipt, projection, collection, observation) =
                boundary.ok_or(ProvenanceProofError::MissingBoundaryLineage)?;
            if self.boundary_lineage_digest.as_deref()
                != Some(boundary_lineage.lineage_digest.as_str())
                || self.observation_record_digest.as_deref()
                    != Some(observation.record_digest.as_str())
            {
                return Err(ProvenanceProofError::ProofDigestMismatch);
            }
            boundary_lineage
                .verify(
                    boundary_receipt,
                    observation,
                    projection,
                    collection,
                    ancestry,
                )
                .map_err(ProvenanceProofError::BoundaryLineage)?;
        } else if boundary.is_some()
            || self.boundary_lineage_digest.is_some()
            || self.observation_record_digest.is_some()
        {
            return Err(ProvenanceProofError::UnexpectedBoundaryLineage);
        }

        if self.proof_digest != self.compute_digest() {
            return Err(ProvenanceProofError::ProofDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        put(&mut h, &self.proof_version);
        put(&mut h, &self.query.query_version);
        put(&mut h, &self.query.target_projection_digest);
        put(&mut h, &self.query.target_output_record_id);
        put(&mut h, &self.query.target_output_record_digest);
        match self.query.scope {
            ProvenanceScope::ClosedProjectionSet => put(&mut h, "closed_projection_set"),
            ProvenanceScope::DepthLimited { max_depth } => {
                put(&mut h, "depth_limited");
                put(&mut h, &max_depth.to_string());
            }
        }
        put(&mut h, if self.query.include_derivation_context { "1" } else { "0" });
        put(
            &mut h,
            if self.query.include_external_observation_boundary { "1" } else { "0" },
        );
        put(&mut h, &self.target_output_lineage_digest);
        for digest in &self.target_projection_ancestry {
            put(&mut h, digest);
        }
        put(&mut h, "complete_within_scope");
        put(&mut h, self.derivation_context_digest.as_deref().unwrap_or(""));
        put(&mut h, self.boundary_lineage_digest.as_deref().unwrap_or(""));
        put(&mut h, self.observation_record_digest.as_deref().unwrap_or(""));
        format!("sha256:{:x}", h.finalize())
    }
}

fn canonical_ancestry(
    ancestry: &[FederatedProjectionReceipt],
) -> Result<Vec<String>, ProvenanceProofError> {
    let mut digests = ancestry
        .iter()
        .map(|p| p.projection_digest.clone())
        .collect::<Vec<_>>();
    digests.sort();
    if digests.windows(2).any(|w| w[0] == w[1] || !valid_digest(&w[0])) {
        return Err(ProvenanceProofError::NonCanonicalAncestry);
    }
    if digests.last().is_some_and(|d| !valid_digest(d)) {
        return Err(ProvenanceProofError::NonCanonicalAncestry);
    }
    Ok(digests)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value.as_bytes()[7..].iter().all(|b| b.is_ascii_hexdigit())
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_rejects_bad_target() {
        assert_eq!(
            FederatedProvenanceQuery::new(
                "not-a-digest",
                "output:1",
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ProvenanceScope::ClosedProjectionSet,
                false,
                false,
            ),
            Err(ProvenanceProofError::InvalidQuery)
        );
    }

    #[test]
    fn depth_zero_is_rejected_until_truncation_receipts_are_supported() {
        assert_eq!(
            FederatedProvenanceQuery::new(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "output:1",
                "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                ProvenanceScope::DepthLimited { max_depth: 0 },
                false,
                false,
            ),
            Err(ProvenanceProofError::InvalidQuery)
        );
    }

    #[test]
    fn query_scope_is_content_distinct() {
        let query_a = FederatedProvenanceQuery::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "output:1",
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ProvenanceScope::ClosedProjectionSet,
            false,
            false,
        )
        .unwrap();
        let query_b = FederatedProvenanceQuery {
            scope: ProvenanceScope::DepthLimited { max_depth: 1 },
            ..query_a.clone()
        };
        assert_ne!(
            serde_json::to_vec(&query_a).unwrap(),
            serde_json::to_vec(&query_b).unwrap()
        );
    }

    #[test]
    fn serde_roundtrip_query() {
        let query = FederatedProvenanceQuery::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "output:1",
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ProvenanceScope::ClosedProjectionSet,
            true,
            false,
        )
        .unwrap();
        let encoded = serde_json::to_vec(&query).unwrap();
        assert_eq!(
            serde_json::from_slice::<FederatedProvenanceQuery>(&encoded).unwrap(),
            query
        );
    }
}
