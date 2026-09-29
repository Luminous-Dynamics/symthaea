//! Content-addressed lineage for individual federated projection outputs.
//!
//! Projection-level ancestry proves that a projection depends on other
//! projections, but does not identify which exact parent record produced a
//! particular output. This receipt supplies that missing granularity.
//!
//! It is provenance-only. It does not qualify evidence, establish truth,
//! authenticate actors, or turn derived records into experimental observations.

use super::cross_dkg_projection::{FederatedProjectionOutput, FederatedProjectionOutputOrigin, FederatedProjectionReceipt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:federated-projection-output-lineage:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProjectionParentOutput {
    pub projection_digest: String,
    pub output_record_id: String,
    pub output_record_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProjectionOutputLineageReceipt {
    pub receipt_version: String,
    pub projection_digest: String,
    pub output_record_id: String,
    pub output_record_digest: String,
    pub parent_outputs: Vec<FederatedProjectionParentOutput>,
    pub lineage_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionOutputLineageError {
    InvalidProjection,
    OutputNotFound,
    OutputOriginMismatch,
    ParentProjectionNotFound,
    ParentOutputNotFound,
    DuplicateParent,
    NonCanonicalParentOrder,
    InvalidDigest,
    LineageDigestMismatch,
    InvalidAncestry,
}

impl FederatedProjectionOutputLineageReceipt {
    /// Commit one output to the exact parent outputs that produced it.
    ///
    /// A SourceReference output must have no parents. A
    /// DerivedFromProjection output must have at least one parent.
    pub fn new(
        projection: &FederatedProjectionReceipt,
        output: &FederatedProjectionOutput,
        parents: &[FederatedProjectionParentOutput],
        ancestry: &[FederatedProjectionReceipt],
    ) -> Result<Self, ProjectionOutputLineageError> {
        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| ProjectionOutputLineageError::InvalidAncestry)?;

        let committed_output = projection
            .outputs
            .iter()
            .find(|candidate| {
                candidate.record_id == output.record_id
                    && candidate.record_digest == output.record_digest
            })
            .ok_or(ProjectionOutputLineageError::OutputNotFound)?;

        if committed_output.origin != output.origin {
            return Err(ProjectionOutputLineageError::OutputOriginMismatch);
        }

        match output.origin {
            FederatedProjectionOutputOrigin::SourceReference if !parents.is_empty() => {
                return Err(ProjectionOutputLineageError::OutputOriginMismatch)
            }
            FederatedProjectionOutputOrigin::DerivedFromProjection if parents.is_empty() => {
                return Err(ProjectionOutputLineageError::OutputOriginMismatch)
            }
            _ => {}
        }

        if !ancestry.iter().any(|p| p.projection_digest == projection.projection_digest) {
            return Err(ProjectionOutputLineageError::InvalidProjection);
        }

        let mut canonical = parents.to_vec();
        canonical.sort_by(|a, b| {
            a.projection_digest
                .cmp(&b.projection_digest)
                .then_with(|| a.output_record_id.cmp(&b.output_record_id))
                .then_with(|| a.output_record_digest.cmp(&b.output_record_digest))
        });
        if canonical.windows(2).any(|w| w[0] == w[1]) {
            return Err(ProjectionOutputLineageError::DuplicateParent);
        }

        for parent in &canonical {
            if !valid_digest(&parent.projection_digest)
                || !valid_digest(&parent.output_record_digest)
            {
                return Err(ProjectionOutputLineageError::InvalidDigest);
            }
            if !projection.parent_projection_digests.iter().any(|d| d == &parent.projection_digest) {
                return Err(ProjectionOutputLineageError::ParentProjectionNotFound);
            }
            let parent_projection = ancestry
                .iter()
                .find(|p| p.projection_digest == parent.projection_digest)
                .ok_or(ProjectionOutputLineageError::ParentProjectionNotFound)?;
            if !parent_projection.outputs.iter().any(|candidate| {
                candidate.record_id == parent.output_record_id
                    && candidate.record_digest == parent.output_record_digest
            }) {
                return Err(ProjectionOutputLineageError::ParentOutputNotFound);
            }
            if parent.projection_digest == projection.projection_digest
                && parent.output_record_id == output.record_id
                && parent.output_record_digest == output.record_digest
            {
                return Err(ProjectionOutputLineageError::InvalidProjection);
            }
        }

        let mut receipt = Self {
            receipt_version: VERSION.into(),
            projection_digest: projection.projection_digest.clone(),
            output_record_id: output.record_id.clone(),
            output_record_digest: output.record_digest.clone(),
            parent_outputs: canonical,
            lineage_digest: String::new(),
        };
        receipt.lineage_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify(
        &self,
        projection: &FederatedProjectionReceipt,
        ancestry: &[FederatedProjectionReceipt],
    ) -> Result<(), ProjectionOutputLineageError> {
        FederatedProjectionReceipt::verify_ancestry(ancestry)
            .map_err(|_| ProjectionOutputLineageError::InvalidAncestry)?;

        if self.receipt_version != VERSION
            || self.projection_digest != projection.projection_digest
            || !valid_digest(&self.projection_digest)
            || !valid_digest(&self.output_record_digest)
        {
            return Err(ProjectionOutputLineageError::InvalidProjection);
        }

        let output = projection
            .outputs
            .iter()
            .find(|candidate| {
                candidate.record_id == self.output_record_id
                    && candidate.record_digest == self.output_record_digest
            })
            .ok_or(ProjectionOutputLineageError::OutputNotFound)?;

        if !ancestry.iter().any(|p| p.projection_digest == projection.projection_digest) {
            return Err(ProjectionOutputLineageError::InvalidProjection);
        }

        match output.origin {
            FederatedProjectionOutputOrigin::SourceReference if !self.parent_outputs.is_empty() => {
                return Err(ProjectionOutputLineageError::OutputOriginMismatch)
            }
            FederatedProjectionOutputOrigin::DerivedFromProjection if self.parent_outputs.is_empty() => {
                return Err(ProjectionOutputLineageError::OutputOriginMismatch)
            }
            _ => {}
        }

        if self.parent_outputs.windows(2).any(|w| {
            (
                w[0].projection_digest.as_str(),
                w[0].output_record_id.as_str(),
                w[0].output_record_digest.as_str(),
            ) >= (
                w[1].projection_digest.as_str(),
                w[1].output_record_id.as_str(),
                w[1].output_record_digest.as_str(),
            )
        }) {
            return Err(ProjectionOutputLineageError::NonCanonicalParentOrder);
        }

        for parent in &self.parent_outputs {
            if !valid_digest(&parent.projection_digest)
                || !valid_digest(&parent.output_record_digest)
            {
                return Err(ProjectionOutputLineageError::InvalidDigest);
            }
            let parent_projection = ancestry
                .iter()
                .find(|p| p.projection_digest == parent.projection_digest)
                .ok_or(ProjectionOutputLineageError::ParentProjectionNotFound)?;
            if !parent_projection.outputs.iter().any(|candidate| {
                candidate.record_id == parent.output_record_id
                    && candidate.record_digest == parent.output_record_digest
            }) {
                return Err(ProjectionOutputLineageError::ParentOutputNotFound);
            }
        }

        if self.lineage_digest != self.compute_digest() {
            return Err(ProjectionOutputLineageError::LineageDigestMismatch);
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
        ] {
            put(&mut h, value);
        }
        for parent in &self.parent_outputs {
            put(&mut h, &parent.projection_digest);
            put(&mut h, &parent.output_record_id);
            put(&mut h, &parent.output_record_digest);
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
    use crate::cross_dkg_adapter::{
        AdapterCapability, AdapterDisposition, CrossDkgAdapterDeclaration, CrossDkgAdapterReceipt,
    };
    use crate::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgRelation};
    use crate::cross_dkg_link_collection::CrossDkgLinkCollection;

    fn setup() -> (FederatedProjectionReceipt, CrossDkgLinkCollection) {
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
        (projection, collection)
    }

    #[test]
    fn source_reference_has_no_parent_outputs() {
        let (projection, _) = setup();
        let output = projection.outputs[0].clone();
        let receipt = FederatedProjectionOutputLineageReceipt::new(
            &projection, &output, &[], &[projection.clone()],
        ).unwrap();
        assert!(receipt.verify(&projection, &[projection]).is_ok());
    }

    #[test]
    fn rejects_parent_for_source_reference() {
        let (projection, _) = setup();
        let output = projection.outputs[0].clone();
        let parent = FederatedProjectionParentOutput {
            projection_digest: projection.projection_digest.clone(),
            output_record_id: output.record_id.clone(),
            output_record_digest: output.record_digest.clone(),
        };
        assert_eq!(
            FederatedProjectionOutputLineageReceipt::new(
                &projection, &output, &[parent], &[projection],
            ),
            Err(ProjectionOutputLineageError::OutputOriginMismatch)
        );
    }

    #[test]
    fn rejects_parent_substitution() {
        let (projection, _) = setup();
        let output = projection.outputs[0].clone();
        let parent = FederatedProjectionParentOutput {
            projection_digest: projection.projection_digest.clone(),
            output_record_id: "missing".into(),
            output_record_digest: output.record_digest.clone(),
        };
        assert_eq!(
            FederatedProjectionOutputLineageReceipt::new(
                &projection, &output, &[parent], &[projection],
            ),
            Err(ProjectionOutputLineageError::ParentOutputNotFound)
        );
    }

    #[test]
    fn rejects_digest_tampering() {
        let (projection, _) = setup();
        let output = projection.outputs[0].clone();
        let receipt = FederatedProjectionOutputLineageReceipt::new(
            &projection, &output, &[], &[projection.clone()],
        ).unwrap();
        let mut tampered = receipt.clone();
        tampered.lineage_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert_eq!(
            tampered.verify(&projection, &[projection]),
            Err(ProjectionOutputLineageError::LineageDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let (projection, _) = setup();
        let output = projection.outputs[0].clone();
        let receipt = FederatedProjectionOutputLineageReceipt::new(
            &projection, &output, &[], &[projection],
        ).unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        assert_eq!(
            serde_json::from_slice::<FederatedProjectionOutputLineageReceipt>(&bytes).unwrap(),
            receipt
        );
    }
}
