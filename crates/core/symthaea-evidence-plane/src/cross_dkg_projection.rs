//! Deterministic receipts for federated cross-DKG projections.
//!
//! A projection receipt commits to the exact link collection, adapter receipt
//! set, and output record identities used to construct a derived view. It is
//! a reproducibility/provenance envelope only: it does not execute adapters,
//! authenticate code, establish truth, transfer semantic authority, or qualify
//! evidence.

use super::cross_dkg_adapter::{
    CrossDkgAdapterDeclaration, CrossDkgAdapterReceipt, AdapterReceiptError,
};
use super::cross_dkg_federation::CrossDkgFederationLink;
use super::cross_dkg_link_collection::{CrossDkgCollectionError, CrossDkgLinkCollection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.0.0";
const DOMAIN: &[u8] = b"symthaea:cross-dkg-projection-receipt:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedProjectionOutputOrigin {
    SourceReference,
    DerivedFromProjection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProjectionOutput {
    pub record_id: String,
    pub record_digest: String,
    pub origin: FederatedProjectionOutputOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederatedProjectionReceipt {
    pub receipt_version: String,
    pub projection_id: String,
    pub projection_version: String,
    pub collection_digest: String,
    pub parent_projection_digests: Vec<String>,
    pub adapter_receipt_digests: Vec<String>,
    pub outputs: Vec<FederatedProjectionOutput>,
    pub projection_digest: String,
}

impl FederatedProjectionReceipt {
    pub fn new(
        projection_id: impl Into<String>,
        projection_version: impl Into<String>,
        collection: &CrossDkgLinkCollection,
        receipts: &[CrossDkgAdapterReceipt],
        parent_projection_digests: &[String],
    ) -> Result<Self, ProjectionReceiptError> {
        collection.verify_integrity()?;
        let projection_id = projection_id.into();
        let projection_version = projection_version.into();
        if projection_id.trim().is_empty() || projection_version.trim().is_empty() {
            return Err(ProjectionReceiptError::MissingProjectionField);
        }
        if receipts.is_empty() {
            return Err(ProjectionReceiptError::EmptyReceipts);
        }
        let mut parents = parent_projection_digests.to_vec();
        parents.sort();
        if parents.windows(2).any(|w| w[0] == w[1]) {
            return Err(ProjectionReceiptError::DuplicateParentProjection);
        }

        let mut receipt_digests = Vec::with_capacity(receipts.len());
        let mut outputs = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            if receipt.receipt_digest.trim().is_empty()
                || receipt.output_record_id.trim().is_empty()
                || receipt.output_record_digest.trim().is_empty()
            {
                return Err(ProjectionReceiptError::InvalidReceipt);
            }
            if !collection.links.iter().any(|l| l.link_digest == receipt.link_digest) {
                return Err(ProjectionReceiptError::ReceiptNotInCollection);
            }
            receipt_digests.push(receipt.receipt_digest.clone());
            outputs.push(FederatedProjectionOutput {
                record_id: receipt.output_record_id.clone(),
                record_digest: receipt.output_record_digest.clone(),
                origin: if parent_projection_digests.is_empty() {
                    FederatedProjectionOutputOrigin::SourceReference
                } else {
                    FederatedProjectionOutputOrigin::DerivedFromProjection
                },
            });
        }

        receipt_digests.sort();
        if receipt_digests.windows(2).any(|w| w[0] == w[1]) {
            return Err(ProjectionReceiptError::DuplicateReceipt);
        }

        outputs.sort_by(|a, b| {
            a.record_id.cmp(&b.record_id)
                .then_with(|| a.record_digest.cmp(&b.record_digest))
        });
        if outputs.windows(2).any(|w| w[0] == w[1]) {
            return Err(ProjectionReceiptError::DuplicateOutput);
        }

        let mut receipt = Self {
            receipt_version: VERSION.into(),
            projection_id,
            projection_version,
            collection_digest: collection.collection_digest.clone(),
            parent_projection_digests: parents,
            adapter_receipt_digests: receipt_digests,
            outputs,
            projection_digest: String::new(),
        };
        receipt.projection_digest = receipt.compute_digest();
        Ok(receipt)
    }

    pub fn verify_integrity(
        &self,
        collection: &CrossDkgLinkCollection,
    ) -> Result<(), ProjectionReceiptError> {
        collection.verify_integrity()?;
        if self.receipt_version != VERSION {
            return Err(ProjectionReceiptError::VersionMismatch);
        }
        if self.projection_id.trim().is_empty() || self.projection_version.trim().is_empty() {
            return Err(ProjectionReceiptError::MissingProjectionField);
        }
        if self.collection_digest != collection.collection_digest {
            return Err(ProjectionReceiptError::CollectionMismatch);
        }
        if self.parent_projection_digests.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ProjectionReceiptError::NonCanonicalParentOrder);
        }
        if self.parent_projection_digests.iter().any(|d| d.trim().is_empty()) {
            return Err(ProjectionReceiptError::InvalidParentProjection);
        }
        if self.parent_projection_digests.iter().any(|d| d == &self.projection_digest) {
            return Err(ProjectionReceiptError::SelfParentProjection);
        }
        if self.adapter_receipt_digests.is_empty() {
            return Err(ProjectionReceiptError::EmptyReceipts);
        }
        if self.adapter_receipt_digests.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ProjectionReceiptError::NonCanonicalReceiptOrder);
        }
        if self.outputs.is_empty() {
            return Err(ProjectionReceiptError::EmptyOutputs);
        }
        if self.outputs.windows(2).any(|w| {
            (w[0].record_id.as_str(), w[0].record_digest.as_str())
                >= (w[1].record_id.as_str(), w[1].record_digest.as_str())
        }) {
            return Err(ProjectionReceiptError::NonCanonicalOutputOrder);
        }
        if self.projection_digest != self.compute_digest() {
            return Err(ProjectionReceiptError::ProjectionDigestMismatch);
        }
        Ok(())
    }

    /// Verifies a supplied closed set of projection receipts is acyclic and
    /// that every declared parent is present in the same provenance closure.
    pub fn verify_ancestry(receipts: &[FederatedProjectionReceipt]) -> Result<(), ProjectionReceiptError> {
        let mut by_digest = std::collections::BTreeMap::new();
        for receipt in receipts {
            if by_digest.insert(receipt.projection_digest.clone(), receipt).is_some() {
                return Err(ProjectionReceiptError::DuplicateProjectionDigest);
            }
        }
        for receipt in receipts {
            for parent in &receipt.parent_projection_digests {
                if !by_digest.contains_key(parent) {
                    return Err(ProjectionReceiptError::MissingParentProjection);
                }
            }
        }
        fn visit(
            digest: &str,
            by_digest: &std::collections::BTreeMap<String, &FederatedProjectionReceipt>,
            visiting: &mut std::collections::BTreeSet<String>,
            visited: &mut std::collections::BTreeSet<String>,
        ) -> bool {
            if visited.contains(digest) { return true; }
            if !visiting.insert(digest.to_owned()) { return false; }
            let ok = by_digest[digest].parent_projection_digests.iter()
                .all(|p| visit(p, by_digest, visiting, visited));
            visiting.remove(digest);
            if ok { visited.insert(digest.to_owned()); }
            ok
        }
        let mut visiting = std::collections::BTreeSet::new();
        let mut visited = std::collections::BTreeSet::new();
        for digest in by_digest.keys() {
            if !visit(digest, &by_digest, &mut visiting, &mut visited) {
                return Err(ProjectionReceiptError::CyclicParentProjection);
            }
        }
        Ok(())
    }
    }

    /// Verifies each receipt against its exact declaration/link pair, then
    /// verifies that the projection receipt commits to precisely that set.
    pub fn verify_receipts(
        &self,
        collection: &CrossDkgLinkCollection,
        pairs: &[(&CrossDkgAdapterDeclaration, &CrossDkgFederationLink)],
        receipts: &[CrossDkgAdapterReceipt],
    ) -> Result<(), ProjectionReceiptError> {
        self.verify_integrity(collection)?;
        if pairs.len() != receipts.len() || receipts.len() != self.adapter_receipt_digests.len() {
            return Err(ProjectionReceiptError::ReceiptSetMismatch);
        }
        let mut verified = Vec::with_capacity(receipts.len());
        for ((declaration, link), receipt) in pairs.iter().zip(receipts) {
            receipt.verify_against(declaration, link)?;
            verified.push(receipt.receipt_digest.clone());
        }
        verified.sort();
        if verified != self.adapter_receipt_digests {
            return Err(ProjectionReceiptError::ReceiptSetMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        put(&mut h, &self.receipt_version);
        put(&mut h, &self.projection_id);
        put(&mut h, &self.projection_version);
        put(&mut h, &self.collection_digest);
        for digest in &self.parent_projection_digests { put(&mut h, digest); }
        for digest in &self.adapter_receipt_digests { put(&mut h, digest); }
        for output in &self.outputs {
            put(&mut h, &output.record_id);
            put(&mut h, &output.record_digest);
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionReceiptError {
    InvalidCollection(CrossDkgCollectionError),
    InvalidReceipt,
    ReceiptNotInCollection,
    DuplicateReceipt,
    EmptyReceipts,
    DuplicateOutput,
    EmptyOutputs,
    MissingProjectionField,
    VersionMismatch,
    CollectionMismatch,
    NonCanonicalReceiptOrder,
    NonCanonicalOutputOrder,
    ProjectionDigestMismatch,
    ReceiptSetMismatch,
    AdapterReceipt(AdapterReceiptError),
    DuplicateParentProjection,
    InvalidParentProjection,
    SelfParentProjection,
    NonCanonicalParentOrder,
    DuplicateProjectionDigest,
    MissingParentProjection,
    CyclicParentProjection,
}

impl From<CrossDkgCollectionError> for ProjectionReceiptError {
    fn from(value: CrossDkgCollectionError) -> Self { Self::InvalidCollection(value) }
}

impl From<AdapterReceiptError> for ProjectionReceiptError {
    fn from(value: AdapterReceiptError) -> Self { Self::AdapterReceipt(value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cross_dkg_adapter::{
        AdapterCapability, AdapterDisposition, CrossDkgAdapterDeclaration,
    };
    use super::super::cross_dkg_federation::CrossDkgRelation;

    fn declaration() -> CrossDkgAdapterDeclaration {
        CrossDkgAdapterDeclaration {
            adapter_id: "mycelix-to-symthaea".into(),
            adapter_version: "1.0".into(),
            source_dkg_id: "mycelix".into(),
            target_dkg_id: "symthaea".into(),
            allowed_relations: vec![CrossDkgRelation::References],
            capability: AdapterCapability::ReferenceOnly,
            policy_ref: "policy:research-reference".into(),
        }
    }

    fn link() -> CrossDkgFederationLink {
        CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:1", "sha256:r1",
            CrossDkgRelation::References, "symthaea", "sha256:g2",
            "observation:1", "sha256:t1",
        ).unwrap()
    }

    #[test]
    fn projection_is_deterministic() {
        let d = declaration();
        let l = link();
        let r = CrossDkgAdapterReceipt::derive_reference(
            &d, &l, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1"
        ).unwrap();
        let c = CrossDkgLinkCollection::new(vec![l.clone()]).unwrap();
        let a = FederatedProjectionReceipt::new("projection:1", "1", &c, &[r.clone()], &[]).unwrap();
        let b = FederatedProjectionReceipt::new("projection:1", "1", &c, &[r], &[]).unwrap();
        assert_eq!(a, b);
        assert!(a.verify_receipts(&c, &[(&d, &l)], &[CrossDkgAdapterReceipt::derive_reference(
            &d, &l, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1"
        ).unwrap()]).is_ok());
    }

    #[test]
    fn output_tampering_is_rejected() {
        let d = declaration();
        let l = link();
        let r = CrossDkgAdapterReceipt::derive_reference(
            &d, &l, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1"
        ).unwrap();
        let c = CrossDkgLinkCollection::new(vec![l]).unwrap();
        let mut p = FederatedProjectionReceipt::new("projection:1", "1", &c, &[r], &[]).unwrap();
        p.outputs[0].record_id = "tampered".into();
        assert_eq!(p.verify_integrity(&c), Err(ProjectionReceiptError::ProjectionDigestMismatch));
    }

    #[test]
    fn receipt_from_other_collection_is_rejected() {
        let d = declaration();
        let l1 = link();
        let l2 = CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:2", "sha256:r2",
            CrossDkgRelation::References, "symthaea", "sha256:g2",
            "observation:2", "sha256:t2",
        ).unwrap();
        let r = CrossDkgAdapterReceipt::derive_reference(
            &d, &l1, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1"
        ).unwrap();
        let c = CrossDkgLinkCollection::new(vec![l2]).unwrap();
        assert_eq!(
            FederatedProjectionReceipt::new("projection:1", "1", &c, &[r], &[]),
            Err(ProjectionReceiptError::ReceiptNotInCollection)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let d = declaration();
        let l = link();
        let r = CrossDkgAdapterReceipt::derive_reference(
            &d, &l, AdapterDisposition::ReferenceOnly, "local:1", "sha256:o1"
        ).unwrap();
        let c = CrossDkgLinkCollection::new(vec![l]).unwrap();
        let p = FederatedProjectionReceipt::new("projection:1", "1", &c, &[r], &[]).unwrap();
        let bytes = serde_json::to_vec(&p).unwrap();
        let decoded: FederatedProjectionReceipt = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(p, decoded);
        assert!(decoded.verify_integrity(&c).is_ok());
    }
}
