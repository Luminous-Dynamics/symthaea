//! Deterministic collection commitment for cross-DKG federation links.
//!
//! The collection is a reproducibility envelope. It does not establish truth,
//! consensus, or transfer semantic authority between the participating DKGs.

use super::cross_dkg_federation::{CrossDkgFederationLink, CrossDkgLinkError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const COLLECTION_VERSION: &str = "1.0.0";
const DIGEST_ALGORITHM: &str = "sha256";
const DIGEST_DOMAIN: &[u8] = b"symthaea:cross-dkg-link-collection:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossDkgLinkCollection {
    pub collection_version: String,
    pub digest_algorithm: String,
    pub links: Vec<CrossDkgFederationLink>,
    pub source_dkg_ids: Vec<String>,
    pub target_dkg_ids: Vec<String>,
    pub collection_digest: String,
}

impl CrossDkgLinkCollection {
    pub fn new(mut links: Vec<CrossDkgFederationLink>) -> Result<Self, CrossDkgCollectionError> {
        for link in &links {
            link.verify_integrity().map_err(CrossDkgCollectionError::InvalidLink)?;
        }
        links.sort_by_key(|l| link_key(l));
        if links.windows(2).any(|w| w[0].link_digest == w[1].link_digest) {
            return Err(CrossDkgCollectionError::DuplicateLink);
        }

        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for link in &links {
            sources.insert(link.source_dkg_id.clone());
            targets.insert(link.target_dkg_id.clone());
        }
        let mut collection = Self {
            collection_version: COLLECTION_VERSION.into(),
            digest_algorithm: DIGEST_ALGORITHM.into(),
            links,
            source_dkg_ids: sources.into_iter().collect(),
            target_dkg_ids: targets.into_iter().collect(),
            collection_digest: String::new(),
        };
        collection.collection_digest = collection.compute_digest();
        Ok(collection)
    }

    pub fn verify_integrity(&self) -> Result<(), CrossDkgCollectionError> {
        if self.collection_version != COLLECTION_VERSION {
            return Err(CrossDkgCollectionError::VersionMismatch);
        }
        if self.digest_algorithm != DIGEST_ALGORITHM {
            return Err(CrossDkgCollectionError::DigestAlgorithmMismatch);
        }
        if self.links.windows(2).any(|w| link_key(&w[0]) >= link_key(&w[1])) {
            return Err(CrossDkgCollectionError::NonCanonicalOrder);
        }
        if self.links.is_empty() {
            return Err(CrossDkgCollectionError::EmptyCollection);
        }
        for link in &self.links {
            link.verify_integrity().map_err(CrossDkgCollectionError::InvalidLink)?;
        }
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for link in &self.links {
            sources.insert(link.source_dkg_id.clone());
            targets.insert(link.target_dkg_id.clone());
        }
        if self.source_dkg_ids != sources.iter().cloned().collect::<Vec<_>>() {
            return Err(CrossDkgCollectionError::SourceInventoryMismatch);
        }
        if self.target_dkg_ids != targets.iter().cloned().collect::<Vec<_>>() {
            return Err(CrossDkgCollectionError::TargetInventoryMismatch);
        }
        if self.collection_digest != self.compute_digest() {
            return Err(CrossDkgCollectionError::CollectionDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(DIGEST_DOMAIN);
        put(&mut h, &self.collection_version);
        put(&mut h, &self.digest_algorithm);
        for link in &self.links {
            put(&mut h, &link.link_digest);
        }
        for id in &self.source_dkg_ids { put(&mut h, id); }
        for id in &self.target_dkg_ids { put(&mut h, id); }
        format!("sha256:{:x}", h.finalize())
    }
}

fn link_key(link: &CrossDkgFederationLink) -> (
    &str, &str, &str, &str, &str, &str, &str, &str, &str, &str,
) {
    (
        &link.source_dkg_id,
        &link.source_graph_digest,
        &link.source_record_id,
        &link.source_record_digest,
        link.relation.as_str(),
        &link.target_dkg_id,
        &link.target_graph_digest,
        &link.target_record_id,
        &link.target_record_digest,
        &link.link_digest,
    )
}

fn put(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossDkgCollectionError {
    InvalidLink(CrossDkgLinkError),
    VersionMismatch,
    DigestAlgorithmMismatch,
    DuplicateLink,
    EmptyCollection,
    NonCanonicalOrder,
    SourceInventoryMismatch,
    TargetInventoryMismatch,
    CollectionDigestMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cross_dkg_federation::CrossDkgRelation;

    fn link(id: &str) -> CrossDkgFederationLink {
        CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", id, "sha256:r", CrossDkgRelation::References,
            "symthaea", "sha256:g2", "observation:1", "sha256:t",
        ).unwrap()
    }

    #[test]
    fn canonicalizes_input_order() {
        let a = link("claim:a");
        let b = link("claim:b");
        let x = CrossDkgLinkCollection::new(vec![b.clone(), a.clone()]).unwrap();
        let y = CrossDkgLinkCollection::new(vec![a, b]).unwrap();
        assert_eq!(x, y);
        assert!(x.verify_integrity().is_ok());
    }

    #[test]
    fn duplicate_link_is_rejected() {
        let a = link("claim:a");
        assert_eq!(
            CrossDkgLinkCollection::new(vec![a.clone(), a]),
            Err(CrossDkgCollectionError::DuplicateLink)
        );
    }

    #[test]
    fn tampering_is_rejected() {
        let a = link("claim:a");
        let mut c = CrossDkgLinkCollection::new(vec![a]).unwrap();
        c.source_dkg_ids = vec!["tampered".into()];
        assert_eq!(c.verify_integrity(), Err(CrossDkgCollectionError::SourceInventoryMismatch));
    }

    #[test]
    #[test]
    fn canonical_key_distinguishes_graph_and_record_digests() {
        let a = CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:1", "sha256:r1", CrossDkgRelation::References,
            "symthaea", "sha256:g2", "observation:1", "sha256:t1",
        ).unwrap();
        let b = CrossDkgFederationLink::new(
            "mycelix", "sha256:g1", "claim:1", "sha256:r1", CrossDkgRelation::References,
            "symthaea", "sha256:g3", "observation:1", "sha256:t1",
        ).unwrap();
        let c = CrossDkgLinkCollection::new(vec![b.clone(), a.clone()]).unwrap();
        assert_ne!(a.link_digest, b.link_digest);
        assert!(c.verify_integrity().is_ok());
        assert_eq!(c.links[0], a);
        assert_eq!(c.links[1], b);
    }

    #[test]
    fn canonical_key_includes_link_digest() {
        let a = link("claim:a");
        let mut b = a.clone();
        b.link_digest = "sha256:tampered".into();
        assert_eq!(
            CrossDkgLinkCollection::new(vec![a, b]),
            Err(CrossDkgCollectionError::InvalidLink(CrossDkgLinkError::DigestMismatch))
        );
    }

    #[test]
    fn serde_roundtrip() {
        let c = CrossDkgLinkCollection::new(vec![link("claim:a")]).unwrap();
        let bytes = serde_json::to_vec(&c).unwrap();
        let d: CrossDkgLinkCollection = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c, d);
        assert!(d.verify_integrity().is_ok());
    }
}
