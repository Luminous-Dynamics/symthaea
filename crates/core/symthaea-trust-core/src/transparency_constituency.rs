// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stable principal-set identities for transparency witness quorums.
//!
//! A witness quorum receipt identifies one concrete observation event. Repeating
//! an observation later, or rotating a key for the same principals, can produce a
//! different quorum receipt without creating a different institutional committee.
//! This module therefore derives a second identity from the root-authorized
//! principal constituency only.
//!
//! Constituency diversity is deliberately weaker than independence. Two distinct
//! principal sets can share organizations, infrastructure, funding, operators, or
//! other latent dependencies. The capability only establishes that the exact
//! root-bound principal sets differ.

use serde::Serialize;

use crate::{FramedDigest, Sha256Digest, VerifiedTransparencyWitnessQuorum};

const WITNESS_QUORUM_CONSTITUENCY_DOMAIN: &str =
    "symthaea.transparency-witness-quorum-constituency.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessQuorumConstituency {
    root_authority_sha256: Sha256Digest,
    principal_ids: Vec<String>,
    organization_ids: Vec<String>,
    region_ids: Vec<String>,
    constituency_sha256: Sha256Digest,
}

impl WitnessQuorumConstituency {
    pub fn root_authority_sha256(&self) -> &Sha256Digest {
        &self.root_authority_sha256
    }

    pub fn principal_ids(&self) -> &[String] {
        &self.principal_ids
    }

    pub fn organization_ids(&self) -> &[String] {
        &self.organization_ids
    }

    pub fn region_ids(&self) -> &[String] {
        &self.region_ids
    }

    pub fn constituency_sha256(&self) -> &Sha256Digest {
        &self.constituency_sha256
    }

    /// The exact root-bound principal set is known and content-addressed.
    pub const fn principal_constituency_established(&self) -> bool {
        true
    }

    /// Different principal-set identities are not proof of causal, operational,
    /// organizational, geographic, financial, or global independence.
    pub const fn global_independence_established(&self) -> bool {
        false
    }
}

pub fn derive_witness_quorum_constituency(
    quorum: &VerifiedTransparencyWitnessQuorum,
) -> WitnessQuorumConstituency {
    let mut principal_ids = quorum.principal_ids().to_vec();
    let mut organization_ids = quorum.organization_ids().to_vec();
    let mut region_ids = quorum.region_ids().to_vec();
    principal_ids.sort();
    principal_ids.dedup();
    organization_ids.sort();
    organization_ids.dedup();
    region_ids.sort();
    region_ids.dedup();

    let constituency_sha256 = constituency_digest(
        quorum.root_authority_sha256(),
        &principal_ids,
        &organization_ids,
        &region_ids,
    );
    WitnessQuorumConstituency {
        root_authority_sha256: quorum.root_authority_sha256().clone(),
        principal_ids,
        organization_ids,
        region_ids,
        constituency_sha256,
    }
}

fn constituency_digest(
    root_authority_sha256: &Sha256Digest,
    principal_ids: &[String],
    organization_ids: &[String],
    region_ids: &[String],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(WITNESS_QUORUM_CONSTITUENCY_DOMAIN);
    digest.text(root_authority_sha256.as_str());
    for principal_id in principal_ids {
        digest.text("principal");
        digest.text(principal_id);
    }
    for organization_id in organization_ids {
        digest.text("organization");
        digest.text(organization_id);
    }
    for region_id in region_ids {
        digest.text("region");
        digest.text(region_id);
    }
    digest.text("event-time-independent");
    digest.text("key-rotation-independent-within-stable-principal-set");
    digest.text("global-independence-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    #[test]
    fn canonical_constituency_identity_is_order_independent() {
        let root = sha("root");
        let mut principals = vec!["p-b".to_string(), "p-a".to_string()];
        let mut organizations = vec!["org-b".to_string(), "org-a".to_string()];
        let mut regions = vec!["r-b".to_string(), "r-a".to_string()];
        principals.sort();
        organizations.sort();
        regions.sort();
        let left = constituency_digest(&root, &principals, &organizations, &regions);

        principals.reverse();
        organizations.reverse();
        regions.reverse();
        principals.sort();
        organizations.sort();
        regions.sort();
        let right = constituency_digest(&root, &principals, &organizations, &regions);
        assert_eq!(left, right);
    }

    #[test]
    fn different_principal_sets_have_different_identities() {
        let root = sha("root");
        let left = constituency_digest(
            &root,
            &["p-a".into()],
            &["org-a".into()],
            &["r-a".into()],
        );
        let right = constituency_digest(
            &root,
            &["p-b".into()],
            &["org-a".into()],
            &["r-a".into()],
        );
        assert_ne!(left, right);
    }
}
