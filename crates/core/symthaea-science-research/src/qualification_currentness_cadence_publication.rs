// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public transparency anchoring for authorized currentness-cadence policy.
//!
//! Publishing the exact authorized policy makes later policy tightening or
//! loosening auditable. Publication alone does not establish that this is the
//! latest policy for the qualification; complete-tail policy-head assessment is
//! a separate layer.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyEntryDraft,
    TransparencyInclusionProof, TrustUsage, verify_transparency_inclusion,
};

use crate::{
    AuthorizedQualificationCurrentnessCadencePolicy, Sha256Digest,
};

pub const SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE: &str =
    "science.qualification-currentness-cadence-policy";
const CURRENTNESS_CADENCE_PUBLICATION_DOMAIN: &str =
    "symthaea.publicly-anchored-currentness-cadence-policy.identity.v1";

pub fn qualification_currentness_cadence_publication_entry(
    authorized: &AuthorizedQualificationCurrentnessCadencePolicy,
) -> TransparencyEntryDraft {
    TransparencyEntryDraft {
        kind: cadence_publication_usage(),
        subject_sha256: bridge_digest(authorized.qualification_sha256()),
        payload_sha256: Some(authorized.authority_sha256().clone()),
        context_sha256: authorized.policy().predecessor_policy_sha256().cloned(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessCadencePublicationError {
    RootAuthorityMismatch,
    WrongEntryKind,
    EntrySubjectMismatch,
    EntryPayloadMismatch,
    EntryContextMismatch,
    InclusionEntryMismatch,
    InclusionTreeSizeMismatch,
    InclusionRootMismatch,
    InvalidInclusionProof,
    PublicationBeforePolicyAuthorization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PubliclyAnchoredQualificationCurrentnessCadencePolicy {
    qualification_sha256: Sha256Digest,
    version: u64,
    policy_sha256: TrustSha256Digest,
    policy_authority_sha256: TrustSha256Digest,
    predecessor_policy_sha256: Option<TrustSha256Digest>,
    root_authority_sha256: TrustSha256Digest,
    namespace_sha256: TrustSha256Digest,
    transparency_entry_sha256: TrustSha256Digest,
    tree_size: u64,
    tree_root_sha256: TrustSha256Digest,
    namespaced_view_sha256: TrustSha256Digest,
    witness_quorum_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
}

impl PubliclyAnchoredQualificationCurrentnessCadencePolicy {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn version(&self) -> u64 { self.version }
    pub fn policy_sha256(&self) -> &TrustSha256Digest { &self.policy_sha256 }
    pub fn policy_authority_sha256(&self) -> &TrustSha256Digest {
        &self.policy_authority_sha256
    }
    pub fn predecessor_policy_sha256(&self) -> Option<&TrustSha256Digest> {
        self.predecessor_policy_sha256.as_ref()
    }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn transparency_entry_sha256(&self) -> &TrustSha256Digest {
        &self.transparency_entry_sha256
    }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn tree_root_sha256(&self) -> &TrustSha256Digest { &self.tree_root_sha256 }
    pub fn namespaced_view_sha256(&self) -> &TrustSha256Digest { &self.namespaced_view_sha256 }
    pub fn witness_quorum_sha256(&self) -> &TrustSha256Digest { &self.witness_quorum_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }

    pub const fn public_anchoring_established(&self) -> bool { true }
    pub const fn latest_policy_established(&self) -> bool { false }
    pub const fn currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_public_currentness_cadence_policy_anchor(
    authorized: &AuthorizedQualificationCurrentnessCadencePolicy,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> Result<PubliclyAnchoredQualificationCurrentnessCadencePolicy, QualificationCurrentnessCadencePublicationError> {
    if authorized.root_authority_sha256() != witnessed_view.root_authority_sha256() {
        return Err(QualificationCurrentnessCadencePublicationError::RootAuthorityMismatch);
    }
    let expected_usage = cadence_publication_usage();
    if entry.kind() != &expected_usage {
        return Err(QualificationCurrentnessCadencePublicationError::WrongEntryKind);
    }
    if entry.subject_sha256() != &bridge_digest(authorized.qualification_sha256()) {
        return Err(QualificationCurrentnessCadencePublicationError::EntrySubjectMismatch);
    }
    if entry.payload_sha256() != Some(authorized.authority_sha256()) {
        return Err(QualificationCurrentnessCadencePublicationError::EntryPayloadMismatch);
    }
    if entry.context_sha256() != authorized.policy().predecessor_policy_sha256() {
        return Err(QualificationCurrentnessCadencePublicationError::EntryContextMismatch);
    }
    if inclusion.entry_sha256() != entry.entry_sha256() {
        return Err(QualificationCurrentnessCadencePublicationError::InclusionEntryMismatch);
    }
    if inclusion.tree_size() != witnessed_view.tree_size() {
        return Err(QualificationCurrentnessCadencePublicationError::InclusionTreeSizeMismatch);
    }
    if inclusion.root_sha256() != witnessed_view.root_sha256() {
        return Err(QualificationCurrentnessCadencePublicationError::InclusionRootMismatch);
    }
    verify_transparency_inclusion(inclusion)
        .map_err(|_| QualificationCurrentnessCadencePublicationError::InvalidInclusionProof)?;

    let (publication_earliest, _) = witnessed_view.consensus_interval();
    if publication_earliest < authorized.authorized_at_unix_s() {
        return Err(
            QualificationCurrentnessCadencePublicationError::PublicationBeforePolicyAuthorization,
        );
    }

    let publication_sha256 = cadence_publication_digest(
        authorized,
        entry,
        inclusion,
        witnessed_view,
    );
    Ok(PubliclyAnchoredQualificationCurrentnessCadencePolicy {
        qualification_sha256: authorized.qualification_sha256().clone(),
        version: authorized.version(),
        policy_sha256: authorized.policy_sha256().clone(),
        policy_authority_sha256: authorized.authority_sha256().clone(),
        predecessor_policy_sha256: authorized.policy().predecessor_policy_sha256().cloned(),
        root_authority_sha256: authorized.root_authority_sha256().clone(),
        namespace_sha256: witnessed_view.namespace_sha256().clone(),
        transparency_entry_sha256: entry.entry_sha256().clone(),
        tree_size: witnessed_view.tree_size(),
        tree_root_sha256: witnessed_view.root_sha256().clone(),
        namespaced_view_sha256: witnessed_view.namespaced_view_sha256().clone(),
        witness_quorum_sha256: witnessed_view
            .witnessed_view()
            .witness_quorum_sha256()
            .clone(),
        publication_sha256,
    })
}

fn cadence_publication_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE)
        .expect("static currentness cadence publication usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn cadence_publication_digest(
    authorized: &AuthorizedQualificationCurrentnessCadencePolicy,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_CADENCE_PUBLICATION_DOMAIN);
    digest.text(authorized.qualification_sha256().as_str());
    digest.text(&authorized.version().to_string());
    digest.text(authorized.policy_sha256().as_str());
    digest.text(authorized.authority_sha256().as_str());
    digest.optional_sha(authorized.policy().predecessor_policy_sha256());
    digest.text(entry.entry_sha256().as_str());
    digest.text(&inclusion.leaf_index().to_string());
    digest.text(&inclusion.tree_size().to_string());
    digest.text(inclusion.root_sha256().as_str());
    digest.text(witnessed_view.namespace_sha256().as_str());
    digest.text(witnessed_view.root_authority_sha256().as_str());
    digest.text(witnessed_view.namespaced_view_sha256().as_str());
    digest.text(witnessed_view.witnessed_view().witness_quorum_sha256().as_str());
    digest.text("public-anchoring-established");
    digest.text("latest-policy-not-established");
    digest.text("currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_publication_usage_is_canonical() {
        assert_eq!(
            cadence_publication_usage().as_str(),
            SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE,
        );
    }
}
