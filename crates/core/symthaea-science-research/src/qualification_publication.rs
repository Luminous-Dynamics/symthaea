// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public transparency anchoring for delegated-root-ratified qualification.
//!
//! Publication is deliberately separate from scientific qualification and from
//! current validity. This layer proves that the exact root-ratified historical
//! qualification is included in an authenticated, witness-quorum-backed
//! transparency view under the same exact root authority. It does not establish
//! global log consistency, evidence freshness, or scientific truth.

use serde::Serialize;
use symthaea_trust_core::{
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyEntryDraft,
    TransparencyInclusionProof, TrustUsage, WitnessedTransparencyCheckpoint,
    verify_transparency_inclusion,
};

use crate::{ResearchId, RootRatifiedScientificQualification, Sha256Digest};

pub const SCIENCE_QUALIFICATION_PUBLICATION_USAGE: &str =
    "science.qualification-ratification";
const PUBLIC_QUALIFICATION_DOMAIN: &str =
    "symthaea.publicly-anchored-scientific-qualification.identity.v1";

pub fn qualification_publication_entry(
    ratified: &RootRatifiedScientificQualification,
) -> TransparencyEntryDraft {
    TransparencyEntryDraft {
        kind: qualification_publication_usage(),
        subject_sha256: bridge_digest(ratified.qualification_sha256()),
        payload_sha256: Some(ratified.ratification_sha256().clone()),
        context_sha256: Some(ratified.root_authority_sha256().clone()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationPublicationError {
    WrongEntryKind,
    EntrySubjectMismatch,
    EntryPayloadMismatch,
    EntryContextMismatch,
    RootAuthorityMismatch,
    InclusionEntryMismatch,
    InclusionTreeSizeMismatch,
    InclusionRootMismatch,
    InvalidInclusionProof,
}

/// Non-forgeable evidence that an exact institutional qualification ratification
/// was included in one exact witnessed transparency view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PubliclyAnchoredScientificQualification {
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    qualification_sha256: Sha256Digest,
    ratification_sha256: TrustSha256Digest,
    root_authority_sha256: TrustSha256Digest,
    transparency_entry_sha256: TrustSha256Digest,
    tree_size: u64,
    tree_root_sha256: TrustSha256Digest,
    witnessed_view_sha256: TrustSha256Digest,
    witness_quorum_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
}

impl PubliclyAnchoredScientificQualification {
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn ratification_sha256(&self) -> &TrustSha256Digest { &self.ratification_sha256 }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn transparency_entry_sha256(&self) -> &TrustSha256Digest {
        &self.transparency_entry_sha256
    }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn tree_root_sha256(&self) -> &TrustSha256Digest { &self.tree_root_sha256 }
    pub fn witnessed_view_sha256(&self) -> &TrustSha256Digest { &self.witnessed_view_sha256 }
    pub fn witness_quorum_sha256(&self) -> &TrustSha256Digest { &self.witness_quorum_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }

    pub const fn public_anchoring_established(&self) -> bool { true }
    pub const fn witnessed_publication_established(&self) -> bool { true }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_public_qualification_anchor(
    ratified: &RootRatifiedScientificQualification,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &WitnessedTransparencyCheckpoint,
) -> Result<PubliclyAnchoredScientificQualification, QualificationPublicationError> {
    let expected_usage = qualification_publication_usage();
    if entry.kind() != &expected_usage {
        return Err(QualificationPublicationError::WrongEntryKind);
    }
    if entry.subject_sha256() != &bridge_digest(ratified.qualification_sha256()) {
        return Err(QualificationPublicationError::EntrySubjectMismatch);
    }
    if entry.payload_sha256() != Some(ratified.ratification_sha256()) {
        return Err(QualificationPublicationError::EntryPayloadMismatch);
    }
    if entry.context_sha256() != Some(ratified.root_authority_sha256()) {
        return Err(QualificationPublicationError::EntryContextMismatch);
    }
    if witnessed_view.root_authority_sha256() != ratified.root_authority_sha256() {
        return Err(QualificationPublicationError::RootAuthorityMismatch);
    }
    if inclusion.entry_sha256() != entry.entry_sha256() {
        return Err(QualificationPublicationError::InclusionEntryMismatch);
    }
    if inclusion.tree_size() != witnessed_view.tree_size() {
        return Err(QualificationPublicationError::InclusionTreeSizeMismatch);
    }
    if inclusion.root_sha256() != witnessed_view.root_sha256() {
        return Err(QualificationPublicationError::InclusionRootMismatch);
    }
    verify_transparency_inclusion(inclusion)
        .map_err(|_| QualificationPublicationError::InvalidInclusionProof)?;

    let publication_sha256 = publication_digest(
        ratified,
        entry,
        witnessed_view,
    );
    Ok(PubliclyAnchoredScientificQualification {
        claim_id: ratified.claim_id().clone(),
        subject_sha256: ratified.subject_sha256().clone(),
        qualification_sha256: ratified.qualification_sha256().clone(),
        ratification_sha256: ratified.ratification_sha256().clone(),
        root_authority_sha256: ratified.root_authority_sha256().clone(),
        transparency_entry_sha256: entry.entry_sha256().clone(),
        tree_size: witnessed_view.tree_size(),
        tree_root_sha256: witnessed_view.root_sha256().clone(),
        witnessed_view_sha256: witnessed_view.view_sha256().clone(),
        witness_quorum_sha256: witnessed_view.witness_quorum_sha256().clone(),
        publication_sha256,
    })
}

fn qualification_publication_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_PUBLICATION_USAGE)
        .expect("static qualification publication usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn publication_digest(
    ratified: &RootRatifiedScientificQualification,
    entry: &TransparencyEntry,
    witnessed_view: &WitnessedTransparencyCheckpoint,
) -> TrustSha256Digest {
    let mut digest = symthaea_trust_core::FramedDigest::new(PUBLIC_QUALIFICATION_DOMAIN);
    digest.text(ratified.claim_id().as_str());
    digest.text(ratified.subject_sha256().as_str());
    digest.text(ratified.qualification_sha256().as_str());
    digest.text(ratified.ratification_sha256().as_str());
    digest.text(ratified.root_authority_sha256().as_str());
    digest.text(entry.entry_sha256().as_str());
    digest.text(&witnessed_view.tree_size().to_string());
    digest.text(witnessed_view.root_sha256().as_str());
    digest.text(witnessed_view.view_sha256().as_str());
    digest.text(witnessed_view.witness_quorum_sha256().as_str());
    digest.text("public-anchoring-established");
    digest.text("global-log-consistency-not-established");
    digest.text("current-validity-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_usage_is_canonical() {
        assert_eq!(
            qualification_publication_usage().as_str(),
            SCIENCE_QUALIFICATION_PUBLICATION_USAGE
        );
    }
}
