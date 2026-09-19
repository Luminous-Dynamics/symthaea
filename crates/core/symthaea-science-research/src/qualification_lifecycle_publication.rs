// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public transparency anchoring for authenticated qualification-lifecycle events.
//!
//! A lifecycle event is public only after the exact authenticated event is
//! included in a witness-quorum-backed transparency view under the same root as
//! the delegated-root-ratified qualification it governs. Publication does not by
//! itself establish that this event is the globally latest lifecycle state.

use serde::Serialize;
use symthaea_trust_core::{
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyEntryDraft,
    TransparencyInclusionProof, TrustUsage, WitnessedTransparencyCheckpoint,
    verify_transparency_inclusion,
};

use crate::{
    AuthenticatedQualificationLifecycleEvent, QualificationLifecycleEventKind,
    RootRatifiedScientificQualification, Sha256Digest,
};

pub const SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE: &str =
    "science.qualification-lifecycle-event";
const LIFECYCLE_PUBLICATION_DOMAIN: &str =
    "symthaea.publicly-anchored-qualification-lifecycle-event.identity.v1";

pub fn qualification_lifecycle_publication_entry(
    event: &AuthenticatedQualificationLifecycleEvent,
) -> TransparencyEntryDraft {
    TransparencyEntryDraft {
        kind: lifecycle_publication_usage(),
        subject_sha256: bridge_digest(event.statement().qualification_sha256()),
        payload_sha256: Some(bridge_digest(event.event_sha256())),
        context_sha256: event.statement().previous_event_sha256().map(bridge_digest),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationLifecyclePublicationError {
    QualificationMismatch,
    RootAuthorityMismatch,
    WrongEntryKind,
    EntrySubjectMismatch,
    EntryPayloadMismatch,
    EntryContextMismatch,
    InclusionEntryMismatch,
    InclusionTreeSizeMismatch,
    InclusionRootMismatch,
    InvalidInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PubliclyAnchoredQualificationLifecycleEvent {
    qualification_sha256: Sha256Digest,
    lifecycle_sequence: u64,
    lifecycle_event_sha256: Sha256Digest,
    previous_event_sha256: Option<Sha256Digest>,
    event_kind: QualificationLifecycleEventKind,
    root_authority_sha256: TrustSha256Digest,
    transparency_entry_sha256: TrustSha256Digest,
    tree_size: u64,
    tree_root_sha256: TrustSha256Digest,
    witnessed_view_sha256: TrustSha256Digest,
    witness_quorum_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
}

impl PubliclyAnchoredQualificationLifecycleEvent {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn lifecycle_sequence(&self) -> u64 { self.lifecycle_sequence }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn previous_event_sha256(&self) -> Option<&Sha256Digest> {
        self.previous_event_sha256.as_ref()
    }
    pub fn event_kind(&self) -> &QualificationLifecycleEventKind { &self.event_kind }
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
    pub const fn globally_latest_lifecycle_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_public_lifecycle_anchor(
    ratified: &RootRatifiedScientificQualification,
    event: &AuthenticatedQualificationLifecycleEvent,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &WitnessedTransparencyCheckpoint,
) -> Result<PubliclyAnchoredQualificationLifecycleEvent, QualificationLifecyclePublicationError> {
    if event.statement().qualification_sha256() != ratified.qualification_sha256() {
        return Err(QualificationLifecyclePublicationError::QualificationMismatch);
    }
    if event.root_authority_sha256() != ratified.root_authority_sha256()
        || witnessed_view.root_authority_sha256() != ratified.root_authority_sha256()
    {
        return Err(QualificationLifecyclePublicationError::RootAuthorityMismatch);
    }

    let expected_usage = lifecycle_publication_usage();
    if entry.kind() != &expected_usage {
        return Err(QualificationLifecyclePublicationError::WrongEntryKind);
    }
    if entry.subject_sha256() != &bridge_digest(ratified.qualification_sha256()) {
        return Err(QualificationLifecyclePublicationError::EntrySubjectMismatch);
    }
    if entry.payload_sha256() != Some(&bridge_digest(event.event_sha256())) {
        return Err(QualificationLifecyclePublicationError::EntryPayloadMismatch);
    }
    let expected_context = event.statement().previous_event_sha256().map(bridge_digest);
    if entry.context_sha256() != expected_context.as_ref() {
        return Err(QualificationLifecyclePublicationError::EntryContextMismatch);
    }
    if inclusion.entry_sha256() != entry.entry_sha256() {
        return Err(QualificationLifecyclePublicationError::InclusionEntryMismatch);
    }
    if inclusion.tree_size() != witnessed_view.tree_size() {
        return Err(QualificationLifecyclePublicationError::InclusionTreeSizeMismatch);
    }
    if inclusion.root_sha256() != witnessed_view.root_sha256() {
        return Err(QualificationLifecyclePublicationError::InclusionRootMismatch);
    }
    verify_transparency_inclusion(inclusion)
        .map_err(|_| QualificationLifecyclePublicationError::InvalidInclusionProof)?;

    let publication_sha256 = lifecycle_publication_digest(
        ratified,
        event,
        entry,
        witnessed_view,
    );
    Ok(PubliclyAnchoredQualificationLifecycleEvent {
        qualification_sha256: ratified.qualification_sha256().clone(),
        lifecycle_sequence: event.statement().sequence(),
        lifecycle_event_sha256: event.event_sha256().clone(),
        previous_event_sha256: event.statement().previous_event_sha256().cloned(),
        event_kind: event.statement().event_kind().clone(),
        root_authority_sha256: ratified.root_authority_sha256().clone(),
        transparency_entry_sha256: entry.entry_sha256().clone(),
        tree_size: witnessed_view.tree_size(),
        tree_root_sha256: witnessed_view.root_sha256().clone(),
        witnessed_view_sha256: witnessed_view.view_sha256().clone(),
        witness_quorum_sha256: witnessed_view.witness_quorum_sha256().clone(),
        publication_sha256,
    })
}

fn lifecycle_publication_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE)
        .expect("static qualification lifecycle publication usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn lifecycle_publication_digest(
    ratified: &RootRatifiedScientificQualification,
    event: &AuthenticatedQualificationLifecycleEvent,
    entry: &TransparencyEntry,
    witnessed_view: &WitnessedTransparencyCheckpoint,
) -> TrustSha256Digest {
    let mut digest = symthaea_trust_core::FramedDigest::new(LIFECYCLE_PUBLICATION_DOMAIN);
    digest.text(ratified.qualification_sha256().as_str());
    digest.text(event.event_sha256().as_str());
    digest.text(&event.statement().sequence().to_string());
    digest.optional_sha(event.statement().previous_event_sha256().map(bridge_digest).as_ref());
    digest.text(entry.entry_sha256().as_str());
    digest.text(ratified.root_authority_sha256().as_str());
    digest.text(&witnessed_view.tree_size().to_string());
    digest.text(witnessed_view.root_sha256().as_str());
    digest.text(witnessed_view.view_sha256().as_str());
    digest.text(witnessed_view.witness_quorum_sha256().as_str());
    digest.text("lifecycle-public-anchoring-established");
    digest.text("globally-latest-lifecycle-not-established");
    digest.text("current-validity-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_publication_usage_is_canonical() {
        assert_eq!(
            lifecycle_publication_usage().as_str(),
            SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE
        );
    }
}
