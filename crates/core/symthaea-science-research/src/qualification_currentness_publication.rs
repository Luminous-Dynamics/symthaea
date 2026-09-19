// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public transparency anchoring for qualification-currentness observations.
//!
//! Both positive institutional-currentness observations and later fail-closed
//! non-ready observations use the same publication format. Publication preserves
//! historical ordering/evidence but does not by itself establish that the
//! published observation is the latest one for the qualification.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyEntryDraft,
    TransparencyInclusionProof, TrustUsage, verify_transparency_inclusion,
};

use crate::{
    QualificationCurrentnessDisposition, QualificationCurrentnessObservation,
    Sha256Digest,
};

pub const SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE: &str =
    "science.qualification-currentness-observation";
const CURRENTNESS_PUBLICATION_DOMAIN: &str =
    "symthaea.publicly-anchored-qualification-currentness-observation.identity.v1";

pub fn qualification_currentness_publication_entry(
    observation: &QualificationCurrentnessObservation,
) -> TransparencyEntryDraft {
    TransparencyEntryDraft {
        kind: currentness_publication_usage(),
        subject_sha256: bridge_digest(observation.qualification_sha256()),
        payload_sha256: Some(observation.observation_sha256().clone()),
        context_sha256: Some(observation.source_sha256().clone()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessPublicationError {
    WrongEntryKind,
    EntrySubjectMismatch,
    EntryPayloadMismatch,
    EntryContextMismatch,
    InclusionEntryMismatch,
    InclusionTreeSizeMismatch,
    InclusionRootMismatch,
    InvalidInclusionProof,
    PublicationNotDefinitelyAfterEvaluation,
}

/// One exact currentness observation proven included in one exact namespaced,
/// witness-backed transparency view.
///
/// Serializable for retained evidence but intentionally not deserializable into
/// authority. Public anchoring does not establish that this observation is the
/// latest published observation for the qualification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PubliclyAnchoredQualificationCurrentnessObservation {
    qualification_sha256: Sha256Digest,
    observation_sha256: TrustSha256Digest,
    source_sha256: TrustSha256Digest,
    disposition: QualificationCurrentnessDisposition,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    namespace_sha256: TrustSha256Digest,
    root_authority_sha256: TrustSha256Digest,
    transparency_entry_sha256: TrustSha256Digest,
    tree_size: u64,
    tree_root_sha256: TrustSha256Digest,
    namespaced_view_sha256: TrustSha256Digest,
    witness_quorum_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
}

impl PubliclyAnchoredQualificationCurrentnessObservation {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn observation_sha256(&self) -> &TrustSha256Digest { &self.observation_sha256 }
    pub fn source_sha256(&self) -> &TrustSha256Digest { &self.source_sha256 }
    pub fn disposition(&self) -> QualificationCurrentnessDisposition { self.disposition }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn transparency_entry_sha256(&self) -> &TrustSha256Digest {
        &self.transparency_entry_sha256
    }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn tree_root_sha256(&self) -> &TrustSha256Digest { &self.tree_root_sha256 }
    pub fn namespaced_view_sha256(&self) -> &TrustSha256Digest { &self.namespaced_view_sha256 }
    pub fn witness_quorum_sha256(&self) -> &TrustSha256Digest { &self.witness_quorum_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }

    pub const fn public_anchoring_established(&self) -> bool { true }
    pub const fn latest_currentness_observation_established(&self) -> bool { false }
    pub const fn durable_anti_rollback_established(&self) -> bool { false }
    pub const fn global_currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_public_currentness_observation_anchor(
    observation: &QualificationCurrentnessObservation,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> Result<PubliclyAnchoredQualificationCurrentnessObservation, QualificationCurrentnessPublicationError> {
    let expected_usage = currentness_publication_usage();
    if entry.kind() != &expected_usage {
        return Err(QualificationCurrentnessPublicationError::WrongEntryKind);
    }
    if entry.subject_sha256() != &bridge_digest(observation.qualification_sha256()) {
        return Err(QualificationCurrentnessPublicationError::EntrySubjectMismatch);
    }
    if entry.payload_sha256() != Some(observation.observation_sha256()) {
        return Err(QualificationCurrentnessPublicationError::EntryPayloadMismatch);
    }
    if entry.context_sha256() != Some(observation.source_sha256()) {
        return Err(QualificationCurrentnessPublicationError::EntryContextMismatch);
    }
    if inclusion.entry_sha256() != entry.entry_sha256() {
        return Err(QualificationCurrentnessPublicationError::InclusionEntryMismatch);
    }
    if inclusion.tree_size() != witnessed_view.tree_size() {
        return Err(QualificationCurrentnessPublicationError::InclusionTreeSizeMismatch);
    }
    if inclusion.root_sha256() != witnessed_view.root_sha256() {
        return Err(QualificationCurrentnessPublicationError::InclusionRootMismatch);
    }
    verify_transparency_inclusion(inclusion)
        .map_err(|_| QualificationCurrentnessPublicationError::InvalidInclusionProof)?;

    let (evaluation_earliest_unix_s, evaluation_latest_unix_s) =
        observation.evaluation_interval();
    let (publication_earliest_unix_s, _) = witnessed_view.consensus_interval();
    if publication_earliest_unix_s < evaluation_latest_unix_s {
        return Err(QualificationCurrentnessPublicationError::PublicationNotDefinitelyAfterEvaluation);
    }

    let publication_sha256 = currentness_publication_digest(
        observation,
        entry,
        inclusion,
        witnessed_view,
    );
    Ok(PubliclyAnchoredQualificationCurrentnessObservation {
        qualification_sha256: observation.qualification_sha256().clone(),
        observation_sha256: observation.observation_sha256().clone(),
        source_sha256: observation.source_sha256().clone(),
        disposition: observation.disposition(),
        evaluation_time_authority_sha256: observation
            .evaluation_time_authority_sha256()
            .clone(),
        evaluation_earliest_unix_s,
        evaluation_latest_unix_s,
        namespace_sha256: witnessed_view.namespace_sha256().clone(),
        root_authority_sha256: witnessed_view.root_authority_sha256().clone(),
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

fn currentness_publication_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE)
        .expect("static qualification currentness publication usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn currentness_publication_digest(
    observation: &QualificationCurrentnessObservation,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_PUBLICATION_DOMAIN);
    digest.text(observation.qualification_sha256().as_str());
    digest.text(observation.observation_sha256().as_str());
    digest.text(observation.source_sha256().as_str());
    digest.text(disposition_tag(observation.disposition()));
    digest.text(observation.evaluation_time_authority_sha256().as_str());
    let (earliest, latest) = observation.evaluation_interval();
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(entry.entry_sha256().as_str());
    digest.text(&inclusion.leaf_index().to_string());
    digest.text(&inclusion.tree_size().to_string());
    digest.text(inclusion.root_sha256().as_str());
    digest.text(witnessed_view.namespace_sha256().as_str());
    digest.text(witnessed_view.root_authority_sha256().as_str());
    digest.text(witnessed_view.namespaced_view_sha256().as_str());
    digest.text(witnessed_view.witnessed_view().witness_quorum_sha256().as_str());
    digest.text("public-anchoring-established");
    digest.text("latest-currentness-observation-not-established");
    digest.text("durable-anti-rollback-not-established");
    digest.text("global-currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

const fn disposition_tag(disposition: QualificationCurrentnessDisposition) -> &'static str {
    match disposition {
        QualificationCurrentnessDisposition::InstitutionallyCurrentAtEvaluation => {
            "institutionally-current-at-evaluation"
        }
        QualificationCurrentnessDisposition::NotActive => "not-active",
        QualificationCurrentnessDisposition::EvidenceChangedRequiresReview => {
            "evidence-changed-requires-review"
        }
        QualificationCurrentnessDisposition::Stale => "stale",
        QualificationCurrentnessDisposition::Incomplete => "incomplete",
        QualificationCurrentnessDisposition::Blocked => "blocked",
        QualificationCurrentnessDisposition::Invalid => "invalid",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_usage_is_canonical() {
        assert_eq!(
            currentness_publication_usage().as_str(),
            SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE
        );
    }
}
