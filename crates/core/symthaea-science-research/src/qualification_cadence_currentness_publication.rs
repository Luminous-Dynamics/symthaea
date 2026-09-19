// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public transparency anchoring for cadence-conformant institutional currentness.
//!
//! The published object is still scoped to the capability's exact authenticated
//! evaluation interval. Publication makes the result durable and auditable; it
//! does not establish that the publication is the latest state for the
//! qualification, that the log is globally visible, or that the science is true.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyEntryDraft,
    TransparencyInclusionProof, TrustUsage, verify_transparency_inclusion,
};

use crate::{CadenceConformantInstitutionalCurrentnessAtEvaluation, Sha256Digest};

pub const SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE: &str =
    "science.qualification-cadence-currentness-at-evaluation";
const CADENCE_CURRENTNESS_PUBLICATION_DOMAIN: &str =
    "symthaea.publicly-anchored-cadence-currentness-at-evaluation.identity.v1";

pub fn cadence_currentness_publication_entry(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
) -> TransparencyEntryDraft {
    TransparencyEntryDraft {
        kind: cadence_currentness_publication_usage(),
        subject_sha256: bridge_digest(current.qualification_sha256()),
        payload_sha256: Some(current.capability_sha256().clone()),
        context_sha256: Some(current.cadence_policy_authority_sha256().clone()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CadenceCurrentnessPublicationError {
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

/// One exact cadence-conformant currentness evaluation proven included in one
/// exact namespaced, witness-backed transparency view.
///
/// Serializable for retained evidence but intentionally not deserializable into
/// authority. Public anchoring does not establish latest-state currentness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation {
    qualification_sha256: Sha256Digest,
    capability_sha256: TrustSha256Digest,
    original_currentness_sha256: TrustSha256Digest,
    cadence_policy_sha256: TrustSha256Digest,
    cadence_policy_authority_sha256: TrustSha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    namespace_sha256: TrustSha256Digest,
    publication_root_authority_sha256: TrustSha256Digest,
    transparency_entry_sha256: TrustSha256Digest,
    tree_size: u64,
    tree_root_sha256: TrustSha256Digest,
    namespaced_view_sha256: TrustSha256Digest,
    witness_quorum_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
}

impl PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn capability_sha256(&self) -> &TrustSha256Digest { &self.capability_sha256 }
    pub fn original_currentness_sha256(&self) -> &TrustSha256Digest {
        &self.original_currentness_sha256
    }
    pub fn cadence_policy_sha256(&self) -> &TrustSha256Digest { &self.cadence_policy_sha256 }
    pub fn cadence_policy_authority_sha256(&self) -> &TrustSha256Digest {
        &self.cadence_policy_authority_sha256
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn publication_root_authority_sha256(&self) -> &TrustSha256Digest {
        &self.publication_root_authority_sha256
    }
    pub fn transparency_entry_sha256(&self) -> &TrustSha256Digest {
        &self.transparency_entry_sha256
    }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn tree_root_sha256(&self) -> &TrustSha256Digest { &self.tree_root_sha256 }
    pub fn namespaced_view_sha256(&self) -> &TrustSha256Digest { &self.namespaced_view_sha256 }
    pub fn witness_quorum_sha256(&self) -> &TrustSha256Digest { &self.witness_quorum_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }

    pub const fn public_anchoring_established(&self) -> bool { true }
    pub const fn latest_cadence_currentness_established(&self) -> bool { false }
    pub const fn currentness_at_later_time_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_public_cadence_currentness_anchor(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> Result<PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation, CadenceCurrentnessPublicationError> {
    let expected_usage = cadence_currentness_publication_usage();
    if entry.kind() != &expected_usage {
        return Err(CadenceCurrentnessPublicationError::WrongEntryKind);
    }
    if entry.subject_sha256() != &bridge_digest(current.qualification_sha256()) {
        return Err(CadenceCurrentnessPublicationError::EntrySubjectMismatch);
    }
    if entry.payload_sha256() != Some(current.capability_sha256()) {
        return Err(CadenceCurrentnessPublicationError::EntryPayloadMismatch);
    }
    if entry.context_sha256() != Some(current.cadence_policy_authority_sha256()) {
        return Err(CadenceCurrentnessPublicationError::EntryContextMismatch);
    }
    if inclusion.entry_sha256() != entry.entry_sha256() {
        return Err(CadenceCurrentnessPublicationError::InclusionEntryMismatch);
    }
    if inclusion.tree_size() != witnessed_view.tree_size() {
        return Err(CadenceCurrentnessPublicationError::InclusionTreeSizeMismatch);
    }
    if inclusion.root_sha256() != witnessed_view.root_sha256() {
        return Err(CadenceCurrentnessPublicationError::InclusionRootMismatch);
    }
    verify_transparency_inclusion(inclusion)
        .map_err(|_| CadenceCurrentnessPublicationError::InvalidInclusionProof)?;

    let (evaluation_earliest_unix_s, evaluation_latest_unix_s) = current.evaluation_interval();
    let (publication_earliest_unix_s, _) = witnessed_view.consensus_interval();
    if publication_earliest_unix_s < evaluation_latest_unix_s {
        return Err(CadenceCurrentnessPublicationError::PublicationNotDefinitelyAfterEvaluation);
    }

    let publication_sha256 = cadence_currentness_publication_digest(
        current,
        entry,
        inclusion,
        witnessed_view,
    );
    Ok(PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation {
        qualification_sha256: current.qualification_sha256().clone(),
        capability_sha256: current.capability_sha256().clone(),
        original_currentness_sha256: current.original_currentness_sha256().clone(),
        cadence_policy_sha256: current.cadence_policy_sha256().clone(),
        cadence_policy_authority_sha256: current.cadence_policy_authority_sha256().clone(),
        evaluation_time_authority_sha256: current.evaluation_time_authority_sha256().clone(),
        evaluation_earliest_unix_s,
        evaluation_latest_unix_s,
        namespace_sha256: witnessed_view.namespace_sha256().clone(),
        publication_root_authority_sha256: witnessed_view.root_authority_sha256().clone(),
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

fn cadence_currentness_publication_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE)
        .expect("static cadence currentness publication usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn cadence_currentness_publication_digest(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    entry: &TransparencyEntry,
    inclusion: &TransparencyInclusionProof,
    witnessed_view: &NamespacedWitnessedTransparencyCheckpoint,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CADENCE_CURRENTNESS_PUBLICATION_DOMAIN);
    digest.text(current.qualification_sha256().as_str());
    digest.text(current.capability_sha256().as_str());
    digest.text(current.original_currentness_sha256().as_str());
    digest.text(current.cadence_policy_sha256().as_str());
    digest.text(current.cadence_policy_authority_sha256().as_str());
    digest.text(current.evaluation_time_authority_sha256().as_str());
    let (earliest, latest) = current.evaluation_interval();
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
    digest.text("latest-cadence-currentness-not-established");
    digest.text("currentness-at-later-time-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_usage_is_canonical() {
        assert_eq!(
            cadence_currentness_publication_usage().as_str(),
            SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE
        );
    }
}
