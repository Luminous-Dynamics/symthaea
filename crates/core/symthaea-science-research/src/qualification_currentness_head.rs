// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complete-tail assessment for qualification-currentness publications.
//!
//! Starting from one already-public currentness observation, this module proves a
//! bounded statement: no *different* later currentness observation for the same
//! qualification appears through one exact fresh, federated witnessed head.
//! A later publication-like entry blocks the older observation from being treated
//! as latest even if the caller has not supplied the later observation object.
//!
//! This establishes anti-rollback only through the exact observed/federated head.
//! It does not establish a globally latest transparency head or scientific truth.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorClosure,
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry,
    TransparencyHeadFederationClosure, TransparencyHeadFederationReceipt,
    TransparencyInclusionProof, verify_transparency_inclusion,
};

use crate::{
    PubliclyAnchoredQualificationCurrentnessObservation,
    SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE, Sha256Digest,
};

const CURRENTNESS_HEAD_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-head.identity.v1";
const FINAL_INCLUSION_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-head-final-inclusion.identity.v1";
pub const MAX_CURRENTNESS_TAIL_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessHeadPolicy {
    pub maximum_tail_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationCurrentnessHeadPolicyIssue {
    ZeroTailCapacity,
    TailCapacityTooLarge,
}

impl QualificationCurrentnessHeadPolicy {
    pub fn validate(&self) -> Result<(), QualificationCurrentnessHeadPolicyIssue> {
        if self.maximum_tail_entries == 0 {
            return Err(QualificationCurrentnessHeadPolicyIssue::ZeroTailCapacity);
        }
        if self.maximum_tail_entries > MAX_CURRENTNESS_TAIL_ENTRIES {
            return Err(QualificationCurrentnessHeadPolicyIssue::TailCapacityTooLarge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessHeadClosure {
    LatestPublishedThroughFederatedHead,
    LaterCurrentnessPublicationObserved,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum QualificationCurrentnessHeadFinding {
    InvalidPolicy,
    NamespaceMismatch,
    PublicationViewMismatch,
    PublicationViewMissingFromMonitor,
    HeadViewMissingFromMonitor,
    MonitorReceiptMismatch,
    MonitorIncomplete,
    MonitorEquivocation,
    MonitorTemporalConflict,
    MonitorInvalid,
    HeadFederationStale,
    HeadFederationIncomplete,
    HeadFederationBlocked,
    HeadFederationInvalid,
    HeadNotFederated,
    HeadTreeSizeMismatch,
    HeadRootMismatch,
    HeadTreeHeadMismatch,
    AnchorEntryMismatch,
    AnchorEntryWrongKind,
    AnchorEntryWrongSubject,
    AnchorEntryWrongPayload,
    AnchorEntryWrongContext,
    AnchorEntryBeyondPublicationCheckpoint,
    HeadBeforeAnchorEntry,
    TailTooLarge,
    TailSequenceGap { expected: u64, actual: u64 },
    TailPredecessorMismatch { sequence: u64 },
    LaterCurrentnessPublicationLikeEntry { sequence: u64 },
    FinalEntryNotAtHead { final_sequence: u64, head_tree_size: u64 },
    FinalInclusionEntryMismatch,
    FinalInclusionIndexMismatch,
    FinalInclusionTreeSizeMismatch,
    FinalInclusionRootMismatch,
    InvalidFinalInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessHeadAssessment {
    qualification_sha256: Sha256Digest,
    anchor_observation_sha256: TrustSha256Digest,
    anchor_publication_sha256: TrustSha256Digest,
    namespace_sha256: TrustSha256Digest,
    head_federation_receipt_sha256: TrustSha256Digest,
    observed_head_namespaced_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    observed_head_root_sha256: TrustSha256Digest,
    scanned_from_sequence: u64,
    scanned_through_sequence: u64,
    scanned_tail_entries: usize,
    findings: Vec<QualificationCurrentnessHeadFinding>,
    closure: QualificationCurrentnessHeadClosure,
    assessment_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessHeadAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn anchor_observation_sha256(&self) -> &TrustSha256Digest {
        &self.anchor_observation_sha256
    }
    pub fn anchor_publication_sha256(&self) -> &TrustSha256Digest {
        &self.anchor_publication_sha256
    }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_receipt_sha256
    }
    pub fn observed_head_namespaced_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_namespaced_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn observed_head_root_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_root_sha256
    }
    pub fn findings(&self) -> &[QualificationCurrentnessHeadFinding] { &self.findings }
    pub fn closure(&self) -> QualificationCurrentnessHeadClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn latest_published_currentness_through_observed_head_established(&self) -> bool {
        self.closure == QualificationCurrentnessHeadClosure::LatestPublishedThroughFederatedHead
    }
    pub fn anti_rollback_through_observed_head_established(&self) -> bool {
        self.latest_published_currentness_through_observed_head_established()
    }
    pub const fn globally_latest_currentness_established(&self) -> bool { false }
    pub const fn currentness_at_later_time_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_published_currentness_head(
    publication: &PubliclyAnchoredQualificationCurrentnessObservation,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion: &TransparencyInclusionProof,
    policy: &QualificationCurrentnessHeadPolicy,
) -> QualificationCurrentnessHeadAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut later_currentness = false;

    if policy.validate().is_err() {
        findings.push(QualificationCurrentnessHeadFinding::InvalidPolicy);
        invalid = true;
    }
    if publication.namespace_sha256() != observed_head.namespace_sha256()
        || publication.namespace_sha256() != publication_view.namespace_sha256()
        || head_federation.namespace_sha256() != observed_head.namespace_sha256()
    {
        findings.push(QualificationCurrentnessHeadFinding::NamespaceMismatch);
        invalid = true;
    }
    if publication.namespaced_view_sha256() != publication_view.namespaced_view_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::PublicationViewMismatch);
        invalid = true;
    }
    if head_federation.monitor_receipt_sha256() != monitor.receipt_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::MonitorReceiptMismatch);
        invalid = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(publication_view.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessHeadFinding::PublicationViewMissingFromMonitor);
        incomplete = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessHeadFinding::HeadViewMissingFromMonitor);
        incomplete = true;
    }

    match monitor.closure() {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {}
        NamespacedTransparencyMonitorClosure::Incomplete
            if publication_view.namespaced_view_sha256() == observed_head.namespaced_view_sha256()
                && monitor.findings().is_empty() => {}
        NamespacedTransparencyMonitorClosure::Incomplete => {
            findings.push(QualificationCurrentnessHeadFinding::MonitorIncomplete);
            incomplete = true;
        }
        NamespacedTransparencyMonitorClosure::EquivocationObserved => {
            findings.push(QualificationCurrentnessHeadFinding::MonitorEquivocation);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved => {
            findings.push(QualificationCurrentnessHeadFinding::MonitorTemporalConflict);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::Invalid => {
            findings.push(QualificationCurrentnessHeadFinding::MonitorInvalid);
            invalid = true;
        }
    }

    match head_federation.closure() {
        TransparencyHeadFederationClosure::ConvergedFreshObservedHead => {}
        TransparencyHeadFederationClosure::Stale => {
            findings.push(QualificationCurrentnessHeadFinding::HeadFederationStale);
            stale = true;
        }
        TransparencyHeadFederationClosure::Incomplete => {
            findings.push(QualificationCurrentnessHeadFinding::HeadFederationIncomplete);
            incomplete = true;
        }
        TransparencyHeadFederationClosure::Blocked => {
            findings.push(QualificationCurrentnessHeadFinding::HeadFederationBlocked);
            blocked = true;
        }
        TransparencyHeadFederationClosure::Invalid => {
            findings.push(QualificationCurrentnessHeadFinding::HeadFederationInvalid);
            invalid = true;
        }
    }
    if !head_federation
        .converged_namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessHeadFinding::HeadNotFederated);
        incomplete = true;
    }
    if head_federation.maximal_tree_size() != observed_head.tree_size() {
        findings.push(QualificationCurrentnessHeadFinding::HeadTreeSizeMismatch);
        invalid = true;
    }
    if head_federation.maximal_root_sha256() != observed_head.root_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::HeadRootMismatch);
        invalid = true;
    }
    if head_federation.maximal_tree_head_sha256()
        != observed_head.witnessed_view().tree_head_sha256()
    {
        findings.push(QualificationCurrentnessHeadFinding::HeadTreeHeadMismatch);
        invalid = true;
    }

    let qualification_subject = bridge_digest(publication.qualification_sha256());
    if anchor_entry.entry_sha256() != publication.transparency_entry_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryMismatch);
        invalid = true;
    }
    if anchor_entry.kind().as_str() != SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryWrongKind);
        invalid = true;
    }
    if anchor_entry.subject_sha256() != &qualification_subject {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryWrongSubject);
        invalid = true;
    }
    if anchor_entry.payload_sha256() != Some(publication.observation_sha256()) {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryWrongPayload);
        invalid = true;
    }
    if anchor_entry.context_sha256() != Some(publication.source_sha256()) {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryWrongContext);
        invalid = true;
    }
    if anchor_entry.sequence() > publication.tree_size() {
        findings.push(QualificationCurrentnessHeadFinding::AnchorEntryBeyondPublicationCheckpoint);
        invalid = true;
    }
    if observed_head.tree_size() < anchor_entry.sequence() {
        findings.push(QualificationCurrentnessHeadFinding::HeadBeforeAnchorEntry);
        invalid = true;
    }
    if subsequent_entries.len() > policy.maximum_tail_entries
        || subsequent_entries.len() > MAX_CURRENTNESS_TAIL_ENTRIES
    {
        findings.push(QualificationCurrentnessHeadFinding::TailTooLarge);
        invalid = true;
    }

    let mut expected_sequence = anchor_entry.sequence().saturating_add(1);
    let mut previous_entry_sha256 = anchor_entry.entry_sha256().clone();
    for entry in subsequent_entries {
        if entry.sequence() != expected_sequence {
            findings.push(QualificationCurrentnessHeadFinding::TailSequenceGap {
                expected: expected_sequence,
                actual: entry.sequence(),
            });
            invalid = true;
        }
        if entry.previous_entry_sha256() != Some(&previous_entry_sha256) {
            findings.push(QualificationCurrentnessHeadFinding::TailPredecessorMismatch {
                sequence: entry.sequence(),
            });
            invalid = true;
        }
        if entry.kind().as_str() == SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE
            && entry.subject_sha256() == &qualification_subject
            && entry.payload_sha256() != Some(publication.observation_sha256())
        {
            findings.push(
                QualificationCurrentnessHeadFinding::LaterCurrentnessPublicationLikeEntry {
                    sequence: entry.sequence(),
                },
            );
            later_currentness = true;
        }
        previous_entry_sha256 = entry.entry_sha256().clone();
        expected_sequence = entry.sequence().saturating_add(1);
    }

    let final_entry = subsequent_entries.last().unwrap_or(anchor_entry);
    if final_entry.sequence() != observed_head.tree_size() {
        findings.push(QualificationCurrentnessHeadFinding::FinalEntryNotAtHead {
            final_sequence: final_entry.sequence(),
            head_tree_size: observed_head.tree_size(),
        });
        incomplete = true;
    }
    if final_inclusion.entry_sha256() != final_entry.entry_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::FinalInclusionEntryMismatch);
        invalid = true;
    }
    if final_inclusion.leaf_index().saturating_add(1) != observed_head.tree_size() {
        findings.push(QualificationCurrentnessHeadFinding::FinalInclusionIndexMismatch);
        invalid = true;
    }
    if final_inclusion.tree_size() != observed_head.tree_size() {
        findings.push(QualificationCurrentnessHeadFinding::FinalInclusionTreeSizeMismatch);
        invalid = true;
    }
    if final_inclusion.root_sha256() != observed_head.root_sha256() {
        findings.push(QualificationCurrentnessHeadFinding::FinalInclusionRootMismatch);
        invalid = true;
    }
    if verify_transparency_inclusion(final_inclusion).is_err() {
        findings.push(QualificationCurrentnessHeadFinding::InvalidFinalInclusionProof);
        invalid = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        QualificationCurrentnessHeadClosure::Invalid
    } else if later_currentness {
        QualificationCurrentnessHeadClosure::LaterCurrentnessPublicationObserved
    } else if blocked {
        QualificationCurrentnessHeadClosure::Blocked
    } else if stale {
        QualificationCurrentnessHeadClosure::Stale
    } else if incomplete {
        QualificationCurrentnessHeadClosure::Incomplete
    } else {
        QualificationCurrentnessHeadClosure::LatestPublishedThroughFederatedHead
    };

    let final_inclusion_sha256 = final_inclusion_digest(final_inclusion);
    let assessment_sha256 = currentness_head_digest(
        publication,
        publication_view,
        observed_head,
        monitor,
        head_federation,
        anchor_entry,
        subsequent_entries,
        &final_inclusion_sha256,
        policy,
        &findings,
        closure,
    );
    QualificationCurrentnessHeadAssessment {
        qualification_sha256: publication.qualification_sha256().clone(),
        anchor_observation_sha256: publication.observation_sha256().clone(),
        anchor_publication_sha256: publication.publication_sha256().clone(),
        namespace_sha256: observed_head.namespace_sha256().clone(),
        head_federation_receipt_sha256: head_federation.receipt_sha256().clone(),
        observed_head_namespaced_view_sha256: observed_head.namespaced_view_sha256().clone(),
        observed_head_tree_size: observed_head.tree_size(),
        observed_head_root_sha256: observed_head.root_sha256().clone(),
        scanned_from_sequence: anchor_entry.sequence(),
        scanned_through_sequence: final_entry.sequence(),
        scanned_tail_entries: subsequent_entries.len(),
        findings,
        closure,
        assessment_sha256,
    }
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn final_inclusion_digest(proof: &TransparencyInclusionProof) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(FINAL_INCLUSION_DOMAIN);
    digest.text(&proof.tree_size().to_string());
    digest.text(&proof.leaf_index().to_string());
    digest.text(proof.entry_sha256().as_str());
    digest.text(proof.root_sha256().as_str());
    for node in proof.path() { digest.text(node.as_str()); }
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn currentness_head_digest(
    publication: &PubliclyAnchoredQualificationCurrentnessObservation,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion_sha256: &TrustSha256Digest,
    policy: &QualificationCurrentnessHeadPolicy,
    findings: &[QualificationCurrentnessHeadFinding],
    closure: QualificationCurrentnessHeadClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_HEAD_DOMAIN);
    digest.text(publication.qualification_sha256().as_str());
    digest.text(publication.observation_sha256().as_str());
    digest.text(publication.publication_sha256().as_str());
    digest.text(publication_view.namespaced_view_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(head_federation.receipt_sha256().as_str());
    digest.text(anchor_entry.entry_sha256().as_str());
    for entry in subsequent_entries {
        digest.text("tail-entry");
        digest.text(entry.entry_sha256().as_str());
    }
    digest.text(final_inclusion_sha256.as_str());
    digest.text(&policy.maximum_tail_entries.to_string());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        QualificationCurrentnessHeadClosure::LatestPublishedThroughFederatedHead => {
            "latest-published-through-federated-head"
        }
        QualificationCurrentnessHeadClosure::LaterCurrentnessPublicationObserved => {
            "later-currentness-publication-observed"
        }
        QualificationCurrentnessHeadClosure::Stale => "stale",
        QualificationCurrentnessHeadClosure::Incomplete => "incomplete",
        QualificationCurrentnessHeadClosure::Blocked => "blocked",
        QualificationCurrentnessHeadClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-currentness-not-established");
    digest.text("currentness-at-later-time-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_finding(digest: &mut FramedDigest, finding: &QualificationCurrentnessHeadFinding) {
    use QualificationCurrentnessHeadFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::PublicationViewMismatch => digest.text("publication-view-mismatch"),
        Finding::PublicationViewMissingFromMonitor => digest.text("publication-view-missing-from-monitor"),
        Finding::HeadViewMissingFromMonitor => digest.text("head-view-missing-from-monitor"),
        Finding::MonitorReceiptMismatch => digest.text("monitor-receipt-mismatch"),
        Finding::MonitorIncomplete => digest.text("monitor-incomplete"),
        Finding::MonitorEquivocation => digest.text("monitor-equivocation"),
        Finding::MonitorTemporalConflict => digest.text("monitor-temporal-conflict"),
        Finding::MonitorInvalid => digest.text("monitor-invalid"),
        Finding::HeadFederationStale => digest.text("head-federation-stale"),
        Finding::HeadFederationIncomplete => digest.text("head-federation-incomplete"),
        Finding::HeadFederationBlocked => digest.text("head-federation-blocked"),
        Finding::HeadFederationInvalid => digest.text("head-federation-invalid"),
        Finding::HeadNotFederated => digest.text("head-not-federated"),
        Finding::HeadTreeSizeMismatch => digest.text("head-tree-size-mismatch"),
        Finding::HeadRootMismatch => digest.text("head-root-mismatch"),
        Finding::HeadTreeHeadMismatch => digest.text("head-tree-head-mismatch"),
        Finding::AnchorEntryMismatch => digest.text("anchor-entry-mismatch"),
        Finding::AnchorEntryWrongKind => digest.text("anchor-entry-wrong-kind"),
        Finding::AnchorEntryWrongSubject => digest.text("anchor-entry-wrong-subject"),
        Finding::AnchorEntryWrongPayload => digest.text("anchor-entry-wrong-payload"),
        Finding::AnchorEntryWrongContext => digest.text("anchor-entry-wrong-context"),
        Finding::AnchorEntryBeyondPublicationCheckpoint => digest.text("anchor-entry-beyond-publication-checkpoint"),
        Finding::HeadBeforeAnchorEntry => digest.text("head-before-anchor-entry"),
        Finding::TailTooLarge => digest.text("tail-too-large"),
        Finding::TailSequenceGap { expected, actual } => {
            digest.text("tail-sequence-gap");
            digest.text(&expected.to_string());
            digest.text(&actual.to_string());
        }
        Finding::TailPredecessorMismatch { sequence } => {
            digest.text("tail-predecessor-mismatch");
            digest.text(&sequence.to_string());
        }
        Finding::LaterCurrentnessPublicationLikeEntry { sequence } => {
            digest.text("later-currentness-publication-like-entry");
            digest.text(&sequence.to_string());
        }
        Finding::FinalEntryNotAtHead { final_sequence, head_tree_size } => {
            digest.text("final-entry-not-at-head");
            digest.text(&final_sequence.to_string());
            digest.text(&head_tree_size.to_string());
        }
        Finding::FinalInclusionEntryMismatch => digest.text("final-inclusion-entry-mismatch"),
        Finding::FinalInclusionIndexMismatch => digest.text("final-inclusion-index-mismatch"),
        Finding::FinalInclusionTreeSizeMismatch => digest.text("final-inclusion-tree-size-mismatch"),
        Finding::FinalInclusionRootMismatch => digest.text("final-inclusion-root-mismatch"),
        Finding::InvalidFinalInclusionProof => digest.text("invalid-final-inclusion-proof"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_head_is_still_bounded_to_observed_federated_head() {
        fn _assert_api(value: &QualificationCurrentnessHeadAssessment) {
            if value.latest_published_currentness_through_observed_head_established() {
                assert!(value.anti_rollback_through_observed_head_established());
                assert!(!value.globally_latest_currentness_established());
                assert!(!value.currentness_at_later_time_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
