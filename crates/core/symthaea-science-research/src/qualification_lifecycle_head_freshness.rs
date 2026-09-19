// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Freshness assessment for the published scientific-qualification lifecycle head.
//!
//! This layer proves a bounded statement: the supplied lifecycle event is the
//! latest non-duplicate lifecycle publication for its qualification through one
//! exact witnessed transparency checkpoint, and that checkpoint is fresh relative
//! to one exact authenticated evaluation-time interval. It does not establish
//! that no newer event exists beyond that checkpoint, that the checkpoint is the
//! globally latest log view, or that the underlying scientific evidence is fresh.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorClosure,
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry, TransparencyInclusionProof,
    TrustedTime, verify_transparency_inclusion,
};

use crate::{
    PubliclyAnchoredQualificationLifecycleEvent, PubliclyAnchoredScientificQualification,
    SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE, Sha256Digest,
};

const LIFECYCLE_HEAD_FRESHNESS_DOMAIN: &str =
    "symthaea.scientific-qualification-lifecycle-head-freshness.identity.v1";
const FINAL_INCLUSION_IDENTITY_DOMAIN: &str =
    "symthaea.scientific-qualification-lifecycle-tail-final-inclusion.identity.v1";
pub const MAX_LIFECYCLE_TAIL_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LifecycleHeadFreshnessPolicy {
    pub maximum_checkpoint_age_s: u64,
    pub maximum_tail_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleHeadFreshnessPolicyError {
    ZeroTailCapacity,
    TailCapacityTooLarge,
}

impl LifecycleHeadFreshnessPolicy {
    pub fn validate(&self) -> Result<(), LifecycleHeadFreshnessPolicyError> {
        if self.maximum_tail_entries == 0 {
            return Err(LifecycleHeadFreshnessPolicyError::ZeroTailCapacity);
        }
        if self.maximum_tail_entries > MAX_LIFECYCLE_TAIL_ENTRIES {
            return Err(LifecycleHeadFreshnessPolicyError::TailCapacityTooLarge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LifecycleHeadFreshnessClosure {
    FreshThroughObservedHead,
    LaterLifecyclePublicationObserved,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum LifecycleHeadFreshnessFinding {
    InvalidPolicy,
    QualificationMismatch,
    QualificationPublicationViewMismatch,
    LifecyclePublicationViewMismatch,
    NamespaceMismatch,
    MonitorNamespaceMismatch,
    QualificationViewMissingFromMonitor,
    LifecycleViewMissingFromMonitor,
    HeadViewMissingFromMonitor,
    MonitorIncomplete,
    MonitorEquivocation,
    MonitorTemporalConflict,
    MonitorInvalid,
    EvaluationRootAuthorityMismatch,
    HeadNotDefinitelyBeforeEvaluation,
    HeadTooOld,
    AnchorEntryMismatch,
    AnchorEntryWrongKind,
    AnchorEntryWrongSubject,
    AnchorEntryBeyondPublicationCheckpoint,
    HeadBeforeAnchorEntry,
    TailTooLarge,
    TailSequenceGap { expected: u64, actual: u64 },
    TailPredecessorMismatch { sequence: u64 },
    LaterLifecyclePublicationLikeEntry { sequence: u64 },
    FinalEntryNotAtHead { final_sequence: u64, head_tree_size: u64 },
    FinalInclusionEntryMismatch,
    FinalInclusionIndexMismatch,
    FinalInclusionTreeSizeMismatch,
    FinalInclusionRootMismatch,
    InvalidFinalInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LifecycleHeadFreshnessAssessment {
    qualification_sha256: Sha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    lifecycle_sequence: u64,
    namespace_sha256: TrustSha256Digest,
    observed_head_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    observed_head_root_sha256: TrustSha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    scanned_from_sequence: u64,
    scanned_through_sequence: u64,
    scanned_tail_entries: usize,
    findings: Vec<LifecycleHeadFreshnessFinding>,
    closure: LifecycleHeadFreshnessClosure,
    assessment_sha256: TrustSha256Digest,
}

impl LifecycleHeadFreshnessAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn lifecycle_sequence(&self) -> u64 { self.lifecycle_sequence }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn observed_head_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn findings(&self) -> &[LifecycleHeadFreshnessFinding] { &self.findings }
    pub fn closure(&self) -> LifecycleHeadFreshnessClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn latest_lifecycle_publication_through_observed_head_established(&self) -> bool {
        self.closure == LifecycleHeadFreshnessClosure::FreshThroughObservedHead
    }
    pub fn checkpoint_freshness_relative_to_evaluation_established(&self) -> bool {
        self.closure == LifecycleHeadFreshnessClosure::FreshThroughObservedHead
    }
    pub const fn globally_latest_lifecycle_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn evidence_freshness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_lifecycle_head_freshness(
    qualification: &PubliclyAnchoredScientificQualification,
    lifecycle: &PubliclyAnchoredQualificationLifecycleEvent,
    qualification_view: &NamespacedWitnessedTransparencyCheckpoint,
    lifecycle_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion: &TransparencyInclusionProof,
    policy: &LifecycleHeadFreshnessPolicy,
) -> LifecycleHeadFreshnessAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut later_lifecycle = false;

    if policy.validate().is_err() {
        findings.push(LifecycleHeadFreshnessFinding::InvalidPolicy);
        invalid = true;
    }
    if lifecycle.qualification_sha256() != qualification.qualification_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::QualificationMismatch);
        invalid = true;
    }
    if qualification_view.witnessed_view().view_sha256() != qualification.witnessed_view_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::QualificationPublicationViewMismatch);
        invalid = true;
    }
    if lifecycle_view.witnessed_view().view_sha256() != lifecycle.witnessed_view_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::LifecyclePublicationViewMismatch);
        invalid = true;
    }
    if qualification_view.namespace_sha256() != lifecycle_view.namespace_sha256()
        || qualification_view.namespace_sha256() != observed_head.namespace_sha256()
    {
        findings.push(LifecycleHeadFreshnessFinding::NamespaceMismatch);
        invalid = true;
    }
    if monitor.namespace_sha256() != Some(observed_head.namespace_sha256()) {
        findings.push(LifecycleHeadFreshnessFinding::MonitorNamespaceMismatch);
        invalid = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(qualification_view.namespaced_view_sha256())
    {
        findings.push(LifecycleHeadFreshnessFinding::QualificationViewMissingFromMonitor);
        incomplete = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(lifecycle_view.namespaced_view_sha256())
    {
        findings.push(LifecycleHeadFreshnessFinding::LifecycleViewMissingFromMonitor);
        incomplete = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(LifecycleHeadFreshnessFinding::HeadViewMissingFromMonitor);
        incomplete = true;
    }

    let all_required_views_same = qualification_view.namespaced_view_sha256()
        == lifecycle_view.namespaced_view_sha256()
        && lifecycle_view.namespaced_view_sha256() == observed_head.namespaced_view_sha256();
    match monitor.closure() {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {}
        NamespacedTransparencyMonitorClosure::Incomplete
            if all_required_views_same && monitor.findings().is_empty() => {}
        NamespacedTransparencyMonitorClosure::Incomplete => {
            findings.push(LifecycleHeadFreshnessFinding::MonitorIncomplete);
            incomplete = true;
        }
        NamespacedTransparencyMonitorClosure::EquivocationObserved => {
            findings.push(LifecycleHeadFreshnessFinding::MonitorEquivocation);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved => {
            findings.push(LifecycleHeadFreshnessFinding::MonitorTemporalConflict);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::Invalid => {
            findings.push(LifecycleHeadFreshnessFinding::MonitorInvalid);
            invalid = true;
        }
    }

    if evaluation_time.root_authority_sha256() != observed_head.root_authority_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::EvaluationRootAuthorityMismatch);
        invalid = true;
    }
    let (head_earliest, head_latest) = observed_head.consensus_interval();
    let (evaluation_earliest, evaluation_latest) = evaluation_time.consensus_interval();
    if head_latest > evaluation_earliest {
        findings.push(LifecycleHeadFreshnessFinding::HeadNotDefinitelyBeforeEvaluation);
        blocked = true;
    }
    if evaluation_latest > head_earliest.saturating_add(policy.maximum_checkpoint_age_s) {
        findings.push(LifecycleHeadFreshnessFinding::HeadTooOld);
        stale = true;
    }

    if anchor_entry.entry_sha256() != lifecycle.transparency_entry_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::AnchorEntryMismatch);
        invalid = true;
    }
    if anchor_entry.kind().as_str() != SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE {
        findings.push(LifecycleHeadFreshnessFinding::AnchorEntryWrongKind);
        invalid = true;
    }
    if anchor_entry.subject_sha256() != &bridge_digest(qualification.qualification_sha256()) {
        findings.push(LifecycleHeadFreshnessFinding::AnchorEntryWrongSubject);
        invalid = true;
    }
    if anchor_entry.sequence() > lifecycle.tree_size() {
        findings.push(LifecycleHeadFreshnessFinding::AnchorEntryBeyondPublicationCheckpoint);
        invalid = true;
    }
    if observed_head.tree_size() < anchor_entry.sequence() {
        findings.push(LifecycleHeadFreshnessFinding::HeadBeforeAnchorEntry);
        invalid = true;
    }
    if subsequent_entries.len() > policy.maximum_tail_entries
        || subsequent_entries.len() > MAX_LIFECYCLE_TAIL_ENTRIES
    {
        findings.push(LifecycleHeadFreshnessFinding::TailTooLarge);
        invalid = true;
    }

    let current_event_payload = bridge_digest(lifecycle.lifecycle_event_sha256());
    let qualification_subject = bridge_digest(qualification.qualification_sha256());
    let mut expected_sequence = anchor_entry.sequence().saturating_add(1);
    let mut previous_entry_sha256 = anchor_entry.entry_sha256().clone();
    for entry in subsequent_entries {
        if entry.sequence() != expected_sequence {
            findings.push(LifecycleHeadFreshnessFinding::TailSequenceGap {
                expected: expected_sequence,
                actual: entry.sequence(),
            });
            invalid = true;
        }
        if entry.previous_entry_sha256() != Some(&previous_entry_sha256) {
            findings.push(LifecycleHeadFreshnessFinding::TailPredecessorMismatch {
                sequence: entry.sequence(),
            });
            invalid = true;
        }
        if entry.kind().as_str() == SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE
            && entry.subject_sha256() == &qualification_subject
            && entry.payload_sha256() != Some(&current_event_payload)
        {
            findings.push(LifecycleHeadFreshnessFinding::LaterLifecyclePublicationLikeEntry {
                sequence: entry.sequence(),
            });
            later_lifecycle = true;
        }
        previous_entry_sha256 = entry.entry_sha256().clone();
        expected_sequence = entry.sequence().saturating_add(1);
    }

    let final_entry = subsequent_entries.last().unwrap_or(anchor_entry);
    if final_entry.sequence() != observed_head.tree_size() {
        findings.push(LifecycleHeadFreshnessFinding::FinalEntryNotAtHead {
            final_sequence: final_entry.sequence(),
            head_tree_size: observed_head.tree_size(),
        });
        incomplete = true;
    }
    if final_inclusion.entry_sha256() != final_entry.entry_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::FinalInclusionEntryMismatch);
        invalid = true;
    }
    if final_inclusion.leaf_index().saturating_add(1) != observed_head.tree_size() {
        findings.push(LifecycleHeadFreshnessFinding::FinalInclusionIndexMismatch);
        invalid = true;
    }
    if final_inclusion.tree_size() != observed_head.tree_size() {
        findings.push(LifecycleHeadFreshnessFinding::FinalInclusionTreeSizeMismatch);
        invalid = true;
    }
    if final_inclusion.root_sha256() != observed_head.root_sha256() {
        findings.push(LifecycleHeadFreshnessFinding::FinalInclusionRootMismatch);
        invalid = true;
    }
    if verify_transparency_inclusion(final_inclusion).is_err() {
        findings.push(LifecycleHeadFreshnessFinding::InvalidFinalInclusionProof);
        invalid = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        LifecycleHeadFreshnessClosure::Invalid
    } else if later_lifecycle {
        LifecycleHeadFreshnessClosure::LaterLifecyclePublicationObserved
    } else if stale {
        LifecycleHeadFreshnessClosure::Stale
    } else if blocked {
        LifecycleHeadFreshnessClosure::Blocked
    } else if incomplete {
        LifecycleHeadFreshnessClosure::Incomplete
    } else {
        LifecycleHeadFreshnessClosure::FreshThroughObservedHead
    };

    let final_inclusion_sha256 = final_inclusion_digest(final_inclusion);
    let assessment_sha256 = lifecycle_head_freshness_digest(
        qualification,
        lifecycle,
        qualification_view,
        lifecycle_view,
        observed_head,
        monitor,
        evaluation_time,
        anchor_entry,
        subsequent_entries,
        &final_inclusion_sha256,
        policy,
        &findings,
        closure,
    );
    LifecycleHeadFreshnessAssessment {
        qualification_sha256: qualification.qualification_sha256().clone(),
        lifecycle_event_sha256: lifecycle.lifecycle_event_sha256().clone(),
        lifecycle_sequence: lifecycle.lifecycle_sequence(),
        namespace_sha256: observed_head.namespace_sha256().clone(),
        observed_head_view_sha256: observed_head.namespaced_view_sha256().clone(),
        observed_head_tree_size: observed_head.tree_size(),
        observed_head_root_sha256: observed_head.root_sha256().clone(),
        evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
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
    let mut digest = FramedDigest::new(FINAL_INCLUSION_IDENTITY_DOMAIN);
    digest.text(&proof.tree_size().to_string());
    digest.text(&proof.leaf_index().to_string());
    digest.text(proof.entry_sha256().as_str());
    digest.text(proof.root_sha256().as_str());
    for node in proof.path() { digest.text(node.as_str()); }
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn lifecycle_head_freshness_digest(
    qualification: &PubliclyAnchoredScientificQualification,
    lifecycle: &PubliclyAnchoredQualificationLifecycleEvent,
    qualification_view: &NamespacedWitnessedTransparencyCheckpoint,
    lifecycle_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    evaluation_time: &TrustedTime,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion_sha256: &TrustSha256Digest,
    policy: &LifecycleHeadFreshnessPolicy,
    findings: &[LifecycleHeadFreshnessFinding],
    closure: LifecycleHeadFreshnessClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(LIFECYCLE_HEAD_FRESHNESS_DOMAIN);
    digest.text(qualification.qualification_sha256().as_str());
    digest.text(qualification.publication_sha256().as_str());
    digest.text(lifecycle.publication_sha256().as_str());
    digest.text(lifecycle.lifecycle_event_sha256().as_str());
    digest.text(&lifecycle.lifecycle_sequence().to_string());
    digest.text(qualification_view.namespaced_view_sha256().as_str());
    digest.text(lifecycle_view.namespaced_view_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    digest.text(anchor_entry.entry_sha256().as_str());
    for entry in subsequent_entries {
        digest.text("tail-entry");
        digest.text(entry.entry_sha256().as_str());
    }
    digest.text(final_inclusion_sha256.as_str());
    digest.text(&policy.maximum_checkpoint_age_s.to_string());
    digest.text(&policy.maximum_tail_entries.to_string());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        LifecycleHeadFreshnessClosure::FreshThroughObservedHead => "fresh-through-observed-head",
        LifecycleHeadFreshnessClosure::LaterLifecyclePublicationObserved => "later-lifecycle-publication-observed",
        LifecycleHeadFreshnessClosure::Stale => "stale",
        LifecycleHeadFreshnessClosure::Incomplete => "incomplete",
        LifecycleHeadFreshnessClosure::Blocked => "blocked",
        LifecycleHeadFreshnessClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-lifecycle-not-established");
    digest.text("current-validity-not-established");
    digest.text("evidence-freshness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_finding(digest: &mut FramedDigest, finding: &LifecycleHeadFreshnessFinding) {
    use LifecycleHeadFreshnessFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::QualificationMismatch => digest.text("qualification-mismatch"),
        Finding::QualificationPublicationViewMismatch => digest.text("qualification-publication-view-mismatch"),
        Finding::LifecyclePublicationViewMismatch => digest.text("lifecycle-publication-view-mismatch"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::MonitorNamespaceMismatch => digest.text("monitor-namespace-mismatch"),
        Finding::QualificationViewMissingFromMonitor => digest.text("qualification-view-missing-from-monitor"),
        Finding::LifecycleViewMissingFromMonitor => digest.text("lifecycle-view-missing-from-monitor"),
        Finding::HeadViewMissingFromMonitor => digest.text("head-view-missing-from-monitor"),
        Finding::MonitorIncomplete => digest.text("monitor-incomplete"),
        Finding::MonitorEquivocation => digest.text("monitor-equivocation"),
        Finding::MonitorTemporalConflict => digest.text("monitor-temporal-conflict"),
        Finding::MonitorInvalid => digest.text("monitor-invalid"),
        Finding::EvaluationRootAuthorityMismatch => digest.text("evaluation-root-authority-mismatch"),
        Finding::HeadNotDefinitelyBeforeEvaluation => digest.text("head-not-definitely-before-evaluation"),
        Finding::HeadTooOld => digest.text("head-too-old"),
        Finding::AnchorEntryMismatch => digest.text("anchor-entry-mismatch"),
        Finding::AnchorEntryWrongKind => digest.text("anchor-entry-wrong-kind"),
        Finding::AnchorEntryWrongSubject => digest.text("anchor-entry-wrong-subject"),
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
        Finding::LaterLifecyclePublicationLikeEntry { sequence } => {
            digest.text("later-lifecycle-publication-like-entry");
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
    fn zero_checkpoint_age_is_valid_but_tail_capacity_must_be_nonzero() {
        assert!(LifecycleHeadFreshnessPolicy {
            maximum_checkpoint_age_s: 0,
            maximum_tail_entries: 1,
        }
        .validate()
        .is_ok());
        assert_eq!(
            LifecycleHeadFreshnessPolicy {
                maximum_checkpoint_age_s: 1,
                maximum_tail_entries: 0,
            }
            .validate(),
            Err(LifecycleHeadFreshnessPolicyError::ZeroTailCapacity)
        );
    }
}
