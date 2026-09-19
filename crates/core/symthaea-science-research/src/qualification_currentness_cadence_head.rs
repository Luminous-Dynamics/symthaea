// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complete published-policy lineage and tail assessment for currentness cadence.
//!
//! A lease must not rely on an older authorized cadence after a newer cadence has
//! been published. This layer first requires a complete published predecessor
//! chain from cadence version 1 to the anchor policy, then scans the exact
//! transparency tail from that anchor through one fresh federated witnessed head.
//!
//! A positive result means the anchor is the latest cadence-policy publication
//! for the qualification through that exact observed head. It does not establish
//! that the head is globally latest or that the policy is scientifically correct.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorClosure,
    NamespacedTransparencyMonitorReceipt, NamespacedWitnessedTransparencyCheckpoint,
    Sha256Digest as TrustSha256Digest, TransparencyEntry,
    TransparencyHeadFederationClosure, TransparencyHeadFederationReceipt,
    TransparencyInclusionProof, verify_transparency_inclusion,
};

use crate::{
    PubliclyAnchoredQualificationCurrentnessCadencePolicy,
    SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE, Sha256Digest,
};

const CADENCE_POLICY_HEAD_DOMAIN: &str =
    "symthaea.scientific-currentness-cadence-policy-head.identity.v1";
const CADENCE_POLICY_FINAL_INCLUSION_DOMAIN: &str =
    "symthaea.scientific-currentness-cadence-policy-head-final-inclusion.identity.v1";
pub const MAX_CADENCE_POLICY_CHAIN: usize = 4096;
pub const MAX_CADENCE_POLICY_TAIL_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessCadenceHeadPolicy {
    pub maximum_tail_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationCurrentnessCadenceHeadPolicyIssue {
    ZeroTailCapacity,
    TailCapacityTooLarge,
}

impl QualificationCurrentnessCadenceHeadPolicy {
    pub fn validate(&self) -> Result<(), QualificationCurrentnessCadenceHeadPolicyIssue> {
        if self.maximum_tail_entries == 0 {
            return Err(QualificationCurrentnessCadenceHeadPolicyIssue::ZeroTailCapacity);
        }
        if self.maximum_tail_entries > MAX_CADENCE_POLICY_TAIL_ENTRIES {
            return Err(QualificationCurrentnessCadenceHeadPolicyIssue::TailCapacityTooLarge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessCadenceHeadClosure {
    LatestPublishedPolicyThroughFederatedHead,
    LaterPolicyPublicationObserved,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessCadenceHeadFinding {
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
    PolicyChainTooLarge,
    PolicyChainLengthMismatch { expected: u64, actual: u64 },
    PolicyChainQualificationMismatch { version: u64 },
    PolicyChainNamespaceMismatch { version: u64 },
    PolicyChainVersionMismatch { expected: u64, actual: u64 },
    PolicyChainPredecessorMismatch { version: u64 },
    PolicyChainViewMissingFromMonitor { version: u64 },
    PolicyChainPublishedAfterAnchor { version: u64 },
    AnchorPredecessorMismatch,
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
    MalformedLaterPolicyPublication { sequence: u64 },
    LaterPolicyPublication { sequence: u64 },
    FinalEntryNotAtHead { final_sequence: u64, head_tree_size: u64 },
    FinalInclusionEntryMismatch,
    FinalInclusionIndexMismatch,
    FinalInclusionTreeSizeMismatch,
    FinalInclusionRootMismatch,
    InvalidFinalInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessCadenceHeadAssessment {
    qualification_sha256: Sha256Digest,
    policy_version: u64,
    policy_sha256: TrustSha256Digest,
    policy_authority_sha256: TrustSha256Digest,
    policy_publication_sha256: TrustSha256Digest,
    namespace_sha256: TrustSha256Digest,
    head_federation_receipt_sha256: TrustSha256Digest,
    observed_head_namespaced_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    predecessor_publication_sha256s: Vec<TrustSha256Digest>,
    findings: Vec<QualificationCurrentnessCadenceHeadFinding>,
    closure: QualificationCurrentnessCadenceHeadClosure,
    assessment_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessCadenceHeadAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn policy_version(&self) -> u64 { self.policy_version }
    pub fn policy_sha256(&self) -> &TrustSha256Digest { &self.policy_sha256 }
    pub fn policy_authority_sha256(&self) -> &TrustSha256Digest {
        &self.policy_authority_sha256
    }
    pub fn policy_publication_sha256(&self) -> &TrustSha256Digest {
        &self.policy_publication_sha256
    }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_receipt_sha256
    }
    pub fn observed_head_namespaced_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_namespaced_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn predecessor_publication_sha256s(&self) -> &[TrustSha256Digest] {
        &self.predecessor_publication_sha256s
    }
    pub fn findings(&self) -> &[QualificationCurrentnessCadenceHeadFinding] { &self.findings }
    pub fn closure(&self) -> QualificationCurrentnessCadenceHeadClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn latest_published_policy_through_observed_head_established(&self) -> bool {
        self.closure
            == QualificationCurrentnessCadenceHeadClosure::LatestPublishedPolicyThroughFederatedHead
    }
    pub const fn globally_latest_policy_established(&self) -> bool { false }
    pub const fn currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_currentness_cadence_policy_head(
    publication: &PubliclyAnchoredQualificationCurrentnessCadencePolicy,
    predecessor_publications: &[PubliclyAnchoredQualificationCurrentnessCadencePolicy],
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion: &TransparencyInclusionProof,
    policy: &QualificationCurrentnessCadenceHeadPolicy,
) -> QualificationCurrentnessCadenceHeadAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut later_policy = false;

    if policy.validate().is_err() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::InvalidPolicy);
        invalid = true;
    }
    if publication.namespace_sha256() != publication_view.namespace_sha256()
        || publication.namespace_sha256() != observed_head.namespace_sha256()
        || publication.namespace_sha256() != head_federation.namespace_sha256()
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::NamespaceMismatch);
        invalid = true;
    }
    if publication.namespaced_view_sha256() != publication_view.namespaced_view_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::PublicationViewMismatch);
        invalid = true;
    }
    if head_federation.monitor_receipt_sha256() != monitor.receipt_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::MonitorReceiptMismatch);
        invalid = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(publication_view.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::PublicationViewMissingFromMonitor);
        incomplete = true;
    }
    if !monitor
        .namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadViewMissingFromMonitor);
        incomplete = true;
    }

    match monitor.closure() {
        NamespacedTransparencyMonitorClosure::AppendOnlyConsistentAcrossAuthorizedRoots => {}
        NamespacedTransparencyMonitorClosure::Incomplete
            if publication_view.namespaced_view_sha256() == observed_head.namespaced_view_sha256()
                && monitor.findings().is_empty() => {}
        NamespacedTransparencyMonitorClosure::Incomplete => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::MonitorIncomplete);
            incomplete = true;
        }
        NamespacedTransparencyMonitorClosure::EquivocationObserved => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::MonitorEquivocation);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::TemporalConflictObserved => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::MonitorTemporalConflict);
            blocked = true;
        }
        NamespacedTransparencyMonitorClosure::Invalid => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::MonitorInvalid);
            invalid = true;
        }
    }

    match head_federation.closure() {
        TransparencyHeadFederationClosure::ConvergedFreshObservedHead => {}
        TransparencyHeadFederationClosure::Stale => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::HeadFederationStale);
            stale = true;
        }
        TransparencyHeadFederationClosure::Incomplete => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::HeadFederationIncomplete);
            incomplete = true;
        }
        TransparencyHeadFederationClosure::Blocked => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::HeadFederationBlocked);
            blocked = true;
        }
        TransparencyHeadFederationClosure::Invalid => {
            findings.push(QualificationCurrentnessCadenceHeadFinding::HeadFederationInvalid);
            invalid = true;
        }
    }
    if !head_federation
        .converged_namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadNotFederated);
        incomplete = true;
    }
    if observed_head.tree_size() != head_federation.maximal_tree_size() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadTreeSizeMismatch);
        invalid = true;
    }
    if observed_head.root_sha256() != head_federation.maximal_root_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadRootMismatch);
        invalid = true;
    }
    if observed_head.witnessed_view().tree_head_sha256()
        != head_federation.maximal_tree_head_sha256()
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadTreeHeadMismatch);
        invalid = true;
    }

    let expected_predecessors = publication.version().saturating_sub(1);
    if expected_predecessors > MAX_CADENCE_POLICY_CHAIN as u64 {
        findings.push(QualificationCurrentnessCadenceHeadFinding::PolicyChainTooLarge);
        invalid = true;
    }
    let actual_predecessors = u64::try_from(predecessor_publications.len()).unwrap_or(u64::MAX);
    if actual_predecessors != expected_predecessors {
        findings.push(QualificationCurrentnessCadenceHeadFinding::PolicyChainLengthMismatch {
            expected: expected_predecessors,
            actual: actual_predecessors,
        });
        incomplete = true;
    }

    let mut previous_policy_sha256: Option<&TrustSha256Digest> = None;
    let mut predecessor_publication_sha256s = Vec::new();
    for (index, predecessor) in predecessor_publications.iter().enumerate() {
        let expected_version = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
        predecessor_publication_sha256s.push(predecessor.publication_sha256().clone());
        if predecessor.qualification_sha256() != publication.qualification_sha256() {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainQualificationMismatch {
                    version: predecessor.version(),
                },
            );
            invalid = true;
        }
        if predecessor.namespace_sha256() != publication.namespace_sha256() {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainNamespaceMismatch {
                    version: predecessor.version(),
                },
            );
            invalid = true;
        }
        if predecessor.version() != expected_version {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainVersionMismatch {
                    expected: expected_version,
                    actual: predecessor.version(),
                },
            );
            invalid = true;
        }
        if predecessor.predecessor_policy_sha256() != previous_policy_sha256 {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainPredecessorMismatch {
                    version: predecessor.version(),
                },
            );
            invalid = true;
        }
        if !monitor
            .namespaced_view_sha256s()
            .contains(predecessor.namespaced_view_sha256())
        {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainViewMissingFromMonitor {
                    version: predecessor.version(),
                },
            );
            incomplete = true;
        }
        if predecessor.tree_size() > publication.tree_size() {
            findings.push(
                QualificationCurrentnessCadenceHeadFinding::PolicyChainPublishedAfterAnchor {
                    version: predecessor.version(),
                },
            );
            invalid = true;
        }
        previous_policy_sha256 = Some(predecessor.policy_sha256());
    }
    if publication.predecessor_policy_sha256() != previous_policy_sha256 {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorPredecessorMismatch);
        invalid = true;
    }

    if anchor_entry.entry_sha256() != publication.transparency_entry_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorEntryMismatch);
        invalid = true;
    }
    if anchor_entry.kind().as_str() != SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorEntryWrongKind);
        invalid = true;
    }
    let qualification_subject = bridge_digest(publication.qualification_sha256());
    if anchor_entry.subject_sha256() != &qualification_subject {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorEntryWrongSubject);
        invalid = true;
    }
    if anchor_entry.payload_sha256() != Some(publication.policy_authority_sha256()) {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorEntryWrongPayload);
        invalid = true;
    }
    if anchor_entry.context_sha256() != publication.predecessor_policy_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::AnchorEntryWrongContext);
        invalid = true;
    }
    if anchor_entry.sequence() > publication.tree_size() {
        findings.push(
            QualificationCurrentnessCadenceHeadFinding::AnchorEntryBeyondPublicationCheckpoint,
        );
        invalid = true;
    }
    if observed_head.tree_size() < anchor_entry.sequence() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::HeadBeforeAnchorEntry);
        invalid = true;
    }
    if subsequent_entries.len() > policy.maximum_tail_entries
        || subsequent_entries.len() > MAX_CADENCE_POLICY_TAIL_ENTRIES
    {
        findings.push(QualificationCurrentnessCadenceHeadFinding::TailTooLarge);
        invalid = true;
    }

    let mut expected_sequence = anchor_entry.sequence().saturating_add(1);
    let mut previous_entry_sha256 = anchor_entry.entry_sha256().clone();
    for entry in subsequent_entries {
        if entry.sequence() != expected_sequence {
            findings.push(QualificationCurrentnessCadenceHeadFinding::TailSequenceGap {
                expected: expected_sequence,
                actual: entry.sequence(),
            });
            invalid = true;
        }
        if entry.previous_entry_sha256() != Some(&previous_entry_sha256) {
            findings.push(QualificationCurrentnessCadenceHeadFinding::TailPredecessorMismatch {
                sequence: entry.sequence(),
            });
            invalid = true;
        }
        if entry.kind().as_str() == SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE
            && entry.subject_sha256() == &qualification_subject
        {
            match entry.payload_sha256() {
                None => {
                    findings.push(
                        QualificationCurrentnessCadenceHeadFinding::MalformedLaterPolicyPublication {
                            sequence: entry.sequence(),
                        },
                    );
                    invalid = true;
                }
                Some(payload) if payload != publication.policy_authority_sha256() => {
                    findings.push(
                        QualificationCurrentnessCadenceHeadFinding::LaterPolicyPublication {
                            sequence: entry.sequence(),
                        },
                    );
                    later_policy = true;
                }
                Some(_) if entry.context_sha256() != publication.predecessor_policy_sha256() => {
                    findings.push(
                        QualificationCurrentnessCadenceHeadFinding::MalformedLaterPolicyPublication {
                            sequence: entry.sequence(),
                        },
                    );
                    invalid = true;
                }
                Some(_) => {}
            }
        }
        previous_entry_sha256 = entry.entry_sha256().clone();
        expected_sequence = entry.sequence().saturating_add(1);
    }

    let final_entry = subsequent_entries.last().unwrap_or(anchor_entry);
    if final_entry.sequence() != observed_head.tree_size() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::FinalEntryNotAtHead {
            final_sequence: final_entry.sequence(),
            head_tree_size: observed_head.tree_size(),
        });
        incomplete = true;
    }
    if final_inclusion.entry_sha256() != final_entry.entry_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::FinalInclusionEntryMismatch);
        invalid = true;
    }
    if final_inclusion.leaf_index().saturating_add(1) != observed_head.tree_size() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::FinalInclusionIndexMismatch);
        invalid = true;
    }
    if final_inclusion.tree_size() != observed_head.tree_size() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::FinalInclusionTreeSizeMismatch);
        invalid = true;
    }
    if final_inclusion.root_sha256() != observed_head.root_sha256() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::FinalInclusionRootMismatch);
        invalid = true;
    }
    if verify_transparency_inclusion(final_inclusion).is_err() {
        findings.push(QualificationCurrentnessCadenceHeadFinding::InvalidFinalInclusionProof);
        invalid = true;
    }

    let closure = if invalid {
        QualificationCurrentnessCadenceHeadClosure::Invalid
    } else if blocked {
        QualificationCurrentnessCadenceHeadClosure::Blocked
    } else if later_policy {
        QualificationCurrentnessCadenceHeadClosure::LaterPolicyPublicationObserved
    } else if stale {
        QualificationCurrentnessCadenceHeadClosure::Stale
    } else if incomplete {
        QualificationCurrentnessCadenceHeadClosure::Incomplete
    } else {
        QualificationCurrentnessCadenceHeadClosure::LatestPublishedPolicyThroughFederatedHead
    };

    let final_inclusion_sha256 = final_inclusion_digest(final_inclusion);
    let assessment_sha256 = cadence_policy_head_digest(
        publication,
        &predecessor_publication_sha256s,
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
    QualificationCurrentnessCadenceHeadAssessment {
        qualification_sha256: publication.qualification_sha256().clone(),
        policy_version: publication.version(),
        policy_sha256: publication.policy_sha256().clone(),
        policy_authority_sha256: publication.policy_authority_sha256().clone(),
        policy_publication_sha256: publication.publication_sha256().clone(),
        namespace_sha256: publication.namespace_sha256().clone(),
        head_federation_receipt_sha256: head_federation.receipt_sha256().clone(),
        observed_head_namespaced_view_sha256: observed_head.namespaced_view_sha256().clone(),
        observed_head_tree_size: observed_head.tree_size(),
        predecessor_publication_sha256s,
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
    let mut digest = FramedDigest::new(CADENCE_POLICY_FINAL_INCLUSION_DOMAIN);
    digest.text(&proof.tree_size().to_string());
    digest.text(&proof.leaf_index().to_string());
    digest.text(proof.entry_sha256().as_str());
    digest.text(proof.root_sha256().as_str());
    for node in proof.path() { digest.text(node.as_str()); }
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn cadence_policy_head_digest(
    publication: &PubliclyAnchoredQualificationCurrentnessCadencePolicy,
    predecessor_publications: &[TrustSha256Digest],
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion_sha256: &TrustSha256Digest,
    policy: &QualificationCurrentnessCadenceHeadPolicy,
    findings: &[QualificationCurrentnessCadenceHeadFinding],
    closure: QualificationCurrentnessCadenceHeadClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CADENCE_POLICY_HEAD_DOMAIN);
    digest.text(publication.qualification_sha256().as_str());
    digest.text(&publication.version().to_string());
    digest.text(publication.policy_sha256().as_str());
    digest.text(publication.policy_authority_sha256().as_str());
    digest.text(publication.publication_sha256().as_str());
    for value in predecessor_publications {
        digest.text("predecessor-publication");
        digest.text(value.as_str());
    }
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
    for finding in findings {
        digest_finding(&mut digest, finding);
    }
    digest.text(match closure {
        QualificationCurrentnessCadenceHeadClosure::LatestPublishedPolicyThroughFederatedHead => {
            "latest-published-policy-through-federated-head"
        }
        QualificationCurrentnessCadenceHeadClosure::LaterPolicyPublicationObserved => {
            "later-policy-publication-observed"
        }
        QualificationCurrentnessCadenceHeadClosure::Stale => "stale",
        QualificationCurrentnessCadenceHeadClosure::Incomplete => "incomplete",
        QualificationCurrentnessCadenceHeadClosure::Blocked => "blocked",
        QualificationCurrentnessCadenceHeadClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-policy-not-established");
    digest.text("currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_finding(
    digest: &mut FramedDigest,
    finding: &QualificationCurrentnessCadenceHeadFinding,
) {
    use QualificationCurrentnessCadenceHeadFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::PublicationViewMismatch => digest.text("publication-view-mismatch"),
        Finding::PublicationViewMissingFromMonitor => digest.text("publication-view-missing"),
        Finding::HeadViewMissingFromMonitor => digest.text("head-view-missing"),
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
        Finding::PolicyChainTooLarge => digest.text("policy-chain-too-large"),
        Finding::PolicyChainLengthMismatch { expected, actual } => {
            digest.text("policy-chain-length-mismatch");
            digest.text(&expected.to_string());
            digest.text(&actual.to_string());
        }
        Finding::PolicyChainQualificationMismatch { version } => {
            digest.text("policy-chain-qualification-mismatch");
            digest.text(&version.to_string());
        }
        Finding::PolicyChainNamespaceMismatch { version } => {
            digest.text("policy-chain-namespace-mismatch");
            digest.text(&version.to_string());
        }
        Finding::PolicyChainVersionMismatch { expected, actual } => {
            digest.text("policy-chain-version-mismatch");
            digest.text(&expected.to_string());
            digest.text(&actual.to_string());
        }
        Finding::PolicyChainPredecessorMismatch { version } => {
            digest.text("policy-chain-predecessor-mismatch");
            digest.text(&version.to_string());
        }
        Finding::PolicyChainViewMissingFromMonitor { version } => {
            digest.text("policy-chain-view-missing-from-monitor");
            digest.text(&version.to_string());
        }
        Finding::PolicyChainPublishedAfterAnchor { version } => {
            digest.text("policy-chain-published-after-anchor");
            digest.text(&version.to_string());
        }
        Finding::AnchorPredecessorMismatch => digest.text("anchor-predecessor-mismatch"),
        Finding::AnchorEntryMismatch => digest.text("anchor-entry-mismatch"),
        Finding::AnchorEntryWrongKind => digest.text("anchor-entry-wrong-kind"),
        Finding::AnchorEntryWrongSubject => digest.text("anchor-entry-wrong-subject"),
        Finding::AnchorEntryWrongPayload => digest.text("anchor-entry-wrong-payload"),
        Finding::AnchorEntryWrongContext => digest.text("anchor-entry-wrong-context"),
        Finding::AnchorEntryBeyondPublicationCheckpoint => {
            digest.text("anchor-entry-beyond-publication-checkpoint")
        }
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
        Finding::MalformedLaterPolicyPublication { sequence } => {
            digest.text("malformed-later-policy-publication");
            digest.text(&sequence.to_string());
        }
        Finding::LaterPolicyPublication { sequence } => {
            digest.text("later-policy-publication");
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
    fn positive_head_still_refuses_global_latest_policy() {
        fn _assert_api(value: &QualificationCurrentnessCadenceHeadAssessment) {
            if value.latest_published_policy_through_observed_head_established() {
                assert!(!value.globally_latest_policy_established());
                assert!(!value.currentness_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
