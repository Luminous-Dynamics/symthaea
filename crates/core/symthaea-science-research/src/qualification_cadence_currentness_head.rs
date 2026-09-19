// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complete-tail anti-rollback assessment for published cadence-conformant currentness.
//!
//! A cadence-conformant positive publication can become stale for several reasons:
//! a newer cadence-currentness evaluation, a newer ordinary currentness observation,
//! a newer qualification-lifecycle event, or a newer cadence policy. This layer
//! scans the complete transparency tail from one published cadence-currentness
//! anchor through a later fresh federated head and fails closed on any different
//! same-qualification publication in those streams.
//!
//! A positive result means only "latest among these relevant publication streams
//! through this exact fresh federated head". It does not extend the scientific
//! currentness evaluation to the later head's time and does not establish global
//! latest visibility or scientific truth.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorReceipt,
    NamespacedWitnessedTransparencyCheckpoint, Sha256Digest as TrustSha256Digest,
    TransparencyEntry, TransparencyHeadFederationClosure, TransparencyHeadFederationReceipt,
    TransparencyInclusionProof, verify_transparency_inclusion,
};

use crate::{
    CadenceConformantInstitutionalCurrentnessAtEvaluation,
    PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation,
    QualificationCurrentnessGuardHeadBundle,
    SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE,
    SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE,
    SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE,
    SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE,
    Sha256Digest,
};

const CADENCE_CURRENTNESS_HEAD_DOMAIN: &str =
    "symthaea.cadence-currentness-publication-head.identity.v1";
const FINAL_INCLUSION_IDENTITY_DOMAIN: &str =
    "symthaea.cadence-currentness-tail-final-inclusion.identity.v1";
pub const MAX_CADENCE_CURRENTNESS_TAIL_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CadenceCurrentnessHeadPolicy {
    pub maximum_tail_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadenceCurrentnessHeadPolicyError {
    ZeroTailCapacity,
    TailCapacityTooLarge,
}

impl CadenceCurrentnessHeadPolicy {
    pub fn validate(&self) -> Result<(), CadenceCurrentnessHeadPolicyError> {
        if self.maximum_tail_entries == 0 {
            return Err(CadenceCurrentnessHeadPolicyError::ZeroTailCapacity);
        }
        if self.maximum_tail_entries > MAX_CADENCE_CURRENTNESS_TAIL_ENTRIES {
            return Err(CadenceCurrentnessHeadPolicyError::TailCapacityTooLarge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CadenceCurrentnessHeadClosure {
    LatestRelevantPublicationThroughFederatedHead,
    LaterRelevantPublicationObserved,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum CadenceCurrentnessHeadFinding {
    InvalidPolicy,
    QualificationMismatch,
    CapabilityMismatch,
    GuardMismatch,
    PublicationViewMismatch,
    NamespaceMismatch,
    HeadFederationNotFresh,
    HeadFederationMonitorMismatch,
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
    LaterCadenceCurrentnessPublication { sequence: u64 },
    LaterOrdinaryCurrentnessPublication { sequence: u64 },
    LaterLifecyclePublication { sequence: u64 },
    LaterCadencePolicyPublication { sequence: u64 },
    FinalEntryNotAtHead { final_sequence: u64, head_tree_size: u64 },
    FinalInclusionEntryMismatch,
    FinalInclusionIndexMismatch,
    FinalInclusionTreeSizeMismatch,
    FinalInclusionRootMismatch,
    InvalidFinalInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CadenceCurrentnessHeadAssessment {
    qualification_sha256: Sha256Digest,
    capability_sha256: TrustSha256Digest,
    publication_sha256: TrustSha256Digest,
    namespace_sha256: TrustSha256Digest,
    head_federation_receipt_sha256: TrustSha256Digest,
    observed_head_namespaced_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    observed_head_root_sha256: TrustSha256Digest,
    scanned_from_sequence: u64,
    scanned_through_sequence: u64,
    scanned_tail_entries: usize,
    findings: Vec<CadenceCurrentnessHeadFinding>,
    closure: CadenceCurrentnessHeadClosure,
    assessment_sha256: TrustSha256Digest,
}

impl CadenceCurrentnessHeadAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn capability_sha256(&self) -> &TrustSha256Digest { &self.capability_sha256 }
    pub fn publication_sha256(&self) -> &TrustSha256Digest { &self.publication_sha256 }
    pub fn namespace_sha256(&self) -> &TrustSha256Digest { &self.namespace_sha256 }
    pub fn head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_receipt_sha256
    }
    pub fn observed_head_namespaced_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_namespaced_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn findings(&self) -> &[CadenceCurrentnessHeadFinding] { &self.findings }
    pub fn closure(&self) -> CadenceCurrentnessHeadClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn latest_relevant_publication_through_federated_head_established(&self) -> bool {
        self.closure == CadenceCurrentnessHeadClosure::LatestRelevantPublicationThroughFederatedHead
    }
    pub const fn currentness_at_head_time_established(&self) -> bool { false }
    pub const fn globally_latest_publication_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_published_cadence_currentness_head(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    publication: &PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation,
    currentness_guard: &QualificationCurrentnessGuardHeadBundle,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion: &TransparencyInclusionProof,
    policy: &CadenceCurrentnessHeadPolicy,
) -> CadenceCurrentnessHeadAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut later_relevant = false;

    if policy.validate().is_err() {
        findings.push(CadenceCurrentnessHeadFinding::InvalidPolicy);
        invalid = true;
    }
    if publication.qualification_sha256() != current.qualification_sha256()
        || currentness_guard.guard().qualification_sha256() != current.qualification_sha256()
    {
        findings.push(CadenceCurrentnessHeadFinding::QualificationMismatch);
        invalid = true;
    }
    if publication.capability_sha256() != current.capability_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::CapabilityMismatch);
        invalid = true;
    }
    if currentness_guard.guard().assessment_sha256()
        != current.currentness_guard_assessment_sha256()
    {
        findings.push(CadenceCurrentnessHeadFinding::GuardMismatch);
        invalid = true;
    }
    if publication_view.namespaced_view_sha256() != publication.namespaced_view_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::PublicationViewMismatch);
        invalid = true;
    }
    if publication.namespace_sha256() != observed_head.namespace_sha256()
        || publication.namespace_sha256() != head_federation.namespace_sha256()
    {
        findings.push(CadenceCurrentnessHeadFinding::NamespaceMismatch);
        invalid = true;
    }

    if head_federation.closure() != TransparencyHeadFederationClosure::ConvergedFreshObservedHead
        || !head_federation.converged_fresh_observed_head_established()
    {
        findings.push(CadenceCurrentnessHeadFinding::HeadFederationNotFresh);
        blocked = true;
    }
    if head_federation.monitor_receipt_sha256() != monitor.receipt_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::HeadFederationMonitorMismatch);
        invalid = true;
    }
    if !head_federation
        .converged_namespaced_view_sha256s()
        .contains(observed_head.namespaced_view_sha256())
    {
        findings.push(CadenceCurrentnessHeadFinding::HeadNotFederated);
        incomplete = true;
    }
    if observed_head.tree_size() != head_federation.maximal_tree_size() {
        findings.push(CadenceCurrentnessHeadFinding::HeadTreeSizeMismatch);
        invalid = true;
    }
    if observed_head.root_sha256() != head_federation.maximal_root_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::HeadRootMismatch);
        invalid = true;
    }
    if observed_head.witnessed_view().tree_head_sha256()
        != head_federation.maximal_tree_head_sha256()
    {
        findings.push(CadenceCurrentnessHeadFinding::HeadTreeHeadMismatch);
        invalid = true;
    }

    if anchor_entry.entry_sha256() != publication.transparency_entry_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryMismatch);
        invalid = true;
    }
    if anchor_entry.kind().as_str() != SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryWrongKind);
        invalid = true;
    }
    let qualification_subject = bridge_digest(current.qualification_sha256());
    if anchor_entry.subject_sha256() != &qualification_subject {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryWrongSubject);
        invalid = true;
    }
    if anchor_entry.payload_sha256() != Some(current.capability_sha256()) {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryWrongPayload);
        invalid = true;
    }
    if anchor_entry.context_sha256() != Some(current.cadence_policy_authority_sha256()) {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryWrongContext);
        invalid = true;
    }
    if anchor_entry.sequence() > publication.tree_size() {
        findings.push(CadenceCurrentnessHeadFinding::AnchorEntryBeyondPublicationCheckpoint);
        invalid = true;
    }
    if observed_head.tree_size() < anchor_entry.sequence() {
        findings.push(CadenceCurrentnessHeadFinding::HeadBeforeAnchorEntry);
        invalid = true;
    }
    if subsequent_entries.len() > policy.maximum_tail_entries
        || subsequent_entries.len() > MAX_CADENCE_CURRENTNESS_TAIL_ENTRIES
    {
        findings.push(CadenceCurrentnessHeadFinding::TailTooLarge);
        invalid = true;
    }

    let original_observation_payload = currentness_guard
        .currentness_head()
        .anchor_observation_sha256();
    let lifecycle_payload = bridge_digest(currentness_guard.guard().lifecycle_event_sha256());
    let cadence_policy_payload = current.cadence_policy_authority_sha256();
    let cadence_currentness_payload = current.capability_sha256();

    let mut expected_sequence = anchor_entry.sequence().saturating_add(1);
    let mut previous_entry_sha256 = anchor_entry.entry_sha256().clone();
    for entry in subsequent_entries {
        if entry.sequence() != expected_sequence {
            findings.push(CadenceCurrentnessHeadFinding::TailSequenceGap {
                expected: expected_sequence,
                actual: entry.sequence(),
            });
            invalid = true;
        }
        if entry.previous_entry_sha256() != Some(&previous_entry_sha256) {
            findings.push(CadenceCurrentnessHeadFinding::TailPredecessorMismatch {
                sequence: entry.sequence(),
            });
            invalid = true;
        }

        if entry.subject_sha256() == &qualification_subject {
            match entry.kind().as_str() {
                SCIENCE_CADENCE_CURRENTNESS_PUBLICATION_USAGE
                    if entry.payload_sha256() != Some(cadence_currentness_payload) =>
                {
                    findings.push(
                        CadenceCurrentnessHeadFinding::LaterCadenceCurrentnessPublication {
                            sequence: entry.sequence(),
                        },
                    );
                    later_relevant = true;
                }
                SCIENCE_QUALIFICATION_CURRENTNESS_OBSERVATION_USAGE
                    if entry.payload_sha256() != Some(original_observation_payload) =>
                {
                    findings.push(
                        CadenceCurrentnessHeadFinding::LaterOrdinaryCurrentnessPublication {
                            sequence: entry.sequence(),
                        },
                    );
                    later_relevant = true;
                }
                SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE
                    if entry.payload_sha256() != Some(&lifecycle_payload) =>
                {
                    findings.push(CadenceCurrentnessHeadFinding::LaterLifecyclePublication {
                        sequence: entry.sequence(),
                    });
                    later_relevant = true;
                }
                SCIENCE_QUALIFICATION_CURRENTNESS_CADENCE_POLICY_USAGE
                    if entry.payload_sha256() != Some(cadence_policy_payload) =>
                {
                    findings.push(CadenceCurrentnessHeadFinding::LaterCadencePolicyPublication {
                        sequence: entry.sequence(),
                    });
                    later_relevant = true;
                }
                _ => {}
            }
        }

        previous_entry_sha256 = entry.entry_sha256().clone();
        expected_sequence = entry.sequence().saturating_add(1);
    }

    let final_entry = subsequent_entries.last().unwrap_or(anchor_entry);
    if final_entry.sequence() != observed_head.tree_size() {
        findings.push(CadenceCurrentnessHeadFinding::FinalEntryNotAtHead {
            final_sequence: final_entry.sequence(),
            head_tree_size: observed_head.tree_size(),
        });
        incomplete = true;
    }
    if final_inclusion.entry_sha256() != final_entry.entry_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::FinalInclusionEntryMismatch);
        invalid = true;
    }
    if final_inclusion.leaf_index().saturating_add(1) != observed_head.tree_size() {
        findings.push(CadenceCurrentnessHeadFinding::FinalInclusionIndexMismatch);
        invalid = true;
    }
    if final_inclusion.tree_size() != observed_head.tree_size() {
        findings.push(CadenceCurrentnessHeadFinding::FinalInclusionTreeSizeMismatch);
        invalid = true;
    }
    if final_inclusion.root_sha256() != observed_head.root_sha256() {
        findings.push(CadenceCurrentnessHeadFinding::FinalInclusionRootMismatch);
        invalid = true;
    }
    if verify_transparency_inclusion(final_inclusion).is_err() {
        findings.push(CadenceCurrentnessHeadFinding::InvalidFinalInclusionProof);
        invalid = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        CadenceCurrentnessHeadClosure::Invalid
    } else if blocked {
        CadenceCurrentnessHeadClosure::Blocked
    } else if later_relevant {
        CadenceCurrentnessHeadClosure::LaterRelevantPublicationObserved
    } else if incomplete {
        CadenceCurrentnessHeadClosure::Incomplete
    } else {
        CadenceCurrentnessHeadClosure::LatestRelevantPublicationThroughFederatedHead
    };

    let final_inclusion_sha256 = final_inclusion_digest(final_inclusion);
    let assessment_sha256 = cadence_currentness_head_digest(
        current,
        publication,
        currentness_guard,
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
    CadenceCurrentnessHeadAssessment {
        qualification_sha256: current.qualification_sha256().clone(),
        capability_sha256: current.capability_sha256().clone(),
        publication_sha256: publication.publication_sha256().clone(),
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
    let mut digest = FramedDigest::new(FINAL_INCLUSION_IDENTITY_DOMAIN);
    digest.text(&proof.tree_size().to_string());
    digest.text(&proof.leaf_index().to_string());
    digest.text(proof.entry_sha256().as_str());
    digest.text(proof.root_sha256().as_str());
    for node in proof.path() { digest.text(node.as_str()); }
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn cadence_currentness_head_digest(
    current: &CadenceConformantInstitutionalCurrentnessAtEvaluation,
    publication: &PubliclyAnchoredCadenceConformantCurrentnessAtEvaluation,
    currentness_guard: &QualificationCurrentnessGuardHeadBundle,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion_sha256: &TrustSha256Digest,
    policy: &CadenceCurrentnessHeadPolicy,
    findings: &[CadenceCurrentnessHeadFinding],
    closure: CadenceCurrentnessHeadClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CADENCE_CURRENTNESS_HEAD_DOMAIN);
    digest.text(current.qualification_sha256().as_str());
    digest.text(current.capability_sha256().as_str());
    digest.text(publication.publication_sha256().as_str());
    digest.text(currentness_guard.guard().assessment_sha256().as_str());
    digest.text(currentness_guard.currentness_head().assessment_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(head_federation.receipt_sha256().as_str());
    digest.text(anchor_entry.entry_sha256().as_str());
    for entry in subsequent_entries { digest.text(entry.entry_sha256().as_str()); }
    digest.text(final_inclusion_sha256.as_str());
    digest.text(&policy.maximum_tail_entries.to_string());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        CadenceCurrentnessHeadClosure::LatestRelevantPublicationThroughFederatedHead => {
            "latest-relevant-publication-through-federated-head"
        }
        CadenceCurrentnessHeadClosure::LaterRelevantPublicationObserved => {
            "later-relevant-publication-observed"
        }
        CadenceCurrentnessHeadClosure::Incomplete => "incomplete",
        CadenceCurrentnessHeadClosure::Blocked => "blocked",
        CadenceCurrentnessHeadClosure::Invalid => "invalid",
    });
    digest.text("currentness-at-head-time-not-established");
    digest.text("globally-latest-publication-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_finding(digest: &mut FramedDigest, finding: &CadenceCurrentnessHeadFinding) {
    use CadenceCurrentnessHeadFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::QualificationMismatch => digest.text("qualification-mismatch"),
        Finding::CapabilityMismatch => digest.text("capability-mismatch"),
        Finding::GuardMismatch => digest.text("guard-mismatch"),
        Finding::PublicationViewMismatch => digest.text("publication-view-mismatch"),
        Finding::NamespaceMismatch => digest.text("namespace-mismatch"),
        Finding::HeadFederationNotFresh => digest.text("head-federation-not-fresh"),
        Finding::HeadFederationMonitorMismatch => digest.text("head-federation-monitor-mismatch"),
        Finding::HeadNotFederated => digest.text("head-not-federated"),
        Finding::HeadTreeSizeMismatch => digest.text("head-tree-size-mismatch"),
        Finding::HeadRootMismatch => digest.text("head-root-mismatch"),
        Finding::HeadTreeHeadMismatch => digest.text("head-tree-head-mismatch"),
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
        Finding::LaterCadenceCurrentnessPublication { sequence } => {
            digest.text("later-cadence-currentness-publication");
            digest.text(&sequence.to_string());
        }
        Finding::LaterOrdinaryCurrentnessPublication { sequence } => {
            digest.text("later-ordinary-currentness-publication");
            digest.text(&sequence.to_string());
        }
        Finding::LaterLifecyclePublication { sequence } => {
            digest.text("later-lifecycle-publication");
            digest.text(&sequence.to_string());
        }
        Finding::LaterCadencePolicyPublication { sequence } => {
            digest.text("later-cadence-policy-publication");
            digest.text(&sequence.to_string());
        }
        Finding::FinalEntryNotAtHead { final_sequence, head_tree_size } => {
            digest.text("final-entry-not-at-head");
            digest.text(&final_sequence.to_string());
            digest.text(&head_tree_size.to_string());
        }
        Finding::FinalInclusionEntryMismatch => digest.text("final-inclusion-entry-mismatch"),
        Finding::FinalInclusionIndexMismatch => digest.text("final-inclusion-index-mismatch"),
        Finding::FinalInclusionTreeSizeMismatch => {
            digest.text("final-inclusion-tree-size-mismatch")
        }
        Finding::FinalInclusionRootMismatch => digest.text("final-inclusion-root-mismatch"),
        Finding::InvalidFinalInclusionProof => digest.text("invalid-final-inclusion-proof"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_policy_rejects_unbounded_configuration() {
        let zero = CadenceCurrentnessHeadPolicy { maximum_tail_entries: 0 };
        assert_eq!(zero.validate(), Err(CadenceCurrentnessHeadPolicyError::ZeroTailCapacity));
        let too_large = CadenceCurrentnessHeadPolicy {
            maximum_tail_entries: MAX_CADENCE_CURRENTNESS_TAIL_ENTRIES + 1,
        };
        assert_eq!(
            too_large.validate(),
            Err(CadenceCurrentnessHeadPolicyError::TailCapacityTooLarge),
        );
    }
}
