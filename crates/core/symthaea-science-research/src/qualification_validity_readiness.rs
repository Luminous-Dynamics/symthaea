// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Aligned readiness assessment for present-tense scientific qualification use.
//!
//! This layer deliberately does not mint `CurrentlyValidQualifiedScientificClaim`.
//! Instead it replays lifecycle-head freshness and evidence freshness using the
//! *same exact `TrustedTime` capability*, then requires the lifecycle head to be
//! one of the fresh converged maximal heads from the transparency federation.
//! This closes the "fresh at different times" composition gap without claiming
//! that the observed head is globally latest or that the evidence search is
//! globally exhaustive.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorReceipt,
    NamespacedWitnessedTransparencyCheckpoint, Sha256Digest as TrustSha256Digest,
    TransparencyEntry, TransparencyHeadFederationClosure,
    TransparencyHeadFederationReceipt, TransparencyInclusionProof, TrustedTime,
};

use crate::{
    ClaimRelationBindingReport, CorpusItemDecision, EvidenceCoverageAuthorityReport,
    EvidenceDecisionBindingReport, EvidenceFreshnessAssessment, EvidenceFreshnessClosure,
    EvidenceFreshnessPolicy, EvidenceRecord, EvidenceRefreshInputs,
    AuthenticatedEvidenceSourceFreshness, FrozenEvidenceCoverageProtocol,
    LifecycleHeadFreshnessAssessment, LifecycleHeadFreshnessClosure,
    LifecycleHeadFreshnessPolicy, PubliclyAnchoredQualificationLifecycleEvent,
    PubliclyAnchoredScientificQualification, QualificationLifecycleEventKind,
    QualifiedScientificClaim, ScientificClaim, Sha256Digest, SourceRetrievalReceipt,
    assess_evidence_freshness, assess_lifecycle_head_freshness,
};

const VALIDITY_READINESS_DOMAIN: &str =
    "symthaea.scientific-qualification-validity-readiness.identity.v1";

pub struct LifecycleValidityReplayInputs<'a> {
    pub qualification: &'a PubliclyAnchoredScientificQualification,
    pub lifecycle: &'a PubliclyAnchoredQualificationLifecycleEvent,
    pub qualification_view: &'a NamespacedWitnessedTransparencyCheckpoint,
    pub lifecycle_view: &'a NamespacedWitnessedTransparencyCheckpoint,
    pub observed_head: &'a NamespacedWitnessedTransparencyCheckpoint,
    pub monitor: &'a NamespacedTransparencyMonitorReceipt,
    pub anchor_entry: &'a TransparencyEntry,
    pub subsequent_entries: &'a [TransparencyEntry],
    pub final_inclusion: &'a TransparencyInclusionProof,
    pub policy: &'a LifecycleHeadFreshnessPolicy,
}

pub struct EvidenceValidityReplayInputs<'a> {
    pub qualified: &'a QualifiedScientificClaim,
    pub claim: &'a ScientificClaim,
    pub claim_binding: &'a ClaimRelationBindingReport,
    pub baseline_protocol: &'a FrozenEvidenceCoverageProtocol,
    pub baseline_coverage: &'a EvidenceCoverageAuthorityReport,
    pub baseline_decision_binding: &'a EvidenceDecisionBindingReport,
    pub refresh_protocol: &'a FrozenEvidenceCoverageProtocol,
    pub refresh_coverage: &'a EvidenceCoverageAuthorityReport,
    pub refresh_decision_binding: &'a EvidenceDecisionBindingReport,
    pub known_evidence: &'a [EvidenceRecord],
    pub receipts: &'a [SourceRetrievalReceipt],
    pub decisions: &'a [CorpusItemDecision],
    pub source_freshness: &'a [AuthenticatedEvidenceSourceFreshness],
    pub policy: &'a EvidenceFreshnessPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationValidityReadinessClosure {
    ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol,
    NotActiveWithinFreshObservedHead,
    EvidenceChangedRequiresReview,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QualificationValidityReadinessFinding {
    QualificationIdentityMismatch,
    HeadFederationNotConverged,
    HeadFederationStale,
    HeadFederationIncomplete,
    HeadFederationBlocked,
    HeadFederationInvalid,
    EvaluationTimeAuthorityMismatch,
    EvaluationIntervalMismatch,
    MonitorReceiptMismatch,
    NamespaceMismatch,
    ObservedHeadNotFederated,
    ObservedHeadTreeSizeMismatch,
    ObservedHeadRootMismatch,
    ObservedHeadTreeHeadMismatch,
    LifecycleFreshnessLaterPublication,
    LifecycleFreshnessStale,
    LifecycleFreshnessIncomplete,
    LifecycleFreshnessBlocked,
    LifecycleFreshnessInvalid,
    LifecycleNotActive,
    EvidenceChanged,
    EvidenceStale,
    EvidenceIncomplete,
    EvidenceBlocked,
    EvidenceInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationValidityReadinessAssessment {
    qualification_sha256: Sha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    head_federation_receipt_sha256: TrustSha256Digest,
    lifecycle_freshness_assessment_sha256: TrustSha256Digest,
    evidence_freshness_assessment_sha256: TrustSha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    observed_namespace_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    observed_head_root_sha256: TrustSha256Digest,
    findings: Vec<QualificationValidityReadinessFinding>,
    closure: QualificationValidityReadinessClosure,
    assessment_sha256: TrustSha256Digest,
}

impl QualificationValidityReadinessAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn head_federation_receipt_sha256(&self) -> &TrustSha256Digest {
        &self.head_federation_receipt_sha256
    }
    pub fn lifecycle_freshness_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_freshness_assessment_sha256
    }
    pub fn evidence_freshness_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.evidence_freshness_assessment_sha256
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn findings(&self) -> &[QualificationValidityReadinessFinding] { &self.findings }
    pub fn closure(&self) -> QualificationValidityReadinessClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn validity_readiness_established(&self) -> bool {
        self.closure
            == QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol
    }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn globally_latest_head_established(&self) -> bool { false }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationValidityReadinessBundle {
    lifecycle_freshness: LifecycleHeadFreshnessAssessment,
    evidence_freshness: EvidenceFreshnessAssessment,
    readiness: QualificationValidityReadinessAssessment,
}

impl QualificationValidityReadinessBundle {
    pub fn lifecycle_freshness(&self) -> &LifecycleHeadFreshnessAssessment {
        &self.lifecycle_freshness
    }
    pub fn evidence_freshness(&self) -> &EvidenceFreshnessAssessment { &self.evidence_freshness }
    pub fn readiness(&self) -> &QualificationValidityReadinessAssessment { &self.readiness }
}

pub fn assess_qualification_validity_readiness(
    lifecycle_inputs: LifecycleValidityReplayInputs<'_>,
    evidence_inputs: EvidenceValidityReplayInputs<'_>,
    head_federation: &TransparencyHeadFederationReceipt,
    evaluation_time: &TrustedTime,
) -> QualificationValidityReadinessBundle {
    let evidence_refresh_inputs = EvidenceRefreshInputs {
        known_evidence: evidence_inputs.known_evidence,
        receipts: evidence_inputs.receipts,
        decisions: evidence_inputs.decisions,
    };
    let lifecycle_freshness = assess_lifecycle_head_freshness(
        lifecycle_inputs.qualification,
        lifecycle_inputs.lifecycle,
        lifecycle_inputs.qualification_view,
        lifecycle_inputs.lifecycle_view,
        lifecycle_inputs.observed_head,
        lifecycle_inputs.monitor,
        evaluation_time,
        lifecycle_inputs.anchor_entry,
        lifecycle_inputs.subsequent_entries,
        lifecycle_inputs.final_inclusion,
        lifecycle_inputs.policy,
    );
    let evidence_freshness = assess_evidence_freshness(
        evidence_inputs.qualified,
        evidence_inputs.claim,
        evidence_inputs.claim_binding,
        evidence_inputs.baseline_protocol,
        evidence_inputs.baseline_coverage,
        evidence_inputs.baseline_decision_binding,
        evidence_inputs.refresh_protocol,
        evidence_inputs.refresh_coverage,
        evidence_inputs.refresh_decision_binding,
        evidence_refresh_inputs,
        evidence_inputs.source_freshness,
        evaluation_time,
        evidence_inputs.policy,
    );

    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut not_active = false;
    let mut evidence_changed = false;

    if lifecycle_inputs.qualification.qualification_sha256()
        != evidence_inputs.qualified.qualification_sha256()
        || lifecycle_inputs.lifecycle.qualification_sha256()
            != evidence_inputs.qualified.qualification_sha256()
    {
        findings.push(QualificationValidityReadinessFinding::QualificationIdentityMismatch);
        invalid = true;
    }

    match head_federation.closure() {
        TransparencyHeadFederationClosure::ConvergedFreshObservedHead => {}
        TransparencyHeadFederationClosure::Stale => {
            findings.push(QualificationValidityReadinessFinding::HeadFederationStale);
            stale = true;
        }
        TransparencyHeadFederationClosure::Incomplete => {
            findings.push(QualificationValidityReadinessFinding::HeadFederationIncomplete);
            incomplete = true;
        }
        TransparencyHeadFederationClosure::Blocked => {
            findings.push(QualificationValidityReadinessFinding::HeadFederationBlocked);
            blocked = true;
        }
        TransparencyHeadFederationClosure::Invalid => {
            findings.push(QualificationValidityReadinessFinding::HeadFederationInvalid);
            invalid = true;
        }
    }
    if !head_federation.converged_fresh_observed_head_established() {
        findings.push(QualificationValidityReadinessFinding::HeadFederationNotConverged);
    }

    let evaluation_interval = evaluation_time.consensus_interval();
    if head_federation.evaluation_time_authority_sha256() != evaluation_time.authority_sha256() {
        findings.push(QualificationValidityReadinessFinding::EvaluationTimeAuthorityMismatch);
        invalid = true;
    }
    if head_federation.evaluation_interval() != evaluation_interval {
        findings.push(QualificationValidityReadinessFinding::EvaluationIntervalMismatch);
        invalid = true;
    }
    if head_federation.monitor_receipt_sha256() != lifecycle_inputs.monitor.receipt_sha256() {
        findings.push(QualificationValidityReadinessFinding::MonitorReceiptMismatch);
        invalid = true;
    }
    if head_federation.namespace_sha256() != lifecycle_inputs.observed_head.namespace_sha256() {
        findings.push(QualificationValidityReadinessFinding::NamespaceMismatch);
        invalid = true;
    }
    if !head_federation
        .converged_namespaced_view_sha256s()
        .contains(lifecycle_inputs.observed_head.namespaced_view_sha256())
    {
        findings.push(QualificationValidityReadinessFinding::ObservedHeadNotFederated);
        incomplete = true;
    }
    if lifecycle_inputs.observed_head.tree_size() != head_federation.maximal_tree_size() {
        findings.push(QualificationValidityReadinessFinding::ObservedHeadTreeSizeMismatch);
        invalid = true;
    }
    if lifecycle_inputs.observed_head.root_sha256() != head_federation.maximal_root_sha256() {
        findings.push(QualificationValidityReadinessFinding::ObservedHeadRootMismatch);
        invalid = true;
    }
    if lifecycle_inputs.observed_head.witnessed_view().tree_head_sha256()
        != head_federation.maximal_tree_head_sha256()
    {
        findings.push(QualificationValidityReadinessFinding::ObservedHeadTreeHeadMismatch);
        invalid = true;
    }

    match lifecycle_freshness.closure() {
        LifecycleHeadFreshnessClosure::FreshThroughObservedHead => {}
        LifecycleHeadFreshnessClosure::LaterLifecyclePublicationObserved => {
            findings.push(QualificationValidityReadinessFinding::LifecycleFreshnessLaterPublication);
            blocked = true;
        }
        LifecycleHeadFreshnessClosure::Stale => {
            findings.push(QualificationValidityReadinessFinding::LifecycleFreshnessStale);
            stale = true;
        }
        LifecycleHeadFreshnessClosure::Incomplete => {
            findings.push(QualificationValidityReadinessFinding::LifecycleFreshnessIncomplete);
            incomplete = true;
        }
        LifecycleHeadFreshnessClosure::Blocked => {
            findings.push(QualificationValidityReadinessFinding::LifecycleFreshnessBlocked);
            blocked = true;
        }
        LifecycleHeadFreshnessClosure::Invalid => {
            findings.push(QualificationValidityReadinessFinding::LifecycleFreshnessInvalid);
            invalid = true;
        }
    }

    if !matches!(
        lifecycle_inputs.lifecycle.event_kind(),
        QualificationLifecycleEventKind::Activate | QualificationLifecycleEventKind::Renew
    ) {
        findings.push(QualificationValidityReadinessFinding::LifecycleNotActive);
        not_active = true;
    }

    match evidence_freshness.closure() {
        EvidenceFreshnessClosure::FreshWithinFrozenProtocol => {}
        EvidenceFreshnessClosure::EvidenceChangedRequiresReview => {
            findings.push(QualificationValidityReadinessFinding::EvidenceChanged);
            evidence_changed = true;
        }
        EvidenceFreshnessClosure::Stale => {
            findings.push(QualificationValidityReadinessFinding::EvidenceStale);
            stale = true;
        }
        EvidenceFreshnessClosure::Incomplete => {
            findings.push(QualificationValidityReadinessFinding::EvidenceIncomplete);
            incomplete = true;
        }
        EvidenceFreshnessClosure::Blocked => {
            findings.push(QualificationValidityReadinessFinding::EvidenceBlocked);
            blocked = true;
        }
        EvidenceFreshnessClosure::Invalid => {
            findings.push(QualificationValidityReadinessFinding::EvidenceInvalid);
            invalid = true;
        }
    }

    let closure = if invalid {
        QualificationValidityReadinessClosure::Invalid
    } else if blocked {
        QualificationValidityReadinessClosure::Blocked
    } else if not_active {
        QualificationValidityReadinessClosure::NotActiveWithinFreshObservedHead
    } else if evidence_changed {
        QualificationValidityReadinessClosure::EvidenceChangedRequiresReview
    } else if stale {
        QualificationValidityReadinessClosure::Stale
    } else if incomplete {
        QualificationValidityReadinessClosure::Incomplete
    } else {
        QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol
    };

    let assessment_sha256 = validity_readiness_digest(
        evidence_inputs.qualified,
        lifecycle_inputs.lifecycle,
        head_federation,
        &lifecycle_freshness,
        &evidence_freshness,
        evaluation_time,
        lifecycle_inputs.observed_head,
        &findings,
        closure,
    );
    let readiness = QualificationValidityReadinessAssessment {
        qualification_sha256: evidence_inputs.qualified.qualification_sha256().clone(),
        lifecycle_event_sha256: lifecycle_inputs.lifecycle.lifecycle_event_sha256().clone(),
        head_federation_receipt_sha256: head_federation.receipt_sha256().clone(),
        lifecycle_freshness_assessment_sha256: lifecycle_freshness.assessment_sha256().clone(),
        evidence_freshness_assessment_sha256: evidence_freshness.assessment_sha256().clone(),
        evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
        evaluation_earliest_unix_s: evaluation_interval.0,
        evaluation_latest_unix_s: evaluation_interval.1,
        observed_namespace_sha256: lifecycle_inputs.observed_head.namespace_sha256().clone(),
        observed_head_tree_size: lifecycle_inputs.observed_head.tree_size(),
        observed_head_root_sha256: lifecycle_inputs.observed_head.root_sha256().clone(),
        findings,
        closure,
        assessment_sha256,
    };

    QualificationValidityReadinessBundle {
        lifecycle_freshness,
        evidence_freshness,
        readiness,
    }
}

#[allow(clippy::too_many_arguments)]
fn validity_readiness_digest(
    qualified: &QualifiedScientificClaim,
    lifecycle: &PubliclyAnchoredQualificationLifecycleEvent,
    head_federation: &TransparencyHeadFederationReceipt,
    lifecycle_freshness: &LifecycleHeadFreshnessAssessment,
    evidence_freshness: &EvidenceFreshnessAssessment,
    evaluation_time: &TrustedTime,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    findings: &[QualificationValidityReadinessFinding],
    closure: QualificationValidityReadinessClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(VALIDITY_READINESS_DOMAIN);
    digest.text(qualified.qualification_sha256().as_str());
    digest.text(lifecycle.lifecycle_event_sha256().as_str());
    digest.text(head_federation.receipt_sha256().as_str());
    digest.text(lifecycle_freshness.assessment_sha256().as_str());
    digest.text(evidence_freshness.assessment_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    let (earliest, latest) = evaluation_time.consensus_interval();
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(observed_head.namespace_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(&observed_head.tree_size().to_string());
    digest.text(observed_head.root_sha256().as_str());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol => {
            "ready-within-observed-federated-head-and-frozen-evidence-protocol"
        }
        QualificationValidityReadinessClosure::NotActiveWithinFreshObservedHead => {
            "not-active-within-fresh-observed-head"
        }
        QualificationValidityReadinessClosure::EvidenceChangedRequiresReview => {
            "evidence-changed-requires-review"
        }
        QualificationValidityReadinessClosure::Stale => "stale",
        QualificationValidityReadinessClosure::Incomplete => "incomplete",
        QualificationValidityReadinessClosure::Blocked => "blocked",
        QualificationValidityReadinessClosure::Invalid => "invalid",
    });
    digest.text("current-validity-not-established");
    digest.text("globally-latest-head-not-established");
    digest.text("global-evidence-exhaustiveness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_finding(digest: &mut FramedDigest, finding: &QualificationValidityReadinessFinding) {
    use QualificationValidityReadinessFinding as Finding;
    digest.text(match finding {
        Finding::QualificationIdentityMismatch => "qualification-identity-mismatch",
        Finding::HeadFederationNotConverged => "head-federation-not-converged",
        Finding::HeadFederationStale => "head-federation-stale",
        Finding::HeadFederationIncomplete => "head-federation-incomplete",
        Finding::HeadFederationBlocked => "head-federation-blocked",
        Finding::HeadFederationInvalid => "head-federation-invalid",
        Finding::EvaluationTimeAuthorityMismatch => "evaluation-time-authority-mismatch",
        Finding::EvaluationIntervalMismatch => "evaluation-interval-mismatch",
        Finding::MonitorReceiptMismatch => "monitor-receipt-mismatch",
        Finding::NamespaceMismatch => "namespace-mismatch",
        Finding::ObservedHeadNotFederated => "observed-head-not-federated",
        Finding::ObservedHeadTreeSizeMismatch => "observed-head-tree-size-mismatch",
        Finding::ObservedHeadRootMismatch => "observed-head-root-mismatch",
        Finding::ObservedHeadTreeHeadMismatch => "observed-head-tree-head-mismatch",
        Finding::LifecycleFreshnessLaterPublication => "lifecycle-freshness-later-publication",
        Finding::LifecycleFreshnessStale => "lifecycle-freshness-stale",
        Finding::LifecycleFreshnessIncomplete => "lifecycle-freshness-incomplete",
        Finding::LifecycleFreshnessBlocked => "lifecycle-freshness-blocked",
        Finding::LifecycleFreshnessInvalid => "lifecycle-freshness-invalid",
        Finding::LifecycleNotActive => "lifecycle-not-active",
        Finding::EvidenceChanged => "evidence-changed",
        Finding::EvidenceStale => "evidence-stale",
        Finding::EvidenceIncomplete => "evidence-incomplete",
        Finding::EvidenceBlocked => "evidence-blocked",
        Finding::EvidenceInvalid => "evidence-invalid",
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_is_explicitly_weaker_than_current_validity() {
        fn _assert_api(assessment: &QualificationValidityReadinessAssessment) {
            if assessment.validity_readiness_established() {
                assert!(!assessment.current_validity_established());
                assert!(!assessment.globally_latest_head_established());
                assert!(!assessment.global_evidence_exhaustiveness_established());
                assert!(!assessment.scientific_truth_established());
            }
        }
    }
}
