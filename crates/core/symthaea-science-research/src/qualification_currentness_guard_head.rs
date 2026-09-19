// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-stream anti-rollback guard for qualification currentness.
//!
//! `qualification_currentness_head` proves that no different *currentness*
//! publication for one qualification appears after an anchor through an exact
//! fresh federated witnessed head. That alone is insufficient: a later
//! qualification-lifecycle publication (for example Suspend or Revoke) could be
//! present before a new currentness assessment has been produced.
//!
//! This layer replays the exact currentness-tail proof and scans the same complete
//! tail for later lifecycle publications. A positive result therefore means that
//! neither a newer currentness publication nor a different lifecycle-event
//! publication for the qualification appears through the exact observed head.
//! It still does not establish a globally latest head, currentness at a later
//! time, global evidence exhaustiveness, or scientific truth.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, NamespacedTransparencyMonitorReceipt,
    NamespacedWitnessedTransparencyCheckpoint, Sha256Digest as TrustSha256Digest,
    TransparencyEntry, TransparencyHeadFederationReceipt, TransparencyInclusionProof,
};

use crate::{
    InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    PubliclyAnchoredQualificationCurrentnessObservation,
    QualificationCurrentnessDisposition, QualificationCurrentnessHeadAssessment,
    QualificationCurrentnessHeadClosure, QualificationCurrentnessHeadPolicy,
    SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE, Sha256Digest,
    assess_published_currentness_head,
};

const CURRENTNESS_GUARD_HEAD_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-guard-head.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessGuardHeadClosure {
    GuardedThroughFederatedHead,
    LaterLifecyclePublicationObserved,
    LaterCurrentnessPublicationObserved,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessGuardHeadFinding {
    QualificationMismatch,
    PublicationNotPositiveCurrentness,
    PublicationSourceMismatch,
    PublicationEvaluationIntervalMismatch,
    CurrentnessHeadLaterPublication,
    CurrentnessHeadStale,
    CurrentnessHeadIncomplete,
    CurrentnessHeadBlocked,
    CurrentnessHeadInvalid,
    MalformedLaterLifecyclePublication { sequence: u64 },
    LaterLifecyclePublication { sequence: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessGuardHeadAssessment {
    qualification_sha256: Sha256Digest,
    currentness_sha256: TrustSha256Digest,
    currentness_publication_sha256: TrustSha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    currentness_head_assessment_sha256: TrustSha256Digest,
    observed_head_namespaced_view_sha256: TrustSha256Digest,
    observed_head_tree_size: u64,
    findings: Vec<QualificationCurrentnessGuardHeadFinding>,
    closure: QualificationCurrentnessGuardHeadClosure,
    assessment_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessGuardHeadAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn currentness_sha256(&self) -> &TrustSha256Digest { &self.currentness_sha256 }
    pub fn currentness_publication_sha256(&self) -> &TrustSha256Digest {
        &self.currentness_publication_sha256
    }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn currentness_head_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.currentness_head_assessment_sha256
    }
    pub fn observed_head_namespaced_view_sha256(&self) -> &TrustSha256Digest {
        &self.observed_head_namespaced_view_sha256
    }
    pub fn observed_head_tree_size(&self) -> u64 { self.observed_head_tree_size }
    pub fn findings(&self) -> &[QualificationCurrentnessGuardHeadFinding] { &self.findings }
    pub fn closure(&self) -> QualificationCurrentnessGuardHeadClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn no_later_currentness_or_lifecycle_publication_through_observed_head_established(
        &self,
    ) -> bool {
        self.closure == QualificationCurrentnessGuardHeadClosure::GuardedThroughFederatedHead
    }
    pub const fn globally_latest_currentness_established(&self) -> bool { false }
    pub const fn currentness_at_later_time_established(&self) -> bool { false }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessGuardHeadBundle {
    currentness_head: QualificationCurrentnessHeadAssessment,
    guard: QualificationCurrentnessGuardHeadAssessment,
}

impl QualificationCurrentnessGuardHeadBundle {
    pub fn currentness_head(&self) -> &QualificationCurrentnessHeadAssessment {
        &self.currentness_head
    }
    pub fn guard(&self) -> &QualificationCurrentnessGuardHeadAssessment { &self.guard }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_currentness_cross_stream_guard(
    current: &InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    publication: &PubliclyAnchoredQualificationCurrentnessObservation,
    publication_view: &NamespacedWitnessedTransparencyCheckpoint,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    monitor: &NamespacedTransparencyMonitorReceipt,
    head_federation: &TransparencyHeadFederationReceipt,
    anchor_entry: &TransparencyEntry,
    subsequent_entries: &[TransparencyEntry],
    final_inclusion: &TransparencyInclusionProof,
    policy: &QualificationCurrentnessHeadPolicy,
) -> QualificationCurrentnessGuardHeadBundle {
    let currentness_head = assess_published_currentness_head(
        publication,
        publication_view,
        observed_head,
        monitor,
        head_federation,
        anchor_entry,
        subsequent_entries,
        final_inclusion,
        policy,
    );

    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut later_currentness = false;
    let mut later_lifecycle = false;

    if publication.qualification_sha256() != current.qualification_sha256() {
        findings.push(QualificationCurrentnessGuardHeadFinding::QualificationMismatch);
        invalid = true;
    }
    if publication.disposition()
        != QualificationCurrentnessDisposition::InstitutionallyCurrentAtEvaluation
    {
        findings.push(QualificationCurrentnessGuardHeadFinding::PublicationNotPositiveCurrentness);
        invalid = true;
    }
    if publication.source_sha256() != current.currentness_sha256() {
        findings.push(QualificationCurrentnessGuardHeadFinding::PublicationSourceMismatch);
        invalid = true;
    }
    if publication.evaluation_interval() != current.evaluation_interval() {
        findings.push(
            QualificationCurrentnessGuardHeadFinding::PublicationEvaluationIntervalMismatch,
        );
        invalid = true;
    }

    match currentness_head.closure() {
        QualificationCurrentnessHeadClosure::LatestPublishedThroughFederatedHead => {}
        QualificationCurrentnessHeadClosure::LaterCurrentnessPublicationObserved => {
            findings.push(QualificationCurrentnessGuardHeadFinding::CurrentnessHeadLaterPublication);
            later_currentness = true;
        }
        QualificationCurrentnessHeadClosure::Stale => {
            findings.push(QualificationCurrentnessGuardHeadFinding::CurrentnessHeadStale);
            stale = true;
        }
        QualificationCurrentnessHeadClosure::Incomplete => {
            findings.push(QualificationCurrentnessGuardHeadFinding::CurrentnessHeadIncomplete);
            incomplete = true;
        }
        QualificationCurrentnessHeadClosure::Blocked => {
            findings.push(QualificationCurrentnessGuardHeadFinding::CurrentnessHeadBlocked);
            blocked = true;
        }
        QualificationCurrentnessHeadClosure::Invalid => {
            findings.push(QualificationCurrentnessGuardHeadFinding::CurrentnessHeadInvalid);
            invalid = true;
        }
    }

    let qualification_subject = bridge_digest(current.qualification_sha256());
    let current_lifecycle_payload = bridge_digest(current.lifecycle_event_sha256());
    for entry in subsequent_entries {
        if entry.kind().as_str() != SCIENCE_QUALIFICATION_LIFECYCLE_PUBLICATION_USAGE
            || entry.subject_sha256() != &qualification_subject
        {
            continue;
        }
        match entry.payload_sha256() {
            None => {
                findings.push(
                    QualificationCurrentnessGuardHeadFinding::MalformedLaterLifecyclePublication {
                        sequence: entry.sequence(),
                    },
                );
                invalid = true;
            }
            Some(payload) if payload != &current_lifecycle_payload => {
                findings.push(
                    QualificationCurrentnessGuardHeadFinding::LaterLifecyclePublication {
                        sequence: entry.sequence(),
                    },
                );
                later_lifecycle = true;
            }
            Some(_) => {}
        }
    }

    let closure = if invalid {
        QualificationCurrentnessGuardHeadClosure::Invalid
    } else if blocked {
        QualificationCurrentnessGuardHeadClosure::Blocked
    } else if later_lifecycle {
        QualificationCurrentnessGuardHeadClosure::LaterLifecyclePublicationObserved
    } else if later_currentness {
        QualificationCurrentnessGuardHeadClosure::LaterCurrentnessPublicationObserved
    } else if stale {
        QualificationCurrentnessGuardHeadClosure::Stale
    } else if incomplete {
        QualificationCurrentnessGuardHeadClosure::Incomplete
    } else {
        QualificationCurrentnessGuardHeadClosure::GuardedThroughFederatedHead
    };

    let assessment_sha256 = guard_head_digest(
        current,
        publication,
        &currentness_head,
        observed_head,
        subsequent_entries,
        &findings,
        closure,
    );
    let guard = QualificationCurrentnessGuardHeadAssessment {
        qualification_sha256: current.qualification_sha256().clone(),
        currentness_sha256: current.currentness_sha256().clone(),
        currentness_publication_sha256: publication.publication_sha256().clone(),
        lifecycle_event_sha256: current.lifecycle_event_sha256().clone(),
        currentness_head_assessment_sha256: currentness_head.assessment_sha256().clone(),
        observed_head_namespaced_view_sha256: observed_head.namespaced_view_sha256().clone(),
        observed_head_tree_size: observed_head.tree_size(),
        findings,
        closure,
        assessment_sha256,
    };
    QualificationCurrentnessGuardHeadBundle { currentness_head, guard }
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn guard_head_digest(
    current: &InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    publication: &PubliclyAnchoredQualificationCurrentnessObservation,
    currentness_head: &QualificationCurrentnessHeadAssessment,
    observed_head: &NamespacedWitnessedTransparencyCheckpoint,
    subsequent_entries: &[TransparencyEntry],
    findings: &[QualificationCurrentnessGuardHeadFinding],
    closure: QualificationCurrentnessGuardHeadClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_GUARD_HEAD_DOMAIN);
    digest.text(current.qualification_sha256().as_str());
    digest.text(current.currentness_sha256().as_str());
    digest.text(current.lifecycle_event_sha256().as_str());
    digest.text(publication.publication_sha256().as_str());
    digest.text(currentness_head.assessment_sha256().as_str());
    digest.text(observed_head.namespaced_view_sha256().as_str());
    digest.text(&observed_head.tree_size().to_string());
    for entry in subsequent_entries {
        digest.text("tail-entry");
        digest.text(entry.entry_sha256().as_str());
    }
    for finding in findings {
        digest_guard_finding(&mut digest, finding);
    }
    digest.text(match closure {
        QualificationCurrentnessGuardHeadClosure::GuardedThroughFederatedHead => {
            "guarded-through-federated-head"
        }
        QualificationCurrentnessGuardHeadClosure::LaterLifecyclePublicationObserved => {
            "later-lifecycle-publication-observed"
        }
        QualificationCurrentnessGuardHeadClosure::LaterCurrentnessPublicationObserved => {
            "later-currentness-publication-observed"
        }
        QualificationCurrentnessGuardHeadClosure::Stale => "stale",
        QualificationCurrentnessGuardHeadClosure::Incomplete => "incomplete",
        QualificationCurrentnessGuardHeadClosure::Blocked => "blocked",
        QualificationCurrentnessGuardHeadClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-currentness-not-established");
    digest.text("currentness-at-later-time-not-established");
    digest.text("global-evidence-exhaustiveness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn digest_guard_finding(
    digest: &mut FramedDigest,
    finding: &QualificationCurrentnessGuardHeadFinding,
) {
    use QualificationCurrentnessGuardHeadFinding as Finding;
    match finding {
        Finding::QualificationMismatch => digest.text("qualification-mismatch"),
        Finding::PublicationNotPositiveCurrentness => {
            digest.text("publication-not-positive-currentness")
        }
        Finding::PublicationSourceMismatch => digest.text("publication-source-mismatch"),
        Finding::PublicationEvaluationIntervalMismatch => {
            digest.text("publication-evaluation-interval-mismatch")
        }
        Finding::CurrentnessHeadLaterPublication => {
            digest.text("currentness-head-later-publication")
        }
        Finding::CurrentnessHeadStale => digest.text("currentness-head-stale"),
        Finding::CurrentnessHeadIncomplete => digest.text("currentness-head-incomplete"),
        Finding::CurrentnessHeadBlocked => digest.text("currentness-head-blocked"),
        Finding::CurrentnessHeadInvalid => digest.text("currentness-head-invalid"),
        Finding::MalformedLaterLifecyclePublication { sequence } => {
            digest.text("malformed-later-lifecycle-publication");
            digest.text(&sequence.to_string());
        }
        Finding::LaterLifecyclePublication { sequence } => {
            digest.text("later-lifecycle-publication");
            digest.text(&sequence.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_result_still_refuses_global_currentness() {
        fn _assert_api(value: &QualificationCurrentnessGuardHeadAssessment) {
            if value.no_later_currentness_or_lifecycle_publication_through_observed_head_established() {
                assert!(!value.globally_latest_currentness_established());
                assert!(!value.currentness_at_later_time_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
