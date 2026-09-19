// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Monotonic local observation tracker for qualification currentness.
//!
//! A retained positive currentness capability is historical evidence about one
//! evaluation interval. This tracker prevents local rollback by accepting newer
//! positive institutional-currentness capabilities *and* newer non-ready
//! readiness assessments, while requiring unambiguous temporal ordering between
//! their authenticated evaluation intervals.
//!
//! The tracker itself is not a durable anti-rollback root. Callers must persist or
//! externally anchor tracker state if they want protection across process/state
//! loss. It also does not establish global latest-head visibility.

use serde::Serialize;
use symthaea_trust_core::{FramedDigest, Sha256Digest as TrustSha256Digest};

use crate::{
    InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    QualificationValidityReadinessAssessment, QualificationValidityReadinessClosure,
    Sha256Digest,
};

const CURRENTNESS_OBSERVATION_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-observation.identity.v1";
const CURRENTNESS_TRACKER_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-tracker.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessDisposition {
    InstitutionallyCurrentAtEvaluation,
    NotActive,
    EvidenceChangedRequiresReview,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessObservation {
    qualification_sha256: Sha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    source_sha256: TrustSha256Digest,
    disposition: QualificationCurrentnessDisposition,
    observation_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessObservation {
    pub fn from_currentness(
        current: &InstitutionallyCurrentAtEvaluationQualifiedScientificClaim,
    ) -> Self {
        let (earliest, latest) = current.evaluation_interval();
        let source_sha256 = current.currentness_sha256().clone();
        let disposition = QualificationCurrentnessDisposition::InstitutionallyCurrentAtEvaluation;
        let observation_sha256 = currentness_observation_digest(
            current.qualification_sha256(),
            current.evaluation_time_authority_sha256(),
            earliest,
            latest,
            &source_sha256,
            disposition,
        );
        Self {
            qualification_sha256: current.qualification_sha256().clone(),
            evaluation_time_authority_sha256: current.evaluation_time_authority_sha256().clone(),
            evaluation_earliest_unix_s: earliest,
            evaluation_latest_unix_s: latest,
            source_sha256,
            disposition,
            observation_sha256,
        }
    }

    pub fn from_nonready_assessment(
        readiness: &QualificationValidityReadinessAssessment,
    ) -> Result<Self, QualificationCurrentnessObservationError> {
        let disposition = match readiness.closure() {
            QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol => {
                return Err(QualificationCurrentnessObservationError::ReadyAssessmentRequiresInstitutionalCurrentness);
            }
            QualificationValidityReadinessClosure::NotActiveWithinFreshObservedHead => {
                QualificationCurrentnessDisposition::NotActive
            }
            QualificationValidityReadinessClosure::EvidenceChangedRequiresReview => {
                QualificationCurrentnessDisposition::EvidenceChangedRequiresReview
            }
            QualificationValidityReadinessClosure::Stale => QualificationCurrentnessDisposition::Stale,
            QualificationValidityReadinessClosure::Incomplete => {
                QualificationCurrentnessDisposition::Incomplete
            }
            QualificationValidityReadinessClosure::Blocked => {
                QualificationCurrentnessDisposition::Blocked
            }
            QualificationValidityReadinessClosure::Invalid => {
                QualificationCurrentnessDisposition::Invalid
            }
        };
        let (earliest, latest) = readiness.evaluation_interval();
        let source_sha256 = readiness.assessment_sha256().clone();
        let observation_sha256 = currentness_observation_digest(
            readiness.qualification_sha256(),
            readiness.evaluation_time_authority_sha256(),
            earliest,
            latest,
            &source_sha256,
            disposition,
        );
        Ok(Self {
            qualification_sha256: readiness.qualification_sha256().clone(),
            evaluation_time_authority_sha256: readiness.evaluation_time_authority_sha256().clone(),
            evaluation_earliest_unix_s: earliest,
            evaluation_latest_unix_s: latest,
            source_sha256,
            disposition,
            observation_sha256,
        })
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn source_sha256(&self) -> &TrustSha256Digest { &self.source_sha256 }
    pub fn disposition(&self) -> QualificationCurrentnessDisposition { self.disposition }
    pub fn observation_sha256(&self) -> &TrustSha256Digest { &self.observation_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessObservationError {
    ReadyAssessmentRequiresInstitutionalCurrentness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessTrackerError {
    QualificationMismatch,
    DuplicateObservation,
    SameIntervalDifferentObservation,
    EvaluationIntervalRollback,
    EvaluationIntervalsOverlap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessTracker {
    qualification_sha256: Sha256Digest,
    sequence: u64,
    latest_observation_sha256: TrustSha256Digest,
    latest_evaluation_time_authority_sha256: TrustSha256Digest,
    latest_evaluation_earliest_unix_s: u64,
    latest_evaluation_latest_unix_s: u64,
    latest_disposition: QualificationCurrentnessDisposition,
    tracker_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessTracker {
    pub fn new(first: &QualificationCurrentnessObservation) -> Self {
        let tracker_sha256 = tracker_digest(
            first.qualification_sha256(),
            1,
            None,
            first,
        );
        Self {
            qualification_sha256: first.qualification_sha256().clone(),
            sequence: 1,
            latest_observation_sha256: first.observation_sha256().clone(),
            latest_evaluation_time_authority_sha256: first
                .evaluation_time_authority_sha256()
                .clone(),
            latest_evaluation_earliest_unix_s: first.evaluation_interval().0,
            latest_evaluation_latest_unix_s: first.evaluation_interval().1,
            latest_disposition: first.disposition(),
            tracker_sha256,
        }
    }

    pub fn observe(
        &mut self,
        next: &QualificationCurrentnessObservation,
    ) -> Result<(), QualificationCurrentnessTrackerError> {
        if next.qualification_sha256() != &self.qualification_sha256 {
            return Err(QualificationCurrentnessTrackerError::QualificationMismatch);
        }
        if next.observation_sha256() == &self.latest_observation_sha256 {
            return Err(QualificationCurrentnessTrackerError::DuplicateObservation);
        }

        let (next_earliest, next_latest) = next.evaluation_interval();
        let current_earliest = self.latest_evaluation_earliest_unix_s;
        let current_latest = self.latest_evaluation_latest_unix_s;
        if next_earliest == current_earliest && next_latest == current_latest {
            return Err(QualificationCurrentnessTrackerError::SameIntervalDifferentObservation);
        }
        if next_latest <= current_earliest {
            return Err(QualificationCurrentnessTrackerError::EvaluationIntervalRollback);
        }
        if next_earliest <= current_latest {
            return Err(QualificationCurrentnessTrackerError::EvaluationIntervalsOverlap);
        }

        let previous_tracker_sha256 = self.tracker_sha256.clone();
        let sequence = self.sequence.saturating_add(1);
        self.tracker_sha256 = tracker_digest(
            &self.qualification_sha256,
            sequence,
            Some(&previous_tracker_sha256),
            next,
        );
        self.sequence = sequence;
        self.latest_observation_sha256 = next.observation_sha256().clone();
        self.latest_evaluation_time_authority_sha256 =
            next.evaluation_time_authority_sha256().clone();
        self.latest_evaluation_earliest_unix_s = next_earliest;
        self.latest_evaluation_latest_unix_s = next_latest;
        self.latest_disposition = next.disposition();
        Ok(())
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn latest_observation_sha256(&self) -> &TrustSha256Digest {
        &self.latest_observation_sha256
    }
    pub fn latest_evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.latest_evaluation_time_authority_sha256
    }
    pub fn latest_evaluation_interval(&self) -> (u64, u64) {
        (
            self.latest_evaluation_earliest_unix_s,
            self.latest_evaluation_latest_unix_s,
        )
    }
    pub fn latest_disposition(&self) -> QualificationCurrentnessDisposition {
        self.latest_disposition
    }
    pub fn tracker_sha256(&self) -> &TrustSha256Digest { &self.tracker_sha256 }

    pub fn institutionally_current_at_latest_observed_evaluation_established(&self) -> bool {
        self.latest_disposition
            == QualificationCurrentnessDisposition::InstitutionallyCurrentAtEvaluation
    }
    pub const fn durable_anti_rollback_established(&self) -> bool { false }
    pub const fn globally_latest_evaluation_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

fn currentness_observation_digest(
    qualification_sha256: &Sha256Digest,
    evaluation_time_authority_sha256: &TrustSha256Digest,
    earliest: u64,
    latest: u64,
    source_sha256: &TrustSha256Digest,
    disposition: QualificationCurrentnessDisposition,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_OBSERVATION_DOMAIN);
    digest.text(qualification_sha256.as_str());
    digest.text(evaluation_time_authority_sha256.as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(source_sha256.as_str());
    digest.text(disposition_tag(disposition));
    digest.digest()
}

fn tracker_digest(
    qualification_sha256: &Sha256Digest,
    sequence: u64,
    previous_tracker_sha256: Option<&TrustSha256Digest>,
    observation: &QualificationCurrentnessObservation,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_TRACKER_DOMAIN);
    digest.text(qualification_sha256.as_str());
    digest.text(&sequence.to_string());
    digest.optional_sha(previous_tracker_sha256);
    digest.text(observation.observation_sha256().as_str());
    digest.text(observation.evaluation_time_authority_sha256().as_str());
    let (earliest, latest) = observation.evaluation_interval();
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text(disposition_tag(observation.disposition()));
    digest.text("durable-anti-rollback-not-established");
    digest.text("globally-latest-evaluation-not-established");
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
    fn disposition_tags_are_distinct() {
        let values = [
            QualificationCurrentnessDisposition::InstitutionallyCurrentAtEvaluation,
            QualificationCurrentnessDisposition::NotActive,
            QualificationCurrentnessDisposition::EvidenceChangedRequiresReview,
            QualificationCurrentnessDisposition::Stale,
            QualificationCurrentnessDisposition::Incomplete,
            QualificationCurrentnessDisposition::Blocked,
            QualificationCurrentnessDisposition::Invalid,
        ];
        let tags: std::collections::BTreeSet<_> =
            values.into_iter().map(disposition_tag).collect();
        assert_eq!(tags.len(), 7);
    }
}
