// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative currentness assessment scoped to observed transparency views.
//!
//! This module intentionally does not mint `CurrentlyValidQualifiedScientificClaim`.
//! It answers a narrower question: does the latest locally accepted lifecycle
//! event agree with an exact publicly anchored lifecycle event, and are the
//! qualification/lifecycle publication views append-only-consistent within the
//! supplied witnessed monitor receipt? Even a positive result does not establish
//! that no newer unseen lifecycle event exists or that underlying evidence is fresh.

use serde::Serialize;
use symthaea_trust_core::{
    FramedDigest, Sha256Digest as TrustSha256Digest, TransparencyMonitorClosure,
    TransparencyMonitorReceipt,
};

use crate::{
    PubliclyAnchoredQualificationLifecycleEvent, PubliclyAnchoredScientificQualification,
    QualificationLifecycleState, QualificationLifecycleTracker, Sha256Digest,
};

const OBSERVED_CURRENTNESS_DOMAIN: &str =
    "symthaea.scientific-qualification-observed-currentness.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ObservedQualificationCurrentnessClosure {
    ActiveWithinObservedViews,
    NotActiveWithinObservedViews,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ObservedQualificationCurrentnessFinding {
    QualificationMismatch,
    RootAuthorityMismatch,
    TrackerQualificationMismatch,
    TrackerRootAuthorityMismatch,
    TrackerLatestEventMismatch,
    TrackerLatestSequenceMismatch,
    QualificationViewMissingFromMonitor,
    LifecycleViewMissingFromMonitor,
    MonitorRootAuthorityMismatch,
    MonitorIncomparableAuthority,
    MonitorIncomplete,
    MonitorEquivocation,
    MonitorTemporalConflict,
    MonitorInvalid,
    LifecycleNotActive { state: QualificationLifecycleState },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ObservedQualificationCurrentnessAssessment {
    qualification_sha256: Sha256Digest,
    root_authority_sha256: TrustSha256Digest,
    qualification_publication_sha256: TrustSha256Digest,
    lifecycle_publication_sha256: TrustSha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    lifecycle_sequence: u64,
    observed_state: Option<QualificationLifecycleState>,
    transparency_monitor_receipt_sha256: TrustSha256Digest,
    findings: Vec<ObservedQualificationCurrentnessFinding>,
    closure: ObservedQualificationCurrentnessClosure,
    assessment_sha256: TrustSha256Digest,
}

impl ObservedQualificationCurrentnessAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn lifecycle_sequence(&self) -> u64 { self.lifecycle_sequence }
    pub fn observed_state(&self) -> Option<QualificationLifecycleState> { self.observed_state }
    pub fn findings(&self) -> &[ObservedQualificationCurrentnessFinding] { &self.findings }
    pub fn closure(&self) -> ObservedQualificationCurrentnessClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }

    pub fn active_within_observed_views_established(&self) -> bool {
        self.closure == ObservedQualificationCurrentnessClosure::ActiveWithinObservedViews
    }
    pub const fn globally_latest_lifecycle_established(&self) -> bool { false }
    pub const fn evidence_freshness_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn assess_observed_qualification_currentness(
    qualification: &PubliclyAnchoredScientificQualification,
    lifecycle: &PubliclyAnchoredQualificationLifecycleEvent,
    tracker: &QualificationLifecycleTracker,
    monitor: &TransparencyMonitorReceipt,
) -> ObservedQualificationCurrentnessAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut integrity_blocked = false;
    let mut incomplete = false;

    if lifecycle.qualification_sha256() != qualification.qualification_sha256() {
        findings.push(ObservedQualificationCurrentnessFinding::QualificationMismatch);
        invalid = true;
    }
    if lifecycle.root_authority_sha256() != qualification.root_authority_sha256() {
        findings.push(ObservedQualificationCurrentnessFinding::RootAuthorityMismatch);
        invalid = true;
    }
    if tracker.qualification_sha256() != Some(qualification.qualification_sha256()) {
        findings.push(ObservedQualificationCurrentnessFinding::TrackerQualificationMismatch);
        invalid = true;
    }
    if tracker.root_authority_sha256() != Some(qualification.root_authority_sha256()) {
        findings.push(ObservedQualificationCurrentnessFinding::TrackerRootAuthorityMismatch);
        invalid = true;
    }
    if tracker.latest_event_sha256() != Some(lifecycle.lifecycle_event_sha256()) {
        findings.push(ObservedQualificationCurrentnessFinding::TrackerLatestEventMismatch);
        invalid = true;
    }
    if tracker.latest_sequence() != Some(lifecycle.lifecycle_sequence()) {
        findings.push(ObservedQualificationCurrentnessFinding::TrackerLatestSequenceMismatch);
        invalid = true;
    }

    if !monitor
        .view_sha256s()
        .contains(qualification.witnessed_view_sha256())
    {
        findings.push(ObservedQualificationCurrentnessFinding::QualificationViewMissingFromMonitor);
        incomplete = true;
    }
    if !monitor
        .view_sha256s()
        .contains(lifecycle.witnessed_view_sha256())
    {
        findings.push(ObservedQualificationCurrentnessFinding::LifecycleViewMissingFromMonitor);
        incomplete = true;
    }
    if monitor.root_authority_sha256s().len() != 1
        || monitor.root_authority_sha256s().first()
            != Some(qualification.root_authority_sha256())
    {
        findings.push(ObservedQualificationCurrentnessFinding::MonitorRootAuthorityMismatch);
        invalid = true;
    }

    match monitor.closure() {
        TransparencyMonitorClosure::AppendOnlyConsistentWithinObservedViews => {}
        TransparencyMonitorClosure::EquivocationObserved => {
            findings.push(ObservedQualificationCurrentnessFinding::MonitorEquivocation);
            integrity_blocked = true;
        }
        TransparencyMonitorClosure::TemporalConflictObserved => {
            findings.push(ObservedQualificationCurrentnessFinding::MonitorTemporalConflict);
            integrity_blocked = true;
        }
        TransparencyMonitorClosure::Incomplete => {
            findings.push(ObservedQualificationCurrentnessFinding::MonitorIncomplete);
            incomplete = true;
        }
        TransparencyMonitorClosure::IncomparableAuthority => {
            findings.push(ObservedQualificationCurrentnessFinding::MonitorIncomparableAuthority);
            incomplete = true;
        }
        TransparencyMonitorClosure::Invalid => {
            findings.push(ObservedQualificationCurrentnessFinding::MonitorInvalid);
            invalid = true;
        }
    }

    let observed_state = tracker.latest_state();
    let lifecycle_not_active = observed_state.is_some_and(|state| state != QualificationLifecycleState::Active);
    if let Some(state) = observed_state {
        if state != QualificationLifecycleState::Active {
            findings.push(ObservedQualificationCurrentnessFinding::LifecycleNotActive { state });
        }
    } else {
        incomplete = true;
    }

    findings.sort_by_key(finding_sort_tag);
    let closure = if invalid {
        ObservedQualificationCurrentnessClosure::Invalid
    } else if integrity_blocked {
        ObservedQualificationCurrentnessClosure::Blocked
    } else if lifecycle_not_active {
        ObservedQualificationCurrentnessClosure::NotActiveWithinObservedViews
    } else if incomplete {
        ObservedQualificationCurrentnessClosure::Incomplete
    } else {
        ObservedQualificationCurrentnessClosure::ActiveWithinObservedViews
    };

    let assessment_sha256 = assessment_digest(
        qualification,
        lifecycle,
        tracker,
        monitor,
        observed_state,
        &findings,
        closure,
    );
    ObservedQualificationCurrentnessAssessment {
        qualification_sha256: qualification.qualification_sha256().clone(),
        root_authority_sha256: qualification.root_authority_sha256().clone(),
        qualification_publication_sha256: qualification.publication_sha256().clone(),
        lifecycle_publication_sha256: lifecycle.publication_sha256().clone(),
        lifecycle_event_sha256: lifecycle.lifecycle_event_sha256().clone(),
        lifecycle_sequence: lifecycle.lifecycle_sequence(),
        observed_state,
        transparency_monitor_receipt_sha256: monitor.receipt_sha256().clone(),
        findings,
        closure,
        assessment_sha256,
    }
}

#[allow(clippy::too_many_arguments)]
fn assessment_digest(
    qualification: &PubliclyAnchoredScientificQualification,
    lifecycle: &PubliclyAnchoredQualificationLifecycleEvent,
    tracker: &QualificationLifecycleTracker,
    monitor: &TransparencyMonitorReceipt,
    state: Option<QualificationLifecycleState>,
    findings: &[ObservedQualificationCurrentnessFinding],
    closure: ObservedQualificationCurrentnessClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(OBSERVED_CURRENTNESS_DOMAIN);
    digest.text(qualification.qualification_sha256().as_str());
    digest.text(qualification.publication_sha256().as_str());
    digest.text(lifecycle.publication_sha256().as_str());
    digest.text(lifecycle.lifecycle_event_sha256().as_str());
    digest.text(&lifecycle.lifecycle_sequence().to_string());
    digest.optional_sha(tracker.latest_event_sha256().map(bridge_digest).as_ref());
    digest.text(monitor.receipt_sha256().as_str());
    digest.text(match state {
        Some(QualificationLifecycleState::Active) => "active",
        Some(QualificationLifecycleState::UnderReview) => "under-review",
        Some(QualificationLifecycleState::Suspended) => "suspended",
        Some(QualificationLifecycleState::Revoked) => "revoked",
        Some(QualificationLifecycleState::Superseded) => "superseded",
        None => "unknown",
    });
    for finding in findings {
        digest_finding(&mut digest, finding);
    }
    digest.text(match closure {
        ObservedQualificationCurrentnessClosure::ActiveWithinObservedViews => "active-observed",
        ObservedQualificationCurrentnessClosure::NotActiveWithinObservedViews => "not-active-observed",
        ObservedQualificationCurrentnessClosure::Incomplete => "incomplete",
        ObservedQualificationCurrentnessClosure::Blocked => "blocked",
        ObservedQualificationCurrentnessClosure::Invalid => "invalid",
    });
    digest.text("globally-latest-lifecycle-not-established");
    digest.text("evidence-freshness-not-established");
    digest.text("current-validity-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn finding_sort_tag(finding: &ObservedQualificationCurrentnessFinding) -> &'static str {
    match finding {
        ObservedQualificationCurrentnessFinding::QualificationMismatch => "01-qualification-mismatch",
        ObservedQualificationCurrentnessFinding::RootAuthorityMismatch => "02-root-authority-mismatch",
        ObservedQualificationCurrentnessFinding::TrackerQualificationMismatch => "03-tracker-qualification-mismatch",
        ObservedQualificationCurrentnessFinding::TrackerRootAuthorityMismatch => "04-tracker-root-authority-mismatch",
        ObservedQualificationCurrentnessFinding::TrackerLatestEventMismatch => "05-tracker-latest-event-mismatch",
        ObservedQualificationCurrentnessFinding::TrackerLatestSequenceMismatch => "06-tracker-latest-sequence-mismatch",
        ObservedQualificationCurrentnessFinding::QualificationViewMissingFromMonitor => "07-qualification-view-missing",
        ObservedQualificationCurrentnessFinding::LifecycleViewMissingFromMonitor => "08-lifecycle-view-missing",
        ObservedQualificationCurrentnessFinding::MonitorRootAuthorityMismatch => "09-monitor-root-authority-mismatch",
        ObservedQualificationCurrentnessFinding::MonitorIncomparableAuthority => "10-monitor-incomparable-authority",
        ObservedQualificationCurrentnessFinding::MonitorIncomplete => "11-monitor-incomplete",
        ObservedQualificationCurrentnessFinding::MonitorEquivocation => "12-monitor-equivocation",
        ObservedQualificationCurrentnessFinding::MonitorTemporalConflict => "13-monitor-temporal-conflict",
        ObservedQualificationCurrentnessFinding::MonitorInvalid => "14-monitor-invalid",
        ObservedQualificationCurrentnessFinding::LifecycleNotActive { .. } => "15-lifecycle-not-active",
    }
}

fn digest_finding(digest: &mut FramedDigest, finding: &ObservedQualificationCurrentnessFinding) {
    match finding {
        ObservedQualificationCurrentnessFinding::QualificationMismatch => digest.text("qualification-mismatch"),
        ObservedQualificationCurrentnessFinding::RootAuthorityMismatch => digest.text("root-authority-mismatch"),
        ObservedQualificationCurrentnessFinding::TrackerQualificationMismatch => digest.text("tracker-qualification-mismatch"),
        ObservedQualificationCurrentnessFinding::TrackerRootAuthorityMismatch => digest.text("tracker-root-authority-mismatch"),
        ObservedQualificationCurrentnessFinding::TrackerLatestEventMismatch => digest.text("tracker-latest-event-mismatch"),
        ObservedQualificationCurrentnessFinding::TrackerLatestSequenceMismatch => digest.text("tracker-latest-sequence-mismatch"),
        ObservedQualificationCurrentnessFinding::QualificationViewMissingFromMonitor => digest.text("qualification-view-missing"),
        ObservedQualificationCurrentnessFinding::LifecycleViewMissingFromMonitor => digest.text("lifecycle-view-missing"),
        ObservedQualificationCurrentnessFinding::MonitorRootAuthorityMismatch => digest.text("monitor-root-authority-mismatch"),
        ObservedQualificationCurrentnessFinding::MonitorIncomparableAuthority => digest.text("monitor-incomparable-authority"),
        ObservedQualificationCurrentnessFinding::MonitorIncomplete => digest.text("monitor-incomplete"),
        ObservedQualificationCurrentnessFinding::MonitorEquivocation => digest.text("monitor-equivocation"),
        ObservedQualificationCurrentnessFinding::MonitorTemporalConflict => digest.text("monitor-temporal-conflict"),
        ObservedQualificationCurrentnessFinding::MonitorInvalid => digest.text("monitor-invalid"),
        ObservedQualificationCurrentnessFinding::LifecycleNotActive { state } => {
            digest.text("lifecycle-not-active");
            digest.text(match state {
                QualificationLifecycleState::Active => "active",
                QualificationLifecycleState::UnderReview => "under-review",
                QualificationLifecycleState::Suspended => "suspended",
                QualificationLifecycleState::Revoked => "revoked",
                QualificationLifecycleState::Superseded => "superseded",
            });
        }
    }
}
