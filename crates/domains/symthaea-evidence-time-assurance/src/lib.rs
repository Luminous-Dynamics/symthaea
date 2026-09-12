// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trusted clock and uncertainty-window assurance for safety-evidence readiness.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt,
};
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineReport, EvidenceQuarantineResolution,
    assess_quarantined_safety_case,
};
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceTimePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_clock_source: String,
    pub expected_clock_domain: String,
    pub expected_epoch_ref: String,
    pub maximum_clock_uncertainty_ms: u64,
    /// Maximum tolerated decrease in wall-clock time between accepted samples.
    /// Monotonic time must still strictly increase.
    pub maximum_wall_clock_backstep_ms: u64,
}

impl EvidenceTimePolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && !self.expected_clock_source.trim().is_empty()
            && !self.expected_clock_domain.trim().is_empty()
            && !self.expected_epoch_ref.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceTimeSample {
    pub sample_id: String,
    pub wall_time_ms: u64,
    pub monotonic_time_ms: u64,
    pub clock_source: String,
    pub clock_domain: String,
    pub epoch_ref: String,
    pub clock_uncertainty_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl EvidenceTimeSample {
    pub fn validate(&self) -> bool {
        !self.sample_id.trim().is_empty()
            && !self.clock_source.trim().is_empty()
            && !self.clock_domain.trim().is_empty()
            && !self.epoch_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceTimeStatus {
    Trusted,
    Untrusted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceTimeIssue {
    InvalidSample(String),
    DuplicateSampleId(String),
    ClockSourceMismatch,
    ClockDomainMismatch,
    EpochMismatch,
    ClockUncertaintyExceeded {
        observed_ms: u64,
        maximum_ms: u64,
    },
    MonotonicTimeDidNotAdvance {
        observed_ms: u64,
        previous_ms: u64,
    },
    WallClockBackstepExceeded {
        observed_ms: u64,
        previous_ms: u64,
        maximum_backstep_ms: u64,
    },
    TimeTrustAlreadyLatched,
}

/// Opaque trusted interval. External callers cannot construct one directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedEvidenceTime {
    nominal_ms: u64,
    earliest_ms: u64,
    latest_ms: u64,
    clock_source: String,
    clock_domain: String,
    epoch_ref: String,
}

impl TrustedEvidenceTime {
    pub const fn nominal_ms(&self) -> u64 {
        self.nominal_ms
    }

    pub const fn earliest_ms(&self) -> u64 {
        self.earliest_ms
    }

    pub const fn latest_ms(&self) -> u64 {
        self.latest_ms
    }

    pub fn clock_source(&self) -> &str {
        &self.clock_source
    }

    pub fn clock_domain(&self) -> &str {
        &self.clock_domain
    }

    pub fn epoch_ref(&self) -> &str {
        &self.epoch_ref
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceTimeReport {
    pub status: EvidenceTimeStatus,
    pub policy_id: String,
    pub issues: Vec<EvidenceTimeIssue>,
    pub trusted_time: Option<TrustedEvidenceTime>,
}

impl EvidenceTimeReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceTimeError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct EvidenceTimeGuard {
    policy: EvidenceTimePolicy,
    seen_sample_ids: BTreeSet<String>,
    last_wall_time_ms: Option<u64>,
    last_monotonic_time_ms: Option<u64>,
    latched_untrusted: bool,
}

impl EvidenceTimeGuard {
    pub fn new(policy: EvidenceTimePolicy) -> Result<Self, EvidenceTimeError> {
        if !policy.validate() {
            return Err(EvidenceTimeError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            seen_sample_ids: BTreeSet::new(),
            last_wall_time_ms: None,
            last_monotonic_time_ms: None,
            latched_untrusted: false,
        })
    }

    pub fn policy(&self) -> &EvidenceTimePolicy {
        &self.policy
    }

    /// Once a time-integrity failure is observed, the guard stays untrusted.
    /// A new guard/epoch must be established through the deployment's reviewed
    /// recovery procedure before safety readiness is trusted again.
    pub fn observe(&mut self, sample: EvidenceTimeSample) -> EvidenceTimeReport {
        let mut issues = Vec::new();

        if !sample.validate() {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::InvalidSample(sample.sample_id));
            return self.report(issues, None);
        }

        if !self.seen_sample_ids.insert(sample.sample_id.clone()) {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::DuplicateSampleId(sample.sample_id));
            return self.report(issues, None);
        }
        if self.latched_untrusted {
            issues.push(EvidenceTimeIssue::TimeTrustAlreadyLatched);
            return self.report(issues, None);
        }
        if sample.clock_source != self.policy.expected_clock_source {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::ClockSourceMismatch);
        }
        if sample.clock_domain != self.policy.expected_clock_domain {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::ClockDomainMismatch);
        }
        if sample.epoch_ref != self.policy.expected_epoch_ref {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::EpochMismatch);
        }
        if sample.clock_uncertainty_ms > self.policy.maximum_clock_uncertainty_ms {
            self.latched_untrusted = true;
            issues.push(EvidenceTimeIssue::ClockUncertaintyExceeded {
                observed_ms: sample.clock_uncertainty_ms,
                maximum_ms: self.policy.maximum_clock_uncertainty_ms,
            });
        }
        if let Some(previous) = self.last_monotonic_time_ms {
            if sample.monotonic_time_ms <= previous {
                self.latched_untrusted = true;
                issues.push(EvidenceTimeIssue::MonotonicTimeDidNotAdvance {
                    observed_ms: sample.monotonic_time_ms,
                    previous_ms: previous,
                });
            }
        }
        if let Some(previous) = self.last_wall_time_ms {
            if sample
                .wall_time_ms
                .saturating_add(self.policy.maximum_wall_clock_backstep_ms)
                < previous
            {
                self.latched_untrusted = true;
                issues.push(EvidenceTimeIssue::WallClockBackstepExceeded {
                    observed_ms: sample.wall_time_ms,
                    previous_ms: previous,
                    maximum_backstep_ms: self.policy.maximum_wall_clock_backstep_ms,
                });
            }
        }

        if self.latched_untrusted {
            return self.report(issues, None);
        }

        self.last_wall_time_ms = Some(sample.wall_time_ms);
        self.last_monotonic_time_ms = Some(sample.monotonic_time_ms);
        let trusted = TrustedEvidenceTime {
            nominal_ms: sample.wall_time_ms,
            earliest_ms: sample.wall_time_ms.saturating_sub(sample.clock_uncertainty_ms),
            latest_ms: sample.wall_time_ms.saturating_add(sample.clock_uncertainty_ms),
            clock_source: sample.clock_source,
            clock_domain: sample.clock_domain,
            epoch_ref: sample.epoch_ref,
        };
        self.report(issues, Some(trusted))
    }

    fn report(
        &self,
        issues: Vec<EvidenceTimeIssue>,
        trusted_time: Option<TrustedEvidenceTime>,
    ) -> EvidenceTimeReport {
        EvidenceTimeReport {
            status: if self.latched_untrusted || trusted_time.is_none() {
                EvidenceTimeStatus::Untrusted
            } else {
                EvidenceTimeStatus::Trusted
            },
            policy_id: self.policy.policy_id.clone(),
            issues,
            trusted_time,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeRobustSafetyReport {
    pub status: StrictSafetyCaseStatus,
    pub earliest_assessment: EvidenceQuarantineReport,
    pub latest_assessment: EvidenceQuarantineReport,
}

impl TimeRobustSafetyReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Require safety readiness to hold at both ends of the trusted clock interval.
///
/// This prevents a receipt from being treated as current when clock uncertainty
/// means it may actually be not-yet-valid or already expired. Lifecycle events
/// that straddle the uncertainty interval also fail closed through the underlying
/// lifecycle evaluator.
pub fn assess_with_trusted_time(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    receipts: &[DeploymentScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    directives: &[EvidenceQuarantineDirective],
    resolutions: &[EvidenceQuarantineResolution],
    trusted_time: &TrustedEvidenceTime,
) -> TimeRobustSafetyReport {
    let earliest_assessment = assess_quarantined_safety_case(
        safety_case,
        context,
        receipts,
        lifecycle_events,
        directives,
        resolutions,
        trusted_time.earliest_ms,
    );
    let latest_assessment = assess_quarantined_safety_case(
        safety_case,
        context,
        receipts,
        lifecycle_events,
        directives,
        resolutions,
        trusted_time.latest_ms,
    );

    let status = if earliest_assessment.status == StrictSafetyCaseStatus::Invalid
        || latest_assessment.status == StrictSafetyCaseStatus::Invalid
    {
        StrictSafetyCaseStatus::Invalid
    } else if earliest_assessment.status == StrictSafetyCaseStatus::Ready
        && latest_assessment.status == StrictSafetyCaseStatus::Ready
    {
        StrictSafetyCaseStatus::Ready
    } else {
        StrictSafetyCaseStatus::Blocked
    };

    TimeRobustSafetyReport {
        status,
        earliest_assessment,
        latest_assessment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_deployment_scope::bind_receipt_to_deployment;
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
    use symthaea_formal_safety::{
        EvidenceKind, ProofObligation, SafetyEvidenceReceipt,
    };

    fn policy() -> EvidenceTimePolicy {
        EvidenceTimePolicy {
            schema_version: "1".into(),
            policy_id: "trusted-time-v1".into(),
            expected_clock_source: "ptp-grandmaster-a".into(),
            expected_clock_domain: "domain-7".into(),
            expected_epoch_ref: "epoch:boot-42".into(),
            maximum_clock_uncertainty_ms: 100,
            maximum_wall_clock_backstep_ms: 5,
        }
    }

    fn sample(id: &str, wall: u64, monotonic: u64, uncertainty: u64) -> EvidenceTimeSample {
        EvidenceTimeSample {
            sample_id: id.into(),
            wall_time_ms: wall,
            monotonic_time_ms: monotonic,
            clock_source: "ptp-grandmaster-a".into(),
            clock_domain: "domain-7".into(),
            epoch_ref: "epoch:boot-42".into(),
            clock_uncertainty_ms: uncertainty,
            evidence_refs: vec![format!("clock-evidence:{id}")],
        }
    }

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("harbor-a");
        case.add_obligation(
            ProofObligation::new("claim", EvidenceKind::Test).discharge("legacy:test"),
        );
        case
    }

    fn context() -> DeploymentEvidenceContext {
        DeploymentEvidenceContext {
            schema_version: "1".into(),
            deployment_id: "node-1".into(),
            configuration_digest: "blake3:config-v1".into(),
            model_manifest_digest: None,
            calibration_manifest_digest: None,
            evidence_refs: vec!["deployment:node-1".into()],
        }
    }

    fn receipt(
        case: &SafetyCase,
        valid_from_ms: u64,
        valid_until_ms: u64,
    ) -> DeploymentScopedSafetyReceipt {
        let obligation = &case.obligations[0];
        let scoped = ScopedSafetyEvidenceReceipt {
            receipt: SafetyEvidenceReceipt {
                receipt_id: "r1".into(),
                obligation_key: obligation.stable_key(),
                evidence_kind: EvidenceKind::Test,
                evidence_ref: "artifact:r1".into(),
                evidence_digest: "blake3:r1".into(),
                verifier_ref: "verifier:safety".into(),
                verified_at_ms: 100,
            },
            contract_digest: case.contract_digest(),
            valid_from_ms,
            valid_until_ms,
            applicability_refs: vec!["deployment:node-1".into()],
        };
        bind_receipt_to_deployment(scoped, &context(), "scope:r1").unwrap()
    }

    #[test]
    fn trusted_sample_yields_uncertainty_interval() {
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        let report = guard.observe(sample("t1", 1_000, 10, 25));
        assert_eq!(report.status, EvidenceTimeStatus::Trusted);
        let time = report.trusted_time.unwrap();
        assert_eq!(time.earliest_ms(), 975);
        assert_eq!(time.latest_ms(), 1_025);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn excessive_uncertainty_latches_untrusted() {
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        let bad = guard.observe(sample("bad", 1_000, 10, 101));
        assert_eq!(bad.status, EvidenceTimeStatus::Untrusted);
        let later = guard.observe(sample("good", 2_000, 20, 10));
        assert_eq!(later.status, EvidenceTimeStatus::Untrusted);
        assert!(later
            .issues
            .contains(&EvidenceTimeIssue::TimeTrustAlreadyLatched));
    }

    #[test]
    fn monotonic_rollback_latches_untrusted() {
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        assert_eq!(
            guard.observe(sample("t1", 1_000, 10, 10)).status,
            EvidenceTimeStatus::Trusted
        );
        let report = guard.observe(sample("t2", 1_100, 10, 10));
        assert_eq!(report.status, EvidenceTimeStatus::Untrusted);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            EvidenceTimeIssue::MonotonicTimeDidNotAdvance { .. }
        )));
    }

    #[test]
    fn wall_clock_backstep_beyond_policy_latches_untrusted() {
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        guard.observe(sample("t1", 1_000, 10, 10));
        let report = guard.observe(sample("t2", 990, 20, 10));
        assert_eq!(report.status, EvidenceTimeStatus::Untrusted);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            EvidenceTimeIssue::WallClockBackstepExceeded { .. }
        )));
    }

    #[test]
    fn receipt_must_be_valid_across_entire_uncertainty_interval() {
        let case = case();
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        let report = guard.observe(sample("t1", 1_000, 10, 100));
        let trusted = report.trusted_time.unwrap();

        // At 900 ms this receipt is not yet valid; at 1100 ms it is valid.
        let readiness = assess_with_trusted_time(
            &case,
            &context(),
            &[receipt(&case, 950, 2_000)],
            &[],
            &[],
            &[],
            &trusted,
        );
        assert_eq!(readiness.earliest_assessment.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(readiness.latest_assessment.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(readiness.status, StrictSafetyCaseStatus::Blocked);
    }

    #[test]
    fn receipt_expiring_inside_uncertainty_interval_is_not_ready() {
        let case = case();
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        let trusted = guard
            .observe(sample("t1", 1_000, 10, 100))
            .trusted_time
            .unwrap();
        let readiness = assess_with_trusted_time(
            &case,
            &context(),
            &[receipt(&case, 100, 1_050)],
            &[],
            &[],
            &[],
            &trusted,
        );
        assert_eq!(readiness.earliest_assessment.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(readiness.latest_assessment.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(readiness.status, StrictSafetyCaseStatus::Blocked);
    }

    #[test]
    fn evidence_valid_across_full_interval_can_be_ready() {
        let case = case();
        let mut guard = EvidenceTimeGuard::new(policy()).unwrap();
        let trusted = guard
            .observe(sample("t1", 1_000, 10, 50))
            .trusted_time
            .unwrap();
        let readiness = assess_with_trusted_time(
            &case,
            &context(),
            &[receipt(&case, 100, 2_000)],
            &[],
            &[],
            &[],
            &trusted,
        );
        assert_eq!(readiness.status, StrictSafetyCaseStatus::Ready);
        assert!(!readiness.grants_physical_authority());
    }
}
