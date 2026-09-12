// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed, hysteretic recovery for degraded assurance states.
//!
//! A favorable sample is not enough to erase a prior restricted/unsafe/incomplete
//! condition. Recovery requires an explicit reviewed authorization plus sustained,
//! distinct nominal evidence over a deployment-reviewed time span.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceState {
    Nominal,
    Restricted,
    Incomplete,
    Unsafe,
    FailClosed,
}

impl AssuranceState {
    const fn restriction_rank(self) -> u8 {
        match self {
            Self::Nominal => 0,
            Self::Restricted => 1,
            Self::Incomplete => 2,
            Self::Unsafe => 3,
            Self::FailClosed => 4,
        }
    }

    const fn is_nominal(self) -> bool {
        matches!(self, Self::Nominal)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequalificationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    /// Must be at least 2: one favorable observation can never restore nominal.
    pub minimum_consecutive_nominal_samples: usize,
    /// Minimum elapsed time between first and latest accepted nominal sample.
    pub minimum_nominal_span_ms: u64,
    /// A larger gap resets recovery progress without clearing the latched restriction.
    pub maximum_nominal_sample_gap_ms: u64,
    /// Maximum age of a reviewed recovery authorization while collecting evidence.
    pub maximum_authorization_age_ms: u64,
}

impl RequalificationPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && self.minimum_consecutive_nominal_samples >= 2
            && self.minimum_nominal_span_ms > 0
            && self.maximum_nominal_sample_gap_ms > 0
            && self.maximum_authorization_age_ms >= self.minimum_nominal_span_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequalificationAuthorization {
    pub authorization_id: String,
    pub procedure_ref: String,
    pub reviewer_ref: String,
    pub authorized_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl RequalificationAuthorization {
    pub fn validate(&self) -> bool {
        !self.authorization_id.trim().is_empty()
            && !self.procedure_ref.trim().is_empty()
            && !self.reviewer_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceSample {
    pub sample_id: String,
    pub observed_at_ms: u64,
    pub state: AssuranceState,
    pub evidence_refs: Vec<String>,
}

impl AssuranceSample {
    pub fn validate(&self) -> bool {
        !self.sample_id.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequalificationPhase {
    Nominal,
    Latched,
    Requalifying,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequalificationIssue {
    InvalidSample(String),
    DuplicateSampleId(String),
    NonMonotonicObservationTime {
        sample_id: String,
        observed_at_ms: u64,
        previous_observed_at_ms: u64,
    },
    RequalificationNotAuthorized,
    AuthorizationExpired {
        authorization_id: String,
        age_ms: u64,
        maximum_age_ms: u64,
    },
    RecoveryGapExceeded {
        observed_gap_ms: u64,
        maximum_gap_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequalificationError {
    InvalidPolicy,
    InvalidAuthorization,
    NoRestrictionLatched,
    AuthorizationPredatesRestriction,
    AuthorizationPredatesLatestObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequalificationReport {
    pub policy_id: String,
    pub effective_state: AssuranceState,
    pub phase: RequalificationPhase,
    pub latched_since_ms: Option<u64>,
    pub authorization_id: Option<String>,
    pub consecutive_nominal_samples: usize,
    pub nominal_span_ms: u64,
    pub last_observed_at_ms: Option<u64>,
    pub issues: Vec<RequalificationIssue>,
}

impl RequalificationReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct RequalificationGate {
    policy: RequalificationPolicy,
    effective_state: AssuranceState,
    phase: RequalificationPhase,
    latched_since_ms: Option<u64>,
    authorization: Option<RequalificationAuthorization>,
    seen_sample_ids: BTreeSet<String>,
    last_observed_at_ms: Option<u64>,
    first_nominal_at_ms: Option<u64>,
    last_nominal_at_ms: Option<u64>,
    consecutive_nominal_samples: usize,
}

impl RequalificationGate {
    pub fn new(policy: RequalificationPolicy) -> Result<Self, RequalificationError> {
        if !policy.validate() {
            return Err(RequalificationError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            effective_state: AssuranceState::Nominal,
            phase: RequalificationPhase::Nominal,
            latched_since_ms: None,
            authorization: None,
            seen_sample_ids: BTreeSet::new(),
            last_observed_at_ms: None,
            first_nominal_at_ms: None,
            last_nominal_at_ms: None,
            consecutive_nominal_samples: 0,
        })
    }

    pub fn policy(&self) -> &RequalificationPolicy {
        &self.policy
    }

    pub fn effective_state(&self) -> AssuranceState {
        self.effective_state
    }

    pub fn begin_requalification(
        &mut self,
        authorization: RequalificationAuthorization,
    ) -> Result<RequalificationReport, RequalificationError> {
        if !authorization.validate() {
            return Err(RequalificationError::InvalidAuthorization);
        }
        let Some(latched_since_ms) = self.latched_since_ms else {
            return Err(RequalificationError::NoRestrictionLatched);
        };
        if authorization.authorized_at_ms < latched_since_ms {
            return Err(RequalificationError::AuthorizationPredatesRestriction);
        }
        if self
            .last_observed_at_ms
            .is_some_and(|latest| authorization.authorized_at_ms < latest)
        {
            return Err(RequalificationError::AuthorizationPredatesLatestObservation);
        }

        self.authorization = Some(authorization);
        self.phase = RequalificationPhase::Requalifying;
        self.reset_nominal_progress();
        Ok(self.report(Vec::new()))
    }

    pub fn observe(&mut self, sample: AssuranceSample) -> RequalificationReport {
        let mut issues = Vec::new();
        if !sample.validate() {
            issues.push(RequalificationIssue::InvalidSample(sample.sample_id));
            return self.report(issues);
        }
        if self.seen_sample_ids.contains(&sample.sample_id) {
            issues.push(RequalificationIssue::DuplicateSampleId(sample.sample_id));
            return self.report(issues);
        }

        // Tombstone a structurally valid sample identity before temporal admission.
        // A rejected non-monotonic sample id therefore cannot be replayed later with
        // modified timing to become favorable recovery evidence.
        self.seen_sample_ids.insert(sample.sample_id.clone());

        if let Some(previous) = self.last_observed_at_ms {
            if sample.observed_at_ms <= previous {
                issues.push(RequalificationIssue::NonMonotonicObservationTime {
                    sample_id: sample.sample_id,
                    observed_at_ms: sample.observed_at_ms,
                    previous_observed_at_ms: previous,
                });
                return self.report(issues);
            }
        }

        self.last_observed_at_ms = Some(sample.observed_at_ms);

        if !sample.state.is_nominal() {
            self.observe_restriction(sample.state, sample.observed_at_ms);
            return self.report(issues);
        }

        if self.effective_state.is_nominal() {
            self.phase = RequalificationPhase::Nominal;
            return self.report(issues);
        }

        let Some(authorization) = self.authorization.as_ref() else {
            issues.push(RequalificationIssue::RequalificationNotAuthorized);
            return self.report(issues);
        };

        let authorization_age_ms = sample
            .observed_at_ms
            .saturating_sub(authorization.authorized_at_ms);
        if authorization_age_ms > self.policy.maximum_authorization_age_ms {
            issues.push(RequalificationIssue::AuthorizationExpired {
                authorization_id: authorization.authorization_id.clone(),
                age_ms: authorization_age_ms,
                maximum_age_ms: self.policy.maximum_authorization_age_ms,
            });
            self.authorization = None;
            self.phase = RequalificationPhase::Latched;
            self.reset_nominal_progress();
            return self.report(issues);
        }

        if let Some(previous_nominal) = self.last_nominal_at_ms {
            let gap = sample.observed_at_ms.saturating_sub(previous_nominal);
            if gap > self.policy.maximum_nominal_sample_gap_ms {
                issues.push(RequalificationIssue::RecoveryGapExceeded {
                    observed_gap_ms: gap,
                    maximum_gap_ms: self.policy.maximum_nominal_sample_gap_ms,
                });
                self.first_nominal_at_ms = Some(sample.observed_at_ms);
                self.last_nominal_at_ms = Some(sample.observed_at_ms);
                self.consecutive_nominal_samples = 1;
                self.phase = RequalificationPhase::Requalifying;
                return self.report(issues);
            }
        }

        if self.consecutive_nominal_samples == 0 {
            self.first_nominal_at_ms = Some(sample.observed_at_ms);
        }
        self.last_nominal_at_ms = Some(sample.observed_at_ms);
        self.consecutive_nominal_samples = self.consecutive_nominal_samples.saturating_add(1);
        self.phase = RequalificationPhase::Requalifying;

        let span = self.nominal_span_ms();
        if self.consecutive_nominal_samples >= self.policy.minimum_consecutive_nominal_samples
            && span >= self.policy.minimum_nominal_span_ms
        {
            self.effective_state = AssuranceState::Nominal;
            self.phase = RequalificationPhase::Nominal;
            self.latched_since_ms = None;
            self.authorization = None;
            self.reset_nominal_progress();
        }

        self.report(issues)
    }

    fn observe_restriction(&mut self, observed: AssuranceState, observed_at_ms: u64) {
        if self.effective_state.is_nominal() {
            self.effective_state = observed;
            self.latched_since_ms = Some(observed_at_ms);
        } else if observed.restriction_rank() > self.effective_state.restriction_rank() {
            self.effective_state = observed;
        }
        self.phase = RequalificationPhase::Latched;
        self.authorization = None;
        self.reset_nominal_progress();
    }

    fn reset_nominal_progress(&mut self) {
        self.first_nominal_at_ms = None;
        self.last_nominal_at_ms = None;
        self.consecutive_nominal_samples = 0;
    }

    fn nominal_span_ms(&self) -> u64 {
        match (self.first_nominal_at_ms, self.last_nominal_at_ms) {
            (Some(first), Some(last)) => last.saturating_sub(first),
            _ => 0,
        }
    }

    fn report(&self, issues: Vec<RequalificationIssue>) -> RequalificationReport {
        RequalificationReport {
            policy_id: self.policy.policy_id.clone(),
            effective_state: self.effective_state,
            phase: self.phase,
            latched_since_ms: self.latched_since_ms,
            authorization_id: self
                .authorization
                .as_ref()
                .map(|authorization| authorization.authorization_id.clone()),
            consecutive_nominal_samples: self.consecutive_nominal_samples,
            nominal_span_ms: self.nominal_span_ms(),
            last_observed_at_ms: self.last_observed_at_ms,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RequalificationPolicy {
        RequalificationPolicy {
            schema_version: "1".into(),
            policy_id: "recovery-v1".into(),
            minimum_consecutive_nominal_samples: 3,
            minimum_nominal_span_ms: 2_000,
            maximum_nominal_sample_gap_ms: 1_500,
            maximum_authorization_age_ms: 10_000,
        }
    }

    fn sample(id: &str, at: u64, state: AssuranceState) -> AssuranceSample {
        AssuranceSample {
            sample_id: id.into(),
            observed_at_ms: at,
            state,
            evidence_refs: vec![format!("evidence:{id}")],
        }
    }

    fn authorization(at: u64) -> RequalificationAuthorization {
        RequalificationAuthorization {
            authorization_id: "reauth-1".into(),
            procedure_ref: "procedure:requalify-v1".into(),
            reviewer_ref: "reviewer:safety-process".into(),
            authorized_at_ms: at,
            evidence_refs: vec!["review:reauth-1".into()],
        }
    }

    #[test]
    fn one_good_sample_cannot_clear_restriction() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Unsafe));
        gate.begin_requalification(authorization(1_001)).unwrap();
        let report = gate.observe(sample("good-1", 2_000, AssuranceState::Nominal));
        assert_eq!(report.effective_state, AssuranceState::Unsafe);
        assert_eq!(report.phase, RequalificationPhase::Requalifying);
        assert_eq!(report.consecutive_nominal_samples, 1);
    }

    #[test]
    fn reviewed_sustained_evidence_can_restore_nominal() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Restricted));
        gate.begin_requalification(authorization(1_001)).unwrap();
        gate.observe(sample("good-1", 2_000, AssuranceState::Nominal));
        gate.observe(sample("good-2", 3_000, AssuranceState::Nominal));
        let report = gate.observe(sample("good-3", 4_000, AssuranceState::Nominal));
        assert_eq!(report.effective_state, AssuranceState::Nominal);
        assert_eq!(report.phase, RequalificationPhase::Nominal);
        assert_eq!(report.authorization_id, None);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn later_less_severe_fault_cannot_relax_latched_unsafe_state() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("unsafe", 1_000, AssuranceState::Unsafe));
        let report = gate.observe(sample("restricted", 2_000, AssuranceState::Restricted));
        assert_eq!(report.effective_state, AssuranceState::Unsafe);
    }

    #[test]
    fn more_severe_fault_worsens_latched_state() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("restricted", 1_000, AssuranceState::Restricted));
        let report = gate.observe(sample("unsafe", 2_000, AssuranceState::Unsafe));
        assert_eq!(report.effective_state, AssuranceState::Unsafe);
    }

    #[test]
    fn good_sample_without_explicit_authorization_does_not_begin_recovery() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Restricted));
        let report = gate.observe(sample("good", 2_000, AssuranceState::Nominal));
        assert_eq!(report.effective_state, AssuranceState::Restricted);
        assert_eq!(report.phase, RequalificationPhase::Latched);
        assert!(report
            .issues
            .contains(&RequalificationIssue::RequalificationNotAuthorized));
    }

    #[test]
    fn expired_authorization_cannot_advance_recovery() {
        let mut short = policy();
        short.maximum_authorization_age_ms = 2_000;
        let mut gate = RequalificationGate::new(short).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Unsafe));
        gate.begin_requalification(authorization(1_001)).unwrap();
        let report = gate.observe(sample("late-good", 4_000, AssuranceState::Nominal));
        assert_eq!(report.effective_state, AssuranceState::Unsafe);
        assert_eq!(report.phase, RequalificationPhase::Latched);
        assert_eq!(report.authorization_id, None);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RequalificationIssue::AuthorizationExpired { .. }
        )));
    }

    #[test]
    fn long_gap_resets_progress_without_clearing_restriction() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Restricted));
        gate.begin_requalification(authorization(1_001)).unwrap();
        gate.observe(sample("good-1", 2_000, AssuranceState::Nominal));
        let report = gate.observe(sample("good-2", 4_000, AssuranceState::Nominal));
        assert_eq!(report.effective_state, AssuranceState::Restricted);
        assert_eq!(report.consecutive_nominal_samples, 1);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            RequalificationIssue::RecoveryGapExceeded { .. }
        )));
    }

    #[test]
    fn fault_during_requalification_cancels_recovery() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Restricted));
        gate.begin_requalification(authorization(1_001)).unwrap();
        gate.observe(sample("good-1", 2_000, AssuranceState::Nominal));
        let report = gate.observe(sample("fault-2", 3_000, AssuranceState::Incomplete));
        assert_eq!(report.effective_state, AssuranceState::Incomplete);
        assert_eq!(report.phase, RequalificationPhase::Latched);
        assert_eq!(report.authorization_id, None);
        assert_eq!(report.consecutive_nominal_samples, 0);
    }

    #[test]
    fn duplicate_or_non_monotonic_samples_cannot_advance_recovery_and_ids_are_tombstoned() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 1_000, AssuranceState::Restricted));
        gate.begin_requalification(authorization(1_001)).unwrap();
        gate.observe(sample("good-1", 2_000, AssuranceState::Nominal));

        let duplicate = gate.observe(sample("good-1", 2_500, AssuranceState::Nominal));
        assert_eq!(duplicate.consecutive_nominal_samples, 1);
        assert!(duplicate
            .issues
            .contains(&RequalificationIssue::DuplicateSampleId("good-1".into())));

        let old = gate.observe(sample("old", 1_500, AssuranceState::Nominal));
        assert_eq!(old.consecutive_nominal_samples, 1);
        assert!(old.issues.iter().any(|issue| matches!(
            issue,
            RequalificationIssue::NonMonotonicObservationTime { .. }
        )));

        let replayed_with_friendlier_time = gate.observe(sample("old", 2_500, AssuranceState::Nominal));
        assert_eq!(replayed_with_friendlier_time.consecutive_nominal_samples, 1);
        assert!(replayed_with_friendlier_time
            .issues
            .contains(&RequalificationIssue::DuplicateSampleId("old".into())));
    }

    #[test]
    fn impossible_authorization_window_is_invalid_policy() {
        let mut invalid = policy();
        invalid.maximum_authorization_age_ms = invalid.minimum_nominal_span_ms - 1;
        assert_eq!(
            RequalificationGate::new(invalid),
            Err(RequalificationError::InvalidPolicy)
        );
    }

    #[test]
    fn authorization_cannot_predate_latched_failure_or_latest_observation() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample("fault", 2_000, AssuranceState::Unsafe));
        assert_eq!(
            gate.begin_requalification(authorization(1_999)),
            Err(RequalificationError::AuthorizationPredatesRestriction)
        );
        gate.observe(sample("later-fault", 3_000, AssuranceState::Unsafe));
        assert_eq!(
            gate.begin_requalification(authorization(2_500)),
            Err(RequalificationError::AuthorizationPredatesLatestObservation)
        );
    }
}
