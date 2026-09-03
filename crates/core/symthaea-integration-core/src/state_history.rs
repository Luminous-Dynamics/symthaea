// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded state history and conservative persistence evidence.
//!
//! A sequence of snapshots cannot reveal the exact time a desired value changed
//! before the first observation. It can establish spans across an uninterrupted
//! *sample sequence*, but it cannot prove that no convergence/regression occurred
//! between samples. Sampled history therefore supports a persistence hypothesis;
//! it does not prove continuous persistence by itself.

use crate::{
    EntityRef, StateAssessmentStatus, StateLimits, StateSnapshot, StateValue,
    TemporalStateAssessment, TemporalStateAssessmentError, TemporalStatePolicy,
    TemporalStateStatus, assess_state_dimension, assess_state_dimension_temporally,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateHistory {
    pub integration_id: String,
    /// Strictly increasing capture times are required. Reordered history is
    /// rejected rather than silently sorted because source ordering is evidence.
    pub snapshots: Vec<StateSnapshot>,
}

impl StateHistory {
    pub fn validate(&self) -> Result<(), StateHistoryError> {
        self.validate_with_limits(&StateHistoryLimits::default())
    }

    pub fn validate_with_limits(&self, limits: &StateHistoryLimits) -> Result<(), StateHistoryError> {
        if self.integration_id.trim().is_empty() {
            return Err(StateHistoryError::EmptyIntegrationId);
        }
        if self.snapshots.len() > limits.max_snapshots {
            return Err(StateHistoryError::TooManySnapshots {
                actual: self.snapshots.len(),
                max: limits.max_snapshots,
            });
        }

        let mut previous = None;
        let mut total_assertions = 0usize;
        for snapshot in &self.snapshots {
            if snapshot.integration_id != self.integration_id {
                return Err(StateHistoryError::SourceMismatch {
                    expected: self.integration_id.clone(),
                    actual: snapshot.integration_id.clone(),
                });
            }
            snapshot
                .validate_with_limits(&limits.per_snapshot_limits)
                .map_err(|error| StateHistoryError::InvalidSnapshot(error.to_string()))?;
            if let Some(previous) = previous {
                if snapshot.collected_at_unix_ms <= previous {
                    return Err(StateHistoryError::NonMonotonicSnapshotTime {
                        previous,
                        current: snapshot.collected_at_unix_ms,
                    });
                }
            }
            previous = Some(snapshot.collected_at_unix_ms);
            total_assertions = total_assertions
                .checked_add(snapshot.assertions.len())
                .ok_or(StateHistoryError::ArithmeticOverflow)?;
            if total_assertions > limits.max_total_assertions {
                return Err(StateHistoryError::TooManyTotalAssertions {
                    actual: total_assertions,
                    max: limits.max_total_assertions,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateHistoryLimits {
    pub max_snapshots: usize,
    pub max_total_assertions: usize,
    pub per_snapshot_limits: StateLimits,
}

impl Default for StateHistoryLimits {
    fn default() -> Self {
        Self {
            max_snapshots: 4_096,
            max_total_assertions: 250_000,
            per_snapshot_limits: StateLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesiredStateContinuity {
    pub value: StateValue,
    pub first_seen_at_unix_ms: u64,
    pub last_seen_at_unix_ms: u64,
    pub consecutive_snapshots: usize,
}

/// Uninterrupted *sample sequence* in which the current desired value remained
/// in disagreement with observed state. This does not prove that the system did
/// not converge and regress between samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftContinuity {
    pub desired_value: StateValue,
    pub first_seen_at_unix_ms: u64,
    pub last_seen_at_unix_ms: u64,
    pub consecutive_snapshots: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalStateAssessment {
    pub current: TemporalStateAssessment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desired_continuity: Option<DesiredStateContinuity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_continuity: Option<DriftContinuity>,
    /// Span from the first through last samples in the uninterrupted sequence
    /// that carry the current desired value. The field name is retained for the
    /// v0.1 wire shape; it is a sampled span, not proof of continuous state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuously_observed_desired_age_lower_bound_ms: Option<u64>,
    /// Span from the first through last samples in the uninterrupted sequence
    /// whose sampled states are all in drift for the same desired value. This is
    /// supporting evidence only; unsampled convergence may still have occurred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuously_observed_drift_age_lower_bound_ms: Option<u64>,
    /// True when the sampled drift span exceeds the convergence window while the
    /// current drift age is otherwise unknown. This supports a persistence
    /// hypothesis but is deliberately weaker than `persistent_drift_proven`.
    pub sampled_persistence_supported: bool,
    /// True only when current temporal evidence itself proves the desired value
    /// became effective before the convergence window and drift exists now.
    /// Sampled history alone never sets this flag.
    pub persistent_drift_proven: bool,
}

pub fn assess_state_dimension_with_history(
    history: &StateHistory,
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
    policy: TemporalStatePolicy,
) -> Result<HistoricalStateAssessment, StateHistoryError> {
    history.validate()?;
    let latest = history
        .snapshots
        .iter()
        .rev()
        .find(|snapshot| snapshot.collected_at_unix_ms <= at_unix_ms)
        .ok_or(StateHistoryError::NoSnapshotAtOrBefore(at_unix_ms))?;

    let current = assess_state_dimension_temporally(
        &latest.assertions,
        subject,
        dimension,
        at_unix_ms,
        policy,
    )
    .map_err(StateHistoryError::TemporalAssessment)?;

    let desired_continuity = desired_state_continuity(history, subject, dimension, at_unix_ms)?;
    let drift_continuity = drift_state_continuity(history, subject, dimension, at_unix_ms)?;
    // A sampled span is supported only by the interval between first and last
    // supporting snapshots. Querying the same history later cannot age evidence.
    let desired_lower_bound = desired_continuity.as_ref().map(|continuity| {
        continuity.last_seen_at_unix_ms - continuity.first_seen_at_unix_ms
    });
    let drift_lower_bound = drift_continuity.as_ref().map(|continuity| {
        continuity.last_seen_at_unix_ms - continuity.first_seen_at_unix_ms
    });
    let sampled_persistence_supported = current.status == TemporalStateStatus::DriftAgeUnknown
        && drift_lower_bound.is_some_and(|age| age > policy.convergence_window_ms);
    let persistent_drift_proven = current.status == TemporalStateStatus::PersistentDrift;

    Ok(HistoricalStateAssessment {
        current,
        desired_continuity,
        drift_continuity,
        continuously_observed_desired_age_lower_bound_ms: desired_lower_bound,
        continuously_observed_drift_age_lower_bound_ms: drift_lower_bound,
        sampled_persistence_supported,
        persistent_drift_proven,
    })
}

/// Find the uninterrupted sampled history of the current desired value.
///
/// Continuity is deliberately exact at the serialized `StateValue` level even
/// if an instantaneous assessment uses a numeric tolerance. This is conservative:
/// a moving desired numeric target must not be treated as one unchanged epoch
/// merely because adjacent values fall within a tolerance band.
pub fn desired_state_continuity(
    history: &StateHistory,
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
) -> Result<Option<DesiredStateContinuity>, StateHistoryError> {
    history.validate()?;
    let eligible = eligible_snapshots(history, at_unix_ms);
    let Some(latest) = eligible.last() else {
        return Ok(None);
    };
    let latest_assessment = exact_assessment(latest, subject, dimension)?;
    if desired_is_unusable(latest_assessment.status) {
        return Ok(None);
    }
    let Some(current_value) = latest_assessment.desired_value else {
        return Ok(None);
    };

    let mut first_seen = latest.collected_at_unix_ms;
    let mut count = 0usize;
    for snapshot in eligible.iter().rev() {
        let assessment = exact_assessment(snapshot, subject, dimension)?;
        if desired_is_unusable(assessment.status)
            || assessment.desired_value.as_ref() != Some(&current_value)
        {
            break;
        }
        first_seen = snapshot.collected_at_unix_ms;
        count += 1;
    }

    Ok(Some(DesiredStateContinuity {
        value: current_value,
        first_seen_at_unix_ms: first_seen,
        last_seen_at_unix_ms: latest.collected_at_unix_ms,
        consecutive_snapshots: count,
    }))
}

/// Find the uninterrupted sample sequence in which the current desired value
/// and each sampled observed state are in drift. Any sampled in-sync, missing,
/// conflicting, or changed-desired state breaks the sequence.
pub fn drift_state_continuity(
    history: &StateHistory,
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
) -> Result<Option<DriftContinuity>, StateHistoryError> {
    history.validate()?;
    let eligible = eligible_snapshots(history, at_unix_ms);
    let Some(latest) = eligible.last() else {
        return Ok(None);
    };
    let latest_assessment = exact_assessment(latest, subject, dimension)?;
    if latest_assessment.status != StateAssessmentStatus::Drift {
        return Ok(None);
    }
    let Some(current_desired) = latest_assessment.desired_value else {
        return Ok(None);
    };

    let mut first_seen = latest.collected_at_unix_ms;
    let mut count = 0usize;
    for snapshot in eligible.iter().rev() {
        let assessment = exact_assessment(snapshot, subject, dimension)?;
        if assessment.status != StateAssessmentStatus::Drift
            || assessment.desired_value.as_ref() != Some(&current_desired)
        {
            break;
        }
        first_seen = snapshot.collected_at_unix_ms;
        count += 1;
    }

    Ok(Some(DriftContinuity {
        desired_value: current_desired,
        first_seen_at_unix_ms: first_seen,
        last_seen_at_unix_ms: latest.collected_at_unix_ms,
        consecutive_snapshots: count,
    }))
}

fn eligible_snapshots(history: &StateHistory, at_unix_ms: u64) -> Vec<&StateSnapshot> {
    history
        .snapshots
        .iter()
        .filter(|snapshot| snapshot.collected_at_unix_ms <= at_unix_ms)
        .collect()
}

fn exact_assessment(
    snapshot: &StateSnapshot,
    subject: &EntityRef,
    dimension: &str,
) -> Result<crate::StateAssessment, StateHistoryError> {
    assess_state_dimension(
        &snapshot.assertions,
        subject,
        dimension,
        snapshot.collected_at_unix_ms,
        crate::StateComparisonPolicy::Exact,
    )
    .map_err(|error| StateHistoryError::InstantaneousAssessment(error.to_string()))
}

fn desired_is_unusable(status: StateAssessmentStatus) -> bool {
    matches!(
        status,
        StateAssessmentStatus::NoEvidence
            | StateAssessmentStatus::MissingDesired
            | StateAssessmentStatus::ConflictingDesired
            | StateAssessmentStatus::ConflictingBoth
    )
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateHistoryError {
    #[error("state history integration id is empty")]
    EmptyIntegrationId,
    #[error("state history has {actual} snapshots; maximum is {max}")]
    TooManySnapshots { actual: usize, max: usize },
    #[error("state history has {actual} total assertions; maximum is {max}")]
    TooManyTotalAssertions { actual: usize, max: usize },
    #[error("state history source mismatch: expected `{expected}`, got `{actual}`")]
    SourceMismatch { expected: String, actual: String },
    #[error("invalid state snapshot in history: {0}")]
    InvalidSnapshot(String),
    #[error("state history times are not strictly increasing: {previous} then {current}")]
    NonMonotonicSnapshotTime { previous: u64, current: u64 },
    #[error("state history size arithmetic overflow")]
    ArithmeticOverflow,
    #[error("no state snapshot exists at or before {0}")]
    NoSnapshotAtOrBefore(u64),
    #[error("state history instantaneous assessment failed: {0}")]
    InstantaneousAssessment(String),
    #[error("state history temporal assessment failed: {0}")]
    TemporalAssessment(#[from] TemporalStateAssessmentError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StateAssertion, StateAssertionSource, StateRole};
    use std::collections::BTreeMap;

    fn subject() -> EntityRef {
        EntityRef::new("k8s:fixture", "k8s_deployment", "deployment-uid")
    }

    fn assertion(id: &str, role: StateRole, value: u64, at: u64) -> StateAssertion {
        StateAssertion {
            assertion_id: id.into(),
            subject: subject(),
            dimension: "workload.replicas".into(),
            role,
            value: StateValue::Unsigned(value),
            source_confidence: 1.0,
            source: StateAssertionSource {
                integration_id: "fixture".into(),
                collector_id: None,
                tenant: None,
            },
            observed_at_unix_ms: at,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
            attributes: BTreeMap::new(),
        }
    }

    fn snapshot(at: u64, desired: Option<u64>, observed: Option<u64>) -> StateSnapshot {
        let mut assertions = Vec::new();
        if let Some(value) = desired {
            assertions.push(assertion(&format!("desired-{at}"), StateRole::Desired, value, at));
        }
        if let Some(value) = observed {
            assertions.push(assertion(&format!("observed-{at}"), StateRole::Observed, value, at));
        }
        StateSnapshot {
            integration_id: "fixture".into(),
            collected_at_unix_ms: at,
            assertions,
        }
    }

    fn history(snapshots: Vec<StateSnapshot>) -> StateHistory {
        StateHistory {
            integration_id: "fixture".into(),
            snapshots,
        }
    }

    fn policy() -> TemporalStatePolicy {
        TemporalStatePolicy {
            comparison: crate::StateComparisonPolicy::Exact,
            max_desired_age_ms: Some(1_000),
            max_observed_age_ms: Some(1_000),
            convergence_window_ms: 150,
        }
    }

    #[test]
    fn continuous_sampled_drift_supports_but_does_not_prove_persistence() {
        let history = history(vec![
            snapshot(100, Some(5), Some(3)),
            snapshot(200, Some(5), Some(3)),
            snapshot(300, Some(5), Some(3)),
        ]);
        let result = assess_state_dimension_with_history(
            &history,
            &subject(),
            "workload.replicas",
            300,
            policy(),
        )
        .unwrap();
        assert_eq!(result.current.status, TemporalStateStatus::DriftAgeUnknown);
        assert_eq!(
            result.continuously_observed_desired_age_lower_bound_ms,
            Some(200)
        );
        assert_eq!(
            result.continuously_observed_drift_age_lower_bound_ms,
            Some(200)
        );
        assert!(result.sampled_persistence_supported);
        assert!(!result.persistent_drift_proven);
    }

    #[test]
    fn intervening_convergence_breaks_sampled_persistence_support() {
        let history = history(vec![
            snapshot(100, Some(5), Some(3)),
            snapshot(200, Some(5), Some(5)),
            snapshot(300, Some(5), Some(3)),
        ]);
        let result = assess_state_dimension_with_history(
            &history,
            &subject(),
            "workload.replicas",
            300,
            policy(),
        )
        .unwrap();
        assert_eq!(
            result.continuously_observed_desired_age_lower_bound_ms,
            Some(200)
        );
        assert_eq!(
            result.continuously_observed_drift_age_lower_bound_ms,
            Some(0)
        );
        assert!(!result.sampled_persistence_supported);
        assert!(!result.persistent_drift_proven);
        assert_eq!(result.drift_continuity.as_ref().unwrap().consecutive_snapshots, 1);
    }

    #[test]
    fn query_time_does_not_age_sampled_continuity() {
        let history = history(vec![
            snapshot(100, Some(4), Some(3)),
            snapshot(300, Some(5), Some(3)),
        ]);
        let result = assess_state_dimension_with_history(
            &history,
            &subject(),
            "workload.replicas",
            350,
            policy(),
        )
        .unwrap();
        assert_eq!(result.current.status, TemporalStateStatus::DriftAgeUnknown);
        assert_eq!(
            result.continuously_observed_drift_age_lower_bound_ms,
            Some(0)
        );
        assert!(!result.sampled_persistence_supported);
        assert!(!result.persistent_drift_proven);
    }

    #[test]
    fn query_time_cannot_inflate_persistence_without_freshness_limits() {
        let history = history(vec![
            snapshot(100, Some(5), Some(3)),
            snapshot(200, Some(5), Some(3)),
        ]);
        let result = assess_state_dimension_with_history(
            &history,
            &subject(),
            "workload.replicas",
            1_000,
            TemporalStatePolicy {
                comparison: crate::StateComparisonPolicy::Exact,
                max_desired_age_ms: None,
                max_observed_age_ms: None,
                convergence_window_ms: 150,
            },
        )
        .unwrap();
        assert_eq!(result.current.status, TemporalStateStatus::DriftAgeUnknown);
        assert_eq!(
            result.continuously_observed_desired_age_lower_bound_ms,
            Some(100)
        );
        assert_eq!(
            result.continuously_observed_drift_age_lower_bound_ms,
            Some(100)
        );
        assert!(!result.sampled_persistence_supported);
        assert!(!result.persistent_drift_proven);
    }

    #[test]
    fn exact_effective_time_can_prove_persistent_drift() {
        let mut latest = snapshot(300, Some(5), Some(3));
        latest
            .assertions
            .iter_mut()
            .find(|assertion| assertion.role == StateRole::Desired)
            .unwrap()
            .valid_from_unix_ms = Some(100);
        let result = assess_state_dimension_with_history(
            &history(vec![latest]),
            &subject(),
            "workload.replicas",
            300,
            policy(),
        )
        .unwrap();
        assert_eq!(result.current.status, TemporalStateStatus::PersistentDrift);
        assert!(!result.sampled_persistence_supported);
        assert!(result.persistent_drift_proven);
    }

    #[test]
    fn missing_desired_snapshot_breaks_continuity() {
        let history = history(vec![
            snapshot(100, Some(5), Some(3)),
            snapshot(200, None, Some(3)),
            snapshot(300, Some(5), Some(3)),
        ]);
        let continuity = desired_state_continuity(
            &history,
            &subject(),
            "workload.replicas",
            300,
        )
        .unwrap()
        .unwrap();
        assert_eq!(continuity.first_seen_at_unix_ms, 300);
        assert_eq!(continuity.consecutive_snapshots, 1);
        assert_eq!(
            drift_state_continuity(&history, &subject(), "workload.replicas", 300)
                .unwrap()
                .unwrap()
                .consecutive_snapshots,
            1
        );
    }

    #[test]
    fn desired_value_change_starts_new_continuity_epoch() {
        let history = history(vec![
            snapshot(100, Some(4), Some(3)),
            snapshot(200, Some(4), Some(3)),
            snapshot(300, Some(5), Some(3)),
            snapshot(400, Some(5), Some(3)),
        ]);
        let continuity = desired_state_continuity(
            &history,
            &subject(),
            "workload.replicas",
            400,
        )
        .unwrap()
        .unwrap();
        assert_eq!(continuity.first_seen_at_unix_ms, 300);
        assert_eq!(continuity.consecutive_snapshots, 2);
    }

    #[test]
    fn non_monotonic_history_fails_closed() {
        let history = history(vec![
            snapshot(200, Some(5), Some(3)),
            snapshot(100, Some(5), Some(3)),
        ]);
        assert!(matches!(
            history.validate(),
            Err(StateHistoryError::NonMonotonicSnapshotTime { .. })
        ));
    }
}
