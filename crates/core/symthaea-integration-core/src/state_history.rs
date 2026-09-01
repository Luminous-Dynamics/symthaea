// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded state history and conservative persistence evidence.
//!
//! A sequence of snapshots cannot reveal the exact time a desired value changed
//! before the first observation. It can prove lower bounds about continuously
//! observed evidence. Critically, unchanged desired state alone does not prove
//! persistent drift: the observed side might have converged and later regressed.
//! Persistence from history therefore requires an uninterrupted run in which the
//! same desired value is present and every sampled state is actually in drift.

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

/// Uninterrupted sampled evidence that the current desired value remained in
/// disagreement with observed state. This is still a sampling lower bound, not
/// proof that no unsampled convergence occurred between captures.
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
    /// Lower bound on how long the current desired value has been continuously
    /// observed. This alone is insufficient to prove persistent drift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuously_observed_desired_age_lower_bound_ms: Option<u64>,
    /// Lower bound from an uninterrupted run of sampled *drift* states with the
    /// same desired value. This is the history evidence used for persistence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuously_observed_drift_age_lower_bound_ms: Option<u64>,
    /// True only when persistence is proved either by an exact desired-effective
    /// timestamp or by a sampled drift lower bound that exceeds the convergence
    /// window. History never proves `Converging`.
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
    let desired_lower_bound = desired_continuity
        .as_ref()
        .map(|continuity| at_unix_ms - continuity.first_seen_at_unix_ms);
    let drift_lower_bound = drift_continuity
        .as_ref()
        .map(|continuity| at_unix_ms - continuity.first_seen_at_unix_ms);
    let lower_bound_proves_persistence = current.status == TemporalStateStatus::DriftAgeUnknown
        && drift_lower_bound.is_some_and(|age| age > policy.convergence_window_ms);
    let persistent_drift_proven = current.status == TemporalStateStatus::PersistentDrift
        || lower_bound_proves_persistence;

    Ok(HistoricalStateAssessment {
        current,
        desired_continuity,
        drift_continuity,
        continuously_observed_desired_age_lower_bound_ms: desired_lower_bound,
        continuously_observed_drift_age_lower_bound_ms: drift_lower_bound,
        persistent_drift_proven,
    })
}

/// Find the uninterrupted history of the current desired value.
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

/// Find the uninterrupted sampled run in which the current desired value and
/// observed state are in actual drift. Any in-sync, missing, conflicting, or
/// changed-desired snapshot breaks the persistence chain.
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
    fn continuous_sampled_drift_can_prove_persistence_lower_bound() {
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
        assert!(result.persistent_drift_proven);
    }

    #[test]
    fn intervening_convergence_breaks_persistent_drift_proof() {
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
        assert!(!result.persistent_drift_proven);
        assert_eq!(result.drift_continuity.as_ref().unwrap().consecutive_snapshots, 1);
    }

    #[test]
    fn recent_first_drift_observation_cannot_prove_convergence_or_persistence() {
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
            Some(50)
        );
        assert!(!result.persistent_drift_proven);
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
