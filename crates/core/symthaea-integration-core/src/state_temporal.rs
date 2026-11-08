// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal qualification for desired/observed state assessments.
//!
//! Instantaneous drift is not automatically persistent drift. This module
//! separates three independent questions:
//! - do desired and observed state agree now?
//! - how fresh is the evidence on each side?
//! - if they disagree, do we actually know when the desired value became
//!   effective, so a convergence window can be measured honestly?
//!
//! Re-reading the same desired value does **not** reset the convergence clock:
//! freshness uses `observed_at_unix_ms`, while drift age requires an explicit
//! `valid_from_unix_ms` on the active desired evidence.

use crate::{
    EntityRef, StateAssertion, StateAssessment, StateAssessmentError, StateAssessmentStatus,
    StateComparisonPolicy, StateRole, assess_state_dimension,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalStatePolicy {
    pub comparison: StateComparisonPolicy,
    /// Maximum accepted age of the freshest desired assertion. `None` disables
    /// desired-side staleness classification.
    pub max_desired_age_ms: Option<u64>,
    /// Maximum accepted age of the freshest observed assertion. `None` disables
    /// observed-side staleness classification.
    pub max_observed_age_ms: Option<u64>,
    /// Time after a *known desired effective time* during which disagreement is
    /// classified as convergence rather than persistent drift.
    pub convergence_window_ms: u64,
}

impl Default for TemporalStatePolicy {
    fn default() -> Self {
        Self {
            comparison: StateComparisonPolicy::Exact,
            max_desired_age_ms: None,
            max_observed_age_ms: None,
            convergence_window_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalStateStatus {
    InSync,
    Converging,
    PersistentDrift,
    /// Desired and observed state disagree, but no trustworthy desired-change
    /// timestamp exists. Persistence must not be inferred from repeated reads.
    DriftAgeUnknown,
    StaleDesired,
    StaleObserved,
    StaleBoth,
    NoEvidence,
    MissingDesired,
    MissingObserved,
    ConflictingDesired,
    ConflictingObserved,
    ConflictingBoth,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalStateAssessment {
    pub instantaneous: StateAssessment,
    pub status: TemporalStateStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desired_freshness_age_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_freshness_age_ms: Option<u64>,
    /// Effective time of the current desired consensus when every active desired
    /// source explicitly supplies the same/latest applicable change boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desired_effective_at_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_age_ms: Option<u64>,
}

pub fn assess_state_dimension_temporally(
    assertions: &[StateAssertion],
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
    policy: TemporalStatePolicy,
) -> Result<TemporalStateAssessment, TemporalStateAssessmentError> {
    // Historical knowledge-time validation must happen before active-at
    // filtering. `StateAssertion::is_active_at` correctly excludes assertions
    // observed after the query time, but silently filtering them here would
    // turn future evidence into an ordinary MissingDesired/MissingObserved
    // result instead of surfacing the chronology violation.
    for assertion in assertions.iter().filter(|assertion| {
        &assertion.subject == subject && assertion.dimension == dimension
    }) {
        if assertion.observed_at_unix_ms > at_unix_ms {
            return Err(TemporalStateAssessmentError::FutureAssertionTimestamp {
                assertion_id: assertion.assertion_id.clone(),
                observed_at_unix_ms: assertion.observed_at_unix_ms,
                assessment_at_unix_ms: at_unix_ms,
            });
        }
    }

    let instantaneous = assess_state_dimension(
        assertions,
        subject,
        dimension,
        at_unix_ms,
        policy.comparison,
    )?;

    let active = assertions
        .iter()
        .filter(|assertion| {
            &assertion.subject == subject
                && assertion.dimension == dimension
                && assertion.is_active_at(at_unix_ms)
        })
        .collect::<Vec<_>>();

    let desired = active
        .iter()
        .copied()
        .filter(|assertion| assertion.role == StateRole::Desired)
        .collect::<Vec<_>>();
    let observed = active
        .iter()
        .copied()
        .filter(|assertion| assertion.role == StateRole::Observed)
        .collect::<Vec<_>>();

    let desired_freshness_age_ms = freshest_age(&desired, at_unix_ms);
    let observed_freshness_age_ms = freshest_age(&observed, at_unix_ms);
    let desired_effective_at_unix_ms = desired_effective_time(&desired);

    let structural_status = match instantaneous.status {
        StateAssessmentStatus::NoEvidence => Some(TemporalStateStatus::NoEvidence),
        StateAssessmentStatus::MissingDesired => Some(TemporalStateStatus::MissingDesired),
        StateAssessmentStatus::MissingObserved => Some(TemporalStateStatus::MissingObserved),
        StateAssessmentStatus::ConflictingDesired => Some(TemporalStateStatus::ConflictingDesired),
        StateAssessmentStatus::ConflictingObserved => {
            Some(TemporalStateStatus::ConflictingObserved)
        }
        StateAssessmentStatus::ConflictingBoth => Some(TemporalStateStatus::ConflictingBoth),
        StateAssessmentStatus::InSync | StateAssessmentStatus::Drift => None,
    };

    if let Some(status) = structural_status {
        return Ok(TemporalStateAssessment {
            instantaneous,
            status,
            desired_freshness_age_ms,
            observed_freshness_age_ms,
            desired_effective_at_unix_ms,
            drift_age_ms: None,
        });
    }

    let desired_stale = is_stale(desired_freshness_age_ms, policy.max_desired_age_ms);
    let observed_stale = is_stale(observed_freshness_age_ms, policy.max_observed_age_ms);
    let stale_status = match (desired_stale, observed_stale) {
        (true, true) => Some(TemporalStateStatus::StaleBoth),
        (true, false) => Some(TemporalStateStatus::StaleDesired),
        (false, true) => Some(TemporalStateStatus::StaleObserved),
        (false, false) => None,
    };
    if let Some(status) = stale_status {
        return Ok(TemporalStateAssessment {
            instantaneous,
            status,
            desired_freshness_age_ms,
            observed_freshness_age_ms,
            desired_effective_at_unix_ms,
            drift_age_ms: None,
        });
    }

    if instantaneous.status == StateAssessmentStatus::InSync {
        return Ok(TemporalStateAssessment {
            instantaneous,
            status: TemporalStateStatus::InSync,
            desired_freshness_age_ms,
            observed_freshness_age_ms,
            desired_effective_at_unix_ms,
            drift_age_ms: None,
        });
    }

    let Some(effective_at) = desired_effective_at_unix_ms else {
        return Ok(TemporalStateAssessment {
            instantaneous,
            status: TemporalStateStatus::DriftAgeUnknown,
            desired_freshness_age_ms,
            observed_freshness_age_ms,
            desired_effective_at_unix_ms: None,
            drift_age_ms: None,
        });
    };
    if effective_at > at_unix_ms {
        return Err(TemporalStateAssessmentError::FutureDesiredEffectiveTime {
            effective_at_unix_ms: effective_at,
            assessment_at_unix_ms: at_unix_ms,
        });
    }
    let drift_age_ms = at_unix_ms - effective_at;
    let status = if drift_age_ms <= policy.convergence_window_ms {
        TemporalStateStatus::Converging
    } else {
        TemporalStateStatus::PersistentDrift
    };

    Ok(TemporalStateAssessment {
        instantaneous,
        status,
        desired_freshness_age_ms,
        observed_freshness_age_ms,
        desired_effective_at_unix_ms: Some(effective_at),
        drift_age_ms: Some(drift_age_ms),
    })
}

fn freshest_age(assertions: &[&StateAssertion], at_unix_ms: u64) -> Option<u64> {
    assertions
        .iter()
        .map(|assertion| assertion.observed_at_unix_ms)
        .max()
        .map(|freshest| at_unix_ms - freshest)
}

fn is_stale(age_ms: Option<u64>, max_age_ms: Option<u64>) -> bool {
    matches!((age_ms, max_age_ms), (Some(age), Some(max)) if age > max)
}

/// Return a trustworthy effective time only when every active desired assertion
/// carries one. Taking the maximum is conservative for agreeing desired sources:
/// it starts the convergence window at the most recent asserted change boundary.
fn desired_effective_time(assertions: &[&StateAssertion]) -> Option<u64> {
    if assertions.is_empty()
        || assertions
            .iter()
            .any(|assertion| assertion.valid_from_unix_ms.is_none())
    {
        return None;
    }
    assertions
        .iter()
        .filter_map(|assertion| assertion.valid_from_unix_ms)
        .max()
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TemporalStateAssessmentError {
    #[error("instantaneous state assessment failed: {0}")]
    State(#[from] StateAssessmentError),
    #[error(
        "state assertion `{assertion_id}` is timestamped in the future: observed {observed_at_unix_ms} > assessment {assessment_at_unix_ms}"
    )]
    FutureAssertionTimestamp {
        assertion_id: String,
        observed_at_unix_ms: u64,
        assessment_at_unix_ms: u64,
    },
    #[error(
        "desired effective time {effective_at_unix_ms} is after assessment time {assessment_at_unix_ms}"
    )]
    FutureDesiredEffectiveTime {
        effective_at_unix_ms: u64,
        assessment_at_unix_ms: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StateAssertionSource, StateValue};
    use std::collections::BTreeMap;

    fn subject() -> EntityRef {
        EntityRef::new("k8s:fixture", "k8s_deployment", "deployment-uid")
    }

    fn assertion(
        id: &str,
        role: StateRole,
        value: u64,
        observed_at_unix_ms: u64,
        valid_from_unix_ms: Option<u64>,
    ) -> StateAssertion {
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
            observed_at_unix_ms,
            valid_from_unix_ms,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
            attributes: BTreeMap::new(),
        }
    }

    fn policy() -> TemporalStatePolicy {
        TemporalStatePolicy {
            comparison: StateComparisonPolicy::Exact,
            max_desired_age_ms: Some(500),
            max_observed_age_ms: Some(500),
            convergence_window_ms: 1_000,
        }
    }

    #[test]
    fn fresh_matching_state_is_in_sync() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, 3, 900, Some(800)),
            assertion("observed", StateRole::Observed, 3, 950, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            1_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::InSync);
        assert_eq!(result.desired_freshness_age_ms, Some(100));
        assert_eq!(result.observed_freshness_age_ms, Some(50));
    }

    #[test]
    fn recent_known_desired_change_is_converging() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, 5, 9_900, Some(9_500)),
            assertion("observed", StateRole::Observed, 3, 9_950, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            10_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::Converging);
        assert_eq!(result.drift_age_ms, Some(500));
    }

    #[test]
    fn old_known_desired_change_is_persistent_drift() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, 5, 9_900, Some(8_000)),
            assertion("observed", StateRole::Observed, 3, 9_950, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            10_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::PersistentDrift);
        assert_eq!(result.drift_age_ms, Some(2_000));
    }

    #[test]
    fn repeated_desired_read_without_change_time_cannot_fake_convergence() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, 5, 9_999, None),
            assertion("observed", StateRole::Observed, 3, 9_999, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            10_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::DriftAgeUnknown);
        assert_eq!(result.drift_age_ms, None);
    }

    #[test]
    fn stale_observed_evidence_overrides_drift_classification() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, 5, 9_900, Some(8_000)),
            assertion("observed", StateRole::Observed, 3, 9_000, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            10_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::StaleObserved);
        assert_eq!(result.observed_freshness_age_ms, Some(1_000));
    }

    #[test]
    fn conflicting_evidence_is_not_relabelled_as_stale_or_drift() {
        let assertions = vec![
            assertion("desired-a", StateRole::Desired, 5, 9_000, Some(8_000)),
            assertion("desired-b", StateRole::Desired, 4, 9_900, Some(8_500)),
            assertion("observed", StateRole::Observed, 4, 9_950, None),
        ];
        let result = assess_state_dimension_temporally(
            &assertions,
            &subject(),
            "workload.replicas",
            10_000,
            policy(),
        )
        .unwrap();
        assert_eq!(result.status, TemporalStateStatus::ConflictingDesired);
    }

    #[test]
    fn future_assertion_timestamp_fails_closed() {
        let assertions = vec![assertion(
            "future",
            StateRole::Desired,
            5,
            10_001,
            Some(9_000),
        )];
        assert!(matches!(
            assess_state_dimension_temporally(
                &assertions,
                &subject(),
                "workload.replicas",
                10_000,
                policy(),
            ),
            Err(TemporalStateAssessmentError::FutureAssertionTimestamp { .. })
        ));
    }
}
