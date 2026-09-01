// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing desired/observed state assertions and drift assessment.
//!
//! This module is deliberately domain-neutral: Kubernetes, Nix, Terraform,
//! cloud control planes, CMDBs, and other integrations can all describe what a
//! resource *should* be and what it *is* without coupling drift semantics to a
//! single adapter. Missing or contradictory evidence is not collapsed into
//! "drift"; those states remain explicit.

use crate::{EntityRef, ObservationId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateRole {
    Desired,
    Observed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateValue {
    Number(f64),
    Integer(i64),
    Unsigned(u64),
    Boolean(bool),
    Text(String),
}

impl StateValue {
    pub fn validate(&self) -> Result<(), StateValidationError> {
        match self {
            Self::Number(value) if !value.is_finite() => {
                Err(StateValidationError::NonFiniteNumber(*value))
            }
            Self::Text(value) if value.trim().is_empty() => {
                Err(StateValidationError::EmptyField("value.text"))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateAssertionSource {
    pub integration_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

/// One source's assertion about one state dimension of one entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateAssertion {
    pub assertion_id: String,
    pub subject: EntityRef,
    /// Stable semantic dimension, e.g. `replicas.desired`, `service.enabled`,
    /// `image.digest`, `config.generation`, or `filesystem.read_only`.
    pub dimension: String,
    pub role: StateRole,
    pub value: StateValue,
    /// Confidence in source extraction/mapping only. It is not a probability
    /// that the desired state is correct or that observed state is healthy.
    pub source_confidence: f32,
    pub source: StateAssertionSource,
    pub observed_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_observation_ids: Vec<ObservationId>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
}

impl StateAssertion {
    pub fn validate(&self) -> Result<(), StateValidationError> {
        require_non_empty("assertion_id", &self.assertion_id)?;
        require_non_empty("subject.namespace", &self.subject.namespace)?;
        require_non_empty("subject.kind", &self.subject.kind)?;
        require_non_empty("subject.id", &self.subject.id)?;
        require_non_empty("dimension", &self.dimension)?;
        require_non_empty("source.integration_id", &self.source.integration_id)?;
        validate_probability("source_confidence", self.source_confidence)?;
        validate_interval(self.valid_from_unix_ms, self.valid_until_unix_ms)?;
        self.value.validate()?;

        let mut evidence = BTreeSet::new();
        for id in &self.evidence_observation_ids {
            require_non_empty("evidence_observation_id", id.as_str())?;
            if !evidence.insert(id.clone()) {
                return Err(StateValidationError::DuplicateEvidenceObservationId(
                    id.clone(),
                ));
            }
        }
        for (key, value) in &self.attributes {
            require_non_empty("attributes.key", key)?;
            require_non_empty("attributes.value", value)?;
        }
        Ok(())
    }

    pub fn is_active_at(&self, at_unix_ms: u64) -> bool {
        self.valid_from_unix_ms
            .is_none_or(|from| at_unix_ms >= from)
            && self
                .valid_until_unix_ms
                .is_none_or(|until| at_unix_ms <= until)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StateComparisonPolicy {
    Exact,
    /// Absolute tolerance for floating-point state such as ratios or voltages.
    /// Integer and boolean/text values remain exact under this policy.
    NumericAbsoluteTolerance { tolerance: f64 },
}

impl Default for StateComparisonPolicy {
    fn default() -> Self {
        Self::Exact
    }
}

impl StateComparisonPolicy {
    pub fn validate(&self) -> Result<(), StateAssessmentError> {
        match self {
            Self::Exact => Ok(()),
            Self::NumericAbsoluteTolerance { tolerance }
                if tolerance.is_finite() && *tolerance >= 0.0 =>
            {
                Ok(())
            }
            Self::NumericAbsoluteTolerance { tolerance } => {
                Err(StateAssessmentError::InvalidTolerance(*tolerance))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateAssessmentStatus {
    InSync,
    Drift,
    MissingDesired,
    MissingObserved,
    ConflictingDesired,
    ConflictingObserved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateAssessment {
    pub subject: EntityRef,
    pub dimension: String,
    pub status: StateAssessmentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desired_value: Option<StateValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_value: Option<StateValue>,
    pub desired_assertion_ids: Vec<String>,
    pub observed_assertion_ids: Vec<String>,
}

/// Assess one entity/dimension at one instant.
///
/// Multiple active assertions on either side are allowed only when they agree
/// under the selected comparison policy. Disagreement is surfaced as explicit
/// source conflict and is never misreported as desired/observed drift.
pub fn assess_state_dimension(
    assertions: &[StateAssertion],
    subject: &EntityRef,
    dimension: &str,
    at_unix_ms: u64,
    policy: StateComparisonPolicy,
) -> Result<StateAssessment, StateAssessmentError> {
    require_non_empty("dimension", dimension).map_err(StateAssessmentError::InvalidAssertion)?;
    policy.validate()?;

    let mut desired = Vec::new();
    let mut observed = Vec::new();
    let mut seen_ids = BTreeSet::new();

    for assertion in assertions {
        assertion
            .validate()
            .map_err(StateAssessmentError::InvalidAssertion)?;
        if !seen_ids.insert(assertion.assertion_id.clone()) {
            return Err(StateAssessmentError::DuplicateAssertionId(
                assertion.assertion_id.clone(),
            ));
        }
        if &assertion.subject != subject
            || assertion.dimension != dimension
            || !assertion.is_active_at(at_unix_ms)
        {
            continue;
        }
        match assertion.role {
            StateRole::Desired => desired.push(assertion),
            StateRole::Observed => observed.push(assertion),
        }
    }

    let desired_ids = sorted_ids(&desired);
    let observed_ids = sorted_ids(&observed);

    if desired.is_empty() {
        return Ok(StateAssessment {
            subject: subject.clone(),
            dimension: dimension.into(),
            status: StateAssessmentStatus::MissingDesired,
            desired_value: None,
            observed_value: consensus_value(&observed, policy)?,
            desired_assertion_ids: desired_ids,
            observed_assertion_ids: observed_ids,
        });
    }
    if observed.is_empty() {
        return Ok(StateAssessment {
            subject: subject.clone(),
            dimension: dimension.into(),
            status: StateAssessmentStatus::MissingObserved,
            desired_value: consensus_value(&desired, policy)?,
            observed_value: None,
            desired_assertion_ids: desired_ids,
            observed_assertion_ids: observed_ids,
        });
    }

    let Some(desired_value) = consensus_value(&desired, policy)? else {
        return Ok(StateAssessment {
            subject: subject.clone(),
            dimension: dimension.into(),
            status: StateAssessmentStatus::ConflictingDesired,
            desired_value: None,
            observed_value: consensus_value(&observed, policy)?,
            desired_assertion_ids: desired_ids,
            observed_assertion_ids: observed_ids,
        });
    };
    let Some(observed_value) = consensus_value(&observed, policy)? else {
        return Ok(StateAssessment {
            subject: subject.clone(),
            dimension: dimension.into(),
            status: StateAssessmentStatus::ConflictingObserved,
            desired_value: Some(desired_value),
            observed_value: None,
            desired_assertion_ids: desired_ids,
            observed_assertion_ids: observed_ids,
        });
    };

    let in_sync = values_match(&desired_value, &observed_value, policy)?;
    Ok(StateAssessment {
        subject: subject.clone(),
        dimension: dimension.into(),
        status: if in_sync {
            StateAssessmentStatus::InSync
        } else {
            StateAssessmentStatus::Drift
        },
        desired_value: Some(desired_value),
        observed_value: Some(observed_value),
        desired_assertion_ids: desired_ids,
        observed_assertion_ids: observed_ids,
    })
}

fn consensus_value(
    assertions: &[&StateAssertion],
    policy: StateComparisonPolicy,
) -> Result<Option<StateValue>, StateAssessmentError> {
    let Some(first) = assertions.first() else {
        return Ok(None);
    };
    for assertion in assertions.iter().skip(1) {
        if !values_match(&first.value, &assertion.value, policy)? {
            return Ok(None);
        }
    }
    Ok(Some(first.value.clone()))
}

fn values_match(
    left: &StateValue,
    right: &StateValue,
    policy: StateComparisonPolicy,
) -> Result<bool, StateAssessmentError> {
    match policy {
        StateComparisonPolicy::Exact => Ok(left == right),
        StateComparisonPolicy::NumericAbsoluteTolerance { tolerance } => match (left, right) {
            (StateValue::Number(left), StateValue::Number(right)) => {
                Ok((left - right).abs() <= tolerance)
            }
            _ => Ok(left == right),
        },
    }
}

fn sorted_ids(assertions: &[&StateAssertion]) -> Vec<String> {
    let mut ids = assertions
        .iter()
        .map(|assertion| assertion.assertion_id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateValidationError {
    #[error("required state field `{0}` is empty")]
    EmptyField(&'static str),
    #[error("state number must be finite, got {0}")]
    NonFiniteNumber(f64),
    #[error("state confidence `{field}` must be finite and within [0,1], got {value}")]
    ConfidenceOutOfRange { field: &'static str, value: f32 },
    #[error("state validity range is inverted: from {from} > until {until}")]
    InvertedValidityRange { from: u64, until: u64 },
    #[error("duplicate evidence observation id `{0}`")]
    DuplicateEvidenceObservationId(ObservationId),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateAssessmentError {
    #[error("invalid state assertion: {0}")]
    InvalidAssertion(StateValidationError),
    #[error("duplicate state assertion id `{0}`")]
    DuplicateAssertionId(String),
    #[error("numeric state tolerance must be finite and >= 0, got {0}")]
    InvalidTolerance(f64),
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), StateValidationError> {
    if value.trim().is_empty() {
        Err(StateValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_probability(field: &'static str, value: f32) -> Result<(), StateValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(StateValidationError::ConfidenceOutOfRange { field, value })
    }
}

fn validate_interval(from: Option<u64>, until: Option<u64>) -> Result<(), StateValidationError> {
    if let (Some(from), Some(until)) = (from, until) {
        if from > until {
            return Err(StateValidationError::InvertedValidityRange { from, until });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: &str) -> StateAssertionSource {
        StateAssertionSource {
            integration_id: id.into(),
            collector_id: None,
            tenant: None,
        }
    }

    fn assertion(id: &str, role: StateRole, value: StateValue) -> StateAssertion {
        StateAssertion {
            assertion_id: id.into(),
            subject: EntityRef::new("k8s:fixture", "k8s_deployment", "deployment-uid"),
            dimension: "replicas".into(),
            role,
            value,
            source_confidence: 1.0,
            source: source("fixture"),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
            attributes: BTreeMap::new(),
        }
    }

    fn subject() -> EntityRef {
        EntityRef::new("k8s:fixture", "k8s_deployment", "deployment-uid")
    }

    #[test]
    fn matching_desired_and_observed_state_is_in_sync() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, StateValue::Unsigned(3)),
            assertion("observed", StateRole::Observed, StateValue::Unsigned(3)),
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::InSync);
    }

    #[test]
    fn replica_mismatch_is_drift() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, StateValue::Unsigned(5)),
            assertion("observed", StateRole::Observed, StateValue::Unsigned(3)),
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::Drift);
    }

    #[test]
    fn contradictory_desired_sources_are_not_misreported_as_drift() {
        let assertions = vec![
            assertion("gitops", StateRole::Desired, StateValue::Unsigned(5)),
            assertion("cluster-spec", StateRole::Desired, StateValue::Unsigned(4)),
            assertion("observed", StateRole::Observed, StateValue::Unsigned(4)),
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(
            assessment.status,
            StateAssessmentStatus::ConflictingDesired
        );
    }

    #[test]
    fn contradictory_observed_sources_are_explicit() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, StateValue::Unsigned(5)),
            assertion("api-status", StateRole::Observed, StateValue::Unsigned(5)),
            assertion("metrics", StateRole::Observed, StateValue::Unsigned(3)),
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(
            assessment.status,
            StateAssessmentStatus::ConflictingObserved
        );
    }

    #[test]
    fn missing_observed_state_is_unknown_not_drift() {
        let assertions = vec![assertion(
            "desired",
            StateRole::Desired,
            StateValue::Unsigned(5),
        )];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::MissingObserved);
    }

    #[test]
    fn floating_state_can_use_explicit_absolute_tolerance() {
        let assertions = vec![
            assertion("desired", StateRole::Desired, StateValue::Number(0.80)),
            assertion("observed", StateRole::Observed, StateValue::Number(0.79)),
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::NumericAbsoluteTolerance { tolerance: 0.02 },
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::InSync);
    }

    #[test]
    fn expired_assertions_do_not_participate() {
        let mut old = assertion("old", StateRole::Observed, StateValue::Unsigned(1));
        old.valid_until_unix_ms = Some(99);
        let assertions = vec![
            assertion("desired", StateRole::Desired, StateValue::Unsigned(5)),
            old,
        ];
        let assessment = assess_state_dimension(
            &assertions,
            &subject(),
            "replicas",
            100,
            StateComparisonPolicy::Exact,
        )
        .unwrap();
        assert_eq!(assessment.status, StateAssessmentStatus::MissingObserved);
    }

    #[test]
    fn duplicate_assertion_ids_fail_closed() {
        let assertions = vec![
            assertion("same", StateRole::Desired, StateValue::Unsigned(5)),
            assertion("same", StateRole::Observed, StateValue::Unsigned(5)),
        ];
        assert!(matches!(
            assess_state_dimension(
                &assertions,
                &subject(),
                "replicas",
                100,
                StateComparisonPolicy::Exact,
            ),
            Err(StateAssessmentError::DuplicateAssertionId(_))
        ));
    }

    #[test]
    fn non_finite_state_is_rejected() {
        let assertion = assertion(
            "desired",
            StateRole::Desired,
            StateValue::Number(f64::NAN),
        );
        assert!(matches!(
            assertion.validate(),
            Err(StateValidationError::NonFiniteNumber(_))
        ));
    }
}
