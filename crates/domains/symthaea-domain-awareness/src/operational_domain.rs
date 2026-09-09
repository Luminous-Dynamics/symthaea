// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operational Design Domain (ODD) evidence and degraded-mode assessment.
//!
//! An ODD defines the environmental, sensing, navigation, communications, model,
//! and human-supervision conditions under which a system has evidence to operate.
//! Leaving the ODD can only restrict capability; it never increases authority.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Modality;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAssuranceState {
    Aligned,
    Restricted,
    Unsafe,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalDesignDomain {
    pub odd_id: Uuid,
    pub name: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    pub minimum_visibility_m: f64,
    pub maximum_wind_speed_mps: f64,
    /// Optional maximum sea state for maritime deployments.
    pub maximum_sea_state: Option<u8>,
    pub minimum_navigation_quality: f64,
    pub required_modalities: Vec<Modality>,
    pub require_communications: bool,
    pub require_operator_available: bool,
    pub authority_ref: String,
    pub evidence_refs: Vec<String>,
}

impl OperationalDesignDomain {
    pub fn validate(&self) -> bool {
        !self.name.trim().is_empty()
            && self.valid_from_ms <= self.valid_until_ms
            && self.minimum_visibility_m.is_finite()
            && self.minimum_visibility_m >= 0.0
            && self.maximum_wind_speed_mps.is_finite()
            && self.maximum_wind_speed_mps >= 0.0
            && self.maximum_sea_state.is_none_or(|value| value <= 9)
            && self.minimum_navigation_quality.is_finite()
            && (0.0..=1.0).contains(&self.minimum_navigation_quality)
            && !self.authority_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub fn active_at(&self, timestamp_ms: u64) -> bool {
        self.validate() && (self.valid_from_ms..=self.valid_until_ms).contains(&timestamp_ms)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalConditions {
    pub observed_at_ms: u64,
    pub visibility_m: Option<f64>,
    pub wind_speed_mps: Option<f64>,
    pub sea_state: Option<u8>,
    pub navigation_quality: Option<f64>,
    pub available_modalities: Vec<Modality>,
    pub communications_available: Option<bool>,
    pub operator_available: Option<bool>,
    pub model_assurance: ModelAssuranceState,
    pub evidence_refs: Vec<String>,
}

impl OperationalConditions {
    pub fn validate(&self) -> bool {
        self.visibility_m
            .is_none_or(|value| value.is_finite() && value >= 0.0)
            && self
                .wind_speed_mps
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && self.sea_state.is_none_or(|value| value <= 9)
            && self
                .navigation_quality
                .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperationalStatus {
    Nominal,
    Degraded,
    Restricted,
    Unavailable,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalIssue {
    InvalidOdd,
    InvalidConditions,
    OddNotActive,
    MissingVisibilityEvidence,
    VisibilityBelowMinimum,
    MissingWindEvidence,
    WindAboveMaximum,
    MissingSeaStateEvidence,
    SeaStateAboveMaximum,
    MissingNavigationEvidence,
    NavigationBelowMinimum,
    RequiredModalityUnavailable,
    MissingCommunicationsEvidence,
    CommunicationsUnavailable,
    MissingOperatorEvidence,
    OperatorUnavailable,
    ModelRestricted,
    ModelUnsafe,
    ModelIncomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalAssessment {
    pub status: OperationalStatus,
    pub issues: Vec<OperationalIssue>,
}

/// Evaluate current evidence against an ODD. The result is monotonic in the
/// safety sense: missing/degraded evidence can only retain or reduce capability.
pub fn assess_operational_domain(
    odd: &OperationalDesignDomain,
    conditions: &OperationalConditions,
) -> OperationalAssessment {
    if !odd.validate() {
        return OperationalAssessment {
            status: OperationalStatus::Incomplete,
            issues: vec![OperationalIssue::InvalidOdd],
        };
    }
    if !conditions.validate() {
        return OperationalAssessment {
            status: OperationalStatus::Incomplete,
            issues: vec![OperationalIssue::InvalidConditions],
        };
    }
    if !odd.active_at(conditions.observed_at_ms) {
        return OperationalAssessment {
            status: OperationalStatus::Unavailable,
            issues: vec![OperationalIssue::OddNotActive],
        };
    }

    let mut status = OperationalStatus::Nominal;
    let mut issues = Vec::new();

    let mut worsen = |new_status: OperationalStatus, issue: OperationalIssue| {
        if new_status > status {
            status = new_status;
        }
        issues.push(issue);
    };

    match conditions.visibility_m {
        None => worsen(
            OperationalStatus::Incomplete,
            OperationalIssue::MissingVisibilityEvidence,
        ),
        Some(value) if value < odd.minimum_visibility_m => worsen(
            OperationalStatus::Restricted,
            OperationalIssue::VisibilityBelowMinimum,
        ),
        Some(_) => {}
    }

    match conditions.wind_speed_mps {
        None => worsen(
            OperationalStatus::Incomplete,
            OperationalIssue::MissingWindEvidence,
        ),
        Some(value) if value > odd.maximum_wind_speed_mps => worsen(
            OperationalStatus::Restricted,
            OperationalIssue::WindAboveMaximum,
        ),
        Some(_) => {}
    }

    if let Some(maximum) = odd.maximum_sea_state {
        match conditions.sea_state {
            None => worsen(
                OperationalStatus::Incomplete,
                OperationalIssue::MissingSeaStateEvidence,
            ),
            Some(value) if value > maximum => worsen(
                OperationalStatus::Restricted,
                OperationalIssue::SeaStateAboveMaximum,
            ),
            Some(_) => {}
        }
    }

    match conditions.navigation_quality {
        None => worsen(
            OperationalStatus::Incomplete,
            OperationalIssue::MissingNavigationEvidence,
        ),
        Some(value) if value < odd.minimum_navigation_quality => worsen(
            OperationalStatus::Restricted,
            OperationalIssue::NavigationBelowMinimum,
        ),
        Some(_) => {}
    }

    if odd
        .required_modalities
        .iter()
        .any(|required| !conditions.available_modalities.contains(required))
    {
        worsen(
            OperationalStatus::Restricted,
            OperationalIssue::RequiredModalityUnavailable,
        );
    }

    if odd.require_communications {
        match conditions.communications_available {
            None => worsen(
                OperationalStatus::Incomplete,
                OperationalIssue::MissingCommunicationsEvidence,
            ),
            Some(false) => worsen(
                OperationalStatus::Restricted,
                OperationalIssue::CommunicationsUnavailable,
            ),
            Some(true) => {}
        }
    }

    if odd.require_operator_available {
        match conditions.operator_available {
            None => worsen(
                OperationalStatus::Incomplete,
                OperationalIssue::MissingOperatorEvidence,
            ),
            Some(false) => worsen(
                OperationalStatus::Unavailable,
                OperationalIssue::OperatorUnavailable,
            ),
            Some(true) => {}
        }
    }

    match conditions.model_assurance {
        ModelAssuranceState::Aligned => {}
        ModelAssuranceState::Restricted => worsen(
            OperationalStatus::Restricted,
            OperationalIssue::ModelRestricted,
        ),
        ModelAssuranceState::Unsafe => worsen(
            OperationalStatus::Unavailable,
            OperationalIssue::ModelUnsafe,
        ),
        ModelAssuranceState::Incomplete => worsen(
            OperationalStatus::Incomplete,
            OperationalIssue::ModelIncomplete,
        ),
    }

    OperationalAssessment { status, issues }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn odd() -> OperationalDesignDomain {
        OperationalDesignDomain {
            odd_id: Uuid::new_v4(),
            name: "harbor-awareness".to_string(),
            valid_from_ms: 0,
            valid_until_ms: 10_000,
            minimum_visibility_m: 500.0,
            maximum_wind_speed_mps: 25.0,
            maximum_sea_state: Some(6),
            minimum_navigation_quality: 0.7,
            required_modalities: vec![Modality::Radar, Modality::ElectroOptical],
            require_communications: true,
            require_operator_available: true,
            authority_ref: "authority:harbor".to_string(),
            evidence_refs: vec!["standard:odd-review".to_string()],
        }
    }

    fn nominal() -> OperationalConditions {
        OperationalConditions {
            observed_at_ms: 1_000,
            visibility_m: Some(5_000.0),
            wind_speed_mps: Some(10.0),
            sea_state: Some(2),
            navigation_quality: Some(0.95),
            available_modalities: vec![Modality::Radar, Modality::ElectroOptical],
            communications_available: Some(true),
            operator_available: Some(true),
            model_assurance: ModelAssuranceState::Aligned,
            evidence_refs: vec!["telemetry:conditions".to_string()],
        }
    }

    #[test]
    fn nominal_conditions_remain_nominal() {
        let result = assess_operational_domain(&odd(), &nominal());
        assert_eq!(result.status, OperationalStatus::Nominal);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn missing_evidence_does_not_default_to_safe() {
        let mut conditions = nominal();
        conditions.visibility_m = None;
        let result = assess_operational_domain(&odd(), &conditions);
        assert_eq!(result.status, OperationalStatus::Incomplete);
        assert!(result
            .issues
            .contains(&OperationalIssue::MissingVisibilityEvidence));
    }

    #[test]
    fn unsafe_model_makes_operation_unavailable() {
        let mut conditions = nominal();
        conditions.model_assurance = ModelAssuranceState::Unsafe;
        let result = assess_operational_domain(&odd(), &conditions);
        assert_eq!(result.status, OperationalStatus::Unavailable);
    }

    #[test]
    fn missing_operator_cannot_increase_capability() {
        let mut conditions = nominal();
        conditions.operator_available = Some(false);
        let result = assess_operational_domain(&odd(), &conditions);
        assert_eq!(result.status, OperationalStatus::Unavailable);
        assert!(result.issues.contains(&OperationalIssue::OperatorUnavailable));
    }
}
