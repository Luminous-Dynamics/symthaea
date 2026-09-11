// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Risk/cost-aware diagnostic utility ranking.
//!
//! This layer ranks *candidate diagnostic tests*. It does not authorize or
//! execute them. Hard planning constraints are applied before utility scoring.
//!
//! ```text
//! high information gain != permission to execute
//! high utility != authorization
//! planning admissibility != live capability
//! low estimated disruption != guaranteed safety
//! ```

use crate::diagnostic_beliefs::{DiagnosticTestId, ExpectedInformationGainV1};
use crate::it_authority::{ItActionKindV1, ItOperationalRiskV1};
use crate::multifault_diagnosis::MultiFaultExpectedInformationGainV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticInformationValueV1 {
    pub test_id: DiagnosticTestId,
    pub information_gain_bits: f64,
}

impl From<ExpectedInformationGainV1> for DiagnosticInformationValueV1 {
    fn from(value: ExpectedInformationGainV1) -> Self {
        Self {
            test_id: value.test_id,
            information_gain_bits: value.information_gain_bits,
        }
    }
}

impl From<MultiFaultExpectedInformationGainV1> for DiagnosticInformationValueV1 {
    fn from(value: MultiFaultExpectedInformationGainV1) -> Self {
        Self {
            test_id: value.test_id,
            information_gain_bits: value.information_gain_bits,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticExecutionProfileV1 {
    pub test_id: DiagnosticTestId,
    pub action_kind: ItActionKindV1,
    pub operational_risk: ItOperationalRiskV1,
    /// Expected wall-clock time for the diagnostic itself.
    pub estimated_duration_ms: u64,
    /// Normalized resource burden [0, 1].
    pub resource_cost: f64,
    /// Normalized expected service/user disruption burden [0, 1].
    pub disruption_cost: f64,
    /// Whether an execution path exists that could later request live authority.
    /// This is not itself authorization.
    pub authority_path_available: bool,
    /// Whether the test is currently available in this environment.
    pub available: bool,
}

impl DiagnosticExecutionProfileV1 {
    pub fn validate(&self) -> Result<(), DiagnosticUtilityErrorV1> {
        if self.test_id.0.trim().is_empty() {
            return Err(DiagnosticUtilityErrorV1::EmptyTestId);
        }
        validate_unit_interval(self.resource_cost, "resource cost")?;
        validate_unit_interval(self.disruption_cost, "disruption cost")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticUtilityWeightsV1 {
    /// Reward per bit of expected information gain.
    pub information_gain_weight: f64,
    /// Penalty applied to duration normalized by `time_scale_ms`.
    pub time_weight: f64,
    pub resource_weight: f64,
    pub disruption_weight: f64,
    /// Duration that corresponds to one unit of normalized time cost.
    pub time_scale_ms: u64,
}

impl DiagnosticUtilityWeightsV1 {
    pub fn validate(&self) -> Result<(), DiagnosticUtilityErrorV1> {
        for (label, value) in [
            ("information gain weight", self.information_gain_weight),
            ("time weight", self.time_weight),
            ("resource weight", self.resource_weight),
            ("disruption weight", self.disruption_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(DiagnosticUtilityErrorV1::InvalidWeight {
                    label,
                    value,
                });
            }
        }
        if self.time_scale_ms == 0 {
            return Err(DiagnosticUtilityErrorV1::InvalidTimeScale);
        }
        Ok(())
    }
}

impl Default for DiagnosticUtilityWeightsV1 {
    fn default() -> Self {
        Self {
            information_gain_weight: 1.0,
            time_weight: 0.15,
            resource_weight: 0.20,
            disruption_weight: 0.50,
            time_scale_ms: 60_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticSelectionPolicyV1 {
    pub max_operational_risk: ItOperationalRiskV1,
    pub allow_active_diagnostics: bool,
    pub max_duration_ms: Option<u64>,
    pub max_resource_cost: Option<f64>,
    pub max_disruption_cost: Option<f64>,
    /// When true, candidates without a path to the separate live-authority layer
    /// are excluded even though this selector never mints authority itself.
    pub require_authority_path: bool,
    pub weights: DiagnosticUtilityWeightsV1,
}

impl DiagnosticSelectionPolicyV1 {
    pub fn validate(&self) -> Result<(), DiagnosticUtilityErrorV1> {
        self.weights.validate()?;
        if let Some(value) = self.max_resource_cost {
            validate_unit_interval(value, "maximum resource cost")?;
        }
        if let Some(value) = self.max_disruption_cost {
            validate_unit_interval(value, "maximum disruption cost")?;
        }
        Ok(())
    }
}

impl Default for DiagnosticSelectionPolicyV1 {
    fn default() -> Self {
        Self {
            max_operational_risk: ItOperationalRiskV1::Low,
            allow_active_diagnostics: false,
            max_duration_ms: None,
            max_resource_cost: Some(1.0),
            max_disruption_cost: Some(0.25),
            require_authority_path: true,
            weights: DiagnosticUtilityWeightsV1::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticExclusionReasonV1 {
    Unavailable,
    ActiveDiagnosticDisabled,
    RiskAbovePolicy {
        requested: ItOperationalRiskV1,
        maximum: ItOperationalRiskV1,
    },
    DurationAbovePolicy {
        estimated_ms: u64,
        maximum_ms: u64,
    },
    ResourceCostAbovePolicy,
    DisruptionCostAbovePolicy,
    AuthorityPathUnavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticUtilityAssessmentV1 {
    pub test_id: DiagnosticTestId,
    pub information_gain_bits: f64,
    pub eligible: bool,
    #[serde(default)]
    pub exclusions: Vec<DiagnosticExclusionReasonV1>,
    pub normalized_time_cost: f64,
    pub resource_cost: f64,
    pub disruption_cost: f64,
    /// Present only for eligible candidates. A positive value is not permission
    /// to execute; it is a planning score among candidates that passed hard gates.
    pub utility: Option<f64>,
}

pub fn rank_diagnostic_tests_by_utility_v1(
    information: &[DiagnosticInformationValueV1],
    profiles: &[DiagnosticExecutionProfileV1],
    policy: &DiagnosticSelectionPolicyV1,
) -> Result<Vec<DiagnosticUtilityAssessmentV1>, DiagnosticUtilityErrorV1> {
    policy.validate()?;

    let mut info_by_test = BTreeMap::new();
    for value in information {
        if value.test_id.0.trim().is_empty() {
            return Err(DiagnosticUtilityErrorV1::EmptyTestId);
        }
        if !value.information_gain_bits.is_finite() || value.information_gain_bits < 0.0 {
            return Err(DiagnosticUtilityErrorV1::InvalidInformationGain {
                test_id: value.test_id.clone(),
                value: value.information_gain_bits,
            });
        }
        if info_by_test
            .insert(value.test_id.clone(), value.information_gain_bits)
            .is_some()
        {
            return Err(DiagnosticUtilityErrorV1::DuplicateInformationValue(
                value.test_id.clone(),
            ));
        }
    }

    let mut profile_by_test = BTreeMap::new();
    for profile in profiles {
        profile.validate()?;
        if profile_by_test
            .insert(profile.test_id.clone(), profile)
            .is_some()
        {
            return Err(DiagnosticUtilityErrorV1::DuplicateExecutionProfile(
                profile.test_id.clone(),
            ));
        }
    }

    if info_by_test.len() != profile_by_test.len()
        || info_by_test
            .keys()
            .any(|test_id| !profile_by_test.contains_key(test_id))
    {
        return Err(DiagnosticUtilityErrorV1::ProfileInformationSetMismatch);
    }

    let mut assessments = Vec::with_capacity(info_by_test.len());
    for (test_id, information_gain_bits) in info_by_test {
        let profile = profile_by_test[&test_id];
        let mut exclusions = Vec::new();

        if !profile.available {
            exclusions.push(DiagnosticExclusionReasonV1::Unavailable);
        }
        if profile.action_kind.is_mutating_or_active() && !policy.allow_active_diagnostics {
            exclusions.push(DiagnosticExclusionReasonV1::ActiveDiagnosticDisabled);
        }
        if profile.operational_risk > policy.max_operational_risk {
            exclusions.push(DiagnosticExclusionReasonV1::RiskAbovePolicy {
                requested: profile.operational_risk,
                maximum: policy.max_operational_risk,
            });
        }
        if let Some(maximum_ms) = policy.max_duration_ms {
            if profile.estimated_duration_ms > maximum_ms {
                exclusions.push(DiagnosticExclusionReasonV1::DurationAbovePolicy {
                    estimated_ms: profile.estimated_duration_ms,
                    maximum_ms,
                });
            }
        }
        if policy
            .max_resource_cost
            .is_some_and(|maximum| profile.resource_cost > maximum)
        {
            exclusions.push(DiagnosticExclusionReasonV1::ResourceCostAbovePolicy);
        }
        if policy
            .max_disruption_cost
            .is_some_and(|maximum| profile.disruption_cost > maximum)
        {
            exclusions.push(DiagnosticExclusionReasonV1::DisruptionCostAbovePolicy);
        }
        if policy.require_authority_path && !profile.authority_path_available {
            exclusions.push(DiagnosticExclusionReasonV1::AuthorityPathUnavailable);
        }

        let normalized_time_cost =
            profile.estimated_duration_ms as f64 / policy.weights.time_scale_ms as f64;
        let eligible = exclusions.is_empty();
        let utility = eligible.then(|| {
            policy.weights.information_gain_weight * information_gain_bits
                - policy.weights.time_weight * normalized_time_cost
                - policy.weights.resource_weight * profile.resource_cost
                - policy.weights.disruption_weight * profile.disruption_cost
        });

        assessments.push(DiagnosticUtilityAssessmentV1 {
            test_id,
            information_gain_bits,
            eligible,
            exclusions,
            normalized_time_cost,
            resource_cost: profile.resource_cost,
            disruption_cost: profile.disruption_cost,
            utility,
        });
    }

    assessments.sort_by(|a, b| {
        b.eligible
            .cmp(&a.eligible)
            .then_with(|| match (a.utility, b.utility) {
                (Some(a_utility), Some(b_utility)) => b_utility
                    .partial_cmp(&a_utility)
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| {
                b.information_gain_bits
                    .partial_cmp(&a.information_gain_bits)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.test_id.0.cmp(&b.test_id.0))
    });

    Ok(assessments)
}

fn validate_unit_interval(
    value: f64,
    label: &'static str,
) -> Result<(), DiagnosticUtilityErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(DiagnosticUtilityErrorV1::InvalidUnitInterval { label, value })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticUtilityErrorV1 {
    EmptyTestId,
    InvalidUnitInterval { label: &'static str, value: f64 },
    InvalidWeight { label: &'static str, value: f64 },
    InvalidTimeScale,
    InvalidInformationGain { test_id: DiagnosticTestId, value: f64 },
    DuplicateInformationValue(DiagnosticTestId),
    DuplicateExecutionProfile(DiagnosticTestId),
    ProfileInformationSetMismatch,
}

impl fmt::Display for DiagnosticUtilityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTestId => write!(f, "empty diagnostic test id"),
            Self::InvalidUnitInterval { label, value } => {
                write!(f, "invalid {label} value {value}; expected [0, 1]")
            }
            Self::InvalidWeight { label, value } => {
                write!(f, "invalid {label} value {value}; expected finite non-negative")
            }
            Self::InvalidTimeScale => write!(f, "diagnostic utility time scale must be non-zero"),
            Self::InvalidInformationGain { test_id, value } => write!(
                f,
                "invalid information gain {value} for diagnostic test {:?}",
                test_id.0
            ),
            Self::DuplicateInformationValue(test_id) => write!(
                f,
                "duplicate information value for diagnostic test {:?}",
                test_id.0
            ),
            Self::DuplicateExecutionProfile(test_id) => write!(
                f,
                "duplicate execution profile for diagnostic test {:?}",
                test_id.0
            ),
            Self::ProfileInformationSetMismatch => write!(
                f,
                "diagnostic information values and execution profiles do not describe the same test set"
            ),
        }
    }
}

impl Error for DiagnosticUtilityErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: &str, bits: f64) -> DiagnosticInformationValueV1 {
        DiagnosticInformationValueV1 {
            test_id: DiagnosticTestId(id.into()),
            information_gain_bits: bits,
        }
    }

    fn profile(
        id: &str,
        action_kind: ItActionKindV1,
        risk: ItOperationalRiskV1,
        duration_ms: u64,
        resource_cost: f64,
        disruption_cost: f64,
    ) -> DiagnosticExecutionProfileV1 {
        DiagnosticExecutionProfileV1 {
            test_id: DiagnosticTestId(id.into()),
            action_kind,
            operational_risk: risk,
            estimated_duration_ms: duration_ms,
            resource_cost,
            disruption_cost,
            authority_path_available: true,
            available: true,
        }
    }

    #[test]
    fn lower_information_test_can_win_when_much_cheaper_and_safer() {
        let information = vec![info("slow", 1.2), info("fast", 0.9)];
        let profiles = vec![
            profile(
                "slow",
                ItActionKindV1::ReadOnlyDiagnostic,
                ItOperationalRiskV1::Low,
                600_000,
                0.8,
                0.2,
            ),
            profile(
                "fast",
                ItActionKindV1::ReadOnlyDiagnostic,
                ItOperationalRiskV1::Passive,
                5_000,
                0.05,
                0.0,
            ),
        ];
        let policy = DiagnosticSelectionPolicyV1 {
            max_operational_risk: ItOperationalRiskV1::Low,
            allow_active_diagnostics: false,
            max_duration_ms: None,
            max_resource_cost: Some(1.0),
            max_disruption_cost: Some(1.0),
            require_authority_path: true,
            weights: DiagnosticUtilityWeightsV1 {
                information_gain_weight: 1.0,
                time_weight: 0.2,
                resource_weight: 0.3,
                disruption_weight: 0.5,
                time_scale_ms: 60_000,
            },
        };
        let ranked = rank_diagnostic_tests_by_utility_v1(&information, &profiles, &policy).unwrap();
        assert_eq!(ranked[0].test_id.0, "fast");
        assert!(ranked[0].eligible);
    }

    #[test]
    fn high_information_active_test_is_excluded_when_active_diagnostics_disabled() {
        let information = vec![info("active", 2.0), info("passive", 0.5)];
        let profiles = vec![
            profile(
                "active",
                ItActionKindV1::ActiveDiagnostic,
                ItOperationalRiskV1::Moderate,
                1_000,
                0.1,
                0.1,
            ),
            profile(
                "passive",
                ItActionKindV1::ReadOnlyDiagnostic,
                ItOperationalRiskV1::Passive,
                2_000,
                0.1,
                0.0,
            ),
        ];
        let policy = DiagnosticSelectionPolicyV1 {
            max_operational_risk: ItOperationalRiskV1::High,
            allow_active_diagnostics: false,
            max_duration_ms: None,
            max_resource_cost: Some(1.0),
            max_disruption_cost: Some(1.0),
            require_authority_path: true,
            weights: DiagnosticUtilityWeightsV1::default(),
        };
        let ranked = rank_diagnostic_tests_by_utility_v1(&information, &profiles, &policy).unwrap();
        assert_eq!(ranked[0].test_id.0, "passive");
        let active = ranked.iter().find(|item| item.test_id.0 == "active").unwrap();
        assert!(!active.eligible);
        assert!(active
            .exclusions
            .contains(&DiagnosticExclusionReasonV1::ActiveDiagnosticDisabled));
        assert!(active.utility.is_none());
    }

    #[test]
    fn authority_path_is_a_hard_planning_gate_not_a_utility_penalty() {
        let information = vec![info("high-info", 4.0), info("bounded", 0.2)];
        let mut unavailable = profile(
            "high-info",
            ItActionKindV1::ReadOnlyDiagnostic,
            ItOperationalRiskV1::Passive,
            1_000,
            0.0,
            0.0,
        );
        unavailable.authority_path_available = false;
        let profiles = vec![
            unavailable,
            profile(
                "bounded",
                ItActionKindV1::ReadOnlyDiagnostic,
                ItOperationalRiskV1::Passive,
                1_000,
                0.0,
                0.0,
            ),
        ];
        let ranked = rank_diagnostic_tests_by_utility_v1(
            &information,
            &profiles,
            &DiagnosticSelectionPolicyV1::default(),
        )
        .unwrap();
        assert_eq!(ranked[0].test_id.0, "bounded");
        let high = ranked.iter().find(|item| item.test_id.0 == "high-info").unwrap();
        assert!(high
            .exclusions
            .contains(&DiagnosticExclusionReasonV1::AuthorityPathUnavailable));
    }

    #[test]
    fn risk_limit_is_hard_constraint() {
        let information = vec![info("risky", 5.0)];
        let profiles = vec![profile(
            "risky",
            ItActionKindV1::ReadOnlyDiagnostic,
            ItOperationalRiskV1::High,
            1_000,
            0.0,
            0.0,
        )];
        let policy = DiagnosticSelectionPolicyV1 {
            max_operational_risk: ItOperationalRiskV1::Low,
            ..DiagnosticSelectionPolicyV1::default()
        };
        let ranked = rank_diagnostic_tests_by_utility_v1(&information, &profiles, &policy).unwrap();
        assert!(!ranked[0].eligible);
        assert!(ranked[0].utility.is_none());
        assert!(ranked[0].exclusions.iter().any(|reason| matches!(
            reason,
            DiagnosticExclusionReasonV1::RiskAbovePolicy { .. }
        )));
    }

    #[test]
    fn test_sets_must_match_exactly() {
        let result = rank_diagnostic_tests_by_utility_v1(
            &[info("a", 1.0)],
            &[profile(
                "b",
                ItActionKindV1::ReadOnlyDiagnostic,
                ItOperationalRiskV1::Passive,
                1,
                0.0,
                0.0,
            )],
            &DiagnosticSelectionPolicyV1::default(),
        );
        assert!(matches!(
            result,
            Err(DiagnosticUtilityErrorV1::ProfileInformationSetMismatch)
        ));
    }
}
