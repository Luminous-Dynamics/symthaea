// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibration of conservative static viability bounds against dynamic observations.
//!
//! Static support-qualified horizons and dynamic simulation outcomes answer different
//! questions. A conservative bound may be exceeded without contradiction, while a
//! failure *before* the guaranteed boundary under unchanged assumptions is evidence
//! that the static model is incomplete or too optimistic. This module makes those
//! distinctions explicit and evidence-bound.

use crate::RegenerativeHorizon;
use serde::{Deserialize, Serialize};

const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;
const MAX_ROLE_OBSERVATIONS: usize = 4096;

/// Relationship between one static horizon and its dynamic observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegenerativeViabilityCalibrationClassV1 {
    /// The dynamic run changed assumptions represented by the static model, so a
    /// direct theorem comparison is invalid even though the observation is useful.
    AssumptionsChanged,
    /// A finite static boundary has not yet been reached by the observation window.
    InconclusiveBeforeFiniteBoundary,
    /// Dynamic unavailability occurred exactly on the first tick after the number
    /// of complete periods guaranteed by the static model.
    CorroboratedFiniteBoundary,
    /// Dynamic unavailability occurred before the finite static guarantee expired.
    EarlyFailureUnderStaticAssumptions,
    /// The dynamic system remained available beyond the conservative guarantee, or
    /// first failed later. This is not a contradiction; it is evidence that may
    /// justify a less conservative future model.
    SurvivedBeyondConservativeBound,
    /// No dynamic failure was observed while the static model remained indefinite
    /// under its stated assumptions.
    NoObservedContradictionToStaticIndefinite,
    /// Dynamic failure occurred despite an indefinite static result while the
    /// caller asserts that the static assumptions remained unchanged.
    ObservedFailureDespiteStaticIndefinite,
}

/// One role-level dynamic observation to compare with a static support horizon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeViabilityDynamicObservationV1 {
    pub role_id: String,
    pub static_horizon: RegenerativeHorizon,
    /// First completed dynamic tick on which the role was unavailable.
    pub first_unavailable_tick: Option<u64>,
    /// Last completed tick included in the observation window. Zero means no tick
    /// has completed yet.
    pub observed_through_tick: u64,
    /// Explicit caller assertion that the assumptions represented by the static
    /// model remained true throughout this observation window.
    pub static_assumptions_held: bool,
    /// Opaque binding to the exact dynamic observation/run evidence.
    pub observation_binding: String,
}

impl RegenerativeViabilityDynamicObservationV1 {
    fn validate(&self) -> Result<(), RegenerativeViabilityCalibrationError> {
        validate_id(&self.role_id)?;
        validate_binding(&self.observation_binding)?;
        if let Some(first_unavailable_tick) = self.first_unavailable_tick {
            if first_unavailable_tick == 0
                || first_unavailable_tick > self.observed_through_tick
            {
                return Err(RegenerativeViabilityCalibrationError::InvalidObservationWindow {
                    role_id: self.role_id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Evidence-bearing calibration result for one viability role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeViabilityRoleCalibrationV1 {
    pub role_id: String,
    pub static_horizon: RegenerativeHorizon,
    /// For finite horizons this is `finite_periods + 1`, the first tick on which
    /// static support is no longer guaranteed. `None` for an indefinite result.
    pub expected_first_unavailable_tick: Option<u64>,
    pub observed_first_unavailable_tick: Option<u64>,
    pub observed_through_tick: u64,
    pub static_assumptions_held: bool,
    pub classification: RegenerativeViabilityCalibrationClassV1,
    /// True only for observations that directly conflict with the static claim
    /// under unchanged assumptions.
    pub static_model_conflict: bool,
    /// True when the observation provides evidence worth using to revise/refine the
    /// static model, whether because it failed too early or proved conservative.
    pub model_refinement_opportunity: bool,
    pub observation_binding: String,
}

/// Deterministic calibration summary across a canonical role set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeViabilityCalibrationReportV1 {
    pub roles: Vec<RegenerativeViabilityRoleCalibrationV1>,
    pub any_static_model_conflict: bool,
    pub conflict_role_ids: Vec<String>,
    pub refinement_role_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeViabilityCalibrationError {
    InvalidIdentifier,
    InvalidBinding,
    InvalidObservationWindow { role_id: String },
    FiniteBoundaryOverflow { role_id: String },
    TooManyRoleObservations,
    NonCanonicalRoleOrder,
}

/// Calibrate one role-level static horizon against one dynamic observation.
pub fn calibrate_regenerative_viability_role(
    observation: &RegenerativeViabilityDynamicObservationV1,
) -> Result<RegenerativeViabilityRoleCalibrationV1, RegenerativeViabilityCalibrationError> {
    observation.validate()?;

    let expected_first_unavailable_tick = match observation.static_horizon {
        RegenerativeHorizon::FinitePeriods(periods) => Some(periods.checked_add(1).ok_or_else(
            || RegenerativeViabilityCalibrationError::FiniteBoundaryOverflow {
                role_id: observation.role_id.clone(),
            },
        )?),
        RegenerativeHorizon::IndefiniteUnderStaticModel => None,
    };

    let classification = if !observation.static_assumptions_held {
        RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
    } else {
        match (
            observation.static_horizon,
            expected_first_unavailable_tick,
            observation.first_unavailable_tick,
        ) {
            (
                RegenerativeHorizon::FinitePeriods(_),
                Some(expected),
                Some(observed),
            ) if observed < expected => {
                RegenerativeViabilityCalibrationClassV1::EarlyFailureUnderStaticAssumptions
            }
            (
                RegenerativeHorizon::FinitePeriods(_),
                Some(expected),
                Some(observed),
            ) if observed == expected => {
                RegenerativeViabilityCalibrationClassV1::CorroboratedFiniteBoundary
            }
            (
                RegenerativeHorizon::FinitePeriods(_),
                Some(_),
                Some(_),
            ) => RegenerativeViabilityCalibrationClassV1::SurvivedBeyondConservativeBound,
            (
                RegenerativeHorizon::FinitePeriods(_),
                Some(expected),
                None,
            ) if observation.observed_through_tick < expected => {
                RegenerativeViabilityCalibrationClassV1::InconclusiveBeforeFiniteBoundary
            }
            (RegenerativeHorizon::FinitePeriods(_), Some(_), None) => {
                RegenerativeViabilityCalibrationClassV1::SurvivedBeyondConservativeBound
            }
            (
                RegenerativeHorizon::IndefiniteUnderStaticModel,
                None,
                Some(_),
            ) => {
                RegenerativeViabilityCalibrationClassV1::ObservedFailureDespiteStaticIndefinite
            }
            (RegenerativeHorizon::IndefiniteUnderStaticModel, None, None) => {
                RegenerativeViabilityCalibrationClassV1::NoObservedContradictionToStaticIndefinite
            }
            _ => unreachable!("horizon and expected-boundary derivation are coupled"),
        }
    };

    let static_model_conflict = matches!(
        classification,
        RegenerativeViabilityCalibrationClassV1::EarlyFailureUnderStaticAssumptions
            | RegenerativeViabilityCalibrationClassV1::ObservedFailureDespiteStaticIndefinite
    );
    let model_refinement_opportunity = matches!(
        classification,
        RegenerativeViabilityCalibrationClassV1::EarlyFailureUnderStaticAssumptions
            | RegenerativeViabilityCalibrationClassV1::SurvivedBeyondConservativeBound
            | RegenerativeViabilityCalibrationClassV1::ObservedFailureDespiteStaticIndefinite
    );

    Ok(RegenerativeViabilityRoleCalibrationV1 {
        role_id: observation.role_id.clone(),
        static_horizon: observation.static_horizon,
        expected_first_unavailable_tick,
        observed_first_unavailable_tick: observation.first_unavailable_tick,
        observed_through_tick: observation.observed_through_tick,
        static_assumptions_held: observation.static_assumptions_held,
        classification,
        static_model_conflict,
        model_refinement_opportunity,
        observation_binding: observation.observation_binding.clone(),
    })
}

/// Calibrate a canonical, strictly role-sorted observation set.
pub fn calibrate_regenerative_viability_roles(
    observations: &[RegenerativeViabilityDynamicObservationV1],
) -> Result<RegenerativeViabilityCalibrationReportV1, RegenerativeViabilityCalibrationError> {
    if observations.len() > MAX_ROLE_OBSERVATIONS {
        return Err(RegenerativeViabilityCalibrationError::TooManyRoleObservations);
    }
    if observations
        .windows(2)
        .any(|pair| pair[0].role_id >= pair[1].role_id)
    {
        return Err(RegenerativeViabilityCalibrationError::NonCanonicalRoleOrder);
    }

    let roles = observations
        .iter()
        .map(calibrate_regenerative_viability_role)
        .collect::<Result<Vec<_>, _>>()?;
    let conflict_role_ids = roles
        .iter()
        .filter(|role| role.static_model_conflict)
        .map(|role| role.role_id.clone())
        .collect::<Vec<_>>();
    let refinement_role_ids = roles
        .iter()
        .filter(|role| role.model_refinement_opportunity)
        .map(|role| role.role_id.clone())
        .collect::<Vec<_>>();

    Ok(RegenerativeViabilityCalibrationReportV1 {
        any_static_model_conflict: !conflict_role_ids.is_empty(),
        conflict_role_ids,
        refinement_role_ids,
        roles,
    })
}

fn validate_id(value: &str) -> Result<(), RegenerativeViabilityCalibrationError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeViabilityCalibrationError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeViabilityCalibrationError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeViabilityCalibrationError::InvalidBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        role_id: &str,
        horizon: RegenerativeHorizon,
        first_unavailable_tick: Option<u64>,
        observed_through_tick: u64,
        assumptions_held: bool,
    ) -> RegenerativeViabilityDynamicObservationV1 {
        RegenerativeViabilityDynamicObservationV1 {
            role_id: role_id.into(),
            static_horizon: horizon,
            first_unavailable_tick,
            observed_through_tick,
            static_assumptions_held: assumptions_held,
            observation_binding: format!("symtropy-run:{role_id}:fixture"),
        }
    }

    #[test]
    fn exact_next_tick_failure_corroborates_finite_static_boundary() {
        let result = calibrate_regenerative_viability_role(&observation(
            "successor_qualification",
            RegenerativeHorizon::FinitePeriods(30),
            Some(31),
            31,
            true,
        ))
        .unwrap();
        assert_eq!(
            result.classification,
            RegenerativeViabilityCalibrationClassV1::CorroboratedFiniteBoundary
        );
        assert_eq!(result.expected_first_unavailable_tick, Some(31));
        assert!(!result.static_model_conflict);
        assert!(!result.model_refinement_opportunity);
    }

    #[test]
    fn early_failure_under_same_assumptions_is_a_static_model_conflict() {
        let result = calibrate_regenerative_viability_role(&observation(
            "successor_qualification",
            RegenerativeHorizon::FinitePeriods(30),
            Some(20),
            20,
            true,
        ))
        .unwrap();
        assert_eq!(
            result.classification,
            RegenerativeViabilityCalibrationClassV1::EarlyFailureUnderStaticAssumptions
        );
        assert!(result.static_model_conflict);
        assert!(result.model_refinement_opportunity);
    }

    #[test]
    fn surviving_beyond_conservative_bound_is_not_a_contradiction() {
        let result = calibrate_regenerative_viability_role(&observation(
            "successor_construction",
            RegenerativeHorizon::FinitePeriods(40),
            None,
            50,
            true,
        ))
        .unwrap();
        assert_eq!(
            result.classification,
            RegenerativeViabilityCalibrationClassV1::SurvivedBeyondConservativeBound
        );
        assert!(!result.static_model_conflict);
        assert!(result.model_refinement_opportunity);
    }

    #[test]
    fn changed_assumptions_do_not_masquerade_as_static_model_refutation() {
        let result = calibrate_regenerative_viability_role(&observation(
            "operation",
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            Some(7),
            7,
            false,
        ))
        .unwrap();
        assert_eq!(
            result.classification,
            RegenerativeViabilityCalibrationClassV1::AssumptionsChanged
        );
        assert!(!result.static_model_conflict);
        assert!(!result.model_refinement_opportunity);
    }

    #[test]
    fn indefinite_static_failure_under_unchanged_assumptions_is_conflicting_evidence() {
        let result = calibrate_regenerative_viability_role(&observation(
            "operation",
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            Some(77),
            77,
            true,
        ))
        .unwrap();
        assert_eq!(
            result.classification,
            RegenerativeViabilityCalibrationClassV1::ObservedFailureDespiteStaticIndefinite
        );
        assert!(result.static_model_conflict);
    }

    #[test]
    fn aggregate_report_preserves_conflict_and_refinement_roles() {
        let observations = vec![
            observation(
                "operation",
                RegenerativeHorizon::FinitePeriods(100),
                Some(101),
                101,
                true,
            ),
            observation(
                "successor_construction",
                RegenerativeHorizon::FinitePeriods(40),
                None,
                50,
                true,
            ),
            observation(
                "successor_qualification",
                RegenerativeHorizon::FinitePeriods(30),
                Some(20),
                20,
                true,
            ),
        ];
        let report = calibrate_regenerative_viability_roles(&observations).unwrap();
        assert!(report.any_static_model_conflict);
        assert_eq!(
            report.conflict_role_ids,
            vec!["successor_qualification"]
        );
        assert_eq!(
            report.refinement_role_ids,
            vec!["successor_construction", "successor_qualification"]
        );
    }
}
