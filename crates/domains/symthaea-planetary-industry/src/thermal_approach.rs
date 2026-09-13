// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-002C conservative thermal approach-temperature screening.
//!
//! This module mirrors the independent `scripts/pie-thermal-approach-oracle.py`
//! theorem without rewriting the simpler PIE-002 temperature-envelope screen.
//! A declared approach-temperature envelope is accounted for explicitly when a
//! claim is intended to represent finite thermal driving force.
//!
//! This remains a first-order screening theorem. It does not establish entropy,
//! exergy, heat-exchanger performance, phase-change behavior, kinetics, thermal
//! network dispatch, process feasibility, equipment qualification, or authority.

use serde::{Deserialize, Serialize};

use crate::{OntologyError, TemperatureRangeK};

/// Conservative thermal approach-temperature classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalApproachStatus {
    /// Every admissible realization has non-negative declared thermal headroom.
    Guaranteed,
    /// The declared uncertainty spans both sufficient and insufficient headroom.
    Possible,
    /// Every admissible realization has negative declared thermal headroom.
    Impossible,
}

/// Semantic basis of the approach-temperature screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalApproachBasis {
    /// Pure algebraic source-vs-demand envelope; zero approach is allowed.
    AlgebraicEnvelope,
    /// Explicit finite-driving-force proposition; approach lower bound must be positive.
    FiniteDrivingForce,
}

/// Signed thermal headroom interval in kelvin.
///
/// Unlike absolute `TemperatureRangeK`, headroom may legitimately be negative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalHeadroomRangeK {
    /// Conservative lower residual bound in kelvin.
    pub min_k: f64,
    /// Conservative upper residual bound in kelvin.
    pub max_k: f64,
}

impl ThermalHeadroomRangeK {
    fn new(min_k: f64, max_k: f64) -> Result<Self, OntologyError> {
        if !min_k.is_finite() || !max_k.is_finite() || min_k > max_k {
            Err(OntologyError::InvalidRange("thermal_headroom_k"))
        } else {
            Ok(Self { min_k, max_k })
        }
    }
}

/// Inputs to one thermal approach-temperature screen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalApproachCase {
    /// Available source-temperature envelope in kelvin; lower bound must be positive.
    pub source_temperature_k: TemperatureRangeK,
    /// Required process-temperature envelope in kelvin; lower bound must be positive.
    pub required_temperature_k: TemperatureRangeK,
    /// Declared minimum approach-temperature envelope in kelvin.
    pub minimum_approach_k: TemperatureRangeK,
    /// Claim basis determining whether zero-inclusive approach is admissible.
    pub basis: ThermalApproachBasis,
}

/// Result of one conservative approach-temperature screen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalApproachReport {
    /// Conservative headroom classification.
    pub status: ThermalApproachStatus,
    /// Signed interval for `source - required - approach`.
    pub headroom_k: ThermalHeadroomRangeK,
    /// Required source-temperature envelope `required + approach`.
    pub required_source_temperature_k: TemperatureRangeK,
    /// Exact claim basis used for this result.
    pub basis: ThermalApproachBasis,
    /// Whether this result is eligible to support the narrow finite-driving-force proposition.
    pub supports_finite_driving_force_claim: bool,
}

fn validate_positive_temperature(
    value: TemperatureRangeK,
    label: &'static str,
) -> Result<(), OntologyError> {
    value.validate()?;
    if value.min.value() > 0.0 {
        Ok(())
    } else {
        Err(OntologyError::InvalidRange(label))
    }
}

fn finite_sum(a: f64, b: f64, label: &'static str) -> Result<f64, OntologyError> {
    let value = a + b;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OntologyError::InvalidQuantity(label))
    }
}

fn finite_difference3(
    a: f64,
    b: f64,
    c: f64,
    label: &'static str,
) -> Result<f64, OntologyError> {
    let value = a - b - c;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OntologyError::InvalidQuantity(label))
    }
}

/// Screen source temperature against required process temperature plus approach.
///
/// The conservative residual is:
///
/// `source - required_process_temperature - minimum_approach`.
///
/// `FiniteDrivingForce` requires a strictly positive lower approach bound.
/// `AlgebraicEnvelope` permits zero approach but never supports the stronger
/// finite-driving-force claim.
pub fn screen_thermal_approach(
    case: &ThermalApproachCase,
) -> Result<ThermalApproachReport, OntologyError> {
    validate_positive_temperature(case.source_temperature_k, "source_temperature_k")?;
    validate_positive_temperature(case.required_temperature_k, "required_temperature_k")?;
    case.minimum_approach_k.validate()?;

    if case.basis == ThermalApproachBasis::FiniteDrivingForce
        && case.minimum_approach_k.min.value() <= 0.0
    {
        return Err(OntologyError::InvalidRange(
            "minimum_approach_k_finite_driving_force",
        ));
    }

    let required_source_temperature_k = TemperatureRangeK::new(
        finite_sum(
            case.required_temperature_k.min.value(),
            case.minimum_approach_k.min.value(),
            "required_source_temperature_k",
        )?,
        finite_sum(
            case.required_temperature_k.max.value(),
            case.minimum_approach_k.max.value(),
            "required_source_temperature_k",
        )?,
    )?;
    validate_positive_temperature(
        required_source_temperature_k,
        "required_source_temperature_k",
    )?;

    let headroom_k = ThermalHeadroomRangeK::new(
        finite_difference3(
            case.source_temperature_k.min.value(),
            case.required_temperature_k.max.value(),
            case.minimum_approach_k.max.value(),
            "thermal_headroom_k",
        )?,
        finite_difference3(
            case.source_temperature_k.max.value(),
            case.required_temperature_k.min.value(),
            case.minimum_approach_k.min.value(),
            "thermal_headroom_k",
        )?,
    )?;

    let status = if headroom_k.min_k >= 0.0 {
        ThermalApproachStatus::Guaranteed
    } else if headroom_k.max_k < 0.0 {
        ThermalApproachStatus::Impossible
    } else {
        ThermalApproachStatus::Possible
    };

    Ok(ThermalApproachReport {
        status,
        headroom_k,
        required_source_temperature_k,
        basis: case.basis,
        supports_finite_driving_force_claim: case.basis
            == ThermalApproachBasis::FiniteDrivingForce
            && case.minimum_approach_k.min.value() > 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(min: f64, max: f64) -> TemperatureRangeK {
        TemperatureRangeK::new(min, max).unwrap()
    }

    fn exact(value: f64) -> TemperatureRangeK {
        t(value, value)
    }

    #[test]
    fn positive_headroom_is_guaranteed() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(520.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: exact(10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(report.status, ThermalApproachStatus::Guaranteed);
        assert_eq!(
            report.headroom_k,
            ThermalHeadroomRangeK {
                min_k: 10.0,
                max_k: 10.0
            }
        );
        assert_eq!(report.required_source_temperature_k, exact(510.0));
        assert!(report.supports_finite_driving_force_claim);
    }

    #[test]
    fn equal_source_and_demand_fail_positive_approach() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(500.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: exact(10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(report.status, ThermalApproachStatus::Impossible);
        assert_eq!(report.headroom_k.min_k, -10.0);
        assert_eq!(report.headroom_k.max_k, -10.0);
    }

    #[test]
    fn zero_approach_is_only_the_weaker_algebraic_claim() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(500.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: exact(0.0),
            basis: ThermalApproachBasis::AlgebraicEnvelope,
        })
        .unwrap();
        assert_eq!(report.status, ThermalApproachStatus::Guaranteed);
        assert!(!report.supports_finite_driving_force_claim);
    }

    #[test]
    fn uncertain_overlap_is_possible() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: t(505.0, 520.0),
            required_temperature_k: t(500.0, 510.0),
            minimum_approach_k: t(5.0, 10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(report.status, ThermalApproachStatus::Possible);
        assert_eq!(report.headroom_k.min_k, -15.0);
        assert_eq!(report.headroom_k.max_k, 15.0);
    }

    #[test]
    fn widening_uncertainty_cannot_strengthen_guaranteed_result() {
        let narrow = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(520.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: exact(10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        let wide = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: t(505.0, 525.0),
            required_temperature_k: t(495.0, 510.0),
            minimum_approach_k: t(5.0, 15.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(narrow.status, ThermalApproachStatus::Guaranteed);
        assert_eq!(wide.status, ThermalApproachStatus::Possible);
    }

    #[test]
    fn clearly_insufficient_source_is_impossible() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: t(450.0, 470.0),
            required_temperature_k: t(500.0, 510.0),
            minimum_approach_k: t(5.0, 10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(report.status, ThermalApproachStatus::Impossible);
    }

    #[test]
    fn zero_upper_headroom_boundary_is_possible_not_impossible() {
        let report = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: t(500.0, 510.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: exact(10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        })
        .unwrap();
        assert_eq!(report.headroom_k.min_k, -10.0);
        assert_eq!(report.headroom_k.max_k, 0.0);
        assert_eq!(report.status, ThermalApproachStatus::Possible);
    }

    #[test]
    fn zero_inclusive_finite_driving_force_approach_fails_closed() {
        let result = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(520.0),
            required_temperature_k: exact(500.0),
            minimum_approach_k: t(0.0, 10.0),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        });
        assert_eq!(
            result,
            Err(OntologyError::InvalidRange(
                "minimum_approach_k_finite_driving_force"
            ))
        );
    }

    #[test]
    fn derived_required_source_overflow_fails_closed() {
        let result = screen_thermal_approach(&ThermalApproachCase {
            source_temperature_k: exact(f64::MAX),
            required_temperature_k: exact(f64::MAX),
            minimum_approach_k: exact(f64::MAX),
            basis: ThermalApproachBasis::FiniteDrivingForce,
        });
        assert_eq!(
            result,
            Err(OntologyError::InvalidQuantity(
                "required_source_temperature_k"
            ))
        );
    }
}
