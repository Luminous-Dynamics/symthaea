// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bound-based capacitor screening that does not confuse statistical uncertainty
//! with guaranteed property bounds.
//!
//! The robust path is enabled only when the supplied relative uncertainties are
//! explicitly labelled hard bounds. It evaluates interval endpoints under an
//! ideal linear-dielectric model. It does not establish physical reliability,
//! probability of failure, manufacturing capability, or product qualification.

use crate::capacitor_screening::{
    CapacitorConstraintViolation, CapacitorScreeningDisposition, CapacitorScreeningError,
    CapacitorScreeningReport, CapacitorScreeningRequirements,
    screen_capacitor_candidate,
};
use crate::dielectric::{
    CapacitorGeometry, DielectricEvidenceKind, DielectricPropertyRecord,
    DielectricValidationError, IdealCapacitorMetrics, PropertyUncertainty,
    PropertyUncertaintySemantics, VACUUM_PERMITTIVITY_F_PER_M,
    calculate_ideal_capacitor_metrics,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// Closed interval derived from an explicitly documented hard relative bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarBounds {
    /// Conservative lower endpoint.
    pub lower: f64,
    /// Conservative upper endpoint.
    pub upper: f64,
}

/// Ideal capacitor metrics bounded using declared hard relative uncertainty intervals.
///
/// The bounds are conservative only with respect to the property intervals and
/// ideal equations declared here. Exact geometry, operating voltage and
/// frequency are assumed; manufacturing variation and model discrepancy are
/// not included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedCapacitorMetrics {
    /// Bounded ideal capacitance, in farads.
    pub capacitance_f: ScalarBounds,
    /// Bounded peak stored energy, in joules.
    pub peak_stored_energy_j: ScalarBounds,
    /// Bounded ideal linear-dielectric energy density at the applied voltage, in J/m³.
    pub ideal_energy_density_j_per_m3: ScalarBounds,
    /// Bounded sinusoidal dielectric-loss estimate, in watts.
    pub dielectric_loss_power_w: ScalarBounds,
    /// Bounded ideal theoretical breakdown-limited energy density, if hard bounds exist for permittivity and breakdown strength.
    pub ideal_breakdown_limited_energy_density_j_per_m3: Option<ScalarBounds>,
    /// Bounded breakdown-field ratio, if hard breakdown bounds exist and applied peak field is positive.
    pub breakdown_field_ratio: Option<ScalarBounds>,
    /// Exact active dielectric volume, in cubic metres; packaging/electrodes are excluded.
    pub active_dielectric_volume_m3: f64,
    /// Exact peak electric field from the supplied voltage and layer thickness, in V/m.
    pub peak_field_v_per_m: f64,
    /// Evidence class for the relative-permittivity and loss-tangent record.
    pub evidence_kind: DielectricEvidenceKind,
    /// Source reference for relative permittivity and loss tangent.
    pub source_reference: String,
}

/// Why a bound-based metric could not be evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustMetricLimitation {
    /// Property uncertainty is absent.
    MissingUncertainty,
    /// Uncertainty is statistical or a confidence interval, not a deterministic bound.
    UncertaintyIsNotAHardBound,
    /// Relative uncertainty around a zero loss estimate cannot express non-zero absolute loss.
    RelativeBoundAroundZeroLoss,
    /// Breakdown value exists but has no explicit hard uncertainty bound.
    BreakdownUncertaintyIsNotAHardBound,
}

/// Errors in bound-based metric calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustMetricError {
    /// Dielectric record, geometry or operating values are invalid.
    InvalidInput(DielectricValidationError),
    /// A required property uncertainty is absent or is not an explicit hard bound.
    UnusableUncertainty {
        /// Name of the property whose uncertainty cannot support robust bounds.
        property: &'static str,
        /// Reason the uncertainty is unusable.
        reason: RobustMetricLimitation,
    },
}

impl fmt::Display for RobustMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(error) => write!(f, "invalid robust capacitor metric input: {error}"),
            Self::UnusableUncertainty { property, reason } => {
                write!(f, "cannot calculate robust bound for {property}: {reason:?}")
            }
        }
    }
}

impl Error for RobustMetricError {}

impl From<DielectricValidationError> for RobustMetricError {
    fn from(value: DielectricValidationError) -> Self {
        Self::InvalidInput(value)
    }
}

fn relative_hard_bounds(
    estimate: f64,
    uncertainty: Option<&PropertyUncertainty>,
    property: &'static str,
) -> Result<ScalarBounds, RobustMetricError> {
    let uncertainty = uncertainty.ok_or(RobustMetricError::UnusableUncertainty {
        property,
        reason: RobustMetricLimitation::MissingUncertainty,
    })?;
    uncertainty
        .validate()
        .map_err(RobustMetricError::InvalidInput)?;
    if uncertainty.semantics != PropertyUncertaintySemantics::HardRelativeBound {
        return Err(RobustMetricError::UnusableUncertainty {
            property,
            reason: RobustMetricLimitation::UncertaintyIsNotAHardBound,
        });
    }
    let radius = uncertainty.relative_uncertainty;
    let bounds = ScalarBounds {
        lower: (estimate * (1.0 - radius)).max(0.0),
        upper: estimate * (1.0 + radius),
    };
    if !bounds.lower.is_finite() || !bounds.upper.is_finite() || bounds.upper < bounds.lower {
        return Err(RobustMetricError::InvalidInput(
            DielectricValidationError::NonFiniteCalculatedMetric,
        ));
    }
    Ok(bounds)
}

/// Calculate conservative ideal capacitor metric intervals from hard property bounds.
///
/// Requires hard relative bounds for relative permittivity and loss tangent. If
/// the record includes a breakdown value, its bound is included only when a hard
/// relative uncertainty is available; otherwise the breakdown-derived metrics
/// are returned as `None` and cannot qualify a minimum-breakdown constraint.
///
/// The function refuses to reinterpret standard uncertainties or confidence
/// intervals as deterministic bounds. `ac_rms_voltage_v` is used for sinusoidal
/// dielectric-loss estimation at the recorded test frequency.
pub fn calculate_bounded_capacitor_metrics(
    material: &DielectricPropertyRecord,
    geometry: &CapacitorGeometry,
    peak_voltage_v: f64,
    ac_rms_voltage_v: f64,
) -> Result<BoundedCapacitorMetrics, RobustMetricError> {
    material.validate()?;
    geometry.validate()?;
    if !peak_voltage_v.is_finite() || peak_voltage_v < 0.0 {
        return Err(DielectricValidationError::InvalidPeakVoltage.into());
    }
    if !ac_rms_voltage_v.is_finite() || ac_rms_voltage_v < 0.0 {
        return Err(DielectricValidationError::InvalidAcRmsVoltage.into());
    }
    if ac_rms_voltage_v > 0.0 && material.test_frequency_hz == 0.0 {
        return Err(
            DielectricValidationError::AcLossRequiresPositiveCharacterizationFrequency.into(),
        );
    }
    if material.loss_tangent == 0.0 {
        return Err(RobustMetricError::UnusableUncertainty {
            property: "loss_tangent",
            reason: RobustMetricLimitation::RelativeBoundAroundZeroLoss,
        });
    }

    let epsilon_bounds = relative_hard_bounds(
        material.relative_permittivity,
        material.permittivity_uncertainty.as_ref(),
        "relative_permittivity",
    )?;
    let loss_bounds = relative_hard_bounds(
        material.loss_tangent,
        material.loss_tangent_uncertainty.as_ref(),
        "loss_tangent",
    )?;

    let geometry_factor = geometry.electrode_overlap_area_m2
        * f64::from(geometry.active_dielectric_layers)
        / geometry.dielectric_thickness_m;
    let capacitance_factor = VACUUM_PERMITTIVITY_F_PER_M * geometry_factor;
    let capacitance_f = ScalarBounds {
        lower: capacitance_factor * epsilon_bounds.lower,
        upper: capacitance_factor * epsilon_bounds.upper,
    };

    let active_dielectric_volume_m3 = geometry.electrode_overlap_area_m2
        * geometry.dielectric_thickness_m
        * f64::from(geometry.active_dielectric_layers);
    let peak_field_v_per_m = peak_voltage_v / geometry.dielectric_thickness_m;
    let energy_factor = 0.5 * capacitance_factor * peak_voltage_v.powi(2);
    let peak_stored_energy_j = ScalarBounds {
        lower: energy_factor * epsilon_bounds.lower,
        upper: energy_factor * epsilon_bounds.upper,
    };
    let energy_density_factor = 0.5
        * VACUUM_PERMITTIVITY_F_PER_M
        * (peak_field_v_per_m.powi(2));
    let ideal_energy_density_j_per_m3 = ScalarBounds {
        lower: energy_density_factor * epsilon_bounds.lower,
        upper: energy_density_factor * epsilon_bounds.upper,
    };

    let loss_factor = 2.0 * PI * material.test_frequency_hz
        * capacitance_factor
        * ac_rms_voltage_v.powi(2);
    let dielectric_loss_power_w = ScalarBounds {
        lower: loss_factor * epsilon_bounds.lower * loss_bounds.lower,
        upper: loss_factor * epsilon_bounds.upper * loss_bounds.upper,
    };

    let breakdown_bounds = match (
        material.breakdown_strength_v_per_m,
        material.breakdown_strength_uncertainty.as_ref(),
    ) {
        (Some(estimate), Some(uncertainty))
            if uncertainty.semantics == PropertyUncertaintySemantics::HardRelativeBound =>
        {
            Some(relative_hard_bounds(
                estimate,
                Some(uncertainty),
                "breakdown_strength_v_per_m",
            )?)
        }
        _ => None,
    };

    let breakdown_field_ratio = breakdown_bounds.and_then(|breakdown| {
        (peak_field_v_per_m > 0.0).then_some(ScalarBounds {
            lower: breakdown.lower / peak_field_v_per_m,
            upper: breakdown.upper / peak_field_v_per_m,
        })
    });
    let ideal_breakdown_limited_energy_density_j_per_m3 =
        breakdown_bounds.map(|breakdown| ScalarBounds {
            lower: 0.5
                * VACUUM_PERMITTIVITY_F_PER_M
                * epsilon_bounds.lower
                * breakdown.lower.powi(2),
            upper: 0.5
                * VACUUM_PERMITTIVITY_F_PER_M
                * epsilon_bounds.upper
                * breakdown.upper.powi(2),
        });

    let scalars = [
        capacitance_f.lower,
        capacitance_f.upper,
        peak_stored_energy_j.lower,
        peak_stored_energy_j.upper,
        ideal_energy_density_j_per_m3.lower,
        ideal_energy_density_j_per_m3.upper,
        dielectric_loss_power_w.lower,
        dielectric_loss_power_w.upper,
        active_dielectric_volume_m3,
        peak_field_v_per_m,
    ];
    if scalars.into_iter().any(|value| !value.is_finite())
        || breakdown_field_ratio.is_some_and(|bounds| {
            !bounds.lower.is_finite() || !bounds.upper.is_finite()
        })
        || ideal_breakdown_limited_energy_density_j_per_m3.is_some_and(|bounds| {
            !bounds.lower.is_finite() || !bounds.upper.is_finite()
        })
    {
        return Err(DielectricValidationError::NonFiniteCalculatedMetric.into());
    }

    Ok(BoundedCapacitorMetrics {
        capacitance_f,
        peak_stored_energy_j,
        ideal_energy_density_j_per_m3,
        dielectric_loss_power_w,
        ideal_breakdown_limited_energy_density_j_per_m3,
        breakdown_field_ratio,
        active_dielectric_volume_m3,
        peak_field_v_per_m,
        evidence_kind: material.evidence_kind,
        source_reference: material.source_reference.clone(),
    })
}

/// Final disposition of a hard-bound capacitor screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustCapacitorScreeningDisposition {
    /// Every declared constraint is satisfied over the supplied hard property bounds.
    /// This remains model-conditional and is not physical qualification.
    PassesConservativeBounds,
    /// At least one declared constraint fails at a conservative bound.
    FailsConservativeBounds,
    /// Available evidence or uncertainty semantics cannot support a bounded decision.
    NotEvaluable,
}

/// Why a candidate cannot be robustly screened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustScreeningLimitation {
    /// The nominal screen itself could not evaluate the characterization/evidence conditions.
    NominalScreenNotEvaluable,
    /// Permittivity uncertainty is missing or is not a hard relative bound.
    PermittivityUncertaintyNotHardBound,
    /// Loss-tangent uncertainty is missing or is not a hard relative bound.
    LossTangentUncertaintyNotHardBound,
    /// A breakdown ratio constraint exists but breakdown uncertainty is not a hard relative bound.
    BreakdownUncertaintyNotHardBound,
    /// Required breakdown-derived interval could not be calculated (for example, applied field is zero).
    BreakdownIntervalUnavailable,
}

/// A declared requirement violated by conservative metric endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustCapacitorConstraintViolation {
    /// Conservative lower capacitance is below the required minimum.
    CapacitanceLowerBoundBelowMinimum,
    /// Conservative upper capacitance exceeds the allowed maximum.
    CapacitanceUpperBoundAboveMaximum,
    /// Deterministic active dielectric volume exceeds the allowed maximum.
    ActiveDielectricVolumeAboveMaximum,
    /// Conservative upper dielectric-loss estimate exceeds the allowed maximum.
    DielectricLossUpperBoundAboveMaximum,
    /// Conservative lower energy density is below the required minimum.
    EnergyDensityLowerBoundBelowMinimum,
    /// Conservative lower breakdown-field ratio is below the required minimum.
    BreakdownRatioLowerBoundBelowMinimum,
}

/// Auditable robust-screen report retaining the original nominal decision and exact bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCapacitorScreeningReport {
    /// Full nominal report, including exact input material, geometry, requirements and voltages.
    pub nominal_report: CapacitorScreeningReport,
    /// Conservative bounded-screen disposition.
    pub disposition: RobustCapacitorScreeningDisposition,
    /// Reasons a robust disposition could not be computed.
    pub limitations: Vec<RobustScreeningLimitation>,
    /// Conservative constraint violations.
    pub violations: Vec<RobustCapacitorConstraintViolation>,
    /// Conservative metric intervals, absent when required hard bounds are unavailable.
    pub bounded_metrics: Option<BoundedCapacitorMetrics>,
}

/// Error when a robust screening input contract is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustScreeningError {
    /// Nominal input validation or requirement validation failed.
    NominalInput(CapacitorScreeningError),
    /// A required hard uncertainty bound is unavailable or unusable.
    Bounds(RobustMetricError),
}

impl fmt::Display for RobustScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NominalInput(error) => write!(f, "robust capacitor screen input error: {error}"),
            Self::Bounds(error) => write!(f, "robust capacitor screen bound error: {error}"),
        }
    }
}

impl Error for RobustScreeningError {}

/// Screen a capacitor against worst-case endpoints of documented hard relative bounds.
///
/// This first robust evaluator is deliberately interval-based and assumes
/// independent property bounds. It is not a probabilistic yield/lifetime model.
/// The caller must separately account for geometry/process variation, correlations,
/// thermal behavior, parasitics and model discrepancy before making qualification claims.
pub fn screen_capacitor_candidate_robust(
    material: &DielectricPropertyRecord,
    geometry: &CapacitorGeometry,
    requirements: &CapacitorScreeningRequirements,
    peak_voltage_v: f64,
    ac_rms_voltage_v: f64,
) -> Result<RobustCapacitorScreeningReport, RobustScreeningError> {
    let nominal_report = screen_capacitor_candidate(
        material,
        geometry,
        requirements,
        peak_voltage_v,
        ac_rms_voltage_v,
    )
    .map_err(RobustScreeningError::NominalInput)?;

    if nominal_report.disposition == CapacitorScreeningDisposition::NotEvaluable {
        return Ok(RobustCapacitorScreeningReport {
            nominal_report,
            disposition: RobustCapacitorScreeningDisposition::NotEvaluable,
            limitations: vec![RobustScreeningLimitation::NominalScreenNotEvaluable],
            violations: Vec::new(),
            bounded_metrics: None,
        });
    }

    let mut limitations = Vec::new();
    let has_hard_bound = |uncertainty: Option<&PropertyUncertainty>| {
        uncertainty.is_some_and(|value| {
            value.semantics == PropertyUncertaintySemantics::HardRelativeBound
        })
    };
    if !has_hard_bound(material.permittivity_uncertainty.as_ref()) {
        limitations.push(RobustScreeningLimitation::PermittivityUncertaintyNotHardBound);
    }
    if !has_hard_bound(material.loss_tangent_uncertainty.as_ref())
        || material.loss_tangent == 0.0
    {
        limitations.push(RobustScreeningLimitation::LossTangentUncertaintyNotHardBound);
    }
    if requirements.min_breakdown_field_ratio.is_some()
        && !has_hard_bound(material.breakdown_strength_uncertainty.as_ref())
    {
        limitations.push(RobustScreeningLimitation::BreakdownUncertaintyNotHardBound);
    }
    if requirements.min_breakdown_field_ratio.is_some() && peak_voltage_v == 0.0 {
        limitations.push(RobustScreeningLimitation::BreakdownIntervalUnavailable);
    }
    if !limitations.is_empty() {
        return Ok(RobustCapacitorScreeningReport {
            nominal_report,
            disposition: RobustCapacitorScreeningDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            bounded_metrics: None,
        });
    }

    let bounds = calculate_bounded_capacitor_metrics(
        material,
        geometry,
        peak_voltage_v,
        ac_rms_voltage_v,
    )
    .map_err(RobustScreeningError::Bounds)?;

    let mut violations = Vec::new();
    if bounds.capacitance_f.lower < requirements.min_capacitance_f {
        violations.push(RobustCapacitorConstraintViolation::CapacitanceLowerBoundBelowMinimum);
    }
    if bounds.capacitance_f.upper > requirements.max_capacitance_f {
        violations.push(RobustCapacitorConstraintViolation::CapacitanceUpperBoundAboveMaximum);
    }
    if bounds.active_dielectric_volume_m3 > requirements.max_active_dielectric_volume_m3 {
        violations.push(RobustCapacitorConstraintViolation::ActiveDielectricVolumeAboveMaximum);
    }
    if bounds.dielectric_loss_power_w.upper > requirements.max_dielectric_loss_power_w {
        violations.push(RobustCapacitorConstraintViolation::DielectricLossUpperBoundAboveMaximum);
    }
    if requirements.min_ideal_energy_density_j_per_m3.is_some_and(|minimum| {
        bounds.ideal_energy_density_j_per_m3.lower < minimum
    }) {
        violations.push(RobustCapacitorConstraintViolation::EnergyDensityLowerBoundBelowMinimum);
    }
    if let Some(minimum_ratio) = requirements.min_breakdown_field_ratio {
        match bounds.breakdown_field_ratio {
            Some(ratio) if ratio.lower >= minimum_ratio => {}
            _ => violations.push(
                RobustCapacitorConstraintViolation::BreakdownRatioLowerBoundBelowMinimum,
            ),
        }
    }

    let disposition = if violations.is_empty() {
        RobustCapacitorScreeningDisposition::PassesConservativeBounds
    } else {
        RobustCapacitorScreeningDisposition::FailsConservativeBounds
    };

    Ok(RobustCapacitorScreeningReport {
        nominal_report,
        disposition,
        limitations,
        violations,
        bounded_metrics: Some(bounds),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacitor_screening::ScreeningLimitation;
    use crate::dielectric::{
        BreakdownTestKind, DielectricEvidenceKind, PropertyUncertaintySemantics,
    };

    fn hard_uncertainty(relative_uncertainty: f64) -> PropertyUncertainty {
        PropertyUncertainty {
            relative_uncertainty,
            semantics: PropertyUncertaintySemantics::HardRelativeBound,
            method_reference: "synthetic declared hard bound".to_string(),
        }
    }

    fn material() -> DielectricPropertyRecord {
        DielectricPropertyRecord {
            material_id: "synthetic-robust-screen-fixture".to_string(),
            relative_permittivity: 1_000.0,
            loss_tangent: 0.01,
            test_temperature_k: 298.15,
            test_frequency_hz: 1_000.0,
            test_field_v_per_m: Some(1.0e7),
            breakdown_strength_v_per_m: Some(1.0e8),
            breakdown_reference: Some("synthetic fixture only".to_string()),
            breakdown_test_temperature_k: Some(298.15),
            breakdown_test_kind: Some(BreakdownTestKind::DcRamp),
            breakdown_evidence_kind: Some(DielectricEvidenceKind::Experimental),
            evidence_kind: DielectricEvidenceKind::Experimental,
            source_reference: "test-fixture://not-real-data".to_string(),
            dataset_version: Some("synthetic-robust-v1".to_string()),
            permittivity_uncertainty: Some(hard_uncertainty(0.05)),
            loss_tangent_uncertainty: Some(hard_uncertainty(0.10)),
            breakdown_strength_uncertainty: Some(hard_uncertainty(0.10)),
        }
    }

    fn geometry() -> CapacitorGeometry {
        CapacitorGeometry {
            electrode_overlap_area_m2: 1.0e-4,
            dielectric_thickness_m: 1.0e-6,
            active_dielectric_layers: 10,
        }
    }

    fn requirements() -> CapacitorScreeningRequirements {
        CapacitorScreeningRequirements {
            operating_temperature_k: 298.15,
            max_temperature_deviation_k: 2.0,
            operating_frequency_hz: 1_000.0,
            max_relative_frequency_deviation: 0.01,
            max_relative_field_deviation: 0.01,
            min_capacitance_f: 5.0e-6,
            max_capacitance_f: 1.0e-3,
            max_active_dielectric_volume_m3: 2.0e-9,
            max_dielectric_loss_power_w: 0.01,
            min_ideal_energy_density_j_per_m3: Some(2.0e5),
            max_breakdown_temperature_deviation_k: 1.0,
            min_breakdown_field_ratio: Some(5.0),
            accepted_breakdown_test_kinds: vec![BreakdownTestKind::DcRamp],
            max_relative_permittivity_uncertainty: Some(0.1),
            max_relative_loss_tangent_uncertainty: Some(0.1),
            max_relative_breakdown_strength_uncertainty: Some(0.1),
            accepted_evidence_kinds: vec![DielectricEvidenceKind::Experimental],
        }
    }

    #[test]
    fn bounds_enclose_nominal_metrics_for_positive_uncertainties() {
        let sample = material();
        let geometry = geometry();
        let nominal =
            calculate_ideal_capacitor_metrics(&sample, &geometry, 10.0, 1.0).unwrap();
        let bounds =
            calculate_bounded_capacitor_metrics(&sample, &geometry, 10.0, 1.0).unwrap();
        assert!(bounds.capacitance_f.lower <= nominal.capacitance_f);
        assert!(bounds.capacitance_f.upper >= nominal.capacitance_f);
        assert!(bounds.dielectric_loss_power_w.lower <= nominal.dielectric_loss_power_w);
        assert!(bounds.dielectric_loss_power_w.upper >= nominal.dielectric_loss_power_w);
        assert!(bounds.breakdown_field_ratio.unwrap().lower <= nominal.breakdown_field_ratio.unwrap());
        assert!(bounds.breakdown_field_ratio.unwrap().upper >= nominal.breakdown_field_ratio.unwrap());
    }

    #[test]
    fn statistical_uncertainty_cannot_be_promoted_to_hard_bound() {
        let mut sample = material();
        sample.permittivity_uncertainty.as_mut().unwrap().semantics =
            PropertyUncertaintySemantics::StatisticalEstimate;
        assert!(matches!(
            calculate_bounded_capacitor_metrics(&sample, &geometry(), 10.0, 1.0),
            Err(RobustMetricError::UnusableUncertainty {
                property: "relative_permittivity",
                reason: RobustMetricLimitation::UncertaintyIsNotAHardBound
            })
        ));
    }

    #[test]
    fn uncertainty_interval_semantics_validate_coverage() {
        let mut sample = material();
        sample.loss_tangent_uncertainty.as_mut().unwrap().semantics =
            PropertyUncertaintySemantics::ConfidenceInterval {
                coverage_probability: 1.0,
            };
        assert_eq!(
            sample.validate(),
            Err(DielectricValidationError::InvalidUncertaintySemantics)
        );
    }

    #[test]
    fn robust_screen_rejects_when_nominal_pass_hides_bound_failure() {
        let mut req = requirements();
        req.max_dielectric_loss_power_w = 0.0006;
        // The nominal metric is near the limit; the bound-scaled upper loss exceeds it.
        let report = screen_capacitor_candidate_robust(
            &material(), &geometry(), &req, 10.0, 1.0
        ).unwrap();
        assert_eq!(
            report.disposition,
            RobustCapacitorScreeningDisposition::FailsConservativeBounds
        );
        assert!(report.violations.contains(
            &RobustCapacitorConstraintViolation::DielectricLossUpperBoundAboveMaximum
        ));
        assert!(report.bounded_metrics.is_some());
    }

    #[test]
    fn missing_breakdown_bound_makes_required_ratio_not_evaluable() {
        let mut sample = material();
        sample.breakdown_strength_uncertainty.as_mut().unwrap().semantics =
            PropertyUncertaintySemantics::StatisticalEstimate;
        let report = screen_capacitor_candidate_robust(
            &sample, &geometry(), &requirements(), 10.0, 1.0
        ).unwrap();
        assert_eq!(
            report.disposition,
            RobustCapacitorScreeningDisposition::NotEvaluable
        );
        assert!(report.limitations.contains(
            &RobustScreeningLimitation::BreakdownUncertaintyNotHardBound
        ));
        assert!(report.bounded_metrics.is_none());
    }

    #[test]
    fn robust_success_is_explicitly_model_conditional() {
        let mut req = requirements();
        req.max_dielectric_loss_power_w = 1.0;
        let report = screen_capacitor_candidate_robust(
            &material(), &geometry(), &req, 10.0, 1.0
        ).unwrap();
        assert_eq!(
            report.disposition,
            RobustCapacitorScreeningDisposition::PassesConservativeBounds
        );
        assert!(report.limitations.is_empty());
        assert!(report.bounded_metrics.is_some());
        assert_eq!(
            report.nominal_report.disposition,
            CapacitorScreeningDisposition::PassesNominalIdealizedScreen
        );
    }

    #[test]
    fn robust_screen_preserves_nominal_failure_but_can_still_compute_bounds() {
        let mut req = requirements();
        req.max_dielectric_loss_power_w = 0.0;
        let report = screen_capacitor_candidate_robust(
            &material(), &geometry(), &req, 10.0, 1.0
        ).unwrap();
        assert_eq!(
            report.nominal_report.disposition,
            CapacitorScreeningDisposition::FailsDeclaredConstraints
        );
        assert_eq!(
            report.disposition,
            RobustCapacitorScreeningDisposition::FailsConservativeBounds
        );
        assert!(report.bounded_metrics.is_some());
        assert!(report.violations.contains(
            &RobustCapacitorConstraintViolation::DielectricLossUpperBoundAboveMaximum
        ));
    }
}
