// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Application-conditioned, fail-closed screening for idealized capacitor candidates.
//!
//! A "pass" means only that the candidate satisfies the caller-supplied screening
//! constraints under an explicitly limited linear-dielectric model. It is not a
//! component qualification, safety assessment, lifetime prediction, or evidence
//! of manufactured performance.

use crate::dielectric::{
    BreakdownTestKind, CapacitorGeometry, DielectricEvidenceKind, DielectricPropertyRecord,
    DielectricValidationError, IdealCapacitorMetrics, calculate_ideal_capacitor_metrics,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Application-specific idealized screening constraints.
///
/// Characterization conditions are checked before any metric is used. This
/// implementation does not extrapolate permittivity or loss over temperature,
/// frequency, or electric field; candidates without condition-matched evidence
/// are reported as not evaluable rather than silently treated as equivalent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorScreeningRequirements {
    /// Intended operating temperature in kelvin.
    pub operating_temperature_k: f64,
    /// Maximum allowed absolute difference between characterization and operating temperature, in kelvin.
    pub max_temperature_deviation_k: f64,
    /// Intended AC operating frequency in hertz. Zero means DC/quasi-static screening.
    pub operating_frequency_hz: f64,
    /// Maximum relative difference between characterization and operating frequency.
    /// When the target frequency is zero, only zero-frequency characterization is accepted.
    pub max_relative_frequency_deviation: f64,
    /// Maximum relative difference between characterized and calculated operating electric field.
    pub max_relative_field_deviation: f64,
    /// Minimum acceptable ideal capacitance in farads.
    pub min_capacitance_f: f64,
    /// Maximum acceptable ideal capacitance in farads.
    pub max_capacitance_f: f64,
    /// Maximum active dielectric volume in cubic metres. This excludes electrodes, packaging, margins, and terminals.
    pub max_active_dielectric_volume_m3: f64,
    /// Maximum ideal sinusoidal dielectric-loss power in watts.
    pub max_dielectric_loss_power_w: f64,
    /// Optional minimum ideal energy density at the supplied applied voltage, in J/m³.
    pub min_ideal_energy_density_j_per_m3: Option<f64>,
    /// Optional maximum absolute difference between breakdown-test temperature and the dielectric-property test temperature, in kelvin.
    pub max_breakdown_temperature_deviation_k: f64,
    /// Optional minimum ratio of reported breakdown field to operating field.
    /// This is a screening threshold only, not a safe operating margin.
    pub min_breakdown_field_ratio: Option<f64>,
    /// Allowed breakdown test protocol families. Breakdown-derived metrics are ineligible if the observed method is not listed.
    pub accepted_breakdown_test_kinds: Vec<BreakdownTestKind>,
    /// Optional maximum allowed relative-permittivity uncertainty; missing uncertainty is not accepted when set.
    pub max_relative_permittivity_uncertainty: Option<f64>,
    /// Optional maximum allowed loss-tangent uncertainty; missing uncertainty is not accepted when set.
    pub max_relative_loss_tangent_uncertainty: Option<f64>,
    /// Optional maximum allowed breakdown-strength uncertainty; missing uncertainty is not accepted when set.
    pub max_relative_breakdown_strength_uncertainty: Option<f64>,
    /// Evidence classes permitted for this screening task. This is an exact allow-list, not an ordinal trust score.
    pub accepted_evidence_kinds: Vec<DielectricEvidenceKind>,
}

impl CapacitorScreeningRequirements {
    /// Validate the application constraint contract.
    pub fn validate(&self) -> Result<(), CapacitorScreeningError> {
        if !self.operating_temperature_k.is_finite() || self.operating_temperature_k <= 0.0 {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "operating temperature must be finite and above absolute zero",
            ));
        }
        if !self.max_temperature_deviation_k.is_finite() || self.max_temperature_deviation_k < 0.0 {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum temperature deviation must be finite and non-negative",
            ));
        }
        if !self.operating_frequency_hz.is_finite() || self.operating_frequency_hz < 0.0 {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "operating frequency must be finite and non-negative",
            ));
        }
        if !self.max_relative_frequency_deviation.is_finite()
            || self.max_relative_frequency_deviation < 0.0
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum relative frequency deviation must be finite and non-negative",
            ));
        }
        if !self.max_relative_field_deviation.is_finite()
            || self.max_relative_field_deviation < 0.0
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum relative field deviation must be finite and non-negative",
            ));
        }
        if !self.min_capacitance_f.is_finite() || self.min_capacitance_f <= 0.0 {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "minimum capacitance must be finite and positive",
            ));
        }
        if !self.max_capacitance_f.is_finite() || self.max_capacitance_f < self.min_capacitance_f {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum capacitance must be finite and at least the minimum",
            ));
        }
        if !self.max_active_dielectric_volume_m3.is_finite()
            || self.max_active_dielectric_volume_m3 <= 0.0
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum active dielectric volume must be finite and positive",
            ));
        }
        if !self.max_dielectric_loss_power_w.is_finite()
            || self.max_dielectric_loss_power_w < 0.0
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum dielectric-loss power must be finite and non-negative",
            ));
        }
        if self
            .min_ideal_energy_density_j_per_m3
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "minimum ideal energy density must be finite and non-negative",
            ));
        }
        if !self.max_breakdown_temperature_deviation_k.is_finite()
            || self.max_breakdown_temperature_deviation_k < 0.0
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "maximum breakdown-temperature deviation must be finite and non-negative",
            ));
        }
        if self
            .min_breakdown_field_ratio
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "minimum breakdown field ratio must be finite and non-negative",
            ));
        }
        if self.min_breakdown_field_ratio.is_some() && self.accepted_breakdown_test_kinds.is_empty() {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "a breakdown ratio constraint requires an explicit allow-list of breakdown test protocols",
            ));
        }
        for uncertainty_limit in [
            self.max_relative_permittivity_uncertainty,
            self.max_relative_loss_tangent_uncertainty,
            self.max_relative_breakdown_strength_uncertainty,
        ]
        .into_iter()
        .flatten()
        {
            if !uncertainty_limit.is_finite() || uncertainty_limit < 0.0 {
                return Err(CapacitorScreeningError::InvalidRequirements(
                    "property-specific uncertainty limits must be finite and non-negative",
                ));
            }
        }
        if self.accepted_evidence_kinds.is_empty() {
            return Err(CapacitorScreeningError::InvalidRequirements(
                "at least one evidence class must be explicitly allowed",
            ));
        }
        Ok(())
    }
}

/// Final disposition of an idealized candidate screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorScreeningDisposition {
    /// Nominal idealized metrics meet the declared constraints and configured evidence gates.
    /// Input uncertainties are thresholded but are not propagated into conservative bounds;
    /// this is not a robust-feasibility guarantee, product qualification, or safety pass.
    PassesNominalIdealizedScreen,
    /// One or more supplied idealized constraints were violated.
    FailsDeclaredConstraints,
    /// A required evidence class, characterization condition, or uncertainty is absent/inapplicable.
    NotEvaluable,
}

/// Reasons a candidate could not be fairly evaluated under the declared application conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScreeningLimitation {
    /// Characterization temperature is outside the explicitly allowed window.
    TemperatureOutsideCharacterization,
    /// Characterization frequency is outside the explicitly allowed window.
    FrequencyOutsideCharacterization,
    /// No field characterization exists for a nonzero operating field.
    MissingFieldCharacterization,
    /// Characterization field is too different from the calculated operating field.
    FieldOutsideCharacterization,
    /// Evidence class was not included in the application's allow-list.
    EvidenceClassNotAllowed,
    /// Permittivity uncertainty was required but not supplied.
    MissingPermittivityUncertainty,
    /// Permittivity uncertainty exceeds the requested limit.
    PermittivityUncertaintyAboveLimit,
    /// Loss-tangent uncertainty was required but not supplied.
    MissingLossTangentUncertainty,
    /// Loss-tangent uncertainty exceeds the requested limit.
    LossTangentUncertaintyAboveLimit,
    /// Breakdown-strength uncertainty was required but not supplied.
    MissingBreakdownUncertainty,
    /// Breakdown-strength uncertainty exceeds the requested limit.
    BreakdownUncertaintyAboveLimit,
    /// A breakdown ratio was required but no referenced breakdown evidence is available.
    MissingBreakdownEvidence,
    /// Breakdown and dielectric-property characterization temperatures differ too much to combine.
    BreakdownTemperatureMismatch,
    /// Breakdown test protocol family was not included in the application's allow-list.
    BreakdownTestKindNotAllowed,
    /// Breakdown evidence class was not included in the application's allow-list.
    BreakdownEvidenceClassNotAllowed,
}

/// A declared idealized requirement that the candidate violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorConstraintViolation {
    /// Calculated ideal capacitance is below the specified minimum.
    CapacitanceBelowMinimum,
    /// Calculated ideal capacitance exceeds the specified maximum.
    CapacitanceAboveMaximum,
    /// Ideal active dielectric volume exceeds the specified maximum.
    ActiveDielectricVolumeAboveMaximum,
    /// Ideal sinusoidal dielectric-loss estimate exceeds the specified maximum.
    DielectricLossAboveMaximum,
    /// Ideal linear-dielectric energy density is below the optional minimum.
    IdealEnergyDensityBelowMinimum,
    /// Reported breakdown-field ratio is below the optional screening threshold.
    BreakdownFieldRatioBelowMinimum,
}

/// Evidence-aware screening report for one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorScreeningReport {
    /// Stable candidate/material identifier.
    pub candidate_id: String,
    /// Exact dielectric-property record used for screening, including provenance and conditions.
    pub material_record: DielectricPropertyRecord,
    /// Exact idealized layer geometry used in calculations.
    pub geometry: CapacitorGeometry,
    /// Exact application requirements used to make the disposition.
    pub requirements: CapacitorScreeningRequirements,
    /// Peak terminal voltage used in the calculation, in volts.
    pub peak_voltage_v: f64,
    /// AC RMS voltage used for the sinusoidal dielectric-loss estimate, in volts.
    pub ac_rms_voltage_v: f64,
    /// Result category, carefully scoped to the idealized model.
    pub disposition: CapacitorScreeningDisposition,
    /// Reasons the candidate was not evaluable. Empty unless disposition is NotEvaluable.
    pub limitations: Vec<ScreeningLimitation>,
    /// Failed hard screening constraints. Empty unless disposition is FailsDeclaredConstraints.
    pub violations: Vec<CapacitorConstraintViolation>,
    /// Calculated ideal metrics, absent if the material characterization did not cover the operating conditions.
    pub metrics: Option<IdealCapacitorMetrics>,
}

/// Errors caused by invalid application contracts or invalid physical inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapacitorScreeningError {
    /// An application-level requirement is malformed.
    InvalidRequirements(&'static str),
    /// A dielectric record, geometry, or supplied operating value is invalid.
    InvalidInput(DielectricValidationError),
}

impl fmt::Display for CapacitorScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequirements(message) => write!(f, "invalid capacitor screening requirements: {message}"),
            Self::InvalidInput(error) => write!(f, "invalid capacitor screening input: {error}"),
        }
    }
}

impl Error for CapacitorScreeningError {}

impl From<DielectricValidationError> for CapacitorScreeningError {
    fn from(value: DielectricValidationError) -> Self {
        Self::InvalidInput(value)
    }
}

/// Screen a candidate using application constraints and condition-matched properties.
///
/// The disposition is nominal rather than risk-bounded: uncertainty gates enforce
/// evidence-quality thresholds but do not transform uncertainty into worst-case
/// predictions or probabilities of constraint violation.
///
/// The function refuses to compute a pass/fail disposition when characterization
/// does not cover the specified temperature, frequency, or operating field.
/// `peak_voltage_v` drives the ideal peak-field/energy calculation and
/// `ac_rms_voltage_v` drives the sinusoidal dielectric-loss estimate.
/// For nonlinear dielectrics, real waveforms, ESR/ESL, package parasitics, thermal
/// rise, and reliability qualification, use appropriate measured models and solvers.
pub fn screen_capacitor_candidate(
    material: &DielectricPropertyRecord,
    geometry: &CapacitorGeometry,
    requirements: &CapacitorScreeningRequirements,
    peak_voltage_v: f64,
    ac_rms_voltage_v: f64,
) -> Result<CapacitorScreeningReport, CapacitorScreeningError> {
    requirements.validate()?;
    material.validate()?;
    geometry.validate()?;
    if !peak_voltage_v.is_finite() || peak_voltage_v < 0.0 {
        return Err(CapacitorScreeningError::InvalidInput(
            DielectricValidationError::InvalidPeakVoltage,
        ));
    }
    if !ac_rms_voltage_v.is_finite() || ac_rms_voltage_v < 0.0 {
        return Err(CapacitorScreeningError::InvalidInput(
            DielectricValidationError::InvalidAcRmsVoltage,
        ));
    }
    if requirements.operating_frequency_hz == 0.0 && ac_rms_voltage_v > 0.0 {
        return Err(CapacitorScreeningError::InvalidInput(
            DielectricValidationError::AcLossRequiresPositiveCharacterizationFrequency,
        ));
    }

    let mut limitations = Vec::new();

    if (material.test_temperature_k - requirements.operating_temperature_k).abs()
        > requirements.max_temperature_deviation_k
    {
        limitations.push(ScreeningLimitation::TemperatureOutsideCharacterization);
    }

    if requirements.operating_frequency_hz == 0.0 {
        if material.test_frequency_hz != 0.0 {
            limitations.push(ScreeningLimitation::FrequencyOutsideCharacterization);
        }
    } else {
        let frequency_delta = (material.test_frequency_hz - requirements.operating_frequency_hz).abs()
            / requirements.operating_frequency_hz;
        if frequency_delta > requirements.max_relative_frequency_deviation {
            limitations.push(ScreeningLimitation::FrequencyOutsideCharacterization);
        }
    }

    let operating_field = peak_voltage_v / geometry.dielectric_thickness_m;
    if !operating_field.is_finite() {
        return Err(CapacitorScreeningError::InvalidInput(
            DielectricValidationError::NonFiniteCalculatedMetric,
        ));
    }
    if operating_field > 0.0 {
        match material.test_field_v_per_m {
            None => limitations.push(ScreeningLimitation::MissingFieldCharacterization),
            Some(test_field) if test_field == 0.0 => {
                limitations.push(ScreeningLimitation::MissingFieldCharacterization)
            }
            Some(test_field) => {
                let field_delta = (test_field - operating_field).abs() / operating_field;
                if field_delta > requirements.max_relative_field_deviation {
                    limitations.push(ScreeningLimitation::FieldOutsideCharacterization);
                }
            }
        }
    }

    if !requirements.accepted_evidence_kinds.contains(&material.evidence_kind) {
        limitations.push(ScreeningLimitation::EvidenceClassNotAllowed);
    }

    if let Some(max_uncertainty) = requirements.max_relative_permittivity_uncertainty {
        match material.permittivity_uncertainty.as_ref() {
            None => limitations.push(ScreeningLimitation::MissingPermittivityUncertainty),
            Some(value) if value.relative_uncertainty > max_uncertainty => {
                limitations.push(ScreeningLimitation::PermittivityUncertaintyAboveLimit)
            }
            Some(_) => {}
        }
    }
    if let Some(max_uncertainty) = requirements.max_relative_loss_tangent_uncertainty {
        match material.loss_tangent_uncertainty.as_ref() {
            None => limitations.push(ScreeningLimitation::MissingLossTangentUncertainty),
            Some(value) if value.relative_uncertainty > max_uncertainty => {
                limitations.push(ScreeningLimitation::LossTangentUncertaintyAboveLimit)
            }
            Some(_) => {}
        }
    }
    if let Some(max_uncertainty) = requirements.max_relative_breakdown_strength_uncertainty {
        match material.breakdown_strength_uncertainty.as_ref() {
            None => limitations.push(ScreeningLimitation::MissingBreakdownUncertainty),
            Some(value) if value.relative_uncertainty > max_uncertainty => {
                limitations.push(ScreeningLimitation::BreakdownUncertaintyAboveLimit)
            }
            Some(_) => {}
        }
    }

    if let Some(breakdown_strength) = material.breakdown_strength_v_per_m {
        if material
            .breakdown_evidence_kind
            .is_some_and(|kind| !requirements.accepted_evidence_kinds.contains(&kind))
        {
            // The report includes breakdown-derived metrics, so their evidence class
            // must satisfy the allow-list even when no minimum-ratio constraint is requested.
            limitations.push(ScreeningLimitation::BreakdownEvidenceClassNotAllowed);
        }
        if material
            .breakdown_test_kind
            .is_some_and(|kind| !requirements.accepted_breakdown_test_kinds.contains(&kind))
        {
            limitations.push(ScreeningLimitation::BreakdownTestKindNotAllowed);
        }
        if material
            .breakdown_test_temperature_k
            .is_some_and(|temperature| {
                (temperature - material.test_temperature_k).abs()
                    > requirements.max_breakdown_temperature_deviation_k
            })
        {
            limitations.push(ScreeningLimitation::BreakdownTemperatureMismatch);
        }
        if requirements.min_breakdown_field_ratio.is_some()
            && (material.breakdown_reference.as_deref().is_none_or(|reference| reference.trim().is_empty())
                || material.breakdown_evidence_kind.is_none()
                || material.breakdown_test_kind.is_none())
        {
            limitations.push(ScreeningLimitation::MissingBreakdownEvidence);
        }
        let _ = breakdown_strength;
    } else if requirements.min_breakdown_field_ratio.is_some() {
        limitations.push(ScreeningLimitation::MissingBreakdownEvidence);
    }

    if !limitations.is_empty() {
        return Ok(CapacitorScreeningReport {
            candidate_id: material.material_id.clone(),
            material_record: material.clone(),
            geometry: *geometry,
            requirements: requirements.clone(),
            peak_voltage_v,
            ac_rms_voltage_v,
            disposition: CapacitorScreeningDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            metrics: None,
        });
    }

    let metrics = calculate_ideal_capacitor_metrics(
        material,
        geometry,
        peak_voltage_v,
        ac_rms_voltage_v,
    )?;

    let mut violations = Vec::new();
    if metrics.capacitance_f < requirements.min_capacitance_f {
        violations.push(CapacitorConstraintViolation::CapacitanceBelowMinimum);
    }
    if metrics.capacitance_f > requirements.max_capacitance_f {
        violations.push(CapacitorConstraintViolation::CapacitanceAboveMaximum);
    }
    if metrics.active_dielectric_volume_m3 > requirements.max_active_dielectric_volume_m3 {
        violations.push(CapacitorConstraintViolation::ActiveDielectricVolumeAboveMaximum);
    }
    if metrics.dielectric_loss_power_w > requirements.max_dielectric_loss_power_w {
        violations.push(CapacitorConstraintViolation::DielectricLossAboveMaximum);
    }
    if requirements
        .min_ideal_energy_density_j_per_m3
        .is_some_and(|minimum| metrics.ideal_energy_density_j_per_m3 < minimum)
    {
        violations.push(CapacitorConstraintViolation::IdealEnergyDensityBelowMinimum);
    }
    if let Some(minimum_ratio) = requirements.min_breakdown_field_ratio {
        if metrics
            .breakdown_field_ratio
            .is_none_or(|ratio| ratio < minimum_ratio)
        {
            violations.push(CapacitorConstraintViolation::BreakdownFieldRatioBelowMinimum);
        }
    }

    let disposition = if violations.is_empty() {
        CapacitorScreeningDisposition::PassesNominalIdealizedScreen
    } else {
        CapacitorScreeningDisposition::FailsDeclaredConstraints
    };

    Ok(CapacitorScreeningReport {
        candidate_id: material.material_id.clone(),
        material_record: material.clone(),
        geometry: *geometry,
        requirements: requirements.clone(),
        peak_voltage_v,
        ac_rms_voltage_v,
        disposition,
        limitations: Vec::new(),
        violations,
        metrics: Some(metrics),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dielectric::VACUUM_PERMITTIVITY_F_PER_M;

    fn material() -> DielectricPropertyRecord {
        DielectricPropertyRecord {
            material_id: "synthetic-screen-fixture".to_string(),
            relative_permittivity: 1_000.0,
            loss_tangent: 0.01,
            test_temperature_k: 298.15,
            test_frequency_hz: 1_000.0,
            test_field_v_per_m: Some(1.0e7),
            breakdown_strength_v_per_m: Some(1.0e8),
            breakdown_reference: Some("synthetic test fixture only".to_string()),
            breakdown_test_temperature_k: Some(298.15),
            breakdown_test_kind: Some(BreakdownTestKind::DcRamp),
            breakdown_evidence_kind: Some(DielectricEvidenceKind::Estimated),
            evidence_kind: DielectricEvidenceKind::Estimated,
            source_reference: "test-fixture://not-real-data".to_string(),
            dataset_version: Some("synthetic-v1".to_string()),
            permittivity_uncertainty: Some(crate::dielectric::PropertyUncertainty {
                relative_uncertainty: 0.05,
                semantics: crate::dielectric::PropertyUncertaintySemantics::HardRelativeBound,
                method_reference: "synthetic uncertainty fixture (hard relative bound)".to_string(),
            }),
            loss_tangent_uncertainty: Some(crate::dielectric::PropertyUncertainty {
                relative_uncertainty: 0.05,
                semantics: crate::dielectric::PropertyUncertaintySemantics::HardRelativeBound,
                method_reference: "synthetic uncertainty fixture (hard relative bound)".to_string(),
            }),
            breakdown_strength_uncertainty: Some(crate::dielectric::PropertyUncertainty {
                relative_uncertainty: 0.05,
                semantics: crate::dielectric::PropertyUncertaintySemantics::HardRelativeBound,
                method_reference: "synthetic uncertainty fixture (hard relative bound)".to_string(),
            }),
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
            min_capacitance_f: 1.0e-5,
            max_capacitance_f: 1.0e-3,
            max_active_dielectric_volume_m3: 2.0e-9,
            max_dielectric_loss_power_w: 0.01,
            min_ideal_energy_density_j_per_m3: Some(1.0e6),
            max_breakdown_temperature_deviation_k: 1.0,
            min_breakdown_field_ratio: Some(5.0),
            accepted_breakdown_test_kinds: vec![BreakdownTestKind::DcRamp],
            max_relative_permittivity_uncertainty: Some(0.1),
            max_relative_loss_tangent_uncertainty: Some(0.1),
            max_relative_breakdown_strength_uncertainty: Some(0.1),
            accepted_evidence_kinds: vec![DielectricEvidenceKind::Estimated],
        }
    }

    #[test]
    fn condition_matched_candidate_passes_only_the_idealized_screen() {
        let report =
            screen_capacitor_candidate(&material(), &geometry(), &requirements(), 10.0, 1.0)
                .unwrap();
        assert_eq!(
            report.disposition,
            CapacitorScreeningDisposition::PassesNominalIdealizedScreen
        );
        assert!(report.violations.is_empty());
        assert!(report.limitations.is_empty());
        assert!(report.metrics.is_some());
    }

    #[test]
    fn off_condition_property_data_are_not_silently_extrapolated() {
        let mut sample = material();
        sample.test_temperature_k = 350.0;
        sample.test_frequency_hz = 100_000.0;
        sample.test_field_v_per_m = None;
        let report =
            screen_capacitor_candidate(&sample, &geometry(), &requirements(), 10.0, 1.0)
                .unwrap();
        assert_eq!(report.disposition, CapacitorScreeningDisposition::NotEvaluable);
        assert!(report.metrics.is_none());
        assert!(report.limitations.contains(&ScreeningLimitation::TemperatureOutsideCharacterization));
        assert!(report.limitations.contains(&ScreeningLimitation::FrequencyOutsideCharacterization));
        assert!(report.limitations.contains(&ScreeningLimitation::MissingFieldCharacterization));
    }

    #[test]
    fn hard_constraint_violations_are_separate_from_missing_evidence() {
        let mut req = requirements();
        req.max_dielectric_loss_power_w = 0.0;
        let report =
            screen_capacitor_candidate(&material(), &geometry(), &req, 10.0, 1.0).unwrap();
        assert_eq!(
            report.disposition,
            CapacitorScreeningDisposition::FailsDeclaredConstraints
        );
        assert!(report.metrics.is_some());
        assert!(report.limitations.is_empty());
        assert!(report.violations.contains(&CapacitorConstraintViolation::DielectricLossAboveMaximum));
    }

    #[test]
    fn evidence_and_uncertainty_gates_fail_closed() {
        let mut req = requirements();
        req.accepted_evidence_kinds = vec![DielectricEvidenceKind::Experimental];
        req.max_relative_permittivity_uncertainty = Some(0.01);
        req.max_relative_loss_tangent_uncertainty = Some(0.01);
        req.max_relative_breakdown_strength_uncertainty = Some(0.01);
        let mut sample = material();
        sample.permittivity_uncertainty = None;
        sample.loss_tangent_uncertainty = None;
        let report =
            screen_capacitor_candidate(&sample, &geometry(), &req, 10.0, 1.0).unwrap();
        assert_eq!(report.disposition, CapacitorScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(&ScreeningLimitation::EvidenceClassNotAllowed));
        assert!(report.limitations.contains(&ScreeningLimitation::BreakdownEvidenceClassNotAllowed));
        assert!(report.limitations.contains(&ScreeningLimitation::MissingPermittivityUncertainty));
        assert!(report.limitations.contains(&ScreeningLimitation::MissingLossTangentUncertainty));
        assert!(report.limitations.contains(&ScreeningLimitation::BreakdownUncertaintyAboveLimit));
    }

    #[test]
    fn invalid_requirements_are_rejected() {
        let mut req = requirements();
        req.min_capacitance_f = 2.0;
        req.max_capacitance_f = 1.0;
        assert!(matches!(
            req.validate(),
            Err(CapacitorScreeningError::InvalidRequirements(_))
        ));

        let mut req = requirements();
        req.accepted_breakdown_test_kinds.clear();
        assert!(matches!(
            req.validate(),
            Err(CapacitorScreeningError::InvalidRequirements(_))
        ));
    }

    #[test]
    fn breakdown_evidence_must_match_temperature_and_protocol_contract() {
        let mut sample = material();
        sample.breakdown_test_temperature_k = Some(330.0);
        sample.breakdown_test_kind = Some(BreakdownTestKind::AcSinusoidal);
        let report =
            screen_capacitor_candidate(&sample, &geometry(), &requirements(), 10.0, 1.0)
                .unwrap();
        assert_eq!(report.disposition, CapacitorScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(&ScreeningLimitation::BreakdownTemperatureMismatch));
        assert!(report.limitations.contains(&ScreeningLimitation::BreakdownTestKindNotAllowed));
        assert!(report.metrics.is_none());
    }

    #[test]
    fn ideal_energy_density_formula_is_consistent_with_capacitance() {
        let report =
            screen_capacitor_candidate(&material(), &geometry(), &requirements(), 10.0, 1.0)
                .unwrap();
        let metrics = report.metrics.unwrap();
        let expected = 0.5 * VACUUM_PERMITTIVITY_F_PER_M * 1_000.0 * 1.0e14;
        assert!((metrics.ideal_energy_density_j_per_m3 - expected).abs() <= expected * 1.0e-12);
    }
}
