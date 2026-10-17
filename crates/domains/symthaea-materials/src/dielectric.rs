// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware dielectric records and first-order capacitor calculations.
//!
//! This module deliberately provides *equations*, not a dielectric-property
//! predictor. Property records must retain their source and characterization
//! conditions. Calculated metrics are ideal linear-dielectric screening values;
//! they are not component ratings, lifetime predictions, or safety approvals.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// Vacuum permittivity in farads per metre (CODATA rounded value).
pub const VACUUM_PERMITTIVITY_F_PER_M: f64 = 8.854_187_812_8e-12;

/// How the recorded dielectric properties were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DielectricEvidenceKind {
    /// Directly measured material/device characterization.
    Experimental,
    /// Manufacturer-provided component or material datasheet.
    ManufacturerDatasheet,
    /// First-principles or other explicitly identified numerical calculation.
    FirstPrinciples,
    /// A fitted or otherwise empirical predictive model.
    EmpiricalModel,
    /// An estimate without a validated domain-specific model.
    Estimated,
}

/// Broad protocol family for dielectric breakdown characterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreakdownTestKind {
    /// Slowly ramped direct-voltage breakdown test.
    DcRamp,
    /// Sinusoidal alternating-voltage breakdown test.
    AcSinusoidal,
    /// Impulse or surge breakdown test.
    Impulse,
    /// Another explicitly documented protocol.
    Other,
}

/// Uncertainty attached to one individual property rather than to the material record as a whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyUncertainty {
    /// Relative uncertainty as a fraction (e.g. 0.05 means 5%).
    pub relative_uncertainty: f64,
    /// Method/definition reference explaining how the uncertainty was calculated.
    pub method_reference: String,
}

impl PropertyUncertainty {
    /// Ensure uncertainty magnitude and its interpretation reference are present.
    pub fn validate(&self) -> Result<(), DielectricValidationError> {
        if !self.relative_uncertainty.is_finite() || self.relative_uncertainty < 0.0 {
            return Err(DielectricValidationError::InvalidRelativeUncertainty);
        }
        if self.method_reference.trim().is_empty() {
            return Err(DielectricValidationError::MissingUncertaintyMethodReference);
        }
        Ok(())
    }
}

/// Dielectric properties measured or calculated at one explicitly recorded point.
///
/// The permittivity and loss tangent share the record's characterization
/// temperature, frequency, and field conditions. Breakdown strength may come
/// from a separate protocol; its own source, evidence class, and uncertainty are
/// therefore stored separately. This type never interpolates or extrapolates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DielectricPropertyRecord {
    /// Stable identifier for the material/composition and, where relevant, processing state.
    pub material_id: String,
    /// Relative permittivity (dimensionless) at the stated test conditions.
    pub relative_permittivity: f64,
    /// Loss tangent (dimensionless, non-negative) at the stated test conditions.
    pub loss_tangent: f64,
    /// Temperature in kelvin at which permittivity and loss tangent apply.
    pub test_temperature_k: f64,
    /// Frequency in hertz at which permittivity and loss tangent apply; zero denotes DC/quasi-static data.
    pub test_frequency_hz: f64,
    /// Optional field strength used to characterize permittivity/loss, in V/m.
    pub test_field_v_per_m: Option<f64>,
    /// Optional reported breakdown field in V/m; not a guaranteed design limit.
    pub breakdown_strength_v_per_m: Option<f64>,
    /// Source/protocol reference for the breakdown value, including specimen/test conditions.
    pub breakdown_reference: Option<String>,
    /// Temperature in kelvin at which breakdown strength was characterized.
    pub breakdown_test_temperature_k: Option<f64>,
    /// Broad protocol family used to characterize breakdown strength.
    pub breakdown_test_kind: Option<BreakdownTestKind>,
    /// Evidence class for the reported breakdown value, when present.
    pub breakdown_evidence_kind: Option<DielectricEvidenceKind>,
    /// Evidence class for relative permittivity and loss tangent.
    pub evidence_kind: DielectricEvidenceKind,
    /// DOI, URL, datasheet ID, dataset row, or another stable provenance reference for permittivity/loss.
    pub source_reference: String,
    /// Optional source dataset/release/version identifier.
    pub dataset_version: Option<String>,
    /// Optional relative-permittivity uncertainty with its calculation method.
    pub permittivity_uncertainty: Option<PropertyUncertainty>,
    /// Optional loss-tangent uncertainty with its calculation method.
    pub loss_tangent_uncertainty: Option<PropertyUncertainty>,
    /// Optional breakdown-strength uncertainty with its calculation method.
    pub breakdown_strength_uncertainty: Option<PropertyUncertainty>,
}

impl DielectricPropertyRecord {
    /// Validate values and minimum provenance requirements.
    pub fn validate(&self) -> Result<(), DielectricValidationError> {
        if self.material_id.trim().is_empty() {
            return Err(DielectricValidationError::EmptyMaterialId);
        }
        if !self.relative_permittivity.is_finite() || self.relative_permittivity <= 0.0 {
            return Err(DielectricValidationError::InvalidRelativePermittivity);
        }
        if !self.loss_tangent.is_finite() || self.loss_tangent < 0.0 {
            return Err(DielectricValidationError::InvalidLossTangent);
        }
        if !self.test_temperature_k.is_finite() || self.test_temperature_k <= 0.0 {
            return Err(DielectricValidationError::InvalidTestTemperature);
        }
        if !self.test_frequency_hz.is_finite() || self.test_frequency_hz < 0.0 {
            return Err(DielectricValidationError::InvalidTestFrequency);
        }
        if self
            .test_field_v_per_m
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(DielectricValidationError::InvalidTestField);
        }
        if self
            .breakdown_strength_v_per_m
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(DielectricValidationError::InvalidBreakdownStrength);
        }
        if self.breakdown_strength_v_per_m.is_some()
            && self
                .breakdown_reference
                .as_deref()
                .is_none_or(|reference| reference.trim().is_empty())
        {
            return Err(DielectricValidationError::MissingBreakdownReference);
        }
        if self.breakdown_strength_v_per_m.is_some() && self.breakdown_evidence_kind.is_none() {
            return Err(DielectricValidationError::MissingBreakdownEvidenceKind);
        }
        if self.breakdown_strength_v_per_m.is_some()
            && self
                .breakdown_test_temperature_k
                .is_none_or(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(DielectricValidationError::InvalidBreakdownTestTemperature);
        }
        if self.breakdown_strength_v_per_m.is_some() && self.breakdown_test_kind.is_none() {
            return Err(DielectricValidationError::MissingBreakdownTestKind);
        }
        if self.breakdown_strength_v_per_m.is_none()
            && (self.breakdown_reference.is_some()
                || self.breakdown_test_temperature_k.is_some()
                || self.breakdown_test_kind.is_some()
                || self.breakdown_evidence_kind.is_some()
                || self.breakdown_strength_uncertainty.is_some())
        {
            return Err(DielectricValidationError::OrphanedBreakdownMetadata);
        }
        for uncertainty in [
            self.permittivity_uncertainty.as_ref(),
            self.loss_tangent_uncertainty.as_ref(),
            self.breakdown_strength_uncertainty.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            uncertainty.validate()?;
        }
        if self.source_reference.trim().is_empty() {
            return Err(DielectricValidationError::MissingSourceReference);
        }
        Ok(())
    }
}

/// Simple parallel-plate / multilayer capacitor geometry.
///
/// `electrode_overlap_area_m2` is the active overlap area of one dielectric
/// layer. Layers are treated as electrically parallel, with each layer having
/// the same thickness and terminal voltage. Fringing, parasitics, defects,
/// electrode margins, and manufacturing variation are not represented.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacitorGeometry {
    /// Electrode overlap area per active layer, in square metres.
    pub electrode_overlap_area_m2: f64,
    /// Dielectric thickness per active layer, in metres.
    pub dielectric_thickness_m: f64,
    /// Number of active dielectric layers in parallel.
    pub active_dielectric_layers: u32,
}

impl CapacitorGeometry {
    /// Validate geometric quantities before calculation.
    pub fn validate(&self) -> Result<(), DielectricValidationError> {
        if !self.electrode_overlap_area_m2.is_finite()
            || self.electrode_overlap_area_m2 <= 0.0
        {
            return Err(DielectricValidationError::InvalidElectrodeArea);
        }
        if !self.dielectric_thickness_m.is_finite() || self.dielectric_thickness_m <= 0.0 {
            return Err(DielectricValidationError::InvalidDielectricThickness);
        }
        if self.active_dielectric_layers == 0 {
            return Err(DielectricValidationError::ZeroActiveLayers);
        }
        Ok(())
    }
}

/// Ideal first-order metrics calculated from a property record and geometry.
///
/// `breakdown_field_ratio`, when present, is only the ratio of a reported
/// breakdown value to the calculated peak field. It is not a pass/fail verdict,
/// a lifetime estimate, or a safe operating margin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdealCapacitorMetrics {
    /// Ideal small-signal capacitance in farads.
    pub capacitance_f: f64,
    /// Applied peak electric field in V/m.
    pub peak_field_v_per_m: f64,
    /// Ideal maximum stored energy for the supplied peak voltage, in joules.
    pub peak_stored_energy_j: f64,
    /// Active dielectric volume, in cubic metres.
    pub active_dielectric_volume_m3: f64,
    /// Ideal linear-dielectric energy density at the applied field, in J/m³.
    pub ideal_energy_density_j_per_m3: f64,
    /// Ideal linear-dielectric energy density evaluated at the reported breakdown field, in J/m³.
    ///
    /// This is a theoretical screening upper bound, not a rated or usable energy density.
    /// It assumes field-independent permittivity, a homogeneous defect-free dielectric,
    /// and no hysteresis, leakage, aging, thermal, or reliability penalties.
    pub ideal_breakdown_limited_energy_density_j_per_m3: Option<f64>,
    /// Estimated sinusoidal dielectric-loss power at the record's test frequency, in watts.
    pub dielectric_loss_power_w: f64,
    /// Reported breakdown strength divided by calculated peak field, if both exist and field is nonzero.
    pub breakdown_field_ratio: Option<f64>,
    /// Temperature at which the input dielectric properties apply, in kelvin.
    pub property_temperature_k: f64,
    /// Frequency at which the input dielectric properties apply, in hertz.
    pub property_frequency_hz: f64,
    /// Temperature associated with the reported breakdown field, if available.
    pub breakdown_test_temperature_k: Option<f64>,
    /// Protocol family used to measure the reported breakdown field, if available.
    pub breakdown_test_kind: Option<BreakdownTestKind>,
    /// Evidence class associated with the breakdown field, if available.
    pub breakdown_evidence_kind: Option<DielectricEvidenceKind>,
    /// Source/protocol reference for the breakdown field, if available.
    pub breakdown_reference: Option<String>,
    /// Property-specific uncertainty for relative permittivity, if recorded.
    pub permittivity_uncertainty: Option<PropertyUncertainty>,
    /// Property-specific uncertainty for loss tangent, if recorded.
    pub loss_tangent_uncertainty: Option<PropertyUncertainty>,
    /// Property-specific uncertainty for breakdown strength, if recorded.
    pub breakdown_strength_uncertainty: Option<PropertyUncertainty>,
    /// Provenance preserved from the input permittivity/loss record.
    pub evidence_kind: DielectricEvidenceKind,
    /// Stable reference preserved from the input permittivity/loss record.
    pub source_reference: String,
}

/// Calculate ideal metrics without predicting or extrapolating material properties.
///
/// The dielectric constant and loss tangent are consumed exactly as recorded.
/// `peak_voltage_v` is the peak terminal voltage used for field/energy; the
/// AC RMS voltage is used for sinusoidal dielectric loss. The loss estimate is
/// based on the record's test frequency and is not valid at other frequencies
/// without suitable characterized properties.
pub fn calculate_ideal_capacitor_metrics(
    material: &DielectricPropertyRecord,
    geometry: &CapacitorGeometry,
    peak_voltage_v: f64,
    ac_rms_voltage_v: f64,
) -> Result<IdealCapacitorMetrics, DielectricValidationError> {
    material.validate()?;
    geometry.validate()?;
    if !peak_voltage_v.is_finite() || peak_voltage_v < 0.0 {
        return Err(DielectricValidationError::InvalidPeakVoltage);
    }
    if !ac_rms_voltage_v.is_finite() || ac_rms_voltage_v < 0.0 {
        return Err(DielectricValidationError::InvalidAcRmsVoltage);
    }
    if ac_rms_voltage_v > 0.0 && material.test_frequency_hz == 0.0 {
        return Err(DielectricValidationError::AcLossRequiresPositiveCharacterizationFrequency);
    }

    let layers = f64::from(geometry.active_dielectric_layers);
    let capacitance_f = VACUUM_PERMITTIVITY_F_PER_M
        * material.relative_permittivity
        * geometry.electrode_overlap_area_m2
        * layers
        / geometry.dielectric_thickness_m;
    let peak_field_v_per_m = peak_voltage_v / geometry.dielectric_thickness_m;
    let peak_stored_energy_j = 0.5 * capacitance_f * peak_voltage_v.powi(2);
    let active_dielectric_volume_m3 = geometry.electrode_overlap_area_m2
        * geometry.dielectric_thickness_m
        * layers;
    let ideal_energy_density_j_per_m3 = peak_stored_energy_j / active_dielectric_volume_m3;
    // This theoretical figure of merit makes the permittivity-versus-breakdown trade-off
    // explicit; it must never be interpreted as an operating rating or safety clearance.
    let ideal_breakdown_limited_energy_density_j_per_m3 = material
        .breakdown_strength_v_per_m
        .map(|breakdown| {
            0.5 * VACUUM_PERMITTIVITY_F_PER_M
                * material.relative_permittivity
                * breakdown.powi(2)
        });
    let dielectric_loss_power_w = 2.0
        * PI
        * material.test_frequency_hz
        * capacitance_f
        * ac_rms_voltage_v.powi(2)
        * material.loss_tangent;
    let breakdown_field_ratio = material
        .breakdown_strength_v_per_m
        .filter(|_| peak_field_v_per_m > 0.0)
        .map(|breakdown| breakdown / peak_field_v_per_m);

    let metrics = IdealCapacitorMetrics {
        capacitance_f,
        peak_field_v_per_m,
        peak_stored_energy_j,
        active_dielectric_volume_m3,
        ideal_energy_density_j_per_m3,
        ideal_breakdown_limited_energy_density_j_per_m3,
        dielectric_loss_power_w,
        breakdown_field_ratio,
        property_temperature_k: material.test_temperature_k,
        property_frequency_hz: material.test_frequency_hz,
        breakdown_test_temperature_k: material.breakdown_test_temperature_k,
        breakdown_test_kind: material.breakdown_test_kind,
        breakdown_evidence_kind: material.breakdown_evidence_kind,
        breakdown_reference: material.breakdown_reference.clone(),
        permittivity_uncertainty: material.permittivity_uncertainty.clone(),
        loss_tangent_uncertainty: material.loss_tangent_uncertainty.clone(),
        breakdown_strength_uncertainty: material.breakdown_strength_uncertainty.clone(),
        evidence_kind: material.evidence_kind,
        source_reference: material.source_reference.clone(),
    };

    let finite = [
        metrics.capacitance_f,
        metrics.peak_field_v_per_m,
        metrics.peak_stored_energy_j,
        metrics.active_dielectric_volume_m3,
        metrics.ideal_energy_density_j_per_m3,
        metrics.dielectric_loss_power_w,
    ]
    .into_iter()
    .all(f64::is_finite);
    if !finite
        || metrics
            .ideal_breakdown_limited_energy_density_j_per_m3
            .is_some_and(|value| !value.is_finite())
        || metrics
            .breakdown_field_ratio
            .is_some_and(|ratio| !ratio.is_finite())
    {
        return Err(DielectricValidationError::NonFiniteCalculatedMetric);
    }
    Ok(metrics)
}

/// Validation errors for dielectric records and ideal capacitor calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DielectricValidationError {
    /// Material identifier is empty.
    EmptyMaterialId,
    /// Relative permittivity must be finite and strictly positive.
    InvalidRelativePermittivity,
    /// Loss tangent must be finite and non-negative.
    InvalidLossTangent,
    /// Characterization temperature must be finite and above absolute zero.
    InvalidTestTemperature,
    /// Characterization frequency must be finite and non-negative.
    InvalidTestFrequency,
    /// Optional characterization field must be finite and non-negative.
    InvalidTestField,
    /// Optional breakdown field must be finite and strictly positive.
    InvalidBreakdownStrength,
    /// A breakdown field requires a source/protocol reference.
    MissingBreakdownReference,
    /// Optional property-level relative uncertainty must be finite and non-negative.
    InvalidRelativeUncertainty,
    /// Property-level uncertainty must cite its definition/calculation method.
    MissingUncertaintyMethodReference,
    /// A reported breakdown value must carry an evidence class.
    MissingBreakdownEvidenceKind,
    /// Breakdown characterization temperature must be finite and above absolute zero.
    InvalidBreakdownTestTemperature,
    /// A reported breakdown value must declare its test protocol family.
    MissingBreakdownTestKind,
    /// Breakdown reference/evidence/uncertainty cannot exist without a breakdown value.
    OrphanedBreakdownMetadata,
    /// Every property record requires a non-empty provenance reference.
    MissingSourceReference,
    /// Electrode overlap area must be finite and strictly positive.
    InvalidElectrodeArea,
    /// Dielectric thickness must be finite and strictly positive.
    InvalidDielectricThickness,
    /// At least one active dielectric layer is required.
    ZeroActiveLayers,
    /// Peak voltage must be finite and non-negative.
    InvalidPeakVoltage,
    /// AC RMS voltage must be finite and non-negative.
    InvalidAcRmsVoltage,
    /// A nonzero AC RMS voltage requires dielectric-loss data at a positive characterization frequency.
    AcLossRequiresPositiveCharacterizationFrequency,
    /// An equation overflowed or produced a non-finite result.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for DielectricValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyMaterialId => "material_id must not be empty",
            Self::InvalidRelativePermittivity => "relative permittivity must be finite and positive",
            Self::InvalidLossTangent => "loss tangent must be finite and non-negative",
            Self::InvalidTestTemperature => "test temperature must be finite and above zero kelvin",
            Self::InvalidTestFrequency => "test frequency must be finite and non-negative",
            Self::InvalidTestField => "test field must be finite and non-negative",
            Self::InvalidBreakdownStrength => "breakdown strength must be finite and positive",
            Self::MissingBreakdownReference => "breakdown strength requires a protocol/source reference",
            Self::InvalidRelativeUncertainty => "relative uncertainty must be finite and non-negative",
            Self::MissingUncertaintyMethodReference => "property uncertainty requires a method reference",
            Self::MissingBreakdownEvidenceKind => "breakdown strength requires an evidence class",
            Self::InvalidBreakdownTestTemperature => "breakdown test temperature must be finite and above zero kelvin",
            Self::MissingBreakdownTestKind => "breakdown strength requires a test protocol kind",
            Self::OrphanedBreakdownMetadata => "breakdown metadata requires a reported breakdown strength",
            Self::MissingSourceReference => "source_reference must not be empty",
            Self::InvalidElectrodeArea => "electrode overlap area must be finite and positive",
            Self::InvalidDielectricThickness => "dielectric thickness must be finite and positive",
            Self::ZeroActiveLayers => "active dielectric layer count must be nonzero",
            Self::InvalidPeakVoltage => "peak voltage must be finite and non-negative",
            Self::InvalidAcRmsVoltage => "AC RMS voltage must be finite and non-negative",
            Self::AcLossRequiresPositiveCharacterizationFrequency => "AC dielectric loss requires a positive characterization frequency",
            Self::NonFiniteCalculatedMetric => "calculated metric is non-finite; check input magnitudes",
        };
        f.write_str(message)
    }
}

impl Error for DielectricValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_record() -> DielectricPropertyRecord {
        DielectricPropertyRecord {
            material_id: "synthetic-fixture-not-a-real-material".to_string(),
            relative_permittivity: 1_000.0,
            loss_tangent: 0.01,
            test_temperature_k: 298.15,
            test_frequency_hz: 1_000.0,
            test_field_v_per_m: Some(1.0e6),
            breakdown_strength_v_per_m: Some(1.0e8),
            breakdown_reference: Some("synthetic test fixture only".to_string()),
            breakdown_test_temperature_k: Some(298.15),
            breakdown_test_kind: Some(BreakdownTestKind::DcRamp),
            breakdown_evidence_kind: Some(DielectricEvidenceKind::Estimated),
            evidence_kind: DielectricEvidenceKind::Estimated,
            source_reference: "test-fixture://not-real-data".to_string(),
            dataset_version: Some("unit-test-v1".to_string()),
            permittivity_uncertainty: Some(PropertyUncertainty {
                relative_uncertainty: 0.05,
                method_reference: "synthetic uncertainty fixture".to_string(),
            }),
            loss_tangent_uncertainty: Some(PropertyUncertainty {
                relative_uncertainty: 0.05,
                method_reference: "synthetic uncertainty fixture".to_string(),
            }),
            breakdown_strength_uncertainty: Some(PropertyUncertainty {
                relative_uncertainty: 0.05,
                method_reference: "synthetic uncertainty fixture".to_string(),
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

    #[test]
    fn parallel_plate_multilayer_capacitance_matches_equation() {
        let metrics = calculate_ideal_capacitor_metrics(
            &synthetic_record(),
            &geometry(),
            10.0,
            1.0,
        )
        .unwrap();
        let expected = VACUUM_PERMITTIVITY_F_PER_M * 1_000.0 * 1.0e-4 * 10.0 / 1.0e-6;
        assert!((metrics.capacitance_f - expected).abs() <= expected * 1.0e-12);
    }

    #[test]
    fn energy_density_matches_linear_dielectric_equation() {
        let metrics = calculate_ideal_capacitor_metrics(
            &synthetic_record(),
            &geometry(),
            10.0,
            0.0,
        )
        .unwrap();
        let expected = 0.5 * VACUUM_PERMITTIVITY_F_PER_M * 1_000.0 * 1.0e14;
        assert!((metrics.ideal_energy_density_j_per_m3 - expected).abs() <= expected * 1.0e-12);
    }

    #[test]
    fn breakdown_limited_energy_density_is_only_an_ideal_upper_bound() {
        let material = synthetic_record();
        let metrics =
            calculate_ideal_capacitor_metrics(&material, &geometry(), 10.0, 0.0).unwrap();
        let breakdown = material.breakdown_strength_v_per_m.unwrap();
        let expected = 0.5
            * VACUUM_PERMITTIVITY_F_PER_M
            * material.relative_permittivity
            * breakdown.powi(2);
        assert_eq!(
            metrics.ideal_breakdown_limited_energy_density_j_per_m3,
            Some(expected)
        );

        let mut no_breakdown = material;
        no_breakdown.breakdown_strength_v_per_m = None;
        no_breakdown.breakdown_reference = None;
        no_breakdown.breakdown_test_temperature_k = None;
        no_breakdown.breakdown_test_kind = None;
        no_breakdown.breakdown_evidence_kind = None;
        no_breakdown.breakdown_strength_uncertainty = None;
        let metrics =
            calculate_ideal_capacitor_metrics(&no_breakdown, &geometry(), 10.0, 0.0).unwrap();
        assert_eq!(metrics.ideal_breakdown_limited_energy_density_j_per_m3, None);
    }

    #[test]
    fn dielectric_loss_uses_characterized_frequency_and_rms_voltage() {
        let material = synthetic_record();
        let metrics =
            calculate_ideal_capacitor_metrics(&material, &geometry(), 0.0, 1.0).unwrap();
        let expected = 2.0
            * PI
            * material.test_frequency_hz
            * metrics.capacitance_f
            * 0.01;
        assert!((metrics.dielectric_loss_power_w - expected).abs() <= expected * 1.0e-12);
    }

    #[test]
    fn calculation_preserves_evidence_provenance() {
        let material = synthetic_record();
        let metrics =
            calculate_ideal_capacitor_metrics(&material, &geometry(), 10.0, 0.0).unwrap();
        assert_eq!(metrics.evidence_kind, material.evidence_kind);
        assert_eq!(metrics.source_reference, material.source_reference);
        assert_eq!(metrics.property_temperature_k, material.test_temperature_k);
        assert_eq!(metrics.property_frequency_hz, material.test_frequency_hz);
        assert_eq!(metrics.breakdown_field_ratio, Some(10.0));
    }

    #[test]
    fn rejects_invalid_property_records() {
        let mut material = synthetic_record();
        material.relative_permittivity = f64::NAN;
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::InvalidRelativePermittivity)
        );

        let mut material = synthetic_record();
        material.breakdown_reference = None;
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::MissingBreakdownReference)
        );

        let mut material = synthetic_record();
        material.source_reference.clear();
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::MissingSourceReference)
        );

        let mut material = synthetic_record();
        material.breakdown_evidence_kind = None;
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::MissingBreakdownEvidenceKind)
        );

        let mut material = synthetic_record();
        material.breakdown_test_temperature_k = None;
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::InvalidBreakdownTestTemperature)
        );

        let mut material = synthetic_record();
        material.breakdown_test_kind = None;
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::MissingBreakdownTestKind)
        );

        let mut material = synthetic_record();
        material.permittivity_uncertainty.as_mut().unwrap().method_reference.clear();
        assert_eq!(
            material.validate(),
            Err(DielectricValidationError::MissingUncertaintyMethodReference)
        );
    }

    #[test]
    fn rejects_invalid_geometry_and_operating_values() {
        let mut bad_geometry = geometry();
        bad_geometry.active_dielectric_layers = 0;
        assert_eq!(
            calculate_ideal_capacitor_metrics(&synthetic_record(), &bad_geometry, 1.0, 0.0),
            Err(DielectricValidationError::ZeroActiveLayers)
        );
        assert_eq!(
            calculate_ideal_capacitor_metrics(&synthetic_record(), &geometry(), f64::INFINITY, 0.0),
            Err(DielectricValidationError::InvalidPeakVoltage)
        );
        assert_eq!(
            calculate_ideal_capacitor_metrics(&synthetic_record(), &geometry(), 1.0, f64::NAN),
            Err(DielectricValidationError::InvalidAcRmsVoltage)
        );
    }

    #[test]
    fn ac_loss_is_not_reported_from_quasistatic_material_data() {
        let mut material = synthetic_record();
        material.test_frequency_hz = 0.0;
        assert_eq!(
            calculate_ideal_capacitor_metrics(&material, &geometry(), 0.0, 1.0),
            Err(DielectricValidationError::AcLossRequiresPositiveCharacterizationFrequency)
        );
    }

    #[test]
    fn zero_peak_field_does_not_create_infinite_breakdown_ratio() {
        let metrics =
            calculate_ideal_capacitor_metrics(&synthetic_record(), &geometry(), 0.0, 0.0).unwrap();
        assert_eq!(metrics.breakdown_field_ratio, None);
    }
}
