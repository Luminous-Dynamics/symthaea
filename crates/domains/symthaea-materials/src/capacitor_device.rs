// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware single-frequency capacitor device screening.
//!
//! This module supplements material-property screening with a simple series RLC
//! device model. It deliberately does not replace ngspice, measured impedance
//! sweeps, electrothermal simulation, or component qualification. A result applies
//! only to the declared model, frequency domain, and characterization temperature.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::f64::consts::PI;

/// Evidence origin of the series-RLC parameters for a capacitor device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorDeviceEvidenceKind {
    /// Impedance or component parameters measured on a physical device.
    MeasuredDevice,
    /// Parameters transcribed from a manufacturer data sheet.
    ManufacturerDatasheet,
    /// Equivalent-circuit parameters fitted to a declared measured sweep.
    FittedEquivalentCircuit,
    /// Estimated parameters not yet validated on the target device.
    Estimated,
}

/// A single series-RLC capacitor model with an explicitly declared applicability domain.
///
/// ESR and ESL are treated as constant over the declared frequency band. This is
/// a useful first-order approximation only; skin/proximity effects, dielectric
/// dispersion, distributed package behavior, mounting inductance, bias dependence,
/// aging, and temperature dependence are not modeled. The source reference should
/// identify the device measurement, manufacturer data, or fitted-model receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesRlcCapacitorModel {
    /// Stable component, specimen, or model identifier.
    pub device_id: String,
    /// Effective capacitance in farads.
    pub capacitance_f: f64,
    /// Effective series resistance in ohms for this model/domain.
    pub esr_ohm: f64,
    /// Effective series inductance in henries for this model/domain.
    pub esl_h: f64,
    /// Temperature in kelvin associated with the model characterization.
    pub characterization_temperature_k: f64,
    /// Lowest frequency at which the model is declared applicable, in hertz.
    pub valid_frequency_min_hz: f64,
    /// Highest frequency at which the model is declared applicable, in hertz.
    pub valid_frequency_max_hz: f64,
    /// Origin of the parameter evidence; this is not an ordinal quality score.
    pub evidence_kind: CapacitorDeviceEvidenceKind,
    /// Stable reference to the source data/model fit and its conditions.
    pub source_reference: String,
}

impl SeriesRlcCapacitorModel {
    /// Validate parameter values, applicability domain, and provenance.
    pub fn validate(&self) -> Result<(), CapacitorDeviceModelError> {
        if self.device_id.trim().is_empty() {
            return Err(CapacitorDeviceModelError::EmptyDeviceId);
        }
        if !self.capacitance_f.is_finite() || self.capacitance_f <= 0.0 {
            return Err(CapacitorDeviceModelError::InvalidCapacitance);
        }
        if !self.esr_ohm.is_finite() || self.esr_ohm < 0.0 {
            return Err(CapacitorDeviceModelError::InvalidEsr);
        }
        if !self.esl_h.is_finite() || self.esl_h < 0.0 {
            return Err(CapacitorDeviceModelError::InvalidEsl);
        }
        if !self.characterization_temperature_k.is_finite()
            || self.characterization_temperature_k <= 0.0
        {
            return Err(CapacitorDeviceModelError::InvalidCharacterizationTemperature);
        }
        if !self.valid_frequency_min_hz.is_finite()
            || !self.valid_frequency_max_hz.is_finite()
            || self.valid_frequency_min_hz <= 0.0
            || self.valid_frequency_max_hz < self.valid_frequency_min_hz
        {
            return Err(CapacitorDeviceModelError::InvalidFrequencyDomain);
        }
        if self.source_reference.trim().is_empty() {
            return Err(CapacitorDeviceModelError::MissingSourceReference);
        }
        Ok(())
    }

    /// Return the ideal series-resonant frequency, if positive ESL is modeled.
    ///
    /// This is an RLC-model calculation, not a measured resonance guarantee.
    pub fn ideal_self_resonant_frequency_hz(&self) -> Option<f64> {
        (self.esl_h > 0.0)
            .then(|| 1.0 / (2.0 * PI * (self.capacitance_f * self.esl_h).sqrt()))
    }
}

/// Constraints for a single-frequency device-level screening decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorDeviceScreeningRequirements {
    /// Intended operating temperature, in kelvin.
    pub operating_temperature_k: f64,
    /// Maximum permitted absolute deviation from the model's characterization temperature, in kelvin.
    pub max_temperature_deviation_k: f64,
    /// Intended single sinusoidal operating frequency, in hertz.
    pub operating_frequency_hz: f64,
    /// RMS sinusoidal ripple current at that frequency, in amperes.
    pub ripple_current_rms_a: f64,
    /// Maximum permitted magnitude of modeled terminal impedance, in ohms.
    pub max_impedance_ohm: f64,
    /// Maximum permitted ESR dissipation for the declared ripple current, in watts.
    pub max_esr_loss_power_w: f64,
    /// Optional thermal resistance from the modeled component to its reference environment, in K/W.
    pub thermal_resistance_k_per_w: Option<f64>,
    /// Optional maximum steady-state temperature rise estimated from ESR loss, in kelvin.
    pub max_temperature_rise_k: Option<f64>,
    /// Optional minimum ideal self-resonant frequency, in hertz.
    pub min_self_resonant_frequency_hz: Option<f64>,
    /// Allowed evidence origins. This is a policy allow-list, not a trust ranking.
    pub accepted_evidence_kinds: Vec<CapacitorDeviceEvidenceKind>,
}

impl CapacitorDeviceScreeningRequirements {
    /// Validate the operating point and hard-constraint contract.
    pub fn validate(&self) -> Result<(), CapacitorDeviceModelError> {
        if !self.operating_temperature_k.is_finite() || self.operating_temperature_k <= 0.0 {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "operating temperature must be finite and above absolute zero",
            ));
        }
        if !self.max_temperature_deviation_k.is_finite()
            || self.max_temperature_deviation_k < 0.0
        {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "temperature deviation must be finite and non-negative",
            ));
        }
        if !self.operating_frequency_hz.is_finite() || self.operating_frequency_hz <= 0.0 {
            return Err(CapacitorDeviceModelError::InvalidOperatingFrequency);
        }
        if !self.ripple_current_rms_a.is_finite() || self.ripple_current_rms_a < 0.0 {
            return Err(CapacitorDeviceModelError::InvalidRippleCurrent);
        }
        if !self.max_impedance_ohm.is_finite() || self.max_impedance_ohm < 0.0 {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "maximum impedance must be finite and non-negative",
            ));
        }
        if !self.max_esr_loss_power_w.is_finite() || self.max_esr_loss_power_w < 0.0 {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "maximum ESR loss must be finite and non-negative",
            ));
        }
        if self
            .thermal_resistance_k_per_w
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(CapacitorDeviceModelError::InvalidThermalResistance);
        }
        if self.max_temperature_rise_k.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "maximum temperature rise must be finite and non-negative",
            ));
        }
        if self.max_temperature_rise_k.is_some() && self.thermal_resistance_k_per_w.is_none() {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "a temperature-rise constraint requires an explicit thermal resistance",
            ));
        }
        if self
            .min_self_resonant_frequency_hz
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "minimum self-resonant frequency must be finite and positive",
            ));
        }
        if self.accepted_evidence_kinds.is_empty() {
            return Err(CapacitorDeviceModelError::InvalidRequirements(
                "at least one device-model evidence origin must be allowed",
            ));
        }
        Ok(())
    }
}

/// Derived single-frequency metrics under the declared series-RLC approximation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorDeviceMetrics {
    /// Frequency used for the calculation, in hertz.
    pub operating_frequency_hz: f64,
    /// Series resistance in ohms.
    pub impedance_real_ohm: f64,
    /// Net series reactance (inductive minus capacitive) in ohms.
    pub impedance_reactive_ohm: f64,
    /// Magnitude of terminal impedance in ohms.
    pub impedance_magnitude_ohm: f64,
    /// Impedance phase in radians; absent when the modeled impedance is exactly zero.
    pub impedance_phase_rad: Option<f64>,
    /// Ideal series-resonant frequency in hertz, absent when ESL is zero.
    pub ideal_self_resonant_frequency_hz: Option<f64>,
    /// RMS terminal ripple voltage for the supplied single-frequency RMS current, in volts.
    pub ripple_voltage_rms_v: f64,
    /// ESR loss under the supplied RMS current, in watts.
    pub esr_loss_power_w: f64,
    /// First-order steady-state temperature rise estimate, in kelvin, if thermal resistance was supplied.
    pub modeled_temperature_rise_k: Option<f64>,
}

/// Overall disposition of the evidence-bounded device-level screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorDeviceScreeningDisposition {
    /// All declared constraints passed under the applicable lumped model.
    PassesLumpedModelScreen,
    /// One or more modeled constraints failed.
    FailsDeclaredConstraints,
    /// Domain, evidence, or a required metric does not support a decision.
    NotEvaluable,
}

/// Why the declared device screen cannot support a final disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorDeviceScreeningLimitation {
    /// Operating temperature is outside the permitted characterization window.
    TemperatureOutsideCharacterization,
    /// Operating frequency is outside the declared model domain.
    FrequencyOutsideModelDomain,
    /// Model evidence origin is not in the requirement allow-list.
    EvidenceKindNotAllowed,
    /// A required resonance constraint cannot be calculated because ESL is absent/zero.
    SelfResonanceUnavailable,
    /// Calculated resonance lies outside the declared model validation band.
    SelfResonanceOutsideModelDomain,
}

/// A hard device-level screening constraint violated by model-derived metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorDeviceConstraintViolation {
    /// Impedance magnitude exceeds the maximum allowed value.
    ImpedanceAboveMaximum,
    /// ESR heating estimate exceeds the maximum allowed power.
    EsrLossAboveMaximum,
    /// First-order temperature-rise estimate exceeds its maximum.
    TemperatureRiseAboveMaximum,
    /// Ideal series-resonant frequency is below the required minimum.
    SelfResonantFrequencyBelowMinimum,
}

/// Auditable device-screen report with original model and operating requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorDeviceScreeningReport {
    /// Exact series-RLC model used for this decision.
    pub model: SeriesRlcCapacitorModel,
    /// Exact operating requirements used for this decision.
    pub requirements: CapacitorDeviceScreeningRequirements,
    /// Model-conditional disposition.
    pub disposition: CapacitorDeviceScreeningDisposition,
    /// Reasons a disposition is unavailable.
    pub limitations: Vec<CapacitorDeviceScreeningLimitation>,
    /// Failed modeled constraints.
    pub violations: Vec<CapacitorDeviceConstraintViolation>,
    /// Derived metrics when the primary applicability domain is satisfied.
    pub metrics: Option<CapacitorDeviceMetrics>,
}

/// Input or numerical error for the evidence-aware lumped capacitor model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapacitorDeviceModelError {
    /// Component or model identifier is empty.
    EmptyDeviceId,
    /// Evidence/source reference is missing.
    MissingSourceReference,
    /// Capacitance is not finite and positive.
    InvalidCapacitance,
    /// ESR is invalid.
    InvalidEsr,
    /// ESL is invalid.
    InvalidEsl,
    /// Characterization temperature is invalid.
    InvalidCharacterizationTemperature,
    /// Frequency domain is malformed.
    InvalidFrequencyDomain,
    /// Operating frequency is invalid.
    InvalidOperatingFrequency,
    /// RMS ripple current is invalid.
    InvalidRippleCurrent,
    /// Thermal resistance is invalid.
    InvalidThermalResistance,
    /// Requirement contract is malformed.
    InvalidRequirements(&'static str),
    /// An operating frequency is outside the model's stated domain.
    OperatingFrequencyOutsideModelDomain,
    /// A derived metric is non-finite.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for CapacitorDeviceModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequirements(message) => {
                write!(f, "invalid capacitor device requirements: {message}")
            }
            other => write!(f, "invalid capacitor device model input: {other:?}"),
        }
    }
}

impl Error for CapacitorDeviceModelError {}

/// Calculate single-frequency metrics under a constant-parameter series-RLC model.
///
/// The equations assume steady-state sinusoidal ripple and model-domain-valid
/// parameters: X = 2πf·ESL - 1/(2πf·C), |Z| = sqrt(ESR² + X²), P_ESR = I_rms²·ESR.
/// The optional thermal result is ΔT = P_ESR·Rθ; it is not a transient, coupled
/// electrothermal, lifetime, or safety model.
pub fn calculate_capacitor_device_metrics(
    model: &SeriesRlcCapacitorModel,
    operating_frequency_hz: f64,
    ripple_current_rms_a: f64,
    thermal_resistance_k_per_w: Option<f64>,
) -> Result<CapacitorDeviceMetrics, CapacitorDeviceModelError> {
    model.validate()?;
    if !operating_frequency_hz.is_finite() || operating_frequency_hz <= 0.0 {
        return Err(CapacitorDeviceModelError::InvalidOperatingFrequency);
    }
    if operating_frequency_hz < model.valid_frequency_min_hz
        || operating_frequency_hz > model.valid_frequency_max_hz
    {
        return Err(CapacitorDeviceModelError::OperatingFrequencyOutsideModelDomain);
    }
    if !ripple_current_rms_a.is_finite() || ripple_current_rms_a < 0.0 {
        return Err(CapacitorDeviceModelError::InvalidRippleCurrent);
    }
    if thermal_resistance_k_per_w.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(CapacitorDeviceModelError::InvalidThermalResistance);
    }

    let angular_frequency = 2.0 * PI * operating_frequency_hz;
    let capacitive_reactance = 1.0 / (angular_frequency * model.capacitance_f);
    let inductive_reactance = angular_frequency * model.esl_h;
    let impedance_reactive_ohm = inductive_reactance - capacitive_reactance;
    let impedance_magnitude_ohm = model.esr_ohm.hypot(impedance_reactive_ohm);
    let impedance_phase_rad = (impedance_magnitude_ohm > 0.0)
        .then(|| impedance_reactive_ohm.atan2(model.esr_ohm));
    let ripple_voltage_rms_v = ripple_current_rms_a * impedance_magnitude_ohm;
    let esr_loss_power_w = ripple_current_rms_a.powi(2) * model.esr_ohm;
    let modeled_temperature_rise_k =
        thermal_resistance_k_per_w.map(|thermal_resistance| esr_loss_power_w * thermal_resistance);
    let ideal_self_resonant_frequency_hz = model.ideal_self_resonant_frequency_hz();

    let scalars = [
        capacitive_reactance,
        inductive_reactance,
        impedance_reactive_ohm,
        impedance_magnitude_ohm,
        ripple_voltage_rms_v,
        esr_loss_power_w,
    ];
    if scalars.into_iter().any(|value| !value.is_finite())
        || impedance_phase_rad.is_some_and(|value| !value.is_finite())
        || modeled_temperature_rise_k.is_some_and(|value| !value.is_finite())
        || ideal_self_resonant_frequency_hz.is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(CapacitorDeviceModelError::NonFiniteCalculatedMetric);
    }

    Ok(CapacitorDeviceMetrics {
        operating_frequency_hz,
        impedance_real_ohm: model.esr_ohm,
        impedance_reactive_ohm,
        impedance_magnitude_ohm,
        impedance_phase_rad,
        ideal_self_resonant_frequency_hz,
        ripple_voltage_rms_v,
        esr_loss_power_w,
        modeled_temperature_rise_k,
    })
}

/// Screen one capacitor device model against an explicit operating point and constraints.
///
/// Out-of-domain or disallowed evidence returns NotEvaluable; it is never
/// converted into a pass/fail ranking using extrapolated parameters. A pass
/// describes only the declared constant-parameter RLC and thermal-resistance
/// approximations, not a physical component qualification.
pub fn screen_capacitor_device(
    model: &SeriesRlcCapacitorModel,
    requirements: &CapacitorDeviceScreeningRequirements,
) -> Result<CapacitorDeviceScreeningReport, CapacitorDeviceModelError> {
    model.validate()?;
    requirements.validate()?;

    let mut limitations = Vec::new();
    if (requirements.operating_temperature_k - model.characterization_temperature_k).abs()
        > requirements.max_temperature_deviation_k
    {
        limitations.push(CapacitorDeviceScreeningLimitation::TemperatureOutsideCharacterization);
    }
    if requirements.operating_frequency_hz < model.valid_frequency_min_hz
        || requirements.operating_frequency_hz > model.valid_frequency_max_hz
    {
        limitations.push(CapacitorDeviceScreeningLimitation::FrequencyOutsideModelDomain);
    }
    if !requirements.accepted_evidence_kinds.contains(&model.evidence_kind) {
        limitations.push(CapacitorDeviceScreeningLimitation::EvidenceKindNotAllowed);
    }
    if !limitations.is_empty() {
        return Ok(CapacitorDeviceScreeningReport {
            model: model.clone(),
            requirements: requirements.clone(),
            disposition: CapacitorDeviceScreeningDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            metrics: None,
        });
    }

    let metrics = calculate_capacitor_device_metrics(
        model,
        requirements.operating_frequency_hz,
        requirements.ripple_current_rms_a,
        requirements.thermal_resistance_k_per_w,
    )?;
    if requirements.min_self_resonant_frequency_hz.is_some() {
        let resonance_limitation = match metrics.ideal_self_resonant_frequency_hz {
            None => Some(CapacitorDeviceScreeningLimitation::SelfResonanceUnavailable),
            Some(frequency)
                if frequency < model.valid_frequency_min_hz
                    || frequency > model.valid_frequency_max_hz =>
            {
                Some(CapacitorDeviceScreeningLimitation::SelfResonanceOutsideModelDomain)
            }
            Some(_) => None,
        };
        if let Some(limitation) = resonance_limitation {
            return Ok(CapacitorDeviceScreeningReport {
                model: model.clone(),
                requirements: requirements.clone(),
                disposition: CapacitorDeviceScreeningDisposition::NotEvaluable,
                limitations: vec![limitation],
                violations: Vec::new(),
                metrics: Some(metrics),
            });
        }
    }

    let mut violations = Vec::new();
    if metrics.impedance_magnitude_ohm > requirements.max_impedance_ohm {
        violations.push(CapacitorDeviceConstraintViolation::ImpedanceAboveMaximum);
    }
    if metrics.esr_loss_power_w > requirements.max_esr_loss_power_w {
        violations.push(CapacitorDeviceConstraintViolation::EsrLossAboveMaximum);
    }
    if requirements.max_temperature_rise_k.is_some_and(|maximum| {
        metrics
            .modeled_temperature_rise_k
            .is_none_or(|rise| rise > maximum)
    }) {
        violations.push(CapacitorDeviceConstraintViolation::TemperatureRiseAboveMaximum);
    }
    if let Some(minimum) = requirements.min_self_resonant_frequency_hz {
        if metrics
            .ideal_self_resonant_frequency_hz
            .is_some_and(|frequency| frequency < minimum)
        {
            violations.push(CapacitorDeviceConstraintViolation::SelfResonantFrequencyBelowMinimum);
        }
    }

    let disposition = if violations.is_empty() {
        CapacitorDeviceScreeningDisposition::PassesLumpedModelScreen
    } else {
        CapacitorDeviceScreeningDisposition::FailsDeclaredConstraints
    };
    Ok(CapacitorDeviceScreeningReport {
        model: model.clone(),
        requirements: requirements.clone(),
        disposition,
        limitations,
        violations,
        metrics: Some(metrics),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> SeriesRlcCapacitorModel {
        SeriesRlcCapacitorModel {
            device_id: "synthetic-series-rlc-fixture".to_string(),
            capacitance_f: 1.0e-6,
            esr_ohm: 0.1,
            esl_h: 10.0e-9,
            characterization_temperature_k: 298.15,
            valid_frequency_min_hz: 1.0e3,
            valid_frequency_max_hz: 10.0e6,
            evidence_kind: CapacitorDeviceEvidenceKind::FittedEquivalentCircuit,
            source_reference: "test-fixture://not-real-component-data".to_string(),
        }
    }

    fn requirements(sample: &SeriesRlcCapacitorModel) -> CapacitorDeviceScreeningRequirements {
        CapacitorDeviceScreeningRequirements {
            operating_temperature_k: 298.15,
            max_temperature_deviation_k: 5.0,
            operating_frequency_hz: sample.ideal_self_resonant_frequency_hz().unwrap(),
            ripple_current_rms_a: 0.5,
            max_impedance_ohm: 0.11,
            max_esr_loss_power_w: 0.03,
            thermal_resistance_k_per_w: Some(10.0),
            max_temperature_rise_k: Some(0.3),
            min_self_resonant_frequency_hz: Some(1.0e6),
            accepted_evidence_kinds: vec![CapacitorDeviceEvidenceKind::FittedEquivalentCircuit],
        }
    }

    #[test]
    fn self_resonance_minimizes_series_rlc_impedance_to_esr() {
        let sample = model();
        let frequency = sample.ideal_self_resonant_frequency_hz().unwrap();
        let metrics = calculate_capacitor_device_metrics(&sample, frequency, 0.5, None).unwrap();
        assert!(metrics.impedance_reactive_ohm.abs() <= 1.0e-10);
        assert!((metrics.impedance_magnitude_ohm - sample.esr_ohm).abs() <= 1.0e-10);
    }

    #[test]
    fn impedance_changes_from_capacitive_to_inductive_across_resonance() {
        let sample = model();
        let resonance = sample.ideal_self_resonant_frequency_hz().unwrap();
        let below = calculate_capacitor_device_metrics(&sample, resonance / 10.0, 1.0, None).unwrap();
        let above = calculate_capacitor_device_metrics(&sample, resonance * 2.0, 1.0, None).unwrap();
        assert!(below.impedance_reactive_ohm < 0.0);
        assert!(above.impedance_reactive_ohm > 0.0);
    }

    #[test]
    fn ripple_loss_and_first_order_thermal_rise_use_rms_current() {
        let sample = model();
        let metrics = calculate_capacitor_device_metrics(&sample, 1.0e6, 0.5, Some(10.0)).unwrap();
        assert!((metrics.esr_loss_power_w - 0.025).abs() <= 1.0e-12);
        assert!((metrics.modeled_temperature_rise_k.unwrap() - 0.25).abs() <= 1.0e-12);
    }

    #[test]
    fn out_of_domain_frequency_is_not_evaluated() {
        let sample = model();
        let mut req = requirements(&sample);
        req.operating_frequency_hz = 20.0e6;
        let report = screen_capacitor_device(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &CapacitorDeviceScreeningLimitation::FrequencyOutsideModelDomain
        ));
        assert!(report.metrics.is_none());
    }

    #[test]
    fn out_of_domain_temperature_is_not_evaluated() {
        let sample = model();
        let mut req = requirements(&sample);
        req.operating_temperature_k = 320.0;
        let report = screen_capacitor_device(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &CapacitorDeviceScreeningLimitation::TemperatureOutsideCharacterization
        ));
    }

    #[test]
    fn evidence_allow_list_is_not_a_quality_ranking() {
        let sample = model();
        let mut req = requirements(&sample);
        req.accepted_evidence_kinds = vec![CapacitorDeviceEvidenceKind::MeasuredDevice];
        let report = screen_capacitor_device(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &CapacitorDeviceScreeningLimitation::EvidenceKindNotAllowed
        ));
    }

    #[test]
    fn zero_esl_does_not_claim_infinite_self_resonance() {
        let mut sample = model();
        sample.esl_h = 0.0;
        assert_eq!(sample.ideal_self_resonant_frequency_hz(), None);
        let report = screen_capacitor_device(&sample, &requirements(&model())).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &CapacitorDeviceScreeningLimitation::SelfResonanceUnavailable
        ));
    }

    #[test]
    fn resonance_outside_validated_band_is_not_used_for_a_pass() {
        let mut sample = model();
        sample.valid_frequency_max_hz = 100.0e3;
        let mut req = requirements(&sample);
        req.operating_frequency_hz = 80.0e3;
        req.min_self_resonant_frequency_hz = Some(500.0e3);
        let report = screen_capacitor_device(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &CapacitorDeviceScreeningLimitation::SelfResonanceOutsideModelDomain
        ));
    }

    #[test]
    fn failed_constraint_is_distinct_from_missing_evidence() {
        let sample = model();
        let mut req = requirements(&sample);
        req.max_impedance_ohm = 0.01;
        let report = screen_capacitor_device(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorDeviceScreeningDisposition::FailsDeclaredConstraints);
        assert!(report.violations.contains(
            &CapacitorDeviceConstraintViolation::ImpedanceAboveMaximum
        ));
    }

    #[test]
    fn invalid_numeric_values_fail_closed() {
        let mut sample = model();
        sample.esl_h = f64::NAN;
        assert_eq!(sample.validate(), Err(CapacitorDeviceModelError::InvalidEsl));
    }

    #[test]
    fn thermal_constraint_requires_explicit_thermal_resistance() {
        let sample = model();
        let mut req = requirements(&sample);
        req.thermal_resistance_k_per_w = None;
        assert!(matches!(
            req.validate(),
            Err(CapacitorDeviceModelError::InvalidRequirements(_))
        ));
    }
}
