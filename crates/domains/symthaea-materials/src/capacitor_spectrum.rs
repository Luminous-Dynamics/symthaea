// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frequency-resolved capacitor impedance and ripple-current screening.
//!
//! This module uses a condition-bound complex-impedance sweep rather than assuming
//! one equivalent series resistance applies at every frequency. It only interpolates
//! between supplied sweep knots and never extrapolates. It combines distinct
//! steady-state AC spectral components by their RMS convention; phase-resolved
//! waveform reconstruction, electrothermal dynamics, lifetime and qualification
//! are deliberately outside this first version.

use crate::capacitor_device::CapacitorDeviceEvidenceKind;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// One complex-impedance observation at a positive frequency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImpedanceSweepPoint {
    /// Frequency in hertz; points in a sweep must be strictly increasing.
    pub frequency_hz: f64,
    /// Real part of impedance in ohms. Negative values are rejected for this passive-device contract.
    pub real_ohm: f64,
    /// Imaginary part of impedance in ohms; positive is inductive under the declared convention.
    pub imaginary_ohm: f64,
}

/// A single device impedance sweep acquired under one declared physical condition.
///
/// The sweep retains the source and method, characterization temperature and optional
/// DC bias. Points are not assumed interchangeable with sweeps at other conditions.
/// The method reference should identify measurement instrumentation/configuration or
/// the equivalent-circuit fitting method, including relevant calibration details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorImpedanceSweep {
    /// Stable component/specimen identifier.
    pub device_id: String,
    /// Temperature in kelvin for this sweep.
    pub characterization_temperature_k: f64,
    /// DC bias in volts during the sweep, if applicable/known.
    pub dc_bias_v: Option<f64>,
    /// Source data reference, e.g. URI plus dataset version and row/file locator.
    pub source_reference: String,
    /// Measurement or fitting method reference, including setup/configuration provenance.
    pub method_reference: String,
    /// Origin of the sweep data; it is not a universal quality ranking.
    pub evidence_kind: CapacitorDeviceEvidenceKind,
    /// Ordered complex impedance observations.
    pub points: Vec<ImpedanceSweepPoint>,
}

impl CapacitorImpedanceSweep {
    /// Validate provenance, condition metadata, passivity and sweep ordering.
    pub fn validate(&self) -> Result<(), CapacitorSpectrumError> {
        if self.device_id.trim().is_empty() {
            return Err(CapacitorSpectrumError::InvalidSweep("device ID is empty"));
        }
        if !self.characterization_temperature_k.is_finite()
            || self.characterization_temperature_k <= 0.0
        {
            return Err(CapacitorSpectrumError::InvalidSweep(
                "characterization temperature must be finite and positive",
            ));
        }
        if self.dc_bias_v.is_some_and(|value| !value.is_finite()) {
            return Err(CapacitorSpectrumError::InvalidSweep(
                "DC bias must be finite when present",
            ));
        }
        if self.source_reference.trim().is_empty() {
            return Err(CapacitorSpectrumError::InvalidSweep(
                "source reference is missing",
            ));
        }
        if self.method_reference.trim().is_empty() {
            return Err(CapacitorSpectrumError::InvalidSweep(
                "measurement/fit method reference is missing",
            ));
        }
        if self.points.len() < 2 {
            return Err(CapacitorSpectrumError::InvalidSweep(
                "at least two impedance knots are required for a bounded sweep",
            ));
        }

        let mut previous_frequency = 0.0;
        for point in &self.points {
            if !point.frequency_hz.is_finite() || point.frequency_hz <= 0.0 {
                return Err(CapacitorSpectrumError::InvalidSweep(
                    "sweep frequencies must be finite and positive",
                ));
            }
            if point.frequency_hz <= previous_frequency {
                return Err(CapacitorSpectrumError::InvalidSweep(
                    "sweep frequencies must be strictly increasing without duplicates",
                ));
            }
            if !point.real_ohm.is_finite() || point.real_ohm < 0.0 {
                return Err(CapacitorSpectrumError::InvalidSweep(
                    "real impedance must be finite and non-negative for a passive device",
                ));
            }
            if !point.imaginary_ohm.is_finite() {
                return Err(CapacitorSpectrumError::InvalidSweep(
                    "imaginary impedance must be finite",
                ));
            }
            previous_frequency = point.frequency_hz;
        }
        Ok(())
    }

    /// Lowest measured frequency in hertz, if the sweep has any points.
    pub fn min_frequency_hz(&self) -> Option<f64> {
        self.points.first().map(|point| point.frequency_hz)
    }

    /// Highest measured frequency in hertz, if the sweep has any points.
    pub fn max_frequency_hz(&self) -> Option<f64> {
        self.points.last().map(|point| point.frequency_hz)
    }

    /// Evaluate impedance at a frequency using log-frequency linear interpolation.
    ///
    /// Exact knots preserve the supplied observation. Between knots, real and
    /// imaginary parts are interpolated linearly in log(f). Frequencies outside
    /// the measured sweep return an error: extrapolation is never performed.
    pub fn evaluate(
        &self,
        frequency_hz: f64,
    ) -> Result<EvaluatedImpedance, CapacitorSpectrumError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(CapacitorSpectrumError::InvalidSpectrum(
                "requested frequency must be finite and positive",
            ));
        }
        let first = self.points.first().expect("validated minimum point count");
        let last = self.points.last().expect("validated minimum point count");
        if frequency_hz < first.frequency_hz || frequency_hz > last.frequency_hz {
            return Err(CapacitorSpectrumError::FrequencyOutsideMeasuredSweep {
                frequency_hz,
                min_frequency_hz: first.frequency_hz,
                max_frequency_hz: last.frequency_hz,
            });
        }

        match self
            .points
            .binary_search_by(|point| point.frequency_hz.total_cmp(&frequency_hz))
        {
            Ok(index) => {
                let point = self.points[index];
                Ok(EvaluatedImpedance {
                    frequency_hz,
                    real_ohm: point.real_ohm,
                    imaginary_ohm: point.imaginary_ohm,
                    evaluation_kind: ImpedanceEvaluationKind::MeasuredKnot,
                    lower_bracket_frequency_hz: point.frequency_hz,
                    upper_bracket_frequency_hz: point.frequency_hz,
                })
            }
            Err(upper_index) => {
                let lower = self.points[upper_index - 1];
                let upper = self.points[upper_index];
                let log_low = lower.frequency_hz.ln();
                let log_high = upper.frequency_hz.ln();
                let fraction = (frequency_hz.ln() - log_low) / (log_high - log_low);
                let real_ohm = lower.real_ohm + fraction * (upper.real_ohm - lower.real_ohm);
                let imaginary_ohm = lower.imaginary_ohm
                    + fraction * (upper.imaginary_ohm - lower.imaginary_ohm);
                if !real_ohm.is_finite() || real_ohm < 0.0 || !imaginary_ohm.is_finite() {
                    return Err(CapacitorSpectrumError::NonFiniteCalculatedMetric);
                }
                Ok(EvaluatedImpedance {
                    frequency_hz,
                    real_ohm,
                    imaginary_ohm,
                    evaluation_kind: ImpedanceEvaluationKind::InterpolatedBetweenKnots,
                    lower_bracket_frequency_hz: lower.frequency_hz,
                    upper_bracket_frequency_hz: upper.frequency_hz,
                })
            }
        }
    }
}

/// Indicates whether an evaluated complex impedance is observed or interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpedanceEvaluationKind {
    /// Requested frequency exactly matches an input sweep knot.
    MeasuredKnot,
    /// Requested frequency lies between two input sweep knots.
    InterpolatedBetweenKnots,
}

/// Impedance at one requested frequency, with the exact observation/interpolation lineage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvaluatedImpedance {
    /// Frequency evaluated, in hertz.
    pub frequency_hz: f64,
    /// Real impedance, in ohms.
    pub real_ohm: f64,
    /// Imaginary impedance, in ohms.
    pub imaginary_ohm: f64,
    /// Whether the value is a measured knot or an interpolation.
    pub evaluation_kind: ImpedanceEvaluationKind,
    /// Lower interpolation bracket frequency, or the measured knot frequency.
    pub lower_bracket_frequency_hz: f64,
    /// Upper interpolation bracket frequency, or the measured knot frequency.
    pub upper_bracket_frequency_hz: f64,
}

impl EvaluatedImpedance {
    /// Magnitude of the complex impedance in ohms.
    pub fn magnitude_ohm(&self) -> f64 {
        self.real_ohm.hypot(self.imaginary_ohm)
    }

    /// Phase in radians under the declared real/imaginary convention.
    pub fn phase_rad(&self) -> f64 {
        self.imaginary_ohm.atan2(self.real_ohm)
    }
}

/// One positive-frequency RMS current component from measurement or simulation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurrentSpectrumComponent {
    /// Positive frequency in hertz.
    pub frequency_hz: f64,
    /// RMS current magnitude at this frequency, in amperes.
    pub current_rms_a: f64,
}

/// A source-bound AC ripple spectrum with explicit extraction-method provenance.
///
/// Frequencies must be strictly increasing and unique. The method reference should
/// identify FFT/windowing, sampling, scaling/RMS conversion, and filtering settings.
/// This first version excludes DC and uses magnitudes only; it cannot reconstruct
/// peak voltage or phase-resolved time waveforms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorCurrentSpectrum {
    /// Source waveform, simulation output, or spectrum dataset reference.
    pub source_reference: String,
    /// Spectrum extraction method/configuration, including window and RMS scaling.
    pub method_reference: String,
    /// Ordered positive-frequency RMS current components.
    pub components: Vec<CurrentSpectrumComponent>,
}

impl CapacitorCurrentSpectrum {
    /// Validate source/method provenance and spectral ordering.
    pub fn validate(&self) -> Result<(), CapacitorSpectrumError> {
        if self.source_reference.trim().is_empty() {
            return Err(CapacitorSpectrumError::InvalidSpectrum(
                "current-spectrum source reference is missing",
            ));
        }
        if self.method_reference.trim().is_empty() {
            return Err(CapacitorSpectrumError::InvalidSpectrum(
                "current-spectrum method reference is missing",
            ));
        }
        if self.components.is_empty() {
            return Err(CapacitorSpectrumError::InvalidSpectrum(
                "current spectrum must contain at least one AC component",
            ));
        }
        let mut previous_frequency = 0.0;
        for component in &self.components {
            if !component.frequency_hz.is_finite() || component.frequency_hz <= 0.0 {
                return Err(CapacitorSpectrumError::InvalidSpectrum(
                    "current-spectrum frequencies must be finite and positive; DC is excluded",
                ));
            }
            if component.frequency_hz <= previous_frequency {
                return Err(CapacitorSpectrumError::InvalidSpectrum(
                    "current-spectrum frequencies must be strictly increasing without duplicates",
                ));
            }
            if !component.current_rms_a.is_finite() || component.current_rms_a < 0.0 {
                return Err(CapacitorSpectrumError::InvalidSpectrum(
                    "current RMS magnitudes must be finite and non-negative",
                ));
            }
            previous_frequency = component.frequency_hz;
        }
        Ok(())
    }
}

/// Temperature and DC-bias applicability requirements for a complex sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorSpectrumScreeningRequirements {
    /// Intended device temperature, in kelvin.
    pub operating_temperature_k: f64,
    /// Maximum absolute deviation from sweep characterization temperature, in kelvin.
    pub max_temperature_deviation_k: f64,
    /// Intended DC bias, in volts; None means only an unbiassed/unspecified sweep may be used.
    pub operating_dc_bias_v: Option<f64>,
    /// Maximum allowed absolute difference between measured and operating DC bias, in volts.
    /// Must be present if and only if operating_dc_bias_v is present.
    pub max_absolute_dc_bias_deviation_v: Option<f64>,
    /// Optional maximum magnitude of impedance at any spectrum frequency, in ohms.
    pub max_impedance_magnitude_ohm: Option<f64>,
    /// Maximum total ESR/real-impedance dissipation under the declared spectrum, in watts.
    pub max_total_loss_power_w: f64,
    /// Optional maximum aggregate RMS ripple voltage, in volts.
    pub max_total_ripple_voltage_rms_v: Option<f64>,
    /// Optional thermal resistance from component to reference environment, in K/W.
    pub thermal_resistance_k_per_w: Option<f64>,
    /// Optional maximum first-order temperature rise, in kelvin.
    pub max_temperature_rise_k: Option<f64>,
    /// Allowed evidence origins; this is a policy allow-list, not a trust ranking.
    pub accepted_evidence_kinds: Vec<CapacitorDeviceEvidenceKind>,
}

impl CapacitorSpectrumScreeningRequirements {
    /// Validate conditions and limits before screening.
    pub fn validate(&self) -> Result<(), CapacitorSpectrumError> {
        if !self.operating_temperature_k.is_finite() || self.operating_temperature_k <= 0.0 {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "operating temperature must be finite and positive",
            ));
        }
        if !self.max_temperature_deviation_k.is_finite()
            || self.max_temperature_deviation_k < 0.0
        {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "maximum temperature deviation must be finite and non-negative",
            ));
        }
        match (self.operating_dc_bias_v, self.max_absolute_dc_bias_deviation_v) {
            (None, None) => {}
            (Some(bias), Some(deviation))
                if bias.is_finite() && deviation.is_finite() && deviation >= 0.0 => {}
            (Some(_), Some(_)) => {
                return Err(CapacitorSpectrumError::InvalidRequirements(
                    "DC bias and its finite non-negative tolerance are required together",
                ));
            }
            _ => {
                return Err(CapacitorSpectrumError::InvalidRequirements(
                    "DC bias and its tolerance must either both be present or both be absent",
                ));
            }
        }
        if self
            .max_impedance_magnitude_ohm
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "maximum impedance must be finite and non-negative",
            ));
        }
        if !self.max_total_loss_power_w.is_finite() || self.max_total_loss_power_w < 0.0 {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "maximum total loss must be finite and non-negative",
            ));
        }
        if self
            .max_total_ripple_voltage_rms_v
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "maximum RMS ripple voltage must be finite and non-negative",
            ));
        }
        if self
            .thermal_resistance_k_per_w
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "thermal resistance must be finite and positive",
            ));
        }
        if self.max_temperature_rise_k.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "maximum temperature rise must be finite and non-negative",
            ));
        }
        if self.max_temperature_rise_k.is_some() && self.thermal_resistance_k_per_w.is_none() {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "a temperature-rise constraint requires explicit thermal resistance",
            ));
        }
        if self.accepted_evidence_kinds.is_empty() {
            return Err(CapacitorSpectrumError::InvalidRequirements(
                "at least one sweep evidence origin must be allowed",
            ));
        }
        Ok(())
    }
}

/// Per-frequency result binding the current component to its impedance source knots.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacitorSpectrumComponentResult {
    /// Frequency, in hertz.
    pub frequency_hz: f64,
    /// RMS current magnitude, in amperes.
    pub current_rms_a: f64,
    /// Interpolated or measured real impedance, in ohms.
    pub impedance_real_ohm: f64,
    /// Interpolated or measured imaginary impedance, in ohms.
    pub impedance_imaginary_ohm: f64,
    /// Impedance magnitude, in ohms.
    pub impedance_magnitude_ohm: f64,
    /// RMS ripple voltage contribution of this distinct frequency, in volts.
    pub ripple_voltage_rms_v: f64,
    /// Real power dissipated at this frequency, in watts.
    pub dissipated_power_w: f64,
    /// Whether impedance was sampled directly or interpolated.
    pub evaluation_kind: ImpedanceEvaluationKind,
    /// Lower impedance-sweep knot used for interpolation, in hertz.
    pub lower_bracket_frequency_hz: f64,
    /// Upper impedance-sweep knot used for interpolation, in hertz.
    pub upper_bracket_frequency_hz: f64,
}

/// Aggregate results calculated from a declared multi-frequency current spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorSpectrumMetrics {
    /// Quadrature total RMS current across the distinct spectral frequencies, in amperes.
    pub total_current_rms_a: f64,
    /// Quadrature aggregate RMS ripple voltage across distinct frequencies, in volts.
    pub total_ripple_voltage_rms_v: f64,
    /// Sum of per-frequency real-power loss, in watts.
    pub total_loss_power_w: f64,
    /// Maximum impedance magnitude over the supplied current-spectrum frequencies, in ohms.
    pub maximum_component_impedance_ohm: f64,
    /// First-order steady-state temperature-rise estimate, if thermal resistance was supplied.
    pub modeled_temperature_rise_k: Option<f64>,
    /// Per-frequency results; each records its measured/interpolated impedance brackets.
    pub components: Vec<CapacitorSpectrumComponentResult>,
}

/// Final disposition of the spectrum-based capacitor screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorSpectrumDisposition {
    /// All declared constraints passed using the condition-matched sweep.
    PassesSpectrumScreen,
    /// One or more declared constraints failed.
    FailsDeclaredConstraints,
    /// The required evidence or condition-domain coverage is insufficient.
    NotEvaluable,
}

/// Reason a spectrum screen cannot be evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorSpectrumLimitation {
    /// Intended temperature is outside the permitted sweep-characterization window.
    TemperatureOutsideCharacterization,
    /// Operating DC bias differs beyond the permitted tolerance, or sweep bias is unknown.
    DcBiasOutsideCharacterization,
    /// One or more current-spectrum frequencies are not covered by the measured sweep.
    FrequencyOutsideMeasuredSweep,
    /// Sweep evidence origin is not allowed by the application requirements.
    EvidenceKindNotAllowed,
}

/// Constraint violations found from the aggregate spectrum calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorSpectrumConstraintViolation {
    /// A component impedance magnitude exceeds its limit.
    ComponentImpedanceAboveMaximum,
    /// Sum of per-frequency real-power loss exceeds the limit.
    TotalLossAboveMaximum,
    /// Aggregate RMS ripple voltage exceeds the limit.
    TotalRippleVoltageAboveMaximum,
    /// First-order temperature rise exceeds its limit.
    TemperatureRiseAboveMaximum,
}

/// Auditable spectrum-screen report retaining exact sweep, spectrum and requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorSpectrumScreeningReport {
    /// Exact input complex-impedance sweep and its provenance.
    pub sweep: CapacitorImpedanceSweep,
    /// Exact current spectrum and its extraction provenance.
    pub current_spectrum: CapacitorCurrentSpectrum,
    /// Exact requirements used for the disposition.
    pub requirements: CapacitorSpectrumScreeningRequirements,
    /// Result scoped to the explicit sweep and spectral model.
    pub disposition: CapacitorSpectrumDisposition,
    /// Reasons an evaluation could not be supported.
    pub limitations: Vec<CapacitorSpectrumLimitation>,
    /// Independent modeled constraint failures.
    pub violations: Vec<CapacitorSpectrumConstraintViolation>,
    /// Metrics are absent when applicability gates block evaluation.
    pub metrics: Option<CapacitorSpectrumMetrics>,
}

/// Input, applicability, or numerical error for sweep-based screening.
#[derive(Debug, Clone, PartialEq)]
pub enum CapacitorSpectrumError {
    /// Invalid impedance sweep or missing source/method context.
    InvalidSweep(&'static str),
    /// Invalid current spectrum or missing source/method context.
    InvalidSpectrum(&'static str),
    /// Invalid screening requirement contract.
    InvalidRequirements(&'static str),
    /// Requested frequency lies outside the observed impedance-sweep band.
    FrequencyOutsideMeasuredSweep {
        /// Frequency requested, in hertz.
        frequency_hz: f64,
        /// Minimum measured sweep frequency, in hertz.
        min_frequency_hz: f64,
        /// Maximum measured sweep frequency, in hertz.
        max_frequency_hz: f64,
    },
    /// Derived values overflowed or became non-finite.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for CapacitorSpectrumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSweep(message) => write!(f, "invalid capacitor impedance sweep: {message}"),
            Self::InvalidSpectrum(message) => write!(f, "invalid capacitor current spectrum: {message}"),
            Self::InvalidRequirements(message) => {
                write!(f, "invalid capacitor spectrum requirements: {message}")
            }
            Self::FrequencyOutsideMeasuredSweep {
                frequency_hz,
                min_frequency_hz,
                max_frequency_hz,
            } => write!(
                f,
                "frequency {frequency_hz} Hz lies outside measured sweep [{min_frequency_hz}, {max_frequency_hz}] Hz"
            ),
            Self::NonFiniteCalculatedMetric => {
                write!(f, "non-finite result in capacitor spectrum calculation")
            }
        }
    }
}

impl Error for CapacitorSpectrumError {}

/// Calculate spectrum-weighted loss and ripple from a condition-matched complex sweep.
///
/// Distinct spectral components are assumed to be steady-state orthogonal sinusoidal
/// components. Their RMS ripple voltages combine by root-sum-square; real-power losses
/// sum as I_rms² times the real impedance evaluated at each component frequency. No
/// phase-resolved peak voltage is inferred. All frequencies must be bracketed by the
/// supplied sweep.
pub fn calculate_capacitor_spectrum_metrics(
    sweep: &CapacitorImpedanceSweep,
    spectrum: &CapacitorCurrentSpectrum,
    thermal_resistance_k_per_w: Option<f64>,
) -> Result<CapacitorSpectrumMetrics, CapacitorSpectrumError> {
    sweep.validate()?;
    spectrum.validate()?;
    if thermal_resistance_k_per_w.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(CapacitorSpectrumError::InvalidRequirements(
            "thermal resistance must be finite and positive",
        ));
    }

    let mut components = Vec::with_capacity(spectrum.components.len());
    let mut sum_current_squared = 0.0;
    let mut sum_voltage_squared = 0.0;
    let mut total_loss_power_w = 0.0;
    let mut maximum_component_impedance_ohm: f64 = 0.0;

    for current in &spectrum.components {
        let impedance = sweep.evaluate(current.frequency_hz)?;
        let magnitude = impedance.magnitude_ohm();
        let ripple_voltage = current.current_rms_a * magnitude;
        let dissipated_power = current.current_rms_a.powi(2) * impedance.real_ohm;

        sum_current_squared += current.current_rms_a.powi(2);
        sum_voltage_squared += ripple_voltage.powi(2);
        total_loss_power_w += dissipated_power;
        maximum_component_impedance_ohm = maximum_component_impedance_ohm.max(magnitude);

        components.push(CapacitorSpectrumComponentResult {
            frequency_hz: current.frequency_hz,
            current_rms_a: current.current_rms_a,
            impedance_real_ohm: impedance.real_ohm,
            impedance_imaginary_ohm: impedance.imaginary_ohm,
            impedance_magnitude_ohm: magnitude,
            ripple_voltage_rms_v: ripple_voltage,
            dissipated_power_w: dissipated_power,
            evaluation_kind: impedance.evaluation_kind,
            lower_bracket_frequency_hz: impedance.lower_bracket_frequency_hz,
            upper_bracket_frequency_hz: impedance.upper_bracket_frequency_hz,
        });
    }

    let total_current_rms_a = sum_current_squared.sqrt();
    let total_ripple_voltage_rms_v = sum_voltage_squared.sqrt();
    let modeled_temperature_rise_k =
        thermal_resistance_k_per_w.map(|resistance| total_loss_power_w * resistance);
    let scalars = [
        total_current_rms_a,
        total_ripple_voltage_rms_v,
        total_loss_power_w,
        maximum_component_impedance_ohm,
    ];
    if scalars.into_iter().any(|value| !value.is_finite())
        || modeled_temperature_rise_k.is_some_and(|value| !value.is_finite())
        || components.iter().any(|component| {
            !component.impedance_magnitude_ohm.is_finite()
                || !component.ripple_voltage_rms_v.is_finite()
                || !component.dissipated_power_w.is_finite()
        })
    {
        return Err(CapacitorSpectrumError::NonFiniteCalculatedMetric);
    }

    Ok(CapacitorSpectrumMetrics {
        total_current_rms_a,
        total_ripple_voltage_rms_v,
        total_loss_power_w,
        maximum_component_impedance_ohm,
        modeled_temperature_rise_k,
        components,
    })
}

/// Screen a current spectrum against a condition-bound complex impedance sweep.
///
/// Any unsupported temperature, bias, evidence class or frequency coverage produces
/// NotEvaluable; no model is extrapolated. A pass remains an analytic spectrum/loss
/// result under the declared measured sweep and steady-state RMS assumptions.
pub fn screen_capacitor_spectrum(
    sweep: &CapacitorImpedanceSweep,
    spectrum: &CapacitorCurrentSpectrum,
    requirements: &CapacitorSpectrumScreeningRequirements,
) -> Result<CapacitorSpectrumScreeningReport, CapacitorSpectrumError> {
    sweep.validate()?;
    spectrum.validate()?;
    requirements.validate()?;

    let mut limitations = Vec::new();
    if (requirements.operating_temperature_k - sweep.characterization_temperature_k).abs()
        > requirements.max_temperature_deviation_k
    {
        limitations.push(CapacitorSpectrumLimitation::TemperatureOutsideCharacterization);
    }
    if let Some(operating_bias) = requirements.operating_dc_bias_v {
        match sweep.dc_bias_v {
            Some(characterized_bias)
                if (operating_bias - characterized_bias).abs()
                    <= requirements.max_absolute_dc_bias_deviation_v.unwrap_or(0.0) => {}
            _ => limitations.push(CapacitorSpectrumLimitation::DcBiasOutsideCharacterization),
        }
    } else if sweep.dc_bias_v.is_some() {
        limitations.push(CapacitorSpectrumLimitation::DcBiasOutsideCharacterization);
    }
    if !requirements.accepted_evidence_kinds.contains(&sweep.evidence_kind) {
        limitations.push(CapacitorSpectrumLimitation::EvidenceKindNotAllowed);
    }
    let min_frequency = sweep.min_frequency_hz().expect("validated sweep has points");
    let max_frequency = sweep.max_frequency_hz().expect("validated sweep has points");
    if spectrum.components.iter().any(|component| {
        component.frequency_hz < min_frequency || component.frequency_hz > max_frequency
    }) {
        limitations.push(CapacitorSpectrumLimitation::FrequencyOutsideMeasuredSweep);
    }

    if !limitations.is_empty() {
        return Ok(CapacitorSpectrumScreeningReport {
            sweep: sweep.clone(),
            current_spectrum: spectrum.clone(),
            requirements: requirements.clone(),
            disposition: CapacitorSpectrumDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            metrics: None,
        });
    }

    let metrics = calculate_capacitor_spectrum_metrics(
        sweep,
        spectrum,
        requirements.thermal_resistance_k_per_w,
    )?;
    let mut violations = Vec::new();
    if requirements.max_impedance_magnitude_ohm.is_some_and(|maximum| {
        metrics.maximum_component_impedance_ohm > maximum
    }) {
        violations.push(CapacitorSpectrumConstraintViolation::ComponentImpedanceAboveMaximum);
    }
    if metrics.total_loss_power_w > requirements.max_total_loss_power_w {
        violations.push(CapacitorSpectrumConstraintViolation::TotalLossAboveMaximum);
    }
    if requirements.max_total_ripple_voltage_rms_v.is_some_and(|maximum| {
        metrics.total_ripple_voltage_rms_v > maximum
    }) {
        violations.push(CapacitorSpectrumConstraintViolation::TotalRippleVoltageAboveMaximum);
    }
    if requirements.max_temperature_rise_k.is_some_and(|maximum| {
        metrics.modeled_temperature_rise_k.is_none_or(|rise| rise > maximum)
    }) {
        violations.push(CapacitorSpectrumConstraintViolation::TemperatureRiseAboveMaximum);
    }
    let disposition = if violations.is_empty() {
        CapacitorSpectrumDisposition::PassesSpectrumScreen
    } else {
        CapacitorSpectrumDisposition::FailsDeclaredConstraints
    };
    Ok(CapacitorSpectrumScreeningReport {
        sweep: sweep.clone(),
        current_spectrum: spectrum.clone(),
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

    fn sweep() -> CapacitorImpedanceSweep {
        CapacitorImpedanceSweep {
            device_id: "synthetic-impedance-sweep".to_string(),
            characterization_temperature_k: 298.15,
            dc_bias_v: Some(1.0),
            source_reference: "test-fixture://not-real-sweep".to_string(),
            method_reference: "synthetic analytic sweep fixture".to_string(),
            evidence_kind: CapacitorDeviceEvidenceKind::MeasuredDevice,
            points: vec![
                ImpedanceSweepPoint { frequency_hz: 1.0e3, real_ohm: 0.1, imaginary_ohm: -10.0 },
                ImpedanceSweepPoint { frequency_hz: 1.0e4, real_ohm: 0.2, imaginary_ohm: -1.0 },
                ImpedanceSweepPoint { frequency_hz: 1.0e5, real_ohm: 0.5, imaginary_ohm: 2.0 },
                ImpedanceSweepPoint { frequency_hz: 1.0e6, real_ohm: 0.8, imaginary_ohm: 10.0 },
            ],
        }
    }

    fn spectrum() -> CapacitorCurrentSpectrum {
        CapacitorCurrentSpectrum {
            source_reference: "test-fixture://not-real-current-spectrum".to_string(),
            method_reference: "synthetic FFT RMS fixture with declared normalization".to_string(),
            components: vec![
                CurrentSpectrumComponent { frequency_hz: 1.0e3, current_rms_a: 0.5 },
                CurrentSpectrumComponent { frequency_hz: 1.0e5, current_rms_a: 0.2 },
            ],
        }
    }

    fn requirements() -> CapacitorSpectrumScreeningRequirements {
        CapacitorSpectrumScreeningRequirements {
            operating_temperature_k: 298.15,
            max_temperature_deviation_k: 2.0,
            operating_dc_bias_v: Some(1.0),
            max_absolute_dc_bias_deviation_v: Some(0.1),
            max_impedance_magnitude_ohm: Some(11.0),
            max_total_loss_power_w: 0.06,
            max_total_ripple_voltage_rms_v: Some(6.0),
            thermal_resistance_k_per_w: Some(10.0),
            max_temperature_rise_k: Some(0.6),
            accepted_evidence_kinds: vec![CapacitorDeviceEvidenceKind::MeasuredDevice],
        }
    }

    #[test]
    fn exact_sweep_knot_preserves_measurement_without_interpolation() {
        let result = sweep().evaluate(1.0e5).unwrap();
        assert_eq!(result.evaluation_kind, ImpedanceEvaluationKind::MeasuredKnot);
        assert_eq!(result.real_ohm, 0.5);
        assert_eq!(result.imaginary_ohm, 2.0);
        assert_eq!(result.lower_bracket_frequency_hz, 1.0e5);
        assert_eq!(result.upper_bracket_frequency_hz, 1.0e5);
    }

    #[test]
    fn log_frequency_interpolation_retains_bracketing_knots() {
        let result = sweep().evaluate(1.0e4_f64.sqrt() * 1.0e3).unwrap();
        assert_eq!(result.evaluation_kind, ImpedanceEvaluationKind::InterpolatedBetweenKnots);
        assert_eq!(result.lower_bracket_frequency_hz, 1.0e3);
        assert_eq!(result.upper_bracket_frequency_hz, 1.0e4);
        assert!((result.real_ohm - 0.15).abs() <= 1.0e-12);
    }

    #[test]
    fn outside_sweep_frequency_cannot_be_extrapolated() {
        assert!(matches!(
            sweep().evaluate(100.0),
            Err(CapacitorSpectrumError::FrequencyOutsideMeasuredSweep { .. })
        ));
        let report = screen_capacitor_spectrum(
            &sweep(),
            &CapacitorCurrentSpectrum {
                components: vec![CurrentSpectrumComponent { frequency_hz: 100.0, current_rms_a: 0.1 }],
                ..spectrum()
            },
            &requirements(),
        )
        .unwrap();
        assert_eq!(report.disposition, CapacitorSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(&CapacitorSpectrumLimitation::FrequencyOutsideMeasuredSweep));
        assert!(report.metrics.is_none());
    }

    #[test]
    fn spectrum_loss_uses_each_frequency_real_impedance() {
        let result = calculate_capacitor_spectrum_metrics(&sweep(), &spectrum(), None).unwrap();
        let expected = 0.5_f64.powi(2) * 0.1 + 0.2_f64.powi(2) * 0.5;
        assert!((result.total_loss_power_w - expected).abs() <= 1.0e-12);
        assert!((result.total_current_rms_a - (0.5_f64.powi(2) + 0.2_f64.powi(2)).sqrt()).abs() <= 1.0e-12);
        assert_eq!(result.components[0].evaluation_kind, ImpedanceEvaluationKind::MeasuredKnot);
        assert_eq!(result.components[1].evaluation_kind, ImpedanceEvaluationKind::MeasuredKnot);
    }

    #[test]
    fn spectrum_ripple_voltage_uses_quadrature_over_distinct_frequencies() {
        let result = calculate_capacitor_spectrum_metrics(&sweep(), &spectrum(), None).unwrap();
        let v0 = 0.5 * 0.1_f64.hypot(10.0);
        let v1 = 0.2 * 0.5_f64.hypot(2.0);
        assert!((result.total_ripple_voltage_rms_v - (v0.powi(2) + v1.powi(2)).sqrt()).abs() <= 1.0e-12);
    }

    #[test]
    fn aggregate_loss_drives_only_declared_first_order_thermal_estimate() {
        let result = calculate_capacitor_spectrum_metrics(&sweep(), &spectrum(), Some(10.0)).unwrap();
        assert!((result.modeled_temperature_rise_k.unwrap() - result.total_loss_power_w * 10.0).abs() <= 1.0e-12);
    }

    #[test]
    fn spectrum_screen_passes_synthetic_in_domain_fixture() {
        let report = screen_capacitor_spectrum(&sweep(), &spectrum(), &requirements()).unwrap();
        assert_eq!(report.disposition, CapacitorSpectrumDisposition::PassesSpectrumScreen);
        assert!(report.violations.is_empty());
        assert!(report.metrics.is_some());
    }

    #[test]
    fn temperature_mismatch_is_not_evaluable() {
        let mut req = requirements();
        req.operating_temperature_k = 320.0;
        let report = screen_capacitor_spectrum(&sweep(), &spectrum(), &req).unwrap();
        assert_eq!(report.disposition, CapacitorSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(&CapacitorSpectrumLimitation::TemperatureOutsideCharacterization));
    }

    #[test]
    fn dc_bias_mismatch_is_not_evaluable() {
        let mut req = requirements();
        req.operating_dc_bias_v = Some(3.0);
        let report = screen_capacitor_spectrum(&sweep(), &spectrum(), &req).unwrap();
        assert_eq!(report.disposition, CapacitorSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(&CapacitorSpectrumLimitation::DcBiasOutsideCharacterization));
    }

    #[test]
    fn descending_or_duplicate_sweep_frequency_is_rejected() {
        let mut sample = sweep();
        sample.points.swap(0, 1);
        assert!(matches!(sample.validate(), Err(CapacitorSpectrumError::InvalidSweep(_))));
        sample = sweep();
        sample.points[1].frequency_hz = sample.points[0].frequency_hz;
        assert!(matches!(sample.validate(), Err(CapacitorSpectrumError::InvalidSweep(_))));
    }

    #[test]
    fn passive_contract_rejects_negative_real_impedance() {
        let mut sample = sweep();
        sample.points[0].real_ohm = -0.1;
        assert!(matches!(sample.validate(), Err(CapacitorSpectrumError::InvalidSweep(_))));
    }

    #[test]
    fn duplicate_or_dc_spectral_frequency_is_rejected() {
        let mut sample = spectrum();
        sample.components[1].frequency_hz = sample.components[0].frequency_hz;
        assert!(matches!(sample.validate(), Err(CapacitorSpectrumError::InvalidSpectrum(_))));
        sample = spectrum();
        sample.components[0].frequency_hz = 0.0;
        assert!(matches!(sample.validate(), Err(CapacitorSpectrumError::InvalidSpectrum(_))));
    }

    #[test]
    fn thermal_rise_constraint_requires_explicit_thermal_resistance() {
        let mut req = requirements();
        req.thermal_resistance_k_per_w = None;
        assert!(matches!(req.validate(), Err(CapacitorSpectrumError::InvalidRequirements(_))));
    }

    #[test]
    fn loss_constraint_is_independent_of_impedance_and_voltage_constraints() {
        let mut req = requirements();
        req.max_total_loss_power_w = 0.01;
        req.max_total_ripple_voltage_rms_v = Some(100.0);
        req.max_impedance_magnitude_ohm = Some(100.0);
        let report = screen_capacitor_spectrum(&sweep(), &spectrum(), &req).unwrap();
        assert_eq!(report.disposition, CapacitorSpectrumDisposition::FailsDeclaredConstraints);
        assert!(report.violations.contains(&CapacitorSpectrumConstraintViolation::TotalLossAboveMaximum));
    }
}
