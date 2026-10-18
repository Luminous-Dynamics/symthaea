// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frequency-dependent impedance-magnitude and ESR sweeps without invented phase.
//!
//! Some published capacitor characterization datasets expose |Z(f)| and ESR(f)
//! but not complex impedance phase. This module models exactly that evidence shape:
//! it does not infer the sign of the imaginary component or re-label magnitude/ESR
//! as a complex impedance measurement. It supports per-frequency ripple-voltage
//! and real-loss estimation when a separately sourced AC current spectrum exists.
//!
//! Log-frequency interpolation is only a nominal approximation between measured
//! knots, never an extrapolation. This module does not replace measured current
//! spectra, ngspice, electrothermal simulation, lifetime prediction or qualification.

use crate::capacitor_device::CapacitorDeviceEvidenceKind;
use crate::capacitor_spectrum::{
    CapacitorCurrentSpectrum, CapacitorSpectrumError, ImpedanceEvaluationKind,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Relative tolerance for consistency checks between measured ESR and impedance magnitude.
const ESR_MAGNITUDE_RELATIVE_TOLERANCE: f64 = 1.0e-6;
/// Absolute tolerance in ohms for ESR/magnitude consistency when measurements round near zero.
const ESR_MAGNITUDE_ABSOLUTE_TOLERANCE_OHM: f64 = 1.0e-9;

/// The DC-bias condition associated with a measured sweep.
///
/// Unknown is not equivalent to zero volts and cannot support a bias-matched screen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CapacitorDcBiasCondition {
    /// The measurement was explicitly taken with no applied DC bias.
    ExplicitlyUnbiased,
    /// The explicitly recorded applied DC bias, in volts.
    AppliedVolts(f64),
    /// The source did not establish the applied DC-bias condition.
    Unknown,
}

impl CapacitorDcBiasCondition {
    /// Validate an applied bias value, leaving explicitly unbiased/unknown states intact.
    pub fn validate(&self) -> Result<(), MagnitudeEsrSweepError> {
        if let Self::AppliedVolts(value) = self {
            if !value.is_finite() {
                return Err(MagnitudeEsrSweepError::InvalidSweep(
                    "applied DC bias must be finite",
                ));
            }
        }
        Ok(())
    }
}

/// One nominal measurement at a positive frequency, without inferred complex phase.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrSweepPoint {
    /// Frequency in hertz; sweep knots must be strictly increasing and unique.
    pub frequency_hz: f64,
    /// Measured impedance magnitude |Z| in ohms.
    pub impedance_magnitude_ohm: f64,
    /// Measured equivalent series resistance in ohms.
    pub esr_ohm: f64,
}

/// One source-bound impedance-magnitude/ESR sweep under a single condition.
///
/// Required source, method, and license references preserve the distinction between
/// the original dataset and any normalized derivative. Unknown specimen identity and
/// unknown bias remain explicit instead of being guessed from filenames or materials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorMagnitudeEsrSweep {
    /// Stable sweep/device identifier from the source manifest.
    pub device_id: String,
    /// Specimen/lot identifier if the source provides one.
    pub specimen_id: Option<String>,
    /// Material/product group identifier if known from source evidence.
    pub group_id: Option<String>,
    /// Characterization temperature in kelvin.
    pub characterization_temperature_k: f64,
    /// Explicit characterization bias state; unknown is not zero.
    pub dc_bias: CapacitorDcBiasCondition,
    /// RMS AC stimulus amplitude used for characterization, in volts, when known.
    pub ac_test_voltage_rms_v: Option<f64>,
    /// Stable source reference including dataset DOI/version and a file/row locator.
    pub source_reference: String,
    /// Method reference including instrument, fixture, frequency span and measurement settings.
    pub method_reference: String,
    /// License reference for the source and any applicable derivative.
    pub license_reference: String,
    /// Dataset or measurement release version.
    pub dataset_version: Option<String>,
    /// Evidence origin; not a universal ordinal quality score.
    pub evidence_kind: CapacitorDeviceEvidenceKind,
    /// Ordered frequency-dependent magnitude/ESR observations.
    pub points: Vec<MagnitudeEsrSweepPoint>,
}

impl CapacitorMagnitudeEsrSweep {
    /// Validate conditions, source/method/license provenance, ordering, and ESR/|Z| consistency.
    pub fn validate(&self) -> Result<(), MagnitudeEsrSweepError> {
        if self.device_id.trim().is_empty() {
            return Err(MagnitudeEsrSweepError::InvalidSweep("sweep/device ID is empty"));
        }
        if self
            .specimen_id
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
            || self.group_id.as_ref().is_some_and(|value| value.trim().is_empty())
        {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "optional specimen/group identifiers must not be empty when present",
            ));
        }
        if !self.characterization_temperature_k.is_finite()
            || self.characterization_temperature_k <= 0.0
        {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "characterization temperature must be finite and positive",
            ));
        }
        self.dc_bias.validate()?;
        if self
            .ac_test_voltage_rms_v
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "AC characterization voltage must be finite and positive when present",
            ));
        }
        if self.source_reference.trim().is_empty() {
            return Err(MagnitudeEsrSweepError::InvalidSweep("source reference is missing"));
        }
        if self.method_reference.trim().is_empty() {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "measurement method reference is missing",
            ));
        }
        if self.license_reference.trim().is_empty() {
            return Err(MagnitudeEsrSweepError::InvalidSweep("license reference is missing"));
        }
        if self
            .dataset_version
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "dataset version must not be empty when present",
            ));
        }
        if self.points.len() < 2 {
            return Err(MagnitudeEsrSweepError::InvalidSweep(
                "at least two frequency knots are required",
            ));
        }

        let mut previous_frequency_hz = 0.0;
        for point in &self.points {
            if !point.frequency_hz.is_finite() || point.frequency_hz <= 0.0 {
                return Err(MagnitudeEsrSweepError::InvalidSweep(
                    "sweep frequencies must be finite and positive",
                ));
            }
            if point.frequency_hz <= previous_frequency_hz {
                return Err(MagnitudeEsrSweepError::InvalidSweep(
                    "sweep frequencies must be strictly increasing without duplicates",
                ));
            }
            if !point.impedance_magnitude_ohm.is_finite()
                || point.impedance_magnitude_ohm < 0.0
            {
                return Err(MagnitudeEsrSweepError::InvalidSweep(
                    "impedance magnitude must be finite and non-negative",
                ));
            }
            if !point.esr_ohm.is_finite() || point.esr_ohm < 0.0 {
                return Err(MagnitudeEsrSweepError::InvalidSweep(
                    "ESR must be finite and non-negative",
                ));
            }
            let tolerance = ESR_MAGNITUDE_ABSOLUTE_TOLERANCE_OHM
                + ESR_MAGNITUDE_RELATIVE_TOLERANCE
                    * point.impedance_magnitude_ohm.max(point.esr_ohm);
            if point.esr_ohm > point.impedance_magnitude_ohm + tolerance {
                return Err(MagnitudeEsrSweepError::EsrAboveMagnitude {
                    frequency_hz: point.frequency_hz,
                    esr_ohm: point.esr_ohm,
                    impedance_magnitude_ohm: point.impedance_magnitude_ohm,
                });
            }
            previous_frequency_hz = point.frequency_hz;
        }
        Ok(())
    }

    /// Lowest measured frequency, in hertz.
    pub fn min_frequency_hz(&self) -> Option<f64> {
        self.points.first().map(|point| point.frequency_hz)
    }

    /// Highest measured frequency, in hertz.
    pub fn max_frequency_hz(&self) -> Option<f64> {
        self.points.last().map(|point| point.frequency_hz)
    }

    /// Evaluate |Z| and ESR at a frequency, interpolating log-frequency only within the sweep.
    ///
    /// The result keeps its exact-vs-interpolated status and source bracket. No phase
    /// or complex impedance is fabricated, and frequencies outside the sweep fail closed.
    pub fn evaluate(
        &self,
        frequency_hz: f64,
    ) -> Result<MagnitudeEsrEvaluation, MagnitudeEsrSweepError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MagnitudeEsrSweepError::InvalidEvaluationFrequency);
        }
        let first = self.points.first().expect("validated point count");
        let last = self.points.last().expect("validated point count");
        if frequency_hz < first.frequency_hz || frequency_hz > last.frequency_hz {
            return Err(MagnitudeEsrSweepError::FrequencyOutsideMeasuredSweep {
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
                Ok(MagnitudeEsrEvaluation {
                    frequency_hz,
                    impedance_magnitude_ohm: point.impedance_magnitude_ohm,
                    esr_ohm: point.esr_ohm,
                    evaluation_kind: ImpedanceEvaluationKind::MeasuredKnot,
                    lower_bracket_frequency_hz: point.frequency_hz,
                    upper_bracket_frequency_hz: point.frequency_hz,
                })
            }
            Err(upper_index) => {
                let lower = self.points[upper_index - 1];
                let upper = self.points[upper_index];
                let fraction = (frequency_hz.ln() - lower.frequency_hz.ln())
                    / (upper.frequency_hz.ln() - lower.frequency_hz.ln());
                let impedance_magnitude_ohm = lower.impedance_magnitude_ohm
                    + fraction * (upper.impedance_magnitude_ohm - lower.impedance_magnitude_ohm);
                let esr_ohm = lower.esr_ohm + fraction * (upper.esr_ohm - lower.esr_ohm);
                if !impedance_magnitude_ohm.is_finite()
                    || impedance_magnitude_ohm < 0.0
                    || !esr_ohm.is_finite()
                    || esr_ohm < 0.0
                {
                    return Err(MagnitudeEsrSweepError::NonFiniteCalculatedMetric);
                }
                let tolerance = ESR_MAGNITUDE_ABSOLUTE_TOLERANCE_OHM
                    + ESR_MAGNITUDE_RELATIVE_TOLERANCE
                        * impedance_magnitude_ohm.max(esr_ohm);
                if esr_ohm > impedance_magnitude_ohm + tolerance {
                    return Err(MagnitudeEsrSweepError::EsrAboveMagnitude {
                        frequency_hz,
                        esr_ohm,
                        impedance_magnitude_ohm,
                    });
                }
                Ok(MagnitudeEsrEvaluation {
                    frequency_hz,
                    impedance_magnitude_ohm,
                    esr_ohm,
                    evaluation_kind: ImpedanceEvaluationKind::InterpolatedBetweenKnots,
                    lower_bracket_frequency_hz: lower.frequency_hz,
                    upper_bracket_frequency_hz: upper.frequency_hz,
                })
            }
        }
    }
}

/// Evaluated impedance magnitude and ESR without any phase inference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrEvaluation {
    /// Frequency evaluated, in hertz.
    pub frequency_hz: f64,
    /// Impedance magnitude |Z| in ohms.
    pub impedance_magnitude_ohm: f64,
    /// ESR in ohms.
    pub esr_ohm: f64,
    /// Exact measured knot or nominal interpolation.
    pub evaluation_kind: ImpedanceEvaluationKind,
    /// Lower source frequency bracket in hertz.
    pub lower_bracket_frequency_hz: f64,
    /// Upper source frequency bracket in hertz.
    pub upper_bracket_frequency_hz: f64,
}

/// Requirements for applying a magnitude/ESR sweep to one operating context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrScreeningRequirements {
    /// Intended operating temperature in kelvin.
    pub operating_temperature_k: f64,
    /// Maximum allowed deviation from characterization temperature, in kelvin.
    pub max_temperature_deviation_k: f64,
    /// Intended DC-bias condition; unknown does not mean zero.
    pub operating_dc_bias: CapacitorDcBiasCondition,
    /// Maximum allowed difference when both sweep and operation have explicit applied bias.
    pub max_absolute_dc_bias_deviation_v: f64,
    /// Allowed sweep evidence origins.
    pub accepted_evidence_kinds: Vec<CapacitorDeviceEvidenceKind>,
    /// Maximum total real loss under the supplied current spectrum, in watts.
    pub max_total_loss_power_w: f64,
    /// Optional maximum aggregate RMS ripple voltage, in volts.
    pub max_total_ripple_voltage_rms_v: Option<f64>,
    /// Optional maximum impedance magnitude for any active spectrum component, in ohms.
    pub max_impedance_magnitude_ohm: Option<f64>,
    /// Optional steady-state thermal resistance in kelvin per watt.
    pub thermal_resistance_k_per_w: Option<f64>,
    /// Optional maximum first-order temperature rise, in kelvin.
    pub max_temperature_rise_k: Option<f64>,
}

impl MagnitudeEsrScreeningRequirements {
    /// Validate the application constraints.
    pub fn validate(&self) -> Result<(), MagnitudeEsrSweepError> {
        if !self.operating_temperature_k.is_finite() || self.operating_temperature_k <= 0.0 {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "operating temperature must be finite and positive",
            ));
        }
        if !self.max_temperature_deviation_k.is_finite()
            || self.max_temperature_deviation_k < 0.0
        {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum temperature deviation must be finite and non-negative",
            ));
        }
        self.operating_dc_bias.validate()?;
        if !self.max_absolute_dc_bias_deviation_v.is_finite()
            || self.max_absolute_dc_bias_deviation_v < 0.0
        {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum DC-bias deviation must be finite and non-negative",
            ));
        }
        if self.accepted_evidence_kinds.is_empty() {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "at least one evidence origin must be allowed",
            ));
        }
        if !self.max_total_loss_power_w.is_finite() || self.max_total_loss_power_w < 0.0 {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum total loss must be finite and non-negative",
            ));
        }
        if self
            .max_total_ripple_voltage_rms_v
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum RMS ripple voltage must be finite and non-negative",
            ));
        }
        if self
            .max_impedance_magnitude_ohm
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum impedance magnitude must be finite and non-negative",
            ));
        }
        if self
            .thermal_resistance_k_per_w
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "thermal resistance must be finite and positive",
            ));
        }
        if self.max_temperature_rise_k.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "maximum temperature rise must be finite and non-negative",
            ));
        }
        if self.max_temperature_rise_k.is_some() && self.thermal_resistance_k_per_w.is_none() {
            return Err(MagnitudeEsrSweepError::InvalidRequirements(
                "a temperature-rise limit requires explicit thermal resistance",
            ));
        }
        Ok(())
    }
}

/// Per-frequency result from magnitude/ESR spectrum evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrSpectrumComponentResult {
    /// Frequency, in hertz.
    pub frequency_hz: f64,
    /// RMS current, in amperes.
    pub current_rms_a: f64,
    /// Evaluated impedance magnitude |Z|, in ohms.
    pub impedance_magnitude_ohm: f64,
    /// Evaluated ESR, in ohms.
    pub esr_ohm: f64,
    /// RMS voltage contribution, in volts.
    pub ripple_voltage_rms_v: f64,
    /// Real loss contribution I_rms² * ESR, in watts.
    pub dissipated_power_w: f64,
    /// Exact knot or nominal interpolation.
    pub evaluation_kind: ImpedanceEvaluationKind,
    /// Lower source bracket, in hertz.
    pub lower_bracket_frequency_hz: f64,
    /// Upper source bracket, in hertz.
    pub upper_bracket_frequency_hz: f64,
}

/// Aggregate RMS/loss outputs from an impedance-magnitude/ESR sweep and current spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrSpectrumMetrics {
    /// Quadrature RMS current over distinct spectral components, in amperes.
    pub total_current_rms_a: f64,
    /// Quadrature aggregate RMS ripple voltage, in volts.
    pub total_ripple_voltage_rms_v: f64,
    /// Sum of per-frequency ESR losses, in watts.
    pub total_loss_power_w: f64,
    /// Maximum evaluated impedance magnitude among active spectrum components, in ohms.
    pub maximum_component_impedance_ohm: f64,
    /// First-order steady-state temperature rise if thermal resistance is supplied, in kelvin.
    pub modeled_temperature_rise_k: Option<f64>,
    /// Individual spectral component results.
    pub components: Vec<MagnitudeEsrSpectrumComponentResult>,
}

/// Final outcome of magnitude/ESR spectrum screening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagnitudeEsrScreeningDisposition {
    /// Every requirement passes in the declared magnitude/ESR approximation.
    PassesMagnitudeEsrScreen,
    /// At least one declared requirement fails.
    FailsDeclaredConstraints,
    /// Conditions/evidence do not support evaluating the operating context.
    NotEvaluable,
}

/// Explicit limitations that block a magnitude/ESR screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagnitudeEsrScreeningLimitation {
    /// Operating temperature is outside the characterization tolerance.
    TemperatureOutsideCharacterization,
    /// Either bias is unknown or applied biases differ beyond tolerance.
    DcBiasOutsideCharacterization,
    /// Evidence origin is not allowed.
    EvidenceKindNotAllowed,
    /// One or more current-spectrum frequencies are outside measured sweep coverage.
    FrequencyOutsideMeasuredSweep,
}

/// Independent magnitude/ESR constraints that fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagnitudeEsrConstraintViolation {
    /// Maximum active component impedance exceeds its limit.
    ImpedanceMagnitudeAboveMaximum,
    /// Total per-frequency ESR loss exceeds its limit.
    TotalLossAboveMaximum,
    /// Aggregate ripple voltage exceeds its limit.
    TotalRippleVoltageAboveMaximum,
    /// First-order temperature rise exceeds its limit.
    TemperatureRiseAboveMaximum,
}

/// Auditable result retaining exact sweep, current spectrum, requirements and outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeEsrScreeningReport {
    /// Exact magnitude/ESR sweep with source conditions and license provenance.
    pub sweep: CapacitorMagnitudeEsrSweep,
    /// Separately sourced current spectrum used in the screen.
    pub current_spectrum: CapacitorCurrentSpectrum,
    /// Exact operating requirements.
    pub requirements: MagnitudeEsrScreeningRequirements,
    /// Result applies only to the declared measurement/model domain.
    pub disposition: MagnitudeEsrScreeningDisposition,
    /// Reasons a disposition could not be supported.
    pub limitations: Vec<MagnitudeEsrScreeningLimitation>,
    /// Independent constraint failures.
    pub violations: Vec<MagnitudeEsrConstraintViolation>,
    /// Derived metrics; absent when a domain/evidence limitation blocks evaluation.
    pub metrics: Option<MagnitudeEsrSpectrumMetrics>,
}

/// Input, condition, or numerical error in magnitude/ESR sweep evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum MagnitudeEsrSweepError {
    /// Malformed sweep or missing provenance.
    InvalidSweep(&'static str),
    /// ESR materially exceeds impedance magnitude at a supplied or interpolated point.
    EsrAboveMagnitude {
        /// Frequency in hertz.
        frequency_hz: f64,
        /// ESR in ohms.
        esr_ohm: f64,
        /// Impedance magnitude in ohms.
        impedance_magnitude_ohm: f64,
    },
    /// Invalid target frequency.
    InvalidEvaluationFrequency,
    /// Requested frequency is outside the measured sweep.
    FrequencyOutsideMeasuredSweep {
        /// Requested frequency in hertz.
        frequency_hz: f64,
        /// Lowest measured frequency in hertz.
        min_frequency_hz: f64,
        /// Highest measured frequency in hertz.
        max_frequency_hz: f64,
    },
    /// Malformed operating requirements.
    InvalidRequirements(&'static str),
    /// A current-spectrum contract failed validation.
    CurrentSpectrum(CapacitorSpectrumError),
    /// Derived scalar is non-finite.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for MagnitudeEsrSweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSweep(message) => write!(f, "invalid magnitude/ESR sweep: {message}"),
            Self::EsrAboveMagnitude {
                frequency_hz,
                esr_ohm,
                impedance_magnitude_ohm,
            } => write!(
                f,
                "ESR {esr_ohm} ohm exceeds impedance magnitude {impedance_magnitude_ohm} ohm at {frequency_hz} Hz"
            ),
            Self::InvalidEvaluationFrequency => {
                write!(f, "evaluation frequency must be finite and positive")
            }
            Self::FrequencyOutsideMeasuredSweep {
                frequency_hz,
                min_frequency_hz,
                max_frequency_hz,
            } => write!(
                f,
                "frequency {frequency_hz} Hz lies outside measured sweep [{min_frequency_hz}, {max_frequency_hz}] Hz"
            ),
            Self::InvalidRequirements(message) => {
                write!(f, "invalid magnitude/ESR requirements: {message}")
            }
            Self::CurrentSpectrum(error) => write!(f, "invalid current spectrum: {error}"),
            Self::NonFiniteCalculatedMetric => {
                write!(f, "non-finite magnitude/ESR spectrum result")
            }
        }
    }
}

impl Error for MagnitudeEsrSweepError {}

impl From<CapacitorSpectrumError> for MagnitudeEsrSweepError {
    fn from(value: CapacitorSpectrumError) -> Self {
        Self::CurrentSpectrum(value)
    }
}

/// Calculate magnitude/ESR-based ripple voltage and loss per frequency.
///
/// For each separate RMS current component, voltage magnitude is I_rms * |Z(f)|
/// and real loss is I_rms² * ESR(f). RMS voltages combine by root-sum-square and
/// real loss sums across distinct frequencies. No phase data are required for
/// these two aggregate quantities under the stated steady-state linear assumptions.
pub fn calculate_magnitude_esr_spectrum_metrics(
    sweep: &CapacitorMagnitudeEsrSweep,
    spectrum: &CapacitorCurrentSpectrum,
    thermal_resistance_k_per_w: Option<f64>,
) -> Result<MagnitudeEsrSpectrumMetrics, MagnitudeEsrSweepError> {
    sweep.validate()?;
    spectrum.validate()?;
    if thermal_resistance_k_per_w.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(MagnitudeEsrSweepError::InvalidRequirements(
            "thermal resistance must be finite and positive when supplied",
        ));
    }

    let mut results = Vec::with_capacity(spectrum.components.len());
    let mut current_squared = 0.0;
    let mut voltage_squared = 0.0;
    let mut loss_power_w = 0.0;
    let mut max_impedance: f64 = 0.0;

    for component in &spectrum.components {
        let evaluated = sweep.evaluate(component.frequency_hz)?;
        let ripple_voltage = component.current_rms_a * evaluated.impedance_magnitude_ohm;
        let dissipated_power = component.current_rms_a.powi(2) * evaluated.esr_ohm;
        current_squared += component.current_rms_a.powi(2);
        voltage_squared += ripple_voltage.powi(2);
        loss_power_w += dissipated_power;
        max_impedance = max_impedance.max(evaluated.impedance_magnitude_ohm);
        results.push(MagnitudeEsrSpectrumComponentResult {
            frequency_hz: component.frequency_hz,
            current_rms_a: component.current_rms_a,
            impedance_magnitude_ohm: evaluated.impedance_magnitude_ohm,
            esr_ohm: evaluated.esr_ohm,
            ripple_voltage_rms_v: ripple_voltage,
            dissipated_power_w: dissipated_power,
            evaluation_kind: evaluated.evaluation_kind,
            lower_bracket_frequency_hz: evaluated.lower_bracket_frequency_hz,
            upper_bracket_frequency_hz: evaluated.upper_bracket_frequency_hz,
        });
    }

    let total_current_rms_a = current_squared.sqrt();
    let total_ripple_voltage_rms_v = voltage_squared.sqrt();
    let modeled_temperature_rise_k =
        thermal_resistance_k_per_w.map(|resistance| loss_power_w * resistance);
    let scalars = [
        total_current_rms_a,
        total_ripple_voltage_rms_v,
        loss_power_w,
        max_impedance,
    ];
    if scalars.into_iter().any(|value| !value.is_finite())
        || modeled_temperature_rise_k.is_some_and(|value| !value.is_finite())
        || results.iter().any(|result| {
            !result.ripple_voltage_rms_v.is_finite() || !result.dissipated_power_w.is_finite()
        })
    {
        return Err(MagnitudeEsrSweepError::NonFiniteCalculatedMetric);
    }

    Ok(MagnitudeEsrSpectrumMetrics {
        total_current_rms_a,
        total_ripple_voltage_rms_v,
        total_loss_power_w: loss_power_w,
        maximum_component_impedance_ohm: max_impedance,
        modeled_temperature_rise_k,
        components: results,
    })
}

/// Screen one current spectrum against the magnitude/ESR sweep and explicit constraints.
///
/// A screen is NotEvaluable when temperature/bias/evidence/frequency is inapplicable.
/// An absence of bias metadata does not become a zero-bias pass. This is nominal
/// screening; the separate hard-bound API must be used for deterministic interval claims.
pub fn screen_magnitude_esr_spectrum(
    sweep: &CapacitorMagnitudeEsrSweep,
    spectrum: &CapacitorCurrentSpectrum,
    requirements: &MagnitudeEsrScreeningRequirements,
) -> Result<MagnitudeEsrScreeningReport, MagnitudeEsrSweepError> {
    sweep.validate()?;
    spectrum.validate()?;
    requirements.validate()?;

    let mut limitations = Vec::new();
    if (requirements.operating_temperature_k - sweep.characterization_temperature_k).abs()
        > requirements.max_temperature_deviation_k
    {
        limitations.push(MagnitudeEsrScreeningLimitation::TemperatureOutsideCharacterization);
    }
    let bias_matches = match (sweep.dc_bias, requirements.operating_dc_bias) {
        (CapacitorDcBiasCondition::ExplicitlyUnbiased, CapacitorDcBiasCondition::ExplicitlyUnbiased) => true,
        (CapacitorDcBiasCondition::AppliedVolts(measured), CapacitorDcBiasCondition::AppliedVolts(operating)) => {
            (measured - operating).abs() <= requirements.max_absolute_dc_bias_deviation_v
        }
        _ => false,
    };
    if !bias_matches {
        limitations.push(MagnitudeEsrScreeningLimitation::DcBiasOutsideCharacterization);
    }
    if !requirements.accepted_evidence_kinds.contains(&sweep.evidence_kind) {
        limitations.push(MagnitudeEsrScreeningLimitation::EvidenceKindNotAllowed);
    }
    let min_frequency = sweep.min_frequency_hz().expect("validated sweep has points");
    let max_frequency = sweep.max_frequency_hz().expect("validated sweep has points");
    if spectrum.components.iter().any(|component| {
        component.frequency_hz < min_frequency || component.frequency_hz > max_frequency
    }) {
        limitations.push(MagnitudeEsrScreeningLimitation::FrequencyOutsideMeasuredSweep);
    }

    if !limitations.is_empty() {
        return Ok(MagnitudeEsrScreeningReport {
            sweep: sweep.clone(),
            current_spectrum: spectrum.clone(),
            requirements: requirements.clone(),
            disposition: MagnitudeEsrScreeningDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            metrics: None,
        });
    }

    let metrics = calculate_magnitude_esr_spectrum_metrics(
        sweep,
        spectrum,
        requirements.thermal_resistance_k_per_w,
    )?;
    let mut violations = Vec::new();
    if requirements.max_impedance_magnitude_ohm.is_some_and(|maximum| {
        metrics.maximum_component_impedance_ohm > maximum
    }) {
        violations.push(MagnitudeEsrConstraintViolation::ImpedanceMagnitudeAboveMaximum);
    }
    if metrics.total_loss_power_w > requirements.max_total_loss_power_w {
        violations.push(MagnitudeEsrConstraintViolation::TotalLossAboveMaximum);
    }
    if requirements.max_total_ripple_voltage_rms_v.is_some_and(|maximum| {
        metrics.total_ripple_voltage_rms_v > maximum
    }) {
        violations.push(MagnitudeEsrConstraintViolation::TotalRippleVoltageAboveMaximum);
    }
    if requirements.max_temperature_rise_k.is_some_and(|maximum| {
        metrics.modeled_temperature_rise_k.is_none_or(|rise| rise > maximum)
    }) {
        violations.push(MagnitudeEsrConstraintViolation::TemperatureRiseAboveMaximum);
    }

    let disposition = if violations.is_empty() {
        MagnitudeEsrScreeningDisposition::PassesMagnitudeEsrScreen
    } else {
        MagnitudeEsrScreeningDisposition::FailsDeclaredConstraints
    };
    Ok(MagnitudeEsrScreeningReport {
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
    use crate::capacitor_spectrum::CurrentSpectrumComponent;

    fn sweep() -> CapacitorMagnitudeEsrSweep {
        CapacitorMagnitudeEsrSweep {
            device_id: "synthetic-magnitude-esr-sweep".to_string(),
            specimen_id: Some("fixture-specimen-01".to_string()),
            group_id: Some("fixture-group".to_string()),
            characterization_temperature_k: 298.15,
            dc_bias: CapacitorDcBiasCondition::AppliedVolts(1.0),
            ac_test_voltage_rms_v: Some(0.1),
            source_reference: "test-fixture://not-real-magnitude-esr-data".to_string(),
            method_reference: "synthetic analytic fixture; not a measurement method".to_string(),
            license_reference: "synthetic fixture terms".to_string(),
            dataset_version: Some("fixture-v1".to_string()),
            evidence_kind: CapacitorDeviceEvidenceKind::SyntheticFixture,
            points: vec![
                MagnitudeEsrSweepPoint {
                    frequency_hz: 1.0e3,
                    impedance_magnitude_ohm: 10.0005,
                    esr_ohm: 0.1,
                },
                MagnitudeEsrSweepPoint {
                    frequency_hz: 1.0e4,
                    impedance_magnitude_ohm: 1.0198039,
                    esr_ohm: 0.2,
                },
                MagnitudeEsrSweepPoint {
                    frequency_hz: 1.0e5,
                    impedance_magnitude_ohm: 2.0615528,
                    esr_ohm: 0.5,
                },
            ],
        }
    }

    fn spectrum() -> CapacitorCurrentSpectrum {
        CapacitorCurrentSpectrum {
            source_reference: "test-fixture://not-real-spectrum".to_string(),
            method_reference: "synthetic harmonic RMS fixture".to_string(),
            components: vec![
                CurrentSpectrumComponent {
                    frequency_hz: 1.0e3,
                    current_rms_a: 0.5,
                },
                CurrentSpectrumComponent {
                    frequency_hz: 1.0e5,
                    current_rms_a: 0.2,
                },
            ],
        }
    }

    fn requirements() -> MagnitudeEsrScreeningRequirements {
        MagnitudeEsrScreeningRequirements {
            operating_temperature_k: 298.15,
            max_temperature_deviation_k: 2.0,
            operating_dc_bias: CapacitorDcBiasCondition::AppliedVolts(1.0),
            max_absolute_dc_bias_deviation_v: 0.1,
            accepted_evidence_kinds: vec![CapacitorDeviceEvidenceKind::SyntheticFixture],
            max_total_loss_power_w: 0.06,
            max_total_ripple_voltage_rms_v: Some(6.0),
            max_impedance_magnitude_ohm: Some(11.0),
            thermal_resistance_k_per_w: Some(10.0),
            max_temperature_rise_k: Some(0.6),
        }
    }

    #[test]
    fn exact_knot_preserves_magnitude_and_esr_without_phase() {
        let result = sweep().evaluate(1.0e5).unwrap();
        assert_eq!(result.evaluation_kind, ImpedanceEvaluationKind::MeasuredKnot);
        assert_eq!(result.impedance_magnitude_ohm, 2.0615528);
        assert_eq!(result.esr_ohm, 0.5);
        assert!(result.lower_bracket_frequency_hz == result.upper_bracket_frequency_hz);
    }

    #[test]
    fn interpolation_is_log_frequency_and_retains_bracketing_knots() {
        let result = sweep().evaluate((1.0e3_f64 * 1.0e4_f64).sqrt()).unwrap();
        assert_eq!(result.evaluation_kind, ImpedanceEvaluationKind::InterpolatedBetweenKnots);
        assert_eq!(result.lower_bracket_frequency_hz, 1.0e3);
        assert_eq!(result.upper_bracket_frequency_hz, 1.0e4);
        assert!((result.esr_ohm - 0.15).abs() <= 1.0e-12);
    }

    #[test]
    fn extrapolation_outside_sweep_is_rejected() {
        assert!(matches!(
            sweep().evaluate(1.0e2),
            Err(MagnitudeEsrSweepError::FrequencyOutsideMeasuredSweep { .. })
        ));
    }

    #[test]
    fn loss_uses_frequency_resolved_esr_not_a_constant_value() {
        let result = calculate_magnitude_esr_spectrum_metrics(&sweep(), &spectrum(), None).unwrap();
        let expected = 0.5_f64.powi(2) * 0.1 + 0.2_f64.powi(2) * 0.5;
        let constant_low_esr = (0.5_f64.powi(2) + 0.2_f64.powi(2)) * 0.1;
        assert!((result.total_loss_power_w - expected).abs() <= 1.0e-12);
        assert!((result.total_loss_power_w - constant_low_esr - 0.016).abs() <= 1.0e-12);
    }

    #[test]
    fn aggregate_ripple_voltage_combines_distinct_frequencies_in_quadrature() {
        let result = calculate_magnitude_esr_spectrum_metrics(&sweep(), &spectrum(), None).unwrap();
        let v0 = 0.5 * 10.0005;
        let v1 = 0.2 * 2.0615528;
        assert!((result.total_ripple_voltage_rms_v - v0.hypot(v1)).abs() <= 1.0e-12);
    }

    #[test]
    fn material_esr_inconsistent_with_magnitude_is_rejected() {
        let mut sample = sweep();
        sample.points[1].esr_ohm = sample.points[1].impedance_magnitude_ohm + 0.1;
        assert!(matches!(
            sample.validate(),
            Err(MagnitudeEsrSweepError::EsrAboveMagnitude { .. })
        ));
    }

    #[test]
    fn unknown_bias_is_not_equated_with_zero_bias() {
        let mut sample = sweep();
        sample.dc_bias = CapacitorDcBiasCondition::Unknown;
        assert!(matches!(
            screen_magnitude_esr_spectrum(&sample, &spectrum(), &requirements()).unwrap().disposition,
            MagnitudeEsrScreeningDisposition::NotEvaluable
        ));
    }

    #[test]
    fn temperature_and_bias_mismatch_are_not_evaluable() {
        let sample = sweep();
        let mut req = requirements();
        req.operating_temperature_k = 320.0;
        req.operating_dc_bias = CapacitorDcBiasCondition::AppliedVolts(3.3);
        let report = screen_magnitude_esr_spectrum(&sample, &spectrum(), &req).unwrap();
        assert_eq!(report.disposition, MagnitudeEsrScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(&MagnitudeEsrScreeningLimitation::TemperatureOutsideCharacterization));
        assert!(report.limitations.contains(&MagnitudeEsrScreeningLimitation::DcBiasOutsideCharacterization));
    }

    #[test]
    fn duplicate_frequency_is_rejected() {
        let mut sample = sweep();
        sample.points[1].frequency_hz = sample.points[0].frequency_hz;
        assert!(matches!(
            sample.validate(),
            Err(MagnitudeEsrSweepError::InvalidSweep(_))
        ));
    }

    #[test]
    fn magnitude_esr_requires_source_method_and_license_provenance() {
        let mut sample = sweep();
        sample.license_reference.clear();
        assert!(matches!(
            sample.validate(),
            Err(MagnitudeEsrSweepError::InvalidSweep(_))
        ));
    }

    #[test]
    fn loss_constraint_is_reported_independently_from_ripple_limit() {
        let sample = sweep();
        let mut req = requirements();
        req.max_total_loss_power_w = 0.01;
        req.max_total_ripple_voltage_rms_v = Some(100.0);
        req.max_impedance_magnitude_ohm = Some(100.0);
        req.max_temperature_rise_k = Some(100.0);
        let report = screen_magnitude_esr_spectrum(&sample, &spectrum(), &req).unwrap();
        assert_eq!(
            report.disposition,
            MagnitudeEsrScreeningDisposition::FailsDeclaredConstraints
        );
        assert!(report.violations.contains(&MagnitudeEsrConstraintViolation::TotalLossAboveMaximum));
    }

    #[test]
    fn thermal_limit_requires_explicit_thermal_resistance() {
        let mut req = requirements();
        req.thermal_resistance_k_per_w = None;
        assert!(matches!(
            req.validate(),
            Err(MagnitudeEsrSweepError::InvalidRequirements(_))
        ));
    }

    #[test]
    fn disallowed_evidence_origin_blocks_screen() {
        let sample = sweep();
        let mut req = requirements();
        req.accepted_evidence_kinds = vec![CapacitorDeviceEvidenceKind::MeasuredDevice];
        let report = screen_magnitude_esr_spectrum(&sample, &spectrum(), &req).unwrap();
        assert_eq!(report.disposition, MagnitudeEsrScreeningDisposition::NotEvaluable);
        assert!(report.limitations.contains(&MagnitudeEsrScreeningLimitation::EvidenceKindNotAllowed));
    }
}
