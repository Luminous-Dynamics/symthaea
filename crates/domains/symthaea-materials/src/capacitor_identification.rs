// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability-gated, in-situ capacitor ESR and parasitic-inductance identification.
//!
//! This module consumes synchronous complex RMS voltage/current phasors measured at
//! multiple frequencies. It does not turn impedance magnitude into phase, and it
//! does not interpret an offline component sweep as deployed telemetry.
//!
//! The fitted model is the first-order series RLC relation
//!
//!     Z(ω) = R_s + j(ω L_s - 1 / (ω C))
//!
//! with capacitance supplied from an independently documented characterization.
//! ESR is estimated from the mean real impedance; series inductance is fitted to
//! the capacitance-corrected imaginary impedance. Frequency-span, current-floor,
//! ESR-dispersion and imaginary-fit gates prevent a numerical result from being
//! mistaken for an identifiable/adequate model. Estimates are nominal point values:
//! measurement uncertainties, phase-calibration uncertainty, thermal normalization,
//! aging attribution, safety limits and protection actions are outside this module.

use crate::capacitor_magnitude_esr::CapacitorDcBiasCondition;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// One synchronous complex RMS voltage/current observation at a positive frequency.
///
/// Voltage and current phasors must use the same frequency, RMS scaling and phase
/// reference. Real and imaginary parts may be signed; all four values must be finite.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacitorPhasorObservation {
    /// Frequency in hertz.
    pub frequency_hz: f64,
    /// Voltage-phasor real part in volts RMS.
    pub voltage_real_rms_v: f64,
    /// Voltage-phasor imaginary part in volts RMS.
    pub voltage_imaginary_rms_v: f64,
    /// Current-phasor real part in amperes RMS.
    pub current_real_rms_a: f64,
    /// Current-phasor imaginary part in amperes RMS.
    pub current_imaginary_rms_a: f64,
}

/// One source- and condition-bound phasor campaign for one capacitor specimen.
///
/// The capacitance value must have independent provenance. The source, extraction
/// method, channel calibration/phase alignment and data-rights references are required
/// so an apparent change in ESR is not detached from its measurement conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorPhasorCampaign {
    /// Stable device/component identifier.
    pub device_id: String,
    /// Independently characterized series capacitance, in farads.
    pub series_capacitance_f: f64,
    /// Characterization temperature, in kelvin.
    pub characterization_temperature_k: f64,
    /// Explicit DC-bias state; unknown is not equivalent to zero volts.
    pub dc_bias: CapacitorDcBiasCondition,
    /// Source reference for synchronized waveform/phasor data.
    pub source_reference: String,
    /// Extraction-method reference, including FFT/lock-in, windowing and RMS scaling.
    pub method_reference: String,
    /// Independent capacitance-characterization source and method.
    pub capacitance_reference: String,
    /// Voltage/current channel calibration and phase-alignment reference.
    pub calibration_reference: String,
    /// License, consent, ownership or other governing data-rights reference.
    pub rights_reference: String,
    /// Observations ordered by strictly increasing frequency.
    pub observations: Vec<CapacitorPhasorObservation>,
}

impl CapacitorPhasorCampaign {
    /// Validate provenance, physical conditions and phasor-domain invariants.
    pub fn validate(&self) -> Result<(), CapacitorIdentificationError> {
        if self.device_id.trim().is_empty() {
            return Err(CapacitorIdentificationError::InvalidCampaign(
                "device ID is missing",
            ));
        }
        if !self.series_capacitance_f.is_finite() || self.series_capacitance_f <= 0.0 {
            return Err(CapacitorIdentificationError::InvalidCampaign(
                "independently characterized capacitance must be finite and positive",
            ));
        }
        if !self.characterization_temperature_k.is_finite()
            || self.characterization_temperature_k <= 0.0
        {
            return Err(CapacitorIdentificationError::InvalidCampaign(
                "characterization temperature must be finite and positive",
            ));
        }
        if let CapacitorDcBiasCondition::AppliedVolts(value) = self.dc_bias {
            if !value.is_finite() {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    "applied DC bias must be finite",
                ));
            }
        }
        for (field, reference) in [
            ("source", self.source_reference.as_str()),
            ("method", self.method_reference.as_str()),
            ("capacitance", self.capacitance_reference.as_str()),
            ("calibration/phase-alignment", self.calibration_reference.as_str()),
            ("data-rights", self.rights_reference.as_str()),
        ] {
            if reference.trim().is_empty() {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    match field {
                        "source" => "source reference is missing",
                        "method" => "phasor extraction method reference is missing",
                        "capacitance" => "capacitance reference is missing",
                        "calibration/phase-alignment" => {
                            "channel calibration and phase-alignment reference is missing"
                        }
                        _ => "data-rights reference is missing",
                    },
                ));
            }
        }
        if self.observations.len() < 2 {
            return Err(CapacitorIdentificationError::InvalidCampaign(
                "at least two frequency observations are required",
            ));
        }

        let mut previous_frequency_hz = 0.0;
        for observation in &self.observations {
            if !observation.frequency_hz.is_finite() || observation.frequency_hz <= 0.0 {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    "observation frequencies must be finite and positive",
                ));
            }
            if observation.frequency_hz <= previous_frequency_hz {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    "observation frequencies must be strictly increasing and unique",
                ));
            }
            for value in [
                observation.voltage_real_rms_v,
                observation.voltage_imaginary_rms_v,
                observation.current_real_rms_a,
                observation.current_imaginary_rms_a,
            ] {
                if !value.is_finite() {
                    return Err(CapacitorIdentificationError::InvalidCampaign(
                        "phasor components must all be finite",
                    ));
                }
            }
            let current_magnitude = observation
                .current_real_rms_a
                .hypot(observation.current_imaginary_rms_a);
            let voltage_magnitude = observation
                .voltage_real_rms_v
                .hypot(observation.voltage_imaginary_rms_v);
            if !current_magnitude.is_finite() || !voltage_magnitude.is_finite() {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    "phasor magnitudes must remain finite",
                ));
            }
            if current_magnitude == 0.0 {
                return Err(CapacitorIdentificationError::InvalidCampaign(
                    "zero-current phasor cannot identify impedance",
                ));
            }
            previous_frequency_hz = observation.frequency_hz;
        }
        Ok(())
    }
}

/// Observable/model-fit gate requirements.
///
/// These are explicit policy inputs, not universal capacitor limits. They should be
/// calibrated against instrument uncertainty and device families before deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorIdentificationRequirements {
    /// Minimum number of distinct frequency observations; at least two.
    pub minimum_observations: usize,
    /// Minimum highest/lowest frequency ratio; must be greater than one.
    pub minimum_frequency_span_ratio: f64,
    /// Minimum RMS current magnitude at every frequency, in amperes.
    pub minimum_current_rms_a: f64,
    /// Maximum relative RMS residual in the capacitance-corrected imaginary model.
    pub maximum_relative_imaginary_fit_rmse: f64,
    /// Maximum relative RMS dispersion of real impedance around the ESR estimate.
    pub maximum_relative_esr_rmse: f64,
}

impl CapacitorIdentificationRequirements {
    /// Validate configurable observability and first-order model-fit gates.
    pub fn validate(&self) -> Result<(), CapacitorIdentificationError> {
        if self.minimum_observations < 2 {
            return Err(CapacitorIdentificationError::InvalidRequirements(
                "minimum observations must be at least two",
            ));
        }
        if !self.minimum_frequency_span_ratio.is_finite()
            || self.minimum_frequency_span_ratio <= 1.0
        {
            return Err(CapacitorIdentificationError::InvalidRequirements(
                "minimum frequency-span ratio must be finite and greater than one",
            ));
        }
        if !self.minimum_current_rms_a.is_finite() || self.minimum_current_rms_a <= 0.0 {
            return Err(CapacitorIdentificationError::InvalidRequirements(
                "minimum current magnitude must be finite and positive",
            ));
        }
        if !self.maximum_relative_imaginary_fit_rmse.is_finite()
            || self.maximum_relative_imaginary_fit_rmse < 0.0
        {
            return Err(CapacitorIdentificationError::InvalidRequirements(
                "maximum imaginary-fit residual must be finite and non-negative",
            ));
        }
        if !self.maximum_relative_esr_rmse.is_finite()
            || self.maximum_relative_esr_rmse < 0.0
        {
            return Err(CapacitorIdentificationError::InvalidRequirements(
                "maximum ESR dispersion must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Result status: a numerical fit is not automatically a trustworthy identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorIdentificationDisposition {
    /// Excitation, current level and first-order model residuals meet caller requirements.
    PassesObservabilityGate,
    /// Available frequencies/current magnitude do not meet the observability requirements.
    NotIdentifiable,
    /// The data are measurable, but the series-RLC residual checks do not meet requirements.
    SeriesRlcModelMismatch,
    /// Nominal fit is available, but required operating-condition context is unknown.
    NotEvaluable,
}

/// Reason a fit is blocked or qualified with an explicit limitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitorIdentificationLimitation {
    /// Too few usable frequency observations for the requested gate.
    InsufficientObservations,
    /// Frequencies do not span the required ratio.
    FrequencySpanTooNarrow,
    /// One or more currents fall below the configured magnitude floor.
    CurrentBelowMinimum,
    /// DC-bias state is unknown, so condition-matched health interpretation is blocked.
    UnknownDcBias,
    /// Real impedance varies too much for the configured constant-ESR model.
    EsrDispersionAboveMaximum,
    /// Imaginary impedance does not fit the capacitance-plus-series-inductance model.
    ImaginaryFitResidualAboveMaximum,
    /// The unconstrained best-fit inductance is negative and cannot be a physical ESL estimate.
    NegativeInductanceEstimate,
}

/// Fit and gate report for one phasor campaign.
///
/// Estimate fields are absent when the basic observability gate fails. When estimates
/// are present, they are nominal point estimates only; no confidence interval or
/// measurement-uncertainty propagation is implied by a passing status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitorIdentificationReport {
    /// Device/component identifier from the input campaign.
    pub device_id: String,
    /// Characterization temperature in kelvin.
    pub characterization_temperature_k: f64,
    /// DC-bias state retained from the input; unknown is not treated as zero.
    pub dc_bias: CapacitorDcBiasCondition,
    /// Assumed independent capacitance, in farads.
    pub series_capacitance_f: f64,
    /// Number of phasor observations in the campaign.
    pub observation_count: usize,
    /// Highest/lowest observed frequency ratio.
    pub frequency_span_ratio: f64,
    /// Lowest current-phasor magnitude across all observations, in amperes RMS.
    pub minimum_observed_current_rms_a: f64,
    /// Nominal fitted ESR, in ohms, absent when the basic observability gate fails.
    pub estimated_esr_ohm: Option<f64>,
    /// Nominal fitted series inductance, in henries, absent when the basic gate fails.
    pub estimated_series_inductance_h: Option<f64>,
    /// RMS residual of real impedance around the ESR estimate, in ohms.
    pub esr_rmse_ohm: Option<f64>,
    /// Relative ESR residual, normalized by max(|ESR|, 1e-9 ohm).
    pub relative_esr_rmse: Option<f64>,
    /// RMS residual of the fitted imaginary impedance, in ohms.
    pub imaginary_fit_rmse_ohm: Option<f64>,
    /// Relative imaginary residual normalized by observed/reactive impedance scale.
    pub relative_imaginary_fit_rmse: Option<f64>,
    /// Overall fit/observability result.
    pub disposition: CapacitorIdentificationDisposition,
    /// Explicit reasons for refusal or model mismatch.
    pub limitations: Vec<CapacitorIdentificationLimitation>,
    /// Source reference retained from the measurement campaign.
    pub source_reference: String,
    /// Measurement/phasor extraction method reference retained from the campaign.
    pub method_reference: String,
    /// Independent capacitance characterization reference retained from the campaign.
    pub capacitance_reference: String,
    /// Calibration and phase-alignment reference retained from the campaign.
    pub calibration_reference: String,
    /// Data-rights reference retained from the campaign.
    pub rights_reference: String,
}

/// Input, metadata or numerical failure while preparing an identification report.
#[derive(Debug, Clone, PartialEq)]
pub enum CapacitorIdentificationError {
    /// Campaign source or physical input is invalid.
    InvalidCampaign(&'static str),
    /// Gate configuration is invalid.
    InvalidRequirements(&'static str),
    /// A measured impedance violates the passive-capacitor real-impedance contract.
    NonPassiveObservation { frequency_hz: f64 },
    /// A calculated value is non-finite or cannot be represented safely.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for CapacitorIdentificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCampaign(reason) => write!(f, "invalid phasor campaign: {reason}"),
            Self::InvalidRequirements(reason) => {
                write!(f, "invalid identification requirements: {reason}")
            }
            Self::NonPassiveObservation { frequency_hz } => write!(
                f,
                "negative real impedance at {frequency_hz} Hz is inconsistent with the passive-capacitor model"
            ),
            Self::NonFiniteCalculatedMetric => {
                write!(f, "identification produced a non-finite calculated metric")
            }
        }
    }
}

impl Error for CapacitorIdentificationError {}

/// Assess whether synchronous multi-frequency phasors support an ESR/series-inductance fit.
///
/// The method assumes the independently characterized capacitance is applicable at the
/// supplied temperature and bias, and that voltage/current channels share a calibrated
/// phase reference. No phase is synthesized from magnitude-only data. If excitation or
/// current-level requirements fail, estimates are withheld rather than reported as a
/// usable result. Unknown DC bias allows diagnostic point estimates but returns
/// NotEvaluable to prevent condition-ambiguous health interpretation.
pub fn assess_capacitor_identification(
    campaign: &CapacitorPhasorCampaign,
    requirements: &CapacitorIdentificationRequirements,
) -> Result<CapacitorIdentificationReport, CapacitorIdentificationError> {
    campaign.validate()?;
    requirements.validate()?;

    let first = campaign
        .observations
        .first()
        .expect("campaign validation guarantees at least two observations");
    let last = campaign
        .observations
        .last()
        .expect("campaign validation guarantees at least two observations");
    let frequency_span_ratio = last.frequency_hz / first.frequency_hz;
    if !frequency_span_ratio.is_finite() {
        return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
    }
    let minimum_observed_current_rms_a = campaign
        .observations
        .iter()
        .map(|observation| {
            observation
                .current_real_rms_a
                .hypot(observation.current_imaginary_rms_a)
        })
        .fold(f64::INFINITY, f64::min);

    let mut limitations = Vec::new();
    if campaign.observations.len() < requirements.minimum_observations {
        limitations.push(CapacitorIdentificationLimitation::InsufficientObservations);
    }
    if frequency_span_ratio < requirements.minimum_frequency_span_ratio {
        limitations.push(CapacitorIdentificationLimitation::FrequencySpanTooNarrow);
    }
    if campaign.observations.iter().any(|observation| {
        observation
            .current_real_rms_a
            .hypot(observation.current_imaginary_rms_a)
            < requirements.minimum_current_rms_a
    }) {
        limitations.push(CapacitorIdentificationLimitation::CurrentBelowMinimum);
    }
    if matches!(campaign.dc_bias, CapacitorDcBiasCondition::Unknown) {
        limitations.push(CapacitorIdentificationLimitation::UnknownDcBias);
    }

    let cannot_fit = limitations.iter().any(|limitation| {
        matches!(
            limitation,
            CapacitorIdentificationLimitation::InsufficientObservations
                | CapacitorIdentificationLimitation::FrequencySpanTooNarrow
                | CapacitorIdentificationLimitation::CurrentBelowMinimum
        )
    });
    if cannot_fit {
        return Ok(report_without_estimates(
            campaign,
            frequency_span_ratio,
            minimum_observed_current_rms_a,
            limitations,
            CapacitorIdentificationDisposition::NotIdentifiable,
        ));
    }

    // Divide V by I without forming |I|^2, which can overflow for finite phasors.
    let mut impedance = Vec::with_capacity(campaign.observations.len());
    for observation in &campaign.observations {
        let current_magnitude = observation
            .current_real_rms_a
            .hypot(observation.current_imaginary_rms_a);
        let unit_current_real = observation.current_real_rms_a / current_magnitude;
        let unit_current_imaginary = observation.current_imaginary_rms_a / current_magnitude;
        let real_ohm = (observation.voltage_real_rms_v * unit_current_real
            + observation.voltage_imaginary_rms_v * unit_current_imaginary)
            / current_magnitude;
        let imaginary_ohm = (observation.voltage_imaginary_rms_v * unit_current_real
            - observation.voltage_real_rms_v * unit_current_imaginary)
            / current_magnitude;
        if !real_ohm.is_finite() || !imaginary_ohm.is_finite() {
            return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
        }
        if real_ohm < 0.0 {
            return Err(CapacitorIdentificationError::NonPassiveObservation {
                frequency_hz: observation.frequency_hz,
            });
        }
        impedance.push((real_ohm, imaginary_ohm));
    }

    let count = impedance.len() as f64;
    let estimated_esr_ohm = impedance.iter().map(|point| point.0).sum::<f64>() / count;
    if !estimated_esr_ohm.is_finite() {
        return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
    }

    // Fit L in y = L*x, where y = (Im(Z) + 1/(ωC))/ω_scale and
    // x = ω/ω_scale. Geometric-mean scaling avoids squaring raw rad/s magnitudes.
    let geometric_mean_frequency_hz =
        (first.frequency_hz.ln() * 0.5 + last.frequency_hz.ln() * 0.5).exp();
    let omega_scale = TAU * geometric_mean_frequency_hz;
    if !omega_scale.is_finite() || omega_scale <= 0.0 {
        return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut imaginary_component_scales = Vec::with_capacity(impedance.len());
    for (observation, (_, imaginary_ohm)) in campaign.observations.iter().zip(&impedance) {
        let omega = TAU * observation.frequency_hz;
        let capacitive_reactance_magnitude =
            1.0 / (omega * campaign.series_capacitance_f);
        let scaled_omega = observation.frequency_hz / geometric_mean_frequency_hz;
        let corrected_imaginary_ohm = imaginary_ohm + capacitive_reactance_magnitude;
        let scaled_corrected_imaginary = corrected_imaginary_ohm / omega_scale;
        if !omega.is_finite()
            || !capacitive_reactance_magnitude.is_finite()
            || !scaled_omega.is_finite()
            || !scaled_corrected_imaginary.is_finite()
        {
            return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
        }
        numerator += scaled_omega * scaled_corrected_imaginary;
        denominator += scaled_omega * scaled_omega;
        imaginary_component_scales.push((omega, capacitive_reactance_magnitude));
    }
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
    }
    let fitted_series_inductance_h = numerator / denominator;
    if !fitted_series_inductance_h.is_finite() {
        return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
    }
    let negative_inductance_estimate = fitted_series_inductance_h < 0.0;
    // A negative unconstrained estimate is evidence against this passive series-RLC
    // model. Use the physical boundary for residual reporting, but do not publish a
    // negative ESL as a component estimate.
    let physical_model_inductance_h = fitted_series_inductance_h.max(0.0);
    if negative_inductance_estimate {
        limitations.push(CapacitorIdentificationLimitation::NegativeInductanceEstimate);
    }

    let esr_squared_residual = impedance
        .iter()
        .map(|point| (point.0 - estimated_esr_ohm).powi(2))
        .sum::<f64>()
        / count;
    let esr_rmse_ohm = esr_squared_residual.sqrt();
    let relative_esr_rmse = esr_rmse_ohm / estimated_esr_ohm.abs().max(1.0e-9);

    let mut imaginary_squared_residual = 0.0;
    let mut imaginary_scale_squared = 0.0;
    for ((_, measured_imaginary_ohm), (omega, xc)) in
        impedance.iter().zip(&imaginary_component_scales)
    {
        let inductive_reactance = omega * physical_model_inductance_h;
        let predicted_imaginary_ohm = inductive_reactance - xc;
        let residual = measured_imaginary_ohm - predicted_imaginary_ohm;
        let scale = inductive_reactance
            .abs()
            .max(xc.abs())
            .max(measured_imaginary_ohm.abs())
            .max(1.0e-12);
        if !predicted_imaginary_ohm.is_finite() || !residual.is_finite() || !scale.is_finite() {
            return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
        }
        imaginary_squared_residual += residual * residual;
        imaginary_scale_squared += scale * scale;
    }
    let imaginary_fit_rmse_ohm = (imaginary_squared_residual / count).sqrt();
    let relative_imaginary_fit_rmse =
        imaginary_fit_rmse_ohm / (imaginary_scale_squared / count).sqrt().max(1.0e-12);
    for value in [
        esr_rmse_ohm,
        relative_esr_rmse,
        imaginary_fit_rmse_ohm,
        relative_imaginary_fit_rmse,
    ] {
        if !value.is_finite() {
            return Err(CapacitorIdentificationError::NonFiniteCalculatedMetric);
        }
    }

    if relative_esr_rmse > requirements.maximum_relative_esr_rmse {
        limitations.push(CapacitorIdentificationLimitation::EsrDispersionAboveMaximum);
    }
    if relative_imaginary_fit_rmse > requirements.maximum_relative_imaginary_fit_rmse {
        limitations.push(CapacitorIdentificationLimitation::ImaginaryFitResidualAboveMaximum);
    }

    let disposition = if matches!(campaign.dc_bias, CapacitorDcBiasCondition::Unknown) {
        CapacitorIdentificationDisposition::NotEvaluable
    } else if limitations.iter().any(|limitation| {
        matches!(
            limitation,
            CapacitorIdentificationLimitation::EsrDispersionAboveMaximum
                | CapacitorIdentificationLimitation::ImaginaryFitResidualAboveMaximum
                | CapacitorIdentificationLimitation::NegativeInductanceEstimate
        )
    }) {
        CapacitorIdentificationDisposition::SeriesRlcModelMismatch
    } else {
        CapacitorIdentificationDisposition::PassesObservabilityGate
    };

    Ok(CapacitorIdentificationReport {
        device_id: campaign.device_id.clone(),
        characterization_temperature_k: campaign.characterization_temperature_k,
        dc_bias: campaign.dc_bias,
        series_capacitance_f: campaign.series_capacitance_f,
        observation_count: campaign.observations.len(),
        frequency_span_ratio,
        minimum_observed_current_rms_a,
        estimated_esr_ohm: Some(estimated_esr_ohm),
        estimated_series_inductance_h: (!negative_inductance_estimate)
            .then_some(fitted_series_inductance_h),
        esr_rmse_ohm: Some(esr_rmse_ohm),
        relative_esr_rmse: Some(relative_esr_rmse),
        imaginary_fit_rmse_ohm: Some(imaginary_fit_rmse_ohm),
        relative_imaginary_fit_rmse: Some(relative_imaginary_fit_rmse),
        disposition,
        limitations,
        source_reference: campaign.source_reference.clone(),
        method_reference: campaign.method_reference.clone(),
        capacitance_reference: campaign.capacitance_reference.clone(),
        calibration_reference: campaign.calibration_reference.clone(),
        rights_reference: campaign.rights_reference.clone(),
    })
}

fn report_without_estimates(
    campaign: &CapacitorPhasorCampaign,
    frequency_span_ratio: f64,
    minimum_observed_current_rms_a: f64,
    limitations: Vec<CapacitorIdentificationLimitation>,
    disposition: CapacitorIdentificationDisposition,
) -> CapacitorIdentificationReport {
    CapacitorIdentificationReport {
        device_id: campaign.device_id.clone(),
        characterization_temperature_k: campaign.characterization_temperature_k,
        dc_bias: campaign.dc_bias,
        series_capacitance_f: campaign.series_capacitance_f,
        observation_count: campaign.observations.len(),
        frequency_span_ratio,
        minimum_observed_current_rms_a,
        estimated_esr_ohm: None,
        estimated_series_inductance_h: None,
        esr_rmse_ohm: None,
        relative_esr_rmse: None,
        imaginary_fit_rmse_ohm: None,
        relative_imaginary_fit_rmse: None,
        disposition,
        limitations,
        source_reference: campaign.source_reference.clone(),
        method_reference: campaign.method_reference.clone(),
        capacitance_reference: campaign.capacitance_reference.clone(),
        calibration_reference: campaign.calibration_reference.clone(),
        rights_reference: campaign.rights_reference.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ESR_OHM: f64 = 0.035;
    const CAPACITANCE_F: f64 = 100.0e-6;
    const INDUCTANCE_H: f64 = 0.5e-6;

    fn observation_for_series_rlc(frequency_hz: f64, current_phase_rad: f64) -> CapacitorPhasorObservation {
        let omega = TAU * frequency_hz;
        let reactance = omega * INDUCTANCE_H - 1.0 / (omega * CAPACITANCE_F);
        let current_magnitude = 2.0;
        let current_real = current_magnitude * current_phase_rad.cos();
        let current_imaginary = current_magnitude * current_phase_rad.sin();
        let voltage_real = ESR_OHM * current_real - reactance * current_imaginary;
        let voltage_imaginary = ESR_OHM * current_imaginary + reactance * current_real;
        CapacitorPhasorObservation {
            frequency_hz,
            voltage_real_rms_v: voltage_real,
            voltage_imaginary_rms_v: voltage_imaginary,
            current_real_rms_a: current_real,
            current_imaginary_rms_a: current_imaginary,
        }
    }

    fn campaign() -> CapacitorPhasorCampaign {
        CapacitorPhasorCampaign {
            device_id: "synthetic-fixture-device-1".to_string(),
            series_capacitance_f: CAPACITANCE_F,
            characterization_temperature_k: 298.15,
            dc_bias: CapacitorDcBiasCondition::AppliedVolts(12.0),
            source_reference: "synthetic://capacitor-identification/fixture-v1".to_string(),
            method_reference: "synthetic coherent RMS phasors; shared phase reference".to_string(),
            capacitance_reference: "synthetic independently supplied C value".to_string(),
            calibration_reference: "synthetic exact channel gain and phase calibration".to_string(),
            rights_reference: "synthetic fixture; no external source data".to_string(),
            observations: vec![
                observation_for_series_rlc(2.0e3, 0.31),
                observation_for_series_rlc(2.0e4, -0.47),
                observation_for_series_rlc(2.0e5, 1.1),
            ],
        }
    }

    fn requirements() -> CapacitorIdentificationRequirements {
        CapacitorIdentificationRequirements {
            minimum_observations: 2,
            minimum_frequency_span_ratio: 5.0,
            minimum_current_rms_a: 0.1,
            maximum_relative_imaginary_fit_rmse: 1.0e-8,
            maximum_relative_esr_rmse: 1.0e-8,
        }
    }

    #[test]
    fn exact_complex_phasors_recover_esr_and_inductance() {
        let report = assess_capacitor_identification(&campaign(), &requirements()).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::PassesObservabilityGate);
        assert!((report.estimated_esr_ohm.unwrap() - ESR_OHM).abs() <= 1.0e-12);
        assert!((report.estimated_series_inductance_h.unwrap() - INDUCTANCE_H).abs() <= 1.0e-15);
        assert!(report.relative_imaginary_fit_rmse.unwrap() <= 1.0e-8);
    }

    #[test]
    fn unknown_bias_retains_diagnostic_fit_but_blocks_evaluation() {
        let mut sample = campaign();
        sample.dc_bias = CapacitorDcBiasCondition::Unknown;
        let report = assess_capacitor_identification(&sample, &requirements()).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::NotEvaluable);
        assert!(report.estimated_esr_ohm.is_some());
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::UnknownDcBias));
    }

    #[test]
    fn narrow_frequency_span_withholds_estimates() {
        let mut sample = campaign();
        sample.observations = vec![
            observation_for_series_rlc(10.0e3, 0.1),
            observation_for_series_rlc(12.0e3, 0.2),
        ];
        let mut req = requirements();
        req.minimum_frequency_span_ratio = 5.0;
        let report = assess_capacitor_identification(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::NotIdentifiable);
        assert_eq!(report.estimated_esr_ohm, None);
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::FrequencySpanTooNarrow));
    }

    #[test]
    fn low_current_at_one_tone_withholds_entire_fit() {
        let mut sample = campaign();
        let observation = &mut sample.observations[1];
        let ratio = 1.0e-8 / observation.current_real_rms_a.hypot(observation.current_imaginary_rms_a);
        observation.current_real_rms_a *= ratio;
        observation.current_imaginary_rms_a *= ratio;
        observation.voltage_real_rms_v *= ratio;
        observation.voltage_imaginary_rms_v *= ratio;
        let mut req = requirements();
        req.minimum_current_rms_a = 1.0e-3;
        let report = assess_capacitor_identification(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::NotIdentifiable);
        assert_eq!(report.estimated_series_inductance_h, None);
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::CurrentBelowMinimum));
    }

    #[test]
    fn non_rlc_imaginary_response_is_rejected_by_residual_gate() {
        let mut sample = campaign();
        sample.observations[1].voltage_imaginary_rms_v += 0.4;
        let mut req = requirements();
        req.maximum_relative_imaginary_fit_rmse = 1.0e-4;
        let report = assess_capacitor_identification(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::SeriesRlcModelMismatch);
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::ImaginaryFitResidualAboveMaximum));
    }

    #[test]
    fn frequency_dependent_esr_is_not_silently_collapsed_to_a_good_constant_fit() {
        let mut sample = campaign();
        sample.observations[1].voltage_real_rms_v += 0.25;
        let mut req = requirements();
        req.maximum_relative_esr_rmse = 1.0e-4;
        let report = assess_capacitor_identification(&sample, &req).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::SeriesRlcModelMismatch);
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::EsrDispersionAboveMaximum));
    }

    #[test]
    fn negative_unconstrained_inductance_is_a_model_mismatch_not_a_published_esl() {
        let mut sample = campaign();
        for observation in &mut sample.observations {
            let omega = TAU * observation.frequency_hz;
            let imaginary_ohm = -1.0 / (omega * CAPACITANCE_F) - omega * INDUCTANCE_H;
            let real_current = observation.current_real_rms_a;
            let imaginary_current = observation.current_imaginary_rms_a;
            observation.voltage_real_rms_v =
                ESR_OHM * real_current - imaginary_ohm * imaginary_current;
            observation.voltage_imaginary_rms_v =
                ESR_OHM * imaginary_current + imaginary_ohm * real_current;
        }
        let report = assess_capacitor_identification(&sample, &requirements()).unwrap();
        assert_eq!(report.disposition, CapacitorIdentificationDisposition::SeriesRlcModelMismatch);
        assert_eq!(report.estimated_series_inductance_h, None);
        assert!(report.limitations.contains(&CapacitorIdentificationLimitation::NegativeInductanceEstimate));
    }

    #[test]
    fn a_shared_phase_rotation_does_not_change_impedance_estimate() {
        let mut sample = campaign();
        for observation in &mut sample.observations {
            let angle = 0.73_f64;
            let (sin, cos) = angle.sin_cos();
            let vr = observation.voltage_real_rms_v;
            let vi = observation.voltage_imaginary_rms_v;
            let ir = observation.current_real_rms_a;
            let ii = observation.current_imaginary_rms_a;
            observation.voltage_real_rms_v = vr * cos - vi * sin;
            observation.voltage_imaginary_rms_v = vr * sin + vi * cos;
            observation.current_real_rms_a = ir * cos - ii * sin;
            observation.current_imaginary_rms_a = ir * sin + ii * cos;
        }
        let report = assess_capacitor_identification(&sample, &requirements()).unwrap();
        assert!((report.estimated_esr_ohm.unwrap() - ESR_OHM).abs() <= 1.0e-12);
        assert!((report.estimated_series_inductance_h.unwrap() - INDUCTANCE_H).abs() <= 1.0e-15);
    }

    #[test]
    fn zero_current_phasor_is_invalid() {
        let mut sample = campaign();
        sample.observations[0].current_real_rms_a = 0.0;
        sample.observations[0].current_imaginary_rms_a = 0.0;
        assert!(matches!(
            sample.validate(),
            Err(CapacitorIdentificationError::InvalidCampaign(_))
        ));
    }

    #[test]
    fn duplicate_frequency_is_invalid() {
        let mut sample = campaign();
        sample.observations[1].frequency_hz = sample.observations[0].frequency_hz;
        assert!(matches!(
            sample.validate(),
            Err(CapacitorIdentificationError::InvalidCampaign(_))
        ));
    }

    #[test]
    fn phase_calibration_and_independent_capacitance_provenance_are_required() {
        let mut sample = campaign();
        sample.calibration_reference.clear();
        assert!(matches!(
            sample.validate(),
            Err(CapacitorIdentificationError::InvalidCampaign(_))
        ));
        let mut sample = campaign();
        sample.capacitance_reference.clear();
        assert!(matches!(
            sample.validate(),
            Err(CapacitorIdentificationError::InvalidCampaign(_))
        ));
    }

    #[test]
    fn negative_real_impedance_is_rejected_for_passive_capacitor_model() {
        let mut sample = campaign();
        for observation in &mut sample.observations {
            let frequency_hz = observation.frequency_hz;
            let impedance_imaginary = {
                let omega = TAU * frequency_hz;
                omega * INDUCTANCE_H - 1.0 / (omega * CAPACITANCE_F)
            };
            observation.voltage_real_rms_v = -0.1 * observation.current_real_rms_a
                - impedance_imaginary * observation.current_imaginary_rms_a;
            observation.voltage_imaginary_rms_v = -0.1 * observation.current_imaginary_rms_a
                + impedance_imaginary * observation.current_real_rms_a;
        }
        assert!(matches!(
            assess_capacitor_identification(&sample, &requirements()),
            Err(CapacitorIdentificationError::NonPassiveObservation { .. })
        ));
    }
}
