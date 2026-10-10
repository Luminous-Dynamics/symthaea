// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated first-order ultrasound design calculations for research use.
//!
//! These functions estimate physical quantities and check a necessary sampling
//! condition. They do not model a complete transducer, tissue propagation,
//! reconstruction pipeline, acoustic exposure, or clinical image quality.
//! A result from this module is not evidence that a device is safe or clinically
//! effective.

use std::fmt;

/// Invalid inputs to the first-order ultrasound calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UltrasoundModelError {
    /// A parameter was NaN or infinite.
    NonFinite(&'static str),
    /// A parameter that must be greater than zero was zero or negative.
    NonPositive(&'static str),
    /// A finite input overflowed while calculating a derived value.
    DerivedValueNonFinite(&'static str),
}

impl fmt::Display for UltrasoundModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::NonPositive(name) => write!(f, "{name} must be greater than zero"),
            Self::DerivedValueNonFinite(name) => {
                write!(f, "derived value {name} is not finite")
            }
        }
    }
}

impl std::error::Error for UltrasoundModelError {}

fn require_positive_finite(
    value: f64,
    name: &'static str,
) -> Result<(), UltrasoundModelError> {
    if !value.is_finite() {
        return Err(UltrasoundModelError::NonFinite(name));
    }
    if value <= 0.0 {
        return Err(UltrasoundModelError::NonPositive(name));
    }
    Ok(())
}

/// Estimate acoustic wavelength λ = c / f in metres.
///
/// The caller must supply a sound speed appropriate to the modeled medium;
/// this function intentionally does not assume a universal tissue sound speed.
pub fn estimate_wavelength_m(
    center_frequency_hz: f64,
    sound_speed_m_s: f64,
) -> Result<f64, UltrasoundModelError> {
    require_positive_finite(center_frequency_hz, "center_frequency_hz")?;
    require_positive_finite(sound_speed_m_s, "sound_speed_m_s")?;

    // Reuse the existing canonical acoustics calculation after validating inputs.
    let wavelength = super::wavelength(center_frequency_hz, sound_speed_m_s);
    if !wavelength.is_finite() || wavelength <= 0.0 {
        return Err(UltrasoundModelError::DerivedValueNonFinite(
            "wavelength_m",
        ));
    }
    Ok(wavelength)
}

/// Estimate idealized pulse-echo axial resolution as half the spatial pulse length.
///
/// For a pulse containing N cycles, spatial pulse length is approximately N * λ,
/// so the idealized separation is N * c / (2 * f) metres. Actual resolution
/// depends on pulse shape, bandwidth convention, transducer response, propagation,
/// reconstruction and the test method. This estimate is not the measured
/// resolution of a real probe.
///
/// Equation basis: [Ultrasound—biophysics mechanisms, §2.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC1995002/),
/// which defines spatial pulse length as Nλ and idealized axial resolution as
/// half the spatial pulse length.
pub fn estimate_axial_resolution_from_pulse_cycles_m(
    center_frequency_hz: f64,
    sound_speed_m_s: f64,
    pulse_cycles: f64,
) -> Result<f64, UltrasoundModelError> {
    require_positive_finite(center_frequency_hz, "center_frequency_hz")?;
    require_positive_finite(sound_speed_m_s, "sound_speed_m_s")?;
    require_positive_finite(pulse_cycles, "pulse_cycles")?;

    let wavelength = super::wavelength(center_frequency_hz, sound_speed_m_s);
    let resolution = pulse_cycles * wavelength / 2.0;
    if !wavelength.is_finite() || !resolution.is_finite() || resolution <= 0.0 {
        return Err(UltrasoundModelError::DerivedValueNonFinite(
            "axial_resolution_m",
        ));
    }
    Ok(resolution)
}

/// The theoretical Nyquist minimum for a real, baseband signal with upper
/// frequency f_max: f_s >= 2 f_max.
///
/// This bound is necessary, not sufficient, for a real acquisition chain.
/// Anti-alias filter response, transition band, ADC performance, clock accuracy
/// and design margin must be evaluated separately.
pub fn minimum_nyquist_sample_rate_hz(
    highest_signal_frequency_hz: f64,
) -> Result<f64, UltrasoundModelError> {
    require_positive_finite(highest_signal_frequency_hz, "highest_signal_frequency_hz")?;
    let minimum = 2.0 * highest_signal_frequency_hz;
    if !minimum.is_finite() {
        return Err(UltrasoundModelError::DerivedValueNonFinite(
            "minimum_nyquist_sample_rate_hz",
        ));
    }
    Ok(minimum)
}

/// A transparent result from a first-pass sampling-plan check.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingPlanAssessment {
    /// The mathematical Nyquist minimum in Hz for the supplied upper frequency.
    pub minimum_nyquist_sample_rate_hz: f64,
    /// The proposed analog-to-digital conversion rate in Hz.
    pub proposed_sample_rate_hz: f64,
    /// True iff the proposed rate is at least the mathematical Nyquist minimum.
    pub nyquist_condition_satisfied: bool,
    /// True iff a non-empty reference to separately reviewed anti-alias evidence was supplied.
    /// This module does not inspect or verify the referenced evidence.
    pub anti_alias_evidence_reference_supplied: bool,
}

impl SamplingPlanAssessment {
    /// Returns true only when both minimum checks pass.
    ///
    /// Passing is not a hardware-safety or medical-device validation claim.
    pub fn passes_minimum_checks(&self) -> bool {
        self.nyquist_condition_satisfied && self.anti_alias_evidence_reference_supplied
    }
}

/// Check a proposed sampling rate against a simple Nyquist condition and the
/// presence of a separately reviewed anti-alias evidence reference.
///
/// Evidence is represented by a reference identifier, not trusted as true by
/// this function. A production workflow must resolve and independently review
/// that evidence and establish adequate margin for the actual acquisition chain.
pub fn assess_sampling_plan(
    proposed_sample_rate_hz: f64,
    highest_signal_frequency_hz: f64,
    anti_alias_evidence_reference: Option<&str>,
) -> Result<SamplingPlanAssessment, UltrasoundModelError> {
    require_positive_finite(proposed_sample_rate_hz, "proposed_sample_rate_hz")?;
    let minimum_nyquist_sample_rate_hz =
        minimum_nyquist_sample_rate_hz(highest_signal_frequency_hz)?;
    let evidence_supplied = anti_alias_evidence_reference
        .map(str::trim)
        .is_some_and(|reference| !reference.is_empty());

    Ok(SamplingPlanAssessment {
        minimum_nyquist_sample_rate_hz,
        proposed_sample_rate_hz,
        nyquist_condition_satisfied: proposed_sample_rate_hz
            >= minimum_nyquist_sample_rate_hz,
        anti_alias_evidence_reference_supplied: evidence_supplied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_wavelength_for_five_mhz_in_1540_m_per_s_medium() {
        let wavelength = estimate_wavelength_m(5_000_000.0, 1_540.0).unwrap();
        assert!((wavelength - 0.000_308).abs() < 1e-12);
    }

    #[test]
    fn known_answer_idealized_pulse_echo_axial_resolution() {
        let resolution =
            estimate_axial_resolution_from_pulse_cycles_m(5_000_000.0, 1_540.0, 2.0).unwrap();
        assert!((resolution - 0.000_308).abs() < 1e-12);
    }

    #[test]
    fn rejects_zero_negative_nan_and_infinite_physical_inputs() {
        assert_eq!(
            estimate_wavelength_m(0.0, 1_540.0),
            Err(UltrasoundModelError::NonPositive("center_frequency_hz"))
        );
        assert_eq!(
            estimate_wavelength_m(5_000_000.0, f64::NAN),
            Err(UltrasoundModelError::NonFinite("sound_speed_m_s"))
        );
        assert_eq!(
            estimate_axial_resolution_from_pulse_cycles_m(5_000_000.0, 1_540.0, f64::INFINITY),
            Err(UltrasoundModelError::NonFinite("pulse_cycles"))
        );
        assert_eq!(
            minimum_nyquist_sample_rate_hz(-1.0),
            Err(UltrasoundModelError::NonPositive(
                "highest_signal_frequency_hz"
            ))
        );
    }

    #[test]
    fn sampling_assessment_fails_when_rate_is_below_nyquist() {
        let assessment = assess_sampling_plan(15_000_000.0, 8_000_000.0, Some("FILTER-REPORT-7"))
            .unwrap();
        assert_eq!(assessment.minimum_nyquist_sample_rate_hz, 16_000_000.0);
        assert!(!assessment.nyquist_condition_satisfied);
        assert!(!assessment.passes_minimum_checks());
    }

    #[test]
    fn sampling_assessment_requires_antialias_evidence_reference() {
        let with_reference =
            assess_sampling_plan(20_000_000.0, 8_000_000.0, Some("FILTER-REPORT-7")).unwrap();
        assert!(with_reference.passes_minimum_checks());

        let without_reference = assess_sampling_plan(20_000_000.0, 8_000_000.0, None).unwrap();
        assert!(without_reference.nyquist_condition_satisfied);
        assert!(!without_reference.anti_alias_evidence_reference_supplied);
        assert!(!without_reference.passes_minimum_checks());

        let blank_reference =
            assess_sampling_plan(20_000_000.0, 8_000_000.0, Some("  ")).unwrap();
        assert!(!blank_reference.passes_minimum_checks());
    }

    #[test]
    fn rejects_non_finite_or_non_positive_sample_rates() {
        assert_eq!(
            assess_sampling_plan(f64::NAN, 8_000_000.0, Some("FILTER-REPORT-7")),
            Err(UltrasoundModelError::NonFinite("proposed_sample_rate_hz"))
        );
        assert_eq!(
            assess_sampling_plan(20_000_000.0, 0.0, Some("FILTER-REPORT-7")),
            Err(UltrasoundModelError::NonPositive(
                "highest_signal_frequency_hz"
            ))
        );
    }
}
