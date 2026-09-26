// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-semiconductor
//!
//! Dependency-light analytical reference models for semiconductor-device engineering.
//! V1 intentionally covers only a bounded idealized diode profile so later TCAD,
//! compact-model, and physical-measurement layers have an independent oracle.
//!
//! Pure `std`, zero dependencies, no external solver execution, no physical I/O,
//! and no fabrication or actuation authority.
//!
//! ```text
//! analytical equation evaluated correctly
//! != physical diode modeled completely
//! != TCAD validated
//! != compact model qualified
//! != measured article characterized
//! ```

/// Exact SI Boltzmann constant, joules per kelvin.
pub const BOLTZMANN_CONSTANT_J_PER_K: f64 = 1.380_649e-23;
/// Exact SI elementary charge, coulombs.
pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

/// Validation failures for a bounded analytical diode profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileError {
    NonFiniteTemperature,
    NonPositiveTemperature,
    NonFiniteSaturationCurrent,
    NonPositiveSaturationCurrent,
    NonFiniteIdealityFactor,
    NonPositiveIdealityFactor,
    NonFiniteVoltageBound,
    ReversedVoltageDomain,
}

impl core::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NonFiniteTemperature => "temperature must be finite",
            Self::NonPositiveTemperature => "absolute temperature must be > 0 K",
            Self::NonFiniteSaturationCurrent => "saturation current must be finite",
            Self::NonPositiveSaturationCurrent => "saturation current must be > 0 A",
            Self::NonFiniteIdealityFactor => "ideality factor must be finite",
            Self::NonPositiveIdealityFactor => "ideality factor must be > 0",
            Self::NonFiniteVoltageBound => "voltage-domain bounds must be finite",
            Self::ReversedVoltageDomain => "voltage-domain minimum must not exceed maximum",
        })
    }
}

impl std::error::Error for ProfileError {}

/// Evaluation failures that remain explicit instead of being silently clamped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationError {
    InvalidProfile(ProfileError),
    NonFiniteVoltage,
    VoltageOutsideDomain {
        voltage: f64,
        minimum: f64,
        maximum: f64,
    },
    NumericalOverflow,
}

impl core::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidProfile(error) => write!(f, "invalid diode profile: {error}"),
            Self::NonFiniteVoltage => f.write_str("voltage must be finite"),
            Self::VoltageOutsideDomain {
                voltage,
                minimum,
                maximum,
            } => write!(
                f,
                "voltage {voltage} V lies outside declared domain [{minimum}, {maximum}] V"
            ),
            Self::NumericalOverflow => f.write_str("analytical diode evaluation overflowed"),
        }
    }
}

impl std::error::Error for EvaluationError {}

/// Exact-bit semantic key for one validated analytical profile.
///
/// This is not a cryptographic evidence identity. It only prevents accidental
/// cache/evaluation aliasing between claim-relevant profile values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnalyticalDiodeProfileKey {
    pub temperature_kelvin_bits: u64,
    pub saturation_current_amps_bits: u64,
    pub ideality_factor_bits: u64,
    pub voltage_min_volts_bits: u64,
    pub voltage_max_volts_bits: u64,
}

/// Bounded idealized diode analytical profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticalDiodeProfile {
    pub temperature_kelvin: f64,
    pub saturation_current_amps: f64,
    pub ideality_factor: f64,
    pub voltage_min_volts: f64,
    pub voltage_max_volts: f64,
}

impl AnalyticalDiodeProfile {
    /// Validate all claim-relevant numerical inputs.
    pub fn validate(self) -> Result<Self, ProfileError> {
        if !self.temperature_kelvin.is_finite() {
            return Err(ProfileError::NonFiniteTemperature);
        }
        if self.temperature_kelvin <= 0.0 {
            return Err(ProfileError::NonPositiveTemperature);
        }
        if !self.saturation_current_amps.is_finite() {
            return Err(ProfileError::NonFiniteSaturationCurrent);
        }
        if self.saturation_current_amps <= 0.0 {
            return Err(ProfileError::NonPositiveSaturationCurrent);
        }
        if !self.ideality_factor.is_finite() {
            return Err(ProfileError::NonFiniteIdealityFactor);
        }
        if self.ideality_factor <= 0.0 {
            return Err(ProfileError::NonPositiveIdealityFactor);
        }
        if !self.voltage_min_volts.is_finite() || !self.voltage_max_volts.is_finite() {
            return Err(ProfileError::NonFiniteVoltageBound);
        }
        if self.voltage_min_volts > self.voltage_max_volts {
            return Err(ProfileError::ReversedVoltageDomain);
        }
        Ok(self)
    }

    /// Build a deterministic semantic key after validation.
    pub fn key(self) -> Result<AnalyticalDiodeProfileKey, ProfileError> {
        let profile = self.validate()?;
        Ok(AnalyticalDiodeProfileKey {
            temperature_kelvin_bits: canonical_bits(profile.temperature_kelvin),
            saturation_current_amps_bits: canonical_bits(profile.saturation_current_amps),
            ideality_factor_bits: canonical_bits(profile.ideality_factor),
            voltage_min_volts_bits: canonical_bits(profile.voltage_min_volts),
            voltage_max_volts_bits: canonical_bits(profile.voltage_max_volts),
        })
    }
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

/// Thermal voltage `V_T = k_B T / q` in volts.
pub fn thermal_voltage(temperature_kelvin: f64) -> Result<f64, ProfileError> {
    if !temperature_kelvin.is_finite() {
        return Err(ProfileError::NonFiniteTemperature);
    }
    if temperature_kelvin <= 0.0 {
        return Err(ProfileError::NonPositiveTemperature);
    }
    Ok(BOLTZMANN_CONSTANT_J_PER_K * temperature_kelvin / ELEMENTARY_CHARGE_C)
}

/// Evaluate the bounded idealized diode relation
/// `I = I_s * (exp(V / (n * V_T)) - 1)`.
///
/// This is an analytical reference relation, not a complete physical diode model.
pub fn ideal_diode_current(
    profile: AnalyticalDiodeProfile,
    voltage_volts: f64,
) -> Result<f64, EvaluationError> {
    let profile = profile.validate().map_err(EvaluationError::InvalidProfile)?;
    validate_voltage(profile, voltage_volts)?;

    let thermal = thermal_voltage(profile.temperature_kelvin)
        .map_err(EvaluationError::InvalidProfile)?;
    let exponent = voltage_volts / (profile.ideality_factor * thermal);
    let current = profile.saturation_current_amps * exponent.exp_m1();

    if !current.is_finite() {
        return Err(EvaluationError::NumericalOverflow);
    }
    Ok(current)
}

/// Small-signal analytical conductance `dI/dV` under the same idealized profile.
pub fn ideal_diode_conductance(
    profile: AnalyticalDiodeProfile,
    voltage_volts: f64,
) -> Result<f64, EvaluationError> {
    let profile = profile.validate().map_err(EvaluationError::InvalidProfile)?;
    validate_voltage(profile, voltage_volts)?;

    let thermal = thermal_voltage(profile.temperature_kelvin)
        .map_err(EvaluationError::InvalidProfile)?;
    let scale = profile.ideality_factor * thermal;
    let exponent = voltage_volts / scale;
    let conductance = profile.saturation_current_amps * exponent.exp() / scale;

    if !conductance.is_finite() {
        return Err(EvaluationError::NumericalOverflow);
    }
    Ok(conductance)
}

fn validate_voltage(
    profile: AnalyticalDiodeProfile,
    voltage_volts: f64,
) -> Result<(), EvaluationError> {
    if !voltage_volts.is_finite() {
        return Err(EvaluationError::NonFiniteVoltage);
    }
    if voltage_volts < profile.voltage_min_volts || voltage_volts > profile.voltage_max_volts {
        return Err(EvaluationError::VoltageOutsideDomain {
            voltage: voltage_volts,
            minimum: profile.voltage_min_volts,
            maximum: profile.voltage_max_volts,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PROFILE: AnalyticalDiodeProfile = AnalyticalDiodeProfile {
        temperature_kelvin: 300.0,
        saturation_current_amps: 1.0e-12,
        ideality_factor: 1.0,
        voltage_min_volts: -0.2,
        voltage_max_volts: 0.8,
    };

    #[test]
    fn thermal_voltage_known_answer_at_300_k() {
        let value = thermal_voltage(300.0).unwrap();
        assert!((value - 0.025_851_999_786_435_535).abs() < 1.0e-15);
    }

    #[test]
    fn zero_bias_current_is_zero() {
        let current = ideal_diode_current(TEST_PROFILE, 0.0).unwrap();
        assert!(current.abs() < 1.0e-30);
    }

    #[test]
    fn forward_current_is_monotonic_in_frozen_domain() {
        let voltages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let mut previous = ideal_diode_current(TEST_PROFILE, voltages[0]).unwrap();
        for voltage in &voltages[1..] {
            let next = ideal_diode_current(TEST_PROFILE, *voltage).unwrap();
            assert!(next > previous);
            previous = next;
        }
    }

    #[test]
    fn small_reverse_bias_approaches_negative_saturation_current() {
        let current = ideal_diode_current(TEST_PROFILE, -0.2).unwrap();
        assert!(current < 0.0);
        assert!((current + TEST_PROFILE.saturation_current_amps).abs() < 5.0e-16);
    }

    #[test]
    fn conductance_matches_finite_difference() {
        let voltage = 0.2;
        let h = 1.0e-7;
        let plus = ideal_diode_current(TEST_PROFILE, voltage + h).unwrap();
        let minus = ideal_diode_current(TEST_PROFILE, voltage - h).unwrap();
        let numerical = (plus - minus) / (2.0 * h);
        let analytical = ideal_diode_conductance(TEST_PROFILE, voltage).unwrap();
        let relative = (analytical - numerical).abs() / analytical.abs();
        assert!(relative < 1.0e-8);
    }

    #[test]
    fn invalid_profiles_fail_closed() {
        assert_eq!(thermal_voltage(0.0), Err(ProfileError::NonPositiveTemperature));
        assert_eq!(thermal_voltage(f64::NAN), Err(ProfileError::NonFiniteTemperature));

        let invalid_ideality = AnalyticalDiodeProfile {
            ideality_factor: 0.0,
            ..TEST_PROFILE
        };
        assert_eq!(
            invalid_ideality.validate(),
            Err(ProfileError::NonPositiveIdealityFactor)
        );

        let invalid_saturation = AnalyticalDiodeProfile {
            saturation_current_amps: 0.0,
            ..TEST_PROFILE
        };
        assert_eq!(
            invalid_saturation.validate(),
            Err(ProfileError::NonPositiveSaturationCurrent)
        );
    }

    #[test]
    fn non_finite_voltage_is_rejected() {
        assert_eq!(
            ideal_diode_current(TEST_PROFILE, f64::NAN),
            Err(EvaluationError::NonFiniteVoltage)
        );
    }

    #[test]
    fn out_of_profile_voltage_is_not_clamped() {
        assert_eq!(
            ideal_diode_current(TEST_PROFILE, 0.9),
            Err(EvaluationError::VoltageOutsideDomain {
                voltage: 0.9,
                minimum: -0.2,
                maximum: 0.8,
            })
        );
    }

    #[test]
    fn overflow_is_explicit() {
        let profile = AnalyticalDiodeProfile {
            voltage_max_volts: 100.0,
            ..TEST_PROFILE
        };
        assert_eq!(
            ideal_diode_current(profile, 100.0),
            Err(EvaluationError::NumericalOverflow)
        );
    }

    #[test]
    fn profile_key_changes_when_claim_relevant_parameter_changes() {
        let base = TEST_PROFILE.key().unwrap();
        let changed = AnalyticalDiodeProfile {
            temperature_kelvin: 301.0,
            ..TEST_PROFILE
        }
        .key()
        .unwrap();
        assert_ne!(base, changed);
    }

    #[test]
    fn profile_key_canonicalizes_signed_zero_bounds() {
        let positive_zero = AnalyticalDiodeProfile {
            voltage_min_volts: 0.0,
            ..TEST_PROFILE
        }
        .key()
        .unwrap();
        let negative_zero = AnalyticalDiodeProfile {
            voltage_min_volts: -0.0,
            ..TEST_PROFILE
        }
        .key()
        .unwrap();
        assert_eq!(positive_zero, negative_zero);
    }

    #[test]
    fn exact_profile_repeats_deterministically() {
        let first = ideal_diode_current(TEST_PROFILE, 0.25).unwrap().to_bits();
        let second = ideal_diode_current(TEST_PROFILE, 0.25).unwrap().to_bits();
        assert_eq!(first, second);
    }
}
