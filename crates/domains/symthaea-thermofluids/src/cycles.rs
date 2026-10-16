// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Idealized thermodynamic-cycle baselines for energy-conversion research.
//!
//! These closed-form models are useful for sanity checks and early design-space
//! screening. They are *not* predictions of real engine performance: they omit
//! mechanical and electrical losses, heat-transfer limits, pressure drops,
//! combustion chemistry, fluid-property variation, packaging, material limits,
//! transients, aging, and balance-of-plant loads.
//!
//! Keep ideal-cycle estimates distinct from measured or calibrated system data.
//! A candidate should not be called better than a production engine until a
//! like-for-like system boundary, independently checked models, and experimental
//! evidence support that claim.

/// Validation failures for ideal-cycle inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleError {
    /// At least one input was NaN or infinite.
    NonFiniteInput,
    /// A reservoir temperature must be above absolute zero.
    NonPositiveTemperature,
    /// The hot reservoir must be hotter than the cold reservoir.
    InvalidTemperatureOrder,
    /// The specific-heat ratio must be finite and greater than one.
    InvalidSpecificHeatRatio,
    /// A Brayton pressure ratio must be finite and greater than one.
    InvalidPressureRatio,
    /// An Otto or Diesel compression ratio must be finite and greater than one.
    InvalidCompressionRatio,
    /// The Diesel cutoff ratio must satisfy 1 < cutoff < compression ratio.
    InvalidCutoffRatio,
    /// Finite inputs caused numeric overflow or an efficiency outside [0, 1].
    NumericalFailure,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::NonFiniteInput => "all cycle inputs must be finite",
            Self::NonPositiveTemperature => "absolute temperatures must be greater than zero kelvin",
            Self::InvalidTemperatureOrder => "hot-reservoir temperature must exceed cold-reservoir temperature",
            Self::InvalidSpecificHeatRatio => "specific-heat ratio gamma must be greater than one",
            Self::InvalidPressureRatio => "Brayton pressure ratio must be greater than one",
            Self::InvalidCompressionRatio => "compression ratio must be greater than one",
            Self::InvalidCutoffRatio => "Diesel cutoff ratio must be greater than one and less than the compression ratio",
            Self::NumericalFailure => "cycle calculation overflowed or produced an efficiency outside [0, 1]",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CycleError {}

fn validate_gamma(gamma: f64) -> Result<(), CycleError> {
    if !gamma.is_finite() {
        return Err(CycleError::NonFiniteInput);
    }
    if gamma <= 1.0 {
        return Err(CycleError::InvalidSpecificHeatRatio);
    }
    Ok(())
}

fn checked_efficiency(efficiency: f64) -> Result<f64, CycleError> {
    if !efficiency.is_finite() || !(0.0..=1.0).contains(&efficiency) {
        return Err(CycleError::NumericalFailure);
    }
    Ok(efficiency)
}

/// Reversible Carnot efficiency bound: eta = 1 - T_cold / T_hot.
///
/// Temperatures are in kelvin. This is an upper bound for a heat engine operating
/// between two reservoirs, not a claim that a real engine can attain the bound.
pub fn carnot_efficiency(t_cold_k: f64, t_hot_k: f64) -> Result<f64, CycleError> {
    if !t_cold_k.is_finite() || !t_hot_k.is_finite() {
        return Err(CycleError::NonFiniteInput);
    }
    if t_cold_k <= 0.0 || t_hot_k <= 0.0 {
        return Err(CycleError::NonPositiveTemperature);
    }
    if t_hot_k <= t_cold_k {
        return Err(CycleError::InvalidTemperatureOrder);
    }
    Ok(1.0 - t_cold_k / t_hot_k)
}

/// Ideal Brayton-cycle thermal efficiency.
///
/// Uses eta = 1 - pressure_ratio^(-(gamma - 1) / gamma) for an ideal-gas
/// working fluid with constant specific heats and isentropic compression and
/// expansion. pressure_ratio is the compressor pressure ratio; gamma = cp/cv.
pub fn ideal_brayton_efficiency(
    pressure_ratio: f64,
    gamma: f64,
) -> Result<f64, CycleError> {
    if !pressure_ratio.is_finite() {
        return Err(CycleError::NonFiniteInput);
    }
    if pressure_ratio <= 1.0 {
        return Err(CycleError::InvalidPressureRatio);
    }
    validate_gamma(gamma)?;

    let efficiency = 1.0 - pressure_ratio.powf(-(gamma - 1.0) / gamma);
    checked_efficiency(efficiency)
}

/// Ideal air-standard Otto-cycle thermal efficiency.
///
/// Uses eta = 1 - compression_ratio^(-(gamma - 1)). Assumes constant specific
/// heats, isentropic compression/expansion, and idealized constant-volume heat
/// addition. It is a cycle benchmark, not a complete spark-ignition engine model.
pub fn air_standard_otto_efficiency(
    compression_ratio: f64,
    gamma: f64,
) -> Result<f64, CycleError> {
    if !compression_ratio.is_finite() {
        return Err(CycleError::NonFiniteInput);
    }
    if compression_ratio <= 1.0 {
        return Err(CycleError::InvalidCompressionRatio);
    }
    validate_gamma(gamma)?;

    let efficiency = 1.0 - compression_ratio.powf(-(gamma - 1.0));
    checked_efficiency(efficiency)
}

/// Ideal air-standard Diesel-cycle thermal efficiency.
///
/// Uses eta = 1 - (1 / r^(gamma - 1)) *
/// ((cutoff_ratio^gamma - 1) / (gamma * (cutoff_ratio - 1))), where r is
/// the compression ratio and cutoff_ratio = V3/V2 is the constant-pressure
/// heat-addition cutoff ratio. Assumes constant specific heats and isentropic
/// compression/expansion. It is not a real engine map or a fuel-consumption model.
pub fn air_standard_diesel_efficiency(
    compression_ratio: f64,
    cutoff_ratio: f64,
    gamma: f64,
) -> Result<f64, CycleError> {
    if !compression_ratio.is_finite() || !cutoff_ratio.is_finite() {
        return Err(CycleError::NonFiniteInput);
    }
    if compression_ratio <= 1.0 {
        return Err(CycleError::InvalidCompressionRatio);
    }
    if cutoff_ratio <= 1.0 || cutoff_ratio >= compression_ratio {
        return Err(CycleError::InvalidCutoffRatio);
    }
    validate_gamma(gamma)?;

    let heat_addition_factor =
        (cutoff_ratio.powf(gamma) - 1.0) / (gamma * (cutoff_ratio - 1.0));
    let efficiency =
        1.0 - heat_addition_factor / compression_ratio.powf(gamma - 1.0);
    checked_efficiency(efficiency)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-4;

    #[test]
    fn carnot_known_value_and_input_validation() {
        let eta = carnot_efficiency(300.0, 600.0).unwrap();
        assert!((eta - 0.5).abs() < 1e-12);
        assert_eq!(
            carnot_efficiency(600.0, 300.0),
            Err(CycleError::InvalidTemperatureOrder)
        );
        assert_eq!(
            carnot_efficiency(0.0, 300.0),
            Err(CycleError::NonPositiveTemperature)
        );
        assert_eq!(
            carnot_efficiency(f64::NAN, 300.0),
            Err(CycleError::NonFiniteInput)
        );
    }

    #[test]
    fn ideal_brayton_matches_reference_value() {
        // gamma=1.4, pressure ratio=8 -> eta ~= 0.44796.
        let eta = ideal_brayton_efficiency(8.0, 1.4).unwrap();
        assert!((eta - 0.447_955_243_2).abs() < TOLERANCE);
        assert!(ideal_brayton_efficiency(12.0, 1.4).unwrap() > eta);
        assert_eq!(
            ideal_brayton_efficiency(1.0, 1.4),
            Err(CycleError::InvalidPressureRatio)
        );
    }

    #[test]
    fn air_standard_otto_matches_reference_value() {
        // gamma=1.4, compression ratio=8 -> eta ~= 0.56472.
        let eta = air_standard_otto_efficiency(8.0, 1.4).unwrap();
        assert!((eta - 0.564_724_718_4).abs() < TOLERANCE);
        assert!(air_standard_otto_efficiency(10.0, 1.4).unwrap() > eta);
        assert_eq!(
            air_standard_otto_efficiency(1.0, 1.4),
            Err(CycleError::InvalidCompressionRatio)
        );
    }

    #[test]
    fn air_standard_diesel_matches_reference_value() {
        // gamma=1.4, compression ratio=18, cutoff ratio=2 -> eta ~= 0.63158.
        let eta = air_standard_diesel_efficiency(18.0, 2.0, 1.4).unwrap();
        assert!((eta - 0.631_577_531_4).abs() < TOLERANCE);
        assert!(air_standard_diesel_efficiency(18.0, 1.5, 1.4).unwrap() > eta);
        assert_eq!(
            air_standard_diesel_efficiency(18.0, 18.0, 1.4),
            Err(CycleError::InvalidCutoffRatio)
        );
    }

    #[test]
    fn nonphysical_specific_heat_ratio_is_rejected() {
        assert_eq!(
            ideal_brayton_efficiency(8.0, 1.0),
            Err(CycleError::InvalidSpecificHeatRatio)
        );
        assert_eq!(
            air_standard_otto_efficiency(8.0, f64::INFINITY),
            Err(CycleError::NonFiniteInput)
        );
        assert_eq!(
            air_standard_diesel_efficiency(1.0e308, 2.0, 1000.0),
            Err(CycleError::NumericalFailure)
        );
    }
}
