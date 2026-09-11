// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Correlator-to-spectrum analysis primitives for lattice spectroscopy.
//!
//! These routines transform already-computed correlators into descriptive
//! effective-mass estimates and evaluate caller-specified constant windows.
//! They do not generate gauge configurations, choose operators, discover a
//! plateau automatically, or establish a physical mass without the surrounding
//! finite-volume / continuum / uncertainty analysis.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeSpectroscopyError {
    InvalidTemporalSpacing(f64),
    TooFewCorrelatorPoints(usize),
    NonFiniteCorrelator { index: usize, value: f64 },
    NonPositiveCorrelator { index: usize, value: f64 },
    InvalidCoshArgument { index: usize, argument: f64 },
    LengthMismatch { values: usize, uncertainties: usize },
    InvalidUncertainty { index: usize, value: f64 },
    InvalidWindow { start: usize, end: usize, len: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectiveMassPoint {
    /// Index corresponding to the central/start time slice used by the estimator.
    pub time_index: usize,
    pub mass: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConstantWindowFit {
    pub start: usize,
    /// Inclusive end index in the supplied values array.
    pub end: usize,
    pub weighted_mean: f64,
    pub standard_error: f64,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

fn validate_spacing(a_t: f64) -> Result<(), LatticeSpectroscopyError> {
    if !a_t.is_finite() || a_t <= 0.0 {
        return Err(LatticeSpectroscopyError::InvalidTemporalSpacing(a_t));
    }
    Ok(())
}

fn validate_correlator(correlator: &[f64]) -> Result<(), LatticeSpectroscopyError> {
    for (index, value) in correlator.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(LatticeSpectroscopyError::NonFiniteCorrelator { index, value });
        }
        if value <= 0.0 {
            return Err(LatticeSpectroscopyError::NonPositiveCorrelator { index, value });
        }
    }
    Ok(())
}

/// Standard log-ratio effective mass:
///
/// `m_eff(t) = ln(C(t) / C(t+1)) / a_t`.
///
/// This estimator is appropriate when a single forward exponential dominates.
pub fn log_effective_mass(
    c_t: f64,
    c_t_plus_1: f64,
    a_t: f64,
) -> Result<f64, LatticeSpectroscopyError> {
    validate_spacing(a_t)?;
    for (index, value) in [(0usize, c_t), (1usize, c_t_plus_1)] {
        if !value.is_finite() {
            return Err(LatticeSpectroscopyError::NonFiniteCorrelator { index, value });
        }
        if value <= 0.0 {
            return Err(LatticeSpectroscopyError::NonPositiveCorrelator { index, value });
        }
    }
    Ok((c_t / c_t_plus_1).ln() / a_t)
}

pub fn log_effective_mass_series(
    correlator: &[f64],
    a_t: f64,
) -> Result<Vec<EffectiveMassPoint>, LatticeSpectroscopyError> {
    validate_spacing(a_t)?;
    validate_correlator(correlator)?;
    if correlator.len() < 2 {
        return Err(LatticeSpectroscopyError::TooFewCorrelatorPoints(
            correlator.len(),
        ));
    }
    correlator
        .windows(2)
        .enumerate()
        .map(|(time_index, pair)| {
            Ok(EffectiveMassPoint {
                time_index,
                mass: log_effective_mass(pair[0], pair[1], a_t)?,
            })
        })
        .collect()
}

/// Periodic/cosh effective mass:
///
/// `m_eff(t) = acosh((C(t-1) + C(t+1)) / (2 C(t))) / a_t`.
pub fn cosh_effective_mass(
    c_t_minus_1: f64,
    c_t: f64,
    c_t_plus_1: f64,
    a_t: f64,
) -> Result<f64, LatticeSpectroscopyError> {
    validate_spacing(a_t)?;
    for (index, value) in [
        (0usize, c_t_minus_1),
        (1usize, c_t),
        (2usize, c_t_plus_1),
    ] {
        if !value.is_finite() {
            return Err(LatticeSpectroscopyError::NonFiniteCorrelator { index, value });
        }
        if value <= 0.0 {
            return Err(LatticeSpectroscopyError::NonPositiveCorrelator { index, value });
        }
    }
    let argument = (c_t_minus_1 + c_t_plus_1) / (2.0 * c_t);
    if !argument.is_finite() || argument < 1.0 {
        return Err(LatticeSpectroscopyError::InvalidCoshArgument {
            index: 1,
            argument,
        });
    }
    Ok(argument.acosh() / a_t)
}

pub fn cosh_effective_mass_series(
    correlator: &[f64],
    a_t: f64,
) -> Result<Vec<EffectiveMassPoint>, LatticeSpectroscopyError> {
    validate_spacing(a_t)?;
    validate_correlator(correlator)?;
    if correlator.len() < 3 {
        return Err(LatticeSpectroscopyError::TooFewCorrelatorPoints(
            correlator.len(),
        ));
    }
    (1..(correlator.len() - 1))
        .map(|time_index| {
            Ok(EffectiveMassPoint {
                time_index,
                mass: cosh_effective_mass(
                    correlator[time_index - 1],
                    correlator[time_index],
                    correlator[time_index + 1],
                    a_t,
                )?,
            })
        })
        .collect()
}

/// Evaluate a caller-selected constant window with independent Gaussian errors.
///
/// This intentionally does **not** search over windows. Correlated covariance
/// fits should replace this helper when a covariance matrix is available.
pub fn weighted_constant_window_fit(
    values: &[f64],
    uncertainties: &[f64],
    start: usize,
    end: usize,
) -> Result<ConstantWindowFit, LatticeSpectroscopyError> {
    if values.len() != uncertainties.len() {
        return Err(LatticeSpectroscopyError::LengthMismatch {
            values: values.len(),
            uncertainties: uncertainties.len(),
        });
    }
    if start > end || end >= values.len() || end - start + 1 < 2 {
        return Err(LatticeSpectroscopyError::InvalidWindow {
            start,
            end,
            len: values.len(),
        });
    }

    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;
    for index in start..=end {
        let value = values[index];
        let sigma = uncertainties[index];
        if !value.is_finite() {
            return Err(LatticeSpectroscopyError::NonFiniteCorrelator { index, value });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(LatticeSpectroscopyError::InvalidUncertainty {
                index,
                value: sigma,
            });
        }
        let weight = 1.0 / (sigma * sigma);
        weight_sum += weight;
        weighted_value_sum += weight * value;
    }

    let weighted_mean = weighted_value_sum / weight_sum;
    let mut chi_square = 0.0;
    for index in start..=end {
        let residual = (values[index] - weighted_mean) / uncertainties[index];
        chi_square += residual * residual;
    }
    let point_count = end - start + 1;
    let degrees_of_freedom = point_count - 1;

    Ok(ConstantWindowFit {
        start,
        end,
        weighted_mean,
        standard_error: (1.0 / weight_sum).sqrt(),
        chi_square,
        degrees_of_freedom,
        chi_square_per_dof: chi_square / degrees_of_freedom as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_effective_mass_recovers_exact_exponential() {
        let mass = 0.73;
        let a_t = 0.12;
        let amplitude = 4.2;
        let correlator: Vec<f64> = (0..8)
            .map(|t| amplitude * (-mass * a_t * t as f64).exp())
            .collect();
        let effective = log_effective_mass_series(&correlator, a_t).unwrap();
        for point in effective {
            assert!((point.mass - mass).abs() < 1e-12);
        }
    }

    #[test]
    fn cosh_effective_mass_recovers_periodic_single_state() {
        let mass = 0.51;
        let a_t = 0.2;
        let temporal_extent = 16usize;
        let correlator: Vec<f64> = (0..temporal_extent)
            .map(|t| {
                let distance = temporal_extent as f64 / 2.0 - t as f64;
                (mass * a_t * distance).cosh()
            })
            .collect();
        let effective = cosh_effective_mass_series(&correlator, a_t).unwrap();
        for point in effective {
            assert!((point.mass - mass).abs() < 1e-12);
        }
    }

    #[test]
    fn constant_window_fit_does_not_choose_window() {
        let values = [1.0, 1.1, 0.9, 8.0];
        let errors = [0.1, 0.1, 0.1, 0.1];
        let fit = weighted_constant_window_fit(&values, &errors, 0, 2).unwrap();
        assert!((fit.weighted_mean - 1.0).abs() < 1e-12);
        assert_eq!(fit.start, 0);
        assert_eq!(fit.end, 2);
        assert_eq!(fit.degrees_of_freedom, 2);
    }

    #[test]
    fn nonpositive_correlators_fail_closed() {
        let correlator = [1.0, 0.0, 0.5];
        assert!(matches!(
            log_effective_mass_series(&correlator, 0.1),
            Err(LatticeSpectroscopyError::NonPositiveCorrelator { index: 1, .. })
        ));
    }

    #[test]
    fn invalid_window_is_rejected() {
        let values = [1.0, 1.0, 1.0];
        let errors = [0.1, 0.1, 0.1];
        assert!(matches!(
            weighted_constant_window_fit(&values, &errors, 1, 1),
            Err(LatticeSpectroscopyError::InvalidWindow { .. })
        ));
    }
}
