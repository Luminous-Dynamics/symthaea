// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical diagnostics for scalar lattice Monte Carlo observables.
//!
//! These routines deliberately do **not** infer thermalization. The caller must
//! choose and record a burn-in; this module then reports descriptive diagnostics
//! for the retained chain: mean, variance, integrated autocorrelation time,
//! effective sample size, autocorrelation-adjusted standard error, and an exact
//! blocking estimate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeStatisticsError {
    TooFewSamples(usize),
    InvalidBurnIn { burn_in: usize, sample_count: usize },
    NonFiniteSample { index: usize, value: f64 },
    ZeroVariance,
    InvalidMaxLag(usize),
    InvalidBlockSize(usize),
    UnevenBlocking { sample_count: usize, block_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChainDiagnostics {
    pub raw_sample_count: usize,
    pub burn_in: usize,
    pub retained_sample_count: usize,
    pub mean: f64,
    pub sample_variance: f64,
    /// Initial-positive-sequence estimate of integrated autocorrelation time.
    pub tau_int: f64,
    /// N / (2 * tau_int). May be below 1 for an extremely correlated short chain.
    pub effective_sample_size: f64,
    /// sqrt(sample_variance / effective_sample_size).
    pub autocorrelation_adjusted_standard_error: f64,
    /// Standard error of exact, non-overlapping block means.
    pub blocking_standard_error: f64,
    pub block_size: usize,
    /// Maximum lag the caller allowed the autocorrelation estimator to inspect.
    /// This is preserved so downstream evidence can distinguish natural IPS
    /// termination from a still-positive correlation tail truncated by policy.
    pub max_lag: usize,
    /// Largest positive lag included in `tau_int`.
    pub positive_lag_count: usize,
}

pub fn sample_mean(samples: &[f64]) -> Result<f64, LatticeStatisticsError> {
    validate_finite(samples)?;
    if samples.is_empty() {
        return Err(LatticeStatisticsError::TooFewSamples(0));
    }
    Ok(samples.iter().sum::<f64>() / samples.len() as f64)
}

pub fn sample_variance(samples: &[f64]) -> Result<f64, LatticeStatisticsError> {
    validate_finite(samples)?;
    if samples.len() < 2 {
        return Err(LatticeStatisticsError::TooFewSamples(samples.len()));
    }
    let mean = sample_mean(samples)?;
    let sum_sq = samples
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>();
    Ok(sum_sq / (samples.len() - 1) as f64)
}

/// Estimate integrated autocorrelation time using an initial positive sequence.
///
/// `tau_int = 1/2 + sum_{t>=1} rho(t)` until the first non-positive lag or
/// `max_lag`, whichever comes first. This is intentionally simple and
/// transparent; production analyses may choose a more sophisticated windowing
/// scheme but must record that choice explicitly.
pub fn integrated_autocorrelation_time(
    samples: &[f64],
    max_lag: usize,
) -> Result<(f64, usize), LatticeStatisticsError> {
    validate_finite(samples)?;
    if samples.len() < 3 {
        return Err(LatticeStatisticsError::TooFewSamples(samples.len()));
    }
    if max_lag == 0 || max_lag >= samples.len() {
        return Err(LatticeStatisticsError::InvalidMaxLag(max_lag));
    }

    let mean = sample_mean(samples)?;
    let centered: Vec<f64> = samples.iter().map(|x| *x - mean).collect();
    let gamma0 = centered.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64;
    if gamma0 <= 0.0 || !gamma0.is_finite() {
        return Err(LatticeStatisticsError::ZeroVariance);
    }

    let mut tau = 0.5;
    let mut positive_lag_count = 0usize;
    for lag in 1..=max_lag {
        let count = samples.len() - lag;
        let covariance = (0..count)
            .map(|i| centered[i] * centered[i + lag])
            .sum::<f64>()
            / count as f64;
        let rho = covariance / gamma0;
        if !rho.is_finite() || rho <= 0.0 {
            break;
        }
        tau += rho;
        positive_lag_count = lag;
    }
    Ok((tau, positive_lag_count))
}

/// Standard error from exact, non-overlapping blocks.
///
/// The retained sample count must divide exactly by `block_size`; this avoids
/// silently discarding a tail and changing the effective dataset.
pub fn blocking_standard_error(
    samples: &[f64],
    block_size: usize,
) -> Result<f64, LatticeStatisticsError> {
    validate_finite(samples)?;
    if block_size == 0 {
        return Err(LatticeStatisticsError::InvalidBlockSize(block_size));
    }
    if samples.len() % block_size != 0 {
        return Err(LatticeStatisticsError::UnevenBlocking {
            sample_count: samples.len(),
            block_size,
        });
    }
    let block_count = samples.len() / block_size;
    if block_count < 2 {
        return Err(LatticeStatisticsError::TooFewSamples(block_count));
    }
    let block_means: Vec<f64> = samples
        .chunks_exact(block_size)
        .map(|block| block.iter().sum::<f64>() / block_size as f64)
        .collect();
    let variance_of_block_means = sample_variance(&block_means)?;
    Ok((variance_of_block_means / block_count as f64).sqrt())
}

pub fn analyze_scalar_chain(
    samples: &[f64],
    burn_in: usize,
    max_lag: usize,
    block_size: usize,
) -> Result<ChainDiagnostics, LatticeStatisticsError> {
    validate_finite(samples)?;
    if samples.len() < 3 {
        return Err(LatticeStatisticsError::TooFewSamples(samples.len()));
    }
    if burn_in >= samples.len() {
        return Err(LatticeStatisticsError::InvalidBurnIn {
            burn_in,
            sample_count: samples.len(),
        });
    }
    let retained = &samples[burn_in..];
    if retained.len() < 3 {
        return Err(LatticeStatisticsError::TooFewSamples(retained.len()));
    }

    let mean = sample_mean(retained)?;
    let variance = sample_variance(retained)?;
    let (tau_int, positive_lag_count) = integrated_autocorrelation_time(retained, max_lag)?;
    let effective_sample_size = retained.len() as f64 / (2.0 * tau_int);
    let autocorrelation_adjusted_standard_error = (variance / effective_sample_size).sqrt();
    let blocking_standard_error = blocking_standard_error(retained, block_size)?;

    Ok(ChainDiagnostics {
        raw_sample_count: samples.len(),
        burn_in,
        retained_sample_count: retained.len(),
        mean,
        sample_variance: variance,
        tau_int,
        effective_sample_size,
        autocorrelation_adjusted_standard_error,
        blocking_standard_error,
        block_size,
        max_lag,
        positive_lag_count,
    })
}

pub fn acceptance_rate(accepted: usize, attempted: usize) -> Option<f64> {
    if attempted == 0 || accepted > attempted {
        None
    } else {
        Some(accepted as f64 / attempted as f64)
    }
}

fn validate_finite(samples: &[f64]) -> Result<(), LatticeStatisticsError> {
    for (index, value) in samples.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(LatticeStatisticsError::NonFiniteSample { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_series(n: usize) -> Vec<f64> {
        let mut state = 0x1234_5678_u64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 11) as f64 / ((1u64 << 53) as f64);
                unit - 0.5
            })
            .collect()
    }

    #[test]
    fn correlated_chain_has_larger_tau_than_reference_noise() {
        let noise = lcg_series(512);
        let mut correlated = Vec::with_capacity(noise.len());
        let mut state = 0.0;
        for eps in &noise {
            state = 0.95 * state + *eps;
            correlated.push(state);
        }
        let (tau_noise, _) = integrated_autocorrelation_time(&noise, 64).unwrap();
        let (tau_corr, _) = integrated_autocorrelation_time(&correlated, 64).unwrap();
        assert!(tau_corr > tau_noise);
        assert!(tau_corr > 1.0);
    }

    #[test]
    fn burn_in_and_autocorrelation_window_are_preserved_in_diagnostics() {
        let samples = lcg_series(128);
        let diagnostics = analyze_scalar_chain(&samples, 32, 24, 8).unwrap();
        assert_eq!(diagnostics.raw_sample_count, 128);
        assert_eq!(diagnostics.burn_in, 32);
        assert_eq!(diagnostics.retained_sample_count, 96);
        assert_eq!(diagnostics.block_size, 8);
        assert_eq!(diagnostics.max_lag, 24);
        assert!(diagnostics.positive_lag_count <= diagnostics.max_lag);
    }

    #[test]
    fn blocking_never_silently_discards_tail() {
        let samples = lcg_series(100);
        assert!(matches!(
            blocking_standard_error(&samples, 16),
            Err(LatticeStatisticsError::UnevenBlocking { .. })
        ));
    }

    #[test]
    fn non_finite_samples_fail_closed() {
        let samples = [0.0, 1.0, f64::NAN, 2.0];
        assert!(matches!(
            sample_mean(&samples),
            Err(LatticeStatisticsError::NonFiniteSample { index: 2, .. })
        ));
    }

    #[test]
    fn acceptance_rate_validates_counts() {
        assert_eq!(acceptance_rate(75, 100), Some(0.75));
        assert_eq!(acceptance_rate(1, 0), None);
        assert_eq!(acceptance_rate(101, 100), None);
    }
}
