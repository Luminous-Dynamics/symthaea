// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-chain diagnostics for lattice Monte Carlo qualification.
//!
//! These routines are descriptive. In particular, this module deliberately
//! defines no universal threshold at which a chain becomes "converged".

use crate::lattice_statistics::{ChainDiagnostics, LatticeStatisticsError, sample_mean, sample_variance};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeConvergenceError {
    Statistics(LatticeStatisticsError),
    TooFewChains(usize),
    TooFewDraws(usize),
    UnequalChainLengths { expected: usize, actual: usize },
    OddChainLength(usize),
    ZeroWithinChainVariance,
    InvalidStandardError(f64),
}

impl From<LatticeStatisticsError> for LatticeConvergenceError {
    fn from(value: LatticeStatisticsError) -> Self {
        Self::Statistics(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplitRhatDiagnostic {
    pub original_chain_count: usize,
    pub split_chain_count: usize,
    pub draws_per_split: usize,
    pub within_chain_variance: f64,
    pub between_chain_variance: f64,
    pub variance_estimate: f64,
    pub r_hat: f64,
}

/// Classical split-R-hat / potential scale reduction diagnostic.
///
/// Every input chain must have the same even length. Each chain is split in
/// half, then the between- and within-chain variances are compared. This is the
/// classical split diagnostic, not rank-normalized/folded R-hat.
pub fn split_r_hat(chains: &[&[f64]]) -> Result<SplitRhatDiagnostic, LatticeConvergenceError> {
    if chains.len() < 2 {
        return Err(LatticeConvergenceError::TooFewChains(chains.len()));
    }
    let draw_count = chains[0].len();
    if draw_count < 4 {
        return Err(LatticeConvergenceError::TooFewDraws(draw_count));
    }
    if draw_count % 2 != 0 {
        return Err(LatticeConvergenceError::OddChainLength(draw_count));
    }
    for chain in &chains[1..] {
        if chain.len() != draw_count {
            return Err(LatticeConvergenceError::UnequalChainLengths {
                expected: draw_count,
                actual: chain.len(),
            });
        }
    }

    let n = draw_count / 2;
    let mut means = Vec::with_capacity(chains.len() * 2);
    let mut variances = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        for half in [ &chain[..n], &chain[n..] ] {
            means.push(sample_mean(half)?);
            variances.push(sample_variance(half)?);
        }
    }

    let m = means.len();
    let within = variances.iter().sum::<f64>() / m as f64;
    if within <= 0.0 || !within.is_finite() {
        return Err(LatticeConvergenceError::ZeroWithinChainVariance);
    }
    let mean_of_means = means.iter().sum::<f64>() / m as f64;
    let between = n as f64
        * means.iter().map(|mean| {
            let d = *mean - mean_of_means;
            d * d
        }).sum::<f64>()
        / (m - 1) as f64;
    let variance_estimate = ((n - 1) as f64 / n as f64) * within + between / n as f64;
    let r_hat = (variance_estimate / within).sqrt();

    Ok(SplitRhatDiagnostic {
        original_chain_count: chains.len(),
        split_chain_count: m,
        draws_per_split: n,
        within_chain_variance: within,
        between_chain_variance: between,
        variance_estimate,
        r_hat,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChainMeanComparison {
    pub first_mean: f64,
    pub second_mean: f64,
    pub difference: f64,
    pub combined_standard_error: f64,
    /// Absolute mean difference divided by the quadrature-combined adjusted SE.
    pub normalized_difference: f64,
}

/// Compare two already-qualified scalar-chain summaries without declaring a
/// pass/fail threshold.
pub fn compare_chain_means(
    first: &ChainDiagnostics,
    second: &ChainDiagnostics,
) -> Result<ChainMeanComparison, LatticeConvergenceError> {
    let a = first.autocorrelation_adjusted_standard_error;
    let b = second.autocorrelation_adjusted_standard_error;
    if !a.is_finite() || a <= 0.0 {
        return Err(LatticeConvergenceError::InvalidStandardError(a));
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(LatticeConvergenceError::InvalidStandardError(b));
    }
    let difference = first.mean - second.mean;
    let combined_standard_error = (a * a + b * b).sqrt();
    Ok(ChainMeanComparison {
        first_mean: first.mean,
        second_mean: second.mean,
        difference,
        combined_standard_error,
        normalized_difference: difference.abs() / combined_standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_statistics::analyze_scalar_chain;

    #[test]
    fn shifted_chains_have_larger_split_rhat_than_aligned_chains() {
        let a = [0.0, 1.0, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7];
        let b = [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6];
        let shifted: Vec<f64> = b.iter().map(|x| *x + 5.0).collect();
        let aligned_rhat = split_r_hat(&[&a, &b]).unwrap().r_hat;
        let shifted_rhat = split_r_hat(&[&a, &shifted]).unwrap().r_hat;
        assert!(shifted_rhat > aligned_rhat);
        assert!(shifted_rhat > 1.0);
    }

    #[test]
    fn odd_length_is_rejected_instead_of_dropping_a_draw() {
        let a = [0.0, 1.0, 0.0, 1.0, 0.0];
        let b = [0.1, 0.9, 0.1, 0.9, 0.1];
        assert!(matches!(
            split_r_hat(&[&a, &b]),
            Err(LatticeConvergenceError::OddChainLength(5))
        ));
    }

    #[test]
    fn chain_mean_comparison_uses_autocorrelation_adjusted_errors() {
        let a: Vec<f64> = (0..96).map(|i| ((i * 37 % 101) as f64) / 101.0).collect();
        let b: Vec<f64> = a.iter().map(|x| x + 0.05).collect();
        let da = analyze_scalar_chain(&a, 0, 16, 8).unwrap();
        let db = analyze_scalar_chain(&b, 0, 16, 8).unwrap();
        let comparison = compare_chain_means(&da, &db).unwrap();
        assert!((comparison.difference + 0.05).abs() < 1e-12);
        assert!(comparison.combined_standard_error > 0.0);
        assert!(comparison.normalized_difference.is_finite());
    }
}
