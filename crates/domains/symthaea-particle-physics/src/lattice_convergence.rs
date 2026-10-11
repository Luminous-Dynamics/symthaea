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
    NonFiniteDraw { chain: usize, draw: usize, value: f64 },
}

impl From<LatticeStatisticsError> for LatticeConvergenceError {
    fn from(value: LatticeStatisticsError) -> Self {
        Self::Statistics(value)
    }
}

fn validate_chains(chains: &[&[f64]]) -> Result<usize, LatticeConvergenceError> {
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
    for (chain_index, chain) in chains.iter().enumerate() {
        if chain.len() != draw_count {
            return Err(LatticeConvergenceError::UnequalChainLengths {
                expected: draw_count,
                actual: chain.len(),
            });
        }
        for (draw_index, value) in chain.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(LatticeConvergenceError::NonFiniteDraw {
                    chain: chain_index,
                    draw: draw_index,
                    value,
                });
            }
        }
    }
    Ok(draw_count)
}

fn split_chains(chains: &[&[f64]], draw_count: usize) -> Vec<Vec<f64>> {
    let n = draw_count / 2;
    let mut split = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        split.push(chain[..n].to_vec());
        split.push(chain[n..].to_vec());
    }
    split
}

fn rhat_for_already_split(chains: &[Vec<f64>]) -> Result<SplitRhatDiagnostic, LatticeConvergenceError> {
    let m = chains.len();
    let n = chains[0].len();
    let mut means = Vec::with_capacity(m);
    let mut variances = Vec::with_capacity(m);
    for chain in chains {
        means.push(sample_mean(chain)?);
        variances.push(sample_variance(chain)?);
    }
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
    Ok(SplitRhatDiagnostic {
        original_chain_count: m / 2,
        split_chain_count: m,
        draws_per_split: n,
        within_chain_variance: within,
        between_chain_variance: between,
        variance_estimate,
        r_hat: (variance_estimate / within).sqrt(),
    })
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
pub fn split_r_hat(chains: &[&[f64]]) -> Result<SplitRhatDiagnostic, LatticeConvergenceError> {
    let draw_count = validate_chains(chains)?;
    rhat_for_already_split(&split_chains(chains, draw_count))
}

/// Acklam-style rational approximation to the inverse standard-normal CDF.
/// Inputs in this module are strictly inside `(0,1)` because they come from the
/// Blom rank transform.
fn inverse_standard_normal(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
         2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
         1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
         2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
         1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
         6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
         4.374_664_141_464_968,
         2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // One-based ranks are start+1 through end inclusive of rank number.
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for position in start..end {
            ranks[order[position]] = average_rank;
        }
        start = end;
    }
    ranks
}

fn rank_normalize_split(split: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = split[0].len();
    let mut pooled = Vec::with_capacity(split.len() * n);
    for chain in split {
        pooled.extend_from_slice(chain);
    }
    let ranks = average_ranks(&pooled);
    let total = pooled.len() as f64;
    let scores: Vec<f64> = ranks
        .into_iter()
        .map(|rank| inverse_standard_normal((rank - 3.0 / 8.0) / (total + 1.0 / 4.0)))
        .collect();
    scores.chunks(n).map(|chunk| chunk.to_vec()).collect()
}

fn pooled_median(chains: &[&[f64]]) -> f64 {
    let mut pooled: Vec<f64> = chains.iter().flat_map(|chain| chain.iter().copied()).collect();
    pooled.sort_by(f64::total_cmp);
    let n = pooled.len();
    if n % 2 == 0 {
        (pooled[n / 2 - 1] + pooled[n / 2]) / 2.0
    } else {
        pooled[n / 2]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RankNormalizedRhatDiagnostic {
    /// Rank-normalized split-R-hat, primarily sensitive to location/bulk mismatch.
    pub bulk_r_hat: f64,
    /// Rank-normalized split-R-hat after folding around the pooled median.
    pub folded_r_hat: f64,
    /// Conservative summary: `max(bulk_r_hat, folded_r_hat)`.
    pub max_r_hat: f64,
}

/// Rank-normalized and folded split-R-hat following Vehtari et al.
///
/// Pooled average ranks are transformed using the Blom normal-score transform
/// `(r - 3/8) / (S + 1/4)`. Folding uses absolute deviations from the pooled
/// median. This function reports diagnostics only and encodes no pass threshold.
pub fn rank_normalized_split_r_hat(
    chains: &[&[f64]],
) -> Result<RankNormalizedRhatDiagnostic, LatticeConvergenceError> {
    let draw_count = validate_chains(chains)?;
    let split = split_chains(chains, draw_count);
    let normalized = rank_normalize_split(&split);
    let bulk = rhat_for_already_split(&normalized)?.r_hat;

    let median = pooled_median(chains);
    let folded_original: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| chain.iter().map(|value| (value - median).abs()).collect())
        .collect();
    let folded_refs: Vec<&[f64]> = folded_original.iter().map(Vec::as_slice).collect();
    let folded_split = split_chains(&folded_refs, draw_count);
    let folded_normalized = rank_normalize_split(&folded_split);
    let folded = rhat_for_already_split(&folded_normalized)?.r_hat;

    Ok(RankNormalizedRhatDiagnostic {
        bulk_r_hat: bulk,
        folded_r_hat: folded,
        max_r_hat: bulk.max(folded),
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
    fn rank_normalized_rhat_is_invariant_to_monotone_transform() {
        let a = [0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.9, 5.3];
        let b = [0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.1, 5.8];
        let exp_a: Vec<f64> = a.iter().copied().map(f64::exp).collect();
        let exp_b: Vec<f64> = b.iter().copied().map(f64::exp).collect();
        let original = rank_normalized_split_r_hat(&[&a, &b]).unwrap();
        let transformed = rank_normalized_split_r_hat(&[&exp_a, &exp_b]).unwrap();
        assert!((original.bulk_r_hat - transformed.bulk_r_hat).abs() < 1e-12);
    }

    #[test]
    fn folding_detects_scale_mismatch_with_similar_location() {
        let a = [-1.2, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.2];
        let b = [-4.8, -3.2, -1.6, -0.4, 0.4, 1.6, 3.2, 4.8];
        let diagnostic = rank_normalized_split_r_hat(&[&a, &b]).unwrap();
        assert!(diagnostic.folded_r_hat > diagnostic.bulk_r_hat);
        assert_eq!(diagnostic.max_r_hat, diagnostic.folded_r_hat);
    }

    #[test]
    fn inverse_normal_is_centered_and_symmetric() {
        assert!(inverse_standard_normal(0.5).abs() < 1e-14);
        let lo = inverse_standard_normal(0.01);
        let hi = inverse_standard_normal(0.99);
        assert!((lo + hi).abs() < 1e-10);
        assert!(lo < 0.0 && hi > 0.0);
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
    fn non_finite_draws_fail_closed() {
        let a = [0.0, 1.0, 0.2, 0.8];
        let b = [0.1, f64::NAN, 0.3, 0.7];
        assert!(matches!(
            rank_normalized_split_r_hat(&[&a, &b]),
            Err(LatticeConvergenceError::NonFiniteDraw { .. })
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
