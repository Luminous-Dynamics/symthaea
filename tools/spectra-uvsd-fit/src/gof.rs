// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Category-resolved descriptive goodness-of-fit diagnostics for Gaussian UVSD.
//!
//! These statistics describe discrepancy between observed category counts
//! and fitted multinomial expectations. No asymptotic p-value or adequacy pass
//! is emitted. Sparse-cell and residual-df information are explicit; finite-
//! sample model-adequacy decisions require a separate parametric-bootstrap
//! calibration layer.

use std::fmt;

use super::normal_log_interval;
use super::sequential::StimulusClass;
use super::{CategoryTable, UvsdEstimate};

const PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-8;
const NEGATIVE_STATISTIC_TOLERANCE: f64 = 1.0e-8;

/// Descriptive computation status. No variant means that the model passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoodnessOfFitStatus {
    Computed,
    NoPositiveResidualDegreesOfFreedom,
}

/// Per-category observed-versus-expected values for both stimulus classes.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryFitResidual {
    pub category_order: i16,
    pub noise_observed: u64,
    pub noise_expected: f64,
    pub noise_pearson_residual: Option<f64>,
    pub signal_observed: u64,
    pub signal_expected: f64,
    pub signal_pearson_residual: Option<f64>,
}

/// Class-specific multinomial discrepancy summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassFitDiagnostics {
    pub stimulus_class: StimulusClass,
    pub observed_total: u64,
    pub expected_total: f64,
    pub pearson_x2: f64,
    pub likelihood_ratio_deviance_g2: f64,
    pub minimum_expected_cell: f64,
    pub expected_cells_below_5: usize,
    pub expected_cells_below_1: usize,
}

/// Category-resolved goodness-of-fit diagnostics for one frozen UVSD table.
///
/// `residual_degrees_of_freedom` is the nominal regular-model count
/// `K - 3`, derived from 2(K-1) unrestricted class probabilities minus
/// (K-1 criteria + mu + sigma). It is descriptive metadata only; this type
/// intentionally does not convert it to an asymptotic p-value.
#[derive(Debug, Clone, PartialEq)]
pub struct UvsdGoodnessOfFitDiagnostics {
    pub status: GoodnessOfFitStatus,
    pub category_count: usize,
    pub residual_degrees_of_freedom: Option<usize>,
    pub noise: ClassFitDiagnostics,
    pub signal: ClassFitDiagnostics,
    pub total_pearson_x2: f64,
    pub total_likelihood_ratio_deviance_g2: f64,
    pub maximum_absolute_pearson_residual: Option<f64>,
    pub expected_cells_below_5_total: usize,
    pub expected_cells_below_1_total: usize,
    pub category_residuals: Vec<CategoryFitResidual>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoodnessOfFitError {
    InvalidEstimate,
    InvalidThresholdCount,
    InvalidThresholdOrder,
    NonFiniteThreshold,
    NonFiniteExpectedProbability,
    ProbabilityMassDoesNotSumToOne,
    PositiveCountWithZeroExpectedProbability,
    NonFiniteStatistic,
    MateriallyNegativeStatistic,
}

impl fmt::Display for GoodnessOfFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidEstimate => "UVSD estimate must have finite mu and positive finite sigma",
            Self::InvalidThresholdCount => "threshold count must be exactly one fewer than category count",
            Self::InvalidThresholdOrder => "UVSD thresholds must be strictly increasing",
            Self::NonFiniteThreshold => "UVSD thresholds must be finite",
            Self::NonFiniteExpectedProbability => "model-implied category probability is non-finite or invalid",
            Self::ProbabilityMassDoesNotSumToOne => "model-implied category probabilities do not sum to one",
            Self::PositiveCountWithZeroExpectedProbability => "a positive observed category count has numerically zero fitted probability",
            Self::NonFiniteStatistic => "goodness-of-fit statistic is non-finite",
            Self::MateriallyNegativeStatistic => "goodness-of-fit statistic is materially negative",
        };
        f.write_str(message)
    }
}

impl std::error::Error for GoodnessOfFitError {}

/// Compute multinomial Pearson and likelihood-ratio discrepancies for one
/// ordered-category UVSD fit.
///
/// The function uses the same standard-normal log-interval primitive as the
/// fitter. It preserves zero-count categories without smoothing. A positive
/// observed count whose fitted probability underflows to zero is
/// `PositiveCountWithZeroExpectedProbability`, not silently repaired.
pub fn diagnose_uvsd_category_fit(
    table: &CategoryTable,
    estimate: &UvsdEstimate,
) -> Result<UvsdGoodnessOfFitDiagnostics, GoodnessOfFitError> {
    if !estimate.mu.is_finite() || !estimate.sigma.is_finite() || estimate.sigma <= 0.0 {
        return Err(GoodnessOfFitError::InvalidEstimate);
    }
    let category_count = table.categories().len();
    if estimate.thresholds.len() + 1 != category_count {
        return Err(GoodnessOfFitError::InvalidThresholdCount);
    }
    if estimate.thresholds.iter().any(|threshold| !threshold.is_finite()) {
        return Err(GoodnessOfFitError::NonFiniteThreshold);
    }
    if estimate.thresholds.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(GoodnessOfFitError::InvalidThresholdOrder);
    }

    let noise_probabilities = category_probabilities(
        &estimate.thresholds,
        0.0,
        1.0,
    )?;
    let signal_probabilities = category_probabilities(
        &estimate.thresholds,
        estimate.mu,
        estimate.sigma,
    )?;

    let noise_observed: Vec<u64> = table.categories().iter().map(|row| row.noise).collect();
    let signal_observed: Vec<u64> = table.categories().iter().map(|row| row.signal).collect();

    let (noise, noise_residuals) =
        class_diagnostics(StimulusClass::Noise, &noise_observed, &noise_probabilities)?;
    let (signal, signal_residuals) =
        class_diagnostics(StimulusClass::Signal, &signal_observed, &signal_probabilities)?;

    let total_pearson_x2 = noise.pearson_x2 + signal.pearson_x2;
    let total_likelihood_ratio_deviance_g2 =
        noise.likelihood_ratio_deviance_g2 + signal.likelihood_ratio_deviance_g2;
    if !total_pearson_x2.is_finite()
        || !total_likelihood_ratio_deviance_g2.is_finite()
    {
        return Err(GoodnessOfFitError::NonFiniteStatistic);
    }

    let mut maximum_absolute_pearson_residual: Option<f64> = None;
    let mut category_residuals = Vec::with_capacity(category_count);
    for index in 0..category_count {
        let noise_residual = noise_residuals[index];
        let signal_residual = signal_residuals[index];
        for residual in [noise_residual, signal_residual].into_iter().flatten() {
            let absolute = residual.abs();
            maximum_absolute_pearson_residual = Some(
                maximum_absolute_pearson_residual.map_or(absolute, |current| current.max(absolute)),
            );
        }
        category_residuals.push(CategoryFitResidual {
            category_order: table.categories()[index].order,
            noise_observed: noise_observed[index],
            noise_expected: noise.expected_cell_at(index, &noise_probabilities),
            noise_pearson_residual: noise_residual,
            signal_observed: signal_observed[index],
            signal_expected: signal.expected_cell_at(index, &signal_probabilities),
            signal_pearson_residual: signal_residual,
        });
    }

    let residual_degrees_of_freedom = category_count.checked_sub(3);
    let status = if residual_degrees_of_freedom.unwrap_or(0) == 0 {
        GoodnessOfFitStatus::NoPositiveResidualDegreesOfFreedom
    } else {
        GoodnessOfFitStatus::Computed
    };

    Ok(UvsdGoodnessOfFitDiagnostics {
        status,
        category_count,
        residual_degrees_of_freedom,
        expected_cells_below_5_total: noise.expected_cells_below_5
            + signal.expected_cells_below_5,
        expected_cells_below_1_total: noise.expected_cells_below_1
            + signal.expected_cells_below_1,
        noise,
        signal,
        total_pearson_x2,
        total_likelihood_ratio_deviance_g2,
        maximum_absolute_pearson_residual,
        category_residuals,
    })
}

impl ClassFitDiagnostics {
    fn expected_cell_at(&self, index: usize, probabilities: &[f64]) -> f64 {
        self.observed_total as f64 * probabilities[index]
    }
}

fn category_probabilities(
    thresholds: &[f64],
    mean: f64,
    sigma: f64,
) -> Result<Vec<f64>, GoodnessOfFitError> {
    if !mean.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return Err(GoodnessOfFitError::InvalidEstimate);
    }
    let mut probabilities = Vec::with_capacity(thresholds.len() + 1);
    for index in 0..=thresholds.len() {
        let lower = if index == 0 {
            f64::NEG_INFINITY
        } else {
            thresholds[index - 1]
        };
        let upper = if index == thresholds.len() {
            f64::INFINITY
        } else {
            thresholds[index]
        };
        let standardized_lower = if lower.is_infinite() {
            lower
        } else {
            (lower - mean) / sigma
        };
        let standardized_upper = if upper.is_infinite() {
            upper
        } else {
            (upper - mean) / sigma
        };
        let log_probability = normal_log_interval(
            standardized_lower,
            standardized_upper,
        );
        if log_probability.is_nan() || log_probability == f64::INFINITY {
            return Err(GoodnessOfFitError::NonFiniteExpectedProbability);
        }
        let probability = log_probability.exp();
        if !probability.is_finite() || probability < 0.0 {
            return Err(GoodnessOfFitError::NonFiniteExpectedProbability);
        }
        probabilities.push(probability);
    }
    let probability_sum: f64 = probabilities.iter().sum();
    if !probability_sum.is_finite()
        || probability_sum <= 0.0
        || (probability_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(GoodnessOfFitError::ProbabilityMassDoesNotSumToOne);
    }
    // Correct only floating-point summation drift after verifying that the
    // unnormalized model mass is already within the declared tolerance.
    for probability in &mut probabilities {
        *probability /= probability_sum;
    }
    Ok(probabilities)
}

fn class_diagnostics(
    stimulus_class: StimulusClass,
    observed: &[u64],
    probabilities: &[f64],
) -> Result<(ClassFitDiagnostics, Vec<Option<f64>>), GoodnessOfFitError> {
    if observed.len() != probabilities.len() || observed.is_empty() {
        return Err(GoodnessOfFitError::InvalidThresholdCount);
    }
    let observed_total: u64 = observed.iter().try_fold(0u64, |total, count| {
        total.checked_add(*count)
    }).ok_or(GoodnessOfFitError::NonFiniteStatistic)?;
    let mut expected_total = 0.0;
    let mut pearson_x2 = 0.0;
    let mut deviance_g2 = 0.0;
    let mut minimum_expected_cell = f64::INFINITY;
    let mut expected_cells_below_5 = 0;
    let mut expected_cells_below_1 = 0;
    let mut residuals = Vec::with_capacity(observed.len());

    for (&observed_count, &probability) in observed.iter().zip(probabilities) {
        let expected = observed_total as f64 * probability;
        if !expected.is_finite() || expected < 0.0 {
            return Err(GoodnessOfFitError::NonFiniteExpectedProbability);
        }
        expected_total += expected;
        minimum_expected_cell = minimum_expected_cell.min(expected);
        if expected < 5.0 {
            expected_cells_below_5 += 1;
        }
        if expected < 1.0 {
            expected_cells_below_1 += 1;
        }

        if expected == 0.0 {
            if observed_count > 0 {
                return Err(GoodnessOfFitError::PositiveCountWithZeroExpectedProbability);
            }
            residuals.push(None);
            continue;
        }
        let observed_value = observed_count as f64;
        let residual = (observed_value - expected) / expected.sqrt();
        pearson_x2 += residual * residual;
        if observed_count > 0 {
            deviance_g2 += 2.0 * observed_value * (observed_value / expected).ln();
        }
        residuals.push(Some(residual));
    }

    if !expected_total.is_finite()
        || !pearson_x2.is_finite()
        || !deviance_g2.is_finite()
    {
        return Err(GoodnessOfFitError::NonFiniteStatistic);
    }
    if deviance_g2 < -NEGATIVE_STATISTIC_TOLERANCE {
        return Err(GoodnessOfFitError::MateriallyNegativeStatistic);
    }
    if deviance_g2 < 0.0 {
        deviance_g2 = 0.0;
    }

    Ok((
        ClassFitDiagnostics {
            stimulus_class,
            observed_total,
            expected_total,
            pearson_x2,
            likelihood_ratio_deviance_g2: deviance_g2,
            minimum_expected_cell,
            expected_cells_below_5,
            expected_cells_below_1,
        },
        residuals,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CategoryCount, UvsdEstimate};

    fn example_estimate() -> UvsdEstimate {
        UvsdEstimate::new(1.4, 1.3, vec![-1.0, -0.3, 0.3, 1.0]).unwrap()
    }

    fn table(
        noise: Vec<u64>,
        signal: Vec<u64>,
    ) -> CategoryTable {
        CategoryTable::new(
            noise.into_iter()
                .zip(signal)
                .enumerate()
                .map(|(index, (noise, signal))| CategoryCount {
                    order: index as i16,
                    signal,
                    noise,
                })
                .collect(),
        ).unwrap()
    }

    fn expected_counts(probabilities: &[f64], total: u64) -> Vec<u64> {
        let mut counts: Vec<u64> = probabilities
            .iter()
            .map(|probability| (probability * total as f64).floor() as u64)
            .collect();
        let assigned: u64 = counts.iter().sum();
        let remainder = (total - assigned) as usize;
        for index in 0..remainder {
            counts[index % counts.len()] += 1;
        }
        counts
    }

    #[test]
    fn fitted_model_counts_have_small_deviance_and_preserve_class_totals() {
        let estimate = example_estimate();
        let noise_probabilities =
            category_probabilities(&estimate.thresholds, 0.0, 1.0).unwrap();
        let signal_probabilities =
            category_probabilities(&estimate.thresholds, estimate.mu, estimate.sigma).unwrap();
        let observed = table(
            expected_counts(&noise_probabilities, 20_000),
            expected_counts(&signal_probabilities, 20_000),
        );
        let diagnostic = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        assert_eq!(diagnostic.noise.observed_total, 20_000);
        assert_eq!(diagnostic.signal.observed_total, 20_000);
        assert!((diagnostic.noise.expected_total - 20_000.0).abs() < 1.0e-6);
        assert!((diagnostic.signal.expected_total - 20_000.0).abs() < 1.0e-6);
        assert_eq!(diagnostic.category_count, 5);
        assert_eq!(diagnostic.residual_degrees_of_freedom, Some(2));
        assert_eq!(diagnostic.status, GoodnessOfFitStatus::Computed);
        assert!(diagnostic.total_pearson_x2 < 0.01);
        assert!(diagnostic.total_likelihood_ratio_deviance_g2 < 0.01);
    }

    #[test]
    fn severe_category_misfit_increases_both_discrepancy_statistics() {
        let estimate = example_estimate();
        let noise_p = category_probabilities(&estimate.thresholds, 0.0, 1.0).unwrap();
        let signal_p = category_probabilities(&estimate.thresholds, estimate.mu, estimate.sigma).unwrap();
        let close = table(
            expected_counts(&noise_p, 4_000),
            expected_counts(&signal_p, 4_000),
        );
        let perturbed = table(
            vec![3_500, 250, 150, 75, 25],
            vec![100, 200, 400, 1_000, 2_300],
        );
        let close_gof = diagnose_uvsd_category_fit(&close, &estimate).unwrap();
        let perturbed_gof = diagnose_uvsd_category_fit(&perturbed, &estimate).unwrap();
        assert!(perturbed_gof.total_pearson_x2 > close_gof.total_pearson_x2);
        assert!(
            perturbed_gof.total_likelihood_ratio_deviance_g2
                > close_gof.total_likelihood_ratio_deviance_g2
        );
    }

    #[test]
    fn zero_observed_cells_are_not_smoothed() {
        let estimate = example_estimate();
        let observed = table(
            vec![100, 0, 50, 0, 100],
            vec![0, 100, 0, 100, 50],
        );
        let diagnostic = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        assert_eq!(diagnostic.category_residuals.len(), 5);
        assert_eq!(diagnostic.category_residuals[1].noise_observed, 0);
        assert!(diagnostic.category_residuals[1].noise_expected > 0.0);
        assert!(diagnostic.category_residuals[1].noise_pearson_residual.is_some());
    }

    #[test]
    fn per_cell_residual_matches_manual_calculation() {
        let estimate = example_estimate();
        let probabilities = category_probabilities(&estimate.thresholds, 0.0, 1.0).unwrap();
        let expected_first = 1_000.0 * probabilities[0];
        let observed = table(
            vec![200, 200, 200, 200, 200],
            vec![200, 200, 200, 200, 200],
        );
        let diagnostic = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        let manual = (200.0 - expected_first) / expected_first.sqrt();
        assert!(
            (diagnostic.category_residuals[0].noise_pearson_residual.unwrap() - manual).abs()
                < 1.0e-10
        );
    }

    #[test]
    fn sparse_expected_cells_are_reported_not_hidden() {
        let estimate = UvsdEstimate::new(10.0, 0.5, vec![-1.0, 0.0, 1.0]).unwrap();
        let observed = table(
            vec![100, 100, 100, 100],
            vec![0, 0, 0, 400],
        );
        let diagnostic = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        assert!(diagnostic.signal.expected_cells_below_5 > 0);
        assert!(diagnostic.expected_cells_below_5_total > 0);
        assert!(diagnostic.signal.minimum_expected_cell < 5.0);
    }

    #[test]
    fn three_categories_report_no_positive_residual_df() {
        let estimate = UvsdEstimate::new(1.0, 1.2, vec![-0.5, 0.5]).unwrap();
        let observed = table(vec![30, 40, 30], vec![20, 40, 40]);
        let diagnostic = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        assert_eq!(diagnostic.residual_degrees_of_freedom, Some(0));
        assert_eq!(
            diagnostic.status,
            GoodnessOfFitStatus::NoPositiveResidualDegreesOfFreedom
        );
    }

    #[test]
    fn malformed_estimates_reject_without_fabricating_diagnostics() {
        let observed = table(vec![30, 40, 30], vec![20, 40, 40]);
        let invalid_sigma = UvsdEstimate {
            mu: 1.0,
            sigma: 0.0,
            d_a: 1.0,
            thresholds: vec![-0.5, 0.5],
        };
        assert_eq!(
            diagnose_uvsd_category_fit(&observed, &invalid_sigma),
            Err(GoodnessOfFitError::InvalidEstimate)
        );
        let invalid_order = UvsdEstimate {
            mu: 1.0,
            sigma: 1.2,
            d_a: 0.9,
            thresholds: vec![0.5, -0.5],
        };
        assert_eq!(
            diagnose_uvsd_category_fit(&observed, &invalid_order),
            Err(GoodnessOfFitError::InvalidThresholdOrder)
        );
        let mismatch = UvsdEstimate::new(1.0, 1.2, vec![-0.5, -0.2, 0.5]).unwrap();
        assert_eq!(
            diagnose_uvsd_category_fit(&observed, &mismatch),
            Err(GoodnessOfFitError::InvalidThresholdCount)
        );
    }

    #[test]
    fn deterministic_category_order_is_preserved() {
        let estimate = example_estimate();
        let observed = table(vec![100, 200, 300, 400, 500], vec![500, 400, 300, 200, 100]);
        let first = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        let second = diagnose_uvsd_category_fit(&observed, &estimate).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.category_residuals.iter().map(|item| item.category_order).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }
}
