// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Common Research Metrics for Epistemic Markets
//!
//! This module provides standardized metrics used across all research studies:
//! - Brier Score and decomposition
//! - Expected Calibration Error (ECE)
//! - Resolution and reliability metrics
//! - Information gain measures
//! - Statistical significance testing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// BRIER SCORE AND DECOMPOSITION
// ============================================================================

/// Brier Score calculation for probability predictions
/// Lower is better: 0 = perfect, 0.25 = random guessing on binary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrierScore {
    /// Raw Brier score value
    pub score: f64,
    /// Number of predictions used
    pub n: usize,
    /// Standard error of the score
    pub standard_error: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
}

impl BrierScore {
    /// Calculate Brier score from predictions and outcomes
    /// predictions: vector of (predicted_probability, actual_outcome)
    /// where actual_outcome is 1.0 if event occurred, 0.0 otherwise
    pub fn calculate(predictions: &[(f64, f64)]) -> Self {
        if predictions.is_empty() {
            return Self {
                score: f64::NAN,
                n: 0,
                standard_error: f64::NAN,
                confidence_interval: (f64::NAN, f64::NAN),
            };
        }

        let n = predictions.len();
        let sum_squared_error: f64 = predictions
            .iter()
            .map(|(p, o)| (p - o).powi(2))
            .sum();
        let score = sum_squared_error / n as f64;

        // Calculate standard error
        let variance: f64 = predictions
            .iter()
            .map(|(p, o)| {
                let error = (p - o).powi(2);
                (error - score).powi(2)
            })
            .sum::<f64>()
            / (n - 1).max(1) as f64;
        let standard_error = (variance / n as f64).sqrt();

        // 95% CI using normal approximation
        let z = 1.96;
        let ci_lower = (score - z * standard_error).max(0.0);
        let ci_upper = (score + z * standard_error).min(1.0);

        Self {
            score,
            n,
            standard_error,
            confidence_interval: (ci_lower, ci_upper),
        }
    }

    /// Calculate Brier score for multi-outcome predictions
    pub fn calculate_multiclass(predictions: &[Vec<f64>], outcomes: &[usize]) -> Self {
        if predictions.is_empty() || predictions.len() != outcomes.len() {
            return Self {
                score: f64::NAN,
                n: 0,
                standard_error: f64::NAN,
                confidence_interval: (f64::NAN, f64::NAN),
            };
        }

        let n = predictions.len();
        let sum_squared_error: f64 = predictions
            .iter()
            .zip(outcomes.iter())
            .map(|(probs, &outcome)| {
                probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let target = if i == outcome { 1.0 } else { 0.0 };
                        (p - target).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum();
        let score = sum_squared_error / n as f64;

        Self {
            score,
            n,
            standard_error: 0.0, // Simplified for multiclass
            confidence_interval: (score, score),
        }
    }
}

/// Murphy decomposition of Brier score
/// Brier = Reliability - Resolution + Uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrierDecomposition {
    /// Reliability (calibration): how well predictions match actual frequencies
    /// Lower is better
    pub reliability: f64,
    /// Resolution: ability to distinguish outcomes
    /// Higher is better
    pub resolution: f64,
    /// Uncertainty: inherent unpredictability of the domain
    /// Not controllable by predictor
    pub uncertainty: f64,
    /// Total Brier score (reliability - resolution + uncertainty)
    pub total: f64,
    /// Calibration curve data points
    pub calibration_curve: Vec<CalibrationPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Midpoint of the confidence bucket
    pub predicted: f64,
    /// Actual observed frequency
    pub actual: f64,
    /// Number of predictions in this bucket
    pub count: usize,
}

impl BrierDecomposition {
    /// Calculate Murphy decomposition
    pub fn calculate(predictions: &[(f64, f64)], num_bins: usize) -> Self {
        if predictions.is_empty() {
            return Self::empty();
        }

        let n = predictions.len() as f64;
        let base_rate: f64 = predictions.iter().map(|(_, o)| *o).sum::<f64>() / n;
        let uncertainty = base_rate * (1.0 - base_rate);

        // Bin predictions
        let mut bins: Vec<Vec<(f64, f64)>> = vec![vec![]; num_bins];
        for &(p, o) in predictions {
            let bin_idx = ((p * num_bins as f64).floor() as usize).min(num_bins - 1);
            bins[bin_idx].push((p, o));
        }

        // Calculate reliability and resolution
        let mut reliability = 0.0;
        let mut resolution = 0.0;
        let mut calibration_curve = Vec::new();

        for (i, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }

            let n_k = bin.len() as f64;
            let f_k: f64 = bin.iter().map(|(p, _)| *p).sum::<f64>() / n_k;
            let o_k: f64 = bin.iter().map(|(_, o)| *o).sum::<f64>() / n_k;

            reliability += n_k * (f_k - o_k).powi(2);
            resolution += n_k * (o_k - base_rate).powi(2);

            calibration_curve.push(CalibrationPoint {
                predicted: (i as f64 + 0.5) / num_bins as f64,
                actual: o_k,
                count: bin.len(),
            });
        }

        reliability /= n;
        resolution /= n;

        Self {
            reliability,
            resolution,
            uncertainty,
            total: reliability - resolution + uncertainty,
            calibration_curve,
        }
    }

    fn empty() -> Self {
        Self {
            reliability: f64::NAN,
            resolution: f64::NAN,
            uncertainty: f64::NAN,
            total: f64::NAN,
            calibration_curve: vec![],
        }
    }
}

// ============================================================================
// EXPECTED CALIBRATION ERROR (ECE)
// ============================================================================

/// Expected Calibration Error - weighted average deviation from perfect calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedCalibrationError {
    /// ECE value (0 = perfect, higher = worse)
    pub ece: f64,
    /// Maximum calibration error across bins
    pub mce: f64,
    /// Number of bins used
    pub num_bins: usize,
    /// Per-bin calibration errors
    pub bin_errors: Vec<BinCalibrationError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinCalibrationError {
    pub bin_lower: f64,
    pub bin_upper: f64,
    pub avg_confidence: f64,
    pub accuracy: f64,
    pub error: f64,
    pub count: usize,
    pub weight: f64,
}

impl ExpectedCalibrationError {
    /// Calculate ECE with specified number of bins
    pub fn calculate(predictions: &[(f64, bool)], num_bins: usize) -> Self {
        if predictions.is_empty() {
            return Self {
                ece: f64::NAN,
                mce: f64::NAN,
                num_bins,
                bin_errors: vec![],
            };
        }

        let n = predictions.len() as f64;
        let bin_width = 1.0 / num_bins as f64;

        let mut bin_errors = Vec::new();
        let mut ece = 0.0;
        let mut mce: f64 = 0.0;

        for bin_idx in 0..num_bins {
            let bin_lower = bin_idx as f64 * bin_width;
            let bin_upper = (bin_idx + 1) as f64 * bin_width;

            let bin_preds: Vec<_> = predictions
                .iter()
                .filter(|(p, _)| *p >= bin_lower && *p < bin_upper)
                .collect();

            if bin_preds.is_empty() {
                continue;
            }

            let count = bin_preds.len();
            let weight = count as f64 / n;
            let avg_confidence: f64 = bin_preds.iter().map(|(p, _)| *p).sum::<f64>() / count as f64;
            let accuracy: f64 =
                bin_preds.iter().map(|(_, o)| if *o { 1.0 } else { 0.0 }).sum::<f64>()
                    / count as f64;
            let error = (avg_confidence - accuracy).abs();

            ece += weight * error;
            mce = mce.max(error);

            bin_errors.push(BinCalibrationError {
                bin_lower,
                bin_upper,
                avg_confidence,
                accuracy,
                error,
                count,
                weight,
            });
        }

        Self {
            ece,
            mce,
            num_bins,
            bin_errors,
        }
    }
}

// ============================================================================
// LOG SCORE AND INFORMATION GAIN
// ============================================================================

/// Logarithmic scoring rule - proper scoring that rewards confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogScore {
    /// Average log score (higher is better, max 0)
    pub score: f64,
    /// Total information gain in bits
    pub total_bits: f64,
    /// Predictions count
    pub n: usize,
}

impl LogScore {
    /// Calculate log score
    pub fn calculate(predictions: &[(f64, bool)]) -> Self {
        if predictions.is_empty() {
            return Self {
                score: f64::NAN,
                total_bits: f64::NAN,
                n: 0,
            };
        }

        let n = predictions.len();
        let epsilon = 1e-15; // Prevent log(0)

        let total_log_score: f64 = predictions
            .iter()
            .map(|(p, outcome)| {
                let p_clamped = p.clamp(epsilon, 1.0 - epsilon);
                if *outcome {
                    p_clamped.ln()
                } else {
                    (1.0 - p_clamped).ln()
                }
            })
            .sum();

        let score = total_log_score / n as f64;
        let total_bits = -total_log_score / 2.0_f64.ln(); // Convert to bits

        Self {
            score,
            total_bits,
            n,
        }
    }
}

/// Information gain relative to a baseline (e.g., base rate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGain {
    /// Bits gained over baseline
    pub bits_gained: f64,
    /// Percentage improvement
    pub relative_improvement: f64,
    /// Baseline log score
    pub baseline_score: f64,
    /// Predictor log score
    pub predictor_score: f64,
}

impl InformationGain {
    /// Calculate information gain vs a baseline probability
    pub fn calculate(predictions: &[(f64, bool)], baseline: f64) -> Self {
        if predictions.is_empty() {
            return Self {
                bits_gained: f64::NAN,
                relative_improvement: f64::NAN,
                baseline_score: f64::NAN,
                predictor_score: f64::NAN,
            };
        }

        let predictor = LogScore::calculate(predictions);

        // Baseline predictions all at the same probability
        let baseline_preds: Vec<_> = predictions.iter().map(|(_, o)| (baseline, *o)).collect();
        let baseline_score = LogScore::calculate(&baseline_preds);

        let bits_gained = baseline_score.total_bits - predictor.total_bits;
        let relative_improvement = if baseline_score.score != 0.0 {
            (predictor.score - baseline_score.score) / baseline_score.score.abs() * 100.0
        } else {
            0.0
        };

        Self {
            bits_gained,
            relative_improvement,
            baseline_score: baseline_score.score,
            predictor_score: predictor.score,
        }
    }
}

// ============================================================================
// STATISTICAL SIGNIFICANCE
// ============================================================================

/// Statistical test results for comparing predictors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    /// Difference in scores (A - B)
    pub score_difference: f64,
    /// Standard error of the difference
    pub standard_error: f64,
    /// Z-score
    pub z_score: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Is difference significant at 0.05 level?
    pub significant_at_05: bool,
    /// Is difference significant at 0.01 level?
    pub significant_at_01: bool,
    /// 95% confidence interval for the difference
    pub confidence_interval: (f64, f64),
}

impl StatisticalComparison {
    /// Compare two sets of Brier scores using paired difference test
    pub fn compare_brier_scores(
        predictions_a: &[(f64, f64)],
        predictions_b: &[(f64, f64)],
    ) -> Self {
        if predictions_a.len() != predictions_b.len() || predictions_a.is_empty() {
            return Self::invalid();
        }

        let n = predictions_a.len();

        // Calculate paired differences in Brier scores
        let differences: Vec<f64> = predictions_a
            .iter()
            .zip(predictions_b.iter())
            .map(|((pa, oa), (pb, _ob))| {
                let brier_a = (pa - oa).powi(2);
                let brier_b = (pb - oa).powi(2); // Use same outcome
                brier_a - brier_b
            })
            .collect();

        let mean_diff: f64 = differences.iter().sum::<f64>() / n as f64;
        let variance: f64 = differences
            .iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1).max(1) as f64;
        let standard_error = (variance / n as f64).sqrt();

        let z_score = if standard_error > 0.0 {
            mean_diff / standard_error
        } else {
            0.0
        };

        // Approximate p-value from z-score
        let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

        let z_95 = 1.96;
        let ci_lower = mean_diff - z_95 * standard_error;
        let ci_upper = mean_diff + z_95 * standard_error;

        Self {
            score_difference: mean_diff,
            standard_error,
            z_score,
            p_value,
            significant_at_05: p_value < 0.05,
            significant_at_01: p_value < 0.01,
            confidence_interval: (ci_lower, ci_upper),
        }
    }

    fn invalid() -> Self {
        Self {
            score_difference: f64::NAN,
            standard_error: f64::NAN,
            z_score: f64::NAN,
            p_value: f64::NAN,
            significant_at_05: false,
            significant_at_01: false,
            confidence_interval: (f64::NAN, f64::NAN),
        }
    }
}

/// Approximate normal CDF using Abramowitz and Stegun approximation
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

// ============================================================================
// AGGREGATION METRICS
// ============================================================================

/// Metrics for evaluating prediction aggregation quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationMetrics {
    /// Brier score of the aggregate
    pub aggregate_brier: BrierScore,
    /// Average individual Brier score
    pub avg_individual_brier: f64,
    /// Best individual Brier score
    pub best_individual_brier: f64,
    /// Worst individual Brier score
    pub worst_individual_brier: f64,
    /// "Wisdom of crowd" ratio: individual/aggregate
    pub wisdom_ratio: f64,
    /// Beat rate: % of individuals worse than aggregate
    pub beat_rate: f64,
    /// Diversity bonus: benefit from aggregating diverse views
    pub diversity_bonus: f64,
}

impl AggregationMetrics {
    /// Calculate aggregation effectiveness metrics
    pub fn calculate(
        individual_predictions: &[Vec<(f64, f64)>],
        aggregate_predictions: &[(f64, f64)],
    ) -> Self {
        if individual_predictions.is_empty() || aggregate_predictions.is_empty() {
            return Self::empty();
        }

        let aggregate_brier = BrierScore::calculate(aggregate_predictions);

        let individual_briers: Vec<f64> = individual_predictions
            .iter()
            .map(|preds| BrierScore::calculate(preds).score)
            .collect();

        let avg_individual = individual_briers.iter().sum::<f64>() / individual_briers.len() as f64;
        let best_individual = individual_briers
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let worst_individual = individual_briers
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let wisdom_ratio = if aggregate_brier.score > 0.0 {
            avg_individual / aggregate_brier.score
        } else {
            0.0
        };

        let beat_count = individual_briers
            .iter()
            .filter(|&&b| b > aggregate_brier.score)
            .count();
        let beat_rate = beat_count as f64 / individual_briers.len() as f64 * 100.0;

        // Diversity bonus: how much better than expected from average quality
        let diversity_bonus = avg_individual - aggregate_brier.score;

        Self {
            aggregate_brier,
            avg_individual_brier: avg_individual,
            best_individual_brier: best_individual,
            worst_individual_brier: worst_individual,
            wisdom_ratio,
            beat_rate,
            diversity_bonus,
        }
    }

    fn empty() -> Self {
        Self {
            aggregate_brier: BrierScore {
                score: f64::NAN,
                n: 0,
                standard_error: f64::NAN,
                confidence_interval: (f64::NAN, f64::NAN),
            },
            avg_individual_brier: f64::NAN,
            best_individual_brier: f64::NAN,
            worst_individual_brier: f64::NAN,
            wisdom_ratio: f64::NAN,
            beat_rate: f64::NAN,
            diversity_bonus: f64::NAN,
        }
    }
}

// ============================================================================
// TIME SERIES METRICS
// ============================================================================

/// Metrics for tracking performance over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetrics {
    /// Rolling Brier scores over time
    pub rolling_brier: Vec<TimestampedValue>,
    /// Rolling ECE over time
    pub rolling_ece: Vec<TimestampedValue>,
    /// Trend direction (-1, 0, 1)
    pub trend: i32,
    /// Slope of improvement/degradation
    pub slope: f64,
    /// Is performance improving significantly?
    pub improving: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedValue {
    pub timestamp: u64,
    pub value: f64,
    pub sample_size: usize,
}

impl TimeSeriesMetrics {
    /// Calculate rolling metrics with specified window size
    pub fn calculate(
        predictions: &[(u64, f64, f64)], // (timestamp, prediction, outcome)
        window_size: usize,
    ) -> Self {
        if predictions.len() < window_size {
            return Self::empty();
        }

        let mut sorted = predictions.to_vec();
        sorted.sort_by_key(|(t, _, _)| *t);

        let mut rolling_brier = Vec::new();

        for window in sorted.windows(window_size) {
            let preds: Vec<(f64, f64)> = window.iter().map(|(_, p, o)| (*p, *o)).collect();
            let brier = BrierScore::calculate(&preds);
            let timestamp = window.last().map(|(t, _, _)| *t).unwrap_or(0);

            rolling_brier.push(TimestampedValue {
                timestamp,
                value: brier.score,
                sample_size: window.len(),
            });
        }

        // Calculate trend using linear regression
        let (slope, improving) = if rolling_brier.len() >= 2 {
            let n = rolling_brier.len() as f64;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean: f64 = rolling_brier.iter().map(|p| p.value).sum::<f64>() / n;

            let numerator: f64 = rolling_brier
                .iter()
                .enumerate()
                .map(|(i, p)| (i as f64 - x_mean) * (p.value - y_mean))
                .sum();
            let denominator: f64 = rolling_brier
                .iter()
                .enumerate()
                .map(|(i, _)| (i as f64 - x_mean).powi(2))
                .sum();

            let slope = if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            };

            (slope, slope < -0.001) // Negative slope = improving (lower Brier is better)
        } else {
            (0.0, false)
        };

        let trend = if slope < -0.001 {
            1 // Improving
        } else if slope > 0.001 {
            -1 // Degrading
        } else {
            0 // Stable
        };

        Self {
            rolling_brier,
            rolling_ece: vec![], // Would calculate similarly
            trend,
            slope,
            improving,
        }
    }

    fn empty() -> Self {
        Self {
            rolling_brier: vec![],
            rolling_ece: vec![],
            trend: 0,
            slope: 0.0,
            improving: false,
        }
    }
}

// ============================================================================
// DOMAIN-SPECIFIC METRICS
// ============================================================================

/// Metrics breakdown by domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMetrics {
    pub domain: String,
    pub brier: BrierScore,
    pub ece: ExpectedCalibrationError,
    pub log_score: LogScore,
    pub prediction_count: usize,
    pub avg_confidence: f64,
    pub accuracy: f64,
}

impl DomainMetrics {
    /// Calculate all metrics for a domain
    pub fn calculate(domain: &str, predictions: &[(f64, bool)]) -> Self {
        if predictions.is_empty() {
            return Self::empty(domain);
        }

        let brier_preds: Vec<(f64, f64)> = predictions
            .iter()
            .map(|(p, o)| (*p, if *o { 1.0 } else { 0.0 }))
            .collect();

        let brier = BrierScore::calculate(&brier_preds);
        let ece = ExpectedCalibrationError::calculate(predictions, 10);
        let log_score = LogScore::calculate(predictions);

        let avg_confidence: f64 = predictions.iter().map(|(p, _)| *p).sum::<f64>()
            / predictions.len() as f64;
        let accuracy: f64 = predictions
            .iter()
            .map(|(_, o)| if *o { 1.0 } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;

        Self {
            domain: domain.to_string(),
            brier,
            ece,
            log_score,
            prediction_count: predictions.len(),
            avg_confidence,
            accuracy,
        }
    }

    fn empty(domain: &str) -> Self {
        Self {
            domain: domain.to_string(),
            brier: BrierScore {
                score: f64::NAN,
                n: 0,
                standard_error: f64::NAN,
                confidence_interval: (f64::NAN, f64::NAN),
            },
            ece: ExpectedCalibrationError {
                ece: f64::NAN,
                mce: f64::NAN,
                num_bins: 0,
                bin_errors: vec![],
            },
            log_score: LogScore {
                score: f64::NAN,
                total_bits: f64::NAN,
                n: 0,
            },
            prediction_count: 0,
            avg_confidence: f64::NAN,
            accuracy: f64::NAN,
        }
    }
}

/// Cross-domain transfer analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferAnalysis {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Correlation between performance in source and target
    pub performance_correlation: f64,
    /// Transfer coefficient: how well calibration transfers
    pub transfer_coefficient: f64,
    /// Does training in source help target?
    pub positive_transfer: bool,
}

impl TransferAnalysis {
    /// Analyze transfer between domains
    pub fn calculate(
        source_domain: &str,
        target_domain: &str,
        predictors: &[PredictorDomainPerformance],
    ) -> Self {
        if predictors.len() < 3 {
            return Self::empty(source_domain, target_domain);
        }

        let source_scores: Vec<f64> = predictors.iter().map(|p| p.source_brier).collect();
        let target_scores: Vec<f64> = predictors.iter().map(|p| p.target_brier).collect();

        let correlation = pearson_correlation(&source_scores, &target_scores);

        // Transfer coefficient: normalized correlation accounting for domain difficulty
        let source_mean: f64 = source_scores.iter().sum::<f64>() / source_scores.len() as f64;
        let target_mean: f64 = target_scores.iter().sum::<f64>() / target_scores.len() as f64;
        let transfer_coefficient = if target_mean > 0.0 {
            correlation * (source_mean / target_mean).sqrt()
        } else {
            0.0
        };

        Self {
            source_domain: source_domain.to_string(),
            target_domain: target_domain.to_string(),
            performance_correlation: correlation,
            transfer_coefficient,
            positive_transfer: correlation > 0.3,
        }
    }

    fn empty(source: &str, target: &str) -> Self {
        Self {
            source_domain: source.to_string(),
            target_domain: target.to_string(),
            performance_correlation: f64::NAN,
            transfer_coefficient: f64::NAN,
            positive_transfer: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorDomainPerformance {
    pub predictor_id: String,
    pub source_brier: f64,
    pub target_brier: f64,
}

/// Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }

    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum();

    let x_var: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
    let y_var: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    if x_var * y_var <= 0.0 {
        return 0.0;
    }

    numerator / (x_var * y_var).sqrt()
}

// ============================================================================
// EXPERIMENT SUMMARY
// ============================================================================

/// Comprehensive summary of an experiment's metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub experiment_id: String,
    pub started_at: u64,
    pub ended_at: Option<u64>,
    pub total_predictions: usize,
    pub total_participants: usize,

    /// Overall metrics
    pub overall_brier: BrierScore,
    pub overall_ece: ExpectedCalibrationError,
    pub overall_log_score: LogScore,

    /// Decomposition
    pub brier_decomposition: BrierDecomposition,

    /// By-domain breakdown
    pub domain_metrics: HashMap<String, DomainMetrics>,

    /// Time series
    pub time_series: TimeSeriesMetrics,

    /// Comparisons (if treatment/control groups)
    pub group_comparisons: HashMap<String, StatisticalComparison>,
}

impl ExperimentSummary {
    /// Create a new experiment summary
    pub fn new(experiment_id: &str) -> Self {
        Self {
            experiment_id: experiment_id.to_string(),
            started_at: 0,
            ended_at: None,
            total_predictions: 0,
            total_participants: 0,
            overall_brier: BrierScore {
                score: 0.0,
                n: 0,
                standard_error: 0.0,
                confidence_interval: (0.0, 0.0),
            },
            overall_ece: ExpectedCalibrationError {
                ece: 0.0,
                mce: 0.0,
                num_bins: 10,
                bin_errors: vec![],
            },
            overall_log_score: LogScore {
                score: 0.0,
                total_bits: 0.0,
                n: 0,
            },
            brier_decomposition: BrierDecomposition {
                reliability: 0.0,
                resolution: 0.0,
                uncertainty: 0.0,
                total: 0.0,
                calibration_curve: vec![],
            },
            domain_metrics: HashMap::new(),
            time_series: TimeSeriesMetrics::empty(),
            group_comparisons: HashMap::new(),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brier_score_perfect() {
        let predictions = vec![(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)];
        let brier = BrierScore::calculate(&predictions);
        assert!((brier.score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_brier_score_worst() {
        let predictions = vec![(0.0, 1.0), (1.0, 0.0)];
        let brier = BrierScore::calculate(&predictions);
        assert!((brier.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brier_score_random() {
        // 50% predictions on balanced outcomes should give ~0.25
        let predictions = vec![(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)];
        let brier = BrierScore::calculate(&predictions);
        assert!((brier.score - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_ece_perfect_calibration() {
        // Predictions that match actual outcomes perfectly
        let predictions = vec![
            (0.1, false),
            (0.1, false),
            (0.9, true),
            (0.9, true),
        ];
        let ece = ExpectedCalibrationError::calculate(&predictions, 10);
        // Should be close to 0
        assert!(ece.ece < 0.2);
    }

    #[test]
    fn test_log_score() {
        let predictions = vec![(0.9, true), (0.1, false)];
        let log = LogScore::calculate(&predictions);
        // Score should be negative (log of probability < 1)
        assert!(log.score < 0.0);
        // But close to 0 for good predictions
        assert!(log.score > -0.2);
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // Perfect negative correlation
        let r = pearson_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_statistical_comparison() {
        // Two predictors with same outcomes but different predictions
        let predictions_a = vec![(0.6, 1.0), (0.4, 0.0), (0.7, 1.0), (0.3, 0.0)];
        let predictions_b = vec![(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)];

        let comparison = StatisticalComparison::compare_brier_scores(&predictions_a, &predictions_b);
        // A should have lower Brier (better calibrated)
        assert!(comparison.score_difference < 0.0);
    }
}
