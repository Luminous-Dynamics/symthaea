// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain performance prediction: pairwise correlations and linear
//! regression models between benchmark metrics.
//!
//! Given multiple seeds/configurations (multi-report) or block-level variance
//! within a single report, this module computes pairwise Pearson correlations
//! between theoretically motivated benchmark pairs and fits simple OLS
//! regression models for cross-domain prediction.
//!
//! Key theoretical pairs are drawn from shared cognitive mechanisms
//! (executive control, working memory, inhibition, social cognition, etc.).

use serde::{Deserialize, Serialize};

use super::analysis::normal_cdf;
use super::report::{BenchmarkReport, key_metric_for_benchmark};
use super::trial_analysis::TrialOutcome;

// ────────────────────────────────────────────────────────────────────
// Shared mechanism classification
// ────────────────────────────────────────────────────────────────────

/// Strength of a shared cognitive mechanism between two benchmark metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharedMechanism {
    /// |r| > 0.5 — strong evidence of a common underlying process.
    Strong,
    /// 0.3 < |r| <= 0.5 — moderate evidence of overlap.
    Moderate,
    /// 0.1 < |r| <= 0.3 — weak shared variance.
    Weak,
    /// |r| <= 0.1 — effectively independent.
    Independent,
}

impl SharedMechanism {
    /// Classify from an absolute Pearson r value.
    pub fn from_r(r: f64) -> Self {
        let abs_r = r.abs();
        if abs_r > 0.5 {
            SharedMechanism::Strong
        } else if abs_r > 0.3 {
            SharedMechanism::Moderate
        } else if abs_r > 0.1 {
            SharedMechanism::Weak
        } else {
            SharedMechanism::Independent
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Domain correlation
// ────────────────────────────────────────────────────────────────────

/// Correlation between two benchmark metrics across multiple seeds/configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCorrelation {
    /// First benchmark name (e.g., "Executive::Stroop").
    pub domain_a: String,
    /// Key metric from first benchmark.
    pub metric_a: String,
    /// Second benchmark name (e.g., "Executive::WCST").
    pub domain_b: String,
    /// Key metric from second benchmark.
    pub metric_b: String,
    /// Pearson r correlation coefficient.
    pub r: f64,
    /// Number of paired observations.
    pub n: usize,
    /// Two-tailed p-value.
    pub p_value: f64,
    /// Inferred shared mechanism strength.
    pub shared_mechanism: SharedMechanism,
}

// ────────────────────────────────────────────────────────────────────
// Prediction model (OLS linear regression)
// ────────────────────────────────────────────────────────────────────

/// Simple linear regression model: target = slope * predictor + intercept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModel {
    /// Predictor benchmark::metric.
    pub predictor: String,
    /// Target benchmark::metric.
    pub target: String,
    /// Regression slope.
    pub slope: f64,
    /// Regression intercept.
    pub intercept: f64,
    /// Coefficient of determination (R^2).
    pub r_squared: f64,
    /// Root mean squared error.
    pub rmse: f64,
}

// ────────────────────────────────────────────────────────────────────
// Statistical helpers
// ────────────────────────────────────────────────────────────────────

/// Compute Pearson r between two vectors, returning (r, n).
fn pearson_r(x: &[f64], y: &[f64]) -> (f64, usize) {
    let n = x.len().min(y.len());
    let r = super::reliability_analysis::pearson_r(x, y);
    (r, n)
}

/// Approximate two-tailed p-value from Pearson r and sample size.
fn p_value_from_r(r: f64, n: usize) -> f64 {
    if n < 3 || (r.abs() - 1.0).abs() < 1e-10 {
        return 1.0;
    }

    let df = n as f64 - 2.0;
    let t = r * (df / (1.0 - r * r)).sqrt();

    // Normal approximation for p (valid when df > ~20, reasonable for smaller).
    let z = t.abs();
    let p = 2.0 * (1.0 - normal_cdf(z));
    p.clamp(0.0, 1.0)
}

/// Fit an ordinary least squares linear regression: y = slope * x + intercept.
///
/// Returns a `PredictionModel` with R^2 and RMSE. Requires at least 2 paired
/// values; returns a zero model otherwise.
pub fn fit_linear(x: &[f64], y: &[f64]) -> PredictionModel {
    let n = x.len().min(y.len());
    if n < 2 {
        return PredictionModel {
            predictor: String::new(),
            target: String::new(),
            slope: 0.0,
            intercept: 0.0,
            r_squared: 0.0,
            rmse: 0.0,
        };
    }

    let nf = n as f64;
    let mean_x = x[..n].iter().sum::<f64>() / nf;
    let mean_y = y[..n].iter().sum::<f64>() / nf;

    let mut ss_xy = 0.0f64;
    let mut ss_xx = 0.0f64;
    let mut ss_yy = 0.0f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    let slope = if ss_xx.abs() > 1e-15 {
        ss_xy / ss_xx
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;

    // R^2 = (SS_xy)^2 / (SS_xx * SS_yy)
    let r_squared = if ss_xx.abs() > 1e-15 && ss_yy.abs() > 1e-15 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        0.0
    };

    // RMSE
    let mut sse = 0.0f64;
    for i in 0..n {
        let predicted = slope * x[i] + intercept;
        let residual = y[i] - predicted;
        sse += residual * residual;
    }
    let rmse = (sse / nf).sqrt();

    PredictionModel {
        predictor: String::new(),
        target: String::new(),
        slope,
        intercept,
        r_squared,
        rmse,
    }
}

// ────────────────────────────────────────────────────────────────────
// Theoretically motivated benchmark pairs
// ────────────────────────────────────────────────────────────────────

/// The 10 theoretically motivated benchmark pairs to correlate.
///
/// Each tuple is (benchmark_a, benchmark_b, theoretical_label).
const KEY_PAIRS: [(&str, &str, &str); 10] = [
    // 1. Shared executive control (conflict monitoring)
    ("Executive::Stroop", "Executive::WCST", "executive_control"),
    // 2. Shared working memory capacity
    ("WorM::N-back", "WorM::DigitSpan", "working_memory"),
    // 3. Shared set-shifting / cognitive flexibility
    (
        "Executive::WCST",
        "CogBench::ReversalLearning",
        "set_shifting",
    ),
    // 4. Shared response inhibition
    ("Inhibition::GoNoGo", "Inhibition::StopSignal", "inhibition"),
    // 5. Shared social cognition / mentalizing
    ("Social::RME", "ToMBench::FalseBelief", "social_cognition"),
    // 6. Shared attentional resources
    (
        "Attention::VisualSearch",
        "SustainedAttention::CPT",
        "attention",
    ),
    // 7. Shared lexical processing
    (
        "Language::LexicalDecision",
        "Language::SemanticPriming",
        "lexical_processing",
    ),
    // 8. Shared motor control
    ("Motor::FittsLaw", "Motor::Bimanual", "motor_control"),
    // 9. Shared metacognitive monitoring
    (
        "Metacognition::Calibration",
        "Metacognition::FeelingOfKnowing",
        "metacognition",
    ),
    // 10. Shared fluid intelligence
    (
        "Reasoning::ArcFluid",
        "Executive::Ravens",
        "fluid_intelligence",
    ),
];

// ────────────────────────────────────────────────────────────────────
// Cross-domain matrix
// ────────────────────────────────────────────────────────────────────

/// Cross-domain correlation and prediction matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainMatrix {
    /// Pairwise correlations between benchmark metrics.
    pub correlations: Vec<DomainCorrelation>,
    /// Linear prediction models (one per pair with sufficient data).
    pub predictions: Vec<PredictionModel>,
}

impl CrossDomainMatrix {
    /// Build cross-domain analysis from multiple seed/configuration reports.
    ///
    /// Each report is assumed to represent an independent run (different seed
    /// or configuration). For each theoretically motivated pair, we extract
    /// the key metric from both benchmarks across all reports and compute
    /// a pairwise Pearson correlation and OLS prediction model.
    pub fn from_multi_seed_reports(reports: &[BenchmarkReport]) -> Self {
        let mut correlations = Vec::new();
        let mut predictions = Vec::new();

        for &(bench_a, bench_b, _label) in &KEY_PAIRS {
            let metric_a_name = key_metric_for_benchmark(bench_a);
            let metric_b_name = key_metric_for_benchmark(bench_b);

            // Collect paired observations across reports.
            let mut vals_a = Vec::new();
            let mut vals_b = Vec::new();

            for report in reports {
                let va = extract_metric(report, bench_a, metric_a_name);
                let vb = extract_metric(report, bench_b, metric_b_name);
                if let (Some(a), Some(b)) = (va, vb) {
                    vals_a.push(a);
                    vals_b.push(b);
                }
            }

            if vals_a.len() >= 3 {
                let (r, n) = pearson_r(&vals_a, &vals_b);
                let p = p_value_from_r(r, n);
                correlations.push(DomainCorrelation {
                    domain_a: bench_a.to_string(),
                    metric_a: metric_a_name.to_string(),
                    domain_b: bench_b.to_string(),
                    metric_b: metric_b_name.to_string(),
                    r,
                    n,
                    p_value: p,
                    shared_mechanism: SharedMechanism::from_r(r),
                });

                let mut model = fit_linear(&vals_a, &vals_b);
                model.predictor = format!("{}::{}", bench_a, metric_a_name);
                model.target = format!("{}::{}", bench_b, metric_b_name);
                predictions.push(model);
            }
        }

        Self {
            correlations,
            predictions,
        }
    }

    /// Build cross-domain analysis from a single report using trial-level data.
    ///
    /// Splits each benchmark's trial trace into blocks and computes per-block
    /// accuracy, then correlates block-level scores between benchmark pairs.
    /// This enables within-subject analysis without needing multiple runs.
    pub fn from_single_report(report: &BenchmarkReport) -> Self {
        let block_size = 10;
        let mut correlations = Vec::new();
        let mut predictions = Vec::new();

        for &(bench_a, bench_b, _label) in &KEY_PAIRS {
            let metric_a_name = key_metric_for_benchmark(bench_a);
            let metric_b_name = key_metric_for_benchmark(bench_b);

            let result_a = report.results.iter().find(|r| r.benchmark == bench_a);
            let result_b = report.results.iter().find(|r| r.benchmark == bench_b);

            let (ra, rb) = match (result_a, result_b) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            // Extract block-level scores from trial traces.
            let blocks_a = block_scores(&ra.trial_trace, block_size);
            let blocks_b = block_scores(&rb.trial_trace, block_size);

            // Use the shorter length for pairing.
            let n_blocks = blocks_a.len().min(blocks_b.len());
            if n_blocks < 3 {
                // Fallback: if no trial traces, try using the metric mean
                // (will yield a single point, so skip correlation).
                continue;
            }

            let (r, n) = pearson_r(&blocks_a[..n_blocks], &blocks_b[..n_blocks]);
            let p = p_value_from_r(r, n);
            correlations.push(DomainCorrelation {
                domain_a: bench_a.to_string(),
                metric_a: metric_a_name.to_string(),
                domain_b: bench_b.to_string(),
                metric_b: metric_b_name.to_string(),
                r,
                n,
                p_value: p,
                shared_mechanism: SharedMechanism::from_r(r),
            });

            let mut model = fit_linear(&blocks_a[..n_blocks], &blocks_b[..n_blocks]);
            model.predictor = format!("{}::{}", bench_a, metric_a_name);
            model.target = format!("{}::{}", bench_b, metric_b_name);
            predictions.push(model);
        }

        Self {
            correlations,
            predictions,
        }
    }

    /// Filter correlations to those with moderate or strong shared mechanisms.
    pub fn shared_mechanisms(&self) -> Vec<&DomainCorrelation> {
        self.correlations
            .iter()
            .filter(|c| {
                matches!(
                    c.shared_mechanism,
                    SharedMechanism::Strong | SharedMechanism::Moderate
                )
            })
            .collect()
    }

    /// Render the matrix as a Markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        // Correlation matrix table.
        out.push_str("## Cross-Domain Correlation Matrix\n\n");
        out.push_str("| Domain A | Metric A | Domain B | Metric B | r | N | p | Mechanism |\n");
        out.push_str(
            "|----------|----------|----------|----------|------|------|------|----------|\n",
        );
        for c in &self.correlations {
            let mech_str = match c.shared_mechanism {
                SharedMechanism::Strong => "Strong",
                SharedMechanism::Moderate => "Moderate",
                SharedMechanism::Weak => "Weak",
                SharedMechanism::Independent => "Independent",
            };
            let sig = if c.p_value < 0.05 { "*" } else { "" };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.3}{} | {} | {:.3} | {} |\n",
                c.domain_a, c.metric_a, c.domain_b, c.metric_b, c.r, sig, c.n, c.p_value, mech_str,
            ));
        }

        // Prediction model summaries.
        if !self.predictions.is_empty() {
            out.push_str("\n## Prediction Models\n\n");
            out.push_str("| Predictor | Target | Slope | Intercept | R^2 | RMSE |\n");
            out.push_str("|-----------|--------|-------|-----------|-----|------|\n");
            for m in &self.predictions {
                out.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.3} | {:.4} |\n",
                    m.predictor, m.target, m.slope, m.intercept, m.r_squared, m.rmse,
                ));
            }
        }

        // Shared mechanisms summary.
        let shared = self.shared_mechanisms();
        if !shared.is_empty() {
            out.push_str("\n## Shared Mechanisms (Moderate+)\n\n");
            for c in &shared {
                out.push_str(&format!(
                    "- {} x {} : r={:.3}, {:?}\n",
                    c.domain_a, c.domain_b, c.r, c.shared_mechanism,
                ));
            }
        }

        out
    }
}

// ────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────

/// Extract a metric value from a report for a given benchmark and metric key.
fn extract_metric(report: &BenchmarkReport, benchmark: &str, metric: &str) -> Option<f64> {
    report
        .results
        .iter()
        .find(|r| r.benchmark == benchmark)
        .and_then(|r| r.metrics.get(metric))
        .map(|mv| mv.mean)
}

/// Compute per-block accuracy scores from a trial trace.
fn block_scores(trace: &[TrialOutcome], block_size: usize) -> Vec<f64> {
    if trace.is_empty() || block_size == 0 {
        return Vec::new();
    }
    trace
        .chunks(block_size)
        .map(|block| {
            let correct = block.iter().filter(|t| t.correct).count();
            correct as f64 / block.len() as f64
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::report::{BenchmarkResult, MetricValue};
    use std::collections::BTreeMap;

    // ── Helpers ──

    fn make_result(name: &str, metric: &str, value: f64) -> BenchmarkResult {
        let mut r = BenchmarkResult::new(name, None);
        r.insert(metric, MetricValue::from_samples(&[value]));
        r
    }

    fn make_result_with_trace(
        name: &str,
        metric: &str,
        value: f64,
        n_trials: usize,
        accuracy: f64,
    ) -> BenchmarkResult {
        let mut r = BenchmarkResult::new(name, None);
        r.insert(metric, MetricValue::from_samples(&[value]));
        // Generate a trial trace with the given accuracy rate.
        let mut trace = Vec::new();
        for i in 0..n_trials {
            let correct = (i as f64 / n_trials as f64) < accuracy;
            trace.push(TrialOutcome {
                trial_idx: i,
                condition: "default".to_string(),
                correct,
                rt_ticks: 5.0,
                similarity: if correct { 0.9 } else { 0.3 },
                confidence: if correct { 0.8 } else { 0.4 },
                response_idx: 0,
                extra: BTreeMap::new(),
            });
        }
        r.trial_trace = trace;
        r
    }

    fn make_report_with_benchmarks(pairs: &[(&str, &str, f64)]) -> BenchmarkReport {
        let mut report = BenchmarkReport {
            results: Vec::new(),
            timestamp: "2026-02-27T00:00:00Z".to_string(),
        };
        for &(bench, metric, value) in pairs {
            report.results.push(make_result(bench, metric, value));
        }
        report
    }

    // ── test_pearson_r_basic ──

    #[test]
    fn test_pearson_r_basic() {
        // Perfect positive correlation.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (r, n) = pearson_r(&x, &y);
        assert_eq!(n, 5);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "expected r=1.0 for perfect positive, got r={}",
            r
        );

        // Perfect negative correlation.
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let (r_neg, _) = pearson_r(&x, &y_neg);
        assert!(
            (r_neg + 1.0).abs() < 1e-10,
            "expected r=-1.0 for perfect negative, got r={}",
            r_neg
        );

        // Zero correlation (constant y).
        let y_const = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (r_zero, _) = pearson_r(&x, &y_const);
        assert!(
            r_zero.abs() < 1e-10,
            "expected r=0 for constant y, got r={}",
            r_zero
        );

        // Too few points.
        let (r_small, n_small) = pearson_r(&[1.0], &[2.0]);
        assert_eq!(n_small, 1);
        assert!((r_small).abs() < 1e-10, "expected 0.0 for n<3");
    }

    // ── test_fit_linear_perfect ──

    #[test]
    fn test_fit_linear_perfect() {
        // y = 2x + 1 exactly.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let model = fit_linear(&x, &y);

        assert!(
            (model.slope - 2.0).abs() < 1e-10,
            "expected slope=2.0, got {}",
            model.slope
        );
        assert!(
            (model.intercept - 1.0).abs() < 1e-10,
            "expected intercept=1.0, got {}",
            model.intercept
        );
        assert!(
            (model.r_squared - 1.0).abs() < 1e-10,
            "expected R^2=1.0, got {}",
            model.r_squared
        );
        assert!(
            model.rmse < 1e-10,
            "expected RMSE~0 for perfect fit, got {}",
            model.rmse
        );
    }

    // ── test_shared_mechanism_classification ──

    #[test]
    fn test_shared_mechanism_classification() {
        assert_eq!(SharedMechanism::from_r(0.0), SharedMechanism::Independent);
        assert_eq!(SharedMechanism::from_r(0.05), SharedMechanism::Independent);
        assert_eq!(SharedMechanism::from_r(0.1), SharedMechanism::Independent);
        assert_eq!(SharedMechanism::from_r(0.15), SharedMechanism::Weak);
        assert_eq!(SharedMechanism::from_r(0.3), SharedMechanism::Weak);
        assert_eq!(SharedMechanism::from_r(0.35), SharedMechanism::Moderate);
        assert_eq!(SharedMechanism::from_r(0.5), SharedMechanism::Moderate);
        assert_eq!(SharedMechanism::from_r(0.55), SharedMechanism::Strong);
        assert_eq!(SharedMechanism::from_r(0.9), SharedMechanism::Strong);
        // Negative values use absolute magnitude.
        assert_eq!(SharedMechanism::from_r(-0.6), SharedMechanism::Strong);
        assert_eq!(SharedMechanism::from_r(-0.4), SharedMechanism::Moderate);
        assert_eq!(SharedMechanism::from_r(-0.2), SharedMechanism::Weak);
        assert_eq!(SharedMechanism::from_r(-0.05), SharedMechanism::Independent);
    }

    // ── test_cross_domain_from_report ──

    #[test]
    fn test_cross_domain_from_report() {
        // Build 5 reports with correlated Stroop and WCST scores.
        let reports: Vec<BenchmarkReport> = (0..5)
            .map(|i| {
                let base = 0.5 + i as f64 * 0.1; // 0.5, 0.6, 0.7, 0.8, 0.9
                make_report_with_benchmarks(&[
                    ("Executive::Stroop", "incongruent_accuracy", base),
                    ("Executive::WCST", "categories_completed", base * 0.9 + 0.05),
                    ("WorM::N-back", "nback_2::accuracy", base * 0.8),
                    ("WorM::DigitSpan", "forward_span", 4.0 + i as f64),
                ])
            })
            .collect();

        let matrix = CrossDomainMatrix::from_multi_seed_reports(&reports);

        // Should have correlations for the pairs present in the data.
        assert!(
            !matrix.correlations.is_empty(),
            "expected at least one correlation from multi-seed reports"
        );

        // Executive::Stroop x Executive::WCST should be present.
        let stroop_wcst = matrix
            .correlations
            .iter()
            .find(|c| c.domain_a == "Executive::Stroop" && c.domain_b == "Executive::WCST");
        assert!(
            stroop_wcst.is_some(),
            "Stroop x WCST correlation should be computed"
        );
        let c = stroop_wcst.unwrap();
        assert_eq!(c.n, 5);
        assert!(c.r.is_finite());
        // These are perfectly correlated by construction.
        assert!(
            c.r > 0.9,
            "expected strong positive r for linearly related scores, got r={}",
            c.r
        );

        // WorM::N-back x WorM::DigitSpan should be present.
        let nback_digit = matrix
            .correlations
            .iter()
            .find(|c| c.domain_a == "WorM::N-back" && c.domain_b == "WorM::DigitSpan");
        assert!(
            nback_digit.is_some(),
            "N-back x DigitSpan correlation should be computed"
        );

        // Prediction models should match correlations.
        assert_eq!(
            matrix.predictions.len(),
            matrix.correlations.len(),
            "each correlation should have a prediction model"
        );
    }

    // ── test_prediction_model_fit ──

    #[test]
    fn test_prediction_model_fit() {
        // y = 0.5x + 0.25 with some noise.
        let x = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = vec![0.25, 0.35, 0.45, 0.55, 0.65, 0.75];
        let model = fit_linear(&x, &y);

        assert!(
            (model.slope - 0.5).abs() < 1e-10,
            "expected slope=0.5, got {}",
            model.slope
        );
        assert!(
            (model.intercept - 0.25).abs() < 1e-10,
            "expected intercept=0.25, got {}",
            model.intercept
        );
        assert!(
            (model.r_squared - 1.0).abs() < 1e-10,
            "R^2 should be 1.0 for perfect linear data, got {}",
            model.r_squared
        );

        // With noise: R^2 < 1.
        let y_noisy = vec![0.30, 0.32, 0.50, 0.48, 0.70, 0.72];
        let model_noisy = fit_linear(&x, &y_noisy);
        assert!(
            model_noisy.r_squared < 1.0 && model_noisy.r_squared > 0.5,
            "noisy data R^2 should be moderate, got {}",
            model_noisy.r_squared
        );
        assert!(
            model_noisy.rmse > 0.0,
            "noisy data RMSE should be positive, got {}",
            model_noisy.rmse
        );
    }

    // ── test_markdown_output ──

    #[test]
    fn test_markdown_output() {
        // Build a minimal matrix manually.
        let matrix = CrossDomainMatrix {
            correlations: vec![DomainCorrelation {
                domain_a: "Executive::Stroop".to_string(),
                metric_a: "incongruent_accuracy".to_string(),
                domain_b: "Executive::WCST".to_string(),
                metric_b: "accuracy".to_string(),
                r: 0.72,
                n: 10,
                p_value: 0.02,
                shared_mechanism: SharedMechanism::Strong,
            }],
            predictions: vec![PredictionModel {
                predictor: "Executive::Stroop::incongruent_accuracy".to_string(),
                target: "Executive::WCST::accuracy".to_string(),
                slope: 0.85,
                intercept: 0.10,
                r_squared: 0.52,
                rmse: 0.05,
            }],
        };

        let md = matrix.to_markdown();

        // Check that all sections are present.
        assert!(
            md.contains("Cross-Domain Correlation Matrix"),
            "missing correlation table header"
        );
        assert!(
            md.contains("Prediction Models"),
            "missing prediction models header"
        );
        assert!(
            md.contains("Shared Mechanisms"),
            "missing shared mechanisms header"
        );

        // Check that data values appear.
        assert!(md.contains("Executive::Stroop"), "missing domain_a");
        assert!(md.contains("Executive::WCST"), "missing domain_b");
        assert!(md.contains("0.720"), "missing r value");
        assert!(md.contains("Strong"), "missing mechanism label");
        assert!(md.contains("0.8500"), "missing slope value");
        assert!(md.contains("0.0500"), "missing RMSE value");
    }

    // ── test_single_report_with_traces ──

    #[test]
    fn test_single_report_with_traces() {
        // Build a report with trial traces for two benchmarks.
        let mut report = BenchmarkReport {
            results: Vec::new(),
            timestamp: "2026-02-27T00:00:00Z".to_string(),
        };

        report.results.push(make_result_with_trace(
            "Executive::Stroop",
            "incongruent_accuracy",
            0.8,
            50,
            0.8,
        ));
        report.results.push(make_result_with_trace(
            "Executive::WCST",
            "categories_completed",
            0.75,
            50,
            0.75,
        ));

        let matrix = CrossDomainMatrix::from_single_report(&report);

        // Should produce at least one correlation (Stroop x WCST).
        let stroop_wcst = matrix
            .correlations
            .iter()
            .find(|c| c.domain_a == "Executive::Stroop" && c.domain_b == "Executive::WCST");
        assert!(
            stroop_wcst.is_some(),
            "from_single_report should produce Stroop x WCST correlation"
        );
        let c = stroop_wcst.unwrap();
        assert!(c.r.is_finite());
        assert!(c.n >= 3, "should have at least 3 blocks, got n={}", c.n);
    }

    // ── test_key_metric_for ──

    #[test]
    fn test_key_metric_for_benchmark() {
        assert_eq!(
            key_metric_for_benchmark("Executive::Stroop"),
            "incongruent_accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("Executive::WCST"),
            "categories_completed"
        );
        assert_eq!(
            key_metric_for_benchmark("WorM::N-back"),
            "nback_2::accuracy"
        );
        assert_eq!(key_metric_for_benchmark("WorM::DigitSpan"), "forward_span");
        assert_eq!(
            key_metric_for_benchmark("Metacognition::FeelingOfKnowing"),
            "fok_gamma"
        );
        assert_eq!(
            key_metric_for_benchmark("Metacognition::Calibration"),
            "calibration_error_ece"
        );
        assert_eq!(
            key_metric_for_benchmark("Motor::FittsLaw"),
            "fitts_r_squared"
        );
        assert_eq!(
            key_metric_for_benchmark("Reasoning::ArcFluid"),
            "transfer_accuracy"
        );
        assert_eq!(
            key_metric_for_benchmark("Unknown::Benchmark"),
            "overall_accuracy"
        );
    }

    // ── test_shared_mechanisms_filter ──

    #[test]
    fn test_shared_mechanisms_filter() {
        let matrix = CrossDomainMatrix {
            correlations: vec![
                DomainCorrelation {
                    domain_a: "A".to_string(),
                    metric_a: "m".to_string(),
                    domain_b: "B".to_string(),
                    metric_b: "m".to_string(),
                    r: 0.6,
                    n: 10,
                    p_value: 0.01,
                    shared_mechanism: SharedMechanism::Strong,
                },
                DomainCorrelation {
                    domain_a: "C".to_string(),
                    metric_a: "m".to_string(),
                    domain_b: "D".to_string(),
                    metric_b: "m".to_string(),
                    r: 0.05,
                    n: 10,
                    p_value: 0.9,
                    shared_mechanism: SharedMechanism::Independent,
                },
                DomainCorrelation {
                    domain_a: "E".to_string(),
                    metric_a: "m".to_string(),
                    domain_b: "F".to_string(),
                    metric_b: "m".to_string(),
                    r: 0.4,
                    n: 10,
                    p_value: 0.05,
                    shared_mechanism: SharedMechanism::Moderate,
                },
            ],
            predictions: Vec::new(),
        };

        let shared = matrix.shared_mechanisms();
        assert_eq!(shared.len(), 2, "should include Strong and Moderate only");
        assert!(shared.iter().any(|c| c.domain_a == "A"));
        assert!(shared.iter().any(|c| c.domain_a == "E"));
        assert!(!shared.iter().any(|c| c.domain_a == "C"));
    }

    // ── test_empty_reports ──

    #[test]
    fn test_empty_reports() {
        let matrix = CrossDomainMatrix::from_multi_seed_reports(&[]);
        assert!(matrix.correlations.is_empty());
        assert!(matrix.predictions.is_empty());

        let empty_report = BenchmarkReport {
            results: Vec::new(),
            timestamp: "2026-02-27T00:00:00Z".to_string(),
        };
        let matrix2 = CrossDomainMatrix::from_single_report(&empty_report);
        assert!(matrix2.correlations.is_empty());
        assert!(matrix2.predictions.is_empty());
    }
}
