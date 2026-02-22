//! Cross-benchmark correlation analysis.
//!
//! Computes Spearman rank correlations between benchmark key metrics across
//! multiple seed runs. Supports construct validity checks (same-domain
//! correlations should exceed 0.3).

use super::report::{domain_of, key_metric_for_benchmark, BenchmarkReport};
use std::collections::BTreeMap;

/// Cross-benchmark analysis from multi-seed runs.
pub struct CrossBenchmarkAnalysis {
    /// Benchmark name → vector of key metric values (one per seed).
    pub values: BTreeMap<String, Vec<f64>>,
}

/// Construct validity result.
pub struct ConstructValidity {
    /// Number of same-domain benchmark pairs.
    pub same_domain_pairs: usize,
    /// Pairs where correlation > 0.3.
    pub convergent_pairs: usize,
    /// Mean within-domain correlation.
    pub mean_within_correlation: f64,
}

impl CrossBenchmarkAnalysis {
    /// Build from multiple seed reports. Each report contributes one value per benchmark.
    pub fn from_multi_seed_reports(reports: &[BenchmarkReport]) -> Self {
        let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for report in reports {
            for result in &report.results {
                let key = key_metric_for_benchmark(&result.benchmark);
                if let Some(metric) = result.metrics.get(key) {
                    values
                        .entry(result.benchmark.clone())
                        .or_default()
                        .push(metric.mean);
                }
            }
        }

        Self { values }
    }

    /// Compute Spearman rank correlation matrix.
    ///
    /// Returns (benchmark_names, matrix) where matrix[i][j] is the correlation
    /// between benchmark i and benchmark j.
    pub fn correlation_matrix(&self) -> (Vec<String>, Vec<Vec<f64>>) {
        let names: Vec<String> = self.values.keys().cloned().collect();
        let n = names.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    let x = &self.values[&names[i]];
                    let y = &self.values[&names[j]];
                    let min_len = x.len().min(y.len());
                    if min_len >= 3 {
                        matrix[i][j] = spearman_correlation(&x[..min_len], &y[..min_len]);
                    }
                }
            }
        }

        (names, matrix)
    }

    /// Check construct validity: same-domain benchmark pairs should correlate > 0.3.
    pub fn construct_validity(&self) -> ConstructValidity {
        let (names, matrix) = self.correlation_matrix();
        let n = names.len();

        let mut same_domain_pairs = 0;
        let mut convergent_pairs = 0;
        let mut within_sum = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let domain_i = domain_of(&names[i]);
                let domain_j = domain_of(&names[j]);
                if domain_i == domain_j {
                    same_domain_pairs += 1;
                    let r = matrix[i][j];
                    within_sum += r;
                    if r > 0.3 {
                        convergent_pairs += 1;
                    }
                }
            }
        }

        let mean_within_correlation = if same_domain_pairs > 0 {
            within_sum / same_domain_pairs as f64
        } else {
            0.0
        };

        ConstructValidity {
            same_domain_pairs,
            convergent_pairs,
            mean_within_correlation,
        }
    }

    /// Split-half test-retest reliability per benchmark.
    ///
    /// Splits seed runs into odd/even halves and computes Spearman correlation
    /// between the halves. Returns a map of benchmark name → reliability coefficient.
    pub fn test_retest_reliability(&self) -> BTreeMap<String, f64> {
        let mut result = BTreeMap::new();
        for (name, vals) in &self.values {
            if vals.len() < 4 {
                // Need at least 2 values per half
                result.insert(name.clone(), 0.0);
                continue;
            }
            let even: Vec<f64> = vals.iter().step_by(2).copied().collect();
            let odd: Vec<f64> = vals.iter().skip(1).step_by(2).copied().collect();
            let min_len = even.len().min(odd.len());
            if min_len >= 3 {
                result.insert(
                    name.clone(),
                    spearman_correlation(&even[..min_len], &odd[..min_len]),
                );
            } else {
                result.insert(name.clone(), 0.0);
            }
        }
        result
    }

    /// Format test-retest reliability as a Markdown table.
    pub fn format_reliability(&self) -> String {
        let reliability = self.test_retest_reliability();
        let mut lines = Vec::new();
        lines.push("| Benchmark | Split-Half r | Interpretation |".to_string());
        lines.push("|-----------|-------------|----------------|".to_string());
        for (name, r) in &reliability {
            let interp = if *r >= 0.75 {
                "Excellent"
            } else if *r >= 0.50 {
                "Moderate"
            } else if *r >= 0.25 {
                "Fair"
            } else {
                "Poor"
            };
            let short = name.split("::").last().unwrap_or(name);
            lines.push(format!("| {} | {:.3} | {} |", short, r, interp));
        }
        lines.join("\n")
    }

    /// Pretty-print the correlation matrix as a Markdown table.
    pub fn format_matrix(&self) -> String {
        let (names, matrix) = self.correlation_matrix();
        if names.is_empty() {
            return "No benchmarks to correlate.".to_string();
        }

        let short_names: Vec<String> = names
            .iter()
            .map(|n| n.split("::").last().unwrap_or(n).to_string())
            .collect();

        let mut lines = Vec::new();

        // Header
        let header = std::iter::once("".to_string())
            .chain(
                short_names
                    .iter()
                    .map(|n| format!("{:>8}", &n[..n.len().min(8)])),
            )
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("| {} |", header));

        let sep = std::iter::once("---".to_string())
            .chain(short_names.iter().map(|_| "--------".to_string()))
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("| {} |", sep));

        // Rows
        for (i, name) in short_names.iter().enumerate() {
            let row = std::iter::once(format!("{:>8}", &name[..name.len().min(8)]))
                .chain(matrix[i].iter().map(|v| format!("{:>8.2}", v)))
                .collect::<Vec<_>>()
                .join(" | ");
            lines.push(format!("| {} |", row));
        }

        lines.join("\n")
    }
}

/// Compute Cohen's d effect size.
///
/// `d = (group_mean - baseline_mean) / pooled_sd`
/// Returns 0.0 if `pooled_sd` is near-zero.
pub fn cohens_d(group_mean: f64, baseline_mean: f64, pooled_sd: f64) -> f64 {
    if pooled_sd.abs() < 1e-15 {
        0.0
    } else {
        (group_mean - baseline_mean) / pooled_sd
    }
}

/// Categorize effect size by magnitude.
///
/// |d| < 0.2 → negligible, 0.2–0.5 → small, 0.5–0.8 → medium, ≥ 0.8 → large.
pub fn effect_size_label(d: f64) -> &'static str {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "negligible"
    } else if abs_d < 0.5 {
        "small"
    } else if abs_d < 0.8 {
        "medium"
    } else {
        "large"
    }
}

/// ICC(2,1) — two-way random, single measures.
///
/// `observations[j]` is the vector of ratings/measurements from judge j.
/// All judges must rate all subjects (balanced design).
/// Returns 0.0 if denominator is near-zero.
/// Values >0.75 indicate excellent reliability.
pub fn icc_2_1(observations: &[Vec<f64>]) -> f64 {
    let k = observations.len(); // number of judges/sessions
    if k < 2 {
        return 0.0;
    }
    let n = observations[0].len(); // number of subjects
    if n < 2 || observations.iter().any(|o| o.len() != n) {
        return 0.0;
    }

    let kf = k as f64;
    let nf = n as f64;
    let grand_mean: f64 = observations.iter().flat_map(|o| o.iter()).sum::<f64>() / (nf * kf);

    // Subject means (across judges)
    let subject_means: Vec<f64> = (0..n)
        .map(|i| observations.iter().map(|o| o[i]).sum::<f64>() / kf)
        .collect();

    // Judge means (across subjects)
    let judge_means: Vec<f64> = observations
        .iter()
        .map(|o| o.iter().sum::<f64>() / nf)
        .collect();

    // Between-subjects MS
    let ss_subjects: f64 = subject_means
        .iter()
        .map(|&m| (m - grand_mean).powi(2))
        .sum::<f64>()
        * kf;
    let bms = ss_subjects / (nf - 1.0);

    // Between-judges MS
    let ss_judges: f64 = judge_means
        .iter()
        .map(|&m| (m - grand_mean).powi(2))
        .sum::<f64>()
        * nf;
    let jms = ss_judges / (kf - 1.0);

    // Error MS (residual)
    let ss_total: f64 = observations
        .iter()
        .flat_map(|o| o.iter())
        .map(|&x| (x - grand_mean).powi(2))
        .sum();
    let ss_error = ss_total - ss_subjects - ss_judges;
    let ems = ss_error / ((nf - 1.0) * (kf - 1.0));

    // ICC(2,1) = (BMS - EMS) / (BMS + (k-1)*EMS + k*(JMS-EMS)/n)
    let denom = bms + (kf - 1.0) * ems + kf * (jms - ems) / nf;
    if denom.abs() < 1e-15 {
        0.0
    } else {
        ((bms - ems) / denom).clamp(-1.0, 1.0)
    }
}

/// Compute Spearman rank correlation between two slices.
///
/// Returns a value in [-1.0, 1.0]. Requires at least 3 data points.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Slices must be equal length");
    let n = x.len();
    if n < 3 {
        return 0.0;
    }

    let rank_x = ranks(x);
    let rank_y = ranks(y);

    // Pearson correlation of ranks
    let mean_rx = rank_x.iter().sum::<f64>() / n as f64;
    let mean_ry = rank_y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = rank_x[i] - mean_rx;
        let dy = rank_y[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Compute fractional ranks (average rank for ties).
fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| a.total_cmp(b));

    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for item in indexed.iter().take(j).skip(i) {
            result[item.0] = avg_rank;
        }
        i = j;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 20.0, 30.0, 40.0, 50.0];
        let r = spearman_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Expected 1.0, got {}", r);
    }

    #[test]
    fn test_spearman_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [50.0, 40.0, 30.0, 20.0, 10.0];
        let r = spearman_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-10, "Expected -1.0, got {}", r);
    }

    #[test]
    fn test_cross_benchmark_population() {
        let mut report1 = BenchmarkReport::new();
        let mut result1 = super::super::report::BenchmarkResult::new("WorM::NBack", None);
        result1.insert(
            "nback_2::accuracy",
            super::super::report::MetricValue::from_samples(&[0.8]),
        );
        report1.add(result1);

        let mut report2 = BenchmarkReport::new();
        let mut result2 = super::super::report::BenchmarkResult::new("WorM::NBack", None);
        result2.insert(
            "nback_2::accuracy",
            super::super::report::MetricValue::from_samples(&[0.9]),
        );
        report2.add(result2);

        let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&[report1, report2]);
        assert_eq!(analysis.values.len(), 1);
        assert_eq!(analysis.values["WorM::NBack"].len(), 2);
    }

    #[test]
    fn test_cohens_d_zero_diff() {
        assert_eq!(cohens_d(0.5, 0.5, 0.1), 0.0);
    }

    #[test]
    fn test_cohens_d_one_sd() {
        let d = cohens_d(1.0, 0.0, 1.0);
        assert!((d - 1.0).abs() < 1e-10, "Expected d=1.0, got {}", d);
    }

    #[test]
    fn test_cohens_d_near_zero_sd() {
        assert_eq!(cohens_d(0.5, 0.3, 0.0), 0.0);
    }

    #[test]
    fn test_effect_size_labels() {
        assert_eq!(effect_size_label(0.1), "negligible");
        assert_eq!(effect_size_label(0.3), "small");
        assert_eq!(effect_size_label(0.6), "medium");
        assert_eq!(effect_size_label(1.2), "large");
        assert_eq!(effect_size_label(-0.9), "large");
    }

    #[test]
    fn test_icc_perfect_agreement() {
        // All judges give identical scores → ICC should be ~1.0
        let obs = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let icc = icc_2_1(&obs);
        assert!(
            icc > 0.99,
            "Perfect agreement should yield ICC~1.0, got {}",
            icc
        );
    }

    #[test]
    fn test_icc_zero_agreement() {
        // Judges give unrelated scores — ICC should be near 0 or negative
        let obs = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]];
        let icc = icc_2_1(&obs);
        assert!(
            icc < 0.5,
            "Reversed judges should yield low ICC, got {}",
            icc
        );
    }

    #[test]
    fn test_icc_known_value() {
        // Two judges with systematic offset but same ranking
        let obs = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]];
        let icc = icc_2_1(&obs);
        // Should be high (same ranking, just shifted)
        assert!(
            icc > 0.7,
            "Systematic offset should still yield decent ICC, got {}",
            icc
        );
    }

    #[test]
    fn test_icc_single_judge() {
        let obs = vec![vec![1.0, 2.0, 3.0]];
        let icc = icc_2_1(&obs);
        assert_eq!(icc, 0.0, "Single judge should return 0.0");
    }

    #[test]
    fn test_format_matrix_non_empty() {
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0]);
        values.insert("A::Y".to_string(), vec![1.0, 2.0, 3.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let fmt = analysis.format_matrix();
        assert!(!fmt.is_empty());
        assert!(fmt.contains("1.00"));
    }
}
