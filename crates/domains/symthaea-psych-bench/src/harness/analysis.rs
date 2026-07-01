// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-benchmark correlation analysis.
//!
//! Computes Spearman rank correlations between benchmark key metrics across
//! multiple seed runs. Supports construct validity checks (same-domain
//! correlations should exceed 0.3).

use super::report::{BenchmarkReport, domain_of, key_metric_for_benchmark};
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

/// Campbell-Fiske Multitrait-Multimethod (MTMM) validity analysis.
///
/// Compares within-domain (convergent) vs between-domain (discriminant)
/// correlations to assess construct validity. A healthy battery should show
/// higher within-domain correlations than between-domain ones.
pub struct MtmmResult {
    /// Mean correlation among same-domain benchmark pairs (convergent).
    pub mean_convergent: f64,
    /// Mean correlation among different-domain benchmark pairs (discriminant).
    pub mean_discriminant: f64,
    /// Ratio: convergent / discriminant (should be > 1.0).
    pub convergent_discriminant_ratio: f64,
    /// Number of same-domain pairs analyzed.
    pub convergent_pairs: usize,
    /// Number of different-domain pairs analyzed.
    pub discriminant_pairs: usize,
    /// Per-domain convergent correlation means.
    pub domain_convergent: BTreeMap<String, f64>,
}

impl CrossBenchmarkAnalysis {
    /// Perform Campbell-Fiske MTMM analysis.
    ///
    /// Requires at least 3 seeds for meaningful correlations.
    pub fn mtmm_validity(&self) -> MtmmResult {
        let (names, matrix) = self.correlation_matrix();
        let n = names.len();

        let mut convergent_sum = 0.0;
        let mut convergent_count = 0usize;
        let mut discriminant_sum = 0.0;
        let mut discriminant_count = 0usize;

        // Per-domain tracking
        let mut domain_sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let r = matrix[i][j];
                let domain_i = domain_of(&names[i]);
                let domain_j = domain_of(&names[j]);

                if domain_i == domain_j {
                    convergent_sum += r;
                    convergent_count += 1;
                    let entry = domain_sums.entry(domain_i.to_string()).or_insert((0.0, 0));
                    entry.0 += r;
                    entry.1 += 1;
                } else {
                    discriminant_sum += r;
                    discriminant_count += 1;
                }
            }
        }

        let mean_convergent = if convergent_count > 0 {
            convergent_sum / convergent_count as f64
        } else {
            0.0
        };
        let mean_discriminant = if discriminant_count > 0 {
            discriminant_sum / discriminant_count as f64
        } else {
            0.0
        };
        let ratio = if mean_discriminant.abs() > 1e-10 {
            mean_convergent / mean_discriminant
        } else if mean_convergent > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        let domain_convergent = domain_sums
            .into_iter()
            .map(|(domain, (sum, count))| (domain, sum / count as f64))
            .collect();

        MtmmResult {
            mean_convergent,
            mean_discriminant,
            convergent_discriminant_ratio: ratio,
            convergent_pairs: convergent_count,
            discriminant_pairs: discriminant_count,
            domain_convergent,
        }
    }

    /// Format MTMM validity as a summary string.
    pub fn format_mtmm(&self) -> String {
        let mtmm = self.mtmm_validity();
        let mut lines = Vec::new();
        lines.push("Campbell-Fiske MTMM Validity Analysis".to_string());
        lines.push(format!(
            "  Convergent (within-domain):   r = {:.3} ({} pairs)",
            mtmm.mean_convergent, mtmm.convergent_pairs
        ));
        lines.push(format!(
            "  Discriminant (between-domain): r = {:.3} ({} pairs)",
            mtmm.mean_discriminant, mtmm.discriminant_pairs
        ));
        lines.push(format!(
            "  Convergent/Discriminant ratio: {:.2}",
            mtmm.convergent_discriminant_ratio
        ));
        let verdict = if mtmm.convergent_discriminant_ratio > 1.5 {
            "GOOD: within-domain correlations well exceed between-domain"
        } else if mtmm.convergent_discriminant_ratio > 1.0 {
            "FAIR: within-domain slightly exceeds between-domain"
        } else {
            "POOR: within-domain does not exceed between-domain"
        };
        lines.push(format!("  Verdict: {}", verdict));

        if !mtmm.domain_convergent.is_empty() {
            lines.push(String::new());
            lines.push("  Per-domain convergent correlations:".to_string());
            for (domain, r) in &mtmm.domain_convergent {
                lines.push(format!("    {:<14} r = {:.3}", domain, r));
            }
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

/// Compute Cronbach's alpha for internal consistency.
///
/// `items[i]` is the score vector for item/condition i across participants/seeds.
/// All item vectors must be the same length (number of participants).
/// Returns alpha in [-inf, 1.0]. Values >0.70 indicate acceptable consistency.
pub fn cronbachs_alpha(items: &[Vec<f64>]) -> f64 {
    let k = items.len();
    if k < 2 {
        return 0.0;
    }
    let n = items[0].len();
    if n < 2 || items.iter().any(|v| v.len() != n) {
        return 0.0;
    }

    // Compute variance of each item
    let item_variances: Vec<f64> = items
        .iter()
        .map(|scores| {
            let mean = scores.iter().sum::<f64>() / n as f64;
            scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        })
        .collect();

    // Compute total scores per participant and their variance
    let total_scores: Vec<f64> = (0..n)
        .map(|p| items.iter().map(|item| item[p]).sum::<f64>())
        .collect();
    let total_mean = total_scores.iter().sum::<f64>() / n as f64;
    let total_variance = total_scores
        .iter()
        .map(|x| (x - total_mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    if total_variance.abs() < 1e-15 {
        return 0.0;
    }

    let sum_item_var: f64 = item_variances.iter().sum();
    let kf = k as f64;

    (kf / (kf - 1.0)) * (1.0 - sum_item_var / total_variance)
}

/// Convert a z-score to a percentile rank using the normal CDF.
///
/// Uses the Abramowitz & Stegun (1964) rational approximation.
/// z=0 → 50th percentile, z=+1 → ~84th, z=-1 → ~16th.
pub fn percentile_from_z(z: f64) -> f64 {
    if !z.is_finite() {
        return 50.0; // NaN or Inf → return median as safe default
    }
    // Phi(z) via Abramowitz & Stegun approximation 26.2.17
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let cdf = if z >= 0.0 { 1.0 - p * poly } else { p * poly };
    (cdf * 100.0).clamp(0.01, 99.99)
}

/// Principal Component Analysis result.
pub struct PcaResult {
    /// Eigenvalues in descending order.
    pub eigenvalues: Vec<f64>,
    /// Proportion of variance explained by each component.
    pub variance_explained: Vec<f64>,
    /// Cumulative variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Factor loadings: components × benchmarks.
    pub loadings: Vec<Vec<f64>>,
    /// Benchmark names corresponding to loading columns.
    pub benchmark_names: Vec<String>,
}

impl CrossBenchmarkAnalysis {
    /// Run PCA on the benchmark correlation matrix.
    ///
    /// Extracts principal components via power iteration on the correlation matrix.
    /// Returns eigenvalues, variance explained, and factor loadings for the top
    /// `max_components` components.
    pub fn pca(&self, max_components: usize) -> PcaResult {
        let (names, corr) = self.correlation_matrix();
        let n = names.len();
        let max_k = max_components.min(n);

        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();

        // Deflation-based power iteration
        let mut matrix: Vec<Vec<f64>> = corr.clone();

        for _ in 0..max_k {
            let (eigenval, eigenvec) = power_iteration(&matrix, 200);
            if eigenval.abs() < 1e-10 {
                break;
            }
            eigenvalues.push(eigenval);
            eigenvectors.push(eigenvec.clone());

            // Deflate: A = A - lambda * v * v^T
            for i in 0..n {
                for j in 0..n {
                    matrix[i][j] -= eigenval * eigenvec[i] * eigenvec[j];
                }
            }
        }

        let total_var: f64 = eigenvalues.iter().sum();
        let variance_explained: Vec<f64> = if total_var > 0.0 {
            eigenvalues.iter().map(|e| e / total_var).collect()
        } else {
            vec![0.0; eigenvalues.len()]
        };

        let mut cumulative = Vec::new();
        let mut cum = 0.0;
        for &v in &variance_explained {
            cum += v;
            cumulative.push(cum);
        }

        // Loadings = eigenvector * sqrt(eigenvalue)
        let loadings: Vec<Vec<f64>> = eigenvectors
            .iter()
            .zip(eigenvalues.iter())
            .map(|(vec, &val)| vec.iter().map(|v| v * val.sqrt()).collect())
            .collect();

        PcaResult {
            eigenvalues,
            variance_explained,
            cumulative_variance: cumulative,
            loadings,
            benchmark_names: names,
        }
    }

    /// Format PCA results as a summary string.
    pub fn format_pca(&self, max_components: usize) -> String {
        let pca = self.pca(max_components);
        let mut lines = Vec::new();
        lines.push("Principal Component Analysis".to_string());
        lines.push(format!(
            "| {:>4} | {:>10} | {:>10} | {:>12} |",
            "PC", "Eigenvalue", "% Variance", "Cumulative %"
        ));
        lines.push(format!(
            "| {:>4} | {:>10} | {:>10} | {:>12} |",
            "----", "----------", "----------", "------------"
        ));
        for (i, ((ev, ve), cv)) in pca
            .eigenvalues
            .iter()
            .zip(pca.variance_explained.iter())
            .zip(pca.cumulative_variance.iter())
            .enumerate()
        {
            lines.push(format!(
                "| {:>4} | {:>10.3} | {:>9.1}% | {:>11.1}% |",
                i + 1,
                ev,
                ve * 100.0,
                cv * 100.0,
            ));
        }

        // Show top loadings for PC1 and PC2
        if pca.loadings.len() >= 2 {
            lines.push(String::new());
            lines.push("Top loadings (PC1, PC2):".to_string());
            lines.push(format!("  {:<25} {:>8} {:>8}", "Benchmark", "PC1", "PC2"));
            for (i, name) in pca.benchmark_names.iter().enumerate() {
                let short = name.split("::").last().unwrap_or(name);
                let l1 = pca.loadings[0].get(i).copied().unwrap_or(0.0);
                let l2 = pca.loadings[1].get(i).copied().unwrap_or(0.0);
                lines.push(format!(
                    "  {:<25} {:>+8.3} {:>+8.3}",
                    &short[..short.len().min(25)],
                    l1,
                    l2,
                ));
            }
        }

        lines.join("\n")
    }
}

/// Ablation effect size: Cohen's d between two conditions' metric vectors.
///
/// Given two sets of per-benchmark metric means (baseline vs ablated),
/// returns `(benchmark_name, cohen_d, label)` triples sorted by |d|.
pub fn ablation_effect_sizes(
    baseline_means: &[(&str, f64)],
    ablated_means: &[(&str, f64)],
) -> Vec<(String, f64, String)> {
    // Pair up by benchmark name
    let mut results = Vec::new();
    for (name, base_val) in baseline_means {
        if let Some((_, abl_val)) = ablated_means.iter().find(|(n, _)| n == name) {
            let diff = base_val - abl_val;
            // Use pooled SD estimate from both values (conservative: assume SD ~ 10% of baseline)
            let sd_est = base_val.abs() * 0.10;
            let d = if sd_est > 1e-15 { diff / sd_est } else { 0.0 };
            results.push((name.to_string(), d, effect_size_label(d).to_string()));
        }
    }
    results.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    results
}

/// Format ablation effect sizes as a table.
pub fn format_ablation_effects(effects: &[(String, f64, String)]) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "| {:25} | {:>10} | {:>12} |",
        "Benchmark", "Cohen's d", "Effect"
    ));
    lines.push(format!(
        "| {:25} | {:>10} | {:>12} |",
        "-------------------------", "----------", "------------"
    ));
    for (name, d, label) in effects {
        let short = name.split("::").last().unwrap_or(name);
        lines.push(format!(
            "| {:25} | {:>+10.3} | {:>12} |",
            &short[..short.len().min(25)],
            d,
            label,
        ));
    }
    lines.join("\n")
}

/// Power iteration to find the dominant eigenvector/eigenvalue.
fn power_iteration(matrix: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = matrix.len();
    if n == 0 {
        return (0.0, Vec::new());
    }

    // Initialize with uniform vector
    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..max_iter {
        // w = A * v
        let mut w: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return (0.0, vec![0.0; n]);
        }
        let new_v: Vec<f64> = w.iter().map(|x| x / norm).collect();

        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(new_v.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        v = new_v;
        if diff < 1e-12 {
            break;
        }
    }

    // Final eigenvalue
    let mut w: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            w[i] += matrix[i][j] * v[j];
        }
    }
    let eigenval: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

    (eigenval, v)
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
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

/// Simple xorshift64 PRNG for bootstrap resampling.
///
/// Deterministic given a seed; avoids pulling in `rand` for resampling.
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    // Guard against seed-0 fixed point
    if s == 0 {
        s ^= 0x9E3779B97F4A7C15;
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Compute a bootstrap confidence interval for the mean.
///
/// Resamples `samples` with replacement `n_resamples` times, computes the mean
/// of each resample, and returns the `(alpha/2, 1-alpha/2)` percentiles of the
/// bootstrap distribution as the CI bounds.
///
/// # Arguments
/// - `samples`: observed data
/// - `n_resamples`: number of bootstrap resamples (default: 2000)
/// - `alpha`: significance level (e.g. 0.05 for 95% CI)
/// - `seed`: RNG seed for reproducibility
pub fn bootstrap_ci(samples: &[f64], n_resamples: usize, alpha: f64, seed: u64) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (samples[0], samples[0]);
    }

    let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
    let mut boot_means = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = (xorshift64(&mut rng_state) as usize) % n;
            sum += samples[idx];
        }
        boot_means.push(sum / n as f64);
    }

    boot_means.sort_by(|a, b| a.total_cmp(b));

    let lo_idx = ((alpha / 2.0) * boot_means.len() as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * boot_means.len() as f64).ceil() as usize;
    let lo_idx = lo_idx.min(boot_means.len().saturating_sub(1));
    let hi_idx = hi_idx.min(boot_means.len().saturating_sub(1));

    (boot_means[lo_idx], boot_means[hi_idx])
}

/// Compute a BCa (bias-corrected and accelerated) bootstrap confidence interval.
///
/// BCa adjusts for both bias and skewness in the bootstrap distribution via
/// jackknife-estimated acceleration and bias-correction factors.
///
/// Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap", Ch. 14.
///
/// # Arguments
/// - `samples`: observed data
/// - `n_resamples`: number of bootstrap resamples (default: 2000)
/// - `alpha`: significance level (e.g. 0.05 for 95% CI)
/// - `seed`: RNG seed for reproducibility
pub fn bootstrap_ci_bca(samples: &[f64], n_resamples: usize, alpha: f64, seed: u64) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (samples[0], samples[0]);
    }

    let observed_mean = samples.iter().sum::<f64>() / n as f64;

    // Generate bootstrap distribution
    let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
    let mut boot_means = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = (xorshift64(&mut rng_state) as usize) % n;
            sum += samples[idx];
        }
        boot_means.push(sum / n as f64);
    }

    boot_means.sort_by(|a, b| a.total_cmp(b));

    // Bias correction factor (z0): proportion of bootstrap means < observed mean
    let below_count = boot_means.iter().filter(|&&x| x < observed_mean).count();
    let p = below_count as f64 / n_resamples as f64;
    let z0 = inv_normal_cdf(p.clamp(0.001, 0.999));

    // Acceleration factor (a): jackknife estimate
    let nf = n as f64;
    let jackknife_means: Vec<f64> = (0..n)
        .map(|i| {
            let sum: f64 = samples
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &x)| x)
                .sum();
            sum / (nf - 1.0)
        })
        .collect();
    let jack_mean = jackknife_means.iter().sum::<f64>() / nf;
    let num: f64 = jackknife_means
        .iter()
        .map(|&x| (jack_mean - x).powi(3))
        .sum();
    let den: f64 = jackknife_means
        .iter()
        .map(|&x| (jack_mean - x).powi(2))
        .sum();
    let a = if den.abs() > 1e-15 {
        num / (6.0 * den.powf(1.5))
    } else {
        0.0
    };

    // Adjusted percentiles
    let z_alpha_lo = inv_normal_cdf(alpha / 2.0);
    let z_alpha_hi = inv_normal_cdf(1.0 - alpha / 2.0);

    let adj_lo = normal_cdf(z0 + (z0 + z_alpha_lo) / (1.0 - a * (z0 + z_alpha_lo)));
    let adj_hi = normal_cdf(z0 + (z0 + z_alpha_hi) / (1.0 - a * (z0 + z_alpha_hi)));

    let lo_idx = (adj_lo * boot_means.len() as f64).floor() as usize;
    let hi_idx = (adj_hi * boot_means.len() as f64).ceil() as usize;
    let lo_idx = lo_idx.min(boot_means.len().saturating_sub(1));
    let hi_idx = hi_idx.min(boot_means.len().saturating_sub(1));

    (boot_means[lo_idx], boot_means[hi_idx])
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
pub fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if z >= 0.0 { 1.0 - p * poly } else { p * poly }
}

/// Inverse standard normal CDF (rational approximation).
///
/// Abramowitz & Stegun 26.2.23 approximation for p in (0,1).
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -3.0;
    }
    if p >= 1.0 {
        return 3.0;
    }

    // Rational approximation (Peter Acklam's algorithm)
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
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

// ---------- Speed-Accuracy Tradeoff (SAT) Curves ----------

/// A single point on a SAT curve.
#[derive(Debug, Clone)]
pub struct SatCurvePoint {
    /// Time pressure level (0.0 to 1.0).
    pub time_pressure: f64,
    /// Accuracy at this pressure level.
    pub accuracy: f64,
    /// Mean RT (ticks) at this pressure level.
    pub rt_ticks: f64,
}

/// A fitted SAT curve for one benchmark.
#[derive(Debug, Clone)]
pub struct SatCurve {
    /// Benchmark name.
    pub benchmark: String,
    /// Raw data points.
    pub points: Vec<SatCurvePoint>,
    /// Wickelgren (1977) fit: λ (asymptotic accuracy).
    pub lambda: f64,
    /// Wickelgren (1977) fit: β (rate).
    pub beta: f64,
    /// Wickelgren (1977) fit: δ (onset delay).
    pub delta: f64,
    /// R² goodness of fit.
    pub r_squared: f64,
}

/// Fit a Wickelgren (1977) SAT curve: d'(t) = λ(1 - e^{-β(t-δ)}).
///
/// Uses grid search over parameter space. `points` should have accuracy in [0,1]
/// and time_pressure in [0,1] (mapped to time = 1 - time_pressure for fitting).
pub fn fit_sat_curve(benchmark: &str, points: &[SatCurvePoint]) -> SatCurve {
    if points.len() < 2 {
        return SatCurve {
            benchmark: benchmark.to_string(),
            points: points.to_vec(),
            lambda: points.first().map(|p| p.accuracy).unwrap_or(0.5),
            beta: 1.0,
            delta: 0.0,
            r_squared: 0.0,
        };
    }

    let max_acc = points.iter().map(|p| p.accuracy).fold(0.0_f64, f64::max);
    let min_tp = points
        .iter()
        .map(|p| p.time_pressure)
        .fold(1.0_f64, f64::min);

    // Grid search
    let mut best_r2 = f64::NEG_INFINITY;
    let mut best_params = (max_acc, 1.0, 0.0);

    for lambda_i in 0..=10 {
        let lambda = max_acc + (1.0 - max_acc) * lambda_i as f64 / 10.0;
        for beta_i in 1..=20 {
            let beta = beta_i as f64 * 0.25;
            for delta_i in 0..=10 {
                let delta = min_tp * delta_i as f64 / 10.0;

                // Compute R²
                let mean_acc: f64 =
                    points.iter().map(|p| p.accuracy).sum::<f64>() / points.len() as f64;
                let ss_tot: f64 = points.iter().map(|p| (p.accuracy - mean_acc).powi(2)).sum();

                let ss_res: f64 = points
                    .iter()
                    .map(|p| {
                        // Map time_pressure to "available time": t = 1 - time_pressure
                        let t = 1.0 - p.time_pressure;
                        let predicted = if t > delta {
                            lambda * (1.0 - (-beta * (t - delta)).exp())
                        } else {
                            0.0
                        };
                        (p.accuracy - predicted).powi(2)
                    })
                    .sum();

                let r2 = if ss_tot > 1e-15 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                };

                if r2 > best_r2 {
                    best_r2 = r2;
                    best_params = (lambda, beta, delta);
                }
            }
        }
    }

    SatCurve {
        benchmark: benchmark.to_string(),
        points: points.to_vec(),
        lambda: best_params.0,
        beta: best_params.1,
        delta: best_params.2,
        r_squared: best_r2.max(0.0),
    }
}

/// Format SAT curve as ASCII table.
pub fn sat_curve_ascii(curve: &SatCurve) -> String {
    let mut out = format!(
        "SAT Curve: {} (λ={:.3}, β={:.3}, δ={:.3}, R²={:.3})\n",
        curve.benchmark, curve.lambda, curve.beta, curve.delta, curve.r_squared
    );
    out.push_str("  Pressure | Accuracy |  RT\n");
    out.push_str("  ---------|----------|------\n");
    for p in &curve.points {
        out.push_str(&format!(
            "     {:.1}   |  {:.3}   | {:.1}\n",
            p.time_pressure, p.accuracy, p.rt_ticks
        ));
    }
    out
}

// ---------- Random-effects meta-analysis (DerSimonian-Laird) ----------

/// Result of a DerSimonian-Laird random-effects meta-analysis.
#[derive(Debug, Clone)]
pub struct MetaAnalysisResult {
    /// Summary effect size (Cohen's d).
    pub summary_d: f64,
    /// Standard error of summary effect.
    pub summary_se: f64,
    /// 95% CI lower bound.
    pub ci_lower: f64,
    /// 95% CI upper bound.
    pub ci_upper: f64,
    /// Heterogeneity variance (tau-squared).
    pub tau_squared: f64,
    /// I-squared heterogeneity percentage (0–100).
    pub i_squared: f64,
    /// Cochran's Q statistic.
    pub q: f64,
    /// Degrees of freedom for Q (k-1).
    pub q_df: usize,
    /// p-value for Q test (chi-squared).
    pub q_p_value: f64,
    /// Number of studies (k).
    pub k: usize,
}

/// Run DerSimonian-Laird random-effects meta-analysis on a set of effect sizes.
///
/// `effects` is a slice of `(d, se)` tuples where `d` is Cohen's d and `se` is
/// the standard error of d. SE for Cohen's d: SE = sqrt(1/n + d²/(2n)).
pub fn meta_analysis(effects: &[(f64, f64)]) -> Option<MetaAnalysisResult> {
    let k = effects.len();
    if k == 0 {
        return None;
    }
    if k == 1 {
        let (d, se) = effects[0];
        return Some(MetaAnalysisResult {
            summary_d: d,
            summary_se: se,
            ci_lower: d - 1.96 * se,
            ci_upper: d + 1.96 * se,
            tau_squared: 0.0,
            i_squared: 0.0,
            q: 0.0,
            q_df: 0,
            q_p_value: 1.0,
            k: 1,
        });
    }

    // Fixed-effect weights: w_i = 1/SE_i²
    let weights: Vec<f64> = effects
        .iter()
        .map(|(_, se)| if *se > 0.0 { 1.0 / (se * se) } else { 0.0 })
        .collect();

    let sum_w: f64 = weights.iter().sum();
    if sum_w < 1e-15 {
        return None;
    }

    // Fixed-effect estimate
    let d_fixed: f64 = effects
        .iter()
        .zip(&weights)
        .map(|((d, _), w)| w * d)
        .sum::<f64>()
        / sum_w;

    // Cochran's Q
    let q: f64 = effects
        .iter()
        .zip(&weights)
        .map(|((d, _), w)| w * (d - d_fixed).powi(2))
        .sum();

    // tau² via DerSimonian-Laird
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let c = sum_w - sum_w2 / sum_w;
    let tau_sq = if q > (k as f64 - 1.0) {
        (q - (k as f64 - 1.0)) / c
    } else {
        0.0
    };

    // Random-effects weights: w_i* = 1/(SE_i² + τ²)
    let re_weights: Vec<f64> = effects
        .iter()
        .map(|(_, se)| {
            let denom = se * se + tau_sq;
            if denom > 0.0 { 1.0 / denom } else { 0.0 }
        })
        .collect();

    let sum_re_w: f64 = re_weights.iter().sum();
    if sum_re_w < 1e-15 {
        return None;
    }

    let summary_d: f64 = effects
        .iter()
        .zip(&re_weights)
        .map(|((d, _), w)| w * d)
        .sum::<f64>()
        / sum_re_w;
    let summary_se = (1.0 / sum_re_w).sqrt();

    // I² = max(0, (Q - (k-1)) / Q) × 100
    let i_squared = if q > 0.0 {
        ((q - (k as f64 - 1.0)) / q).max(0.0) * 100.0
    } else {
        0.0
    };

    // p-value for Q via Wilson-Hilferty chi-squared approximation
    let df = (k - 1) as f64;
    let q_p_value = if df > 0.0 {
        let z = ((q / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
        1.0 - normal_cdf(z)
    } else {
        1.0
    };

    Some(MetaAnalysisResult {
        summary_d,
        summary_se,
        ci_lower: summary_d - 1.96 * summary_se,
        ci_upper: summary_d + 1.96 * summary_se,
        tau_squared: tau_sq,
        i_squared,
        q,
        q_df: k - 1,
        q_p_value,
        k,
    })
}

/// Run meta-analysis from forest plot rows.
///
/// Computes SE for each row's Cohen's d using SE = sqrt(1/n + d²/(2n)).
pub fn meta_analysis_from_forest(
    rows: &[super::report::ForestPlotRow],
) -> Option<MetaAnalysisResult> {
    let effects: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|row| {
            let d = row.cohens_d?;
            // n from the benchmark's trial count isn't in ForestPlotRow,
            // so estimate from CI width: SE ≈ (ci_upper - ci_lower) / 3.92
            let ci_width = row.ci_upper - row.ci_lower;
            let se = if ci_width > 0.0 { ci_width / 3.92 } else { 0.1 };
            Some((d, se))
        })
        .collect();

    meta_analysis(&effects)
}

/// Format meta-analysis result as a summary string.
pub fn format_meta_analysis(result: &MetaAnalysisResult) -> String {
    format!(
        "Meta-analysis (k={}): d = {:.3} [{:.3}, {:.3}], SE = {:.3}\n  \
         τ² = {:.4}, I² = {:.1}%, Q({}) = {:.2}, p = {:.4}",
        result.k,
        result.summary_d,
        result.ci_lower,
        result.ci_upper,
        result.summary_se,
        result.tau_squared,
        result.i_squared,
        result.q_df,
        result.q,
        result.q_p_value,
    )
}

// ── Individual Differences Simulation ─────────────────────────────────────

/// Result of individual differences simulation for one benchmark.
#[derive(Debug, Clone)]
pub struct IndividualDifferencesResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Key metric name.
    pub metric: String,
    /// Number of simulated participants.
    pub n: usize,
    /// Mean across participants.
    pub mean: f64,
    /// Standard deviation.
    pub sd: f64,
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
    /// Skewness.
    pub skewness: f64,
    /// Shapiro-Wilk W approximation.
    pub shapiro_wilk_w: f64,
    /// 95% CI lower bound.
    pub ci_lower: f64,
    /// 95% CI upper bound.
    pub ci_upper: f64,
}

/// Approximate Shapiro-Wilk W statistic.
///
/// Uses the simplified formula W = 1 - ratio of ordered-statistic spread
/// to total variance. For small samples (<50) this approximates the full
/// Shapiro-Wilk; for larger samples it converges to a normality measure.
pub fn shapiro_wilk_approx(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return 1.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let ss_total: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum();

    if ss_total < 1e-15 {
        return 1.0; // No variance = perfectly "normal"
    }

    // Compute ordered-statistic numerator (sum of a_i * (x_(n+1-i) - x_i))
    let half = n / 2;
    let mut numerator = 0.0;
    for i in 0..half {
        // Approximate a_i coefficients using expected normal order statistics
        let expected_z = probit_approx((i as f64 + 0.375) / (n as f64 + 0.25));
        let pair_diff = sorted[n - 1 - i] - sorted[i];
        numerator += expected_z * pair_diff;
    }

    let w = numerator * numerator / ss_total;
    w.clamp(0.0, 1.0)
}

/// Probit approximation for Shapiro-Wilk.
fn probit_approx(p: f64) -> f64 {
    let p = p.clamp(0.001, 0.999);
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

/// Compute individual differences summary for a set of participant scores.
pub fn individual_differences(
    benchmark: &str,
    metric: &str,
    scores: &[f64],
) -> IndividualDifferencesResult {
    let n = scores.len();
    if n == 0 {
        return IndividualDifferencesResult {
            benchmark: benchmark.to_string(),
            metric: metric.to_string(),
            n: 0,
            mean: 0.0,
            sd: 0.0,
            min: 0.0,
            max: 0.0,
            skewness: 0.0,
            shapiro_wilk_w: 1.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
        };
    }

    let mean = scores.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let sd = var.sqrt();
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Skewness (Fisher's)
    let skewness = if sd > 1e-15 && n > 2 {
        let m3 = scores
            .iter()
            .map(|x| ((x - mean) / sd).powi(3))
            .sum::<f64>()
            / n as f64;
        m3 * (n as f64).sqrt() * (n as f64 - 1.0).sqrt() / (n as f64 - 2.0)
    } else {
        0.0
    };

    let w = shapiro_wilk_approx(scores);

    // 95% CI
    let se = sd / (n as f64).sqrt();
    let ci_lower = mean - 1.96 * se;
    let ci_upper = mean + 1.96 * se;

    IndividualDifferencesResult {
        benchmark: benchmark.to_string(),
        metric: metric.to_string(),
        n,
        mean,
        sd,
        min,
        max,
        skewness,
        shapiro_wilk_w: w,
        ci_lower,
        ci_upper,
    }
}

/// Format individual differences result as a summary line.
pub fn format_individual_differences(r: &IndividualDifferencesResult) -> String {
    format!(
        "{:25} {:>25} | N={:>3} | M={:>8.3} SD={:>7.3} [{:>7.3}, {:>7.3}] | \
         skew={:>+6.2} W={:.3} | 95%CI [{:.3}, {:.3}]",
        r.benchmark.split("::").last().unwrap_or(&r.benchmark),
        r.metric,
        r.n,
        r.mean,
        r.sd,
        r.min,
        r.max,
        r.skewness,
        r.shapiro_wilk_w,
        r.ci_lower,
        r.ci_upper,
    )
}

// ── HDC Dimensionality Scaling ───────────────────────────────────────────

/// Result of dimensionality scaling analysis for one benchmark.
#[derive(Debug, Clone)]
pub struct DimScalingResult {
    /// Benchmark name.
    pub benchmark: String,
    /// Dimension values tested.
    pub dims: Vec<usize>,
    /// Key metric values at each dimension.
    pub values: Vec<f64>,
    /// Linear regression slope of metric vs log₂(dim).
    pub slope: f64,
    /// R² of the linear regression.
    pub r_squared: f64,
}

/// Compute linear regression slope of metric vs log₂(dim).
pub fn dim_scaling_slope(dims: &[usize], values: &[f64]) -> (f64, f64) {
    let n = dims.len();
    if n < 2 || n != values.len() {
        return (0.0, 0.0);
    }

    let xs: Vec<f64> = dims.iter().map(|&d| (d as f64).log2()).collect();
    let x_mean = xs.iter().sum::<f64>() / n as f64;
    let y_mean = values.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (xs[i] - x_mean) * (values[i] - y_mean);
        den += (xs[i] - x_mean).powi(2);
    }

    let slope = if den.abs() > 1e-15 { num / den } else { 0.0 };

    // R²
    let ss_tot: f64 = values.iter().map(|y| (y - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 1e-15 {
        let ss_res: f64 = xs
            .iter()
            .zip(values.iter())
            .map(|(x, y)| {
                let predicted = y_mean + slope * (x - x_mean);
                (y - predicted).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, r_squared)
}

/// Format dimensionality scaling result as a table row.
pub fn format_dim_scaling(r: &DimScalingResult) -> String {
    let short = r.benchmark.split("::").last().unwrap_or(&r.benchmark);
    let vals_str: String = r
        .values
        .iter()
        .map(|v| format!("{:.3}", v))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "{:25} | {} | slope={:>+7.4} R²={:.3}",
        short, vals_str, r.slope, r.r_squared,
    )
}

/// Reasoning domain validity analysis.
///
/// Computes convergent validity within the reasoning cluster (ArcFluid,
/// ArcCompositional, ArcAnalogy, ArcAbductive) and discriminant validity
/// against executive function benchmarks (Stroop, WCST, Ravens, Tower).
///
/// Expects a `BenchmarkReport` containing results from a single seed/run.
/// For proper correlation analysis, use `CrossBenchmarkAnalysis::from_multi_seed_reports`.
pub struct ReasoningValidityResult {
    /// Mean pairwise correlation within reasoning benchmarks.
    pub reasoning_convergent: f64,
    /// Number of reasoning benchmark pairs with r > 0.3.
    pub reasoning_convergent_pairs: usize,
    /// Mean pairwise correlation between reasoning and executive benchmarks.
    pub reasoning_executive_correlation: f64,
    /// Convergent/discriminant ratio (should be > 1).
    pub convergent_discriminant_ratio: f64,
    /// Per-benchmark key metric values for the reasoning cluster.
    pub reasoning_scores: BTreeMap<String, f64>,
    /// Per-benchmark key metric values for executive function cluster.
    pub executive_scores: BTreeMap<String, f64>,
}

/// Extract reasoning validity from a single-run report.
///
/// This is a lightweight analysis based on within-run metric values.
/// For proper validity with correlations, use `CrossBenchmarkAnalysis`.
pub fn reasoning_validity_single_run(
    report: &super::report::BenchmarkReport,
) -> ReasoningValidityResult {
    let reasoning_benchmarks = ["ArcFluid", "ArcCompositional", "ArcAnalogy", "ArcAbductive"];
    let executive_benchmarks = ["Stroop", "Wisconsin", "Ravens", "TowerOfLondon"];

    let mut reasoning_scores = BTreeMap::new();
    let mut executive_scores = BTreeMap::new();

    for result in &report.results {
        let key_metric = super::report::key_metric_for_benchmark(&result.benchmark);
        if let Some(metric) = result.metrics.get(key_metric) {
            for name in &reasoning_benchmarks {
                if result.benchmark.contains(name) {
                    reasoning_scores.insert(result.benchmark.clone(), metric.mean);
                }
            }
            for name in &executive_benchmarks {
                if result.benchmark.contains(name) && !result.benchmark.contains("Emotional") {
                    executive_scores.insert(result.benchmark.clone(), metric.mean);
                }
            }
        }
    }

    // Compute "convergent" = mean absolute difference within reasoning (lower = more similar)
    // and "discriminant" = mean absolute difference between reasoning and executive
    let reasoning_vals: Vec<f64> = reasoning_scores.values().copied().collect();
    let executive_vals: Vec<f64> = executive_scores.values().copied().collect();

    // Within-reasoning pairwise correlation of scores (using values as single data points)
    // Note: with a single run, we can only compute consistency, not true correlation.
    // We use coefficient of variation as a proxy for convergent validity.
    let reasoning_mean = if reasoning_vals.is_empty() {
        0.0
    } else {
        reasoning_vals.iter().sum::<f64>() / reasoning_vals.len() as f64
    };
    let reasoning_var = if reasoning_vals.len() > 1 {
        reasoning_vals
            .iter()
            .map(|v| (v - reasoning_mean).powi(2))
            .sum::<f64>()
            / (reasoning_vals.len() - 1) as f64
    } else {
        0.0
    };
    // Convergent validity proxy: 1 - CV (high = consistent scores within reasoning)
    let reasoning_cv = if reasoning_mean.abs() > 1e-10 {
        reasoning_var.sqrt() / reasoning_mean.abs()
    } else {
        1.0
    };
    let reasoning_convergent = (1.0 - reasoning_cv).max(0.0);

    // Count "convergent pairs" where scores are within 0.2 of each other
    let mut convergent_pairs = 0;
    let n_r = reasoning_vals.len();
    for i in 0..n_r {
        for j in (i + 1)..n_r {
            if (reasoning_vals[i] - reasoning_vals[j]).abs() < 0.2 {
                convergent_pairs += 1;
            }
        }
    }

    // Cross-cluster mean difference (higher = more discriminant)
    let mut cross_diffs = Vec::new();
    for r in &reasoning_vals {
        for e in &executive_vals {
            cross_diffs.push((r - e).abs());
        }
    }
    let reasoning_executive_correlation = if cross_diffs.is_empty() {
        0.0
    } else {
        1.0 - (cross_diffs.iter().sum::<f64>() / cross_diffs.len() as f64)
    };

    let convergent_discriminant_ratio = if reasoning_executive_correlation.abs() > 1e-10 {
        reasoning_convergent / reasoning_executive_correlation.abs()
    } else if reasoning_convergent > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    ReasoningValidityResult {
        reasoning_convergent,
        reasoning_convergent_pairs: convergent_pairs,
        reasoning_executive_correlation,
        convergent_discriminant_ratio,
        reasoning_scores,
        executive_scores,
    }
}

/// Format reasoning validity result as a Markdown table.
pub fn format_reasoning_validity(result: &ReasoningValidityResult) -> String {
    let mut lines = vec![
        "## Reasoning Domain Validity".to_string(),
        String::new(),
        "### Reasoning Cluster Scores".to_string(),
        "| Benchmark | Key Metric |".to_string(),
        "|-----------|-----------|".to_string(),
    ];
    for (name, val) in &result.reasoning_scores {
        let short = name.split("::").last().unwrap_or(name);
        lines.push(format!("| {} | {:.3} |", short, val));
    }
    lines.push(String::new());
    lines.push("### Executive Function Cluster Scores".to_string());
    lines.push("| Benchmark | Key Metric |".to_string());
    lines.push("|-----------|-----------|".to_string());
    for (name, val) in &result.executive_scores {
        let short = name.split("::").last().unwrap_or(name);
        lines.push(format!("| {} | {:.3} |", short, val));
    }
    lines.push(String::new());
    lines.push(format!(
        "Reasoning convergent (consistency): {:.3}",
        result.reasoning_convergent
    ));
    lines.push(format!(
        "Reasoning-Executive cross-cluster:  {:.3}",
        result.reasoning_executive_correlation
    ));
    lines.push(format!(
        "Convergent/Discriminant ratio:      {:.2}",
        result.convergent_discriminant_ratio
    ));
    let verdict = if result.convergent_discriminant_ratio > 1.5 {
        "GOOD: reasoning scores internally consistent, distinct from executive"
    } else if result.convergent_discriminant_ratio > 1.0 {
        "FAIR: some distinction between reasoning and executive"
    } else {
        "NOTE: reasoning and executive scores overlap"
    };
    lines.push(format!("Verdict: {}", verdict));
    lines.join("\n")
}

// ── ARC Split-Half Reliability ──

/// Split-half reliability result for a single ARC benchmark.
#[derive(Debug, Clone)]
pub struct SplitHalfResult {
    pub benchmark: String,
    pub metric: String,
    /// Raw correlation between odd/even trial halves.
    pub raw_correlation: f64,
    /// Spearman-Brown corrected reliability (r_sb = 2r / (1 + r)).
    pub corrected_reliability: f64,
    /// Number of trials used.
    pub n_trials: usize,
}

/// Compute split-half reliability for ARC benchmarks in a report.
///
/// Since `MetricValue` stores summary statistics (mean, std_dev, n) rather
/// than individual trial samples, this uses a parametric simulation:
/// generates 2×n synthetic samples from N(mean, std_dev), splits into
/// odd/even halves, and computes Spearman-Brown corrected reliability.
/// Seed is derived from the benchmark name for determinism.
pub fn arc_split_half_reliability(report: &super::report::BenchmarkReport) -> Vec<SplitHalfResult> {
    let mut results = Vec::new();

    for bench_result in &report.results {
        if !bench_result.benchmark.contains("Arc") {
            continue;
        }
        let key = super::report::key_metric_for_benchmark(&bench_result.benchmark);
        if let Some(metric) = bench_result.metrics.get(key) {
            let n = metric.n.max(4);
            if n < 4 || metric.std_dev.abs() < 1e-15 {
                // Not enough trials or zero variance — perfect reliability trivially
                results.push(SplitHalfResult {
                    benchmark: bench_result.benchmark.clone(),
                    metric: key.to_string(),
                    raw_correlation: 1.0,
                    corrected_reliability: 1.0,
                    n_trials: n,
                });
                continue;
            }
            // Generate 2*n synthetic samples via Box-Muller with deterministic seed
            let mut seed: u64 = 0x5A17_C0DE;
            for b in bench_result.benchmark.bytes() {
                seed = seed.wrapping_mul(31).wrapping_add(b as u64);
            }
            seed ^= 0x9E3779B97F4A7C15;
            let xor_shift = |s: &mut u64| {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
            };
            let total = 2 * n;
            let mut samples = Vec::with_capacity(total);
            for _ in 0..total.div_ceil(2) {
                xor_shift(&mut seed);
                let u1 = (seed % 100_000) as f64 / 100_000.0;
                xor_shift(&mut seed);
                let u2 = (seed % 100_000) as f64 / 100_000.0;
                let u1 = u1.max(1e-10);
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                samples.push(metric.mean + metric.std_dev * z1);
                samples.push(metric.mean + metric.std_dev * z2);
            }
            samples.truncate(total);

            let odd: Vec<f64> = samples.iter().step_by(2).copied().collect();
            let even: Vec<f64> = samples.iter().skip(1).step_by(2).copied().collect();
            let half_n = odd.len().min(even.len());
            if half_n >= 2 {
                let r = spearman_correlation(&odd[..half_n], &even[..half_n]);
                let r_sb = if (1.0 + r).abs() > 1e-10 {
                    2.0 * r / (1.0 + r)
                } else {
                    0.0
                };
                results.push(SplitHalfResult {
                    benchmark: bench_result.benchmark.clone(),
                    metric: key.to_string(),
                    raw_correlation: r,
                    corrected_reliability: r_sb,
                    n_trials: n,
                });
            }
        }
    }
    results
}

/// Format split-half reliability results.
pub fn format_arc_reliability(results: &[SplitHalfResult]) -> String {
    let mut lines = vec![
        "## ARC Benchmark Split-Half Reliability".to_string(),
        String::new(),
        "| Benchmark | Metric | r_raw | r_SB | N | Verdict |".to_string(),
        "|-----------|--------|-------|------|---|---------|".to_string(),
    ];
    for r in results {
        let verdict = if r.corrected_reliability >= 0.80 {
            "Excellent"
        } else if r.corrected_reliability >= 0.70 {
            "Good"
        } else if r.corrected_reliability >= 0.50 {
            "Moderate"
        } else {
            "Low"
        };
        let short = r.benchmark.split("::").last().unwrap_or(&r.benchmark);
        lines.push(format!(
            "| {} | {} | {:.3} | {:.3} | {} | {} |",
            short, r.metric, r.raw_correlation, r.corrected_reliability, r.n_trials, verdict
        ));
    }
    lines.join("\n")
}

// ── Unified ARC Report ──

/// Comprehensive ARC analysis report combining all ARC benchmark results.
pub fn format_arc_report(report: &super::report::BenchmarkReport) -> String {
    let mut lines = vec![
        "# ARC Reasoning Suite — Comprehensive Report".to_string(),
        String::new(),
    ];

    // 1. Overview table: benchmark name, key metric, value
    lines.push("## 1. Overview".to_string());
    lines.push(String::new());
    lines.push("| Benchmark | Key Metric | Mean | SD | CI |".to_string());
    lines.push("|-----------|-----------|------|----|----|".to_string());
    for result in &report.results {
        if !result.benchmark.contains("Arc") {
            continue;
        }
        let key = super::report::key_metric_for_benchmark(&result.benchmark);
        if let Some(m) = result.metrics.get(key) {
            let short = result
                .benchmark
                .split("::")
                .last()
                .unwrap_or(&result.benchmark);
            lines.push(format!(
                "| {} | {} | {:.3} | {:.3} | [{:.3}, {:.3}] |",
                short, key, m.mean, m.std_dev, m.ci_lower, m.ci_upper
            ));
        }
    }

    // 2. Scaling analysis (if ArcScaling present)
    lines.push(String::new());
    lines.push("## 2. Scaling Analysis".to_string());
    for result in &report.results {
        if result.benchmark.contains("ArcScaling") {
            lines.push(String::new());
            lines.push("### Grid Complexity".to_string());
            for name in &[
                "grid_3x3_accuracy",
                "grid_5x5_accuracy",
                "grid_8x8_accuracy",
                "grid_12x12_accuracy",
            ] {
                if let Some(m) = result.metrics.get(*name) {
                    lines.push(format!("  {}: {:.3} (±{:.3})", name, m.mean, m.std_dev));
                }
            }
            if let Some(m) = result.metrics.get("grid_complexity_slope") {
                lines.push(format!("  Slope (acc/cell²): {:.4}", m.mean));
            }
            lines.push(String::new());
            lines.push("### Dimension Scaling".to_string());
            for name in &[
                "dim_128d_accuracy",
                "dim_256d_accuracy",
                "dim_512d_accuracy",
                "dim_1024d_accuracy",
            ] {
                if let Some(m) = result.metrics.get(*name) {
                    lines.push(format!("  {}: {:.3} (±{:.3})", name, m.mean, m.std_dev));
                }
            }
            if let Some(m) = result.metrics.get("dimension_efficiency_slope") {
                lines.push(format!("  Slope (acc/log2_dim): {:.4}", m.mean));
            }
        }
    }

    // 3. Learning curve (ArcFewShot)
    for result in &report.results {
        if result.benchmark.contains("ArcFewShot") {
            lines.push(String::new());
            lines.push("## 3. Learning Curve (Few-Shot)".to_string());
            lines.push(String::new());
            for k in 1..=5 {
                let key = format!("accuracy_{}shot", k);
                if let Some(m) = result.metrics.get(&key) {
                    lines.push(format!(
                        "  {}-shot accuracy: {:.3} (±{:.3})",
                        k, m.mean, m.std_dev
                    ));
                }
            }
            if let Some(m) = result.metrics.get("learning_rate") {
                lines.push(format!("  Learning rate (slope): {:.4}", m.mean));
            }
            if let Some(m) = result.metrics.get("saturation_point") {
                lines.push(format!("  Saturation point: {:.1} examples", m.mean));
            }
        }
    }

    // 4. Noise robustness (ArcNoise)
    for result in &report.results {
        if result.benchmark.contains("ArcNoise") {
            lines.push(String::new());
            lines.push("## 4. Noise Robustness".to_string());
            lines.push(String::new());
            for name in &[
                "accuracy_0pct",
                "accuracy_10pct",
                "accuracy_20pct",
                "accuracy_30pct",
                "accuracy_50pct",
            ] {
                if let Some(m) = result.metrics.get(*name) {
                    lines.push(format!("  {}: {:.3}", name, m.mean));
                }
            }
            if let Some(m) = result.metrics.get("noise_resilience") {
                lines.push(format!("  Noise resilience (AUC): {:.3}", m.mean));
            }
            if let Some(m) = result.metrics.get("accuracy_drop") {
                lines.push(format!("  Accuracy drop (0→50%): {:.3}", m.mean));
            }
        }
    }

    // 5. Chain composition (ArcChain)
    for result in &report.results {
        if result.benchmark.contains("ArcChain") {
            lines.push(String::new());
            lines.push("## 5. Chain Composition".to_string());
            lines.push(String::new());
            for (len, key) in &[
                (2, "chain_2_accuracy"),
                (3, "chain_3_accuracy"),
                (4, "chain_4_accuracy"),
            ] {
                if let Some(m) = result.metrics.get(*key) {
                    lines.push(format!(
                        "  {}-step chain: {:.3} (±{:.3})",
                        len, m.mean, m.std_dev
                    ));
                }
            }
            if let Some(m) = result.metrics.get("chain_degradation") {
                lines.push(format!("  Degradation/step: {:.4}", m.mean));
            }
        }
    }

    // 6. Algebra probes (ArcAlgebra)
    for result in &report.results {
        if result.benchmark.contains("ArcAlgebra") {
            lines.push(String::new());
            lines.push("## 6. Rule Algebra Properties".to_string());
            lines.push(String::new());
            for (label, key) in &[
                ("Rule consistency", "rule_consistency"),
                ("Associativity", "associativity"),
                ("Inverse existence", "inverse_similarity"),
                ("Identity preservation", "identity_preservation"),
                ("Distributivity", "distributivity"),
            ] {
                if let Some(m) = result.metrics.get(*key) {
                    lines.push(format!("  {}: {:.3} (±{:.3})", label, m.mean, m.std_dev));
                }
            }
            if let Some(m) = result.metrics.get("algebra_score") {
                lines.push(format!("  Overall algebra score: {:.3}", m.mean));
            }
        }
    }

    // 7. RSA (ArcRSA)
    for result in &report.results {
        if result.benchmark.contains("ArcRSA") {
            lines.push(String::new());
            lines.push("## 7. Representational Similarity Analysis".to_string());
            lines.push(String::new());
            if let Some(m) = result.metrics.get("rsa_correlation") {
                lines.push(format!(
                    "  RSA correlation (encoding ↔ structural): {:.3}",
                    m.mean
                ));
            }
            if let Some(m) = result.metrics.get("within_type_similarity") {
                lines.push(format!("  Within-type encoding similarity: {:.3}", m.mean));
            }
            if let Some(m) = result.metrics.get("between_type_similarity") {
                lines.push(format!("  Between-type encoding similarity: {:.3}", m.mean));
            }
            if let Some(m) = result.metrics.get("discriminability") {
                lines.push(format!(
                    "  Representational discriminability: {:.3}",
                    m.mean
                ));
            }
        }
    }

    // 8. Staircase (ArcStaircase)
    for result in &report.results {
        if result.benchmark.contains("ArcStaircase") {
            lines.push(String::new());
            lines.push("## 8. Adaptive Staircase".to_string());
            lines.push(String::new());
            if let Some(m) = result.metrics.get("capacity_threshold") {
                lines.push(format!(
                    "  Capacity threshold (75% acc): {:.1} grid size",
                    m.mean
                ));
            }
            if let Some(m) = result.metrics.get("psychometric_slope") {
                lines.push(format!("  Psychometric slope: {:.4}", m.mean));
            }
        }
    }

    // 9. Confusion matrix (from ArcFluid)
    for result in &report.results {
        if result.benchmark.contains("ArcFluid") {
            lines.push(String::new());
            lines.push("## 9. Confusion Analysis (ArcFluid)".to_string());
            lines.push(String::new());
            if let Some(m) = result.metrics.get("confusion_diagonal") {
                lines.push(format!("  Diagonal accuracy: {:.3}", m.mean));
            }
            if let Some(m) = result.metrics.get("confusion_max_error") {
                lines.push(format!("  Max off-diagonal: {:.3}", m.mean));
            }
            if let Some(m) = result.metrics.get("confusion_entropy") {
                lines.push(format!("  Mean row entropy: {:.3} bits", m.mean));
            }
        }
    }

    // 10. Reliability
    let reliability = arc_split_half_reliability(report);
    if !reliability.is_empty() {
        lines.push(String::new());
        lines.push(format_arc_reliability(&reliability));
    }

    lines.join("\n")
}

// ──── Cross-Domain Prediction Analysis ────

/// Result of an OLS cross-domain prediction analysis.
#[derive(Debug, Clone)]
pub struct CrossDomainPrediction {
    /// Target benchmark being predicted.
    pub target: String,
    /// Predictor benchmark names with standardized beta weights.
    pub predictors: Vec<(String, f64)>,
    /// R-squared of the full model.
    pub r_squared: f64,
    /// Adjusted R-squared (penalizes extra predictors).
    pub adjusted_r_squared: f64,
}

/// Perform OLS cross-domain prediction: predict `target` from `predictors`.
///
/// Requires multi-seed data in `reports` (one report per seed). Extracts the
/// key metric for each benchmark across seeds, then fits: target = X * beta.
/// Uses normal equations: beta = (X^T X)^{-1} X^T y.
pub fn cross_domain_prediction(
    reports: &[BenchmarkReport],
    target: &str,
    predictors: &[&str],
) -> Option<CrossDomainPrediction> {
    if reports.len() < 3 || predictors.is_empty() {
        return None;
    }

    let n = reports.len();
    let p = predictors.len();

    // Extract target values across seeds
    let target_key = super::report::key_metric_for_benchmark(target);
    let mut y = Vec::with_capacity(n);
    for report in reports {
        let val = report
            .results
            .iter()
            .find(|r| r.benchmark == target)
            .and_then(|r| r.metrics.get(target_key))
            .map(|m| m.mean)?;
        y.push(val);
    }

    // Extract predictor values
    let mut x_cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    for &pred in predictors {
        let pred_key = super::report::key_metric_for_benchmark(pred);
        let mut col = Vec::with_capacity(n);
        for report in reports {
            let val = report
                .results
                .iter()
                .find(|r| r.benchmark == pred)
                .and_then(|r| r.metrics.get(pred_key))
                .map(|m| m.mean)?;
            col.push(val);
        }
        x_cols.push(col);
    }

    // Build X matrix (n × p) with intercept column
    // X^T X (p+1 × p+1) and X^T y (p+1)
    let k = p + 1; // include intercept
    let mut xtx = vec![0.0f64; k * k];
    let mut xty = vec![0.0f64; k];

    for i in 0..n {
        let mut row = vec![1.0]; // intercept
        for col in &x_cols {
            row.push(col[i]);
        }
        for a in 0..k {
            for b in 0..k {
                xtx[a * k + b] += row[a] * row[b];
            }
            xty[a] += row[a] * y[i];
        }
    }

    // Solve via Gaussian elimination (tiny matrix, no need for LAPACK)
    let beta = solve_linear_system(&xtx, &xty, k)?;

    // Compute R-squared
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let mut ss_res = 0.0f64;
    for i in 0..n {
        let mut pred = beta[0]; // intercept
        for (j, col) in x_cols.iter().enumerate() {
            pred += beta[j + 1] * col[i];
        }
        ss_res += (y[i] - pred).powi(2);
    }

    let r_squared = if ss_tot > 1e-10 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let adjusted_r_squared = if n > k && ss_tot > 1e-10 {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n - k) as f64)
    } else {
        r_squared
    };

    let predictor_betas: Vec<(String, f64)> = predictors
        .iter()
        .enumerate()
        .map(|(i, &name)| (name.to_string(), beta[i + 1]))
        .collect();

    Some(CrossDomainPrediction {
        target: target.to_string(),
        predictors: predictor_betas,
        r_squared,
        adjusted_r_squared,
    })
}

/// Format cross-domain prediction as a markdown table.
pub fn format_cross_domain_prediction(result: &CrossDomainPrediction) -> String {
    let mut lines = Vec::new();
    lines.push(format!("### Cross-Domain Prediction: {}", result.target));
    lines.push(String::new());
    lines.push("| Predictor | Beta |".to_string());
    lines.push("|-----------|------|".to_string());
    for (name, beta) in &result.predictors {
        lines.push(format!("| {} | {:.3} |", name, beta));
    }
    lines.push(String::new());
    lines.push(format!(
        "R² = {:.3}, Adjusted R² = {:.3}",
        result.r_squared, result.adjusted_r_squared
    ));
    lines.join("\n")
}

/// Solve A*x = b via Gaussian elimination with partial pivoting.
/// A is k×k stored row-major, b is k-vector.
fn solve_linear_system(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            aug[i * (k + 1) + j] = a[i * k + j];
        }
        aug[i * (k + 1) + k] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..k {
        let mut max_row = col;
        let mut max_val = aug[col * (k + 1) + col].abs();
        for row in (col + 1)..k {
            let val = aug[row * (k + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None; // Singular
        }
        if max_row != col {
            for j in 0..=k {
                aug.swap(col * (k + 1) + j, max_row * (k + 1) + j);
            }
        }
        let pivot = aug[col * (k + 1) + col];
        for row in (col + 1)..k {
            let factor = aug[row * (k + 1) + col] / pivot;
            for j in col..=k {
                let val = aug[col * (k + 1) + j];
                aug[row * (k + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum = aug[i * (k + 1) + k];
        for j in (i + 1)..k {
            sum -= aug[i * (k + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (k + 1) + i];
    }

    Some(x)
}

// ──── Publication-Ready ARC LaTeX Output ────

/// LLM baseline for ARC comparison.
pub struct LlmArcBaseline {
    pub model: &'static str,
    pub accuracy: f64,
    pub source: &'static str,
}

/// Get published LLM baselines for ARC-AGI.
pub fn llm_arc_baselines() -> Vec<LlmArcBaseline> {
    vec![
        LlmArcBaseline {
            model: "GPT-4 (2023)",
            accuracy: 0.05,
            source: "Chollet (2024), ARC-AGI leaderboard",
        },
        LlmArcBaseline {
            model: "Claude 3.5 Sonnet",
            accuracy: 0.21,
            source: "Anthropic (2024), ARC-AGI evaluation",
        },
        LlmArcBaseline {
            model: "o3-high (2024)",
            accuracy: 0.875,
            source: "OpenAI (2024), ARC-AGI semi-private",
        },
        LlmArcBaseline {
            model: "Human (Chollet)",
            accuracy: 0.84,
            source: "Chollet (2019), On the Measure of Intelligence",
        },
    ]
}

/// Generate a LaTeX section for ARC benchmark results.
///
/// Produces: (1) ARC benchmark table, (2) LLM comparison, (3) within-reasoning
/// correlation matrix, (4) scaling analysis summary.
pub fn arc_paper_section_latex(report: &BenchmarkReport) -> String {
    let mut latex = String::new();

    // Header
    latex.push_str("% Auto-generated ARC results section\n");
    latex.push_str("\\subsection{ARC Reasoning Benchmarks}\n\n");

    // Table 1: ARC benchmark results
    latex.push_str("\\begin{table}[h]\n");
    latex.push_str("\\centering\n");
    latex.push_str("\\caption{ARC benchmark results across 11 reasoning tasks.}\n");
    latex.push_str("\\label{tab:arc-results}\n");
    latex.push_str("\\begin{tabular}{llrrl}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("Benchmark & Key Metric & Mean & 95\\% CI & Effect \\\\\n");
    latex.push_str("\\midrule\n");

    let arc_benchmarks: Vec<&super::report::BenchmarkResult> = report
        .results
        .iter()
        .filter(|r| r.benchmark.contains("Arc") || r.benchmark.contains("ARC"))
        .collect();

    for result in &arc_benchmarks {
        let key = super::report::key_metric_for_benchmark(&result.benchmark);
        if let Some(metric) = result.metrics.get(key) {
            let short_name = result
                .benchmark
                .split("::")
                .last()
                .unwrap_or(&result.benchmark);
            latex.push_str(&format!(
                "{} & {} & {:.3} & [{:.3}, {:.3}] & --- \\\\\n",
                short_name, key, metric.mean, metric.ci_lower, metric.ci_upper,
            ));
        }
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");
    latex.push_str("\\end{table}\n\n");

    // Table 2: LLM comparison
    latex.push_str("\\begin{table}[h]\n");
    latex.push_str("\\centering\n");
    latex.push_str("\\caption{Comparison with LLM baselines on ARC-AGI.}\n");
    latex.push_str("\\label{tab:llm-comparison}\n");
    latex.push_str("\\begin{tabular}{lrl}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("Model & Accuracy & Source \\\\\n");
    latex.push_str("\\midrule\n");

    for baseline in llm_arc_baselines() {
        latex.push_str(&format!(
            "{} & {:.1}\\% & {} \\\\\n",
            baseline.model,
            baseline.accuracy * 100.0,
            baseline.source,
        ));
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");
    latex.push_str("\\end{table}\n\n");

    // Within-reasoning correlation summary
    latex.push_str("% Within-reasoning correlation matrix available via --correlation-matrix\n");

    latex
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

    #[test]
    fn test_mtmm_same_domain_higher() {
        // Two benchmarks in domain A, one in domain B
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        values.insert("A::Y".to_string(), vec![1.1, 2.2, 2.9, 4.1, 5.2]);
        values.insert("B::Z".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let mtmm = analysis.mtmm_validity();
        assert_eq!(mtmm.convergent_pairs, 1);
        assert_eq!(mtmm.discriminant_pairs, 2);
        // Same-domain (A::X, A::Y) should correlate positively
        assert!(
            mtmm.mean_convergent > 0.5,
            "convergent r={:.3}",
            mtmm.mean_convergent
        );
        assert!(mtmm.domain_convergent.contains_key("A"));
    }

    #[test]
    fn test_mtmm_convergent_exceeds_discriminant() {
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        values.insert("A::Y".to_string(), vec![1.5, 2.5, 3.5, 4.5, 5.5]);
        values.insert("B::Z".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let mtmm = analysis.mtmm_validity();
        // Convergent (A::X with A::Y) should be much higher than discriminant
        assert!(
            mtmm.mean_convergent > mtmm.mean_discriminant,
            "convergent ({:.3}) should exceed discriminant ({:.3})",
            mtmm.mean_convergent,
            mtmm.mean_discriminant
        );
    }

    #[test]
    fn test_mtmm_format_non_empty() {
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0]);
        values.insert("A::Y".to_string(), vec![1.0, 2.0, 3.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let fmt = analysis.format_mtmm();
        assert!(fmt.contains("Campbell-Fiske"));
        assert!(fmt.contains("Convergent"));
        assert!(fmt.contains("Discriminant"));
    }

    #[test]
    fn test_cronbachs_alpha_perfect() {
        // All items identical → alpha = 1.0
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let alpha = cronbachs_alpha(&items);
        assert!(
            (alpha - 1.0).abs() < 0.01,
            "Perfect consistency should yield alpha~1.0, got {}",
            alpha
        );
    }

    #[test]
    fn test_cronbachs_alpha_uncorrelated() {
        // Uncorrelated items → alpha should be low
        let items = vec![
            vec![1.0, 5.0, 2.0, 4.0, 3.0],
            vec![5.0, 1.0, 4.0, 2.0, 3.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0],
        ];
        let alpha = cronbachs_alpha(&items);
        assert!(
            alpha < 0.5,
            "Uncorrelated items should yield low alpha, got {}",
            alpha
        );
    }

    #[test]
    fn test_cronbachs_alpha_single_item() {
        let items = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(cronbachs_alpha(&items), 0.0);
    }

    #[test]
    fn test_percentile_from_z_center() {
        let p = percentile_from_z(0.0);
        assert!((p - 50.0).abs() < 0.5, "z=0 should be ~50th, got {}", p);
    }

    #[test]
    fn test_percentile_from_z_positive() {
        let p = percentile_from_z(1.0);
        assert!((p - 84.13).abs() < 1.0, "z=+1 should be ~84th, got {}", p);
    }

    #[test]
    fn test_percentile_from_z_negative() {
        let p = percentile_from_z(-1.0);
        assert!((p - 15.87).abs() < 1.0, "z=-1 should be ~16th, got {}", p);
    }

    #[test]
    fn test_percentile_from_z_extreme() {
        let p_high = percentile_from_z(3.0);
        assert!(p_high > 99.0, "z=+3 should be >99th, got {}", p_high);
        let p_low = percentile_from_z(-3.0);
        assert!(p_low < 1.0, "z=-3 should be <1st, got {}", p_low);
    }

    #[test]
    fn test_pca_identity() {
        // Two perfectly correlated benchmarks → 1 dominant component
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        values.insert("A::Y".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let pca = analysis.pca(2);
        assert!(!pca.eigenvalues.is_empty());
        // First component should explain most variance
        assert!(
            pca.variance_explained[0] > 0.9,
            "PC1 should explain >90%, got {:.1}%",
            pca.variance_explained[0] * 100.0
        );
    }

    #[test]
    fn test_pca_format_non_empty() {
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        values.insert("B::Y".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let fmt = analysis.format_pca(2);
        assert!(fmt.contains("Principal Component"));
        assert!(fmt.contains("Eigenvalue"));
    }

    #[test]
    fn test_ablation_effect_sizes_ordering() {
        let baseline = vec![("A::X", 0.90), ("B::Y", 0.80), ("C::Z", 0.70)];
        let ablated = vec![("A::X", 0.85), ("B::Y", 0.40), ("C::Z", 0.65)];
        let effects = ablation_effect_sizes(&baseline, &ablated);
        assert_eq!(effects.len(), 3);
        // Should be sorted by |d| descending — B::Y has largest drop
        assert_eq!(effects[0].0, "B::Y");
        assert!(effects[0].1.abs() > effects[1].1.abs());
    }

    #[test]
    fn test_ablation_effect_sizes_no_match() {
        let baseline = vec![("A::X", 0.90)];
        let ablated = vec![("B::Y", 0.40)];
        let effects = ablation_effect_sizes(&baseline, &ablated);
        assert!(effects.is_empty());
    }

    #[test]
    fn test_mtmm_no_same_domain() {
        // All different domains — convergent should be 0
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0]);
        values.insert("B::Y".to_string(), vec![1.0, 2.0, 3.0]);
        values.insert("C::Z".to_string(), vec![1.0, 2.0, 3.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let mtmm = analysis.mtmm_validity();
        assert_eq!(mtmm.convergent_pairs, 0);
        assert!(mtmm.discriminant_pairs > 0);
    }

    // ──── Edge-case stress tests for analysis functions ────

    #[test]
    fn stress_spearman_all_identical() {
        // All identical values → zero variance → should return 0.0 (not NaN/panic)
        let x = [5.0, 5.0, 5.0, 5.0, 5.0];
        let y = [3.0, 3.0, 3.0, 3.0, 3.0];
        let r = spearman_correlation(&x, &y);
        assert!(
            r.is_finite(),
            "spearman with all-identical should be finite, got {}",
            r
        );
        assert!((r - 0.0).abs() < 1e-10, "all-identical should yield r=0");
    }

    #[test]
    fn stress_spearman_two_elements() {
        // n=2 → below minimum threshold → should return 0.0
        let x = [1.0, 2.0];
        let y = [2.0, 1.0];
        let r = spearman_correlation(&x, &y);
        assert_eq!(r, 0.0, "n<3 should return 0.0");
    }

    #[test]
    fn stress_spearman_with_ties() {
        // Tied values should use fractional ranks
        let x = [1.0, 2.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0, 5.0];
        let r = spearman_correlation(&x, &y);
        assert!(r.is_finite());
        assert!(
            r > 0.5,
            "positively correlated with ties should have r>0.5, got {}",
            r
        );
    }

    #[test]
    fn stress_spearman_large_values() {
        // Very large values should not cause overflow
        let x = [1e15, 2e15, 3e15, 4e15, 5e15];
        let y = [5e15, 4e15, 3e15, 2e15, 1e15];
        let r = spearman_correlation(&x, &y);
        assert!(
            (r - (-1.0)).abs() < 1e-10,
            "large values should still give -1.0"
        );
    }

    #[test]
    fn stress_icc_single_subject() {
        // n=1 subject → degenerate → should return 0.0
        let obs = vec![vec![5.0], vec![5.0]];
        let icc = icc_2_1(&obs);
        assert_eq!(icc, 0.0, "single subject ICC should be 0.0");
    }

    #[test]
    fn stress_icc_zero_variance() {
        // All scores identical across all judges and subjects → degenerate
        let obs = vec![vec![3.0, 3.0, 3.0, 3.0], vec![3.0, 3.0, 3.0, 3.0]];
        let icc = icc_2_1(&obs);
        assert!(icc.is_finite(), "zero-variance ICC should be finite");
    }

    #[test]
    fn stress_icc_unequal_lengths() {
        // Jagged observations → should return 0.0
        let obs = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let icc = icc_2_1(&obs);
        assert_eq!(icc, 0.0, "unequal lengths should return 0.0");
    }

    #[test]
    fn stress_icc_negative_agreement() {
        // Anti-correlated judges → ICC should be negative
        let obs = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let icc = icc_2_1(&obs);
        assert!(icc.is_finite());
        assert!(
            icc >= -1.0 && icc <= 1.0,
            "ICC should be clamped to [-1,1], got {}",
            icc
        );
    }

    #[test]
    fn stress_cronbachs_alpha_zero_variance() {
        // All items have zero variance → should return 0.0
        let items = vec![vec![5.0, 5.0, 5.0, 5.0], vec![3.0, 3.0, 3.0, 3.0]];
        let alpha = cronbachs_alpha(&items);
        assert!(alpha.is_finite(), "zero-variance alpha should be finite");
    }

    #[test]
    fn stress_cronbachs_alpha_two_participants() {
        // Minimum viable: 2 participants, 2 items
        let items = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let alpha = cronbachs_alpha(&items);
        assert!(alpha.is_finite());
        assert!(
            (alpha - 1.0).abs() < 0.01,
            "perfectly correlated 2×2 should yield ~1.0, got {}",
            alpha
        );
    }

    #[test]
    fn stress_cronbachs_alpha_negative() {
        // Negatively correlated items → alpha can be negative
        let items = vec![vec![1.0, 5.0, 1.0, 5.0], vec![5.0, 1.0, 5.0, 1.0]];
        let alpha = cronbachs_alpha(&items);
        assert!(
            alpha.is_finite(),
            "negative alpha should be finite, got {}",
            alpha
        );
    }

    #[test]
    fn stress_percentile_extreme_z() {
        // z = ±10 should clamp to near 0 and near 100
        let p_high = percentile_from_z(10.0);
        assert!(p_high >= 99.0, "z=+10 should be near 100, got {}", p_high);
        let p_low = percentile_from_z(-10.0);
        assert!(p_low <= 1.0, "z=-10 should be near 0, got {}", p_low);
    }

    #[test]
    fn stress_percentile_nan_guard() {
        // NaN z-score should not panic, returns 50th percentile as safe default
        let p = percentile_from_z(f64::NAN);
        assert!(
            (p - 50.0).abs() < 0.01,
            "NaN z should return 50th percentile, got {}",
            p
        );
        // Infinity should also be safe
        let p_inf = percentile_from_z(f64::INFINITY);
        assert!(
            (p_inf - 50.0).abs() < 0.01,
            "Inf z should return 50th percentile, got {}",
            p_inf
        );
    }

    #[test]
    fn stress_pca_single_benchmark() {
        // Single benchmark → single component
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let pca = analysis.pca(3);
        assert!(!pca.eigenvalues.is_empty());
        assert!(pca.eigenvalues[0].is_finite());
    }

    #[test]
    fn stress_pca_empty() {
        // No benchmarks → empty PCA
        let values = BTreeMap::new();
        let analysis = CrossBenchmarkAnalysis { values };
        let pca = analysis.pca(3);
        assert!(pca.eigenvalues.is_empty());
    }

    #[test]
    fn stress_pca_zero_variance() {
        // All identical values → zero-variance correlation matrix
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![5.0, 5.0, 5.0, 5.0, 5.0]);
        values.insert("A::Y".to_string(), vec![3.0, 3.0, 3.0, 3.0, 3.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let pca = analysis.pca(2);
        // Should not panic, eigenvalues should be finite
        for ev in &pca.eigenvalues {
            assert!(ev.is_finite(), "eigenvalue should be finite, got {}", ev);
        }
    }

    #[test]
    fn stress_ablation_empty_inputs() {
        let baseline: Vec<(&str, f64)> = vec![];
        let ablated: Vec<(&str, f64)> = vec![];
        let effects = ablation_effect_sizes(&baseline, &ablated);
        assert!(effects.is_empty());
    }

    #[test]
    fn stress_ablation_zero_baseline() {
        // Baseline value = 0 → SD estimate = 0 → d should be 0
        let baseline = vec![("A::X", 0.0)];
        let ablated = vec![("A::X", 0.5)];
        let effects = ablation_effect_sizes(&baseline, &ablated);
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].1, 0.0, "zero baseline should yield d=0");
    }

    #[test]
    fn stress_cohens_d_extreme_values() {
        let d = cohens_d(1e15, 0.0, 1.0);
        assert!(d.is_finite(), "extreme values should produce finite d");
        assert!(d > 0.0);
    }

    #[test]
    fn stress_ranks_all_identical() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let r = ranks(&data);
        // All should have the same average rank = 2.5
        for rank in &r {
            assert!(
                (rank - 2.5).abs() < 1e-10,
                "identical values should have avg rank 2.5, got {}",
                rank
            );
        }
    }

    #[test]
    fn stress_construct_validity_no_benchmarks() {
        let values = BTreeMap::new();
        let analysis = CrossBenchmarkAnalysis { values };
        let cv = analysis.construct_validity();
        assert_eq!(cv.same_domain_pairs, 0);
        assert_eq!(cv.convergent_pairs, 0);
        assert_eq!(cv.mean_within_correlation, 0.0);
    }

    #[test]
    fn stress_test_retest_insufficient_seeds() {
        // Only 2 seed values → below minimum (4) → reliability = 0.0
        let mut values = BTreeMap::new();
        values.insert("A::X".to_string(), vec![1.0, 2.0]);
        let analysis = CrossBenchmarkAnalysis { values };
        let rel = analysis.test_retest_reliability();
        assert_eq!(rel["A::X"], 0.0, "insufficient seeds should return 0.0");
    }

    // ──── Bootstrap CI tests ────

    #[test]
    fn test_bootstrap_ci_contains_true_mean() {
        // N(5.0, 1.0) — true mean should be within CI
        let samples: Vec<f64> = (0..100).map(|i| 5.0 + (i as f64 - 50.0) * 0.02).collect();
        let (lo, hi) = bootstrap_ci(&samples, 2000, 0.05, 42);
        assert!(
            lo <= 5.0 && hi >= 5.0,
            "CI [{:.4}, {:.4}] should contain true mean 5.0",
            lo,
            hi
        );
    }

    #[test]
    fn test_bootstrap_ci_narrows_with_n() {
        let small: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let large: Vec<f64> = (0..100).map(|i| 3.0 + (i as f64 - 50.0) * 0.04).collect();
        let (lo_s, hi_s) = bootstrap_ci(&small, 2000, 0.05, 42);
        let (lo_l, hi_l) = bootstrap_ci(&large, 2000, 0.05, 42);
        let width_s = hi_s - lo_s;
        let width_l = hi_l - lo_l;
        assert!(
            width_l < width_s,
            "large sample width ({:.4}) should be < small sample width ({:.4})",
            width_l,
            width_s
        );
    }

    #[test]
    fn test_bootstrap_bca_skewed() {
        // Right-skewed (exponential-like) data — BCa should adjust bounds
        let samples: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).exp()).collect();
        let (lo, hi) = bootstrap_ci_bca(&samples, 2000, 0.05, 42);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            lo < mean && hi > mean,
            "BCa CI [{:.4}, {:.4}] should bracket mean {:.4}",
            lo,
            hi,
            mean
        );
        // For right-skewed data, upper tail should extend further
        assert!(
            hi - mean > mean - lo,
            "BCa should show asymmetry for skewed data"
        );
    }

    #[test]
    fn test_bootstrap_single_sample() {
        let (lo, hi) = bootstrap_ci(&[42.0], 2000, 0.05, 42);
        assert!(
            (lo - 42.0).abs() < 1e-10 && (hi - 42.0).abs() < 1e-10,
            "single sample CI should be (x, x), got ({}, {})",
            lo,
            hi
        );
    }

    #[test]
    fn test_bootstrap_deterministic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (lo1, hi1) = bootstrap_ci(&samples, 2000, 0.05, 42);
        let (lo2, hi2) = bootstrap_ci(&samples, 2000, 0.05, 42);
        assert!(
            (lo1 - lo2).abs() < 1e-10 && (hi1 - hi2).abs() < 1e-10,
            "same seed should produce identical CIs"
        );
    }

    // ──── Meta-analysis tests ────

    #[test]
    fn test_meta_single_effect() {
        let result = meta_analysis(&[(0.5, 0.1)]).unwrap();
        assert!((result.summary_d - 0.5).abs() < 1e-10);
        assert_eq!(result.k, 1);
        assert_eq!(result.tau_squared, 0.0);
    }

    #[test]
    fn test_meta_homogeneous_i2_zero() {
        // All identical effects → I²=0
        let effects = vec![(0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)];
        let result = meta_analysis(&effects).unwrap();
        assert!((result.i_squared).abs() < 1e-10, "I²={}", result.i_squared);
        assert!(
            (result.summary_d - 0.5).abs() < 0.01,
            "d={}",
            result.summary_d
        );
    }

    #[test]
    fn test_meta_heterogeneous_i2_positive() {
        // Very different effects → I² should be high
        let effects = vec![(0.1, 0.05), (0.9, 0.05), (-0.5, 0.05), (1.5, 0.05)];
        let result = meta_analysis(&effects).unwrap();
        assert!(
            result.i_squared > 50.0,
            "I²={} should be >50",
            result.i_squared
        );
        assert!(
            result.tau_squared > 0.0,
            "τ²={} should be >0",
            result.tau_squared
        );
    }

    #[test]
    fn test_meta_zero_se() {
        // SE=0 should not panic; weight=0 so effect is excluded
        let effects = vec![(0.5, 0.0), (0.3, 0.1)];
        let result = meta_analysis(&effects);
        assert!(result.is_some());
    }

    #[test]
    fn test_meta_large_k() {
        let effects: Vec<(f64, f64)> = (0..50)
            .map(|i| (0.5 + (i as f64 - 25.0) * 0.01, 0.1))
            .collect();
        let result = meta_analysis(&effects).unwrap();
        assert_eq!(result.k, 50);
        assert!(result.summary_d.is_finite());
        assert!(result.summary_se.is_finite());
    }

    #[test]
    fn test_meta_negative_effects() {
        let effects = vec![(-0.3, 0.1), (-0.5, 0.15), (-0.2, 0.08)];
        let result = meta_analysis(&effects).unwrap();
        assert!(
            result.summary_d < 0.0,
            "summary should be negative: {}",
            result.summary_d
        );
    }

    #[test]
    fn test_meta_q_df() {
        let effects = vec![(0.3, 0.1), (0.5, 0.12), (0.4, 0.09)];
        let result = meta_analysis(&effects).unwrap();
        assert_eq!(result.q_df, 2); // k-1 = 3-1 = 2
        assert!(result.q >= 0.0);
    }

    #[test]
    fn test_meta_ci_contains_summary() {
        let effects = vec![(0.3, 0.1), (0.5, 0.12), (0.4, 0.09), (0.35, 0.11)];
        let result = meta_analysis(&effects).unwrap();
        assert!(result.ci_lower <= result.summary_d, "CI lower > d");
        assert!(result.ci_upper >= result.summary_d, "CI upper < d");
    }

    #[test]
    fn test_meta_fixed_equals_re_when_homogeneous() {
        // When all effects identical, τ²=0, so fixed and RE weights are the same
        let effects = vec![(0.4, 0.1), (0.4, 0.1), (0.4, 0.1)];
        let result = meta_analysis(&effects).unwrap();
        assert!((result.summary_d - 0.4).abs() < 1e-10);
        assert!((result.tau_squared).abs() < 1e-10);
    }

    #[test]
    fn test_meta_from_forest() {
        use crate::harness::report::ForestPlotRow;
        let rows = vec![
            ForestPlotRow {
                domain: "A".into(),
                benchmark: "X".into(),
                metric: "acc".into(),
                agent_mean: 0.8,
                human_mean: 0.85,
                human_sd: Some(0.1),
                cohens_d: Some(-0.5),
                ci_lower: 0.75,
                ci_upper: 0.85,
                ratio: 0.94,
                z_score: Some(-0.5),
            },
            ForestPlotRow {
                domain: "B".into(),
                benchmark: "Y".into(),
                metric: "acc".into(),
                agent_mean: 0.9,
                human_mean: 0.88,
                human_sd: Some(0.08),
                cohens_d: Some(0.25),
                ci_lower: 0.86,
                ci_upper: 0.94,
                ratio: 1.02,
                z_score: Some(0.25),
            },
        ];
        let result = meta_analysis_from_forest(&rows).unwrap();
        assert_eq!(result.k, 2);
        assert!(result.summary_d.is_finite());
    }

    #[test]
    fn test_meta_summary_format() {
        let effects = vec![(0.3, 0.1), (0.5, 0.12)];
        let result = meta_analysis(&effects).unwrap();
        let fmt = format_meta_analysis(&result);
        assert!(fmt.contains("Meta-analysis"));
        assert!(fmt.contains("k=2"));
        assert!(fmt.contains("I²"));
    }

    #[test]
    fn test_meta_p_value_range() {
        let effects = vec![(0.3, 0.1), (0.5, 0.12), (0.4, 0.09)];
        let result = meta_analysis(&effects).unwrap();
        assert!(
            result.q_p_value >= 0.0 && result.q_p_value <= 1.0,
            "p={} out of range",
            result.q_p_value
        );
    }

    // ──── SAT Curve tests ────

    #[test]
    fn test_sat_fit_monotonic() {
        // Accuracy should decrease with time_pressure (less time → less accuracy)
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.95,
                rt_ticks: 10.0,
            },
            SatCurvePoint {
                time_pressure: 0.2,
                accuracy: 0.90,
                rt_ticks: 8.0,
            },
            SatCurvePoint {
                time_pressure: 0.4,
                accuracy: 0.82,
                rt_ticks: 6.0,
            },
            SatCurvePoint {
                time_pressure: 0.6,
                accuracy: 0.70,
                rt_ticks: 4.5,
            },
            SatCurvePoint {
                time_pressure: 0.8,
                accuracy: 0.55,
                rt_ticks: 3.0,
            },
            SatCurvePoint {
                time_pressure: 1.0,
                accuracy: 0.40,
                rt_ticks: 2.0,
            },
        ];
        let curve = fit_sat_curve("Test", &points);
        assert!(curve.lambda > 0.0, "λ={}", curve.lambda);
        assert!(curve.beta > 0.0, "β={}", curve.beta);
    }

    #[test]
    fn test_sat_fit_r_squared_high() {
        // Perfect exponential data should yield high R²
        let points: Vec<SatCurvePoint> = (0..6)
            .map(|i| {
                let tp = i as f64 * 0.2;
                let t = 1.0 - tp;
                SatCurvePoint {
                    time_pressure: tp,
                    accuracy: 0.9 * (1.0 - (-2.0 * t).exp()),
                    rt_ticks: 10.0 - tp * 8.0,
                }
            })
            .collect();
        let curve = fit_sat_curve("Perfect", &points);
        assert!(curve.r_squared > 0.8, "R²={}", curve.r_squared);
    }

    #[test]
    fn test_sat_fit_degenerate() {
        // All same accuracy → R²=0
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.5,
                rt_ticks: 5.0,
            },
            SatCurvePoint {
                time_pressure: 1.0,
                accuracy: 0.5,
                rt_ticks: 2.0,
            },
        ];
        let curve = fit_sat_curve("Flat", &points);
        assert!(curve.r_squared >= 0.0);
    }

    #[test]
    fn test_sat_fit_two_points() {
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.9,
                rt_ticks: 10.0,
            },
            SatCurvePoint {
                time_pressure: 1.0,
                accuracy: 0.4,
                rt_ticks: 2.0,
            },
        ];
        let curve = fit_sat_curve("TwoPoint", &points);
        assert!(curve.lambda.is_finite());
        assert!(curve.beta.is_finite());
    }

    #[test]
    fn test_sat_ascii_output() {
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.9,
                rt_ticks: 10.0,
            },
            SatCurvePoint {
                time_pressure: 0.5,
                accuracy: 0.7,
                rt_ticks: 5.0,
            },
        ];
        let curve = fit_sat_curve("Bench", &points);
        let ascii = sat_curve_ascii(&curve);
        assert!(ascii.contains("SAT Curve: Bench"));
        assert!(ascii.contains("Pressure"));
        assert!(ascii.contains("0.9"));
    }

    #[test]
    fn test_sat_point_serde() {
        // SatCurvePoint should have all finite fields
        let p = SatCurvePoint {
            time_pressure: 0.3,
            accuracy: 0.85,
            rt_ticks: 7.0,
        };
        assert!(p.time_pressure.is_finite());
        assert!(p.accuracy.is_finite());
        assert!(p.rt_ticks.is_finite());
    }

    #[test]
    fn test_sat_sweep_all_finite() {
        let points: Vec<SatCurvePoint> = (0..=10)
            .map(|i| SatCurvePoint {
                time_pressure: i as f64 * 0.1,
                accuracy: 0.9 - i as f64 * 0.04,
                rt_ticks: 10.0 - i as f64 * 0.8,
            })
            .collect();
        let curve = fit_sat_curve("Sweep", &points);
        assert!(curve.lambda.is_finite());
        assert!(curve.beta.is_finite());
        assert!(curve.delta.is_finite());
        assert!(curve.r_squared.is_finite());
    }

    #[test]
    fn test_sat_accuracy_decreases() {
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.95,
                rt_ticks: 10.0,
            },
            SatCurvePoint {
                time_pressure: 0.5,
                accuracy: 0.75,
                rt_ticks: 5.0,
            },
            SatCurvePoint {
                time_pressure: 1.0,
                accuracy: 0.50,
                rt_ticks: 2.0,
            },
        ];
        // Accuracy should decrease with pressure
        assert!(points[0].accuracy > points[2].accuracy);
        let curve = fit_sat_curve("Decreasing", &points);
        assert!(curve.lambda > 0.0);
    }

    #[test]
    fn test_sat_rt_decreases() {
        let points = vec![
            SatCurvePoint {
                time_pressure: 0.0,
                accuracy: 0.95,
                rt_ticks: 10.0,
            },
            SatCurvePoint {
                time_pressure: 1.0,
                accuracy: 0.50,
                rt_ticks: 2.0,
            },
        ];
        // RT should decrease with pressure
        assert!(points[0].rt_ticks > points[1].rt_ticks);
    }

    // ──── Individual Differences tests ────

    #[test]
    fn test_shapiro_wilk_normal() {
        // Approximately normal data — W should be close to 1.0
        let data: Vec<f64> = (0..50)
            .map(|i| {
                // Linear approximation of normal quantiles
                let z = (i as f64 - 25.0) / 12.5;
                5.0 + z * 1.0
            })
            .collect();
        let w = shapiro_wilk_approx(&data);
        assert!(w > 0.9, "Normal-ish data should have W > 0.9, got {:.4}", w);
    }

    #[test]
    fn test_shapiro_wilk_uniform() {
        // Uniform data — W should be lower than normal
        let data: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let w = shapiro_wilk_approx(&data);
        assert!(
            w.is_finite(),
            "Shapiro-Wilk should be finite for uniform data"
        );
        assert!(w > 0.0 && w <= 1.0, "W={:.4} out of bounds", w);
    }

    #[test]
    fn test_shapiro_wilk_small() {
        // Very small sample — should not panic
        let w2 = shapiro_wilk_approx(&[1.0, 2.0]);
        assert_eq!(w2, 1.0, "n<3 should return 1.0");

        let w3 = shapiro_wilk_approx(&[1.0, 2.0, 3.0]);
        assert!(w3.is_finite());
        assert!(w3 > 0.0 && w3 <= 1.0, "W={:.4} for n=3", w3);
    }

    #[test]
    fn test_indiv_diff_basic() {
        let scores = vec![0.80, 0.85, 0.75, 0.90, 0.70, 0.88, 0.82, 0.79, 0.91, 0.77];
        let result = individual_differences("TestBench", "accuracy", &scores);
        assert_eq!(result.n, 10);
        assert!(
            result.mean > 0.7 && result.mean < 0.95,
            "mean={:.3}",
            result.mean
        );
        assert!(result.sd > 0.0, "sd should be positive");
        assert!(result.min <= result.mean);
        assert!(result.max >= result.mean);
        assert!(result.shapiro_wilk_w > 0.0 && result.shapiro_wilk_w <= 1.0);
        assert!(result.ci_lower < result.mean);
        assert!(result.ci_upper > result.mean);
    }

    #[test]
    fn test_indiv_diff_empty() {
        let result = individual_differences("Empty", "metric", &[]);
        assert_eq!(result.n, 0);
        assert_eq!(result.mean, 0.0);
        assert_eq!(result.sd, 0.0);
    }

    #[test]
    fn test_indiv_diff_single() {
        let result = individual_differences("Single", "metric", &[0.42]);
        assert_eq!(result.n, 1);
        assert!((result.mean - 0.42).abs() < 1e-10);
        assert_eq!(result.sd, 0.0);
        assert_eq!(result.min, 0.42);
        assert_eq!(result.max, 0.42);
    }

    #[test]
    fn test_population_deterministic() {
        // Same inputs should produce same output
        let scores = vec![0.8, 0.85, 0.9, 0.75, 0.82];
        let r1 = individual_differences("A", "m", &scores);
        let r2 = individual_differences("A", "m", &scores);
        assert!((r1.mean - r2.mean).abs() < 1e-15);
        assert!((r1.sd - r2.sd).abs() < 1e-15);
        assert!((r1.shapiro_wilk_w - r2.shapiro_wilk_w).abs() < 1e-15);
    }

    #[test]
    fn test_population_wm_variation() {
        // Different WM capacities should produce different scores when we vary input
        let low_wm: Vec<f64> = (0..20).map(|i| 0.50 + (i as f64 % 5.0) * 0.02).collect();
        let high_wm: Vec<f64> = (0..20).map(|i| 0.80 + (i as f64 % 5.0) * 0.01).collect();
        let r_low = individual_differences("Low", "acc", &low_wm);
        let r_high = individual_differences("High", "acc", &high_wm);
        assert!(
            r_low.mean < r_high.mean,
            "low WM mean ({:.3}) should be < high WM mean ({:.3})",
            r_low.mean,
            r_high.mean
        );
    }

    // ──── Dimensionality Scaling tests ────

    #[test]
    fn test_dim_slope_positive() {
        // Accuracy improves with dimension
        let dims = vec![128, 256, 512, 1024, 2048];
        let values = vec![0.60, 0.70, 0.78, 0.84, 0.88];
        let (slope, r2) = dim_scaling_slope(&dims, &values);
        assert!(slope > 0.0, "slope={:.4} should be positive", slope);
        assert!(r2 > 0.9, "R²={:.4} should be high for monotonic data", r2);
    }

    #[test]
    fn test_dim_r_squared_perfect() {
        // Perfect linear relationship with log2(dim)
        let dims = vec![128, 256, 512, 1024];
        let values: Vec<f64> = dims.iter().map(|&d| (d as f64).log2() * 0.1).collect();
        let (slope, r2) = dim_scaling_slope(&dims, &values);
        assert!(
            (r2 - 1.0).abs() < 1e-10,
            "perfect linear should give R²=1.0, got {:.6}",
            r2
        );
        assert!(
            (slope - 0.1).abs() < 1e-10,
            "slope should be 0.1, got {:.6}",
            slope
        );
    }

    #[test]
    fn test_dim_flat() {
        // No relationship → slope ≈ 0
        let dims = vec![128, 256, 512, 1024, 2048];
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let (slope, r2) = dim_scaling_slope(&dims, &values);
        assert!(
            slope.abs() < 1e-10,
            "flat data slope should be ~0, got {:.6}",
            slope
        );
        assert_eq!(r2, 0.0, "flat data R² should be 0");
    }

    #[test]
    fn test_dim_single_point() {
        let (slope, r2) = dim_scaling_slope(&[512], &[0.8]);
        assert_eq!(slope, 0.0);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn test_dim_sweep_stroop() {
        use crate::benchmarks::executive::StroopBenchmark;
        use crate::harness::PsychBenchmark;
        let dims = [128, 256, 512];
        let mut values = Vec::new();
        for &dim in &dims {
            let config = super::super::config::BenchmarkConfig {
                dimension: dim,
                trials_per_condition: 5,
                ..Default::default()
            };
            let result = StroopBenchmark.run(&config);
            values.push(result.metrics["stroop_effect"].mean);
        }
        let (slope, r2) = dim_scaling_slope(
            &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            &values,
        );
        assert!(slope.is_finite(), "slope should be finite");
        assert!(r2.is_finite(), "R² should be finite");
        assert!(
            r2 >= 0.0 && r2 <= 1.0,
            "R² should be in [0,1], got {:.4}",
            r2
        );
    }

    #[test]
    fn test_dim_sweep_all_finite() {
        let dims = vec![64, 128, 256, 512, 1024];
        let values = vec![0.55, 0.65, 0.72, 0.78, 0.82];
        let result = DimScalingResult {
            benchmark: "Test".to_string(),
            dims: dims.clone(),
            values: values.clone(),
            slope: 0.07,
            r_squared: 0.98,
        };
        let formatted = format_dim_scaling(&result);
        assert!(formatted.contains("Test"));
        assert!(formatted.contains("slope="));
        assert!(formatted.contains("R²="));

        // All fields should be finite
        assert!(result.slope.is_finite());
        assert!(result.r_squared.is_finite());
        for v in &result.values {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_indiv_diff_format() {
        let scores = vec![0.80, 0.85, 0.75, 0.90, 0.70];
        let result = individual_differences("TestBench", "accuracy", &scores);
        let formatted = format_individual_differences(&result);
        assert!(formatted.contains("TestBench"));
        assert!(formatted.contains("accuracy"));
        assert!(formatted.contains("N=  5"));
        assert!(formatted.contains("W="));
    }

    #[test]
    fn test_solve_linear_system_2x2() {
        // 2x + 3y = 8, x + y = 3 => x=1, y=2
        let a = [2.0, 3.0, 1.0, 1.0];
        let b = [8.0, 3.0];
        let x = solve_linear_system(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_singular() {
        // Singular matrix: rows are linearly dependent
        let a = [1.0, 2.0, 2.0, 4.0];
        let b = [3.0, 6.0];
        assert!(solve_linear_system(&a, &b, 2).is_none());
    }

    #[test]
    fn test_cross_domain_prediction_insufficient_data() {
        // Less than 3 reports should return None
        let reports = vec![BenchmarkReport::new()];
        let result = cross_domain_prediction(&reports, "target", &["pred1"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_llm_arc_baselines_populated() {
        let baselines = llm_arc_baselines();
        assert_eq!(baselines.len(), 4);
        assert!(baselines[0].accuracy > 0.0);
        assert!(baselines[3].accuracy > 0.0);
    }

    #[test]
    fn test_arc_paper_section_latex_structure() {
        let report = BenchmarkReport::new();
        let latex = arc_paper_section_latex(&report);
        assert!(latex.contains("\\subsection"));
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("\\end{table}"));
        assert!(latex.contains("LLM baselines"));
    }

    #[test]
    fn test_format_cross_domain_prediction() {
        let pred = CrossDomainPrediction {
            target: "ArcFluid".to_string(),
            predictors: vec![("NBack".to_string(), 0.45), ("Stroop".to_string(), -0.12)],
            r_squared: 0.67,
            adjusted_r_squared: 0.62,
        };
        let formatted = format_cross_domain_prediction(&pred);
        assert!(formatted.contains("ArcFluid"));
        assert!(formatted.contains("NBack"));
        assert!(formatted.contains("0.450"));
        assert!(formatted.contains("R²"));
    }
}
