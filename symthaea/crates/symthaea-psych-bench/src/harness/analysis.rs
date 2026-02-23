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
                    let entry = domain_sums
                        .entry(domain_i.to_string())
                        .or_insert((0.0, 0));
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
    let total_variance =
        total_scores.iter().map(|x| (x - total_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

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
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
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
            lines.push(format!(
                "  {:<25} {:>8} {:>8}",
                "Benchmark", "PC1", "PC2"
            ));
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

        // eigenvalue = v^T * w
        let _eigenval: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

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
        assert!(alpha < 0.5, "Uncorrelated items should yield low alpha, got {}", alpha);
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
        assert!(r.is_finite(), "spearman with all-identical should be finite, got {}", r);
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
        assert!(r > 0.5, "positively correlated with ties should have r>0.5, got {}", r);
    }

    #[test]
    fn stress_spearman_large_values() {
        // Very large values should not cause overflow
        let x = [1e15, 2e15, 3e15, 4e15, 5e15];
        let y = [5e15, 4e15, 3e15, 2e15, 1e15];
        let r = spearman_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-10, "large values should still give -1.0");
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
        let obs = vec![
            vec![3.0, 3.0, 3.0, 3.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ];
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
        assert!(icc >= -1.0 && icc <= 1.0, "ICC should be clamped to [-1,1], got {}", icc);
    }

    #[test]
    fn stress_cronbachs_alpha_zero_variance() {
        // All items have zero variance → should return 0.0
        let items = vec![
            vec![5.0, 5.0, 5.0, 5.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ];
        let alpha = cronbachs_alpha(&items);
        assert!(alpha.is_finite(), "zero-variance alpha should be finite");
    }

    #[test]
    fn stress_cronbachs_alpha_two_participants() {
        // Minimum viable: 2 participants, 2 items
        let items = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        let alpha = cronbachs_alpha(&items);
        assert!(alpha.is_finite());
        assert!((alpha - 1.0).abs() < 0.01, "perfectly correlated 2×2 should yield ~1.0, got {}", alpha);
    }

    #[test]
    fn stress_cronbachs_alpha_negative() {
        // Negatively correlated items → alpha can be negative
        let items = vec![
            vec![1.0, 5.0, 1.0, 5.0],
            vec![5.0, 1.0, 5.0, 1.0],
        ];
        let alpha = cronbachs_alpha(&items);
        assert!(alpha.is_finite(), "negative alpha should be finite, got {}", alpha);
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
        assert!((p - 50.0).abs() < 0.01, "NaN z should return 50th percentile, got {}", p);
        // Infinity should also be safe
        let p_inf = percentile_from_z(f64::INFINITY);
        assert!((p_inf - 50.0).abs() < 0.01, "Inf z should return 50th percentile, got {}", p_inf);
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
            assert!((rank - 2.5).abs() < 1e-10, "identical values should have avg rank 2.5, got {}", rank);
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
}
