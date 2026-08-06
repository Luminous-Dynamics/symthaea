// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trial-level neuromodulator telemetry correlation analysis.
//!
//! Correlates per-trial DA/NE/5-HT/ACh levels with behavioral metrics
//! (RT, accuracy, error bursts) from live cognitive loop runs.
//!
//! Connects Phase 1A (per-trial traces) with Phase 4 (neuromod profiles).

use super::analysis::normal_cdf;
use super::live_runner::LoopTrialResult;
use serde::{Deserialize, Serialize};

/// Pearson correlation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correlation {
    /// Variable X name.
    pub x_name: String,
    /// Variable Y name.
    pub y_name: String,
    /// Pearson r coefficient (-1.0 to 1.0).
    pub r: f64,
    /// Number of data points.
    pub n: usize,
    /// t-statistic for significance testing.
    pub t_stat: f64,
    /// Two-tailed p-value (approximate for large N).
    pub p_value: f64,
    /// Effect size interpretation.
    pub effect: EffectSize,
}

/// Effect size interpretation (Cohen, 1988).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectSize {
    /// |r| < 0.10
    Negligible,
    /// 0.10 <= |r| < 0.30
    Small,
    /// 0.30 <= |r| < 0.50
    Medium,
    /// |r| >= 0.50
    Large,
}

impl EffectSize {
    fn from_r(r: f64) -> Self {
        let abs_r = r.abs();
        if abs_r < 0.10 {
            EffectSize::Negligible
        } else if abs_r < 0.30 {
            EffectSize::Small
        } else if abs_r < 0.50 {
            EffectSize::Medium
        } else {
            EffectSize::Large
        }
    }
}

/// Compute Pearson r between two vectors, returning (r, n).
fn pearson_r(x: &[f64], y: &[f64]) -> (f64, usize) {
    let n = x.len().min(y.len());
    let r = super::reliability_analysis::pearson_r(x, y);
    (r, n)
}

/// Compute t-statistic and approximate p-value for a correlation.
fn significance(r: f64, n: usize) -> (f64, f64) {
    if n < 3 || (r.abs() - 1.0).abs() < 1e-10 {
        return (0.0, 1.0);
    }

    let df = n as f64 - 2.0;
    let t = r * (df / (1.0 - r * r)).sqrt();

    // Approximate two-tailed p-value using t-distribution
    // For large N, use normal approximation
    let p = if df > 30.0 {
        // Normal approximation: 2 * (1 - Phi(|t|))
        let z = t.abs();
        2.0 * (1.0 - normal_cdf(z))
    } else {
        // Rough approximation for small df
        let x = df / (df + t * t);
        incomplete_beta(df / 2.0, 0.5, x)
    };

    (t, p.clamp(0.0, 1.0))
}

/// Incomplete beta function approximation (for p-value with small df).
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Simple numerical integration (trapezoidal, 100 points)
    let n = 100;
    let dx = x / n as f64;
    let mut sum = 0.0f64;
    for i in 0..n {
        let t0 = i as f64 * dx;
        let t1 = (i + 1) as f64 * dx;
        let f0 = t0.powf(a - 1.0) * (1.0 - t0).powf(b - 1.0);
        let f1 = t1.powf(a - 1.0) * (1.0 - t1).powf(b - 1.0);
        sum += (f0 + f1) * dx / 2.0;
    }
    // Normalize by beta function (approximated)
    let beta = gamma_approx(a) * gamma_approx(b) / gamma_approx(a + b);
    (sum / beta).clamp(0.0, 1.0)
}

/// Stirling's approximation for ln(Gamma(x)).
fn gamma_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_gamma =
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x);
    ln_gamma.exp()
}

/// Compute a single correlation between a named neuromod signal and a behavioral metric.
fn compute_correlation(x_name: &str, y_name: &str, x: &[f64], y: &[f64]) -> Correlation {
    let (r, n) = pearson_r(x, y);
    let (t_stat, p_value) = significance(r, n);
    Correlation {
        x_name: x_name.to_string(),
        y_name: y_name.to_string(),
        r,
        n,
        t_stat,
        p_value,
        effect: EffectSize::from_r(r),
    }
}

/// Full neuromod–behavior correlation matrix from loop trial results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodCorrelationMatrix {
    /// All computed correlations.
    pub correlations: Vec<Correlation>,
}

impl NeuromodCorrelationMatrix {
    /// Compute correlations from loop trial results.
    ///
    /// Correlates DA, NE, 5-HT, ACh with:
    /// - RT (cycle_time_us)
    /// - Accuracy (correct as 0/1)
    /// - Prediction error
    pub fn from_trials(trials: &[LoopTrialResult]) -> Self {
        if trials.is_empty() {
            return Self {
                correlations: Vec::new(),
            };
        }

        let da: Vec<f64> = trials.iter().map(|t| t.da as f64).collect();
        let ne: Vec<f64> = trials.iter().map(|t| t.ne as f64).collect();
        let sht: Vec<f64> = trials.iter().map(|t| t.sht as f64).collect();
        let ach: Vec<f64> = trials.iter().map(|t| t.ach as f64).collect();
        let rt: Vec<f64> = trials.iter().map(|t| t.cycle_time_us as f64).collect();
        let acc: Vec<f64> = trials
            .iter()
            .map(|t| if t.correct { 1.0 } else { 0.0 })
            .collect();
        let pe: Vec<f64> = trials.iter().map(|t| t.prediction_error as f64).collect();

        let neuromods = [("DA", &da), ("NE", &ne), ("5-HT", &sht), ("ACh", &ach)];
        let behaviors = [("RT", &rt), ("Accuracy", &acc), ("PredError", &pe)];

        let mut correlations = Vec::new();
        for (nm_name, nm_vals) in &neuromods {
            for (beh_name, beh_vals) in &behaviors {
                correlations.push(compute_correlation(nm_name, beh_name, nm_vals, beh_vals));
            }
        }

        Self { correlations }
    }

    /// Get significant correlations (p < alpha).
    pub fn significant(&self, alpha: f64) -> Vec<&Correlation> {
        self.correlations
            .iter()
            .filter(|c| c.p_value < alpha)
            .collect()
    }

    /// Get correlations with at least medium effect size.
    pub fn medium_or_larger(&self) -> Vec<&Correlation> {
        self.correlations
            .iter()
            .filter(|c| matches!(c.effect, EffectSize::Medium | EffectSize::Large))
            .collect()
    }

    /// Format as markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("## Neuromodulator × Behavior Correlations\n\n");
        out.push_str("| Neuromod | Behavior | r | N | t | p | Effect |\n");
        out.push_str("|----------|----------|------|------|------|------|--------|\n");
        for c in &self.correlations {
            let effect_str = match c.effect {
                EffectSize::Negligible => "-",
                EffectSize::Small => "S",
                EffectSize::Medium => "M",
                EffectSize::Large => "L",
            };
            let sig = if c.p_value < 0.05 { "*" } else { "" };
            out.push_str(&format!(
                "| {} | {} | {:.3}{} | {} | {:.2} | {:.3} | {} |\n",
                c.x_name, c.y_name, c.r, sig, c.n, c.t_stat, c.p_value, effect_str
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trials(n: usize) -> Vec<LoopTrialResult> {
        (0..n)
            .map(|i| LoopTrialResult {
                response_idx: 0,
                correct: i % 3 != 0, // 67% accuracy
                prediction_error: 0.5 - (i as f32 * 0.01),
                da: 0.3 + (i as f32 * 0.02),
                ne: 0.5,
                sht: 0.5,
                ach: 0.5,
                psi: 0.3,
                cycle_time_us: 1000 + (i as u64 * 50),
                learning_occurred: i % 5 == 0,
                reward: 0.0,
                // Added 2026-07-31 to unblock the crate: another session's
                // LoopTrialResult gained `cycle_reward` and this test fixture was
                // collateral damage. Synthetic data, so 0.0 matches the
                // neighbouring `reward`/`moral_score` fixtures.
                cycle_reward: 0.0,
                oxytocin: 0.3,
                moral_score: 0.0,
                bath_entropy: 0.5,
                allostatic_load: 0.1,
                consciousness_mod: 1.0,
                social_coherence: 1.0,
                structural_micro_phi: 0.0,
                structural_emergence_ratio: 0.0,
            })
            .collect()
    }

    #[test]
    fn test_pearson_r_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (r, n) = pearson_r(&x, &y);
        assert_eq!(n, 5);
        assert!((r - 1.0).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_pearson_r_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let (r, _) = pearson_r(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_pearson_r_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // constant
        let (r, _) = pearson_r(&x, &y);
        assert!((r).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_effect_size_classification() {
        assert_eq!(EffectSize::from_r(0.05), EffectSize::Negligible);
        assert_eq!(EffectSize::from_r(0.15), EffectSize::Small);
        assert_eq!(EffectSize::from_r(0.35), EffectSize::Medium);
        assert_eq!(EffectSize::from_r(0.65), EffectSize::Large);
        assert_eq!(EffectSize::from_r(-0.55), EffectSize::Large);
    }

    #[test]
    fn test_correlation_matrix_from_trials() {
        let trials = make_trials(30);
        let matrix = NeuromodCorrelationMatrix::from_trials(&trials);
        // 4 neuromods × 3 behaviors = 12 correlations
        assert_eq!(matrix.correlations.len(), 12);
        for c in &matrix.correlations {
            assert!(c.r.is_finite());
            assert!(c.r >= -1.0 && c.r <= 1.0);
            assert_eq!(c.n, 30);
        }
    }

    #[test]
    fn test_da_rt_correlation_direction() {
        // DA increases with trial index, RT increases with trial index
        // So DA-RT should be positively correlated
        let trials = make_trials(50);
        let matrix = NeuromodCorrelationMatrix::from_trials(&trials);
        let da_rt = matrix
            .correlations
            .iter()
            .find(|c| c.x_name == "DA" && c.y_name == "RT")
            .unwrap();
        assert!(da_rt.r > 0.0, "DA-RT should be positive: r={}", da_rt.r);
    }

    #[test]
    fn test_significant_filter() {
        let trials = make_trials(100);
        let matrix = NeuromodCorrelationMatrix::from_trials(&trials);
        let sig = matrix.significant(0.05);
        for c in &sig {
            assert!(c.p_value < 0.05);
        }
    }

    #[test]
    fn test_empty_trials() {
        let matrix = NeuromodCorrelationMatrix::from_trials(&[]);
        assert!(matrix.correlations.is_empty());
    }

    #[test]
    fn test_markdown_output() {
        let trials = make_trials(20);
        let matrix = NeuromodCorrelationMatrix::from_trials(&trials);
        let md = matrix.to_markdown();
        assert!(md.contains("DA"));
        assert!(md.contains("RT"));
        assert!(md.contains("Neuromod"));
    }
}
