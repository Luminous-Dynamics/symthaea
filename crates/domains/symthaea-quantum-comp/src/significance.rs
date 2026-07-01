//! Small dependency-free significance helpers for replicated probe reports.
//!
//! These helpers are intentionally conservative. They are designed to support
//! lightweight local audits and lab notes, not publication-grade inference.

use crate::statistics::SampleSummary;

/// Summary of paired differences between two methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedDifferenceSummary {
    /// Number of paired observations included in the summary.
    pub count: usize,
    /// Number of pairs where `a > b` by more than `tie_tolerance`.
    pub a_wins: usize,
    /// Number of pairs where `b > a` by more than `tie_tolerance`.
    pub b_wins: usize,
    /// Number of pairs treated as ties.
    pub ties: usize,
    /// Summary of `a - b` differences.
    pub delta: SampleSummary,
    /// Two-sided exact sign-test p-value ignoring ties.
    pub sign_test_p_two_sided: Option<f64>,
}

impl PairedDifferenceSummary {
    /// Builds a paired difference summary.
    pub fn from_pairs(a: &[f32], b: &[f32], tie_tolerance: f32) -> Option<Self> {
        if a.len() != b.len() || a.is_empty() || tie_tolerance < 0.0 {
            return None;
        }
        let mut diffs = Vec::with_capacity(a.len());
        let mut a_wins = 0usize;
        let mut b_wins = 0usize;
        let mut ties = 0usize;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = x - y;
            diffs.push(d);
            if d > tie_tolerance {
                a_wins += 1;
            } else if d < -tie_tolerance {
                b_wins += 1;
            } else {
                ties += 1;
            }
        }
        let delta = SampleSummary::from_samples(&diffs)?;
        let effective_n = a_wins + b_wins;
        let sign_test_p_two_sided = if effective_n == 0 {
            None
        } else {
            Some(exact_two_sided_sign_test_p_value(
                a_wins.min(b_wins),
                effective_n,
            ))
        };
        Some(Self {
            count: a.len(),
            a_wins,
            b_wins,
            ties,
            delta,
            sign_test_p_two_sided,
        })
    }

    /// Returns a compact text summary.
    pub fn to_text(&self, label_a: &str, label_b: &str) -> String {
        let (lo, hi) = self.delta.approximate_95_ci();
        format!(
            "{label_a}_vs_{label_b}: n={} {label_a}_wins={} {label_b}_wins={} ties={} mean_delta={:.6} ci95=[{:.6},{:.6}] sign_test_p_two_sided={:?}",
            self.count,
            self.a_wins,
            self.b_wins,
            self.ties,
            self.delta.mean,
            lo,
            hi,
            self.sign_test_p_two_sided,
        )
    }
}

/// Computes a conservative exact two-sided sign-test p-value for a fair coin.
///
/// The input `minority_successes` should be `min(wins_a, wins_b)`, and `n` is
/// the number of non-tied paired observations.
pub fn exact_two_sided_sign_test_p_value(minority_successes: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let tail = (0..=minority_successes)
        .map(|k| binomial_probability(n, k))
        .sum::<f64>();
    (2.0 * tail).min(1.0)
}

fn binomial_probability(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut coeff = 1.0f64;
    for i in 0..k {
        coeff *= (n - i) as f64;
        coeff /= (i + 1) as f64;
    }
    coeff / 2.0f64.powi(n as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_reports_extreme_difference() {
        let p = exact_two_sided_sign_test_p_value(0, 4);
        assert!((p - 0.125).abs() < 1e-12);
    }

    #[test]
    fn paired_summary_counts_wins_and_ties() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.0, 2.0, 2.0, 5.0];
        let s = PairedDifferenceSummary::from_pairs(&a, &b, 1e-6).unwrap();
        assert_eq!(s.a_wins, 2);
        assert_eq!(s.b_wins, 1);
        assert_eq!(s.ties, 1);
    }
}
