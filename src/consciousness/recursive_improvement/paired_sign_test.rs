// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Dependency-free exact paired sign-test arithmetic shared by RSI analysis
//! and the independent REUSE evaluation gate. The p-value is crate-internal;
//! candidate-facing REUSE APIs must return only a coarse decision.

/// Exact one-sided sign-test upper-tail probability under Binomial(n, 1/2).
///
/// Returns None when the counts overflow or there are no non-tied pairs.
/// Ties are excluded from n. The caller decides how an all-tie comparison is
/// treated; REUSE defines it as p=1 and therefore rejects it.
pub(crate) fn exact_sign_test_upper_tail(wins: usize, losses: usize) -> Option<f64> {
    let n = wins.checked_add(losses)?;
    if n == 0 {
        return None;
    }
    if wins == 0 {
        return Some(1.0);
    }

    let log_probability_at = |k: usize| -> f64 {
        let terms = k.min(n - k);
        let mut log_choose = 0.0f64;
        for i in 1..=terms {
            log_choose += ((n - terms + i) as f64).ln() - (i as f64).ln();
        }
        log_choose - (n as f64) * std::f64::consts::LN_2
    };

    if wins > n / 2 {
        // Sum the upper tail from P(X = wins), then scale by its log probability.
        let mut relative_term = 1.0f64;
        let mut relative_sum = 1.0f64;
        for k in wins..n {
            relative_term *= (n - k) as f64 / (k + 1) as f64;
            relative_sum += relative_term;
        }
        Some((log_probability_at(wins).exp() * relative_sum).clamp(0.0, 1.0))
    } else {
        // The requested tail is at least one half. Compute its complement
        // from P(X = wins - 1), which is in the lower half of the distribution.
        let boundary = wins - 1;
        let mut relative_term = 1.0f64;
        let mut relative_sum = 1.0f64;
        for k in (1..=boundary).rev() {
            relative_term *= k as f64 / (n - k + 1) as f64;
            relative_sum += relative_term;
        }
        let lower_tail = log_probability_at(boundary).exp() * relative_sum;
        Some((1.0 - lower_tail).clamp(0.0, 1.0))
    }
}


#[cfg(test)]
mod tests {
    use super::exact_sign_test_upper_tail;

    #[test]
    fn exact_upper_tail_matches_small_binomial_fixtures() {
        assert!((exact_sign_test_upper_tail(4, 0).unwrap() - 1.0 / 16.0).abs() < 1e-12);
        assert!((exact_sign_test_upper_tail(3, 1).unwrap() - 5.0 / 16.0).abs() < 1e-12);
        assert_eq!(exact_sign_test_upper_tail(0, 3), Some(1.0));
    }

    #[test]
    fn zero_disagreements_has_no_sign_test_estimate() {
        assert_eq!(exact_sign_test_upper_tail(0, 0), None);
    }
}
