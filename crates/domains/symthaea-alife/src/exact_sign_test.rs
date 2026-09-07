// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact one-sided paired sign test for small deterministic ALife seed panels.
//!
//! This module deliberately tests direction only, not effect magnitude. For paired differences
//! `d_i`, the preregistered alternative is `P(d_i > 0) > 0.5`. Exact zeroes are ties and are
//! excluded from the binomial denominator. No normal approximation is used.
//!
//! The implementation retains the exact binomial-tail numerator and denominator in `u128` so the
//! fixed alpha=0.05 decision does not depend on floating-point rounding. The public floating
//! p-value is descriptive convenience only.

/// Maximum non-tied panel size supported by the exact integer implementation.
///
/// At this cap the largest possible tail numerator/denominator is `2^120`. The exact alpha check
/// multiplies the tail by 20, so its worst case is `20 * 2^120 < 2^125 < 2^128`. The iterative
/// binomial-coefficient intermediates also remain inside `u128` for `n <= 120`.
pub const MAX_EXACT_SIGN_TEST_NON_TIES: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactPositiveSignTestV1 {
    pub total_observations: usize,
    pub positive: usize,
    pub negative: usize,
    pub ties: usize,
    pub non_ties: usize,
    /// Exact numerator of `P[X >= positive]` for `X ~ Binomial(non_ties, 0.5)`.
    pub tail_numerator: u128,
    /// Exact denominator, `2^non_ties`.
    pub denominator: u128,
    pub one_sided_p_value: f64,
    /// Exact decision for alpha = 1/20 = 0.05.
    pub reject_at_alpha_0_05: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactSignTestErrorV1 {
    NonFiniteObservation { index: usize },
    TooManyNonTies { observed: usize, maximum: usize },
}

/// Exact one-sided sign test for the preregistered positive-direction alternative.
///
/// Positive observations count as candidate-favoring, negative observations as
/// reference-favoring, and exact `+0.0`/`-0.0` values as ties. When all observations are ties,
/// the exact p-value is 1 and the null is not rejected.
pub fn exact_positive_sign_test(
    observations: &[f64],
) -> Result<ExactPositiveSignTestV1, ExactSignTestErrorV1> {
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut ties = 0usize;

    for (index, &value) in observations.iter().enumerate() {
        if !value.is_finite() {
            return Err(ExactSignTestErrorV1::NonFiniteObservation { index });
        }
        if value > 0.0 {
            positive += 1;
        } else if value < 0.0 {
            negative += 1;
        } else {
            ties += 1;
        }
    }

    let non_ties = positive + negative;
    if non_ties > MAX_EXACT_SIGN_TEST_NON_TIES {
        return Err(ExactSignTestErrorV1::TooManyNonTies {
            observed: non_ties,
            maximum: MAX_EXACT_SIGN_TEST_NON_TIES,
        });
    }

    let denominator = 1u128 << non_ties;
    let tail_numerator = (positive..=non_ties)
        .map(|k| binomial_coefficient(non_ties, k))
        .sum::<u128>();
    let one_sided_p_value = tail_numerator as f64 / denominator as f64;
    let reject_at_alpha_0_05 = tail_numerator * 20 <= denominator;

    Ok(ExactPositiveSignTestV1 {
        total_observations: observations.len(),
        positive,
        negative,
        ties,
        non_ties,
        tail_numerator,
        denominator,
        one_sided_p_value,
        reject_at_alpha_0_05,
    })
}

fn binomial_coefficient(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        result = result * (n - k + i) as u128 / i as u128;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_of_eight_positive_is_exactly_one_over_256() {
        let result = exact_positive_sign_test(&[1.0; 8]).expect("finite panel");
        assert_eq!(result.positive, 8);
        assert_eq!(result.negative, 0);
        assert_eq!(result.ties, 0);
        assert_eq!(result.tail_numerator, 1);
        assert_eq!(result.denominator, 256);
        assert_eq!(result.one_sided_p_value, 1.0 / 256.0);
        assert!(result.reject_at_alpha_0_05);
    }

    #[test]
    fn seven_of_eight_positive_passes_exact_point_zero_five_gate() {
        let result = exact_positive_sign_test(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
            .expect("finite panel");
        assert_eq!(result.tail_numerator, 9);
        assert_eq!(result.denominator, 256);
        assert!(result.reject_at_alpha_0_05);
    }

    #[test]
    fn six_of_eight_positive_does_not_pass_gate() {
        let result = exact_positive_sign_test(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
            .expect("finite panel");
        assert_eq!(result.tail_numerator, 37);
        assert_eq!(result.denominator, 256);
        assert!(!result.reject_at_alpha_0_05);
    }

    #[test]
    fn exact_zeroes_are_ties_and_leave_a_smaller_exact_denominator() {
        let result = exact_positive_sign_test(&[1.0, 1.0, 0.0, -0.0, -1.0])
            .expect("finite panel");
        assert_eq!(result.positive, 2);
        assert_eq!(result.negative, 1);
        assert_eq!(result.ties, 2);
        assert_eq!(result.non_ties, 3);
        assert_eq!(result.tail_numerator, 4); // C(3,2) + C(3,3)
        assert_eq!(result.denominator, 8);
        assert!(!result.reject_at_alpha_0_05);
    }

    #[test]
    fn all_ties_return_p_one() {
        let result = exact_positive_sign_test(&[0.0, -0.0]).expect("finite panel");
        assert_eq!(result.non_ties, 0);
        assert_eq!(result.tail_numerator, 1);
        assert_eq!(result.denominator, 1);
        assert_eq!(result.one_sided_p_value, 1.0);
        assert!(!result.reject_at_alpha_0_05);
    }

    #[test]
    fn non_finite_observation_fails_closed() {
        assert_eq!(
            exact_positive_sign_test(&[1.0, f64::NAN]),
            Err(ExactSignTestErrorV1::NonFiniteObservation { index: 1 })
        );
    }

    #[test]
    fn maximum_supported_all_negative_panel_keeps_exact_worst_case_tail_in_range() {
        let observations = [-1.0; MAX_EXACT_SIGN_TEST_NON_TIES];
        let result = exact_positive_sign_test(&observations).expect("maximum supported panel");
        let denominator = 1u128 << MAX_EXACT_SIGN_TEST_NON_TIES;
        assert_eq!(result.positive, 0);
        assert_eq!(result.negative, MAX_EXACT_SIGN_TEST_NON_TIES);
        assert_eq!(result.tail_numerator, denominator);
        assert_eq!(result.denominator, denominator);
        assert_eq!(result.one_sided_p_value, 1.0);
        assert!(!result.reject_at_alpha_0_05);
        assert!(result.tail_numerator.checked_mul(20).is_some());
    }

    #[test]
    fn maximum_supported_all_positive_panel_is_exactly_one_over_two_to_120() {
        let observations = [1.0; MAX_EXACT_SIGN_TEST_NON_TIES];
        let result = exact_positive_sign_test(&observations).expect("maximum supported panel");
        assert_eq!(result.positive, MAX_EXACT_SIGN_TEST_NON_TIES);
        assert_eq!(result.tail_numerator, 1);
        assert_eq!(
            result.denominator,
            1u128 << MAX_EXACT_SIGN_TEST_NON_TIES
        );
        assert!(result.reject_at_alpha_0_05);
    }

    #[test]
    fn panel_above_exact_integer_cap_fails_before_binomial_arithmetic() {
        let observations = vec![1.0; MAX_EXACT_SIGN_TEST_NON_TIES + 1];
        assert_eq!(
            exact_positive_sign_test(&observations),
            Err(ExactSignTestErrorV1::TooManyNonTies {
                observed: MAX_EXACT_SIGN_TEST_NON_TIES + 1,
                maximum: MAX_EXACT_SIGN_TEST_NON_TIES,
            })
        );
    }
}
