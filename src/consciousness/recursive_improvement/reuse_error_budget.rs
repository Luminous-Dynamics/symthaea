// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pure allocation arithmetic for REUSE-style adaptive evaluation.
//!
//! This module computes one per-round/per-promotion-count error allocation.
//! It is deliberately **not** a sequential test, holdout service, promotion
//! gate, proof of a simultaneous guarantee, or candidate-facing feedback API.
//! A complete implementation must enforce the decision-only feedback
//! assumption, persist the full adaptive history, bind the audit corpus and
//! grader, and independently verify every decision outside the editable agent.
//!
//! The allocation follows the REUSE schedule:
//!
//!   delta[t,p] = alpha * w[t] * v[p]
//!                / (C(t-1,p) * K^(p+1))
//!
//! where w[t] = 1/(t(t+1)), v[p] = 1/((p+1)(p+2)), t is one-based,
//! p is the number of earlier promotions, and K is the maximum candidates
//! considered in the round. The allocated delta is split between the paired
//! test and the cumulative lower-bound certificate by rho.
//!
//! Extremely small allocations that underflow binary64 are rejected. They are
//! never rounded to zero because zero would falsely imply an error-free test.

use std::fmt;

/// Bound on round indexing for the bounded log-binomial implementation.
/// Larger histories require a reviewed arbitrary-precision/log-gamma path.
pub const MAX_REUSE_ROUND: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseBudgetError {
    InvalidAlpha,
    InvalidRho,
    InvalidRound,
    InvalidCandidateLimit,
    PromotionCountExceedsHistory,
    HistoryTooLarge,
    AllocationUnderflow,
}

impl fmt::Display for ReuseBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be finite and strictly between 0 and 1"),
            Self::InvalidRho => write!(f, "rho must be finite and strictly between 0 and 1"),
            Self::InvalidRound => write!(f, "round is one-based and must be at least 1"),
            Self::InvalidCandidateLimit => write!(f, "candidate limit K must be at least 1"),
            Self::PromotionCountExceedsHistory => {
                write!(f, "prior promotions cannot exceed the number of completed rounds")
            }
            Self::HistoryTooLarge => write!(
                f,
                "round exceeds MAX_REUSE_ROUND; use a separately reviewed allocation implementation"
            ),
            Self::AllocationUnderflow => write!(
                f,
                "error allocation underflowed binary64; evaluation is not numerically admissible"
            ),
        }
    }
}

impl std::error::Error for ReuseBudgetError {}

/// One REUSE-style error allocation. All values are strictly positive and
/// finite; the two component budgets sum to total_error_budget within the
/// rounding tolerance of IEEE-754 binary64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReuseErrorAllocation {
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    total_error_budget: f64,
    paired_test_error_budget: f64,
    lower_bound_error_budget: f64,
}

impl ReuseErrorAllocation {
    pub fn round(&self) -> usize {
        self.round
    }

    pub fn prior_promotions(&self) -> usize {
        self.prior_promotions
    }

    pub fn candidates_per_round(&self) -> usize {
        self.candidates_per_round
    }

    pub fn total_error_budget(&self) -> f64 {
        self.total_error_budget
    }

    pub fn paired_test_error_budget(&self) -> f64 {
        self.paired_test_error_budget
    }

    pub fn lower_bound_error_budget(&self) -> f64 {
        self.lower_bound_error_budget
    }
}

/// Allocate the per-comparison REUSE error budget for one adaptive round.
///
/// Arguments:
/// - alpha: global family-wise false-promotion budget, fixed before search;
/// - round: one-based round number;
/// - prior_promotions: number of earlier accepted promotions;
/// - candidates_per_round: preregistered maximum K proposed candidates;
/// - rho: fraction of this allocation assigned to the paired test. The
///   remaining (1-rho) is assigned to the cumulative lower-bound certificate.
///
/// This function validates arithmetic inputs only. It does not know whether
/// alpha, K, the promotion history or rho were frozen by an independent
/// authority, and it cannot enforce that the proposer sees only a decision.
/// Do not use this function alone to authorize promotion or claim the REUSE
/// simultaneous guarantee.
pub fn allocate_reuse_error_budget(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    rho: f64,
) -> Result<ReuseErrorAllocation, ReuseBudgetError> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(ReuseBudgetError::InvalidAlpha);
    }
    if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
        return Err(ReuseBudgetError::InvalidRho);
    }
    if round == 0 {
        return Err(ReuseBudgetError::InvalidRound);
    }
    if candidates_per_round == 0 {
        return Err(ReuseBudgetError::InvalidCandidateLimit);
    }
    if round > MAX_REUSE_ROUND {
        return Err(ReuseBudgetError::HistoryTooLarge);
    }
    if prior_promotions > round - 1 {
        return Err(ReuseBudgetError::PromotionCountExceedsHistory);
    }

    let n = round - 1;
    let log_choose = log_binomial_coefficient(n, prior_promotions);

    // Compute entirely in log space to avoid integer binomial coefficients and
    // intermediate K^(p+1) overflow.
    let log_alpha = alpha.ln();
    let log_round_weight = -((round as f64).ln() + ((round + 1) as f64).ln());
    let log_promotion_weight =
        -(((prior_promotions + 1) as f64).ln() + ((prior_promotions + 2) as f64).ln());
    let log_candidate_factor =
        (prior_promotions + 1) as f64 * (candidates_per_round as f64).ln();
    let log_total =
        log_alpha + log_round_weight + log_promotion_weight - log_choose - log_candidate_factor;

    let total_error_budget = log_total.exp();
    let paired_test_error_budget = total_error_budget * rho;
    let lower_bound_error_budget = total_error_budget * (1.0 - rho);

    if !log_total.is_finite()
        || !total_error_budget.is_finite()
        || total_error_budget <= 0.0
        || !paired_test_error_budget.is_finite()
        || paired_test_error_budget <= 0.0
        || !lower_bound_error_budget.is_finite()
        || lower_bound_error_budget <= 0.0
    {
        return Err(ReuseBudgetError::AllocationUnderflow);
    }

    Ok(ReuseErrorAllocation {
        round,
        prior_promotions,
        candidates_per_round,
        total_error_budget,
        paired_test_error_budget,
        lower_bound_error_budget,
    })
}

/// log(C(n,k)) using a bounded sum; callers validate n and k first.
fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    let terms = k.min(n - k);
    let mut log_choose = 0.0_f64;
    for i in 1..=terms {
        log_choose += ((n - terms + i) as f64).ln() - (i as f64).ln();
    }
    log_choose
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_round_known_allocation_matches_hand_calculation() {
        // alpha * (1/2) * (1/2) / (1 * 8) = 0.0015625.
        let allocation = allocate_reuse_error_budget(0.05, 1, 0, 8, 0.5).unwrap();
        assert!((allocation.total_error_budget() - 0.0015625).abs() < 1e-15);
        assert!((allocation.paired_test_error_budget() - 0.00078125).abs() < 1e-15);
        assert!((allocation.lower_bound_error_budget() - 0.00078125).abs() < 1e-15);
        assert!((allocation.paired_test_error_budget()
            + allocation.lower_bound_error_budget()
            - allocation.total_error_budget())
            .abs()
            < 1e-15);
    }

    #[test]
    fn promotion_history_known_allocation_matches_hand_calculation() {
        // For t=3,p=1,K=2:
        // delta = 0.05 * (1/12) * (1/6) / (C(2,1) * 2^2)
        //       = 0.05 / 576.
        let allocation = allocate_reuse_error_budget(0.05, 3, 1, 2, 0.25).unwrap();
        let expected = 0.05 / 576.0;
        assert!((allocation.total_error_budget() - expected).abs() < 1e-18);
        assert!((allocation.paired_test_error_budget() - expected * 0.25).abs() < 1e-18);
        assert!((allocation.lower_bound_error_budget() - expected * 0.75).abs() < 1e-18);
    }

    #[test]
    fn allocation_decreases_with_round_promotion_count_and_candidate_limit() {
        let first = allocate_reuse_error_budget(0.05, 5, 0, 2, 0.5).unwrap();
        let later = allocate_reuse_error_budget(0.05, 6, 0, 2, 0.5).unwrap();
        let more_promotions = allocate_reuse_error_budget(0.05, 5, 1, 2, 0.5).unwrap();
        let more_candidates = allocate_reuse_error_budget(0.05, 5, 0, 4, 0.5).unwrap();

        assert!(later.total_error_budget() < first.total_error_budget());
        assert!(more_promotions.total_error_budget() < first.total_error_budget());
        assert!(more_candidates.total_error_budget() < first.total_error_budget());
    }

    #[test]
    fn rejects_invalid_alpha_rho_round_candidate_limit_and_history() {
        assert_eq!(
            allocate_reuse_error_budget(0.0, 1, 0, 1, 0.5),
            Err(ReuseBudgetError::InvalidAlpha)
        );
        assert_eq!(
            allocate_reuse_error_budget(f64::NAN, 1, 0, 1, 0.5),
            Err(ReuseBudgetError::InvalidAlpha)
        );
        assert_eq!(
            allocate_reuse_error_budget(0.05, 1, 0, 1, 0.0),
            Err(ReuseBudgetError::InvalidRho)
        );
        assert_eq!(
            allocate_reuse_error_budget(0.05, 0, 0, 1, 0.5),
            Err(ReuseBudgetError::InvalidRound)
        );
        assert_eq!(
            allocate_reuse_error_budget(0.05, 1, 0, 0, 0.5),
            Err(ReuseBudgetError::InvalidCandidateLimit)
        );
        assert_eq!(
            allocate_reuse_error_budget(0.05, 2, 2, 1, 0.5),
            Err(ReuseBudgetError::PromotionCountExceedsHistory)
        );
    }

    #[test]
    fn rejects_histories_beyond_the_reviewed_computation_bound() {
        assert_eq!(
            allocate_reuse_error_budget(0.05, MAX_REUSE_ROUND + 1, 0, 1, 0.5),
            Err(ReuseBudgetError::HistoryTooLarge)
        );
    }

    #[test]
    fn rejects_underflow_instead_of_returning_a_zero_error_budget() {
        assert_eq!(
            allocate_reuse_error_budget(0.05, 100_000, 99_999, 100, 0.5),
            Err(ReuseBudgetError::AllocationUnderflow)
        );
    }

    #[test]
    fn log_binomial_matches_small_exact_coefficients() {
        assert!((log_binomial_coefficient(2, 1) - 2.0_f64.ln()).abs() < 1e-12);
        assert!((log_binomial_coefficient(5, 2) - 10.0_f64.ln()).abs() < 1e-12);
        assert_eq!(log_binomial_coefficient(10, 0), 0.0);
        assert_eq!(log_binomial_coefficient(10, 10), 0.0);
    }
}
