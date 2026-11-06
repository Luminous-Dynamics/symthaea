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

use super::reuse_confidence_certificate::{
    MAX_CERTIFICATE_SAMPLE_COUNT, BoundedScoreConfidenceCertificate, ReuseCertificateError,
    ReuseConfidenceCertificate, bounded_score_confidence_certificate,
    pass_fail_confidence_certificate,
};
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
    ComparisonCountOverflow,
    InvalidCandidateEvidence,
    InvalidBoundedCandidateEvidence,
    InvalidBoundedDifference,
    CandidateSampleTooLarge { actual: usize, maximum: usize },
    InvalidPairedDifference(i8),
    CandidateCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ReuseBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be finite and strictly between 0 and 1"),
            Self::InvalidRho => write!(f, "rho must be finite and strictly between 0 and 1"),
            Self::InvalidRound => write!(f, "round is one-based and must be at least 1"),
            Self::InvalidCandidateLimit => write!(f, "candidate limit K must be at least 1"),
            Self::PromotionCountExceedsHistory => {
                write!(
                    f,
                    "prior promotions cannot exceed the number of completed rounds"
                )
            }
            Self::HistoryTooLarge => write!(
                f,
                "round exceeds MAX_REUSE_ROUND; use a separately reviewed allocation implementation"
            ),
            Self::AllocationUnderflow => write!(
                f,
                "error allocation underflowed binary64; evaluation is not numerically admissible"
            ),
            Self::ComparisonCountOverflow => {
                write!(f, "paired win/loss counts overflow usize")
            }
            Self::InvalidCandidateEvidence => {
                write!(
                    f,
                    "candidate evidence requires a non-empty paired outcome vector"
                )
            }
            Self::InvalidBoundedCandidateEvidence => {
                write!(
                    f,
                    "bounded-score candidate evidence requires a non-empty paired outcome vector"
                )
            }
            Self::InvalidBoundedDifference => {
                write!(
                    f,
                    "bounded-score paired differences must be finite and within [-1, 1]"
                )
            }
            Self::CandidateSampleTooLarge { actual, maximum } => {
                write!(
                    f,
                    "candidate sample count {actual} exceeds maximum {maximum}"
                )
            }
            Self::InvalidPairedDifference(value) => {
                write!(f, "paired outcome must be -1, 0 or +1, got {value}")
            }
            Self::CandidateCountMismatch { expected, actual } => {
                write!(f, "expected {expected} candidate summaries, got {actual}")
            }
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

/// Private per-candidate evaluation evidence held by the independent evaluator.
///
/// Counts and the empirical mean are derived from the same complete paired
/// outcome vector. Callers cannot supply unrelated aggregate statistics.
/// The proposer must not receive this structure or its contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ReuseCandidateEvidence {
    paired_differences: Vec<i8>,
    wins: usize,
    losses: usize,
    empirical_improvement: f64,
}

impl ReuseCandidateEvidence {
    /// Construct candidate evidence from the full matched pass/fail outcomes.
    ///
    /// Each element is candidate_success - incumbent_success in {-1, 0, +1}.
    /// The vector must cover the complete preregistered eligible task universe.
    /// Its rows must represent the independent paired units for REUSE; clustered
    /// task rows must be aggregated to the preregistered independent unit before
    /// constructing this value.
    pub fn new(paired_differences: impl Into<Vec<i8>>) -> Result<Self, ReuseBudgetError> {
        let paired_differences = paired_differences.into();
        if paired_differences.is_empty() {
            return Err(ReuseBudgetError::InvalidCandidateEvidence);
        }
        if paired_differences.len() > MAX_CERTIFICATE_SAMPLE_COUNT {
            return Err(ReuseBudgetError::CandidateSampleTooLarge {
                actual: paired_differences.len(),
                maximum: MAX_CERTIFICATE_SAMPLE_COUNT,
            });
        }

        let mut wins = 0usize;
        let mut losses = 0usize;
        for &difference in &paired_differences {
            match difference {
                1 => wins += 1,
                -1 => losses += 1,
                0 => {}
                other => return Err(ReuseBudgetError::InvalidPairedDifference(other)),
            }
        }
        let empirical_improvement = (wins as f64 - losses as f64) / paired_differences.len() as f64;

        Ok(Self {
            paired_differences,
            wins,
            losses,
            empirical_improvement,
        })
    }
}



/// Candidate evidence for one predeclared bounded-score estimand.
///
/// Each value is candidate-minus-incumbent in [-1, 1]. The empirical mean is
/// derived internally; callers cannot supply detached summary statistics.
/// Rows must already be reduced to the preregistered independent paired unit
/// and supplied in canonical order by the trusted evidence adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedScoreCandidateEvidence {
    paired_differences: Vec<f64>,
    empirical_improvement: f64,
}

impl BoundedScoreCandidateEvidence {
    pub fn new(paired_differences: impl Into<Vec<f64>>) -> Result<Self, ReuseBudgetError> {
        let paired_differences = paired_differences.into();
        if paired_differences.is_empty() {
            return Err(ReuseBudgetError::InvalidBoundedCandidateEvidence);
        }
        if paired_differences.len() > MAX_CERTIFICATE_SAMPLE_COUNT {
            return Err(ReuseBudgetError::CandidateSampleTooLarge {
                actual: paired_differences.len(),
                maximum: MAX_CERTIFICATE_SAMPLE_COUNT,
            });
        }

        let mut sum = 0.0_f64;
        for &difference in &paired_differences {
            if !difference.is_finite() || !(-1.0..=1.0).contains(&difference) {
                return Err(ReuseBudgetError::InvalidBoundedDifference);
            }
            sum += difference;
        }
        let empirical_improvement = sum / paired_differences.len() as f64;
        if !empirical_improvement.is_finite() {
            return Err(ReuseBudgetError::InvalidBoundedDifference);
        }

        Ok(Self {
            paired_differences,
            empirical_improvement,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.paired_differences.len()
    }

    pub fn empirical_improvement(&self) -> f64 {
        self.empirical_improvement
    }
}

/// Candidate-facing output for one complete REUSE round. This is the paper's
/// y_t in {0, ..., K}: RejectAll represents zero, and PromoteCandidate uses
/// a one-based candidate number in 1..=K. A proposer receives only this one
/// round decision, never individual pass/fail outcomes, p-values, scores or
/// confidence certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseRoundDecision {
    RejectAll,
    PromoteCandidate { candidate_number: usize },
}

/// Trusted evaluator audit record. This type is crate-private so the cumulative
/// certificate is not accidentally exposed through the proposer-facing API.
#[derive(Debug, Clone, PartialEq)]
pub struct ReuseRoundAudit {
    decision: ReuseRoundDecision,
    selected_certificate: Option<ReuseConfidenceCertificate>,
    cumulative_lower_bound: f64,
}

impl ReuseRoundAudit {
    pub fn decision(&self) -> ReuseRoundDecision {
        self.decision
    }

    pub fn selected_certificate(&self) -> Option<&ReuseConfidenceCertificate> {
        self.selected_certificate.as_ref()
    }

    pub fn cumulative_lower_bound(&self) -> f64 {
        self.cumulative_lower_bound
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReuseRoundError {
    Budget(ReuseBudgetError),
    ConfidenceCertificate(ReuseCertificateError),
    InvalidCumulativeLowerBound,
}

impl fmt::Display for ReuseRoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => write!(f, "REUSE round input error: {error}"),
            Self::ConfidenceCertificate(error) => {
                write!(f, "REUSE confidence certificate failed: {error}")
            }
            Self::InvalidCumulativeLowerBound => {
                write!(
                    f,
                    "prior cumulative lower bound must be finite and nonnegative"
                )
            }
        }
    }
}

impl std::error::Error for ReuseRoundError {}

impl From<ReuseBudgetError> for ReuseRoundError {
    fn from(value: ReuseBudgetError) -> Self {
        Self::Budget(value)
    }
}

impl From<ReuseCertificateError> for ReuseRoundError {
    fn from(value: ReuseCertificateError) -> Self {
        Self::ConfidenceCertificate(value)
    }
}

/// Evaluate all K candidates and return only the paper's one-based round
/// decision y_t. The selected certificate is computed and retained internally
/// by the trusted evaluator; it is never returned through this API.
pub fn evaluate_reuse_round(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    rho: f64,
    candidates: &[ReuseCandidateEvidence],
) -> Result<ReuseRoundDecision, ReuseRoundError> {
    Ok(evaluate_reuse_round_audited(
        alpha,
        round,
        prior_promotions,
        candidates_per_round,
        rho,
        candidates,
        0.0,
    )?
    .decision())
}



/// Evaluate exactly K bounded-score candidates and return only the coarse
/// REUSE round decision. Detailed evidence and certificates remain internal.
pub fn evaluate_bounded_score_reuse_round(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    practical_improvement_threshold: f64,
    candidates: &[BoundedScoreCandidateEvidence],
) -> Result<ReuseRoundDecision, BoundedScoreRoundError> {
    Ok(evaluate_bounded_score_reuse_round_audited(
        alpha,
        round,
        prior_promotions,
        candidates_per_round,
        practical_improvement_threshold,
        candidates,
        0.0,
    )?.decision())
}

/// In-crate audit path for a trusted bounded-score evaluator.
///
/// The same REUSE allocation covers each possible candidate/history comparison.
/// A candidate is eligible only when its lower confidence bound exceeds the
/// preregistered gamma. Among eligible candidates, the highest empirical mean
/// is selected. Only the selected lower bound is added to the private progress
/// ledger. This helper is not a persisted or independently operated service.
pub(crate) fn evaluate_bounded_score_reuse_round_audited(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    practical_improvement_threshold: f64,
    candidates: &[BoundedScoreCandidateEvidence],
    prior_cumulative_lower_bound: f64,
) -> Result<BoundedScoreRoundAudit, BoundedScoreRoundError> {
    if !practical_improvement_threshold.is_finite() || practical_improvement_threshold < 0.0 {
        return Err(BoundedScoreRoundError::InvalidPracticalImprovementThreshold);
    }
    if !prior_cumulative_lower_bound.is_finite() || prior_cumulative_lower_bound < 0.0 {
        return Err(BoundedScoreRoundError::InvalidCumulativeLowerBound);
    }
    if candidates.len() != candidates_per_round {
        return Err(ReuseBudgetError::CandidateCountMismatch {
            expected: candidates_per_round,
            actual: candidates.len(),
        }
        .into());
    }

    let delta = compute_reuse_total_error_budget(
        alpha,
        round,
        prior_promotions,
        candidates_per_round,
    )?;

    let mut best: Option<(usize, usize, f64, BoundedScoreConfidenceCertificate)> = None;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let certificate =
            bounded_score_confidence_certificate(&candidate.paired_differences, delta)?;
        if certificate.lower_bound() > practical_improvement_threshold
            && best.as_ref().is_none_or(|(_, _, best_mean, _)| {
                candidate.empirical_improvement > *best_mean
            })
        {
            best = Some((
                candidate_index,
                candidate_index + 1,
                candidate.empirical_improvement,
                certificate,
            ));
        }
    }

    let Some((_candidate_index, candidate_number, _mean, certificate)) = best else {
        return Ok(BoundedScoreRoundAudit {
            decision: ReuseRoundDecision::RejectAll,
            selected_certificate: None,
            cumulative_lower_bound: prior_cumulative_lower_bound,
        });
    };

    let cumulative_lower_bound = prior_cumulative_lower_bound + certificate.lower_bound();
    if !cumulative_lower_bound.is_finite() {
        return Err(BoundedScoreRoundError::InvalidCumulativeLowerBound);
    }

    Ok(BoundedScoreRoundAudit {
        decision: ReuseRoundDecision::PromoteCandidate { candidate_number },
        selected_certificate: Some(certificate),
        cumulative_lower_bound,
    })
}

/// In-crate audit path for the independent evaluator service.
///
/// The candidate evidence contains the full paired outcomes. This function:
/// 1. applies the history-aware sign test to all K candidates;
/// 2. chooses the passing candidate with the highest empirical improvement;
/// 3. constructs the selected candidate's (1-rho)*delta confidence certificate;
/// 4. advances the cumulative lower bound by max(ell, 0).
///
/// Only the decision from the returned audit record may cross the feedback
/// firewall. The service must persist the round/promotion history and audit
/// record independently; this helper is not itself a sequential service.
pub(crate) fn evaluate_reuse_round_audited(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    rho: f64,
    candidates: &[ReuseCandidateEvidence],
    prior_cumulative_lower_bound: f64,
) -> Result<ReuseRoundAudit, ReuseRoundError> {
    if !prior_cumulative_lower_bound.is_finite() || prior_cumulative_lower_bound < 0.0 {
        return Err(ReuseRoundError::InvalidCumulativeLowerBound);
    }
    if candidates.len() != candidates_per_round {
        return Err(ReuseBudgetError::CandidateCountMismatch {
            expected: candidates_per_round,
            actual: candidates.len(),
        }
        .into());
    }
    let allocation =
        allocate_reuse_error_budget(alpha, round, prior_promotions, candidates_per_round, rho)?;

    let mut best: Option<(usize, usize, f64)> = None;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let discordant_count = candidate
            .wins
            .checked_add(candidate.losses)
            .ok_or(ReuseBudgetError::ComparisonCountOverflow)?;
        if discordant_count == 0 {
            continue;
        }

        let p_value =
            super::paired_sign_test::exact_sign_test_upper_tail(candidate.wins, candidate.losses)
                .ok_or(ReuseBudgetError::ComparisonCountOverflow)?;

        if p_value <= allocation.paired_test_error_budget
            && best.is_none_or(|(_, _, best_improvement)| {
                candidate.empirical_improvement > best_improvement
            })
        {
            best = Some((
                candidate_index,
                candidate_index + 1,
                candidate.empirical_improvement,
            ));
        }
    }

    let Some((candidate_index, candidate_number, _)) = best else {
        return Ok(ReuseRoundAudit {
            decision: ReuseRoundDecision::RejectAll,
            selected_certificate: None,
            cumulative_lower_bound: prior_cumulative_lower_bound,
        });
    };

    let certificate = pass_fail_confidence_certificate(
        &candidates[candidate_index].paired_differences,
        allocation.lower_bound_error_budget,
    )?;
    let cumulative_lower_bound = prior_cumulative_lower_bound + certificate.cumulative_increment();
    if !cumulative_lower_bound.is_finite() {
        return Err(ReuseRoundError::InvalidCumulativeLowerBound);
    }

    Ok(ReuseRoundAudit {
        decision: ReuseRoundDecision::PromoteCandidate { candidate_number },
        selected_certificate: Some(certificate),
        cumulative_lower_bound,
    })
}



/// Trusted evaluator audit record for the bounded-score REUSE route.
/// Full certificate and cumulative progress remain outside proposer feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedScoreRoundAudit {
    decision: ReuseRoundDecision,
    selected_certificate: Option<BoundedScoreConfidenceCertificate>,
    cumulative_lower_bound: f64,
}

impl BoundedScoreRoundAudit {
    pub fn decision(&self) -> ReuseRoundDecision {
        self.decision
    }

    pub fn selected_certificate(&self) -> Option<&BoundedScoreConfidenceCertificate> {
        self.selected_certificate.as_ref()
    }

    pub fn cumulative_lower_bound(&self) -> f64 {
        self.cumulative_lower_bound
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundedScoreRoundError {
    Budget(ReuseBudgetError),
    ConfidenceCertificate(ReuseCertificateError),
    InvalidPracticalImprovementThreshold,
    InvalidCumulativeLowerBound,
}

impl fmt::Display for BoundedScoreRoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => write!(f, "bounded-score REUSE input error: {error}"),
            Self::ConfidenceCertificate(error) => {
                write!(f, "bounded-score confidence certificate failed: {error}")
            }
            Self::InvalidPracticalImprovementThreshold => {
                write!(f, "practical improvement threshold gamma must be finite and nonnegative")
            }
            Self::InvalidCumulativeLowerBound => {
                write!(f, "prior cumulative lower bound must be finite and nonnegative")
            }
        }
    }
}

impl std::error::Error for BoundedScoreRoundError {}

impl From<ReuseBudgetError> for BoundedScoreRoundError {
    fn from(value: ReuseBudgetError) -> Self {
        Self::Budget(value)
    }
}

impl From<ReuseCertificateError> for BoundedScoreRoundError {
    fn from(value: ReuseCertificateError) -> Self {
        Self::ConfidenceCertificate(value)
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
/// Compute the complete per-comparison REUSE allocation before any rho split.
/// Used by the bounded-score route, which spends the complete allocation on
/// its single lower-confidence-bound decision.
fn compute_reuse_total_error_budget(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
) -> Result<f64, ReuseBudgetError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ReuseBudgetError::InvalidAlpha);
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
    let log_alpha = alpha.ln();
    let log_round_weight = -((round as f64).ln() + ((round + 1) as f64).ln());
    let log_promotion_weight =
        -(((prior_promotions + 1) as f64).ln() + ((prior_promotions + 2) as f64).ln());
    let log_candidate_factor =
        (prior_promotions + 1) as f64 * (candidates_per_round as f64).ln();
    let log_total =
        log_alpha + log_round_weight + log_promotion_weight - log_choose - log_candidate_factor;
    let total_error_budget = log_total.exp();

    if !log_total.is_finite()
        || !total_error_budget.is_finite()
        || total_error_budget <= 0.0
    {
        return Err(ReuseBudgetError::AllocationUnderflow);
    }
    Ok(total_error_budget)
}

/// Allocate the per-comparison REUSE error budget for one adaptive pass/fail round.
pub fn allocate_reuse_error_budget(
    alpha: f64,
    round: usize,
    prior_promotions: usize,
    candidates_per_round: usize,
    rho: f64,
) -> Result<ReuseErrorAllocation, ReuseBudgetError> {
    if !rho.is_finite() || rho <= 0.0 || rho >= 1.0 {
        return Err(ReuseBudgetError::InvalidRho);
    }
    let total_error_budget =
        compute_reuse_total_error_budget(alpha, round, prior_promotions, candidates_per_round)?;
    let paired_test_error_budget = total_error_budget * rho;
    let lower_bound_error_budget = total_error_budget * (1.0 - rho);

    if !paired_test_error_budget.is_finite()
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

    fn evidence(wins: usize, losses: usize, total: usize) -> ReuseCandidateEvidence {
        assert!(wins + losses <= total);
        let mut outcomes = vec![1_i8; wins];
        outcomes.extend(vec![-1_i8; losses]);
        outcomes.extend(vec![0_i8; total - wins - losses]);
        ReuseCandidateEvidence::new(outcomes).unwrap()
    }

    #[test]
    fn first_round_known_allocation_matches_hand_calculation() {
        // alpha * (1/2) * (1/2) / (1 * 8) = 0.0015625.
        let allocation = allocate_reuse_error_budget(0.05, 1, 0, 8, 0.5).unwrap();
        assert!((allocation.total_error_budget() - 0.0015625).abs() < 1e-15);
        assert!((allocation.paired_test_error_budget() - 0.00078125).abs() < 1e-15);
        assert!((allocation.lower_bound_error_budget() - 0.00078125).abs() < 1e-15);
        assert!(
            (allocation.paired_test_error_budget() + allocation.lower_bound_error_budget()
                - allocation.total_error_budget())
            .abs()
                < 1e-15
        );
    }

    #[test]
    fn promotion_history_known_allocation_matches_hand_calculation() {
        // For t=3,p=1,K=2:
        // delta = 0.05 * (1/12) * (1/6) / (C(2,1) * 2^2) = 0.05 / 576.
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
            allocate_reuse_error_budget(1.0, 1, 0, 1, 0.5),
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
            allocate_reuse_error_budget(0.05, 1, 0, 1, 1.0),
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

    #[test]
    fn candidate_evidence_derives_counts_and_mean_from_full_outcomes() {
        let candidate = evidence(40, 10, 100);
        assert_eq!(candidate.wins, 40);
        assert_eq!(candidate.losses, 10);
        assert!((candidate.empirical_improvement - 0.3).abs() < 1e-15);
        assert_eq!(candidate.paired_differences.len(), 100);
    }

    #[test]
    fn candidate_evidence_rejects_empty_and_invalid_outcome_vectors() {
        assert_eq!(
            ReuseCandidateEvidence::new(Vec::<i8>::new()),
            Err(ReuseBudgetError::InvalidCandidateEvidence)
        );
        assert_eq!(
            ReuseCandidateEvidence::new(vec![1, 0, 2]),
            Err(ReuseBudgetError::InvalidPairedDifference(2))
        );
    }

    #[test]
    fn round_selector_promotes_best_empirical_candidate_that_passes() {
        // For K=3 the paired-test threshold is 0.0020833. Candidate 2 has
        // 9/100 wins and no losses (p=1/512): it passes and beats candidate 1
        // on empirical gain. Candidate 3 has 8/100 wins (p=1/256) and fails.
        let candidates = [
            evidence(10, 5, 100),
            evidence(9, 0, 100),
            evidence(8, 0, 100),
        ];
        assert_eq!(
            evaluate_reuse_round(0.05, 1, 0, 3, 0.5, &candidates).unwrap(),
            ReuseRoundDecision::PromoteCandidate {
                candidate_number: 2
            }
        );
    }

    #[test]
    fn round_selector_rejects_all_ties_and_all_candidates_that_fail() {
        let tied = [evidence(0, 0, 100), evidence(0, 0, 100)];
        assert_eq!(
            evaluate_reuse_round(0.05, 1, 0, 2, 0.5, &tied).unwrap(),
            ReuseRoundDecision::RejectAll
        );

        let insufficient = [evidence(3, 2, 100), evidence(4, 3, 100)];
        assert_eq!(
            evaluate_reuse_round(0.05, 1, 0, 2, 0.5, &insufficient).unwrap(),
            ReuseRoundDecision::RejectAll
        );
    }

    #[test]
    fn round_selector_breaks_equal_empirical_gains_by_input_order() {
        let candidates = [evidence(10, 0, 100), evidence(11, 1, 100)];
        assert_eq!(
            evaluate_reuse_round(0.1, 1, 0, 2, 0.5, &candidates).unwrap(),
            ReuseRoundDecision::PromoteCandidate {
                candidate_number: 1
            }
        );
    }

    #[test]
    fn round_selector_rejects_candidate_count_mismatch() {
        let one = [evidence(10, 0, 100)];
        assert_eq!(
            evaluate_reuse_round(0.05, 1, 0, 2, 0.5, &one),
            Err(ReuseRoundError::Budget(
                ReuseBudgetError::CandidateCountMismatch {
                    expected: 2,
                    actual: 1
                }
            ))
        );
    }

    #[test]
    fn audit_retains_selected_certificate_and_advances_cumulative_lower_bound() {
        let candidates = [evidence(40, 10, 100)];
        let audit = evaluate_reuse_round_audited(0.05, 1, 0, 1, 0.5, &candidates, 0.25).unwrap();
        assert_eq!(
            audit.decision(),
            ReuseRoundDecision::PromoteCandidate {
                candidate_number: 1
            }
        );
        let certificate = audit.selected_certificate().unwrap();
        assert!(certificate.lower_bound() > 0.0);
        assert!(
            (audit.cumulative_lower_bound() - (0.25 + certificate.cumulative_increment())).abs()
                < 1e-15
        );
    }

    #[test]
    fn audit_rejection_keeps_cumulative_lower_bound_and_has_no_certificate() {
        let candidates = [evidence(0, 0, 100)];
        let audit = evaluate_reuse_round_audited(0.05, 1, 0, 1, 0.5, &candidates, 0.75).unwrap();
        assert_eq!(audit.decision(), ReuseRoundDecision::RejectAll);
        assert!(audit.selected_certificate().is_none());
        assert_eq!(audit.cumulative_lower_bound(), 0.75);
    }

    #[test]
    fn audit_rejects_invalid_cumulative_lower_bound() {
        let candidates = [evidence(10, 0, 100)];
        assert_eq!(
            evaluate_reuse_round_audited(0.05, 1, 0, 1, 0.5, &candidates, f64::NAN),
            Err(ReuseRoundError::InvalidCumulativeLowerBound)
        );
        assert_eq!(
            evaluate_reuse_round_audited(0.05, 1, 0, 1, 0.5, &candidates, -1.0),
            Err(ReuseRoundError::InvalidCumulativeLowerBound)
        );
    }
}


#[cfg(test)]
mod bounded_score_round_tests {
    use super::*;

    fn evidence(positive: f64, n: usize) -> BoundedScoreCandidateEvidence {
        let rows = (0..n)
            .map(|i| if i % 4 == 3 { -0.1 } else { positive })
            .collect::<Vec<_>>();
        BoundedScoreCandidateEvidence::new(rows).unwrap()
    }

    #[test]
    fn bounded_score_round_selects_highest_mean_among_candidates_over_gamma() {
        let candidates = [evidence(0.2, 1_000), evidence(0.3, 1_000)];
        let decision = evaluate_bounded_score_reuse_round(
            0.05, 1, 0, 2, 0.05, &candidates,
        ).unwrap();
        assert_eq!(
            decision,
            ReuseRoundDecision::PromoteCandidate { candidate_number: 2 }
        );
    }

    #[test]
    fn bounded_score_round_rejects_when_no_candidate_clears_gamma() {
        let candidates = [BoundedScoreCandidateEvidence::new(vec![0.0; 100]).unwrap()];
        let decision = evaluate_bounded_score_reuse_round(
            0.05, 1, 0, 1, 0.0, &candidates,
        ).unwrap();
        assert_eq!(decision, ReuseRoundDecision::RejectAll);
    }

    #[test]
    fn bounded_score_round_requires_exact_candidate_count() {
        let candidates = [evidence(0.2, 1_000)];
        let result = evaluate_bounded_score_reuse_round(0.05, 1, 0, 2, 0.0, &candidates);
        assert!(matches!(
            result,
            Err(BoundedScoreRoundError::Budget(
                ReuseBudgetError::CandidateCountMismatch { expected: 2, actual: 1 }
            ))
        ));
    }

    #[test]
    fn bounded_score_round_rejects_invalid_gamma_and_history() {
        let candidates = [evidence(0.2, 1_000)];
        assert!(matches!(
            evaluate_bounded_score_reuse_round(0.05, 1, 0, 1, f64::NAN, &candidates),
            Err(BoundedScoreRoundError::InvalidPracticalImprovementThreshold)
        ));
        assert!(matches!(
            evaluate_bounded_score_reuse_round(0.05, 1, 1, 1, 0.0, &candidates),
            Err(BoundedScoreRoundError::Budget(ReuseBudgetError::PromotionCountExceedsHistory))
        ));
    }

    #[test]
    fn bounded_score_candidate_evidence_rejects_invalid_values() {
        assert_eq!(
            BoundedScoreCandidateEvidence::new(Vec::<f64>::new()),
            Err(ReuseBudgetError::InvalidBoundedCandidateEvidence)
        );
        assert_eq!(
            BoundedScoreCandidateEvidence::new(vec![0.0, f64::NAN]),
            Err(ReuseBudgetError::InvalidBoundedDifference)
        );
        assert_eq!(
            BoundedScoreCandidateEvidence::new(vec![1.1]),
            Err(ReuseBudgetError::InvalidBoundedDifference)
        );
        assert_eq!(
            BoundedScoreCandidateEvidence::new(vec![-1.1]),
            Err(ReuseBudgetError::InvalidBoundedDifference)
        );
    }
}

