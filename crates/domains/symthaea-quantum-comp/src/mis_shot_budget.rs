//! Adaptive shot-budget policy for confidence-aware MIS tie-breaking.
//!
//! This module plans measurement allocation. It does not execute quantum circuits,
//! produce samples, or certify maximum-independent-set membership. The caller must
//! provide estimates from a documented sampling protocol and record every allocation.

use core::fmt;

use crate::mis_informer::MisSelectionMode;

/// Empirical inclusion count for one eligible vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisCandidateEstimate {
    /// Stable zero-based vertex identity.
    pub vertex: usize,
    /// Number of independent sample observations attributed to this candidate.
    pub shots: u64,
    /// Number of those observations in which the vertex was included.
    pub inclusion_count: u64,
}

impl MisCandidateEstimate {
    /// Create an estimate, preserving zero shots as missing evidence.
    pub fn new(vertex: usize, shots: u64, inclusion_count: u64) -> Result<Self, MisShotBudgetError> {
        if inclusion_count > shots {
            return Err(MisShotBudgetError::InclusionCountExceedsShots {
                vertex,
                inclusion_count,
                shots,
            });
        }
        Ok(Self { vertex, shots, inclusion_count })
    }
}

/// Limits for sequential measurement allocation over the current candidate set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisShotBudgetConfig {
    /// First allocation to each candidate when all candidates are unsampled.
    pub initial_shots_per_candidate: u64,
    /// Maximum additional shots per unresolved candidate at each policy step.
    pub batch_shots_per_candidate: u64,
    /// Maximum sum of per-candidate shots for this decision.
    pub max_total_shots: u64,
    /// Total family-wise error budget across candidates and all adaptive looks.
    /// The planner allocates this risk across at most max_total_shots + 1 looks.
    pub familywise_error_probability: f64,
}

impl MisShotBudgetConfig {
    /// Construct a valid configuration.
    pub fn new(
        initial_shots_per_candidate: u64,
        batch_shots_per_candidate: u64,
        max_total_shots: u64,
        familywise_error_probability: f64,
    ) -> Result<Self, MisShotBudgetError> {
        let config = Self {
            initial_shots_per_candidate,
            batch_shots_per_candidate,
            max_total_shots,
            familywise_error_probability,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<(), MisShotBudgetError> {
        if self.initial_shots_per_candidate == 0 {
            return Err(MisShotBudgetError::ZeroInitialShots);
        }
        if self.batch_shots_per_candidate == 0 {
            return Err(MisShotBudgetError::ZeroBatchShots);
        }
        if self.max_total_shots == 0 {
            return Err(MisShotBudgetError::ZeroTotalBudget);
        }
        if !self.familywise_error_probability.is_finite()
            || self.familywise_error_probability <= 0.0
            || self.familywise_error_probability >= 1.0
        {
            return Err(MisShotBudgetError::InvalidConfidenceLevel);
        }
        Ok(())
    }
}

/// Additional shots requested for one candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisShotAllocation {
    /// Candidate vertex to measure.
    pub vertex: usize,
    /// Number of additional samples requested.
    pub additional_shots: u64,
}

/// Why adaptive sampling stopped without a confidence-separated choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisShotAbstentionReason {
    /// The caller supplied fewer than two eligible candidates.
    OnlyOneCandidate,
    /// The declared per-decision budget is exhausted before scores separate.
    BudgetExhausted,
}

/// A deterministic next step for a bounded adaptive measurement loop.
#[derive(Debug, Clone, PartialEq)]
pub enum MisShotBudgetDecision {
    /// The empirical best candidate is separated from every competitor's interval.
    Selected {
        /// Selected vertex according to the requested preference mode.
        vertex: usize,
        /// Number of samples contributing to this candidate estimate.
        shots: u64,
        /// Number of included observations for the selected vertex.
        inclusion_count: u64,
        /// Lower simultaneous confidence bound for the selected inclusion rate.
        lower_bound: f64,
        /// Upper simultaneous confidence bound for the selected inclusion rate.
        upper_bound: f64,
        /// Upper bound for best rival (prefer-in), or lower bound (prefer-out).
        competitor_bound: f64,
        /// Sum of samples consumed across the candidate set.
        total_shots: u64,
    },
    /// More measurements could resolve the score ordering; execute these allocations.
    RequestMoreShots {
        /// Candidate-specific next allocations, ordered by ascending vertex index.
        allocations: Vec<MisShotAllocation>,
        /// Sum of already consumed samples.
        total_shots: u64,
        /// Remaining budget before applying the requested allocations.
        remaining_budget: u64,
    },
    /// The policy declines to guess; caller may use a declared classical fallback.
    Abstain {
        /// Reason the policy stopped.
        reason: MisShotAbstentionReason,
        /// Samples consumed before stopping.
        total_shots: u64,
    },
}

/// Invalid input or budget configuration for adaptive MIS sampling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisShotBudgetError {
    /// Initial shots per candidate must be non-zero.
    ZeroInitialShots,
    /// The next-batch size per candidate must be non-zero.
    ZeroBatchShots,
    /// The decision must have a positive total shot budget.
    ZeroTotalBudget,
    /// Family-wise error probability must be finite and strictly in (0, 1).
    InvalidConfidenceLevel,
    /// An inclusion count cannot exceed that candidate's sample count.
    InclusionCountExceedsShots {
        /// Candidate vertex.
        vertex: usize,
        /// Reported inclusions.
        inclusion_count: u64,
        /// Declared sample count.
        shots: u64,
    },
    /// A vertex was listed more than once.
    DuplicateCandidate {
        /// Repeated vertex index.
        vertex: usize,
    },
    /// Sample-count sum overflowed a u64.
    TotalShotsOverflow,
    /// Already consumed samples exceed the configured decision budget.
    ExistingSamplesExceedBudget {
        /// Sum of samples already used.
        total_shots: u64,
        /// Configured maximum.
        max_total_shots: u64,
    },
    /// No candidate estimates were supplied.
    EmptyCandidates,
}

impl fmt::Display for MisShotBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInitialShots => write!(f, "initial shots per candidate must be non-zero"),
            Self::ZeroBatchShots => write!(f, "batch shots per candidate must be non-zero"),
            Self::ZeroTotalBudget => write!(f, "total shot budget must be non-zero"),
            Self::InvalidConfidenceLevel => {
                write!(f, "family-wise error probability must be finite and strictly between 0 and 1")
            }
            Self::InclusionCountExceedsShots { vertex, inclusion_count, shots } => write!(
                f,
                "vertex {vertex} has {inclusion_count} inclusions in only {shots} shots"
            ),
            Self::DuplicateCandidate { vertex } => write!(f, "candidate vertex {vertex} is repeated"),
            Self::TotalShotsOverflow => write!(f, "sum of candidate sample counts overflowed u64"),
            Self::ExistingSamplesExceedBudget { total_shots, max_total_shots } => write!(
                f,
                "existing sample count {total_shots} exceeds budget {max_total_shots}"
            ),
            Self::EmptyCandidates => write!(f, "no eligible candidates were supplied"),
        }
    }
}

impl std::error::Error for MisShotBudgetError {}

#[derive(Debug, Clone, Copy)]
struct CandidateInterval {
    estimate: MisCandidateEstimate,
    rate: f64,
    lower: f64,
    upper: f64,
}

fn interval(
    estimate: MisCandidateEstimate,
    eligible_count: usize,
    alpha_per_look: f64,
) -> CandidateInterval {
    if estimate.shots == 0 {
        return CandidateInterval { estimate, rate: 0.5, lower: 0.0, upper: 1.0 };
    }
    let rate = estimate.inclusion_count as f64 / estimate.shots as f64;
    let radius = ((2.0 * eligible_count as f64 / alpha_per_look).ln()
        / (2.0 * estimate.shots as f64))
        .sqrt();
    CandidateInterval {
        estimate,
        rate,
        lower: (rate - radius).max(0.0),
        upper: (rate + radius).min(1.0),
    }
}

fn preferred(
    candidate: CandidateInterval,
    current: CandidateInterval,
    mode: MisSelectionMode,
) -> bool {
    match mode {
        MisSelectionMode::PreferInSet => {
            candidate.rate > current.rate
                || (candidate.rate == current.rate
                    && candidate.estimate.vertex < current.estimate.vertex)
        }
        MisSelectionMode::PreferOutSet => {
            candidate.rate < current.rate
                || (candidate.rate == current.rate
                    && candidate.estimate.vertex < current.estimate.vertex)
        }
    }
}

/// Plan one deterministic step for an adaptive shot-allocation loop.
///
/// `estimates` should contain exactly the structurally eligible candidates for the
/// current MIS decision (for example, the current minimum-degree vertices). A zero-shot
/// candidate has no estimate and receives the full uncertainty interval [0, 1]; it is
/// never interpreted as an observed zero inclusion probability.
///
/// The Hoeffding/union-bound intervals require an appropriate independent-sampling
/// model. To account for repeated looks during adaptive sampling, the configured error
/// probability is split across at most max_total_shots + 1 looks as well as the
/// candidate set; this is conservative but prevents repeated fixed-sample checks from
/// silently reusing the full error budget. Device drift, correlated observations, or
/// calibration changes can invalidate the sampling model; this planner cannot detect
/// such violations. Counts are conservatively summed per candidate; the planner
/// does not coalesce identical rooted-topology circuits or account for shared cached
/// measurements, which must be handled by a separate scheduler with explicit cache
/// identity/invalidation. It plans allocations only: no quantum execution, provenance,
/// or maximum-set certificate is produced.
pub fn plan_mis_shots(
    estimates: &[MisCandidateEstimate],
    mode: MisSelectionMode,
    config: MisShotBudgetConfig,
) -> Result<MisShotBudgetDecision, MisShotBudgetError> {
    config.validate()?;
    if estimates.is_empty() {
        return Err(MisShotBudgetError::EmptyCandidates);
    }

    let mut sorted = estimates.to_vec();
    sorted.sort_unstable_by_key(|estimate| estimate.vertex);
    for pair in sorted.windows(2) {
        if pair[0].vertex == pair[1].vertex {
            return Err(MisShotBudgetError::DuplicateCandidate { vertex: pair[0].vertex });
        }
    }
    for estimate in &sorted {
        if estimate.inclusion_count > estimate.shots {
            return Err(MisShotBudgetError::InclusionCountExceedsShots {
                vertex: estimate.vertex,
                inclusion_count: estimate.inclusion_count,
                shots: estimate.shots,
            });
        }
    }

    let total_shots = sorted.iter().try_fold(0_u64, |total, estimate| {
        total.checked_add(estimate.shots).ok_or(MisShotBudgetError::TotalShotsOverflow)
    })?;
    if total_shots > config.max_total_shots {
        return Err(MisShotBudgetError::ExistingSamplesExceedBudget {
            total_shots,
            max_total_shots: config.max_total_shots,
        });
    }
    if sorted.len() == 1 {
        return Ok(MisShotBudgetDecision::Abstain {
            reason: MisShotAbstentionReason::OnlyOneCandidate,
            total_shots,
        });
    }

    let remaining_budget = config.max_total_shots - total_shots;
    if sorted.iter().all(|estimate| estimate.shots == 0) {
        return request_shots(
            &sorted,
            config.initial_shots_per_candidate,
            total_shots,
            remaining_budget,
        );
    }

    // Bonferroni spend across all possible sample counts/looks. A decision may be
    // reevaluated after every batch, so reusing alpha at each look would inflate the
    // overall false-selection probability under optional stopping.
    let alpha_per_look =
        config.familywise_error_probability / (config.max_total_shots as f64 + 1.0);
    let intervals = sorted.iter().copied()
        .map(|estimate| interval(estimate, sorted.len(), alpha_per_look))
        .collect::<Vec<_>>();
    let mut winner = intervals[0];
    for &candidate in intervals.iter().skip(1) {
        if preferred(candidate, winner, mode) {
            winner = candidate;
        }
    }

    let competitor = intervals.iter().copied()
        .filter(|candidate| candidate.estimate.vertex != winner.estimate.vertex)
        .reduce(|best, candidate| {
            if preferred(candidate, best, mode) { candidate } else { best }
        })
        .expect("at least two candidates were checked above");

    let separated = match mode {
        MisSelectionMode::PreferInSet => winner.lower > competitor.upper,
        MisSelectionMode::PreferOutSet => winner.upper < competitor.lower,
    };
    if separated {
        return Ok(MisShotBudgetDecision::Selected {
            vertex: winner.estimate.vertex,
            shots: winner.estimate.shots,
            inclusion_count: winner.estimate.inclusion_count,
            lower_bound: winner.lower,
            upper_bound: winner.upper,
            competitor_bound: match mode {
                MisSelectionMode::PreferInSet => competitor.upper,
                MisSelectionMode::PreferOutSet => competitor.lower,
            },
            total_shots,
        });
    }
    if remaining_budget == 0 {
        return Ok(MisShotBudgetDecision::Abstain {
            reason: MisShotAbstentionReason::BudgetExhausted,
            total_shots,
        });
    }

    // Candidates whose intervals could still beat the current empirical winner
    // receive additional measurements. The budget is divided evenly so the plan
    // is deterministic and cannot exceed the declared total.
    let contenders = intervals.iter().copied()
        .filter(|candidate| match mode {
            MisSelectionMode::PreferInSet => candidate.upper >= winner.lower,
            MisSelectionMode::PreferOutSet => candidate.lower <= winner.upper,
        })
        .map(|candidate| candidate.estimate)
        .collect::<Vec<_>>();
    request_shots(
        &contenders,
        config.batch_shots_per_candidate,
        total_shots,
        remaining_budget,
    )
}

fn request_shots(
    estimates: &[MisCandidateEstimate],
    requested_per_candidate: u64,
    total_shots: u64,
    remaining_budget: u64,
) -> Result<MisShotBudgetDecision, MisShotBudgetError> {
    let per_candidate = requested_per_candidate.min(remaining_budget / estimates.len() as u64);
    if per_candidate == 0 {
        return Ok(MisShotBudgetDecision::Abstain {
            reason: MisShotAbstentionReason::BudgetExhausted,
            total_shots,
        });
    }
    let allocations = estimates.iter()
        .map(|estimate| MisShotAllocation {
            vertex: estimate.vertex,
            additional_shots: per_candidate,
        })
        .collect();
    Ok(MisShotBudgetDecision::RequestMoreShots {
        allocations,
        total_shots,
        remaining_budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_total_shots: u64) -> MisShotBudgetConfig {
        MisShotBudgetConfig::new(8, 8, max_total_shots, 0.05).expect("valid config")
    }

    #[test]
    fn requests_initial_shots_without_treating_missing_as_zero() {
        let estimates = [
            MisCandidateEstimate::new(4, 0, 0).expect("empty estimate"),
            MisCandidateEstimate::new(2, 0, 0).expect("empty estimate"),
        ];
        assert_eq!(
            plan_mis_shots(&estimates, MisSelectionMode::PreferInSet, config(32))
                .expect("valid plan"),
            MisShotBudgetDecision::RequestMoreShots {
                allocations: vec![
                    MisShotAllocation { vertex: 2, additional_shots: 8 },
                    MisShotAllocation { vertex: 4, additional_shots: 8 },
                ],
                total_shots: 0,
                remaining_budget: 32,
            }
        );
    }

    #[test]
    fn requests_more_samples_when_score_intervals_overlap() {
        let estimates = [
            MisCandidateEstimate::new(0, 1000, 600).expect("valid estimate"),
            MisCandidateEstimate::new(1, 1000, 590).expect("valid estimate"),
            MisCandidateEstimate::new(2, 1000, 100).expect("valid estimate"),
        ];
        match plan_mis_shots(&estimates, MisSelectionMode::PreferInSet, config(10_000))
            .expect("valid plan")
        {
            MisShotBudgetDecision::RequestMoreShots { allocations, total_shots, .. } => {
                assert_eq!(total_shots, 3000);
                assert!(allocations.iter().any(|allocation| allocation.vertex == 0));
                assert!(allocations.iter().any(|allocation| allocation.vertex == 1));
                assert!(!allocations.iter().any(|allocation| allocation.vertex == 2));
            }
            other => panic!("expected additional shots, got {other:?}"),
        }
    }

    #[test]
    fn selects_only_after_score_intervals_separate() {
        let estimates = [
            MisCandidateEstimate::new(0, 1000, 950).expect("valid estimate"),
            MisCandidateEstimate::new(1, 1000, 20).expect("valid estimate"),
        ];
        match plan_mis_shots(&estimates, MisSelectionMode::PreferInSet, config(3000))
            .expect("valid plan")
        {
            MisShotBudgetDecision::Selected { vertex, lower_bound, competitor_bound, .. } => {
                assert_eq!(vertex, 0);
                assert!(lower_bound > competitor_bound);
            }
            other => panic!("expected separated winner, got {other:?}"),
        }
    }

    #[test]
    fn abstains_when_budget_is_exhausted_before_separation() {
        let estimates = [
            MisCandidateEstimate::new(0, 10, 6).expect("valid estimate"),
            MisCandidateEstimate::new(1, 10, 5).expect("valid estimate"),
        ];
        assert_eq!(
            plan_mis_shots(&estimates, MisSelectionMode::PreferInSet, config(20))
                .expect("valid plan"),
            MisShotBudgetDecision::Abstain {
                reason: MisShotAbstentionReason::BudgetExhausted,
                total_shots: 20,
            }
        );
    }

    #[test]
    fn rejects_duplicate_vertices_and_invalid_counts() {
        assert_eq!(
            plan_mis_shots(
                &[
                    MisCandidateEstimate::new(1, 10, 5).expect("valid"),
                    MisCandidateEstimate::new(1, 10, 4).expect("valid"),
                ],
                MisSelectionMode::PreferInSet,
                config(100),
            ).expect_err("duplicate candidate"),
            MisShotBudgetError::DuplicateCandidate { vertex: 1 }
        );
        assert_eq!(
            MisCandidateEstimate::new(2, 3, 4).expect_err("invalid counts"),
            MisShotBudgetError::InclusionCountExceedsShots {
                vertex: 2,
                inclusion_count: 4,
                shots: 3,
            }
        );
    }

    #[test]
    fn rejects_zero_initial_budget_parameters() {
        assert_eq!(
            MisShotBudgetConfig::new(0, 8, 100, 0.05).expect_err("initial batch must be positive"),
            MisShotBudgetError::ZeroInitialShots
        );
        assert_eq!(
            MisShotBudgetConfig::new(8, 0, 100, 0.05).expect_err("incremental batch must be positive"),
            MisShotBudgetError::ZeroBatchShots
        );
    }

    #[test]
    fn rejects_sample_count_overflow_instead_of_wrapping_budget() {
        assert_eq!(
            plan_mis_shots(
                &[
                    MisCandidateEstimate::new(0, u64::MAX, 0).expect("valid maximum count"),
                    MisCandidateEstimate::new(1, 1, 0).expect("valid estimate"),
                ],
                MisSelectionMode::PreferInSet,
                config(u64::MAX),
            ).expect_err("sum of per-candidate counts overflows"),
            MisShotBudgetError::TotalShotsOverflow
        );
    }

    #[test]
    fn refuses_to_hide_a_total_budget_overrun() {
        assert_eq!(
            plan_mis_shots(
                &[
                    MisCandidateEstimate::new(0, 10, 5).expect("valid"),
                    MisCandidateEstimate::new(1, 10, 5).expect("valid"),
                ],
                MisSelectionMode::PreferInSet,
                config(19),
            ).expect_err("existing shots exceed configured total"),
            MisShotBudgetError::ExistingSamplesExceedBudget {
                total_shots: 20,
                max_total_shots: 19,
            }
        );
    }
}
