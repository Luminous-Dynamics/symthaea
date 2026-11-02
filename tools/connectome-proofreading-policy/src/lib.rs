#![forbid(unsafe_code)]

//! Deterministic, risk-aware review-budget allocation for connectome proofreading.
//!
//! This is a policy primitive, not an image classifier, edit engine, or biological
//! inference system. It ranks cases by expected avoidable error cost per reviewer-second.
//! The current allocator is a greedy heuristic; it is not an exact knapsack solver.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofreadingTask {
    SplitError,
    MergeError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReviewCandidate {
    pub operation_id: String,
    pub task: ProofreadingTask,
    /// Calibrated probability that this operation contains an error, in [0, 1].
    pub error_probability: f64,
    /// Preregistered missed-error cost in the declared cost profile.
    pub missed_error_cost_units: f64,
    /// Probability that expert review resolves the error, conditional on review.
    /// Use an independently measured value; 1.0 is an ideal-review upper bound.
    pub review_resolution_probability: f64,
    /// Review duration in integer seconds, measured under the declared protocol.
    pub review_seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReviewBudget {
    pub seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMethod {
    GreedyExpectedAvoidedCostPerSecond,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReviewPlan {
    /// Selected operation IDs, sorted lexicographically for stable receipts.
    pub selected_operation_ids: Vec<String>,
    /// Deferred operation IDs, sorted lexicographically for stable receipts.
    pub deferred_operation_ids: Vec<String>,
    pub spent_seconds: u64,
    pub budget_seconds: u32,
    /// Sum(error_probability * missed_error_cost_units) over all candidates.
    pub expected_baseline_error_cost_units: f64,
    /// Sum(probability * missed cost * review resolution probability) for selected cases.
    pub expected_avoided_error_cost_units: f64,
    /// Baseline expected cost less the expected benefit of selected review cases.
    pub expected_residual_error_cost_units: f64,
    pub allocation_method: AllocationMethod,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    EmptyOperationId,
    NonCanonicalOperationId(String),
    DuplicateOperationId(String),
    InvalidProbability {
        operation_id: String,
        field: &'static str,
    },
    InvalidCost(String),
    ZeroReviewDuration(String),
    NonFiniteExpectedCost(String),
    ExpectedCostOverflow,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOperationId => write!(f, "operation ID must not be empty"),
            Self::NonCanonicalOperationId(id) => {
                write!(f, "operation ID has leading or trailing whitespace: {id:?}")
            }
            Self::DuplicateOperationId(id) => write!(f, "duplicate operation ID: {id}"),
            Self::InvalidProbability { operation_id, field } => {
                write!(f, "invalid {field} for operation {operation_id}")
            }
            Self::InvalidCost(id) => write!(f, "invalid missed-error cost for operation {id}"),
            Self::ZeroReviewDuration(id) => {
                write!(f, "review duration must be positive for operation {id}")
            }
            Self::NonFiniteExpectedCost(id) => {
                write!(f, "expected cost is not finite for operation {id}")
            }
            Self::ExpectedCostOverflow => write!(f, "aggregate expected cost overflowed"),
        }
    }
}

impl std::error::Error for PolicyError {}

#[derive(Debug, Clone)]
struct ScoredCandidate<'a> {
    candidate: &'a ReviewCandidate,
    expected_baseline_cost: f64,
    expected_avoided_cost: f64,
    avoided_cost_per_second: f64,
}

/// Allocate a reviewer-time budget using a deterministic greedy priority rule.
///
/// Priority is:
///
/// `error_probability * missed_error_cost_units * review_resolution_probability
///     / review_seconds`
///
/// Candidates are ordered by descending priority, then ascending operation ID. A candidate
/// that does not fit in the remaining budget is skipped so that shorter cases can still fit.
/// Zero-benefit candidates are deferred.
///
/// **Qualification:** value-density greedy allocation is not globally optimal for every
/// indivisible-budget/variable-duration instance. Compare it with an exact small-cohort
/// knapsack oracle in benchmark evaluation. For equal review durations this ranking selects
/// the highest expected-benefit cases. The returned expected residual cost assumes the
/// supplied review-resolution probabilities are calibrated and transportable.
///
/// The function never inspects hidden labels and never claims that review was completed.
pub fn plan_reviews(
    candidates: &[ReviewCandidate],
    budget: ReviewBudget,
) -> Result<ReviewPlan, PolicyError> {
    let mut seen = BTreeSet::new();
    let mut scored = Vec::with_capacity(candidates.len());
    let mut expected_baseline_error_cost_units = 0.0_f64;

    for candidate in candidates {
        let id = candidate.operation_id.as_str();
        if id.trim().is_empty() {
            return Err(PolicyError::EmptyOperationId);
        }
        if id.trim() != id {
            return Err(PolicyError::NonCanonicalOperationId(id.to_owned()));
        }
        if !seen.insert(id.to_owned()) {
            return Err(PolicyError::DuplicateOperationId(id.to_owned()));
        }
        validate_probability(id, "error_probability", candidate.error_probability)?;
        validate_probability(
            id,
            "review_resolution_probability",
            candidate.review_resolution_probability,
        )?;
        if !candidate.missed_error_cost_units.is_finite()
            || candidate.missed_error_cost_units < 0.0
        {
            return Err(PolicyError::InvalidCost(id.to_owned()));
        }
        if candidate.review_seconds == 0 {
            return Err(PolicyError::ZeroReviewDuration(id.to_owned()));
        }

        let expected_baseline_cost =
            candidate.error_probability * candidate.missed_error_cost_units;
        let expected_avoided_cost =
            expected_baseline_cost * candidate.review_resolution_probability;
        let avoided_cost_per_second =
            expected_avoided_cost / f64::from(candidate.review_seconds);

        if !expected_baseline_cost.is_finite()
            || !expected_avoided_cost.is_finite()
            || !avoided_cost_per_second.is_finite()
        {
            return Err(PolicyError::NonFiniteExpectedCost(id.to_owned()));
        }
        expected_baseline_error_cost_units += expected_baseline_cost;
        if !expected_baseline_error_cost_units.is_finite() {
            return Err(PolicyError::ExpectedCostOverflow);
        }

        scored.push(ScoredCandidate {
            candidate,
            expected_baseline_cost,
            expected_avoided_cost,
            avoided_cost_per_second,
        });
    }

    scored.sort_by(|left, right| {
        right
            .avoided_cost_per_second
            .total_cmp(&left.avoided_cost_per_second)
            .then_with(|| {
                left.candidate
                    .operation_id
                    .cmp(&right.candidate.operation_id)
            })
    });

    let mut remaining_seconds = budget.seconds;
    let mut spent_seconds = 0_u64;
    let mut expected_avoided_error_cost_units = 0.0_f64;
    let mut selected_operation_ids = Vec::new();
    let mut deferred_operation_ids = Vec::new();

    for scored_candidate in scored {
        let candidate = scored_candidate.candidate;
        let cost = candidate.review_seconds;

        if scored_candidate.expected_avoided_cost > 0.0 && cost <= remaining_seconds {
            remaining_seconds -= cost;
            spent_seconds += u64::from(cost);
            expected_avoided_error_cost_units += scored_candidate.expected_avoided_cost;
            selected_operation_ids.push(candidate.operation_id.clone());
        } else {
            deferred_operation_ids.push(candidate.operation_id.clone());
        }
    }

    selected_operation_ids.sort();
    deferred_operation_ids.sort();

    if !expected_avoided_error_cost_units.is_finite() {
        return Err(PolicyError::ExpectedCostOverflow);
    }

    let expected_residual_error_cost_units =
        (expected_baseline_error_cost_units - expected_avoided_error_cost_units).max(0.0);

    Ok(ReviewPlan {
        selected_operation_ids,
        deferred_operation_ids,
        spent_seconds,
        budget_seconds: budget.seconds,
        expected_baseline_error_cost_units,
        expected_avoided_error_cost_units,
        expected_residual_error_cost_units,
        allocation_method: AllocationMethod::GreedyExpectedAvoidedCostPerSecond,
    })
}

fn validate_probability(
    operation_id: &str,
    field: &'static str,
    value: f64,
) -> Result<(), PolicyError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PolicyError::InvalidProbability {
            operation_id: operation_id.to_owned(),
            field,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        operation_id: &str,
        error_probability: f64,
        missed_cost: f64,
        resolution_probability: f64,
        review_seconds: u32,
    ) -> ReviewCandidate {
        ReviewCandidate {
            operation_id: operation_id.to_owned(),
            task: ProofreadingTask::SplitError,
            error_probability,
            missed_error_cost_units: missed_cost,
            review_resolution_probability: resolution_probability,
            review_seconds,
        }
    }

    #[test]
    fn known_answer_selects_highest_expected_benefit_for_equal_duration() {
        let candidates = vec![
            candidate("op-A", 0.95, 10.0, 1.0, 120),
            candidate("op-B", 0.80, 10.0, 1.0, 120),
            candidate("op-C", 0.40, 10.0, 1.0, 120),
            candidate("op-D", 0.10, 10.0, 1.0, 120),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 240 }).unwrap();

        assert_eq!(plan.selected_operation_ids, vec!["op-A", "op-B"]);
        assert_eq!(plan.deferred_operation_ids, vec!["op-C", "op-D"]);
        assert_eq!(plan.spent_seconds, 240);
        assert!((plan.expected_baseline_error_cost_units - 22.5).abs() < 1e-12);
        assert!((plan.expected_avoided_error_cost_units - 17.5).abs() < 1e-12);
        assert!((plan.expected_residual_error_cost_units - 5.0).abs() < 1e-12);
    }

    #[test]
    fn cost_aware_priority_can_differ_from_uncertainty_only() {
        let candidates = vec![
            candidate("low-cost-high-probability", 0.90, 1.0, 1.0, 60),
            candidate("high-cost-lower-probability", 0.40, 5.0, 1.0, 60),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 60 }).unwrap();

        assert_eq!(
            plan.selected_operation_ids,
            vec!["high-cost-lower-probability"]
        );
    }

    #[test]
    fn review_resolution_probability_scales_expected_benefit() {
        let candidates = vec![
            candidate("certain-review", 0.5, 10.0, 1.0, 60),
            candidate("uncertain-review", 0.9, 10.0, 0.1, 60),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 60 }).unwrap();

        assert_eq!(plan.selected_operation_ids, vec!["certain-review"]);
        assert!((plan.expected_avoided_error_cost_units - 5.0).abs() < 1e-12);
    }

    #[test]
    fn equal_priority_uses_lexicographic_operation_id_tie_break() {
        let candidates = vec![
            candidate("zeta", 0.5, 2.0, 1.0, 30),
            candidate("alpha", 0.5, 2.0, 1.0, 30),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 30 }).unwrap();

        assert_eq!(plan.selected_operation_ids, vec!["alpha"]);
        assert_eq!(plan.deferred_operation_ids, vec!["zeta"]);
    }

    #[test]
    fn zero_budget_defers_every_candidate_without_dropping_expected_risk() {
        let candidates = vec![candidate("case-1", 0.5, 4.0, 1.0, 60)];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 0 }).unwrap();

        assert!(plan.selected_operation_ids.is_empty());
        assert_eq!(plan.deferred_operation_ids, vec!["case-1"]);
        assert_eq!(plan.expected_baseline_error_cost_units, 2.0);
        assert_eq!(plan.expected_avoided_error_cost_units, 0.0);
        assert_eq!(plan.expected_residual_error_cost_units, 2.0);
    }

    #[test]
    fn skips_case_that_does_not_fit_and_uses_remaining_budget() {
        let candidates = vec![
            candidate("long", 1.0, 10.0, 1.0, 90),
            candidate("short-a", 0.5, 10.0, 1.0, 30),
            candidate("short-b", 0.4, 10.0, 1.0, 30),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 60 }).unwrap();

        assert_eq!(plan.selected_operation_ids, vec!["short-a", "short-b"]);
        assert_eq!(plan.spent_seconds, 60);
    }

    #[test]
    fn documents_greedy_is_not_an_exact_knapsack_solver() {
        let candidates = vec![
            candidate("one-large", 1.0, 100.0, 1.0, 70),
            candidate("two-small-a", 1.0, 60.0, 1.0, 40),
            candidate("two-small-b", 1.0, 60.0, 1.0, 40),
        ];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 70 }).unwrap();

        // Greedy value-density chooses a 60-unit case. An exact knapsack solver would
        // choose the 100-unit case. This is intentional disclosure, not a correctness bug.
        assert_eq!(plan.selected_operation_ids, vec!["two-small-a"]);
        assert_eq!(plan.expected_avoided_error_cost_units, 60.0);
    }

    #[test]
    fn rejects_duplicate_operation_ids() {
        let candidates = vec![
            candidate("same", 0.5, 1.0, 1.0, 30),
            candidate("same", 0.4, 1.0, 1.0, 30),
        ];

        assert_eq!(
            plan_reviews(&candidates, ReviewBudget { seconds: 60 }),
            Err(PolicyError::DuplicateOperationId("same".to_owned()))
        );
    }

    #[test]
    fn rejects_invalid_probabilities_costs_durations_and_noncanonical_ids() {
        assert!(matches!(
            plan_reviews(
                &[candidate("bad", f64::NAN, 1.0, 1.0, 30)],
                ReviewBudget { seconds: 30 }
            ),
            Err(PolicyError::InvalidProbability { .. })
        ));
        assert_eq!(
            plan_reviews(
                &[candidate("bad", 0.5, -1.0, 1.0, 30)],
                ReviewBudget { seconds: 30 }
            ),
            Err(PolicyError::InvalidCost("bad".to_owned()))
        );
        assert_eq!(
            plan_reviews(
                &[candidate("bad", 0.5, 1.0, 1.0, 0)],
                ReviewBudget { seconds: 30 }
            ),
            Err(PolicyError::ZeroReviewDuration("bad".to_owned()))
        );
        assert_eq!(
            plan_reviews(
                &[candidate(" padded ", 0.5, 1.0, 1.0, 30)],
                ReviewBudget { seconds: 30 }
            ),
            Err(PolicyError::NonCanonicalOperationId(" padded ".to_owned()))
        );
        assert!(matches!(
            plan_reviews(
                &[candidate("bad", 0.5, 1.0, f64::INFINITY, 30)],
                ReviewBudget { seconds: 30 }
            ),
            Err(PolicyError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn rejects_expected_cost_overflow() {
        let candidates = vec![candidate("overflow", 1.0, f64::MAX, 1.0, 1)];

        assert_eq!(
            plan_reviews(&candidates, ReviewBudget { seconds: 1 }),
            Err(PolicyError::NonFiniteExpectedCost("overflow".to_owned()))
        );
    }

    #[test]
    fn zero_benefit_candidates_are_not_reviewed() {
        let candidates = vec![candidate("zero", 0.0, 100.0, 1.0, 10)];

        let plan = plan_reviews(&candidates, ReviewBudget { seconds: 10 }).unwrap();

        assert!(plan.selected_operation_ids.is_empty());
        assert_eq!(plan.deferred_operation_ids, vec!["zero"]);
    }
}
