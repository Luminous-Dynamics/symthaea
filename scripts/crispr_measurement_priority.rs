//! Decision-critical assay allocation for partial-label CRISPR ranking comparisons.
//!
//! Compile and run without Cargo or network access:
//!   rustc --edition=2024 --test scripts/crispr_measurement_priority.rs -o /tmp/crispr-measurement-tests
//!   /tmp/crispr-measurement-tests
//!
//! This is a narrow measurement-planning heuristic, not an expected-value-of-information
//! estimator. It prioritizes UNKNOWN outcomes whose top-K membership differs between two
//! frozen policies, because observing such an outcome can resolve one unit of uncertainty
//! in the paired top-K positive-hit numerator. It does not resolve unknown outcomes common
//! to both policies, denominator uncertainty, positives omitted from the candidate universe,
//! or any biological/clinical safety question.
//!
//! The input is one shared, complete candidate universe with full one-based ranks under both
//! policies. Source rights, artifact hashes, rank reconstruction, biological interpretation,
//! and a fair comparison with random/score-only allocation baselines are separate gates.

use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LabelState {
    Positive,
    AssayDefinedNegative,
    UnknownUnassayed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssayCandidate {
    pub id: String,
    pub context: String,
    /// Full, one-based rank in policy A over the shared frozen universe.
    pub rank_a: usize,
    /// Full, one-based rank in policy B over the same shared frozen universe.
    pub rank_b: usize,
    pub label: LabelState,
    /// Positive integer cost proxy declared before allocation (for example, normalized
    /// review/measurement units). It is not a dollar or laboratory-cost claim.
    pub assay_cost_units: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DifferentialSide {
    /// Candidate is in A's top K and outside B's top K.
    PolicyAOnly,
    /// Candidate is in B's top K and outside A's top K.
    PolicyBOnly,
}

impl DifferentialSide {
    /// Contribution to (A positive hits - B positive hits) if the candidate is positive.
    pub fn positive_hit_delta(self) -> i8 {
        match self {
            Self::PolicyAOnly => 1,
            Self::PolicyBOnly => -1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedAssay {
    pub candidate_id: String,
    pub context: String,
    pub cost_units: u64,
    pub differential_side: DifferentialSide,
    pub positive_hit_delta_if_positive: i8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationPlan {
    pub review_budget_k: usize,
    pub budget_units: u64,
    pub spent_units: u64,
    pub unspent_units: u64,
    pub initial_unknown_count: usize,
    pub initially_differential_unknown_count: usize,
    pub unknown_a_only_count: usize,
    pub unknown_b_only_count: usize,
    pub unknown_same_membership_count: usize,
    pub selected: Vec<SelectedAssay>,
    /// Remaining unknown outcomes that could directly change the paired top-K positive-hit
    /// numerator. It is not the total remaining uncertainty in recall or policy superiority.
    pub unresolved_differential_unknown_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllocationError {
    NoCandidates,
    ZeroReviewBudget,
    TooFewCandidates { requested: usize, available: usize },
    EmptyIdentity { candidate_id: String },
    DuplicateCandidateId(String),
    DuplicateRank { policy: char, rank: usize },
    NonContiguousRanks { policy: char, expected: usize },
    ZeroAssayCost { candidate_id: String },
    InvalidInput(String),
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidates => write!(f, "candidate universe must not be empty"),
            Self::ZeroReviewBudget => write!(f, "top-K review budget must be greater than zero"),
            Self::TooFewCandidates { requested, available } => write!(
                f,
                "top-K budget {requested} is not estimable with only {available} candidates"
            ),
            Self::EmptyIdentity { candidate_id } => {
                write!(f, "candidate {candidate_id:?} must have a non-empty id and context")
            }
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate id: {id}"),
            Self::DuplicateRank { policy, rank } => {
                write!(f, "policy {policy} has duplicate rank {rank}")
            }
            Self::NonContiguousRanks { policy, expected } => write!(
                f,
                "policy {policy} ranking must contain each one-based rank 1..=N; missing rank {expected}"
            ),
            Self::ZeroAssayCost { candidate_id } => {
                write!(f, "candidate {candidate_id} must have positive declared cost units")
            }
            Self::InvalidInput(message) => write!(f, "invalid allocation input: {message}"),
        }
    }
}

impl std::error::Error for AllocationError {}

fn validate_ranks(
    candidates: &[AssayCandidate],
    policy: char,
) -> Result<(), AllocationError> {
    let mut ranks = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        let rank = if policy == 'A' {
            candidate.rank_a
        } else {
            candidate.rank_b
        };
        if !ranks.insert(rank) {
            return Err(AllocationError::DuplicateRank { policy, rank });
        }
    }

    for expected in 1..=candidates.len() {
        if !ranks.contains(&expected) {
            return Err(AllocationError::NonContiguousRanks { policy, expected });
        }
    }
    Ok(())
}

fn validate_universe(candidates: &[AssayCandidate], k: usize) -> Result<(), AllocationError> {
    if candidates.is_empty() {
        return Err(AllocationError::NoCandidates);
    }
    if k == 0 {
        return Err(AllocationError::ZeroReviewBudget);
    }
    if candidates.len() < k {
        return Err(AllocationError::TooFewCandidates {
            requested: k,
            available: candidates.len(),
        });
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.id.trim().is_empty() || candidate.context.trim().is_empty() {
            return Err(AllocationError::EmptyIdentity {
                candidate_id: candidate.id.clone(),
            });
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(AllocationError::DuplicateCandidateId(candidate.id.clone()));
        }
        if candidate.assay_cost_units == 0 {
            return Err(AllocationError::ZeroAssayCost {
                candidate_id: candidate.id.clone(),
            });
        }
    }

    validate_ranks(candidates, 'A')?;
    validate_ranks(candidates, 'B')?;
    Ok(())
}

/// Build a cost-bounded plan that maximizes the number of directly decision-differential
/// unknown outcomes measured, for the fixed top-K comparison of policies A and B.
///
/// Since every eligible unknown outcome resolves exactly one unit of top-K hit-count
/// ambiguity, the optimal cardinality plan is the cheapest eligible candidates first.
/// Ties are deterministic (context, candidate ID). The planner intentionally does not
/// spend leftover budget on same-membership unknowns; those may be valuable for other
/// objectives but do not directly affect this paired numerator component.
pub fn plan_decision_differential_assays(
    candidates: &[AssayCandidate],
    review_budget_k: usize,
    budget_units: u64,
) -> Result<AllocationPlan, AllocationError> {
    validate_universe(candidates, review_budget_k)?;

    let mut initial_unknown_count = 0_usize;
    let mut unknown_a_only_count = 0_usize;
    let mut unknown_b_only_count = 0_usize;
    let mut unknown_same_membership_count = 0_usize;
    let mut eligible: Vec<(&AssayCandidate, DifferentialSide)> = Vec::new();

    for candidate in candidates {
        if candidate.label != LabelState::UnknownUnassayed {
            continue;
        }
        initial_unknown_count += 1;
        let in_a_top_k = candidate.rank_a <= review_budget_k;
        let in_b_top_k = candidate.rank_b <= review_budget_k;
        match (in_a_top_k, in_b_top_k) {
            (true, false) => {
                unknown_a_only_count += 1;
                eligible.push((candidate, DifferentialSide::PolicyAOnly));
            }
            (false, true) => {
                unknown_b_only_count += 1;
                eligible.push((candidate, DifferentialSide::PolicyBOnly));
            }
            _ => unknown_same_membership_count += 1,
        }
    }

    let initially_differential_unknown_count = eligible.len();
    eligible.sort_by(|(left, _), (right, _)| {
        left.assay_cost_units
            .cmp(&right.assay_cost_units)
            .then_with(|| left.context.cmp(&right.context))
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut spent_units = 0_u64;
    let mut selected = Vec::new();
    for (candidate, side) in eligible {
        if candidate.assay_cost_units > budget_units - spent_units {
            // Eligible candidates are sorted by nondecreasing cost; no later candidate can fit.
            break;
        }
        spent_units += candidate.assay_cost_units;
        selected.push(SelectedAssay {
            candidate_id: candidate.id.clone(),
            context: candidate.context.clone(),
            cost_units: candidate.assay_cost_units,
            differential_side: side,
            positive_hit_delta_if_positive: side.positive_hit_delta(),
        });
    }

    let unresolved_differential_unknown_count =
        initially_differential_unknown_count - selected.len();
    Ok(AllocationPlan {
        review_budget_k,
        budget_units,
        spent_units,
        unspent_units: budget_units - spent_units,
        initial_unknown_count,
        initially_differential_unknown_count,
        unknown_a_only_count,
        unknown_b_only_count,
        unknown_same_membership_count,
        selected,
        unresolved_differential_unknown_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &str,
        context: &str,
        rank_a: usize,
        rank_b: usize,
        label: LabelState,
        cost: u64,
    ) -> AssayCandidate {
        AssayCandidate {
            id: id.to_owned(),
            context: context.to_owned(),
            rank_a,
            rank_b,
            label,
            assay_cost_units: cost,
        }
    }

    fn fixture() -> Vec<AssayCandidate> {
        vec![
            candidate("u-a", "guide-a", 1, 5, LabelState::UnknownUnassayed, 2),
            candidate("u-b", "guide-a", 5, 2, LabelState::UnknownUnassayed, 1),
            candidate("u-common", "guide-a", 2, 1, LabelState::UnknownUnassayed, 3),
            candidate("p1", "guide-a", 3, 3, LabelState::Positive, 1),
            candidate("n1", "guide-b", 4, 4, LabelState::AssayDefinedNegative, 1),
            candidate("u-neither", "guide-b", 6, 6, LabelState::UnknownUnassayed, 1),
        ]
    }

    #[test]
    fn prioritizes_unknown_cases_with_differing_top_k_membership() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 3).unwrap();
        assert_eq!(plan.initial_unknown_count, 4);
        assert_eq!(plan.initially_differential_unknown_count, 2);
        assert_eq!(plan.unknown_a_only_count, 1);
        assert_eq!(plan.unknown_b_only_count, 1);
        assert_eq!(plan.unknown_same_membership_count, 2);
    }

    #[test]
    fn cost_bounded_plan_selects_cheapest_discordant_case() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 2).unwrap();
        assert_eq!(plan.selected.len(), 1);
        assert_eq!(plan.selected[0].candidate_id, "u-b");
        assert_eq!(plan.spent_units, 1);
        assert_eq!(plan.unspent_units, 1);
        assert_eq!(plan.unresolved_differential_unknown_count, 1);
    }

    #[test]
    fn budget_that_covers_both_discordant_labels_resolves_both() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 3).unwrap();
        assert_eq!(
            plan.selected.iter().map(|x| x.candidate_id.as_str()).collect::<Vec<_>>(),
            vec!["u-b", "u-a"]
        );
        assert_eq!(plan.spent_units, 3);
        assert_eq!(plan.unresolved_differential_unknown_count, 0);
    }

    #[test]
    fn common_top_k_unknown_does_not_enter_numerator_priority_set() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 10).unwrap();
        assert!(!plan.selected.iter().any(|x| x.candidate_id == "u-common"));
        assert_eq!(plan.unknown_same_membership_count, 2);
    }

    #[test]
    fn outside_both_top_k_unknown_is_not_mislabeled_as_direct_numerator_value() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 10).unwrap();
        assert!(!plan.selected.iter().any(|x| x.candidate_id == "u-neither"));
    }

    #[test]
    fn known_outcomes_are_never_selected_for_measurement() {
        let mut cases = fixture();
        for item in &mut cases {
            if item.id.starts_with('u') {
                item.label = LabelState::AssayDefinedNegative;
            }
        }
        let plan = plan_decision_differential_assays(&cases, 3, 100).unwrap();
        assert!(plan.selected.is_empty());
        assert_eq!(plan.initial_unknown_count, 0);
    }

    #[test]
    fn zero_budget_returns_empty_plan_without_spending() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 0).unwrap();
        assert!(plan.selected.is_empty());
        assert_eq!(plan.spent_units, 0);
        assert_eq!(plan.unresolved_differential_unknown_count, 2);
    }

    #[test]
    fn no_discordant_unknowns_returns_empty_plan() {
        let cases = vec![
            candidate("u1", "g", 1, 1, LabelState::UnknownUnassayed, 1),
            candidate("u2", "g", 2, 2, LabelState::UnknownUnassayed, 1),
            candidate("p", "g", 3, 3, LabelState::Positive, 1),
        ];
        let plan = plan_decision_differential_assays(&cases, 2, 10).unwrap();
        assert!(plan.selected.is_empty());
        assert_eq!(plan.initially_differential_unknown_count, 0);
    }

    #[test]
    fn empty_universe_is_rejected() {
        assert_eq!(
            plan_decision_differential_assays(&[], 1, 1),
            Err(AllocationError::NoCandidates)
        );
    }

    #[test]
    fn zero_k_is_rejected() {
        assert_eq!(
            plan_decision_differential_assays(&fixture(), 0, 1),
            Err(AllocationError::ZeroReviewBudget)
        );
    }

    #[test]
    fn underfilled_universe_is_not_scored_at_k() {
        assert_eq!(
            plan_decision_differential_assays(&fixture()[..2], 3, 1),
            Err(AllocationError::TooFewCandidates { requested: 3, available: 2 })
        );
    }

    #[test]
    fn duplicate_candidate_ids_fail_closed() {
        let mut cases = fixture();
        cases[1].id = cases[0].id.clone();
        assert_eq!(
            plan_decision_differential_assays(&cases, 3, 1),
            Err(AllocationError::DuplicateCandidateId("u-a".to_owned()))
        );
    }

    #[test]
    fn duplicate_rank_in_policy_a_fails_closed() {
        let mut cases = fixture();
        cases[1].rank_a = cases[0].rank_a;
        assert!(matches!(
            plan_decision_differential_assays(&cases, 3, 1),
            Err(AllocationError::DuplicateRank { policy: 'A', .. })
        ));
    }

    #[test]
    fn duplicate_rank_in_policy_b_fails_closed() {
        let mut cases = fixture();
        cases[1].rank_b = cases[0].rank_b;
        assert!(matches!(
            plan_decision_differential_assays(&cases, 3, 1),
            Err(AllocationError::DuplicateRank { policy: 'B', .. })
        ));
    }

    #[test]
    fn non_contiguous_ranks_fail_closed() {
        let mut cases = fixture();
        cases[1].rank_a = 7;
        assert!(matches!(
            plan_decision_differential_assays(&cases, 3, 1),
            Err(AllocationError::NonContiguousRanks { policy: 'A', .. })
        ));
    }

    #[test]
    fn zero_assay_cost_fails_closed() {
        let mut cases = fixture();
        cases[0].assay_cost_units = 0;
        assert_eq!(
            plan_decision_differential_assays(&cases, 3, 10),
            Err(AllocationError::ZeroAssayCost { candidate_id: "u-a".to_owned() })
        );
    }

    #[test]
    fn blank_identity_fails_closed() {
        let mut cases = fixture();
        cases[0].context = " ".to_owned();
        assert_eq!(
            plan_decision_differential_assays(&cases, 3, 1),
            Err(AllocationError::EmptyIdentity { candidate_id: "u-a".to_owned() })
        );
    }

    #[test]
    fn selected_policy_sign_is_explicit() {
        let plan = plan_decision_differential_assays(&fixture(), 3, 3).unwrap();
        let a = plan.selected.iter().find(|x| x.candidate_id == "u-a").unwrap();
        let b = plan.selected.iter().find(|x| x.candidate_id == "u-b").unwrap();
        assert_eq!(a.positive_hit_delta_if_positive, 1);
        assert_eq!(b.positive_hit_delta_if_positive, -1);
    }

    #[test]
    fn equal_cost_ties_have_deterministic_context_then_id_order() {
        let cases = vec![
            candidate("z", "b", 1, 3, LabelState::UnknownUnassayed, 1),
            candidate("y", "a", 3, 1, LabelState::UnknownUnassayed, 1),
            candidate("a", "a", 2, 2, LabelState::Positive, 1),
        ];
        let plan = plan_decision_differential_assays(&cases, 1, 2).unwrap();
        assert_eq!(
            plan.selected.iter().map(|x| x.candidate_id.as_str()).collect::<Vec<_>>(),
            vec!["y", "z"]
        );
    }
}
