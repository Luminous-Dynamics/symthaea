// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic baselines for choosing the next acoustic design experiment.
//!
//! This module ranks already-eligible design × fidelity options. It does not
//! predict uncertainty reduction, infer objective values, validate evidence, or
//! authorize physical actuation. The caller supplies declared heuristic inputs
//! and must freeze their meaning in an evaluation profile. Hard feasibility,
//! equipment safety, evidence validity and budget checks remain upstream gates.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;

/// One eligible experiment option: a design evaluated at a particular fidelity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentOption {
    /// Stable unique identifier for this design × fidelity evaluation.
    pub id: String,
    /// Stable identifier of the design being evaluated.
    pub design_id: String,
    /// Stable identifier of the solver, measurement tier or fidelity source.
    pub fidelity_id: String,
    /// Evaluation profile fixing how the relevance estimates are defined.
    pub evaluation_profile_id: String,
    /// Unit shared by all estimated costs in this evaluation.
    pub cost_unit_id: String,
    /// Estimated cost in the declared cost unit.
    pub estimated_cost_units: u64,
    /// Count of currently blocking constraint questions this option may inform.
    ///
    /// This is a relevance heuristic, not the probability of resolving a constraint.
    pub blocking_constraints_addressed: u32,
    /// Count of unresolved candidate-pair comparisons this option may inform.
    ///
    /// The same pair may be addressed by multiple options; this is not a
    /// count of independently resolved or guaranteed frontier improvements.
    pub frontier_pairs_addressed: u32,
    /// Caller-defined estimated reduction in uncertainty units under the frozen
    /// evaluation profile. Not necessarily a probability or a calibrated interval.
    pub uncertainty_reduction_units: u64,
}

impl ExperimentOption {
    /// Construct an experiment option with explicit cost and relevance estimates.
    pub fn new(
        id: impl Into<String>,
        design_id: impl Into<String>,
        fidelity_id: impl Into<String>,
        estimated_cost_units: u64,
        blocking_constraints_addressed: u32,
        frontier_pairs_addressed: u32,
        uncertainty_reduction_units: u64,
    ) -> Self {
        Self {
            id: id.into(),
            design_id: design_id.into(),
            fidelity_id: fidelity_id.into(),
            evaluation_profile_id: String::new(),
            cost_unit_id: String::new(),
            estimated_cost_units,
            blocking_constraints_addressed,
            frontier_pairs_addressed,
            uncertainty_reduction_units,
        }
    }

    /// Bind the estimates to their versioned interpretation and common cost unit.
    ///
    /// Ranking rejects an option whose contract is empty or differs from the
    /// expected ranking contract. This prevents cost ratios from comparing
    /// incompatible units or relevance scores from different versions.
    pub fn with_contract(
        mut self,
        evaluation_profile_id: impl Into<String>,
        cost_unit_id: impl Into<String>,
    ) -> Self {
        self.evaluation_profile_id = evaluation_profile_id.into();
        self.cost_unit_id = cost_unit_id.into();
        self
    }
}

/// Policy used to create a full deterministic ranking of experiment options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentSelectionPolicy {
    /// Lowest estimated cost first; stable ID breaks ties.
    CheapestFirst,
    /// Resolve the most consequential declared questions first; cost breaks ties.
    FrontierRelevanceFirst,
    /// Highest declared uncertainty-reduction proxy per cost unit first.
    UncertaintyReductionPerCost,
    /// Reproducible pseudo-random baseline, ordered by a seeded stable hash.
    SeededPseudoRandom { seed: u64 },
}

/// Invalid experiment options or mismatched evaluation contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentSelectionError {
    EmptyEvaluationProfileId,
    EmptyCostUnitId,
    EmptyOptionId,
    EmptyDesignId,
    EmptyFidelityId,
    OptionEvaluationProfileMismatch,
    OptionCostUnitMismatch,
    DuplicateOptionId,
    ZeroEstimatedCost,
    RankingOptionPoolMismatch,
    InvalidRankingPermutation,
}

impl fmt::Display for ExperimentSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyEvaluationProfileId => "evaluation profile identifier must not be empty",
            Self::EmptyCostUnitId => "cost unit identifier must not be empty",
            Self::EmptyOptionId => "experiment option identifiers must not be empty",
            Self::EmptyDesignId => "design identifiers must not be empty",
            Self::EmptyFidelityId => "fidelity identifiers must not be empty",
            Self::OptionEvaluationProfileMismatch => {
                "experiment option uses a different evaluation profile"
            }
            Self::OptionCostUnitMismatch => "experiment option uses a different cost unit",
            Self::DuplicateOptionId => "experiment option identifiers must be unique",
            Self::ZeroEstimatedCost => "estimated experiment cost must be greater than zero",
            Self::RankingOptionPoolMismatch => {
                "ranking does not match this exact input option pool and order"
            }
            Self::InvalidRankingPermutation => {
                "ranking indices must be a permutation of the input option indices"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for ExperimentSelectionError {}

/// An auditable ranking of input option indices under a frozen profile and policy.
///
/// The indices refer to the exact input slice passed to rank_experiments.
/// Persist the profile ID, cost unit, policy and ordered option IDs with each
/// benchmark receipt. Rankings are deterministic when inputs are unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentRanking {
    pub option_indices: Vec<usize>,
    /// Snapshot of option IDs in original input order; guards index reuse.
    pub input_option_ids: Vec<String>,
    pub policy: ExperimentSelectionPolicy,
    pub evaluation_profile_id: String,
    pub cost_unit_id: String,
}

/// Budget-constrained plan produced from one ranking.
///
/// Selected/skipped IDs, the profile, policy, cost unit and estimated costs form
/// an auditable planning receipt. This plan is not an authorization to execute
/// a physical measurement or hardware change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedExperimentPlan {
    pub selected_option_indices: Vec<usize>,
    pub selected_option_ids: Vec<String>,
    pub skipped_option_ids: Vec<String>,
    pub policy: ExperimentSelectionPolicy,
    pub evaluation_profile_id: String,
    pub cost_unit_id: String,
    pub budget_units: u64,
    pub estimated_cost_units: u64,
    pub unused_budget_units: u64,
}

/// Rank pre-screened experiment options using one explicit baseline policy.
///
/// Every option must use cost values in the common cost unit declared by the
/// caller, and relevance estimates must follow the same evaluation profile.
/// No expected-value or probability semantics are inferred by this function.
/// It returns a complete order so the benchmark harness can compare policies
/// while enforcing the same budget and eligibility gates.
pub fn rank_experiments(
    options: &[ExperimentOption],
    evaluation_profile_id: &str,
    cost_unit_id: &str,
    policy: ExperimentSelectionPolicy,
) -> Result<ExperimentRanking, ExperimentSelectionError> {
    if evaluation_profile_id.trim().is_empty() {
        return Err(ExperimentSelectionError::EmptyEvaluationProfileId);
    }
    if cost_unit_id.trim().is_empty() {
        return Err(ExperimentSelectionError::EmptyCostUnitId);
    }
    validate_options(options, evaluation_profile_id, cost_unit_id)?;

    let mut indices: Vec<usize> = (0..options.len()).collect();
    indices.sort_by(|left_index, right_index| {
        let left = &options[*left_index];
        let right = &options[*right_index];

        let policy_order = match policy {
            ExperimentSelectionPolicy::CheapestFirst => left
                .estimated_cost_units
                .cmp(&right.estimated_cost_units),
            ExperimentSelectionPolicy::FrontierRelevanceFirst => right
                .blocking_constraints_addressed
                .cmp(&left.blocking_constraints_addressed)
                .then_with(|| {
                    right
                        .frontier_pairs_addressed
                        .cmp(&left.frontier_pairs_addressed)
                })
                .then_with(|| compare_reduction_per_cost(left, right))
                .then_with(|| left.estimated_cost_units.cmp(&right.estimated_cost_units)),
            ExperimentSelectionPolicy::UncertaintyReductionPerCost => {
                compare_reduction_per_cost(left, right)
                    .then_with(|| {
                        right
                            .frontier_pairs_addressed
                            .cmp(&left.frontier_pairs_addressed)
                    })
                    .then_with(|| left.estimated_cost_units.cmp(&right.estimated_cost_units))
            }
            ExperimentSelectionPolicy::SeededPseudoRandom { seed } => {
                stable_hash(&left.id, seed).cmp(&stable_hash(&right.id, seed))
            }
        };

        policy_order.then_with(|| left.id.cmp(&right.id))
    });

    Ok(ExperimentRanking {
        option_indices: indices,
        input_option_ids: options.iter().map(|option| option.id.clone()).collect(),
        policy,
        evaluation_profile_id: evaluation_profile_id.to_owned(),
        cost_unit_id: cost_unit_id.to_owned(),
    })
}

/// Build a deterministic, budget-feasible plan from a ranking.
///
/// Options that do not fit the remaining budget are skipped, not treated as
/// terminal, so a cheaper later option can still be selected. The caller must
/// enforce the same budget semantics when executing or benchmarking the plan.
/// The ranking is accepted only for the exact option-ID sequence that created it.
pub fn select_under_budget(
    options: &[ExperimentOption],
    ranking: &ExperimentRanking,
    budget_units: u64,
) -> Result<BudgetedExperimentPlan, ExperimentSelectionError> {
    validate_options(
        options,
        &ranking.evaluation_profile_id,
        &ranking.cost_unit_id,
    )?;
    let input_option_ids: Vec<_> = options.iter().map(|option| option.id.clone()).collect();
    if input_option_ids != ranking.input_option_ids {
        return Err(ExperimentSelectionError::RankingOptionPoolMismatch);
    }
    if ranking.option_indices.len() != options.len() {
        return Err(ExperimentSelectionError::InvalidRankingPermutation);
    }

    let mut seen = vec![false; options.len()];
    for &index in &ranking.option_indices {
        if index >= options.len() || seen[index] {
            return Err(ExperimentSelectionError::InvalidRankingPermutation);
        }
        seen[index] = true;
    }
    if seen.iter().any(|value| !value) {
        return Err(ExperimentSelectionError::InvalidRankingPermutation);
    }

    let mut selected_option_indices = Vec::new();
    let mut selected_option_ids = Vec::new();
    let mut skipped_option_ids = Vec::new();
    let mut remaining_budget = budget_units;

    for &index in &ranking.option_indices {
        let option = &options[index];
        if option.estimated_cost_units <= remaining_budget {
            remaining_budget -= option.estimated_cost_units;
            selected_option_indices.push(index);
            selected_option_ids.push(option.id.clone());
        } else {
            skipped_option_ids.push(option.id.clone());
        }
    }

    Ok(BudgetedExperimentPlan {
        selected_option_indices,
        selected_option_ids,
        skipped_option_ids,
        policy: ranking.policy,
        evaluation_profile_id: ranking.evaluation_profile_id.clone(),
        cost_unit_id: ranking.cost_unit_id.clone(),
        budget_units,
        estimated_cost_units: budget_units - remaining_budget,
        unused_budget_units: remaining_budget,
    })
}

fn validate_options(
    options: &[ExperimentOption],
    expected_profile_id: &str,
    expected_cost_unit_id: &str,
) -> Result<(), ExperimentSelectionError> {
    let mut ids = HashSet::with_capacity(options.len());
    for option in options {
        if option.id.trim().is_empty() {
            return Err(ExperimentSelectionError::EmptyOptionId);
        }
        if option.design_id.trim().is_empty() {
            return Err(ExperimentSelectionError::EmptyDesignId);
        }
        if option.fidelity_id.trim().is_empty() {
            return Err(ExperimentSelectionError::EmptyFidelityId);
        }
        if option.evaluation_profile_id.trim().is_empty() {
            return Err(ExperimentSelectionError::EmptyEvaluationProfileId);
        }
        if option.evaluation_profile_id != expected_profile_id {
            return Err(ExperimentSelectionError::OptionEvaluationProfileMismatch);
        }
        if option.cost_unit_id.trim().is_empty() {
            return Err(ExperimentSelectionError::EmptyCostUnitId);
        }
        if option.cost_unit_id != expected_cost_unit_id {
            return Err(ExperimentSelectionError::OptionCostUnitMismatch);
        }
        if !ids.insert(option.id.as_str()) {
            return Err(ExperimentSelectionError::DuplicateOptionId);
        }
        if option.estimated_cost_units == 0 {
            return Err(ExperimentSelectionError::ZeroEstimatedCost);
        }
    }
    Ok(())
}

/// Compare reduction-per-cost ratios without floating point or overflow.
fn compare_reduction_per_cost(left: &ExperimentOption, right: &ExperimentOption) -> Ordering {
    let left_scaled =
        u128::from(left.uncertainty_reduction_units) * u128::from(right.estimated_cost_units);
    let right_scaled =
        u128::from(right.uncertainty_reduction_units) * u128::from(left.estimated_cost_units);
    right_scaled.cmp(&left_scaled)
}

/// Stable FNV-1a-style seeded ordering key; does not use process-randomized hashing.
fn stable_hash(id: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64 ^ seed;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = "synthetic-acoustics-v1";
    const COST_UNIT: &str = "normalized_cost_credit";

    fn options() -> Vec<ExperimentOption> {
        vec![
            ExperimentOption::new("physical-a", "design-a", "lab", 100, 2, 5, 100)
                .with_contract(PROFILE, COST_UNIT),
            ExperimentOption::new("analytic-b", "design-b", "analytic", 2, 0, 1, 5)
                .with_contract(PROFILE, COST_UNIT),
            ExperimentOption::new("simulation-c", "design-c", "simulation", 20, 1, 3, 80)
                .with_contract(PROFILE, COST_UNIT),
            ExperimentOption::new("physical-d", "design-d", "lab", 80, 3, 2, 120)
                .with_contract(PROFILE, COST_UNIT),
        ]
    }

    fn ids(options: &[ExperimentOption], ranking: &ExperimentRanking) -> Vec<String> {
        ranking
            .option_indices
            .iter()
            .map(|index| options[*index].id.clone())
            .collect()
    }

    #[test]
    fn cheapest_first_uses_cost_then_stable_id() {
        let options = options();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();

        assert_eq!(
            ids(&options, &ranking),
            vec!["analytic-b", "simulation-c", "physical-d", "physical-a"]
        );
    }

    #[test]
    fn frontier_relevance_prioritizes_blocking_questions_then_frontier_pairs() {
        let options = options();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::FrontierRelevanceFirst,
        )
        .unwrap();

        assert_eq!(
            ids(&options, &ranking),
            vec!["physical-d", "physical-a", "simulation-c", "analytic-b"]
        );
    }

    #[test]
    fn uncertainty_reduction_per_cost_uses_exact_integer_ratio() {
        let options = options();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::UncertaintyReductionPerCost,
        )
        .unwrap();

        assert_eq!(
            ids(&options, &ranking),
            vec!["simulation-c", "analytic-b", "physical-d", "physical-a"]
        );
    }

    #[test]
    fn seeded_pseudo_random_baseline_is_reproducible_and_a_permutation() {
        let options = options();
        let policy = ExperimentSelectionPolicy::SeededPseudoRandom { seed: 42 };
        let first = rank_experiments(&options, PROFILE, COST_UNIT, policy).unwrap();
        let second = rank_experiments(&options, PROFILE, COST_UNIT, policy).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.option_indices.len(), options.len());
        let unique: HashSet<_> = first.option_indices.iter().copied().collect();
        assert_eq!(unique.len(), options.len());
        assert_eq!(
            first.input_option_ids,
            options.iter().map(|option| option.id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn seeded_order_is_independent_of_input_order_for_unique_ids() {
        let options = options();
        let mut reversed = options.clone();
        reversed.reverse();
        let policy = ExperimentSelectionPolicy::SeededPseudoRandom { seed: 7 };
        let first = rank_experiments(&options, PROFILE, COST_UNIT, policy).unwrap();
        let second = rank_experiments(&reversed, PROFILE, COST_UNIT, policy).unwrap();

        assert_eq!(ids(&options, &first), ids(&reversed, &second));
    }

    #[test]
    fn empty_option_set_is_valid_under_a_nonempty_contract() {
        let ranking = rank_experiments(
            &[],
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();

        assert!(ranking.option_indices.is_empty());
        assert_eq!(ranking.evaluation_profile_id, PROFILE);
        assert_eq!(ranking.cost_unit_id, COST_UNIT);
    }


    #[test]
    fn budgeted_plan_never_exceeds_budget_and_skips_unaffordable_options() {
        let options = options();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        let plan = select_under_budget(&options, &ranking, 22).unwrap();

        assert_eq!(plan.selected_option_ids, vec!["analytic-b", "simulation-c"]);
        assert_eq!(plan.skipped_option_ids, vec!["physical-d", "physical-a"]);
        assert_eq!(plan.estimated_cost_units, 22);
        assert_eq!(plan.unused_budget_units, 0);
        assert!(plan.estimated_cost_units <= plan.budget_units);
    }

    #[test]
    fn budgeted_plan_checks_exact_option_pool_and_order() {
        let options = options();
        let ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        let mut reordered = options.clone();
        reordered.reverse();

        assert_eq!(
            select_under_budget(&reordered, &ranking, 100),
            Err(ExperimentSelectionError::RankingOptionPoolMismatch)
        );
    }

    #[test]
    fn budgeted_plan_rejects_mutated_non_permutation_rankings() {
        let options = options();
        let mut ranking = rank_experiments(
            &options,
            PROFILE,
            COST_UNIT,
            ExperimentSelectionPolicy::CheapestFirst,
        )
        .unwrap();
        ranking.option_indices[1] = ranking.option_indices[0];

        assert_eq!(
            select_under_budget(&options, &ranking, 100),
            Err(ExperimentSelectionError::InvalidRankingPermutation)
        );
    }

    #[test]
    fn invalid_contracts_and_options_fail_closed() {
        assert_eq!(
            rank_experiments(&[], " ", COST_UNIT, ExperimentSelectionPolicy::CheapestFirst),
            Err(ExperimentSelectionError::EmptyEvaluationProfileId)
        );
        assert_eq!(
            rank_experiments(&[], PROFILE, " ", ExperimentSelectionPolicy::CheapestFirst),
            Err(ExperimentSelectionError::EmptyCostUnitId)
        );
        assert_eq!(
            rank_experiments(
                &[ExperimentOption::new("x", "design", "lab", 0, 0, 0, 0)
                    .with_contract(PROFILE, COST_UNIT)],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::ZeroEstimatedCost)
        );
        assert_eq!(
            rank_experiments(
                &[ExperimentOption::new("x", "design", " ", 1, 0, 0, 0)
                    .with_contract(PROFILE, COST_UNIT)],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::EmptyFidelityId)
        );

        assert_eq!(
            rank_experiments(
                &[ExperimentOption::new("wrong-profile", "d", "lab", 1, 0, 0, 0)
                    .with_contract("another-profile", COST_UNIT)],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::OptionEvaluationProfileMismatch)
        );
        assert_eq!(
            rank_experiments(
                &[ExperimentOption::new("wrong-cost-unit", "d", "lab", 1, 0, 0, 0)
                    .with_contract(PROFILE, "seconds")],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::OptionCostUnitMismatch)
        );
        assert_eq!(
            rank_experiments(
                &[
                    ExperimentOption::new("x", "design", "lab", 1, 0, 0, 0)
                        .with_contract(PROFILE, COST_UNIT),
                    ExperimentOption::new("x", "other-design", "simulation", 2, 0, 0, 0)
                        .with_contract(PROFILE, COST_UNIT),
                ],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::DuplicateOptionId)
        );
        assert_eq!(
            rank_experiments(
                &[ExperimentOption::new("x", " ", "lab", 1, 0, 0, 0)
                    .with_contract(PROFILE, COST_UNIT)],
                PROFILE,
                COST_UNIT,
                ExperimentSelectionPolicy::CheapestFirst
            ),
            Err(ExperimentSelectionError::EmptyDesignId)
        );
    }
}
