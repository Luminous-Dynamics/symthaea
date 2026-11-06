// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic interval algebra for bounded few-for-many portfolios.
//!
//! Consumes externally qualified, jointly simultaneous action × objective
//! intervals. It does not estimate means or establish confidence coverage.
//! The caller must freeze the feasible action universe and objective
//! coefficients before these deterministic bounds can support a decision.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const FEW_FOR_MANY_PORTFOLIO_BOUND_METHOD_ID: &str =
    "spectra-few-for-many-portfolio-degradation-interval-v1";

/// Direction-normalized, pre-registered positive coefficient for an objective.
/// The coefficient must already account for any unit normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioObjectiveSpec {
    pub objective_id: u16,
    pub coefficient: f64,
}

/// Simultaneous interval for one feasible action and one objective.
/// Larger values are better; orient minimization objectives before ingestion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionObjectiveInterval {
    pub action_id: u32,
    pub objective_id: u16,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Frozen contract. feasible_action_ids must contain only actions whose
/// mandatory feasibility gates were qualified upstream. This module does not
/// authenticate those gates or their provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct FewForManyPortfolioContract {
    pub feasible_action_ids: Vec<u32>,
    pub objectives: Vec<PortfolioObjectiveSpec>,
    pub max_portfolio_size: usize,
    pub epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveDegradationBound {
    pub objective_id: u16,
    pub ideal_lower_bound: f64,
    pub ideal_upper_bound: f64,
    pub portfolio_best_lower_bound: f64,
    pub portfolio_best_upper_bound: f64,
    pub degradation_lower_bound: f64,
    pub degradation_upper_bound: f64,
    pub coefficient: f64,
    pub weighted_lower_bound: f64,
    pub weighted_upper_bound: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortfolioDegradationStatus {
    CertifiedWithinThreshold,
    CertifiedAboveThreshold,
    Unresolved,
}

/// Deterministic result conditional on the upstream simultaneous intervals
/// and feasibility contract. This is not a cryptographic receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct FewForManyPortfolioBound {
    pub method_id: &'static str,
    pub portfolio_action_ids: Vec<u32>,
    pub max_portfolio_size: usize,
    pub epsilon: f64,
    pub objective_bounds: Vec<ObjectiveDegradationBound>,
    pub degradation_lower_bound: f64,
    pub degradation_upper_bound: f64,
    pub status: PortfolioDegradationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FewForManyPortfolioError {
    EmptyFeasibleActionSet,
    DuplicateFeasibleAction,
    EmptyObjectiveSet,
    DuplicateObjective,
    InvalidObjectiveCoefficient,
    InvalidPortfolioSize,
    InvalidEpsilon,
    EmptyIntervalGrid,
    DuplicateActionObjectiveInterval,
    UnknownIntervalAction,
    UnknownIntervalObjective,
    MissingActionObjectiveInterval,
    InvalidUtilityInterval,
    EmptyPortfolio,
    DuplicatePortfolioAction,
    UnknownPortfolioAction,
    PortfolioTooLarge,
    EmptyEnumerationUniverse,
    DuplicateEnumerationAction,
    InvalidEnumerationLimit,
    EnumerationCountOverflow,
    EnumerationBudgetExceeded,
}

impl fmt::Display for FewForManyPortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyFeasibleActionSet => "the frozen feasible action set must be nonempty",
            Self::DuplicateFeasibleAction => "the frozen feasible action set contains a duplicate action ID",
            Self::EmptyObjectiveSet => "the objective specification must be nonempty",
            Self::DuplicateObjective => "the objective specification contains a duplicate objective ID",
            Self::InvalidObjectiveCoefficient => "objective coefficients must be finite and strictly positive",
            Self::InvalidPortfolioSize => "maximum portfolio size must be at least one",
            Self::InvalidEpsilon => "portfolio degradation threshold must be finite and nonnegative",
            Self::EmptyIntervalGrid => "the action × objective interval grid must be nonempty",
            Self::DuplicateActionObjectiveInterval => "duplicate action × objective interval",
            Self::UnknownIntervalAction => "interval references an action outside the frozen feasible action set",
            Self::UnknownIntervalObjective => "interval references an objective outside the frozen objective set",
            Self::MissingActionObjectiveInterval => "the frozen action × objective interval grid is incomplete",
            Self::InvalidUtilityInterval => "utility interval must be finite and ordered",
            Self::EmptyPortfolio => "a portfolio must contain at least one action",
            Self::DuplicatePortfolioAction => "portfolio contains a duplicate action ID",
            Self::UnknownPortfolioAction => "portfolio references an action outside the frozen feasible action set",
            Self::PortfolioTooLarge => "portfolio exceeds the frozen maximum portfolio size",
            Self::EmptyEnumerationUniverse => "portfolio enumeration requires a nonempty action universe",
            Self::DuplicateEnumerationAction => "portfolio enumeration action IDs must be unique",
            Self::InvalidEnumerationLimit => "maximum portfolio size and enumeration budget must be positive",
            Self::EnumerationCountOverflow => "bounded portfolio count overflowed usize",
            Self::EnumerationBudgetExceeded => "exact portfolio enumeration exceeds the declared candidate budget",
        };
        f.write_str(message)
    }
}
impl std::error::Error for FewForManyPortfolioError {}

/// Bound the weighted worst-objective degradation of one nonempty portfolio.
///
/// On the event that all supplied action × objective intervals are
/// simultaneously valid, for every nonempty S in the frozen feasible action
/// universe and every objective m:
///
/// max(0, max_a L[a,m] - max_{a in S} U[a,m])
///   <= ideal[m] - max_{a in S} mean[a,m]
///   <= max(0, max_a U[a,m] - max_{a in S} L[a,m]).
///
/// Positive-coefficient max aggregation preserves these inequalities. Every
/// portfolio is deterministic post-processing of the same base intervals,
/// so no additional union bound over portfolios is needed. The upstream
/// action × objective interval family still needs valid multiplicity control.
pub fn bound_few_for_many_portfolio_degradation(
    intervals: &[ActionObjectiveInterval],
    contract: &FewForManyPortfolioContract,
    portfolio_action_ids: &[u32],
) -> Result<FewForManyPortfolioBound, FewForManyPortfolioError> {
    let (action_ids, objective_specs, by_pair) = validate_inputs(intervals, contract)?;
    if portfolio_action_ids.is_empty() {
        return Err(FewForManyPortfolioError::EmptyPortfolio);
    }
    if portfolio_action_ids.len() > contract.max_portfolio_size {
        return Err(FewForManyPortfolioError::PortfolioTooLarge);
    }

    let mut portfolio = BTreeSet::new();
    for action_id in portfolio_action_ids {
        if !action_ids.contains(action_id) {
            return Err(FewForManyPortfolioError::UnknownPortfolioAction);
        }
        if !portfolio.insert(*action_id) {
            return Err(FewForManyPortfolioError::DuplicatePortfolioAction);
        }
    }

    let mut objective_bounds = Vec::with_capacity(objective_specs.len());
    let mut degradation_lower_bound = 0.0_f64;
    let mut degradation_upper_bound = 0.0_f64;

    for objective in &objective_specs {
        let mut ideal_lower = f64::NEG_INFINITY;
        let mut ideal_upper = f64::NEG_INFINITY;
        let mut portfolio_lower = f64::NEG_INFINITY;
        let mut portfolio_upper = f64::NEG_INFINITY;
        for action_id in &action_ids {
            let interval = by_pair
                .get(&(*action_id, objective.objective_id))
                .copied()
                .ok_or(FewForManyPortfolioError::MissingActionObjectiveInterval)?;
            ideal_lower = ideal_lower.max(interval.lower_bound);
            ideal_upper = ideal_upper.max(interval.upper_bound);
            if portfolio.contains(action_id) {
                portfolio_lower = portfolio_lower.max(interval.lower_bound);
                portfolio_upper = portfolio_upper.max(interval.upper_bound);
            }
        }

        let loss_lower = (ideal_lower - portfolio_upper).max(0.0);
        let loss_upper = (ideal_upper - portfolio_lower).max(0.0);
        let weighted_lower = objective.coefficient * loss_lower;
        let weighted_upper = objective.coefficient * loss_upper;
        if !loss_lower.is_finite()
            || !loss_upper.is_finite()
            || !weighted_lower.is_finite()
            || !weighted_upper.is_finite()
            || weighted_lower > weighted_upper
        {
            return Err(FewForManyPortfolioError::InvalidUtilityInterval);
        }

        degradation_lower_bound = degradation_lower_bound.max(weighted_lower);
        degradation_upper_bound = degradation_upper_bound.max(weighted_upper);
        objective_bounds.push(ObjectiveDegradationBound {
            objective_id: objective.objective_id,
            ideal_lower_bound: ideal_lower,
            ideal_upper_bound: ideal_upper,
            portfolio_best_lower_bound: portfolio_lower,
            portfolio_best_upper_bound: portfolio_upper,
            degradation_lower_bound: loss_lower,
            degradation_upper_bound: loss_upper,
            coefficient: objective.coefficient,
            weighted_lower_bound: weighted_lower,
            weighted_upper_bound: weighted_upper,
        });
    }

    let status = if degradation_upper_bound <= contract.epsilon {
        PortfolioDegradationStatus::CertifiedWithinThreshold
    } else if degradation_lower_bound > contract.epsilon {
        PortfolioDegradationStatus::CertifiedAboveThreshold
    } else {
        PortfolioDegradationStatus::Unresolved
    };

    Ok(FewForManyPortfolioBound {
        method_id: FEW_FOR_MANY_PORTFOLIO_BOUND_METHOD_ID,
        portfolio_action_ids: portfolio.into_iter().collect(),
        max_portfolio_size: contract.max_portfolio_size,
        epsilon: contract.epsilon,
        objective_bounds,
        degradation_lower_bound,
        degradation_upper_bound,
        status,
    })
}

/// Exact small-K oracle. Enumerates all nonempty portfolios of size <= cap,
/// in lexicographic action-ID order, only if the candidate count is <= budget.
pub fn enumerate_bounded_portfolios(
    action_ids: &[u32],
    max_portfolio_size: usize,
    max_candidates: usize,
) -> Result<Vec<Vec<u32>>, FewForManyPortfolioError> {
    if action_ids.is_empty() {
        return Err(FewForManyPortfolioError::EmptyEnumerationUniverse);
    }
    if max_portfolio_size == 0 || max_candidates == 0 {
        return Err(FewForManyPortfolioError::InvalidEnumerationLimit);
    }
    let mut ids = action_ids.to_vec();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(FewForManyPortfolioError::DuplicateEnumerationAction);
    }

    let limit = max_portfolio_size.min(ids.len());
    let mut total = 0_usize;
    for size in 1..=limit {
        total = total
            .checked_add(binomial(ids.len(), size)?)
            .ok_or(FewForManyPortfolioError::EnumerationCountOverflow)?;
        if total > max_candidates {
            return Err(FewForManyPortfolioError::EnumerationBudgetExceeded);
        }
    }

    let mut output = Vec::with_capacity(total);
    for size in 1..=limit {
        enumerate_combinations(&ids, size, 0, &mut Vec::with_capacity(size), &mut output);
    }
    Ok(output)
}

fn validate_inputs<'a>(
    intervals: &'a [ActionObjectiveInterval],
    contract: &FewForManyPortfolioContract,
) -> Result<
    (
        BTreeSet<u32>,
        Vec<PortfolioObjectiveSpec>,
        BTreeMap<(u32, u16), &'a ActionObjectiveInterval>,
    ),
    FewForManyPortfolioError,
> {
    if contract.feasible_action_ids.is_empty() {
        return Err(FewForManyPortfolioError::EmptyFeasibleActionSet);
    }
    if contract.max_portfolio_size == 0 {
        return Err(FewForManyPortfolioError::InvalidPortfolioSize);
    }
    if !contract.epsilon.is_finite() || contract.epsilon < 0.0 {
        return Err(FewForManyPortfolioError::InvalidEpsilon);
    }
    if contract.objectives.is_empty() {
        return Err(FewForManyPortfolioError::EmptyObjectiveSet);
    }
    if intervals.is_empty() {
        return Err(FewForManyPortfolioError::EmptyIntervalGrid);
    }

    let action_ids: BTreeSet<u32> = contract.feasible_action_ids.iter().copied().collect();
    if action_ids.len() != contract.feasible_action_ids.len() {
        return Err(FewForManyPortfolioError::DuplicateFeasibleAction);
    }
    let mut objective_specs = contract.objectives.clone();
    objective_specs.sort_by_key(|objective| objective.objective_id);
    let objective_ids: BTreeSet<u16> =
        objective_specs.iter().map(|objective| objective.objective_id).collect();
    if objective_ids.len() != objective_specs.len() {
        return Err(FewForManyPortfolioError::DuplicateObjective);
    }
    if objective_specs.iter().any(|objective| {
        !objective.coefficient.is_finite() || objective.coefficient <= 0.0
    }) {
        return Err(FewForManyPortfolioError::InvalidObjectiveCoefficient);
    }

    let expected = action_ids
        .len()
        .checked_mul(objective_ids.len())
        .ok_or(FewForManyPortfolioError::MissingActionObjectiveInterval)?;
    if intervals.len() != expected {
        return Err(FewForManyPortfolioError::MissingActionObjectiveInterval);
    }

    let mut by_pair = BTreeMap::new();
    for interval in intervals {
        if !action_ids.contains(&interval.action_id) {
            return Err(FewForManyPortfolioError::UnknownIntervalAction);
        }
        if !objective_ids.contains(&interval.objective_id) {
            return Err(FewForManyPortfolioError::UnknownIntervalObjective);
        }
        if !interval.lower_bound.is_finite()
            || !interval.upper_bound.is_finite()
            || interval.lower_bound > interval.upper_bound
        {
            return Err(FewForManyPortfolioError::InvalidUtilityInterval);
        }
        if by_pair
            .insert((interval.action_id, interval.objective_id), interval)
            .is_some()
        {
            return Err(FewForManyPortfolioError::DuplicateActionObjectiveInterval);
        }
    }
    if by_pair.len() != expected {
        return Err(FewForManyPortfolioError::MissingActionObjectiveInterval);
    }
    Ok((action_ids, objective_specs, by_pair))
}

fn binomial(n: usize, k: usize) -> Result<usize, FewForManyPortfolioError> {
    let k = k.min(n - k);
    let mut result = 1_usize;
    for index in 0..k {
        let numerator = n - index;
        let denominator = index + 1;
        let divisor = gcd(numerator, denominator);
        let reduced_numerator = numerator / divisor;
        let reduced_denominator = denominator / divisor;
        result /= reduced_denominator;
        result = result
            .checked_mul(reduced_numerator)
            .ok_or(FewForManyPortfolioError::EnumerationCountOverflow)?;
    }
    Ok(result)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn enumerate_combinations(
    ids: &[u32],
    remaining: usize,
    start: usize,
    current: &mut Vec<u32>,
    output: &mut Vec<Vec<u32>>,
) {
    if remaining == 0 {
        output.push(current.clone());
        return;
    }
    for index in start..=(ids.len() - remaining) {
        current.push(ids[index]);
        enumerate_combinations(ids, remaining - 1, index + 1, current, output);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(cap: usize, epsilon: f64) -> FewForManyPortfolioContract {
        FewForManyPortfolioContract {
            feasible_action_ids: vec![1, 2, 3],
            objectives: vec![
                PortfolioObjectiveSpec { objective_id: 10, coefficient: 1.0 },
                PortfolioObjectiveSpec { objective_id: 20, coefficient: 2.0 },
            ],
            max_portfolio_size: cap,
            epsilon,
        }
    }

    fn points() -> Vec<ActionObjectiveInterval> {
        vec![
            ActionObjectiveInterval { action_id: 1, objective_id: 10, lower_bound: 1.0, upper_bound: 1.0 },
            ActionObjectiveInterval { action_id: 1, objective_id: 20, lower_bound: 0.0, upper_bound: 0.0 },
            ActionObjectiveInterval { action_id: 2, objective_id: 10, lower_bound: 0.0, upper_bound: 0.0 },
            ActionObjectiveInterval { action_id: 2, objective_id: 20, lower_bound: 1.0, upper_bound: 1.0 },
            ActionObjectiveInterval { action_id: 3, objective_id: 10, lower_bound: 0.4, upper_bound: 0.4 },
            ActionObjectiveInterval { action_id: 3, objective_id: 20, lower_bound: 0.4, upper_bound: 0.4 },
        ]
    }

    #[test]
    fn a_portfolio_covering_each_objective_ideal_has_zero_degradation() {
        let bound = bound_few_for_many_portfolio_degradation(&points(), &contract(2, 0.0), &[2, 1]).unwrap();
        assert_eq!(bound.portfolio_action_ids, vec![1, 2]);
        assert_eq!(bound.degradation_lower_bound, 0.0);
        assert_eq!(bound.degradation_upper_bound, 0.0);
        assert_eq!(bound.status, PortfolioDegradationStatus::CertifiedWithinThreshold);
    }

    #[test]
    fn weighted_max_degradation_and_above_threshold_are_distinguished() {
        let bound = bound_few_for_many_portfolio_degradation(&points(), &contract(1, 1.0), &[3]).unwrap();
        assert!((bound.objective_bounds[0].degradation_lower_bound - 0.6).abs() < 1.0e-12);
        assert!((bound.objective_bounds[1].degradation_lower_bound - 0.6).abs() < 1.0e-12);
        assert!((bound.degradation_lower_bound - 1.2).abs() < 1.0e-12);
        assert_eq!(bound.status, PortfolioDegradationStatus::CertifiedAboveThreshold);
    }

    #[test]
    fn overlapping_intervals_remain_unresolved() {
        let intervals = vec![
            ActionObjectiveInterval { action_id: 1, objective_id: 10, lower_bound: 0.8, upper_bound: 1.0 },
            ActionObjectiveInterval { action_id: 1, objective_id: 20, lower_bound: 0.0, upper_bound: 0.4 },
            ActionObjectiveInterval { action_id: 2, objective_id: 10, lower_bound: 0.0, upper_bound: 0.5 },
            ActionObjectiveInterval { action_id: 2, objective_id: 20, lower_bound: 0.7, upper_bound: 1.0 },
            ActionObjectiveInterval { action_id: 3, objective_id: 10, lower_bound: 0.2, upper_bound: 0.7 },
            ActionObjectiveInterval { action_id: 3, objective_id: 20, lower_bound: 0.2, upper_bound: 0.7 },
        ];
        let bound = bound_few_for_many_portfolio_degradation(&intervals, &contract(1, 0.6), &[3]).unwrap();
        assert!(bound.degradation_lower_bound <= 0.6);
        assert!(bound.degradation_upper_bound > 0.6);
        assert_eq!(bound.status, PortfolioDegradationStatus::Unresolved);
    }

    #[test]
    fn interval_and_contract_order_do_not_change_results() {
        let first = bound_few_for_many_portfolio_degradation(&points(), &contract(2, 0.5), &[1, 3]).unwrap();
        let mut intervals = points();
        intervals.reverse();
        let mut reordered = contract(2, 0.5);
        reordered.feasible_action_ids.reverse();
        reordered.objectives.reverse();
        let second = bound_few_for_many_portfolio_degradation(&intervals, &reordered, &[3, 1]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn exact_small_k_enumerator_is_lexicographic_and_budget_limited() {
        assert_eq!(
            enumerate_bounded_portfolios(&[3, 1, 2], 2, 6).unwrap(),
            vec![vec![1], vec![2], vec![3], vec![1, 2], vec![1, 3], vec![2, 3]]
        );
        assert_eq!(enumerate_bounded_portfolios(&[1, 2, 3], 3, 7).unwrap().len(), 7);
        assert_eq!(
            enumerate_bounded_portfolios(&[1, 2, 3], 3, 6),
            Err(FewForManyPortfolioError::EnumerationBudgetExceeded)
        );
    }

    #[test]
    fn invalid_grids_and_portfolios_fail_closed() {
        let mut incomplete = points();
        incomplete.pop();
        assert_eq!(
            bound_few_for_many_portfolio_degradation(&incomplete, &contract(2, 0.1), &[1]),
            Err(FewForManyPortfolioError::MissingActionObjectiveInterval)
        );

        let mut duplicate = points();
        duplicate[1] = duplicate[0];
        assert_eq!(
            bound_few_for_many_portfolio_degradation(&duplicate, &contract(2, 0.1), &[1]),
            Err(FewForManyPortfolioError::DuplicateActionObjectiveInterval)
        );
        assert_eq!(
            bound_few_for_many_portfolio_degradation(&points(), &contract(2, 0.1), &[99]),
            Err(FewForManyPortfolioError::UnknownPortfolioAction)
        );
        assert_eq!(
            bound_few_for_many_portfolio_degradation(&points(), &contract(1, 0.1), &[1, 2]),
            Err(FewForManyPortfolioError::PortfolioTooLarge)
        );
        assert_eq!(
            bound_few_for_many_portfolio_degradation(&points(), &contract(2, f64::NAN), &[1]),
            Err(FewForManyPortfolioError::InvalidEpsilon)
        );
        assert_eq!(
            enumerate_bounded_portfolios(&[1, 1], 1, 10),
            Err(FewForManyPortfolioError::DuplicateEnumerationAction)
        );
    }

    #[test]
    fn deterministic_property_check_contains_true_degradation_for_all_small_portfolios() {
        let mut state = 0x5eed_f0f0_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / ((1_u64 << 53) as f64)
        };

        for campaign in 0..128 {
            let action_count = 2 + (campaign % 5) as usize;
            let objective_count = 1 + (campaign % 4) as usize;
            let action_ids = (0..action_count).map(|id| id as u32 + 1).collect::<Vec<_>>();
            let objectives = (0..objective_count)
                .map(|id| PortfolioObjectiveSpec {
                    objective_id: id as u16 + 10,
                    coefficient: 0.25 + 2.0 * next_unit(),
                })
                .collect::<Vec<_>>();
            let mut intervals = Vec::new();
            let mut truth = BTreeMap::new();
            for action_id in &action_ids {
                for objective in &objectives {
                    let mean = next_unit();
                    let lower = (mean - 0.3 * next_unit()).max(0.0);
                    let upper = (mean + 0.3 * next_unit()).min(1.0);
                    intervals.push(ActionObjectiveInterval {
                        action_id: *action_id,
                        objective_id: objective.objective_id,
                        lower_bound: lower,
                        upper_bound: upper,
                    });
                    truth.insert((*action_id, objective.objective_id), mean);
                }
            }

            let contract = FewForManyPortfolioContract {
                feasible_action_ids: action_ids.clone(),
                objectives: objectives.clone(),
                max_portfolio_size: action_count.min(3),
                epsilon: 0.5,
            };
            let portfolios = enumerate_bounded_portfolios(&action_ids, contract.max_portfolio_size, 100).unwrap();
            for portfolio in portfolios {
                let bound = bound_few_for_many_portfolio_degradation(&intervals, &contract, &portfolio).unwrap();
                let true_degradation = objectives
                    .iter()
                    .map(|objective| {
                        let ideal = action_ids.iter()
                            .map(|action_id| truth[&(*action_id, objective.objective_id)])
                            .fold(f64::NEG_INFINITY, f64::max);
                        let covered = portfolio.iter()
                            .map(|action_id| truth[&(*action_id, objective.objective_id)])
                            .fold(f64::NEG_INFINITY, f64::max);
                        objective.coefficient * (ideal - covered).max(0.0)
                    })
                    .fold(0.0_f64, f64::max);
                assert!(bound.degradation_lower_bound <= true_degradation + 1.0e-12);
                assert!(true_degradation <= bound.degradation_upper_bound + 1.0e-12);
            }
        }
    }
}
