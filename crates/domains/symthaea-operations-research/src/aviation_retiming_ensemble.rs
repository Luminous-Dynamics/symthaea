// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario-ensemble evaluation of greedy versus bounded retiming recovery.
//!
//! This adapter evaluates both methods on the same frozen synthetic scenarios
//! and feeds paired cost deltas to the crate's existing finite-support
//! distributionally robust evaluator. It measures algorithmic resilience over
//! the declared ensemble; it is not a guarantee about scenarios outside it.

use crate::aviation_distributional_robustness::{
    evaluate_tv_robust_cost_delta, DistributionalRobustError, DistributionalRobustReport,
    ScenarioCostDelta, PROBABILITY_SCALE_PPM,
};
use crate::aviation_retiming_recovery::{
    solve_retiming_branch_and_bound, solve_retiming_greedy, verify_retiming_plan,
    RetimingScenario, RetimingSearchOptions, RetimingSearchStatus,
    RetimingVerificationStatus,
};

/// One immutable synthetic scenario and its preregistered nominal probability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingEnsembleCase {
    pub scenario: RetimingScenario,
    pub nominal_probability_ppm: u32,
}

/// Compute budgets and finite-support distributional robustness radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetimingEnsembleOptions {
    pub max_expanded_nodes_per_scenario: u64,
    pub max_generated_nodes_per_scenario: u64,
    pub max_absolute_optimality_gap: u64,
    pub tv_radius_ppm: u32,
}

/// Candidate-minus-greedy score for one frozen scenario.
/// Negative values favor branch-and-bound; positive values mean it returned a
/// higher-cost verified plan. The reported search status preserves whether its
/// objective-gap certificate was completed before the budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetimingEnsembleScenarioReport {
    pub scenario_id: String,
    pub input_digest: String,
    pub nominal_probability_ppm: u32,
    pub greedy_cost_units: u64,
    pub search_cost_units: u64,
    pub candidate_minus_greedy_cost_units: i64,
    pub search_status: RetimingSearchStatus,
    pub search_lower_bound_cost_units: Option<u64>,
    pub search_absolute_gap_units: Option<u64>,
    pub expanded_nodes: u64,
    pub generated_nodes: u64,
    pub candidate_actions_considered: u64,
    pub lower_bound_evaluations: u64,
    pub availability_relaxation_checks: u64,
    pub greedy_delay_minutes: u64,
    pub search_delay_minutes: u64,
    pub greedy_cancelled_flights: u64,
    pub search_cancelled_flights: u64,
}

/// Paired method comparison across a finite, probability-weighted scenario set.
#[derive(Debug, Clone, PartialEq)]
pub struct RetimingEnsembleReport {
    pub scenario_count: usize,
    pub tv_radius_ppm: u32,
    /// Sorted by scenario ID so the report does not depend on input row order.
    pub scenarios: Vec<RetimingEnsembleScenarioReport>,
    pub distributionally_robust_cost_delta: DistributionalRobustReport,
}

/// Fail-closed failure while building an ensemble comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetimingEnsembleError {
    EmptyScenarioSet,
    EmptyScenarioId,
    DuplicateScenarioId { scenario_id: String },
    EmptyInputDigest { scenario_id: String },
    ProbabilityMassMismatch { actual_ppm: u64 },
    NoGreedyPlan { scenario_id: String, reason: String },
    NoSearchPlanWithinBudget { scenario_id: String },
    SearchSolverError { scenario_id: String, reason: String },
    GreedyVerificationFailed { scenario_id: String },
    SearchVerificationFailed { scenario_id: String },
    CostDeltaOutOfRange { scenario_id: String },
    ArithmeticOverflow { scenario_id: String },
    DistributionalRobustness(DistributionalRobustError),
}

fn summarize_plan(
    scenario: &RetimingScenario,
    assignments: &[crate::aviation_retiming_recovery::RetimingAssignment],
) -> Result<(u64, u64), RetimingEnsembleError> {
    let mut delay_minutes = 0_u64;
    let mut cancelled_flights = 0_u64;
    for assignment in assignments {
        if assignment.aircraft_id.is_none() {
            cancelled_flights = cancelled_flights
                .checked_add(1)
                .ok_or_else(|| RetimingEnsembleError::ArithmeticOverflow {
                    scenario_id: scenario.id.clone(),
                })?;
        } else {
            delay_minutes = delay_minutes
                .checked_add(u64::from(assignment.delay_minutes))
                .ok_or_else(|| RetimingEnsembleError::ArithmeticOverflow {
                    scenario_id: scenario.id.clone(),
                })?;
        }
    }
    Ok((delay_minutes, cancelled_flights))
}

/// Evaluate the greedy baseline and bounded search on every scenario using the
/// same model and objective, then report nominal and total-variation worst-case
/// candidate-minus-baseline cost. The TV guarantee is conditional on the
/// supplied scenario support and nominal masses; it is not out-of-support safety.
pub fn evaluate_retiming_ensemble(
    cases: &[RetimingEnsembleCase],
    options: RetimingEnsembleOptions,
) -> Result<RetimingEnsembleReport, RetimingEnsembleError> {
    if cases.is_empty() {
        return Err(RetimingEnsembleError::EmptyScenarioSet);
    }

    let mut seen = std::collections::BTreeSet::<String>::new();
    let mut probability_sum = 0_u64;
    for case in cases {
        if case.scenario.id.trim().is_empty() {
            return Err(RetimingEnsembleError::EmptyScenarioId);
        }
        if !seen.insert(case.scenario.id.clone()) {
            return Err(RetimingEnsembleError::DuplicateScenarioId {
                scenario_id: case.scenario.id.clone(),
            });
        }
        if case.scenario.input_digest.trim().is_empty() {
            return Err(RetimingEnsembleError::EmptyInputDigest {
                scenario_id: case.scenario.id.clone(),
            });
        }
        case.scenario.validate().map_err(|reason| RetimingEnsembleError::NoGreedyPlan {
            scenario_id: case.scenario.id.clone(),
            reason,
        })?;
        probability_sum = probability_sum
            .checked_add(u64::from(case.nominal_probability_ppm))
            .ok_or_else(|| RetimingEnsembleError::ArithmeticOverflow {
                scenario_id: case.scenario.id.clone(),
            })?;
    }
    if probability_sum != u64::from(PROBABILITY_SCALE_PPM) {
        return Err(RetimingEnsembleError::ProbabilityMassMismatch {
            actual_ppm: probability_sum,
        });
    }

    let mut scenario_reports = Vec::with_capacity(cases.len());
    let mut cost_deltas = Vec::with_capacity(cases.len());
    for case in cases {
        let scenario = &case.scenario;
        let greedy = solve_retiming_greedy(scenario).map_err(|reason| {
            RetimingEnsembleError::NoGreedyPlan {
                scenario_id: scenario.id.clone(),
                reason,
            }
        })?;
        if verify_retiming_plan(scenario, &greedy.plan).status
            != RetimingVerificationStatus::Verified
        {
            return Err(RetimingEnsembleError::GreedyVerificationFailed {
                scenario_id: scenario.id.clone(),
            });
        }

        let search = solve_retiming_branch_and_bound(
            scenario,
            RetimingSearchOptions {
                max_expanded_nodes: options.max_expanded_nodes_per_scenario,
                max_generated_nodes: options.max_generated_nodes_per_scenario,
                max_absolute_optimality_gap: options.max_absolute_optimality_gap,
            },
        )
        .map_err(|reason| RetimingEnsembleError::SearchSolverError {
            scenario_id: scenario.id.clone(),
            reason,
        })?;
        let search_plan = search.best_plan.as_ref().ok_or_else(|| {
            RetimingEnsembleError::NoSearchPlanWithinBudget {
                scenario_id: scenario.id.clone(),
            }
        })?;
        if verify_retiming_plan(scenario, search_plan).status
            != RetimingVerificationStatus::Verified
        {
            return Err(RetimingEnsembleError::SearchVerificationFailed {
                scenario_id: scenario.id.clone(),
            });
        }
        let delta = i128::from(search_plan.total_cost) - i128::from(greedy.plan.total_cost);
        let delta_i64 = i64::try_from(delta).map_err(|_| {
            RetimingEnsembleError::CostDeltaOutOfRange {
                scenario_id: scenario.id.clone(),
            }
        })?;
        let (greedy_delay, greedy_cancelled) =
            summarize_plan(scenario, &greedy.plan.assignments)?;
        let (search_delay, search_cancelled) =
            summarize_plan(scenario, &search_plan.assignments)?;

        cost_deltas.push(ScenarioCostDelta {
            scenario_id: scenario.id.clone(),
            nominal_probability_ppm: case.nominal_probability_ppm,
            candidate_minus_baseline_cost_units: delta_i64,
        });
        scenario_reports.push(RetimingEnsembleScenarioReport {
            scenario_id: scenario.id.clone(),
            input_digest: scenario.input_digest.clone(),
            nominal_probability_ppm: case.nominal_probability_ppm,
            greedy_cost_units: greedy.plan.total_cost,
            search_cost_units: search_plan.total_cost,
            candidate_minus_greedy_cost_units: delta_i64,
            search_status: search.status,
            search_lower_bound_cost_units: search.global_lower_bound_cost,
            search_absolute_gap_units: search.absolute_optimality_gap,
            expanded_nodes: search.stats.expanded_nodes,
            generated_nodes: search.stats.generated_nodes,
            candidate_actions_considered: search.stats.candidate_actions_considered,
            lower_bound_evaluations: search.stats.lower_bound_evaluations,
            availability_relaxation_checks: search.stats.availability_relaxation_checks,
            greedy_delay_minutes: greedy_delay,
            search_delay_minutes: search_delay,
            greedy_cancelled_flights: greedy_cancelled,
            search_cancelled_flights: search_cancelled,
        });
    }
    scenario_reports.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let robust = evaluate_tv_robust_cost_delta(&cost_deltas, options.tv_radius_ppm)
        .map_err(RetimingEnsembleError::DistributionalRobustness)?;

    Ok(RetimingEnsembleReport {
        scenario_count: scenario_reports.len(),
        tv_radius_ppm: options.tv_radius_ppm,
        scenarios: scenario_reports,
        distributionally_robust_cost_delta: robust,
    })
}
