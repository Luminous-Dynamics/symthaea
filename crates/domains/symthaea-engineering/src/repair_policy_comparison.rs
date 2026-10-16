// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired evaluation of frozen repair plans versus rolling-horizon dispatch.
//!
//! Both approaches receive the same nominal jobs, crews, parts, duration scenarios,
//! weights, and horizon. The fixed-plan side delegates to the robust replay evaluator;
//! the adaptive side dispatches from the observable state at each event. Per-scenario
//! traces are retained so a better aggregate cannot conceal a harmful case.
//!
//! Scenario weights are caller supplied. They are not empirical probabilities unless
//! supported by evidence. This is a deterministic evaluation harness, not calibration.

use crate::repair_crew_scheduling::{
    RepairCrew, RepairJob, RepairPartStock, RepairSchedulingPolicy,
};
use crate::robust_repair_evaluation::{
    evaluate_repair_policies_under_duration_scenarios, DurationScenarioOutcome,
    RepairDurationScenario, ReplayedRepair, RobustRepairConfig,
    RobustRepairError,
};
use crate::rolling_horizon_repair::{
    dispatch_repairs_rolling_horizon, RealizedRepairDuration, RollingHorizonRepairError,
    RollingHorizonRepairReport,
};

const EPSILON: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct PairedRepairScenarioComparison {
    pub scenario_id: String,
    pub scenario_weight: f64,
    pub fixed_plan_loss: f64,
    pub rolling_horizon_loss: f64,
    /// Negative means rolling horizon lost less weighted service; positive means worse.
    pub loss_delta_adaptive_minus_fixed: f64,
    pub fixed_plan_schedule: Vec<ReplayedRepair>,
    pub rolling_horizon_report: RollingHorizonRepairReport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairPolicyComparisonSummary {
    pub policy: RepairSchedulingPolicy,
    pub fixed_expected_loss: f64,
    pub rolling_horizon_expected_loss: f64,
    pub expected_loss_delta_adaptive_minus_fixed: f64,
    /// (fixed - adaptive) / fixed; None if the fixed baseline loss is zero.
    pub expected_loss_reduction_fraction: Option<f64>,
    pub fixed_worst_case_loss: f64,
    pub rolling_horizon_worst_case_loss: f64,
    pub fixed_cvar_loss: f64,
    pub rolling_horizon_cvar_loss: f64,
    pub improved_scenario_count: usize,
    pub worsened_scenario_count: usize,
    pub unchanged_scenario_count: usize,
    /// Sum of caller-supplied weights, not necessarily an empirical probability.
    pub improved_scenario_weight: f64,
    pub worsened_scenario_weight: f64,
    pub scenarios: Vec<PairedRepairScenarioComparison>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepairPolicyComparisonReport {
    pub horizon_steps: usize,
    pub cvar_alpha: f64,
    pub policy_summaries: Vec<RepairPolicyComparisonSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepairPolicyComparisonError {
    FixedPlanEvaluation(RobustRepairError),
    RollingHorizonEvaluation {
        scenario_id: String,
        policy: RepairSchedulingPolicy,
        source: RollingHorizonRepairError,
    },
    MissingFixedScenario(String),
}

impl std::fmt::Display for RepairPolicyComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedPlanEvaluation(source) => write!(f, "fixed-plan evaluation failed: {source}"),
            Self::RollingHorizonEvaluation { scenario_id, policy, source } => {
                write!(f, "rolling-horizon evaluation failed for scenario {scenario_id} and policy {policy:?}: {source}")
            }
            Self::MissingFixedScenario(id) => write!(f, "fixed-plan evaluator omitted scenario {id}"),
        }
    }
}
impl std::error::Error for RepairPolicyComparisonError {}

fn upper_tail_cvar(losses: &[(f64, f64)], alpha: f64) -> f64 {
    let mut ranked = losses.to_vec();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut remaining = 1.0 - alpha;
    let mut weighted_loss = 0.0_f64;
    for (loss, weight) in ranked {
        if remaining <= EPSILON { break; }
        let included = weight.min(remaining);
        weighted_loss += loss * included;
        remaining -= included;
    }
    weighted_loss / (1.0 - alpha)
}


/// Compare fixed nominal-plan replay with event-triggered dispatch on paired cases.
///
/// All three existing policies are evaluated. Every adaptive scenario receives the
/// same duration realization as its fixed-plan counterpart. Report both aggregate
/// risk measures and per-scenario deltas; do not declare adaptation universally better
/// on the basis of the aggregate alone.
pub fn compare_fixed_and_rolling_horizon_policies(
    nominal_jobs: &[RepairJob],
    crews: &[RepairCrew],
    parts: &[RepairPartStock],
    scenarios: &[RepairDurationScenario],
    config: &RobustRepairConfig,
) -> Result<RepairPolicyComparisonReport, RepairPolicyComparisonError> {
    let fixed = evaluate_repair_policies_under_duration_scenarios(
        nominal_jobs, crews, parts, scenarios, config,
    ).map_err(RepairPolicyComparisonError::FixedPlanEvaluation)?;

    let mut comparisons = Vec::with_capacity(fixed.policy_summaries.len());
    for fixed_policy in &fixed.policy_summaries {
        let mut paired = Vec::with_capacity(scenarios.len());
        let mut adaptive_losses = Vec::with_capacity(scenarios.len());

        for scenario in scenarios {
            let realized: Vec<RealizedRepairDuration> = scenario.durations.iter()
                .map(|duration| RealizedRepairDuration::new(duration.job_id.clone(), duration.duration_steps))
                .collect();
            let adaptive = dispatch_repairs_rolling_horizon(
                nominal_jobs,
                crews,
                parts,
                &realized,
                &crate::rolling_horizon_repair::RollingHorizonConfig::new(
                    config.horizon_steps,
                    fixed_policy.policy,
                ),
            ).map_err(|source| RepairPolicyComparisonError::RollingHorizonEvaluation {
                scenario_id: scenario.id.clone(),
                policy: fixed_policy.policy,
                source,
            })?;

            let fixed_scenario: &DurationScenarioOutcome = fixed_policy.scenarios
                .iter()
                .find(|outcome| outcome.scenario_id == scenario.id)
                .ok_or_else(|| RepairPolicyComparisonError::MissingFixedScenario(scenario.id.clone()))?;
            adaptive_losses.push((adaptive.total_weighted_service_unit_steps_lost, scenario.weight));
            paired.push(PairedRepairScenarioComparison {
                scenario_id: scenario.id.clone(),
                scenario_weight: scenario.weight,
                fixed_plan_loss: fixed_scenario.total_weighted_service_unit_steps_lost,
                rolling_horizon_loss: adaptive.total_weighted_service_unit_steps_lost,
                loss_delta_adaptive_minus_fixed:
                    adaptive.total_weighted_service_unit_steps_lost
                        - fixed_scenario.total_weighted_service_unit_steps_lost,
                fixed_plan_schedule: fixed_scenario.schedule.clone(),
                rolling_horizon_report: adaptive,
            });
        }

        let adaptive_expected = adaptive_losses.iter().map(|(loss, weight)| loss * weight).sum::<f64>();
        let adaptive_worst = adaptive_losses.iter().map(|(loss, _)| *loss).fold(0.0_f64, f64::max);
        let adaptive_cvar = upper_tail_cvar(&adaptive_losses, config.cvar_alpha);
        let mut improved = 0usize;
        let mut worsened = 0usize;
        let mut unchanged = 0usize;
        let mut improved_weight = 0.0;
        let mut worsened_weight = 0.0;
        for case in &paired {
            if case.loss_delta_adaptive_minus_fixed < -EPSILON {
                improved += 1;
                improved_weight += case.scenario_weight;
            } else if case.loss_delta_adaptive_minus_fixed > EPSILON {
                worsened += 1;
                worsened_weight += case.scenario_weight;
            } else {
                unchanged += 1;
            }
        }

        let reduction = if fixed_policy.expected_weighted_service_unit_steps_lost.abs() <= EPSILON {
            None
        } else {
            Some((fixed_policy.expected_weighted_service_unit_steps_lost - adaptive_expected)
                / fixed_policy.expected_weighted_service_unit_steps_lost)
        };
        comparisons.push(RepairPolicyComparisonSummary {
            policy: fixed_policy.policy,
            fixed_expected_loss: fixed_policy.expected_weighted_service_unit_steps_lost,
            rolling_horizon_expected_loss: adaptive_expected,
            expected_loss_delta_adaptive_minus_fixed:
                adaptive_expected - fixed_policy.expected_weighted_service_unit_steps_lost,
            expected_loss_reduction_fraction: reduction,
            fixed_worst_case_loss: fixed_policy.worst_case_weighted_service_unit_steps_lost,
            rolling_horizon_worst_case_loss: adaptive_worst,
            fixed_cvar_loss: fixed_policy.cvar_weighted_service_unit_steps_lost,
            rolling_horizon_cvar_loss: adaptive_cvar,
            improved_scenario_count: improved,
            worsened_scenario_count: worsened,
            unchanged_scenario_count: unchanged,
            improved_scenario_weight: improved_weight,
            worsened_scenario_weight: worsened_weight,
            scenarios: paired,
        });
    }

    Ok(RepairPolicyComparisonReport {
        horizon_steps: config.horizon_steps,
        cvar_alpha: config.cvar_alpha,
        policy_summaries: comparisons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair_crew_scheduling::RepairPartRequirement;

    fn job(id: &str, nominal: usize, priority: u32, value: f64, outage: usize) -> RepairJob {
        RepairJob::new(
            id, "general", nominal, priority, value, outage,
            Vec::<String>::new(), Vec::<RepairPartRequirement>::new(),
        )
    }
    fn crew() -> RepairCrew { RepairCrew::new("crew", ["general"], 0) }
    fn durations(items: &[(&str, usize)]) -> Vec<crate::robust_repair_evaluation::JobDuration> {
        items.iter().map(|(id, d)| crate::robust_repair_evaluation::JobDuration::new(*id, *d)).collect()
    }
    fn scenario(id: &str, weight: f64, d: &[(&str, usize)]) -> RepairDurationScenario {
        RepairDurationScenario::new(id, weight, durations(d))
    }
    fn config() -> RobustRepairConfig { RobustRepairConfig::new(20, 0.9) }

    fn witness_jobs() -> Vec<RepairJob> {
        vec![
            job("A", 1, 0, 1.0, 0),
            job("B", 2, 1, 100.0, 5),
            job("C", 2, 20, 1.0, 0),
        ]
    }

    #[test]
    fn paired_witness_shows_adaptive_gain_without_future_information() {
        let result = compare_fixed_and_rolling_horizon_policies(
            &witness_jobs(), &[crew()], &[],
            &[scenario("delay-A", 1.0, &[("A", 10), ("B", 2), ("C", 2)])],
            &config(),
        ).unwrap();
        let summary = result.policy_summaries.iter()
            .find(|summary| summary.policy == RepairSchedulingPolicy::PriorityFirst).unwrap();
        assert!((summary.fixed_expected_loss - 922.0).abs() < 1.0e-9);
        assert!((summary.rolling_horizon_expected_loss - 724.0).abs() < 1.0e-9);
        assert!((summary.expected_loss_reduction_fraction.unwrap() - (198.0 / 922.0)).abs() < 1.0e-9);
        assert_eq!(summary.improved_scenario_count, 1);
        assert_eq!(summary.worsened_scenario_count, 0);
        assert_eq!(summary.scenarios[0].rolling_horizon_report.schedule.iter().map(|r| r.job_id.as_str()).collect::<Vec<_>>(), vec!["A", "B", "C"]);
    }

    #[test]
    fn paired_evaluation_retains_every_scenario_for_each_policy() {
        let scenarios = [
            scenario("ordinary", 0.75, &[("A", 1), ("B", 2), ("C", 2)]),
            scenario("slow-a", 0.25, &[("A", 10), ("B", 2), ("C", 2)]),
        ];
        let result = compare_fixed_and_rolling_horizon_policies(&witness_jobs(), &[crew()], &[], &scenarios, &config()).unwrap();
        assert_eq!(result.policy_summaries.len(), 3);
        for summary in &result.policy_summaries {
            assert_eq!(summary.scenarios.iter().map(|s| s.scenario_id.as_str()).collect::<Vec<_>>(), vec!["ordinary", "slow-a"]);
            assert_eq!(summary.improved_scenario_count + summary.worsened_scenario_count + summary.unchanged_scenario_count, 2);
            assert!(summary.scenarios.iter().all(|case| case.rolling_horizon_report.decisions.len() > 0));
        }
    }

    #[test]
    fn fixed_and_adaptive_cvar_are_reported_separately() {
        let result = compare_fixed_and_rolling_horizon_policies(
            &[job("pump", 1, 0, 1.0, 0)], &[crew()], &[],
            &[scenario("normal", 0.9, &[("pump", 1)]), scenario("slow", 0.1, &[("pump", 9)])],
            &config(),
        ).unwrap();
        let summary = result.policy_summaries.iter().find(|s| s.policy == RepairSchedulingPolicy::PriorityFirst).unwrap();
        assert!(summary.fixed_cvar_loss.is_finite());
        assert!(summary.rolling_horizon_cvar_loss.is_finite());
        assert_eq!(summary.fixed_cvar_loss, summary.rolling_horizon_cvar_loss);
    }

    #[test]
    fn invalid_scenario_weights_fail_closed_through_fixed_baseline() {
        let err = compare_fixed_and_rolling_horizon_policies(
            &[job("pump", 1, 0, 1.0, 0)], &[crew()], &[],
            &[scenario("a", 0.8, &[("pump", 1)]), scenario("b", 0.3, &[("pump", 2)])],
            &config(),
        ).unwrap_err();
        assert!(matches!(err, RepairPolicyComparisonError::FixedPlanEvaluation(RobustRepairError::ScenarioWeightsDoNotSumToOne)));
    }

    #[test]
    fn identical_cases_produce_identical_comparison_report() {
        let input_jobs = witness_jobs();
        let scenarios = [scenario("case", 1.0, &[("A", 10), ("B", 2), ("C", 2)])];
        let first = compare_fixed_and_rolling_horizon_policies(&input_jobs, &[crew()], &[], &scenarios, &config()).unwrap();
        let second = compare_fixed_and_rolling_horizon_policies(&input_jobs, &[crew()], &[], &scenarios, &config()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn per_scenario_deltas_are_fixed_baseline_paired_differences() {
        let result = compare_fixed_and_rolling_horizon_policies(
            &witness_jobs(), &[crew()], &[],
            &[scenario("delay-A", 1.0, &[("A", 10), ("B", 2), ("C", 2)])],
            &config(),
        ).unwrap();
        let summary = result.policy_summaries.iter().find(|s| s.policy == RepairSchedulingPolicy::PriorityFirst).unwrap();
        let case = &summary.scenarios[0];
        assert!((case.loss_delta_adaptive_minus_fixed - (case.rolling_horizon_loss - case.fixed_plan_loss)).abs() < 1.0e-9);
        assert!(case.loss_delta_adaptive_minus_fixed < 0.0);
        assert!((summary.expected_loss_delta_adaptive_minus_fixed - case.loss_delta_adaptive_minus_fixed).abs() < 1.0e-9);
    }
}
