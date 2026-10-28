// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Finite-ensemble robust evaluation for aviation network recovery policies.
//!
//! This module compares a candidate against the incumbent across a frozen,
//! declared scenario set. It prevents a favorable mean from hiding a scenario
//! regression or a per-actor harm-cap violation. Its guarantees apply only to
//! the supplied finite scenario set; they are not distribution-free guarantees
//! about unseen disruptions.
//!
//! Inputs must come from a separate feasibility checker. Scenario IDs must bind
//! to immutable input digests, and objective weights must be frozen before the
//! evaluation. No forecasts, costs or feasibility labels are verified here.

use std::collections::{BTreeMap, BTreeSet};

use crate::aviation_network_externalities::{
    compare_network_outcomes, ActorDomain, FeasibilityStatus, NetworkComparisonError,
    NetworkObjective, NetworkOutcome,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustSetError {
    EmptyScenarioSet,
    DuplicateScenarioId { side: &'static str, scenario_id: String },
    ScenarioSetMismatch {
        missing_from_candidate: Vec<String>,
        extra_in_candidate: Vec<String>,
    },
    ActorUniverseMismatch { scenario_id: String, missing: Vec<String>, extra: Vec<String> },
    ActorDomainChanged { actor_id: String, expected: ActorDomain, actual: ActorDomain },
    InvalidTailFraction,
    OutcomeComparison(NetworkComparisonError),
    ArithmeticOverflow { context: &'static str },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorWorstRegression {
    pub actor_id: String,
    pub domain: ActorDomain,
    pub worst_regression_units: u128,
    pub worst_scenario_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioRisk {
    pub scenario_id: String,
    /// Positive means the candidate reduces the network objective.
    pub net_system_benefit_units: i128,
    /// Positive is a candidate system-cost regression for this scenario.
    pub system_cost_regression_units: u128,
    pub worst_actor_regression_units: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustNetworkReport {
    pub scenario_count: usize,
    pub objective_id: String,
    pub mean_net_system_benefit_units: f64,
    pub worst_case_net_system_benefit_units: i128,
    /// Mean of the worst upper-tail fraction of positive system-cost regressions.
    /// Scenarios with no regression contribute zero to the loss distribution.
    pub upper_tail_system_regression_units: f64,
    pub worst_actor_regression_units: u128,
    pub scenarios_with_system_regression: usize,
    pub scenarios_with_system_improvement: usize,
    pub scenarios_unchanged: usize,
    pub actor_worst_regressions: Vec<ActorWorstRegression>,
    /// Sorted by scenario ID for deterministic reporting.
    pub scenario_risks: Vec<ScenarioRisk>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustNetworkConstraints {
    /// Hard cap enforced for every actor in every supplied scenario.
    pub max_actor_regression_units: u128,
    /// Minimum mean benefit across the full scenario set.
    pub min_mean_system_benefit_units: f64,
    /// Minimum single-scenario benefit; negative values permit bounded regressions.
    pub min_worst_case_system_benefit_units: i128,
    /// Maximum empirical upper-tail mean of positive system-cost regression.
    pub max_upper_tail_system_regression_units: f64,
    /// Fraction of scenarios in the upper regression tail, in (0, 1].
    pub tail_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RobustExclusion {
    ActorHarmCap {
        actor_id: String,
        scenario_id: String,
        regression_units: u128,
        maximum_allowed_units: u128,
    },
    MeanBenefitBelowMinimum { actual: f64, minimum: f64 },
    WorstCaseBenefitBelowMinimum { actual: i128, minimum: i128 },
    UpperTailRegressionAboveMaximum { actual: f64, maximum: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustCandidateAssessment {
    pub eligible: bool,
    pub exclusions: Vec<RobustExclusion>,
    pub report: RobustNetworkReport,
}

fn checked_sum_i128(a: i128, b: i128, context: &'static str) -> Result<i128, RobustSetError> {
    a.checked_add(b).ok_or(RobustSetError::ArithmeticOverflow { context })
}

fn positive_regression(benefit: i128) -> u128 {
    if benefit < 0 { benefit.unsigned_abs() } else { 0 }
}

fn upper_tail_mean(values: &[u128], fraction: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().map(|x| *x as f64).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mass = sorted.len() as f64 * fraction;
    let n = mass.floor() as usize;
    let fractional = mass - n as f64;
    let full_sum: f64 = sorted.iter().take(n).sum();
    let partial = if fractional > 0.0 && n < sorted.len() {
        sorted[n] * fractional
    } else {
        0.0
    };
    (full_sum + partial) / mass
}

fn actor_universe(outcomes: &[NetworkOutcome]) -> Result<BTreeMap<String, ActorDomain>, RobustSetError> {
    let first = outcomes.first().ok_or(RobustSetError::EmptyScenarioSet)?;
    let expected: BTreeMap<String, ActorDomain> = first.actors.iter()
        .map(|a| (a.actor_id.clone(), a.domain)).collect();
    let expected_ids: BTreeSet<String> = expected.keys().cloned().collect();

    for outcome in outcomes.iter().skip(1) {
        let actual: BTreeMap<String, ActorDomain> = outcome.actors.iter()
            .map(|a| (a.actor_id.clone(), a.domain)).collect();
        let actual_ids: BTreeSet<String> = actual.keys().cloned().collect();
        if expected_ids != actual_ids {
            return Err(RobustSetError::ActorUniverseMismatch {
                scenario_id: outcome.scenario_id.clone(),
                missing: expected_ids.difference(&actual_ids).cloned().collect(),
                extra: actual_ids.difference(&expected_ids).cloned().collect(),
            });
        }
        for (id, domain) in &expected {
            let actual_domain = actual[id];
            if *domain != actual_domain {
                return Err(RobustSetError::ActorDomainChanged {
                    actor_id: id.clone(),
                    expected: *domain,
                    actual: actual_domain,
                });
            }
        }
    }
    Ok(expected)
}

fn index_scenarios<'a>(
    outcomes: &'a [NetworkOutcome],
    side: &'static str,
) -> Result<BTreeMap<&'a str, &'a NetworkOutcome>, RobustSetError> {
    if outcomes.is_empty() {
        return Err(RobustSetError::EmptyScenarioSet);
    }
    let mut indexed = BTreeMap::new();
    for outcome in outcomes {
        if indexed.insert(outcome.scenario_id.as_str(), outcome).is_some() {
            return Err(RobustSetError::DuplicateScenarioId {
                side,
                scenario_id: outcome.scenario_id.clone(),
            });
        }
    }
    Ok(indexed)
}

/// Evaluate one candidate on a frozen finite scenario set against an incumbent.
///
/// Both sets must contain exactly the same unique scenario IDs and the same
/// actor universe/domain mapping. Every paired plan must be marked verified
/// feasible by a separate checker. The function computes a worst-case actor
/// regression across all actor/scenario pairs, mean system benefit, worst-case
/// system benefit and upper-tail mean positive system-cost regression.
///
/// The output is descriptive for the declared finite ensemble. It does not
/// guarantee performance on unobserved or adversarially chosen future events.
pub fn evaluate_robust_network_candidate(
    objective: &NetworkObjective,
    baseline: &[NetworkOutcome],
    candidate: &[NetworkOutcome],
    tail_fraction: f64,
) -> Result<RobustNetworkReport, RobustSetError> {
    if !tail_fraction.is_finite() || tail_fraction <= 0.0 || tail_fraction > 1.0 {
        return Err(RobustSetError::InvalidTailFraction);
    }
    let base_by_id = index_scenarios(baseline, "baseline")?;
    let candidate_by_id = index_scenarios(candidate, "candidate")?;
    let base_ids: BTreeSet<&str> = base_by_id.keys().copied().collect();
    let candidate_ids: BTreeSet<&str> = candidate_by_id.keys().copied().collect();
    if base_ids != candidate_ids {
        return Err(RobustSetError::ScenarioSetMismatch {
            missing_from_candidate: base_ids.difference(&candidate_ids).map(|x| (*x).to_owned()).collect(),
            extra_in_candidate: candidate_ids.difference(&base_ids).map(|x| (*x).to_owned()).collect(),
        });
    }
    let universe = actor_universe(baseline)?;
    let candidate_universe = actor_universe(candidate)?;
    if universe != candidate_universe {
        // Pairwise comparison below also checks each actor set. This whole-set check
        // catches consistent-but-wrong actor-domain changes across all candidate cases.
        return Err(RobustSetError::ActorUniverseMismatch {
            scenario_id: String::new(),
            missing: universe.keys().filter(|k| !candidate_universe.contains_key(*k)).cloned().collect(),
            extra: candidate_universe.keys().filter(|k| !universe.contains_key(*k)).cloned().collect(),
        });
    }

    let mut total_benefit = 0_i128;
    let mut worst_benefit = i128::MAX;
    let mut system_losses = Vec::with_capacity(base_by_id.len());
    let mut worst_actor: BTreeMap<String, (u128, String)> = BTreeMap::new();
    let mut scenario_risks = Vec::with_capacity(base_by_id.len());
    let mut regressions = 0;
    let mut improvements = 0;
    let mut unchanged = 0;

    for (scenario_id, base) in &base_by_id {
        let cand = candidate_by_id[scenario_id];
        let report = compare_network_outcomes(objective, base, cand)
            .map_err(RobustSetError::OutcomeComparison)?;
        total_benefit = checked_sum_i128(total_benefit, report.net_system_benefit_units, "scenario benefit sum")?;
        worst_benefit = worst_benefit.min(report.net_system_benefit_units);
        let loss = positive_regression(report.net_system_benefit_units);
        system_losses.push(loss);

        match report.net_system_benefit_units.cmp(&0) {
            std::cmp::Ordering::Less => regressions += 1,
            std::cmp::Ordering::Greater => improvements += 1,
            std::cmp::Ordering::Equal => unchanged += 1,
        }
        for actor in &report.actor_deltas {
            let regression = if actor.candidate_delta_units > 0 {
                actor.candidate_delta_units as u128
            } else {
                0
            };
            let entry = worst_actor.entry(actor.actor_id.clone()).or_insert((0, (*scenario_id).to_owned()));
            if regression > entry.0 {
                *entry = (regression, (*scenario_id).to_owned());
            }
        }
        scenario_risks.push(ScenarioRisk {
            scenario_id: (*scenario_id).to_owned(),
            net_system_benefit_units: report.net_system_benefit_units,
            system_cost_regression_units: loss,
            worst_actor_regression_units: report.worst_actor_regression_units,
        });
    }
    scenario_risks.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));

    let n = base_by_id.len() as f64;
    let mean = total_benefit as f64 / n;
    let worst_actor_regression_units = worst_actor.values().map(|x| x.0).max().unwrap_or(0);
    let actor_worst_regressions = worst_actor.into_iter().map(|(actor_id, (units, scenario_id))| {
        ActorWorstRegression {
            domain: universe[&actor_id],
            actor_id,
            worst_regression_units: units,
            worst_scenario_id: scenario_id,
        }
    }).collect();

    Ok(RobustNetworkReport {
        scenario_count: base_by_id.len(),
        objective_id: objective.id.clone(),
        mean_net_system_benefit_units: mean,
        worst_case_net_system_benefit_units: worst_benefit,
        upper_tail_system_regression_units: upper_tail_mean(&system_losses, tail_fraction),
        worst_actor_regression_units,
        scenarios_with_system_regression: regressions,
        scenarios_with_system_improvement: improvements,
        scenarios_unchanged: unchanged,
        actor_worst_regressions,
        scenario_risks,
    })
}

/// Apply predeclared robust constraints to a report. A policy is eligible only
/// if the per-actor harm cap holds in every observed scenario, the mean benefit
/// and worst-case benefit meet their thresholds, and the upper-tail regression
/// stays under the configured maximum.
pub fn assess_robust_network_candidate(
    report: RobustNetworkReport,
    constraints: RobustNetworkConstraints,
) -> Result<RobustCandidateAssessment, RobustSetError> {
    if !constraints.tail_fraction.is_finite()
        || constraints.tail_fraction <= 0.0
        || constraints.tail_fraction > 1.0
        || !constraints.min_mean_system_benefit_units.is_finite()
        || !constraints.max_upper_tail_system_regression_units.is_finite()
    {
        return Err(RobustSetError::InvalidTailFraction);
    }

    let mut exclusions = Vec::new();
    for actor in &report.actor_worst_regressions {
        if actor.worst_regression_units > constraints.max_actor_regression_units {
            exclusions.push(RobustExclusion::ActorHarmCap {
                actor_id: actor.actor_id.clone(),
                scenario_id: actor.worst_scenario_id.clone(),
                regression_units: actor.worst_regression_units,
                maximum_allowed_units: constraints.max_actor_regression_units,
            });
        }
    }
    if report.mean_net_system_benefit_units < constraints.min_mean_system_benefit_units {
        exclusions.push(RobustExclusion::MeanBenefitBelowMinimum {
            actual: report.mean_net_system_benefit_units,
            minimum: constraints.min_mean_system_benefit_units,
        });
    }
    if report.worst_case_net_system_benefit_units < constraints.min_worst_case_system_benefit_units {
        exclusions.push(RobustExclusion::WorstCaseBenefitBelowMinimum {
            actual: report.worst_case_net_system_benefit_units,
            minimum: constraints.min_worst_case_system_benefit_units,
        });
    }
    if report.upper_tail_system_regression_units > constraints.max_upper_tail_system_regression_units {
        exclusions.push(RobustExclusion::UpperTailRegressionAboveMaximum {
            actual: report.upper_tail_system_regression_units,
            maximum: constraints.max_upper_tail_system_regression_units,
        });
    }
    Ok(RobustCandidateAssessment { eligible: exclusions.is_empty(), exclusions, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aviation_network_externalities::{ActorOutcome, FeasibilityStatus};

    fn objective() -> NetworkObjective {
        NetworkObjective {
            id: "robust-v1".into(),
            delay_cost_per_minute_units: 0,
            cancellation_cost_per_flight_units: 0,
        }
    }

    fn outcome(id: &str, airline: u64, airport: u64) -> NetworkOutcome {
        NetworkOutcome {
            scenario_id: id.into(),
            objective_id: "robust-v1".into(),
            feasibility: FeasibilityStatus::VerifiedFeasible,
            actors: vec![
                ActorOutcome {
                    actor_id: "airline".into(),
                    domain: ActorDomain::Airline,
                    direct_cost_units: airline,
                    non_cancelled_delay_minutes: 0,
                    cancelled_flights: 0,
                    co2e_kg: 0,
                },
                ActorOutcome {
                    actor_id: "airport".into(),
                    domain: ActorDomain::Airport,
                    direct_cost_units: airport,
                    non_cancelled_delay_minutes: 0,
                    cancelled_flights: 0,
                    co2e_kg: 0,
                },
            ],
        }
    }

    fn base() -> Vec<NetworkOutcome> {
        vec![outcome("S1", 100, 100), outcome("S2", 100, 100), outcome("S3", 100, 100)]
    }

    #[test]
    fn positive_mean_does_not_hide_a_worst_case_loss() {
        let candidate = vec![outcome("S1", 70, 100), outcome("S2", 100, 140), outcome("S3", 70, 100)];
        let report = evaluate_robust_network_candidate(&objective(), &base(), &candidate, 1.0 / 3.0).unwrap();
        assert!(report.mean_net_system_benefit_units > 0.0);
        assert_eq!(report.worst_case_net_system_benefit_units, -40);
        assert_eq!(report.scenarios_with_system_regression, 1);
        assert_eq!(report.upper_tail_system_regression_units, 40.0);
    }

    #[test]
    fn actor_harm_cap_applies_to_every_scenario() {
        let candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 130), outcome("S3", 80, 110)];
        let report = evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5).unwrap();
        let assessment = assess_robust_network_candidate(report, RobustNetworkConstraints {
            max_actor_regression_units: 20,
            min_mean_system_benefit_units: 0.0,
            min_worst_case_system_benefit_units: -100,
            max_upper_tail_system_regression_units: 100.0,
            tail_fraction: 0.5,
        }).unwrap();
        assert!(!assessment.eligible);
        assert!(assessment.exclusions.iter().any(|x| matches!(x, RobustExclusion::ActorHarmCap { actor_id, regression_units: 30, .. } if actor_id == "airport")));
    }

    #[test]
    fn accepts_when_every_finite_set_gate_passes() {
        let candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 110), outcome("S3", 80, 110)];
        let report = evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5).unwrap();
        let assessment = assess_robust_network_candidate(report, RobustNetworkConstraints {
            max_actor_regression_units: 20,
            min_mean_system_benefit_units: 5.0,
            min_worst_case_system_benefit_units: 0,
            max_upper_tail_system_regression_units: 0.0,
            tail_fraction: 0.5,
        }).unwrap();
        assert!(assessment.eligible);
        assert!(assessment.exclusions.is_empty());
    }

    #[test]
    fn scenario_order_does_not_change_result() {
        let baseline = base();
        let candidate = vec![outcome("S3", 80, 110), outcome("S1", 80, 110), outcome("S2", 80, 110)];
        let a = evaluate_robust_network_candidate(&objective(), &baseline, &candidate, 0.5).unwrap();
        let b = evaluate_robust_network_candidate(&objective(), &baseline, &candidate, 0.5).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.scenario_risks[0].scenario_id, "S1");
    }

    #[test]
    fn rejects_duplicate_scenario_identifiers() {
        let mut candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 110), outcome("S2", 80, 110)];
        assert!(matches!(
            evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5),
            Err(RobustSetError::DuplicateScenarioId { side: "candidate", .. })
        ));
        candidate.clear();
    }

    #[test]
    fn rejects_scenario_set_mismatch() {
        let candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 110), outcome("S4", 80, 110)];
        assert!(matches!(
            evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5),
            Err(RobustSetError::ScenarioSetMismatch { .. })
        ));
    }

    #[test]
    fn rejects_actor_universe_drift_between_scenarios() {
        let mut candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 110), outcome("S3", 80, 110)];
        candidate[2].actors.pop();
        assert!(matches!(
            evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5),
            Err(RobustSetError::OutcomeComparison(NetworkComparisonError::ActorSetMismatch { .. }))
                | Err(RobustSetError::ActorUniverseMismatch { .. })
        ));
    }

    #[test]
    fn rejects_unverified_outcomes() {
        let baseline = base();
        let mut candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 110), outcome("S3", 80, 110)];
        candidate[1].feasibility = FeasibilityStatus::Indeterminate;
        assert!(matches!(
            evaluate_robust_network_candidate(&objective(), &baseline, &candidate, 0.5),
            Err(RobustSetError::OutcomeComparison(NetworkComparisonError::OutcomeNotVerified { .. }))
        ));
    }

    #[test]
    fn rejects_invalid_tail_fraction() {
        for fraction in [0.0, -0.1, 1.1, f64::NAN] {
            assert!(matches!(
                evaluate_robust_network_candidate(&objective(), &base(), &base(), fraction),
                Err(RobustSetError::InvalidTailFraction)
            ));
        }
    }

    #[test]
    fn mean_and_tail_constraints_fail_closed_on_non_finite_values() {
        let report = evaluate_robust_network_candidate(&objective(), &base(), &base(), 0.5).unwrap();
        let result = assess_robust_network_candidate(report, RobustNetworkConstraints {
            max_actor_regression_units: 10,
            min_mean_system_benefit_units: f64::NAN,
            min_worst_case_system_benefit_units: 0,
            max_upper_tail_system_regression_units: 10.0,
            tail_fraction: 0.5,
        });
        assert!(matches!(result, Err(RobustSetError::InvalidTailFraction)));
    }

    #[test]
    fn maximum_actor_regression_is_reported_with_scenario_id() {
        let candidate = vec![outcome("S1", 80, 110), outcome("S2", 80, 130), outcome("S3", 80, 110)];
        let report = evaluate_robust_network_candidate(&objective(), &base(), &candidate, 0.5).unwrap();
        let airport = report.actor_worst_regressions.iter().find(|a| a.actor_id == "airport").unwrap();
        assert_eq!(airport.worst_regression_units, 30);
        assert_eq!(airport.worst_scenario_id, "S2");
    }
}
