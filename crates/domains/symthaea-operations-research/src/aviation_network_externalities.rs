// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired network-externality accounting and harm-constrained candidate selection.
//!
//! This module evaluates declared outcomes; it does not generate operational
//! actions or independently verify aircraft/dispatch/maintenance feasibility.
//! Cost weights are synthetic unless explicitly calibrated and reviewed.
//! Cancelled-flight delay must be excluded from non_cancelled_delay_minutes to
//! avoid counting the same disruption twice. CO2e remains a separate ledger.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActorDomain {
    Airline,
    Airport,
    AirNavigationProvider,
    Passenger,
    MaintenanceProvider,
    Supplier,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibilityStatus {
    VerifiedFeasible,
    Rejected,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkObjective {
    /// Frozen, versioned objective identifier shared by all paired outcomes.
    pub id: String,
    pub delay_cost_per_minute_units: u64,
    pub cancellation_cost_per_flight_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorOutcome {
    pub actor_id: String,
    pub domain: ActorDomain,
    pub direct_cost_units: u64,
    /// Must exclude cancelled-flight delay minutes; cancellations are counted separately.
    pub non_cancelled_delay_minutes: u64,
    pub cancelled_flights: u64,
    /// Separate kg CO2-equivalent ledger; not monetized by this objective.
    pub co2e_kg: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkOutcome {
    /// Must identify one immutable scenario/input digest.
    pub scenario_id: String,
    pub objective_id: String,
    pub feasibility: FeasibilityStatus,
    /// The same complete set of actor IDs/domains must appear in baseline and candidate.
    pub actors: Vec<ActorOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySide {
    Baseline,
    Candidate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkComparisonError {
    EmptyIdentifier { field: &'static str },
    EmptyActorSet,
    DuplicateActorId { side: PolicySide, actor_id: String },
    ActorSetMismatch { missing_from_candidate: Vec<String>, extra_in_candidate: Vec<String> },
    ActorDomainMismatch { actor_id: String, baseline: ActorDomain, candidate: ActorDomain },
    ScenarioMismatch { baseline: String, candidate: String },
    ObjectiveMismatch { expected: String, actual: String, side: PolicySide },
    OutcomeNotVerified { side: PolicySide, status: FeasibilityStatus },
    ArithmeticOverflow { context: &'static str },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorDelta {
    pub actor_id: String,
    pub domain: ActorDomain,
    pub baseline_cost_units: u128,
    pub candidate_cost_units: u128,
    /// Positive means this actor is worse off under the candidate.
    pub candidate_delta_units: i128,
    pub baseline_non_cancelled_delay_minutes: u64,
    pub candidate_non_cancelled_delay_minutes: u64,
    pub baseline_cancelled_flights: u64,
    pub candidate_cancelled_flights: u64,
    /// Positive means candidate emissions are higher.
    pub candidate_co2e_delta_kg: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkExternalityReport {
    pub scenario_id: String,
    pub objective_id: String,
    pub actor_count: usize,
    pub baseline_total_cost_units: u128,
    pub candidate_total_cost_units: u128,
    /// Positive means the candidate reduces system-wide modeled cost.
    pub net_system_benefit_units: i128,
    pub gross_actor_benefit_units: u128,
    pub gross_actor_burden_units: u128,
    /// Exact rational ratio (gross burden / gross benefit), None when benefit is zero.
    pub spillover_ratio: Option<(u128, u128)>,
    pub actors_helped: usize,
    pub actors_harmed: usize,
    pub actors_unchanged: usize,
    pub worst_actor_regression_units: u128,
    pub baseline_non_cancelled_delay_minutes: u128,
    pub candidate_non_cancelled_delay_minutes: u128,
    pub non_cancelled_delay_minutes_avoided: i128,
    pub baseline_cancelled_flights: u128,
    pub candidate_cancelled_flights: u128,
    pub cancellations_avoided: i128,
    pub baseline_total_co2e_kg: u128,
    pub candidate_total_co2e_kg: u128,
    pub co2e_kg_avoided: i128,
    /// Sorted by actor ID, independent of input row order.
    pub actor_deltas: Vec<ActorDelta>,
}

fn index_actors<'a>(
    actors: &'a [ActorOutcome],
    side: PolicySide,
) -> Result<BTreeMap<&'a str, &'a ActorOutcome>, NetworkComparisonError> {
    let mut result = BTreeMap::new();
    for actor in actors {
        if actor.actor_id.trim().is_empty() {
            return Err(NetworkComparisonError::EmptyIdentifier { field: "actor_id" });
        }
        if result.insert(actor.actor_id.as_str(), actor).is_some() {
            return Err(NetworkComparisonError::DuplicateActorId {
                side,
                actor_id: actor.actor_id.clone(),
            });
        }
    }
    Ok(result)
}

fn add_u128(a: u128, b: u128, context: &'static str) -> Result<u128, NetworkComparisonError> {
    a.checked_add(b).ok_or(NetworkComparisonError::ArithmeticOverflow { context })
}

fn to_i128(value: u128, context: &'static str) -> Result<i128, NetworkComparisonError> {
    i128::try_from(value).map_err(|_| NetworkComparisonError::ArithmeticOverflow { context })
}

fn actor_cost(
    objective: &NetworkObjective,
    actor: &ActorOutcome,
) -> Result<u128, NetworkComparisonError> {
    let delay = u128::from(actor.non_cancelled_delay_minutes)
        .checked_mul(u128::from(objective.delay_cost_per_minute_units))
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "delay cost product" })?;
    let cancellations = u128::from(actor.cancelled_flights)
        .checked_mul(u128::from(objective.cancellation_cost_per_flight_units))
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "cancellation cost product" })?;
    let cost = u128::from(actor.direct_cost_units)
        .checked_add(delay)
        .and_then(|x| x.checked_add(cancellations))
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "actor cost sum" })?;
    to_i128(cost, "actor cost")?;
    Ok(cost)
}

/// Compare baseline and candidate outcomes for one scenario, objective, and actor universe.
/// Feasibility status must be supplied by a separate checker; this function does not verify it.
pub fn compare_network_outcomes(
    objective: &NetworkObjective,
    baseline: &NetworkOutcome,
    candidate: &NetworkOutcome,
) -> Result<NetworkExternalityReport, NetworkComparisonError> {
    if objective.id.trim().is_empty() {
        return Err(NetworkComparisonError::EmptyIdentifier { field: "objective.id" });
    }
    if baseline.scenario_id.trim().is_empty() || candidate.scenario_id.trim().is_empty() {
        return Err(NetworkComparisonError::EmptyIdentifier { field: "scenario_id" });
    }
    if baseline.scenario_id != candidate.scenario_id {
        return Err(NetworkComparisonError::ScenarioMismatch {
            baseline: baseline.scenario_id.clone(),
            candidate: candidate.scenario_id.clone(),
        });
    }
    for (side, outcome) in [(PolicySide::Baseline, baseline), (PolicySide::Candidate, candidate)] {
        if outcome.objective_id != objective.id {
            return Err(NetworkComparisonError::ObjectiveMismatch {
                expected: objective.id.clone(),
                actual: outcome.objective_id.clone(),
                side,
            });
        }
        if outcome.feasibility != FeasibilityStatus::VerifiedFeasible {
            return Err(NetworkComparisonError::OutcomeNotVerified { side, status: outcome.feasibility });
        }
        if outcome.actors.is_empty() {
            return Err(NetworkComparisonError::EmptyActorSet);
        }
    }

    let base = index_actors(&baseline.actors, PolicySide::Baseline)?;
    let cand = index_actors(&candidate.actors, PolicySide::Candidate)?;
    let base_ids: BTreeSet<&str> = base.keys().copied().collect();
    let cand_ids: BTreeSet<&str> = cand.keys().copied().collect();
    if base_ids != cand_ids {
        return Err(NetworkComparisonError::ActorSetMismatch {
            missing_from_candidate: base_ids.difference(&cand_ids).map(|x| (*x).to_owned()).collect(),
            extra_in_candidate: cand_ids.difference(&base_ids).map(|x| (*x).to_owned()).collect(),
        });
    }

    let mut actor_deltas = Vec::with_capacity(base.len());
    let mut baseline_cost = 0_u128;
    let mut candidate_cost = 0_u128;
    let mut gross_benefit = 0_u128;
    let mut gross_burden = 0_u128;
    let mut helped = 0;
    let mut harmed = 0;
    let mut unchanged = 0;
    let mut worst_regression = 0_u128;
    let mut base_delay = 0_u128;
    let mut cand_delay = 0_u128;
    let mut base_cancel = 0_u128;
    let mut cand_cancel = 0_u128;
    let mut base_co2 = 0_u128;
    let mut cand_co2 = 0_u128;

    for (actor_id, b) in &base {
        let c = cand.get(actor_id).expect("actor sets checked equal");
        if b.domain != c.domain {
            return Err(NetworkComparisonError::ActorDomainMismatch {
                actor_id: (*actor_id).to_owned(),
                baseline: b.domain,
                candidate: c.domain,
            });
        }
        let bc = actor_cost(objective, b)?;
        let cc = actor_cost(objective, c)?;
        let delta = to_i128(cc, "candidate actor cost")?
            .checked_sub(to_i128(bc, "baseline actor cost")?)
            .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "actor delta" })?;
        baseline_cost = add_u128(baseline_cost, bc, "baseline system cost")?;
        candidate_cost = add_u128(candidate_cost, cc, "candidate system cost")?;
        base_delay = add_u128(base_delay, u128::from(b.non_cancelled_delay_minutes), "baseline delay")?;
        cand_delay = add_u128(cand_delay, u128::from(c.non_cancelled_delay_minutes), "candidate delay")?;
        base_cancel = add_u128(base_cancel, u128::from(b.cancelled_flights), "baseline cancellations")?;
        cand_cancel = add_u128(cand_cancel, u128::from(c.cancelled_flights), "candidate cancellations")?;
        base_co2 = add_u128(base_co2, u128::from(b.co2e_kg), "baseline emissions")?;
        cand_co2 = add_u128(cand_co2, u128::from(c.co2e_kg), "candidate emissions")?;

        if delta < 0 {
            gross_benefit = add_u128(gross_benefit, (-delta) as u128, "gross benefit")?;
            helped += 1;
        } else if delta > 0 {
            gross_burden = add_u128(gross_burden, delta as u128, "gross burden")?;
            worst_regression = worst_regression.max(delta as u128);
            harmed += 1;
        } else {
            unchanged += 1;
        }
        actor_deltas.push(ActorDelta {
            actor_id: (*actor_id).to_owned(),
            domain: b.domain,
            baseline_cost_units: bc,
            candidate_cost_units: cc,
            candidate_delta_units: delta,
            baseline_non_cancelled_delay_minutes: b.non_cancelled_delay_minutes,
            candidate_non_cancelled_delay_minutes: c.non_cancelled_delay_minutes,
            baseline_cancelled_flights: b.cancelled_flights,
            candidate_cancelled_flights: c.cancelled_flights,
            candidate_co2e_delta_kg: i128::from(c.co2e_kg) - i128::from(b.co2e_kg),
        });
    }

    let benefit = to_i128(baseline_cost, "baseline total")?
        .checked_sub(to_i128(candidate_cost, "candidate total")?)
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "system benefit" })?;
    let delay_avoided = to_i128(base_delay, "baseline delay total")?
        .checked_sub(to_i128(cand_delay, "candidate delay total")?)
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "delay difference" })?;
    let cancellations_avoided = to_i128(base_cancel, "baseline cancellation total")?
        .checked_sub(to_i128(cand_cancel, "candidate cancellation total")?)
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "cancellation difference" })?;
    let co2_avoided = to_i128(base_co2, "baseline emissions")?
        .checked_sub(to_i128(cand_co2, "candidate emissions")?)
        .ok_or(NetworkComparisonError::ArithmeticOverflow { context: "emissions difference" })?;

    Ok(NetworkExternalityReport {
        scenario_id: baseline.scenario_id.clone(),
        objective_id: objective.id.clone(),
        actor_count: actor_deltas.len(),
        baseline_total_cost_units: baseline_cost,
        candidate_total_cost_units: candidate_cost,
        net_system_benefit_units: benefit,
        gross_actor_benefit_units: gross_benefit,
        gross_actor_burden_units: gross_burden,
        spillover_ratio: if gross_benefit == 0 { None } else { Some((gross_burden, gross_benefit)) },
        actors_helped: helped,
        actors_harmed: harmed,
        actors_unchanged: unchanged,
        worst_actor_regression_units: worst_regression,
        baseline_non_cancelled_delay_minutes: base_delay,
        candidate_non_cancelled_delay_minutes: cand_delay,
        non_cancelled_delay_minutes_avoided: delay_avoided,
        baseline_cancelled_flights: base_cancel,
        candidate_cancelled_flights: cand_cancel,
        cancellations_avoided,
        baseline_total_co2e_kg: base_co2,
        candidate_total_co2e_kg: cand_co2,
        co2e_kg_avoided: co2_avoided,
        actor_deltas,
    })
}

/// Predeclared policy constraints. These are governance choices, not universal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkSelectionConstraints {
    pub max_actor_regression_units: u128,
    pub minimum_net_system_benefit_units: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateExclusionReason {
    InvalidLedger(NetworkComparisonError),
    ActorHarmCeilingExceeded { actor_id: String, regression_units: u128, maximum_allowed_units: u128 },
    InsufficientSystemBenefit { benefit_units: i128, minimum_required_units: i128 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAssessment {
    pub candidate_index: usize,
    pub eligible: bool,
    pub exclusion: Option<CandidateExclusionReason>,
    pub report: Option<NetworkExternalityReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkSelection {
    /// None means all options failed at least one predeclared gate; caller must escalate.
    pub selected_candidate_index: Option<usize>,
    pub assessments: Vec<CandidateAssessment>,
}

/// Choose the lowest-cost eligible candidate, breaking ties by lower worst-actor
/// regression and then original index. Invalid candidates are recorded; an invalid
/// baseline is a hard error. Caller must not treat None as permission to execute a plan.
pub fn select_network_candidate(
    objective: &NetworkObjective,
    baseline: &NetworkOutcome,
    candidates: &[NetworkOutcome],
    constraints: NetworkSelectionConstraints,
) -> Result<NetworkSelection, NetworkComparisonError> {
    let _ = compare_network_outcomes(objective, baseline, baseline)?;
    let mut assessments = Vec::with_capacity(candidates.len());
    let mut eligible = Vec::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let report = match compare_network_outcomes(objective, baseline, candidate) {
            Ok(report) => report,
            Err(error) => {
                assessments.push(CandidateAssessment {
                    candidate_index: index,
                    eligible: false,
                    exclusion: Some(CandidateExclusionReason::InvalidLedger(error)),
                    report: None,
                });
                continue;
            }
        };
        if let Some(actor) = report.actor_deltas.iter().find(|x| {
            x.candidate_delta_units > 0 && (x.candidate_delta_units as u128) > constraints.max_actor_regression_units
        }) {
            assessments.push(CandidateAssessment {
                candidate_index: index,
                eligible: false,
                exclusion: Some(CandidateExclusionReason::ActorHarmCeilingExceeded {
                    actor_id: actor.actor_id.clone(),
                    regression_units: actor.candidate_delta_units as u128,
                    maximum_allowed_units: constraints.max_actor_regression_units,
                }),
                report: Some(report),
            });
            continue;
        }
        if report.net_system_benefit_units < constraints.minimum_net_system_benefit_units {
            assessments.push(CandidateAssessment {
                candidate_index: index,
                eligible: false,
                exclusion: Some(CandidateExclusionReason::InsufficientSystemBenefit {
                    benefit_units: report.net_system_benefit_units,
                    minimum_required_units: constraints.minimum_net_system_benefit_units,
                }),
                report: Some(report),
            });
            continue;
        }
        eligible.push((report.candidate_total_cost_units, report.worst_actor_regression_units, index));
        assessments.push(CandidateAssessment { candidate_index: index, eligible: true, exclusion: None, report: Some(report) });
    }

    eligible.sort_unstable();
    Ok(NetworkSelection {
        selected_candidate_index: eligible.first().map(|(_, _, index)| *index),
        assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective() -> NetworkObjective {
        NetworkObjective { id: "v1".into(), delay_cost_per_minute_units: 2, cancellation_cost_per_flight_units: 100 }
    }
    fn actor(id: &str, domain: ActorDomain, direct: u64, delay: u64, cancel: u64, co2: u64) -> ActorOutcome {
        ActorOutcome {
            actor_id: id.into(), domain, direct_cost_units: direct,
            non_cancelled_delay_minutes: delay, cancelled_flights: cancel, co2e_kg: co2,
        }
    }
    fn outcome(actors: Vec<ActorOutcome>) -> NetworkOutcome {
        NetworkOutcome { scenario_id: "S1".into(), objective_id: "v1".into(), feasibility: FeasibilityStatus::VerifiedFeasible, actors }
    }
    fn base() -> NetworkOutcome {
        outcome(vec![
            actor("airline", ActorDomain::Airline, 100, 40, 0, 1000),
            actor("airport", ActorDomain::Airport, 20, 20, 0, 300),
            actor("ansp", ActorDomain::AirNavigationProvider, 10, 10, 0, 100),
        ])
    }
    fn externalizing() -> NetworkOutcome {
        outcome(vec![
            actor("airline", ActorDomain::Airline, 100, 10, 0, 900),
            actor("airport", ActorDomain::Airport, 20, 35, 0, 330),
            actor("ansp", ActorDomain::AirNavigationProvider, 10, 15, 0, 120),
        ])
    }
    fn bounded_harm() -> NetworkOutcome {
        outcome(vec![
            actor("airline", ActorDomain::Airline, 100, 25, 0, 950),
            actor("airport", ActorDomain::Airport, 20, 30, 0, 310),
            actor("ansp", ActorDomain::AirNavigationProvider, 10, 10, 0, 100),
        ])
    }

    #[test]
    fn computes_network_benefit_and_displaced_burden() {
        let r = compare_network_outcomes(&objective(), &base(), &externalizing()).unwrap();
        assert_eq!(r.baseline_total_cost_units, 270);
        assert_eq!(r.candidate_total_cost_units, 250);
        assert_eq!(r.net_system_benefit_units, 20);
        assert_eq!(r.gross_actor_benefit_units, 60);
        assert_eq!(r.gross_actor_burden_units, 40);
        assert_eq!(r.worst_actor_regression_units, 30);
    }
    #[test]
    fn co2e_remains_a_separate_ledger() {
        let r = compare_network_outcomes(&objective(), &base(), &externalizing()).unwrap();
        assert_eq!(r.co2e_kg_avoided, 50);
    }
    #[test]
    fn rejects_candidate_that_omits_an_affected_actor() {
        let mut c = externalizing();
        c.actors.pop();
        assert!(matches!(compare_network_outcomes(&objective(), &base(), &c), Err(NetworkComparisonError::ActorSetMismatch { .. })));
    }
    #[test]
    fn rejects_unverified_or_indeterminate_candidate() {
        for status in [FeasibilityStatus::Rejected, FeasibilityStatus::Indeterminate] {
            let mut c = externalizing();
            c.feasibility = status;
            assert!(matches!(compare_network_outcomes(&objective(), &base(), &c), Err(NetworkComparisonError::OutcomeNotVerified { side: PolicySide::Candidate, .. })));
        }
    }
    #[test]
    fn rejects_duplicate_actor_ids() {
        let mut c = externalizing();
        c.actors.push(c.actors[0].clone());
        assert!(matches!(compare_network_outcomes(&objective(), &base(), &c), Err(NetworkComparisonError::DuplicateActorId { side: PolicySide::Candidate, .. })));
    }
    #[test]
    fn rejects_scenario_and_objective_mismatch() {
        let mut c = externalizing();
        c.scenario_id = "S2".into();
        assert!(matches!(compare_network_outcomes(&objective(), &base(), &c), Err(NetworkComparisonError::ScenarioMismatch { .. })));
        c = externalizing();
        c.objective_id = "v2".into();
        assert!(matches!(compare_network_outcomes(&objective(), &base(), &c), Err(NetworkComparisonError::ObjectiveMismatch { .. })));
    }
    #[test]
    fn constrained_selector_prefers_bounded_harm_candidate() {
        let result = select_network_candidate(
            &objective(), &base(), &[externalizing(), bounded_harm()],
            NetworkSelectionConstraints { max_actor_regression_units: 20, minimum_net_system_benefit_units: 1 },
        ).unwrap();
        assert_eq!(result.selected_candidate_index, Some(1));
        assert!(!result.assessments[0].eligible);
        assert!(matches!(result.assessments[0].exclusion, Some(CandidateExclusionReason::ActorHarmCeilingExceeded { .. })));
    }
    #[test]
    fn constrained_selector_returns_none_when_all_candidates_violate_harm_cap() {
        let result = select_network_candidate(
            &objective(), &base(), &[externalizing()],
            NetworkSelectionConstraints { max_actor_regression_units: 0, minimum_net_system_benefit_units: 1 },
        ).unwrap();
        assert_eq!(result.selected_candidate_index, None);
    }
    #[test]
    fn selector_records_invalid_candidates_and_fails_closed_on_invalid_baseline() {
        let mut invalid = externalizing();
        invalid.actors.pop();
        let result = select_network_candidate(
            &objective(), &base(), &[invalid, bounded_harm()],
            NetworkSelectionConstraints { max_actor_regression_units: 20, minimum_net_system_benefit_units: 1 },
        ).unwrap();
        assert_eq!(result.selected_candidate_index, Some(1));
        assert!(matches!(result.assessments[0].exclusion, Some(CandidateExclusionReason::InvalidLedger(_))));

        let mut bad_base = base();
        bad_base.feasibility = FeasibilityStatus::Indeterminate;
        assert!(matches!(
            select_network_candidate(&objective(), &bad_base, &[bounded_harm()],
                NetworkSelectionConstraints { max_actor_regression_units: 20, minimum_net_system_benefit_units: 1 }),
            Err(NetworkComparisonError::OutcomeNotVerified { side: PolicySide::Baseline, .. })
        ));
    }
}
