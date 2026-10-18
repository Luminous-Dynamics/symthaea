//! Scenario-aligned robust Pareto analysis for culinary candidates.
//!
//! Each scenario is a named, versioned set of joint assumptions shared by all
//! candidates in a comparison. The caller supplies a scenario digest; this module
//! records it but does not authenticate the digest or validate the underlying data.
//! A candidate enters a scenario frontier only when all hard gates pass and all
//! objectives for that scenario have finite, provenance-backed values.
//!
//! Robust dominance is stricter: both candidates must be fully rankable in every
//! registered scenario, and the putative dominator must be no worse on every
//! objective in every scenario, with at least one strict improvement somewhere.
//! Missing scenario values are never carried forward from the baseline.

use crate::pareto::{
    evaluate_frontier, CulinaryCandidate, GateDisposition, GateEvidence, ObjectiveDirection,
    ObjectiveMetric, ParetoError, ParetoReport, RankingDisposition,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioSpec {
    /// Stable identifier such as observed_basket_2026-10.
    pub id: String,
    /// Caller-computed digest/version binding the joint scenario assumptions.
    pub digest: String,
    /// Exactly one scenario in a comparison must be the nominal baseline.
    pub is_baseline: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioObjective {
    pub key: String,
    pub direction: ObjectiveDirection,
    /// Explicitly comparable unit, e.g. USD/serving or g_CO2e/serving.
    pub unit: String,
    /// None means missing under this scenario, never "reuse baseline".
    pub value: Option<f64>,
    /// Required when value is Some; a reference is not proof of source truth.
    pub evidence_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateScenarioMetrics {
    pub scenario_id: String,
    pub objectives: Vec<ScenarioObjective>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioCandidate {
    pub id: String,
    pub gates: Vec<GateEvidence>,
    /// Must contain exactly one objective set for every registered scenario.
    pub scenario_metrics: Vec<CandidateScenarioMetrics>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveDefinition {
    pub key: String,
    pub direction: ObjectiveDirection,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioFrontier {
    pub scenario: ScenarioSpec,
    /// Canonical objective schema, including units omitted by the v1 point evaluator.
    pub objective_definitions: Vec<ObjectiveDefinition>,
    pub pareto: ParetoReport,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustCandidateAssessment {
    pub id: String,
    /// True only when gates pass and every objective is present in every scenario.
    pub robustly_rankable: bool,
    pub frontier_scenario_count: usize,
    pub ranked_scenario_count: usize,
    pub dominated_scenario_count: usize,
    pub incomplete_scenario_count: usize,
    pub blocked_scenario_count: usize,
    pub indeterminate_scenario_count: usize,
    pub frontier_in: Vec<String>,
    pub dominated_in: Vec<String>,
    /// Scenario-wide robust dominance relationships, not baseline-only relationships.
    pub robust_dominates: Vec<String>,
    pub robust_dominated_by: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustParetoReport {
    pub baseline_scenario_id: String,
    /// Ordered by scenario ID. Each scenario keeps its complete point-evaluator report.
    pub scenarios: Vec<ScenarioFrontier>,
    /// Ordered by candidate ID.
    pub candidates: Vec<RobustCandidateAssessment>,
    /// Fully evidenced candidates not robustly dominated by another fully evidenced one.
    pub robust_frontier_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RobustParetoError {
    InvalidInput(String),
    Pareto(ParetoError),
}

impl fmt::Display for RobustParetoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "{message}"),
            Self::Pareto(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for RobustParetoError {}

impl From<ParetoError> for RobustParetoError {
    fn from(error: ParetoError) -> Self {
        Self::Pareto(error)
    }
}

/// Evaluate nominal and scenario-specific frontiers plus robust dominance.
///
/// This function validates scenario alignment and objective keys/directions/units.
/// It does not generate plausible scenarios, estimate their probabilities, validate
/// the source evidence, or decide which scenarios stakeholders should require.
pub fn evaluate_robust_frontier(
    scenarios: &[ScenarioSpec],
    candidates: &[ScenarioCandidate],
) -> Result<RobustParetoReport, RobustParetoError> {
    if scenarios.is_empty() {
        return Err(invalid("at least one scenario is required"));
    }
    if candidates.is_empty() {
        return Err(invalid("at least one candidate is required"));
    }

    let mut sorted_scenarios = scenarios.to_vec();
    sorted_scenarios.sort_by(|left, right| left.id.cmp(&right.id));

    let mut scenario_ids = BTreeSet::new();
    for scenario in &sorted_scenarios {
        if scenario.id.trim().is_empty() {
            return Err(invalid("scenario ID must not be empty"));
        }
        if !scenario_ids.insert(scenario.id.clone()) {
            return Err(invalid(format!("duplicate scenario ID: {}", scenario.id)));
        }
        if scenario.digest.trim().is_empty() {
            return Err(invalid(format!("scenario {} has an empty digest", scenario.id)));
        }
    }

    let baseline_ids: Vec<&str> = sorted_scenarios
        .iter()
        .filter(|scenario| scenario.is_baseline)
        .map(|scenario| scenario.id.as_str())
        .collect();
    if baseline_ids.len() != 1 {
        return Err(invalid(format!(
            "exactly one baseline scenario is required; found {}",
            baseline_ids.len()
        )));
    }
    let baseline_scenario_id = baseline_ids[0].to_owned();
    let expected_scenario_ids: BTreeSet<String> =
        sorted_scenarios.iter().map(|scenario| scenario.id.clone()).collect();

    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|left, right| left.id.cmp(&right.id));
    let mut candidate_ids = BTreeSet::new();

    for candidate in &mut sorted_candidates {
        if candidate.id.trim().is_empty() {
            return Err(invalid("candidate ID must not be empty"));
        }
        if !candidate_ids.insert(candidate.id.clone()) {
            return Err(invalid(format!("duplicate candidate ID: {}", candidate.id)));
        }

        candidate
            .scenario_metrics
            .sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
        let mut found_scenarios = BTreeSet::new();

        for metrics in &candidate.scenario_metrics {
            if !expected_scenario_ids.contains(&metrics.scenario_id) {
                return Err(invalid(format!(
                    "candidate {} references unknown scenario {}",
                    candidate.id, metrics.scenario_id
                )));
            }
            if !found_scenarios.insert(metrics.scenario_id.clone()) {
                return Err(invalid(format!(
                    "candidate {} has duplicate metrics for scenario {}",
                    candidate.id, metrics.scenario_id
                )));
            }
        }

        if found_scenarios != expected_scenario_ids {
            let missing: Vec<String> = expected_scenario_ids
                .difference(&found_scenarios)
                .cloned()
                .collect();
            return Err(invalid(format!(
                "candidate {} is missing required scenario metrics: {}",
                candidate.id,
                missing.join(", ")
            )));
        }
    }

    let mut canonical_definitions: Option<Vec<ObjectiveDefinition>> = None;
    for candidate in &sorted_candidates {
        for metrics in &candidate.scenario_metrics {
            let definitions = validate_objectives(&candidate.id, metrics)?;
            match &canonical_definitions {
                None => canonical_definitions = Some(definitions),
                Some(canonical) if *canonical == definitions => {}
                Some(_) => {
                    return Err(invalid(format!(
                        "candidate {} has objective key/direction/unit mismatch in scenario {}",
                        candidate.id, metrics.scenario_id
                    )));
                }
            }
        }
    }
    let objective_definitions = canonical_definitions
        .ok_or_else(|| invalid("at least one objective definition is required"))?;

    let mut scenario_reports = Vec::with_capacity(sorted_scenarios.len());
    for scenario in &sorted_scenarios {
        let point_candidates: Vec<CulinaryCandidate> = sorted_candidates
            .iter()
            .map(|candidate| {
                let metrics = candidate
                    .scenario_metrics
                    .iter()
                    .find(|metrics| metrics.scenario_id == scenario.id)
                    .expect("scenario coverage was validated");
                CulinaryCandidate {
                    id: candidate.id.clone(),
                    gates: candidate.gates.clone(),
                    objectives: metrics
                        .objectives
                        .iter()
                        .map(|objective| ObjectiveMetric {
                            key: objective.key.clone(),
                            direction: objective.direction,
                            value: objective.value,
                            evidence_ref: objective.evidence_ref.clone(),
                        })
                        .collect(),
                }
            })
            .collect();

        let pareto = evaluate_frontier(&point_candidates)?;
        scenario_reports.push(ScenarioFrontier {
            scenario: scenario.clone(),
            objective_definitions: objective_definitions.clone(),
            pareto,
        });
    }

    let candidate_lookup: BTreeMap<&str, &ScenarioCandidate> = sorted_candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect();

    let mut assessments = Vec::with_capacity(sorted_candidates.len());
    for candidate in &sorted_candidates {
        let mut frontier_in = Vec::new();
        let mut dominated_in = Vec::new();
        let mut ranked_scenario_count = 0;
        let mut dominated_scenario_count = 0;
        let mut incomplete_scenario_count = 0;
        let mut blocked_scenario_count = 0;
        let mut indeterminate_scenario_count = 0;

        for scenario_report in &scenario_reports {
            let assessment = scenario_report
                .pareto
                .candidates
                .iter()
                .find(|assessment| assessment.id == candidate.id)
                .expect("the point evaluator reports every validated candidate");

            if assessment.on_frontier {
                frontier_in.push(scenario_report.scenario.id.clone());
            }
            if assessment.ranking_disposition == RankingDisposition::Ranked {
                ranked_scenario_count += 1;
                if !assessment.dominated_by.is_empty() {
                    dominated_scenario_count += 1;
                    dominated_in.push(scenario_report.scenario.id.clone());
                }
            }
            if assessment.ranking_disposition == RankingDisposition::IncompleteMetrics {
                incomplete_scenario_count += 1;
            }
            if assessment.gate_disposition == GateDisposition::Blocked {
                blocked_scenario_count += 1;
            }
            if assessment.gate_disposition == GateDisposition::Indeterminate {
                indeterminate_scenario_count += 1;
            }
        }

        assessments.push(RobustCandidateAssessment {
            id: candidate.id.clone(),
            robustly_rankable: ranked_scenario_count == scenario_reports.len(),
            frontier_scenario_count: frontier_in.len(),
            ranked_scenario_count,
            dominated_scenario_count,
            incomplete_scenario_count,
            blocked_scenario_count,
            indeterminate_scenario_count,
            frontier_in,
            dominated_in,
            robust_dominates: Vec::new(),
            robust_dominated_by: Vec::new(),
        });
    }

    for left_index in 0..sorted_candidates.len() {
        if !assessments[left_index].robustly_rankable {
            continue;
        }
        for right_index in (left_index + 1)..sorted_candidates.len() {
            if !assessments[right_index].robustly_rankable {
                continue;
            }
            let left = candidate_lookup[sorted_candidates[left_index].id.as_str()];
            let right = candidate_lookup[sorted_candidates[right_index].id.as_str()];

            if robustly_dominates(left, right, &sorted_scenarios) {
                assessments[left_index]
                    .robust_dominates
                    .push(right.id.clone());
                assessments[right_index]
                    .robust_dominated_by
                    .push(left.id.clone());
            } else if robustly_dominates(right, left, &sorted_scenarios) {
                assessments[right_index]
                    .robust_dominates
                    .push(left.id.clone());
                assessments[left_index]
                    .robust_dominated_by
                    .push(right.id.clone());
            }
        }
    }

    for assessment in &mut assessments {
        assessment.robust_dominates.sort();
        assessment.robust_dominated_by.sort();
    }

    let robust_frontier_ids = assessments
        .iter()
        .filter(|assessment| {
            assessment.robustly_rankable && assessment.robust_dominated_by.is_empty()
        })
        .map(|assessment| assessment.id.clone())
        .collect();

    Ok(RobustParetoReport {
        baseline_scenario_id,
        scenarios: scenario_reports,
        candidates: assessments,
        robust_frontier_ids,
    })
}

fn validate_objectives(
    candidate_id: &str,
    metrics: &CandidateScenarioMetrics,
) -> Result<Vec<ObjectiveDefinition>, RobustParetoError> {
    if metrics.objectives.is_empty() {
        return Err(invalid(format!(
            "candidate {candidate_id} has no objectives in scenario {}",
            metrics.scenario_id
        )));
    }

    let mut definitions = Vec::with_capacity(metrics.objectives.len());
    let mut keys = BTreeSet::new();

    for objective in &metrics.objectives {
        if objective.key.trim().is_empty() {
            return Err(invalid(format!(
                "candidate {candidate_id} has an empty objective key in scenario {}",
                metrics.scenario_id
            )));
        }
        if !keys.insert(objective.key.clone()) {
            return Err(invalid(format!(
                "candidate {candidate_id} repeats objective {} in scenario {}",
                objective.key, metrics.scenario_id
            )));
        }
        if objective.unit.trim().is_empty() {
            return Err(invalid(format!(
                "objective {} for candidate {candidate_id} has an empty unit",
                objective.key
            )));
        }
        if let Some(value) = objective.value {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "objective {} for candidate {candidate_id} is non-finite",
                    objective.key
                )));
            }
            match objective.evidence_ref.as_deref() {
                None => {
                    return Err(invalid(format!(
                        "objective {} for candidate {candidate_id} has no evidence reference",
                        objective.key
                    )));
                }
                Some(reference) if reference.trim().is_empty() => {
                    return Err(invalid(format!(
                        "objective {} for candidate {candidate_id} has an empty evidence reference",
                        objective.key
                    )));
                }
                Some(_) => {}
            }
        }

        definitions.push(ObjectiveDefinition {
            key: objective.key.clone(),
            direction: objective.direction,
            unit: objective.unit.clone(),
        });
    }

    definitions.sort_by(|left, right| left.key.cmp(&right.key));
    Ok(definitions)
}

fn robustly_dominates(
    left: &ScenarioCandidate,
    right: &ScenarioCandidate,
    scenarios: &[ScenarioSpec],
) -> bool {
    let mut strictly_better_somewhere = false;

    for scenario in scenarios {
        let Some(left_metrics) = left
            .scenario_metrics
            .iter()
            .find(|metrics| metrics.scenario_id == scenario.id)
        else {
            return false;
        };
        let Some(right_metrics) = right
            .scenario_metrics
            .iter()
            .find(|metrics| metrics.scenario_id == scenario.id)
        else {
            return false;
        };

        for left_objective in &left_metrics.objectives {
            let Some(right_objective) = right_metrics
                .objectives
                .iter()
                .find(|objective| objective.key == left_objective.key)
            else {
                return false;
            };
            let (Some(left_value), Some(right_value)) =
                (left_objective.value, right_objective.value)
            else {
                return false;
            };

            let (no_worse, strictly_better) = match left_objective.direction {
                ObjectiveDirection::Minimize => (
                    left_value <= right_value,
                    left_value < right_value,
                ),
                ObjectiveDirection::Maximize => (
                    left_value >= right_value,
                    left_value > right_value,
                ),
            };
            if !no_worse {
                return false;
            }
            strictly_better_somewhere |= strictly_better;
        }
    }

    strictly_better_somewhere
}

fn invalid(message: impl Into<String>) -> RobustParetoError {
    RobustParetoError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenarios() -> Vec<ScenarioSpec> {
        vec![
            ScenarioSpec {
                id: "observed_basket".to_owned(),
                digest: "sha256:observed-basket-v1".to_owned(),
                is_baseline: true,
            },
            ScenarioSpec {
                id: "seasonal_price_shock".to_owned(),
                digest: "sha256:seasonal-shock-v1".to_owned(),
                is_baseline: false,
            },
        ]
    }

    fn candidate(
        id: &str,
        observed_cost: Option<f64>,
        shock_cost: Option<f64>,
    ) -> ScenarioCandidate {
        ScenarioCandidate {
            id: id.to_owned(),
            gates: vec![GateEvidence::pass(
                "allergen_reviewed",
                format!("gate:{id}:allergens:v1"),
            )],
            scenario_metrics: vec![
                CandidateScenarioMetrics {
                    scenario_id: "observed_basket".to_owned(),
                    objectives: vec![ScenarioObjective {
                        key: "cost_per_serving".to_owned(),
                        direction: ObjectiveDirection::Minimize,
                        unit: "USD/serving".to_owned(),
                        value: observed_cost,
                        evidence_ref: observed_cost.map(|_| format!("price:{id}:observed:v1")),
                    }],
                },
                CandidateScenarioMetrics {
                    scenario_id: "seasonal_price_shock".to_owned(),
                    objectives: vec![ScenarioObjective {
                        key: "cost_per_serving".to_owned(),
                        direction: ObjectiveDirection::Minimize,
                        unit: "USD/serving".to_owned(),
                        value: shock_cost,
                        evidence_ref: shock_cost.map(|_| format!("price:{id}:shock:v1")),
                    }],
                },
            ],
        }
    }

    #[test]
    fn reports_baseline_and_shock_frontiers_when_dominance_reverses() {
        let report = evaluate_robust_frontier(
            &scenarios(),
            &[candidate("a", Some(5.0), Some(8.0)), candidate("b", Some(6.0), Some(6.0))],
        )
        .unwrap();

        let baseline = report
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario.is_baseline)
            .unwrap();
        let shock = report
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario.id == "seasonal_price_shock")
            .unwrap();

        assert_eq!(baseline.pareto.frontier_ids, vec!["a"]);
        assert_eq!(shock.pareto.frontier_ids, vec!["b"]);
        assert_eq!(report.robust_frontier_ids, vec!["a", "b"]);
        assert_eq!(
            report.candidates.iter().find(|item| item.id == "a").unwrap().frontier_scenario_count,
            1
        );
        assert_eq!(
            report.candidates.iter().find(|item| item.id == "b").unwrap().frontier_scenario_count,
            1
        );
    }

    #[test]
    fn missing_scenario_value_is_not_carried_forward_from_baseline() {
        let report = evaluate_robust_frontier(
            &scenarios(),
            &[candidate("complete", Some(5.0), Some(8.0)), candidate("missing", Some(6.0), None)],
        )
        .unwrap();

        let missing = report.candidates.iter().find(|item| item.id == "missing").unwrap();
        assert!(!missing.robustly_rankable);
        assert_eq!(missing.incomplete_scenario_count, 1);
        assert!(!report.robust_frontier_ids.contains(&"missing".to_owned()));
    }

    #[test]
    fn robust_dominance_requires_no_worse_values_in_every_scenario() {
        let report = evaluate_robust_frontier(
            &scenarios(),
            &[candidate("stable_winner", Some(5.0), Some(5.0)), candidate("worse", Some(6.0), Some(7.0))],
        )
        .unwrap();

        assert_eq!(report.robust_frontier_ids, vec!["stable_winner"]);
        let winner = report.candidates.iter().find(|item| item.id == "stable_winner").unwrap();
        assert_eq!(winner.robust_dominates, vec!["worse"]);
    }

    #[test]
    fn units_must_match_across_scenarios_and_candidates() {
        let mut invalid_candidate = candidate("b", Some(6.0), Some(6.0));
        invalid_candidate.scenario_metrics[1].objectives[0].unit = "cents/serving".to_owned();

        let result = evaluate_robust_frontier(
            &scenarios(),
            &[candidate("a", Some(5.0), Some(5.0)), invalid_candidate],
        );

        assert!(matches!(result, Err(RobustParetoError::InvalidInput(_))));
    }

    #[test]
    fn blocked_candidates_never_enter_any_frontier() {
        let mut blocked = candidate("blocked", Some(0.0), Some(0.0));
        blocked.gates[0] =
            GateEvidence::fail("allergen_reviewed", "gate:known-allergen-conflict");

        let report = evaluate_robust_frontier(
            &scenarios(),
            &[candidate("eligible", Some(5.0), Some(5.0)), blocked],
        )
        .unwrap();

        assert!(report
            .scenarios
            .iter()
            .all(|scenario| !scenario.pareto.frontier_ids.contains(&"blocked".to_owned())));
        assert!(!report.robust_frontier_ids.contains(&"blocked".to_owned()));
    }

    #[test]
    fn candidate_and_scenario_input_order_do_not_change_report() {
        let candidates = vec![
            candidate("z", Some(9.0), Some(9.0)),
            candidate("a", Some(5.0), Some(5.0)),
        ];
        let forward = evaluate_robust_frontier(&scenarios(), &candidates).unwrap();
        let mut reversed_scenarios = scenarios();
        reversed_scenarios.reverse();
        let reverse = evaluate_robust_frontier(
            &reversed_scenarios,
            &candidates.iter().cloned().rev().collect::<Vec<_>>(),
        )
        .unwrap();

        assert_eq!(forward, reverse);
    }

    #[test]
    fn rejects_candidates_missing_a_registered_scenario() {
        let mut incomplete = candidate("incomplete", Some(5.0), Some(5.0));
        incomplete.scenario_metrics.pop();

        assert!(matches!(
            evaluate_robust_frontier(&scenarios(), &[incomplete]),
            Err(RobustParetoError::InvalidInput(_))
        ));
    }
}
