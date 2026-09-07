// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-stage shadow benchmark protocol for resource-planning selectors.
//!
//! Stage 1 runs selectors without any outcome input. Stage 2 later joins a complete
//! outcome set for every feasible candidate and evaluates those decisions under an
//! exact, independently declared metric schema. This API separation prevents
//! realized/holdout outcomes from leaking into baseline, Pareto, or HDC selection.
//!
//! Evaluation reports exact outcome vectors and Pareto ranks only. It does not
//! collapse outcomes into a scalar winner or make a superiority claim.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_operations_research::{
    rank_pareto, ObjectiveDirection, ParetoCandidate, ParetoError, ParetoObjective,
    ParetoRanking,
};
use symthaea_resource_feasible_set::DeterministicFeasibleSet;
use symthaea_resource_hdc_shadow::{
    plan_hdc_shadow, HdcPreferenceProfile, HdcShadowError, HdcShadowPlan,
};
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricError, ObjectiveMetricSchema, ObjectiveMetricSchemaError,
};
use symthaea_resource_objective_normalization::ObjectiveNormalizationProfile;
use symthaea_resource_pareto::ResourceParetoRanking;
use symthaea_resource_recommendation::{
    RecommendationError, ValidatedResourcePlannerDecision,
};
use symthaea_resource_selectors::{
    validated_canonical_feasible, validated_pareto_canonical, validated_pareto_unique,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SelectorKind {
    CanonicalFeasible,
    ParetoUniqueFrontier,
    ParetoCanonicalFrontier,
    HdcShadow,
}

/// Stage-1 result. No outcome data is accepted by this constructor.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowDecisionSet {
    ranking_context: ResourceParetoRanking,
    decisions: BTreeMap<SelectorKind, ValidatedResourcePlannerDecision>,
    hdc_plan: HdcShadowPlan,
}

impl ShadowDecisionSet {
    pub fn ranking_context(&self) -> &ResourceParetoRanking {
        &self.ranking_context
    }

    pub fn decision(&self, selector: SelectorKind) -> Option<&ValidatedResourcePlannerDecision> {
        self.decisions.get(&selector)
    }

    pub fn decisions(
        &self,
    ) -> impl Iterator<Item = (SelectorKind, &ValidatedResourcePlannerDecision)> {
        self.decisions.iter().map(|(kind, decision)| (*kind, decision))
    }

    pub fn hdc_plan(&self) -> &HdcShadowPlan {
        &self.hdc_plan
    }
}

/// Run all benchmark selectors against exactly the same planning context.
///
/// Outcome/realization data is intentionally absent from this API.
pub fn run_shadow_decisions(
    ranking: &ResourceParetoRanking,
    normalization: &ObjectiveNormalizationProfile,
    hdc_preferences: &HdcPreferenceProfile,
) -> Result<ShadowDecisionSet, ShadowDecisionError> {
    let feasible_set = ranking.evidence_set().feasible_set();
    let baseline = validated_canonical_feasible(feasible_set)
        .map_err(ShadowDecisionError::Recommendation)?;
    let pareto_unique = validated_pareto_unique(ranking)
        .map_err(ShadowDecisionError::Recommendation)?;
    let pareto_canonical = validated_pareto_canonical(ranking)
        .map_err(ShadowDecisionError::Recommendation)?;
    let hdc_plan = plan_hdc_shadow(ranking, normalization, hdc_preferences)
        .map_err(ShadowDecisionError::Hdc)?;

    let mut decisions = BTreeMap::new();
    decisions.insert(SelectorKind::CanonicalFeasible, baseline);
    decisions.insert(SelectorKind::ParetoUniqueFrontier, pareto_unique);
    decisions.insert(SelectorKind::ParetoCanonicalFrontier, pareto_canonical);
    decisions.insert(
        SelectorKind::HdcShadow,
        hdc_plan.validated_decision().clone(),
    );

    Ok(ShadowDecisionSet {
        ranking_context: ranking.clone(),
        decisions,
        hdc_plan,
    })
}

#[derive(Debug, Error)]
pub enum ShadowDecisionError {
    #[error("control recommendation validation failed: {0}")]
    Recommendation(#[source] RecommendationError),
    #[error("HDC shadow planning failed: {0}")]
    Hdc(#[source] HdcShadowError),
}

/// Descriptive provenance of one benchmark outcome scalar. No ordering is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OutcomeEvidenceClass {
    Simulated,
    Observed,
    Attested,
}

/// One exact candidate-level evaluation outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateOutcomeEvidence {
    pub candidate_id: String,
    pub metric: ObjectiveMetric,
    pub value: f64,
    pub evidence_class: OutcomeEvidenceClass,
    pub evidence_ref: String,
}

/// Outcome evidence bound to one exact deterministic feasible state.
///
/// Multiple exact metrics may share a display name, but they remain distinct keys.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceOutcomeSet {
    feasible_set: DeterministicFeasibleSet,
    outcomes: BTreeMap<String, BTreeMap<ObjectiveMetric, CandidateOutcomeEvidence>>,
}

impl ResourceOutcomeSet {
    pub fn new(feasible_set: DeterministicFeasibleSet) -> Self {
        Self {
            feasible_set,
            outcomes: BTreeMap::new(),
        }
    }

    pub fn feasible_set(&self) -> &DeterministicFeasibleSet {
        &self.feasible_set
    }

    pub fn insert(
        &mut self,
        outcome: CandidateOutcomeEvidence,
    ) -> Result<(), OutcomeEvidenceError> {
        validate_outcome(&self.feasible_set, &outcome)?;
        let candidate = self
            .outcomes
            .entry(outcome.candidate_id.clone())
            .or_default();
        if candidate.contains_key(&outcome.metric) {
            return Err(OutcomeEvidenceError::DuplicateExactMetric {
                candidate_id: outcome.candidate_id,
                metric_id: outcome.metric.metric_id,
            });
        }
        candidate.insert(outcome.metric.clone(), outcome);
        Ok(())
    }

    pub fn get(
        &self,
        candidate_id: &str,
        metric: &ObjectiveMetric,
    ) -> Option<&CandidateOutcomeEvidence> {
        self.outcomes.get(candidate_id)?.get(metric)
    }

    pub fn outcomes_for(
        &self,
        candidate_id: &str,
    ) -> impl Iterator<Item = &CandidateOutcomeEvidence> {
        self.outcomes
            .get(candidate_id)
            .into_iter()
            .flat_map(|metrics| metrics.values())
    }
}

fn validate_outcome(
    feasible_set: &DeterministicFeasibleSet,
    outcome: &CandidateOutcomeEvidence,
) -> Result<(), OutcomeEvidenceError> {
    if outcome.candidate_id.trim().is_empty() {
        return Err(OutcomeEvidenceError::EmptyCandidateId);
    }
    outcome
        .metric
        .validate()
        .map_err(OutcomeEvidenceError::InvalidMetric)?;
    if !outcome.value.is_finite() {
        return Err(OutcomeEvidenceError::NonFiniteValue {
            metric_id: outcome.metric.metric_id.clone(),
            value: outcome.value,
        });
    }
    if outcome.evidence_ref.trim().is_empty() {
        return Err(OutcomeEvidenceError::EmptyEvidenceRef);
    }
    if feasible_set
        .feasible_candidate(&outcome.candidate_id)
        .is_some()
    {
        return Ok(());
    }
    if feasible_set.rejection(&outcome.candidate_id).is_some() {
        return Err(OutcomeEvidenceError::RejectedCandidate(
            outcome.candidate_id.clone(),
        ));
    }
    Err(OutcomeEvidenceError::CandidateOutsideUniverse(
        outcome.candidate_id.clone(),
    ))
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum OutcomeEvidenceError {
    #[error("outcome candidate id must not be empty")]
    EmptyCandidateId,
    #[error("invalid outcome metric: {0}")]
    InvalidMetric(ObjectiveMetricError),
    #[error("outcome metric {metric_id} has non-finite value {value}")]
    NonFiniteValue { metric_id: String, value: f64 },
    #[error("outcome evidence ref must not be empty")]
    EmptyEvidenceRef,
    #[error("candidate {0} exists in the universe but failed feasibility")]
    RejectedCandidate(String),
    #[error("candidate {0} is outside the retained candidate universe")]
    CandidateOutsideUniverse(String),
    #[error("duplicate exact outcome metric {metric_id} for candidate {candidate_id}")]
    DuplicateExactMetric {
        candidate_id: String,
        metric_id: String,
    },
}

/// Exact evaluation metric plus preferred outcome direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationObjective {
    pub metric: ObjectiveMetric,
    pub direction: ObjectiveDirection,
}

impl EvaluationObjective {
    pub fn new(metric: ObjectiveMetric, direction: ObjectiveDirection) -> Self {
        Self { metric, direction }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluatedSelectorDecision {
    Recommended {
        selector_id: String,
        candidate_id: String,
        outcome_values: BTreeMap<String, CandidateOutcomeEvidence>,
        outcome_pareto_rank: usize,
    },
    Abstained {
        selector_id: String,
    },
}

/// Stage-2 benchmark result. It reports raw exact outcome evidence and Pareto rank;
/// it deliberately exposes no scalar winner or aggregate superiority score.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowBenchmarkEvaluation {
    decisions: ShadowDecisionSet,
    outcomes: ResourceOutcomeSet,
    evaluation_schema: ObjectiveMetricSchema,
    evaluation_objectives: Vec<EvaluationObjective>,
    outcome_ranking: ParetoRanking,
    evaluated_decisions: BTreeMap<SelectorKind, EvaluatedSelectorDecision>,
}

impl ShadowBenchmarkEvaluation {
    pub fn decisions(&self) -> &ShadowDecisionSet {
        &self.decisions
    }

    pub fn outcomes(&self) -> &ResourceOutcomeSet {
        &self.outcomes
    }

    pub fn evaluation_schema(&self) -> &ObjectiveMetricSchema {
        &self.evaluation_schema
    }

    pub fn evaluation_objectives(&self) -> &[EvaluationObjective] {
        &self.evaluation_objectives
    }

    pub fn outcome_ranking(&self) -> &ParetoRanking {
        &self.outcome_ranking
    }

    pub fn evaluated_decision(
        &self,
        selector: SelectorKind,
    ) -> Option<&EvaluatedSelectorDecision> {
        self.evaluated_decisions.get(&selector)
    }

    pub fn outcome_frontier_ids(&self) -> Vec<&str> {
        self.outcome_ranking
            .first_front()
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect()
    }
}

/// Evaluate already-frozen selector decisions against complete candidate outcomes.
pub fn evaluate_shadow_decisions(
    decisions: &ShadowDecisionSet,
    outcomes: &ResourceOutcomeSet,
    objectives: &[EvaluationObjective],
) -> Result<ShadowBenchmarkEvaluation, BenchmarkEvaluationError> {
    let planning_feasible = decisions.ranking_context.evidence_set().feasible_set();
    if outcomes.feasible_set() != planning_feasible {
        return Err(BenchmarkEvaluationError::PlanningStateMismatch);
    }
    if objectives.is_empty() {
        return Err(BenchmarkEvaluationError::NoEvaluationObjectives);
    }

    let evaluation_schema = ObjectiveMetricSchema::new(
        objectives.iter().map(|objective| objective.metric.clone()),
    )
    .map_err(BenchmarkEvaluationError::InvalidEvaluationSchema)?;

    let pareto_objectives: Vec<ParetoObjective> = objectives
        .iter()
        .map(|objective| {
            ParetoObjective::new(
                objective.metric.objective_name.clone(),
                objective.direction,
            )
        })
        .collect();

    // Complete candidate outcome coverage is required before any selector is
    // evaluated, including outcomes for alternatives no selector happened to pick.
    let mut outcome_vectors: BTreeMap<String, BTreeMap<String, CandidateOutcomeEvidence>> =
        BTreeMap::new();
    let mut pareto_candidates = Vec::new();
    for feasible in planning_feasible.feasible() {
        let candidate_id = feasible.candidate_id();
        let mut values = Vec::with_capacity(objectives.len());
        let mut exact = BTreeMap::new();
        for objective in objectives {
            let evidence = outcomes
                .get(candidate_id, &objective.metric)
                .ok_or_else(|| BenchmarkEvaluationError::MissingOutcome {
                    candidate_id: candidate_id.to_owned(),
                    metric_id: objective.metric.metric_id.clone(),
                })?
                .clone();
            values.push(evidence.value);
            exact.insert(objective.metric.objective_name.clone(), evidence);
        }
        outcome_vectors.insert(candidate_id.to_owned(), exact);
        pareto_candidates.push(ParetoCandidate::new(candidate_id, values));
    }

    let outcome_ranking = rank_pareto(&pareto_objectives, &pareto_candidates)
        .map_err(BenchmarkEvaluationError::Pareto)?;

    let mut evaluated_decisions = BTreeMap::new();
    for (kind, decision) in decisions.decisions() {
        let evaluated = evaluate_one_decision(decision, &outcome_vectors, &outcome_ranking)?;
        evaluated_decisions.insert(kind, evaluated);
    }

    Ok(ShadowBenchmarkEvaluation {
        decisions: decisions.clone(),
        outcomes: outcomes.clone(),
        evaluation_schema,
        evaluation_objectives: objectives.to_vec(),
        outcome_ranking,
        evaluated_decisions,
    })
}

fn evaluate_one_decision(
    decision: &ValidatedResourcePlannerDecision,
    outcome_vectors: &BTreeMap<String, BTreeMap<String, CandidateOutcomeEvidence>>,
    ranking: &ParetoRanking,
) -> Result<EvaluatedSelectorDecision, BenchmarkEvaluationError> {
    match decision {
        ValidatedResourcePlannerDecision::Recommend(recommendation) => {
            let candidate_id = recommendation.candidate_id();
            let outcome_values = outcome_vectors
                .get(candidate_id)
                .cloned()
                .ok_or_else(|| BenchmarkEvaluationError::MissingCandidateOutcomeVector(
                    candidate_id.to_owned(),
                ))?;
            let outcome_pareto_rank = ranking
                .rank_of(candidate_id)
                .ok_or_else(|| BenchmarkEvaluationError::CandidateMissingFromOutcomeRanking(
                    candidate_id.to_owned(),
                ))?;
            Ok(EvaluatedSelectorDecision::Recommended {
                selector_id: recommendation.selector_id().to_owned(),
                candidate_id: candidate_id.to_owned(),
                outcome_values,
                outcome_pareto_rank,
            })
        }
        ValidatedResourcePlannerDecision::Abstain(abstention) => {
            Ok(EvaluatedSelectorDecision::Abstained {
                selector_id: abstention.selector_id().to_owned(),
            })
        }
    }
}

#[derive(Debug, Error)]
pub enum BenchmarkEvaluationError {
    #[error("outcome set is bound to a different deterministic feasible state")]
    PlanningStateMismatch,
    #[error("at least one evaluation objective is required")]
    NoEvaluationObjectives,
    #[error("invalid evaluation metric schema: {0}")]
    InvalidEvaluationSchema(ObjectiveMetricSchemaError),
    #[error("missing outcome for {candidate_id}/{metric_id}")]
    MissingOutcome {
        candidate_id: String,
        metric_id: String,
    },
    #[error("outcome Pareto ranking failed: {0}")]
    Pareto(#[source] ParetoError),
    #[error("missing complete outcome vector for selected candidate {0}")]
    MissingCandidateOutcomeVector(String),
    #[error("selected candidate {0} is missing from complete outcome ranking")]
    CandidateMissingFromOutcomeRanking(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::enumerate_feasible_set;
    use symthaea_resource_hdc_shadow::{HdcObjectivePreference, HdcPreferenceProfile};
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_objective_evidence::{
        CandidateObjectiveEvidence, ObjectiveEvidenceClass, ObjectiveEvidenceScope,
        ResourceObjectiveEvidenceSet,
    };
    use symthaea_resource_objective_metric::ObjectiveStatistic;
    use symthaea_resource_objective_normalization::{
        ObjectiveNormalizationBand, OutOfRangePolicy,
    };
    use symthaea_resource_objective_qualification::{
        ObjectiveEvidenceIdentityPolicy, ObjectiveMultipleEvidenceRule,
        ObjectiveQualificationPolicy, ObjectiveQualificationRequest,
    };
    use symthaea_resource_pareto::{rank_feasible_resources, ResourceParetoObjective};
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
        ResourceQualityRequirement,
    };
    use symthaea_resource_quality_evidence::{
        QualityEvidenceClass, QualityEvidenceSet, QualityEvidenceWindow, QualitySubject,
    };
    use symthaea_resource_quality_qualification::{
        EvidenceIdentityPolicy, MultipleEvidenceRule, QualityQualificationPolicy,
    };
    use symthaea_resource_topology::{ResourceLink, ResourceTopology};

    fn t0() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn resilience() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "resilience",
            "resilience.fraction.v1",
            "ratio.fraction.v1",
            ObjectiveStatistic::CandidateFraction,
        )
        .unwrap()
    }

    fn realized_cost() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "realized_cost",
            "cost.realized.usd_micro.v1",
            "currency.usd_micro.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn planning_context() -> (
        ResourceParetoRanking,
        ObjectiveNormalizationProfile,
        HdcPreferenceProfile,
    ) {
        let candidates = [("a", 20.0, 0.6), ("b", 60.0, 0.95)];
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [ResourcePort {
                    id: "out".into(),
                    direction: PortDirection::Output,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [ResourcePort {
                    id: "in".into(),
                    direction: PortDirection::Input,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();

        let mut capacities = CapacitySchedule::default();
        for (id, subject) in [
            (
                "source-cap",
                CapacitySubject::Port {
                    node_id: "source".into(),
                    port_id: "out".into(),
                },
            ),
            (
                "link-cap",
                CapacitySubject::Link {
                    link_id: "line".into(),
                },
            ),
            (
                "sink-cap",
                CapacitySubject::Port {
                    node_id: "sink".into(),
                    port_id: "in".into(),
                },
            ),
        ] {
            capacities
                .add_window(
                    &topology,
                    CapacityWindow {
                        id: id.into(),
                        subject,
                        valid_from: t0(),
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }

        let mut quality_profile = ResourceQualityProfile::new(power(1.0).key);
        quality_profile
            .set_numeric(QualityMetric::TemperatureCelsius, 40.0)
            .unwrap();
        let mut quality_evidence = QualityEvidenceSet::default();
        quality_evidence
            .insert(
                &topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link {
                        link_id: "line".into(),
                    },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile: quality_profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        let mut quality_requirement = ResourceQualityRequirement::new(power(1.0).key);
        quality_requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(20.0),
                maximum: Some(60.0),
            })
            .unwrap();
        let quality_policy = QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        let feasible = enumerate_feasible_set(
            &topology,
            &capacities,
            &AllocationBook::default(),
            &quality_evidence,
            &quality_requirement,
            &quality_policy,
            candidates
                .iter()
                .enumerate()
                .map(|(index, (id, _, _))| PlannedAllocation {
                    id: (*id).into(),
                    link_id: "line".into(),
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    sent: power(20.0 + index as f64),
                })
                .collect(),
            8,
        )
        .unwrap();

        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible);
        for (candidate_id, energy_value, resilience_value) in candidates {
            for (suffix, metric, value) in [
                ("energy", energy(), energy_value),
                ("resilience", resilience(), resilience_value),
            ] {
                evidence
                    .insert(CandidateObjectiveEvidence {
                        id: format!("{candidate_id}-{suffix}"),
                        candidate_id: candidate_id.into(),
                        metric,
                        value,
                        evidence_class: ObjectiveEvidenceClass::Observed,
                        evidence_ref: format!("planner:{candidate_id}:{suffix}"),
                        scope: ObjectiveEvidenceScope::CandidateAggregate,
                    })
                    .unwrap();
            }
        }

        let objective_policy = ObjectiveQualificationPolicy {
            allowed_classes: [ObjectiveEvidenceClass::Observed].into_iter().collect(),
            identity_policy: ObjectiveEvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: ObjectiveMultipleEvidenceRule::RejectMultiple,
        };
        let ranking = rank_feasible_resources(
            &evidence,
            &[
                ResourceParetoObjective::new(energy(), ObjectiveDirection::Minimize),
                ResourceParetoObjective::new(resilience(), ObjectiveDirection::Maximize),
            ],
            &[
                ObjectiveQualificationRequest {
                    metric: energy(),
                    policy: objective_policy.clone(),
                },
                ObjectiveQualificationRequest {
                    metric: resilience(),
                    policy: objective_policy,
                },
            ],
        )
        .unwrap();

        let normalization = ObjectiveNormalizationProfile::new([
            ObjectiveNormalizationBand {
                metric: energy(),
                direction: ObjectiveDirection::Minimize,
                ideal: 0.0,
                worst: 100.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
            ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 1.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
        ])
        .unwrap();
        let preferences = HdcPreferenceProfile::new([
            HdcObjectivePreference {
                metric: energy(),
                weight: 1,
            },
            HdcObjectivePreference {
                metric: resilience(),
                weight: 1,
            },
        ])
        .unwrap();
        (ranking, normalization, preferences)
    }

    fn decisions() -> ShadowDecisionSet {
        let (ranking, normalization, preferences) = planning_context();
        run_shadow_decisions(&ranking, &normalization, &preferences).unwrap()
    }

    fn outcomes(
        decisions: &ShadowDecisionSet,
        include_b: bool,
    ) -> ResourceOutcomeSet {
        let feasible = decisions
            .ranking_context()
            .evidence_set()
            .feasible_set()
            .clone();
        let mut outcomes = ResourceOutcomeSet::new(feasible);
        outcomes
            .insert(CandidateOutcomeEvidence {
                candidate_id: "a".into(),
                metric: realized_cost(),
                value: 5_000_000.0,
                evidence_class: OutcomeEvidenceClass::Simulated,
                evidence_ref: "holdout:a:cost".into(),
            })
            .unwrap();
        if include_b {
            outcomes
                .insert(CandidateOutcomeEvidence {
                    candidate_id: "b".into(),
                    metric: realized_cost(),
                    value: 1_000_000.0,
                    evidence_class: OutcomeEvidenceClass::Simulated,
                    evidence_ref: "holdout:b:cost".into(),
                })
                .unwrap();
        }
        outcomes
    }

    #[test]
    fn selector_stage_contains_all_controls_without_outcome_input() {
        let decisions = decisions();
        assert!(decisions.decision(SelectorKind::CanonicalFeasible).is_some());
        assert!(decisions.decision(SelectorKind::ParetoUniqueFrontier).is_some());
        assert!(decisions.decision(SelectorKind::ParetoCanonicalFrontier).is_some());
        assert!(decisions.decision(SelectorKind::HdcShadow).is_some());
    }

    #[test]
    fn incomplete_counterfactual_outcomes_fail_complete_evaluation() {
        let decisions = decisions();
        let incomplete = outcomes(&decisions, false);
        assert!(matches!(
            evaluate_shadow_decisions(
                &decisions,
                &incomplete,
                &[EvaluationObjective::new(realized_cost(), ObjectiveDirection::Minimize)],
            ),
            Err(BenchmarkEvaluationError::MissingOutcome { candidate_id, .. }) if candidate_id == "b"
        ));
    }

    #[test]
    fn evaluation_schema_may_be_distinct_from_planning_schema() {
        let decisions = decisions();
        let complete = outcomes(&decisions, true);
        let evaluation = evaluate_shadow_decisions(
            &decisions,
            &complete,
            &[EvaluationObjective::new(realized_cost(), ObjectiveDirection::Minimize)],
        )
        .unwrap();
        assert_eq!(evaluation.evaluation_schema().get("realized_cost"), Some(&realized_cost()));
        assert_eq!(evaluation.outcome_frontier_ids(), vec!["b"]);
        assert_eq!(evaluation.outcome_ranking().rank_of("a"), Some(1));
        assert_eq!(evaluation.outcome_ranking().rank_of("b"), Some(0));
    }

    #[test]
    fn evaluation_reports_rank_for_recommendations_without_declaring_a_winner() {
        let decisions = decisions();
        let complete = outcomes(&decisions, true);
        let evaluation = evaluate_shadow_decisions(
            &decisions,
            &complete,
            &[EvaluationObjective::new(realized_cost(), ObjectiveDirection::Minimize)],
        )
        .unwrap();
        let Some(EvaluatedSelectorDecision::Recommended {
            candidate_id,
            outcome_pareto_rank,
            outcome_values,
            ..
        }) = evaluation.evaluated_decision(SelectorKind::CanonicalFeasible)
        else {
            panic!("canonical baseline should recommend");
        };
        assert_eq!(candidate_id, "a");
        assert_eq!(*outcome_pareto_rank, 1);
        assert_eq!(outcome_values["realized_cost"].evidence_class, OutcomeEvidenceClass::Simulated);
    }

    #[test]
    fn outcome_set_is_bound_to_exact_feasible_candidate_identity() {
        let decisions = decisions();
        let feasible = decisions
            .ranking_context()
            .evidence_set()
            .feasible_set()
            .clone();
        let mut set = ResourceOutcomeSet::new(feasible);
        let error = set
            .insert(CandidateOutcomeEvidence {
                candidate_id: "manufactured".into(),
                metric: realized_cost(),
                value: 1.0,
                evidence_class: OutcomeEvidenceClass::Observed,
                evidence_ref: "outcome:manufactured".into(),
            })
            .unwrap_err();
        assert!(matches!(
            error,
            OutcomeEvidenceError::CandidateOutsideUniverse(id) if id == "manufactured"
        ));
    }
}
