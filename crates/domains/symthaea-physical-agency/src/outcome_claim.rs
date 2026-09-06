// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared simulation outcome claims for strict Physical Agency evidence.
//!
//! A structurally valid solver result is not automatically evidence that the
//! proposed physical effect succeeded. PA-13 freezes a typed claim before the
//! strict simulation executes, carries that same claim through the run, and
//! evaluates exact metric/unit criteria conservatively under uncertainty.
//!
//! Successful evaluation produces a non-serializable runtime receipt. This is
//! still simulation evidence only: it is not a safety proof, backend
//! authentication, actuator command, or physical execution authority.

use crate::deliberation::SelectedCandidate;
use crate::strict_context::{
    CanonicalRequestTranscript, SimulationContextRef, StrictSimulationRegistry,
};
use crate::strict_selection::{
    PreparedSelectedSimulation, SelectionBoundSimulationError, SelectionBoundSimulationEvidence,
    prepare_selected_simulation, run_prepared_selected_simulation,
};
use serde::{Deserialize, Serialize};
use symthaea_sim_bridge::{Interval, SimulationMetric, SimulationRequest};
use thiserror::Error;

pub const OUTCOME_CLAIM_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricUncertaintyPolicy {
    RequireInterval,
    AllowPointEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricPredicate {
    AtLeast(f64),
    AtMost(f64),
    InsideClosedInterval { lower: f64, upper: f64 },
    OutsideOpenInterval { lower: f64, upper: f64 },
}

impl MetricPredicate {
    fn validate(&self) -> Result<(), OutcomeClaimError> {
        match self {
            Self::AtLeast(value) | Self::AtMost(value) => {
                if !value.is_finite() {
                    return Err(OutcomeClaimError::InvalidClaim(
                        "metric threshold must be finite".into(),
                    ));
                }
            }
            Self::InsideClosedInterval { lower, upper }
            | Self::OutsideOpenInterval { lower, upper } => {
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    return Err(OutcomeClaimError::InvalidClaim(
                        "metric interval bounds must be finite and ordered".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricCriterion {
    pub metric_name: String,
    pub unit: String,
    pub predicate: MetricPredicate,
    pub uncertainty_policy: MetricUncertaintyPolicy,
}

impl MetricCriterion {
    pub fn validate(&self) -> Result<(), OutcomeClaimError> {
        if self.metric_name.trim().is_empty() {
            return Err(OutcomeClaimError::InvalidClaim(
                "criterion metric_name cannot be empty".into(),
            ));
        }
        if self.unit.trim().is_empty() {
            return Err(OutcomeClaimError::InvalidClaim(
                "criterion unit cannot be empty".into(),
            ));
        }
        self.predicate.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimAggregation {
    AllCriteria,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationOutcomeClaim {
    pub schema_version: u16,
    pub claim_id: String,
    pub transition_id: String,
    pub proposal_id: String,
    pub criteria: Vec<MetricCriterion>,
    pub aggregation: ClaimAggregation,
}

impl SimulationOutcomeClaim {
    pub fn all_criteria(
        claim_id: impl Into<String>,
        transition_id: impl Into<String>,
        proposal_id: impl Into<String>,
        criteria: Vec<MetricCriterion>,
    ) -> Self {
        Self {
            schema_version: OUTCOME_CLAIM_SCHEMA_VERSION,
            claim_id: claim_id.into(),
            transition_id: transition_id.into(),
            proposal_id: proposal_id.into(),
            criteria,
            aggregation: ClaimAggregation::AllCriteria,
        }
    }

    pub fn validate(&self) -> Result<(), OutcomeClaimError> {
        if self.schema_version != OUTCOME_CLAIM_SCHEMA_VERSION {
            return Err(OutcomeClaimError::UnsupportedClaimSchema(
                self.schema_version,
            ));
        }
        if self.claim_id.trim().is_empty()
            || self.transition_id.trim().is_empty()
            || self.proposal_id.trim().is_empty()
        {
            return Err(OutcomeClaimError::InvalidClaim(
                "claim id, transition id, and proposal id are required".into(),
            ));
        }
        if self.criteria.is_empty() {
            return Err(OutcomeClaimError::InvalidClaim(
                "strict confirmatory claim requires at least one criterion".into(),
            ));
        }
        for criterion in &self.criteria {
            criterion.validate()?;
        }
        Ok(())
    }

    fn validate_for_selection(
        &self,
        selected: &SelectedCandidate,
    ) -> Result<(), OutcomeClaimError> {
        self.validate()?;
        if self.transition_id != selected.transition().id {
            return Err(OutcomeClaimError::TransitionMismatch {
                claim: self.transition_id.clone(),
                selected: selected.transition().id.clone(),
            });
        }
        if self.proposal_id != selected.assessment().proposal.id {
            return Err(OutcomeClaimError::ProposalMismatch {
                claim: self.proposal_id.clone(),
                selected: selected.assessment().proposal.id.clone(),
            });
        }
        Ok(())
    }
}

/// Non-serializable preregistration receipt. The claim is frozen together with
/// the selection-bound strict request before solver execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedConfirmatorySimulation {
    prepared: PreparedSelectedSimulation,
    claim: SimulationOutcomeClaim,
}

impl PreparedConfirmatorySimulation {
    pub fn prepared(&self) -> &PreparedSelectedSimulation {
        &self.prepared
    }

    pub fn claim(&self) -> &SimulationOutcomeClaim {
        &self.claim
    }
}

pub fn prepare_confirmatory_simulation(
    selected: &SelectedCandidate,
    request: SimulationRequest,
    claim: SimulationOutcomeClaim,
) -> Result<PreparedConfirmatorySimulation, OutcomeClaimError> {
    claim.validate_for_selection(selected)?;
    let prepared = prepare_selected_simulation(selected, request)
        .map_err(OutcomeClaimError::SelectionBound)?;
    Ok(PreparedConfirmatorySimulation { prepared, claim })
}

/// Non-serializable strict solver evidence that retains the exact preregistered
/// claim used to prepare the run.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfirmatorySimulationEvidence {
    selection_evidence: SelectionBoundSimulationEvidence,
    claim: SimulationOutcomeClaim,
}

impl ConfirmatorySimulationEvidence {
    pub fn selection_evidence(&self) -> &SelectionBoundSimulationEvidence {
        &self.selection_evidence
    }

    pub fn claim(&self) -> &SimulationOutcomeClaim {
        &self.claim
    }
}

pub fn run_confirmatory_simulation(
    registry: &StrictSimulationRegistry,
    prepared: &PreparedConfirmatorySimulation,
) -> Result<ConfirmatorySimulationEvidence, OutcomeClaimError> {
    let selection_evidence = run_prepared_selected_simulation(registry, prepared.prepared())
        .map_err(OutcomeClaimError::SelectionBound)?;
    Ok(ConfirmatorySimulationEvidence {
        selection_evidence,
        claim: prepared.claim().clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCriterionResult {
    Satisfied,
    NotSatisfied,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionEvidenceTier {
    IntervalBound,
    PointEstimate,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatedCriterion {
    pub criterion: MetricCriterion,
    pub result: ClaimCriterionResult,
    pub evidence_tier: CriterionEvidenceTier,
    pub observed_value: Option<f64>,
    pub observed_interval: Option<Interval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimEvidenceTier {
    IntervalBound,
    PointEstimateAllowed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimEvaluation {
    pub claim_id: String,
    pub criteria: Vec<EvaluatedCriterion>,
    pub overall: ClaimCriterionResult,
}

/// Non-serializable receipt that the exact preregistered claim was satisfied by
/// the exact strict simulation evidence lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct SatisfiedSimulationClaim {
    claim: SimulationOutcomeClaim,
    evaluation: ClaimEvaluation,
    evidence_tier: ClaimEvidenceTier,
    simulation_request_id: String,
    output_digest: String,
    request_transcript: CanonicalRequestTranscript,
    contexts: Vec<SimulationContextRef>,
}

impl SatisfiedSimulationClaim {
    pub fn claim(&self) -> &SimulationOutcomeClaim {
        &self.claim
    }

    pub fn evaluation(&self) -> &ClaimEvaluation {
        &self.evaluation
    }

    pub fn evidence_tier(&self) -> ClaimEvidenceTier {
        self.evidence_tier
    }

    pub fn simulation_request_id(&self) -> &str {
        &self.simulation_request_id
    }

    pub fn output_digest(&self) -> &str {
        &self.output_digest
    }

    pub fn request_transcript(&self) -> &CanonicalRequestTranscript {
        &self.request_transcript
    }

    pub fn contexts(&self) -> &[SimulationContextRef] {
        &self.contexts
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfirmatoryClaimOutcome {
    Satisfied(SatisfiedSimulationClaim),
    NotSatisfied(ClaimEvaluation),
    Indeterminate(ClaimEvaluation),
}

pub fn evaluate_confirmatory_claim(
    evidence: &ConfirmatorySimulationEvidence,
) -> Result<ConfirmatoryClaimOutcome, OutcomeClaimError> {
    let selected = evidence.selection_evidence().selected();
    evidence.claim().validate_for_selection(selected)?;

    let validated = evidence.selection_evidence().validated();
    let result = validated.result();
    let evaluations = evidence
        .claim()
        .criteria
        .iter()
        .map(|criterion| evaluate_criterion(criterion, &result.metrics))
        .collect::<Result<Vec<_>, _>>()?;

    let overall = match evidence.claim().aggregation {
        ClaimAggregation::AllCriteria => {
            if evaluations
                .iter()
                .any(|evaluation| evaluation.result == ClaimCriterionResult::NotSatisfied)
            {
                ClaimCriterionResult::NotSatisfied
            } else if evaluations
                .iter()
                .any(|evaluation| evaluation.result == ClaimCriterionResult::Indeterminate)
            {
                ClaimCriterionResult::Indeterminate
            } else {
                ClaimCriterionResult::Satisfied
            }
        }
    };

    let evaluation = ClaimEvaluation {
        claim_id: evidence.claim().claim_id.clone(),
        criteria: evaluations,
        overall,
    };

    match overall {
        ClaimCriterionResult::NotSatisfied => {
            Ok(ConfirmatoryClaimOutcome::NotSatisfied(evaluation))
        }
        ClaimCriterionResult::Indeterminate => {
            Ok(ConfirmatoryClaimOutcome::Indeterminate(evaluation))
        }
        ClaimCriterionResult::Satisfied => {
            let output_digest = result
                .evidence
                .output_digest
                .clone()
                .ok_or(OutcomeClaimError::MissingOutputDigest)?;
            let evidence_tier = if evaluation
                .criteria
                .iter()
                .any(|criterion| criterion.evidence_tier == CriterionEvidenceTier::PointEstimate)
            {
                ClaimEvidenceTier::PointEstimateAllowed
            } else {
                ClaimEvidenceTier::IntervalBound
            };
            Ok(ConfirmatoryClaimOutcome::Satisfied(
                SatisfiedSimulationClaim {
                    claim: evidence.claim().clone(),
                    evaluation,
                    evidence_tier,
                    simulation_request_id: result.request_id.clone(),
                    output_digest,
                    request_transcript: validated.request_transcript().clone(),
                    contexts: validated.contexts().to_vec(),
                },
            ))
        }
    }
}

fn evaluate_criterion(
    criterion: &MetricCriterion,
    metrics: &[SimulationMetric],
) -> Result<EvaluatedCriterion, OutcomeClaimError> {
    criterion.validate()?;
    let matching = metrics
        .iter()
        .filter(|metric| metric.name == criterion.metric_name && metric.unit == criterion.unit)
        .collect::<Vec<_>>();

    if matching.len() != 1 {
        return Ok(EvaluatedCriterion {
            criterion: criterion.clone(),
            result: ClaimCriterionResult::Indeterminate,
            evidence_tier: CriterionEvidenceTier::Missing,
            observed_value: None,
            observed_interval: None,
        });
    }

    let metric = matching[0];
    match criterion.uncertainty_policy {
        MetricUncertaintyPolicy::RequireInterval => {
            let interval = metric.uncertainty.and_then(|uncertainty| uncertainty.interval);
            let Some(interval) = interval else {
                return Ok(EvaluatedCriterion {
                    criterion: criterion.clone(),
                    result: ClaimCriterionResult::Indeterminate,
                    evidence_tier: CriterionEvidenceTier::Missing,
                    observed_value: Some(metric.value),
                    observed_interval: None,
                });
            };
            Ok(EvaluatedCriterion {
                criterion: criterion.clone(),
                result: evaluate_interval(&criterion.predicate, interval),
                evidence_tier: CriterionEvidenceTier::IntervalBound,
                observed_value: Some(metric.value),
                observed_interval: Some(interval),
            })
        }
        MetricUncertaintyPolicy::AllowPointEstimate => {
            if let Some(interval) = metric.uncertainty.and_then(|uncertainty| uncertainty.interval) {
                Ok(EvaluatedCriterion {
                    criterion: criterion.clone(),
                    result: evaluate_interval(&criterion.predicate, interval),
                    evidence_tier: CriterionEvidenceTier::IntervalBound,
                    observed_value: Some(metric.value),
                    observed_interval: Some(interval),
                })
            } else {
                Ok(EvaluatedCriterion {
                    criterion: criterion.clone(),
                    result: evaluate_point(&criterion.predicate, metric.value),
                    evidence_tier: CriterionEvidenceTier::PointEstimate,
                    observed_value: Some(metric.value),
                    observed_interval: None,
                })
            }
        }
    }
}

fn evaluate_point(predicate: &MetricPredicate, value: f64) -> ClaimCriterionResult {
    match predicate {
        MetricPredicate::AtLeast(threshold) => {
            bool_result(value >= *threshold)
        }
        MetricPredicate::AtMost(threshold) => {
            bool_result(value <= *threshold)
        }
        MetricPredicate::InsideClosedInterval { lower, upper } => {
            bool_result(value >= *lower && value <= *upper)
        }
        MetricPredicate::OutsideOpenInterval { lower, upper } => {
            bool_result(value <= *lower || value >= *upper)
        }
    }
}

fn bool_result(satisfied: bool) -> ClaimCriterionResult {
    if satisfied {
        ClaimCriterionResult::Satisfied
    } else {
        ClaimCriterionResult::NotSatisfied
    }
}

fn evaluate_interval(predicate: &MetricPredicate, interval: Interval) -> ClaimCriterionResult {
    match predicate {
        MetricPredicate::AtLeast(threshold) => {
            if interval.lower >= *threshold {
                ClaimCriterionResult::Satisfied
            } else if interval.upper < *threshold {
                ClaimCriterionResult::NotSatisfied
            } else {
                ClaimCriterionResult::Indeterminate
            }
        }
        MetricPredicate::AtMost(threshold) => {
            if interval.upper <= *threshold {
                ClaimCriterionResult::Satisfied
            } else if interval.lower > *threshold {
                ClaimCriterionResult::NotSatisfied
            } else {
                ClaimCriterionResult::Indeterminate
            }
        }
        MetricPredicate::InsideClosedInterval { lower, upper } => {
            if interval.lower >= *lower && interval.upper <= *upper {
                ClaimCriterionResult::Satisfied
            } else if interval.upper < *lower || interval.lower > *upper {
                ClaimCriterionResult::NotSatisfied
            } else {
                ClaimCriterionResult::Indeterminate
            }
        }
        MetricPredicate::OutsideOpenInterval { lower, upper } => {
            if interval.upper <= *lower || interval.lower >= *upper {
                ClaimCriterionResult::Satisfied
            } else if interval.lower > *lower && interval.upper < *upper {
                ClaimCriterionResult::NotSatisfied
            } else {
                ClaimCriterionResult::Indeterminate
            }
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum OutcomeClaimError {
    #[error("unsupported simulation outcome claim schema version {0}")]
    UnsupportedClaimSchema(u16),
    #[error("invalid simulation outcome claim: {0}")]
    InvalidClaim(String),
    #[error("claim transition {claim:?} does not match selected transition {selected:?}")]
    TransitionMismatch { claim: String, selected: String },
    #[error("claim proposal {claim:?} does not match selected proposal {selected:?}")]
    ProposalMismatch { claim: String, selected: String },
    #[error("selection-bound strict simulation failed: {0}")]
    SelectionBound(SelectionBoundSimulationError),
    #[error("strict external solver evidence unexpectedly lacks an output digest")]
    MissingOutputDigest,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deliberation::{
        DeliberationOutcome, SnapshotDigestAlgorithm, WorldSnapshotRef, deliberate,
    };
    use crate::portfolio::{
        CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
    };
    use crate::strict_context::{
        ContextAwareSimulationBackend, ContextBoundSimulationRequest, ContextBoundSimulationResult,
        ContextConsumptionEvidence,
    };
    use symthaea_physical_effects::{
        AuthorityClass, DesiredTransition, EffectKind, MechanismRef, PhysicalModality,
        PredictedOutcome, ProposedIntervention, TargetRegion,
    };
    use symthaea_sim_bridge::{
        EngineeringDomain, ExecutionMode, SimulationEvidence, SimulationError, SimulationMetric,
        SimulationResult, SolverKind, UncertaintyEstimate,
    };

    #[derive(Debug)]
    struct FixtureBackend {
        metrics: Vec<SimulationMetric>,
    }

    impl ContextAwareSimulationBackend for FixtureBackend {
        fn name(&self) -> &'static str {
            "confirmatory-fixture"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::Custom]
        }

        fn run_context_bound(
            &self,
            request: &ContextBoundSimulationRequest,
        ) -> Result<ContextBoundSimulationResult, SimulationError> {
            let mut result = SimulationResult::converged(&request.request.id, 0.97)
                .with_uncertainty(UncertaintyEstimate::new(0.02, 0.01))
                .with_external_evidence(SimulationEvidence {
                    mode: ExecutionMode::ExternalSolver,
                    backend: Some(self.name().into()),
                    solver_version: Some("fixture-1".into()),
                    input_digest: Some("input-digest".into()),
                    output_digest: Some("output-digest".into()),
                    parser_version: Some("parser-1".into()),
                });
            result.metrics = self.metrics.clone();
            Ok(ContextBoundSimulationResult {
                result,
                consumption: ContextConsumptionEvidence {
                    request_transcript: request
                        .canonical_transcript()
                        .map_err(|error| SimulationError::Adapter(error.to_string()))?,
                    consumed_contexts: request.contexts.clone(),
                },
            })
        }
    }

    fn selected() -> SelectedCandidate {
        let transition = DesiredTransition::simulation_only(
            "claim-t0",
            "confirm diagnostic quality in simulation",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let assessment = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "claim-p0".into(),
                transition_id: "claim-t0".into(),
                mechanism: MechanismRef {
                    backend: "fixture-model".into(),
                    mechanism: "diagnostic".into(),
                    modality: PhysicalModality::Acoustic,
                },
                required_authority: AuthorityClass::SimulationOnly,
                predicted_outcome: PredictedOutcome {
                    success_probability: 0.9,
                    epistemic_uncertainty: 0.08,
                    aleatoric_uncertainty: 0.03,
                },
            },
            model_predictions: vec![
                ModelPrediction {
                    model_id: "model-a".into(),
                    success_probability: 0.9,
                },
                ModelPrediction {
                    model_id: "model-b".into(),
                    success_probability: 0.88,
                },
            ],
            expected_energy_j: 1.0,
            expected_power_w: None,
            expected_duration_ms: 100,
            information_gain: 0.8,
            reversibility_score: 1.0,
            safety_margin: 0.95,
        };
        let portfolio = CandidatePortfolio {
            transition,
            candidates: vec![assessment],
        };
        let snapshot = WorldSnapshotRef::cryptographic(
            "world",
            SnapshotDigestAlgorithm::Blake3,
            "a".repeat(64),
        );
        let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };
        frontier.select("claim-p0").unwrap()
    }

    fn request() -> SimulationRequest {
        SimulationRequest::new(
            "claim-run-0",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "confirm preregistered diagnostic quality",
        )
    }

    fn criterion(policy: MetricUncertaintyPolicy) -> MetricCriterion {
        MetricCriterion {
            metric_name: "diagnostic_quality".into(),
            unit: "1".into(),
            predicate: MetricPredicate::AtLeast(0.8),
            uncertainty_policy: policy,
        }
    }

    fn claim(policy: MetricUncertaintyPolicy) -> SimulationOutcomeClaim {
        SimulationOutcomeClaim::all_criteria(
            "claim-0",
            "claim-t0",
            "claim-p0",
            vec![criterion(policy)],
        )
    }

    fn metric(value: f64, interval: Option<(f64, f64)>) -> SimulationMetric {
        SimulationMetric {
            name: "diagnostic_quality".into(),
            value,
            unit: "1".into(),
            uncertainty: interval.map(|(lower, upper)| {
                UncertaintyEstimate::new(0.05, 0.02)
                    .with_interval(Interval::new(lower, upper))
            }),
        }
    }

    #[test]
    fn confirmatory_claim_is_frozen_before_run_and_can_mint_satisfied_receipt() {
        let selected = selected();
        let prepared =
            prepare_confirmatory_simulation(&selected, request(), claim(MetricUncertaintyPolicy::RequireInterval))
                .unwrap();
        assert_eq!(prepared.claim().claim_id, "claim-0");

        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend {
            metrics: vec![metric(0.86, Some((0.82, 0.91)))],
        });
        let evidence = run_confirmatory_simulation(&registry, &prepared).unwrap();
        assert_eq!(evidence.claim(), prepared.claim());

        let satisfied = match evaluate_confirmatory_claim(&evidence).unwrap() {
            ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
            other => panic!("expected satisfied claim, got {other:?}"),
        };
        assert_eq!(satisfied.claim().claim_id, "claim-0");
        assert_eq!(satisfied.output_digest(), "output-digest");
        assert_eq!(satisfied.evidence_tier(), ClaimEvidenceTier::IntervalBound);
        assert_eq!(satisfied.contexts().len(), 1);
    }

    #[test]
    fn uncertainty_interval_that_straddles_threshold_is_indeterminate() {
        let evaluation = evaluate_criterion(
            &criterion(MetricUncertaintyPolicy::RequireInterval),
            &[metric(0.85, Some((0.72, 0.93)))],
        )
        .unwrap();
        assert_eq!(evaluation.result, ClaimCriterionResult::Indeterminate);
    }

    #[test]
    fn wrong_unit_or_duplicate_metric_is_indeterminate() {
        let mut wrong_unit = metric(0.9, Some((0.85, 0.95)));
        wrong_unit.unit = "percent".into();
        assert_eq!(
            evaluate_criterion(
                &criterion(MetricUncertaintyPolicy::RequireInterval),
                &[wrong_unit]
            )
            .unwrap()
            .result,
            ClaimCriterionResult::Indeterminate
        );

        let duplicate = metric(0.9, Some((0.85, 0.95)));
        assert_eq!(
            evaluate_criterion(
                &criterion(MetricUncertaintyPolicy::RequireInterval),
                &[duplicate.clone(), duplicate]
            )
            .unwrap()
            .result,
            ClaimCriterionResult::Indeterminate
        );
    }

    #[test]
    fn missing_required_interval_is_indeterminate_but_point_policy_is_explicitly_weaker() {
        let point = metric(0.9, None);
        assert_eq!(
            evaluate_criterion(
                &criterion(MetricUncertaintyPolicy::RequireInterval),
                std::slice::from_ref(&point)
            )
            .unwrap()
            .result,
            ClaimCriterionResult::Indeterminate
        );
        let weaker = evaluate_criterion(
            &criterion(MetricUncertaintyPolicy::AllowPointEstimate),
            &[point],
        )
        .unwrap();
        assert_eq!(weaker.result, ClaimCriterionResult::Satisfied);
        assert_eq!(weaker.evidence_tier, CriterionEvidenceTier::PointEstimate);
    }

    #[test]
    fn neighboring_claim_cannot_be_preregistered_for_selected_proposal() {
        let selected = selected();
        let neighboring = SimulationOutcomeClaim::all_criteria(
            "neighbor",
            "claim-t0",
            "different-proposal",
            vec![criterion(MetricUncertaintyPolicy::RequireInterval)],
        );
        assert!(matches!(
            prepare_confirmatory_simulation(&selected, request(), neighboring),
            Err(OutcomeClaimError::ProposalMismatch { .. })
        ));
    }
}
