// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered safety-case lineage for strict confirmatory simulation.
//!
//! PA-13 freezes outcome criteria before the solver runs. PA-15 applies the
//! same anti-post-hoc principle to the safety argument: the obligation set must
//! exist before solver execution, and only obligation status/evidence may evolve
//! afterward.
//!
//! This remains structural simulation qualification. It does not authenticate
//! the truth of an evidence reference, turn `SafetyCase` into an authority root,
//! or introduce HAL/actuator capability.

use crate::outcome_claim::{
    ConfirmatorySimulationEvidence, OutcomeClaimError, PreparedConfirmatorySimulation,
    SatisfiedSimulationClaim, run_confirmatory_simulation,
};
use crate::strict_context::StrictSimulationRegistry;
use crate::strict_qualification::{
    StrictConfirmatoryQualificationError, StrictConfirmatorySimulationQualification,
    qualify_confirmatory_simulation, required_confirmatory_safety_evidence_ref,
};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_formal_safety::{EvidenceKind, ObligationStatus, SafetyCase};
use thiserror::Error;

/// Frozen identity of one safety obligation as it existed before simulation.
///
/// Intentionally non-serializable. Evidence references and mutable status are
/// absent because those are the fields allowed to evolve after execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenSafetyObligation {
    id: String,
    claim: String,
    expected_evidence: EvidenceKind,
}

impl FrozenSafetyObligation {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn claim(&self) -> &str {
        &self.claim
    }

    pub fn expected_evidence(&self) -> EvidenceKind {
        self.expected_evidence
    }
}

/// Non-serializable preregistration receipt for the safety argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreregisteredSafetyPlan {
    case_id: String,
    subject: String,
    obligations: Vec<FrozenSafetyObligation>,
}

impl PreregisteredSafetyPlan {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn obligations(&self) -> &[FrozenSafetyObligation] {
        &self.obligations
    }
}

/// Confirmatory request with both the outcome claim and safety-obligation set
/// frozen before solver execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSafetyConfirmatorySimulation {
    prepared: PreparedConfirmatorySimulation,
    safety_plan: PreregisteredSafetyPlan,
}

impl PreparedSafetyConfirmatorySimulation {
    pub fn prepared(&self) -> &PreparedConfirmatorySimulation {
        &self.prepared
    }

    pub fn safety_plan(&self) -> &PreregisteredSafetyPlan {
        &self.safety_plan
    }
}

/// Strict solver evidence that retains the safety plan that existed before the
/// run. The receipt is intentionally non-serializable.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetyPreregisteredConfirmatoryEvidence {
    confirmatory: ConfirmatorySimulationEvidence,
    safety_plan: PreregisteredSafetyPlan,
}

impl SafetyPreregisteredConfirmatoryEvidence {
    pub fn confirmatory(&self) -> &ConfirmatorySimulationEvidence {
        &self.confirmatory
    }

    pub fn safety_plan(&self) -> &PreregisteredSafetyPlan {
        &self.safety_plan
    }
}

/// Freeze a safety case before the confirmatory solver run.
///
/// Strict preregistration requires:
/// - exact proposal subject identity;
/// - at least one simulation obligation for exact-run evidence;
/// - at least one non-simulation obligation so solver success is not the entire
///   safety argument;
/// - unique obligation ids, non-empty claims, Open status, and no attached
///   evidence before execution.
pub fn preregister_confirmatory_safety(
    prepared: PreparedConfirmatorySimulation,
    safety_case: &SafetyCase,
) -> Result<PreparedSafetyConfirmatorySimulation, SafetyPreregistrationError> {
    let proposal = &prepared.prepared().selected().assessment().proposal.id;
    if safety_case.subject != *proposal {
        return Err(SafetyPreregistrationError::SafetyCaseSubjectMismatch {
            safety_subject: safety_case.subject.clone(),
            proposal: proposal.clone(),
        });
    }
    if safety_case.obligations.is_empty() {
        return Err(SafetyPreregistrationError::EmptySafetyPlan);
    }

    let mut ids = BTreeSet::new();
    let mut has_simulation = false;
    let mut has_independent = false;
    let mut obligations = Vec::with_capacity(safety_case.obligations.len());

    for obligation in &safety_case.obligations {
        let id = obligation.id.to_string();
        if !ids.insert(id.clone()) {
            return Err(SafetyPreregistrationError::DuplicateObligationId(id));
        }
        if obligation.claim.trim().is_empty() {
            return Err(SafetyPreregistrationError::EmptyObligationClaim(id));
        }
        if obligation.status != ObligationStatus::Open {
            return Err(SafetyPreregistrationError::ObligationNotOpen(id));
        }
        if !obligation.evidence_refs.is_empty() {
            return Err(SafetyPreregistrationError::PrematureEvidence(id));
        }

        if obligation.expected_evidence == EvidenceKind::Simulation {
            has_simulation = true;
        } else {
            has_independent = true;
        }

        obligations.push(FrozenSafetyObligation {
            id,
            claim: obligation.claim.clone(),
            expected_evidence: obligation.expected_evidence,
        });
    }

    if !has_simulation {
        return Err(SafetyPreregistrationError::MissingSimulationObligation);
    }
    if !has_independent {
        return Err(SafetyPreregistrationError::MissingIndependentObligation);
    }

    Ok(PreparedSafetyConfirmatorySimulation {
        prepared,
        safety_plan: PreregisteredSafetyPlan {
            case_id: safety_case.id.to_string(),
            subject: safety_case.subject.clone(),
            obligations,
        },
    })
}

/// Execute only after both outcome and safety criteria have been frozen.
pub fn run_preregistered_safety_confirmatory_simulation(
    registry: &StrictSimulationRegistry,
    prepared: &PreparedSafetyConfirmatorySimulation,
) -> Result<SafetyPreregisteredConfirmatoryEvidence, SafetyPreregistrationError> {
    let confirmatory = run_confirmatory_simulation(registry, prepared.prepared())?;
    Ok(SafetyPreregisteredConfirmatoryEvidence {
        confirmatory,
        safety_plan: prepared.safety_plan.clone(),
    })
}

/// Exact PA-14 v2 simulation evidence reference to attach to one of the
/// preregistered `EvidenceKind::Simulation` obligations after the run.
pub fn required_preregistered_safety_evidence_ref(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
) -> Result<String, SafetyPreregistrationError> {
    Ok(required_confirmatory_safety_evidence_ref(
        evidence.confirmatory(),
        satisfied,
    )?)
}

/// Qualify a confirmatory simulation only if the completed safety case is the
/// same case and same obligation set that was frozen before execution.
///
/// Obligation status may move from Open to Discharged and evidence references
/// may be attached. Ids, claims, evidence kinds, case id, subject, obligation
/// count, and obligation membership may not change.
pub fn qualify_preregistered_safety_confirmatory_simulation(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
    completed_safety_case: &SafetyCase,
) -> Result<StrictConfirmatorySimulationQualification, SafetyPreregistrationError> {
    validate_completed_safety_case(evidence.safety_plan(), completed_safety_case)?;

    let required_ref = required_preregistered_safety_evidence_ref(evidence, satisfied)?;
    let completed = completed_obligation_map(completed_safety_case)?;

    let has_exact_simulation_evidence = evidence
        .safety_plan()
        .obligations()
        .iter()
        .filter(|obligation| obligation.expected_evidence == EvidenceKind::Simulation)
        .any(|obligation| {
            completed
                .get(obligation.id())
                .is_some_and(|actual| actual.evidence_refs.iter().any(|item| item == &required_ref))
        });
    if !has_exact_simulation_evidence {
        return Err(SafetyPreregistrationError::MissingExactSimulationEvidence);
    }

    // Every independently-typed obligation must have at least one evidence
    // reference other than the exact simulation-lineage label. This prevents a
    // single solver receipt from being reused as the entire safety argument.
    for obligation in evidence
        .safety_plan()
        .obligations()
        .iter()
        .filter(|obligation| obligation.expected_evidence != EvidenceKind::Simulation)
    {
        let actual = completed
            .get(obligation.id())
            .expect("validated safety plan must contain every frozen obligation");
        if !actual
            .evidence_refs
            .iter()
            .any(|item| !item.trim().is_empty() && item != &required_ref)
        {
            return Err(SafetyPreregistrationError::IndependentEvidenceMissing(
                obligation.id().to_string(),
            ));
        }
    }

    Ok(qualify_confirmatory_simulation(
        evidence.confirmatory(),
        satisfied,
        completed_safety_case,
    )?)
}

fn validate_completed_safety_case(
    plan: &PreregisteredSafetyPlan,
    completed: &SafetyCase,
) -> Result<(), SafetyPreregistrationError> {
    if completed.id.to_string() != plan.case_id {
        return Err(SafetyPreregistrationError::SafetyCaseIdMismatch {
            preregistered: plan.case_id.clone(),
            completed: completed.id.to_string(),
        });
    }
    if completed.subject != plan.subject {
        return Err(SafetyPreregistrationError::CompletedSubjectMismatch {
            preregistered: plan.subject.clone(),
            completed: completed.subject.clone(),
        });
    }
    if completed.obligations.len() != plan.obligations.len() {
        return Err(SafetyPreregistrationError::ObligationSetChanged);
    }

    let actual = completed_obligation_map(completed)?;
    for frozen in &plan.obligations {
        let obligation = actual
            .get(frozen.id())
            .ok_or(SafetyPreregistrationError::ObligationSetChanged)?;
        if obligation.claim != frozen.claim
            || obligation.expected_evidence != frozen.expected_evidence
        {
            return Err(SafetyPreregistrationError::ObligationDefinitionChanged(
                frozen.id.clone(),
            ));
        }
        if obligation.status != ObligationStatus::Discharged {
            return Err(SafetyPreregistrationError::ObligationNotDischarged(
                frozen.id.clone(),
            ));
        }
        if obligation.evidence_refs.is_empty()
            || obligation.evidence_refs.iter().any(|item| item.trim().is_empty())
        {
            return Err(SafetyPreregistrationError::MissingObligationEvidence(
                frozen.id.clone(),
            ));
        }
    }

    if !completed.is_discharged() {
        return Err(SafetyPreregistrationError::SafetyCaseNotDischarged);
    }
    Ok(())
}

fn completed_obligation_map<'a>(
    safety_case: &'a SafetyCase,
) -> Result<BTreeMap<String, &'a symthaea_formal_safety::ProofObligation>, SafetyPreregistrationError>
{
    let mut map = BTreeMap::new();
    for obligation in &safety_case.obligations {
        let id = obligation.id.to_string();
        if map.insert(id.clone(), obligation).is_some() {
            return Err(SafetyPreregistrationError::DuplicateObligationId(id));
        }
    }
    Ok(map)
}

#[derive(Debug, Error)]
pub enum SafetyPreregistrationError {
    #[error("confirmatory simulation failed: {0}")]
    Outcome(#[from] OutcomeClaimError),
    #[error("strict confirmatory qualification failed: {0}")]
    StrictQualification(#[from] StrictConfirmatoryQualificationError),
    #[error("safety case subject {safety_subject:?} does not match proposal {proposal:?}")]
    SafetyCaseSubjectMismatch {
        safety_subject: String,
        proposal: String,
    },
    #[error("strict safety preregistration requires at least one obligation")]
    EmptySafetyPlan,
    #[error("duplicate safety obligation id {0:?}")]
    DuplicateObligationId(String),
    #[error("safety obligation {0:?} has an empty claim")]
    EmptyObligationClaim(String),
    #[error("safety obligation {0:?} must be Open at preregistration")]
    ObligationNotOpen(String),
    #[error("safety obligation {0:?} already contains evidence before execution")]
    PrematureEvidence(String),
    #[error("strict safety preregistration requires a Simulation obligation")]
    MissingSimulationObligation,
    #[error("strict safety preregistration requires at least one independent non-Simulation obligation")]
    MissingIndependentObligation,
    #[error("completed safety case id {completed:?} does not match preregistered id {preregistered:?}")]
    SafetyCaseIdMismatch {
        preregistered: String,
        completed: String,
    },
    #[error("completed safety subject {completed:?} does not match preregistered subject {preregistered:?}")]
    CompletedSubjectMismatch {
        preregistered: String,
        completed: String,
    },
    #[error("completed safety case changed the preregistered obligation set")]
    ObligationSetChanged,
    #[error("safety obligation {0:?} changed claim or expected evidence after preregistration")]
    ObligationDefinitionChanged(String),
    #[error("safety obligation {0:?} is not Discharged")]
    ObligationNotDischarged(String),
    #[error("safety obligation {0:?} has no concrete evidence reference")]
    MissingObligationEvidence(String),
    #[error("completed safety case is not fully discharged")]
    SafetyCaseNotDischarged,
    #[error("preregistered Simulation obligations do not cite the exact confirmatory lineage")]
    MissingExactSimulationEvidence,
    #[error("independent obligation {0:?} has no evidence distinct from the exact simulation lineage")]
    IndependentEvidenceMissing(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deliberation::{
        DeliberationOutcome, SnapshotDigestAlgorithm, WorldSnapshotRef, deliberate,
    };
    use crate::outcome_claim::{
        ConfirmatoryClaimOutcome, MetricCriterion, MetricPredicate, MetricUncertaintyPolicy,
        SimulationOutcomeClaim, evaluate_confirmatory_claim, prepare_confirmatory_simulation,
    };
    use crate::portfolio::{
        CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
    };
    use crate::strict_context::{
        ContextAwareSimulationBackend, ContextBoundSimulationRequest, ContextBoundSimulationResult,
        ContextConsumptionEvidence,
    };
    use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyCase};
    use symthaea_physical_effects::{
        AuthorityClass, DesiredTransition, EffectKind, MechanismRef, PhysicalModality,
        PredictedOutcome, ProposedIntervention, TargetRegion,
    };
    use symthaea_sim_bridge::{
        EngineeringDomain, ExecutionMode, Interval, SimulationEvidence, SimulationError,
        SimulationMetric, SimulationRequest, SimulationResult, SolverKind, UncertaintyEstimate,
    };

    #[derive(Debug)]
    struct FixtureBackend;

    impl ContextAwareSimulationBackend for FixtureBackend {
        fn name(&self) -> &'static str {
            "safety-preregistration-fixture"
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
                    input_digest: Some("safety-input".into()),
                    output_digest: Some("safety-output".into()),
                    parser_version: Some("parser-1".into()),
                });
            result.metrics = vec![SimulationMetric {
                name: "diagnostic_quality".into(),
                value: 0.9,
                unit: "1".into(),
                uncertainty: Some(
                    UncertaintyEstimate::new(0.03, 0.02)
                        .with_interval(Interval::new(0.86, 0.94)),
                ),
            }];
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

    fn selected() -> crate::deliberation::SelectedCandidate {
        let transition = DesiredTransition::simulation_only(
            "safety-t0",
            "safety preregistration fixture",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let assessment = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "safety-p0".into(),
                transition_id: "safety-t0".into(),
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
            "b".repeat(64),
        );
        let frontier = match deliberate(&portfolio, &snapshot, PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };
        frontier.select("safety-p0").unwrap()
    }

    fn prepared() -> PreparedConfirmatorySimulation {
        let selected = selected();
        let claim = SimulationOutcomeClaim::all_criteria(
            "safety-claim-0",
            "safety-t0",
            "safety-p0",
            vec![MetricCriterion {
                metric_name: "diagnostic_quality".into(),
                unit: "1".into(),
                predicate: MetricPredicate::AtLeast(0.8),
                uncertainty_policy: MetricUncertaintyPolicy::RequireInterval,
            }],
        );
        prepare_confirmatory_simulation(
            &selected,
            SimulationRequest::new(
                "safety-run-0",
                EngineeringDomain::Systems,
                SolverKind::Custom,
                "strict safety preregistration fixture",
            ),
            claim,
        )
        .unwrap()
    }

    fn open_safety_case() -> SafetyCase {
        let mut safety = SafetyCase::new("safety-p0");
        safety.add_obligation(ProofObligation::new(
            "exact confirmatory simulation supports the claimed outcome",
            EvidenceKind::Simulation,
        ));
        safety.add_obligation(ProofObligation::new(
            "independent invariant review is complete",
            EvidenceKind::FormalProof,
        ));
        safety
    }

    fn run_with_plan(
    ) -> (
        SafetyPreregisteredConfirmatoryEvidence,
        SatisfiedSimulationClaim,
        SafetyCase,
    ) {
        let safety = open_safety_case();
        let mut completed = safety.clone();
        let prepared = preregister_confirmatory_safety(prepared(), &safety).unwrap();
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend);
        let evidence = run_preregistered_safety_confirmatory_simulation(&registry, &prepared).unwrap();
        let satisfied = match evaluate_confirmatory_claim(evidence.confirmatory()).unwrap() {
            ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
            other => panic!("expected satisfied claim, got {other:?}"),
        };
        let exact = required_preregistered_safety_evidence_ref(&evidence, &satisfied).unwrap();
        for obligation in &mut completed.obligations {
            obligation.status = ObligationStatus::Discharged;
            obligation.evidence_refs.push(if obligation.expected_evidence == EvidenceKind::Simulation {
                exact.clone()
            } else {
                "independent-proof:fixture".into()
            });
        }
        (evidence, satisfied, completed)
    }

    #[test]
    fn preregistered_case_can_complete_and_strictly_qualify() {
        let (evidence, satisfied, completed) = run_with_plan();
        let qualified = qualify_preregistered_safety_confirmatory_simulation(
            &evidence,
            &satisfied,
            &completed,
        )
        .unwrap();
        assert_eq!(qualified.assessment().proposal.id, "safety-p0");
        assert_eq!(qualified.output_digest(), "safety-output");
    }

    #[test]
    fn simulation_only_safety_argument_is_rejected_before_execution() {
        let mut safety = SafetyCase::new("safety-p0");
        safety.add_obligation(ProofObligation::new(
            "simulation says it is safe",
            EvidenceKind::Simulation,
        ));
        assert!(matches!(
            preregister_confirmatory_safety(prepared(), &safety),
            Err(SafetyPreregistrationError::MissingIndependentObligation)
        ));
    }

    #[test]
    fn pre_discharged_obligation_cannot_be_preregistered() {
        let mut safety = open_safety_case();
        let first = safety.obligations.remove(0).discharge("post-hoc-looking-evidence");
        safety.obligations.insert(0, first);
        assert!(matches!(
            preregister_confirmatory_safety(prepared(), &safety),
            Err(SafetyPreregistrationError::ObligationNotOpen(_))
        ));
    }

    #[test]
    fn posthoc_new_safety_case_cannot_replace_preregistered_case() {
        let (evidence, satisfied, _completed) = run_with_plan();
        let exact = required_preregistered_safety_evidence_ref(&evidence, &satisfied).unwrap();
        let mut posthoc = SafetyCase::new("safety-p0");
        posthoc.add_obligation(
            ProofObligation::new("invented after run", EvidenceKind::Simulation).discharge(exact),
        );
        posthoc.add_obligation(
            ProofObligation::new("invented review", EvidenceKind::FormalProof)
                .discharge("independent-proof:posthoc"),
        );
        assert!(matches!(
            qualify_preregistered_safety_confirmatory_simulation(
                &evidence,
                &satisfied,
                &posthoc,
            ),
            Err(SafetyPreregistrationError::SafetyCaseIdMismatch { .. })
        ));
    }

    #[test]
    fn obligation_claim_cannot_change_after_preregistration() {
        let (evidence, satisfied, mut completed) = run_with_plan();
        completed.obligations[0].claim.push_str(" changed after run");
        assert!(matches!(
            qualify_preregistered_safety_confirmatory_simulation(
                &evidence,
                &satisfied,
                &completed,
            ),
            Err(SafetyPreregistrationError::ObligationDefinitionChanged(_))
        ));
    }

    #[test]
    fn independent_obligation_cannot_use_only_the_simulation_lineage() {
        let safety = open_safety_case();
        let mut completed = safety.clone();
        let prepared = preregister_confirmatory_safety(prepared(), &safety).unwrap();
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend);
        let evidence = run_preregistered_safety_confirmatory_simulation(&registry, &prepared).unwrap();
        let satisfied = match evaluate_confirmatory_claim(evidence.confirmatory()).unwrap() {
            ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
            other => panic!("expected satisfied claim, got {other:?}"),
        };
        let exact = required_preregistered_safety_evidence_ref(&evidence, &satisfied).unwrap();
        for obligation in &mut completed.obligations {
            obligation.status = ObligationStatus::Discharged;
            obligation.evidence_refs.push(exact.clone());
        }
        assert!(matches!(
            qualify_preregistered_safety_confirmatory_simulation(
                &evidence,
                &satisfied,
                &completed,
            ),
            Err(SafetyPreregistrationError::IndependentEvidenceMissing(_))
        ));
    }
}
