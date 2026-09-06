// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict confirmatory simulation qualification.
//!
//! PA-14 converges the two independent evidence questions that must both be
//! answered before a simulation candidate can be called strictly qualified:
//!
//! 1. Did the exact preregistered outcome claim pass with interval-backed
//!    evidence on this exact strict simulation lineage?
//! 2. Is the proposal-specific `SafetyCase` fully discharged and explicitly
//!    bound to this exact confirmatory run?
//!
//! The resulting receipt is still **simulation qualification only**. A
//! `SafetyCase` is currently a structural proof-obligation container, not an
//! authenticated authority root; this module does not upgrade it into one. No
//! HAL path, actuator command, or physical execution permit is introduced.

use crate::deliberation::{SelectedCandidate, SnapshotDigestAlgorithm, WorldSnapshotRef};
use crate::outcome_claim::{
    ClaimCriterionResult, ClaimEvidenceTier, ConfirmatorySimulationEvidence,
    SatisfiedSimulationClaim,
};
use crate::portfolio::{CandidateAssessment, PortfolioPolicy};
use crate::strict_context::{CanonicalRequestTranscript, SimulationContextRef};
use symthaea_formal_safety::{EvidenceKind, SafetyCase};
use thiserror::Error;

/// Non-serializable receipt for an interval-backed, claim-satisfied,
/// safety-case-discharged simulation candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct StrictConfirmatorySimulationQualification {
    selected: SelectedCandidate,
    satisfied_claim: SatisfiedSimulationClaim,
    safety_case_id: String,
    backend: String,
    solver_version: String,
    parser_version: String,
    input_digest: String,
    output_digest: String,
}

impl StrictConfirmatorySimulationQualification {
    pub fn selected(&self) -> &SelectedCandidate {
        &self.selected
    }

    pub fn assessment(&self) -> &CandidateAssessment {
        self.selected.assessment()
    }

    pub fn selection_policy(&self) -> PortfolioPolicy {
        self.selected.policy()
    }

    pub fn world_snapshot(&self) -> &WorldSnapshotRef {
        self.selected.world_snapshot()
    }

    pub fn satisfied_claim(&self) -> &SatisfiedSimulationClaim {
        &self.satisfied_claim
    }

    pub fn safety_case_id(&self) -> &str {
        &self.safety_case_id
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn solver_version(&self) -> &str {
        &self.solver_version
    }

    pub fn parser_version(&self) -> &str {
        &self.parser_version
    }

    pub fn input_digest(&self) -> &str {
        &self.input_digest
    }

    pub fn output_digest(&self) -> &str {
        &self.output_digest
    }

    pub fn request_transcript(&self) -> &CanonicalRequestTranscript {
        self.satisfied_claim.request_transcript()
    }

    pub fn contexts(&self) -> &[SimulationContextRef] {
        self.satisfied_claim.contexts()
    }
}

/// Canonical structural evidence reference a discharged simulation obligation
/// must contain for PA-14 qualification.
///
/// This reference binds the proposal, request, external input/output digests,
/// preregistered claim id, and selected world snapshot. It is an exact lineage
/// label, not a signature, attestation, or authorization token.
pub fn required_confirmatory_safety_evidence_ref(
    evidence: &ConfirmatorySimulationEvidence,
    satisfied: &SatisfiedSimulationClaim,
) -> Result<String, StrictConfirmatoryQualificationError> {
    validate_claim_run_lineage(evidence, satisfied)?;
    let selected = evidence.selection_evidence().selected();
    let result = evidence.selection_evidence().validated().result();
    let input_digest = result
        .evidence
        .input_digest
        .as_deref()
        .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?;
    let output_digest = result
        .evidence
        .output_digest
        .as_deref()
        .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?;

    let mut reference = String::from("physical-agency-confirmatory:v1");
    for field in [
        selected.assessment().proposal.id.as_str(),
        result.request_id.as_str(),
        input_digest,
        output_digest,
        satisfied.claim().claim_id.as_str(),
        snapshot_algorithm_tag(selected.world_snapshot().digest_algorithm()),
        selected.world_snapshot().snapshot_digest(),
    ] {
        reference.push('|');
        reference.push_str(&field.len().to_string());
        reference.push(':');
        reference.push_str(field);
    }
    Ok(reference)
}

pub fn qualify_confirmatory_simulation(
    evidence: &ConfirmatorySimulationEvidence,
    satisfied: &SatisfiedSimulationClaim,
    safety_case: &SafetyCase,
) -> Result<StrictConfirmatorySimulationQualification, StrictConfirmatoryQualificationError> {
    validate_claim_run_lineage(evidence, satisfied)?;

    if satisfied.evidence_tier() != ClaimEvidenceTier::IntervalBound {
        return Err(StrictConfirmatoryQualificationError::EvidenceTierTooWeak);
    }
    if satisfied.evaluation().overall != ClaimCriterionResult::Satisfied {
        return Err(StrictConfirmatoryQualificationError::ClaimNotSatisfied);
    }

    let selected = evidence.selection_evidence().selected();
    if safety_case.subject != selected.assessment().proposal.id {
        return Err(
            StrictConfirmatoryQualificationError::SafetyCaseSubjectMismatch {
                safety_subject: safety_case.subject.clone(),
                proposal: selected.assessment().proposal.id.clone(),
            },
        );
    }
    if !safety_case.is_discharged() {
        return Err(StrictConfirmatoryQualificationError::SafetyCaseNotDischarged);
    }

    let required_ref = required_confirmatory_safety_evidence_ref(evidence, satisfied)?;
    let exact_simulation_evidence = safety_case.obligations.iter().any(|obligation| {
        obligation.expected_evidence == EvidenceKind::Simulation
            && obligation
                .evidence_refs
                .iter()
                .any(|reference| reference == &required_ref)
    });
    if !exact_simulation_evidence {
        return Err(StrictConfirmatoryQualificationError::SafetyCaseMissingExactEvidence);
    }

    let validated = evidence.selection_evidence().validated();
    let result = validated.result();
    let provenance = &result.evidence;
    Ok(StrictConfirmatorySimulationQualification {
        selected: selected.clone(),
        satisfied_claim: satisfied.clone(),
        safety_case_id: safety_case.id.to_string(),
        backend: validated.backend().to_string(),
        solver_version: provenance
            .solver_version
            .clone()
            .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?,
        parser_version: provenance
            .parser_version
            .clone()
            .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?,
        input_digest: provenance
            .input_digest
            .clone()
            .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?,
        output_digest: provenance
            .output_digest
            .clone()
            .ok_or(StrictConfirmatoryQualificationError::IncompleteExternalProvenance)?,
    })
}

fn validate_claim_run_lineage(
    evidence: &ConfirmatorySimulationEvidence,
    satisfied: &SatisfiedSimulationClaim,
) -> Result<(), StrictConfirmatoryQualificationError> {
    if satisfied.claim() != evidence.claim() {
        return Err(StrictConfirmatoryQualificationError::ClaimLineageMismatch);
    }

    let validated = evidence.selection_evidence().validated();
    let result = validated.result();
    if satisfied.simulation_request_id() != result.request_id
        || satisfied.request_transcript() != validated.request_transcript()
        || satisfied.contexts() != validated.contexts()
        || result.evidence.output_digest.as_deref() != Some(satisfied.output_digest())
    {
        return Err(StrictConfirmatoryQualificationError::RunLineageMismatch);
    }
    Ok(())
}

fn snapshot_algorithm_tag(algorithm: SnapshotDigestAlgorithm) -> &'static str {
    match algorithm {
        SnapshotDigestAlgorithm::LegacyOpaque => "legacy-opaque",
        SnapshotDigestAlgorithm::Blake3 => "blake3-256",
        SnapshotDigestAlgorithm::Sha256 => "sha256",
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StrictConfirmatoryQualificationError {
    #[error("satisfied claim does not match the confirmatory evidence claim")]
    ClaimLineageMismatch,
    #[error("satisfied claim request/output/context lineage does not match confirmatory evidence")]
    RunLineageMismatch,
    #[error("strict confirmatory qualification requires interval-backed claim evidence")]
    EvidenceTierTooWeak,
    #[error("claim receipt is not satisfied")]
    ClaimNotSatisfied,
    #[error("external solver provenance is incomplete")]
    IncompleteExternalProvenance,
    #[error("safety case subject {safety_subject:?} does not match proposal {proposal:?}")]
    SafetyCaseSubjectMismatch {
        safety_subject: String,
        proposal: String,
    },
    #[error("safety case still contains open proof obligations")]
    SafetyCaseNotDischarged,
    #[error("safety case does not cite the exact confirmatory simulation evidence lineage")]
    SafetyCaseMissingExactEvidence,
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
        run_confirmatory_simulation,
    };
    use crate::portfolio::{
        CandidateAssessment, CandidatePortfolio, ModelPrediction, PortfolioPolicy,
    };
    use crate::strict_context::{
        ContextAwareSimulationBackend, ContextBoundSimulationRequest, ContextBoundSimulationResult,
        ContextConsumptionEvidence, StrictSimulationRegistry,
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
    struct FixtureBackend {
        interval: Option<Interval>,
    }

    impl ContextAwareSimulationBackend for FixtureBackend {
        fn name(&self) -> &'static str {
            "strict-qualification-fixture"
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
                    input_digest: Some(format!("input:{}", request.request.id)),
                    output_digest: Some(format!("output:{}", request.request.id)),
                    parser_version: Some("parser-1".into()),
                });
            result.metrics = vec![SimulationMetric {
                name: "diagnostic_quality".into(),
                value: 0.88,
                unit: "1".into(),
                uncertainty: self.interval.map(|interval| {
                    UncertaintyEstimate::new(0.04, 0.02).with_interval(interval)
                }),
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

    fn selected() -> SelectedCandidate {
        let transition = DesiredTransition::simulation_only(
            "qualification-t0",
            "strict confirmatory diagnostic simulation",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let assessment = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "qualification-p0".into(),
                transition_id: "qualification-t0".into(),
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
        frontier.select("qualification-p0").unwrap()
    }

    fn claim(policy: MetricUncertaintyPolicy) -> SimulationOutcomeClaim {
        SimulationOutcomeClaim::all_criteria(
            "qualification-claim-0",
            "qualification-t0",
            "qualification-p0",
            vec![MetricCriterion {
                metric_name: "diagnostic_quality".into(),
                unit: "1".into(),
                predicate: MetricPredicate::AtLeast(0.8),
                uncertainty_policy: policy,
            }],
        )
    }

    fn confirmatory_run(
        request_id: &str,
        interval: Option<Interval>,
        policy: MetricUncertaintyPolicy,
    ) -> (ConfirmatorySimulationEvidence, SatisfiedSimulationClaim) {
        let selected = selected();
        let request = SimulationRequest::new(
            request_id,
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "strict qualification fixture",
        );
        let prepared = prepare_confirmatory_simulation(&selected, request, claim(policy)).unwrap();
        let mut registry = StrictSimulationRegistry::new();
        registry.register(FixtureBackend { interval });
        let evidence = run_confirmatory_simulation(&registry, &prepared).unwrap();
        let satisfied = match evaluate_confirmatory_claim(&evidence).unwrap() {
            ConfirmatoryClaimOutcome::Satisfied(receipt) => receipt,
            other => panic!("expected satisfied claim, got {other:?}"),
        };
        (evidence, satisfied)
    }

    fn discharged_case(
        evidence: &ConfirmatorySimulationEvidence,
        satisfied: &SatisfiedSimulationClaim,
    ) -> SafetyCase {
        let reference = required_confirmatory_safety_evidence_ref(evidence, satisfied).unwrap();
        let proposal = evidence.selection_evidence().selected().assessment().proposal.id.clone();
        let mut safety = SafetyCase::new(&proposal);
        safety.add_obligation(
            ProofObligation::new("exact confirmatory evidence", EvidenceKind::Simulation)
                .discharge(reference),
        );
        safety
    }

    #[test]
    fn interval_backed_claim_plus_exact_discharged_safety_case_qualifies() {
        let (evidence, satisfied) = confirmatory_run(
            "strict-run-a",
            Some(Interval::new(0.84, 0.92)),
            MetricUncertaintyPolicy::RequireInterval,
        );
        let safety = discharged_case(&evidence, &satisfied);
        let qualified = qualify_confirmatory_simulation(&evidence, &satisfied, &safety).unwrap();

        assert_eq!(qualified.assessment().proposal.id, "qualification-p0");
        assert_eq!(qualified.backend(), "strict-qualification-fixture");
        assert_eq!(qualified.output_digest(), "output:strict-run-a");
        assert_eq!(qualified.world_snapshot().snapshot_digest(), "a".repeat(64));
    }

    #[test]
    fn point_estimate_only_claim_is_too_weak_for_strict_qualification() {
        let (evidence, satisfied) = confirmatory_run(
            "strict-run-point",
            None,
            MetricUncertaintyPolicy::AllowPointEstimate,
        );
        let safety = discharged_case(&evidence, &satisfied);
        assert_eq!(
            qualify_confirmatory_simulation(&evidence, &satisfied, &safety).unwrap_err(),
            StrictConfirmatoryQualificationError::EvidenceTierTooWeak
        );
    }

    #[test]
    fn safety_case_must_be_bound_to_exact_confirmatory_lineage() {
        let (evidence, satisfied) = confirmatory_run(
            "strict-run-safety",
            Some(Interval::new(0.84, 0.92)),
            MetricUncertaintyPolicy::RequireInterval,
        );
        let proposal = evidence.selection_evidence().selected().assessment().proposal.id.clone();
        let mut safety = SafetyCase::new(&proposal);
        safety.add_obligation(
            ProofObligation::new("neighboring simulation", EvidenceKind::Simulation)
                .discharge("not-the-required-lineage"),
        );
        assert_eq!(
            qualify_confirmatory_simulation(&evidence, &satisfied, &safety).unwrap_err(),
            StrictConfirmatoryQualificationError::SafetyCaseMissingExactEvidence
        );
    }

    #[test]
    fn satisfied_claim_from_neighboring_run_cannot_be_substituted() {
        let (_evidence_a, satisfied_a) = confirmatory_run(
            "strict-run-a2",
            Some(Interval::new(0.84, 0.92)),
            MetricUncertaintyPolicy::RequireInterval,
        );
        let (evidence_b, _satisfied_b) = confirmatory_run(
            "strict-run-b2",
            Some(Interval::new(0.84, 0.92)),
            MetricUncertaintyPolicy::RequireInterval,
        );
        let safety = SafetyCase::new("qualification-p0");
        assert_eq!(
            qualify_confirmatory_simulation(&evidence_b, &satisfied_a, &safety).unwrap_err(),
            StrictConfirmatoryQualificationError::RunLineageMismatch
        );
    }
}
