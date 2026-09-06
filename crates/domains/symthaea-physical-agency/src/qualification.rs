// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simulation-only evidence qualification for physical-agency candidates.
//!
//! This module deliberately does **not** create physical execution authority.
//! It separates two non-serializable capabilities:
//!
//! ```text
//! SimulationRegistry::run
//!     -> VerifiedSimulationEvidence
//!     -> exact proposal/run safety-case binding
//!     -> QualifiedSimulationCandidate
//! ```
//!
//! Callers cannot inject a deserialized or manually assembled `SimulationResult`
//! into the qualification boundary.

use serde::{Deserialize, Serialize};
use symthaea_formal_safety::{EvidenceKind, SafetyCase};
use symthaea_physical_effects::{
    AuthorityClass, DesiredTransition, EffectValidationError, ProposedIntervention,
};
use symthaea_sim_bridge::{SimulationRegistry, SimulationRequest};
use thiserror::Error;

/// Explicit binding between an unqualified proposal and the solver request/backend
/// expected to provide evidence for that proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationEvidenceBinding {
    pub proposal_id: String,
    pub simulation_request_id: String,
    pub expected_backend: String,
}

impl SimulationEvidenceBinding {
    pub fn validate(&self) -> Result<(), SimulationQualificationError> {
        if self.proposal_id.trim().is_empty() {
            return Err(SimulationQualificationError::EmptyField(
                "binding.proposal_id",
            ));
        }
        if self.simulation_request_id.trim().is_empty() {
            return Err(SimulationQualificationError::EmptyField(
                "binding.simulation_request_id",
            ));
        }
        if self.expected_backend.trim().is_empty() {
            return Err(SimulationQualificationError::EmptyField(
                "binding.expected_backend",
            ));
        }
        Ok(())
    }
}

/// Registry-produced external-solver evidence.
///
/// This type intentionally implements neither `Serialize` nor `Deserialize`.
/// Its fields are private and the only constructor path in this crate is
/// [`execute_verified_simulation`].
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedSimulationEvidence {
    request_id: String,
    backend: String,
    solver_version: String,
    parser_version: String,
    input_digest: String,
    output_digest: String,
    confidence: f64,
    epistemic_uncertainty: f64,
    aleatoric_uncertainty: f64,
}

impl VerifiedSimulationEvidence {
    pub fn request_id(&self) -> &str {
        &self.request_id
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

    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    pub fn epistemic_uncertainty(&self) -> f64 {
        self.epistemic_uncertainty
    }

    pub fn aleatoric_uncertainty(&self) -> f64 {
        self.aleatoric_uncertainty
    }

    /// Canonical reference a discharged simulation obligation must contain to
    /// bind the safety case to this exact normalized solver output.
    pub fn safety_evidence_ref(&self) -> String {
        format!(
            "physical-agency-simulation:{}:{}:{}",
            self.backend, self.request_id, self.output_digest
        )
    }
}

/// Execute a normalized request through the existing registry and convert only
/// real external-solver engineering evidence into a non-serializable receipt.
///
/// This is the only place in PA-04 that runs a solver. It cannot reach HAL or
/// any physical actuator boundary.
pub fn execute_verified_simulation(
    registry: &SimulationRegistry,
    request: &SimulationRequest,
) -> Result<VerifiedSimulationEvidence, SimulationQualificationError> {
    let result = registry
        .run(request)
        .map_err(|error| SimulationQualificationError::SimulationRun(error.to_string()))?;

    if !result.is_engineering_evidence() {
        return Err(SimulationQualificationError::NotEngineeringEvidence);
    }

    let evidence = &result.evidence;
    let backend = evidence
        .backend
        .clone()
        .ok_or(SimulationQualificationError::IncompleteProvenance)?;
    let solver_version = evidence
        .solver_version
        .clone()
        .ok_or(SimulationQualificationError::IncompleteProvenance)?;
    let parser_version = evidence
        .parser_version
        .clone()
        .ok_or(SimulationQualificationError::IncompleteProvenance)?;
    let input_digest = evidence
        .input_digest
        .clone()
        .ok_or(SimulationQualificationError::IncompleteProvenance)?;
    let output_digest = evidence
        .output_digest
        .clone()
        .ok_or(SimulationQualificationError::IncompleteProvenance)?;

    Ok(VerifiedSimulationEvidence {
        request_id: result.request_id,
        backend,
        solver_version,
        parser_version,
        input_digest,
        output_digest,
        confidence: result.confidence,
        epistemic_uncertainty: result.uncertainty.epistemic,
        aleatoric_uncertainty: result.uncertainty.aleatoric,
    })
}

/// Evidence-backed marker for a proposal that remains simulation-only.
///
/// This type intentionally implements neither `Serialize` nor `Deserialize`.
/// Its fields are private and it can only be constructed by
/// [`qualify_simulation_candidate`]. Serialized bytes therefore cannot mint a
/// qualified value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedSimulationCandidate {
    transition_id: String,
    proposal_id: String,
    simulation_request_id: String,
    backend: String,
    solver_version: String,
    parser_version: String,
    input_digest: String,
    output_digest: String,
    safety_case_id: String,
}

impl QualifiedSimulationCandidate {
    pub fn transition_id(&self) -> &str {
        &self.transition_id
    }

    pub fn proposal_id(&self) -> &str {
        &self.proposal_id
    }

    pub fn simulation_request_id(&self) -> &str {
        &self.simulation_request_id
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

    pub fn safety_case_id(&self) -> &str {
        &self.safety_case_id
    }
}

/// Bind a simulation-only proposal to registry-produced evidence and a
/// discharged, proposal-specific safety case that cites the exact output digest.
pub fn qualify_simulation_candidate(
    transition: &DesiredTransition,
    proposal: &ProposedIntervention,
    binding: &SimulationEvidenceBinding,
    evidence: &VerifiedSimulationEvidence,
    safety_case: &SafetyCase,
) -> Result<QualifiedSimulationCandidate, SimulationQualificationError> {
    transition
        .validate()
        .map_err(SimulationQualificationError::Effect)?;
    proposal
        .validate()
        .map_err(SimulationQualificationError::Effect)?;
    binding.validate()?;

    if transition.required_authority != AuthorityClass::SimulationOnly
        || proposal.required_authority != AuthorityClass::SimulationOnly
    {
        return Err(SimulationQualificationError::NonSimulationAuthority);
    }

    if proposal.transition_id != transition.id {
        return Err(SimulationQualificationError::TransitionMismatch {
            transition: transition.id.clone(),
            proposal_transition: proposal.transition_id.clone(),
        });
    }

    if !transition
        .allowed_modalities
        .contains(&proposal.mechanism.modality)
    {
        return Err(SimulationQualificationError::ModalityNotAllowed);
    }

    if binding.proposal_id != proposal.id {
        return Err(SimulationQualificationError::ProposalBindingMismatch {
            binding: binding.proposal_id.clone(),
            proposal: proposal.id.clone(),
        });
    }

    if binding.simulation_request_id != evidence.request_id {
        return Err(SimulationQualificationError::SimulationBindingMismatch {
            binding: binding.simulation_request_id.clone(),
            evidence: evidence.request_id.clone(),
        });
    }

    if binding.expected_backend != evidence.backend {
        return Err(SimulationQualificationError::BackendBindingMismatch {
            expected: binding.expected_backend.clone(),
            actual: evidence.backend.clone(),
        });
    }

    let predicted = proposal.predicted_outcome;
    if predicted.success_probability < transition.uncertainty.min_confidence
        || predicted.epistemic_uncertainty > transition.uncertainty.max_epistemic
        || predicted.aleatoric_uncertainty > transition.uncertainty.max_aleatoric
    {
        return Err(SimulationQualificationError::ProposalOutsideUncertaintyBudget);
    }

    if evidence.confidence < transition.uncertainty.min_confidence
        || evidence.epistemic_uncertainty > transition.uncertainty.max_epistemic
        || evidence.aleatoric_uncertainty > transition.uncertainty.max_aleatoric
    {
        return Err(SimulationQualificationError::ResultOutsideUncertaintyBudget);
    }

    if safety_case.subject != proposal.id {
        return Err(SimulationQualificationError::SafetyCaseSubjectMismatch {
            safety_subject: safety_case.subject.clone(),
            proposal: proposal.id.clone(),
        });
    }
    if !safety_case.is_discharged() {
        return Err(SimulationQualificationError::SafetyCaseNotDischarged);
    }

    let required_ref = evidence.safety_evidence_ref();
    let exact_evidence_bound = safety_case.obligations.iter().any(|obligation| {
        obligation.expected_evidence == EvidenceKind::Simulation
            && obligation.evidence_refs.iter().any(|reference| reference == &required_ref)
    });
    if !exact_evidence_bound {
        return Err(SimulationQualificationError::SafetyCaseMissingExactEvidence);
    }

    Ok(QualifiedSimulationCandidate {
        transition_id: transition.id.clone(),
        proposal_id: proposal.id.clone(),
        simulation_request_id: evidence.request_id.clone(),
        backend: evidence.backend.clone(),
        solver_version: evidence.solver_version.clone(),
        parser_version: evidence.parser_version.clone(),
        input_digest: evidence.input_digest.clone(),
        output_digest: evidence.output_digest.clone(),
        safety_case_id: safety_case.id.to_string(),
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SimulationQualificationError {
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("invalid physical-effect value: {0}")]
    Effect(EffectValidationError),
    #[error("simulation registry execution failed: {0}")]
    SimulationRun(String),
    #[error("simulation qualification accepts SimulationOnly authority only")]
    NonSimulationAuthority,
    #[error("proposal transition {proposal_transition:?} does not match {transition:?}")]
    TransitionMismatch {
        transition: String,
        proposal_transition: String,
    },
    #[error("proposal mechanism modality is not allowed by the desired transition")]
    ModalityNotAllowed,
    #[error("binding proposal {binding:?} does not match proposal {proposal:?}")]
    ProposalBindingMismatch { binding: String, proposal: String },
    #[error("binding simulation request {binding:?} does not match verified evidence {evidence:?}")]
    SimulationBindingMismatch { binding: String, evidence: String },
    #[error("binding expected backend {expected:?} but verified evidence came from {actual:?}")]
    BackendBindingMismatch { expected: String, actual: String },
    #[error("proposal prediction falls outside the transition uncertainty budget")]
    ProposalOutsideUncertaintyBudget,
    #[error("simulation result is not external-solver engineering evidence")]
    NotEngineeringEvidence,
    #[error("simulation result falls outside the transition uncertainty budget")]
    ResultOutsideUncertaintyBudget,
    #[error("safety case subject {safety_subject:?} is not proposal {proposal:?}")]
    SafetyCaseSubjectMismatch {
        safety_subject: String,
        proposal: String,
    },
    #[error("proposal-bound safety case is not fully discharged")]
    SafetyCaseNotDischarged,
    #[error("proposal-bound safety case does not cite the exact verified simulation output")]
    SafetyCaseMissingExactEvidence,
    #[error("external-solver evidence has incomplete provenance")]
    IncompleteProvenance,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation};
    use symthaea_physical_effects::{
        EffectKind, MechanismRef, PhysicalModality, PredictedOutcome, TargetRegion,
    };
    use symthaea_sim_bridge::{
        ExecutionMode, SimulationBackend, SimulationError, SimulationEvidence, SimulationResult,
        SolverKind, UncertaintyEstimate,
    };

    #[derive(Debug)]
    struct ExternalBackend {
        epistemic: f64,
        aleatoric: f64,
    }

    impl SimulationBackend for ExternalBackend {
        fn name(&self) -> &'static str {
            "reference-solver"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::FiniteElement]
        }

        fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
            Ok(SimulationResult::converged(&request.id, 0.95)
                .with_uncertainty(UncertaintyEstimate::new(self.epistemic, self.aleatoric))
                .with_metric("diagnostic_score", 0.9, "1")
                .with_external_evidence(SimulationEvidence {
                    mode: ExecutionMode::ExternalSolver,
                    backend: Some(self.name().into()),
                    solver_version: Some("1.0".into()),
                    input_digest: Some("input-digest".into()),
                    output_digest: Some("output-digest".into()),
                    parser_version: Some("parser-1".into()),
                }))
        }
    }

    #[derive(Debug)]
    struct DryBackend;

    impl SimulationBackend for DryBackend {
        fn name(&self) -> &'static str {
            "dry-solver"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::FiniteElement]
        }

        fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
            Ok(SimulationResult::dry_run(&request.id, self.name(), 1.0)
                .with_metric("diagnostic_score", 1.0, "1"))
        }
    }

    fn request() -> SimulationRequest {
        SimulationRequest::new(
            "sim-1",
            symthaea_sim_bridge::EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
            "simulation-only diagnostic evidence",
        )
    }

    fn verified_evidence() -> VerifiedSimulationEvidence {
        let mut registry = SimulationRegistry::new();
        registry.register(ExternalBackend {
            epistemic: 0.1,
            aleatoric: 0.05,
        });
        execute_verified_simulation(&registry, &request()).unwrap()
    }

    fn transition() -> DesiredTransition {
        let mut transition = DesiredTransition::simulation_only(
            "t-1",
            "compare a simulated diagnostic mechanism",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        transition.uncertainty.min_confidence = 0.8;
        transition.uncertainty.max_epistemic = 0.2;
        transition.uncertainty.max_aleatoric = 0.2;
        transition
    }

    fn proposal() -> ProposedIntervention {
        ProposedIntervention {
            id: "p-1".into(),
            transition_id: "t-1".into(),
            mechanism: MechanismRef {
                backend: "reference".into(),
                mechanism: "simulated-acoustic-probe".into(),
                modality: PhysicalModality::Acoustic,
            },
            required_authority: AuthorityClass::SimulationOnly,
            predicted_outcome: PredictedOutcome {
                success_probability: 0.9,
                epistemic_uncertainty: 0.1,
                aleatoric_uncertainty: 0.05,
            },
        }
    }

    fn binding() -> SimulationEvidenceBinding {
        SimulationEvidenceBinding {
            proposal_id: "p-1".into(),
            simulation_request_id: "sim-1".into(),
            expected_backend: "reference-solver".into(),
        }
    }

    fn discharged_case(subject: &str, exact_ref: &str) -> SafetyCase {
        let mut case = SafetyCase::new(subject);
        case.add_obligation(
            ProofObligation::new("simulation evidence reviewed", EvidenceKind::Simulation)
                .discharge(exact_ref),
        );
        case
    }

    #[test]
    fn registry_verified_evidence_and_exact_bound_case_can_qualify() {
        let evidence = verified_evidence();
        let case = discharged_case("p-1", &evidence.safety_evidence_ref());
        let qualified = qualify_simulation_candidate(
            &transition(),
            &proposal(),
            &binding(),
            &evidence,
            &case,
        )
        .unwrap();

        assert_eq!(qualified.proposal_id(), "p-1");
        assert_eq!(qualified.transition_id(), "t-1");
        assert_eq!(qualified.backend(), "reference-solver");
        assert_eq!(qualified.output_digest(), "output-digest");
    }

    #[test]
    fn dry_run_cannot_become_verified_evidence() {
        let mut registry = SimulationRegistry::new();
        registry.register(DryBackend);
        assert_eq!(
            execute_verified_simulation(&registry, &request()),
            Err(SimulationQualificationError::NotEngineeringEvidence)
        );
    }

    #[test]
    fn safety_case_must_cite_exact_verified_output() {
        let evidence = verified_evidence();
        let case = discharged_case("p-1", "solver-run:some-neighboring-result");
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &evidence,
                &case,
            ),
            Err(SimulationQualificationError::SafetyCaseMissingExactEvidence)
        );
    }

    #[test]
    fn unrelated_safety_case_cannot_be_reused() {
        let evidence = verified_evidence();
        let case = discharged_case("another-proposal", &evidence.safety_evidence_ref());
        assert!(matches!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &evidence,
                &case,
            ),
            Err(SimulationQualificationError::SafetyCaseSubjectMismatch { .. })
        ));
    }

    #[test]
    fn higher_authority_proposal_is_rejected_by_simulation_qualifier() {
        let evidence = verified_evidence();
        let case = discharged_case("p-1", &evidence.safety_evidence_ref());
        let mut proposal = proposal();
        proposal.required_authority = AuthorityClass::DiagnosticExcitation;
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal,
                &binding(),
                &evidence,
                &case,
            ),
            Err(SimulationQualificationError::NonSimulationAuthority)
        );
    }

    #[test]
    fn expected_backend_is_bound_exactly() {
        let evidence = verified_evidence();
        let case = discharged_case("p-1", &evidence.safety_evidence_ref());
        let mut binding = binding();
        binding.expected_backend = "another-backend".into();
        assert!(matches!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding,
                &evidence,
                &case,
            ),
            Err(SimulationQualificationError::BackendBindingMismatch { .. })
        ));
    }

    #[test]
    fn verified_result_outside_uncertainty_budget_is_rejected() {
        let mut registry = SimulationRegistry::new();
        registry.register(ExternalBackend {
            epistemic: 0.4,
            aleatoric: 0.1,
        });
        let evidence = execute_verified_simulation(&registry, &request()).unwrap();
        let case = discharged_case("p-1", &evidence.safety_evidence_ref());
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &evidence,
                &case,
            ),
            Err(SimulationQualificationError::ResultOutsideUncertaintyBudget)
        );
    }
}
