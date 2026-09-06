// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simulation-only evidence qualification for physical-agency candidates.
//!
//! This module deliberately does **not** create physical execution authority.
//! Its strongest output is a non-serializable marker that a simulation-only
//! proposal is bound to external-solver evidence and a discharged safety case.

use serde::{Deserialize, Serialize};
use symthaea_formal_safety::SafetyCase;
use symthaea_physical_effects::{
    AuthorityClass, DesiredTransition, EffectValidationError, ProposedIntervention,
};
use symthaea_sim_bridge::SimulationResult;
use thiserror::Error;

/// Explicit binding between an unqualified proposal and the solver request used
/// as evidence for that proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationEvidenceBinding {
    pub proposal_id: String,
    pub simulation_request_id: String,
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
        Ok(())
    }
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

/// Bind a simulation-only proposal to solver-backed evidence and a discharged,
/// proposal-specific safety case.
///
/// This function does not run a solver and cannot produce an actuator permit.
pub fn qualify_simulation_candidate(
    transition: &DesiredTransition,
    proposal: &ProposedIntervention,
    binding: &SimulationEvidenceBinding,
    result: &SimulationResult,
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

    if binding.simulation_request_id != result.request_id {
        return Err(SimulationQualificationError::SimulationBindingMismatch {
            binding: binding.simulation_request_id.clone(),
            result: result.request_id.clone(),
        });
    }

    let predicted = proposal.predicted_outcome;
    if predicted.success_probability < transition.uncertainty.min_confidence
        || predicted.epistemic_uncertainty > transition.uncertainty.max_epistemic
        || predicted.aleatoric_uncertainty > transition.uncertainty.max_aleatoric
    {
        return Err(SimulationQualificationError::ProposalOutsideUncertaintyBudget);
    }

    if !result.is_engineering_evidence() {
        return Err(SimulationQualificationError::NotEngineeringEvidence);
    }

    if result.confidence < transition.uncertainty.min_confidence
        || result.uncertainty.epistemic > transition.uncertainty.max_epistemic
        || result.uncertainty.aleatoric > transition.uncertainty.max_aleatoric
    {
        return Err(SimulationQualificationError::ResultOutsideUncertaintyBudget);
    }

    // Bind the safety case to this exact proposal, not merely to a generic
    // system or neighboring candidate.
    if safety_case.subject != proposal.id {
        return Err(SimulationQualificationError::SafetyCaseSubjectMismatch {
            safety_subject: safety_case.subject.clone(),
            proposal: proposal.id.clone(),
        });
    }
    if !safety_case.is_discharged() {
        return Err(SimulationQualificationError::SafetyCaseNotDischarged);
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

    Ok(QualifiedSimulationCandidate {
        transition_id: transition.id.clone(),
        proposal_id: proposal.id.clone(),
        simulation_request_id: result.request_id.clone(),
        backend,
        solver_version,
        parser_version,
        input_digest,
        output_digest,
        safety_case_id: safety_case.id.to_string(),
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SimulationQualificationError {
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("invalid physical-effect value: {0}")]
    Effect(EffectValidationError),
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
    #[error("binding simulation request {binding:?} does not match result {result:?}")]
    SimulationBindingMismatch { binding: String, result: String },
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
    use symthaea_sim_bridge::{ExecutionMode, SimulationEvidence, UncertaintyEstimate};

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
        }
    }

    fn external_result() -> SimulationResult {
        SimulationResult::converged("sim-1", 0.9)
            .with_uncertainty(UncertaintyEstimate::new(0.1, 0.05))
            .with_metric("diagnostic_score", 0.9, "1")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("reference-solver".into()),
                solver_version: Some("1.0".into()),
                input_digest: Some("input-digest".into()),
                output_digest: Some("output-digest".into()),
                parser_version: Some("parser-1".into()),
            })
    }

    fn discharged_case(subject: &str) -> SafetyCase {
        let mut case = SafetyCase::new(subject);
        case.add_obligation(
            ProofObligation::new("simulation evidence reviewed", EvidenceKind::Simulation)
                .discharge("solver-run:sim-1"),
        );
        case
    }

    #[test]
    fn external_evidence_and_bound_discharged_case_can_qualify_simulation_only() {
        let qualified = qualify_simulation_candidate(
            &transition(),
            &proposal(),
            &binding(),
            &external_result(),
            &discharged_case("p-1"),
        )
        .unwrap();

        assert_eq!(qualified.proposal_id(), "p-1");
        assert_eq!(qualified.transition_id(), "t-1");
        assert_eq!(qualified.backend(), "reference-solver");
        assert_eq!(qualified.solver_version(), "1.0");
    }

    #[test]
    fn dry_run_cannot_qualify() {
        let dry = SimulationResult::dry_run("sim-1", "fixture", 1.0)
            .with_metric("diagnostic_score", 1.0, "1");
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &dry,
                &discharged_case("p-1"),
            ),
            Err(SimulationQualificationError::NotEngineeringEvidence)
        );
    }

    #[test]
    fn unrelated_safety_case_cannot_be_reused() {
        assert!(matches!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &external_result(),
                &discharged_case("another-proposal"),
            ),
            Err(SimulationQualificationError::SafetyCaseSubjectMismatch { .. })
        ));
    }

    #[test]
    fn open_safety_case_cannot_qualify() {
        let mut case = SafetyCase::new("p-1");
        case.add_obligation(ProofObligation::new(
            "still open",
            EvidenceKind::Simulation,
        ));
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &external_result(),
                &case,
            ),
            Err(SimulationQualificationError::SafetyCaseNotDischarged)
        );
    }

    #[test]
    fn higher_authority_proposal_is_rejected_by_simulation_qualifier() {
        let mut proposal = proposal();
        proposal.required_authority = AuthorityClass::DiagnosticExcitation;
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal,
                &binding(),
                &external_result(),
                &discharged_case("p-1"),
            ),
            Err(SimulationQualificationError::NonSimulationAuthority)
        );
    }

    #[test]
    fn evidence_binding_must_match_exact_result() {
        let bad_binding = SimulationEvidenceBinding {
            proposal_id: "p-1".into(),
            simulation_request_id: "different-run".into(),
        };
        assert!(matches!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &bad_binding,
                &external_result(),
                &discharged_case("p-1"),
            ),
            Err(SimulationQualificationError::SimulationBindingMismatch { .. })
        ));
    }

    #[test]
    fn result_outside_uncertainty_budget_is_rejected() {
        let result = SimulationResult::converged("sim-1", 0.5)
            .with_uncertainty(UncertaintyEstimate::new(0.4, 0.1))
            .with_metric("diagnostic_score", 0.5, "1")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("reference-solver".into()),
                solver_version: Some("1.0".into()),
                input_digest: Some("input-digest".into()),
                output_digest: Some("output-digest".into()),
                parser_version: Some("parser-1".into()),
            });
        assert_eq!(
            qualify_simulation_candidate(
                &transition(),
                &proposal(),
                &binding(),
                &result,
                &discharged_case("p-1"),
            ),
            Err(SimulationQualificationError::ResultOutsideUncertaintyBudget)
        );
    }
}
