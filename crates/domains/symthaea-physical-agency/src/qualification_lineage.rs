// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deliberation-bound simulation qualification.
//!
//! This is the public simulation-qualification path for PA-07. It consumes a
//! non-serializable [`SelectedCandidate`] receipt rather than an arbitrary
//! caller-supplied proposal, preserving the evaluated model/transition lineage
//! through qualification.
//!
//! The strongest output here is still simulation-only. No HAL dependency,
//! actuator command, or physical execution authority is introduced.

use crate::deliberation::SelectedCandidate;
use crate::portfolio::{CandidateAssessment, PortfolioError, PortfolioPolicy};
use crate::qualification::{
    QualifiedSimulationCandidate, SimulationEvidenceBinding, SimulationQualificationError,
    VerifiedSimulationEvidence, qualify_simulation_candidate,
};
use symthaea_formal_safety::SafetyCase;
use thiserror::Error;

/// Simulation-qualified candidate that retains the exact deliberative
/// assessment and policy that produced its selection.
///
/// Intentionally implements neither `Serialize` nor `Deserialize`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliberationBoundSimulationCandidate {
    qualified: QualifiedSimulationCandidate,
    assessment: CandidateAssessment,
    selection_policy: PortfolioPolicy,
}

impl DeliberationBoundSimulationCandidate {
    pub fn transition_id(&self) -> &str {
        self.qualified.transition_id()
    }

    pub fn proposal_id(&self) -> &str {
        self.qualified.proposal_id()
    }

    pub fn simulation_request_id(&self) -> &str {
        self.qualified.simulation_request_id()
    }

    pub fn backend(&self) -> &str {
        self.qualified.backend()
    }

    pub fn solver_version(&self) -> &str {
        self.qualified.solver_version()
    }

    pub fn parser_version(&self) -> &str {
        self.qualified.parser_version()
    }

    pub fn input_digest(&self) -> &str {
        self.qualified.input_digest()
    }

    pub fn output_digest(&self) -> &str {
        self.qualified.output_digest()
    }

    pub fn safety_case_id(&self) -> &str {
        self.qualified.safety_case_id()
    }

    pub fn assessment(&self) -> &CandidateAssessment {
        &self.assessment
    }

    pub fn selection_policy(&self) -> PortfolioPolicy {
        self.selection_policy
    }
}

/// Qualify a candidate that actually survived portfolio deliberation.
///
/// In addition to the PA-04 simulation checks, this requires the selected
/// ensemble's conservative success and effective epistemic uncertainty to fit
/// inside the transition's uncertainty budget. Model disagreement therefore
/// cannot disappear between Pareto selection and simulation qualification.
pub fn qualify_selected_simulation_candidate(
    selected: &SelectedCandidate,
    binding: &SimulationEvidenceBinding,
    evidence: &VerifiedSimulationEvidence,
    safety_case: &SafetyCase,
) -> Result<DeliberationBoundSimulationCandidate, DeliberationQualificationError> {
    let transition = selected.transition();
    let assessment = selected.assessment();

    // Revalidate the assessment even though the selection receipt could only
    // have been created from a successful portfolio evaluation. This keeps the
    // qualification function fail-closed if the internal representation later
    // evolves.
    assessment
        .validate()
        .map_err(DeliberationQualificationError::Portfolio)?;

    let conservative_success = assessment
        .conservative_success_probability()
        .map_err(DeliberationQualificationError::Portfolio)?;
    if conservative_success < transition.uncertainty.min_confidence {
        return Err(
            DeliberationQualificationError::EnsembleSuccessOutsideTransitionBudget {
                conservative_success,
                required: transition.uncertainty.min_confidence,
            },
        );
    }

    let effective_epistemic = assessment
        .effective_epistemic_uncertainty()
        .map_err(DeliberationQualificationError::Portfolio)?;
    if effective_epistemic > transition.uncertainty.max_epistemic {
        return Err(
            DeliberationQualificationError::EnsembleUncertaintyOutsideTransitionBudget {
                effective_epistemic,
                maximum: transition.uncertainty.max_epistemic,
            },
        );
    }

    let qualified = qualify_simulation_candidate(
        transition,
        &assessment.proposal,
        binding,
        evidence,
        safety_case,
    )
    .map_err(DeliberationQualificationError::Simulation)?;

    Ok(DeliberationBoundSimulationCandidate {
        qualified,
        assessment: assessment.clone(),
        selection_policy: selected.policy(),
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DeliberationQualificationError {
    #[error("invalid deliberative portfolio lineage: {0}")]
    Portfolio(PortfolioError),
    #[error("simulation qualification failed: {0}")]
    Simulation(SimulationQualificationError),
    #[error(
        "conservative ensemble success {conservative_success} is below transition minimum {required}"
    )]
    EnsembleSuccessOutsideTransitionBudget {
        conservative_success: f64,
        required: f64,
    },
    #[error(
        "effective epistemic uncertainty {effective_epistemic} exceeds transition maximum {maximum}"
    )]
    EnsembleUncertaintyOutsideTransitionBudget {
        effective_epistemic: f64,
        maximum: f64,
    },
}
