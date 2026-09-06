// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deliberation-bound simulation qualification.
//!
//! This is the public simulation-qualification path for PA-08. It consumes a
//! non-serializable [`SelectedCandidate`] receipt rather than an arbitrary
//! caller-supplied proposal, preserving the evaluated model/transition/world
//! snapshot lineage through qualification.
//!
//! The strongest output here is still simulation-only. No HAL dependency,
//! actuator command, or physical execution authority is introduced.

use crate::deliberation::{SelectedCandidate, WorldSnapshotRef};
use crate::portfolio::{CandidateAssessment, PortfolioError, PortfolioPolicy};
use crate::qualification::{
    QualifiedSimulationCandidate, SimulationEvidenceBinding, SimulationQualificationError,
    VerifiedSimulationEvidence, qualify_simulation_candidate,
};
use serde::{Deserialize, Serialize};
use symthaea_formal_safety::SafetyCase;
use thiserror::Error;

/// Public binding between a selected deliberation receipt and one simulation
/// request/backend/world-snapshot lineage.
///
/// This is ordinary serializable planning data, not authority. The strict
/// qualifier checks it against the non-serializable `SelectedCandidate` before
/// translating it to the crate-private PA-04 evidence binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliberationSimulationBinding {
    proposal_id: String,
    simulation_request_id: String,
    expected_backend: String,
    world_snapshot_digest: String,
}

impl DeliberationSimulationBinding {
    pub fn new(
        proposal_id: impl Into<String>,
        simulation_request_id: impl Into<String>,
        expected_backend: impl Into<String>,
        world_snapshot_digest: impl Into<String>,
    ) -> Self {
        Self {
            proposal_id: proposal_id.into(),
            simulation_request_id: simulation_request_id.into(),
            expected_backend: expected_backend.into(),
            world_snapshot_digest: world_snapshot_digest.into(),
        }
    }

    pub fn proposal_id(&self) -> &str {
        &self.proposal_id
    }

    pub fn simulation_request_id(&self) -> &str {
        &self.simulation_request_id
    }

    pub fn expected_backend(&self) -> &str {
        &self.expected_backend
    }

    pub fn world_snapshot_digest(&self) -> &str {
        &self.world_snapshot_digest
    }

    fn validate(&self) -> Result<(), DeliberationQualificationError> {
        for (field, value) in [
            ("binding.proposal_id", self.proposal_id.as_str()),
            (
                "binding.simulation_request_id",
                self.simulation_request_id.as_str(),
            ),
            ("binding.expected_backend", self.expected_backend.as_str()),
            (
                "binding.world_snapshot_digest",
                self.world_snapshot_digest.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(DeliberationQualificationError::EmptyBindingField(field));
            }
        }
        Ok(())
    }
}

/// Simulation-qualified candidate that retains the exact deliberative
/// assessment, policy, and world snapshot that produced its selection.
///
/// Intentionally implements neither `Serialize` nor `Deserialize`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliberationBoundSimulationCandidate {
    qualified: QualifiedSimulationCandidate,
    world_snapshot: WorldSnapshotRef,
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

    pub fn world_snapshot(&self) -> &WorldSnapshotRef {
        &self.world_snapshot
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
/// inside the transition's uncertainty budget and the simulation binding to
/// cite the exact world snapshot used at deliberation. Model disagreement and
/// world-state identity therefore cannot disappear before qualification.
pub fn qualify_selected_simulation_candidate(
    selected: &SelectedCandidate,
    binding: &DeliberationSimulationBinding,
    evidence: &VerifiedSimulationEvidence,
    safety_case: &SafetyCase,
) -> Result<DeliberationBoundSimulationCandidate, DeliberationQualificationError> {
    binding.validate()?;

    let transition = selected.transition();
    let assessment = selected.assessment();

    if binding.world_snapshot_digest != selected.world_snapshot().snapshot_digest() {
        return Err(DeliberationQualificationError::WorldSnapshotBindingMismatch {
            selected: selected.world_snapshot().snapshot_digest().to_string(),
            binding: binding.world_snapshot_digest.clone(),
        });
    }

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

    let private_binding = SimulationEvidenceBinding {
        proposal_id: binding.proposal_id.clone(),
        simulation_request_id: binding.simulation_request_id.clone(),
        expected_backend: binding.expected_backend.clone(),
    };
    let qualified = qualify_simulation_candidate(
        transition,
        &assessment.proposal,
        &private_binding,
        evidence,
        safety_case,
    )
    .map_err(DeliberationQualificationError::Simulation)?;

    Ok(DeliberationBoundSimulationCandidate {
        qualified,
        world_snapshot: selected.world_snapshot().clone(),
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
    #[error("required deliberation binding field is empty: {0}")]
    EmptyBindingField(&'static str),
    #[error(
        "simulation binding world snapshot {binding:?} does not match selected snapshot {selected:?}"
    )]
    WorldSnapshotBindingMismatch { selected: String, binding: String },
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
