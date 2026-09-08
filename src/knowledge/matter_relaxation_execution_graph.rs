// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Restart-aware execution graph for structural relaxation evidence.
//!
//! A converged crystal structure may require more than one external solver run.
//! This module models a relaxation as an ordered chain of leaf executions whose
//! structure outputs feed the next step. The final admitted force diagnostic must
//! be bound to the final step; an earlier, easier step cannot satisfy the result.
//!
//! All chronology remains declared only. Process success is not convergence: the
//! numerical criterion admitted by `matter_crystal_phase` remains authoritative.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::MatterClaim;
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    Sha256Digest,
};

use super::matter_crystal_evidence_binding::{
    bind_crystal_phase_evidence, CrystalEvidenceBindingError, EvidenceBoundCrystalPhaseMatterClaim,
};
use super::matter_crystal_execution::{
    CrystalExecutionError, CrystalSolverExecutionBinding, CrystalSolverExecutionReceipt,
    CrystalSolverTask,
};
use super::matter_crystal_leaf_binding::bind_crystal_solver_leaf;
use super::matter_crystal_phase::{CrystalPhaseDiagnosticSnapshot, CrystalPhaseEvidenceSlot};

#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationStepReceipt {
    pub step_index: u32,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionReceipt,
    pub input_structure_artifact_id: String,
    pub output_structure_artifact_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralRelaxationResultReceipt {
    pub diagnostic_artifact_id: String,
    pub final_structure_artifact_id: String,
    pub declared_result_yyyymmdd: u32,
    pub max_force_bits: u64,
    pub force_tolerance_bits: u64,
}

impl StructuralRelaxationResultReceipt {
    pub fn validate(&self) -> Result<(), RelaxationExecutionGraphError> {
        for (field, value) in [
            ("diagnostic_artifact_id", self.diagnostic_artifact_id.as_str()),
            ("final_structure_artifact_id", self.final_structure_artifact_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RelaxationExecutionGraphError::EmptyField(field));
            }
        }
        if self.declared_result_yyyymmdd == 0 {
            return Err(RelaxationExecutionGraphError::MissingResultDate);
        }
        let force = f64::from_bits(self.max_force_bits);
        let tolerance = f64::from_bits(self.force_tolerance_bits);
        if !force.is_finite() || force < 0.0 {
            return Err(RelaxationExecutionGraphError::InvalidMaxForce);
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(RelaxationExecutionGraphError::InvalidForceTolerance);
        }
        if force > tolerance {
            return Err(RelaxationExecutionGraphError::ForceCriterionNotMet);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralRelaxationExecutionGraphReceipt {
    pub target_claim_id: String,
    pub steps: Vec<RelaxationStepReceipt>,
    pub result: StructuralRelaxationResultReceipt,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationStepBinding {
    pub step_index: u32,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionBinding,
    pub input_structure_evidence_id: String,
    pub input_structure_sha256: Sha256Digest,
    pub output_structure_evidence_id: String,
    pub output_structure_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralRelaxationResultBinding {
    pub diagnostic_evidence_id: String,
    pub diagnostic_sha256: Sha256Digest,
    pub final_structure_evidence_id: String,
    pub final_structure_sha256: Sha256Digest,
    pub result_claimed_date: ClaimedUtcDate,
    pub max_force_bits: u64,
    pub force_tolerance_bits: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralRelaxationExecutionGraphBinding {
    pub evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    pub target_claim_id: String,
    pub steps: Vec<RelaxationStepBinding>,
    pub result: StructuralRelaxationResultBinding,
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

pub fn bind_structural_relaxation_execution_graph(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    graph: StructuralRelaxationExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<StructuralRelaxationExecutionGraphBinding, RelaxationExecutionGraphError> {
    let rebound = bind_crystal_phase_evidence(evidence_bound.admitted.clone(), bundle)?;
    if rebound != evidence_bound {
        return Err(RelaxationExecutionGraphError::ParentEvidenceBindingMismatch);
    }

    let relaxation_snapshot = evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
        .ok_or(RelaxationExecutionGraphError::MissingRelaxationClaim)?;
    let target_claim = &relaxation_snapshot.claim;
    if graph.target_claim_id != target_claim.claim_id {
        return Err(RelaxationExecutionGraphError::TargetClaimIdentityMismatch {
            graph: graph.target_claim_id.clone(),
            claim: target_claim.claim_id.clone(),
        });
    }

    let (expected_force_bits, expected_tolerance_bits) = match &relaxation_snapshot.diagnostic {
        CrystalPhaseDiagnosticSnapshot::StructuralRelaxation {
            max_force_bits,
            force_tolerance_bits,
            ..
        } => (*max_force_bits, *force_tolerance_bits),
        _ => return Err(RelaxationExecutionGraphError::RelaxationDiagnosticMismatch),
    };

    graph.result.validate()?;
    if graph.result.max_force_bits != expected_force_bits {
        return Err(RelaxationExecutionGraphError::MaxForceMismatch);
    }
    if graph.result.force_tolerance_bits != expected_tolerance_bits {
        return Err(RelaxationExecutionGraphError::ForceToleranceMismatch);
    }
    if graph.steps.is_empty() {
        return Err(RelaxationExecutionGraphError::EmptyRelaxationSequence);
    }

    let criterion = evidence_bound
        .criterion_bindings
        .iter()
        .find(|binding| binding.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
        .ok_or(RelaxationExecutionGraphError::MissingRelaxationCriterion)?;

    validate_step_shape(&graph.steps, &graph.target_claim_id)?;

    let mut bound_steps = Vec::with_capacity(graph.steps.len());
    let mut chronology = Vec::with_capacity(graph.steps.len() + 1);
    let mut execution_ids = BTreeSet::new();
    let mut execution_digests = BTreeSet::new();
    let mut any_step_ood = false;
    let mut previous_finished_ms: Option<u64> = None;
    let mut previous_output_digest: Option<Sha256Digest> = None;

    for step in &graph.steps {
        if step.execution.task != CrystalSolverTask::StructuralRelaxation {
            return Err(RelaxationExecutionGraphError::StepIsNotRelaxationTask);
        }
        if !execution_ids.insert(step.execution.execution_evidence_id.clone()) {
            return Err(RelaxationExecutionGraphError::DuplicateExecutionIdentity(
                step.execution.execution_evidence_id.clone(),
            ));
        }

        let execution = bind_crystal_solver_leaf(&step.execution, &step.claim, bundle)?;
        if !execution_digests.insert(execution.execution_content_sha256.as_str().to_string()) {
            return Err(RelaxationExecutionGraphError::DuplicateExecutionContent);
        }

        if let Some(previous_finished_ms) = previous_finished_ms {
            if execution.started_unix_ms < previous_finished_ms {
                return Err(RelaxationExecutionGraphError::OverlappingRelaxationSteps {
                    step_index: step.step_index,
                });
            }
        }
        previous_finished_ms = Some(execution.finished_unix_ms);

        let input = bundle.require_role(
            &step.input_structure_artifact_id,
            EvidenceRole::ArtifactContent,
        )?;
        let output = bundle.require_role(
            &step.output_structure_artifact_id,
            EvidenceRole::ArtifactContent,
        )?;
        let input_date = input.claimed_utc_date.ok_or_else(|| {
            RelaxationExecutionGraphError::MissingClaimedDate(
                step.input_structure_artifact_id.clone(),
            )
        })?;
        let output_date = output.claimed_utc_date.ok_or_else(|| {
            RelaxationExecutionGraphError::MissingClaimedDate(
                step.output_structure_artifact_id.clone(),
            )
        })?;
        if input_date > execution.execution_claimed_date {
            return Err(RelaxationExecutionGraphError::InputStructurePostdatesExecution {
                step_index: step.step_index,
            });
        }
        if output_date < execution.execution_claimed_date {
            return Err(RelaxationExecutionGraphError::OutputStructurePredatesExecution {
                step_index: step.step_index,
            });
        }

        if let Some(previous_digest) = &previous_output_digest {
            if previous_digest != &input.content_sha256 {
                return Err(RelaxationExecutionGraphError::BrokenStructureHandoff {
                    step_index: step.step_index,
                });
            }
        }
        previous_output_digest = Some(output.content_sha256.clone());

        if step
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
        {
            any_step_ood = true;
        }

        chronology.push(bundle.require_preregistration_before(
            &criterion.evidence_id,
            &execution.execution_evidence_id,
        )?);
        bound_steps.push(RelaxationStepBinding {
            step_index: step.step_index,
            claim: step.claim.clone(),
            execution,
            input_structure_evidence_id: input.id.as_str().to_string(),
            input_structure_sha256: input.content_sha256.clone(),
            output_structure_evidence_id: output.id.as_str().to_string(),
            output_structure_sha256: output.content_sha256.clone(),
        });
    }

    if any_step_ood
        && !target_claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(RelaxationExecutionGraphError::StepOodNotPropagated);
    }

    let final_step = bound_steps
        .last()
        .ok_or(RelaxationExecutionGraphError::EmptyRelaxationSequence)?;
    if final_step.claim.claim_id != target_claim.claim_id {
        return Err(RelaxationExecutionGraphError::FinalStepIsNotTargetClaim);
    }
    if graph.result.final_structure_artifact_id != final_step.output_structure_evidence_id {
        return Err(RelaxationExecutionGraphError::FinalStructureIdentityMismatch);
    }

    let diagnostic = bundle.require_role(
        &graph.result.diagnostic_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let final_structure = bundle.require_role(
        &graph.result.final_structure_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let result_date = diagnostic.claimed_utc_date.ok_or_else(|| {
        RelaxationExecutionGraphError::MissingClaimedDate(
            graph.result.diagnostic_artifact_id.clone(),
        )
    })?;
    if result_date.yyyymmdd() != graph.result.declared_result_yyyymmdd {
        return Err(RelaxationExecutionGraphError::ResultDateMismatch {
            receipt_yyyymmdd: graph.result.declared_result_yyyymmdd,
            evidence_yyyymmdd: result_date.yyyymmdd(),
        });
    }
    if result_date < final_step.execution.execution_claimed_date {
        return Err(RelaxationExecutionGraphError::ResultPredatesFinalExecution);
    }
    if diagnostic.content_sha256 == final_structure.content_sha256 {
        return Err(RelaxationExecutionGraphError::DiagnosticAliasesFinalStructure);
    }
    if !target_claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == diagnostic.id.as_str())
    {
        return Err(RelaxationExecutionGraphError::DiagnosticMissingFromTargetClaim);
    }
    if !target_claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == final_structure.id.as_str())
    {
        return Err(RelaxationExecutionGraphError::FinalStructureMissingFromTargetClaim);
    }

    chronology.push(bundle.require_preregistration_before(
        &criterion.evidence_id,
        diagnostic.id.as_str(),
    )?);

    Ok(StructuralRelaxationExecutionGraphBinding {
        evidence_bound,
        target_claim_id: graph.target_claim_id,
        steps: bound_steps,
        result: StructuralRelaxationResultBinding {
            diagnostic_evidence_id: diagnostic.id.as_str().to_string(),
            diagnostic_sha256: diagnostic.content_sha256.clone(),
            final_structure_evidence_id: final_structure.id.as_str().to_string(),
            final_structure_sha256: final_structure.content_sha256.clone(),
            result_claimed_date: result_date,
            max_force_bits: graph.result.max_force_bits,
            force_tolerance_bits: graph.result.force_tolerance_bits,
        },
        criterion_chronology: chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn validate_step_shape(
    steps: &[RelaxationStepReceipt],
    target_claim_id: &str,
) -> Result<(), RelaxationExecutionGraphError> {
    let mut claim_ids = BTreeSet::new();
    for (expected_index, step) in steps.iter().enumerate() {
        let expected_index = u32::try_from(expected_index)
            .map_err(|_| RelaxationExecutionGraphError::StepCountOverflow)?;
        if step.step_index != expected_index {
            return Err(RelaxationExecutionGraphError::NonContiguousStepIndex {
                expected: expected_index,
                actual: step.step_index,
            });
        }
        if step.input_structure_artifact_id.trim().is_empty()
            || step.output_structure_artifact_id.trim().is_empty()
        {
            return Err(RelaxationExecutionGraphError::EmptyStructureIdentity);
        }
        if !claim_ids.insert(step.claim.claim_id.clone()) {
            return Err(RelaxationExecutionGraphError::DuplicateStepClaimIdentity(
                step.claim.claim_id.clone(),
            ));
        }
        if expected_index + 1 < steps.len() as u32 && step.claim.claim_id == target_claim_id {
            return Err(RelaxationExecutionGraphError::TargetClaimAppearsBeforeFinalStep);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelaxationExecutionGraphError {
    CrystalEvidence(CrystalEvidenceBindingError),
    ExternalEvidence(ExternalEvidenceError),
    LeafExecution(CrystalExecutionError),
    EmptyField(&'static str),
    MissingResultDate,
    InvalidMaxForce,
    InvalidForceTolerance,
    ForceCriterionNotMet,
    ParentEvidenceBindingMismatch,
    MissingRelaxationClaim,
    TargetClaimIdentityMismatch { graph: String, claim: String },
    RelaxationDiagnosticMismatch,
    MaxForceMismatch,
    ForceToleranceMismatch,
    EmptyRelaxationSequence,
    MissingRelaxationCriterion,
    StepCountOverflow,
    NonContiguousStepIndex { expected: u32, actual: u32 },
    EmptyStructureIdentity,
    DuplicateStepClaimIdentity(String),
    TargetClaimAppearsBeforeFinalStep,
    StepIsNotRelaxationTask,
    DuplicateExecutionIdentity(String),
    DuplicateExecutionContent,
    OverlappingRelaxationSteps { step_index: u32 },
    MissingClaimedDate(String),
    InputStructurePostdatesExecution { step_index: u32 },
    OutputStructurePredatesExecution { step_index: u32 },
    BrokenStructureHandoff { step_index: u32 },
    StepOodNotPropagated,
    FinalStepIsNotTargetClaim,
    FinalStructureIdentityMismatch,
    ResultDateMismatch { receipt_yyyymmdd: u32, evidence_yyyymmdd: u32 },
    ResultPredatesFinalExecution,
    DiagnosticAliasesFinalStructure,
    DiagnosticMissingFromTargetClaim,
    FinalStructureMissingFromTargetClaim,
}

impl From<CrystalEvidenceBindingError> for RelaxationExecutionGraphError {
    fn from(value: CrystalEvidenceBindingError) -> Self {
        Self::CrystalEvidence(value)
    }
}

impl From<ExternalEvidenceError> for RelaxationExecutionGraphError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl From<CrystalExecutionError> for RelaxationExecutionGraphError {
    fn from(value: CrystalExecutionError) -> Self {
        Self::LeafExecution(value)
    }
}

impl fmt::Display for RelaxationExecutionGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "structural relaxation execution graph rejected: {self:?}")
    }
}

impl std::error::Error for RelaxationExecutionGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_receipt_requires_admitted_force_bound() {
        let good = StructuralRelaxationResultReceipt {
            diagnostic_artifact_id: "diag".into(),
            final_structure_artifact_id: "structure".into(),
            declared_result_yyyymmdd: 20260908,
            max_force_bits: 0.01_f64.to_bits(),
            force_tolerance_bits: 0.02_f64.to_bits(),
        };
        good.validate().unwrap();

        let bad = StructuralRelaxationResultReceipt {
            max_force_bits: 0.03_f64.to_bits(),
            ..good
        };
        assert_eq!(
            bad.validate().unwrap_err(),
            RelaxationExecutionGraphError::ForceCriterionNotMet
        );
    }

    #[test]
    fn sequence_shape_is_zero_based_and_contiguous() {
        let claim = MatterClaim::local_computational(
            "claim:step",
            "step",
            symthaea_epistemic_types::MatterScale::Crystal,
            symthaea_epistemic_types::MatterValidationStage::PhysicsSimulated,
            symthaea_epistemic_types::MatterSolverProfile {
                name: "solver".into(),
                version: "1".into(),
                method: "relax".into(),
                capabilities: Vec::new(),
                limitations: Vec::new(),
            },
        )
        .unwrap();
        let execution = CrystalSolverExecutionReceipt {
            claim_id: "claim:step".into(),
            task: CrystalSolverTask::StructuralRelaxation,
            execution_evidence_id: "exec".into(),
            declared_execution_yyyymmdd: 20260908,
            started_unix_ms: 1,
            finished_unix_ms: 2,
            exit_code: 0,
            solver_name: "solver".into(),
            solver_version: "1".into(),
            adapter_name: "adapter".into(),
            adapter_version: "1".into(),
            input_artifact_id: "input".into(),
            configuration_artifact_id: "config".into(),
            environment_snapshot_id: "env".into(),
            parser_snapshot_id: "parser".into(),
            output_artifact_ids: vec!["out".into()],
            dependency_snapshot_ids: Vec::new(),
        };
        let step = RelaxationStepReceipt {
            step_index: 1,
            claim,
            execution,
            input_structure_artifact_id: "s0".into(),
            output_structure_artifact_id: "s1".into(),
        };
        assert!(matches!(
            validate_step_shape(&[step], "claim:step"),
            Err(RelaxationExecutionGraphError::NonContiguousStepIndex { .. })
        ));
    }
}
