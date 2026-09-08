// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physical-completeness boundary for structural relaxation evidence.
//!
//! A force criterion is sufficient only under an explicitly declared fixed-cell
//! optimization. Variable-cell optimization must additionally satisfy declared
//! stress and cell-change tolerances, with both criteria preregistered before
//! every relaxation execution. This layer does not claim trusted chronology or
//! prove that a declared fixed-cell mode was faithfully implemented by a solver.

use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    Sha256Digest,
};

use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_crystal_phase::{CrystalPhaseEvidenceSlot, CrystalPhaseDiagnosticSnapshot};
use super::matter_periodic_convergence_binding::bind_strictly_converged_crystal_execution_plan;

pub use super::matter_periodic_convergence_binding::{
    ConvergenceBoundCrystalExecutionPlan, CrystalCapabilityGateError,
    CrystalCapabilityInterpretation, CrystalClaimCapabilityBinding, CrystalClaimCapabilityReceipt,
    CrystalScientificCapability, MixedCrystalExecutionPlanReceipt, PeriodicConvergenceError,
    PeriodicConvergenceInterpretation, PeriodicEnergyConvergenceSampleBinding,
    PeriodicEnergyConvergenceSampleReceipt, PeriodicEnergyConvergenceStudyBinding,
    PeriodicEnergyConvergenceStudyReceipt, StrictPeriodicConvergenceError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxationCellMode {
    FixedCell,
    VariableCell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelaxationCompletenessInterpretation {
    /// Forces are within the parent criterion under a producer-declared fixed
    /// cell constraint. The declaration itself is not independently verified.
    ForceCriterionUnderDeclaredFixedCell,
    /// Force, stress, and cell-change diagnostics are within the separately
    /// declared tolerances for the final variable-cell result.
    ForceStressAndCellCriteriaWithinDeclaredTolerances,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableCellCompletenessReceipt {
    /// Must be the exact diagnostic artifact already bound to the final
    /// structural-relaxation result.
    pub diagnostic_artifact_id: String,
    pub maximum_absolute_stress_gpa_bits: u64,
    pub stress_tolerance_gpa_bits: u64,
    pub maximum_relative_cell_change_bits: u64,
    pub cell_change_tolerance_bits: u64,
    pub stress_criterion_evidence_id: String,
    pub cell_change_criterion_evidence_id: String,
}

impl VariableCellCompletenessReceipt {
    fn validate(&self) -> Result<(), RelaxationCompletenessError> {
        for (field, value) in [
            ("diagnostic_artifact_id", self.diagnostic_artifact_id.as_str()),
            ("stress_criterion_evidence_id", self.stress_criterion_evidence_id.as_str()),
            (
                "cell_change_criterion_evidence_id",
                self.cell_change_criterion_evidence_id.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(RelaxationCompletenessError::EmptyField(field));
            }
        }
        let stress = finite_nonnegative(
            self.maximum_absolute_stress_gpa_bits,
            "maximum absolute stress",
        )?;
        let stress_tolerance =
            finite_nonnegative(self.stress_tolerance_gpa_bits, "stress tolerance")?;
        let cell_change = finite_nonnegative(
            self.maximum_relative_cell_change_bits,
            "maximum relative cell change",
        )?;
        let cell_tolerance =
            finite_nonnegative(self.cell_change_tolerance_bits, "cell-change tolerance")?;
        if stress > stress_tolerance {
            return Err(RelaxationCompletenessError::StressCriterionNotMet);
        }
        if cell_change > cell_tolerance {
            return Err(RelaxationCompletenessError::CellChangeCriterionNotMet);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationCompletenessReceipt {
    pub target_claim_id: String,
    pub mode: RelaxationCellMode,
    /// Canonical artifact declaring whether the optimization was fixed-cell or
    /// variable-cell. Every restart step must retain this exact reference.
    pub mode_declaration_artifact_id: String,
    /// Required exactly for variable-cell relaxation and forbidden for fixed-cell.
    pub variable_cell: Option<VariableCellCompletenessReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelaxationModeBinding {
    pub mode: RelaxationCellMode,
    pub evidence_id: String,
    pub content_sha256: Sha256Digest,
    pub claimed_date: ClaimedUtcDate,
    pub reference_interpretation: EvidenceReferenceInterpretation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableCellCompletenessBinding {
    pub diagnostic_evidence_id: String,
    pub diagnostic_sha256: Sha256Digest,
    pub maximum_absolute_stress_gpa_bits: u64,
    pub stress_tolerance_gpa_bits: u64,
    pub maximum_relative_cell_change_bits: u64,
    pub cell_change_tolerance_bits: u64,
    pub stress_criterion_evidence_id: String,
    pub stress_criterion_sha256: Sha256Digest,
    pub cell_change_criterion_evidence_id: String,
    pub cell_change_criterion_sha256: Sha256Digest,
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationCompleteCrystalExecutionPlan {
    pub converged: ConvergenceBoundCrystalExecutionPlan,
    pub mode: RelaxationModeBinding,
    pub variable_cell: Option<VariableCellCompletenessBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
    pub completeness_interpretation: RelaxationCompletenessInterpretation,
}

/// Strict public route above periodic-energy convergence.
///
/// The parent plan is rebuilt from raw receipts before relaxation completeness
/// is evaluated, avoiding trust in a hand-constructed parent binding.
pub fn bind_relaxation_complete_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    capability_receipts: &[CrystalClaimCapabilityReceipt],
    studies: &[PeriodicEnergyConvergenceStudyReceipt],
    relaxation: RelaxationCompletenessReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<RelaxationCompleteCrystalExecutionPlan, RelaxationCompletenessError> {
    let converged = bind_strictly_converged_crystal_execution_plan(
        evidence_bound,
        plan_receipt,
        capability_receipts,
        studies,
        bundle,
    )?;

    let graph = &converged.capability_bound.plan.relaxation;
    if graph.steps.is_empty() {
        return Err(RelaxationCompletenessError::EmptyRelaxationGraph);
    }
    if relaxation.target_claim_id != graph.target_claim_id {
        return Err(RelaxationCompletenessError::TargetClaimIdentityMismatch);
    }
    if relaxation.mode_declaration_artifact_id.trim().is_empty() {
        return Err(RelaxationCompletenessError::EmptyField(
            "mode_declaration_artifact_id",
        ));
    }

    let target_snapshot = graph
        .evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
        .ok_or(RelaxationCompletenessError::MissingRelaxationClaim)?;
    if !matches!(
        target_snapshot.diagnostic,
        CrystalPhaseDiagnosticSnapshot::StructuralRelaxation { .. }
    ) {
        return Err(RelaxationCompletenessError::RelaxationDiagnosticMismatch);
    }
    let target_claim = &target_snapshot.claim;

    let mode_ref = bundle.require_role(
        &relaxation.mode_declaration_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let mode_date = mode_ref.claimed_utc_date.ok_or_else(|| {
        RelaxationCompletenessError::MissingClaimedDate(
            relaxation.mode_declaration_artifact_id.clone(),
        )
    })?;
    let first_execution_date = graph.steps[0].execution.execution_claimed_date;
    if mode_date > first_execution_date {
        return Err(RelaxationCompletenessError::ModeDeclarationPostdatesRelaxation);
    }

    for step in &graph.steps {
        if !claim_retains(&step.claim, mode_ref.id.as_str()) {
            return Err(RelaxationCompletenessError::ModeDeclarationMissingFromStepClaim {
                step_index: step.step_index,
            });
        }
    }
    if !claim_retains(target_claim, mode_ref.id.as_str()) {
        return Err(RelaxationCompletenessError::ModeDeclarationMissingFromTargetClaim);
    }
    if mode_ref.content_sha256 == graph.result.diagnostic_sha256
        || mode_ref.content_sha256 == graph.result.final_structure_sha256
    {
        return Err(RelaxationCompletenessError::ModeDeclarationAliasesResultArtifact);
    }

    let mode_binding = RelaxationModeBinding {
        mode: relaxation.mode,
        evidence_id: mode_ref.id.as_str().to_string(),
        content_sha256: mode_ref.content_sha256.clone(),
        claimed_date: mode_date,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
    };

    let (variable_cell, interpretation) = match (relaxation.mode, relaxation.variable_cell) {
        (RelaxationCellMode::FixedCell, None) => (
            None,
            RelaxationCompletenessInterpretation::ForceCriterionUnderDeclaredFixedCell,
        ),
        (RelaxationCellMode::FixedCell, Some(_)) => {
            return Err(RelaxationCompletenessError::VariableMetricsForbiddenForFixedCell)
        }
        (RelaxationCellMode::VariableCell, None) => {
            return Err(RelaxationCompletenessError::VariableMetricsRequired)
        }
        (RelaxationCellMode::VariableCell, Some(receipt)) => {
            let binding = bind_variable_cell(&receipt, graph, target_claim, bundle)?;
            (
                Some(binding),
                RelaxationCompletenessInterpretation::ForceStressAndCellCriteriaWithinDeclaredTolerances,
            )
        }
    };

    Ok(RelaxationCompleteCrystalExecutionPlan {
        converged,
        mode: mode_binding,
        variable_cell,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
        completeness_interpretation: interpretation,
    })
}

fn bind_variable_cell(
    receipt: &VariableCellCompletenessReceipt,
    graph: &super::matter_relaxation_execution_binding::StructuralRelaxationExecutionGraphBinding,
    target_claim: &symthaea_epistemic_types::MatterClaim,
    bundle: &ExternalEvidenceBundle,
) -> Result<VariableCellCompletenessBinding, RelaxationCompletenessError> {
    receipt.validate()?;
    if receipt.diagnostic_artifact_id != graph.result.diagnostic_evidence_id {
        return Err(RelaxationCompletenessError::VariableDiagnosticIdentityMismatch);
    }
    let diagnostic = bundle.require_role(
        &receipt.diagnostic_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    if diagnostic.content_sha256 != graph.result.diagnostic_sha256 {
        return Err(RelaxationCompletenessError::VariableDiagnosticContentMismatch);
    }

    let stress_criterion = bundle.require_role(
        &receipt.stress_criterion_evidence_id,
        EvidenceRole::Preregistration,
    )?;
    let cell_criterion = bundle.require_role(
        &receipt.cell_change_criterion_evidence_id,
        EvidenceRole::Preregistration,
    )?;
    if stress_criterion.content_sha256 == cell_criterion.content_sha256
        && stress_criterion.id != cell_criterion.id
    {
        return Err(RelaxationCompletenessError::DistinctCriterionIdsAliasContent);
    }
    for criterion in [stress_criterion, cell_criterion] {
        if !claim_retains(target_claim, criterion.id.as_str()) {
            return Err(RelaxationCompletenessError::CriterionMissingFromTargetClaim(
                criterion.id.as_str().to_string(),
            ));
        }
        for step in &graph.steps {
            if !claim_retains(&step.claim, criterion.id.as_str()) {
                return Err(RelaxationCompletenessError::CriterionMissingFromStepClaim {
                    step_index: step.step_index,
                    criterion_id: criterion.id.as_str().to_string(),
                });
            }
        }
    }

    let mut chronology = Vec::with_capacity(graph.steps.len() * 2);
    for step in &graph.steps {
        chronology.push(bundle.require_preregistration_before(
            stress_criterion.id.as_str(),
            &step.execution.execution_evidence_id,
        )?);
        chronology.push(bundle.require_preregistration_before(
            cell_criterion.id.as_str(),
            &step.execution.execution_evidence_id,
        )?);
    }

    Ok(VariableCellCompletenessBinding {
        diagnostic_evidence_id: diagnostic.id.as_str().to_string(),
        diagnostic_sha256: diagnostic.content_sha256.clone(),
        maximum_absolute_stress_gpa_bits: receipt.maximum_absolute_stress_gpa_bits,
        stress_tolerance_gpa_bits: receipt.stress_tolerance_gpa_bits,
        maximum_relative_cell_change_bits: receipt.maximum_relative_cell_change_bits,
        cell_change_tolerance_bits: receipt.cell_change_tolerance_bits,
        stress_criterion_evidence_id: stress_criterion.id.as_str().to_string(),
        stress_criterion_sha256: stress_criterion.content_sha256.clone(),
        cell_change_criterion_evidence_id: cell_criterion.id.as_str().to_string(),
        cell_change_criterion_sha256: cell_criterion.content_sha256.clone(),
        criterion_chronology: chronology,
    })
}

fn claim_retains(claim: &symthaea_epistemic_types::MatterClaim, evidence_id: &str) -> bool {
    claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == evidence_id)
}

fn finite_nonnegative(bits: u64, field: &'static str) -> Result<f64, RelaxationCompletenessError> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(RelaxationCompletenessError::InvalidMetric(field));
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelaxationCompletenessError {
    Parent(StrictPeriodicConvergenceError),
    ExternalEvidence(ExternalEvidenceError),
    EmptyField(&'static str),
    InvalidMetric(&'static str),
    EmptyRelaxationGraph,
    TargetClaimIdentityMismatch,
    MissingRelaxationClaim,
    RelaxationDiagnosticMismatch,
    MissingClaimedDate(String),
    ModeDeclarationPostdatesRelaxation,
    ModeDeclarationMissingFromStepClaim { step_index: u32 },
    ModeDeclarationMissingFromTargetClaim,
    ModeDeclarationAliasesResultArtifact,
    VariableMetricsForbiddenForFixedCell,
    VariableMetricsRequired,
    StressCriterionNotMet,
    CellChangeCriterionNotMet,
    VariableDiagnosticIdentityMismatch,
    VariableDiagnosticContentMismatch,
    DistinctCriterionIdsAliasContent,
    CriterionMissingFromTargetClaim(String),
    CriterionMissingFromStepClaim { step_index: u32, criterion_id: String },
}

impl From<StrictPeriodicConvergenceError> for RelaxationCompletenessError {
    fn from(value: StrictPeriodicConvergenceError) -> Self {
        Self::Parent(value)
    }
}

impl From<ExternalEvidenceError> for RelaxationCompletenessError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for RelaxationCompletenessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "relaxation completeness rejected: {self:?}")
    }
}

impl std::error::Error for RelaxationCompletenessError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_metrics_reject_stress_above_tolerance() {
        let receipt = VariableCellCompletenessReceipt {
            diagnostic_artifact_id: "diag".into(),
            maximum_absolute_stress_gpa_bits: 0.21_f64.to_bits(),
            stress_tolerance_gpa_bits: 0.20_f64.to_bits(),
            maximum_relative_cell_change_bits: 1.0e-4_f64.to_bits(),
            cell_change_tolerance_bits: 2.0e-4_f64.to_bits(),
            stress_criterion_evidence_id: "stress".into(),
            cell_change_criterion_evidence_id: "cell".into(),
        };
        assert_eq!(
            receipt.validate().unwrap_err(),
            RelaxationCompletenessError::StressCriterionNotMet
        );
    }

    #[test]
    fn variable_metrics_reject_nonfinite_cell_change() {
        let receipt = VariableCellCompletenessReceipt {
            diagnostic_artifact_id: "diag".into(),
            maximum_absolute_stress_gpa_bits: 0.1_f64.to_bits(),
            stress_tolerance_gpa_bits: 0.2_f64.to_bits(),
            maximum_relative_cell_change_bits: f64::NAN.to_bits(),
            cell_change_tolerance_bits: 2.0e-4_f64.to_bits(),
            stress_criterion_evidence_id: "stress".into(),
            cell_change_criterion_evidence_id: "cell".into(),
        };
        assert_eq!(
            receipt.validate().unwrap_err(),
            RelaxationCompletenessError::InvalidMetric("maximum relative cell change")
        );
    }
}
