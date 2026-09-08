// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict public boundary for restart-aware structural relaxation graphs.
//!
//! The raw graph module is crate-private. This wrapper closes provenance links
//! that are easy to omit when describing restart chains: each structure identity
//! must be retained by the step claim, each output structure must actually be an
//! output artifact of that solver execution, and the final diagnostic must come
//! from the final solver execution rather than an unrelated artifact.

use symthaea_evidence_plane::external_receipt::ExternalEvidenceBundle;

pub use super::matter_relaxation_execution_graph::{
    RelaxationExecutionGraphError, RelaxationStepBinding, RelaxationStepReceipt,
    StructuralRelaxationExecutionGraphBinding, StructuralRelaxationExecutionGraphReceipt,
    StructuralRelaxationResultBinding, StructuralRelaxationResultReceipt,
};

/// The only public relaxation-graph admission path.
pub fn bind_structural_relaxation_execution_graph(
    evidence_bound: super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim,
    graph: StructuralRelaxationExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<StructuralRelaxationExecutionGraphBinding, StrictRelaxationExecutionError> {
    validate_explicit_step_provenance(&graph)?;
    let bound = super::matter_relaxation_execution_graph::bind_structural_relaxation_execution_graph(
        evidence_bound,
        graph,
        bundle,
    )?;
    Ok(bound)
}

fn validate_explicit_step_provenance(
    graph: &StructuralRelaxationExecutionGraphReceipt,
) -> Result<(), StrictRelaxationExecutionError> {
    if graph.steps.is_empty() {
        return Ok(()); // raw graph emits the canonical empty-sequence error.
    }

    for step in &graph.steps {
        for structure_id in [
            step.input_structure_artifact_id.as_str(),
            step.output_structure_artifact_id.as_str(),
        ] {
            if !step
                .claim
                .evidence
                .iter()
                .any(|evidence| evidence.evidence_id == structure_id)
            {
                return Err(StrictRelaxationExecutionError::StructureMissingFromStepClaim {
                    step_index: step.step_index,
                    structure_id: structure_id.to_string(),
                });
            }
        }

        if !step
            .execution
            .output_artifact_ids
            .iter()
            .any(|id| id == &step.output_structure_artifact_id)
        {
            return Err(StrictRelaxationExecutionError::OutputStructureNotExecutionOutput {
                step_index: step.step_index,
                structure_id: step.output_structure_artifact_id.clone(),
            });
        }
    }

    let final_step = graph.steps.last().expect("non-empty checked above");
    if !final_step
        .execution
        .output_artifact_ids
        .iter()
        .any(|id| id == &graph.result.diagnostic_artifact_id)
    {
        return Err(StrictRelaxationExecutionError::DiagnosticNotFinalExecutionOutput(
            graph.result.diagnostic_artifact_id.clone(),
        ));
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrictRelaxationExecutionError {
    Graph(RelaxationExecutionGraphError),
    StructureMissingFromStepClaim {
        step_index: u32,
        structure_id: String,
    },
    OutputStructureNotExecutionOutput {
        step_index: u32,
        structure_id: String,
    },
    DiagnosticNotFinalExecutionOutput(String),
}

impl From<RelaxationExecutionGraphError> for StrictRelaxationExecutionError {
    fn from(value: RelaxationExecutionGraphError) -> Self {
        Self::Graph(value)
    }
}

impl std::fmt::Display for StrictRelaxationExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "strict structural-relaxation graph rejected: {self:?}")
    }
}

impl std::error::Error for StrictRelaxationExecutionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{
        MatterClaim, MatterEvidenceRef, MatterScale, MatterSolverProfile, MatterValidationStage,
    };
    use super::super::matter_crystal_execution::{
        CrystalSolverExecutionReceipt, CrystalSolverTask,
    };

    fn evidence(id: &str) -> MatterEvidenceRef {
        MatterEvidenceRef {
            evidence_id: id.into(),
            run_id: None,
            config_hash: None,
            dataset_ids: Vec::new(),
        }
    }

    fn graph() -> StructuralRelaxationExecutionGraphReceipt {
        let claim = MatterClaim::local_computational(
            "claim:relax",
            "relaxation fixture",
            MatterScale::Crystal,
            MatterValidationStage::PhysicsSimulated,
            MatterSolverProfile {
                name: "solver".into(),
                version: "1".into(),
                method: "relax".into(),
                capabilities: Vec::new(),
                limitations: Vec::new(),
            },
        )
        .unwrap()
        .with_evidence(evidence("exec"))
        .with_evidence(evidence("s0"))
        .with_evidence(evidence("s1"))
        .with_evidence(evidence("diag"));

        StructuralRelaxationExecutionGraphReceipt {
            target_claim_id: "claim:relax".into(),
            steps: vec![RelaxationStepReceipt {
                step_index: 0,
                claim,
                execution: CrystalSolverExecutionReceipt {
                    claim_id: "claim:relax".into(),
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
                    input_artifact_id: "rendered-input".into(),
                    configuration_artifact_id: "config".into(),
                    environment_snapshot_id: "env".into(),
                    parser_snapshot_id: "parser".into(),
                    output_artifact_ids: vec!["s1".into(), "diag".into()],
                    dependency_snapshot_ids: Vec::new(),
                },
                input_structure_artifact_id: "s0".into(),
                output_structure_artifact_id: "s1".into(),
            }],
            result: StructuralRelaxationResultReceipt {
                diagnostic_artifact_id: "diag".into(),
                final_structure_artifact_id: "s1".into(),
                declared_result_yyyymmdd: 20260908,
                max_force_bits: 0.01_f64.to_bits(),
                force_tolerance_bits: 0.02_f64.to_bits(),
            },
        }
    }

    #[test]
    fn strict_shape_accepts_linked_structure_and_diagnostic_outputs() {
        validate_explicit_step_provenance(&graph()).unwrap();
    }

    #[test]
    fn output_structure_must_be_an_execution_output() {
        let mut graph = graph();
        graph.steps[0].execution.output_artifact_ids.retain(|id| id != "s1");
        assert!(matches!(
            validate_explicit_step_provenance(&graph),
            Err(StrictRelaxationExecutionError::OutputStructureNotExecutionOutput { .. })
        ));
    }

    #[test]
    fn final_diagnostic_must_come_from_final_execution() {
        let mut graph = graph();
        graph.steps[0].execution.output_artifact_ids.retain(|id| id != "diag");
        assert!(matches!(
            validate_explicit_step_provenance(&graph),
            Err(StrictRelaxationExecutionError::DiagnosticNotFinalExecutionOutput(_))
        ));
    }
}
