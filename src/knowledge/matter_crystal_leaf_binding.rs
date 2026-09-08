// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reusable canonical binding for one crystal-solver leaf execution.
//!
//! This module is the shared leaf theorem for task-specific execution graphs.
//! It deliberately does not decide relaxation convergence, phase stability, or
//! phonon stability. It validates only that one computational Matter claim is
//! tied to one successful declared solver execution and to canonical external
//! evidence with internally consistent roles, content identities, and dates.

use std::collections::BTreeMap;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterSolverCapability, MatterValidationStage,
};
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, EvidenceReferenceInterpretation,
    EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceReference,
};

use super::matter_crystal_execution::{
    CrystalExecutionError, CrystalSolverExecutionBinding, CrystalSolverExecutionReceipt,
    CrystalSolverTask,
};

/// Bind one solver execution to one computational Matter claim.
///
/// Future workflow graphs should call this function rather than reimplementing
/// solver identity, evidence-role, content-distinctness, or chronology checks.
pub fn bind_crystal_solver_leaf(
    receipt: &CrystalSolverExecutionReceipt,
    claim: &MatterClaim,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalSolverExecutionBinding, CrystalExecutionError> {
    receipt.validate()?;
    validate_receipt_against_claim(receipt, claim)?;
    bind_execution_references(receipt, bundle)
}

fn validate_receipt_against_claim(
    receipt: &CrystalSolverExecutionReceipt,
    claim: &MatterClaim,
) -> Result<(), CrystalExecutionError> {
    if receipt.claim_id != claim.claim_id {
        return Err(CrystalExecutionError::ClaimIdentityMismatch {
            receipt: receipt.claim_id.clone(),
            claim: claim.claim_id.clone(),
        });
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(CrystalExecutionError::ClaimScaleMismatch(claim.scale));
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(CrystalExecutionError::ClaimNotScientific);
    }
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(CrystalExecutionError::ClaimStageNotComputational(
            claim.validation_stage,
        ));
    }
    let solver = claim
        .solver
        .as_ref()
        .ok_or(CrystalExecutionError::MissingSolverProfile)?;
    if solver.name != receipt.solver_name || solver.version != receipt.solver_version {
        return Err(CrystalExecutionError::SolverIdentityMismatch);
    }
    if receipt.task == CrystalSolverTask::PeriodicElectronicStructure
        && !solver.supports(MatterSolverCapability::PeriodicElectronicStructure)
    {
        return Err(CrystalExecutionError::PeriodicCapabilityMissing);
    }
    if !claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == receipt.execution_evidence_id)
    {
        return Err(CrystalExecutionError::ExecutionEvidenceMissingFromClaim);
    }
    Ok(())
}

fn bind_execution_references(
    receipt: &CrystalSolverExecutionReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalSolverExecutionBinding, CrystalExecutionError> {
    let execution = bundle.require_role(
        &receipt.execution_evidence_id,
        EvidenceRole::SolverExecution,
    )?;
    let input = bundle.require_role(&receipt.input_artifact_id, EvidenceRole::ArtifactContent)?;
    let configuration = bundle.require_role(
        &receipt.configuration_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let environment = bundle.require_role(
        &receipt.environment_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;
    let parser = bundle.require_role(
        &receipt.parser_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;

    let execution_date = require_date(execution)?;
    if execution_date.yyyymmdd() != receipt.declared_execution_yyyymmdd {
        return Err(CrystalExecutionError::ExecutionDateMismatch {
            receipt_yyyymmdd: receipt.declared_execution_yyyymmdd,
            evidence_yyyymmdd: execution_date.yyyymmdd(),
        });
    }

    let mut semantic_refs = vec![execution, input, configuration, environment, parser];
    let mut output_refs = Vec::new();
    for id in &receipt.output_artifact_ids {
        let reference = bundle.require_role(id, EvidenceRole::ArtifactContent)?;
        output_refs.push(reference);
        semantic_refs.push(reference);
    }
    let mut dependency_refs = Vec::new();
    for id in &receipt.dependency_snapshot_ids {
        let reference = bundle.require_role(id, EvidenceRole::DependencySnapshot)?;
        dependency_refs.push(reference);
        semantic_refs.push(reference);
    }
    ensure_distinct_content(&semantic_refs)?;

    for reference in [input, configuration, environment, parser]
        .into_iter()
        .chain(dependency_refs.iter().copied())
    {
        let date = require_date(reference)?;
        if date > execution_date {
            return Err(CrystalExecutionError::DependencyPostdatesExecution {
                evidence_id: reference.id.as_str().to_string(),
                dependency_yyyymmdd: date.yyyymmdd(),
                execution_yyyymmdd: execution_date.yyyymmdd(),
            });
        }
    }
    for reference in &output_refs {
        let date = require_date(reference)?;
        if date < execution_date {
            return Err(CrystalExecutionError::OutputPredatesExecution {
                evidence_id: reference.id.as_str().to_string(),
                output_yyyymmdd: date.yyyymmdd(),
                execution_yyyymmdd: execution_date.yyyymmdd(),
            });
        }
    }

    Ok(CrystalSolverExecutionBinding {
        claim_id: receipt.claim_id.clone(),
        task: receipt.task,
        execution_evidence_id: execution.id.as_str().to_string(),
        execution_content_sha256: execution.content_sha256.clone(),
        execution_claimed_date: execution_date,
        started_unix_ms: receipt.started_unix_ms,
        finished_unix_ms: receipt.finished_unix_ms,
        solver_name: receipt.solver_name.clone(),
        solver_version: receipt.solver_version.clone(),
        adapter_name: receipt.adapter_name.clone(),
        adapter_version: receipt.adapter_version.clone(),
        input_content_sha256: input.content_sha256.clone(),
        configuration_content_sha256: configuration.content_sha256.clone(),
        environment_content_sha256: environment.content_sha256.clone(),
        parser_content_sha256: parser.content_sha256.clone(),
        output_content_sha256: output_refs
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        dependency_content_sha256: dependency_refs
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn ensure_distinct_content(
    references: &[&ExternalEvidenceReference],
) -> Result<(), CrystalExecutionError> {
    let mut seen = BTreeMap::<String, String>::new();
    for reference in references {
        let digest = reference.content_sha256.as_str().to_string();
        if let Some(first_id) = seen.insert(digest, reference.id.as_str().to_string()) {
            return Err(CrystalExecutionError::DuplicateContentIdentity {
                first_id,
                second_id: reference.id.as_str().to_string(),
            });
        }
    }
    Ok(())
}

fn require_date(
    reference: &ExternalEvidenceReference,
) -> Result<ClaimedUtcDate, CrystalExecutionError> {
    reference
        .claimed_utc_date
        .ok_or_else(|| CrystalExecutionError::MissingClaimedDate(reference.id.as_str().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{MatterEvidenceRef, MatterSolverProfile};
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(
        id: &str,
        role: EvidenceRole,
        digest_char: char,
        date: u32,
    ) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(digest_char),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            Some(date),
            Some("leaf fixture".into()),
        )
        .unwrap()
    }

    fn claim() -> MatterClaim {
        MatterClaim::local_computational(
            "claim:periodic-leaf",
            "periodic leaf fixture",
            MatterScale::Crystal,
            MatterValidationStage::HighFidelitySimulated,
            MatterSolverProfile {
                name: "solver".into(),
                version: "1".into(),
                method: "periodic".into(),
                capabilities: vec![MatterSolverCapability::PeriodicElectronicStructure],
                limitations: vec!["fixture".into()],
            },
        )
        .unwrap()
        .with_evidence(MatterEvidenceRef {
            evidence_id: "exec".into(),
            run_id: None,
            config_hash: None,
            dataset_ids: Vec::new(),
        })
    }

    fn receipt() -> CrystalSolverExecutionReceipt {
        CrystalSolverExecutionReceipt {
            claim_id: "claim:periodic-leaf".into(),
            task: CrystalSolverTask::PeriodicElectronicStructure,
            execution_evidence_id: "exec".into(),
            declared_execution_yyyymmdd: 20260907,
            started_unix_ms: 1_000,
            finished_unix_ms: 2_000,
            exit_code: 0,
            solver_name: "solver".into(),
            solver_version: "1".into(),
            adapter_name: "adapter".into(),
            adapter_version: "1".into(),
            input_artifact_id: "input".into(),
            configuration_artifact_id: "config".into(),
            environment_snapshot_id: "env".into(),
            parser_snapshot_id: "parser".into(),
            output_artifact_ids: vec!["output".into()],
            dependency_snapshot_ids: vec!["dep".into()],
        }
    }

    fn bundle() -> ExternalEvidenceBundle {
        ExternalEvidenceBundle::new(vec![
            reference("exec", EvidenceRole::SolverExecution, 'a', 20260907),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260906),
            reference("config", EvidenceRole::ArtifactContent, 'c', 20260906),
            reference("env", EvidenceRole::ImplementationSnapshot, 'd', 20260906),
            reference("parser", EvidenceRole::ImplementationSnapshot, 'e', 20260906),
            reference("output", EvidenceRole::ArtifactContent, 'f', 20260907),
            reference("dep", EvidenceRole::DependencySnapshot, '1', 20260905),
        ])
        .unwrap()
    }

    #[test]
    fn reusable_leaf_binds_exact_execution() {
        let bound = bind_crystal_solver_leaf(&receipt(), &claim(), &bundle()).unwrap();
        assert_eq!(bound.execution_evidence_id, "exec");
        assert_eq!(bound.execution_claimed_date.yyyymmdd(), 20260907);
        assert_eq!(bound.reference_interpretation, EvidenceReferenceInterpretation::ReferenceOnly);
    }

    #[test]
    fn claim_must_retain_execution_identity() {
        let mut claim = claim();
        claim.evidence.clear();
        assert_eq!(
            bind_crystal_solver_leaf(&receipt(), &claim, &bundle()).unwrap_err(),
            CrystalExecutionError::ExecutionEvidenceMissingFromClaim
        );
    }

    #[test]
    fn duplicate_content_across_semantic_slots_fails_closed() {
        let mut refs = bundle().references().to_vec();
        refs.retain(|reference| reference.id.as_str() != "config");
        refs.push(reference("config", EvidenceRole::ArtifactContent, 'b', 20260906));
        let bad = ExternalEvidenceBundle::new(refs).unwrap();
        assert!(matches!(
            bind_crystal_solver_leaf(&receipt(), &claim(), &bad),
            Err(CrystalExecutionError::DuplicateContentIdentity { .. })
        ));
    }
}
