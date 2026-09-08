// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Batched q-point execution graph for crystal lattice-dynamics evidence.
//!
//! One lattice-dynamics solver execution may produce one or many q-point
//! dynamical matrices. The graph therefore models *execution batches* rather
//! than assuming one process per q-point. This avoids either rejecting valid
//! grid-style workflows or counting one solver execution several times.
//!
//! The final dynamical-stability claim is an aggregate over all uniquely sampled
//! q-points plus an explicit named/versioned aggregation step. Exact minimum
//! frequency, tolerance, and sampled-q count must reconcile with the diagnostic
//! already admitted by `matter_crystal_phase`.
//!
//! This is computational evidence only. It does not establish Brillouin-zone
//! completeness, solver correctness, experiment, replication, or trusted time.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterValidationStage,
};
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    ExternalEvidenceReference, Sha256Digest,
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

/// One q-point result produced inside a lattice-dynamics execution batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeQPointSampleReceipt {
    pub q_point_id: String,
    /// Exact IEEE-754 bits of fractional reciprocal coordinates. The graph does
    /// not infer a reciprocal-coordinate convention or symmetry expansion.
    pub q_fractional_bits: [u64; 3],
    pub dynamical_matrix_artifact_id: String,
}

/// One external lattice-dynamics execution and all q-point matrices it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeQPointBatchReceipt {
    pub batch_id: String,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionReceipt,
    pub samples: Vec<LatticeQPointSampleReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeDynamicsAggregationReceipt {
    pub aggregator_name: String,
    pub aggregator_version: String,
    pub implementation_snapshot_id: String,
    pub configuration_artifact_id: String,
    pub output_artifact_id: String,
    pub dependency_snapshot_ids: Vec<String>,
    /// Must be exactly the set of dynamical-matrix evidence ids produced by all
    /// q-point batches in this graph. Ordering has no authority.
    pub input_dynamical_matrix_artifact_ids: Vec<String>,
    pub declared_output_yyyymmdd: u32,
    pub minimum_frequency_bits: u64,
    pub imaginary_frequency_tolerance_bits: u64,
    pub sampled_q_point_count: u32,
}

impl LatticeDynamicsAggregationReceipt {
    pub fn validate(&self) -> Result<(), LatticeDynamicsGraphError> {
        for (field, value) in [
            ("aggregator_name", self.aggregator_name.as_str()),
            ("aggregator_version", self.aggregator_version.as_str()),
            (
                "implementation_snapshot_id",
                self.implementation_snapshot_id.as_str(),
            ),
            (
                "configuration_artifact_id",
                self.configuration_artifact_id.as_str(),
            ),
            ("output_artifact_id", self.output_artifact_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(LatticeDynamicsGraphError::EmptyField(field));
            }
        }
        if self.declared_output_yyyymmdd == 0 {
            return Err(LatticeDynamicsGraphError::MissingAggregationDate);
        }
        if self.sampled_q_point_count == 0 {
            return Err(LatticeDynamicsGraphError::NoQPoints);
        }
        if !f64::from_bits(self.minimum_frequency_bits).is_finite() {
            return Err(LatticeDynamicsGraphError::InvalidMinimumFrequency);
        }
        let tolerance = f64::from_bits(self.imaginary_frequency_tolerance_bits);
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LatticeDynamicsGraphError::InvalidImaginaryFrequencyTolerance);
        }
        if self.input_dynamical_matrix_artifact_ids.is_empty() {
            return Err(LatticeDynamicsGraphError::NoDynamicalMatrixInputs);
        }
        if self
            .dependency_snapshot_ids
            .iter()
            .chain(self.input_dynamical_matrix_artifact_ids.iter())
            .any(|id| id.trim().is_empty())
        {
            return Err(LatticeDynamicsGraphError::EmptyArtifactIdentity);
        }

        let mut semantic_ids = BTreeSet::new();
        for id in [
            self.implementation_snapshot_id.as_str(),
            self.configuration_artifact_id.as_str(),
            self.output_artifact_id.as_str(),
        ]
        .into_iter()
        .chain(self.dependency_snapshot_ids.iter().map(String::as_str))
        {
            if !semantic_ids.insert(id) {
                return Err(LatticeDynamicsGraphError::DuplicateAggregationIdentity(
                    id.to_string(),
                ));
            }
        }

        let mut matrix_ids = BTreeSet::new();
        for id in &self.input_dynamical_matrix_artifact_ids {
            if !matrix_ids.insert(id) {
                return Err(LatticeDynamicsGraphError::DuplicateDynamicalMatrixIdentity(
                    id.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatticeDynamicsExecutionGraphReceipt {
    pub target_claim_id: String,
    pub batches: Vec<LatticeQPointBatchReceipt>,
    pub aggregation: LatticeDynamicsAggregationReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeQPointSampleBinding {
    pub q_point_id: String,
    pub q_fractional_bits: [u64; 3],
    pub dynamical_matrix_evidence_id: String,
    pub dynamical_matrix_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatticeQPointBatchBinding {
    pub batch_id: String,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionBinding,
    pub samples: Vec<LatticeQPointSampleBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeDynamicsAggregationBinding {
    pub aggregator_name: String,
    pub aggregator_version: String,
    pub implementation_evidence_id: String,
    pub implementation_sha256: Sha256Digest,
    pub configuration_evidence_id: String,
    pub configuration_sha256: Sha256Digest,
    pub output_evidence_id: String,
    pub output_sha256: Sha256Digest,
    pub output_claimed_date: ClaimedUtcDate,
    pub dependency_evidence_ids: Vec<String>,
    pub dependency_sha256: Vec<Sha256Digest>,
    pub input_dynamical_matrix_evidence_ids: Vec<String>,
    pub minimum_frequency_bits: u64,
    pub imaginary_frequency_tolerance_bits: u64,
    pub sampled_q_point_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatticeDynamicsExecutionGraphBinding {
    pub evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    pub target_claim_id: String,
    pub batches: Vec<LatticeQPointBatchBinding>,
    pub aggregation: LatticeDynamicsAggregationBinding,
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

pub fn bind_lattice_dynamics_execution_graph(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    graph: LatticeDynamicsExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<LatticeDynamicsExecutionGraphBinding, LatticeDynamicsGraphError> {
    let rebound = bind_crystal_phase_evidence(evidence_bound.admitted.clone(), bundle)?;
    if rebound != evidence_bound {
        return Err(LatticeDynamicsGraphError::ParentEvidenceBindingMismatch);
    }

    let snapshot = evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::DynamicalStability)
        .ok_or(LatticeDynamicsGraphError::MissingDynamicalStabilityClaim)?;
    let target_claim = &snapshot.claim;
    if graph.target_claim_id != target_claim.claim_id {
        return Err(LatticeDynamicsGraphError::TargetClaimIdentityMismatch {
            graph: graph.target_claim_id.clone(),
            claim: target_claim.claim_id.clone(),
        });
    }

    let (expected_minimum_frequency_bits, expected_tolerance_bits, expected_q_count) =
        match &snapshot.diagnostic {
            CrystalPhaseDiagnosticSnapshot::DynamicalStability {
                minimum_frequency_bits,
                imaginary_frequency_tolerance_bits,
                sampled_q_point_count,
                ..
            } => (
                *minimum_frequency_bits,
                *imaginary_frequency_tolerance_bits,
                *sampled_q_point_count,
            ),
            _ => return Err(LatticeDynamicsGraphError::DynamicalDiagnosticMismatch),
        };
    let expected_q_count = u32::try_from(expected_q_count)
        .map_err(|_| LatticeDynamicsGraphError::QPointCountOverflow)?;

    graph.aggregation.validate()?;
    if graph.aggregation.minimum_frequency_bits != expected_minimum_frequency_bits {
        return Err(LatticeDynamicsGraphError::MinimumFrequencyMismatch);
    }
    if graph.aggregation.imaginary_frequency_tolerance_bits != expected_tolerance_bits {
        return Err(LatticeDynamicsGraphError::ImaginaryFrequencyToleranceMismatch);
    }
    if graph.aggregation.sampled_q_point_count != expected_q_count {
        return Err(LatticeDynamicsGraphError::QPointCountMismatch {
            expected: expected_q_count,
            actual: graph.aggregation.sampled_q_point_count,
        });
    }

    let actual_sample_count = graph.batches.iter().try_fold(0usize, |total, batch| {
        total
            .checked_add(batch.samples.len())
            .ok_or(LatticeDynamicsGraphError::QPointCountOverflow)
    })?;
    if actual_sample_count != expected_q_count as usize {
        return Err(LatticeDynamicsGraphError::QPointSampleCountMismatch {
            expected: expected_q_count,
            actual: actual_sample_count,
        });
    }

    validate_target_claim(target_claim, &graph.aggregation)?;
    validate_batch_shape(&graph.batches, &graph.target_claim_id)?;

    let criterion = evidence_bound
        .criterion_bindings
        .iter()
        .find(|binding| binding.slot == CrystalPhaseEvidenceSlot::DynamicalStability)
        .ok_or(LatticeDynamicsGraphError::MissingDynamicalCriterion)?;

    let mut bound_batches = Vec::with_capacity(graph.batches.len());
    let mut chronology = Vec::with_capacity(graph.batches.len() + 1);
    let mut execution_ids = BTreeSet::new();
    let mut execution_digests = BTreeSet::new();
    let mut matrix_ids = BTreeSet::new();
    let mut matrix_digests = BTreeSet::new();
    let mut q_ids = BTreeSet::new();
    let mut q_coordinates = BTreeSet::new();
    let mut any_batch_ood = false;

    for batch in &graph.batches {
        if batch.execution.task != CrystalSolverTask::LatticeDynamics {
            return Err(LatticeDynamicsGraphError::BatchIsNotLatticeDynamicsTask);
        }
        validate_computational_claim_shape(&batch.claim)?;

        if !execution_ids.insert(batch.execution.execution_evidence_id.clone()) {
            return Err(LatticeDynamicsGraphError::DuplicateExecutionIdentity(
                batch.execution.execution_evidence_id.clone(),
            ));
        }
        let execution = bind_crystal_solver_leaf(&batch.execution, &batch.claim, bundle)?;
        if !execution_digests.insert(execution.execution_content_sha256.as_str().to_string()) {
            return Err(LatticeDynamicsGraphError::DuplicateExecutionContent);
        }

        let mut bound_samples = Vec::with_capacity(batch.samples.len());
        for sample in &batch.samples {
            validate_sample_shape(sample)?;
            if !q_ids.insert(sample.q_point_id.clone()) {
                return Err(LatticeDynamicsGraphError::DuplicateQPointId(
                    sample.q_point_id.clone(),
                ));
            }
            if !q_coordinates.insert(sample.q_fractional_bits) {
                return Err(LatticeDynamicsGraphError::DuplicateQPointCoordinates);
            }
            if !matrix_ids.insert(sample.dynamical_matrix_artifact_id.clone()) {
                return Err(LatticeDynamicsGraphError::DuplicateDynamicalMatrixIdentity(
                    sample.dynamical_matrix_artifact_id.clone(),
                ));
            }
            if !batch
                .execution
                .output_artifact_ids
                .iter()
                .any(|id| id == &sample.dynamical_matrix_artifact_id)
            {
                return Err(LatticeDynamicsGraphError::DynamicalMatrixNotExecutionOutput {
                    batch_id: batch.batch_id.clone(),
                    q_point_id: sample.q_point_id.clone(),
                });
            }
            if !batch
                .claim
                .evidence
                .iter()
                .any(|evidence| evidence.evidence_id == sample.dynamical_matrix_artifact_id)
            {
                return Err(LatticeDynamicsGraphError::DynamicalMatrixMissingFromClaim {
                    batch_id: batch.batch_id.clone(),
                    q_point_id: sample.q_point_id.clone(),
                });
            }

            let matrix = bundle.require_role(
                &sample.dynamical_matrix_artifact_id,
                EvidenceRole::ArtifactContent,
            )?;
            let matrix_date = require_date(matrix)?;
            if matrix_date < execution.execution_claimed_date {
                return Err(LatticeDynamicsGraphError::DynamicalMatrixPredatesExecution {
                    batch_id: batch.batch_id.clone(),
                    q_point_id: sample.q_point_id.clone(),
                });
            }
            if !matrix_digests.insert(matrix.content_sha256.as_str().to_string()) {
                return Err(LatticeDynamicsGraphError::DuplicateDynamicalMatrixContent);
            }

            bound_samples.push(LatticeQPointSampleBinding {
                q_point_id: sample.q_point_id.clone(),
                q_fractional_bits: sample.q_fractional_bits,
                dynamical_matrix_evidence_id: matrix.id.as_str().to_string(),
                dynamical_matrix_sha256: matrix.content_sha256.clone(),
            });
        }

        if batch
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
        {
            any_batch_ood = true;
        }

        chronology.push(bundle.require_preregistration_before(
            &criterion.evidence_id,
            &execution.execution_evidence_id,
        )?);
        bound_batches.push(LatticeQPointBatchBinding {
            batch_id: batch.batch_id.clone(),
            claim: batch.claim.clone(),
            execution,
            samples: bound_samples,
        });
    }

    if any_batch_ood
        && !target_claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(LatticeDynamicsGraphError::BatchOodNotPropagated);
    }

    let aggregation = bind_aggregation(
        target_claim,
        &graph.aggregation,
        &bound_batches,
        bundle,
    )?;
    chronology.push(bundle.require_preregistration_before(
        &criterion.evidence_id,
        &aggregation.output_evidence_id,
    )?);

    Ok(LatticeDynamicsExecutionGraphBinding {
        evidence_bound,
        target_claim_id: graph.target_claim_id,
        batches: bound_batches,
        aggregation,
        criterion_chronology: chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn validate_target_claim(
    claim: &MatterClaim,
    aggregation: &LatticeDynamicsAggregationReceipt,
) -> Result<(), LatticeDynamicsGraphError> {
    validate_computational_claim_shape(claim)?;
    let solver = claim
        .solver
        .as_ref()
        .ok_or(LatticeDynamicsGraphError::MissingAggregateSolverProfile)?;
    if solver.name != aggregation.aggregator_name || solver.version != aggregation.aggregator_version {
        return Err(LatticeDynamicsGraphError::AggregateSolverIdentityMismatch);
    }
    if !claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == aggregation.output_artifact_id)
    {
        return Err(LatticeDynamicsGraphError::AggregateOutputMissingFromClaim);
    }
    Ok(())
}

fn validate_batch_shape(
    batches: &[LatticeQPointBatchReceipt],
    target_claim_id: &str,
) -> Result<(), LatticeDynamicsGraphError> {
    if batches.is_empty() {
        return Err(LatticeDynamicsGraphError::NoExecutionBatches);
    }
    let mut batch_ids = BTreeSet::new();
    let mut claim_ids = BTreeSet::new();
    for batch in batches {
        if batch.batch_id.trim().is_empty() {
            return Err(LatticeDynamicsGraphError::EmptyBatchId);
        }
        if batch.samples.is_empty() {
            return Err(LatticeDynamicsGraphError::EmptyQPointBatch(batch.batch_id.clone()));
        }
        if !batch_ids.insert(batch.batch_id.clone()) {
            return Err(LatticeDynamicsGraphError::DuplicateBatchId(batch.batch_id.clone()));
        }
        if batch.claim.claim_id == target_claim_id {
            return Err(LatticeDynamicsGraphError::BatchAliasesAggregateClaim {
                batch_id: batch.batch_id.clone(),
                claim_id: batch.claim.claim_id.clone(),
            });
        }
        if !claim_ids.insert(batch.claim.claim_id.clone()) {
            return Err(LatticeDynamicsGraphError::DuplicateBatchClaimId(
                batch.claim.claim_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_sample_shape(sample: &LatticeQPointSampleReceipt) -> Result<(), LatticeDynamicsGraphError> {
    if sample.q_point_id.trim().is_empty() {
        return Err(LatticeDynamicsGraphError::EmptyQPointId);
    }
    if sample.dynamical_matrix_artifact_id.trim().is_empty() {
        return Err(LatticeDynamicsGraphError::EmptyArtifactIdentity);
    }
    if sample
        .q_fractional_bits
        .into_iter()
        .map(f64::from_bits)
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(LatticeDynamicsGraphError::InvalidQPointCoordinate);
    }
    Ok(())
}

fn validate_computational_claim_shape(claim: &MatterClaim) -> Result<(), LatticeDynamicsGraphError> {
    if claim.claim_id.trim().is_empty() || claim.statement.trim().is_empty() {
        return Err(LatticeDynamicsGraphError::MalformedMatterClaim);
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(LatticeDynamicsGraphError::ClaimScaleMismatch(claim.scale));
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(LatticeDynamicsGraphError::ClaimNotScientific);
    }
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(LatticeDynamicsGraphError::ClaimStageNotComputational(
            claim.validation_stage,
        ));
    }
    Ok(())
}

fn bind_aggregation(
    target_claim: &MatterClaim,
    receipt: &LatticeDynamicsAggregationReceipt,
    batches: &[LatticeQPointBatchBinding],
    bundle: &ExternalEvidenceBundle,
) -> Result<LatticeDynamicsAggregationBinding, LatticeDynamicsGraphError> {
    let implementation = bundle.require_role(
        &receipt.implementation_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;
    let configuration = bundle.require_role(
        &receipt.configuration_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let output = bundle.require_role(&receipt.output_artifact_id, EvidenceRole::ArtifactContent)?;
    let output_date = require_date(output)?;
    if output_date.yyyymmdd() != receipt.declared_output_yyyymmdd {
        return Err(LatticeDynamicsGraphError::AggregationDateMismatch {
            receipt_yyyymmdd: receipt.declared_output_yyyymmdd,
            evidence_yyyymmdd: output_date.yyyymmdd(),
        });
    }

    let expected_matrix_ids: BTreeSet<String> = batches
        .iter()
        .flat_map(|batch| batch.samples.iter())
        .map(|sample| sample.dynamical_matrix_evidence_id.clone())
        .collect();
    let supplied_matrix_ids: BTreeSet<String> = receipt
        .input_dynamical_matrix_artifact_ids
        .iter()
        .cloned()
        .collect();
    if expected_matrix_ids != supplied_matrix_ids {
        return Err(LatticeDynamicsGraphError::AggregationInputSetMismatch);
    }

    let mut semantic_refs = vec![implementation, configuration, output];
    let mut dependency_refs = Vec::new();
    for id in &receipt.dependency_snapshot_ids {
        let reference = bundle.require_role(id, EvidenceRole::DependencySnapshot)?;
        dependency_refs.push(reference);
        semantic_refs.push(reference);
    }
    ensure_distinct_content(&semantic_refs)?;

    for reference in [implementation, configuration]
        .into_iter()
        .chain(dependency_refs.iter().copied())
    {
        let date = require_date(reference)?;
        if date > output_date {
            return Err(LatticeDynamicsGraphError::AggregationDependencyPostdatesOutput(
                reference.id.as_str().to_string(),
            ));
        }
    }

    for batch in batches {
        if batch.execution.execution_claimed_date > output_date {
            return Err(LatticeDynamicsGraphError::AggregationPredatesBatch(
                batch.batch_id.clone(),
            ));
        }
        for sample in &batch.samples {
            if sample.dynamical_matrix_sha256 == output.content_sha256 {
                return Err(LatticeDynamicsGraphError::AggregationOutputAliasesMatrix {
                    batch_id: batch.batch_id.clone(),
                    q_point_id: sample.q_point_id.clone(),
                });
            }
        }
    }

    if !target_claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == output.id.as_str())
    {
        return Err(LatticeDynamicsGraphError::AggregateOutputMissingFromClaim);
    }

    Ok(LatticeDynamicsAggregationBinding {
        aggregator_name: receipt.aggregator_name.clone(),
        aggregator_version: receipt.aggregator_version.clone(),
        implementation_evidence_id: implementation.id.as_str().to_string(),
        implementation_sha256: implementation.content_sha256.clone(),
        configuration_evidence_id: configuration.id.as_str().to_string(),
        configuration_sha256: configuration.content_sha256.clone(),
        output_evidence_id: output.id.as_str().to_string(),
        output_sha256: output.content_sha256.clone(),
        output_claimed_date: output_date,
        dependency_evidence_ids: dependency_refs
            .iter()
            .map(|reference| reference.id.as_str().to_string())
            .collect(),
        dependency_sha256: dependency_refs
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        input_dynamical_matrix_evidence_ids: receipt
            .input_dynamical_matrix_artifact_ids
            .clone(),
        minimum_frequency_bits: receipt.minimum_frequency_bits,
        imaginary_frequency_tolerance_bits: receipt.imaginary_frequency_tolerance_bits,
        sampled_q_point_count: receipt.sampled_q_point_count,
    })
}

fn ensure_distinct_content(
    references: &[&ExternalEvidenceReference],
) -> Result<(), LatticeDynamicsGraphError> {
    let mut seen = BTreeMap::<String, String>::new();
    for reference in references {
        let digest = reference.content_sha256.as_str().to_string();
        if let Some(first_id) = seen.insert(digest, reference.id.as_str().to_string()) {
            return Err(LatticeDynamicsGraphError::DuplicateAggregationContent {
                first_id,
                second_id: reference.id.as_str().to_string(),
            });
        }
    }
    Ok(())
}

fn require_date(
    reference: &ExternalEvidenceReference,
) -> Result<ClaimedUtcDate, LatticeDynamicsGraphError> {
    reference
        .claimed_utc_date
        .ok_or_else(|| LatticeDynamicsGraphError::MissingClaimedDate(reference.id.as_str().to_string()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeDynamicsGraphError {
    CrystalEvidence(CrystalEvidenceBindingError),
    ExternalEvidence(ExternalEvidenceError),
    LeafExecution(CrystalExecutionError),
    EmptyField(&'static str),
    MissingAggregationDate,
    NoQPoints,
    InvalidMinimumFrequency,
    InvalidImaginaryFrequencyTolerance,
    NoDynamicalMatrixInputs,
    EmptyArtifactIdentity,
    DuplicateAggregationIdentity(String),
    DuplicateDynamicalMatrixIdentity(String),
    ParentEvidenceBindingMismatch,
    MissingDynamicalStabilityClaim,
    TargetClaimIdentityMismatch { graph: String, claim: String },
    DynamicalDiagnosticMismatch,
    QPointCountOverflow,
    MinimumFrequencyMismatch,
    ImaginaryFrequencyToleranceMismatch,
    QPointCountMismatch { expected: u32, actual: u32 },
    QPointSampleCountMismatch { expected: u32, actual: usize },
    MissingAggregateSolverProfile,
    AggregateSolverIdentityMismatch,
    AggregateOutputMissingFromClaim,
    MissingDynamicalCriterion,
    NoExecutionBatches,
    EmptyBatchId,
    EmptyQPointBatch(String),
    DuplicateBatchId(String),
    BatchAliasesAggregateClaim { batch_id: String, claim_id: String },
    DuplicateBatchClaimId(String),
    EmptyQPointId,
    DuplicateQPointId(String),
    DuplicateQPointCoordinates,
    InvalidQPointCoordinate,
    BatchIsNotLatticeDynamicsTask,
    DuplicateExecutionIdentity(String),
    DuplicateExecutionContent,
    DynamicalMatrixNotExecutionOutput { batch_id: String, q_point_id: String },
    DynamicalMatrixMissingFromClaim { batch_id: String, q_point_id: String },
    DynamicalMatrixPredatesExecution { batch_id: String, q_point_id: String },
    DuplicateDynamicalMatrixContent,
    BatchOodNotPropagated,
    MalformedMatterClaim,
    ClaimScaleMismatch(MatterScale),
    ClaimNotScientific,
    ClaimStageNotComputational(MatterValidationStage),
    AggregationInputSetMismatch,
    AggregationDateMismatch { receipt_yyyymmdd: u32, evidence_yyyymmdd: u32 },
    AggregationDependencyPostdatesOutput(String),
    AggregationPredatesBatch(String),
    AggregationOutputAliasesMatrix { batch_id: String, q_point_id: String },
    DuplicateAggregationContent { first_id: String, second_id: String },
    MissingClaimedDate(String),
}

impl From<CrystalEvidenceBindingError> for LatticeDynamicsGraphError {
    fn from(value: CrystalEvidenceBindingError) -> Self {
        Self::CrystalEvidence(value)
    }
}

impl From<ExternalEvidenceError> for LatticeDynamicsGraphError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl From<CrystalExecutionError> for LatticeDynamicsGraphError {
    fn from(value: CrystalExecutionError) -> Self {
        Self::LeafExecution(value)
    }
}

impl fmt::Display for LatticeDynamicsGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lattice-dynamics execution graph rejected: {self:?}")
    }
}

impl std::error::Error for LatticeDynamicsGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, x: f64) -> LatticeQPointSampleReceipt {
        LatticeQPointSampleReceipt {
            q_point_id: id.into(),
            q_fractional_bits: [x.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()],
            dynamical_matrix_artifact_id: format!("dynmat:{id}"),
        }
    }

    #[test]
    fn aggregation_requires_finite_frequency_tolerance_and_nonzero_q_count() {
        let good = LatticeDynamicsAggregationReceipt {
            aggregator_name: "phonon-aggregate".into(),
            aggregator_version: "1".into(),
            implementation_snapshot_id: "impl".into(),
            configuration_artifact_id: "config".into(),
            output_artifact_id: "output".into(),
            dependency_snapshot_ids: Vec::new(),
            input_dynamical_matrix_artifact_ids: vec!["dynmat:q0".into()],
            declared_output_yyyymmdd: 20260908,
            minimum_frequency_bits: 0.2_f64.to_bits(),
            imaginary_frequency_tolerance_bits: 0.01_f64.to_bits(),
            sampled_q_point_count: 1,
        };
        good.validate().unwrap();

        let bad = LatticeDynamicsAggregationReceipt {
            sampled_q_point_count: 0,
            ..good
        };
        assert_eq!(bad.validate().unwrap_err(), LatticeDynamicsGraphError::NoQPoints);
    }

    #[test]
    fn q_coordinates_must_be_finite() {
        let bad = LatticeQPointSampleReceipt {
            q_fractional_bits: [0.0_f64.to_bits(), f64::NAN.to_bits(), 0.5_f64.to_bits()],
            ..sample("q0", 0.0)
        };
        assert_eq!(
            validate_sample_shape(&bad).unwrap_err(),
            LatticeDynamicsGraphError::InvalidQPointCoordinate
        );
    }

    #[test]
    fn batch_shape_allows_multiple_q_points_per_execution() {
        use symthaea_epistemic_types::{MatterScale, MatterSolverProfile};
        let claim = MatterClaim::local_computational(
            "claim:batch",
            "two q points in one solver execution",
            MatterScale::Crystal,
            MatterValidationStage::PhysicsSimulated,
            MatterSolverProfile {
                name: "solver".into(),
                version: "1".into(),
                method: "phonon batch".into(),
                capabilities: Vec::new(),
                limitations: Vec::new(),
            },
        )
        .unwrap();
        let batch = LatticeQPointBatchReceipt {
            batch_id: "batch:0".into(),
            claim,
            execution: CrystalSolverExecutionReceipt {
                claim_id: "claim:batch".into(),
                task: CrystalSolverTask::LatticeDynamics,
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
                output_artifact_ids: vec!["dynmat:q0".into(), "dynmat:q1".into()],
                dependency_snapshot_ids: Vec::new(),
            },
            samples: vec![sample("q0", 0.0), sample("q1", 0.5)],
        };
        validate_batch_shape(&[batch], "claim:aggregate").unwrap();
    }
}
