// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Many-execution evidence graph for thermodynamic phase competition.
//!
//! A convex-hull / energy-above-hull claim is not one solver invocation. It is
//! an aggregate over a target-phase energy, one or more competing-phase energies,
//! and an explicit phase-diagram aggregation method. This module keeps those
//! objects separate and binds every numerical leaf to canonical solver evidence.
//!
//! The graph is intentionally limited to phase competition. Relaxation restart
//! sequences and multi-q-point lattice-dynamics workflows need their own typed
//! graph semantics instead of being smuggled into a generic "workflow" bucket.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterSolverCapability, MatterValidationStage,
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
use super::matter_crystal_phase::{CrystalPhaseDiagnosticSnapshot, CrystalPhaseEvidenceSlot};

/// Semantic role of one periodic-energy leaf in the phase-comparison set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhaseEnergyRole {
    TargetPhase,
    CompetingPhase,
}

/// One independently identified phase-energy claim plus the exact solver run
/// that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseEnergyLeafReceipt {
    pub phase_id: String,
    pub role: PhaseEnergyRole,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionReceipt,
}

/// Exact aggregation operation that turns phase energies into the stored
/// energy-above-hull result.
///
/// This is deliberately not a `SolverExecution`: phase-diagram construction is
/// a transformation over already-computed energies. The output is an exact
/// artifact produced by a named/versioned implementation and configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseDiagramAggregationReceipt {
    pub aggregator_name: String,
    pub aggregator_version: String,
    pub implementation_snapshot_id: String,
    pub configuration_artifact_id: String,
    pub output_artifact_id: String,
    pub dependency_snapshot_ids: Vec<String>,
    pub declared_aggregation_yyyymmdd: u32,
    /// Exact IEEE-754 bits of the aggregate energy-above-hull quantity.
    pub energy_above_hull_bits: u64,
    /// Number of distinct competing phases represented by this graph.
    pub competing_phase_count: u32,
}

impl PhaseDiagramAggregationReceipt {
    pub fn validate(&self) -> Result<(), PhaseCompetitionGraphError> {
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
                return Err(PhaseCompetitionGraphError::EmptyField(field));
            }
        }
        if self.declared_aggregation_yyyymmdd == 0 {
            return Err(PhaseCompetitionGraphError::MissingAggregationDate);
        }
        if self.competing_phase_count == 0 {
            return Err(PhaseCompetitionGraphError::NoCompetingPhases);
        }
        let energy = f64::from_bits(self.energy_above_hull_bits);
        if !energy.is_finite() || energy < 0.0 {
            return Err(PhaseCompetitionGraphError::InvalidEnergyAboveHull);
        }
        if self
            .dependency_snapshot_ids
            .iter()
            .any(|identity| identity.trim().is_empty())
        {
            return Err(PhaseCompetitionGraphError::EmptyDependencyIdentity);
        }
        let mut ids = BTreeSet::new();
        for id in [
            self.implementation_snapshot_id.as_str(),
            self.configuration_artifact_id.as_str(),
            self.output_artifact_id.as_str(),
        ]
        .into_iter()
        .chain(self.dependency_snapshot_ids.iter().map(String::as_str))
        {
            if !ids.insert(id) {
                return Err(PhaseCompetitionGraphError::DuplicateAggregationIdentity(
                    id.to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Complete phase-competition derivation supplied by an external workflow or
/// local phase-diagram evaluator.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseCompetitionExecutionGraphReceipt {
    /// Must equal the canonical phase-competition Matter claim id from #792/#801.
    pub target_claim_id: String,
    pub leaves: Vec<PhaseEnergyLeafReceipt>,
    pub aggregation: PhaseDiagramAggregationReceipt,
}

/// Canonically bound periodic leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseEnergyLeafBinding {
    pub phase_id: String,
    pub role: PhaseEnergyRole,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionBinding,
}

/// Bound phase-diagram transformation artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseDiagramAggregationBinding {
    pub aggregator_name: String,
    pub aggregator_version: String,
    pub implementation_evidence_id: String,
    pub implementation_content_sha256: Sha256Digest,
    pub configuration_evidence_id: String,
    pub configuration_content_sha256: Sha256Digest,
    pub output_evidence_id: String,
    pub output_content_sha256: Sha256Digest,
    pub output_claimed_date: ClaimedUtcDate,
    pub dependency_evidence_ids: Vec<String>,
    pub dependency_content_sha256: Vec<Sha256Digest>,
    pub energy_above_hull_bits: u64,
    pub competing_phase_count: u32,
}

/// Fully bound many-execution phase-competition graph.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseCompetitionExecutionGraphBinding {
    pub evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    pub target_claim_id: String,
    pub leaves: Vec<PhaseEnergyLeafBinding>,
    pub aggregation: PhaseDiagramAggregationBinding,
    /// Preregistered hull criterion must precede every energy leaf and the
    /// aggregate output at the evidence plane's declared day granularity.
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

/// Bind a realistic many-calculation phase-competition graph to the canonical
/// crystal evidence object.
pub fn bind_phase_competition_execution_graph(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    graph: PhaseCompetitionExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<PhaseCompetitionExecutionGraphBinding, PhaseCompetitionGraphError> {
    let rebound = bind_crystal_phase_evidence(evidence_bound.admitted.clone(), bundle)?;
    if rebound != evidence_bound {
        return Err(PhaseCompetitionGraphError::ParentEvidenceBindingMismatch);
    }

    let phase_snapshot = evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition)
        .ok_or(PhaseCompetitionGraphError::MissingPhaseCompetitionClaim)?;
    let target_claim = &phase_snapshot.claim;
    if graph.target_claim_id != target_claim.claim_id {
        return Err(PhaseCompetitionGraphError::TargetClaimIdentityMismatch {
            graph: graph.target_claim_id.clone(),
            claim: target_claim.claim_id.clone(),
        });
    }

    let (expected_energy_bits, expected_competing_count) = match &phase_snapshot.diagnostic {
        CrystalPhaseDiagnosticSnapshot::ThermodynamicPhaseCompetition {
            energy_above_hull_bits,
            competing_phase_count,
            ..
        } => (*energy_above_hull_bits, *competing_phase_count),
        _ => return Err(PhaseCompetitionGraphError::PhaseDiagnosticMismatch),
    };
    graph.aggregation.validate()?;
    if graph.aggregation.energy_above_hull_bits != expected_energy_bits {
        return Err(PhaseCompetitionGraphError::EnergyAboveHullMismatch);
    }
    if graph.aggregation.competing_phase_count != expected_competing_count {
        return Err(PhaseCompetitionGraphError::CompetingPhaseCountMismatch {
            expected: expected_competing_count,
            actual: graph.aggregation.competing_phase_count,
        });
    }

    let criterion = evidence_bound
        .criterion_bindings
        .iter()
        .find(|binding| binding.slot == CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition)
        .ok_or(PhaseCompetitionGraphError::MissingHullCriterion)?;

    validate_target_aggregate_claim(target_claim, &graph.aggregation)?;
    let expected_competitors = usize::try_from(expected_competing_count)
        .map_err(|_| PhaseCompetitionGraphError::CompetingPhaseCountOverflow)?;
    let (target_count, competitor_count) = validate_leaf_set_shape(
        &graph.leaves,
        &graph.target_claim_id,
        expected_competitors,
    )?;
    debug_assert_eq!(target_count, 1);
    debug_assert_eq!(competitor_count, expected_competitors);

    let mut leaf_bindings = Vec::with_capacity(graph.leaves.len());
    let mut criterion_chronology = Vec::with_capacity(graph.leaves.len() + 1);
    let mut execution_ids = BTreeSet::new();
    let mut execution_digests = BTreeMap::<String, String>::new();
    let mut any_leaf_ood = false;

    for leaf in &graph.leaves {
        validate_phase_energy_leaf(leaf)?;
        if !execution_ids.insert(leaf.execution.execution_evidence_id.clone()) {
            return Err(PhaseCompetitionGraphError::DuplicateLeafExecutionIdentity(
                leaf.execution.execution_evidence_id.clone(),
            ));
        }
        let binding = bind_leaf_execution_references(&leaf.execution, bundle)?;
        let digest = binding.execution_content_sha256.as_str().to_string();
        if let Some(previous_phase) = execution_digests.insert(digest, leaf.phase_id.clone()) {
            return Err(PhaseCompetitionGraphError::DuplicateLeafExecutionContent {
                first_phase: previous_phase,
                second_phase: leaf.phase_id.clone(),
            });
        }
        if leaf
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
        {
            any_leaf_ood = true;
        }
        criterion_chronology.push(bundle.require_preregistration_before(
            &criterion.evidence_id,
            &binding.execution_evidence_id,
        )?);
        leaf_bindings.push(PhaseEnergyLeafBinding {
            phase_id: leaf.phase_id.clone(),
            role: leaf.role,
            claim: leaf.claim.clone(),
            execution: binding,
        });
    }

    if any_leaf_ood
        && !target_claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(PhaseCompetitionGraphError::LeafOodNotPropagated);
    }

    let aggregation = bind_aggregation(
        target_claim,
        &graph.aggregation,
        &leaf_bindings,
        bundle,
    )?;
    criterion_chronology.push(bundle.require_preregistration_before(
        &criterion.evidence_id,
        &aggregation.output_evidence_id,
    )?);

    Ok(PhaseCompetitionExecutionGraphBinding {
        evidence_bound,
        target_claim_id: graph.target_claim_id,
        leaves: leaf_bindings,
        aggregation,
        criterion_chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn validate_target_aggregate_claim(
    claim: &MatterClaim,
    aggregation: &PhaseDiagramAggregationReceipt,
) -> Result<(), PhaseCompetitionGraphError> {
    validate_computational_claim_shape(claim)?;
    let solver = claim
        .solver
        .as_ref()
        .ok_or(PhaseCompetitionGraphError::MissingAggregateSolverProfile)?;
    if solver.name != aggregation.aggregator_name || solver.version != aggregation.aggregator_version {
        return Err(PhaseCompetitionGraphError::AggregateSolverIdentityMismatch);
    }
    if !claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == aggregation.output_artifact_id)
    {
        return Err(PhaseCompetitionGraphError::AggregateOutputMissingFromClaim);
    }
    Ok(())
}

fn validate_leaf_set_shape(
    leaves: &[PhaseEnergyLeafReceipt],
    aggregate_claim_id: &str,
    expected_competitors: usize,
) -> Result<(usize, usize), PhaseCompetitionGraphError> {
    if leaves.len() != expected_competitors.saturating_add(1) {
        return Err(PhaseCompetitionGraphError::LeafCountMismatch {
            expected: expected_competitors.saturating_add(1),
            actual: leaves.len(),
        });
    }
    let mut phase_ids = BTreeSet::new();
    let mut claim_ids = BTreeSet::new();
    let mut target_count = 0usize;
    let mut competitor_count = 0usize;
    for leaf in leaves {
        if leaf.phase_id.trim().is_empty() {
            return Err(PhaseCompetitionGraphError::EmptyPhaseId);
        }
        if !phase_ids.insert(leaf.phase_id.clone()) {
            return Err(PhaseCompetitionGraphError::DuplicatePhaseId(
                leaf.phase_id.clone(),
            ));
        }
        if leaf.claim.claim_id == aggregate_claim_id {
            return Err(PhaseCompetitionGraphError::LeafAliasesAggregateClaim(
                leaf.claim.claim_id.clone(),
            ));
        }
        if !claim_ids.insert(leaf.claim.claim_id.clone()) {
            return Err(PhaseCompetitionGraphError::DuplicateLeafClaimId(
                leaf.claim.claim_id.clone(),
            ));
        }
        match leaf.role {
            PhaseEnergyRole::TargetPhase => target_count += 1,
            PhaseEnergyRole::CompetingPhase => competitor_count += 1,
        }
    }
    if target_count != 1 {
        return Err(PhaseCompetitionGraphError::TargetPhaseCardinality(target_count));
    }
    if competitor_count != expected_competitors {
        return Err(PhaseCompetitionGraphError::CompetingPhaseCountMismatch {
            expected: expected_competitors as u32,
            actual: competitor_count as u32,
        });
    }
    Ok((target_count, competitor_count))
}

fn validate_phase_energy_leaf(leaf: &PhaseEnergyLeafReceipt) -> Result<(), PhaseCompetitionGraphError> {
    leaf.execution
        .validate()
        .map_err(PhaseCompetitionGraphError::LeafExecution)?;
    if leaf.execution.task != CrystalSolverTask::PeriodicElectronicStructure {
        return Err(PhaseCompetitionGraphError::LeafIsNotPeriodicEnergyTask);
    }
    if leaf.execution.claim_id != leaf.claim.claim_id {
        return Err(PhaseCompetitionGraphError::LeafClaimIdentityMismatch {
            receipt: leaf.execution.claim_id.clone(),
            claim: leaf.claim.claim_id.clone(),
        });
    }
    validate_computational_claim_shape(&leaf.claim)?;
    let solver = leaf
        .claim
        .solver
        .as_ref()
        .ok_or(PhaseCompetitionGraphError::MissingLeafSolverProfile)?;
    if solver.name != leaf.execution.solver_name || solver.version != leaf.execution.solver_version {
        return Err(PhaseCompetitionGraphError::LeafSolverIdentityMismatch);
    }
    if !solver.supports(MatterSolverCapability::PeriodicElectronicStructure) {
        return Err(PhaseCompetitionGraphError::LeafPeriodicCapabilityMissing);
    }
    if !leaf
        .claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == leaf.execution.execution_evidence_id)
    {
        return Err(PhaseCompetitionGraphError::LeafExecutionMissingFromClaim);
    }
    Ok(())
}

fn validate_computational_claim_shape(claim: &MatterClaim) -> Result<(), PhaseCompetitionGraphError> {
    if claim.claim_id.trim().is_empty() || claim.statement.trim().is_empty() {
        return Err(PhaseCompetitionGraphError::MalformedMatterClaim);
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(PhaseCompetitionGraphError::ClaimScaleMismatch(claim.scale));
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(PhaseCompetitionGraphError::ClaimNotScientific);
    }
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(PhaseCompetitionGraphError::ClaimStageNotComputational(
            claim.validation_stage,
        ));
    }
    Ok(())
}

fn bind_leaf_execution_references(
    receipt: &CrystalSolverExecutionReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalSolverExecutionBinding, PhaseCompetitionGraphError> {
    let execution = bundle.require_role(&receipt.execution_evidence_id, EvidenceRole::SolverExecution)?;
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
        return Err(PhaseCompetitionGraphError::LeafExecutionDateMismatch {
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
            return Err(PhaseCompetitionGraphError::LeafDependencyPostdatesExecution {
                evidence_id: reference.id.as_str().to_string(),
                dependency_yyyymmdd: date.yyyymmdd(),
                execution_yyyymmdd: execution_date.yyyymmdd(),
            });
        }
    }
    for reference in &output_refs {
        let date = require_date(reference)?;
        if date < execution_date {
            return Err(PhaseCompetitionGraphError::LeafOutputPredatesExecution {
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

fn bind_aggregation(
    target_claim: &MatterClaim,
    receipt: &PhaseDiagramAggregationReceipt,
    leaves: &[PhaseEnergyLeafBinding],
    bundle: &ExternalEvidenceBundle,
) -> Result<PhaseDiagramAggregationBinding, PhaseCompetitionGraphError> {
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
    if output_date.yyyymmdd() != receipt.declared_aggregation_yyyymmdd {
        return Err(PhaseCompetitionGraphError::AggregationDateMismatch {
            receipt_yyyymmdd: receipt.declared_aggregation_yyyymmdd,
            evidence_yyyymmdd: output_date.yyyymmdd(),
        });
    }

    let mut semantic_refs = vec![implementation, configuration, output];
    let mut dependencies = Vec::new();
    for id in &receipt.dependency_snapshot_ids {
        let reference = bundle.require_role(id, EvidenceRole::DependencySnapshot)?;
        dependencies.push(reference);
        semantic_refs.push(reference);
    }
    ensure_distinct_content(&semantic_refs)?;

    for reference in [implementation, configuration]
        .into_iter()
        .chain(dependencies.iter().copied())
    {
        let date = require_date(reference)?;
        if date > output_date {
            return Err(PhaseCompetitionGraphError::AggregationDependencyPostdatesOutput {
                evidence_id: reference.id.as_str().to_string(),
                dependency_yyyymmdd: date.yyyymmdd(),
                output_yyyymmdd: output_date.yyyymmdd(),
            });
        }
    }
    for leaf in leaves {
        if leaf.execution.execution_claimed_date > output_date {
            return Err(PhaseCompetitionGraphError::AggregationPredatesLeafExecution {
                phase_id: leaf.phase_id.clone(),
                leaf_yyyymmdd: leaf.execution.execution_claimed_date.yyyymmdd(),
                output_yyyymmdd: output_date.yyyymmdd(),
            });
        }
        if leaf
            .execution
            .output_content_sha256
            .iter()
            .any(|digest| digest == &output.content_sha256)
        {
            return Err(PhaseCompetitionGraphError::AggregationOutputAliasesLeafOutput(
                leaf.phase_id.clone(),
            ));
        }
    }

    if !target_claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == output.id.as_str())
    {
        return Err(PhaseCompetitionGraphError::AggregateOutputMissingFromClaim);
    }

    Ok(PhaseDiagramAggregationBinding {
        aggregator_name: receipt.aggregator_name.clone(),
        aggregator_version: receipt.aggregator_version.clone(),
        implementation_evidence_id: implementation.id.as_str().to_string(),
        implementation_content_sha256: implementation.content_sha256.clone(),
        configuration_evidence_id: configuration.id.as_str().to_string(),
        configuration_content_sha256: configuration.content_sha256.clone(),
        output_evidence_id: output.id.as_str().to_string(),
        output_content_sha256: output.content_sha256.clone(),
        output_claimed_date: output_date,
        dependency_evidence_ids: dependencies
            .iter()
            .map(|reference| reference.id.as_str().to_string())
            .collect(),
        dependency_content_sha256: dependencies
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        energy_above_hull_bits: receipt.energy_above_hull_bits,
        competing_phase_count: receipt.competing_phase_count,
    })
}

fn ensure_distinct_content(
    references: &[&ExternalEvidenceReference],
) -> Result<(), PhaseCompetitionGraphError> {
    let mut seen = BTreeMap::<String, String>::new();
    for reference in references {
        let digest = reference.content_sha256.as_str().to_string();
        if let Some(first_id) = seen.insert(digest, reference.id.as_str().to_string()) {
            return Err(PhaseCompetitionGraphError::DuplicateContentIdentity {
                first_id,
                second_id: reference.id.as_str().to_string(),
            });
        }
    }
    Ok(())
}

fn require_date(
    reference: &ExternalEvidenceReference,
) -> Result<ClaimedUtcDate, PhaseCompetitionGraphError> {
    reference
        .claimed_utc_date
        .ok_or_else(|| PhaseCompetitionGraphError::MissingClaimedDate(reference.id.as_str().to_string()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhaseCompetitionGraphError {
    CrystalEvidence(CrystalEvidenceBindingError),
    ExternalEvidence(ExternalEvidenceError),
    LeafExecution(CrystalExecutionError),
    ParentEvidenceBindingMismatch,
    MissingPhaseCompetitionClaim,
    TargetClaimIdentityMismatch { graph: String, claim: String },
    PhaseDiagnosticMismatch,
    MissingHullCriterion,
    EmptyField(&'static str),
    MissingAggregationDate,
    InvalidEnergyAboveHull,
    NoCompetingPhases,
    EmptyDependencyIdentity,
    DuplicateAggregationIdentity(String),
    EnergyAboveHullMismatch,
    CompetingPhaseCountMismatch { expected: u32, actual: u32 },
    CompetingPhaseCountOverflow,
    MissingAggregateSolverProfile,
    AggregateSolverIdentityMismatch,
    AggregateOutputMissingFromClaim,
    LeafCountMismatch { expected: usize, actual: usize },
    EmptyPhaseId,
    DuplicatePhaseId(String),
    LeafAliasesAggregateClaim(String),
    DuplicateLeafClaimId(String),
    TargetPhaseCardinality(usize),
    DuplicateLeafExecutionIdentity(String),
    DuplicateLeafExecutionContent { first_phase: String, second_phase: String },
    LeafIsNotPeriodicEnergyTask,
    LeafClaimIdentityMismatch { receipt: String, claim: String },
    MalformedMatterClaim,
    ClaimScaleMismatch(MatterScale),
    ClaimNotScientific,
    ClaimStageNotComputational(MatterValidationStage),
    MissingLeafSolverProfile,
    LeafSolverIdentityMismatch,
    LeafPeriodicCapabilityMissing,
    LeafExecutionMissingFromClaim,
    LeafOodNotPropagated,
    LeafExecutionDateMismatch { receipt_yyyymmdd: u32, evidence_yyyymmdd: u32 },
    DuplicateContentIdentity { first_id: String, second_id: String },
    MissingClaimedDate(String),
    LeafDependencyPostdatesExecution { evidence_id: String, dependency_yyyymmdd: u32, execution_yyyymmdd: u32 },
    LeafOutputPredatesExecution { evidence_id: String, output_yyyymmdd: u32, execution_yyyymmdd: u32 },
    AggregationDateMismatch { receipt_yyyymmdd: u32, evidence_yyyymmdd: u32 },
    AggregationDependencyPostdatesOutput { evidence_id: String, dependency_yyyymmdd: u32, output_yyyymmdd: u32 },
    AggregationPredatesLeafExecution { phase_id: String, leaf_yyyymmdd: u32, output_yyyymmdd: u32 },
    AggregationOutputAliasesLeafOutput(String),
}

impl From<CrystalEvidenceBindingError> for PhaseCompetitionGraphError {
    fn from(value: CrystalEvidenceBindingError) -> Self {
        Self::CrystalEvidence(value)
    }
}

impl From<ExternalEvidenceError> for PhaseCompetitionGraphError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for PhaseCompetitionGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrystalEvidence(error) => write!(f, "parent crystal evidence rejected: {error}"),
            Self::ExternalEvidence(error) => write!(f, "{error}"),
            Self::LeafExecution(error) => write!(f, "phase-energy leaf execution rejected: {error}"),
            Self::ParentEvidenceBindingMismatch => write!(f, "stored crystal evidence does not equal a fresh #801 rebind"),
            Self::MissingPhaseCompetitionClaim => write!(f, "crystal evidence has no thermodynamic phase-competition slot"),
            Self::TargetClaimIdentityMismatch { graph, claim } => write!(f, "phase graph target claim `{graph}` does not match canonical phase claim `{claim}`"),
            Self::PhaseDiagnosticMismatch => write!(f, "phase-competition slot carries the wrong diagnostic shape"),
            Self::MissingHullCriterion => write!(f, "phase-competition graph has no canonical hull preregistration binding"),
            Self::EmptyField(field) => write!(f, "phase graph field `{field}` is empty"),
            Self::MissingAggregationDate => write!(f, "phase-diagram aggregation date is missing"),
            Self::InvalidEnergyAboveHull => write!(f, "energy above hull must be finite and non-negative"),
            Self::NoCompetingPhases => write!(f, "phase competition requires at least one competing phase"),
            Self::EmptyDependencyIdentity => write!(f, "phase aggregation dependency identity is empty"),
            Self::DuplicateAggregationIdentity(id) => write!(f, "phase aggregation reuses evidence id `{id}`"),
            Self::EnergyAboveHullMismatch => write!(f, "phase aggregation energy-above-hull bits differ from the admitted crystal diagnostic"),
            Self::CompetingPhaseCountMismatch { expected, actual } => write!(f, "phase graph competing-phase count {actual} does not match admitted count {expected}"),
            Self::CompetingPhaseCountOverflow => write!(f, "competing-phase count cannot be represented on this platform"),
            Self::MissingAggregateSolverProfile => write!(f, "phase-competition aggregate claim has no solver/aggregator profile"),
            Self::AggregateSolverIdentityMismatch => write!(f, "phase aggregation name/version does not match the aggregate Matter claim"),
            Self::AggregateOutputMissingFromClaim => write!(f, "aggregate Matter claim does not retain the phase-diagram output evidence id"),
            Self::LeafCountMismatch { expected, actual } => write!(f, "phase graph has {actual} leaves; expected exactly {expected}"),
            Self::EmptyPhaseId => write!(f, "phase graph phase id is empty"),
            Self::DuplicatePhaseId(id) => write!(f, "phase graph phase id `{id}` occurs more than once"),
            Self::LeafAliasesAggregateClaim(id) => write!(f, "phase-energy leaf claim `{id}` aliases the aggregate stability claim"),
            Self::DuplicateLeafClaimId(id) => write!(f, "phase-energy leaf claim id `{id}` occurs more than once"),
            Self::TargetPhaseCardinality(count) => write!(f, "phase graph requires exactly one target phase, found {count}"),
            Self::DuplicateLeafExecutionIdentity(id) => write!(f, "phase-energy execution id `{id}` occurs more than once"),
            Self::DuplicateLeafExecutionContent { first_phase, second_phase } => write!(f, "phase-energy leaves `{first_phase}` and `{second_phase}` resolve to the same execution content"),
            Self::LeafIsNotPeriodicEnergyTask => write!(f, "phase-energy leaf must be a periodic-electronic-structure execution"),
            Self::LeafClaimIdentityMismatch { receipt, claim } => write!(f, "phase-energy receipt claim `{receipt}` does not match leaf Matter claim `{claim}`"),
            Self::MalformedMatterClaim => write!(f, "phase graph contains an empty Matter claim id or statement"),
            Self::ClaimScaleMismatch(scale) => write!(f, "phase graph Matter claim has non-crystal scale {scale:?}"),
            Self::ClaimNotScientific => write!(f, "phase graph Matter claim is not in Scientific context"),
            Self::ClaimStageNotComputational(stage) => write!(f, "phase graph Matter claim stage {stage:?} is not computational"),
            Self::MissingLeafSolverProfile => write!(f, "phase-energy leaf has no solver profile"),
            Self::LeafSolverIdentityMismatch => write!(f, "phase-energy solver name/version does not match its execution receipt"),
            Self::LeafPeriodicCapabilityMissing => write!(f, "phase-energy solver does not advertise PeriodicElectronicStructure"),
            Self::LeafExecutionMissingFromClaim => write!(f, "phase-energy Matter claim does not retain its execution evidence id"),
            Self::LeafOodNotPropagated => write!(f, "out-of-domain phase-energy evidence was not propagated to the aggregate phase claim"),
            Self::LeafExecutionDateMismatch { receipt_yyyymmdd, evidence_yyyymmdd } => write!(f, "phase leaf receipt date {receipt_yyyymmdd} differs from canonical execution date {evidence_yyyymmdd}"),
            Self::DuplicateContentIdentity { first_id, second_id } => write!(f, "phase graph semantic evidence `{first_id}` and `{second_id}` resolve to identical content"),
            Self::MissingClaimedDate(id) => write!(f, "phase graph evidence `{id}` has no claimed UTC date"),
            Self::LeafDependencyPostdatesExecution { evidence_id, dependency_yyyymmdd, execution_yyyymmdd } => write!(f, "phase leaf dependency `{evidence_id}` date {dependency_yyyymmdd} postdates execution {execution_yyyymmdd}"),
            Self::LeafOutputPredatesExecution { evidence_id, output_yyyymmdd, execution_yyyymmdd } => write!(f, "phase leaf output `{evidence_id}` date {output_yyyymmdd} predates execution {execution_yyyymmdd}"),
            Self::AggregationDateMismatch { receipt_yyyymmdd, evidence_yyyymmdd } => write!(f, "phase aggregation date {receipt_yyyymmdd} differs from output evidence date {evidence_yyyymmdd}"),
            Self::AggregationDependencyPostdatesOutput { evidence_id, dependency_yyyymmdd, output_yyyymmdd } => write!(f, "phase aggregation dependency `{evidence_id}` date {dependency_yyyymmdd} postdates output {output_yyyymmdd}"),
            Self::AggregationPredatesLeafExecution { phase_id, leaf_yyyymmdd, output_yyyymmdd } => write!(f, "phase aggregation output {output_yyyymmdd} predates leaf `{phase_id}` execution {leaf_yyyymmdd}"),
            Self::AggregationOutputAliasesLeafOutput(phase_id) => write!(f, "phase aggregation output aliases a raw output from phase `{phase_id}`"),
        }
    }
}

impl std::error::Error for PhaseCompetitionGraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{MatterEvidenceRef, MatterSolverProfile};
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(id: &str, role: EvidenceRole, ch: char, date: u32) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(ch),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            Some(date),
            Some("phase-graph-test".to_string()),
        )
        .unwrap()
    }

    fn periodic_claim(id: &str, execution_id: &str) -> MatterClaim {
        MatterClaim::local_computational(
            id,
            format!("periodic energy for {id}"),
            MatterScale::Crystal,
            MatterValidationStage::HighFidelitySimulated,
            MatterSolverProfile {
                name: "periodic-solver".into(),
                version: "1".into(),
                method: "periodic energy".into(),
                capabilities: vec![MatterSolverCapability::PeriodicElectronicStructure],
                limitations: Vec::new(),
            },
        )
        .unwrap()
        .with_evidence(MatterEvidenceRef {
            evidence_id: execution_id.into(),
            run_id: None,
            config_hash: None,
            dataset_ids: Vec::new(),
        })
    }

    fn execution(claim_id: &str, execution_id: &str) -> CrystalSolverExecutionReceipt {
        CrystalSolverExecutionReceipt {
            claim_id: claim_id.into(),
            task: CrystalSolverTask::PeriodicElectronicStructure,
            execution_evidence_id: execution_id.into(),
            declared_execution_yyyymmdd: 20260902,
            started_unix_ms: 1,
            finished_unix_ms: 2,
            exit_code: 0,
            solver_name: "periodic-solver".into(),
            solver_version: "1".into(),
            adapter_name: "test-adapter".into(),
            adapter_version: "1".into(),
            input_artifact_id: format!("{execution_id}:input"),
            configuration_artifact_id: format!("{execution_id}:config"),
            environment_snapshot_id: format!("{execution_id}:env"),
            parser_snapshot_id: format!("{execution_id}:parser"),
            output_artifact_ids: vec![format!("{execution_id}:output")],
            dependency_snapshot_ids: vec![format!("{execution_id}:dep")],
        }
    }

    #[test]
    fn aggregation_receipt_rejects_invalid_hull_energy() {
        let receipt = PhaseDiagramAggregationReceipt {
            aggregator_name: "phase-evaluator".into(),
            aggregator_version: "1".into(),
            implementation_snapshot_id: "impl".into(),
            configuration_artifact_id: "config".into(),
            output_artifact_id: "output".into(),
            dependency_snapshot_ids: vec![],
            declared_aggregation_yyyymmdd: 20260903,
            energy_above_hull_bits: (-0.1f64).to_bits(),
            competing_phase_count: 1,
        };
        assert_eq!(
            receipt.validate().unwrap_err(),
            PhaseCompetitionGraphError::InvalidEnergyAboveHull
        );
    }

    #[test]
    fn leaf_set_requires_one_target_and_exact_competitor_count() {
        let target = PhaseEnergyLeafReceipt {
            phase_id: "target".into(),
            role: PhaseEnergyRole::TargetPhase,
            claim: periodic_claim("claim:target-energy", "exec-target"),
            execution: execution("claim:target-energy", "exec-target"),
        };
        let competitor = PhaseEnergyLeafReceipt {
            phase_id: "competitor".into(),
            role: PhaseEnergyRole::CompetingPhase,
            claim: periodic_claim("claim:competitor-energy", "exec-competitor"),
            execution: execution("claim:competitor-energy", "exec-competitor"),
        };
        assert_eq!(
            validate_leaf_set_shape(&[target.clone(), competitor.clone()], "aggregate", 1)
                .unwrap(),
            (1, 1)
        );
        let duplicate_target = PhaseEnergyLeafReceipt {
            phase_id: "other-target".into(),
            role: PhaseEnergyRole::TargetPhase,
            claim: periodic_claim("claim:other-target", "exec-other"),
            execution: execution("claim:other-target", "exec-other"),
        };
        assert_eq!(
            validate_leaf_set_shape(&[target, duplicate_target], "aggregate", 1).unwrap_err(),
            PhaseCompetitionGraphError::TargetPhaseCardinality(2)
        );
    }

    #[test]
    fn phase_energy_leaf_requires_periodic_capability_and_claim_binding() {
        let leaf = PhaseEnergyLeafReceipt {
            phase_id: "target".into(),
            role: PhaseEnergyRole::TargetPhase,
            claim: periodic_claim("claim:target", "exec"),
            execution: execution("claim:target", "exec"),
        };
        validate_phase_energy_leaf(&leaf).unwrap();

        let mut wrong = leaf.clone();
        wrong.execution.claim_id = "different".into();
        assert!(matches!(
            validate_phase_energy_leaf(&wrong),
            Err(PhaseCompetitionGraphError::LeafClaimIdentityMismatch { .. })
        ));
    }

    #[test]
    fn leaf_binding_preserves_exact_roles_and_declared_chronology() {
        let receipt = execution("claim:target", "exec");
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("exec", EvidenceRole::SolverExecution, 'a', 20260902),
            reference("exec:input", EvidenceRole::ArtifactContent, 'b', 20260901),
            reference("exec:config", EvidenceRole::ArtifactContent, 'c', 20260901),
            reference("exec:env", EvidenceRole::ImplementationSnapshot, 'd', 20260901),
            reference("exec:parser", EvidenceRole::ImplementationSnapshot, 'e', 20260901),
            reference("exec:output", EvidenceRole::ArtifactContent, 'f', 20260902),
            reference("exec:dep", EvidenceRole::DependencySnapshot, '1', 20260901),
        ])
        .unwrap();
        let bound = bind_leaf_execution_references(&receipt, &bundle).unwrap();
        assert_eq!(bound.execution_evidence_id, "exec");
        assert_eq!(
            bound.reference_interpretation,
            EvidenceReferenceInterpretation::ReferenceOnly
        );
        assert_eq!(
            bound.chronology_interpretation,
            DeclaredChronologyInterpretation::DeclaredChronologyOnly
        );
    }
}
