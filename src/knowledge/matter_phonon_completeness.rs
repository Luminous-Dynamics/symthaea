// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reciprocal-grid completeness for lattice-dynamics evidence.
//!
//! The parent lattice graph proves exact q-point sample lineage and aggregate
//! frequency/tolerance bits. This boundary adds the declared reciprocal grid,
//! maps every sampled q point to one integer grid representative within a
//! preregistered coordinate tolerance, reconstructs full-grid multiplicity, and
//! records the exact ASR/no-ASR aggregation configuration.
//!
//! Symmetry correctness and ASR physical appropriateness remain declared rather
//! than independently proven by this layer.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle,
    ExternalEvidenceError, Sha256Digest,
};

use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_relaxation_completeness::bind_relaxation_complete_crystal_execution_plan;

pub use super::matter_relaxation_completeness::{
    CrystalClaimCapabilityReceipt, MixedCrystalExecutionPlanReceipt,
    PeriodicEnergyConvergenceStudyReceipt, RelaxationCellMode,
    RelaxationCompletenessError, RelaxationCompletenessInterpretation,
    RelaxationCompletenessReceipt, RelaxationCompleteCrystalExecutionPlan,
    VariableCellCompletenessBinding, VariableCellCompletenessReceipt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryReductionMode {
    None,
    DeclaredIrreducibleReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticSumRuleTreatment {
    NoneDeclared,
    Translational,
    TranslationalAndRotational,
    CustomDeclared,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QPointGridMappingReceipt {
    pub q_point_id: String,
    pub grid_index: [u32; 3],
    /// Full-grid multiplicity represented by this sampled q point.
    pub multiplicity: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetryReductionReceipt {
    pub mode: SymmetryReductionMode,
    /// `DependencySnapshot` required exactly for an irreducible reduction.
    pub symmetry_snapshot_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticSumRuleReceipt {
    pub treatment: AcousticSumRuleTreatment,
    /// Must equal the lattice aggregation's exact configuration artifact.
    pub aggregation_configuration_artifact_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhononCompletenessReceipt {
    pub target_claim_id: String,
    pub reciprocal_grid: [u32; 3],
    pub reciprocal_shift_fractional_bits: [u64; 3],
    pub coordinate_tolerance_bits: u64,
    pub coordinate_criterion_evidence_id: String,
    /// `DependencySnapshot` retained by every phonon batch.
    pub reciprocal_grid_snapshot_id: String,
    pub symmetry: SymmetryReductionReceipt,
    pub mappings: Vec<QPointGridMappingReceipt>,
    pub acoustic_sum_rule: AcousticSumRuleReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QPointGridMappingBinding {
    pub q_point_id: String,
    pub grid_index: [u32; 3],
    pub multiplicity: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetryReductionBinding {
    pub mode: SymmetryReductionMode,
    pub symmetry_snapshot_id: Option<String>,
    pub symmetry_snapshot_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticSumRuleBinding {
    pub treatment: AcousticSumRuleTreatment,
    pub aggregation_configuration_evidence_id: String,
    pub aggregation_configuration_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhononCompletenessInterpretation {
    /// Grid membership/multiplicity are checked; group-theoretic correctness of
    /// a declared symmetry reduction is not independently proven here.
    DeclaredReciprocalGridCoverage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhononCompleteCrystalExecutionPlan {
    pub relaxation_complete: RelaxationCompleteCrystalExecutionPlan,
    pub reciprocal_grid: [u32; 3],
    pub reciprocal_shift_fractional_bits: [u64; 3],
    pub coordinate_tolerance_bits: u64,
    pub coordinate_criterion_evidence_id: String,
    pub coordinate_criterion_sha256: Sha256Digest,
    pub reciprocal_grid_snapshot_id: String,
    pub reciprocal_grid_snapshot_sha256: Sha256Digest,
    pub mappings: Vec<QPointGridMappingBinding>,
    pub symmetry: SymmetryReductionBinding,
    pub acoustic_sum_rule: AcousticSumRuleBinding,
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
    pub completeness_interpretation: PhononCompletenessInterpretation,
}

/// Strict public route above relaxation completeness.
pub fn bind_phonon_complete_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    capability_receipts: &[CrystalClaimCapabilityReceipt],
    studies: &[PeriodicEnergyConvergenceStudyReceipt],
    relaxation_receipt: RelaxationCompletenessReceipt,
    phonon: PhononCompletenessReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<PhononCompleteCrystalExecutionPlan, PhononCompletenessError> {
    let raw_lattice = plan_receipt.lattice_dynamics_graph.clone();
    let parent = bind_relaxation_complete_crystal_execution_plan(
        evidence_bound,
        plan_receipt,
        capability_receipts,
        studies,
        relaxation_receipt,
        bundle,
    )?;
    let lattice = &parent.converged.capability_bound.plan.lattice_dynamics;
    if phonon.target_claim_id != lattice.target_claim_id {
        return Err(PhononCompletenessError::TargetClaimIdentityMismatch);
    }

    let full_grid_count = full_grid_count(phonon.reciprocal_grid)?;
    let shift = validate_shift(phonon.reciprocal_shift_fractional_bits)?;
    let coordinate_tolerance = finite_nonnegative(
        phonon.coordinate_tolerance_bits,
        "coordinate tolerance",
    )?;
    for (field, value) in [
        (
            "coordinate_criterion_evidence_id",
            phonon.coordinate_criterion_evidence_id.as_str(),
        ),
        (
            "reciprocal_grid_snapshot_id",
            phonon.reciprocal_grid_snapshot_id.as_str(),
        ),
        (
            "aggregation_configuration_artifact_id",
            phonon
                .acoustic_sum_rule
                .aggregation_configuration_artifact_id
                .as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(PhononCompletenessError::EmptyField(field));
        }
    }

    let target_claim = lattice
        .evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| {
            snapshot.slot
                == super::matter_crystal_phase::CrystalPhaseEvidenceSlot::DynamicalStability
        })
        .ok_or(PhononCompletenessError::MissingDynamicalClaim)?
        .claim
        .clone();

    let grid_snapshot = bundle.require_role(
        &phonon.reciprocal_grid_snapshot_id,
        EvidenceRole::DependencySnapshot,
    )?;
    let grid_date = grid_snapshot.claimed_utc_date.ok_or_else(|| {
        PhononCompletenessError::MissingClaimedDate(
            phonon.reciprocal_grid_snapshot_id.clone(),
        )
    })?;
    if !claim_retains(&target_claim, grid_snapshot.id.as_str()) {
        return Err(PhononCompletenessError::GridSnapshotMissingFromTargetClaim);
    }

    let coordinate_criterion = bundle.require_role(
        &phonon.coordinate_criterion_evidence_id,
        EvidenceRole::Preregistration,
    )?;
    if !claim_retains(&target_claim, coordinate_criterion.id.as_str()) {
        return Err(PhononCompletenessError::CoordinateCriterionMissingFromTargetClaim);
    }

    let symmetry = bind_symmetry(&phonon.symmetry, &target_claim, bundle)?;
    let asr = bind_asr(&phonon.acoustic_sum_rule, &target_claim, &raw_lattice, bundle)?;

    // Build the exact bound q-point set, retaining the coordinate bits already
    // proven by the parent lattice graph.
    let mut samples = BTreeMap::new();
    for batch in &lattice.batches {
        if grid_date > batch.execution.execution_claimed_date {
            return Err(PhononCompletenessError::GridSnapshotPostdatesExecution);
        }
        if !claim_retains(&batch.claim, grid_snapshot.id.as_str()) {
            return Err(PhononCompletenessError::GridSnapshotMissingFromBatchClaim(
                batch.batch_id.clone(),
            ));
        }
        if !claim_retains(&batch.claim, coordinate_criterion.id.as_str()) {
            return Err(PhononCompletenessError::CoordinateCriterionMissingFromBatchClaim(
                batch.batch_id.clone(),
            ));
        }
        if let Some(symmetry_id) = symmetry.symmetry_snapshot_id.as_deref() {
            if !claim_retains(&batch.claim, symmetry_id) {
                return Err(PhononCompletenessError::SymmetrySnapshotMissingFromBatchClaim(
                    batch.batch_id.clone(),
                ));
            }
        }
        for sample in &batch.samples {
            if samples
                .insert(sample.q_point_id.clone(), sample.q_fractional_bits)
                .is_some()
            {
                return Err(PhononCompletenessError::DuplicateQPointIdentity(
                    sample.q_point_id.clone(),
                ));
            }
        }
    }

    validate_execution_dependencies(
        &raw_lattice,
        grid_snapshot.id.as_str(),
        symmetry.symmetry_snapshot_id.as_deref(),
    )?;

    if phonon.mappings.len() != samples.len() {
        return Err(PhononCompletenessError::MappingSetSizeMismatch {
            mappings: phonon.mappings.len(),
            samples: samples.len(),
        });
    }

    let mut seen_ids = BTreeSet::new();
    let mut seen_indices = BTreeSet::new();
    let mut multiplicity_sum = 0_u64;
    let mut mappings = Vec::with_capacity(phonon.mappings.len());
    for mapping in &phonon.mappings {
        if mapping.q_point_id.trim().is_empty() {
            return Err(PhononCompletenessError::EmptyField("q_point_id"));
        }
        if mapping.multiplicity == 0 {
            return Err(PhononCompletenessError::ZeroMultiplicity(
                mapping.q_point_id.clone(),
            ));
        }
        if !seen_ids.insert(mapping.q_point_id.clone()) {
            return Err(PhononCompletenessError::DuplicateMappingQPoint(
                mapping.q_point_id.clone(),
            ));
        }
        for axis in 0..3 {
            if mapping.grid_index[axis] >= phonon.reciprocal_grid[axis] {
                return Err(PhononCompletenessError::GridIndexOutOfBounds {
                    q_point_id: mapping.q_point_id.clone(),
                    axis,
                });
            }
        }
        if !seen_indices.insert(mapping.grid_index) {
            return Err(PhononCompletenessError::DuplicateRepresentativeGridIndex(
                mapping.grid_index,
            ));
        }
        let actual_bits = samples
            .get(&mapping.q_point_id)
            .ok_or_else(|| {
                PhononCompletenessError::UnknownMappedQPoint(mapping.q_point_id.clone())
            })?;
        validate_q_coordinate(
            &mapping.q_point_id,
            *actual_bits,
            mapping.grid_index,
            phonon.reciprocal_grid,
            shift,
            coordinate_tolerance,
        )?;
        multiplicity_sum = multiplicity_sum
            .checked_add(u64::from(mapping.multiplicity))
            .ok_or(PhononCompletenessError::MultiplicityOverflow)?;
        mappings.push(QPointGridMappingBinding {
            q_point_id: mapping.q_point_id.clone(),
            grid_index: mapping.grid_index,
            multiplicity: mapping.multiplicity,
        });
    }
    if seen_ids.len() != samples.len() {
        return Err(PhononCompletenessError::MappingSetMismatch);
    }

    match phonon.symmetry.mode {
        SymmetryReductionMode::None => {
            if u64::try_from(phonon.mappings.len()).ok() != Some(full_grid_count)
                || phonon.mappings.iter().any(|mapping| mapping.multiplicity != 1)
            {
                return Err(PhononCompletenessError::UnreducedGridCoverageMismatch);
            }
        }
        SymmetryReductionMode::DeclaredIrreducibleReduction => {
            if u64::try_from(phonon.mappings.len()).unwrap_or(u64::MAX) >= full_grid_count {
                return Err(PhononCompletenessError::ReductionDoesNotReduceGrid);
            }
        }
    }
    if multiplicity_sum != full_grid_count {
        return Err(PhononCompletenessError::MultiplicityDoesNotReconstructFullGrid {
            expected: full_grid_count,
            actual: multiplicity_sum,
        });
    }

    let mut chronology = Vec::with_capacity(lattice.batches.len());
    for batch in &lattice.batches {
        chronology.push(bundle.require_preregistration_before(
            coordinate_criterion.id.as_str(),
            &batch.execution.execution_evidence_id,
        )?);
    }

    mappings.sort_by(|a, b| a.q_point_id.cmp(&b.q_point_id));
    Ok(PhononCompleteCrystalExecutionPlan {
        relaxation_complete: parent,
        reciprocal_grid: phonon.reciprocal_grid,
        reciprocal_shift_fractional_bits: phonon.reciprocal_shift_fractional_bits,
        coordinate_tolerance_bits: phonon.coordinate_tolerance_bits,
        coordinate_criterion_evidence_id: coordinate_criterion.id.as_str().to_string(),
        coordinate_criterion_sha256: coordinate_criterion.content_sha256.clone(),
        reciprocal_grid_snapshot_id: grid_snapshot.id.as_str().to_string(),
        reciprocal_grid_snapshot_sha256: grid_snapshot.content_sha256.clone(),
        mappings,
        symmetry,
        acoustic_sum_rule: asr,
        criterion_chronology: chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
        completeness_interpretation:
            PhononCompletenessInterpretation::DeclaredReciprocalGridCoverage,
    })
}

fn bind_symmetry(
    receipt: &SymmetryReductionReceipt,
    target_claim: &symthaea_epistemic_types::MatterClaim,
    bundle: &ExternalEvidenceBundle,
) -> Result<SymmetryReductionBinding, PhononCompletenessError> {
    match (receipt.mode, receipt.symmetry_snapshot_id.as_deref()) {
        (SymmetryReductionMode::None, None) => Ok(SymmetryReductionBinding {
            mode: receipt.mode,
            symmetry_snapshot_id: None,
            symmetry_snapshot_sha256: None,
        }),
        (SymmetryReductionMode::None, Some(_)) => {
            Err(PhononCompletenessError::SymmetrySnapshotForbiddenWithoutReduction)
        }
        (SymmetryReductionMode::DeclaredIrreducibleReduction, None) => {
            Err(PhononCompletenessError::SymmetrySnapshotRequired)
        }
        (SymmetryReductionMode::DeclaredIrreducibleReduction, Some(id)) => {
            if id.trim().is_empty() {
                return Err(PhononCompletenessError::EmptyField("symmetry_snapshot_id"));
            }
            let snapshot = bundle.require_role(id, EvidenceRole::DependencySnapshot)?;
            if !claim_retains(target_claim, snapshot.id.as_str()) {
                return Err(PhononCompletenessError::SymmetrySnapshotMissingFromTargetClaim);
            }
            Ok(SymmetryReductionBinding {
                mode: receipt.mode,
                symmetry_snapshot_id: Some(snapshot.id.as_str().to_string()),
                symmetry_snapshot_sha256: Some(snapshot.content_sha256.clone()),
            })
        }
    }
}

fn bind_asr(
    receipt: &AcousticSumRuleReceipt,
    target_claim: &symthaea_epistemic_types::MatterClaim,
    raw_lattice: &super::matter_lattice_dynamics_binding::LatticeDynamicsExecutionGraphReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<AcousticSumRuleBinding, PhononCompletenessError> {
    if receipt.aggregation_configuration_artifact_id.trim().is_empty() {
        return Err(PhononCompletenessError::EmptyField(
            "aggregation_configuration_artifact_id",
        ));
    }
    if receipt.aggregation_configuration_artifact_id
        != raw_lattice.aggregation.configuration_artifact_id
    {
        return Err(PhononCompletenessError::AsrConfigurationIsNotAggregationConfiguration);
    }
    let config = bundle.require_role(
        &receipt.aggregation_configuration_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    if !claim_retains(target_claim, config.id.as_str()) {
        return Err(PhononCompletenessError::AsrConfigurationMissingFromTargetClaim);
    }
    Ok(AcousticSumRuleBinding {
        treatment: receipt.treatment,
        aggregation_configuration_evidence_id: config.id.as_str().to_string(),
        aggregation_configuration_sha256: config.content_sha256.clone(),
    })
}

fn validate_execution_dependencies(
    raw_lattice: &super::matter_lattice_dynamics_binding::LatticeDynamicsExecutionGraphReceipt,
    grid_id: &str,
    symmetry_id: Option<&str>,
) -> Result<(), PhononCompletenessError> {
    for batch in &raw_lattice.batches {
        if !batch.execution.dependency_snapshot_ids.iter().any(|id| id == grid_id) {
            return Err(PhononCompletenessError::GridSnapshotNotExecutionDependency(
                batch.batch_id.clone(),
            ));
        }
        if let Some(symmetry_id) = symmetry_id {
            if !batch
                .execution
                .dependency_snapshot_ids
                .iter()
                .any(|id| id == symmetry_id)
            {
                return Err(PhononCompletenessError::SymmetrySnapshotNotExecutionDependency(
                    batch.batch_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn full_grid_count(grid: [u32; 3]) -> Result<u64, PhononCompletenessError> {
    if grid.iter().any(|n| *n == 0) {
        return Err(PhononCompletenessError::ZeroGridDimension);
    }
    grid.into_iter().try_fold(1_u64, |acc, n| {
        acc.checked_mul(u64::from(n))
            .ok_or(PhononCompletenessError::GridSizeOverflow)
    })
}

fn validate_shift(bits: [u64; 3]) -> Result<[f64; 3], PhononCompletenessError> {
    let mut shift = [0.0; 3];
    for axis in 0..3 {
        let value = f64::from_bits(bits[axis]);
        if !value.is_finite() || !(0.0..1.0).contains(&value) {
            return Err(PhononCompletenessError::InvalidReciprocalShift { axis });
        }
        shift[axis] = value;
    }
    Ok(shift)
}

fn validate_q_coordinate(
    q_point_id: &str,
    actual_bits: [u64; 3],
    index: [u32; 3],
    grid: [u32; 3],
    shift: [f64; 3],
    tolerance: f64,
) -> Result<(), PhononCompletenessError> {
    for axis in 0..3 {
        let actual = f64::from_bits(actual_bits[axis]);
        if !actual.is_finite() {
            return Err(PhononCompletenessError::InvalidQCoordinate {
                q_point_id: q_point_id.to_string(),
                axis,
            });
        }
        let expected = (f64::from(index[axis]) + shift[axis]) / f64::from(grid[axis]);
        if periodic_unit_distance(actual, expected) > tolerance {
            return Err(PhononCompletenessError::QCoordinateOutsideTolerance {
                q_point_id: q_point_id.to_string(),
                axis,
            });
        }
    }
    Ok(())
}

fn periodic_unit_distance(a: f64, b: f64) -> f64 {
    let raw = (a - b).rem_euclid(1.0);
    raw.min(1.0 - raw)
}

fn finite_nonnegative(bits: u64, field: &'static str) -> Result<f64, PhononCompletenessError> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(PhononCompletenessError::InvalidMetric(field));
    }
    Ok(value)
}

fn claim_retains(claim: &symthaea_epistemic_types::MatterClaim, id: &str) -> bool {
    claim.evidence.iter().any(|evidence| evidence.evidence_id == id)
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhononCompletenessError {
    Parent(RelaxationCompletenessError),
    ExternalEvidence(ExternalEvidenceError),
    EmptyField(&'static str),
    InvalidMetric(&'static str),
    TargetClaimIdentityMismatch,
    MissingDynamicalClaim,
    MissingClaimedDate(String),
    ZeroGridDimension,
    GridSizeOverflow,
    InvalidReciprocalShift { axis: usize },
    GridSnapshotPostdatesExecution,
    GridSnapshotMissingFromTargetClaim,
    GridSnapshotMissingFromBatchClaim(String),
    GridSnapshotNotExecutionDependency(String),
    CoordinateCriterionMissingFromTargetClaim,
    CoordinateCriterionMissingFromBatchClaim(String),
    SymmetrySnapshotForbiddenWithoutReduction,
    SymmetrySnapshotRequired,
    SymmetrySnapshotMissingFromTargetClaim,
    SymmetrySnapshotMissingFromBatchClaim(String),
    SymmetrySnapshotNotExecutionDependency(String),
    AsrConfigurationIsNotAggregationConfiguration,
    AsrConfigurationMissingFromTargetClaim,
    MappingSetSizeMismatch { mappings: usize, samples: usize },
    MappingSetMismatch,
    DuplicateQPointIdentity(String),
    DuplicateMappingQPoint(String),
    UnknownMappedQPoint(String),
    ZeroMultiplicity(String),
    GridIndexOutOfBounds { q_point_id: String, axis: usize },
    DuplicateRepresentativeGridIndex([u32; 3]),
    InvalidQCoordinate { q_point_id: String, axis: usize },
    QCoordinateOutsideTolerance { q_point_id: String, axis: usize },
    MultiplicityOverflow,
    MultiplicityDoesNotReconstructFullGrid { expected: u64, actual: u64 },
    UnreducedGridCoverageMismatch,
    ReductionDoesNotReduceGrid,
}

impl From<RelaxationCompletenessError> for PhononCompletenessError {
    fn from(value: RelaxationCompletenessError) -> Self {
        Self::Parent(value)
    }
}

impl From<ExternalEvidenceError> for PhononCompletenessError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for PhononCompletenessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "phonon completeness rejected: {self:?}")
    }
}

impl std::error::Error for PhononCompletenessError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_distance_accepts_wrapped_equivalents() {
        assert!(periodic_unit_distance(-0.25, 0.75) < 1.0e-14);
    }

    #[test]
    fn zero_grid_dimension_is_rejected() {
        assert_eq!(
            full_grid_count([4, 0, 4]).unwrap_err(),
            PhononCompletenessError::ZeroGridDimension
        );
    }

    #[test]
    fn shift_must_lie_in_unit_interval() {
        assert_eq!(
            validate_shift([
                0.0_f64.to_bits(),
                1.0_f64.to_bits(),
                0.0_f64.to_bits(),
            ])
            .unwrap_err(),
            PhononCompletenessError::InvalidReciprocalShift { axis: 1 }
        );
    }
}
