// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic ordered periodic-cell identity for Matter crystal lineages.
//!
//! v1 canonicalizes one explicitly ordered periodic cell: exact lattice-vector
//! bits (with signed zero normalized), fully occupied atomic species, and
//! fractional coordinates wrapped into [0,1) then deterministically site-sorted.
//! It deliberately does **not** identify cells related by arbitrary integer basis
//! transforms, primitive/conventional-cell choices, origin shifts, or disorder.
//!
//! The cell values are locally canonicalized, but the mapping from those values
//! to an external structure artifact is still a declared mapping: this module
//! does not parse and re-derive the cell from arbitrary CIF/POSCAR/QE bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle,
    ExternalEvidenceError, Sha256Digest,
};

use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_phase_competition_graph::PhaseEnergyRole;
use super::matter_phonon_completeness::bind_phonon_complete_crystal_execution_plan;

pub use super::matter_phonon_completeness::{
    AcousticSumRuleReceipt, AcousticSumRuleTreatment, CrystalClaimCapabilityReceipt,
    MixedCrystalExecutionPlanReceipt, PeriodicEnergyConvergenceStudyReceipt,
    PhononCompleteCrystalExecutionPlan, PhononCompletenessError, PhononCompletenessReceipt,
    QPointGridMappingReceipt, RelaxationCellMode, RelaxationCompletenessReceipt,
    SymmetryReductionMode, SymmetryReductionReceipt, VariableCellCompletenessReceipt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedPeriodicSiteReceipt {
    pub atomic_number: u16,
    pub fractional_bits: [u64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedPeriodicCellReceipt {
    pub structure_artifact_id: String,
    /// Three ordered direct-lattice vectors in ångström, stored as exact f64 bits.
    pub lattice_vectors_angstrom_bits: [[u64; 3]; 3],
    /// Fully occupied sites only in v1. Partial/disordered occupancy is out of scope.
    pub sites: Vec<OrderedPeriodicSiteReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrderedPeriodicSiteIdentity {
    pub atomic_number: u16,
    pub fractional_bits: [u64; 3],
}

/// The representation key used for exact v1 cell equality.
///
/// Lattice-vector order is intentionally retained. Therefore cells related only
/// by a basis permutation or another GL(3,Z) transform are *not* declared equal
/// by v1 unless their ordered representation is already identical.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrderedPeriodicCellKey {
    pub lattice_vectors_angstrom_bits: [[u64; 3]; 3],
    pub sites: Vec<OrderedPeriodicSiteIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedPeriodicCellIdentityBinding {
    pub structure_artifact_id: String,
    pub structure_artifact_sha256: Sha256Digest,
    pub key: OrderedPeriodicCellKey,
    /// Sorted `(atomic_number, site_count)` composition.
    pub composition: Vec<(u16, u32)>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub identity_interpretation: OrderedCellIdentityInterpretation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedCellIdentityInterpretation {
    /// Receipt values were canonicalized locally, while their mapping to the
    /// external artifact remains declared rather than independently re-parsed.
    LocallyCanonicalizedDeclaredArtifactMappingV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseOrderedCellIdentityBinding {
    pub phase_id: String,
    pub role: PhaseEnergyRole,
    pub claim_id: String,
    pub structure_artifact_id: String,
    pub key: OrderedPeriodicCellKey,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentityBoundCrystalExecutionPlan {
    pub phonon_complete: PhononCompleteCrystalExecutionPlan,
    pub cells: Vec<OrderedPeriodicCellIdentityBinding>,
    pub final_relaxed_structure_artifact_id: String,
    pub final_relaxed_key: OrderedPeriodicCellKey,
    pub main_periodic_structure_artifact_id: String,
    pub phase_identities: Vec<PhaseOrderedCellIdentityBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub identity_interpretation: OrderedCellIdentityInterpretation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedStructureArtifact {
    sha256: Sha256Digest,
}

pub fn bind_ordered_periodic_crystal_identities(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    capability_receipts: &[CrystalClaimCapabilityReceipt],
    studies: &[PeriodicEnergyConvergenceStudyReceipt],
    relaxation_receipt: RelaxationCompletenessReceipt,
    phonon_receipt: PhononCompletenessReceipt,
    cells: &[OrderedPeriodicCellReceipt],
    bundle: &ExternalEvidenceBundle,
) -> Result<IdentityBoundCrystalExecutionPlan, CrystalIdentityError> {
    let parent = bind_phonon_complete_crystal_execution_plan(
        evidence_bound,
        plan_receipt,
        capability_receipts,
        studies,
        relaxation_receipt,
        phonon_receipt,
        bundle,
    )?;

    let expected = collect_expected_structure_artifacts(&parent)?;
    let mut supplied = BTreeMap::<String, &OrderedPeriodicCellReceipt>::new();
    for receipt in cells {
        if receipt.structure_artifact_id.trim().is_empty() {
            return Err(CrystalIdentityError::EmptyStructureArtifactId);
        }
        if supplied
            .insert(receipt.structure_artifact_id.clone(), receipt)
            .is_some()
        {
            return Err(CrystalIdentityError::DuplicateStructureIdentityReceipt(
                receipt.structure_artifact_id.clone(),
            ));
        }
    }
    let expected_ids: BTreeSet<_> = expected.keys().cloned().collect();
    let supplied_ids: BTreeSet<_> = supplied.keys().cloned().collect();
    if expected_ids != supplied_ids {
        return Err(CrystalIdentityError::StructureReceiptSetMismatch);
    }

    let mut identities = BTreeMap::<String, OrderedPeriodicCellIdentityBinding>::new();
    let mut digest_to_key = BTreeMap::<String, OrderedPeriodicCellKey>::new();
    for (id, expected_artifact) in &expected {
        let receipt = supplied
            .get(id)
            .expect("exact structure receipt set checked above");
        let structure = bundle.require_role(id, EvidenceRole::ArtifactContent)?;
        if structure.content_sha256 != expected_artifact.sha256 {
            return Err(CrystalIdentityError::ParentStructureDigestMismatch(id.clone()));
        }
        let (key, composition) = canonicalize_cell(receipt)?;
        let digest = structure.content_sha256.as_str().to_string();
        match digest_to_key.get(&digest) {
            Some(existing) if existing != &key => {
                return Err(CrystalIdentityError::SameArtifactContentDifferentCellIdentity)
            }
            Some(_) => {}
            None => {
                digest_to_key.insert(digest, key.clone());
            }
        }
        identities.insert(
            id.clone(),
            OrderedPeriodicCellIdentityBinding {
                structure_artifact_id: id.clone(),
                structure_artifact_sha256: structure.content_sha256.clone(),
                key,
                composition,
                reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
                identity_interpretation:
                    OrderedCellIdentityInterpretation::LocallyCanonicalizedDeclaredArtifactMappingV1,
            },
        );
    }

    let plan = &parent.relaxation_complete.converged.capability_bound.plan;
    let final_relaxed_id = plan.relaxation.result.final_structure_evidence_id.clone();
    let final_relaxed = identity(&identities, &final_relaxed_id)?;

    let target_elements: BTreeSet<u16> = plan
        .evidence_bound
        .admitted
        .target
        .element_atomic_numbers
        .iter()
        .copied()
        .collect();
    let final_elements: BTreeSet<u16> = final_relaxed
        .composition
        .iter()
        .map(|(z, _)| *z)
        .collect();
    if final_elements != target_elements {
        return Err(CrystalIdentityError::FinalStructureElementSetMismatch);
    }
    let final_relaxed_key = final_relaxed.key.clone();

    let studies_by_execution: BTreeMap<_, _> = parent
        .relaxation_complete
        .converged
        .periodic_energy_studies
        .iter()
        .map(|study| (study.production_execution_evidence_id.as_str(), study))
        .collect();

    let main_study = studies_by_execution
        .get(plan.periodic_execution.execution_evidence_id.as_str())
        .ok_or(CrystalIdentityError::MissingMainPeriodicConvergenceStudy)?;
    let main_structure_id = final_kpoint_structure_id(main_study)?;
    let main_structure = identity(&identities, main_structure_id)?;
    if main_structure.key != final_relaxed_key {
        return Err(CrystalIdentityError::MainPeriodicStructureIsNotFinalRelaxedCell);
    }

    let mut phase_identities = Vec::with_capacity(plan.phase_competition.leaves.len());
    let mut phase_keys = BTreeMap::<OrderedPeriodicCellKey, String>::new();
    let mut saw_target = false;
    for leaf in &plan.phase_competition.leaves {
        let study = studies_by_execution
            .get(leaf.execution.execution_evidence_id.as_str())
            .ok_or_else(|| {
                CrystalIdentityError::MissingPhaseConvergenceStudy(leaf.phase_id.clone())
            })?;
        let structure_id = final_kpoint_structure_id(study)?;
        let cell = identity(&identities, structure_id)?;
        let element_set: BTreeSet<u16> = cell.composition.iter().map(|(z, _)| *z).collect();
        if !element_set.is_subset(&target_elements) {
            return Err(CrystalIdentityError::PhaseContainsElementOutsideTargetSystem(
                leaf.phase_id.clone(),
            ));
        }
        match leaf.role {
            PhaseEnergyRole::TargetPhase => {
                if saw_target {
                    return Err(CrystalIdentityError::MultipleTargetPhaseIdentities);
                }
                saw_target = true;
                if cell.key != final_relaxed_key {
                    return Err(CrystalIdentityError::TargetPhaseIsNotFinalRelaxedCell);
                }
            }
            PhaseEnergyRole::CompetingPhase => {}
        }
        if let Some(previous) = phase_keys.insert(cell.key.clone(), leaf.phase_id.clone()) {
            if previous != leaf.phase_id {
                return Err(CrystalIdentityError::DuplicateOrderedCellUnderDifferentPhaseLabels {
                    first: previous,
                    second: leaf.phase_id.clone(),
                });
            }
        }
        phase_identities.push(PhaseOrderedCellIdentityBinding {
            phase_id: leaf.phase_id.clone(),
            role: leaf.role,
            claim_id: leaf.claim.claim_id.clone(),
            structure_artifact_id: structure_id.to_string(),
            key: cell.key.clone(),
        });
    }
    if !saw_target {
        return Err(CrystalIdentityError::MissingTargetPhaseIdentity);
    }

    // Lattice dynamics must explicitly retain an artifact whose ordered-cell
    // identity equals the final relaxed target cell. This is declaration-level
    // linkage until a solver adapter exposes the structure as an exact input slot.
    let target_aliases: BTreeSet<String> = identities
        .values()
        .filter(|cell| cell.key == final_relaxed_key)
        .map(|cell| cell.structure_artifact_id.clone())
        .collect();
    let lattice = &plan.lattice_dynamics;
    let dynamical_target = lattice
        .evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| {
            snapshot.slot
                == super::matter_crystal_phase::CrystalPhaseEvidenceSlot::DynamicalStability
        })
        .ok_or(CrystalIdentityError::MissingDynamicalStabilityClaim)?;
    require_claim_retains_one_structure(&dynamical_target.claim, &target_aliases)?;
    for batch in &lattice.batches {
        require_claim_retains_one_structure(&batch.claim, &target_aliases).map_err(|_| {
            CrystalIdentityError::LatticeBatchMissingTargetStructure(batch.batch_id.clone())
        })?;
    }

    // Every relaxation handoff has both sides locally canonicalized. Digest
    // equality from the parent must therefore also imply exact ordered-cell equality.
    for window in plan.relaxation.steps.windows(2) {
        let left = identity(&identities, &window[0].output_structure_evidence_id)?;
        let right = identity(&identities, &window[1].input_structure_evidence_id)?;
        if left.key != right.key {
            return Err(CrystalIdentityError::RelaxationHandoffCellIdentityMismatch {
                next_step_index: window[1].step_index,
            });
        }
    }

    let mut bound_cells: Vec<_> = identities.into_values().collect();
    bound_cells.sort_by(|a, b| a.structure_artifact_id.cmp(&b.structure_artifact_id));
    phase_identities.sort_by(|a, b| a.phase_id.cmp(&b.phase_id));

    Ok(IdentityBoundCrystalExecutionPlan {
        phonon_complete: parent,
        cells: bound_cells,
        final_relaxed_structure_artifact_id: final_relaxed_id,
        final_relaxed_key,
        main_periodic_structure_artifact_id: main_structure_id.to_string(),
        phase_identities,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        identity_interpretation:
            OrderedCellIdentityInterpretation::LocallyCanonicalizedDeclaredArtifactMappingV1,
    })
}

fn collect_expected_structure_artifacts(
    parent: &PhononCompleteCrystalExecutionPlan,
) -> Result<BTreeMap<String, ExpectedStructureArtifact>, CrystalIdentityError> {
    let mut expected = BTreeMap::new();
    let plan = &parent.relaxation_complete.converged.capability_bound.plan;
    for step in &plan.relaxation.steps {
        insert_expected(
            &mut expected,
            &step.input_structure_evidence_id,
            &step.input_structure_sha256,
        )?;
        insert_expected(
            &mut expected,
            &step.output_structure_evidence_id,
            &step.output_structure_sha256,
        )?;
    }
    insert_expected(
        &mut expected,
        &plan.relaxation.result.final_structure_evidence_id,
        &plan.relaxation.result.final_structure_sha256,
    )?;
    for study in &parent.relaxation_complete.converged.periodic_energy_studies {
        for sample in study.cutoff_series.iter().chain(study.kpoint_series.iter()) {
            insert_expected(
                &mut expected,
                &sample.normalized_structure_evidence_id,
                &sample.normalized_structure_sha256,
            )?;
        }
    }
    if expected.is_empty() {
        return Err(CrystalIdentityError::NoStructureArtifacts);
    }
    Ok(expected)
}

fn insert_expected(
    expected: &mut BTreeMap<String, ExpectedStructureArtifact>,
    id: &str,
    digest: &Sha256Digest,
) -> Result<(), CrystalIdentityError> {
    match expected.get(id) {
        Some(existing) if existing.sha256 != *digest => {
            Err(CrystalIdentityError::SameStructureIdDifferentDigest(id.to_string()))
        }
        Some(_) => Ok(()),
        None => {
            expected.insert(
                id.to_string(),
                ExpectedStructureArtifact {
                    sha256: digest.clone(),
                },
            );
            Ok(())
        }
    }
}

fn canonicalize_cell(
    receipt: &OrderedPeriodicCellReceipt,
) -> Result<(OrderedPeriodicCellKey, Vec<(u16, u32)>), CrystalIdentityError> {
    if receipt.sites.is_empty() {
        return Err(CrystalIdentityError::EmptyCell);
    }

    let mut lattice = [[0_u64; 3]; 3];
    let mut lattice_values = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let value = f64::from_bits(receipt.lattice_vectors_angstrom_bits[i][j]);
            if !value.is_finite() {
                return Err(CrystalIdentityError::NonFiniteLatticeComponent { vector: i, axis: j });
            }
            let normalized = normalize_signed_zero(value);
            lattice[i][j] = normalized.to_bits();
            lattice_values[i][j] = normalized;
        }
    }
    let det = determinant(lattice_values);
    if !det.is_finite() || det == 0.0 {
        return Err(CrystalIdentityError::SingularOrInvalidLattice);
    }

    let mut sites = Vec::with_capacity(receipt.sites.len());
    let mut coordinates = BTreeSet::new();
    let mut composition = BTreeMap::<u16, u32>::new();
    for site in &receipt.sites {
        if site.atomic_number == 0 {
            return Err(CrystalIdentityError::InvalidAtomicNumber);
        }
        let mut fractional_bits = [0_u64; 3];
        for axis in 0..3 {
            let value = f64::from_bits(site.fractional_bits[axis]);
            if !value.is_finite() {
                return Err(CrystalIdentityError::NonFiniteFractionalCoordinate { axis });
            }
            let wrapped = normalize_signed_zero(value.rem_euclid(1.0));
            fractional_bits[axis] = wrapped.to_bits();
        }
        if !coordinates.insert(fractional_bits) {
            return Err(CrystalIdentityError::CoincidentOrderedSites);
        }
        let count = composition.entry(site.atomic_number).or_default();
        *count = count
            .checked_add(1)
            .ok_or(CrystalIdentityError::CompositionCountOverflow)?;
        sites.push(OrderedPeriodicSiteIdentity {
            atomic_number: site.atomic_number,
            fractional_bits,
        });
    }
    sites.sort();

    Ok((
        OrderedPeriodicCellKey {
            lattice_vectors_angstrom_bits: lattice,
            sites,
        },
        composition.into_iter().collect(),
    ))
}

fn normalize_signed_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn determinant(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn final_kpoint_structure_id(
    study: &super::matter_periodic_convergence_binding::PeriodicEnergyConvergenceStudyBinding,
) -> Result<&str, CrystalIdentityError> {
    study
        .kpoint_series
        .last()
        .map(|sample| sample.normalized_structure_evidence_id.as_str())
        .ok_or(CrystalIdentityError::MissingFinalKPointStructure)
}

fn identity<'a>(
    identities: &'a BTreeMap<String, OrderedPeriodicCellIdentityBinding>,
    id: &str,
) -> Result<&'a OrderedPeriodicCellIdentityBinding, CrystalIdentityError> {
    identities
        .get(id)
        .ok_or_else(|| CrystalIdentityError::MissingCanonicalIdentity(id.to_string()))
}

fn require_claim_retains_one_structure(
    claim: &symthaea_epistemic_types::MatterClaim,
    allowed: &BTreeSet<String>,
) -> Result<(), CrystalIdentityError> {
    if claim
        .evidence
        .iter()
        .any(|evidence| allowed.contains(&evidence.evidence_id))
    {
        Ok(())
    } else {
        Err(CrystalIdentityError::DynamicalClaimMissingTargetStructure)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrystalIdentityError {
    Parent(PhononCompletenessError),
    ExternalEvidence(ExternalEvidenceError),
    EmptyStructureArtifactId,
    DuplicateStructureIdentityReceipt(String),
    StructureReceiptSetMismatch,
    ParentStructureDigestMismatch(String),
    SameArtifactContentDifferentCellIdentity,
    NoStructureArtifacts,
    SameStructureIdDifferentDigest(String),
    EmptyCell,
    NonFiniteLatticeComponent { vector: usize, axis: usize },
    SingularOrInvalidLattice,
    InvalidAtomicNumber,
    NonFiniteFractionalCoordinate { axis: usize },
    CoincidentOrderedSites,
    CompositionCountOverflow,
    MissingCanonicalIdentity(String),
    FinalStructureElementSetMismatch,
    MissingMainPeriodicConvergenceStudy,
    MissingFinalKPointStructure,
    MainPeriodicStructureIsNotFinalRelaxedCell,
    MissingPhaseConvergenceStudy(String),
    PhaseContainsElementOutsideTargetSystem(String),
    MultipleTargetPhaseIdentities,
    TargetPhaseIsNotFinalRelaxedCell,
    DuplicateOrderedCellUnderDifferentPhaseLabels { first: String, second: String },
    MissingTargetPhaseIdentity,
    MissingDynamicalStabilityClaim,
    DynamicalClaimMissingTargetStructure,
    LatticeBatchMissingTargetStructure(String),
    RelaxationHandoffCellIdentityMismatch { next_step_index: u32 },
}

impl From<PhononCompletenessError> for CrystalIdentityError {
    fn from(value: PhononCompletenessError) -> Self {
        Self::Parent(value)
    }
}

impl From<ExternalEvidenceError> for CrystalIdentityError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for CrystalIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ordered periodic-cell identity rejected: {self:?}")
    }
}

impl std::error::Error for CrystalIdentityError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cell(x: f64) -> OrderedPeriodicCellReceipt {
        OrderedPeriodicCellReceipt {
            structure_artifact_id: "cell".into(),
            lattice_vectors_angstrom_bits: [
                [1.0_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()],
                [0.0_f64.to_bits(), 1.0_f64.to_bits(), 0.0_f64.to_bits()],
                [0.0_f64.to_bits(), 0.0_f64.to_bits(), 1.0_f64.to_bits()],
            ],
            sites: vec![OrderedPeriodicSiteReceipt {
                atomic_number: 14,
                fractional_bits: [x.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()],
            }],
        }
    }

    #[test]
    fn fractional_coordinates_wrap_periodically() {
        let (a, _) = canonicalize_cell(&simple_cell(-0.25)).unwrap();
        let (b, _) = canonicalize_cell(&simple_cell(0.75)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_site_order_is_input_order_independent() {
        let mut a = simple_cell(0.25);
        a.sites.push(OrderedPeriodicSiteReceipt {
            atomic_number: 8,
            fractional_bits: [0.5_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()],
        });
        let mut b = a.clone();
        b.sites.reverse();
        assert_eq!(canonicalize_cell(&a).unwrap(), canonicalize_cell(&b).unwrap());
    }

    #[test]
    fn duplicate_periodic_site_is_rejected() {
        let mut cell = simple_cell(0.0);
        cell.sites.push(OrderedPeriodicSiteReceipt {
            atomic_number: 8,
            fractional_bits: [1.0_f64.to_bits(), 0.0_f64.to_bits(), 0.0_f64.to_bits()],
        });
        assert_eq!(
            canonicalize_cell(&cell).unwrap_err(),
            CrystalIdentityError::CoincidentOrderedSites
        );
    }

    #[test]
    fn singular_lattice_is_rejected() {
        let mut cell = simple_cell(0.0);
        cell.lattice_vectors_angstrom_bits[2] = cell.lattice_vectors_angstrom_bits[1];
        assert_eq!(
            canonicalize_cell(&cell).unwrap_err(),
            CrystalIdentityError::SingularOrInvalidLattice
        );
    }
}
